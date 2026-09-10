"""Seed a new experiment's PRIOR and skills from another experiment's memory.

``inherit_from`` copies a frozen strategy; ``inherit_memory_from`` copies what
the source learned -- its latest PRIOR and its current skills generation -- so
an arm re-created on a new PIT seed or calendar starts from the closed
families and negative results its predecessor paid for, whether or not that
predecessor ever froze a strategy. Both are copied at creation time as this
experiment's own read-only generations and are read exactly like a Meta
publication: the first Meta keeps or replaces the PRIOR, the first Fold and
Meta mount the skills, and a session row then takes over as the head.
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

from .hitl_state import read_json
from .ledger import ExperimentLedger
from .prior import ExperimentPriorStore, latest_prior_text
from .skills import (
    SkillsSnapshot,
    import_skills_generation,
    latest_skills_snapshot,
    skills_snapshot_from_ref,
)

# The ``hitl/params.json`` key the creation writes and the worker reads back;
# console-managed like ``_inherited_artifact``, never a request parameter.
INHERITED_MEMORY_PARAM = "_inherited_memory"


@dataclass(frozen=True)
class InheritedMemory:
    source_experiment_id: str
    prior_generation_id: str
    prior_text: str
    skills: SkillsSnapshot | None


def import_inherited_memory(
    experiment_dir: str | Path, source_dir: str | Path, *, source_id: str
) -> dict[str, object]:
    """Copy the source's latest PRIOR and current skills into the new
    experiment as its own read-only generations; the params.json payload.

    The source's ledger is the authority on both, as it is for the source's
    own sessions. A source that never published a PRIOR has no memory to
    inherit and the creation fails here.
    """

    experiment = Path(experiment_dir)
    records = ExperimentLedger(
        Path(source_dir) / "ledgers" / "experiment_ledger.jsonl"
    ).read()
    prior = latest_prior_text(records)
    if not prior:
        raise ValueError(
            f"source experiment {source_id!r} has no published PRIOR to inherit"
        )
    generation = f"inherited_{source_id}"
    published = ExperimentPriorStore(experiment).publish(prior, generation_id=generation)
    metas = [row for row in records if row.get("record_type") == "meta_learning"]
    payload: dict[str, object] = {
        "source_experiment_id": source_id,
        "prior_generation_id": published.generation_id,
        "prior_ref": published.prior_ref,
        "prior_source_generation_id": str(metas[-1].get("prior_generation_id") or "")
        or None,
        "skills_ref": None,
        "skills_generation_id": None,
        "skills_source_generation_id": None,
    }
    source_skills = latest_skills_snapshot(records, experiment_dir=source_dir)
    if source_skills.root is not None:
        skills = import_skills_generation(
            experiment, source_skills.root, generation_id=generation
        )
        if skills.published:
            payload.update(
                skills_ref=skills.skills_ref,
                skills_generation_id=skills.generation_id,
                skills_source_generation_id=source_skills.generation_id,
            )
    return payload


def load_inherited_memory(experiment_dir: str | Path) -> InheritedMemory | None:
    """The memory seeded at creation, re-validated rather than trusted.

    Like the inherited parent: a generation that is missing, empty or
    writable again is a tampered seed and stops the run instead of starting a
    session from unverified memory. None for an experiment created without
    ``inherit_memory_from``.
    """

    experiment = Path(experiment_dir)
    payload = read_json(experiment / "hitl" / "params.json").get(INHERITED_MEMORY_PARAM)
    if not isinstance(payload, dict):
        return None
    store = ExperimentPriorStore(experiment)
    generation = str(payload.get("prior_generation_id") or "")
    path = store.generation_path(generation)
    if not path.is_file():
        raise RuntimeError(f"inherited PRIOR generation is missing: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError(f"inherited PRIOR generation is writable: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"inherited PRIOR generation is empty: {path}")
    skills_ref = str(payload.get("skills_ref") or "")
    skills = (
        skills_snapshot_from_ref(
            experiment,
            skills_ref,
            generation_id=str(payload.get("skills_generation_id") or ""),
        )
        if skills_ref
        else None
    )
    return InheritedMemory(
        source_experiment_id=str(payload.get("source_experiment_id") or ""),
        prior_generation_id=generation,
        prior_text=text,
        skills=skills,
    )


__all__ = [
    "INHERITED_MEMORY_PARAM",
    "InheritedMemory",
    "import_inherited_memory",
    "load_inherited_memory",
]
