"""Seed a new experiment's PRIOR and skills from another experiment's memory.

``inherit_from`` copies a frozen strategy; ``inherit_memory_from`` copies what
the source learned -- its latest PRIOR and its current skills generation -- so
an arm re-created on a new PIT seed or calendar starts from the closed
families and negative results its predecessor paid for, whether or not that
predecessor ever froze a strategy. Both are copied at creation time as this
experiment's own read-only generations and are read exactly like a Meta
publication: the first Meta keeps or replaces the PRIOR, the first Fold and
Meta mount the skills, and a session row then takes over as the head.

Read exactly like a Meta publication is what a session cannot infer on its
own: the inherited PRIOR names folds and artifacts of the source experiment's
ledger, which this experiment's ledger has never heard of. So this module is
also the single source of the provenance every surface that shows inherited
memory projects -- the run facts, ``meta_context.json`` and the mount-time
header of the read-only ``PRIOR.md`` -- and the seeded snapshots stay
byte-identical to what the source published.
"""

from __future__ import annotations

import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import utc_now_iso

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

# What "inherited" means for each artifact, in one sentence, carried by the
# provenance block itself so the facts, meta_context.json and the mounted
# PRIOR header all state it identically.
INHERITED_PRIOR_NOTE = (
    "本 PRIOR 由另一个实验的 Meta 写成、在本实验创建时整份继承："
    "它引用的折 id、产物 id 与账本数值属于那个实验的账本，本实验账本里没有这些记录，"
    "它提到的产物也没有挂载到本会话；其中的机制关闭与负面结果按先验知识沿用，"
    "父本、冻结、预算一类的状态断言一律以本会话运行事实为准。"
)
INHERITED_SKILLS_NOTE = (
    "可写 skills/ 的初始内容同样整份继承自那个实验的同一份记忆，不是本实验写下的："
    "它是可改写的先验知识，与本窗口证据冲突时以证据为准。"
)


@dataclass(frozen=True)
class InheritedMemory:
    source_experiment_id: str
    inherited_at: str
    prior_generation_id: str
    prior_source_generation_id: str
    prior_text: str
    skills: SkillsSnapshot | None
    skills_source_generation_id: str


def _provenance(
    memory: InheritedMemory,
    *,
    source_generation_id: str,
    note: str,
    ref_store: AgentRefStore,
) -> dict[str, str]:
    return {
        "source_experiment": memory.source_experiment_id,
        # The source's generation id embeds its Epoch/fold labels, so it is
        # projected through the reference store like every other cross-session
        # identity the Agent sees.
        "source_generation_id": ref_store.get_or_create("meta", source_generation_id),
        "inherited_at": memory.inherited_at,
        "note": note,
    }


def prior_provenance(
    memory: InheritedMemory | None, prior_text: str, *, ref_store: AgentRefStore
) -> dict[str, str] | None:
    """Where the PRIOR now in force came from, or ``None`` when it is this
    experiment's own.

    The body itself is the test: the inherited snapshot stays in force until a
    Meta of this experiment publishes something different, and a first Meta
    that keeps it unchanged has not made it any less foreign.
    """

    if memory is None or prior_text.strip() != memory.prior_text:
        return None
    return _provenance(
        memory,
        source_generation_id=(
            memory.prior_source_generation_id or memory.prior_generation_id
        ),
        note=INHERITED_PRIOR_NOTE,
        ref_store=ref_store,
    )


def skills_provenance(
    memory: InheritedMemory | None,
    skills_source_ref: str | Path,
    *,
    ref_store: AgentRefStore,
) -> dict[str, str] | None:
    """The same for the skills generation a session is about to mount."""

    if memory is None or memory.skills is None or not str(skills_source_ref).strip():
        return None
    if Path(skills_source_ref).resolve() != memory.skills.root:
        return None
    return _provenance(
        memory,
        source_generation_id=(
            memory.skills_source_generation_id or memory.skills.generation_id
        ),
        note=INHERITED_SKILLS_NOTE,
        ref_store=ref_store,
    )


def inherited_prior_header(
    provenance: Mapping[str, object], *, has_parent: bool
) -> str:
    """The provenance line a session's read-only ``PRIOR.md`` is mounted with.

    Generated at mount time and never written back: the inherited generation
    under ``artifacts/prior/generations/`` stays byte-identical to the text the
    source experiment published. The parent clause is the one state assertion
    the header answers itself, because contradicting it is what a stale-looking
    PRIOR does first.
    """

    inherited_at = str(provenance.get("inherited_at") or "")
    return (
        "> 继承说明（宿主挂载时生成，不属于 PRIOR 正文）："
        f"{provenance['note']}"
        f"来源实验 `{provenance['source_experiment']}`、"
        f"来源代次 `{provenance['source_generation_id']}`"
        f"{'、继承于 ' + inherited_at if inherited_at else ''}。"
        + (
            "本实验已有自己的冻结父产物，以运行事实 `artifact_contract.parent` 为准。"
            if has_parent
            else "本实验尚无冻结父产物，基线锚点规则在本实验重新适用。"
        )
    )


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
        "inherited_at": utc_now_iso(),
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
        # Absent for a seed written before the creation recorded it; the rest
        # of the provenance stands on its own and the field is simply dropped.
        inherited_at=str(payload.get("inherited_at") or ""),
        prior_generation_id=generation,
        prior_source_generation_id=str(
            payload.get("prior_source_generation_id") or ""
        ),
        prior_text=text,
        skills=skills,
        skills_source_generation_id=str(
            payload.get("skills_source_generation_id") or ""
        ),
    )


__all__ = [
    "INHERITED_MEMORY_PARAM",
    "INHERITED_PRIOR_NOTE",
    "INHERITED_SKILLS_NOTE",
    "InheritedMemory",
    "import_inherited_memory",
    "inherited_prior_header",
    "load_inherited_memory",
    "prior_provenance",
    "skills_provenance",
]
