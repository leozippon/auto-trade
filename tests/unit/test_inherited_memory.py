"""``inherit_memory_from``: a new experiment starts from another's PRIOR and skills."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from autotrade.pipelines.inherited_memory import (
    INHERITED_MEMORY_PARAM,
    import_inherited_memory,
    load_inherited_memory,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.prior import ExperimentPriorStore, latest_prior_text
from autotrade.pipelines.skills import ExperimentSkillsStore, latest_skills_snapshot
from autotrade.pipelines.worker import _restore_prior_store

PRIOR = "Closed: small-cap reversal (null percentile 0.5). Next: post-event drift with a matched control."


def _meta_row(**fields: object) -> dict[str, object]:
    return {
        "record_type": "meta_learning",
        "experiment_id": "src",
        "epoch_id": "epoch_001",
        "fold_id": "meta_001",
        "run_id": "run_m",
        **fields,
    }


def _source(tmp_path: Path, *, prior: str = PRIOR, skills: bool = True) -> Path:
    """A source experiment as its own ledger leaves it: one Meta row naming
    the PRIOR generation it published and (optionally) the skills generation
    that is its head."""
    source = tmp_path / "experiments" / "src"
    row = _meta_row(prior=prior, prior_generation_id="gen_1" if prior else None)
    if prior:
        ExperimentPriorStore(source).publish(prior, generation_id="gen_1")
    if skills:
        tree = tmp_path / "skills_src" / "skills"
        (tree / "closed-families").mkdir(parents=True)
        (tree / "closed-families" / "SKILL.md").write_text(
            "# Closed Families\n\nSmall-cap reversal is closed; do not re-test it.\n",
            encoding="utf-8",
        )
        published = ExperimentSkillsStore(source).publish(tree, generation_id="gen_1")
        row.update(
            skills_ref=published.skills_ref,
            skills_generation_id=published.generation_id,
            **published.stats.ledger_fields(),
            skills_published=True,
        )
    ExperimentLedger(source / "ledgers" / "experiment_ledger.jsonl").append(row)
    return source


def _new_experiment(tmp_path: Path) -> Path:
    new = tmp_path / "experiments" / "new"
    (new / "hitl").mkdir(parents=True)
    _write_params(new, None)
    return new


def _write_params(experiment: Path, payload: dict[str, object] | None) -> None:
    params: dict[str, object] = {"experiment_id": experiment.name}
    if payload is not None:
        params[INHERITED_MEMORY_PARAM] = payload
    (experiment / "hitl" / "params.json").write_text(json.dumps(params), encoding="utf-8")


def _writable(path: Path) -> bool:
    return bool(stat.S_IMODE(path.stat().st_mode) & 0o222)


def test_import_seeds_read_only_generations_the_worker_reads_as_a_meta_publication(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    new = _new_experiment(tmp_path)
    assert load_inherited_memory(new) is None

    payload = import_inherited_memory(new, source, source_id="src")
    assert payload["source_experiment_id"] == "src"
    assert payload["prior_generation_id"] == "inherited_src"
    assert payload["prior_source_generation_id"] == "gen_1"
    assert payload["skills_ref"] == "artifacts/skills/generations/inherited_src/skills"
    assert payload["skills_generation_id"] == "inherited_src"
    assert payload["skills_source_generation_id"] == "gen_1"
    store = ExperimentPriorStore(new)
    assert store.current_generation_id() == "inherited_src"
    assert store.current_text().strip() == PRIOR
    assert not _writable(Path(str(payload["prior_ref"])))
    skills_root = new / str(payload["skills_ref"])
    assert not _writable(skills_root / "closed-families" / "SKILL.md")

    _write_params(new, payload)
    memory = load_inherited_memory(new)
    assert memory is not None
    assert memory.prior_text == PRIOR
    assert memory.skills is not None
    assert memory.skills.root == skills_root.resolve()
    assert memory.skills.generation_id == "inherited_src"
    assert memory.skills.stats.count == 1

    # Worker start before the first Meta: CURRENT stays on the inherited
    # generation (the first Meta that keeps the PRIOR records this ref), the
    # PRIOR text is what the ledger cannot yet provide, and the skills head is
    # the inherited snapshot until a session row exists.
    ledger = ExperimentLedger(new / "ledgers" / "experiment_ledger.jsonl")
    _restore_prior_store(new, ledger, fallback_generation_id=memory.prior_generation_id)
    assert store.current_generation_id() == "inherited_src"
    assert latest_prior_text(ledger.read("meta_learning")) == ""
    assert (
        latest_skills_snapshot(ledger.read(), experiment_dir=new, inherited=memory.skills)
        == memory.skills
    )
    # A Meta row then takes over on both counts, exactly as after any Meta.
    store.publish("revised direction", generation_id="gen_meta")
    ledger.append(
        _meta_row(prior="revised direction", prior_generation_id="gen_meta", skills_ref=None)
    )
    _restore_prior_store(new, ledger, fallback_generation_id=memory.prior_generation_id)
    assert store.current_generation_id() == "gen_meta"
    assert latest_prior_text(ledger.read("meta_learning")) == "revised direction"
    assert (
        latest_skills_snapshot(ledger.read(), experiment_dir=new, inherited=memory.skills).root
        is None
    )


def test_a_source_without_a_published_prior_cannot_seed_memory(tmp_path: Path) -> None:
    source = _source(tmp_path, prior="")
    new = _new_experiment(tmp_path)
    with pytest.raises(ValueError, match="no published PRIOR"):
        import_inherited_memory(new, source, source_id="src")
    assert not (new / "artifacts").exists()


def test_a_source_without_skills_seeds_the_prior_alone(tmp_path: Path) -> None:
    source = _source(tmp_path, skills=False)
    new = _new_experiment(tmp_path)
    payload = import_inherited_memory(new, source, source_id="src")
    assert payload["skills_ref"] is None and payload["skills_generation_id"] is None
    _write_params(new, payload)
    memory = load_inherited_memory(new)
    assert memory is not None and memory.skills is None
    assert memory.prior_text == PRIOR


def test_a_tampered_inherited_seed_stops_the_run(tmp_path: Path) -> None:
    source = _source(tmp_path)
    new = _new_experiment(tmp_path)
    payload = import_inherited_memory(new, source, source_id="src")
    _write_params(new, payload)
    prior_path = Path(str(payload["prior_ref"]))
    prior_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="writable"):
        load_inherited_memory(new)
    prior_path.chmod(0o444)
    assert load_inherited_memory(new) is not None
    skill = new / str(payload["skills_ref"]) / "closed-families" / "SKILL.md"
    skill.chmod(0o644)
    with pytest.raises(ValueError, match="published skills path is writable"):
        load_inherited_memory(new)
    skill.chmod(0o444)
    prior_path.parent.chmod(0o755)
    prior_path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        load_inherited_memory(new)
