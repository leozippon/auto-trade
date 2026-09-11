"""``inherit_memory_from``: a new experiment starts from another's PRIOR and skills."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.inherited_memory import (
    INHERITED_MEMORY_PARAM,
    INHERITED_PRIOR_NOTE,
    INHERITED_SKILLS_NOTE,
    import_inherited_memory,
    inherited_prior_header,
    load_inherited_memory,
    prior_provenance,
    skills_provenance,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.meta_inputs import select_meta_review_folds
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


def _fold_facts(ref_store: AgentRefStore, **provenance: object) -> dict[str, object]:
    return build_experiment_facts(
        manifest={"kind": "fold", "experiment_id": "new", "run_id": "run_1"},
        ref_store=ref_store,
        **provenance,
    )


def test_every_surface_that_shows_inherited_memory_states_where_it_came_from(
    tmp_path: Path,
) -> None:
    """The defect this closes: the mounted PRIOR cites folds and artifacts of
    the source experiment's ledger, so a session that is not told the body is
    inherited reads its own empty ledger as the inconsistency."""

    source = _source(tmp_path)
    new = _new_experiment(tmp_path)
    payload = import_inherited_memory(new, source, source_id="src")
    _write_params(new, payload)
    memory = load_inherited_memory(new)
    assert memory is not None
    refs = AgentRefStore(new)

    origin = prior_provenance(memory, PRIOR, ref_store=refs)
    assert origin == {
        "source_experiment": "src",
        # Opaque like every other cross-session id, not the raw generation.
        "source_generation_id": refs.get_or_create("meta", "gen_1"),
        "inherited_at": str(payload["inherited_at"]),
        "note": INHERITED_PRIOR_NOTE,
    }
    assert origin["source_generation_id"] != "gen_1"
    skills_origin = skills_provenance(memory, memory.skills.root, ref_store=refs)
    assert skills_origin is not None and skills_origin["note"] == INHERITED_SKILLS_NOTE

    # The run facts carry both, and carry neither for an experiment that owns
    # its memory.
    facts = _fold_facts(refs, prior_provenance=origin, skills_provenance=skills_origin)
    assert facts["prior_provenance"] == origin
    assert facts["skills_provenance"] == skills_origin
    bare = _fold_facts(refs)
    assert "prior_provenance" not in bare and "skills_provenance" not in bare

    # The mount-time header names the source and answers the state assertion a
    # foreign PRIOR contradicts first; the read-only generation never gets it.
    header = inherited_prior_header(origin, has_parent=False)
    assert header.startswith("> 继承说明")
    assert "src" in header and str(payload["inherited_at"]) in header
    assert "基线锚点" in header
    assert "artifact_contract.parent" in inherited_prior_header(origin, has_parent=True)
    snapshot = Path(str(payload["prior_ref"]))
    assert snapshot.read_text(encoding="utf-8") == PRIOR + "\n"

    # The Meta's review window names that generation instead of leaving
    # previous_meta_ref null beside a PRIOR that plainly has a predecessor.
    folds, window = select_meta_review_folds([], ref_store=refs, inherited_prior=origin)
    assert folds == [] and window["fold_count"] == 0
    assert window["previous_meta_ref"] == origin["source_generation_id"]
    assert window["prior_provenance"] == origin

    # Once this experiment's own Meta publishes a PRIOR the provenance is gone
    # on every surface: the body is no longer the inherited one, and the window
    # names this experiment's Meta row.
    assert prior_provenance(memory, "our own direction", ref_store=refs) is None
    _, own = select_meta_review_folds(
        [_meta_row(prior="our own direction", meta_learning_id="meta_002")],
        ref_store=refs,
        inherited_prior=None,
    )
    assert own["previous_meta_ref"] == refs.get_or_create("meta", "meta_002")
    assert "prior_provenance" not in own


def test_no_provenance_without_an_inherited_seed_or_for_another_generation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    new = _new_experiment(tmp_path)
    payload = import_inherited_memory(new, source, source_id="src")
    _write_params(new, payload)
    memory = load_inherited_memory(new)
    assert memory is not None
    refs = AgentRefStore(new)

    assert prior_provenance(None, PRIOR, ref_store=refs) is None
    assert skills_provenance(None, memory.skills.root, ref_store=refs) is None
    # A session mounting some other skills generation is not mounting this one.
    assert skills_provenance(memory, "", ref_store=refs) is None
    assert skills_provenance(memory, new / "artifacts", ref_store=refs) is None
    # A source that seeded no skills has no skills provenance to state.
    other = _source(tmp_path / "second", skills=False)
    plain = _new_experiment(tmp_path / "second")
    _write_params(plain, import_inherited_memory(plain, other, source_id="src"))
    plain_memory = load_inherited_memory(plain)
    assert plain_memory is not None and plain_memory.skills is None
    assert (
        skills_provenance(plain_memory, memory.skills.root, ref_store=AgentRefStore(plain))
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
