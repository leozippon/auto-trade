"""The experiment-level PRIOR handoff between research sessions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autotrade.agent.prompts import build_system_prompt
from autotrade.environment.tools import SafeWorkspace, WriteFileTool
from autotrade.environment.tools.prior_policy import PRIOR_MAX_CHARS
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.prior import (
    ExperimentPriorStore,
    latest_prior_text,
    restore_current_from_records,
)


def test_the_session_prompt_injects_the_prior_full_text_as_a_handoff() -> None:
    prior = "先用 grep 定向，再抽样 parquet。\n不要并行委托。"
    prompt = build_system_prompt(prior_prompt=prior)
    assert prior in prompt
    assert "上一会话的交接" in prompt
    assert "结束前改写它就是交给下一会话的交接" in prompt


def test_session_write_tools_cannot_overwrite_the_published_prior(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    store.root.mkdir(parents=True)
    legacy_current = store.root / "CURRENT.md"
    legacy_current.write_text("stale duplicate\n", encoding="utf-8")
    published = store.publish("sample then count", generation_id="meta_001_run_a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    WriteFileTool(SafeWorkspace(workspace)).invoke(
        {"path": "PRIOR.md", "content": "tampered workspace copy"}
    )
    assert (workspace / "PRIOR.md").read_text(encoding="utf-8") == "tampered workspace copy"
    assert store.current_text().strip() == "sample then count"
    assert legacy_current.read_text(encoding="utf-8") == "stale duplicate\n"
    assert Path(published.prior_ref).read_text(encoding="utf-8").strip() == "sample then count"
    assert not hasattr(published, "sha256")


def test_a_session_publishes_keeps_and_rejects_an_overlong_prior(tmp_path: Path) -> None:
    from autotrade.pipelines.experiment import RollingExperimentPipeline

    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    first = store.publish("first workflow", generation_id="gen_1")
    pipeline = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    pipeline.config = type("Cfg", (), {"experiment_dir": experiment})()

    published = pipeline._publish_or_keep_prior(
        "updated workflow notes", previous=first.text, generation_id="gen_2"
    )
    assert published["prior_published"] is True
    assert store.current_text().strip() == "updated workflow notes"
    assert Path(published["prior_ref"]).is_file()

    # An empty or unchanged PRIOR.md keeps the generation in force.
    for candidate, generation in (("", "gen_3"), ("updated workflow notes\n", "gen_same")):
        kept = pipeline._publish_or_keep_prior(
            candidate, previous="updated workflow notes", generation_id=generation
        )
        assert kept["prior"] == "updated workflow notes"
        assert kept["prior_published"] is False
        assert kept["prior_generation_id"] == "gen_2"
    assert store.current_generation_id() == "gen_2"

    # Before any PRIOR exists, an empty handoff is simply empty.
    empty = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    empty.config = type("Cfg", (), {"experiment_dir": tmp_path / "empty"})()
    assert empty._publish_or_keep_prior("", previous="", generation_id="gen_first") == {
        "prior": "",
        "prior_published": False,
        "prior_ref": None,
        "prior_generation_id": None,
        "prior_chars": 0,
    }

    with pytest.raises(ValueError, match="characters"):
        pipeline._publish_or_keep_prior(
            "x" * (PRIOR_MAX_CHARS + 1),
            previous="updated workflow notes",
            generation_id="gen_overlong",
        )
    with pytest.raises(FileExistsError):
        pipeline._publish_or_keep_prior(
            "collision", previous="updated workflow notes", generation_id="gen_2"
        )


def test_latest_prior_reads_the_last_research_session_record() -> None:
    records = [
        {"record_type": "meta_learning", "prior": "ignored"},
        {"record_type": "research_session", "prior": "first"},
        {"record_type": "research_session", "prior": "second", "prior_published": False},
        {"record_type": "forward", "prior": "ignored"},
    ]
    assert latest_prior_text(records) == "second"
    assert latest_prior_text([]) == ""
    assert latest_prior_text([{"record_type": "meta_learning", "prior": "ignored"}]) == ""


def test_prior_store_restore_points_current_at_earlier_generation(
    tmp_path: Path,
) -> None:
    store = ExperimentPriorStore(tmp_path / "experiment")
    first = store.publish("first workflow", generation_id="gen_1")
    store.publish("second workflow", generation_id="gen_2")
    restored = store.restore("gen_1")
    assert store.current_text().strip() == "first workflow"
    assert store.current_generation_id() == "gen_1"
    assert restored.prior_ref == first.prior_ref
    assert Path(first.prior_ref).read_text(encoding="utf-8").strip() == "first workflow"
    assert (
        Path(store.root / "generations" / "gen_2" / "PRIOR.md")
        .read_text(encoding="utf-8")
        .strip()
        == "second workflow"
    )


def test_prior_store_rejects_malformed_or_dangling_current_ref(tmp_path: Path) -> None:
    store = ExperimentPriorStore(tmp_path / "experiment")
    store.root.mkdir(parents=True)
    store.current_pointer_path.write_text("../outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generation_id"):
        store.current_text()
    store.current_pointer_path.write_text("missing\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        store.current_text()


def test_prior_store_serializes_concurrent_publications(tmp_path: Path) -> None:
    store = ExperimentPriorStore(tmp_path / "experiment")
    publications = (("first", "gen_1"), ("second", "gen_2"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda item: store.publish(item[0], generation_id=item[1]), publications)
        )
    current_generation = store.current_generation_id()
    assert current_generation in {"gen_1", "gen_2"}
    expected = {result.generation_id: result.text for result in results}
    assert store.current_text().strip() == expected[current_generation]
    assert all(Path(result.prior_ref).is_file() for result in results)


def _append_session(
    ledger: ExperimentLedger,
    *,
    run_id: str,
    generation_id: str = "",
    prior: str = "",
) -> None:
    ledger.append(
        {
            "record_type": "research_session",
            "experiment_id": "exp",
            "epoch_id": "research",
            "fold_id": "s1",
            "run_id": run_id,
            "prior": prior,
            "prior_generation_id": generation_id or None,
        }
    )


def test_restore_prior_store_rewinds_current_after_rollback(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    store.publish("first workflow", generation_id="gen_1")
    store.publish("second workflow", generation_id="gen_2")
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _append_session(
        ledger, run_id="run_1", generation_id="gen_1", prior="first workflow"
    )
    restore_current_from_records(experiment, ledger.read())
    assert store.current_generation_id() == "gen_1"
    assert store.current_text().strip() == "first workflow"
    assert (
        Path(store.root / "generations" / "gen_2" / "PRIOR.md")
        .read_text(encoding="utf-8")
        .strip()
        == "second workflow"
    )


def test_restore_prior_store_clears_current_when_no_generation_remains(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    store.publish("later workflow", generation_id="gen_2")
    legacy_current = store.root / "CURRENT.md"
    legacy_current.write_text("ignored legacy copy\n", encoding="utf-8")
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    restore_current_from_records(experiment, ledger.read())
    assert store.current_generation_id() == ""
    assert store.current_text() == ""
    assert legacy_current.read_text(encoding="utf-8") == "ignored legacy copy\n"
    assert (
        Path(store.root / "generations" / "gen_2" / "PRIOR.md")
        .read_text(encoding="utf-8")
        .strip()
        == "later workflow"
    )


def test_restore_prior_store_fails_if_generation_is_missing(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _append_session(ledger, run_id="run_ghost", generation_id="ghost", prior="gone")
    with pytest.raises(FileNotFoundError, match="ghost"):
        restore_current_from_records(experiment, ledger.read())
