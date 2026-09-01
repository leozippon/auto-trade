from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines import (
    ArtifactRevision,
    EvaluationResult,
    FoldSessionResult,
    FrozenArtifact,
    RollingExperimentConfig,
    RollingExperimentPipeline,
    StepResult,
)
from autotrade.pipelines.config import (
    MetaSessionResult,
    fold_session_deadline_seconds,
)
from autotrade.pipelines.experiment import _session_budgets
from autotrade.pipelines.folds import build_fold_schedule
from autotrade.pipelines.hitl_state import fold_session_key
from autotrade.pipelines.ledger import (
    INTERRUPTED_RUN_ERROR,
    UNKNOWN_MARKER_LINK_KEY,
    UNREADABLE_RUN_MARKER_ERROR,
    ExperimentLedger,
    FrozenArtifactMutated,
    FrozenArtifactRestoreFailed,
    RunMarkers,
    latest_fold_records,
    latest_heldout_records,
)
from autotrade.pipelines.skills import install_workspace_skills


class Snapshots:
    def prepare(self, *, fold, phase, start, end, decision_time):
        from autotrade.pipelines.config import SnapshotBundle

        return SnapshotBundle(f"{phase}_{start}_{end}", "decision", "replay")


class Artifacts:
    def __init__(self, revision: ArtifactRevision, root: Path):
        self.revisions = {revision.revision_id: revision}
        self.root = root

    def revision(self, revision_id):
        return self.revisions[revision_id]

    def freeze_revision(self, revision_id, **values):
        source = self.revisions[revision_id]
        target = self.root / values["artifact_id"]
        shutil.copytree(source.output_path, target)
        models = None
        if source.models_path is not None and Path(source.models_path).is_dir():
            models = self.root / f"{values['artifact_id']}_models"
            shutil.copytree(source.models_path, models)
        return FrozenArtifact(
            values["artifact_id"], target, models, values["run_id"], values["fold_id"], values["step_id"]
        )


class Evaluator:
    def evaluate(self, request):
        return EvaluationResult(
            {"total_return": 0.02, "max_drawdown": -0.03, "filled_orders": 1},
            f"result/{request.mode}",
        )


def test_rolling_pipeline_runs_meta_fold_test_and_heldout(tmp_path: Path):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    revision = ArtifactRevision("revision_1", revision_dir)

    def developer(request):
        return FoldSessionResult(
            "conversation_1",
            (
                StepResult(
                    "step_1",
                    revision.revision_id,
                    EvaluationResult({"total_return": 0.05, "max_drawdown": -0.02}, "result/valid"),
                    True,
                ),
            ),
            "step_1",
        )

    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2025Q4",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        test_stage=True,
        epochs=1,
    )
    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=Evaluator(),
        developer=developer,
        meta_learner=lambda facts: MetaSessionResult(prior="prefer simple daily signals"),
        ledger=ledger,
    )
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2025-09-29", "2026-06-30")]
    fold = build_fold_schedule(
        "2025Q4", "2026Q1", days, window_months=24, test_stage=True
    )[0]
    # The same call order the interactive worker drives: epoch-start Meta, then
    # the Fold, then one Held-out pass over the resulting frontier.
    prior, parent = pipeline.run_meta_session("epoch_001", 0, fold, parent=None)
    assert prior == "prefer simple daily signals"
    outcome = pipeline.run_fold("epoch_001", fold, parent=parent, prior=prior)
    final = outcome.frozen
    assert final is not None
    heldout_runs = pipeline.run_heldout("epoch_001", final, days)
    assert heldout_runs == 1
    records = ledger.read()
    assert [record["record_type"] for record in records] == ["meta_learning", "fold", "heldout"]
    assert final.artifact_id.startswith("strategy_")
    heldout = records[-1]
    assert heldout["result"]["total_return"] == 0.02
    assert heldout["strategy_artifact_id"] == final.artifact_id
    fold_record = next(record for record in records if record["record_type"] == "fold")
    assert "state_changed_during_test" not in fold_record
    assert "state_changed_during_test" not in heldout


def test_successful_fold_publishes_skills_and_next_fold_noops_by_bytes(
    tmp_path: Path,
):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    revision = ArtifactRevision("revision_1", revision_dir)
    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        epochs=1,
    )
    seen_sources: list[str] = []

    def developer(request):
        seen_sources.append(request.skills_source_ref)
        workspace = config.experiment_dir / "artifacts" / request.run_id / "workspace"
        workspace.mkdir(parents=True)
        install_workspace_skills(request.skills_source_ref or None, workspace)
        if not request.skills_source_ref:
            item = workspace / "skills" / "schema-notes"
            item.mkdir()
            (item / "SKILL.md").write_text(
                "# Schema Notes\n\nRead schema before selecting columns.\n",
                encoding="utf-8",
            )
        return FoldSessionResult(
            "conversation_1",
            (
                StepResult(
                    "step_1",
                    revision.revision_id,
                    EvaluationResult(
                        {"total_return": 0.05, "max_drawdown": -0.02},
                        "result/valid",
                    ),
                    True,
                ),
            ),
            "step_1",
            skills_source_ref=str(workspace / "skills"),
        )

    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=Evaluator(),
        developer=developer,
        ledger=ledger,
    )
    days = [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range("2025-09-29", "2026-06-30")
    ]
    fold = build_fold_schedule("2026Q1", "2026Q1", days, window_months=24)[0]
    first = pipeline.run_fold("epoch_001", fold, parent=None)
    second = pipeline.run_fold("epoch_001", fold, parent=first.frozen)

    records = ledger.read("fold")
    assert records[0]["skills_published"] is True
    assert records[0]["skills_count"] == 1
    assert records[0]["skills_files"] == 1
    assert str(records[0]["skills_ref"]).startswith(
        "artifacts/skills/generations/"
    )
    assert records[1]["skills_published"] is False
    assert records[1]["skills_ref"] == records[0]["skills_ref"]
    assert records[1]["skills_generation_id"] == records[0]["skills_generation_id"]
    assert seen_sources[0] == ""
    assert Path(seen_sources[1]) == config.experiment_dir / str(records[0]["skills_ref"])
    assert second.frozen is not None


def test_meta_session_retains_only_the_authorized_test_diagnostic(tmp_path: Path):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n",
        encoding="utf-8",
    )
    revision = ArtifactRevision("revision_1", revision_dir)
    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        epochs=1,
    )
    AgentRefStore(config.experiment_dir)
    ledger = ExperimentLedger(config.ledger_path)
    ledger.append(
        {
            "record_type": "meta_learning",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "epoch_001",
            "run_id": "run_meta0",
            "session_key": "epoch_001/meta_learning",
            "meta_learning_id": "epoch_001",
            "prior": "initial",
            "status": "prior_only",
        }
    )
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2025Q4",
            "run_id": "run_prior",
            "fold_status": "frozen",
            "validation_period": "20251001..20251231",
            "validation_result": {"total_return": 0.03},
            "test_result": {
                "total_return": -0.02,
                "sharpe": -0.5,
                "max_drawdown": -0.08,
                "filled_orders": 4,
                "rejected_orders": 1,
                "private_detail": "must not enter Meta",
            },
            "accept_reasons": [],
            "accept_warnings": [],
        }
    )
    captured: dict[str, object] = {}

    def meta_learner(facts):
        captured.update(facts)
        return MetaSessionResult(prior="prefer robust signals")

    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=Evaluator(),
        developer=lambda _request: None,
        meta_learner=meta_learner,
        ledger=ledger,
    )
    days = [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range("2025-09-29", "2026-06-30")
    ]
    visible_fold = build_fold_schedule("2026Q1", "2026Q1", days, window_months=24)[0]

    prior = pipeline.run_meta_session(
        "epoch_001",
        1,
        visible_fold,
        parent=None,
        previous_prior="initial",
        session_context={
            "session_timing": lambda: {
                "run_wall_seconds": 12.34,
                "researcher_wait_seconds": 1.26,
            }
        },
    )

    # run_meta_session returns (PRIOR, next_parent): a regularized artifact
    # becomes the next Fold's parent, and a PRIOR-only session leaves it unchanged.
    assert prior == ("prefer robust signals", None)
    meta_record = ledger.read("meta_learning")[-1]
    assert meta_record["run_wall_seconds"] == 12.3
    assert meta_record["researcher_wait_seconds"] == 1.3
    history = captured["development_history"]
    assert isinstance(history, dict)
    assert set(history) == {
        "evaluation_contract",
        "fold_backtest_summaries",
        "fold_reviews",
        "fold_validation_history",
        "review_window",
        "meta_learning",
    }
    # Every completed Fold so far, not only the review window, reaches Meta as
    # a compact Validation summary.
    assert len(history["fold_validation_history"]) == len(latest_fold_records(ledger.read("fold")))
    window = history["review_window"]
    assert isinstance(window, dict)
    assert window["fold_count"] == 1
    assert "fold_2025Q4" not in str(window)
    assert captured.get("review_window") == window
    assert meta_record["review_window"] == window
    reviews = history["fold_reviews"]
    assert isinstance(reviews, list)
    assert reviews[0]["test_result"] == {
        "total_return": -0.02,
        "sharpe": -0.5,
        "max_drawdown": -0.08,
    }
    assert "private_detail" not in str(reviews)
    summaries = history["fold_backtest_summaries"]
    assert isinstance(summaries, list)
    # Only the compact frozen-test metric whitelist crosses into Meta.
    assert summaries[0]["test_result"] == {
        "total_return": -0.02,
        "sharpe": -0.5,
        "max_drawdown": -0.08,
    }
    assert summaries[0]["fold_id"].startswith("fold_ref_")
    # Each summary names the window it was replayed on, so a benchmark figure
    # is never read against a neighbouring node's period.
    assert summaries[0]["validation_period"] == "20251001..20251231"
    assert reviews[0]["validation_period"] == "20251001..20251231"
    assert "2025Q4" not in str(history)
    assert "private_detail" not in str(history)


def _meta_only_pipeline(tmp_path: Path, ledger: ExperimentLedger, captured: dict[str, object]):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    revision = ArtifactRevision("revision_1", revision_dir)
    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        epochs=1,
    )
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=Evaluator(),
        developer=lambda _request: None,
        meta_learner=lambda facts: (captured.update(facts) or MetaSessionResult(prior="next")),
        ledger=ledger,
    )
    days = [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range("2025-09-29", "2026-06-30")
    ]
    return pipeline, build_fold_schedule("2026Q1", "2026Q1", days, window_months=24)[0]


def test_first_meta_session_has_empty_review_window(tmp_path: Path):
    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        epochs=1,
    )
    AgentRefStore(config.experiment_dir)
    ledger = ExperimentLedger(config.ledger_path)
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2025Q4",
            "run_id": "run_prior",
            "fold_status": "frozen",
            "validation_result": {"total_return": 0.03},
        }
    )
    captured: dict[str, object] = {}
    pipeline, visible_fold = _meta_only_pipeline(tmp_path, ledger, captured)
    pipeline.run_meta_session("epoch_001", 0, visible_fold, parent=None, previous_prior="")
    history = captured["development_history"]
    assert isinstance(history, dict)
    assert history["fold_reviews"] == []
    assert history["fold_backtest_summaries"] == []
    window = history["review_window"]
    assert window == {
        "previous_meta_ref": None,
        "fold_run_refs": [],
        "fold_count": 0,
    }
    assert ledger.read("meta_learning")[-1]["review_window"] == window


def test_meta_session_window_skips_folds_before_previous_meta(tmp_path: Path):
    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        epochs=1,
    )
    AgentRefStore(config.experiment_dir)
    ledger = ExperimentLedger(config.ledger_path)
    ledger.append(
        {
            "record_type": "meta_learning",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "epoch_001",
            "run_id": "run_meta0",
            "meta_learning_id": "epoch_001",
            "status": "prior_only",
        }
    )
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "fold_old",
            "run_id": "run_old",
            "fold_status": "frozen",
            "validation_result": {"total_return": 0.01},
        }
    )
    ledger.append(
        {
            "record_type": "meta_learning",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "epoch_001_after_fold_001",
            "run_id": "run_meta1",
            "meta_learning_id": "epoch_001_after_fold_001",
            "status": "prior_only",
        }
    )
    ledger.append(
        {
            "record_type": "fold",
            "experiment_id": "experiment_a",
            "epoch_id": "epoch_001",
            "fold_id": "fold_new",
            "run_id": "run_new",
            "fold_status": "frozen",
            "validation_result": {"total_return": 0.04},
        }
    )
    captured: dict[str, object] = {}
    pipeline, visible_fold = _meta_only_pipeline(tmp_path, ledger, captured)
    pipeline.run_meta_session("epoch_001", 2, visible_fold, parent=None, previous_prior="")
    history = captured["development_history"]
    assert isinstance(history, dict)
    window = history["review_window"]
    assert isinstance(window, dict)
    assert window["fold_count"] == 1
    assert "fold_old" not in str(window)
    assert "fold_new" not in str(window)
    reviews = history["fold_reviews"]
    summaries = history["fold_backtest_summaries"]
    assert isinstance(reviews, list) and len(reviews) == 1
    assert isinstance(summaries, list) and len(summaries) == 1
    assert summaries[0]["validation_result"]["total_return"] == 0.04


def _pipeline_capturing_fold_requests(tmp_path: Path, captured: list, **config_overrides):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir(parents=True)
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    revision = ArtifactRevision("revision_1", revision_dir)

    def developer(request):
        captured.append(request)
        return FoldSessionResult(
            "conversation_1",
            (
                StepResult(
                    "step_1",
                    revision.revision_id,
                    EvaluationResult({"total_return": 0.05, "max_drawdown": -0.02}, "result/valid"),
                    True,
                ),
            ),
            "step_1",
        )

    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        epochs=1,
        **config_overrides,
    )
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=Evaluator(),
        developer=developer,
        meta_learner=lambda facts: MetaSessionResult(prior="unused"),
        ledger=ExperimentLedger(config.ledger_path),
    )
    days = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2025-09-29", "2026-06-30")]
    return pipeline, build_fold_schedule("2026Q1", "2026Q1", days, window_months=24)[0]


def test_run_fold_forwards_the_consoles_gpu_allocation_to_the_session_request(tmp_path: Path):
    """The seam between the HITL session context and the developer backend.

    `set_gpu_count` is only useful if the number the researcher typed at the
    approval gate reaches the request the sandbox is built from.
    """
    captured: list = []
    pipeline, fold = _pipeline_capturing_fold_requests(tmp_path, captured)
    pipeline.run_fold(
        "epoch_001", fold, parent=None, prior="", session_context={"sandbox_gpu_count": 3}
    )
    assert captured[-1].sandbox_gpu_count == 3
    pipeline.run_fold("epoch_001", fold, parent=None, prior="", session_context={})
    assert captured[-1].sandbox_gpu_count is None


def test_run_fold_refuses_a_gpu_override_that_is_not_in_0_to_4(tmp_path: Path):
    captured: list = []
    pipeline, fold = _pipeline_capturing_fold_requests(tmp_path, captured)
    for bogus in (-1, True, "2", 2.0, 5):
        with pytest.raises(ValueError, match="0..4"):
            pipeline.run_fold(
                "epoch_001", fold, parent=None, prior="",
                session_context={"sandbox_gpu_count": bogus},
            )
    pipeline.run_fold(
        "epoch_001", fold, parent=None, prior="", session_context={"sandbox_gpu_count": 0}
    )
    assert captured[-1].sandbox_gpu_count == 0


class ModeMutatingEvaluator:
    def __init__(self, mutate_mode: str, target: str = "output") -> None:
        self.mutate_mode = mutate_mode
        self.target = target

    def evaluate(self, request):
        if request.mode == self.mutate_mode:
            if self.target == "output":
                main = request.revision.output_path / "main.py"
                main.write_text(
                    main.read_text(encoding="utf-8") + "# mutated\n",
                    encoding="utf-8",
                )
            else:
                models = request.revision.models_path
                assert models is not None
                (models / "weights.json").write_text('{"w": 1}\n', encoding="utf-8")
        return EvaluationResult(
            {"total_return": 0.02, "max_drawdown": -0.03},
            f"result/{request.mode}",
        )


class CallbackMutatingEvaluator:
    def __init__(self, mutate_mode: str, mutate) -> None:
        self.mutate_mode = mutate_mode
        self.mutate = mutate

    def evaluate(self, request):
        if request.mode == self.mutate_mode:
            self.mutate(request)
        return EvaluationResult(
            {"total_return": 0.02, "max_drawdown": -0.03},
            f"result/{request.mode}",
        )


def _days() -> list[str]:
    return [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range("2025-09-29", "2026-06-30")
    ]


def _pipeline_with_evaluator(
    tmp_path: Path,
    evaluator,
    *,
    models: bool = False,
    extra_file: bool = False,
    **config_overrides,
):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    if extra_file:
        (revision_dir / "notes.txt").write_text("keep\n", encoding="utf-8")
    models_path = None
    if models:
        models_path = tmp_path / "revision_models"
        models_path.mkdir()
        (models_path / "weights.json").write_text("{}\n", encoding="utf-8")
    revision = ArtifactRevision("revision_1", revision_dir, models_path)

    def developer(request):
        return FoldSessionResult(
            "conversation_1",
            (
                StepResult(
                    "step_1",
                    revision.revision_id,
                    EvaluationResult(
                        {"total_return": 0.05, "max_drawdown": -0.02}, "result/valid"
                    ),
                    True,
                ),
            ),
            "step_1",
        )

    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2025Q4",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        test_stage=True,
        epochs=1,
        **config_overrides,
    )
    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=evaluator,
        developer=developer,
        meta_learner=lambda facts: MetaSessionResult(prior="prefer simple daily signals"),
        ledger=ledger,
    )
    fold = build_fold_schedule(
        "2025Q4", "2026Q1", _days(), window_months=24, test_stage=True
    )[0]
    return pipeline, fold, ledger


def _assert_frozen_main_restored(
    tmp_path: Path, original: str = "def generate_orders(context):\n    return []\n"
) -> None:
    mains = list((tmp_path / "frozen").rglob("main.py"))
    assert mains
    for main in mains:
        assert main.read_text(encoding="utf-8") == original
        assert not (main.stat().st_mode & 0o222)


def test_session_budgets_honor_nondefault_deadline_grace(tmp_path: Path):
    for minutes in (0, 20):
        config = RollingExperimentConfig(
            "experiment_a",
            tmp_path / f"grace_{minutes}",
            "2026Q1",
            "2026Q1",
            "2026Q2",
            "2026Q2",
            fold_period="quarter",
            epochs=1,
            deadline_grace_minutes=minutes,
        )
        budgets = _session_budgets(config, None)
        assert budgets["deadline_grace_seconds"] == minutes * 60
        assert budgets["deadline_seconds"] == fold_session_deadline_seconds(
            config.max_fold_minutes, minutes
        )
        overridden = _session_budgets(config, {"deadline_seconds": 1200})
        assert overridden["deadline_seconds"] == 1200 + minutes * 60
        assert overridden["deadline_grace_seconds"] == minutes * 60
        with pytest.raises(ValueError, match="unknown resource override"):
            _session_budgets(config, {"deadline_grace_seconds": 1})


def test_fold_session_request_carries_configured_deadline_grace(tmp_path: Path):
    for minutes in (0, 20):
        captured: list = []
        pipeline, fold = _pipeline_capturing_fold_requests(
            tmp_path / f"req_{minutes}",
            captured,
            deadline_grace_minutes=minutes,
        )
        pipeline.run_fold("epoch_001", fold, parent=None)
        request = captured[-1]
        assert request.deadline_grace_seconds == minutes * 60
        assert request.deadline_seconds == fold_session_deadline_seconds(
            pipeline.config.max_fold_minutes, minutes
        )


def test_frozen_test_fails_fast_when_output_changes(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(
        tmp_path, ModeMutatingEvaluator("frozen_test", "output")
    )
    with pytest.raises(FrozenArtifactMutated, match="changed during frozen test"):
        pipeline.run_fold("epoch_001", fold, parent=None)
    records = ledger.read()
    folds = [record for record in records if record["record_type"] == "fold"]
    assert len(folds) == 1
    assert folds[0]["state_changed_during_test"] is True
    assert not any(record["record_type"] == "attempt_failed" for record in records)
    assert latest_fold_records(records) == {}
    _assert_frozen_main_restored(tmp_path)
    with pytest.raises(FrozenArtifactMutated):
        pipeline.run_fold("epoch_001", fold, parent=None)
    assert [row.get("run_id") for row in ledger.read()] == [
        row.get("run_id") for row in records
    ]


def test_frozen_test_omits_state_changed_when_trees_are_stable(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    pipeline.run_fold("epoch_001", fold, parent=None)
    fold_record = ledger.read("fold")[0]
    assert "state_changed_during_test" not in fold_record


def test_heldout_fails_fast_when_models_change(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(
        tmp_path, ModeMutatingEvaluator("heldout", "models"), models=True
    )
    frozen = pipeline.run_fold("epoch_001", fold, parent=None).frozen
    assert frozen is not None
    with pytest.raises(FrozenArtifactMutated, match="changed during held-out"):
        pipeline.run_heldout("epoch_001", frozen, _days())
    records = ledger.read()
    heldout = [record for record in records if record["record_type"] == "heldout"]
    assert len(heldout) == 1
    assert heldout[0]["state_changed_during_test"] is True
    fold_record = next(record for record in records if record["record_type"] == "fold")
    assert "state_changed_during_test" not in fold_record
    assert latest_heldout_records(records) == []
    dummy = FrozenArtifact("x", tmp_path, None, "", "", "")
    with pytest.raises(FrozenArtifactMutated):
        pipeline.run_heldout("epoch_001", dummy, _days())
    assert [row.get("run_id") for row in ledger.read()] == [
        row.get("run_id") for row in records
    ]


def test_heldout_omits_state_changed_when_trees_are_stable(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    frozen = pipeline.run_fold("epoch_001", fold, parent=None).frozen
    assert frozen is not None
    assert pipeline.run_heldout("epoch_001", frozen, _days()) == 1
    heldout = ledger.read("heldout")[0]
    assert "state_changed_during_test" not in heldout
    assert sorted(RunMarkers(pipeline.config.experiment_dir).root.glob("*.json")) == []


def _add_output_file(request) -> None:
    (request.revision.output_path / "extra.py").write_text("x = 1\n", encoding="utf-8")


def _delete_output_file(request) -> None:
    (request.revision.output_path / "notes.txt").unlink()


def _delete_models_dir(request) -> None:
    models = request.revision.models_path
    assert models is not None
    shutil.rmtree(models)


def _add_models_dir(request) -> None:
    models = request.revision.output_path.parent / "models"
    models.mkdir()
    (models / "weights.json").write_text("{}\n", encoding="utf-8")


def _delete_output_dir(request) -> None:
    shutil.rmtree(request.revision.output_path)


@pytest.mark.parametrize(
    ("mutate", "models", "extra_file"),
    (
        (_add_output_file, False, False),
        (_delete_output_file, False, True),
        (_delete_models_dir, True, False),
        (_add_models_dir, False, False),
        (_delete_output_dir, False, False),
    ),
    ids=(
        "add_file",
        "delete_file",
        "missing_models",
        "new_models",
        "compare_error",
    ),
)
def test_frozen_test_records_integrity_failure_for_tree_mutations(
    tmp_path: Path, mutate, models: bool, extra_file: bool
):
    pipeline, fold, ledger = _pipeline_with_evaluator(
        tmp_path,
        CallbackMutatingEvaluator("frozen_test", mutate),
        models=models,
        extra_file=extra_file,
    )
    with pytest.raises(FrozenArtifactMutated, match="changed during frozen test"):
        pipeline.run_fold("epoch_001", fold, parent=None)
    records = ledger.read()
    folds = [record for record in records if record["record_type"] == "fold"]
    assert len(folds) == 1
    assert folds[0]["state_changed_during_test"] is True
    assert not any(record["record_type"] == "attempt_failed" for record in records)
    mains = list((tmp_path / "frozen").rglob("main.py"))
    assert mains
    if extra_file:
        notes = list((tmp_path / "frozen").rglob("notes.txt"))
        assert notes and notes[0].read_text(encoding="utf-8") == "keep\n"
    else:
        _assert_frozen_main_restored(tmp_path)


def test_frozen_restore_copy_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import autotrade.pipelines.experiment as experiment

    def boom(**_kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(experiment, "restore_frozen_artifact_trees", boom)
    pipeline, fold, ledger = _pipeline_with_evaluator(
        tmp_path, ModeMutatingEvaluator("frozen_test", "output")
    )
    with pytest.raises(FrozenArtifactRestoreFailed, match="restoring"):
        pipeline.run_fold("epoch_001", fold, parent=None)
    records = ledger.read()
    folds = [record for record in records if record["record_type"] == "fold"]
    assert len(folds) == 1
    assert folds[0]["state_changed_during_test"] is True
    mains = list((tmp_path / "frozen").rglob("main.py"))
    assert any("# mutated" in path.read_text(encoding="utf-8") for path in mains)
    with pytest.raises(FrozenArtifactMutated):
        pipeline.run_fold("epoch_001", fold, parent=None)
    assert [row.get("run_id") for row in ledger.read()] == [
        row.get("run_id") for row in records
    ]


class BenchmarkedEvaluator:
    """Replay summaries carrying the frozen benchmark block Held-out is judged on."""

    def __init__(self, total_return: float, sharpe: float, max_drawdown: float, benchmark: float):
        self.summary = {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "benchmark": {"label": "CSI 300", "benchmark_return": benchmark},
        }

    def evaluate(self, request):
        return EvaluationResult(dict(self.summary), f"result/{request.mode}")


def _single_window_pipeline(tmp_path: Path, evaluator):
    """One explicit-range development Fold, no Test stage."""
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    revision = ArtifactRevision("revision_1", revision_dir)

    def developer(request):
        return FoldSessionResult(
            "conversation_1",
            (
                StepResult(
                    "step_1",
                    revision.revision_id,
                    EvaluationResult({"total_return": 0.05, "max_drawdown": -0.02}, "result/valid"),
                    True,
                ),
            ),
            "step_1",
        )

    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "20251001..20260331",
        "20251001..20260331",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
    )
    assert config.test_stage is False
    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=evaluator,
        developer=developer,
        meta_learner=lambda facts: MetaSessionResult(prior="prefer simple daily signals"),
        ledger=ledger,
    )
    folds = build_fold_schedule(
        "20251001..20260331", "20251001..20260331", _days(), window_months=24
    )
    assert [fold.fold_id for fold in folds] == ["fold_20251001..20260331"]
    return pipeline, folds[0], ledger


class RecordingEvaluator:
    """Benchmarked summaries keyed by the revision replayed; records every call."""

    def __init__(self, returns: dict[str, float], *, fail_on: set[str] = frozenset()):
        self.returns = returns
        self.fail_on = set(fail_on)
        self.calls: list[tuple[str, str, str, str]] = []

    def evaluate(self, request):
        revision_id = request.revision.revision_id
        self.calls.append((request.mode, revision_id, request.start, request.end))
        if revision_id in self.fail_on:
            raise TimeoutError(f"replay of {revision_id} exceeded its wall clock")
        return EvaluationResult(
            {
                "total_return": self.returns[revision_id],
                "sharpe": 1.0,
                "max_drawdown": -0.05,
                "benchmark": {"label": "CSI 300", "benchmark_return": 0.02},
            },
            f"result/{request.mode}/{revision_id}",
        )


def _regular_fold_pipeline(tmp_path: Path, evaluator, *, max_steps: int = 1, test_stage: bool = False):
    """Two regular quarterly Folds (or two rolling Folds) driven by a fake developer.

    The developer freezes its own revision on every Fold and echoes the parent
    control it was handed, so the test can see exactly what reached it.
    """
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
    (revision_dir / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    revision = ArtifactRevision("revision_1", revision_dir)
    seen: list[FoldSessionResult | None] = []
    requests: list = []

    def developer(request):
        requests.append(request)
        steps = [
            StepResult(
                f"step_{len(requests)}_{index}",
                revision.revision_id,
                EvaluationResult(
                    {"total_return": 0.05, "max_drawdown": -0.02},
                    "result/valid",
                ),
                selected=index == 0,
            )
            for index in range(max_steps)
        ]
        if request.parent_control is not None:
            steps.insert(
                0,
                StepResult(
                    f"parent_control_{len(requests)}",
                    "revision_parent_copy",
                    request.parent_control,
                    parent_control=True,
                ),
            )
        result = FoldSessionResult("conversation", tuple(steps), steps[-1].step_id)
        seen.append(result)
        return result

    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2025Q4",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        fold_period="quarter",
        test_stage=test_stage,
        max_steps_per_fold=max_steps,
    )
    ledger = ExperimentLedger(config.ledger_path)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=Artifacts(revision, tmp_path / "frozen"),
        evaluator=evaluator,
        developer=developer,
        meta_learner=None,
        ledger=ledger,
    )
    folds = build_fold_schedule(
        config.development_first_period,
        config.development_last_period,
        _days(),
        window_months=24,
        test_stage=test_stage,
    )
    return pipeline, folds, ledger, requests


def test_parent_control_replays_the_inherited_parent_once_per_fold_off_budget(tmp_path: Path):
    """Regular Folds without a Test stage: the host replays the previous Fold's
    frozen strategy on the next Fold's Validation window before the session,
    records it as the Fold's ``parent_control`` and charges no budget."""
    evaluator = RecordingEvaluator({"revision_1": 0.05})
    pipeline, folds, ledger, requests = _regular_fold_pipeline(tmp_path, evaluator, max_steps=1)
    assert [fold.fold_id for fold in folds] == ["fold_2025Q4", "fold_2026Q1"]
    assert all(not fold.has_test for fold in folds)

    first = pipeline.run_fold("epoch_001", folds[0], parent=None)
    assert first.frozen is not None
    # No parent: nothing to control, nothing evaluated by the host.
    assert requests[0].parent_control is None
    assert evaluator.calls == []
    assert ledger.read("fold")[0]["parent_control"] is None

    # A recording evaluator keyed by artifact id: the frozen parent is what
    # the host replays, on exactly the second Fold's Validation window.
    evaluator.returns[first.frozen.artifact_id] = 0.04
    second = pipeline.run_fold("epoch_001", folds[1], parent=first.frozen)
    assert evaluator.calls == [("valid", first.frozen.artifact_id, "20260101", "20260331")]
    control = requests[1].parent_control
    assert control is not None
    assert control.summary["total_return"] == 0.04
    # The developer returned the control step plus max_steps=1 own Step: the
    # control is not charged against the Step budget.
    assert second.fold_status == "frozen"
    record = ledger.read("fold")[1]
    assert record["parent_control"] == {
        "status": "ok",
        "parent_strategy_artifact_id": first.frozen.artifact_id,
        "step_id": "parent_control_2",
        "validation_result": control.summary,
        "validation_result_ref": control.result_ref,
    }
    assert [step["parent_control"] for step in record["steps"]] == [True, False]
    assert record["test_period"] is None and record["test_result"] is None


def test_a_failed_parent_control_is_recorded_and_the_fold_proceeds(tmp_path: Path):
    evaluator = RecordingEvaluator({"revision_1": 0.05})
    pipeline, folds, ledger, requests = _regular_fold_pipeline(tmp_path, evaluator)
    first = pipeline.run_fold("epoch_001", folds[0], parent=None)
    assert first.frozen is not None
    evaluator.fail_on.add(first.frozen.artifact_id)
    second = pipeline.run_fold("epoch_001", folds[1], parent=first.frozen)
    assert second.fold_status == "frozen"
    assert requests[1].parent_control is None
    control = ledger.read("fold")[1]["parent_control"]
    assert control["status"] == "failed"
    assert control["parent_strategy_artifact_id"] == first.frozen.artifact_id
    assert "TimeoutError" in control["error"]


def test_the_agents_own_steps_still_count_against_the_step_budget(tmp_path: Path):
    evaluator = RecordingEvaluator({"revision_1": 0.05})
    pipeline, folds, _ledger, _requests = _regular_fold_pipeline(tmp_path, evaluator, max_steps=1)
    first = pipeline.run_fold("epoch_001", folds[0], parent=None)
    assert first.frozen is not None
    evaluator.returns[first.frozen.artifact_id] = 0.04

    def greedy(request):
        return FoldSessionResult(
            "conversation",
            (
                StepResult("control", "revision_parent_copy", request.parent_control, parent_control=True),
                StepResult("step_a", "revision_1", EvaluationResult({"total_return": 0.05, "max_drawdown": -0.02}, "r")),
                StepResult("step_b", "revision_1", EvaluationResult({"total_return": 0.05, "max_drawdown": -0.02}, "r")),
            ),
            "step_b",
        )

    pipeline.developer = greedy
    with pytest.raises(RuntimeError, match="exceeded the Step budget"):
        pipeline.run_fold("epoch_001", folds[1], parent=first.frozen)


def test_walk_forward_term_reaches_the_held_out_verdict(tmp_path: Path):
    """Graduation term (b) from the final Epoch's parent controls: with one
    transition whose inherited strategy trailed the benchmark, a Held-out that
    passes term (a) is still discarded, and the reason names the counts."""
    from autotrade.pipelines.ledger import experiment_verdict

    evaluator = RecordingEvaluator({"revision_1": 0.05})
    pipeline, folds, ledger, _requests = _regular_fold_pipeline(tmp_path, evaluator)
    first = pipeline.run_fold("epoch_001", folds[0], parent=None)
    assert first.frozen is not None
    evaluator.returns[first.frozen.artifact_id] = 0.01  # below the 0.02 benchmark
    second = pipeline.run_fold("epoch_001", folds[1], parent=first.frozen)
    assert second.frozen is not None
    evaluator.returns[second.frozen.artifact_id] = 0.10  # Held-out itself passes
    pipeline.run_heldout("epoch_001", second.frozen, _days())
    verdict = experiment_verdict(ledger.read())
    assert verdict is not None
    assert verdict["status"] == "discarded"
    assert verdict["reasons"] == ["walkforward_excess_inconsistent(0/1<1)"]
    assert verdict["periods"][0]["walk_forward"] == {
        "status": "inconsistent",
        "source": "parent_control",
        "transitions": 1,
        "positive_excess": 0,
        "required": 1,
    }


def test_a_positive_walk_forward_transition_lets_a_passing_held_out_graduate(tmp_path: Path):
    from autotrade.pipelines.ledger import experiment_verdict

    evaluator = RecordingEvaluator({"revision_1": 0.05})
    pipeline, folds, ledger, _requests = _regular_fold_pipeline(tmp_path, evaluator)
    first = pipeline.run_fold("epoch_001", folds[0], parent=None)
    evaluator.returns[first.frozen.artifact_id] = 0.03  # beats the 0.02 benchmark
    second = pipeline.run_fold("epoch_001", folds[1], parent=first.frozen)
    evaluator.returns[second.frozen.artifact_id] = 0.10
    pipeline.run_heldout("epoch_001", second.frozen, _days())
    verdict = experiment_verdict(ledger.read())
    assert verdict["status"] == "graduated"
    assert verdict["periods"][0]["walk_forward"]["status"] == "consistent"


def test_a_test_stage_schedule_uses_the_frozen_tests_as_walk_forward_evidence(tmp_path: Path):
    """With rolling Folds every Fold already has an out-of-sample frozen Test,
    so term (b) counts those instead of parent controls."""
    from autotrade.pipelines.ledger import experiment_verdict

    class FrozenTestEvaluator(RecordingEvaluator):
        def evaluate(self, request):
            result = super().evaluate(request)
            if request.mode == "frozen_test":
                # The frozen Test trails the benchmark; Held-out itself passes.
                return EvaluationResult({**result.summary, "total_return": 0.01}, result.result_ref)
            return result

    evaluator = FrozenTestEvaluator({"revision_1": 0.05})
    pipeline, folds, ledger, _requests = _regular_fold_pipeline(tmp_path, evaluator, test_stage=True)
    assert [(fold.fold_id, fold.has_test) for fold in folds] == [("fold_2026Q1", True)]
    outcome = pipeline.run_fold("epoch_001", folds[0], parent=None)
    assert outcome.frozen is not None
    evaluator.returns[outcome.frozen.artifact_id] = 0.10
    pipeline.run_heldout("epoch_001", outcome.frozen, _days())
    verdict = experiment_verdict(ledger.read())
    assert verdict["status"] == "discarded"
    assert verdict["reasons"] == ["walkforward_excess_inconsistent(0/1<1)"]
    block = verdict["periods"][0]["walk_forward"]
    assert block["source"] == "frozen_test"
    assert (block["transitions"], block["positive_excess"], block["required"]) == (1, 0, 1)


def test_single_window_fold_has_no_frozen_test_and_held_out_graduates(tmp_path: Path):
    from autotrade.pipelines.ledger import experiment_verdict

    evaluator = BenchmarkedEvaluator(0.08, 1.2, -0.05, 0.03)
    pipeline, fold, ledger = _single_window_pipeline(tmp_path, evaluator)
    assert not fold.has_test
    assert fold.validation_start == "20251001" and fold.validation_end == "20260331"
    assert experiment_verdict(ledger.read()) is None
    outcome = pipeline.run_fold("epoch_001", fold, parent=None)
    assert outcome.fold_status == "frozen"
    assert outcome.frozen is not None
    # No Test stage: nothing was evaluated or recorded for a test region.
    assert outcome.test_summary is None
    record = ledger.read("fold")[0]
    assert record["test_period"] is None
    assert record["test_decision_time"] is None
    assert record["test_result"] is None
    assert record["test_result_ref"] is None
    assert record["snapshot_ids"] == {
        "valid_decision_input": "valid_20251001_20260331",
        "test_decision_input": None,
    }
    assert pipeline.run_heldout("epoch_001", outcome.frozen, _days()) == 1
    heldout = ledger.read("heldout")[0]
    assert heldout["verdict"] == {
        "status": "graduated",
        "reasons": [],
        "excess_return": pytest.approx(0.05),
        "sharpe": 1.2,
        "max_drawdown": -0.05,
        "max_drawdown_limit": 0.25,
        # A single development Fold has no walk-forward transition: term (b)
        # is not applicable and Held-out alone decides.
        "walk_forward": {"status": "not_applicable", "transitions": 0},
    }
    verdict = experiment_verdict(ledger.read())
    assert verdict is not None
    assert verdict["status"] == "graduated"
    assert verdict["reasons"] == []
    assert verdict["periods"][0]["period"] == "2026Q2"


def test_a_frozen_test_never_runs_without_a_test_stage(tmp_path: Path):
    """The evaluator must never be asked for a frozen_test replay of a
    single-window Fold, and the ledger must not carry a placeholder."""
    pipeline, fold, ledger = _single_window_pipeline(
        tmp_path, ModeMutatingEvaluator("frozen_test", "output")
    )
    outcome = pipeline.run_fold("epoch_001", fold, parent=None)
    assert outcome.fold_status == "frozen"
    assert ledger.read("fold")[0]["test_result"] is None
    assert "state_changed_during_test" not in ledger.read("fold")[0]


@pytest.mark.parametrize(
    ("summary", "reasons"),
    (
        ((0.02, 0.5, -0.10, 0.03), ["excess_return_not_positive"]),
        ((0.08, 0.0, -0.10, 0.03), ["sharpe_not_positive"]),
        ((0.08, 0.9, -0.30, 0.03), ["max_drawdown_exceeded"]),
        (
            (-0.02, -0.4, -0.40, 0.03),
            ["excess_return_not_positive", "sharpe_not_positive", "max_drawdown_exceeded"],
        ),
    ),
)
def test_held_out_discards_with_every_failing_reason(tmp_path: Path, summary, reasons):
    from autotrade.pipelines.ledger import experiment_verdict

    pipeline, fold, ledger = _single_window_pipeline(tmp_path, BenchmarkedEvaluator(*summary))
    frozen = pipeline.run_fold("epoch_001", fold, parent=None).frozen
    assert frozen is not None
    pipeline.run_heldout("epoch_001", frozen, _days())
    verdict = experiment_verdict(ledger.read())
    assert verdict is not None
    assert verdict["status"] == "discarded"
    assert verdict["reasons"] == reasons


def test_held_out_without_a_benchmark_block_cannot_graduate():
    from autotrade.pipelines.config import AcceptanceRules

    rules = AcceptanceRules()
    verdict = rules.heldout_verdict({"total_return": 0.2, "sharpe": 2.0, "max_drawdown": -0.01})
    assert verdict["status"] == "discarded"
    assert verdict["reasons"] == ["missing_benchmark_return"]
    assert verdict["excess_return"] is None
    failed = rules.heldout_verdict({"status": "failed", "error": "boom"})
    assert failed["status"] == "discarded"
    assert failed["reasons"][0] == "heldout_failed"
    nan = rules.heldout_verdict(
        {"total_return": float("nan"), "sharpe": 1.0, "max_drawdown": -0.01,
         "benchmark": {"benchmark_return": 0.0}}
    )
    assert "missing_total_return" in nan["reasons"]


def _killed_marker(fold_id: str, run_id: str) -> dict[str, object]:
    """What a SIGKILLed fold run leaves behind: a marker and no ledger row."""
    return {
        "experiment_id": "experiment_a",
        "epoch_id": "epoch_001",
        "fold_id": fold_id,
        "run_id": run_id,
        "session_key": fold_session_key("epoch_001", fold_id),
        "phase": "fold",
    }


def test_a_recorded_fold_run_leaves_nothing_for_interrupted_run_recovery(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    pipeline.run_fold("epoch_001", fold, parent=None)
    markers = RunMarkers(pipeline.config.experiment_dir)
    assert sorted(markers.root.glob("*.json")) == []
    assert markers.recover(ledger) == []
    assert not any(row["record_type"] == "attempt_failed" for row in ledger.read())


def test_a_failing_fold_run_records_attempt_failed_and_clears_its_marker(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())

    def crashing_developer(_request):
        raise RuntimeError("session crashed")

    pipeline.developer = crashing_developer
    with pytest.raises(RuntimeError, match="session crashed"):
        pipeline.run_fold("epoch_001", fold, parent=None)
    failed = [row for row in ledger.read() if row["record_type"] == "attempt_failed"]
    assert len(failed) == 1
    assert failed[0]["phase"] == "fold"
    assert failed[0]["session_key"] == fold_session_key("epoch_001", fold.fold_id)
    assert failed[0]["error"] == "RuntimeError: session crashed"
    markers = RunMarkers(pipeline.config.experiment_dir)
    assert sorted(markers.root.glob("*.json")) == []
    # The in-process record is the only one; recovery must not add a second.
    assert markers.recover(ledger) == []


def test_a_failing_heldout_run_records_attempt_failed_and_clears_its_marker(
    tmp_path: Path,
):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    frozen = pipeline.run_fold("epoch_001", fold, parent=None).frozen
    assert frozen is not None
    markers = RunMarkers(pipeline.config.experiment_dir)
    in_flight: list[list[str]] = []

    class CrashingHeldout:
        def evaluate(self, request):
            in_flight.append(sorted(path.name for path in markers.root.glob("*.json")))
            raise RuntimeError("held-out replay crashed")

    pipeline.evaluator = CrashingHeldout()
    with pytest.raises(RuntimeError, match="held-out replay crashed"):
        pipeline.run_heldout("epoch_001", frozen, _days())

    # The marker is on disk while the replay runs: a SIGKILL here is still
    # recoverable evidence, exactly as for a Fold or Meta run.
    assert len(in_flight) == 1 and len(in_flight[0]) == 1
    failed = [row for row in ledger.read() if row["record_type"] == "attempt_failed"]
    assert len(failed) == 1
    assert failed[0]["phase"] == "heldout"
    assert failed[0]["session_key"] == "heldout"
    assert failed[0]["error"] == "RuntimeError: held-out replay crashed"
    # No held-out result was invented, and the period stays unfinished.
    assert ledger.read("heldout") == []
    assert sorted(markers.root.glob("*.json")) == []
    # The in-process record is the only one; recovery must not add a second.
    assert markers.recover(ledger) == []


def test_a_killed_run_is_recorded_as_attempt_failed_once_at_the_next_start(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    markers = RunMarkers(pipeline.config.experiment_dir)
    markers.begin(_killed_marker(fold.fold_id, "run_killed"))

    appended = markers.recover(ledger)

    assert [row["run_id"] for row in appended] == ["run_killed"]
    records = ledger.read()
    failed = [row for row in records if row["record_type"] == "attempt_failed"]
    assert len(failed) == 1
    assert failed[0]["fold_id"] == fold.fold_id
    assert failed[0]["session_key"] == fold_session_key("epoch_001", fold.fold_id)
    assert failed[0]["phase"] == "fold"
    assert failed[0]["error"] == INTERRUPTED_RUN_ERROR
    assert failed[0]["started_at"]
    # Restarting the worker again must not duplicate the evidence.
    assert markers.recover(ledger) == []
    assert ledger.read() == records
    # An attempt_failed row is not work: it neither completes nor blocks a rerun.
    assert latest_fold_records(records) == {}


def test_recovery_ignores_a_marker_whose_run_already_reached_the_ledger(tmp_path: Path):
    pipeline, fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    outcome = pipeline.run_fold("epoch_001", fold, parent=None)
    markers = RunMarkers(pipeline.config.experiment_dir)
    # A kill between the ledger append and the marker cleanup.
    markers.begin(_killed_marker(fold.fold_id, outcome.run_id))

    assert markers.recover(ledger) == []

    assert not any(row["record_type"] == "attempt_failed" for row in ledger.read())
    assert sorted(markers.root.glob("*.json")) == []


def test_an_unreadable_run_marker_becomes_one_attempt_failed_and_is_cleared(
    tmp_path: Path,
):
    """A torn marker is evidence, not a brick.

    ``write_json_atomic`` renames without fsync, so the very events the marker
    exists to survive can leave it zero-length or truncated. Such a marker must
    still become exactly one ``attempt_failed`` and then be gone; if it stayed,
    every later resume would fail on the same file.
    """
    pipeline, _fold, ledger = _pipeline_with_evaluator(tmp_path, Evaluator())
    markers = RunMarkers(pipeline.config.experiment_dir)
    markers.root.mkdir(parents=True, exist_ok=True)
    (markers.root / "run_empty.json").write_text("", encoding="utf-8")
    (markers.root / "run_truncated.json").write_text(
        '{"experiment_id": "experiment_a", "epoch_id": "epoch_0',
        encoding="utf-8",
    )

    appended = markers.recover(ledger)

    assert [row["run_id"] for row in appended] == ["run_empty", "run_truncated"]
    failed = [row for row in ledger.read() if row["record_type"] == "attempt_failed"]
    assert len(failed) == 2
    for row, marker_name in zip(failed, ("run_empty.json", "run_truncated.json")):
        # Well-formed for every ledger reader: the link keys the marker could
        # not supply are named unknown, never invented.
        assert row["experiment_id"] == "experiment_a"
        assert row["epoch_id"] == UNKNOWN_MARKER_LINK_KEY
        assert row["fold_id"] == UNKNOWN_MARKER_LINK_KEY
        assert row["error"].startswith(UNREADABLE_RUN_MARKER_ERROR)
        assert marker_name in row["error"]
    assert sorted(markers.root.glob("*.json")) == []
    assert latest_fold_records(ledger.read()) == {}
    # Restarting the worker again must not duplicate the evidence.
    records = ledger.read()
    assert markers.recover(ledger) == []
    assert ledger.read() == records


def test_a_run_marker_without_link_keys_is_refused(tmp_path: Path):
    markers = RunMarkers(tmp_path / "experiments" / "experiment_a")
    with pytest.raises(ValueError, match="run marker missing link keys"):
        markers.begin({"experiment_id": "experiment_a", "run_id": "run_1"})
    assert not markers.root.exists()
