from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from autotrade.pipelines import (
    ArtifactRevision,
    EvaluationResult,
    FoldSessionResult,
    FrozenArtifact,
    RollingExperimentConfig,
    RollingExperimentPipeline,
    StepResult,
)
from autotrade.pipelines.config import MetaSessionResult
from autotrade.pipelines.folds import build_fold_schedule
from autotrade.pipelines.ledger import ExperimentLedger
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
        return FrozenArtifact(
            values["artifact_id"], target, None, values["run_id"], values["fold_id"], values["step_id"]
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
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
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
    result = pipeline.run(days)
    assert result["heldout_runs"] == 1
    records = ledger.read()
    assert [record["record_type"] for record in records] == ["meta_learning", "fold", "heldout"]
    assert result["final_strategy_artifact"].startswith("strategy_")
    heldout = records[-1]
    assert heldout["result"]["total_return"] == 0.02
    assert heldout["strategy_artifact_id"] == result["final_strategy_artifact"]


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
    fold = build_fold_schedule("2026Q1", "2026Q1", days)[0]
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
        epochs=1,
    )
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
    visible_fold = build_fold_schedule("2026Q1", "2026Q1", days)[0]

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
        "review_window",
        "meta_learning",
    }
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
    return pipeline, build_fold_schedule("2026Q1", "2026Q1", days)[0]


def test_first_meta_session_has_empty_review_window(tmp_path: Path):
    config = RollingExperimentConfig(
        "experiment_a",
        tmp_path / "experiments",
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
        epochs=1,
    )
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
        epochs=1,
    )
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


def _pipeline_capturing_fold_requests(tmp_path: Path, captured: list):
    revision_dir = tmp_path / "revision"
    revision_dir.mkdir()
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
        "experiment_a", tmp_path / "experiments", "2026Q1", "2026Q1", "2026Q2", "2026Q2", epochs=1
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
    return pipeline, build_fold_schedule("2026Q1", "2026Q1", days)[0]


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
