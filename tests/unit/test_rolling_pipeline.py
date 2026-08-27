from __future__ import annotations

import json
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
    DEFAULT_DEADLINE_GRACE_MINUTES,
    MetaSessionResult,
    fold_session_deadline_seconds,
)
from autotrade.pipelines.experiment import _session_budgets
from autotrade.pipelines.folds import build_fold_schedule
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    FrozenArtifactMutated,
    FrozenArtifactRestoreFailed,
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
    fold = build_fold_schedule("2026Q1", "2026Q1", days)[0]
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
        "2026Q1",
        "2026Q1",
        "2026Q2",
        "2026Q2",
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
    fold = build_fold_schedule("2026Q1", "2026Q1", _days())[0]
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
            epochs=1,
            deadline_grace_minutes=minutes,
        )
        budgets = _session_budgets(config, None)
        assert budgets["deadline_grace_seconds"] == minutes * 60
        assert budgets["deadline_seconds"] == fold_session_deadline_seconds(
            240, minutes
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
        assert request.deadline_seconds == fold_session_deadline_seconds(240, minutes)


def test_prompt_preview_deadline_matches_live_fold_budget(tmp_path: Path):
    from autotrade.webui.prompt_preview import build_prompt_preview

    cases = (
        ({}, fold_session_deadline_seconds(240, DEFAULT_DEADLINE_GRACE_MINUTES)),
        (
            {"deadline_grace_minutes": 0, "max_fold_minutes": 240},
            fold_session_deadline_seconds(240, DEFAULT_DEADLINE_GRACE_MINUTES),
        ),
        (
            {"deadline_grace_minutes": 20, "max_fold_minutes": 90},
            fold_session_deadline_seconds(90, DEFAULT_DEADLINE_GRACE_MINUTES),
        ),
    )
    for index, (extra, expected) in enumerate(cases):
        experiment = tmp_path / f"preview_{index}"
        hitl = experiment / "hitl"
        hitl.mkdir(parents=True)
        (hitl / "params.json").write_text(
            json.dumps({"strategy_period": "day", **extra}),
            encoding="utf-8",
        )
        (hitl / "schedule.json").write_text(
            json.dumps(
                {
                    "sessions": [
                        {
                            "session_key": "epoch_001/fold_x",
                            "kind": "fold",
                            "epoch_id": "epoch_001",
                            "fold_id": "fold_x",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        preview = build_prompt_preview(experiment, "epoch_001/fold_x", "")
        assert f'"deadline_seconds": {int(expected)}' in str(preview["prompt"])


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
