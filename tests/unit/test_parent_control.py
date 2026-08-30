"""The host's parent control: the inherited parent replayed unchanged on the
new Fold's Validation window before the Agent starts.

At the session level it must land in the step tree as an in-session node the
Agent can keep as the explicit keep-parent, charge no Validation slot and no
Step, and reach the run manifest; at the ledger level the final Epoch's
transitions must count exactly the way graduation term (b) reads them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
    ModificationConstraints,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import chmod_tree, write_json_atomic
from autotrade.environment.step_tree import StepTree, node_in_session
from autotrade.environment.time_budget import InferenceTimeBudget
from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.finish_fold import FinishFoldTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.pipelines.config import (
    BrokerProfile,
    EvaluationResult,
    FoldSessionRequest,
    FoldSpec,
    SnapshotBundle,
    StrategySchedule,
)
from autotrade.pipelines.ledger import walk_forward_transitions
from autotrade.pipelines.local_backend import (
    PARENT_CONTROL_RESULT_NAME,
    FoldBacktestTool,
    record_parent_control,
)

PARENT_SOURCE = "def generate_orders(context):\n    return []\n"


@pytest.fixture(autouse=True)
def _writable_tmp(tmp_path: Path):
    """Revisions and step nodes are published read-only; hand the temporary
    tree back writable so pytest's own basetemp housekeeping can remove it."""
    yield
    chmod_tree(tmp_path, file_mode=0o644, dir_mode=0o755)


CHALLENGER_SOURCE = "def generate_orders(context):\n    threshold = 0.02\n    return []\n"


class _Evaluator:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def evaluate(self, request, max_days=None):
        del max_days
        self.calls += 1
        summary = {"total_return": 0.03, "sharpe": 0.9, "max_drawdown": 0.04}
        target = self.root / f"valid_{self.calls:03d}" / "result.json"
        write_json_atomic(target, {"stats": summary})
        return EvaluationResult(summary=dict(summary), result_ref=str(target))


class _Manifest:
    def __init__(self) -> None:
        self.summaries: list[dict[str, object]] = []

    def append_backtest_summary(self, summary: dict[str, object]) -> None:
        self.summaries.append(dict(summary))


def _session(root: Path, *, max_backtests: int = 2, max_steps: int = 2):
    workspace = root / "workspace"
    output, models, parent = workspace / "output", workspace / "models", root / "parent"
    for directory in (output, models, parent):
        directory.mkdir(parents=True)
    (parent / "main.py").write_text(PARENT_SOURCE, encoding="utf-8")
    (output / "main.py").write_text(PARENT_SOURCE, encoding="utf-8")
    moment = datetime(2022, 12, 30, 23, 59, 59, tzinfo=UTC)
    request = FoldSessionRequest(
        experiment_id="exp",
        epoch_id="epoch_001",
        fold=FoldSpec(
            fold_id="fold_2023",
            input_window_start="20210101",
            input_window_end="20221231",
            validation_start="20230101",
            validation_end="20231231",
            valid_decision_time=moment,
        ),
        run_id="run_control",
        parent=None,
        snapshot=SnapshotBundle("snapshot", "decision", "replay"),
        max_steps=max_steps,
        max_backtests=max_backtests,
        max_llm_calls=10,
        deadline_seconds=600.0,
    )
    tree = StepTree(root / "steps")
    ref_store = AgentRefStore(root / "experiment")
    manifest = _Manifest()
    backtest = FoldBacktestTool(
        request=request,
        output_dir=output,
        models_dir=models,
        modification_check=ModificationCheckTool(
            output, parent_dir=parent, models_dir=models, constraints=ModificationConstraints()
        ),
        artifact_store=FilesystemArtifactStore(root / "revisions"),
        evaluator=_Evaluator(root / "results"),
        tree=tree,
        schedule=StrategySchedule(),
        broker_profile=BrokerProfile(),
        time_budget=InferenceTimeBudget(duration_seconds=600.0),
        ref_store=ref_store,
        manifest=manifest,  # type: ignore[arg-type]
    )
    finish = FinishFoldTool(
        tree,
        fold_id=ref_store.get_or_create("fold", "fold_2023"),
        run_id=ref_store.get_or_create("run", "run_control"),
        parent_main_py=parent / "main.py",
        current_output=output,
        current_models=models,
    )
    return backtest, tree, finish, manifest, output


def _control_result(root: Path) -> EvaluationResult:
    target = root / "host_results" / "parent_control" / "result.json"
    summary = {"total_return": 0.02, "sharpe": 0.5, "max_drawdown": 0.03}
    write_json_atomic(target, {"stats": summary})
    return EvaluationResult(summary=summary, result_ref=str(target))


def test_the_control_is_an_in_session_step_node_that_charges_no_budget(tmp_path: Path):
    backtest, tree, _finish, manifest, output = _session(tmp_path)
    step = record_parent_control(
        backtest, _control_result(tmp_path), output_dir=output, models_dir=output.parent / "models"
    )
    assert step.parent_control is True
    node = tree.get_node(step.step_id)
    assert node["result_name"] == PARENT_CONTROL_RESULT_NAME
    assert node["complete_validation"] is True
    assert node["metadata"]["parent_control"] is True
    assert node_in_session(
        node,
        fold_id=backtest.ref_store.get_or_create("fold", "fold_2023"),
        run_id=backtest.ref_store.get_or_create("run", "run_control"),
    )
    # The node snapshot is the parent byte for byte, with its Validation attached.
    assert (tree.node_output_dir(step.step_id) / "main.py").read_text(encoding="utf-8") == PARENT_SOURCE
    assert any("validation" in name for name in node["attachments"])
    assert tree.current_node_id == step.step_id
    # No Validation slot and no Step spent; the manifest carries the summary.
    assert backtest.backtests == 0
    assert backtest.steps == []
    assert manifest.summaries[-1]["result_name"] == PARENT_CONTROL_RESULT_NAME
    assert manifest.summaries[-1]["parent_control"] is True
    assert manifest.summaries[-1]["total_return"] == 0.02
    # The Agent still has its whole budget: two full Validations fit.
    assert backtest.reserve_validations(2) == ["valid_001", "valid_002"]


def test_the_control_node_is_the_explicit_keep_parent_after_a_different_hypothesis(tmp_path: Path):
    backtest, tree, finish, _manifest, output = _session(tmp_path)
    control = record_parent_control(
        backtest, _control_result(tmp_path), output_dir=output, models_dir=output.parent / "models"
    )
    # Keep-parent first is still refused: no different hypothesis was tried.
    with pytest.raises(ToolError, match="differs from the parent"):
        finish.invoke({"node_id": control.step_id})
    (output / "main.py").write_text(CHALLENGER_SOURCE, encoding="utf-8")
    challenger = backtest.invoke({})
    assert challenger.ok
    assert backtest.backtests == 1 and len(backtest.steps) == 1
    # Now the control node is selectable as the explicit keep-parent once the
    # working copy matches it again, without a second backtest of the parent.
    (output / "main.py").write_text(PARENT_SOURCE, encoding="utf-8")
    result = finish.invoke({"node_id": control.step_id})
    assert result.finish is True
    assert result.value["node_id"] == control.step_id
    assert backtest.backtests == 1
    assert tree.current_node_id == control.step_id


def _fold(epoch_id, fold_id, period, *, control=None, test=None):
    return {
        "record_type": "fold",
        "experiment_id": "e",
        "epoch_id": epoch_id,
        "fold_id": fold_id,
        "run_id": f"run_{epoch_id}_{fold_id}",
        "fold_status": "frozen",
        "validation_period": period,
        "parent_control": control,
        "test_result": test,
    }


def _ok(total_return: float, benchmark: float = 0.02):
    return {
        "status": "ok",
        "validation_result": {
            "total_return": total_return,
            "benchmark": {"benchmark_return": benchmark},
        },
    }


def test_walk_forward_transitions_count_the_final_epochs_parent_controls():
    records = [
        # Epoch 1 is not the final Epoch: ignored entirely.
        _fold("epoch_001", "fold_2022", "20220101..20221231"),
        _fold("epoch_001", "fold_2023", "20230101..20231231", control=_ok(0.10)),
        # Final Epoch: the first Fold's control (previous Epoch's frontier) is
        # not a transition; the other three are, whatever their outcome.
        _fold("epoch_002", "fold_2022", "20220101..20221231", control=_ok(0.10)),
        _fold("epoch_002", "fold_2023", "20230101..20231231", control=_ok(0.05)),
        _fold("epoch_002", "fold_2024", "20240101..20241231", control=_ok(0.01)),
        _fold("epoch_002", "fold_2025", "20250101..20251231", control={"status": "failed", "error": "x"}),
    ]
    assert walk_forward_transitions(records, epoch_id="epoch_002", test_stage=False) == {
        "source": "parent_control",
        "epoch_id": "epoch_002",
        "transitions": 3,
        "positive_excess": 1,
    }
    # A rerun appends a superseding record; only the latest counts.
    records.append(_fold("epoch_002", "fold_2025", "20250101..20251231", control=_ok(0.09)))
    assert walk_forward_transitions(records, epoch_id="epoch_002", test_stage=False)["positive_excess"] == 2
    # One Fold: no transition at all.
    single = [_fold("epoch_001", "fold_20220101..20251231", "20220101..20251231")]
    assert walk_forward_transitions(single, epoch_id="epoch_001", test_stage=False)["transitions"] == 0


def test_walk_forward_transitions_use_frozen_tests_with_a_test_stage():
    records = [
        _fold("epoch_001", "fold_2023", "20220101..20221231", test={"total_return": 0.05, "benchmark": {"benchmark_return": 0.02}}),
        _fold("epoch_001", "fold_2024", "20230101..20231231", test={"total_return": 0.01, "benchmark": {"benchmark_return": 0.02}}),
        _fold("epoch_001", "fold_2025", "20240101..20241231", test={"status": "failed", "error": "boom"}),
    ]
    assert walk_forward_transitions(records, epoch_id="epoch_001", test_stage=True) == {
        "source": "frozen_test",
        "epoch_id": "epoch_001",
        "transitions": 3,
        "positive_excess": 1,
    }
    # Without a benchmark block an excess return cannot be shown at all.
    records[0]["test_result"] = {"total_return": 0.05}
    assert walk_forward_transitions(records, epoch_id="epoch_001", test_stage=True)["positive_excess"] == 0
    assert json.dumps(walk_forward_transitions(records, epoch_id="epoch_001", test_stage=True))
