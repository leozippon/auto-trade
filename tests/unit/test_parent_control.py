"""The host's parent control: the inherited parent replayed unchanged on the
new Fold's Validation window before the Agent starts.

At the session level it must land in the step tree as an in-session node the
Agent can keep as the explicit keep-parent, charge no Validation slot and no
Step, and reach the run manifest projected exactly like the Fold's own
candidates -- that manifest row is what the next Meta session reads the parent
baseline from; at the ledger level the final Epoch's transitions must count
exactly the way graduation term (b) reads them, scored on the Fold's new period
alone once the Validation window trails over several.
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
from autotrade.pipelines.agent_views import compact_fold_history
from autotrade.pipelines.config import (
    BrokerProfile,
    EvaluationResult,
    FoldSessionRequest,
    FoldSpec,
    SnapshotBundle,
    StrategySchedule,
)
from autotrade.pipelines.experiment import _step_result
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
        summary = {
            "total_return": 0.03,
            "sharpe": 0.9,
            "max_drawdown": 0.04,
            # One structured block that belongs in the manifest row and one
            # replay-scaled block that must stay in the referenced result.json.
            "benchmark": {"beta": 0.664, "neutralized_excess_return": -0.1815},
            "per_stock": {"000001.SZ": [0.1] * 8},
        }
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
    summary = {
        "total_return": 0.02,
        "sharpe": 0.5,
        "max_drawdown": 0.03,
        "benchmark": {"beta": 0.645, "neutralized_excess_return": -0.2975},
        "per_stock": {"000001.SZ": [0.2] * 8},
    }
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


def test_the_control_row_is_projected_into_the_manifest_like_any_candidate(tmp_path: Path):
    """The control contributes the same summary projection as a candidate.

    ``compact_fold_history`` reads these rows into the next Meta session's Fold
    history, where the control sits beside the Fold's own Validations. A control
    row that drops the ``benchmark`` block every sibling carries leaves the
    parent baseline with no neutralized-excess anchor, and a reader then takes a
    neighbour's block for it.
    """
    backtest, _tree, _finish, manifest, output = _session(tmp_path)
    record_parent_control(
        backtest, _control_result(tmp_path), output_dir=output, models_dir=output.parent / "models"
    )
    (output / "main.py").write_text(CHALLENGER_SOURCE, encoding="utf-8")
    assert backtest.invoke({}).ok
    control_row, candidate_row = manifest.summaries
    assert control_row["result_name"] == PARENT_CONTROL_RESULT_NAME
    # Its own block, never the candidate's, and the same fields as a candidate.
    assert control_row["benchmark"] == {"beta": 0.645, "neutralized_excess_return": -0.2975}
    assert candidate_row["benchmark"] == {"beta": 0.664, "neutralized_excess_return": -0.1815}
    assert set(control_row) - {"parent_control"} == set(candidate_row)
    # Replay-scaled series stay in the referenced result.json for both.
    assert "per_stock" not in control_row and "per_stock" not in candidate_row


def test_meta_fold_history_never_borrows_another_nodes_benchmark_block(tmp_path: Path):
    """Every Meta-visible row is built from that row's own manifest entry.

    A run whose control row has no ``benchmark`` must stay without one: filling
    it from the Fold's selected node would hand Meta the challenger's
    neutralized excess as the parent's baseline.
    """
    manifest_ref = tmp_path / "artifacts" / "run_1" / "run_manifest.json"
    selected = {"beta": 0.664, "neutralized_excess_return": -0.1815}
    write_json_atomic(
        manifest_ref,
        {
            "backtest_summaries": [
                {
                    "result_name": PARENT_CONTROL_RESULT_NAME,
                    "mode": "valid",
                    "status": "ok",
                    "complete_validation": True,
                    "total_return": -0.2315,
                },
                {
                    "result_name": "valid_001",
                    "mode": "valid",
                    "status": "ok",
                    "complete_validation": True,
                    "total_return": -0.1317,
                    "benchmark": selected,
                },
            ]
        },
    )
    compact = compact_fold_history(
        {
            "record_type": "fold",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2024",
            "fold_status": "frozen",
            "run_manifest_ref": str(manifest_ref),
            "validation_period": "20240101..20241231",
            "validation_result": {"total_return": -0.1317, "benchmark": selected},
        },
        ref_store=AgentRefStore(tmp_path / "experiment"),
    )
    control_row, candidate_row = compact["backtest_summaries"]
    assert control_row["result_name"] == PARENT_CONTROL_RESULT_NAME
    assert control_row["total_return"] == -0.2315
    assert "benchmark" not in control_row
    assert candidate_row["benchmark"] == selected
    # The Fold-level ``validation_result`` is the selected node's own block too
    # (``agent_visible_metrics`` keeps its own narrower field whitelist).
    assert compact["validation_result"]["benchmark"]["beta"] == 0.664


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


def _sub_window(label, start, end, *, ret, benchmark, turnover=5.0):
    return {
        "kind": "quarter",
        "label": label,
        "start": start,
        "end": end,
        "trade_days": 60,
        "partial": False,
        "return": ret,
        "benchmark_return": benchmark,
        "excess_return": ret - benchmark,
        "sharpe": 0.5,
        "max_drawdown": 0.03,
        "turnover": turnover,
        "trade_count": 7,
    }


_TRAILING_SUMMARY = {
    "total_return": 0.20,
    "turnover": 20.0,
    "benchmark": {"benchmark_return": 0.05, "excess_return": 0.15},
    "sub_windows": [
        _sub_window("2022Q3", "20220701", "20220930", ret=0.10, benchmark=0.01),
        _sub_window("2022Q4", "20221010", "20221230", ret=0.06, benchmark=0.01),
        _sub_window("2023Q1", "20230103", "20230331", ret=0.05, benchmark=0.01),
        _sub_window("2023Q2", "20230403", "20230630", ret=0.02, benchmark=0.01),
    ],
}


def _rolling_fold(*, step: bool):
    """One Fold of a trailing four-quarter window, or the single-period default."""
    return FoldSpec(
        fold_id="fold_2023Q2",
        input_window_start="20200701",
        input_window_end="20220630",
        validation_start="20220701" if step else "20230401",
        validation_end="20230630",
        valid_decision_time=datetime(2022, 6, 30, 23, 59, 59, tzinfo=UTC),
        step_start="20230401" if step else None,
        step_end="20230630" if step else None,
    )


def test_the_step_result_is_the_new_periods_sub_window_priced_like_a_result():
    # The parent was developed on the first three quarters of this window; only
    # 2023Q2 is new, and it is already in the control's own sub-window table.
    step = _step_result(_TRAILING_SUMMARY, _rolling_fold(step=True), 5.0)
    assert step["label"] == "2023Q2"
    assert (step["start"], step["end"]) == ("20230403", "20230630")
    # Result-shaped, so the ledger and the report read it like any other result.
    assert step["total_return"] == 0.02
    assert step["benchmark"] == {"benchmark_return": 0.01, "excess_return": 0.01}
    assert step["trade_count"] == 7
    # Priced on the step's own turnover, not the whole window's.
    cost = step["cost_sensitivity"]
    assert cost["cost_per_bp_per_side"] == pytest.approx(0.0005)
    assert cost["excess_at_2x_slippage"] == pytest.approx(0.01 - 5.0 * 0.0005)
    assert json.dumps(step)


def test_a_single_period_fold_has_no_separate_step():
    # Nothing to project: the whole window is the transition, exactly as before.
    assert _step_result(_TRAILING_SUMMARY, _rolling_fold(step=False), 5.0) is None
    # A window whose sub-window table does not cover the step is not invented.
    narrowed = {**_TRAILING_SUMMARY, "sub_windows": _TRAILING_SUMMARY["sub_windows"][:3]}
    assert _step_result(narrowed, _rolling_fold(step=True), 5.0) is None


def test_a_transition_is_graded_on_the_step_when_the_window_carries_one():
    # The four-quarter window beat the benchmark, its new quarter did not: the
    # transition must count the quarter, which is the only out-of-sample part.
    control = {
        "status": "ok",
        "parent_strategy_artifact_id": "strategy_a",
        "validation_result": {
            "total_return": 0.20,
            "benchmark": {"benchmark_return": 0.05},
        },
        "step_result": {
            "label": "2023Q2",
            "total_return": 0.02,
            "benchmark": {"benchmark_return": 0.04},
        },
    }
    records = [
        _fold("epoch_001", "fold_2023Q1", "20220401..20230331"),
        _fold("epoch_001", "fold_2023Q2", "20220701..20230630", control=control),
    ]
    counted = walk_forward_transitions(records, epoch_id="epoch_001", test_stage=False)
    assert counted == {
        "source": "parent_control",
        "epoch_id": "epoch_001",
        "transitions": 1,
        "positive_excess": 0,
    }
    # An older ledger has no step_result: the whole window stays the transition.
    del control["step_result"]
    assert (
        walk_forward_transitions(records, epoch_id="epoch_001", test_stage=False)[
            "positive_excess"
        ]
        == 1
    )
