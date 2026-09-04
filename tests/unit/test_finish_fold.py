from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.base import ToolError, ToolRegistry
from autotrade.environment.tools.finish_fold import (
    EARLY_STOP_REASON_MAX_CHARS,
    FinishFoldTool,
    FoldBudgetStatus,
    executable_source_structure,
)

PARENT = "def generate_orders(context):\n    return []\n"
COMMENT_ONLY = "def generate_orders(context):\n    # try a new idea\n    return []\n"
DOCSTRING_ONLY = (
    '"""module copy"""\n'
    "def generate_orders(context):\n"
    '    """still parent"""\n'
    "    return []\n"
)
LOGIC = "def generate_orders(context):\n    threshold = 0.02\n    return []\n"
CLASS_PARENT = (
    "class Signal:\n"
    "    def score(self, row):\n"
    "        return 0.0\n"
    "\n"
    "def generate_orders(context):\n"
    "    return []\n"
)
CLASS_DOCSTRING_ONLY = (
    "class Signal:\n"
    '    """still parent"""\n'
    "    def score(self, row):\n"
    "        return 0.0\n"
    "\n"
    "def generate_orders(context):\n"
    "    return []\n"
)


def test_structure_ignores_comments_docstrings_and_whitespace():
    parent = executable_source_structure(PARENT)
    assert executable_source_structure(COMMENT_ONLY) == parent
    assert executable_source_structure(DOCSTRING_ONLY) == parent
    assert executable_source_structure("def generate_orders(context):\n\n    return []\n") == parent
    assert executable_source_structure(LOGIC) != parent
    # A class docstring is documentation like the other two: adding one must
    # not let a parent-identical package pass the different-hypothesis gate.
    assert executable_source_structure(CLASS_DOCSTRING_ONLY) == (
        executable_source_structure(CLASS_PARENT)
    )


def _record(
    tree: StepTree,
    output: Path,
    *,
    source: str,
    result_name: str,
    fold_id: str = "fold_ref_ab",
    run_id: str = "run_x",
) -> str:
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text(source, encoding="utf-8")
    return tree.record_step(
        output,
        epoch_id="epoch_001",
        fold_id=fold_id,
        run_id=run_id,
        result_name=result_name,
        revision_id=new_revision_id("revision"),
        metrics={},
    )


def _tool(tree: StepTree, parent_main: Path) -> FinishFoldTool:
    return FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        parent_main_py=parent_main,
    )


def test_finish_fold_fails_when_all_validations_match_parent(tmp_path: Path):
    parent_main = tmp_path / "parent" / "main.py"
    parent_main.parent.mkdir()
    parent_main.write_text(PARENT, encoding="utf-8")
    tree = StepTree(tmp_path / "steps")
    clone = _record(tree, tmp_path / "clone", source=COMMENT_ONLY, result_name="valid_000")
    finish = _tool(tree, parent_main)
    with pytest.raises(ToolError, match="differs from the parent"):
        finish.invoke({"node_id": clone})


def test_finish_fold_succeeds_when_a_different_hyp_validation_exists(tmp_path: Path):
    parent_main = tmp_path / "parent" / "main.py"
    parent_main.parent.mkdir()
    parent_main.write_text(PARENT, encoding="utf-8")
    tree = StepTree(tmp_path / "steps")
    clone = _record(tree, tmp_path / "clone", source=COMMENT_ONLY, result_name="valid_000")
    changed = _record(tree, tmp_path / "changed", source=LOGIC, result_name="valid_001")
    finish = _tool(tree, parent_main)
    selected = finish.invoke({"node_id": changed})
    assert selected.ok and selected.finish
    assert selected.value["node_id"] == changed
    kept = finish.invoke({"node_id": clone})
    assert kept.ok and kept.value["node_id"] == clone


def test_finish_fold_rejects_when_current_output_differs_from_selected_revision(
    tmp_path: Path,
):
    parent_main = tmp_path / "parent" / "main.py"
    parent_main.parent.mkdir()
    parent_main.write_text(PARENT, encoding="utf-8")
    tree = StepTree(tmp_path / "steps")
    changed = _record(tree, tmp_path / "changed", source=LOGIC, result_name="valid_001")
    working = tmp_path / "working"
    working.mkdir()
    (working / "main.py").write_text(PARENT, encoding="utf-8")
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        parent_main_py=parent_main,
        current_output=working,
    )
    with pytest.raises(ToolError, match="current output to match"):
        finish.invoke({"node_id": changed})
    (working / "main.py").write_text(LOGIC, encoding="utf-8")
    selected = finish.invoke({"node_id": changed})
    assert selected.ok and selected.value["node_id"] == changed


def _record_round(
    tree: StepTree,
    root: Path,
    *,
    batch_id: str,
    marker: str,
    metrics: dict[str, object] | None = None,
) -> str:
    """One recorded batch candidate: a Step node carrying the batch id."""
    output = root / f"cand_{batch_id}_{marker}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text(
        f"def generate_orders(context):\n    _ = {marker}\n    return []\n",
        encoding="utf-8",
    )
    return tree.record_step(
        output,
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name=f"valid_{batch_id}_{marker}",
        revision_id=new_revision_id("revision"),
        metrics=metrics or {},
        metadata={"batch_id": batch_id, "candidate": marker, "hypothesis": "h"},
    )


def test_finish_fold_requires_two_batch_rounds_while_another_fits(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    first = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x", min_batch_rounds=2)
    with pytest.raises(ToolError, match="1 of the 2 batch_validate rounds"):
        finish.invoke({"node_id": first})
    # A second candidate of the SAME batch is still one round.
    _record_round(tree, tmp_path, batch_id="b1", marker="2")
    with pytest.raises(ToolError, match="1 of the 2 batch_validate rounds"):
        finish.invoke({"node_id": first})
    second = _record_round(tree, tmp_path, batch_id="b2", marker="3")
    assert finish.invoke({"node_id": second}).finish
    # Selecting the first round's node after two rounds is equally fine.
    assert finish.invoke({"node_id": first}).finish


def test_finish_fold_waives_the_round_floor_when_no_round_fits(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        min_batch_rounds=2,
        another_round_fits=lambda: False,
    )
    assert finish.invoke({"node_id": node}).finish


def test_a_falsified_round_counts_and_other_sessions_rounds_do_not(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    # Another session's batch is evidence, not one of this session's rounds.
    tree.record_step(
        tree.node_output_dir(node),
        epoch_id="epoch_001",
        fold_id="fold_ref_other",
        run_id="run_y",
        result_name="valid_other",
        revision_id=new_revision_id("revision"),
        metrics={},
        metadata={"batch_id": "foreign"},
    )
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x", min_batch_rounds=2)
    with pytest.raises(ToolError, match="1 of the 2 batch_validate rounds"):
        finish.invoke({"node_id": node})
    # A round whose candidates all failed their replay still completed.
    tree.record_failed_attempt(
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="valid_b2_dead",
        error="generate_orders exceeded the per-decision timeout",
        metadata={"batch_id": "b2", "candidate": "dead", "hypothesis": "h"},
    )
    assert finish.invoke({"node_id": node}).finish


def _budget(remaining: int, total: int = 30) -> FoldBudgetStatus:
    return FoldBudgetStatus(
        backtests_remaining=remaining,
        backtests_total=total,
        steps_remaining=remaining,
        steps_total=total,
        inference_seconds_remaining=5400.0,
    )


def test_finish_fold_requires_a_reason_for_a_voluntary_early_finish(tmp_path: Path):
    """More than a third of the backtest budget left while another round
    still fits: the finish must say why, and the reason rides in the result."""
    tree = StepTree(tmp_path / "steps")
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    finish = FinishFoldTool(
        tree, fold_id="fold_ref_ab", run_id="run_x", budget_status=lambda: _budget(20)
    )
    with pytest.raises(ToolError, match="early_stop_reason") as refused:
        finish.invoke({"node_id": node})
    message = str(refused.value)
    assert "20/30 backtests" in message and "20/30 Steps" in message and "90 min" in message
    assert refused.value.details["backtests_remaining"] == 20
    assert refused.value.retry_hint
    reason = "H3 (unlock-pressure overlay) untested: the events domain is empty this window"
    finished = finish.invoke({"node_id": node, "early_stop_reason": reason})
    assert finished.finish
    assert finished.value["early_stop_reason"] == reason
    assert finished.value["budget_at_finish"]["backtests_remaining"] == 20
    assert finished.value["budget_at_finish"]["inference_seconds_remaining"] == 5400.0


def test_finish_fold_early_stop_reason_lapses_with_the_waiver_or_a_spent_budget(
    tmp_path: Path,
):
    tree = StepTree(tmp_path / "steps")
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    waived = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        budget_status=lambda: _budget(20),
        another_round_fits=lambda: False,
    )
    result = waived.invoke({"node_id": node})
    assert result.finish and "early_stop_reason" not in result.value
    assert result.value["budget_at_finish"]["backtests_total"] == 30
    # Exactly a third left is not early; a reason given anyway is still recorded.
    spent = FinishFoldTool(
        tree, fold_id="fold_ref_ab", run_id="run_x", budget_status=lambda: _budget(10)
    )
    assert "early_stop_reason" not in spent.invoke({"node_id": node}).value
    explained = spent.invoke({"node_id": node, "early_stop_reason": "done"})
    assert explained.value["early_stop_reason"] == "done"
    # Without a wired budget the tool cannot judge an early finish, only record.
    unwired = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    plain = unwired.invoke({"node_id": node})
    assert plain.finish and "budget_at_finish" not in plain.value


def test_finish_fold_bounds_the_early_stop_reason(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    registry = ToolRegistry([FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")])
    overlong = registry.invoke(
        "finish_fold",
        {"node_id": node, "early_stop_reason": "x" * (EARLY_STOP_REASON_MAX_CHARS + 1)},
    )
    assert overlong.ok is False and "early_stop_reason is too long" in overlong.error
    assert registry.invoke("finish_fold", {"node_id": node, "early_stop_reason": "x"}).ok


def test_finish_fold_bare_call_is_refused_on_the_parent_of_a_batch_round(tmp_path: Path):
    """``batch_validate`` leaves the position on the round's parent; a bare
    call there must not freeze the parent silently."""
    tree = StepTree(tmp_path / "steps")
    parent = _record(tree, tmp_path / "parent_node", source=LOGIC, result_name="valid_000")
    winner_dir = tmp_path / "cand_win"
    winner_dir.mkdir()
    (winner_dir / "main.py").write_text(
        "def generate_orders(context):\n    _ = 'win'\n    return []\n", encoding="utf-8"
    )
    winner = tree.record_step(
        winner_dir,
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="valid_001",
        revision_id=new_revision_id("revision"),
        metrics={"total_return": 0.12, "sharpe": 1.5},
        metadata={"batch_id": "b1", "candidate": "win", "hypothesis": "h"},
    )
    tree.set_position(parent)  # every candidate of a batch branches off the parent
    loser = _record_round(tree, tmp_path, batch_id="b1", marker="2")
    tree.set_position(parent)
    tree.record_failed_attempt(
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="valid_b1_dead",
        error="generate_orders exceeded the per-decision timeout",
        metadata={"batch_id": "b1", "candidate": "dead", "hypothesis": "h"},
    )
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    with pytest.raises(ToolError, match="explicit node_id") as refused:
        finish.invoke({})
    message = str(refused.value)
    assert f"{winner} (win: total_return=0.1200 sharpe=1.5000)" in message
    assert f"{loser} (2: complete)" in message and "(dead: failed)" in message
    assert refused.value.details["tree_position"] == parent
    assert [row["candidate"] for row in refused.value.details["candidates"]] == [
        "win",
        "2",
        "dead",
    ]
    # Explicit choices, the parent included, go through.
    assert finish.invoke({"node_id": winner}).value["node_id"] == winner
    assert finish.invoke({"node_id": parent}).value["node_id"] == parent


def test_finish_fold_bare_call_takes_the_position_when_no_batch_hangs_below(
    tmp_path: Path,
):
    tree = StepTree(tmp_path / "steps")
    node = _record(tree, tmp_path / "node", source=LOGIC, result_name="valid_000")
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    assert finish.invoke({}).value["node_id"] == node


def test_finish_fold_rejects_an_absent_or_snapshotless_node(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "main.py").write_text(PARENT, encoding="utf-8")
    tree = StepTree(tmp_path / "steps")
    node_id = _record(tree, output, source=PARENT, result_name="valid_000")
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    with pytest.raises(ToolError, match="absent"):
        finish.invoke({"node_id": "missing_node"})
    snapshot_dir = tree.node_output_dir(node_id)
    snapshot_dir.chmod(0o755)
    (snapshot_dir / "main.py").unlink()
    with pytest.raises(ToolError, match="snapshot is absent"):
        finish.invoke({"node_id": node_id})


def _written(directory: Path, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "main.py").write_text(source, encoding="utf-8")
    return directory


def _acceptance_check():
    """The Pipeline's own hard rules, wired the way the Fold session wires them."""
    from autotrade.pipelines.local_backend import acceptance_hard_rule_check

    check = acceptance_hard_rule_check({"max_drawdown": 0.25})
    assert check is not None
    return check


def _metrics(max_drawdown: float) -> dict[str, object]:
    return {"total_return": 0.1, "max_drawdown": max_drawdown, "sharpe": 0.4}


def test_finish_fold_refuses_a_hard_reject_while_another_node_passes(tmp_path: Path):
    """The reviewed failure: a 28.5 % drawdown node was accepted, the Pipeline
    then refused to freeze it and recorded baseline_missing while a sibling at
    24.7 % would have frozen. Outside the deadline window that is a refusal
    naming the nodes that pass, not a silent nomination."""

    tree = StepTree(tmp_path / "steps")
    breaching = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.285)
    )
    passing = _record_round(
        tree, tmp_path, batch_id="b2", marker="2", metrics=_metrics(0.2469)
    )
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        hard_rule_check=_acceptance_check(),
    )
    with pytest.raises(ToolError, match="hard acceptance rules") as refused:
        finish.invoke({"node_id": breaching})
    assert refused.value.error_type == "acceptance_hard_reject"
    assert passing in str(refused.value)
    assert refused.value.details["hard_reject_reasons"] == ["max_drawdown_exceeded"]
    verdicts = {
        row["node_id"]: row["passes_hard_rules"]
        for row in refused.value.details["candidates"]
    }
    assert verdicts == {breaching: False, passing: True}
    # The node that passes goes through untouched.
    accepted = finish.invoke({"node_id": passing})
    assert accepted.finish and "acceptance_hard_reject_reasons" not in accepted.value


def test_finish_fold_lists_parent_control_among_the_passing_nodes(tmp_path: Path):
    """Keeping the parent is done by selecting the host's control node, so it
    has to appear in the refusal exactly like the session's own Validations."""

    tree = StepTree(tmp_path / "steps")
    control = tree.record_step(
        _written(tmp_path / "control", PARENT),
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="parent_control",
        revision_id=new_revision_id("revision"),
        metrics=_metrics(0.12),
    )
    breaching = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.4)
    )
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        hard_rule_check=_acceptance_check(),
    )
    with pytest.raises(ToolError) as refused:
        finish.invoke({"node_id": breaching})
    listed = {
        row["node_id"]: row
        for row in refused.value.details["candidates"]
        if row["passes_hard_rules"]
    }
    assert set(listed) == {control}
    assert listed[control]["result_name"] == "parent_control"
    assert listed[control]["max_drawdown"] == 0.12
    assert control in str(refused.value)


def test_finish_fold_accepts_a_hard_reject_and_states_what_the_pipeline_will_do(
    tmp_path: Path,
):
    """Inside the deadline window, or with nothing recorded that passes, the
    nomination stands — but the result says the Pipeline will not freeze it, so
    the session's own early_stop_reason and the Meta review read the truth."""

    tree = StepTree(tmp_path / "steps")
    breaching = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.373)
    )
    passing = _record_round(
        tree, tmp_path, batch_id="b2", marker="2", metrics=_metrics(0.1)
    )
    check = _acceptance_check()
    in_window = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        hard_rule_check=check,
        another_round_fits=lambda: False,
    )
    accepted = in_window.invoke({"node_id": breaching})
    assert accepted.finish
    assert accepted.value["acceptance_hard_reject_reasons"] == ["max_drawdown_exceeded"]
    assert accepted.value["pipeline_will_freeze"] is False
    # No parent artifact was wired, so there is nothing to fall back to.
    assert accepted.value["pipeline_fold_status"] == "baseline_missing"
    # With a parent the Pipeline keeps it instead of recording nothing.
    parent_main = _written(tmp_path / "parent", PARENT) / "main.py"
    with_parent = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        parent_main_py=parent_main,
        hard_rule_check=check,
        another_round_fits=lambda: False,
    )
    assert (
        with_parent.invoke({"node_id": breaching}).value["pipeline_fold_status"]
        == "no_update"
    )
    # Outside the window the sibling that passes still forces the refusal.
    outside = FinishFoldTool(
        tree, fold_id="fold_ref_ab", run_id="run_x", hard_rule_check=check
    )
    with pytest.raises(ToolError, match="hard acceptance rules"):
        outside.invoke({"node_id": breaching})
    assert outside.invoke({"node_id": passing}).finish


def test_finish_fold_accepts_a_hard_reject_when_nothing_recorded_passes(
    tmp_path: Path,
):
    """open_mechanism fold_2022Q4/2023Q1: every candidate breached the cap.
    There is nothing to redirect the session to, so the nomination is accepted
    with the notice rather than refused into a dead end."""

    tree = StepTree(tmp_path / "steps")
    first = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.373)
    )
    _record_round(tree, tmp_path, batch_id="b2", marker="2", metrics=_metrics(0.344))
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        hard_rule_check=_acceptance_check(),
    )
    accepted = finish.invoke({"node_id": first})
    assert accepted.finish
    assert accepted.value["pipeline_fold_status"] == "baseline_missing"


def test_finish_fold_without_wired_rules_checks_nothing(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    node = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.9)
    )
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    accepted = finish.invoke({"node_id": node})
    assert accepted.finish and "pipeline_fold_status" not in accepted.value
