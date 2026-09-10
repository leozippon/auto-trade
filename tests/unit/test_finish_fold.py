from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.base import ToolError, ToolRegistry
from autotrade.environment.tools.finish_fold import (
    EARLY_STOP_REASON_MAX_CHARS,
    NO_EDGE_REASON_MIN_CHARS,
    FinishFoldTool,
    FoldBudgetStatus,
    executable_source_structure,
    mechanism_difference,
    mechanism_structure,
)
from autotrade.environment.tools.modification_check import ModificationCheckTool

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


# A graduated mechanism with its declared knobs: the values a deployment
# refit may change without changing what the mechanism is.
MECHANISM = (
    "import numpy as np\n"
    "TOP_N = 20\n"
    "BOARDS = ['main', 'gem']\n"
    "HOLD_DAYS: int | None = None\n"
    "def score(rows):\n"
    "    if rows['pct_chg'] > 0.02 and rows['turnover'] < -1.5:\n"
    "        return np.log(rows['amount']) * 0.5\n"
    "    return 0.0\n"
    "def generate_orders(context):\n"
    "    use_cap = True\n"
    "    return [] if use_cap else None\n"
)


def _mechanism(**edits: str) -> str:
    source = MECHANISM
    for before, after in edits.items():
        assert before in source, before
        source = source.replace(before, after)
    return source


def _package(root: Path, source: str, *, extra: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(source, encoding="utf-8")
    for name, body in (extra or {}).items():
        (root / name).write_text(body, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "edits",
    (
        {"0.02": "0.035"},  # a threshold
        {"< -1.5": "< 1.5"},  # a signed threshold's sign
        {"TOP_N = 20": "TOP_N = 35"},  # a declared knob
        {"['main', 'gem']": "['main', 'gem', 'star']"},  # a declared list knob
        {"HOLD_DAYS: int | None = None": "HOLD_DAYS: int | None = 3"},  # None -> int knob
        {"TOP_N = 20": "TOP_N = 20.0"},  # a declared knob is one placeholder whatever its shape
        {"use_cap = True": "use_cap = False"},  # a boolean
        {"* 0.5": "* 0.5  # damped"},  # comment only
    ),
)
def test_mechanism_structure_ignores_declared_knobs(tmp_path: Path, edits: dict[str, str]):
    parent = _package(tmp_path / "parent", MECHANISM)
    child = _package(tmp_path / "child", _mechanism(**edits))
    assert mechanism_structure(child) == mechanism_structure(parent)
    assert mechanism_difference(parent, child) is None


@pytest.mark.parametrize(
    "edits",
    (
        {"rows['pct_chg']": "rows['pct_change']"},  # an inline column name
        {"and rows['turnover'] < -1.5": ""},  # a dropped comparison
        {"np.log(rows['amount'])": "np.sqrt(rows['amount'])"},  # another call
        {"import numpy as np": "import numpy as np\nimport pandas as pd"},  # an import
        {"    return 0.0\n": "    if rows['amount'] > 1e8:\n        return 1.0\n    return 0.0\n"},  # a branch
        {"* 0.5": "* 1"},  # an inline literal of another type
        {"TOP_N = 20": "TOP_N = len(BOARDS)"},  # a knob that is no longer a literal
        {"BOARDS = ['main', 'gem']": "boards = ['main', 'gem']"},  # an undeclared list
    ),
)
def test_mechanism_structure_sees_every_logic_change(tmp_path: Path, edits: dict[str, str]):
    parent = _package(tmp_path / "parent", MECHANISM)
    child = _package(tmp_path / "child", _mechanism(**edits))
    assert mechanism_structure(child) != mechanism_structure(parent)
    assert mechanism_difference(parent, child) is not None


def test_mechanism_difference_names_a_file_added_removed_or_changed(tmp_path: Path):
    parent = _package(tmp_path / "parent", MECHANISM, extra={"lib.py": "K = 1\n"})
    same = _package(tmp_path / "same", MECHANISM, extra={"lib.py": "K = 2\n"})
    assert mechanism_difference(parent, same) is None
    added = _package(tmp_path / "added", MECHANISM, extra={"lib.py": "K = 1\n", "more.py": "x = 1\n"})
    assert mechanism_difference(parent, added) == "more.py added"
    removed = _package(tmp_path / "removed", MECHANISM)
    assert mechanism_difference(parent, removed) == "lib.py removed"
    changed = _package(tmp_path / "changed", MECHANISM, extra={"lib.py": "K = 1\ndef f():\n    return K\n"})
    assert mechanism_difference(parent, changed) == (
        "lib.py changes the executable logic beyond its declared knobs"
    )


def test_finish_fold_same_mechanism_accepts_knob_edits_and_the_parent_only(tmp_path: Path):
    """A deployment adjustment nominates a refit of the graduated mechanism:
    a knob-only node or the parent itself is accepted, a node that changes the
    mechanism is refused, and the different-hypothesis rule does not apply."""
    parent_main = tmp_path / "parent" / "main.py"
    _package(parent_main.parent, MECHANISM)
    tree = StepTree(tmp_path / "steps")
    kept = _record(tree, tmp_path / "kept", source=MECHANISM, result_name="valid_000")
    refit = _record(tree, tmp_path / "refit", source=_mechanism(**{"0.02": "0.03"}), result_name="valid_001")
    changed = _record(
        tree,
        tmp_path / "changed",
        source=_mechanism(**{"rows['pct_chg']": "rows['pct_change']"}),
        result_name="valid_002",
    )
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        parent_main_py=parent_main,
        same_mechanism=True,
    )
    assert finish.invoke({"node_id": refit}).value["node_id"] == refit
    assert finish.invoke({"node_id": kept}).value["node_id"] == kept
    with pytest.raises(ToolError, match="changes the graduated mechanism") as refused:
        finish.invoke({"node_id": changed})
    assert refused.value.error_type == "mechanism_changed"
    # Without a parent there is no mechanism to keep.
    with pytest.raises(ValueError, match="same_mechanism"):
        FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x", same_mechanism=True)


def test_modification_check_refuses_a_mechanism_change_before_any_replay(tmp_path: Path):
    parent = _package(tmp_path / "parent", MECHANISM)
    parent_models = tmp_path / "parent_models"
    parent_models.mkdir()
    (parent_models / "weights.json").write_text('{"w": 1}\n', encoding="utf-8")
    work = _package(tmp_path / "work", _mechanism(**{"TOP_N = 20": "TOP_N = 30"}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "weights.json").write_text('{"w": 2}\n', encoding="utf-8")
    tool = ModificationCheckTool(
        work,
        parent_dir=parent,
        models_dir=models,
        parent_models_dir=parent_models,
        mechanism_parent=parent,
    )
    # A retrained models/ tree beside a knob edit is exactly the refit.
    accepted = ToolRegistry([tool]).invoke("modification_check", {})
    assert accepted.ok, accepted.error
    (work / "main.py").write_text(
        _mechanism(**{"np.log(rows['amount'])": "np.sqrt(rows['amount'])"}), encoding="utf-8"
    )
    refused = ToolRegistry([tool]).invoke("modification_check", {})
    assert not refused.ok
    assert refused.value["error_type"] == "artifact_constraint"
    assert "keeps the graduated mechanism" in str(refused.error)


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


def test_one_completed_round_is_enough_to_finish(tmp_path: Path):
    """No round floor: the environment counts no rounds before a finish.

    Sustained exploration is asked for in the prompts, not enforced here, so a
    node from the session's first round is a legal nomination.
    """

    tree = StepTree(tmp_path / "steps")
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1")
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
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
    """A stand-in for the Pipeline's hard rules: metrics -> reject reasons.

    ``finish_fold`` reports whatever the injected callable returns, whichever
    rules the Pipeline treats as hard at the time. Its current hard rule — a
    non-finite metric — cannot be staged here, because tree.json refuses to
    serialize one, so these paths use a same-shaped stub. That the real wiring
    builds a callable is asserted below; which rules it makes hard belongs to
    test_pipeline_config.
    """

    from autotrade.pipelines.local_backend import acceptance_hard_rule_check

    assert acceptance_hard_rule_check({"max_drawdown": 0.25}) is not None

    def check(metrics: dict[str, object]) -> list[str]:
        drawdown = metrics.get("max_drawdown")
        if isinstance(drawdown, (int, float)) and abs(float(drawdown)) > 0.25:
            return ["max_drawdown_exceeded"]
        return []

    return check


def _metrics(max_drawdown: float) -> dict[str, object]:
    return {"total_return": 0.1, "max_drawdown": max_drawdown, "sharpe": 0.4}


def test_finish_fold_refuses_a_hard_reject_while_another_node_passes(tmp_path: Path):
    """The reviewed failure: a node the hard rules reject was accepted, the
    Pipeline then refused to freeze it and recorded baseline_missing while a
    sibling would have frozen. Outside the deadline window that is a refusal
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
    """open_mechanism fold_2022Q4/2023Q1: every candidate failed the rules.
    There is nothing to redirect the session to, so the nomination is accepted
    with the notice rather than refused into a dead end."""

    tree = StepTree(tmp_path / "steps")
    first = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.373)
    )
    _record_round(
        tree, tmp_path, batch_id="b2", marker="2", metrics=_metrics(0.344)
    )
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
    assert accepted.finish and "acceptance_hard_reject_reasons" not in accepted.value
    # No rule to breach: the Pipeline freezes the nomination, and the result
    # says so instead of staying silent.
    assert accepted.value["pipeline_fold_status"] == "frozen"
    assert accepted.value["pipeline_will_freeze"] is True


def test_every_accepted_nomination_states_what_the_pipeline_will_freeze(tmp_path: Path):
    """The reviewed defect: a session nominated "the least-bad node" believing
    nothing would be frozen, and the Pipeline froze it. Every accepted call now
    states the fold status and what will be frozen, in the ledger's words."""

    tree = StepTree(tmp_path / "steps")
    passing = _record_round(
        tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.1)
    )
    finish = FinishFoldTool(
        tree, fold_id="fold_ref_ab", run_id="run_x", hard_rule_check=_acceptance_check()
    )
    accepted = finish.invoke({"node_id": passing})
    assert accepted.value["outcome"] == "select"
    assert accepted.value["pipeline_fold_status"] == "frozen"
    assert accepted.value["pipeline_will_freeze"] is True
    assert accepted.value["pipeline_outcome"] == (
        f"Fold will freeze {passing} (valid_b1_1) as this Fold's strategy"
    )
    # A hard-rejected nomination accepted inside the window says the same
    # thing the ledger will: nothing frozen, and why.
    breaching = _record_round(
        tree, tmp_path, batch_id="b2", marker="2", metrics=_metrics(0.4)
    )
    in_window = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        hard_rule_check=_acceptance_check(),
        another_round_fits=lambda: False,
    )
    rejected = in_window.invoke({"node_id": breaching})
    assert rejected.value["pipeline_will_freeze"] is False
    assert rejected.value["pipeline_fold_status"] == "baseline_missing"
    assert rejected.value["pipeline_outcome"].startswith("No candidate frozen: ")
    assert "max_drawdown_exceeded" in rejected.value["pipeline_outcome"]
    assert "baseline_missing" in rejected.value["pipeline_outcome"]


NO_EDGE_REASON = (
    "all four candidates: neutralized excess about 0 and beats_parent=false; "
    "the new quarter is negative for every one of them"
)


def test_finish_fold_no_edge_records_the_fallback_status_without_a_nomination(
    tmp_path: Path,
):
    """A Fold that found no edge finishes without nominating anything: the
    parent, when there is one, stays the lineage head (no_update); a parentless
    Fold whose every candidate fails the hard rules records baseline_missing.
    Neither path freezes a node."""

    tree = StepTree(tmp_path / "steps")
    # Beyond the stand-in drawdown cap: hard-rejected, so nothing a parentless
    # Fold could anchor its lineage on.
    _record_round(tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.4))
    parentless = FinishFoldTool(
        tree, fold_id="fold_ref_ab", run_id="run_x", hard_rule_check=_acceptance_check()
    )
    result = parentless.invoke({"outcome": "no_edge", "reason": NO_EDGE_REASON})
    assert result.ok and result.finish
    assert "node_id" not in result.value and "revision_id" not in result.value
    assert result.value["outcome"] == "no_edge"
    assert result.value["reason"] == NO_EDGE_REASON
    assert result.value["candidates_evaluated"] == 1
    assert result.value["fold_status"] == "pending_pipeline_review"
    assert result.value["pipeline_fold_status"] == "baseline_missing"
    assert result.value["pipeline_will_freeze"] is False
    assert result.value["pipeline_outcome"].startswith(
        "No candidate frozen; there is no parent"
    )
    parent_main = _written(tmp_path / "parent", PARENT) / "main.py"
    with_parent = FinishFoldTool(
        tree, fold_id="fold_ref_ab", run_id="run_x", parent_main_py=parent_main
    )
    kept = with_parent.invoke({"outcome": "no_edge", "reason": NO_EDGE_REASON})
    assert kept.value["pipeline_fold_status"] == "no_update"
    assert kept.value["pipeline_outcome"] == (
        "No candidate frozen; the inherited parent stays the lineage head (no_update)"
    )


def test_finish_fold_no_edge_is_refused_without_a_parent_while_a_candidate_passes(
    tmp_path: Path,
):
    """The baseline anchor rule. Three arms spent every Fold on abstentions:
    with no frozen parent there was never a parent_control, a vs_parent or a
    walk-forward transition, so the Meta review saw no forward evidence at all.
    Without a parent, an abstention is refused while a complete Validation
    passes the hard rules, and the refusal lists those candidates with the
    figures the choice is made on; the choice itself stays the Agent's."""

    tree = StepTree(tmp_path / "steps")
    passing = _record_round(
        tree,
        tmp_path,
        batch_id="b1",
        marker="1",
        metrics={**_metrics(0.1), "benchmark": {"neutralized_excess_return": 0.0123}},
    )
    rejected = _record_round(tree, tmp_path, batch_id="b1", marker="2", metrics=_metrics(0.4))
    finish = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        hard_rule_check=_acceptance_check(),
        null_controls=lambda: {passing: {"excess_percentile": 0.62}},
        # The anchor refusal comes before the early-stop justification: no
        # reason could make this abstention legal.
        budget_status=lambda: _budget(20),
    )
    with pytest.raises(ToolError) as error:
        finish.invoke({"outcome": "no_edge", "reason": NO_EDGE_REASON})
    assert error.value.error_type == "baseline_anchor_required"
    message = str(error.value)
    assert "no frozen parent" in message and "baseline anchor" in message
    listed = message.split("Passing candidates: ")[1]
    assert listed.startswith(
        f"{passing} (valid_b1_1, neutralized_excess_return=0.0123, "
        "null_excess_percentile=0.6200)"
    )
    assert rejected not in listed
    rows = {row["node_id"]: row for row in error.value.details["candidates"]}
    assert rows[passing]["passes_hard_rules"] is True
    assert rows[rejected]["hard_reject_reasons"] == ["max_drawdown_exceeded"]
    # Nominating the passing node is the way out, and the Pipeline freezes it.
    selected = finish.invoke({"node_id": passing, "early_stop_reason": "anchor"})
    assert selected.finish and selected.value["pipeline_fold_status"] == "frozen"
    # Without wired rules every complete Validation passes, so a bare
    # parentless tool refuses the same way.
    bare = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    with pytest.raises(ToolError, match="baseline anchor"):
        bare.invoke({"outcome": "no_edge", "reason": NO_EDGE_REASON})


def test_finish_fold_no_edge_refuses_a_node_a_thin_reason_or_an_empty_session(
    tmp_path: Path,
):
    tree = StepTree(tmp_path / "steps")
    finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
    # Nothing validated yet: there is no evidence to have found no edge in,
    # and the host's parent control is the baseline, not a candidate.
    tree.record_step(
        _written(tmp_path / "control", PARENT),
        epoch_id="epoch_001",
        fold_id="fold_ref_ab",
        run_id="run_x",
        result_name="parent_control",
        revision_id=new_revision_id("revision"),
        metrics=_metrics(0.1),
        metadata={"parent_control": True},
    )
    with pytest.raises(ToolError, match="at least one complete Validation"):
        finish.invoke({"outcome": "no_edge", "reason": NO_EDGE_REASON})
    node = _record_round(tree, tmp_path, batch_id="b1", marker="1", metrics=_metrics(0.1))
    with pytest.raises(ToolError, match="requires reason"):
        finish.invoke({"outcome": "no_edge"})
    with pytest.raises(ToolError, match="requires reason"):
        finish.invoke({"outcome": "no_edge", "reason": "no edge"})
    with pytest.raises(ToolError, match="node_id must be absent"):
        finish.invoke({"outcome": "no_edge", "node_id": node, "reason": NO_EDGE_REASON})
    # A nomination does not take the no-edge reason, and the schema refuses
    # any other outcome before the tool runs.
    with pytest.raises(ToolError, match='belongs to outcome="no_edge"'):
        finish.invoke({"node_id": node, "reason": NO_EDGE_REASON})
    registry = ToolRegistry([finish])
    assert registry.invoke("finish_fold", {"outcome": "abstain"}).ok is False
    # The schema carries the same floor as the runtime check, so a thin reason
    # is refused with the field rule instead of reaching the tool.
    thin = registry.invoke("finish_fold", {"outcome": "no_edge", "reason": "no edge"})
    assert thin.ok is False
    assert f"minimum {NO_EDGE_REASON_MIN_CHARS}" in thin.error
    # The early-finish gate applies to an abstention exactly as to a nomination
    # (with a parent: without one the anchor rule refuses the abstention first).
    budgeted = FinishFoldTool(
        tree,
        fold_id="fold_ref_ab",
        run_id="run_x",
        parent_main_py=_written(tmp_path / "parent", PARENT) / "main.py",
        budget_status=lambda: _budget(20),
    )
    with pytest.raises(ToolError, match="early_stop_reason"):
        budgeted.invoke({"outcome": "no_edge", "reason": NO_EDGE_REASON})
    finished = budgeted.invoke(
        {
            "outcome": "no_edge",
            "reason": "x" * NO_EDGE_REASON_MIN_CHARS,
            "early_stop_reason": "H3 untested: the events domain is empty this window",
        }
    )
    assert finished.finish and finished.value["early_stop_reason"]
    assert finished.value["budget_at_finish"]["backtests_remaining"] == 20
