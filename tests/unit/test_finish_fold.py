from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.finish_fold import (
    FinishFoldTool,
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


def test_structure_ignores_comments_docstrings_and_whitespace():
    parent = executable_source_structure(PARENT)
    assert executable_source_structure(COMMENT_ONLY) == parent
    assert executable_source_structure(DOCSTRING_ONLY) == parent
    assert executable_source_structure("def generate_orders(context):\n\n    return []\n") == parent
    assert executable_source_structure(LOGIC) != parent


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


def _record_round(tree: StepTree, root: Path, *, batch_id: str, marker: str) -> str:
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
        metrics={},
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
