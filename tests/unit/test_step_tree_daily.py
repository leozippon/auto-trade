from __future__ import annotations

import stat
from pathlib import Path

from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import ToolRegistry
from autotrade.environment.tools.finish_fold import FinishFoldTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.step_rollback import StepRollbackTool


def test_step_revision_can_be_selected_and_restored(tmp_path: Path):
    revision = tmp_path / "revision"
    revision_models = tmp_path / "revision-models"
    revision.mkdir()
    revision_models.mkdir()
    (revision / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    (revision_models / "weights.bin").write_bytes(b"selected")
    tree = StepTree(tmp_path / "steps")
    node = tree.record_step(
        revision,
        epoch_id="epoch_001",
        fold_id="fold_2026Q1",
        run_id="run_a",
        result_name="valid_000",
        revision_id="revision_a",
        metrics={"total_return": 0.1},
        models_root=revision_models,
    )
    work = tmp_path / "work"
    work_models = tmp_path / "work-models"
    work.mkdir()
    work_models.mkdir()
    (work / "main.py").write_text("broken", encoding="utf-8")
    (work_models / "weights.bin").write_bytes(b"broken")
    rollback = StepRollbackTool(
        tree, work, work_models, fold_id="fold_2026Q1", run_id="run_a"
    )
    restored = rollback.invoke({"node_id": node})
    assert restored.ok
    assert "generate_orders" in (work / "main.py").read_text(encoding="utf-8")
    assert (work_models / "weights.bin").read_bytes() == b"selected"
    assert stat.S_IMODE(work.stat().st_mode) == 0o777
    assert stat.S_IMODE((work / "main.py").stat().st_mode) == 0o666
    assert stat.S_IMODE(work_models.stat().st_mode) == 0o777
    assert stat.S_IMODE((work_models / "weights.bin").stat().st_mode) == 0o666
    assert stat.S_IMODE(tree.node_models_dir(node).stat().st_mode) == 0o555
    assert stat.S_IMODE((tree.node_models_dir(node) / "weights.bin").stat().st_mode) == 0o444
    finished = FinishFoldTool(tree, fold_id="fold_2026Q1", run_id="run_a").invoke({})
    assert finished.finish and finished.value["revision_id"] == "revision_a"


def test_step_rollback_refuses_a_node_outside_the_current_fold_session(tmp_path: Path):
    """Another Fold's or run's node is evidence only, exactly as finish_fold treats it.

    The experiment-level tree is handed whole to every Fold, so a foreign
    ``node_id`` is readable; restoring one would rebase this Fold's work copy
    and lineage onto an artifact it may not submit.
    """

    revision = tmp_path / "revision"
    revision.mkdir()
    (revision / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    tree = StepTree(tmp_path / "steps")
    foreign_fold = tree.record_step(
        revision,
        epoch_id="epoch_001",
        fold_id="fold_2025Q4",
        run_id="run_old",
        result_name="valid_000",
        revision_id="revision_old",
        metrics={"total_return": 0.4},
    )
    earlier_run = tree.record_step(
        revision,
        epoch_id="epoch_001",
        fold_id="fold_2026Q1",
        run_id="run_crashed",
        result_name="valid_000",
        revision_id="revision_crashed",
        metrics={"total_return": 0.2},
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "main.py").write_text("current work copy", encoding="utf-8")
    registry = ToolRegistry(
        [
            StepRollbackTool(tree, work, fold_id="fold_2026Q1", run_id="run_a"),
            FinishFoldTool(tree, fold_id="fold_2026Q1", run_id="run_a"),
        ]
    )
    position = tree.current_node_id

    for node_id in (foreign_fold, earlier_run):
        result = registry.invoke("step_rollback", {"node_id": node_id})
        assert not result.ok
        assert "current Fold session" in result.error
        assert (work / "main.py").read_text(encoding="utf-8") == "current work copy"
        assert StepTree(tmp_path / "steps").current_node_id == position
        finished = registry.invoke("finish_fold", {"node_id": node_id})
        assert not finished.ok
        assert "current Fold session" in finished.error

    # An absent node is shaped like its finish_fold sibling: a typed tool error
    # the model can act on, not an untyped ValueError leaking through.
    for tool in ("step_rollback", "finish_fold"):
        absent = registry.invoke(tool, {"node_id": "step_missing"})
        assert not absent.ok, tool
        assert absent.value["error_type"] == "tool_error", tool
        assert "absent Step" in absent.error, tool


def test_modification_check_keeps_daily_json_entry(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    result = ModificationCheckTool(output).invoke({})
    assert result.ok
    assert result.value["strategy_entry"] == "generate_orders"


def test_strategy_path_violation_is_repairable_at_agent_tool_boundary(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    strategy = output / "main.py"
    strategy.write_text(
        "import pandas as pd\n"
        "def generate_orders(context):\n"
        "    path = context.asof_dir\n"
        "    pd.read_parquet(path + '/daily')\n"
        "    return []\n",
        encoding="utf-8",
    )
    registry = ToolRegistry([ModificationCheckTool(output)])

    result = registry.invoke("modification_check", {})
    assert not result.ok
    assert "only below context.snapshot_dir or context.asof_dir" in result.error

    strategy.write_text(
        "import pandas as pd\n"
        "def generate_orders(context):\n"
        "    pd.read_parquet(context.asof_dir + '/daily')\n"
        "    return []\n",
        encoding="utf-8",
    )
    assert registry.invoke("modification_check", {}).ok
