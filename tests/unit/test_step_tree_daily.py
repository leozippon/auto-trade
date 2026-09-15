from __future__ import annotations

import stat
from pathlib import Path

from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools import ToolRegistry
from autotrade.environment.tools.finish_session import FinishSessionTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.step_rollback import StepRollbackTool


def _passing_gate(node_id: str) -> dict[str, object]:
    """A freeze gate every node passes."""
    return {"passed": True, "reasons": []}


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
        session_ref="fold_2026Q1",
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
        tree, work, work_models, session_ref="fold_2026Q1", run_id="run_a"
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
    finished = FinishSessionTool(tree, session_ref="fold_2026Q1", run_ref="run_a", freeze_gate=_passing_gate).invoke({"outcome": "freeze"})
    assert finished.finish and finished.value["revision_id"] == "revision_a"


def test_step_rollback_refuses_a_node_outside_the_current_session(tmp_path: Path):
    """Another session's or run's node is evidence only, exactly as finish_session treats it.

    The experiment-level tree is handed whole to every session, so a foreign
    ``node_id`` is readable; restoring one would rebase this session's work
    copy and lineage onto a node it may not submit.
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
        session_ref="fold_2025Q4",
        run_id="run_old",
        result_name="valid_000",
        revision_id="revision_old",
        metrics={"total_return": 0.4},
    )
    earlier_run = tree.record_step(
        revision,
        epoch_id="epoch_001",
        session_ref="fold_2026Q1",
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
            StepRollbackTool(tree, work, session_ref="fold_2026Q1", run_id="run_a"),
            FinishSessionTool(tree, session_ref="fold_2026Q1", run_ref="run_a", freeze_gate=_passing_gate),
        ]
    )
    position = tree.current_node_id

    for node_id in (foreign_fold, earlier_run):
        result = registry.invoke("step_rollback", {"node_id": node_id})
        assert not result.ok
        assert "current session" in result.error
        assert (work / "main.py").read_text(encoding="utf-8") == "current work copy"
        assert StepTree(tmp_path / "steps").current_node_id == position
        finished = registry.invoke("finish_session", {"outcome": "freeze", "node_id": node_id})
        assert not finished.ok
        assert "not a Step of this session" in finished.error

    # An absent node is shaped like its finish_session sibling: a typed tool error
    # the model can act on, not an untyped ValueError leaking through.
    for tool, arguments in (
        ("step_rollback", {"node_id": "step_missing"}),
        ("finish_session", {"outcome": "freeze", "node_id": "step_missing"}),
    ):
        absent = registry.invoke(tool, arguments)
        assert not absent.ok, tool
        assert absent.value["error_type"] == "tool_error", tool
        assert "absent Step" in absent.error or "not a Step node" in absent.error, tool


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
        "    pd.read_parquet('/mnt/snapshot/daily.parquet')\n"
        "    return []\n",
        encoding="utf-8",
    )
    registry = ToolRegistry([ModificationCheckTool(output)])

    result = registry.invoke("modification_check", {})
    assert not result.ok
    assert "absolute path literal to read_parquet" in result.error

    # The repair only has to root the path at a context directory; how the
    # strategy builds it from there is its own business.
    strategy.write_text(
        "import pandas as pd\n"
        "def generate_orders(context):\n"
        "    path = context.asof_dir\n"
        "    pd.read_parquet(path + '/daily')\n"
        "    return []\n",
        encoding="utf-8",
    )
    assert registry.invoke("modification_check", {}).ok
