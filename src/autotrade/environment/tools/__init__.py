"""Agent-facing tool contracts dispatched by the session runner.

Heavy tool modules are imported lazily so internal helpers can reuse
``tools.base`` without pulling the backtest/NL stack into a circular import.
"""

from __future__ import annotations

from .base import (
    CommandResult,
    CommandRunner,
    SessionInterrupt,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSchemaError,
    ToolSpec,
)
from .files import EditFileTool, WriteFileTool
from .finish_fold import FinishFoldTool
from .hitl import AskUserTool
from .modification_check import ModificationCheckTool
from .search import SEARCH_ROOTS, GlobTool, GrepTool, ReadFileTool, SearchRoots
from .shell import ReadOnlyShellTool, SandboxShellTool
from .step_rollback import StepRollbackTool
from .strategy_validation import StrategyValidationTool
from .todo import TodoTool
from .workspace import SafeWorkspace

__all__ = [
    "AskUserTool",
    "BacktestTool",
    "CommandResult",
    "CommandRunner",
    "EditFileTool",
    "FinishFoldTool",
    "GlobTool",
    "GrepTool",
    "ModificationCheckTool",
    "ReadFileTool",
    "ReadOnlyShellTool",
    "SEARCH_ROOTS",
    "SafeWorkspace",
    "SandboxShellTool",
    "SearchRoots",
    "SessionInterrupt",
    "StepRollbackTool",
    "StrategyValidationTool",
    "TodoTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSchemaError",
    "ToolSpec",
    "WriteFileTool",
]


def __getattr__(name: str):
    if name == "BacktestTool":
        from .backtest import BacktestTool

        return BacktestTool
    raise AttributeError(name)
