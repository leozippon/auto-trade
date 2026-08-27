"""Agent-facing tool contracts dispatched by the session runner.

The skills tools are imported lazily so internal helpers can reuse
``tools.base`` without pulling the pipeline package into a circular import.
"""

from __future__ import annotations

from .base import (
    SEQUENTIAL_TOOL_NAMES,
    CommandResult,
    CommandRunner,
    SessionInterrupt,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSchemaError,
    ToolSpec,
    is_sequential_tool,
)
from .files import EditFileTool, WriteFileTool
from .finish_fold import FinishFoldTool
from .finish_meta import FinishMetaTool
from .hitl import AskUserTool
from .modification_check import ModificationCheckTool
from .search import SEARCH_ROOTS, GlobTool, GrepTool, ReadFileTool, SearchRoots
from .shell import SandboxShellTool
from .step_rollback import StepRollbackTool
from .workspace import SafeWorkspace

__all__ = [
    "AskUserTool",
    "CommandResult",
    "CommandRunner",
    "EditFileTool",
    "FinishFoldTool",
    "FinishMetaTool",
    "GlobTool",
    "GrepTool",
    "ModificationCheckTool",
    "ReadFileTool",
    "SEARCH_ROOTS",
    "SEQUENTIAL_TOOL_NAMES",
    "SafeWorkspace",
    "SandboxShellTool",
    "SearchRoots",
    "SessionInterrupt",
    "StepRollbackTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSchemaError",
    "ToolSpec",
    "WriteFileTool",
    "WriteSkillTool",
    "DeleteSkillTool",
    "is_sequential_tool",
]


def __dir__() -> list[str]:
    return sorted({*globals(), "WriteSkillTool", "DeleteSkillTool"})


def __getattr__(name: str):
    if name in {"WriteSkillTool", "DeleteSkillTool"}:
        from autotrade.pipelines.skills import DeleteSkillTool, WriteSkillTool

        return {"WriteSkillTool": WriteSkillTool, "DeleteSkillTool": DeleteSkillTool}[name]
    raise AttributeError(name)
