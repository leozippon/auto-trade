"""Finish a Meta session once the fixed PRIOR.md passes the content policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .base import ToolError, ToolResult, ToolSpec
from .prior_policy import prior_policy_violation
from .workspace import SafeWorkspace


class FinishMetaTool:
    spec = ToolSpec(
        "finish_meta",
        "Finish local Meta learning after maintaining the fixed PRIOR.md.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        workspace: str | Path | SafeWorkspace,
        *,
        window_dates: set[str] | None = None,
    ) -> None:
        self.workspace = (
            workspace
            if isinstance(workspace, SafeWorkspace)
            else SafeWorkspace(workspace)
        )
        self.window_dates = set(window_dates or ())

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        violation = prior_policy_violation(
            self.workspace.root / "PRIOR.md", window_dates=self.window_dates
        )
        if violation:
            raise ToolError(violation, error_type="prior_policy")
        return ToolResult(
            True,
            value={"status": "meta_learning_done"},
            finish=True,
        )


__all__ = ["FinishMetaTool"]
