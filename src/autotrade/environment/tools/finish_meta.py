"""Finish a Meta session once the fixed PRIOR.md passes the content policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .base import ToolError, ToolResult, ToolSpec
from .prior_policy import PRIOR_MAX_CHARS, prior_policy_violation
from .workspace import SafeWorkspace

# The gate is a set of per-line pattern checks, not a semantic review; the
# description says exactly what trips them so a first attempt can pass.
_PRIOR_REDLINES = (
    f"Refused while PRIOR.md is missing, empty, or over {PRIOR_MAX_CHARS} chars, or "
    "any single line (a) writes a calendar date — a year welded to 年/-MM/.MM/Qn/"
    "季度, an 8-digit YYYYMMDD, or this window's bare year or endpoint — "
    "(b) mentions held-out/holdout/持有期外/隐藏区间, (c) puts Test/测试 next to a "
    "performance figure (a return/sharpe/drawdown/IC-style word beside a number, or "
    "a signed percentage), or (d) chooses or keeps a strategy because of Test. "
    "A line carrying a prohibition word (不得/禁止/不可见/排除…) is exempt from "
    "(b)-(d), but only while it states no figure of its own. Checks are line-level "
    "patterns: the error names the line and quotes the matched words; reword it "
    "qualitatively (cadence words such as 季度/月 and plain counts are fine) and "
    "call finish_meta again."
)


class FinishMetaTool:
    spec = ToolSpec(
        "finish_meta",
        "Finish local Meta learning after maintaining the fixed PRIOR.md; the call "
        "starts only after every background sub-agent has finished, and once it "
        "succeeds the remaining tool calls of this assistant turn are cancelled. "
        + _PRIOR_REDLINES,
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
            raise ToolError(
                violation,
                error_type="prior_policy",
                retry_hint=(
                    "edit_file the named line of PRIOR.md, then call finish_meta "
                    "again; the checks are line-level patterns, not a semantic "
                    "review, so keep hidden-stage words, Test figures and calendar "
                    "dates off ordinary lines"
                ),
            )
        return ToolResult(
            True,
            value={"status": "meta_learning_done"},
            finish=True,
        )


__all__ = ["FinishMetaTool"]
