"""``compact``: the Agent replaces its conversation with its own summary.

The tool itself only validates the summary; the session runner rebuilds the
conversation after the call (``agent.compact.compact_with_summary``) and
records the compaction in the trace, which the Agent reads back through the
``trace`` root.
"""

from __future__ import annotations

from collections.abc import Mapping

from .base import ToolError, ToolResult, ToolSpec

COMPACT_SUMMARY_MIN_CHARS = 200
# About 10,000 tokens, the order of what the compaction model may write: at
# 16,000 a session's own summary of a 221,184-token context ran into the limit.
COMPACT_SUMMARY_MAX_CHARS = 32_000


class CompactTool:
    spec = ToolSpec(
        "compact",
        "Replace this conversation with your own summary of it. After the call the "
        "conversation is the system prompt (facts included), one message carrying "
        "summary verbatim, and the most recent messages; everything older leaves the "
        "context but stays in the trace, which read_file and grep read under root "
        "trace. Write the summary for yourself: the state of output/ and the "
        "candidates, every decision and rejected direction with its numbers and node "
        "ids, the open threads and what comes next, and pointers into the trace "
        "(call indices or grep terms) for details you may need again. Call it when "
        "the context_notice observation arrives, or at a round boundary once its "
        "results are digested; when the threshold is reached without it, the host "
        f"compacts with a model instead. {COMPACT_SUMMARY_MIN_CHARS}-"
        f"{COMPACT_SUMMARY_MAX_CHARS} characters; a longer summary keeps only its "
        "whole lines within the limit and ends on a line saying it was cut.",
        {
            "type": "object",
            "properties": {
                # No schema maxLength: a strict-mode provider enforces it while
                # decoding and hands back a summary stopped mid-item at exactly
                # the limit, indistinguishable from a whole one (round
                # 20261113b). The rebuild cuts an over-long one at a line and
                # says so (``agent.compact.compact_with_summary``).
                "summary": {
                    "type": "string",
                    "minLength": COMPACT_SUMMARY_MIN_CHARS,
                    "description": (
                        "Your working state, written for yourself: strategy state, "
                        "decisions and rejected directions with numbers and node ids, "
                        "open threads, next steps, trace pointers."
                    ),
                }
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        example={"summary": "<state of the strategy, decisions with numbers, open threads>"},
    )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        summary = str(arguments.get("summary") or "").strip()
        if len(summary) < COMPACT_SUMMARY_MIN_CHARS:
            raise ToolError(
                f"compact: summary has {len(summary)} characters after trimming; write at "
                f"least {COMPACT_SUMMARY_MIN_CHARS} so the rebuilt context can carry the "
                "session's state",
                error_type="schema_error",
                blocked_target="summary",
            )
        return ToolResult(True, value={"status": "compacted", "summary_chars": len(summary)})


__all__ = ["COMPACT_SUMMARY_MAX_CHARS", "COMPACT_SUMMARY_MIN_CHARS", "CompactTool"]
