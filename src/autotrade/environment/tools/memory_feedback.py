"""Report how mounted operating memory held up, without ever rewriting it.

Mounted memory is provenance-tagged advice from other experiments and from the
researcher, not a rule this session must obey. A session may doubt it, ignore
it, and say so — but it may never edit or delete it, because those entries are
another experiment's immutable artifacts or the researcher's curated library.

``memory_feedback`` is the whole return path: one verdict per entry per session,
recorded in the run manifest and, like every tool call, in the session trace.
The console aggregates those manifests across experiments; promotion, exclusion
and every other change to the library stay the researcher's decision.
"""

from __future__ import annotations

from collections.abc import Mapping

from autotrade.environment.runtime import RunManifest, utc_now_iso

from .base import ToolError, ToolResult, ToolSpec
from .prior_policy import (
    calendar_policy_violation,
    strict_transferable_content_violation,
    visible_window_dates,
)
from .workspace import SafeWorkspace

MEMORY_FEEDBACK_VERDICTS = ("confirmed", "outdated", "wrong")
MAX_MEMORY_FEEDBACK_NOTE_CHARS = 500


class MemoryFeedbackTool:
    spec = ToolSpec(
        "memory_feedback",
        "Record how one mounted operating-memory entry held up against this "
        "session's own evidence. entry is <source>/<name> exactly as the "
        "operating_memory section of inputs/skills_index.json lists it "
        "(curated/<name> or <experiment_id>/<name>). verdict is confirmed, "
        "outdated or wrong. note is one short transferable sentence saying what "
        "the evidence showed, with no calendar date and no Test/Held-out figure. "
        "This records a judgement only: the mounted entry is never changed, and "
        "reporting the same entry again replaces this session's earlier verdict.",
        {
            "type": "object",
            "properties": {
                "entry": {"type": "string", "minLength": 3, "maxLength": 200},
                "verdict": {
                    "type": "string",
                    "enum": list(MEMORY_FEEDBACK_VERDICTS),
                },
                "note": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_MEMORY_FEEDBACK_NOTE_CHARS,
                },
            },
            "required": ["entry", "verdict", "note"],
            "additionalProperties": False,
        },
        mutating=True,
        example={
            "entry": "curated/pit-read-budget",
            "verdict": "outdated",
            "note": "按它的读取顺序取不到当前数据合同里的摘要字段，改用逐域摘要后才可用。",
        },
    )

    def __init__(self, workspace: SafeWorkspace, manifest: RunManifest) -> None:
        self.workspace = workspace
        self.manifest = manifest

    def _mounted_entry(self, reference: str) -> tuple[str, str]:
        """The reference must name something this session actually mounted."""

        # Imported here: pipelines.skills builds on this tools package, so the
        # dependency only exists while a call is running, never at import time.
        from autotrade.pipelines.skills import (
            OPERATING_MEMORY_DIRNAME,
            validate_memory_entry_ref,
        )

        source, name = validate_memory_entry_ref(reference)
        mounted = self.workspace.root / OPERATING_MEMORY_DIRNAME / source / name
        if not mounted.is_dir():
            raise ValueError(
                f"entry is not mounted in this session: {reference}; use a "
                "<source>/<name> listed in the operating_memory section of "
                "inputs/skills_index.json"
            )
        return source, name

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            source, name = self._mounted_entry(str(arguments["entry"]))
            verdict = str(arguments["verdict"]).strip()
            if verdict not in MEMORY_FEEDBACK_VERDICTS:
                raise ValueError(
                    "verdict must be one of " + ", ".join(MEMORY_FEEDBACK_VERDICTS)
                )
            note = str(arguments["note"]).strip()
            if not note:
                raise ValueError("note must say what the evidence showed")
            if len(note) > MAX_MEMORY_FEEDBACK_NOTE_CHARS:
                raise ValueError(
                    f"note exceeds {MAX_MEMORY_FEEDBACK_NOTE_CHARS} characters"
                )
            # The note travels to other experiments through the console, so it
            # passes the same transferable-content gate as PRIOR and skills.
            leak = calendar_policy_violation(
                note, window_dates=visible_window_dates(self.manifest.data)
            ) or strict_transferable_content_violation(note)
            if leak:
                raise ValueError(f"note {leak}; state it qualitatively instead")
            record: dict[str, object] = {
                "entry": f"{source}/{name}",
                "source": source,
                "name": name,
                "verdict": verdict,
                "note": note,
                "recorded_at": utc_now_iso(),
            }
            recorded = self.manifest.record_memory_feedback(record)
        except ValueError as exc:
            raise ToolError(str(exc), error_type="memory_feedback_policy") from exc
        return ToolResult(
            True, value={**record, "entries_reported_this_session": len(recorded)}
        )


__all__ = [
    "MAX_MEMORY_FEEDBACK_NOTE_CHARS",
    "MEMORY_FEEDBACK_VERDICTS",
    "MemoryFeedbackTool",
]
