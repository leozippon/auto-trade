"""File a defect the session itself discovered, so it reaches the operators.

Trace audits keep finding sessions that correctly diagnose a real problem with
the environment, a trusted tool's output, the mounted data, or the mounted
documentation — and that knowledge dies in the trace until a human rereads it.
``report_issue`` is the first-class return path: the research session's parent
(never a sub-agent) appends one redacted JSON line per report to the
experiment's ``ledgers/issue_reports.jsonl``, where the console lists them for
the researcher.

Host-side telemetry only: a report changes no artifact, budget, or result, and
is never mounted or projected back into any session's inputs — it is operator
telemetry, not memory and not a results channel.

The researcher answers a report in the same log: ``resolve_issue.py`` appends a
``record_type: "resolution"`` line naming the report, the outcome and a short
note. Nothing is rewritten — the filed report stays exactly as the session
wrote it, and the console joins the two on read.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.runtime import (
    RunManifest,
    append_versioned_jsonl,
    new_id,
)

from .base import ToolError, ToolResult, ToolSpec
from .feedback_log import (
    append_resolution,
    read_feedback_log,
    session_link_fields,
)

# Stamped on every appended report; bump when the record shape changes.
ISSUE_REPORT_SCHEMA_VERSION = 1
ISSUE_CATEGORIES = ("tool_output", "environment", "data", "docs", "other")
# The researcher's verdict on one report. ``fixed`` names the repair, the other
# two close a report that needed no code change; the note carries the reason
# (and, for a fix, the commit).
ISSUE_OUTCOMES = ("fixed", "not_a_defect", "accepted_limitation")
# Names this log's records in the shared reader's errors.
ISSUE_REPORT_LABEL = "issue report"
MAX_ISSUE_SUMMARY_CHARS = 500
MAX_ISSUE_EVIDENCE_CHARS = 4_000
MAX_ISSUE_REPORTS_PER_SESSION = 16
ISSUE_REPORTS_NAME = "issue_reports.jsonl"


def issue_reports_path(experiment_dir: str | Path) -> Path:
    """Single source for the writer and the console reader."""

    return Path(experiment_dir) / "ledgers" / ISSUE_REPORTS_NAME


def append_issue_report(
    path: str | Path, record: Mapping[str, object]
) -> dict[str, object]:
    """Append one sanitized report line, durable like the experiment ledger."""

    return append_versioned_jsonl(
        path, record, schema_version=ISSUE_REPORT_SCHEMA_VERSION
    )


def read_issue_reports(path: str | Path) -> list[dict[str, object]]:
    """All reports in append order, each carrying its resolution when it has one."""

    return read_feedback_log(
        path,
        schema_version=ISSUE_REPORT_SCHEMA_VERSION,
        label=ISSUE_REPORT_LABEL,
        outcomes=ISSUE_OUTCOMES,
    )


def append_issue_resolution(
    path: str | Path, *, report_id: str, outcome: str, note: str
) -> dict[str, object]:
    """Record how one filed report was answered; refuse anything else."""

    return append_resolution(
        path,
        schema_version=ISSUE_REPORT_SCHEMA_VERSION,
        label=ISSUE_REPORT_LABEL,
        outcomes=ISSUE_OUTCOMES,
        report_id=report_id,
        outcome=outcome,
        note=note,
    )


class ReportIssueTool:
    spec = ToolSpec(
        "report_issue",
        "Report a suspected defect in the environment, a trusted tool's output, "
        "the mounted data, or the mounted documentation to the human operators. "
        "category is tool_output (a trusted tool or command returned something "
        "misleading or malformed), environment (sandbox/replay/timeout/"
        "infrastructure behavior), data (PIT data, field or unit surprises), "
        "docs (mounted docs or reference text wrong or misleading), or other. "
        "summary states the issue in one or two sentences; evidence gives the "
        "concrete reproduction: what was run or read, expected vs observed, "
        "workspace-relative paths. Host-side telemetry only: it changes no "
        "artifact, budget, or result, is never read back by any session, and is "
        "not a research-notes or results channel. At most "
        f"{MAX_ISSUE_REPORTS_PER_SESSION} reports per session.",
        {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(ISSUE_CATEGORIES)},
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_ISSUE_SUMMARY_CHARS,
                },
                "evidence": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_ISSUE_EVIDENCE_CHARS,
                },
            },
            "required": ["category", "summary", "evidence"],
            "additionalProperties": False,
        },
        mutating=False,
        example={
            "category": "tool_output",
            "summary": "batch_validate 的失败行把宿主争用错误报成候选代码异常。",
            "evidence": (
                "同一候选目录连续两次提交：第一次整批返回 candidate error，"
                "第二次原样通过；失败行的原文是超时而非策略栈异常。"
                "复现：candidates/c1 目录未改动，间隔 5 分钟重跑。"
            ),
        },
    )

    def __init__(self, path: str | Path, manifest: RunManifest) -> None:
        self.path = Path(path)
        self.manifest = manifest
        # Concurrency-safe cap: the spec is non-mutating, so calls in one
        # assistant turn may dispatch in parallel. The counter is per tool
        # instance, which is per run: a retried session is a new run with its
        # own run_id, so the ledger holds no line this counter could inherit.
        self._lock = threading.Lock()
        self._filed = 0

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            category = str(arguments["category"]).strip()
            if category not in ISSUE_CATEGORIES:
                raise ValueError(
                    "category must be one of " + ", ".join(ISSUE_CATEGORIES)
                )
            summary = str(arguments["summary"]).strip()
            if not summary:
                raise ValueError("summary must state the issue")
            if len(summary) > MAX_ISSUE_SUMMARY_CHARS:
                raise ValueError(
                    f"summary exceeds {MAX_ISSUE_SUMMARY_CHARS} characters"
                )
            evidence = str(arguments["evidence"]).strip()
            if not evidence:
                raise ValueError("evidence must give the concrete reproduction")
            if len(evidence) > MAX_ISSUE_EVIDENCE_CHARS:
                raise ValueError(
                    f"evidence exceeds {MAX_ISSUE_EVIDENCE_CHARS} characters"
                )
            with self._lock:
                if self._filed >= MAX_ISSUE_REPORTS_PER_SESSION:
                    raise ValueError(
                        "issue report cap reached: this session already filed "
                        f"{MAX_ISSUE_REPORTS_PER_SESSION} reports"
                    )
                record: dict[str, object] = {
                    "report_id": new_id("issue"),
                    **session_link_fields(self.manifest),
                    "category": category,
                    "summary": summary,
                    "evidence": evidence,
                }
                recorded = append_issue_report(self.path, record)
                self._filed += 1
                filed = self._filed
        except ValueError as exc:
            raise ToolError(str(exc), error_type="issue_report_policy") from exc
        return ToolResult(
            True,
            value={
                "report_id": record["report_id"],
                "category": category,
                "recorded_at": recorded["recorded_at"],
                "reports_this_session": filed,
                "max_reports_per_session": MAX_ISSUE_REPORTS_PER_SESSION,
            },
        )


__all__ = [
    "ISSUE_CATEGORIES",
    "ISSUE_OUTCOMES",
    "ISSUE_REPORTS_NAME",
    "ISSUE_REPORT_LABEL",
    "ISSUE_REPORT_SCHEMA_VERSION",
    "MAX_ISSUE_EVIDENCE_CHARS",
    "MAX_ISSUE_REPORTS_PER_SESSION",
    "MAX_ISSUE_SUMMARY_CHARS",
    "ReportIssueTool",
    "append_issue_report",
    "append_issue_resolution",
    "issue_reports_path",
    "read_issue_reports",
]
