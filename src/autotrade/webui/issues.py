"""Console projection of Agent-filed issue reports and their resolutions.

Sessions file suspected environment/tool/data/docs defects with the
``report_issue`` tool; each report is one redacted JSON line in the owning
experiment's ``ledgers/issue_reports.jsonl``. This module is the only place
they are read back together — for the researcher, never for a session. Like
every other projection, Agent-authored text leaves the host through
:class:`PublicIdentity`, and an experiment whose log cannot be read is named
rather than silently dropped.

A report the researcher has answered (``scripts/experiments/resolve_issue.py``
appends the resolution line) carries its outcome and note here; the listing
itself follows the one rule in :mod:`autotrade.webui.feedback_page`.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.tools.report_issue import (
    ISSUE_CATEGORIES,
    issue_reports_path,
    read_issue_reports,
)

from .feedback_page import feedback_page
from .public_identity import PublicIdentity

# One bounded page, newest first, is the whole surface. It is a real cut, not a
# formality: at 16 reports per session, a round of five experiments running
# roughly 24 parent sessions each can file hundreds. ``total`` says how many
# matched, so the console shows "最近 N 条，共 M 条" rather than implying the
# page is everything.
MAX_ISSUE_REPORTS_PAGE = 200
_UNREADABLE_REPORTS = "issue reports are unreadable"


def _experiment_reports(directory: Path) -> list[dict[str, object]]:
    """One experiment's reports, projected past the host boundary."""

    records = read_issue_reports(issue_reports_path(directory))
    if not records:
        return []
    identity = PublicIdentity(directory)
    rows: list[dict[str, object]] = []
    for record in records:
        category = str(record.get("category") or "")
        if category not in ISSUE_CATEGORIES:
            # The version gate already passed, so within schema 1 an unknown
            # category is corruption; name it instead of guessing a label.
            raise ValueError(f"unknown issue category: {category!r}")
        label = ""
        raw_session = str(record.get("session_key") or "")
        if raw_session:
            try:
                label = identity.public_session_key(raw_session)
            except (KeyError, ValueError):
                # A key the plan does not name still counts, just unlabelled.
                label = ""
        rows.append(
            {
                "experiment_id": directory.name,
                "report_id": str(record.get("report_id") or ""),
                "category": category,
                "kind": str(record.get("kind") or ""),
                "session_label": label,
                # Agent-authored text leaves the host through the same
                # projection as every other traced string.
                "summary": identity.public_text(str(record.get("summary") or "")),
                "evidence": identity.public_text(str(record.get("evidence") or "")),
                "recorded_at": str(record.get("recorded_at") or ""),
                # The resolution is the researcher's own text, written on the
                # host and read on the host: it carries no Agent identity and
                # is shown as written, commit hashes and paths included.
                "resolved_at": str(record.get("resolved_at") or ""),
                "outcome": str(record.get("outcome") or ""),
                "resolution": str(record.get("resolution") or ""),
            }
        )
    return rows


def issue_reports(
    experiments_root: Path,
    *,
    experiment_id: str | None = None,
    include_resolved: bool = False,
    limit: int = MAX_ISSUE_REPORTS_PAGE,
) -> dict[str, object]:
    """This channel's page of :func:`feedback_page`."""

    return feedback_page(
        experiments_root,
        project=_experiment_reports,
        unreadable_error=_UNREADABLE_REPORTS,
        experiment_id=experiment_id,
        include_resolved=include_resolved,
        limit=limit,
        max_limit=MAX_ISSUE_REPORTS_PAGE,
    )


__all__ = ["MAX_ISSUE_REPORTS_PAGE", "issue_reports"]
