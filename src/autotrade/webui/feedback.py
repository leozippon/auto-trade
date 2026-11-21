"""The researcher's two inboxes of Agent feedback, one page each.

Sessions file suspected environment, tool, data or docs defects with the
``report_issue`` tool (``ledgers/issue_reports.jsonl``) and contradict a mounted
skill-library entry with ``skill_feedback`` (``ledgers/skill_feedback.jsonl``):
two channels of the same thing, an append-only redacted JSONL per experiment
that the researcher answers with a resolution line
(``scripts/experiments/resolve_feedback.py issue|skill``). This module is the
only place they are read back together -- for the researcher, never for a
session -- and both listings obey one rule: newest first, one bounded page,
resolved reports out of the default view but counted either way, and an
experiment whose log cannot be read named rather than silently dropped. Each
channel supplies only its own fields. Agent-authored text leaves the host
through :class:`PublicIdentity`, like every other projection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from autotrade.environment.tools.report_issue import (
    ISSUE_CATEGORIES,
    issue_reports_path,
    read_issue_reports,
)
from autotrade.environment.tools.skill_feedback import (
    SKILL_CLAIMS,
    read_skill_feedback,
    skill_feedback_path,
)
from autotrade.pipelines.skills import validate_memory_entry_ref

from .public_identity import PublicIdentity
from .registry import experiment_directories, resolve_experiment_dir

# One bounded page, newest first, is the whole surface. It is a real cut for
# the issue reports, not a formality: at 16 reports per session, a round of
# five experiments running roughly 24 parent sessions each can file hundreds.
# ``total`` says how many matched, so the console shows "最近 N 条，共 M 条"
# rather than implying the page is everything.
FEEDBACK_PAGE = 200


def _issue_fields(record: Mapping[str, object]) -> dict[str, object]:
    category = str(record.get("category") or "")
    if category not in ISSUE_CATEGORIES:
        # The version gate already passed, so within schema 1 an unknown
        # category is corruption; name it instead of guessing a label.
        raise ValueError(f"unknown issue category: {category!r}")
    return {"category": category}


def _skill_fields(record: Mapping[str, object]) -> dict[str, object]:
    claim = str(record.get("claim") or "")
    if claim not in SKILL_CLAIMS:
        raise ValueError(f"unknown skill feedback claim: {claim!r}")
    skill = str(record.get("skill") or "")
    # The stored reference is the one fact; the mount's own parser splits it
    # so the page can never address an entry differently from the session.
    source, name = validate_memory_entry_ref(skill)
    origin = str(record.get("origin") or "")
    if origin != "curated":
        raise ValueError(f"unknown skill feedback origin: {origin!r}")
    return {"claim": claim, "skill": skill, "source": source, "name": name, "origin": origin}


def _session_label(identity: PublicIdentity, record: Mapping[str, object]) -> str:
    raw = str(record.get("session_key") or "")
    if not raw:
        return ""
    try:
        return identity.public_session_key(raw)
    except (KeyError, ValueError):
        return ""  # a key the plan does not name still counts, just unlabelled


def _projected(
    directory: Path,
    records: list[dict[str, object]],
    fields: Callable[[Mapping[str, object]], dict[str, object]],
    texts: tuple[str, ...],
) -> list[dict[str, object]]:
    """One experiment's reports, projected past the host boundary: the
    channel's own ``fields`` and its Agent-authored ``texts`` around what every
    report carries."""

    if not records:
        return []
    identity = PublicIdentity(directory)
    return [
        {
            "experiment_id": directory.name,
            "report_id": str(record.get("report_id") or ""),
            **fields(record),
            "kind": str(record.get("kind") or ""),
            "session_label": _session_label(identity, record),
            # Agent-authored text leaves the host through the same projection
            # as every other traced string.
            **{key: identity.public_text(str(record.get(key) or "")) for key in texts},
            "recorded_at": str(record.get("recorded_at") or ""),
            # The resolution is the researcher's own text, written on the host
            # and read on the host: it carries no Agent identity and is shown
            # as written, commit hashes and paths included.
            "resolved_at": str(record.get("resolved_at") or ""),
            "outcome": str(record.get("outcome") or ""),
            "resolution": str(record.get("resolution") or ""),
        }
        for record in records
    ]


def _page(
    experiments_root: Path,
    project: Callable[[Path], list[dict[str, object]]],
    unreadable_error: str,
    experiment_id: str | None,
    include_resolved: bool,
) -> dict[str, object]:
    """Reports across experiments (or one), newest first, one page.

    ``total`` counts what the listing is drawn from, so it follows the resolved
    filter; ``resolved`` counts the resolved reports in scope either way, which
    is what tells an empty open list apart from an experiment that never filed
    anything.
    """

    root = Path(experiments_root)
    reports: list[dict[str, object]] = []
    unreadable: list[dict[str, object]] = []
    directories = (
        [resolve_experiment_dir(root, experiment_id)] if experiment_id is not None else experiment_directories(root)
    )
    for directory in directories:
        try:
            reports.extend(project(directory))
        except (OSError, TypeError, ValueError) as exc:
            unreadable.append({"experiment_id": directory.name, "error": f"{type(exc).__name__}: {unreadable_error}"})
    reports.sort(key=lambda item: str(item.get("recorded_at")), reverse=True)
    resolved = sum(1 for item in reports if item["outcome"])
    if not include_resolved:
        reports = [item for item in reports if not item["outcome"]]
    return {
        "reports": reports[:FEEDBACK_PAGE],
        "total": len(reports),
        "resolved": resolved,
        "limit": FEEDBACK_PAGE,
        "unreadable": unreadable,
    }


def issue_reports(
    experiments_root: Path, *, experiment_id: str | None = None, include_resolved: bool = False
) -> dict[str, object]:
    return _page(
        experiments_root,
        lambda directory: _projected(
            directory, read_issue_reports(issue_reports_path(directory)), _issue_fields, ("summary", "evidence")
        ),
        "issue reports are unreadable",
        experiment_id,
        include_resolved,
    )


def skill_feedback(
    experiments_root: Path, *, experiment_id: str | None = None, include_resolved: bool = False
) -> dict[str, object]:
    return _page(
        experiments_root,
        lambda directory: _projected(
            directory, read_skill_feedback(skill_feedback_path(directory)), _skill_fields, ("evidence",)
        ),
        "skill feedback is unreadable",
        experiment_id,
        include_resolved,
    )


__all__ = ["FEEDBACK_PAGE", "issue_reports", "skill_feedback"]
