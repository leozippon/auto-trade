"""Console projection of Agent-filed skill feedback and its resolutions.

Sessions contradict a mounted operating-memory entry with the ``skill_feedback``
tool; each report is one redacted JSON line in the owning experiment's
``ledgers/skill_feedback.jsonl``. This module is the only place they are read
back together — for the researcher, never for a session. Like every other
projection, Agent-authored text leaves the host through :class:`PublicIdentity`,
and an experiment whose log cannot be read is named rather than silently
dropped.

A report the researcher has answered (``scripts/experiments/resolve_skill_feedback.py``
appends the resolution line) carries its outcome and note here. Resolved reports
leave the default page: what is still open is the working list, and the resolved
count says how much the toggle would add back.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.tools.skill_feedback import (
    SKILL_CLAIMS,
    read_skill_feedback,
    skill_feedback_path,
)
from autotrade.pipelines.skills import validate_memory_entry_ref

from .public_identity import PublicIdentity
from .registry import resolve_experiment_dir

# One bounded page, newest first, like the issue reports beside it. The channel
# is far quieter — one report per mounted entry per session — so this cap is a
# ceiling rather than a working limit; ``total`` still says how many matched.
MAX_SKILL_FEEDBACK_PAGE = 200
_UNREADABLE_FEEDBACK = "skill feedback is unreadable"


def _experiment_feedback(directory: Path) -> list[dict[str, object]]:
    """One experiment's reports, projected past the host boundary."""

    records = read_skill_feedback(skill_feedback_path(directory))
    if not records:
        return []
    identity = PublicIdentity(directory)
    rows: list[dict[str, object]] = []
    for record in records:
        claim = str(record.get("claim") or "")
        if claim not in SKILL_CLAIMS:
            # The version gate already passed, so within schema 1 an unknown
            # claim is corruption; name it instead of guessing a label.
            raise ValueError(f"unknown skill feedback claim: {claim!r}")
        skill = str(record.get("skill") or "")
        # The stored reference is the one fact; the mount's own parser splits it
        # so the page can never address an entry differently from the session.
        source, name = validate_memory_entry_ref(skill)
        origin = str(record.get("origin") or "")
        if origin not in {"curated", "graduated"}:
            raise ValueError(f"unknown skill feedback origin: {origin!r}")
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
                "claim": claim,
                "skill": skill,
                "source": source,
                "name": name,
                "origin": origin,
                "kind": str(record.get("kind") or ""),
                "session_label": label,
                # Agent-authored text leaves the host through the same
                # projection as every other traced string.
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


def skill_feedback(
    experiments_root: Path,
    *,
    experiment_id: str | None = None,
    include_resolved: bool = False,
    limit: int = MAX_SKILL_FEEDBACK_PAGE,
) -> dict[str, object]:
    """Reports across experiments (or one), newest first, bounded to one page.

    ``total`` counts what the listing is drawn from, so it follows the resolved
    filter; ``resolved`` counts the resolved reports in scope either way, which
    is what tells an empty open list apart from an experiment that never filed
    anything.
    """

    if not 1 <= limit <= MAX_SKILL_FEEDBACK_PAGE:
        raise ValueError(f"limit must be between 1 and {MAX_SKILL_FEEDBACK_PAGE}")
    root = Path(experiments_root)
    reports: list[dict[str, object]] = []
    unreadable: list[dict[str, object]] = []
    if experiment_id is not None:
        directories = [resolve_experiment_dir(root, experiment_id)]
    elif root.is_dir():
        directories = sorted(
            (
                directory
                for directory in root.iterdir()
                if directory.is_dir() and not directory.name.startswith(".")
            ),
            key=lambda path: path.name,
        )
    else:
        directories = []
    for directory in directories:
        try:
            reports.extend(_experiment_feedback(directory))
        except (OSError, TypeError, ValueError) as exc:
            unreadable.append(
                {
                    "experiment_id": directory.name,
                    "error": f"{type(exc).__name__}: {_UNREADABLE_FEEDBACK}",
                }
            )
    reports.sort(key=lambda item: str(item.get("recorded_at")), reverse=True)
    resolved = sum(1 for item in reports if item["outcome"])
    if not include_resolved:
        reports = [item for item in reports if not item["outcome"]]
    return {
        "reports": reports[:limit],
        "total": len(reports),
        "resolved": resolved,
        "limit": limit,
        "unreadable": unreadable,
    }


__all__ = ["MAX_SKILL_FEEDBACK_PAGE", "skill_feedback"]
