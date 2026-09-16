"""One bounded page of the Agent's feedback ledgers, for the researcher.

``report_issue`` and ``skill_feedback`` are two channels of the same thing: an
append-only JSONL per experiment that the researcher answers with a resolution
line. Their listings therefore obey one rule — newest first, one bounded page,
resolved reports out of the default view but counted either way, and an
experiment whose log cannot be read named rather than silently dropped. That
rule lives here once; each channel supplies only its own per-record projection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .registry import resolve_experiment_dir


def feedback_page(
    experiments_root: Path,
    *,
    project: Callable[[Path], list[dict[str, object]]],
    unreadable_error: str,
    experiment_id: str | None,
    include_resolved: bool,
    limit: int,
    max_limit: int,
) -> dict[str, object]:
    """Reports across experiments (or one), newest first, bounded to one page.

    ``total`` counts what the listing is drawn from, so it follows the resolved
    filter; ``resolved`` counts the resolved reports in scope either way, which
    is what tells an empty open list apart from an experiment that never filed
    anything.
    """

    if not 1 <= limit <= max_limit:
        raise ValueError(f"limit must be between 1 and {max_limit}")
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
            reports.extend(project(directory))
        except (OSError, TypeError, ValueError) as exc:
            unreadable.append(
                {
                    "experiment_id": directory.name,
                    "error": f"{type(exc).__name__}: {unreadable_error}",
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


__all__ = ["feedback_page"]
