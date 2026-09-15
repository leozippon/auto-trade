"""The append-only shape both Agent→operator feedback channels are written in.

``report_issue`` files a suspected defect in the environment, a tool's output,
the mounted data or the mounted docs; ``skill_feedback`` contradicts a mounted
operating-memory entry. Both write one redacted, version-stamped JSON line per
report into the owning experiment's ledger, and the researcher answers in the
same log with a ``resolution`` line naming the report. Nothing is ever
rewritten — the filed report stays exactly as the session wrote it, and the
console joins the two on read.

Only the ledger's schema version, its record label and its outcome vocabulary
differ between the two channels; everything below is the shared mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from autotrade.environment.runtime import (
    RunManifest,
    append_versioned_jsonl,
    read_versioned_jsonl,
)

# Link keys copied verbatim from the host run manifest when present, so a
# report correlates with the trace and ledger records of the run that filed it.
SESSION_LINK_KEYS = (
    "experiment_id",
    "epoch_id",
    "fold_id",
    "run_id",
    "session_key",
    "kind",
)
# Within a schema version this key is the discriminator: a line without it is a
# report, because every line written before resolutions existed is one.
RESOLUTION_RECORD_TYPE = "resolution"
MAX_RESOLUTION_NOTE_CHARS = 500


def session_link_fields(manifest: RunManifest) -> dict[str, str]:
    """The run's identity as a filed report carries it."""

    return {
        key: str(manifest.get(key)) for key in SESSION_LINK_KEYS if manifest.get(key)
    }


def read_feedback_log(
    path: str | Path,
    *,
    schema_version: int,
    label: str,
    outcomes: Sequence[str],
) -> list[dict[str, object]]:
    """All reports in append order, each carrying its resolution when it has one.

    A resolved report gains ``resolved_at`` (the resolution line's own log
    stamp, so the two can never disagree), ``outcome`` and ``resolution``. A
    foreign or newer format, or a resolution that names no report or the same
    report twice, fails fast: within one schema version that is corruption, not
    history.
    """

    records = read_versioned_jsonl(path, schema_version=schema_version, label=label)
    reports = [
        record
        for record in records
        if record.get("record_type") != RESOLUTION_RECORD_TYPE
    ]
    by_id = {str(record.get("report_id") or ""): record for record in reports}
    for record in records:
        if record.get("record_type") != RESOLUTION_RECORD_TYPE:
            continue
        report_id = str(record.get("report_id") or "")
        report = by_id.get(report_id) if report_id else None
        if report is None:
            raise ValueError(f"resolution names no filed {label}: {report_id!r}")
        if report.get("outcome"):
            raise ValueError(f"{label} resolved twice: {report_id}")
        outcome = str(record.get("outcome") or "")
        if outcome not in outcomes:
            raise ValueError(f"unknown {label} outcome: {outcome!r}")
        report["resolved_at"] = record.get("recorded_at")
        report["outcome"] = outcome
        report["resolution"] = record.get("note")
    return reports


def append_resolution(
    path: str | Path,
    *,
    schema_version: int,
    label: str,
    outcomes: Sequence[str],
    report_id: str,
    outcome: str,
    note: str,
) -> dict[str, object]:
    """Record how one filed report was answered; refuse anything else.

    The write goes through the same versioned append — and therefore the same
    ``flock`` — as a report, so a session filing a report while the researcher
    resolves one cannot interleave with this line.
    """

    if outcome not in outcomes:
        raise ValueError("outcome must be one of " + ", ".join(outcomes))
    text = note.strip()
    if not text:
        raise ValueError("note must say how the report was answered")
    if len(text) > MAX_RESOLUTION_NOTE_CHARS:
        raise ValueError(f"note exceeds {MAX_RESOLUTION_NOTE_CHARS} characters")
    reports = read_feedback_log(
        path, schema_version=schema_version, label=label, outcomes=outcomes
    )
    match = next(
        (item for item in reports if str(item.get("report_id") or "") == report_id),
        None,
    )
    if match is None:
        raise ValueError(f"unknown {label}: {report_id!r}")
    if match.get("outcome"):
        raise ValueError(
            f"{label} {report_id} is already resolved as {match['outcome']}"
        )
    return append_versioned_jsonl(
        path,
        {
            "record_type": RESOLUTION_RECORD_TYPE,
            "report_id": report_id,
            "outcome": outcome,
            "note": text,
        },
        schema_version=schema_version,
    )


__all__ = [
    "MAX_RESOLUTION_NOTE_CHARS",
    "RESOLUTION_RECORD_TYPE",
    "SESSION_LINK_KEYS",
    "append_resolution",
    "read_feedback_log",
    "session_link_fields",
]
