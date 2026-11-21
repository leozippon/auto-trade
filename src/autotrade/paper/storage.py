"""Crash-tolerant journals of the local Paper account.

Paper's JSON object files go through ``autotrade.common.records`` like every
other record: ``read_json_object(missing_ok=True)`` and owner-only
``write_json_atomic(private=True)``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

# The account's journals: one file per session and family
# (``<prefix><YYYYMMDD>.jsonl``) -- what each decision ordered, what each
# session filled and each session's ex-date settlements -- and the one equity
# journal, a row per settled session.
ORDERS_PREFIX = "orders_"
EXECUTIONS_PREFIX = "executions_"
CORPORATE_ACTIONS_PREFIX = "corporate_actions_"
EQUITY_JOURNAL_NAME = "equity_daily.jsonl"


def journal_name(prefix: str, day: str) -> str:
    """The file of one session's journal of a family."""

    return f"{prefix}{day}.jsonl"


def journal_dates(root: Path, prefix: str) -> list[str]:
    """The sessions one journal family holds a file for, oldest first."""

    return sorted(
        day
        for path in Path(root).glob(f"{prefix}[0-9]*.jsonl")
        if len(day := path.stem.removeprefix(prefix)) == 8 and day.isdigit()
    )


def read_jsonl(path: Path) -> tuple[list[dict[str, object]], int]:
    """Tolerant reader for the Paper JSONL journals (orders/executions/equity).

    Single parse of the on-disk journal contract shared by the paper engine and
    the console read-model. Returns ``(records, skipped_lines)``: a
    missing/unreadable file reads as empty (normal before the first write);
    non-JSON or non-object lines are counted, never fatal. A journal is an
    append-only stream that a crash can truncate mid-line, so one damaged line
    must not withhold every good record before it — the count is the report.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 0
    rows: list[dict[str, object]] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            skipped += 1
    return rows, skipped


def append_jsonl_once(path: Path, payload: dict[str, object], *, identity_key: str = "event_id") -> None:
    identity = payload.get(identity_key)
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"journal payload requires {identity_key}")
    rows, _skipped = read_jsonl(path)
    if any(row.get(identity_key) == identity for row in rows):
        return
    append_jsonl(path, [payload])


def append_jsonl(path: Path, payloads: Sequence[dict[str, object]]) -> None:
    """Append ``payloads`` as JSON lines in one write, fsynced: lines that
    belong together land together instead of one at a time."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = b"".join(
        json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8") + b"\n"
        for payload in payloads
    )
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


__all__ = [
    "CORPORATE_ACTIONS_PREFIX",
    "EQUITY_JOURNAL_NAME",
    "EXECUTIONS_PREFIX",
    "ORDERS_PREFIX",
    "append_jsonl",
    "append_jsonl_once",
    "journal_dates",
    "journal_name",
    "read_jsonl",
]
