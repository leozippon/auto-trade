"""Crash-tolerant journals of the local Paper account.

Paper's JSON object files go through ``autotrade.common.records`` like every
other record: ``read_json_object(missing_ok=True)`` and owner-only
``write_json_atomic(private=True)``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


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
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8") + b"\n"
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


__all__ = ["append_jsonl_once", "read_jsonl"]
