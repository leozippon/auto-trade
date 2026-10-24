"""TuShare raw-file IO helpers."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The sidecar read contract and the footer identity are owned by the
# environment's PIT layer.
from autotrade.environment.data.pit import (
    WRITE_ID_METADATA_KEY,
    CorruptSidecarError,
    concat_rows,
    parquet_meta,
    parquet_write_id,
)

_unique_jsonl_lock = threading.Lock()
_unique_jsonl_state: dict[tuple[Path, str], tuple[int, int, int, set[str]]] = {}
def frames_content_equal(old_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
    """Order-insensitive whole-frame equality over the union of columns.

    A differing column SET is never equal: padding both sides would let a
    schema change (added/dropped column with empty-string values) pass as
    identical content."""
    if len(old_df) != len(new_df):
        return False
    if set(old_df.columns) != set(new_df.columns):
        return False
    columns = sorted(set(old_df.columns) | set(new_df.columns))

    def canon(df: pd.DataFrame) -> list[tuple[str, ...]]:
        normalized = df.copy()
        for column in columns:
            if column not in normalized.columns:
                normalized[column] = ""
        return sorted(normalized[columns].astype(str).itertuples(index=False, name=None))

    return canon(old_df) == canon(new_df)


def write_parquet(
    path: Path,
    df: pd.DataFrame,
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    extra_metadata: dict[str, Any] | None = None,
    write_id: str | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one Parquet/sidecar pair with a shared UUID identity.

    The parquet and the sidecar are each staged under a temporary name and
    renamed into place, parquet first. ``at`` is the commit instant recorded
    as ``fetched_at`` (now when omitted); ``write_id`` the commit identity
    (fresh when omitted). The versioned write path (``versions.publish``)
    supplies both so that what it records about the commit agrees with it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    write_id = write_id or str(uuid.uuid4())
    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[WRITE_ID_METADATA_KEY] = write_id.encode("ascii")
    table = table.replace_schema_metadata(schema_metadata)
    tmp = path.with_name(f".{path.name}.{write_id}.tmp")
    pq.write_table(table, tmp)
    meta = {
        "api_name": api_name,
        "params": dict(params),
        "fields": list(fields),
        "row_count": len(df),
        "write_id": write_id,
        "fetched_at": at or datetime.now(UTC).isoformat(),
        "format": "parquet",
    }
    if extra_metadata:
        overlap = set(meta).intersection(extra_metadata)
        if overlap:
            raise ValueError(f"extra metadata cannot override commit fields: {sorted(overlap)}")
        meta.update(extra_metadata)
    meta_tmp = meta_path.with_name(f".{meta_path.name}.{write_id}.tmp")
    meta_tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    os.replace(meta_tmp, meta_path)
    return meta


def append_jsonl_unique(path: Path, payload: dict[str, Any], *, key: str) -> bool:
    """Append one logical record once, without rescanning on every write."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"unique JSONL record requires a non-empty string {key!r}")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = (path, key)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"

    def identity(record: dict[str, Any]) -> str:
        if key != "event_id":
            return str(record.get(key) or "")
        stable = {
            field: item
            for field, item in record.items()
            if field not in {"event_id", "detected_at", "write_id"}
        }
        return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    logical_value = identity(payload)
    with _unique_jsonl_lock, path.open("a+", encoding="utf-8") as handle:
        flock(handle.fileno(), LOCK_EX)
        try:
            stat = os.fstat(handle.fileno())
            state = _unique_jsonl_state.get(cache_key)
            if state is None or state[:2] != (stat.st_dev, stat.st_ino) or stat.st_size < state[2]:
                offset, values = 0, set()
            else:
                offset, values = state[2], state[3]
            handle.seek(offset)
            while line := handle.readline():
                try:
                    existing_record = json.loads(line)
                except (json.JSONDecodeError, AttributeError):
                    continue
                if isinstance(existing_record, dict):
                    existing = identity(existing_record)
                    if existing:
                        values.add(existing)
            offset = os.fstat(handle.fileno()).st_size
            if logical_value in values:
                _unique_jsonl_state[cache_key] = (stat.st_dev, stat.st_ino, offset, values)
                return False
            handle.write(encoded)
            handle.flush()
            values.add(logical_value)
            offset = os.fstat(handle.fileno()).st_size
            _unique_jsonl_state[cache_key] = (stat.st_dev, stat.st_ino, offset, values)
            return True
        finally:
            flock(handle.fileno(), LOCK_UN)


def read_many(files: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in files]
    return concat_rows(frames) if frames else pd.DataFrame()


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def committed_partition_intact(path: Path) -> bool:
    """True when footer and sidecar identify the same complete committed write."""
    try:
        meta = parquet_meta(path)
    except CorruptSidecarError:
        return False
    if not meta or type(meta.get("row_count")) is not int:
        return False
    footer_id = parquet_write_id(path)
    sidecar_id = str(meta.get("write_id") or "")
    try:
        footer_uuid = uuid.UUID(footer_id)
        row_count = parquet_rows(path)
    except (ValueError, AttributeError, OSError, pa.ArrowException):
        return False
    return bool(
        str(footer_uuid) == footer_id
        and sidecar_id == footer_id
        and meta["row_count"] == row_count
    )


def has_pagination_probe(path: Path) -> bool:
    """Whether the sidecar proves this partition was paged past one page.

    A corrupt sidecar cannot prove a probe: answer False (the caller then
    conservatively flags exact-limit row counts as cap risk) instead of
    crashing the whole audit -- the corruption itself is reported as an
    error finding by the dedicated sidecar-inventory check."""
    try:
        meta = parquet_meta(path)
    except CorruptSidecarError:
        return False
    pagination = (meta.get("params") or {}).get("pagination") or {}
    return int(pagination.get("pages") or 0) > 1


UPDATER_LOCK_FD_ENV = "TUSHARE_UPDATE_LOCK_FD"


def updater_lock_path(repo_root: Path) -> Path:
    """The one updater lock file the cron runner flocks exclusively."""
    return repo_root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"


def inherited_updater_lock_fd(repo_root: Path) -> int | None:
    """The runner's updater-lock fd this process inherited, or None.

    The cron runner holds the exclusive updater flock and passes that very file
    descriptor to its children (``pass_fds``) plus its number in
    ``TUSHARE_UPDATE_LOCK_FD``. A child proves it really is a runner child by
    showing all three:

      1. the named fd is open in THIS process,
      2. it refers to the same file as the updater lock path (dev+ino), and
      3. that lock is currently held exclusively (a fresh shared, non-blocking
         probe fails).

    An environment variable alone proves nothing, and neither does "somebody
    holds the lock" -- a manual run could hold it itself. Honest boundary: an
    operator who deliberately opens the lock file, flocks it exclusively and
    exports the fd number has reimplemented the runner; this is a discipline
    boundary against accident and drift, not a security boundary against a
    determined local user.
    """
    raw = os.environ.get(UPDATER_LOCK_FD_ENV, "")
    if not raw.isdigit():
        return None
    fd = int(raw)
    lock_path = updater_lock_path(repo_root)
    try:
        inherited = os.fstat(fd)
        on_disk = os.stat(str(lock_path))
    except OSError:
        return None
    if (inherited.st_dev, inherited.st_ino) != (on_disk.st_dev, on_disk.st_ino):
        return None
    try:
        probe = os.open(str(lock_path), os.O_RDONLY)
    except OSError:
        return None
    try:
        flock(probe, LOCK_SH | LOCK_NB)
    except BlockingIOError:
        return fd  # exclusively held: the runner's lock is live
    else:
        flock(probe, LOCK_UN)
        return None
    finally:
        os.close(probe)
