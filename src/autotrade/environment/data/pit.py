from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.environment.data.contracts import CN_TZ, LOCALLY_DERIVED_DATASETS


def concat_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Row-union for parquet-sourced frames — deliberately NOT a ``pd.concat``
    drop-in: always a fresh RangeIndex over the outer column union, column
    order follows the non-empty inputs (schema-only columns from empty inputs
    append after), and no join/axis/key options exist.

    Empty inputs are excluded from the concat (their inclusion in dtype
    inference is deprecated and warns per call) but still contribute schema.
    Real frames may carry all-NA columns — genuine sparse schema, not
    removable — for which pandas' future behavior (typed all-NA columns
    participate in inference) equals today's outcome because the columns are
    parquet-typed; that residual FutureWarning is accepted noise and
    suppressed at this single boundary."""
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        # Preserve the column union and dtypes of the empty inputs instead of
        # collapsing to a columnless frame (writers persist the schema; readers
        # select columns on the result).
        columns: dict[str, object] = {}
        for frame in frames:
            for column in frame.columns:
                columns.setdefault(column, frame[column].dtype)
        return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in columns.items()})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*empty or all-NA entries.*")
        merged = pd.concat(non_empty, ignore_index=True, sort=False)
    # Empty inputs still contribute schema: columns only they carry are added
    # as typed all-NA via reindex (integer dtypes promote to float, exactly as
    # a plain concat would have produced).
    for frame in frames:
        if frame.empty:
            for column in frame.columns:
                if column not in merged.columns:
                    merged[column] = pd.Series(dtype=frame[column].dtype).reindex(merged.index)
    return merged


class CorruptSidecarError(RuntimeError):
    """A parquet sidecar exists but cannot be parsed: torn-write evidence."""


# A keyed raw partition ``<name>.parquet`` may carry an observation table
# beside it, ``<name>.parquet.first_seen``: the instant each business key first
# became part of the visible partition (written by the ingest adapter's
# versioned write path, data_sources/tushare/versions.py; read here).
FIRST_SEEN_SUFFIX = ".first_seen"
FIRST_SEEN_COLUMN = "first_seen_at"


def parquet_meta(path: Path) -> dict[str, Any]:
    """Read a raw-lake parquet's ``<file>.meta.json`` sidecar (empty if absent).

    The sidecar scheme is part of the raw-lake contract: the ingest adapter
    writes it, PIT consumers read it through this single helper. An ABSENT
    sidecar reads as {} (pre-scheme files); a PRESENT but unparseable one is
    torn-write evidence and raises -- returning {} here made corruption
    indistinguishable from absence. The writer stamps one UUID ``write_id``
    into the Parquet footer and sidecar so the audit can detect a torn pair
    without reading the file contents."""
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorruptSidecarError(f"unreadable parquet sidecar: {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorruptSidecarError(f"parquet sidecar is not a JSON object: {meta_path}")
    return payload


WRITE_ID_METADATA_KEY = b"autotrade.write_id"


def parquet_write_id(path: Path) -> str:
    """Return the UUID stamped in a Parquet footer, or ``""`` for legacy files."""
    try:
        metadata = pq.read_metadata(path).metadata or {}
    except (OSError, pa.ArrowException):
        return ""
    value = metadata.get(WRITE_ID_METADATA_KEY, b"")
    try:
        return value.decode("ascii")
    except (AttributeError, UnicodeDecodeError):
        return ""


def complete_interrupted_commit(path: Path) -> None:
    """Roll forward a commit that renamed its parquet but not its sidecar.

    The writer stages the sidecar as ``.<sidecar>.<write_id>.tmp`` before the
    parquet rename, so the sidecar belonging to the visible parquet is still
    there when a crash fell between the two renames."""
    path = Path(path)
    if not path.exists():
        return
    footer = parquet_write_id(path)
    if not footer:
        return
    try:
        current = parquet_meta(path).get("write_id")
    except CorruptSidecarError:
        current = None
    if current == footer:
        return
    meta = path.with_suffix(path.suffix + ".meta.json")
    pending = meta.with_name(f".{meta.name}.{footer}.tmp")
    try:
        os.replace(pending, meta)
    except FileNotFoundError:
        pass


# What a writer stages beside a partition before its renames: the parquet, its
# observation table and its sidecar as ``.<file>.<token>.tmp``, and the raw
# generation record as ``.raw_generation.json.tmp<pid>``.
_STAGED_PARTITION_FILE = re.compile(r"^\.(?P<file>.+\.parquet(?:\.meta\.json|" + re.escape(FIRST_SEEN_SUFFIX) + r")?)\.(?P<token>[0-9A-Za-z-]+)\.tmp$")
_STAGED_GENERATION_FILE = re.compile(r"^\.raw_generation\.json\.tmp\d+$")


def complete_interrupted_commits(raw_dir: str | Path) -> dict[str, int]:
    """Finish or clear what crashed writers left anywhere in ``raw_dir``.

    A staged sidecar whose parquet is already visible completes its commit
    (:func:`complete_interrupted_commit`); every other staged file belongs to a
    write that never became visible and is removed. Call it only where no
    writer can be running -- under the updater lock, held exclusively or
    shared -- before reading partitions as consistent pairs: a leftover would
    otherwise refuse every release until its partition is written again.
    Idempotent; files of any other shape, and the directories of datasets
    written without the lock, are left for the reader to refuse."""
    finished = removed = 0
    root = Path(raw_dir)
    for directory, dirs, files in os.walk(root):
        if Path(directory) == root:
            dirs[:] = [name for name in dirs if name not in LOCALLY_DERIVED_DATASETS]
        for name in files:
            staged = _STAGED_PARTITION_FILE.match(name)
            if staged is None and not _STAGED_GENERATION_FILE.match(name):
                continue
            leftover = Path(directory) / name
            if staged is not None and staged["file"].endswith(".meta.json"):
                partition = Path(directory) / staged["file"][: -len(".meta.json")]
                complete_interrupted_commit(partition)
                if not leftover.exists():
                    finished += 1
                    continue
            try:
                leftover.unlink()
            except FileNotFoundError:
                continue
            removed += 1
    return {"finished": finished, "removed": removed}



# ---------------------------------------------------------------- observation tables
#
# The table holds the business-key columns in canonical text and
# ``first_seen_at``, a UTC instant. Keys visible before versioning started have
# no row. It is written before the partition it describes, so a row later than
# the partition's ``visible_since`` belongs to a write that never became
# visible and is not read. The PIT views make a row visible from the later of
# its vendor-rule ``available_at`` and the first-seen time of its key
# (docs/data-documentation.md §3.3).

KEY_SEPARATOR = "\x1f"


def canonical_revision_value(value: Any) -> str:
    """One text form per value, so a re-serialised number or an empty marker
    compares equal to the same value written another way."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError):
        return text
    if not decimal.is_finite():
        return text
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def canonical_series(series: pd.Series) -> pd.Series:
    values = series.astype(object)
    try:
        mapping = {value: canonical_revision_value(value) for value in pd.unique(values)}
        out = values.map(mapping)
    except TypeError:  # unhashable cells
        out = values.map(canonical_revision_value)
    return out.where(values.notna(), "").astype(str)


def canonical_keys(frame: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    """The business-key columns of ``frame`` in canonical text, the form the
    observation table stores; a missing key column reads as empty text."""
    return pd.DataFrame(
        {
            column: canonical_series(frame[column]) if column in frame.columns else pd.Series("", index=frame.index)
            for column in key_columns
        },
        index=frame.index,
    )


def join_columns(frame: pd.DataFrame, columns: Sequence[str], separator: str = KEY_SEPARATOR) -> pd.Series:
    if not columns:
        return pd.Series("", index=frame.index, dtype=object)
    first, *rest = columns
    if not rest:
        return frame[first].astype(str)
    return frame[first].str.cat([frame[column] for column in rest], sep=separator)


def key_strings(frame: pd.DataFrame, key_columns: Sequence[str]) -> pd.Series:
    return join_columns(canonical_keys(frame, key_columns), list(key_columns))


def observation_path(path: Path) -> Path:
    return path.with_name(path.name + FIRST_SEEN_SUFFIX)


def read_observations(path: Path, meta: dict[str, Any] | None = None) -> pd.DataFrame:
    """The observation rows of the visible partition at ``path``: its canonical
    business keys and ``first_seen_at``. Rows written by a commit that never
    became visible are left out; a partition without a table has none.

    The one rule for which rows of a table count, used by the versioned write
    path (which passes the sidecar ``meta`` it has already read) and by the PIT
    views alike."""
    path = Path(path)
    obs = observation_path(path)
    if not obs.exists():
        return pd.DataFrame(columns=[FIRST_SEEN_COLUMN])
    frame = pq.read_table(obs).to_pandas()
    if not path.exists():
        # A first landing that crashed before its parquet rename.
        return frame.iloc[:0]
    since = _required_visible_since(path, parquet_meta(path) if meta is None else meta, rows=len(frame))
    return _observations_up_to(frame, since)


def _required_visible_since(path: Path, meta: dict[str, Any], *, rows: int) -> datetime | None:
    """The visible partition's ``visible_since``. A table holding rows beside a
    partition whose sidecar does not say when it became visible (missing,
    unreadable or pre-versioning) cannot be filtered: refuse loudly rather than
    read or rewrite it without its recorded first-seen times."""
    try:
        since = datetime.fromisoformat(str(meta["visible_since"])) if meta.get("visible_since") else None
    except ValueError:
        since = None
    if since is None and rows:
        raise RuntimeError(
            f"{observation_path(path)} holds {rows} first-seen rows but the sidecar of {path} records no "
            "visible_since; repair the sidecar, or rebuild the table with "
            "`scripts/data/tushare_lake.py rebuild-observations`"
        )
    return since


def _observations_up_to(frame: pd.DataFrame, since: datetime | None) -> pd.DataFrame:
    """The rows of an observation table first seen no later than ``since``."""
    if frame.empty:
        return frame
    if since is None:
        return frame.iloc[:0]
    stamps = pd.to_datetime(frame[FIRST_SEEN_COLUMN], utc=True, format="ISO8601")
    return frame.loc[stamps <= pd.Timestamp(since)].reset_index(drop=True)


def _first_seen_cn(stamps: pd.Series) -> pd.Series:
    """Observation instants (UTC text) as CN timestamps rounded up to the
    second, the precision of every view stamp: rounding up never shows a row
    before the lake held it."""
    return pd.to_datetime(stamps, utc=True, format="ISO8601").dt.ceil("s").dt.tz_convert(CN_TZ)


def first_seen_later(path: Path, frame: pd.DataFrame, available_at: pd.Series) -> pd.Series | None:
    """When the lake first held each row of ``frame``, the visible partition at
    ``path``, where that is later than the row's vendor-rule ``available_at``.

    A CN timestamp per row, NaT where the key was visible before versioning
    started (no observation row) or was first seen no later than its stamp:
    such a row keeps the vendor rule. ``None`` when no row is first seen later."""
    observed = read_observations(path)
    if observed.empty:
        return None
    keys = [column for column in observed.columns if column != FIRST_SEEN_COLUMN]
    seen_at = _first_seen_cn(observed[FIRST_SEEN_COLUMN]).set_axis(
        pd.Index(join_columns(observed[keys].astype(str), keys))
    )
    seen = key_strings(frame, keys).map(seen_at)
    later = seen > to_cn_timestamps(available_at)
    return seen.where(later) if later.any() else None


def first_seen_between(path: Path, start: datetime, end: datetime) -> bool:
    """Whether the partition at ``path`` records a key first seen inside
    ``[start, end]``: such a partition can hold rows a view in that window
    shows, whatever dates its name covers."""
    obs = observation_path(Path(path))
    if not obs.exists():
        return False
    seen = _first_seen_cn(pq.read_table(obs, columns=[FIRST_SEEN_COLUMN]).column(0).to_pandas())
    return bool(((seen >= start) & (seen <= end)).any())


def visible_from(frame: pd.DataFrame, available: pd.Series | None = None) -> pd.Series:
    """When a view may first show each row of a frame that carries
    ``available_at`` and possibly ``first_seen_at``: the later of the two.
    ``available`` is the frame's ``available_at`` already parsed, if at hand."""
    if available is None:
        available = to_cn_timestamps(frame["available_at"])
    if FIRST_SEEN_COLUMN not in frame.columns:
        return available
    present = frame[FIRST_SEEN_COLUMN].notna()
    if not present.any():
        return available
    seen = to_cn_timestamps(frame.loc[present, FIRST_SEEN_COLUMN]).reindex(frame.index)
    return available.mask(seen > available, seen)


def stamp_text(stamps: pd.Series) -> pd.Series:
    """CN timestamps as the views' stamp text, ``None`` for NaT."""
    return stamps.dt.strftime("%Y-%m-%d %H:%M:%S+08:00").where(stamps.notna(), None)


def to_cn_timestamps(series: pd.Series) -> pd.Series:
    """Parse available_at values to Asia/Shanghai timestamps.

    Raw datasets mix tz-aware ISO strings (e.g. margin) and tz-naive Beijing
    wall-clock strings (e.g. anns_d rec_time); naive values must be localized
    to CN, never treated as UTC.

    Stamp forms are normalised first: the raw lake writes
    "YYYY-MM-DD HH:MM:SS+08:00", builder stamps write the "T" form, and a bare
    wall-clock value is CN by contract. A union column mixing forms must not
    lose any of them — pandas infers one format from the first value and
    coerces the others to NaT, which the PIT wall would then hide silently.

    Fast path next: replay/snapshot columns are usually uniform CN-offset
    strings, and pandas' generic tz-aware parser costs ~5µs/row — a quarter
    of minute bars is tens of millions of rows. The fixed-format naive parse
    plus one localize is ~10x faster and applies only when a vectorized
    suffix check proves the offset is uniform CN.
    """
    if series.dtype == object and len(series):
        text = series.astype(str)
        separated = text.str.replace("T", " ", n=1, regex=False)
        naive = separated.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        separated = separated.where(~naive, separated + "+08:00")
        if separated.str.endswith("+08:00").all():
            fast = pd.to_datetime(separated.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
            if not fast.isna().any():
                # Localize via a UTC shift: tz_localize(CN_TZ) with a ZoneInfo
                # walks rows one by one, the UTC route is metadata-only.
                return (fast - pd.Timedelta(hours=8)).dt.tz_localize("UTC").dt.tz_convert(CN_TZ)
        series = separated
    try:
        parsed = pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        parsed = None
    if parsed is not None and getattr(parsed.dtype, "tz", None) is not None:
        return parsed.dt.tz_convert(CN_TZ)
    if parsed is not None and parsed.dtype != object:
        return parsed.dt.tz_localize(CN_TZ)
    # Mixed aware/naive values: normalize element-wise.
    fallback = pd.to_datetime(series, errors="coerce", utc=False)
    if fallback.dtype != object:
        if getattr(fallback.dtype, "tz", None) is not None:
            return fallback.dt.tz_convert(CN_TZ)
        return fallback.dt.tz_localize(CN_TZ)
    return fallback.map(
        lambda value: (value.tz_localize(CN_TZ) if value.tzinfo is None else value.tz_convert(CN_TZ))
        if pd.notna(value)
        else pd.NaT
    )


def yyyymmdd(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return text
    try:
        return pd.Timestamp(text).strftime("%Y%m%d")
    except Exception as exc:
        raise ValueError(f"cannot parse date value as YYYYMMDD: {value!r}") from exc


def parse_partition_date(path: Path) -> str:
    stem = path.stem
    return stem.split("=", 1)[1] if "=" in stem else ""


@dataclass(frozen=True)
class PITDataStore:
    raw_dir: Path

    def dataset_dir(self, dataset: str) -> Path:
        path = self.raw_dir / dataset
        if not path.exists():
            raise FileNotFoundError(f"missing dataset directory: {path}")
        return path

    def trade_dates(self, dataset: str) -> list[str]:
        return sorted(
            key
            for key in (parse_partition_date(path) for path in self.dataset_dir(dataset).glob("trade_date=*.parquet"))
            if key
        )

    def read_trade_date(self, dataset: str, trade_date: date | str, columns: list[str] | None = None) -> pd.DataFrame:
        key = yyyymmdd(trade_date)
        path = self.dataset_dir(dataset) / f"trade_date={key}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing partition: {path}")
        return pd.read_parquet(path, columns=columns)

    def read_trade_range(self, dataset: str, start: date | str, end: date | str, columns: list[str] | None = None) -> pd.DataFrame:
        start_key = yyyymmdd(start)
        end_key = yyyymmdd(end)
        if start_key > end_key:
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
        frames = []
        for key in self.trade_dates(dataset):
            if start_key <= key <= end_key:
                frame = self.read_trade_date(dataset, key, columns=columns)
                if not frame.empty:
                    frames.append(frame)
        if frames:
            return concat_rows(frames)
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
