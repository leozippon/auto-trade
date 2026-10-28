"""Versioned raw-lake partitions: the one path that changes a visible partition.

A partition is ``<raw>/<dataset>/[<key>=<value>/...]<name>.parquet`` with its
``.meta.json`` sidecar; readers, audits and releases see only those. Every
change of a partition goes through :func:`publish` (a vendor answer) or
:func:`restore` (an operator's revert or promotion), which keep three promises:

* The visible pair is replaced by atomic renames and never edited, so a reader
  or a release hard link never sees a torn or missing parquet. A crash between
  the parquet and the sidecar rename is rolled forward by the next write.
* A change that removes or rewrites existing business keys first keeps the
  partition it replaces as an immutable *superseded* version (a hard link, so
  it costs no space while a release still holds the same file). A pure append,
  in which every existing key is present with unchanged values, keeps no copy:
  nothing is lost.
* An answer that would remove more than half of the current keys is written as
  a *staged* version and not made visible; an operator promotes it. An answer
  the caller marks truncated (cut at the vendor's per-call row cap) is exempt:
  the keys it lacks prove nothing, so it is published like any other answer.

Versions live outside the raw tree, under
``<raw>/../raw_versions/<dataset>/<partition>/<version>.parquet`` (with the
version's sidecar), so no partition glob, audit inventory or release clone can
pick them up. A version is named by the UTC instant it left the visible slot
or was pulled, and by its kind: ``20261024T014501123456Z-superseded`` or
``20261024T014501123456Z-staged``. The pipeline never edits or deletes a
version; a crash leaves at most one extra version behind.

A keyed partition of the versioned era also carries an observation table next
to it, ``<name>.parquet.first_seen``: Parquet with the business-key columns (as
canonical text) and ``first_seen_at``, the UTC instant at which the key first
became part of the visible partition. The versioned era starts on the date the
generation record names as ``versioned_since``, the day versioning began; a
partition belongs to it when the period its id names (``month=``, ``year=``,
``trade_date=``, ``period=`` ...; :func:`partition_period_end`) ends on or
after that date, or when its id names no period (whole-history files such as
``ts_code=<code>`` or ``share_float_complete``). Older periods never get rows,
whenever their keys arrive -- a backfill, a history extension or a re-pull
adding keys to a 2022 month follows the vendor's dates, like all history before
versioning -- and neither do keys already visible when versioning started.
The table only grows, and it is written before the partition it describes, so a
row whose ``first_seen_at`` is later than the partition's ``visible_since``
belongs to a write that never became visible and is ignored
(:func:`read_observations`) and dropped by the next write.

Every visible sidecar written here carries ``visible_since`` (when this content
became the visible partition), the business ``key_columns`` and, once a
restatement or a restore has replaced its history, ``restated_write_id``: the
write that started the content's current version. A dataset's vintage id hashes
those, so it moves when history changes and not when rows are appended.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.environment.data.contracts import RAW_GENERATION_FILENAME
from autotrade.environment.data.pit import (
    FIRST_SEEN_COLUMN,
    KEY_SEPARATOR,
    CorruptSidecarError,
    canonical_keys,
    canonical_revision_value,
    canonical_series,
    complete_interrupted_commit,
    concat_rows,
    join_columns,
    key_strings,
    observation_path,
    parquet_meta,
    read_observations,
)

from .io import (
    append_jsonl_unique,
    committed_partition_intact,
    frames_content_equal,
    write_parquet,
)

VERSION_ROOT_NAME = "raw_versions"
SUPERSEDED = "superseded"
STAGED = "staged"
# An answer removing more than this share of the partition's current keys is
# staged for an operator instead of being made visible.
STAGE_REMOVED_SHARE = 0.5
# Business-key fields the vendor legitimately fills in later. They stay in the
# key (two rows of one filing differ by them), but a fill is not a removal:
# when the H1 reporting season closed TuShare populated
# ``disclosure_date.actual_date`` on 330 rows of period=20260630 and backfilled
# ``dividend.ann_date`` on 1990s records, and no row had disappeared. The
# staging rule therefore counts removals on the key minus these fields.
VENDOR_BACKFILLED_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "disclosure_date": ("actual_date",),
    "dividend": ("ann_date",),
}
# The runner names a per-run JSONL file here; every partition outcome of its
# child processes is appended to it (one line each), and the runner folds them
# into the generation record when the job ends. ``complete`` is the one
# dataset-level outcome: the dataset's whole pull of this run returned without
# an error (a defect in one of its partitions is reported separately).
RUN_REPORT_ENV = "TUSHARE_RUN_REPORT"
OUTCOMES = ("unchanged", "appended", "replaced", "restated", "staged", "defect", "complete")

REVISION_EVENT_SCHEMA_VERSION = 3
EVENT_KINDS = ("restatement", "staged", "defect", "deduplicated", "promotion", "revert", "probe")
REVISION_EVENT_FIELDS = (
    "schema_version",
    "event_id",
    "detected_at",
    "source",
    "kind",
    "dataset",
    "partition",
    "key_columns",
    "old_rows",
    "new_rows",
    "old_keys",
    "new_keys",
    "added_keys",
    "removed_keys",
    "changed_keys",
    "changed_columns",
    "duplicate_rows_dropped",
    "detail",
    "version",
    "superseded_version",
    "write_id",
)

_ROW_SEPARATOR = "\x1e"


class DefectiveAnswer(RuntimeError):
    """The vendor answered, but the answer cannot be trusted: a requested field
    is missing, fields changed while paging, a page repeated or a page boundary
    did not hold. The partition keeps its current version and the defect is
    recorded as an issue; the job goes on with the next partition."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class KeyDiff:
    """Two frames compared on business keys. A key whose rows repeat compares
    as the multiset of its rows, so duplicate keys stay decidable."""

    old_keys: int
    new_keys: int
    added: pd.Index
    removed: pd.Index
    changed: pd.Index
    changed_columns: dict[str, int]

    @property
    def restates(self) -> bool:
        return bool(len(self.removed) or len(self.changed))

    @property
    def empty(self) -> bool:
        return not (len(self.added) or len(self.removed) or len(self.changed))


def diff_frames(old: pd.DataFrame, new: pd.DataFrame, key_columns: Sequence[str]) -> KeyDiff:
    keys = list(key_columns)
    value_columns = sorted((set(map(str, old.columns)) | set(map(str, new.columns))) - set(keys))
    columns = keys + value_columns

    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reset_index(drop=True)
        return pd.DataFrame(
            {
                column: canonical_series(frame[column]) if column in frame.columns else pd.Series("", index=frame.index)
                for column in columns
            },
            index=frame.index,
        )

    def signatures(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        keyed = join_columns(frame, keys)
        values = join_columns(frame, value_columns, KEY_SEPARATOR)
        if keyed.is_unique:
            return pd.Series(values.to_numpy(), index=keyed.to_numpy(), dtype=object), keyed
        grouped = (
            pd.DataFrame({"k": keyed, "v": values})
            .sort_values(["k", "v"], kind="stable")
            .groupby("k", sort=False)["v"]
            .agg(_ROW_SEPARATOR.join)
        )
        return grouped, keyed

    old_c, new_c = canonical(old), canonical(new)
    old_sig, old_keyed = signatures(old_c)
    new_sig, new_keyed = signatures(new_c)
    removed = old_sig.index.difference(new_sig.index)
    added = new_sig.index.difference(old_sig.index)
    common = old_sig.index.intersection(new_sig.index)
    changed = common[old_sig.loc[common].to_numpy() != new_sig.loc[common].to_numpy()]
    changed_columns: dict[str, int] = {}
    if len(changed) and old_keyed.is_unique and new_keyed.is_unique and value_columns:
        before = old_c.set_index(old_keyed.to_numpy()).loc[changed, value_columns].to_numpy()
        after = new_c.set_index(new_keyed.to_numpy()).loc[changed, value_columns].to_numpy()
        counts = (before != after).sum(axis=0)
        changed_columns = {column: int(count) for column, count in zip(value_columns, counts, strict=True) if count}
    return KeyDiff(len(old_sig), len(new_sig), added, removed, changed, changed_columns)


def deduplicate_keys(frame: pd.DataFrame, key_columns: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """One row per business key, chosen the same way whatever order the vendor
    served the rows in: the row whose canonical text is smallest, column by
    column in the frame's own order. Returns the frame and the rows dropped."""
    keys = list(key_columns)
    if frame.empty or not set(keys) <= set(frame.columns):
        return frame, 0
    frame = frame.reset_index(drop=True)
    canonical = pd.DataFrame({column: canonical_series(frame[column]) for column in frame.columns}, index=frame.index)
    if not canonical.duplicated(keys).any():
        return frame, 0
    order = canonical.sort_values(list(canonical.columns), kind="stable")
    keep = ~order.duplicated(keys, keep="first")
    kept = frame.loc[sorted(order.index[keep.to_numpy()])]
    return kept.reset_index(drop=True), len(frame) - len(kept)


# ---------------------------------------------------------------- layout


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def dataset_dir(path: Path) -> Path:
    """The dataset directory of a partition: its nearest ancestor that is not a
    ``<key>=<value>`` partition directory."""
    for parent in Path(path).parents:
        if "=" not in parent.name:
            return parent
    raise ValueError(f"not a raw-lake partition path: {path}")


def partition_id(path: Path) -> str:
    """The partition relative to its dataset directory, without ``.parquet``."""
    return Path(path).relative_to(dataset_dir(path)).with_suffix("").as_posix()


_DAY_PERIODS = frozenset({"trade_date", "date", "ann_date", "period"})
_MONTH_PERIODS = frozenset({"month", "ann_month"})


def partition_period_end(partition: str) -> str | None:
    """The last day (YYYYMMDD) of the period a partition id names, or None
    when it names none (a whole-history file, a code, an open-ended
    ``range=<start>_latest``). ``period=`` is a fiscal period end."""
    end = None
    for part in partition.split("/"):
        key, _, value = part.partition("=")
        if key in _DAY_PERIODS and re.fullmatch(r"\d{8}", value):
            end = value
        elif key in _MONTH_PERIODS and re.fullmatch(r"\d{6}", value):
            end = f"{value}{calendar.monthrange(int(value[:4]), int(value[4:]))[1]:02d}"
        elif key == "year" and re.fullmatch(r"\d{4}", value):
            end = f"{value}1231"
        elif key == "range" and (closed := re.fullmatch(r"\d{6,8}_(\d{6}|\d{8})", value)):
            last = closed.group(1)
            end = last if len(last) == 8 else f"{last}{calendar.monthrange(int(last[:4]), int(last[4:]))[1]:02d}"
    return end


def versioned_since(raw_dir: Path) -> str | None:
    """The first day (YYYYMMDD) of the lake's versioned era, as the generation
    record names it; None for a record that names none."""
    record = Path(raw_dir) / RAW_GENERATION_FILENAME
    if not record.exists():
        return None
    return json.loads(record.read_text(encoding="utf-8")).get("versioned_since") or None


def records_observations(path: Path) -> bool:
    """Whether the partition at ``path`` belongs to the versioned era and so
    keeps first-seen rows (see the module docstring)."""
    since = versioned_since(dataset_dir(Path(path)).parent)
    end = partition_period_end(partition_id(Path(path)))
    return since is not None and (end is None or end >= since)


def version_dir(path: Path) -> Path:
    ds = dataset_dir(path)
    return ds.parent.parent / VERSION_ROOT_NAME / ds.name / partition_id(path)


def version_root(raw_dir: Path) -> Path:
    return Path(raw_dir).parent / VERSION_ROOT_NAME


def _version_stamp(instant: str) -> str:
    return datetime.fromisoformat(instant).astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def list_versions(path: Path) -> list[str]:
    directory = version_dir(path)
    if not directory.is_dir():
        return []
    return sorted(item.name[: -len(".parquet")] for item in directory.glob("*.parquet"))


def _keep_version(path: Path, instant: str, kind: str) -> str:
    """Hard-link the visible pair into the version directory; the version id."""
    version = f"{_version_stamp(instant)}-{kind}"
    target = version_dir(path) / f"{version}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(path, target)
    if sidecar_path(path).exists():
        os.link(sidecar_path(path), sidecar_path(target))
    return version


# ---------------------------------------------------------------- ledger and run report


def revision_event(
    kind: str,
    *,
    dataset: str,
    partition: str,
    source: str,
    key_columns: Sequence[str] = (),
    diff: KeyDiff | None = None,
    old_rows: int | None = None,
    new_rows: int | None = None,
    duplicate_rows_dropped: int = 0,
    detail: str = "",
    version: str = "",
    superseded_version: str = "",
    write_id: str = "",
) -> dict[str, Any]:
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown revision event kind: {kind}")
    return {
        "schema_version": REVISION_EVENT_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "detected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "kind": kind,
        "dataset": dataset,
        "partition": partition,
        "key_columns": list(key_columns),
        "old_rows": old_rows,
        "new_rows": new_rows,
        "old_keys": diff.old_keys if diff else None,
        "new_keys": diff.new_keys if diff else None,
        "added_keys": len(diff.added) if diff else None,
        "removed_keys": len(diff.removed) if diff else None,
        "changed_keys": len(diff.changed) if diff else None,
        "changed_columns": dict(diff.changed_columns) if diff else {},
        "duplicate_rows_dropped": int(duplicate_rows_dropped),
        "detail": detail,
        "version": version,
        "superseded_version": superseded_version,
        "write_id": write_id,
    }


def append_event(ledger: Path | str | None, event: dict[str, Any]) -> None:
    if ledger:
        append_jsonl_unique(Path(ledger), event, key="event_id")
    print("REVISION_ALERT " + json.dumps(
        {key: event[key] for key in ("kind", "dataset", "partition", "detail", "version", "superseded_version")},
        ensure_ascii=False, sort_keys=True,
    ))


def report_outcome(path: Path, outcome: str, *, detail: str = "", version: str = "") -> None:
    """Append one partition outcome to the runner's run report, if any."""
    if outcome not in OUTCOMES or outcome == "complete":
        raise ValueError(f"unknown partition outcome: {outcome}")
    _report(dataset_dir(path).name, partition_id(path), outcome, detail=detail, version=version)


def report_complete(dataset: str) -> None:
    """Record in the run report that ``dataset``'s whole pull returned without
    an error. Only such a dataset, and only without a defective or staged
    partition in the same run, counts as refreshed."""
    _report(dataset, "", "complete")


def _report(dataset: str, partition: str, outcome: str, *, detail: str = "", version: str = "") -> None:
    report = os.environ.get(RUN_REPORT_ENV, "")
    if not report:
        return
    line = json.dumps(
        {"dataset": dataset, "partition": partition, "outcome": outcome, "detail": detail, "version": version, "at": utc_now()},
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    with open(report, "a", encoding="utf-8") as handle:
        handle.write(line)


def record_defect(path: Path, detail: str, *, ledger: Path | str | None, source: str) -> str:
    path = Path(path)
    append_event(ledger, revision_event(
        "defect", dataset=dataset_dir(path).name, partition=partition_id(path), source=source, detail=detail[:1000],
    ))
    report_outcome(path, "defect", detail=detail[:300])
    return "defect"


@contextmanager
def answer_defects(path: Path, *, ledger: Path | str | None, source: str = "force_refresh") -> Iterator[None]:
    """Turn a :class:`DefectiveAnswer` for ``path`` into a recorded defect.

    The partition keeps its current version and the loop around this block
    goes on with the next partition."""
    try:
        yield
    except DefectiveAnswer as exc:
        print(f"defective answer for {path}: {exc}")
        record_defect(path, str(exc), ledger=ledger, source=source)


# ---------------------------------------------------------------- observations


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    pq.write_table(pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False), tmp)
    os.replace(tmp, path)


def _record_observations(
    path: Path,
    new_keys: pd.Index,
    key_columns: Sequence[str],
    instant: str,
    meta: dict[str, Any] | None,
) -> None:
    """Give every key in ``new_keys`` without a row the first-seen ``instant``.

    ``meta`` is the visible partition's sidecar, ``None`` when there is no
    visible partition yet (rows a crashed first landing left are dropped)."""
    keys = list(key_columns)
    obs = observation_path(path)
    existing = read_observations(path, meta)
    stale = obs.exists() and len(existing) != pq.ParquetFile(obs).metadata.num_rows
    known = pd.Index(join_columns(existing.astype({column: str for column in keys}), keys)) if len(existing) else pd.Index([])
    fresh = pd.Index(new_keys).difference(known)
    if fresh.empty and not stale:
        return
    rows = pd.DataFrame([key.split(KEY_SEPARATOR) for key in fresh], columns=keys, dtype=str) if len(fresh) else pd.DataFrame(columns=keys, dtype=str)
    rows[FIRST_SEEN_COLUMN] = instant
    frame = concat_rows([existing[[*keys, FIRST_SEEN_COLUMN]], rows]) if len(existing) else rows
    _write_table(obs, frame[[*keys, FIRST_SEEN_COLUMN]].astype(str))


def rebuild_observations(path: Path) -> dict[str, int]:
    """Rebuild the observation table of ``path`` from its version history.

    The retained states are walked in order -- every superseded version, then
    the visible partition -- and a key gets the ``visible_since`` of the first
    retained state that newly holds it. A retained state written before
    versioning started (its sidecar has no ``visible_since``) contributes keys
    without rows. Appends keep no copy, so a rebuilt time is that of the next
    retained state, and a key whose pre-versioning state was replaced by an
    append gets a row too: a rebuilt row is never earlier than the recorded
    one, only later."""
    path = Path(path)
    meta = parquet_meta(path)
    keys = list(meta.get("key_columns") or [])
    if not keys:
        raise RuntimeError(f"{path} records no business key columns; it has no observation table")
    if not records_observations(path):
        # A period before the versioned era keeps no rows, rebuilt or not.
        observation_path(path).unlink(missing_ok=True)
        return {"states": 0, "rows": 0}
    states = [version_dir(path) / f"{version}.parquet" for version in list_versions(path) if version.endswith(SUPERSEDED)]
    rows: dict[str, str] = {}
    previous: set[str] = set()
    for state in [*states, path]:
        state_meta = json.loads(sidecar_path(state).read_text(encoding="utf-8"))
        held = set(key_strings(pd.read_parquet(state, columns=[c for c in keys if c in pq.ParquetFile(state).schema_arrow.names]), keys))
        since = state_meta.get("visible_since")
        if since:
            for key in sorted(held - previous):
                rows.setdefault(key, str(since))
        previous = held
    frame = pd.DataFrame([key.split(KEY_SEPARATOR) for key in rows], columns=keys, dtype=str)
    frame[FIRST_SEEN_COLUMN] = list(rows.values())
    _write_table(observation_path(path), frame[[*keys, FIRST_SEEN_COLUMN]].astype(str))
    return {"states": len(states) + 1, "rows": len(frame)}


# ---------------------------------------------------------------- publish


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        return parquet_meta(path)
    except CorruptSidecarError as exc:
        # A torn sidecar the roll-forward could not repair: this write
        # replaces both files, loudly.
        print("REVISION_ALERT " + json.dumps(
            {"note": "corrupt_sidecar_replaced", "path": str(path), "error": str(exc)[:300]},
            ensure_ascii=False, sort_keys=True,
        ))
        return {}


def _refresh_sidecar(path: Path, meta: dict[str, Any], params: dict[str, Any], fields: list[str]) -> None:
    """Record a wider request over unchanged content: same commit identity."""
    if meta.get("params") == params and meta.get("fields") == list(fields):
        return
    updated = {**meta, "params": dict(params), "fields": list(fields)}
    target = sidecar_path(path)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4()}.tmp")
    tmp.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, target)


def publish(
    path: Path,
    df: pd.DataFrame,
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    key_columns: Sequence[str],
    ledger: Path | str | None,
    source: str = "force_refresh",
    deduplicate: bool = False,
    extra_metadata: dict[str, Any] | None = None,
    compare: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    current: pd.DataFrame | None = None,
    truncated: bool = False,
) -> str:
    """Make ``df`` the visible partition at ``path`` if the answer allows it.

    Returns the outcome: ``unchanged`` (nothing written), ``appended`` (new
    keys only, or a first landing), ``replaced`` (an unkeyed partition
    changed, or an interrupted commit rewritten with the content it already
    holds), ``restated`` (existing keys removed or rewritten; the replaced
    partition is kept as a superseded version), ``staged`` (more than half
    of the current keys would go; written as a staged version only) or
    ``defect`` (an empty answer for a non-empty partition; nothing written).

    ``compare`` restricts the key comparison to a slice the caller proved
    holds every key the write affects (old rows, new rows); such a caller
    merges the answer into the partition itself and owns the shrink judgement,
    so a compared write is never staged. ``current`` is the visible frame when
    the caller has already read it. ``truncated`` marks an answer cut at the
    vendor's per-call row cap: the keys it lacks are not a withdrawal, so it is
    never staged and the caller owns recovering them."""
    path = Path(path)
    keys = list(key_columns)
    dataset, partition = dataset_dir(path).name, partition_id(path)
    complete_interrupted_commit(path)
    exists = path.exists()
    if exists and current is None:
        current = pd.read_parquet(path)
    meta = _read_meta(path) if exists else {}
    dropped = 0
    if deduplicate and keys:
        df, dropped = deduplicate_keys(df, keys)
    if exists and len(current) and df.empty:
        return record_defect(path, f"empty answer for a partition holding {len(current)} rows", ledger=ledger, source=source)

    diff: KeyDiff | None = None
    if not exists:
        outcome = "appended"
    elif not keys:
        outcome = "unchanged" if frames_content_equal(current, df) else "replaced"
    else:
        old_part, new_part = compare if compare is not None else (current, df)
        diff = diff_frames(old_part, new_part, keys)
        same_columns = set(map(str, current.columns)) == set(map(str, df.columns))
        if diff.empty and same_columns:
            outcome = "unchanged"
        else:
            outcome = "restated" if diff.restates else "appended"
            if compare is None and not truncated and _removes_most_keys(api_name, current, df, keys, diff):
                return _stage(path, df, api_name=api_name, params=params, fields=fields, keys=keys, diff=diff,
                              ledger=ledger, source=source, dropped=dropped, old_rows=len(current),
                              extra_metadata=extra_metadata, meta=meta)
    if dropped and outcome != "restated":
        append_event(ledger, revision_event(
            "deduplicated", dataset=dataset, partition=partition, source=source, key_columns=keys,
            old_rows=len(current) if current is not None else 0, new_rows=len(df), duplicate_rows_dropped=dropped,
            detail="duplicate business keys in the answer were dropped",
        ))
    if outcome == "unchanged" and not committed_partition_intact(path):
        # The visible pair is not one complete commit (no sidecar, or another
        # write's): the same content is committed again to heal it.
        outcome = "replaced"
    if outcome == "unchanged":
        _refresh_sidecar(path, meta, params, fields)
        report_outcome(path, outcome)
        return outcome

    instant = utc_now()
    write_id = str(uuid.uuid4())
    superseded = _keep_version(path, instant, SUPERSEDED) if outcome == "restated" else ""
    metadata = _with_availability({**(extra_metadata or {}), "visible_since": instant}, meta, instant, len(df))
    if keys:
        metadata["key_columns"] = keys
        restated = write_id if outcome == "restated" else meta.get("restated_write_id")
        if restated:
            metadata["restated_write_id"] = restated
    if keys and records_observations(path):
        new_keys = diff.added if diff is not None else pd.Index(key_strings(df, keys).unique())
        _record_observations(path, new_keys, keys, instant, meta if exists else None)
    write_parquet(path, df, api_name=api_name, params=params, fields=fields, extra_metadata=metadata, write_id=write_id, at=instant)
    if outcome == "restated":
        append_event(ledger, revision_event(
            "restatement", dataset=dataset, partition=partition, source=source, key_columns=keys, diff=diff,
            old_rows=len(current), new_rows=len(df), duplicate_rows_dropped=dropped,
            superseded_version=superseded, write_id=write_id,
        ))
    report_outcome(path, outcome, version=superseded)
    return outcome


def _with_availability(metadata: dict[str, Any], meta: dict[str, Any], instant: str, rows: int) -> dict[str, Any]:
    """``metadata`` with an availability record when the caller gave none but
    the visible partition has one: first-landing evidence describes the old
    payload, and a changed payload was not knowable before this fetch."""
    previous = meta.get("availability")
    if previous and "availability" not in metadata:
        metadata["availability"] = {**previous, "available_at": instant, "rule": "observed:source_revision_fetch", "row_count": rows}
    return metadata


def _removes_most_keys(api_name: str, current: pd.DataFrame, df: pd.DataFrame, keys: list[str], diff: KeyDiff) -> bool:
    """Whether the answer removes more than ``STAGE_REMOVED_SHARE`` of the
    partition's keys, counted on the key minus vendor-backfilled fields."""
    if not len(diff.removed):
        return False
    stable = [column for column in keys if column not in VENDOR_BACKFILLED_KEY_FIELDS.get(api_name, ())]
    if stable == keys:
        removed, held = len(diff.removed), diff.old_keys
    else:
        before, after = set(key_strings(current, stable)), set(key_strings(df, stable))
        removed, held = len(before - after), len(before)
    return bool(held) and removed > STAGE_REMOVED_SHARE * held


def _stage(
    path: Path,
    df: pd.DataFrame,
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    keys: list[str],
    diff: KeyDiff,
    ledger: Path | str | None,
    source: str,
    dropped: int,
    old_rows: int,
    extra_metadata: dict[str, Any] | None,
    meta: dict[str, Any],
) -> str:
    """Keep a shrunk answer as a staged version with the metadata its
    publication would have carried (an observed availability record
    included), so a promotion restores exactly that commit."""
    staged = [version for version in list_versions(path) if version.endswith(STAGED)]
    if staged:
        latest = version_dir(path) / f"{staged[-1]}.parquet"
        if diff_frames(pd.read_parquet(latest), df, keys).empty:
            # The vendor serves the same shrunk answer again: the issue stays
            # open on the version already staged.
            report_outcome(path, "staged", version=staged[-1])
            return "staged"
    instant = utc_now()
    version = f"{_version_stamp(instant)}-{STAGED}"
    write_id = str(uuid.uuid4())
    target = version_dir(path) / f"{version}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = _with_availability(
        {**(extra_metadata or {}), "key_columns": keys, "restated_write_id": write_id}, meta, instant, len(df)
    )
    write_parquet(target, df, api_name=api_name, params=params, fields=fields, extra_metadata=metadata, write_id=write_id, at=instant)
    append_event(ledger, revision_event(
        "staged", dataset=dataset_dir(path).name, partition=partition_id(path), source=source, key_columns=keys,
        diff=diff, old_rows=old_rows, new_rows=len(df), duplicate_rows_dropped=dropped, version=version,
        detail=f"would remove {len(diff.removed)} of {diff.old_keys} keys; staged for an operator",
    ))
    report_outcome(path, "staged", version=version)
    return "staged"


def restore(path: Path, version: str, *, ledger: Path | str | None, source: str = "operator") -> dict[str, Any]:
    """Make a kept version the visible partition again (a revert of a
    superseded version, a promotion of a staged one); the replaced partition is
    kept as a superseded version first."""
    path = Path(path)
    kind = "promotion" if version.endswith(f"-{STAGED}") else "revert" if version.endswith(f"-{SUPERSEDED}") else ""
    if not kind:
        raise ValueError(f"not a version id: {version!r} (expected <stamp>-{SUPERSEDED} or <stamp>-{STAGED})")
    source_file = version_dir(path) / f"{version}.parquet"
    if not source_file.is_file() or not sidecar_path(source_file).is_file():
        raise FileNotFoundError(f"version {version} of {path} does not exist: {source_file}")
    complete_interrupted_commit(path)
    version_meta = json.loads(sidecar_path(source_file).read_text(encoding="utf-8"))
    meta = _read_meta(path) if path.exists() else {}
    keys = list(version_meta.get("key_columns") or meta.get("key_columns") or [])
    restored = pd.read_parquet(source_file)
    current = pd.read_parquet(path) if path.exists() else restored.iloc[:0]
    diff = diff_frames(current, restored, keys) if keys else None
    instant = utc_now()
    superseded = _keep_version(path, instant, SUPERSEDED) if path.exists() else ""
    if keys and records_observations(path):
        _record_observations(path, diff.added, keys, instant, meta if path.exists() else None)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    os.link(source_file, tmp)
    restored_meta = {**version_meta, "visible_since": instant}
    if keys:
        restored_meta["key_columns"] = keys
    meta_target = sidecar_path(path)
    meta_tmp = meta_target.with_name(f".{meta_target.name}.{restored_meta['write_id']}.tmp")
    meta_tmp.write_text(json.dumps(restored_meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
    os.replace(meta_tmp, meta_target)
    event = revision_event(
        kind, dataset=dataset_dir(path).name, partition=partition_id(path), source=source, key_columns=keys, diff=diff,
        old_rows=len(current), new_rows=len(restored), version=version, superseded_version=superseded,
        write_id=str(restored_meta.get("write_id") or ""),
    )
    append_event(ledger, event)
    return {"kind": kind, "version": version, "superseded_version": superseded, "rows": len(restored)}


# ---------------------------------------------------------------- vintages


def dataset_vintage(dataset_path: Path) -> str:
    """The dataset's vintage id: a hash over the restated partitions' version
    pointers (``restated_write_id``). Appends and first landings leave it
    unchanged; a restatement, promotion or revert moves it, and a revert back
    to an earlier state restores the earlier id."""
    dataset_path = Path(dataset_path)
    entries: list[str] = []
    for meta_path in sorted(dataset_path.rglob("*.parquet.meta.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("restated_write_id"):
            entries.append(f"{meta_path.relative_to(dataset_path).as_posix()}\t{payload['restated_write_id']}")
    digest = hashlib.sha256("\n".join([dataset_path.name, *entries]).encode("utf-8"))
    return digest.hexdigest()[:16]


__all__ = [
    "FIRST_SEEN_COLUMN",
    "RUN_REPORT_ENV",
    "STAGE_REMOVED_SHARE",
    "VERSION_ROOT_NAME",
    "DefectiveAnswer",
    "KeyDiff",
    "answer_defects",
    "canonical_keys",
    "canonical_revision_value",
    "complete_interrupted_commit",
    "dataset_vintage",
    "deduplicate_keys",
    "diff_frames",
    "list_versions",
    "observation_path",
    "partition_period_end",
    "publish",
    "read_observations",
    "rebuild_observations",
    "record_defect",
    "records_observations",
    "report_complete",
    "restore",
    "revision_event",
    "version_dir",
    "versioned_since",
]
