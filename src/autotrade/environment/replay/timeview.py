"""Rolling Timeview (docs/environment-design.md).

Replays the real local-DB refresh cadence so a strategy at any tick sees only the
data its landing cron job has already written. Each agent-readable parquet domain
(daily, events, macro, fundamentals, intraday minute history, text_index) is
exposed under ``ctx.asof_dir/<domain>/`` as a directory of plain parquet parts
that ``pandas.read_parquet`` concatenates into one table:

  * part 0 is the frozen research snapshot for that domain, hardlinked in
    (zero-copy);
  * later parts are write-once replay-slot increments, appended only when the
    simulation clock crosses a refresh node that covers the domain
    (``REFRESH_NODES`` in data/contracts.py).

Visibility only grows forward in time, so each replay row is written exactly once
and unchanged domains cost nothing. Between the next pending refresh or row-level
boundary (including observed auction availability), ``refresh`` is an O(1) no-op.
``ctx.asof_version`` bumps whenever any new part lands. It identifies the global
view, not an individual domain; heavy single-domain strategy features should use a
narrower dependency key so minute updates do not invalidate unrelated work.

Text bodies live under ``ctx.asof_dir/text_library``. Frozen snapshot body shards
are hardlinked at start; replay body shards are copied only for newly visible
``text_index`` rows, so direct text processing has the same PIT wall as ``ctx.nl``.
Durable part reuse is enabled only after the PIT backend directly validates the
stash contract for the exact release, snapshot, replay slot, configuration, and
schedule. Each part is then published under its own process lock.

One view can run on across consecutive replay slots (``continue_into``): each
domain keeps one segment per slot, and a part that lands after a slot boundary
lists the earlier slot's rows first. Slots partition rows by ``available_at``,
so the parts are those one long slot laid out in slot order would write.

Replay rows stay in the slot's parquet files (``ReplayRows``): a domain reads
only its gating columns to build its cursors and reads the row groups holding
a part's rows when it encodes that part, so a stash hit reads nothing and no
decoded slot is ever resident.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.environment.data.contracts import (
    domain_next_visible_boundary,
    domain_visible_cutoff,
    event_dataset_next_visible_boundary,
    event_dataset_visible_cutoff,
    macro_dataset_next_visible_boundary,
    macro_dataset_visible_cutoff,
    text_dataset_next_visible_boundary,
    text_dataset_visible_cutoff,
)
from autotrade.environment.data.pit import to_cn_timestamps

_EMPTY_INDICES = np.array([], dtype=np.int64)
_ROW_AVAILABLE_AT = "row_available_at"
# Rows per row group of a written part: pyarrow's ``write_table`` default, so
# a part streamed row group by row group is byte for byte the file one
# ``write_table`` call over the whole part produces.
_PART_ROW_GROUP_ROWS = 1 << 20


@dataclass
class _PendingReplayPartition:
    frame: pd.DataFrame
    keys: np.ndarray


def _utc_ns(values: pd.Series) -> np.ndarray:
    """tz-aware timestamps as UTC-naive datetime64[ns] (searchsorted keys)."""
    return values.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def _cutoff_ns(cutoff: object) -> np.datetime64:
    ts = pd.Timestamp(cutoff)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts, "ns")


class _SortedCursor:
    """Visibility only grows forward, so each cutoff advance is a binary search
    over available_at-sorted row positions instead of a whole-series boolean
    mask (the mask was O(rows) per node crossing — ~44M rows for a quarter of
    minute bars)."""

    __slots__ = ("indices", "keys", "pos")

    def __init__(self, indices: np.ndarray, keys: np.ndarray) -> None:
        order = np.argsort(keys, kind="stable")
        self.indices = indices[order]
        self.keys = keys[order]
        self.pos = 0

    def advance(self, cutoff: np.datetime64) -> np.ndarray:
        new_pos = int(np.searchsorted(self.keys, cutoff, side="right"))
        if new_pos <= self.pos:
            return _EMPTY_INDICES
        newly = self.indices[self.pos:new_pos]
        self.pos = new_pos
        return newly

    def has_pending(self) -> bool:
        return self.pos < len(self.keys)

    def next_key(self) -> np.datetime64 | None:
        return self.keys[self.pos] if self.has_pending() else None


class ReplayRows:
    """One replay domain's rows, left on disk until a roll takes them.

    Building the cursors needs only the gating columns (``available_at`` and,
    for dataset-gated domains, ``dataset``), so those are all that is read up
    front. The rows a part publishes are read from the row groups that hold
    them when that part is encoded -- never on a stash hit -- and each row
    group's rows take the same pandas round trip a slice of the whole decoded
    frame took, so a part is byte-identical to one cut from that frame (a
    string column with no value in the slice still becomes a null column)
    without the decoded slot (21-24 GiB as pandas for one year of events)
    ever being resident.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = pq.ParquetFile(self.path)
        metadata = self._file.metadata
        self.columns: list[str] = list(self._file.schema_arrow.names)
        self.num_rows = int(metadata.num_rows)
        self._row_group_ends = np.cumsum(
            [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
        )

    @property
    def empty(self) -> bool:
        return self.num_rows == 0

    def column(self, name: str) -> pd.Series:
        """One column of every row, as the decoded frame's column."""
        return self._file.read(columns=[name]).to_pandas()[name]

    def take(self, indices: np.ndarray) -> Iterator[pa.Table]:
        """The rows at ``indices`` (ascending), one table per row group holding them.

        Each table is what ``frame.iloc[rows]`` of the decoded file would
        encode; one row group is the most decoded at once. The tables unify on
        the types their round trips inferred (a null column beside a typed one
        takes its type), which is what the whole slice would have inferred.
        """
        groups = np.searchsorted(self._row_group_ends, indices, side="right")
        for group in np.unique(groups):
            first = int(self._row_group_ends[group - 1]) if group else 0
            rows = self._file.read_row_group(int(group)).take(pa.array(indices[groups == group] - first))
            yield pa.Table.from_pandas(rows.to_pandas(), preserve_index=False)


@dataclass
class _ReplaySegment:
    """One replay slot's rows of a domain and the cursors that roll them.

    ``cursor`` gates the whole domain (or each row); ``cursors`` gate it
    dataset by dataset. Exactly one of the two is in use. ``library_dir`` is
    the slot's text body library (text index segments only).
    """

    rows: ReplayRows
    cursor: _SortedCursor | None
    cursors: dict[str, _SortedCursor]
    library_dir: Path | None = None

    def has_pending(self) -> bool:
        """Whether any row of the slot still waits for its refresh node."""
        live = [self.cursor] if self.cursor is not None else list(self.cursors.values())
        return any(cursor.has_pending() for cursor in live)


def _range_indexed(frame: pd.DataFrame) -> pd.DataFrame:
    # Parquet-loaded frames already carry a clean RangeIndex, and reset_index
    # would copy the whole frame (a multi-million-row object union costs tens
    # of minutes to copy).
    if isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1:
        return frame
    return frame.reset_index(drop=True)


def _concat_slices(slices: list[pd.DataFrame]) -> pd.DataFrame:
    return slices[0] if len(slices) == 1 else pd.concat(slices, ignore_index=True)


def _dataset_cursors(datasets: np.ndarray, valid_indices: np.ndarray, keys_all: np.ndarray) -> dict[str, _SortedCursor]:
    cursors: dict[str, _SortedCursor] = {}
    valid_datasets = datasets[valid_indices]
    for name in np.unique(valid_datasets):
        selection = valid_indices[valid_datasets == name]
        cursors[str(name)] = _SortedCursor(selection, keys_all[selection])
    return cursors

# Agent-readable parquet domains: (view name, snapshot/replay file, whole-domain
# cutoff key or None for per-dataset gating).
_DOMAINS: tuple[tuple[str, str, str | None], ...] = (
    ("daily", "daily.parquet", "daily"),
    ("events", "events.parquet", None),
    # Macro gates per dataset too: the global tier lands on its own
    # natural-day node while domestic series stay on the trading-evening
    # node — a whole-domain cutoff exposed weekend-stamped domestic rows
    # after weekend global runs that never ingested them.
    ("macro", "macro.parquet", None),
    ("fundamentals", "fundamentals.parquet", "fundamentals"),
    ("intraday_1min", "intraday_1min.parquet", "intraday_1min"),
    # Auction rows carry the uniform official 09:29 publish stamp, so this
    # domain rolls directly on each row instead of a cron-node approximation.
    ("auction", "auction.parquet", _ROW_AVAILABLE_AT),
)

# Every domain the rolling view exposes as a DIRECTORY of parquet parts under
# ``ctx.asof_dir``. Single source for the static strategy check that rejects
# reading one of them as a flat ``<domain>.parquet`` file (the shape the frozen
# decision snapshot has). ``text_library`` is deliberately absent: its shards
# really are individual ``<name>.parquet`` files.
ASOF_DOMAIN_NAMES: tuple[str, ...] = tuple(
    sorted({name for name, _file, _key in _DOMAINS} | {"text_index", "universe"})
)

# Per-dataset cutoff/boundary resolvers for domains gated dataset-by-dataset.
_DATASET_CUTOFFS = {
    "events": (event_dataset_visible_cutoff, event_dataset_next_visible_boundary),
    "macro": (macro_dataset_visible_cutoff, macro_dataset_next_visible_boundary),
}


class Timeview:
    """Builds and rolls the as-of view for one replay."""

    def __init__(
        self,
        *,
        host_dir: Path,
        snapshot_dir: Path,
        replay: Mapping[str, ReplayRows],
        replay_text_library_dir: Path | None = None,
        incremental_domains: set[str] | frozenset[str] | None = None,
        stash_dir: Path | None = None,
    ) -> None:
        self.host_dir = Path(host_dir)
        self.snapshot_dir = Path(snapshot_dir)
        # Fresh per backtest: no stale parts from an earlier run leak in. Keep
        # the root inode because formal Docker mounts this directory read-only
        # before the host starts rolling parts into it.
        self.host_dir.mkdir(parents=True, exist_ok=True)
        for child in self.host_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        self._version = 0
        # After the first refresh, ordinary inference calls need one timestamp comparison
        # only. Domain/node traversal resumes when the simulation clock reaches
        # the earliest pending row or refresh-node boundary.
        self._boundary_gate_ready = False
        self._next_boundary: np.datetime64 | None = None
        self._domains: dict[str, _DomainView] = {}
        stash_root = _validated_stash_root(stash_dir)
        incremental = frozenset(incremental_domains or ())
        for name, filename, cutoff_key in _DOMAINS:
            self._domains[name] = _DomainView(
                name=name,
                cutoff_key=cutoff_key,
                out_dir=self.host_dir / name,
                frozen_file=self.snapshot_dir / filename,
                replay=replay.get(name),
                incremental=name in incremental,
                stash_dir=(stash_root / name) if stash_root is not None else None,
            )
        self._text = _TextView(
            out_index_dir=self.host_dir / "text_index",
            out_library_dir=self.host_dir / "text_library",
            frozen_index_file=self.snapshot_dir / "text_index.parquet",
            frozen_library_dir=self.snapshot_dir / "text_library",
            replay_index=replay.get("text_index"),
            replay_library_dir=Path(replay_text_library_dir) if replay_text_library_dir is not None else None,
            stash_index_dir=(stash_root / "text_index") if stash_root is not None else None,
            stash_library_dir=(stash_root / "text_library") if stash_root is not None else None,
        )
        # The universe never rolls. Expose it as a parts directory so
        # ``asof_dir + "/universe"`` matches every other domain.
        universe = self.snapshot_dir / "universe.parquet"
        if universe.exists():
            universe_dir = self.host_dir / "universe"
            universe_dir.mkdir(parents=True, exist_ok=True)
            _link_or_copy(universe, universe_dir / "part_0000.parquet")

    def append_replay_partition(self, domain: str, replay: pd.DataFrame) -> None:
        """Add one bounded source partition to an incremental replay domain."""
        try:
            view = self._domains[str(domain)]
        except KeyError as exc:
            raise ValueError(f"unknown Timeview domain: {domain}") from exc
        view.append_replay_partition(replay)
        # A newly added partition may already be visible at the current node;
        # force one domain traversal on the next refresh before restoring O(1).
        self._boundary_gate_ready = False

    def continue_into(
        self,
        open_slot: Callable[[], Mapping[str, ReplayRows]],
        *,
        replay_text_library_dir: Path | None,
        stash_dir: Path | None,
    ) -> None:
        """Run this view on into the next consecutive replay slot.

        The caller guarantees the slot continues the earlier ones: every row it
        holds becomes available after every row they hold. Every domain first
        drops the slots it has published entirely, then ``open_slot`` opens the
        next slot's domains, which read only their gating columns. Parts keep
        their numbering and from here on publish under the next slot's
        ``stash_dir``.
        """
        stash_root = _validated_stash_root(stash_dir)
        for view in self._domains.values():
            view.release_published()
        self._text.release_published()
        replay = open_slot()
        for name, _filename, _key in _DOMAINS:
            self._domains[name].continue_slot(
                replay.get(name),
                stash_dir=(stash_root / name) if stash_root is not None else None,
            )
        self._text.continue_slot(
            replay.get("text_index"),
            library_dir=Path(replay_text_library_dir) if replay_text_library_dir is not None else None,
            stash_index_dir=(stash_root / "text_index") if stash_root is not None else None,
            stash_library_dir=(stash_root / "text_library") if stash_root is not None else None,
        )
        self._boundary_gate_ready = False

    def refresh(self, when: pd.Timestamp) -> tuple[str, str]:
        """Append any newly-visible rows at ``when`` and return the container-mapped
        ``asof_dir`` plus the current ``asof_version`` (bumps on each new part)."""
        now = _cutoff_ns(when)
        if self._boundary_gate_ready and (
            self._next_boundary is None or now < self._next_boundary
        ):
            return self._result()
        for view in self._domains.values():
            if view.roll(when):
                self._version += 1
        if self._text.roll(when):
            self._version += 1
        boundaries = [view.next_boundary(when) for view in self._domains.values()]
        boundaries.append(self._text.next_boundary(when))
        pending = [boundary for boundary in boundaries if boundary is not None]
        self._next_boundary = min(pending) if pending else None
        self._boundary_gate_ready = True
        return self._result()

    def _result(self) -> tuple[str, str]:
        return str(self.host_dir), str(self._version)


class _DomainView:
    """One domain's growing directory of parquet parts."""

    def __init__(
        self,
        *,
        name: str,
        cutoff_key: str | None,
        out_dir: Path,
        frozen_file: Path,
        replay: ReplayRows | None,
        incremental: bool = False,
        stash_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.cutoff_key = cutoff_key
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.incremental = bool(incremental)
        self._stash_dir = stash_dir
        self._pending: list[_PendingReplayPartition] = []
        # One segment per replay slot a non-incremental domain has been given.
        self._segments: list[_ReplaySegment] = []
        self._dataset_names: list[str] = []
        self._part_seq = 0
        self._last_signature: object = object()  # sentinel: force the first roll
        self._frozen_schema: pa.Schema | None = None
        self._columns = self._init_frozen_part(frozen_file)
        self.continue_slot(replay, stash_dir=stash_dir)

    def continue_slot(self, rows: ReplayRows | None, *, stash_dir: Path | None) -> None:
        """Add one replay slot's rows; parts from here on publish under ``stash_dir``."""
        self._stash_dir = stash_dir
        if rows is None or rows.empty:
            return
        if self.incremental:
            raise ValueError(f"Timeview domain {self.name} is incremental and takes partitions only")
        # A replay domain can only roll if it carries the row-level available_at the
        # node gate needs; without it the domain stays frozen-only (conservative).
        if "available_at" not in rows.columns:
            return
        # The agent-facing schema drops the gating-only available_at unless the
        # frozen domain already carries it (events/macro/fundamentals do; daily
        # does not).
        if not self._columns:
            self._columns = [column for column in rows.columns if column != "available_at"]
        self._require_schema_covers(rows.columns)
        available_at = to_cn_timestamps(rows.column("available_at"))
        valid = np.flatnonzero(available_at.notna().to_numpy())  # NaT rows never become visible
        keys = _utc_ns(available_at)
        if self.cutoff_key is not None:
            self._segments.append(_ReplaySegment(rows, _SortedCursor(valid, keys[valid]), {}))
        elif "dataset" in rows.columns:
            cursors = _dataset_cursors(rows.column("dataset").astype(str).to_numpy(), valid, keys)
            self._segments.append(_ReplaySegment(rows, None, cursors))
        self._segments_changed()

    def release_published(self) -> None:
        """Drop the slots whose every row a refresh has published."""
        self._segments = [segment for segment in self._segments if segment.has_pending()]
        self._segments_changed()

    def _segments_changed(self) -> None:
        self._dataset_names = sorted(
            {name for segment in self._segments for name in segment.cursors}
        )
        self._last_signature = object()

    def append_replay_partition(self, replay: pd.DataFrame) -> None:
        if not self.incremental:
            raise ValueError(f"Timeview domain {self.name} is not incremental")
        if replay.empty or "available_at" not in replay.columns:
            return
        frame = _range_indexed(replay)
        available_at = to_cn_timestamps(frame["available_at"])
        valid = np.flatnonzero(available_at.notna().to_numpy())
        if valid.size == 0:
            return
        keys_all = _utc_ns(available_at)
        if valid.size != len(frame):
            frame = frame.iloc[valid].reset_index(drop=True)
            keys = keys_all[valid]
        else:
            keys = keys_all
        if not self._columns:
            self._columns = [column for column in frame.columns if column != "available_at"]
        self._require_schema_covers(list(frame.columns))
        self._pending.append(_PendingReplayPartition(frame=frame, keys=keys))
        self._last_signature = object()

    def _require_schema_covers(self, columns: Sequence[str]) -> None:
        """Surface replay columns the roll's projection will drop.

        The canonical schema is fixed by the frozen part (parts must share one
        schema for the documented DuckDB ``dir/*.parquet`` read), and union
        domains can legitimately carry replay-only columns when a dataset has
        rows only inside the replay window — so this warns loudly instead of
        failing, but never drops silently."""
        if not columns or not self._columns:
            return
        # available_at / available_at_rule are gating annotations, not strategy
        # data — the frozen schema drops them by design (intraday replay rows
        # carry the rule tag from the auction correction, 469 warnings/replay
        # otherwise). Warn once per domain: the schema cannot change mid-replay.
        if getattr(self, "_schema_drop_warned", False):
            return
        extra = sorted(set(columns) - set(self._columns) - {"available_at", "available_at_rule"})
        if extra:
            self._schema_drop_warned = True
            warnings.warn(
                f"Timeview domain {self.name!r}: replay columns {extra} are absent from the "
                "frozen snapshot schema and will not appear in the rolling view; rebuild the "
                "frozen snapshot if strategies need them",
                RuntimeWarning,
                stacklevel=2,
            )

    def _init_frozen_part(self, frozen_file: Path) -> list[str]:
        """Seed part 0 from the frozen snapshot domain and fix the canonical schema.

        A non-empty frozen file is hardlinked unchanged and its columns become the
        canonical schema. A zero-row file is also hardlinked when its footer already
        owns a concrete, non-null Arrow schema (for example canonical empty auction).
        Legacy zero-column/null-typed files write no part because they cannot unify
        safely with typed replay parts appended later.

        Emptiness and schema come from the parquet FOOTER (num_rows + arrow
        schema): the frozen daily/minute domains run to gigabytes and reading
        them whole here dominated Timeview init."""
        frozen_columns: list[str] = []
        if frozen_file.exists():
            footer = pq.ParquetFile(frozen_file)
            frozen_columns = list(footer.schema_arrow.names)
            schema = footer.schema_arrow
            typed_empty = (
                footer.metadata.num_rows == 0
                and len(schema) > 0
                and all(field.type != pa.null() for field in schema)
            )
            if footer.metadata.num_rows > 0 or typed_empty:
                _link_or_copy(frozen_file, self.out_dir / "part_0000.parquet")
                self._part_seq = 1
                # Types for canonical columns a replay frame does not carry:
                # every part in the directory has to unify with part 0.
                self._frozen_schema = schema
        # Without frozen columns the first replay rows fix the schema (continue_slot).
        return frozen_columns

    def roll(self, when: pd.Timestamp) -> bool:
        """Append a part for rows newly visible at ``when``; return True if written."""
        if self.incremental:
            return self._roll_incremental(when)
        if not self._segments:
            return False
        signature = self._signature(when)
        if signature == self._last_signature:
            return False  # this domain's covering node(s) have not advanced
        self._last_signature = signature
        taken: list[tuple[_ReplaySegment, np.ndarray]] = []
        for segment in self._segments:
            newly = self._newly_visible(segment, when)
            if newly.size:
                newly.sort()  # original frame order: parts read back exactly as the frame slice
                taken.append((segment, newly))
        if not taken:
            return False
        # The rows are read only if the part has to be encoded: a stash hit
        # hardlinks the part and touches no row group.
        self._write_part(
            sum(int(newly.size) for _segment, newly in taken),
            lambda: (table for segment, newly in taken for table in segment.rows.take(newly)),
        )
        return True

    def _roll_incremental(self, when: pd.Timestamp) -> bool:
        if not self._pending:
            return False
        signature = self._signature(when)
        if signature == self._last_signature:
            return False
        self._last_signature = signature
        cutoff = domain_visible_cutoff(str(self.cutoff_key), when) if self.cutoff_key is not None else None
        if cutoff is None:
            return False
        cutoff_key = _cutoff_ns(cutoff)
        newly: list[pd.DataFrame] = []
        pending: list[_PendingReplayPartition] = []
        for partition in self._pending:
            visible = partition.keys <= cutoff_key
            if bool(np.all(visible)):
                newly.append(partition.frame)
            elif bool(np.any(visible)):
                newly.append(partition.frame.iloc[np.flatnonzero(visible)])
                hidden = np.flatnonzero(~visible)
                pending.append(
                    _PendingReplayPartition(
                        frame=partition.frame.iloc[hidden].reset_index(drop=True),
                        keys=partition.keys[hidden],
                    )
                )
            else:
                pending.append(partition)
        self._pending = pending
        if not newly:
            return False

        self._write_part(
            sum(len(frame) for frame in newly),
            lambda: (pa.Table.from_pandas(frame, preserve_index=False) for frame in newly),
        )
        return True

    def _part_schema(self, unified: pa.Schema) -> pa.Schema:
        """The canonical part schema, typed by the slices where they carry a column.

        Canonical columns the slices lack take the frozen part's type: every
        part in the directory has to unify with part 0.
        """
        fields: list[pa.Field] = []
        for name in self._columns:
            index = unified.get_field_index(name)
            if index >= 0:
                fields.append(unified.field(index))
                continue
            if self._frozen_schema is None:
                raise RuntimeError(
                    f"Timeview domain {self.name!r}: replay slice has no column {name!r} and "
                    "no frozen part fixes its type"
                )
            fields.append(self._frozen_schema.field(self._frozen_schema.get_field_index(name)))
        # Without the pandas metadata of the source: it still describes the
        # dropped gating columns.
        return pa.schema(fields)

    @staticmethod
    def _project(table: pa.Table, schema: pa.Schema) -> pa.Table:
        """One slice on the part schema.

        In Arrow: ``DataFrame.reindex(columns=...)`` rebuilt every block of
        the slice, 18-73 s on one day of minute bars (~700k rows over seven
        object columns) and 95 % of replay wall, against 0.3 s for selecting
        the converted arrays.
        """
        arrays = [
            table.column(name).cast(field.type)
            if name in table.column_names
            else pa.nulls(table.num_rows, type=field.type)
            for name, field in zip(schema.names, schema, strict=True)
        ]
        return pa.Table.from_arrays(arrays, schema=schema)

    def _write_slices(self, slices: Callable[[], Iterator[pa.Table]], row_count: int, path: Path) -> None:
        """Encode one part from its slices, one row group resident at a time.

        A part's schema is what the whole slice infers -- a column null in
        one slice and typed in another is typed -- and the writer needs it
        before the first row group. A part within one row group is unified
        and written whole. A larger one (a window opening years into a span
        publishes everything before it as its first part, tens of millions
        of rows) is converted twice, once for its schema and once for its
        rows, and written in the row groups one ``write_table`` call would
        cut, so the file is byte-identical without the part ever being
        resident. Every table is written with one chunk per column, as the
        whole slice was: the writer cuts its pages on chunk boundaries too,
        so a column chunked by row group encodes to different bytes.
        """
        if row_count <= _PART_ROW_GROUP_ROWS:
            unified = pa.concat_tables(list(slices()), promote_options="default")
            pq.write_table(self._project(unified, self._part_schema(unified.schema)).combine_chunks(), path)
            return
        schema = self._part_schema(
            pa.unify_schemas([table.schema for table in slices()], promote_options="default")
        )
        with pq.ParquetWriter(path, schema) as writer:
            buffered: list[pa.Table] = []
            rows = 0
            for table in slices():
                buffered.append(self._project(table, schema))
                rows += table.num_rows
                while rows >= _PART_ROW_GROUP_ROWS:
                    whole = pa.concat_tables(buffered)
                    writer.write_table(
                        whole.slice(0, _PART_ROW_GROUP_ROWS).combine_chunks(),
                        row_group_size=_PART_ROW_GROUP_ROWS,
                    )
                    rest = whole.slice(_PART_ROW_GROUP_ROWS)
                    buffered = [rest] if rest.num_rows else []
                    rows = rest.num_rows
            if buffered:
                writer.write_table(pa.concat_tables(buffered).combine_chunks(), row_group_size=_PART_ROW_GROUP_ROWS)

    def _write_part(self, row_count: int, slices: Callable[[], Iterator[pa.Table]]) -> None:
        """Write the next part, reusing the run-level stash when one is present.

        The PIT backend has already bound the stash to the exact semantic
        contract. For that fixed snapshot, replay slot, configuration, and
        refresh schedule, a part written by an earlier backtest is identical
        and can be hardlinked instead of sliced and re-encoded. The footer row
        count is still compared against the newly-visible count: any mismatch
        must fail rather than expose a wrong visibility slice. Publication and
        reuse are serialized per part so concurrent evaluations never observe
        a partial parquet file.
        """
        name = f"part_{self._part_seq:04d}.parquet"
        out = self.out_dir / name
        write = partial(self._write_slices, slices, row_count)
        if self._stash_dir is None:
            write(out)
        else:
            self._stash_dir.mkdir(parents=True, exist_ok=True)
            with _exclusive_part_lock(self._stash_dir / f".{name}.lock"):
                _publish_checked_part(
                    self._stash_dir / name,
                    out,
                    row_count=row_count,
                    write=write,
                    label=f"domain {self.name!r} {name}",
                )
        self._part_seq += 1

    def _newly_visible(self, segment: _ReplaySegment, when: pd.Timestamp) -> np.ndarray:
        if segment.cursor is not None:
            if self.cutoff_key == _ROW_AVAILABLE_AT:
                return segment.cursor.advance(_cutoff_ns(when))
            cutoff = domain_visible_cutoff(self.cutoff_key, when)
            return segment.cursor.advance(_cutoff_ns(cutoff)) if cutoff is not None else _EMPTY_INDICES
        if not segment.cursors:
            return _EMPTY_INDICES
        dataset_cutoff, _ = _DATASET_CUTOFFS[self.name]
        parts = [
            indices
            for name, cursor in segment.cursors.items()
            if (cutoff := dataset_cutoff(name, when)) is not None
            and (indices := cursor.advance(_cutoff_ns(cutoff))).size
        ]
        return np.concatenate(parts) if parts else _EMPTY_INDICES

    def next_boundary(self, when: pd.Timestamp) -> np.datetime64 | None:
        """Earliest instant after ``when`` at which this view may grow."""
        if self.incremental:
            if not self._pending or self.cutoff_key is None:
                return None
            boundary = domain_next_visible_boundary(str(self.cutoff_key), when)
            return _cutoff_ns(boundary) if boundary is not None else None
        boundaries = [
            boundary
            for segment in self._segments
            if (boundary := self._segment_boundary(segment, when)) is not None
        ]
        return min(boundaries) if boundaries else None

    def _segment_boundary(self, segment: _ReplaySegment, when: pd.Timestamp) -> np.datetime64 | None:
        if segment.cursor is not None:
            if not segment.cursor.has_pending():
                return None
            if self.cutoff_key == _ROW_AVAILABLE_AT:
                return segment.cursor.next_key()
            boundary = domain_next_visible_boundary(self.cutoff_key, when)
            return _cutoff_ns(boundary) if boundary is not None else None
        if not segment.cursors:
            return None
        _, dataset_boundary = _DATASET_CUTOFFS[self.name]
        boundaries = [
            boundary
            for name, cursor in segment.cursors.items()
            if cursor.has_pending()
            and (boundary := dataset_boundary(name, when)) is not None
        ]
        return min(map(_cutoff_ns, boundaries)) if boundaries else None

    def _signature(self, when: pd.Timestamp) -> object:
        if self.cutoff_key is not None:
            if self.cutoff_key == _ROW_AVAILABLE_AT:
                return _cutoff_ns(when)
            return str(domain_visible_cutoff(self.cutoff_key, when))
        if not self._dataset_names:
            return ()
        dataset_cutoff, _ = _DATASET_CUTOFFS[self.name]
        return tuple((d, str(dataset_cutoff(d, when))) for d in self._dataset_names)


class _TextView:
    """Rolling text index plus visible body shards.

    The frozen research library is fully visible at replay start, but replay-slot
    bodies are only copied into the view once their matching text_index rows pass
    the text refresh-node gate. This lets strategies do their own NLP from
    ``ctx.asof_dir`` without exposing future text bodies.

    Both part kinds go through the run-level stash under their own
    ``text_index``/``text_library`` names, so the filtered library read that
    cuts a body part is paid once per slot rather than once per candidate.
    """

    def __init__(
        self,
        *,
        out_index_dir: Path,
        out_library_dir: Path,
        frozen_index_file: Path,
        frozen_library_dir: Path,
        replay_index: ReplayRows | None,
        replay_library_dir: Path | None,
        stash_index_dir: Path | None = None,
        stash_library_dir: Path | None = None,
    ) -> None:
        self.out_index_dir = out_index_dir
        self.out_library_dir = out_library_dir
        self.out_index_dir.mkdir(parents=True, exist_ok=True)
        self.out_library_dir.mkdir(parents=True, exist_ok=True)
        # One segment per replay slot; each reads bodies from its own library.
        self._segments: list[_ReplaySegment] = []
        self._dataset_names: list[str] = []
        self._part_seq = 0
        self._last_signature: object = object()
        self._init_frozen(frozen_index_file, frozen_library_dir)
        self.continue_slot(
            replay_index,
            library_dir=replay_library_dir,
            stash_index_dir=stash_index_dir,
            stash_library_dir=stash_library_dir,
        )

    def continue_slot(
        self,
        rows: ReplayRows | None,
        *,
        library_dir: Path | None,
        stash_index_dir: Path | None,
        stash_library_dir: Path | None,
    ) -> None:
        """Add one replay slot's text index; parts from here on publish under its stash."""
        # Own stash names, so a stash written by an earlier code version holds
        # only the numeric domains and is neither read nor invalidated here.
        self._stash_index_dir = stash_index_dir
        self._stash_library_dir = stash_library_dir
        required = {"available_at", "dataset", "text_id"}
        if rows is not None and not rows.empty and required.issubset(rows.columns):
            available_at = to_cn_timestamps(rows.column("available_at"))
            valid = np.flatnonzero(available_at.notna().to_numpy())
            cursors = _dataset_cursors(rows.column("dataset").astype(str).to_numpy(), valid, _utc_ns(available_at))
            self._segments.append(_ReplaySegment(rows, None, cursors, library_dir))
        self._segments_changed()

    def release_published(self) -> None:
        """Drop the slots whose every index row a refresh has published."""
        self._segments = [segment for segment in self._segments if segment.has_pending()]
        self._segments_changed()

    def _segments_changed(self) -> None:
        self._dataset_names = sorted(
            {name for segment in self._segments for name in segment.cursors}
        )
        self._last_signature = object()

    def _init_frozen(self, frozen_index_file: Path, frozen_library_dir: Path) -> None:
        if frozen_index_file.exists():
            _link_or_copy(frozen_index_file, self.out_index_dir / "part_0000.parquet")
            self._part_seq = 1
        if frozen_library_dir.exists():
            for src in sorted(frozen_library_dir.glob("*.parquet")):
                _link_or_copy(src, self.out_library_dir / src.name)

    def roll(self, when: pd.Timestamp) -> bool:
        if not self._segments:
            return False
        signature = tuple((d, str(text_dataset_visible_cutoff(d, when))) for d in self._dataset_names)
        if signature == self._last_signature:
            return False
        self._last_signature = signature
        visible: list[tuple[_ReplaySegment, pd.DataFrame]] = []
        for segment in self._segments:
            parts = [
                indices
                for name, cursor in segment.cursors.items()
                if (cutoff := text_dataset_visible_cutoff(name, when)) is not None
                and (indices := cursor.advance(_cutoff_ns(cutoff))).size
            ]
            if parts:
                newly = np.concatenate(parts)
                newly.sort()  # original frame order: parts read back exactly as the frame slice
                visible.append((segment, pa.concat_tables(list(segment.rows.take(newly))).to_pandas()))
        if not visible:
            return False
        rows = _concat_slices([index_rows for _segment, index_rows in visible]).copy()
        datasets = rows["dataset"].astype(str)
        # Every row this roll makes visible is relabelled onto the body part
        # this roll writes for its dataset, so the whole column is one string
        # concatenation instead of a per-row lookup (0.30 s/day measured on a
        # real slot, ~73 s per replay).
        suffix = f"__part_{self._part_seq:04d}.parquet"
        rows["library_file"] = datasets + suffix
        # A dataset's body part is cut from the library of each slot whose
        # index rows it makes visible, in slot order.
        groups = [
            (
                dataset,
                [
                    (segment.library_dir, text_ids)
                    for segment, index_rows in visible
                    if (
                        text_ids := set(
                            index_rows.loc[
                                index_rows["dataset"].astype(str) == dataset, "text_id"
                            ].astype(str)
                        )
                    )
                ],
            )
            for dataset in sorted(set(datasets))
        ]
        index_name = f"part_{self._part_seq:04d}.parquet"
        if self._stash_index_dir is None or self._stash_library_dir is None:
            for dataset, group in groups:
                self._write_bodies(dataset, group, self.out_library_dir / f"{dataset}{suffix}")
            _write_frame(rows, self.out_index_dir / index_name)
        else:
            self._publish_roll(rows, groups, index_name=index_name, suffix=suffix)
        self._part_seq += 1
        return True

    def _publish_roll(
        self,
        rows: pd.DataFrame,
        groups: list[tuple[str, list[tuple[Path | None, set[str]]]]],
        *,
        index_name: str,
        suffix: str,
    ) -> None:
        """Publish one roll's body parts and its index part through the stash.

        Each part is a pure function of the semantic contract the PIT backend
        already bound, so a part an earlier backtest encoded is hardlinked
        instead of re-read and re-encoded. The whole roll is published under
        one lock, and the index part -- which fixes the visibility slice the
        body parts are cut from -- carries the same footer row-count check the
        numeric domains use.
        """

        stash_index = self._stash_index_dir
        stash_library = self._stash_library_dir
        assert stash_index is not None and stash_library is not None
        stash_index.mkdir(parents=True, exist_ok=True)
        stash_library.mkdir(parents=True, exist_ok=True)
        with _exclusive_part_lock(stash_index / f".{index_name}.lock"):
            for dataset, group in groups:
                name = f"{dataset}{suffix}"
                _publish_part(
                    stash_library / name,
                    self.out_library_dir / name,
                    write=partial(self._write_bodies, dataset, group),
                )
            _publish_checked_part(
                stash_index / index_name,
                self.out_index_dir / index_name,
                row_count=len(rows),
                write=partial(_write_frame, rows),
                label=f"text index {index_name}",
            )

    def next_boundary(self, when: pd.Timestamp) -> np.datetime64 | None:
        """Earliest pending text refresh-node boundary after ``when``."""
        boundaries = [
            boundary
            for segment in self._segments
            for name, cursor in segment.cursors.items()
            if cursor.has_pending()
            and (boundary := text_dataset_next_visible_boundary(name, when)) is not None
        ]
        return min(map(_cutoff_ns, boundaries)) if boundaries else None

    def _write_bodies(
        self, dataset: str, sources: list[tuple[Path | None, set[str]]], path: Path
    ) -> None:
        """Write the visible body rows one dataset contributes to one roll."""

        bodies = [
            rows
            for library_dir, text_ids in sources
            if not (rows := self._read_body_rows(library_dir, dataset, text_ids)).empty
        ]
        body = _concat_slices(bodies) if bodies else pd.DataFrame()
        if body.empty or "text_id" not in body.columns:
            body = pd.DataFrame(columns=["text_id", "body"])
        else:
            body = body[[c for c in ("text_id", "body") if c in body.columns]]
        _write_frame(body, path)

    @staticmethod
    def _read_body_rows(
        library_dir: Path | None, dataset: str, text_ids: set[str]
    ) -> pd.DataFrame:
        path = library_dir / f"{dataset}.parquet" if library_dir is not None else None
        if path is None or not path.exists() or not text_ids:
            return pd.DataFrame(columns=["text_id", "body"])
        try:
            return pd.read_parquet(path, filters=[("text_id", "in", sorted(text_ids))])
        except (TypeError, NotImplementedError):
            # Engine without predicate pushdown only. Corruption must raise:
            # a silent fallback would turn a broken partition into a full
            # scan of the multi-GB text library and hide the damage.
            body = pd.read_parquet(path)
            if body.empty or "text_id" not in body.columns:
                return pd.DataFrame(columns=["text_id", "body"])
            return body.loc[body["text_id"].astype(str).isin(text_ids)]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _publish_part(stash: Path, out: Path, *, write: Callable[[Path], None]) -> None:
    """Materialise ``stash`` if it is absent and hardlink it to ``out``.

    The caller holds the publication lock. A part already in the stash was
    written under the same bound semantic contract and is therefore identical,
    so it is linked rather than re-encoded.
    """

    if not stash.exists():
        tmp = stash.parent / f".{stash.name}.{uuid.uuid4().hex}.tmp"
        try:
            write(tmp)
            os.replace(tmp, stash)
        finally:
            if tmp.exists():
                tmp.unlink()
    os.link(stash, out)


def _publish_checked_part(
    stash: Path,
    out: Path,
    *,
    row_count: int,
    write: Callable[[Path], None],
    label: str,
) -> None:
    """Publish one stash part whose newly-visible row count is known.

    The footer row count of a reused part is compared against the fresh slice:
    any mismatch must fail rather than expose a wrong visibility slice.
    """

    if stash.exists():
        stashed_rows = int(pq.ParquetFile(stash).metadata.num_rows)
        if stashed_rows != row_count:
            raise RuntimeError(
                f"Timeview stash mismatch for {label}: stashed {stashed_rows} rows, "
                f"expected {row_count}"
            )
    _publish_part(stash, out, write=write)


@contextmanager
def _exclusive_part_lock(path: Path) -> Iterator[None]:
    """Serialize publication and validation of one shared stash part."""

    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validated_stash_root(stash_dir: Path | None) -> Path | None:
    stash_root = Path(stash_dir) if stash_dir is not None else None
    if stash_root is not None and not (stash_root / "contract.json").is_file():
        raise RuntimeError(
            f"Timeview stash has no validated semantic contract: {stash_root}"
        )
    return stash_root


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` (zero-copy on the same filesystem); copy on a
    cross-device or already-linked failure."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
