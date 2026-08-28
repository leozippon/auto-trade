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
"""

from __future__ import annotations

import fcntl
import os
import shutil
import uuid
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
        executor=None,
        snapshot_dir: Path,
        replay_frames: dict[str, pd.DataFrame],
        replay_text_library_dir: Path | None = None,
        incremental_domains: set[str] | frozenset[str] | None = None,
        stash_dir: Path | None = None,
    ) -> None:
        self.host_dir = Path(host_dir)
        self.executor = executor
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
        # The container mapping of host_dir is invariant for the whole replay;
        # resolving it on every refresh costs several filesystem walks (map_path does
        # multiple Path.resolve() calls), so it is computed once, lazily.
        self._mapped_dir: str | None = None
        self._domains: dict[str, _DomainView] = {}
        stash_root = Path(stash_dir) if stash_dir is not None else None
        if stash_root is not None and not (stash_root / "contract.json").is_file():
            raise RuntimeError(
                f"Timeview stash has no validated semantic contract: {stash_root}"
            )
        incremental = frozenset(incremental_domains or ())
        for name, filename, cutoff_key in _DOMAINS:
            replay = replay_frames.get(name)
            self._domains[name] = _DomainView(
                name=name,
                cutoff_key=cutoff_key,
                out_dir=self.host_dir / name,
                frozen_file=self.snapshot_dir / filename,
                replay=replay if replay is not None else pd.DataFrame(),
                incremental=name in incremental,
                stash_dir=(stash_root / name) if stash_root is not None else None,
            )
        self._text = _TextView(
            out_index_dir=self.host_dir / "text_index",
            out_library_dir=self.host_dir / "text_library",
            frozen_index_file=self.snapshot_dir / "text_index.parquet",
            frozen_library_dir=self.snapshot_dir / "text_library",
            replay_index=replay_frames.get("text_index", pd.DataFrame()),
            replay_library_dir=Path(replay_text_library_dir) if replay_text_library_dir is not None else None,
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
        if self._mapped_dir is None:
            self._mapped_dir = (
                self.executor.map_path(self.host_dir)
                if self.executor is not None
                else str(self.host_dir)
            )
        return self._mapped_dir, str(self._version)


class _DomainView:
    """One domain's growing directory of parquet parts."""

    def __init__(
        self,
        *,
        name: str,
        cutoff_key: str | None,
        out_dir: Path,
        frozen_file: Path,
        replay: pd.DataFrame,
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
        # Same guard as append_replay_partition: parquet-loaded frames already
        # carry a clean RangeIndex, and reset_index would copy the whole frame
        # (a multi-million-row object union costs tens of minutes to copy).
        if not (
            isinstance(replay.index, pd.RangeIndex)
            and replay.index.start == 0
            and replay.index.step == 1
        ):
            replay = replay.reset_index(drop=True)
        # A replay frame can only roll if it carries the row-level available_at the
        # node gate needs; without it the domain stays frozen-only (conservative).
        if replay.empty or "available_at" not in replay.columns:
            replay = pd.DataFrame()
        self.replay = pd.DataFrame() if self.incremental else replay
        self._part_seq = 0
        self._last_signature: object = object()  # sentinel: force the first roll
        self._cursor: _SortedCursor | None = None
        self._cursors: dict[str, _SortedCursor] = {}
        if not self.incremental and not replay.empty:
            available_at = to_cn_timestamps(replay["available_at"])
            valid = np.flatnonzero(available_at.notna().to_numpy())  # NaT rows never become visible
            keys = _utc_ns(available_at)
            if cutoff_key is not None:
                self._cursor = _SortedCursor(valid, keys[valid])
            elif "dataset" in replay.columns:
                self._cursors = _dataset_cursors(replay["dataset"].astype(str).to_numpy(), valid, keys)
        self._dataset_names: list[str] = sorted(self._cursors)
        self._columns = self._init_frozen_part(frozen_file)
        if not self.incremental:
            self._require_schema_covers(self.replay)
        if self.incremental and not replay.empty:
            self.append_replay_partition(replay)

    def append_replay_partition(self, replay: pd.DataFrame) -> None:
        if not self.incremental:
            raise ValueError(f"Timeview domain {self.name} is not incremental")
        if replay.empty or "available_at" not in replay.columns:
            return
        frame = replay
        if not (
            isinstance(frame.index, pd.RangeIndex)
            and frame.index.start == 0
            and frame.index.step == 1
        ):
            frame = frame.reset_index(drop=True)
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
        self._require_schema_covers(frame)
        self._pending.append(_PendingReplayPartition(frame=frame, keys=keys))
        self._last_signature = object()

    def _require_schema_covers(self, replay: pd.DataFrame) -> None:
        """Surface replay columns the roll's ``reindex`` will drop.

        The canonical schema is fixed by the frozen part (parts must share one
        schema for the documented DuckDB ``dir/*.parquet`` read), and union
        domains can legitimately carry replay-only columns when a dataset has
        rows only inside the replay window — so this warns loudly instead of
        failing, but never drops silently."""
        if replay is None or replay.empty or not self._columns:
            return
        # available_at / available_at_rule are gating annotations, not strategy
        # data — the frozen schema drops them by design (intraday replay rows
        # carry the rule tag from the auction correction, 469 warnings/replay
        # otherwise). Warn once per domain: the schema cannot change mid-replay.
        if getattr(self, "_schema_drop_warned", False):
            return
        extra = sorted(
            set(replay.columns) - set(self._columns) - {"available_at", "available_at_rule"}
        )
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
                return frozen_columns
        # The agent-facing schema drops the gating-only available_at unless the frozen
        # domain already carries it (events/macro/fundamentals do; daily does not).
        columns = frozen_columns or list(self.replay.columns)
        if "available_at" not in frozen_columns and "available_at" in columns:
            columns = [c for c in columns if c != "available_at"]
        return columns

    def roll(self, when: pd.Timestamp) -> bool:
        """Append a part for rows newly visible at ``when``; return True if written."""
        if self.incremental:
            return self._roll_incremental(when)
        if self.replay.empty:
            return False
        signature = self._signature(when)
        if signature == self._last_signature:
            return False  # this domain's covering node(s) have not advanced
        self._last_signature = signature
        newly = self._newly_visible(when)
        if newly.size == 0:
            return False
        newly.sort()  # original frame order: parts read back exactly as the frame slice
        self._write_part(int(newly.size), lambda: self.replay.iloc[newly].reindex(columns=self._columns))
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

        def build() -> pd.DataFrame:
            rows = newly[0] if len(newly) == 1 else pd.concat(newly, ignore_index=True)
            return rows.reindex(columns=self._columns)

        self._write_part(sum(len(frame) for frame in newly), build)
        return True

    def _write_part(self, row_count: int, build: Callable[[], pd.DataFrame]) -> None:
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
        stash = self._stash_dir / name if self._stash_dir is not None else None
        if stash is not None:
            stash.parent.mkdir(parents=True, exist_ok=True)
            lock = stash.parent / f".{name}.lock"
            with _exclusive_part_lock(lock):
                if stash.exists():
                    stashed_rows = int(pq.ParquetFile(stash).metadata.num_rows)
                    if stashed_rows != row_count:
                        raise RuntimeError(
                            f"Timeview stash mismatch for domain {self.name!r} {name}: stashed "
                            f"{stashed_rows} rows, expected {row_count}"
                        )
                else:
                    tmp = stash.parent / f".{name}.{uuid.uuid4().hex}.tmp"
                    try:
                        build().to_parquet(tmp, index=False)
                        os.replace(tmp, stash)
                    finally:
                        if tmp.exists():
                            tmp.unlink()
                os.link(stash, out)
        else:
            build().to_parquet(out, index=False)
        self._part_seq += 1

    def _newly_visible(self, when: pd.Timestamp) -> np.ndarray:
        if self._cursor is not None:
            if self.cutoff_key == _ROW_AVAILABLE_AT:
                return self._cursor.advance(_cutoff_ns(when))
            cutoff = domain_visible_cutoff(self.cutoff_key, when)
            return self._cursor.advance(_cutoff_ns(cutoff)) if cutoff is not None else _EMPTY_INDICES
        if not self._cursors:
            return _EMPTY_INDICES
        dataset_cutoff, _ = _DATASET_CUTOFFS[self.name]
        parts = [
            indices
            for name, cursor in self._cursors.items()
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
        if self._cursor is not None:
            if not self._cursor.has_pending():
                return None
            if self.cutoff_key == _ROW_AVAILABLE_AT:
                return self._cursor.next_key()
            boundary = domain_next_visible_boundary(self.cutoff_key, when)
            return _cutoff_ns(boundary) if boundary is not None else None
        if not self._cursors:
            return None
        _, dataset_boundary = _DATASET_CUTOFFS[self.name]
        boundaries = [
            boundary
            for name, cursor in self._cursors.items()
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
    """

    def __init__(
        self,
        *,
        out_index_dir: Path,
        out_library_dir: Path,
        frozen_index_file: Path,
        frozen_library_dir: Path,
        replay_index: pd.DataFrame,
        replay_library_dir: Path | None,
    ) -> None:
        self.out_index_dir = out_index_dir
        self.out_library_dir = out_library_dir
        self.out_index_dir.mkdir(parents=True, exist_ok=True)
        self.out_library_dir.mkdir(parents=True, exist_ok=True)
        self.replay_index = replay_index.reset_index(drop=True) if replay_index is not None else pd.DataFrame()
        self.replay_library_dir = replay_library_dir
        self._part_seq = 0
        self._last_signature: object = object()
        required = {"available_at", "dataset", "text_id"}
        if self.replay_index.empty or not required.issubset(self.replay_index.columns):
            self.replay_index = pd.DataFrame()
        self._cursors: dict[str, _SortedCursor] = {}
        if not self.replay_index.empty:
            available_at = to_cn_timestamps(self.replay_index["available_at"])
            valid = np.flatnonzero(available_at.notna().to_numpy())
            self._cursors = _dataset_cursors(
                self.replay_index["dataset"].astype(str).to_numpy(), valid, _utc_ns(available_at)
            )
        self._dataset_names = sorted(self._cursors)
        self._init_frozen(frozen_index_file, frozen_library_dir)

    def _init_frozen(self, frozen_index_file: Path, frozen_library_dir: Path) -> None:
        if frozen_index_file.exists():
            _link_or_copy(frozen_index_file, self.out_index_dir / "part_0000.parquet")
            self._part_seq = 1
        if frozen_library_dir.exists():
            for src in sorted(frozen_library_dir.glob("*.parquet")):
                _link_or_copy(src, self.out_library_dir / src.name)

    def roll(self, when: pd.Timestamp) -> bool:
        if self.replay_index.empty:
            return False
        signature = tuple((d, str(text_dataset_visible_cutoff(d, when))) for d in self._dataset_names)
        if signature == self._last_signature:
            return False
        self._last_signature = signature
        parts = [
            indices
            for name, cursor in self._cursors.items()
            if (cutoff := text_dataset_visible_cutoff(name, when)) is not None
            and (indices := cursor.advance(_cutoff_ns(cutoff))).size
        ]
        if not parts:
            return False
        newly = np.concatenate(parts)
        newly.sort()  # original frame order: parts read back exactly as the frame slice
        rows = self.replay_index.iloc[newly].copy()
        if "library_file" not in rows.columns:
            rows["library_file"] = rows["dataset"].astype(str) + ".parquet"
        library_files: dict[tuple[str, str], str] = {}
        for dataset, group in rows.groupby(rows["dataset"].astype(str), sort=True):
            part_name = f"{dataset}__part_{self._part_seq:04d}.parquet"
            self._write_body_part(dataset, set(group["text_id"].astype(str)), part_name)
            for source_file in group["library_file"].astype(str).unique():
                library_files[(dataset, source_file)] = part_name
        rows["library_file"] = [
            library_files.get(
                (str(row.get("dataset", "")), str(row.get("library_file", ""))),
                str(row.get("library_file", "")),
            )
            for _, row in rows.iterrows()
        ]
        rows.to_parquet(self.out_index_dir / f"part_{self._part_seq:04d}.parquet", index=False)
        self._part_seq += 1
        return True

    def next_boundary(self, when: pd.Timestamp) -> np.datetime64 | None:
        """Earliest pending text refresh-node boundary after ``when``."""
        boundaries = [
            boundary
            for name, cursor in self._cursors.items()
            if cursor.has_pending()
            and (boundary := text_dataset_next_visible_boundary(name, when)) is not None
        ]
        return min(map(_cutoff_ns, boundaries)) if boundaries else None

    def _write_body_part(self, dataset: str, text_ids: set[str], part_name: str) -> None:
        body = self._read_body_rows(dataset, text_ids)
        if body.empty or "text_id" not in body.columns:
            pd.DataFrame(columns=["text_id", "body"]).to_parquet(self.out_library_dir / part_name, index=False)
            return
        part = body[[c for c in ("text_id", "body") if c in body.columns]]
        part.to_parquet(self.out_library_dir / part_name, index=False)

    def _read_body_rows(self, dataset: str, text_ids: set[str]) -> pd.DataFrame:
        path = self.replay_library_dir / f"{dataset}.parquet" if self.replay_library_dir is not None else None
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


@contextmanager
def _exclusive_part_lock(path: Path) -> Iterator[None]:
    """Serialize publication and validation of one shared stash part."""

    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst`` (zero-copy on the same filesystem); copy on a
    cross-device or already-linked failure."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
