"""PIT inputs of one Paper run: the newest committed release, one book window.

A run reads the release the data updater committed last, so each morning sees
the previous session's data. The window is laid out like a research replay of
the book: a decision view anchored two sessions before the book's first
decision and one replay slot from the session before it through the target
session, whose own bars do not exist yet. Views are cached per release
generation under the book's state root, and a run drops the caches of every
older generation.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterable
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from autotrade.environment.data.research_release import pin_research_release
from autotrade.environment.data.snapshot import SnapshotConfig, load_snapshot_manifest
from autotrade.environment.llm import LLMProxy
from autotrade.environment.nl import NLConfig, NLService
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.replay.timeview import Timeview
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.engine import PaperDataNotReady
from autotrade.pipelines.calendar import load_sse_trading_days
from autotrade.pipelines.pit_backend import (
    HistoricalMinuteSource,
    ResearchPITSnapshotProvider,
    _AsOfReadOnlyView,
    _discard_ephemeral_asof,
    _load_replay_frames,
    _require_read_only_tree,
    load_slot_corporate_actions,
    required_release_raw_datasets,
)

PIT_CACHE_NAME = "pit"
LIVE_GENERATION = "live"


class BookPITData:
    def __init__(
        self,
        *,
        state_root: str | Path,
        raw_dir: str | Path,
        fundamental_events_root: str | Path,
        fundamental_events_status: str | Path,
        snapshot_config: SnapshotConfig,
        start: str,
        trade_date: str,
        nl_llm: LLMProxy | None = None,
        nl_config: NLConfig | None = None,
        nl_failure_policy: str = "return_error_with_audit",
        max_intraday_row_group_rows: int = 2_000_000,
    ) -> None:
        cache_root = Path(state_root).resolve() / PIT_CACHE_NAME
        generation_dir = _pin_newest_release(
            cache_root,
            raw_dir=raw_dir,
            fundamental_events_root=fundamental_events_root,
            fundamental_events_status=fundamental_events_status,
            snapshot_config=snapshot_config,
        )
        provider = ResearchPITSnapshotProvider(
            experiment_dir=generation_dir,
            raw_dir=raw_dir,
            fundamental_events_root=fundamental_events_root,
            fundamental_events_status=fundamental_events_status,
            config=snapshot_config,
        )
        self.generation_id = provider.release.generation_id or LIVE_GENERATION
        if generation_dir.name != self.generation_id:
            raise RuntimeError(f"release pin {generation_dir} resolved to generation {self.generation_id}")
        for entry in cache_root.iterdir():
            if entry != generation_dir:
                _remove_tree(entry)
        self.sessions = tuple(load_sse_trading_days(provider.release.raw_dir))
        self.release_end = provider.trading_days[-1]
        prior = [day for day in self.sessions if day < trade_date]
        if prior and self.release_end < prior[-1]:
            # Checked before the views are built: a late morning must not spend
            # the view build (minutes, tens of GB) only to be refused.
            raise PaperDataNotReady(
                f"committed data ends at {self.release_end} (generation {self.generation_id}); "
                f"the decision for {trade_date} needs sessions through {prior[-1]}"
            )
        # The slot opens one session before the book's first decision: a slot
        # needs daily bars, and on the first morning the decision session has
        # none yet. The rows of that extra session reach the Timeview through
        # the slot instead of the frozen view, so every decision sees exactly
        # what a replay anchored the session before would show.
        earlier = [day for day in self.sessions if day < start]
        if len(earlier) < 2:
            raise RuntimeError(f"the exchange calendar has fewer than two sessions before {start}")
        window_start, anchor = earlier[-1], earlier[-2]
        bundle = provider.prepare(
            fold=None,
            phase="paper",
            start=window_start,
            end=trade_date,
            decision_time=datetime.combine(pd.Timestamp(anchor).date(), time(23, 59, 59), tzinfo=CN_TZ),
        )
        self.snapshot_dir = Path(bundle.decision_ref).resolve(strict=True)
        replay_dir = Path(bundle.replay_ref).resolve(strict=True)
        _require_read_only_tree(self.snapshot_dir)
        manifest = load_snapshot_manifest(replay_dir)
        frames = _load_replay_frames(
            replay_dir,
            generation_id=provider.release.generation_id,
            replay_manifest=manifest,
            cache=provider._replay_frame_cache,
        )
        self.market = DailyMarketData(frames["daily"], load_slot_corporate_actions(replay_dir, manifest))
        self._daily = frames["daily"]
        runtime = generation_dir / "runtime" / trade_date
        if runtime.exists():
            _remove_tree(runtime)
        runtime.mkdir(parents=True)
        self.asof_dir = runtime / "asof"
        minute_path = replay_dir / "intraday_1min.parquet"
        self.minute_source = (
            HistoricalMinuteSource(minute_path, max_row_group_rows=max_intraday_row_group_rows)
            if minute_path.exists() and pq.ParquetFile(minute_path).metadata.num_rows
            else None
        )
        self.timeview = Timeview(
            host_dir=self.asof_dir,
            snapshot_dir=self.snapshot_dir,
            replay_frames=frames,
            replay_text_library_dir=replay_dir / "text_library",
            incremental_domains={"intraday_1min"} if self.minute_source is not None else None,
            # Paper inference timestamps come from the live calendar, so no
            # schedule-bound stash applies.
            stash_dir=None,
        )
        self._lock = _AsOfReadOnlyView(self.asof_dir)
        self._lock.lock()
        window = sum(1 for day in self.sessions if start <= day <= trade_date)
        self.nl_service = NLService.from_snapshot(
            self.asof_dir,
            llm=nl_llm,
            config=(nl_config or NLConfig()).for_replay(window),
            failure_policy=nl_failure_policy,
        )
        self.nl_query = self.nl_service.query
        self._refreshed: set[str] = set()

    def context_data(self, inference_at: datetime) -> StrategyDataView:
        key = inference_at.isoformat()
        if key in self._refreshed:
            raise RuntimeError(f"Paper Timeview refresh was requested twice: {key}")
        self._refreshed.add(key)
        self._lock.unlock_directories()
        try:
            if self.minute_source is not None:
                self.minute_source.append_visible(self.timeview, inference_at)
            path, version = self.timeview.refresh(pd.Timestamp(inference_at))
        finally:
            self._lock.lock()
        return StrategyDataView(str(self.snapshot_dir), path, version)

    def execution_price(self, symbol: str, when: datetime) -> float | None:
        if self.minute_source is None:
            return None
        return self.minute_source.price_at(symbol, when)

    def references(self, symbols: Iterable[str], before: str) -> dict[str, dict[str, object]]:
        """Name and last raw close before ``before`` of each symbol: host-side
        display quotes for the order sheet, never a strategy input."""

        wanted = sorted(set(symbols))
        if not wanted:
            return {}
        universe = pd.read_parquet(self.snapshot_dir / "universe.parquet", columns=["ts_code", "name"])
        names = dict(zip(universe["ts_code"].astype(str), universe["name"].astype(str), strict=True))
        floor = (pd.Timestamp(before) - pd.Timedelta(days=45)).strftime("%Y%m%d")
        history = pd.read_parquet(self.snapshot_dir / "daily.parquet", columns=["ts_code", "trade_date", "close"])
        frames = [history, self._daily[["ts_code", "trade_date", "close"]]]
        quotes = pd.concat(frames, ignore_index=True)
        quotes["trade_date"] = quotes["trade_date"].astype(str).str.replace("-", "").str[:8]
        quotes = quotes[
            quotes["ts_code"].astype(str).isin(wanted)
            & (quotes["trade_date"] >= floor)
            & (quotes["trade_date"] < before)
        ].sort_values(["ts_code", "trade_date"])
        last = quotes.groupby("ts_code").tail(1).set_index("ts_code")
        return {
            symbol: {
                "name": names.get(symbol, ""),
                "close": float(last.at[symbol, "close"]) if symbol in last.index else None,
                "close_date": str(last.at[symbol, "trade_date"]) if symbol in last.index else None,
            }
            for symbol in wanted
        }

    def close(self) -> None:
        self.nl_service.close()
        _discard_ephemeral_asof(self.asof_dir)


def _pin_newest_release(
    cache_root: Path,
    *,
    raw_dir: str | Path,
    fundamental_events_root: str | Path,
    fundamental_events_status: str | Path,
    snapshot_config: SnapshotConfig,
) -> Path:
    """Pin the newest committed release into a directory named by its generation."""

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = cache_root / f".pin-{uuid.uuid4().hex}"
    release = pin_research_release(
        experiment_dir=staging,
        raw_dir=raw_dir,
        fundamental_events_root=fundamental_events_root,
        fundamental_events_status=fundamental_events_status,
        required_raw_datasets=required_release_raw_datasets(snapshot_config),
    )
    # A lake without the updater's lock and generation marker (a local or
    # synthetic root) is read live, as pin_research_release defines it.
    target = cache_root / (release.generation_id or LIVE_GENERATION)
    if target.exists():
        _remove_tree(staging)
    else:
        staging.rename(target)
    return target


def _remove_tree(root: Path) -> None:
    """Remove a cache tree. Only directories are made writable: the view files
    are hardlinked between the cache's own slots, and a file mode is the inode's."""

    if root.is_file() or root.is_symlink():
        root.unlink()
        return
    if not root.exists():
        return
    for directory, _dirnames, _filenames in os.walk(root):
        Path(directory).chmod(0o755)
    shutil.rmtree(root)


__all__ = ["PIT_CACHE_NAME", "BookPITData"]
