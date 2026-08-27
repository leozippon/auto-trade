"""Research-release snapshots and scheduled PIT evaluation.

``generate_orders(context)`` runs only at the configured user schedule.
Historical minute rows can enter the read-only PIT research view and can also
serve as trusted, exact-time price observations for submitted orders; neither
use creates strategy ticks or a minute-driven environment loop.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from autotrade.environment.data.contracts import domain_visible_cutoff
from autotrade.environment.data.pit import PITDataStore, to_cn_timestamps
from autotrade.environment.data.research_release import (
    ResearchRelease,
    pin_research_release,
)
from autotrade.environment.data.snapshot import (
    SnapshotBuilder,
    SnapshotConfig,
    load_snapshot_manifest,
)
from autotrade.environment.data.summary import write_agent_data_summary
from autotrade.environment.executor import DockerStrategyExecutor
from autotrade.environment.nl import NLConfig, NLService
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.style import replay_style_analysis, write_style_rollup
from autotrade.environment.replay.timeview import Timeview
from autotrade.environment.runtime import chmod_tree, write_json_atomic
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import CN_TZ, StrategySchedule
from autotrade.environment.strategy_loader import validate_strategy_source

from .config import (
    SNAPSHOT_CACHE_FORMAT_VERSION,
    EvaluationRequest,
    EvaluationResult,
    SnapshotBundle,
    StrategyExperimentConfig,
)
from .experiment import DailyStrategyPipeline
from .pit_views_seed import pit_cache_provider_record, seed_pit_views

_PHASES = frozenset({"meta", "valid", "frozen_test", "heldout", "paper"})
_CORE_RAW_DATASETS = ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d")


def required_release_raw_datasets(config: SnapshotConfig) -> tuple[str, ...]:
    """Raw directories that the exact snapshot configuration can consume."""

    return tuple(
        dict.fromkeys(
            (
                *_CORE_RAW_DATASETS,
                *config.fundamental_datasets,
                *config.events_datasets,
                *config.macro_datasets,
                *config.text_datasets,
                *(("stk_mins_1min_by_date",) if config.include_intraday else ()),
            )
        )
    )


class ResearchPITSnapshotProvider:
    """Pin one committed release and cache immutable phase data views.

    Cache identities are explicit semantic path components over one pinned
    release and one exact ``SnapshotConfig``. A completed
    directory is accepted only when its manifest restates the requested time
    boundary; partial or conflicting directories fail explicitly.
    """

    def __init__(
        self,
        *,
        experiment_dir: str | Path,
        raw_dir: str | Path,
        fundamental_events_root: str | Path,
        fundamental_events_status: str | Path,
        config: SnapshotConfig | None = None,
        cache_root: str | Path | None = None,
        pit_views_seed: str | Path | None = None,
        pit_views_seed_required: bool = False,
    ) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.config = config or SnapshotConfig()
        self.release: ResearchRelease = pin_research_release(
            experiment_dir=self.experiment_dir,
            raw_dir=raw_dir,
            fundamental_events_root=fundamental_events_root,
            fundamental_events_status=fundamental_events_status,
            required_raw_datasets=required_release_raw_datasets(self.config),
        )
        self.cache_root = Path(cache_root).resolve() if cache_root is not None else self.experiment_dir / "pit_views"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._replay_frame_cache: _ReplayFrameCache = {}
        record = self._bind_cache_contract()
        if pit_views_seed is not None:
            seed_pit_views(
                self.cache_root,
                Path(pit_views_seed),
                expected_provider=record,
                required=pit_views_seed_required,
            )
        self.builder = SnapshotBuilder(
            self.release.raw_dir,
            self.release.fundamental_events_root,
            self.release.fundamental_events_status,
        )
        self.trading_days = PITDataStore(self.release.raw_dir).trade_dates("daily")
        if not self.trading_days:
            raise RuntimeError("pinned research release has no daily trading dates")

    def prepare(
        self,
        *,
        fold,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle:
        del fold
        if phase not in _PHASES:
            raise ValueError(f"unsupported PIT snapshot phase: {phase}")
        decision = _cn_datetime(decision_time)
        start_key = _date_key(start)
        end_key = _date_key(end)
        if start_key > end_key:
            raise ValueError("PIT snapshot phase start cannot be after end")
        decision_key = decision.strftime("%Y%m%dT%H%M%S%z")
        decision_dir = self.cache_root / "decision" / decision_key
        replay_dir = (
            self.cache_root
            / "replay"
            / phase
            / f"{start_key}_{end_key}_{decision_key}"
        )
        decision_manifest = self._decision_view(decision_dir, decision)
        replay_dir = self._replay_view(replay_dir, start_key, end_key, decision, phase)
        summary_dir = self.cache_root / "bundles" / phase / f"{start_key}_{end_key}_{decision_key}"
        summary_path = summary_dir / "data_summary.json"
        with _exclusive_lock(summary_dir.with_suffix(".lock")):
            if not summary_path.exists():
                summary_dir.mkdir(parents=True, exist_ok=True)
                write_agent_data_summary(
                    summary_path,
                    kind=phase,
                    fold_id=None,
                    views={"snapshot": (decision_dir, "/mnt/snapshot")},
                )
                chmod_tree(summary_dir, file_mode=0o444, dir_mode=0o555)
        return SnapshotBundle(
            snapshot_id=str(decision_manifest.get("snapshot_id") or ""),
            decision_ref=str(decision_dir),
            replay_ref=str(replay_dir),
            data_summary_ref=str(summary_path),
            generation_id=self.release.generation_id,
        )

    def _bind_cache_contract(self) -> dict[str, object]:
        path = self.cache_root / "provider.json"
        # The cache-format version is part of the binding contract: a view
        # built under an older on-disk contract is refused, never reused.
        record = pit_cache_provider_record(
            generation_id=self.release.generation_id,
            release_raw_dir=self.release.raw_dir,
            snapshot_config=self.config,
        )
        with _exclusive_lock(self.cache_root / ".provider.lock"):
            if path.exists():
                existing = _read_json(path)
                if existing != record:
                    raise RuntimeError("PIT view cache is already bound to a different release or configuration")
            else:
                write_json_atomic(path, record)
        return record

    def _decision_view(self, target: Path, decision: datetime) -> dict[str, object]:
        with _exclusive_lock(target.with_suffix(".lock")):
            if target.exists():
                manifest = load_snapshot_manifest(target)
                if manifest.get("kind") != "decision_input" or _cn_datetime(
                    datetime.fromisoformat(str(manifest.get("decision_time")))
                ) != decision:
                    raise RuntimeError(f"conflicting cached decision snapshot: {target}")
                return manifest
            staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                manifest = self.builder.build_decision_snapshot(
                    decision,
                    staging,
                    self.config,
                    prior_events=self._prior_decision_events(decision),
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, target)
                chmod_tree(target, file_mode=0o444, dir_mode=0o555)
                return manifest
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

    def _prior_decision_events(
        self, decision: datetime
    ) -> tuple[Path, datetime] | None:
        root = self.cache_root / "decision"
        if not root.is_dir():
            return None
        best_path: Path | None = None
        best_time: datetime | None = None
        for path in root.iterdir():
            if path.name.startswith(".") or not path.is_dir():
                continue
            events = path / "events.parquet"
            if not events.is_file():
                continue
            try:
                manifest = load_snapshot_manifest(path)
                when = _cn_datetime(
                    datetime.fromisoformat(str(manifest.get("decision_time")))
                )
            except (OSError, TypeError, ValueError):
                continue
            if manifest.get("kind") != "decision_input" or when >= decision:
                continue
            if best_time is None or when > best_time:
                best_path, best_time = events, when
        if best_path is None or best_time is None:
            return None
        return best_path, best_time

    def _replay_view(
        self,
        target: Path,
        start: str,
        end: str,
        decision: datetime,
        phase: str,
    ) -> Path:
        with _exclusive_lock(target.with_suffix(".lock")):
            if target.exists():
                manifest = load_snapshot_manifest(target)
                if not _replay_manifest_matches(
                    manifest, start=start, end=end, decision=decision, phase=phase
                ):
                    raise RuntimeError(f"conflicting cached replay slot: {target}")
                return target
            unphased = self.cache_root / "replay" / target.name
            if unphased.exists() and unphased.resolve() != target.resolve():
                try:
                    alt = load_snapshot_manifest(unphased)
                except (OSError, TypeError, ValueError):
                    alt = {}
                if _replay_manifest_matches(
                    alt, start=start, end=end, decision=decision, phase=phase
                ):
                    return unphased
            staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                self.builder.build_replay_slot(
                    start,
                    end,
                    staging,
                    label=phase,
                    config=self.config,
                    available_from=decision,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, target)
                chmod_tree(target, file_mode=0o444, dir_mode=0o555)
                return target
            finally:
                if staging.exists():
                    shutil.rmtree(staging)


@dataclass(frozen=True)
class _MinuteRowGroup:
    index: int
    first_available: pd.Timestamp
    last_available: pd.Timestamp
    rows: int


class HistoricalMinuteSource:
    """Bounded static minute data for PIT features and exact order prices."""

    def __init__(self, path: str | Path, *, max_row_group_rows: int = 2_000_000) -> None:
        if isinstance(max_row_group_rows, bool) or max_row_group_rows <= 0:
            raise ValueError("max_row_group_rows must be a positive integer")
        self.path = Path(path)
        self.max_row_group_rows = int(max_row_group_rows)
        self.parquet = pq.ParquetFile(self.path)
        names = list(self.parquet.schema_arrow.names)
        if "available_at" not in names:
            raise ValueError(f"historical minute data has no available_at column: {self.path}")
        column_index = names.index("available_at")
        groups: list[_MinuteRowGroup] = []
        previous_first: pd.Timestamp | None = None
        for index in range(self.parquet.metadata.num_row_groups):
            group = self.parquet.metadata.row_group(index)
            rows = int(group.num_rows)
            if rows > self.max_row_group_rows:
                raise ValueError(
                    f"historical minute row group {index} has {rows} rows, above the bounded limit "
                    f"{self.max_row_group_rows}: {self.path}"
                )
            statistics = group.column(column_index).statistics
            if rows and (statistics is None or not statistics.has_min_max):
                raise ValueError(
                    f"historical minute row group {index} lacks available_at statistics; "
                    "refusing an unbounded fallback read"
                )
            if rows:
                first = _cn_timestamp(statistics.min)
                last = _cn_timestamp(statistics.max)
                if last < first or last - first > pd.Timedelta(days=2):
                    raise ValueError(
                        f"historical minute row group {index} is not one bounded date partition: "
                        f"{first}..{last}"
                    )
                if previous_first is not None and first < previous_first:
                    raise ValueError("historical minute row groups are not ordered by available_at")
                previous_first = first
                groups.append(_MinuteRowGroup(index, first, last, rows))
        self.groups = tuple(groups)
        self._loaded: set[int] = set()
        self.loaded_rows = 0
        self.max_loaded_partition_rows = 0
        required = {"trade_time", "close"}
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"historical minute data is missing columns {missing}: {self.path}")
        if "ts_code" in names:
            self._symbol_column = "ts_code"
        elif "symbol" in names:
            self._symbol_column = "symbol"
        else:
            raise ValueError(f"historical minute data has no symbol column: {self.path}")
        self._quote_group_index: int | None = None
        self._quote_group: pd.DataFrame | None = None

    @property
    def total_rows(self) -> int:
        return int(self.parquet.metadata.num_rows)

    @property
    def loaded_groups(self) -> int:
        return len(self._loaded)

    def append_visible(self, timeview: Timeview, when: datetime) -> None:
        cutoff = domain_visible_cutoff("intraday_1min", when)
        if cutoff is None:
            return
        cutoff_stamp = pd.Timestamp(cutoff)
        for group in self.groups:
            if group.index in self._loaded or group.first_available > cutoff_stamp:
                continue
            frame = self.parquet.read_row_group(group.index).to_pandas()
            if len(frame) != group.rows:
                raise RuntimeError(f"minute row-group size changed while reading {self.path}")
            timeview.append_replay_partition("intraday_1min", frame)
            self._loaded.add(group.index)
            self.loaded_rows += len(frame)
            self.max_loaded_partition_rows = max(self.max_loaded_partition_rows, len(frame))

    def price_at(self, symbol: str, when: datetime) -> float | None:
        """Return the close recorded at one exact historical minute.

        No rounding, forward fill, or next-event fallback is permitted. The
        returned observation stays on the trusted execution side and is never
        added to the strategy context ahead of its PIT availability.
        """

        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("execution timestamp must include a timezone")
        local = when.astimezone(CN_TZ)
        if local.second or local.microsecond:
            return None
        stamp = pd.Timestamp(local)
        matches: list[object] = []
        for group in self.groups:
            if stamp < group.first_available or stamp > group.last_available:
                continue
            frame = self._quote_frame(group.index)
            times = to_cn_timestamps(frame["trade_time"])
            available = to_cn_timestamps(frame["available_at"])
            selected = frame[
                frame[self._symbol_column].astype(str).str.strip().eq(str(symbol).strip())
                & times.eq(stamp)
                & available.le(stamp)
            ]
            matches.extend(selected["close"].tolist())
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                f"duplicate historical minute price for {symbol} at {local.isoformat()}: "
                f"rows={len(matches)}"
            )
        try:
            price = float(matches[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid historical minute price for {symbol} at {local.isoformat()}"
            ) from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError(
                f"invalid historical minute price for {symbol} at {local.isoformat()}: {price!r}"
            )
        return price

    def _quote_frame(self, index: int) -> pd.DataFrame:
        if self._quote_group_index != index or self._quote_group is None:
            self._quote_group = self.parquet.read_row_group(
                index,
                columns=[self._symbol_column, "trade_time", "available_at", "close"],
            ).to_pandas()
            self._quote_group_index = index
        return self._quote_group


class PITDailyEvaluationBackend:
    """Evaluate one daily strategy only against its supplied PIT bundle."""

    def __init__(
        self,
        results_root: str | Path,
        *,
        execution_mode: str,
        sandbox: SandboxConfig | None = None,
        nl_llm=None,
        nl_config: NLConfig | None = None,
        nl_failure_policy: str = "return_error_with_audit",
        max_intraday_row_group_rows: int = 2_000_000,
    ) -> None:
        if execution_mode not in {"sandbox", "trusted"}:
            raise ValueError("execution_mode must be sandbox or trusted")
        self.results_root = Path(results_root).resolve()
        self.execution_mode = execution_mode
        self.sandbox = sandbox or SandboxConfig()
        self.nl_llm = nl_llm
        self.nl_config = nl_config or NLConfig()
        self.nl_failure_policy = nl_failure_policy
        self.max_intraday_row_group_rows = int(max_intraday_row_group_rows)
        self._replay_frame_cache: _ReplayFrameCache = {}

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.mode not in {"valid", "frozen_test", "heldout"}:
            raise ValueError(f"unsupported PIT evaluation mode: {request.mode}")
        strategy_path = Path(request.revision.output_path) / "main.py"
        if not strategy_path.is_file():
            raise FileNotFoundError(f"strategy revision has no main.py: {strategy_path}")
        validate_strategy_source(strategy_path.read_text(encoding="utf-8"), filename="main.py")
        snapshot_dir = Path(request.snapshot.decision_ref).resolve(strict=True)
        replay_dir = Path(request.snapshot.replay_ref).resolve(strict=True)
        decision_manifest, replay_manifest = self._validate_bundle(
            request, snapshot_dir, replay_dir
        )
        _require_read_only_tree(snapshot_dir)
        stash_dir = _bind_asof_stash_contract(
            snapshot_dir=snapshot_dir,
            replay_dir=replay_dir,
            schedule=request.schedule,
            phase=request.mode,
            generation_id=request.snapshot.generation_id,
            decision_manifest=decision_manifest,
            replay_manifest=replay_manifest,
        )

        result_id = f"{request.mode}_{uuid.uuid4().hex}"
        result_dir = self.results_root / result_id
        result_dir.mkdir(parents=True, exist_ok=False)
        asof_dir = result_dir / "asof"
        keep_result_dir = False
        frames = _load_replay_frames(
            replay_dir,
            generation_id=request.snapshot.generation_id,
            replay_manifest=replay_manifest,
            cache=self._replay_frame_cache,
        )
        daily = frames["daily"]
        daily = daily[
            (daily["trade_date"].map(_date_key) >= _date_key(request.start))
            & (daily["trade_date"].map(_date_key) <= _date_key(request.end))
        ].copy()
        if daily.empty:
            raise ValueError(f"PIT daily replay is empty for {request.start}..{request.end}")

        minute_path = replay_dir / "intraday_1min.parquet"
        minute_source = (
            HistoricalMinuteSource(
                minute_path,
                max_row_group_rows=self.max_intraday_row_group_rows,
            )
            if minute_path.exists() and pq.ParquetFile(minute_path).metadata.num_rows
            else None
        )
        timeview = Timeview(
            host_dir=asof_dir,
            snapshot_dir=snapshot_dir,
            replay_frames={key: value for key, value in frames.items() if key != "daily"} | {"daily": daily},
            replay_text_library_dir=(replay_dir / "text_library"),
            incremental_domains={"intraday_1min"} if minute_source is not None else None,
            stash_dir=stash_dir,
        )
        lock = _AsOfReadOnlyView(asof_dir)
        lock.lock()
        nl_service = NLService.from_snapshot(
            asof_dir,
            llm=self.nl_llm,
            config=self.nl_config,
            failure_policy=self.nl_failure_policy,
        )
        refreshed: set[str] = set()

        def context_data(inference_at: datetime) -> StrategyDataView:
            key = inference_at.isoformat()
            if key in refreshed:
                raise RuntimeError(f"Timeview refresh was requested twice for one daily inference: {key}")
            refreshed.add(key)
            lock.unlock_directories()
            try:
                if minute_source is not None:
                    minute_source.append_visible(timeview, inference_at)
                path, version = timeview.refresh(pd.Timestamp(inference_at))
            finally:
                lock.lock()
            return StrategyDataView(str(snapshot_dir), path, version)

        config = StrategyExperimentConfig(
            strategy_path=strategy_path,
            schedule=request.schedule,
            broker_profile=request.broker_profile,
            execution_mode=self.execution_mode,  # type: ignore[arg-type]
            sandbox=self.sandbox,
        )
        executor_factory = None
        if self.execution_mode == "sandbox":
            executor_factory = lambda cfg: DockerStrategyExecutor(
                cfg.strategy_path,
                cfg.sandbox,
                snapshot_dir=snapshot_dir,
                asof_dir=asof_dir,
            )
        try:
            try:
                replay = DailyStrategyPipeline(
                    config,
                    nl_query=nl_service.query,
                    context_data=context_data,
                    execution_price=minute_source.price_at if minute_source is not None else None,
                    executor_factory=executor_factory,
                ).run(daily)
                record = replay.to_record()
            finally:
                nl_service.close()
                lock.lock()
            record["pit"] = {
                "snapshot_id": request.snapshot.snapshot_id,
                "generation_id": request.snapshot.generation_id,
                "decision_ref": str(snapshot_dir),
                "replay_ref": str(replay_dir),
                "refresh_calls": len(refreshed),
                "minute_row_groups_loaded": minute_source.loaded_groups if minute_source is not None else 0,
                "minute_rows_loaded": minute_source.loaded_rows if minute_source is not None else 0,
                "minute_max_loaded_partition_rows": (
                    minute_source.max_loaded_partition_rows if minute_source is not None else 0
                ),
                "minute_total_rows": minute_source.total_rows if minute_source is not None else 0,
            }
            target = result_dir / "result.json"
            write_json_atomic(target, record)
            write_style_rollup(
                result_dir,
                replay_style_analysis(
                    replay,
                    daily,
                    replay_dir=replay_dir,
                    snapshot_dir=snapshot_dir,
                    mode=request.mode,
                ),
            )
            summary = record.get("stats")
            if not isinstance(summary, dict):
                raise TypeError("daily replay omitted stats")
            keep_result_dir = True
            return EvaluationResult(dict(summary), str(target), complete=True)
        finally:
            _discard_ephemeral_asof(asof_dir)
            if not keep_result_dir:
                shutil.rmtree(result_dir, ignore_errors=True)

    @staticmethod
    def _validate_bundle(
        request: EvaluationRequest, snapshot_dir: Path, replay_dir: Path
    ) -> tuple[dict[str, object], dict[str, object]]:
        decision = load_snapshot_manifest(snapshot_dir)
        replay = load_snapshot_manifest(replay_dir)
        if decision.get("kind") != "decision_input":
            raise ValueError("EvaluationRequest decision_ref is not a decision snapshot")
        if replay.get("kind") != "replay_slot":
            raise ValueError("EvaluationRequest replay_ref is not a replay slot")
        if str(replay.get("period_start")) != _date_key(request.start) or str(
            replay.get("period_end")
        ) != _date_key(request.end):
            raise ValueError("EvaluationRequest range does not match its immutable replay slot")
        if str(replay.get("label") or "") != request.mode:
            raise ValueError("EvaluationRequest mode does not match its immutable replay slot")
        snapshot_id = str(decision.get("snapshot_id") or "")
        if snapshot_id != request.snapshot.snapshot_id:
            raise ValueError("EvaluationRequest snapshot_id does not match decision manifest")
        return decision, replay


class PaperPITData:
    """One-day Paper adapter over the same pinned release and Timeview contract."""

    def __init__(
        self,
        provider: ResearchPITSnapshotProvider,
        *,
        trade_date: str,
        runtime_root: str | Path,
        nl_llm=None,
        nl_config: NLConfig | None = None,
        nl_failure_policy: str = "return_error_with_audit",
        max_intraday_row_group_rows: int = 2_000_000,
    ) -> None:
        day = _date_key(trade_date)
        prior = [value for value in provider.trading_days if value < day]
        if not prior:
            raise RuntimeError(f"Paper PIT requires a prior trading day before {day}")
        prior_day = datetime.strptime(prior[-1], "%Y%m%d").replace(tzinfo=CN_TZ).date()
        decision_time = datetime.combine(prior_day, time(23, 59, 59), tzinfo=CN_TZ)
        self.bundle = provider.prepare(
            fold=None,
            phase="paper",
            start=day,
            end=day,
            decision_time=decision_time,
        )
        self.snapshot_dir = Path(self.bundle.decision_ref).resolve(strict=True)
        self.replay_dir = Path(self.bundle.replay_ref).resolve(strict=True)
        _require_read_only_tree(self.snapshot_dir)
        replay_manifest = load_snapshot_manifest(self.replay_dir)
        frames = _load_replay_frames(
            self.replay_dir,
            generation_id=self.bundle.generation_id,
            replay_manifest=replay_manifest,
            cache=provider._replay_frame_cache,
        )
        self.daily = frames["daily"]
        self.daily = self.daily[self.daily["trade_date"].map(_date_key) == day].copy()
        if self.daily.empty:
            raise RuntimeError(f"Paper PIT replay slot has no daily market rows for {day}")
        runtime = Path(runtime_root).resolve() / day
        runtime.mkdir(parents=True, exist_ok=True)
        self.asof_dir = runtime / "asof"
        minute_path = self.replay_dir / "intraday_1min.parquet"
        self.minute_source = (
            HistoricalMinuteSource(
                minute_path,
                max_row_group_rows=max_intraday_row_group_rows,
            )
            if minute_path.exists() and pq.ParquetFile(minute_path).metadata.num_rows
            else None
        )
        self.timeview = Timeview(
            host_dir=self.asof_dir,
            snapshot_dir=self.snapshot_dir,
            replay_frames={key: value for key, value in frames.items() if key != "daily"}
            | {"daily": self.daily},
            replay_text_library_dir=self.replay_dir / "text_library",
            incremental_domains={"intraday_1min"} if self.minute_source is not None else None,
            # Paper inference timestamps are supplied by the live caller rather
            # than one frozen StrategySchedule, so no schedule-bound part stash
            # can be reused safely here.
            stash_dir=None,
        )
        self._lock = _AsOfReadOnlyView(self.asof_dir)
        self._lock.lock()
        self.nl_service = NLService.from_snapshot(
            self.asof_dir,
            llm=nl_llm,
            config=nl_config or NLConfig(),
            failure_policy=nl_failure_policy,
        )
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

    def close(self) -> None:
        self.nl_service.close()
        self._lock.lock()


def _discard_ephemeral_asof(asof_dir: Path) -> None:
    """Drop the per-replay Timeview scratch. Durable evidence is result.json
    plus the immutable decision/replay refs; keeping asof here copies the
    text library and minutes into every Validation."""

    if not asof_dir.exists():
        return
    try:
        # Directories must be writable to unlink children. Do not chmod files:
        # Timeview hardlinks snapshot parquet into asof, so mode changes would
        # unfreeze the decision snapshot and fail the next Validation.
        for path in [asof_dir, *(item for item in asof_dir.rglob("*") if item.is_dir())]:
            path.chmod(0o755)
        shutil.rmtree(asof_dir)
    except OSError as exc:
        raise RuntimeError(f"cannot discard ephemeral asof view: {asof_dir}: {exc}") from exc


class _AsOfReadOnlyView:
    """Keep a trusted strategy's rolling view read-only between refreshes."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def unlock_directories(self) -> None:
        for path in [self.root, *(item for item in self.root.rglob("*") if item.is_dir())]:
            path.chmod(0o755)

    def lock(self) -> None:
        chmod_tree(self.root, file_mode=0o444, dir_mode=0o555)


_STASH_CONTRACT_SCHEMA_VERSION = 1
_ReplayFrameCache = dict[
    str, tuple[dict[str, object], dict[str, pd.DataFrame]]
]


def _asof_stash_dir(
    snapshot_dir: Path,
    replay_dir: Path,
    schedule: StrategySchedule,
    phase: str,
) -> Path:
    """Return the complete semantic stash hierarchy for one scheduled replay."""

    snapshot = Path(snapshot_dir).resolve()
    replay = Path(replay_dir).resolve()
    if (
        snapshot.parent.name != "decision"
        or replay.parent.parent.name != "replay"
        or snapshot.parent.parent != replay.parent.parent.parent
    ):
        raise RuntimeError(
            "PIT stash inputs must be decision/replay slots under one cache root"
        )
    if phase not in _PHASES:
        raise ValueError(f"unsupported PIT stash phase: {phase}")
    safe_phase = _safe_path_component(phase, label="evaluation phase")
    if replay.parent.name != safe_phase:
        raise RuntimeError("PIT replay slot phase does not match the evaluation phase")
    decision_slot = _safe_path_component(snapshot.name, label="decision slot")
    replay_slot = _safe_path_component(replay.name, label="replay slot")
    schedule_record = schedule.to_record()
    period = _safe_path_component(schedule_record["period"], label="schedule period")
    try:
        hour, minute = schedule_record["inference_time"].split(":")
    except ValueError as exc:
        raise RuntimeError("invalid StrategySchedule inference_time") from exc
    hour = _safe_path_component(hour, label="inference hour")
    minute = _safe_path_component(minute, label="inference minute")
    root = snapshot.parent.parent / "asof_stash"
    target = (
        root
        / "decision"
        / decision_slot
        / "replay"
        / replay_slot
        / "phase"
        / safe_phase
        / "schedule"
        / f"period={period}"
        / "inference_time"
        / f"hour={hour}"
        / f"minute={minute}"
    )
    if not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"PIT stash path escapes its cache root: {target}")
    return target


def _bind_asof_stash_contract(
    *,
    snapshot_dir: Path,
    replay_dir: Path,
    schedule: StrategySchedule,
    phase: str,
    generation_id: str,
    decision_manifest: dict[str, object],
    replay_manifest: dict[str, object],
) -> Path:
    """Bind a part stash to complete, directly-comparable PIT semantics."""

    stash_dir = _asof_stash_dir(snapshot_dir, replay_dir, schedule, phase)
    cache_root = Path(snapshot_dir).resolve().parent.parent
    provider_record = _read_json(cache_root / "provider.json")
    expected_provider_keys = {
        "schema_version",
        "generation_id",
        "release_raw_dir",
        "snapshot_config",
    }
    if set(provider_record) != expected_provider_keys:
        raise RuntimeError(
            "PIT provider contract has missing or unknown fields: "
            f"{sorted(set(provider_record) ^ expected_provider_keys)}"
        )
    if provider_record.get("schema_version") != SNAPSHOT_CACHE_FORMAT_VERSION:
        raise RuntimeError("PIT provider contract has an incompatible cache schema")
    configured_generation = str(provider_record.get("generation_id") or "")
    if not generation_id or configured_generation != generation_id:
        raise RuntimeError("PIT stash generation does not match the provider contract")
    snapshot_config = provider_record.get("snapshot_config")
    if not isinstance(snapshot_config, dict):
        raise TypeError("PIT provider contract has no SnapshotConfig record")
    _require_record_shape(
        snapshot_config,
        SnapshotConfig().to_record(),
        label="PIT provider SnapshotConfig",
    )
    safe_phase = _safe_path_component(phase, label="evaluation phase")
    if str(replay_manifest.get("label") or "") != safe_phase:
        raise RuntimeError("PIT replay manifest phase does not match the evaluation phase")
    for label, manifest in (
        ("decision", decision_manifest),
        ("replay", replay_manifest),
    ):
        raw_generation = manifest.get("raw_generation")
        if not isinstance(raw_generation, dict):
            raise TypeError(f"PIT {label} manifest has no raw generation record")
        manifest_generation = str(raw_generation.get("generation_id") or "")
        if manifest_generation != generation_id:
            raise RuntimeError(
                f"PIT {label} raw generation does not match the requested release"
            )
    contract: dict[str, object] = {
        "schema_version": _STASH_CONTRACT_SCHEMA_VERSION,
        "snapshot_cache_schema_version": SNAPSHOT_CACHE_FORMAT_VERSION,
        "generation_id": generation_id,
        "release_raw_dir": provider_record["release_raw_dir"],
        "snapshot_config": snapshot_config,
        "phase": safe_phase,
        "schedule": schedule.to_record(),
        "decision_slot": {
            "path": str(Path(snapshot_dir).resolve()),
            "manifest": decision_manifest,
        },
        "replay_slot": {
            "path": str(Path(replay_dir).resolve()),
            "manifest": replay_manifest,
        },
    }
    contract_path = stash_dir / "contract.json"
    lock_path = stash_dir.parent / f".{stash_dir.name}.contract.lock"
    with _exclusive_lock(lock_path):
        if stash_dir.exists():
            if not stash_dir.is_dir():
                raise RuntimeError(f"PIT stash path is not a directory: {stash_dir}")
            existing = _read_json(contract_path)
            if existing != contract:
                raise RuntimeError(
                    f"PIT stash contract conflicts with requested semantics: {contract_path}"
                )
        else:
            stash_dir.mkdir(parents=True)
            write_json_atomic(contract_path, contract)
    return stash_dir


def _safe_path_component(value: object, *, label: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or os.sep in component
        or (os.altsep is not None and os.altsep in component)
        or "\x00" in component
    ):
        raise RuntimeError(f"unsafe {label} in PIT stash path: {component!r}")
    return component


def _require_record_shape(
    record: dict[str, object], template: dict[str, object], *, label: str
) -> None:
    if set(record) != set(template):
        raise RuntimeError(
            f"{label} has missing or unknown fields: {sorted(set(record) ^ set(template))}"
        )
    for key, expected in template.items():
        actual = record[key]
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise TypeError(f"{label}.{key} must be an object")
            _require_record_shape(actual, expected, label=f"{label}.{key}")
        elif isinstance(expected, list) and not isinstance(actual, list):
            raise TypeError(f"{label}.{key} must be an array")


def _load_replay_frames(
    replay_dir: Path,
    *,
    generation_id: str,
    replay_manifest: dict[str, object],
    cache: _ReplayFrameCache,
) -> dict[str, pd.DataFrame]:
    key = str(replay_dir.resolve())
    identity: dict[str, object] = {
        "generation_id": generation_id,
        "replay_manifest": replay_manifest,
    }
    entry = cache.get(key)
    cached = entry[1] if entry is not None and entry[0] == identity else None
    if cached is None:
        frames: dict[str, pd.DataFrame] = {}
        for name, filename in (
            ("daily", "daily.parquet"),
            ("events", "events.parquet"),
            ("macro", "macro.parquet"),
            ("fundamentals", "fundamentals.parquet"),
            ("auction", "auction.parquet"),
            ("text_index", "text_index.parquet"),
        ):
            path = replay_dir / filename
            if path.exists():
                frames[name] = pd.read_parquet(path)
            elif name == "daily":
                raise FileNotFoundError(f"replay slot has no daily.parquet: {replay_dir}")
            else:
                frames[name] = pd.DataFrame()
        cache[key] = (identity, frames)
        cached = frames
    frames = dict(cached)
    frames["daily"] = cached["daily"].copy()
    return frames


def _require_read_only_tree(root: Path) -> None:
    writable = []
    for path in (root, *root.rglob("*")):
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as exc:
            raise RuntimeError(f"cannot inspect decision snapshot permissions: {path}: {exc}") from exc
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable.append(path)
            if len(writable) >= 3:
                break
    if writable:
        raise RuntimeError(f"decision snapshot is not read-only: {[str(path) for path in writable]}")


def _cn_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=CN_TZ)
    return value.astimezone(CN_TZ)


def _optional_cn_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    return _cn_datetime(datetime.fromisoformat(str(value)))


def _replay_manifest_matches(
    manifest: Mapping[str, object],
    *,
    start: str,
    end: str,
    decision: datetime,
    phase: str,
) -> bool:
    return (
        manifest.get("kind") == "replay_slot"
        and str(manifest.get("period_start")) == start
        and str(manifest.get("period_end")) == end
        and _optional_cn_datetime(manifest.get("available_from")) == decision
        and str(manifest.get("label") or "") == phase
    )


def _cn_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(CN_TZ)
    return stamp.tz_convert(CN_TZ)


def _date_key(value: object) -> str:
    return pd.Timestamp(str(value)).strftime("%Y%m%d")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid PIT cache record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"PIT cache record is not an object: {path}")
    return value


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "HistoricalMinuteSource",
    "PITDailyEvaluationBackend",
    "PaperPITData",
    "ResearchPITSnapshotProvider",
    "required_release_raw_datasets",
]
