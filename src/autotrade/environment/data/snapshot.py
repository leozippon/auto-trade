"""PIT snapshot construction (docs/environment-design.md §1).

Builds the seven domain files plus universe and manifest for one decision time:

    manifest.json, daily.parquet, intraday_1min.parquet, auction.parquet,
    fundamentals.parquet, events.parquet, macro.parquet, text_index.parquet,
    text_library/, universe.parquet

Every row satisfies ``available_at <= decision_time``. Datasets whose raw rows
carry an ``available_at`` column (events/macro/text/minute) are filtered on it;
the daily core uses the dataset contracts. The normalized trading-file unit
contract covers daily, minute and auction data — every conversion is recorded
in the manifest; events/macro/fundamentals/text keep TuShare per-source units
and their domain meta carries ``units="source"`` (env docs §1.4). Replay slots
(valid/test) are built separately and are NOT PIT-filtered: they are the
replay regions read only by backtest_tool.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.data_quality import read_quality_report
from autotrade.environment.data import PITDataStore, default_tushare_contracts
from autotrade.environment.data.auction import (
    AUCTION_CORRECTION_RULE_ID,
    AuctionCorrectionConfig,
    apply_open_auction_correction,
)
from autotrade.environment.data.contracts import (
    BOARD_TRADING_DATASETS,
    CN_TZ,
    STK_AUCTION_PRICE_ABS_TOLERANCE,
    read_committed_raw_generation,
)
from autotrade.environment.data.fundamental_events import (
    FUNDAMENTAL_EVENT_DATASETS,
    FUNDAMENTAL_SIDECAR_COLUMNS,
    read_fundamental_events,
)
from autotrade.environment.data.pit import concat_rows, to_cn_timestamps, yyyymmdd
from autotrade.environment.data.research_release import (
    DOMAIN_REPORT_TYPES,
    DOMAIN_STATUS_FILES,
)
from autotrade.environment.data.summary import (
    DATE_COLUMNS as PROFILE_DATE_COLUMNS,
)
from autotrade.environment.data.summary import (
    NULL_COUNT_COLUMNS as PROFILE_NULL_COLUMNS,
)
from autotrade.environment.data.summary import (
    iter_column_statistics,
    scalar_to_text,
)
from autotrade.environment.data.units import (
    normalize_auction_units,
    normalize_daily_units,
    validate_snapshot_units,
)
from autotrade.environment.runtime import new_id, utc_now_iso

SNAPSHOT_DOMAIN_WORKERS = 3
# Independent event/macro datasets are mostly parquet IO; overlap them so a
# 30-dataset union is not 30 serial scans.
_DATASET_UNION_WORKERS = 8
# Per-dataset partition fan-in: daily-partitioned event datasets hold thousands
# of small files whose parquet decode releases the GIL.
_PARTITION_READ_WORKERS = 8
# Replay minute partitions are much larger than event partitions.  Keep only
# three transformed daily frames in flight so reads and auction correction can
# overlap without turning a quarter into an unbounded in-memory concat.
_MINUTE_PARTITION_WORKERS = 3
# Board-domain gate enablement follows the board dataset contract directly.
_BOARD_TRADING_DATASETS = frozenset(BOARD_TRADING_DATASETS)
# Instrument registries in the macro domain (contract/bond basics): rows stay
# valid for the instrument's whole life, so the macro window floor must not
# truncate them; per-row available_at (list date) still enforces PIT.
MACRO_REGISTRY_DATASETS = frozenset({"fut_basic", "opt_basic", "cb_basic"})
_REGISTRY_WINDOW_FLOOR = pd.Timestamp("1990-01-01", tz=CN_TZ)

# Forward-scheduled event registries announce future events years ahead (IPO
# lockup expiries), so the DECISION snapshot windows them on the event date as
# well as announcement recency -- windowing on available_at alone silently
# deleted the largest near-term unlocks (measured 2026-08-10: 56.8% of the
# share-weighted unlock supply due within 90 days was invisible). Replay slots
# must NOT: their rows union with the frozen snapshot, so a slot only needs
# rows newly announced inside the replay window.
FORWARD_EVENT_DATE_COLUMNS = {"share_float_complete": "float_date"}

# Auction visibility is one constant for every trade date in every vintage:
# 09:29 of the trade date, the exchange's official publication deadline. This
# models SOURCE publication truth, exactly like the margin family's official
# rule stamps: the exchange published the data at 09:29 whether or not our
# capture pipeline landed it on time, and historical multi-day gaps were our
# own script defects, not source unavailability — the user's explicit
# decision is to backtest against the corrected world rather than replicate
# pipeline outages. Observed landing times stay recorded in the sidecars as
# operational evidence; they no longer influence visibility.
_AUCTION_PUBLISH_CLOCK = "09:29:00"
# Columns that must never enter frozen research inputs. The raw lake keeps
# them and the unit registry still classifies them for the data audit; they
# are dropped per dataset before the domain union, so a same-named column of
# another dataset (limit_list_ths.limit_amount, daily_info.total_share) is
# unaffected. Two removal reasons, both permanent:
#
# leakage — the column would carry post-decision state into history.
# unusable — the column is dead, dead for the recent years, mostly zero with
#   undocumented semantics, inconsistent with its declared unit, or carries
#   irreconcilable regimes inside one series. Evidence below comes from a
#   per-year raw-lake scan (2026-08-29) recorded in data docs §4. Columns that
#   are merely sparse by nature (block_trade.amount) or populated per vendor
#   category (kpl_list bid_*, the limit_list_ths pool fields) are NOT removed:
#   they are complete within the rows they describe.
SNAPSHOT_EXCLUDED_COLUMNS: dict[str, tuple[str, ...]] = {
    # avg_turnover: always present but 95.9% zeros, max 0.04, semantics
    # undocumented. interval_3/interval_6: last populated 2018-09, all-NA in
    # every later partition (8.0% of scanned rows overall).
    "bak_daily": ("avg_turnover", "interval_3", "interval_6"),
    # 板上成交金额: the source rewrites history to null and early-2020
    # magnitudes are anomalous — audit-only by contract (data docs §3.2).
    "limit_list_d": ("limit_amount",),
    # All-NA in every limit_type pool and every year; the pool-conditional
    # siblings (limit_order/limit_amount/turnover/rise_rate/sum_float) stay.
    "limit_list_ths": ("lu_limit_order",),
    # Populated 2020-2022 only, all-NA 2023+, and the count basis (笔 vs 万笔)
    # was never confirmed.
    "daily_info": ("trans_count",),
    # All-NA 2023+ (populated 78-100% 2020-2021, 27-34% in 2022); the
    # exchange-level amount/market-value columns of this dataset are complete.
    "sz_daily_info": ("vol", "total_share", "float_share"),
    # All-NA locally across 2020-2026 (0 of 1652 scanned rows).
    "us_tltr": ("e_factor",),
    # nt_accu alternates between a cumulative index level (~100.x) and a
    # cumulative percent change (0.2 in 202411-202412, -0.1 in 202502) with no
    # regime marker; town_accu/cnt_accu hold index levels throughout and stay.
    "cn_cpi": ("nt_accu",),
    # Free-text vendor fields pooled over 3136 distinct events: 71.5%/91.5%/
    # 36.5% non-empty but only 18.4%/21.9%/8.9% parse as numbers, and the scale
    # is per event (rig counts, PMI, indices, percentages). The calendar's
    # date/event/country columns stay.
    "eco_cal": ("value", "pre_value", "fore_value"),
    # All-NA except two Hong Kong indexes (300 of 16341 scanned rows); the
    # per-market vol column stays.
    "index_global": ("amount",),
    # Nightly CURRENT-STATE registry refresh ingested by list_date: these
    # mutable fields would leak post-decision state into history (data docs
    # §3.3). The static issuance terms stay.
    "cb_basic": ("conv_price", "remain_size", "newest_rating", "delist_date"),
    # Declared a CNY impairment amount, but 96.4% of values are |x| < 10 with
    # a median of 0.0005 — the unit contradicts the declared meaning.
    "fina_indicator_vip": ("impai_ttm",),
}
DomainBuildResult = tuple[dict[str, object], dict[str, object]]
DomainBuildTask = tuple[
    str,
    tuple[str, ...],
    Callable[[Mapping[str, DomainBuildResult]], DomainBuildResult],
]

# ---- Agent-visible dataset scope -------------------------------------------
# Two levels over one source. ``SELECTABLE_DATASETS`` is every dataset a domain
# can load: the raw lake downloads and audits all of them and the fail-closed
# unit registry classifies every one of their columns, so an experiment may
# select any subset. ``DEFAULT_DATASETS`` is the narrower scope built when an
# experiment selects nothing — the A-share daily/weekly stock material, one
# series per concept. Selectable-but-not-default datasets stay downloadable,
# auditable and opt-in per experiment; the scope and the evidence behind it are
# in docs/data-documentation.md.
#
# Never selectable (raw keeps downloading; re-enable once the stated condition
# is met):
# - pledge_detail: 24% of in-window business keys carry contradictory versions
#   (is_release/is_buyback/amount conflicts) with no version column — excluded
#   until source identity rules exist (data docs §4).
# - repurchase: same defect class at ~3% (conflicting amounts/price caps for
#   one announcement) — excluded with the same re-inclusion condition.
# - hm_list: daily-refreshed reference without PIT timestamps; contributed 0
#   rows to every snapshot ever built and injected two all-null columns.
# - slb_len_mm / slb_len: 转融通 feeds dead at source (last rows 2025-07-25,
#   zero-row ever since); datasets decommissioned.
# - cn_schedule: the source keeps only the current months, so it contributes
#   nothing to a historical replay.
SELECTABLE_DATASETS: dict[str, tuple[str, ...]] = {
    "events": (
        # 两融
        "margin",
        "margin_detail",
        "margin_secs",
        # per-stock money flow (one official series plus two vendor variants)
        # and the industry/concept aggregates derived from them
        "moneyflow",
        "moneyflow_dc",
        "moneyflow_ths",
        "moneyflow_ind_dc",
        "moneyflow_ind_ths",
        "moneyflow_cnt_ths",
        # per-stock trading structure
        "cyq_perf",
        "bak_daily",
        "block_trade",
        # shareholder structure and share supply
        "stk_holdernumber",
        "stk_holdertrade",
        "top10_holders",
        "top10_floatholders",
        "stk_surv",
        "new_share",
        "share_float_complete",
        # 龙虎榜 and the official full-market 涨跌停/炸板 list (history since
        # 2020-01; row-level available_at at trade-date 16:00)
        "top_list",
        "top_inst",
        "limit_list_d",
        # board-trading / sentiment cluster (row-level available_at; day-end or
        # next-morning labels — descriptive sentiment signals, never a truth
        # source for fills, tradability or risk)
        "kpl_list",
        "kpl_concept_cons",
        "dc_index",
        "dc_member",
        "limit_step",
        "limit_cpt_list",
        "limit_list_ths",
        "ths_hot",
        "dc_hot",
        "hm_detail",
    ),
    "macro": (
        # CN macro releases
        "cn_gdp",
        "cn_cpi",
        "cn_ppi",
        "cn_pmi",
        "cn_m",
        "sf_month",
        "eco_cal",
        # money market and policy rates
        "shibor",
        "shibor_quote",
        "shibor_lpr",
        "repo_daily",
        # A-share index and industry context
        "index_daily",
        "index_dailybasic",
        "sw_daily",
        "ci_daily",
        "ths_daily",
        # exchange aggregates and market-level money flow
        "daily_info",
        "sz_daily_info",
        "moneyflow_mkt_dc",
        "broker_recommend",
        # offshore equity, FX and US yield curves
        "index_global",
        "fx_daily",
        "us_tycr",
        "us_trycr",
        "us_tbr",
        "us_tltr",
        # derivatives and convertible bonds (non-tradable context; the Agent
        # computes basis, PCR and conversion premium itself)
        "fut_basic",
        "fut_mapping",
        "fut_daily",
        "opt_basic",
        "opt_daily",
        "cb_basic",
        "cb_daily",
        "cb_call",
    ),
    "text": (
        # per-stock text (every row carries ts_code)
        "anns_d",
        "research_report",
        "report_rc",
        "irm_qa_sh",
        "irm_qa_sz",
        # market-wide newswire (no ts_code on any row)
        "major_news",
        "news",
        "cctv_news",
        "npr",
    ),
    "fundamentals": FUNDAMENTAL_EVENT_DATASETS,
}

# What a snapshot carries when the experiment selects nothing. Excluded by
# default and why (measured on the 2025-12-31 decision snapshot; the share is
# of that domain's rows):
# - events: dc_member (55.1%) and kpl_concept_cons (6.6%) are concept-board
#   membership tables, ths_hot/dc_hot (5.6%) are hot-list rankings, moneyflow_dc
#   /moneyflow_ths (7.4%) duplicate moneyflow per stock and the moneyflow_ind_*
#   /moneyflow_cnt_ths series aggregate it by industry, margin_secs (3.0%) is
#   the eligibility roster behind margin_detail, top10_holders /
#   top10_floatholders / stk_surv are quarterly holder and survey tables, and
#   limit_step / limit_cpt_list / limit_list_ths / hm_detail extend the board
#   cluster past the official limit_list_d. Together 81% of the domain's rows.
# - macro: opt_*/fut_*/cb_* derivatives and convertible bonds (54.9%),
#   ths_daily and ci_daily duplicate sw_daily's industry-index role (32.7%),
#   and repo_daily, fx_daily, us_* curves, index_global, shibor_quote,
#   daily_info, sz_daily_info, moneyflow_mkt_dc, broker_recommend and eco_cal
#   are cross-asset or aggregate series a stock strategy can rebuild from the
#   daily domain. Together 93% of the domain's rows.
# - text: the four market-wide newswire feeds carry no ts_code on any row and
#   hold 83% of the text library's bytes.
DEFAULT_DATASETS: dict[str, tuple[str, ...]] = {
    "events": (
        "margin",
        "margin_detail",
        "moneyflow",
        "cyq_perf",
        "bak_daily",
        "block_trade",
        "stk_holdernumber",
        "stk_holdertrade",
        "new_share",
        "share_float_complete",
        "top_list",
        "top_inst",
        "limit_list_d",
        "kpl_list",
    ),
    "macro": (
        "cn_gdp",
        "cn_cpi",
        "cn_ppi",
        "cn_pmi",
        "cn_m",
        "sf_month",
        "shibor",
        "shibor_lpr",
        "index_daily",
        "index_dailybasic",
        "sw_daily",
    ),
    "text": (
        "anns_d",
        "research_report",
        "report_rc",
        "irm_qa_sh",
        "irm_qa_sz",
    ),
    "fundamentals": FUNDAMENTAL_EVENT_DATASETS,
}


@dataclass(frozen=True)
class SnapshotConfig:
    # Same base window the research calendar uses (RollingExperimentConfig).
    window_months: int = 24
    daily_window_months: int | None = None
    fundamentals_window_months: int | None = None
    events_window_months: int | None = None
    macro_window_months: int | None = None
    text_window_months: int | None = None
    # One trading month of decision-input minute bars; valid/test replay minute
    # windows are sized by the fold periods, not this field.
    intraday_trade_days: int = 21
    events_datasets: tuple[str, ...] = DEFAULT_DATASETS["events"]
    macro_datasets: tuple[str, ...] = DEFAULT_DATASETS["macro"]
    text_datasets: tuple[str, ...] = DEFAULT_DATASETS["text"]
    # Newswire knobs, effective only when the opt-in ``news`` dataset is
    # selected: every src= partition on disk and the full text window. Its
    # cross-source content dedup always applies — measured 43% of full-window
    # rows are duplicates (4.56M -> 2.60M, ~0.4GB library).
    news_sources: tuple[str, ...] = ()  # empty = all sources present on disk
    news_window_months: int | None = None  # None = follow the text window
    fundamental_datasets: tuple[str, ...] = DEFAULT_DATASETS["fundamentals"]
    # Minute bars are opt-in: they are the single largest domain to build and
    # carry, and a daily/weekly stock strategy reads them only to price an
    # order at an exact historical minute. With them off, execution resolves at
    # the open and the close and any other execute_at is rejected as
    # ``missing_execution_price``.
    include_intraday: bool = False
    include_industry: bool = True
    text_body_chars: int = 4000
    replay_include_events: bool = True
    replay_include_text: bool = True
    replay_include_minutes: bool = False  # follows include_intraday
    replay_include_macro: bool = True
    replay_include_fundamentals: bool = True
    # ---- universe screening (experiment-level research universe) ----
    # Applied to every per-stock domain (universe/daily/minutes/auction/events/
    # fundamentals) at snapshot AND replay-slot build, using only decision-time
    # knowledge (as-of names, list_date, latest daily_basic <= anchor). The set
    # is frozen at the decision anchor: codes turning ST / delisting inside the
    # replay period keep their data (they were eligible when chosen). Empty /
    # default values disable screening entirely (zero overhead).
    screen_exclude_st: bool = False
    screen_exclude_new_listed_days: int = 0
    screen_min_circ_mv_yi: float | None = None  # 流通市值下限（亿元）
    screen_max_circ_mv_yi: float | None = None  # 流通市值上限（亿元）
    screen_min_price: float | None = None
    screen_max_price: float | None = None
    screen_boards: tuple[str, ...] = ()  # subset of main/gem/star/bj; empty = all

    def screening_active(self) -> bool:
        return bool(
            self.screen_exclude_st
            or self.screen_exclude_new_listed_days > 0
            or self.screen_min_circ_mv_yi is not None
            or self.screen_max_circ_mv_yi is not None
            or self.screen_min_price is not None
            or self.screen_max_price is not None
            or self.screen_boards
        )

    def __post_init__(self) -> None:
        for domain, selected in (
            ("events", self.events_datasets),
            ("macro", self.macro_datasets),
            ("text", self.text_datasets),
            ("fundamentals", self.fundamental_datasets),
        ):
            unknown = sorted(set(selected) - set(SELECTABLE_DATASETS[domain]))
            if unknown:
                raise ValueError(f"unsupported {domain} datasets: {unknown}")
        for domain in ("daily", "fundamentals", "events", "macro", "text"):
            if self.months_for(domain) <= 0:
                raise ValueError(f"{domain}_months must be positive")
        if self.intraday_trade_days <= 0:
            raise ValueError("intraday_trade_days must be positive")
        if self.screen_exclude_new_listed_days < 0:
            raise ValueError("screen_exclude_new_listed_days must be >= 0")
        unknown_boards = set(self.screen_boards) - {"main", "gem", "star", "bj"}
        if unknown_boards:
            raise ValueError(f"unknown screen_boards: {sorted(unknown_boards)}")
        for low, high, label in (
            (self.screen_min_circ_mv_yi, self.screen_max_circ_mv_yi, "screen_circ_mv_yi"),
            (self.screen_min_price, self.screen_max_price, "screen_price"),
        ):
            if low is not None and high is not None and low > high:
                raise ValueError(f"{label}: min must be <= max")

    def months_for(self, domain: str) -> int:
        overrides = {
            "daily": self.daily_window_months,
            "fundamentals": self.fundamentals_window_months,
            "events": self.events_window_months,
            "macro": self.macro_window_months,
            "text": self.text_window_months,
        }
        if domain not in overrides:
            raise ValueError(f"unknown snapshot window domain: {domain}")
        return int(overrides[domain] if overrides[domain] is not None else self.window_months)

    def window_start_for(self, decision_time: datetime, domain: str) -> pd.Timestamp:
        return _window_start(decision_time, self.months_for(domain))

    def to_record(self) -> dict[str, object]:
        return {
            "decision_windows": {
                "daily_months": self.months_for("daily"),
                "fundamentals_months": self.months_for("fundamentals"),
                "events_months": self.months_for("events"),
                "macro_months": self.months_for("macro"),
                "text_months": self.months_for("text"),
                "intraday_trade_days": self.intraday_trade_days,
            },
            "datasets": {
                "events": list(self.events_datasets),
                "macro": list(self.macro_datasets),
                "text": list(self.text_datasets),
                "fundamentals": list(self.fundamental_datasets),
            },
            # Both change what gets built; leaving them out of the record left
            # snapshots that could not prove which news slice they contained.
            "news_sources": list(self.news_sources),
            "news_window_months": self.news_window_months,
            "include_intraday": self.include_intraday,
            "include_industry": self.include_industry,
            "text_body_chars": self.text_body_chars,
            "universe_screen": {
                "exclude_st": self.screen_exclude_st,
                "exclude_new_listed_days": self.screen_exclude_new_listed_days,
                "min_circ_mv_yi": self.screen_min_circ_mv_yi,
                "max_circ_mv_yi": self.screen_max_circ_mv_yi,
                "min_price": self.screen_min_price,
                "max_price": self.screen_max_price,
                "boards": list(self.screen_boards),
            },
            "replay": {
                "include_events": self.replay_include_events,
                "include_text": self.replay_include_text,
                "include_minutes": self.replay_include_minutes,
                "include_macro": self.replay_include_macro,
                "include_fundamentals": self.replay_include_fundamentals,
            },
        }


@dataclass
class SnapshotBuilder:
    raw_dir: Path
    fundamental_events_root: Path
    fundamental_events_status: Path | None

    def __init__(
        self,
        raw_dir: str | Path,
        fundamental_events_root: str | Path,
        fundamental_events_status: str | Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.fundamental_events_root = Path(fundamental_events_root)
        self.fundamental_events_status = Path(fundamental_events_status) if fundamental_events_status is not None else None
        # The per-domain audit status files live next to the fundamental one
        # (results/data_quality/); no status path disables the domain gates too.
        self.data_quality_dir = self.fundamental_events_status.parent if self.fundamental_events_status is not None else None
        self.contracts = default_tushare_contracts()
        self.store = PITDataStore(self.raw_dir)

    @contextmanager
    def _raw_lake_guard(self):
        """Shared flock over the cron updater's exclusive lock plus a
        generation double-read: a snapshot build never overlaps a raw-lake
        mutation run, and fails fast if the lake changed under it anyway
        (a writer bypassing the lock). Lakes without the cron lock file
        (manual/test raw dirs) have no updater to exclude."""
        lock_path = self.raw_dir.parent.parent / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        fd = None
        if lock_path.exists():
            fd = os.open(lock_path, os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            generation = read_committed_raw_generation(self.raw_dir)
            yield generation
            if read_committed_raw_generation(self.raw_dir) != generation:
                raise RuntimeError(f"raw lake generation changed during snapshot build under {self.raw_dir}")
        finally:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    # ---- decision-input snapshot ----

    def build_decision_snapshot(
        self,
        decision_time: datetime,
        output_dir: str | Path,
        config: SnapshotConfig | None = None,
        *,
        prior_events: tuple[Path, datetime] | None = None,
    ) -> dict[str, object]:
        with self._raw_lake_guard() as raw_generation:
            manifest = self._build_decision_snapshot_impl(
                decision_time,
                output_dir,
                config,
                raw_generation,
                prior_events=prior_events,
            )
        return manifest

    def _build_decision_snapshot_impl(
        self,
        decision_time: datetime,
        output_dir: str | Path,
        config: SnapshotConfig | None,
        raw_generation: dict[str, object] | None,
        *,
        prior_events: tuple[Path, datetime] | None = None,
    ) -> dict[str, object]:
        config = config or SnapshotConfig()
        decision_time = decision_time if decision_time.tzinfo else decision_time.replace(tzinfo=CN_TZ)
        decision_time = decision_time.astimezone(CN_TZ)
        daily_window_start = config.window_start_for(decision_time, "daily")
        fundamentals_window_start = config.window_start_for(decision_time, "fundamentals")
        events_window_start = config.window_start_for(decision_time, "events")
        macro_window_start = config.window_start_for(decision_time, "macro")
        text_window_start = config.window_start_for(decision_time, "text")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_quality_warnings = self._domain_status_gates(config)
        for domain, note in data_quality_warnings.items():
            print(f"snapshot data-quality note [{domain}]: {note}")
        domains: dict[str, dict[str, object]] = {}
        profiles: dict[str, dict[str, object]] = {}
        total_started = time.perf_counter()

        # Research-universe screen: one decision-time set restricts every
        # per-stock domain below (None = screening off, zero overhead).
        screened = self._screened_codes(decision_time, config)

        def build_daily(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            daily, meta = self._build_daily(decision_time, daily_window_start)
            daily = self._apply_screen(daily, screened)
            meta = {**meta, "rows": int(len(daily))}
            profile = _write_with_profile(
                output_dir / "daily.parquet", daily, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_intraday(completed: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            if config.include_intraday:
                daily_meta = completed["daily"][0]
                intraday, meta = self._build_intraday(decision_time, daily_meta["trade_dates"], config)
                intraday = self._apply_screen(intraday, screened)
                meta = {**meta, "rows": int(len(intraday))}
            else:
                intraday, meta = pd.DataFrame(), {"rows": 0, "datasets": [], "skipped": True}
            profile = _write_with_profile(
                output_dir / "intraday_1min.parquet", intraday, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_auction(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            auction, meta = self._build_auction(
                daily_window_start.strftime("%Y%m%d"), decision_time.strftime("%Y%m%d")
            )
            auction = self._apply_screen(auction, screened)
            if not auction.empty:
                available_at = to_cn_timestamps(auction["available_at"])
                auction = auction[available_at <= decision_time].reset_index(drop=True)
                meta = {**meta, "rows": int(len(auction))}
            profile = _write_with_profile(
                output_dir / "auction.parquet", auction, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_fundamentals(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            if config.fundamental_datasets:
                self._assert_fundamental_event_status_ok(
                    fundamentals_window_start, tuple(config.fundamental_datasets)
                )
            fundamentals = read_fundamental_events(
                self.fundamental_events_root,
                decision_time.isoformat(),
                datasets=config.fundamental_datasets,
                min_available_at=fundamentals_window_start.isoformat(),
                require_partitions=bool(config.fundamental_datasets),
            )
            fundamentals = self._apply_screen(fundamentals, screened)
            fundamentals, dataset_columns = _apply_fundamental_exclusions(
                fundamentals,
                _fundamental_dataset_columns(self.raw_dir, tuple(config.fundamental_datasets)),
            )
            profile = _write_with_profile(
                output_dir / "fundamentals.parquet",
                fundamentals,
                build_seconds=time.perf_counter() - started,
            )
            return {
                "rows": int(len(fundamentals)),
                "datasets": list(config.fundamental_datasets),
                "units": "source",
                "dataset_columns": dataset_columns,
            }, profile

        def build_events(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            events, meta = self._build_events_domain(
                config.events_datasets,
                decision_time,
                events_window_start,
                screen=screened,
                prior_events=prior_events,
            )
            meta = {**meta, "rows": int(len(events))}
            profile = _write_with_profile(
                output_dir / "events.parquet", events, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_macro(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            macro, meta = self._build_available_at_domain(
                config.macro_datasets, decision_time, macro_window_start, lifetime_registries=True
            )
            profile = _write_with_profile(
                output_dir / "macro.parquet", macro, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_text(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            text_index, meta = self._build_text(config, decision_time, text_window_start, output_dir)
            profile = _write_with_profile(
                output_dir / "text_index.parquet", text_index, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_universe(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            universe = self._build_universe(decision_time, config)
            universe = self._apply_screen(universe, screened)
            profile = _write_with_profile(
                output_dir / "universe.parquet", universe, build_seconds=time.perf_counter() - started
            )
            return {"rows": int(len(universe))}, profile

        # Ready tasks occupy worker slots in list order: text/macro before auction.
        tasks: list[DomainBuildTask] = [
            ("daily", (), build_daily),
            ("intraday", ("daily",), build_intraday),
            ("text", (), build_text),
            ("macro", (), build_macro),
            ("auction", (), build_auction),
            ("fundamentals", (), build_fundamentals),
            ("universe", (), build_universe),
        ]
        harvest = [
            ("daily", "daily", "daily.parquet"),
            ("auction", "auction", "auction.parquet"),
            ("intraday", "intraday_1min", "intraday_1min.parquet"),
            ("fundamentals", "fundamentals", "fundamentals.parquet"),
            ("macro", "macro", "macro.parquet"),
            ("text", "text", "text_index.parquet"),
            ("universe", "universe", "universe.parquet"),
        ]
        if config.events_datasets:
            tasks.append(("events", (), build_events))
            harvest.append(("events", "events", "events.parquet"))
        else:
            domains["events"] = {"rows": 0, "datasets": [], "skipped": True}
        task_results = _run_domain_tasks(tasks)
        for task_name, domain_name, file_name in harvest:
            domains[domain_name], profiles[file_name] = task_results[task_name]
        domains["universe_screen"] = {
            "active": screened is not None,
            "codes": len(screened) if screened is not None else None,
            "config": config.to_record()["universe_screen"],
        }

        manifest = {
            "snapshot_id": new_id("snap"),
            "kind": "decision_input",
            "created_at": utc_now_iso(),
            "decision_time": decision_time.isoformat(),
            "window_start": daily_window_start.isoformat(),
            "window_months": config.months_for("daily"),
            "window_config": config.to_record()["decision_windows"],
            "domain_windows": {
                "daily": {"window_start": daily_window_start.isoformat(), "window_months": config.months_for("daily")},
                "fundamentals": {"window_start": fundamentals_window_start.isoformat(), "window_months": config.months_for("fundamentals")},
                "events": {"window_start": events_window_start.isoformat(), "window_months": config.months_for("events")},
                "macro": {"window_start": macro_window_start.isoformat(), "window_months": config.months_for("macro")},
                "text": {"window_start": text_window_start.isoformat(), "window_months": config.months_for("text")},
                "intraday_1min": {"trade_days": config.intraday_trade_days},
            },
            "domains": domains,
            "data_quality_warnings": data_quality_warnings,
            "raw_generation": _raw_generation_identity(raw_generation),
            "build_profile": {
                "total_seconds": round(time.perf_counter() - total_started, 3),
                "domains": _profile_timings(profiles),
            },
            "data_profile": {"files": profiles},
        }
        # Every column in every written file must classify in the unit
        # registry before the snapshot becomes consumable.
        validate_snapshot_units(output_dir, manifest)
        _write_manifest(output_dir, manifest)
        return manifest

    # Enabled-domain data-quality gates over the audit status files. Execution-
    # critical domains (daily bars, intraday minutes; fundamentals has its own
    # stricter gate) hard-fail on a missing/unreadable/error status — bad
    # execution data invalidates every fill. Research domains (events,
    # board-trading, macro, text) degrade to a manifest warning: their audits flag source-level
    # sparsity and calibration artifacts that should not block an experiment.
    # A report whose own status is "warning" is recorded (manifest + build log)
    # for every domain, so the semantic risks it names stay visible downstream.
    _DOMAIN_STATUS_FILES: tuple[tuple[str, str, bool], ...] = (
        ("daily", DOMAIN_STATUS_FILES["daily"], True),
        ("intraday_1min", DOMAIN_STATUS_FILES["intraday_1min"], True),
        # Raw financial data is input-critical only when fundamentals are
        # enabled: its completeness is enforced nowhere else (the PIT index
        # gate audits the derived tree), and before the core/finance split a
        # raw-finance error hard-failed every experiment via the daily gate.
        ("fundamentals_raw", DOMAIN_STATUS_FILES["fundamentals_raw"], True),
        ("events", DOMAIN_STATUS_FILES["events"], False),
        ("board_trading", DOMAIN_STATUS_FILES["board_trading"], False),
        ("macro", DOMAIN_STATUS_FILES["macro"], False),
        ("text", DOMAIN_STATUS_FILES["text"], False),
    )

    def _domain_status_gates(self, config: SnapshotConfig) -> dict[str, str]:
        """Check each enabled domain's audit status; return research-domain warnings."""
        if self.data_quality_dir is None:
            return {}
        enabled = {
            "daily": True,
            "intraday_1min": bool(config.include_intraday),
            "fundamentals_raw": bool(config.fundamental_datasets),
            "events": bool(config.events_datasets),
            "board_trading": bool(_BOARD_TRADING_DATASETS.intersection(config.events_datasets)),
            "macro": bool(config.macro_datasets),
            "text": bool(config.text_datasets),
        }
        # Reports written before the current raw generation prove the PREVIOUS
        # lake. Deliberately a warning, not a hard gate: audits re-run on a
        # nightly cadence after each mutating job, so hard-gating would lock
        # experiments out for hours every night; the shared flock and the
        # generation-keyed snapshot cache carry the hard guarantees.
        generation = read_committed_raw_generation(self.raw_dir) or {}
        generation_at = str(generation.get("completed_at", ""))
        warnings: dict[str, str] = {}
        for domain, filename, critical in self._DOMAIN_STATUS_FILES:
            if not enabled[domain]:
                continue
            path = self.data_quality_dir / filename
            problem = ""
            stale = ""
            warning_status = ""
            if not path.exists():
                problem = "status file missing"
            else:
                try:
                    payload = read_quality_report(
                        path, expected_report_type=DOMAIN_REPORT_TYPES[domain]
                    )
                except (OSError, TypeError, ValueError) as exc:
                    problem = f"status report invalid: {exc}"
                else:
                    status = str(payload.get("status", "")).lower()
                    if status == "error":
                        problem = "audit status is error"
                    elif status == "warning":
                        # data docs §3.1: warning reports must be explicitly
                        # handled downstream — record them so they leave a
                        # trace in the manifest instead of vanishing.
                        counts = payload.get("finding_counts") or {}
                        warning_status = (
                            f"audit status is warning ({filename}): "
                            f"{int(counts.get('warning', 0))} warning findings"
                        )
                    created_at = str(payload.get("created_at", ""))
                    if generation_at and created_at and created_at < generation_at:
                        stale = f"audit status predates current raw generation ({filename})"
            if problem:
                if critical:
                    raise ValueError(
                        f"data-quality gate failed for execution-critical domain {domain!r}: {problem} ({path})"
                    )
                warnings[domain] = f"{problem} ({filename})"
            elif stale:
                warnings[domain] = stale
            elif warning_status:
                warnings[domain] = warning_status
        return warnings

    def _assert_fundamental_event_status_ok(
        self, window_start: datetime, datasets: tuple[str, ...]
    ) -> None:
        if self.fundamental_events_status is None:
            raise ValueError("PIT fundamental events status is required when fundamental datasets are enabled")
        if not self.fundamental_events_status.exists():
            raise FileNotFoundError(f"missing PIT fundamental events status: {self.fundamental_events_status}")
        report = read_quality_report(
            self.fundamental_events_status,
            expected_report_type=DOMAIN_REPORT_TYPES["fundamentals"],
        )
        errors = int(report["finding_counts"]["error"])
        status = str(report.get("status", "")).lower()
        if status == "error" or errors > 0:
            raise ValueError(
                f"PIT fundamental events audit is not usable: "
                f"status={report.get('status')!r} errors={errors} path={self.fundamental_events_status}"
            )
        # The status only proves what it audited: the audit scope must cover
        # every dataset and every partition month this build is about to load,
        # or unaudited history would pass the gate on the strength of a
        # narrower report.
        scope = report.get("scope") or {}
        audited_datasets = {str(name) for name in (scope.get("datasets") or ())}
        unaudited = sorted(set(datasets) - audited_datasets)
        if unaudited:
            raise ValueError(
                f"PIT fundamental events audit does not cover datasets {unaudited} "
                f"(path={self.fundamental_events_status})"
            )
        first_loadable_month = min(
            (
                path.stem.split("=", 1)[1]
                for dataset in datasets
                for path in (self.fundamental_events_root / dataset).glob("available_month=*.parquet")
            ),
            default="",
        )
        first_needed_month = max(
            pd.Timestamp(window_start).strftime("%Y%m"), first_loadable_month
        )
        audited_start = str(scope.get("start_date", "")).replace("-", "")
        if not first_loadable_month:
            return  # nothing to load; require_partitions fails the build explicitly
        if len(audited_start) < 6 or not audited_start[:6].isdigit() or audited_start[:6] > first_needed_month:
            raise ValueError(
                f"PIT fundamental events audit window starts at {audited_start or '<missing>'} "
                f"but this build loads partitions from {first_needed_month}; "
                f"the audited range must cover everything the snapshot can load "
                f"(path={self.fundamental_events_status})"
            )

    # ---- replay slot (valid/test region; not PIT-filtered) ----

    def build_replay_slot(
        self,
        start_date: str,
        end_date: str,
        output_dir: str | Path,
        *,
        label: str,
        config: SnapshotConfig | None = None,
        available_from: datetime | None = None,
    ) -> dict[str, object]:
        """Replay region data: daily bars plus the events/text/minutes/macro/
        fundamentals published inside the period, every domain carrying row-level
        ``available_at`` so the per-inference PIT view can expose each dataset at its
        refresh node. Read only by backtest_tool; never PIT-filtered up front.

        ``available_from`` is the fold's decision anchor (last trading day before
        the period, 23:59:59). Rows published between that anchor and calendar
        midnight of the period start — weekend/holiday news, events, macro —
        belong to the replay's first pre-open refresh, so the availability floor
        uses the anchor, not period-start midnight."""
        with self._raw_lake_guard() as raw_generation:
            manifest = self._build_replay_slot_impl(
                start_date, end_date, output_dir, label, config, raw_generation, available_from
            )
        return manifest

    def _build_replay_slot_impl(
        self,
        start_date: str,
        end_date: str,
        output_dir: str | Path,
        label: str,
        config: SnapshotConfig | None,
        raw_generation: dict[str, object] | None,
        available_from: datetime | None = None,
    ) -> dict[str, object]:
        config = config or SnapshotConfig()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start_key, end_key = yyyymmdd(start_date), yyyymmdd(end_date)
        period_start = pd.Timestamp(start_key, tz=CN_TZ)
        period_end = pd.Timestamp(end_key, tz=CN_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        anchor = pd.Timestamp(available_from) if available_from is not None else None
        if anchor is not None and anchor.tzinfo is None:
            anchor = anchor.tz_localize(CN_TZ)
        # Availability floor for the published-inside-the-period domains.
        window_floor = anchor if anchor is not None and anchor < period_start else period_start
        domains: dict[str, dict[str, object]] = {}
        profiles: dict[str, dict[str, object]] = {}
        total_started = time.perf_counter()

        # Same screened set the agent's decision snapshot used: anchored strictly
        # BEFORE the period, frozen across it (no intra-period re-screening).
        screen_anchor = (
            anchor.to_pydatetime() if anchor is not None
            else period_start.to_pydatetime() - timedelta(seconds=1)
        )
        screened = self._screened_codes(screen_anchor, config) if config.screening_active() else None

        def build_daily(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            daily = self._daily_join(start_key, end_key)
            daily = self._apply_screen(daily, screened)
            daily, conversions = normalize_daily_units(daily)
            daily = _stamp_daily_available_at(daily, self.contracts["daily"])
            profile = _write_with_profile(
                output_dir / "daily.parquet", daily, build_seconds=time.perf_counter() - started
            )
            return {"rows": int(len(daily)), "unit_conversions": conversions}, profile

        def build_macro(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            macro, meta = self._build_available_at_domain(config.macro_datasets, period_end, window_floor)
            profile = _write_with_profile(
                output_dir / "macro.parquet", macro, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_fundamentals(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            # Not the formal PIT decision boundary: take fundamentals published
            # inside the period without requiring partitions or the audit status,
            # so a slot still builds where a fundamental window happens to be empty.
            fundamentals = read_fundamental_events(
                self.fundamental_events_root,
                period_end.isoformat(),
                datasets=config.fundamental_datasets,
                min_available_at=window_floor.isoformat(),
                require_partitions=False,
            )
            fundamentals = self._apply_screen(fundamentals, screened)
            fundamentals, dataset_columns = _apply_fundamental_exclusions(
                fundamentals,
                _fundamental_dataset_columns(self.raw_dir, tuple(config.fundamental_datasets)),
            )
            profile = _write_with_profile(
                output_dir / "fundamentals.parquet", fundamentals, build_seconds=time.perf_counter() - started
            )
            return {
                "rows": int(len(fundamentals)),
                "datasets": list(config.fundamental_datasets),
                "dataset_columns": dataset_columns,
            }, profile

        def build_events(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            events, meta = self._build_available_at_domain(
                config.events_datasets, period_end, window_floor, screen=screened
            )
            meta = {**meta, "rows": int(len(events))}
            profile = _write_with_profile(
                output_dir / "events.parquet", events, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_text(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            text_index, meta = self._build_text(config, period_end, window_floor, output_dir)
            profile = _write_with_profile(
                output_dir / "text_index.parquet", text_index, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_minutes(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            meta, profile = self._write_minutes_range(
                start_key,
                end_key,
                output_dir / "intraday_1min.parquet",
                screened,
            )
            return meta, profile

        def build_actions(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            actions, meta = self._build_corporate_actions(start_key, end_key, period_end)
            profile = _write_with_profile(
                output_dir / "corporate_actions.parquet", actions, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        def build_auction(_: Mapping[str, DomainBuildResult]) -> DomainBuildResult:
            started = time.perf_counter()
            auction, meta = self._build_auction(start_key, end_key)
            auction = self._apply_screen(auction, screened)
            meta = {**meta, "rows": int(len(auction))}
            profile = _write_with_profile(
                output_dir / "auction.parquet", auction, build_seconds=time.perf_counter() - started
            )
            return meta, profile

        tasks: list[DomainBuildTask] = []
        if config.replay_include_minutes:
            tasks.append(("minutes", (), build_minutes))
        if config.replay_include_events:
            tasks.append(("events", (), build_events))
        if config.replay_include_text:
            tasks.append(("text", (), build_text))
        tasks.append(("daily", (), build_daily))
        if config.replay_include_fundamentals and config.fundamental_datasets:
            tasks.append(("fundamentals", (), build_fundamentals))
        if config.replay_include_macro:
            tasks.append(("macro", (), build_macro))
        tasks.extend(
            [
                ("actions", (), build_actions),
                ("auction", (), build_auction),
            ]
        )
        task_results = _run_domain_tasks(tasks)
        for task_name, domain_name, file_name in (
            ("daily", "daily", "daily.parquet"),
            ("macro", "macro", "macro.parquet"),
            ("fundamentals", "fundamentals", "fundamentals.parquet"),
            ("events", "events", "events.parquet"),
            ("text", "text", "text_index.parquet"),
            ("minutes", "intraday_1min", "intraday_1min.parquet"),
            ("actions", "corporate_actions", "corporate_actions.parquet"),
            ("auction", "auction", "auction.parquet"),
        ):
            if task_name in task_results:
                domains[domain_name], profiles[file_name] = task_results[task_name]
        domains["universe_screen"] = {
            "active": screened is not None,
            "codes": len(screened) if screened is not None else None,
            "config": config.to_record()["universe_screen"],
        }

        manifest = {
            "snapshot_id": new_id("replay"),
            "kind": "replay_slot",
            "label": label,
            "created_at": utc_now_iso(),
            "period_start": start_key,
            "period_end": end_key,
            "available_from": anchor.isoformat() if anchor is not None else None,
            "domains": domains,
            "raw_generation": _raw_generation_identity(raw_generation),
            "build_profile": {
                "total_seconds": round(time.perf_counter() - total_started, 3),
                "domains": _profile_timings(profiles),
            },
            "data_profile": {"files": profiles},
        }
        # Every column in every written file must classify in the unit
        # registry before the snapshot becomes consumable.
        validate_snapshot_units(output_dir, manifest)
        _write_manifest(output_dir, manifest)
        return manifest

    _AUCTION_COLUMNS = (
        "ts_code", "trade_date", "session", "price", "vol", "amount", "pre_close",
        "turnover_rate", "volume_ratio", "float_share", "available_at", "available_at_rule",
    )
    _AUCTION_STRING_COLUMNS = {
        "ts_code", "trade_date", "session", "available_at", "available_at_rule",
    }

    def _build_auction(self, start_key: str, end_key: str) -> tuple[pd.DataFrame, dict[str, object]]:
        """Exact opening call-auction results available from 2025-01-16."""
        frame = self.store.read_trade_range("stk_auction", start_key, end_key)
        price_quality = {
            "source_price_rows": 0,
            "derived_price_rows": 0,
            "no_trade_rows": 0,
            "unobserved_rows_dropped": 0,
        }
        if frame.empty:
            # Object-typed empty columns serialize as Arrow ``null``. A later
            # string trade-date predicate then fails before the backtest can
            # observe that the file has no rows. Keep the empty payload, but
            # preserve the same physical string/double schema as populated slots.
            auction = pd.DataFrame(
                {
                    column: pd.Series(
                        dtype="string" if column in self._AUCTION_STRING_COLUMNS else "float64"
                    )
                    for column in self._AUCTION_COLUMNS
                }
            )
        else:
            auction = frame.copy()
            price = pd.to_numeric(auction["price"], errors="coerce")
            volume = pd.to_numeric(auction["vol"], errors="coerce")
            amount = pd.to_numeric(auction["amount"], errors="coerce")
            # Rows with price, vol AND amount all missing carry no auction
            # observation: the source lists suspended codes this way, and around
            # the 2025-08 BSE renumbering it published three whole days of BJ
            # rows (retired 43/83/87xxxx aliases AND trading 920xxx codes) as
            # all-NaN. Such rows are equivalent to a missing row (the Broker
            # already falls back to the labelled 09:30 proxy), so they are
            # dropped and counted; genuinely inconsistent combinations below
            # still fail the build.
            unobserved = price.isna() & volume.isna() & amount.isna()
            if unobserved.any():
                price_quality["unobserved_rows_dropped"] = int(unobserved.sum())
                keep = ~unobserved
                auction = auction.loc[keep]
                price = price.loc[keep]
                volume = volume.loc[keep]
                amount = amount.loc[keep]
            valid_quantities = (
                pd.Series(np.isfinite(volume), index=auction.index)
                & pd.Series(np.isfinite(amount), index=auction.index)
                & volume.ge(0)
                & amount.ge(0)
            )
            traded = valid_quantities & volume.gt(0) & amount.gt(0)
            no_trade = valid_quantities & volume.eq(0) & amount.eq(0)
            inconsistent = ~(traded | no_trade)
            if inconsistent.any():
                sample = auction.loc[inconsistent, ["trade_date", "ts_code", "price", "vol", "amount"]]
                raise ValueError(f"stk_auction has invalid quantity combinations: {sample.head(5).to_dict('records')}")
            finite_price = pd.Series(np.isfinite(price), index=auction.index)
            valid_price = traded & finite_price & price.gt(0)
            derived_price = traded & ~valid_price
            recovered = amount.div(volume)
            recoverable = (
                pd.Series(np.isfinite(recovered), index=auction.index) & recovered.gt(0)
            )
            unrecoverable = traded & ~recoverable
            if unrecoverable.any():
                sample = auction.loc[unrecoverable, ["trade_date", "ts_code", "price", "vol", "amount"]]
                raise ValueError(f"stk_auction has unrecoverable clearing prices: {sample.head(5).to_dict('records')}")
            price_mismatch = (
                valid_price
                & recoverable
                & ~pd.Series(
                    np.isclose(
                        price,
                        recovered,
                        rtol=1e-9,
                        atol=STK_AUCTION_PRICE_ABS_TOLERANCE,
                    ),
                    index=auction.index,
                )
            )
            if price_mismatch.any():
                sample = auction.loc[price_mismatch, ["trade_date", "ts_code", "price", "vol", "amount"]]
                raise ValueError(f"stk_auction has inconsistent clearing prices: {sample.head(5).to_dict('records')}")
            auction.loc[derived_price, "price"] = recovered[derived_price]
            # A source price without matched quantity must never become a hidden
            # Broker fill. Preserve the row but normalize it to the no-trade sentinel.
            auction.loc[no_trade, "price"] = float("nan")
            price_quality.update(
                source_price_rows=int(valid_price.sum()),
                derived_price_rows=int(derived_price.sum()),
                no_trade_rows=int(no_trade.sum()),
            )
            auction["session"] = "open"
            availability = {
                trade_date: self._auction_partition_availability(trade_date)
                for trade_date in auction["trade_date"].astype(str).unique()
            }
            auction["available_at"] = [availability[str(day)][0] for day in auction["trade_date"]]
            auction["available_at_rule"] = [availability[str(day)][1] for day in auction["trade_date"]]
            auction = auction[list(self._AUCTION_COLUMNS)]
            auction = auction.sort_values(["trade_date", "session", "ts_code"]).reset_index(drop=True)
        auction, conversions = normalize_auction_units(auction)
        metadata: dict[str, object] = {
            "rows": int(len(auction)),
            "datasets": ["stk_auction"],
            "units": "unit_contract",
            "unit_conversions": conversions,
            "clearing_price_fields": {"open": "price", "close": "15:00 bar close"},
            "precoverage_fallback": "labelled 09:30 minute proxy; Shenzhen vol/amount use configured correction",
            "price_quality": price_quality,
        }
        if not auction.empty:
            visible_dates = sorted(auction["trade_date"].astype(str).unique())
            metadata["coverage_start"] = visible_dates[0]
            metadata["coverage_end"] = visible_dates[-1]
        return auction, metadata

    def _auction_partition_availability(self, trade_date: str) -> tuple[str, str]:
        # See _AUCTION_PUBLISH_CLOCK: one official-publication constant for
        # every partition, whatever the sidecar's landing evidence says.
        return (
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T{_AUCTION_PUBLISH_CLOCK}+08:00",
            "official:latest_publish_time",
        )

    _CORPORATE_ACTION_COLUMNS = (
        "ts_code", "ex_date", "record_date", "pay_date", "div_listdate",
        "cash_per_share", "stock_per_share",
    )

    def _build_corporate_actions(
        self, start_key: str, end_key: str, period_end: pd.Timestamp
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Implemented dividend events with an ex-date inside the replay window,
        one row per (ts_code, ex_date): SimBroker's ex-date corporate-action truth
        (docs/environment-design.md §3.2). Not an agent input — agent visibility of
        dividends stays announcement-gated via the PIT fundamental events.

        ``cash_per_share`` is the gross (税前) per-share cash amount and
        ``stock_per_share`` the combined 送转 ratio. Announcements are read without
        a lower available_at bound (an ex-date can trail its 实施公告 by weeks), a
        row announced only after its own ex-date is dropped as a revision artifact,
        and same-day events for one code are summed (they share the record-date
        share base)."""
        raw = read_fundamental_events(
            self.fundamental_events_root,
            period_end.isoformat(),
            datasets=("dividend",),
            require_partitions=False,
        )
        empty = pd.DataFrame(columns=list(self._CORPORATE_ACTION_COLUMNS))
        dropped = {"missing_ex_date": 0, "announced_after_ex_date": 0}
        meta: dict[str, object] = {"rows": 0, "datasets": ["dividend"], "dropped": dropped}
        if raw.empty:
            return empty, meta
        required = {
            "ts_code", "end_date", "div_proc", "ex_date", "available_at",
            "cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate",
            "record_date", "pay_date", "div_listdate",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"dividend events missing columns: {missing}")
        frame = raw[raw["div_proc"].astype(str) == "实施"].copy()
        for column in ("ex_date", "record_date", "pay_date", "div_listdate"):
            frame[column] = frame[column].astype("string").str.strip().fillna("")
        dropped["missing_ex_date"] = int((frame["ex_date"] == "").sum())
        frame = frame[(frame["ex_date"] >= start_key) & (frame["ex_date"] <= end_key)]
        if frame.empty:
            return empty, meta
        announced = frame["available_at"].astype(str).str[:10].str.replace("-", "", regex=False)
        late = announced > frame["ex_date"]
        dropped["announced_after_ex_date"] = int(late.sum())
        frame = frame[~late]
        # Latest announced version per dividend event, then per-share amounts with
        # the documented fallbacks (cash_div_tax is gross; stk_div is the combined
        # 送股+转增 ratio; audit.py records the unit semantics).
        frame = frame.sort_values("available_at").drop_duplicates(["ts_code", "end_date", "ex_date"], keep="last")
        cash = pd.to_numeric(frame["cash_div_tax"], errors="coerce")
        cash = cash.fillna(pd.to_numeric(frame["cash_div"], errors="coerce")).fillna(0.0)
        bo = pd.to_numeric(frame["stk_bo_rate"], errors="coerce").fillna(0.0)
        co = pd.to_numeric(frame["stk_co_rate"], errors="coerce").fillna(0.0)
        stock = pd.to_numeric(frame["stk_div"], errors="coerce").fillna(bo + co)
        frame = frame.assign(cash_per_share=cash.clip(lower=0.0), stock_per_share=stock.clip(lower=0.0))
        frame = frame[(frame["cash_per_share"] > 0.0) | (frame["stock_per_share"] > 0.0)]
        if frame.empty:
            return empty, meta
        out = (
            frame.groupby(["ts_code", "ex_date"], as_index=False)
            .agg(
                cash_per_share=("cash_per_share", "sum"),
                stock_per_share=("stock_per_share", "sum"),
                record_date=("record_date", "first"),
                pay_date=("pay_date", "first"),
                div_listdate=("div_listdate", "max"),
            )
            .sort_values(["ex_date", "ts_code"], ignore_index=True)
        )
        meta["rows"] = int(len(out))
        return out[list(self._CORPORATE_ACTION_COLUMNS)], meta

    def _minute_partition_paths(self, start_key: str, end_key: str) -> list[Path]:
        dataset_dir = self.raw_dir / "stk_mins_1min_by_date"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"missing intraday by-date dataset: {dataset_dir}")
        return [
            path
            for path in sorted(dataset_dir.glob("trade_date=*.parquet"))
            if start_key <= path.stem.split("=", 1)[1] <= end_key
        ]

    def _write_minutes_range(
        self,
        start_key: str,
        end_key: str,
        output_path: Path,
        screened: frozenset[str] | None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Prefetch daily transforms and append them in source order.

        The prior replay path retained every daily frame and a second full
        concatenated frame before writing.  Daily Parquet row groups preserve
        the same single-file consumer contract.  A bounded pool overlaps source
        reads, screening and auction correction; one writer still consumes the
        transformed frames in sorted partition order, bounding working memory
        to the prefetch window plus Arrow's write buffers.
        """
        paths = self._minute_partition_paths(start_key, end_key)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        schema: pa.Schema | None = None
        empty_template: pd.DataFrame | None = None
        rows = 0
        columns: list[str] = []
        build_seconds = 0.0
        write_seconds = 0.0

        def prepare(source_path: Path) -> tuple[Path, pd.DataFrame, float]:
            started = time.perf_counter()
            frame = pd.read_parquet(source_path)
            if not frame.empty:
                frame = apply_open_auction_correction(frame)
            frame = self._apply_screen(frame, screened)
            return source_path, frame, time.perf_counter() - started

        try:
            if paths:
                workers = min(_MINUTE_PARTITION_WORKERS, len(paths))
                executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="snapshot-minute")
                running: dict[Future[tuple[Path, pd.DataFrame, float]], int] = {}
                ready: dict[int, tuple[Path, pd.DataFrame, float]] = {}
                next_submit = 0
                next_write = 0

                def submit_window() -> None:
                    nonlocal next_submit
                    while next_submit < len(paths) and next_submit - next_write < workers:
                        running[executor.submit(prepare, paths[next_submit])] = next_submit
                        next_submit += 1

                try:
                    submit_window()
                    while next_write < len(paths):
                        completed, _ = wait(running, return_when=FIRST_COMPLETED)
                        for future in completed:
                            index = running.pop(future)
                            # Resolve every completed future before writing more
                            # row groups: a failed prefetched partition aborts the
                            # staging file as soon as the pool observes it.
                            ready[index] = future.result()

                        while next_write in ready:
                            source_path, frame, elapsed = ready.pop(next_write)
                            build_seconds += elapsed
                            if empty_template is None or len(frame.columns) > len(empty_template.columns):
                                empty_template = frame.iloc[0:0]
                            if not frame.empty:
                                started = time.perf_counter()
                                table = pa.Table.from_pandas(frame, preserve_index=False)
                                if writer is None:
                                    schema = table.schema
                                    columns = list(frame.columns)
                                    writer = pq.ParquetWriter(tmp, schema, compression="snappy")
                                elif not table.schema.equals(schema, check_metadata=False):
                                    try:
                                        table = table.cast(schema, safe=True)
                                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError) as exc:
                                        raise ValueError(
                                            f"minute partition schema drift at {source_path}: "
                                            f"expected={schema} actual={table.schema}"
                                        ) from exc
                                else:
                                    table = table.replace_schema_metadata(schema.metadata)
                                writer.write_table(table)
                                write_seconds += time.perf_counter() - started
                                rows += len(frame)
                            next_write += 1
                            submit_window()
                finally:
                    for future in running:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)

            if writer is None:
                empty = empty_template if empty_template is not None else pd.DataFrame()
                profile = _write_with_profile(output_path, empty, build_seconds=build_seconds)
                return (
                    {"rows": 0, "datasets": ["stk_mins_1min_by_date"], "files": len(paths)},
                    profile,
                )

            started = time.perf_counter()
            writer.close()
            writer = None
            tmp.replace(output_path)
            write_seconds += time.perf_counter() - started
            profile = _parquet_file_profile(
                output_path,
                rows=rows,
                columns=columns,
                build_seconds=build_seconds,
                write_seconds=write_seconds,
            )
            return (
                {"rows": rows, "datasets": ["stk_mins_1min_by_date"], "files": len(paths)},
                profile,
            )
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001 - preserve the original build error
                    pass
            tmp.unlink(missing_ok=True)

    # ---- domain builders ----

    def _build_daily(self, decision_time: datetime, window_start: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
        daily_datasets = ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d")
        visible_by_dataset = {
            dataset: self._visible_trade_dates(dataset, decision_time, window_start) for dataset in daily_datasets
        }
        visible_dates = visible_by_dataset["daily"]
        if not visible_dates:
            raise ValueError(f"no visible daily trade dates before {decision_time.isoformat()}")
        frame = self._daily_join(visible_dates[0], visible_dates[-1], visible_dates_by_dataset=visible_by_dataset)
        frame, conversions = normalize_daily_units(frame)
        meta = {
            "rows": int(len(frame)),
            "datasets": list(daily_datasets),
            "coverage_start": visible_dates[0],
            "coverage_end": visible_dates[-1],
            "trade_dates": visible_dates,
            "visible_trade_dates_by_dataset": visible_by_dataset,
            "units": "unit_contract",
            "unit_conversions": conversions,
            "availability_rule": "per-dataset daily contracts; joins include only partitions visible at the decision time",
        }
        return frame, meta

    def _visible_trade_dates(self, dataset: str, decision_time: datetime, window_start: pd.Timestamp) -> list[str]:
        contract = self.contracts[dataset]
        return [
            key
            for key in self.store.trade_dates(dataset)
            if contract.available_at(datetime.strptime(key, "%Y%m%d").date()) <= decision_time
            and pd.Timestamp(key, tz=CN_TZ) >= window_start
        ]

    def _daily_join(
        self,
        start: str,
        end: str,
        *,
        visible_dates_by_dataset: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        daily = self.store.read_trade_range("daily", start, end)
        if daily.empty:
            raise ValueError(f"daily raw data empty for {start}..{end}")
        basic = self.store.read_trade_range("daily_basic", start, end)
        limits = self.store.read_trade_range("stk_limit", start, end)
        adj = self.store.read_trade_range("adj_factor", start, end)
        suspend = self.store.read_trade_range("suspend_d", start, end, columns=["trade_date", "ts_code"])
        if visible_dates_by_dataset is not None:
            daily = _filter_trade_dates(daily, visible_dates_by_dataset.get("daily", []))
            basic = _filter_trade_dates(basic, visible_dates_by_dataset.get("daily_basic", []))
            limits = _filter_trade_dates(limits, visible_dates_by_dataset.get("stk_limit", []))
            adj = _filter_trade_dates(adj, visible_dates_by_dataset.get("adj_factor", []))
            suspend = _filter_trade_dates(suspend, visible_dates_by_dataset.get("suspend_d", []))
            if daily.empty:
                raise ValueError(f"daily raw data empty after PIT filter for {start}..{end}")
        for name, frame in (("daily", daily), ("daily_basic", basic), ("stk_limit", limits), ("adj_factor", adj)):
            if frame.duplicated(["trade_date", "ts_code"]).any():
                raise ValueError(f"{name} has duplicate (trade_date, ts_code) keys in {start}..{end}")
        # The join keys are (trade_date, ts_code), so daily_basic.close and
        # stk_limit.pre_close only ever survived as the suffixed duplicates
        # close_basic / pre_close_limit. Measured on the 2025-12-31 snapshot
        # (1,291,986 rows): close_basic equalled close in every row and
        # pre_close_limit equalled pre_close wherever stk_limit had a row, so
        # they were a second name for one number in the Agent's schema.
        basic = basic.drop(columns=["close"], errors="ignore")
        limits = limits.drop(columns=["pre_close"], errors="ignore")
        out = daily.merge(basic, on=["trade_date", "ts_code"], how="left", suffixes=("", "_basic"))
        out = out.merge(limits, on=["trade_date", "ts_code"], how="left", suffixes=("", "_limit"))
        if not adj.empty:
            out = out.merge(adj[["trade_date", "ts_code", "adj_factor"]], on=["trade_date", "ts_code"], how="left")
        suspended = set(zip(suspend.get("trade_date", []), suspend.get("ts_code", [])))
        out["is_suspended"] = [(d, c) in suspended for d, c in zip(out["trade_date"], out["ts_code"])]
        out["trade_date"] = out["trade_date"].astype(str)
        out["ts_code"] = out["ts_code"].astype(str)
        return out

    def _build_intraday(
        self, decision_time: datetime, visible_daily_dates: list[str], config: SnapshotConfig
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        dataset_dir = self.raw_dir / "stk_mins_1min_by_date"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"missing intraday by-date dataset: {dataset_dir}")
        recent = visible_daily_dates[-config.intraday_trade_days :]
        frames = []
        for key in recent:
            path = dataset_dir / f"trade_date={key}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing intraday partition: {path}")
            frames.append(pd.read_parquet(path))
        minute = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not minute.empty:
            available = to_cn_timestamps(minute["available_at"])
            minute = minute[available <= decision_time].reset_index(drop=True)
            minute = apply_open_auction_correction(minute)
            # For minute bars available_at == the bar close (trade_time), so it is an
            # internal gating column, not agent information. Drop it (as daily does) to
            # keep the agent-facing intraday schema clean; the replay slot keeps its own
            # available_at as the Timeview gate.
            minute = minute.drop(columns=["available_at", "available_at_rule"], errors="ignore")
        correction = AuctionCorrectionConfig()
        meta = {
            "rows": int(len(minute)),
            "datasets": ["stk_mins_1min_by_date"],
            "trade_dates": recent,
            "availability_rule": "available_at=bar close time (trade_time)",
            "auction_correction": {
                "rule_id": AUCTION_CORRECTION_RULE_ID,
                "factors": {
                    "00*.SZ": correction.volume_factors["sz_main_00"],
                    "30*.SZ": correction.volume_factors["sz_gem_30"],
                    "other": 1.0,
                },
                "applies_to": "09:30 SZ bars as live stk_auction proxy columns only",
            },
        }
        return minute, meta

    def _build_events_domain(
        self,
        datasets: tuple[str, ...],
        decision_time: datetime,
        window_start: pd.Timestamp,
        *,
        screen: frozenset[str] | None,
        prior_events: tuple[Path, datetime] | None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if prior_events is not None:
            prior_path, prior_time = prior_events
            prior_time = (
                prior_time
                if prior_time.tzinfo
                else prior_time.replace(tzinfo=CN_TZ)
            ).astimezone(CN_TZ)
            if prior_path.is_file() and prior_time < decision_time:
                try:
                    prior = pd.read_parquet(prior_path)
                except (OSError, ValueError):
                    prior = None
                if prior is not None and "available_at" in prior.columns:
                    return self._extend_prior_events(
                        prior,
                        prior_time,
                        datasets,
                        decision_time,
                        window_start,
                        screen=screen,
                    )
        return self._build_available_at_domain(
            datasets,
            decision_time,
            window_start,
            forward_events=True,
            screen=screen,
        )

    def _extend_prior_events(
        self,
        prior: pd.DataFrame,
        prior_time: datetime,
        datasets: tuple[str, ...],
        decision_time: datetime,
        window_start: pd.Timestamp,
        *,
        screen: frozenset[str] | None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Reuse an earlier decision events table and union only the new as-of slice.

        Kept prior rows use the same availability / forward-event window as a
        full rebuild. Newly screened-in names do not receive events from before
        ``prior_time``; the manifest records that limitation.
        """

        prior_at = to_cn_timestamps(prior["available_at"])
        keep = prior_at >= window_start
        start_day = window_start.strftime("%Y%m%d")
        for dataset in datasets:
            event_column = FORWARD_EVENT_DATE_COLUMNS.get(dataset)
            if event_column is None:
                continue
            if event_column not in prior.columns:
                raise ValueError(
                    f"prior events file has no {event_column} column; cannot window forward events"
                )
            event_in_window = prior[event_column].fillna("").astype(str) >= start_day
            if "dataset" in prior.columns:
                event_in_window = event_in_window & (prior["dataset"].astype(str) == dataset)
            keep = keep | event_in_window
        kept = prior[keep].copy()
        delta, meta = self._build_available_at_domain(
            datasets,
            decision_time,
            pd.Timestamp(prior_time),
            forward_events=True,
            screen=screen,
        )
        if not delta.empty and "available_at" in delta.columns:
            delta_at = to_cn_timestamps(delta["available_at"])
            delta = delta[delta_at > prior_time].reset_index(drop=True)
        # Do not drop all-NA columns: they are schema contributed by empty
        # datasets in this slice. Stripping them made dataset_columns declare
        # fields that the written parquet no longer had.
        frames = [frame for frame in (kept, delta) if not frame.empty]
        merged = concat_rows(frames) if frames else pd.DataFrame()
        if not merged.empty:
            merged = merged.drop_duplicates(ignore_index=True)
            merged = self._apply_screen(merged, screen)
        physical = set(merged.columns)
        dataset_columns = meta.get("dataset_columns")
        if isinstance(dataset_columns, dict):
            meta = {
                **meta,
                "dataset_columns": {
                    dataset: [column for column in columns if column in physical]
                    for dataset, columns in dataset_columns.items()
                    if isinstance(columns, list)
                },
            }
        return merged, {
            **meta,
            "incremental": True,
            "prior_decision_time": prior_time.isoformat(),
            "prior_rows_kept": int(len(kept)),
            "delta_rows": int(len(delta)),
            "new_universe_names_lack_pre_prior_events": True,
        }

    def _build_available_at_domain(
        self,
        datasets: tuple[str, ...],
        decision_time: datetime,
        window_start: pd.Timestamp,
        *,
        lifetime_registries: bool = False,
        forward_events: bool = False,
        screen: frozenset[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        def load_one(dataset: str) -> dict[str, object]:
            dataset_started = time.perf_counter()
            dataset_dir = self.raw_dir / dataset
            if not dataset_dir.exists():
                raise FileNotFoundError(f"missing configured dataset directory: {dataset_dir}")
            # Registry datasets (contract/bond basics) stay valid for the
            # instrument's whole life and are tiny: the DECISION snapshot
            # exempts them from the window floor (PIT still enforced per row
            # via the list-date available_at). Replay slots must NOT: their
            # rows are unioned with the frozen snapshot by the Timeview, so a
            # second full-life copy would duplicate every registry row in the
            # agent's backtest view — the slot only needs registries newly
            # listed inside the replay window.
            exempt = lifetime_registries and dataset in MACRO_REGISTRY_DATASETS
            floor = _REGISTRY_WINDOW_FLOOR if exempt else window_start
            forward_column = FORWARD_EVENT_DATE_COLUMNS.get(dataset) if forward_events else None
            local_nat: dict[str, int] = {}
            rows, read_profile = self._read_dataset_window(
                dataset_dir, decision_time, floor, local_nat, forward_event_column=forward_column
            )
            excluded = SNAPSHOT_EXCLUDED_COLUMNS.get(dataset, ())
            if excluded:
                rows = rows.drop(columns=[column for column in excluded if column in rows.columns])
            had_visible_rows = not rows.empty
            # Overlapping partition files (the pre-canonical macro range
            # family) repeat identical rows; a duplicated series distorts
            # every frequency/aggregate a strategy computes on it.
            started = time.perf_counter()
            deduped = rows.drop_duplicates(ignore_index=True)
            deduplicate_seconds = time.perf_counter() - started
            duplicate_count = int(len(rows) - len(deduped))
            if duplicate_count:
                rows = deduped
            # Screen while the frame is still narrow (its own columns only). A
            # post-union screen materializes a rows × union-columns take that
            # goes super-linear once the wide 2025+ membership datasets enter
            # the window (a ~19M-row union spent hours in one frame[mask]).
            rows_before_screen = len(rows)
            started = time.perf_counter()
            rows = self._apply_screen(rows, screen)
            screen_seconds = time.perf_counter() - started
            schema = None
            frame = None
            columns: list[str]
            if had_visible_rows and len(rows):
                rows.insert(0, "dataset", dataset)
                frame = rows
                columns = list(rows.columns)
            else:
                schema = _dataset_footer_schema(dataset_dir)
                if schema is not None and excluded:
                    # Keep the zero-row schema contribution consistent with
                    # the rows path, or frozen and replay parts would
                    # disagree on the excluded columns.
                    schema = pa.schema([field for field in schema if field.name not in excluded])
                columns = ["dataset", *schema.names] if schema is not None else ["dataset"]
            return {
                "dataset": dataset,
                "frame": frame,
                "schema": schema,
                "columns": columns,
                "nat": local_nat.get(dataset, 0),
                "duplicate_count": duplicate_count,
                "profile": {
                    **read_profile,
                    "rows_output": int(len(rows)),
                    "duplicate_rows_dropped": duplicate_count,
                    "screen_rows_dropped": int(rows_before_screen - len(rows)),
                    "total_seconds": round(time.perf_counter() - dataset_started, 3),
                    "phases": {
                        **read_profile["phases"],
                        "deduplicate_seconds": round(deduplicate_seconds, 3),
                        "screen_seconds": round(screen_seconds, 3),
                    },
                },
            }

        if len(datasets) > 1:
            with ThreadPoolExecutor(
                max_workers=min(_DATASET_UNION_WORKERS, len(datasets)),
                thread_name_prefix="snapshot-dataset",
            ) as pool:
                loaded = list(pool.map(load_one, datasets))
        else:
            loaded = [load_one(dataset) for dataset in datasets]

        frames: list[pd.DataFrame] = []
        schema_only: dict[str, object] = {}
        rules: dict[str, str] = {}
        dataset_columns: dict[str, list[str]] = {}
        dataset_build_profile: dict[str, dict[str, object]] = {}
        duplicate_rows_dropped: dict[str, int] = {}
        nat_counts: dict[str, int] = {}
        for item in loaded:
            dataset = str(item["dataset"])
            rules[dataset] = "raw available_at column"
            dataset_columns[dataset] = list(item["columns"])
            dataset_build_profile[dataset] = item["profile"]  # type: ignore[assignment]
            if int(item["duplicate_count"]):
                duplicate_rows_dropped[dataset] = int(item["duplicate_count"])
            if int(item["nat"]):
                nat_counts[dataset] = int(item["nat"])
            if item["frame"] is not None:
                frames.append(item["frame"])  # type: ignore[arg-type]
            elif item["schema"] is not None:
                schema_only[dataset] = item["schema"]
        merged = concat_rows(frames) if frames else pd.DataFrame()
        merged = _pad_union_schema(merged, schema_only)
        # Manifest attribution must match the frame that will be written.
        # Empty-window datasets still contribute footer columns via padding;
        # if a footer field never lands in `merged`, do not declare it.
        physical = set(merged.columns)
        dataset_columns = {
            dataset: [column for column in columns if column in physical]
            for dataset, columns in dataset_columns.items()
        }
        # units="source": heterogeneous unions keep TuShare per-source units —
        # the daily-domain unit contract does NOT extend to same-named fields
        # here (env docs §1.4; raw unit table in data docs §1.2).
        meta = {
            "rows": int(len(merged)),
            "datasets": list(datasets),
            "units": "source",
            "availability_rules": rules,
            "dataset_columns": dataset_columns,
            "dataset_build_profile": dataset_build_profile,
        }
        if duplicate_rows_dropped:
            meta["duplicate_rows_dropped"] = duplicate_rows_dropped
        if nat_counts:
            meta["unparseable_available_at_dropped"] = nat_counts
        return merged, meta

    def _read_dataset_window(
        self,
        dataset_dir: Path,
        decision_time: datetime,
        window_start: pd.Timestamp,
        nat_counts: dict[str, int] | None = None,
        forward_event_column: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        started = time.perf_counter()
        start_day = window_start.strftime("%Y%m%d")
        end_day = decision_time.strftime("%Y%m%d")
        paths = [
            path
            for path in sorted(dataset_dir.rglob("*.parquet"))
            if _partition_overlaps(path.stem, start_day, end_day)
        ]
        discover_seconds = time.perf_counter() - started

        def load(path: Path) -> tuple[pd.DataFrame | None, int, int]:
            frame = pd.read_parquet(path)
            source_rows = len(frame)
            if frame.empty:
                return None, 0, source_rows
            if "available_at" not in frame.columns:
                raise ValueError(f"{path} has no available_at column; cannot enforce the PIT wall")
            available = to_cn_timestamps(frame["available_at"])
            # Unparseable timestamps become NaT and fail both bounds: dropped in
            # the conservative direction (hidden, never leaked), but counted so
            # an ingestion defect surfaces in the manifest instead of silently.
            nat = int(available.isna().sum())
            in_window = available >= window_start
            if forward_event_column is not None:
                if forward_event_column not in frame.columns:
                    raise ValueError(f"{path} has no {forward_event_column} column; cannot window forward events")
                # Long-announced future events stay visible: a row belongs to
                # the window if EITHER it was announced inside it or its event
                # date has not fallen out of it. The PIT wall is unchanged.
                in_window |= frame[forward_event_column].fillna("").astype(str) >= start_day
            keep = frame[(available <= decision_time) & in_window]
            return (keep if not keep.empty else None), nat, source_rows

        # Partition files number in the thousands for daily-partitioned event
        # datasets; parquet decode releases the GIL, so a bounded thread pool
        # cuts the wall while pool.map preserves the sorted partition order
        # (the concat below must stay byte-identical to a serial read).
        started = time.perf_counter()
        if len(paths) > 1:
            with ThreadPoolExecutor(
                max_workers=min(_PARTITION_READ_WORKERS, len(paths)), thread_name_prefix="snapshot-read"
            ) as pool:
                results = list(pool.map(load, paths))
        else:
            results = [load(path) for path in paths]
        read_filter_seconds = time.perf_counter() - started
        frames = [frame for frame, _, _ in results if frame is not None]
        nat_dropped = sum(nat for _, nat, _ in results)
        if nat_counts is not None and nat_dropped:
            nat_counts[dataset_dir.name] = nat_counts.get(dataset_dir.name, 0) + nat_dropped
        started = time.perf_counter()
        merged = concat_rows(frames) if frames else pd.DataFrame()
        concat_seconds = time.perf_counter() - started
        return merged, {
            "partition_files": len(paths),
            "source_rows": int(sum(source_rows for _, _, source_rows in results)),
            "rows_after_visibility": int(len(merged)),
            "phases": {
                "discover_seconds": round(discover_seconds, 3),
                "read_filter_seconds": round(read_filter_seconds, 3),
                "concat_seconds": round(concat_seconds, 3),
            },
        }

    def _build_text(
        self, config: SnapshotConfig, decision_time: datetime, window_start: pd.Timestamp, output_dir: Path
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Text index plus per-dataset body shards under text_library/.

        Bodies are stored as one parquet per dataset keyed by text_id (not one
        file per document) so multi-million-row text windows stay tractable.
        """
        library_dir = output_dir / "text_library"
        library_dir.mkdir(parents=True, exist_ok=True)
        index_frames: list[pd.DataFrame] = []
        for dataset in config.text_datasets:
            dataset_dir = self.raw_dir / dataset
            if not dataset_dir.exists():
                raise FileNotFoundError(f"missing configured text dataset: {dataset_dir}")
            if dataset == "news":
                news_start = window_start
                if config.news_window_months is not None:
                    news_start = max(
                        window_start, pd.Timestamp(decision_time) - pd.DateOffset(months=config.news_window_months)
                    )
                if config.news_sources:
                    source_dirs = [dataset_dir / f"src={source}" for source in config.news_sources]
                    for source_dir in source_dirs:
                        if not source_dir.exists():
                            raise FileNotFoundError(f"missing configured news source: {source_dir}")
                else:
                    source_dirs = sorted(p for p in dataset_dir.glob("src=*") if p.is_dir())
                source_frames = []
                for source_dir in source_dirs:
                    source_rows, _ = self._read_dataset_window(source_dir, decision_time, news_start)
                    if not source_rows.empty:
                        source_frames.append(source_rows.assign(src=source_dir.name.split("=", 1)[1]))
                rows = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
            else:
                rows, _ = self._read_dataset_window(dataset_dir, decision_time, window_start)
            if rows.empty:
                continue
            if dataset in {"irm_qa_sh", "irm_qa_sz"}:
                title_column = "q" if "q" in rows.columns else None
                body_columns = [c for c in ("q", "a") if c in rows.columns]
            else:
                title_column = next((c for c in ("title", "report_title", "name") if c in rows.columns), None)
                body_columns = [
                    c for c in ("title", "report_title", "abstr", "content", "content_html", "url")
                    if c in rows.columns
                ]
            if title_column is None or not body_columns:
                raise ValueError(f"text dataset {dataset} has no usable title/body columns: {list(rows.columns)}")
            titles = rows[title_column].fillna("").astype(str)
            if "content" in rows.columns:
                titles = titles.where(titles.str.len() > 0, rows["content"].fillna("").astype(str))
            bodies = rows[body_columns[0]].fillna("").astype(str)
            for key in body_columns[1:]:
                bodies = bodies + "\n" + rows[key].fillna("").astype(str)
            bodies = bodies.str.slice(0, config.text_body_chars)
            # Same-document duplicates collapse to the earliest copy BEFORE
            # text_id assignment: re-ingested rows otherwise get distinct
            # text_ids that occupy bounded retrieval slots and inflate evidence
            # counts (measured over a 21-month window: anns_d 38%, major_news
            # 9% duplicate rows). Only datasets with a measured duplication
            # mechanism and a safe identity are deduplicated.
            if dataset == "news":
                # Cross-source duplicate flashes; identity is the truncated
                # body content compared directly.
                identity = bodies
            elif dataset == "anns_d":
                # body = title + url (the url carries the filing id; measured
                # 0 blank urls); ts_code keeps joint announcements distinct.
                identity = rows["ts_code"].fillna("").astype(str) + "|" + bodies
            elif dataset == "major_news":
                # body = title + content; literal re-ingested pushes.
                identity = bodies
            else:
                identity = None
            if identity is not None:
                order = rows["available_at"].astype(str).sort_values(kind="stable").index
                keep = order[~identity.loc[order].duplicated().values]
                rows, titles, bodies = rows.loc[keep], titles.loc[keep], bodies.loc[keep]
            available = rows["available_at"].astype(str)
            text_ids = [
                f"{dataset}:{avail}:{position}"
                for position, avail in enumerate(available)
            ]
            library_file = f"{dataset}.parquet"
            _write(library_dir / library_file, pd.DataFrame({"text_id": text_ids, "body": bodies.values}))
            index_frames.append(
                pd.DataFrame(
                    {
                        "text_id": text_ids,
                        "dataset": dataset,
                        "ts_codes": rows.get("ts_code", pd.Series("", index=rows.index)).fillna("").astype(str).values,
                        "title": titles.str.slice(0, 200).values,
                        "available_at": available.values,
                        "library_file": library_file,
                    }
                )
            )
        index = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame(
            columns=["text_id", "dataset", "ts_codes", "title", "available_at", "library_file"]
        )
        meta = {"rows": int(len(index)), "datasets": list(config.text_datasets), "library_dir": "text_library"}
        return index, meta

    _BOARD_PREFIXES = {
        "main": ("600", "601", "603", "605", "000", "001", "002", "003"),
        "gem": ("300", "301", "302"),
        "star": ("688", "689"),
        "bj": (),  # matched by the .BJ suffix instead
    }

    def _screened_codes(self, decision_time: datetime, config: SnapshotConfig) -> frozenset[str] | None:
        """Research-universe screen, evaluated with decision-time knowledge only.

        Returns None when screening is off. ST status comes from the as-of name
        (namechange), listing age from stock_basic list_date, cap/price bands
        from the latest daily_basic row at or before the anchor day. Codes with
        a missing attribute fail closed for that filter (an unnamed or unpriced
        code cannot prove eligibility)."""
        if not config.screening_active():
            return None
        universe = self._build_universe(decision_time, replace(config, include_industry=False))
        day = decision_time.strftime("%Y%m%d")
        keep = universe[["ts_code"]].copy()
        keep["name"] = universe.get("name")
        keep["list_date"] = universe.get("list_date")
        if config.screen_exclude_st:
            names = keep["name"].fillna("").astype(str).str.upper()
            keep = keep[(names != "") & ~names.str.contains("ST")]
        if config.screen_exclude_new_listed_days > 0:
            cutoff = (decision_time - timedelta(days=config.screen_exclude_new_listed_days)).strftime("%Y%m%d")
            listed = keep["list_date"].fillna("").astype(str)
            keep = keep[(listed != "") & (listed <= cutoff)]
        if config.screen_boards:
            codes = keep["ts_code"].astype(str)
            allowed_boards = set(config.screen_boards)
            prefixes = tuple(p for board in allowed_boards for p in self._BOARD_PREFIXES[board])
            mask = codes.str.startswith(prefixes) if prefixes else pd.Series(False, index=codes.index)
            if "bj" in allowed_boards:
                mask = mask | codes.str.endswith(".BJ")
            keep = keep[mask]
        needs_basic = any(
            value is not None
            for value in (config.screen_min_circ_mv_yi, config.screen_max_circ_mv_yi,
                          config.screen_min_price, config.screen_max_price)
        )
        if needs_basic:
            contract = self.contracts["daily_basic"]
            basic_dates = [
                d
                for d in self.store.trade_dates("daily_basic")
                if d <= day and contract.available_at(datetime.strptime(d, "%Y%m%d").date()) <= decision_time
            ]
            if not basic_dates:
                raise FileNotFoundError(f"universe screening needs a daily_basic partition at or before {day}")
            basic = self.store.read_trade_date("daily_basic", basic_dates[-1], columns=["ts_code", "close", "circ_mv"])
            keep = keep.merge(basic, on="ts_code", how="left")
            circ_mv_yi = pd.to_numeric(keep["circ_mv"], errors="coerce") / 1e4  # 万元 -> 亿元
            close = pd.to_numeric(keep["close"], errors="coerce")
            if config.screen_min_circ_mv_yi is not None:
                keep = keep[circ_mv_yi.reindex(keep.index) >= config.screen_min_circ_mv_yi]
            if config.screen_max_circ_mv_yi is not None:
                keep = keep[circ_mv_yi.reindex(keep.index) <= config.screen_max_circ_mv_yi]
            if config.screen_min_price is not None:
                keep = keep[close.reindex(keep.index) >= config.screen_min_price]
            if config.screen_max_price is not None:
                keep = keep[close.reindex(keep.index) <= config.screen_max_price]
        screened = frozenset(keep["ts_code"].astype(str))
        if not screened:
            raise ValueError(
                "universe screening left ZERO eligible codes at the decision anchor - "
                "loosen the screen_* configuration (this would otherwise surface later "
                "as an empty replay region)"
            )
        return screened

    @staticmethod
    def _apply_screen(frame: pd.DataFrame, allowed: frozenset[str] | None) -> pd.DataFrame:
        """Restrict per-stock rows to the screened set.

        Only A-share-coded rows (``\\d{6}.SH/.SZ/.BJ``) are subject to the screen:
        concept/industry/index rows (``881xxx.TI``, ``BKxxxx.DC``, ``000242.KP``,
        null ts_code, ...) are market-level context that no stock screen can
        legitimately empty — screening them against a stock universe silently
        deleted whole sentiment datasets."""
        if allowed is None or frame.empty or "ts_code" not in frame.columns:
            return frame
        codes = frame["ts_code"].astype(str)
        is_stock = codes.str.match(r"\d{6}\.(?:SH|SZ|BJ)$")
        return frame[~is_stock | codes.isin(allowed)].reset_index(drop=True)

    def _build_universe(self, decision_time: datetime, config: SnapshotConfig) -> pd.DataFrame:
        """Stocks listed as of the decision day (delistings after it included).

        Building from the current L partition alone would drop names delisted
        later than the decision day and inject survivorship bias. Point-in-time
        columns: ``name`` is the name in force at the decision day (from
        namechange — the current stock_basic name may be a future rename), and
        ``delist_date`` is dropped after filtering — every survivor's delisting
        is after the decision day, i.e. future information.
        """
        day = decision_time.strftime("%Y%m%d")
        frames = []
        for status in ("L", "D", "P"):
            path = self.raw_dir / "stock_basic" / f"list_status={status}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            raise FileNotFoundError(f"missing stock_basic partitions under {self.raw_dir / 'stock_basic'}")
        basic = pd.concat(frames, ignore_index=True)
        keep = [col for col in ("ts_code", "exchange", "list_date", "delist_date", "market") if col in basic.columns]
        universe = basic[keep].copy()
        universe["ts_code"] = universe["ts_code"].astype(str)
        universe = universe.drop_duplicates("ts_code", keep="first")
        if "list_date" in universe.columns:
            universe = universe[universe["list_date"].fillna("").astype(str) <= day]
        if "delist_date" in universe.columns:
            delist = universe["delist_date"].fillna("").astype(str)
            universe = universe[(delist == "") | (delist == "None") | (delist > day)]
            universe = universe.drop(columns=["delist_date"])
        universe = universe.merge(self._names_as_of(decision_time), on="ts_code", how="left")
        if config.include_industry:
            industry = self._industry_membership(decision_time.strftime("%Y%m%d"))
            if not industry.empty:
                universe = universe.merge(industry, on="ts_code", how="left")
        return universe.reset_index(drop=True)

    def _names_as_of(self, decision_time: datetime) -> pd.DataFrame:
        """``ts_code -> name`` in force at the decision day (announced by then).

        The namechange dataset carries every code's listing name, so a null
        merge result is a genuine data gap, not a normal case; the current
        stock_basic name is never used as a fallback — it may be a rename the
        market had not seen at the decision day."""
        path = self.raw_dir / "namechange" / "namechange.parquet"
        if not path.exists():
            raise FileNotFoundError(f"namechange dataset required for as-of universe names: {path}")
        names = pd.read_parquet(path)
        day = decision_time.strftime("%Y%m%d")
        if "ann_date" in names.columns:
            names = names[names["ann_date"].astype(str).str.strip().le(day) | names["ann_date"].isna()]
        names = names[names["start_date"].astype(str) <= day]
        names = names.sort_values("start_date").drop_duplicates("ts_code", keep="last")
        return names[["ts_code", "name"]]

    def _industry_membership(self, decision_day: str) -> pd.DataFrame:
        """As-of SW level-1 membership: in_date <= decision day < out_date.

        Decision days before the SW2021 index switch use the frozen SW2014
        membership (legacy index_member partitions) so each day is classified
        by the scheme the market actually used then; later days use the
        SW2021 scheme (index_member_all partitions)."""
        if decision_day < SW2021_EFFECTIVE_DAY:
            return self._sw2014_membership(decision_day)
        dataset_dir = self.raw_dir / "index_member_all"
        if not dataset_dir.exists():
            return pd.DataFrame()
        frames = []
        for path in sorted(dataset_dir.glob("l1_code=*.parquet")):
            frame = pd.read_parquet(path)
            cols = [col for col in ("ts_code", "l1_code", "l1_name", "in_date", "out_date") if col in frame.columns]
            if "ts_code" in cols:
                frames.append(frame[cols])
        if not frames:
            return pd.DataFrame()
        merged = _membership_as_of(concat_rows(frames), decision_day)
        return merged.drop_duplicates("ts_code", keep="last")[
            [col for col in ("ts_code", "l1_code", "l1_name") if col in merged.columns]
        ]

    def _sw2014_membership(self, decision_day: str) -> pd.DataFrame:
        dataset_dir = self.raw_dir / "index_member"
        classify_path = self.raw_dir / "index_classify" / "src=SW2014.parquet"
        if not dataset_dir.exists() or not classify_path.exists():
            return pd.DataFrame()
        frames = [
            pd.read_parquet(path, columns=["index_code", "con_code", "in_date", "out_date"])
            for path in sorted(dataset_dir.glob("l1_code=*.parquet"))
        ]
        if not frames:
            return pd.DataFrame()
        merged = _membership_as_of(concat_rows(frames), decision_day)
        classify = pd.read_parquet(classify_path, columns=["index_code", "industry_name", "level"])
        names = classify.loc[classify["level"].astype(str) == "L1", ["index_code", "industry_name"]]
        merged = merged.merge(names, on="index_code", how="left")
        merged = merged.rename(columns={"con_code": "ts_code", "index_code": "l1_code", "industry_name": "l1_name"})
        return merged.drop_duplicates("ts_code", keep="last")[["ts_code", "l1_code", "l1_name"]]


SW2021_EFFECTIVE_DAY = "20211213"  # Shenwan indices switched to the 2021 classification on this day


def _membership_as_of(merged: pd.DataFrame, decision_day: str) -> pd.DataFrame:
    """The as-of membership rule shared by both SW vintages:
    in_date <= decision day < out_date; the latest in_date wins on dedup."""
    if "in_date" in merged.columns:
        merged = merged[merged["in_date"].fillna("").astype(str) <= decision_day]
    if "out_date" in merged.columns:
        out_date = merged["out_date"].fillna("").astype(str)
        merged = merged[(out_date == "") | (out_date == "None") | (out_date > decision_day)]
    return merged.sort_values("in_date" if "in_date" in merged.columns else "ts_code")


def _stamp_daily_available_at(daily: pd.DataFrame, contract) -> pd.DataFrame:
    """Add a row-level ``available_at`` to replay daily bars (the daily core's
    publish time, ``trade_date`` close). The Timeview gates the whole daily domain
    on the evening refresh node, so any time before that night's 23:35 makes the
    row roll in from the next day; this column carries that timestamp explicitly."""
    if daily.empty or "trade_date" not in daily.columns:
        return daily
    out = daily.copy()
    out["available_at"] = [
        contract.available_at(datetime.strptime(str(date), "%Y%m%d").date()).isoformat()
        for date in out["trade_date"].astype(str)
    ]
    return out


def _filter_trade_dates(frame: pd.DataFrame, visible_dates: list[str]) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame.copy()
    visible = set(visible_dates)
    out = frame[frame["trade_date"].astype(str).isin(visible)].copy()
    return out


def _run_domain_tasks(tasks: list[DomainBuildTask]) -> dict[str, DomainBuildResult]:
    """Run independent snapshot domains with a small dependency-aware pool."""
    if len({name for name, _, _ in tasks}) != len(tasks):
        raise ValueError("snapshot domain task names must be unique")
    remaining = list(tasks)
    results: dict[str, DomainBuildResult] = {}
    running: dict[Future[DomainBuildResult], str] = {}
    executor = ThreadPoolExecutor(max_workers=SNAPSHOT_DOMAIN_WORKERS, thread_name_prefix="snapshot-domain")
    try:
        while remaining or running:
            while len(running) < SNAPSHOT_DOMAIN_WORKERS:
                ready_index = next(
                    (
                        index
                        for index, (_, dependencies, _) in enumerate(remaining)
                        if all(dependency in results for dependency in dependencies)
                    ),
                    None,
                )
                if ready_index is None:
                    break
                name, _, build = remaining.pop(ready_index)
                running[executor.submit(build, dict(results))] = name

            if not running:
                unresolved = {name: dependencies for name, dependencies, _ in remaining}
                raise ValueError(f"snapshot domain dependencies cannot be resolved: {unresolved}")

            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                name = running.pop(future)
                results[name] = future.result()
    except Exception:
        for future in running:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results


def _pad_union_schema(merged: pd.DataFrame, schema_only: Mapping[str, object]) -> pd.DataFrame:
    """Add zero-row dataset columns in one Arrow pass.

    Assigning each missing column through pandas on an 8M-row union recopies
    the frame per column. Arrow appends null arrays without rewriting row data.
    """
    missing: list[pa.Field] = []
    seen: set[str] = set(merged.columns)
    if schema_only and "dataset" not in seen:
        missing.append(pa.field("dataset", pa.string()))
        seen.add("dataset")
    for schema in schema_only.values():
        if not isinstance(schema, pa.Schema):
            continue
        for field in schema:
            if field.name not in seen:
                missing.append(field)
                seen.add(field.name)
    if not missing:
        return merged
    if merged.empty:
        for field in missing:
            merged[field.name] = pd.Series(dtype=pd.ArrowDtype(field.type))
        return merged
    table = pa.Table.from_pandas(merged, preserve_index=False)
    n = table.num_rows
    for field in missing:
        table = table.append_column(field.name, pa.nulls(n, type=field.type))
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def _window_start(decision_time: datetime, months: int) -> pd.Timestamp:
    window_start = (pd.Timestamp(decision_time) - pd.DateOffset(months=months)).tz_localize(None)
    return window_start.tz_localize(CN_TZ)


def finalize_snapshot_dir(snapshot_dir: str | Path, **fields: object) -> dict[str, object]:
    """Stamp an externally assembled snapshot directory with an immutable manifest.

    Directories containing union files must supply the builder's
    ``domains[...]["dataset_columns"]`` explicitly via ``fields`` — dataset
    ownership is never inferred from file content, and validation below fails
    without it. Same gate as the builder: every column must classify in the
    unit registry.
    """
    snapshot_dir = Path(snapshot_dir)
    manifest: dict[str, object] = {"snapshot_id": new_id("snap"), "created_at": utc_now_iso(), **fields}
    validate_snapshot_units(snapshot_dir, manifest)
    _write_manifest(snapshot_dir, manifest, trim_trade_dates=False)
    return manifest


def load_snapshot_manifest(snapshot_dir: str | Path) -> dict[str, object]:
    path = Path(snapshot_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"snapshot manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_overlaps(stem: str, start_day: str, end_day: str) -> bool:
    """Cheap pre-filter on partition file names; unknown layouts are read fully."""
    if "=" not in stem:
        return True
    key, value = stem.split("=", 1)
    if key in {"trade_date", "date", "ann_date"} and len(value) == 8 and value.isdigit():
        return start_day <= value <= end_day
    if key in {"month", "ann_month"} and len(value) == 6 and value.isdigit():
        return start_day[:6] <= value <= end_day[:6]
    if key == "year" and len(value) == 4 and value.isdigit():
        return start_day[:4] <= value <= end_day[:4]
    return True


# Profiled column sets are shared with the agent data summary (summary.py),
# imported above as PROFILE_DATE_COLUMNS / PROFILE_NULL_COLUMNS.
PROFILE_FULL_SCAN_MAX_ROWS = 1_000_000


def _write_with_profile(path: Path, frame: pd.DataFrame, *, build_seconds: float) -> dict[str, object]:
    started = time.perf_counter()
    _write(path, frame)
    return _frame_profile(path, frame, build_seconds=build_seconds, write_seconds=time.perf_counter() - started)


def _profile_timings(profiles: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        name: {"build_seconds": item["build_seconds"], "write_seconds": item["write_seconds"]}
        for name, item in profiles.items()
    }


def _frame_profile(
    path: Path,
    frame: pd.DataFrame,
    *,
    build_seconds: float,
    write_seconds: float,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "file": path.name,
        "rows": int(len(frame)),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "column_count": int(len(frame.columns)),
        "columns": [str(col) for col in frame.columns],
        "build_seconds": round(float(build_seconds), 3),
        "write_seconds": round(float(write_seconds), 3),
    }
    if not frame.empty:
        footer_profile = _parquet_footer_profile(path) if len(frame) > PROFILE_FULL_SCAN_MAX_ROWS else None
        date_ranges = footer_profile[0] if footer_profile is not None else _profile_date_ranges(frame)
        if date_ranges:
            profile["date_ranges"] = date_ranges
        key_nulls = footer_profile[1] if footer_profile is not None else _profile_key_nulls(frame)
        if key_nulls:
            profile["key_nulls"] = key_nulls
        if "dataset" in frame.columns and len(frame) <= PROFILE_FULL_SCAN_MAX_ROWS:
            counts = frame["dataset"].fillna("").astype(str).value_counts().head(50)
            profile["dataset_counts"] = {str(key): int(value) for key, value in counts.items()}
        elif "dataset" in frame.columns:
            profile["dataset_counts"] = "skipped_large_frame"
    return profile


def _parquet_file_profile(
    path: Path,
    *,
    rows: int,
    columns: list[str],
    build_seconds: float,
    write_seconds: float,
) -> dict[str, object]:
    """Profile a streamed Parquet file from footer statistics only."""
    profile: dict[str, object] = {
        "file": path.name,
        "rows": int(rows),
        "size_bytes": int(path.stat().st_size),
        "column_count": len(columns),
        "columns": [str(column) for column in columns],
        "build_seconds": round(float(build_seconds), 3),
        "write_seconds": round(float(write_seconds), 3),
    }
    if rows:
        footer_profile = _parquet_footer_profile(path)
        if footer_profile is None:
            raise ValueError(f"streamed Parquet footer lacks required profile statistics: {path}")
        date_ranges, key_nulls = footer_profile
        if date_ranges:
            profile["date_ranges"] = date_ranges
        if key_nulls:
            profile["key_nulls"] = key_nulls
    return profile


def _parquet_footer_profile(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, int]] | None:
    """Read exact large-frame range/null statistics from Parquet metadata.

    Pandas/pyarrow snapshot writes include per-row-group statistics. If a
    different writer omits any statistic needed by the existing manifest
    contract, return ``None`` so the caller keeps the prior DataFrame scan.
    """
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    columns = {name: index for index, name in enumerate(parquet.schema_arrow.names)}
    date_ranges: dict[str, dict[str, str]] = {}
    key_nulls: dict[str, int] = {}

    for column in PROFILE_DATE_COLUMNS:
        index = columns.get(column)
        if index is None:
            continue
        # The prior manifest contract stringified the in-memory values. Parquet
        # timestamp statistics may normalize timezone-aware values to UTC, which
        # is the same instant but a different manifest string. Keep the footer
        # fast path for production string date columns and fall back to the
        # DataFrame scan for temporal/other representations.
        field_type = parquet.schema_arrow.field(column).type
        if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
            return None
        minimums: list[str] = []
        maximums: list[str] = []
        for group, statistics in iter_column_statistics(metadata, index):
            if statistics is None or statistics.null_count is None:
                return None
            if int(statistics.null_count) == int(group.num_rows):
                continue
            if not statistics.has_min_max:
                return None
            minimums.append(scalar_to_text(statistics.min))
            maximums.append(scalar_to_text(statistics.max))
        if minimums:
            date_ranges[column] = {"min": min(minimums), "max": max(maximums)}

    for column in PROFILE_NULL_COLUMNS:
        index = columns.get(column)
        if index is None:
            continue
        null_count = 0
        for _group, statistics in iter_column_statistics(metadata, index):
            if statistics is None or statistics.null_count is None:
                return None
            null_count += int(statistics.null_count)
        key_nulls[column] = null_count
    return date_ranges, key_nulls


def _profile_date_ranges(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    ranges: dict[str, dict[str, str]] = {}
    for column in PROFILE_DATE_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        text = values.astype(str)
        ranges[column] = {"min": str(text.min()), "max": str(text.max())}
    return ranges


def _profile_key_nulls(frame: pd.DataFrame) -> dict[str, int]:
    nulls: dict[str, int] = {}
    for column in PROFILE_NULL_COLUMNS:
        if column in frame.columns:
            nulls[column] = int(frame[column].isna().sum())
    return nulls


def _dataset_footer_schema(dataset_dir: Path):
    """Arrow schema from the newest partition footer: the dataset's schema
    contribution when no rows are visible in the build window."""
    latest = max(dataset_dir.rglob("*.parquet"), default=None, key=lambda p: p.name)
    if latest is None:
        return None
    return pq.read_schema(latest)


def _write(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, tmp)
    tmp.replace(path)


def _fundamental_dataset_columns(raw_dir: Path, datasets: tuple[str, ...]) -> dict[str, list[str]]:
    """Per-dataset columns from the vendor raw schemas plus PIT sidecars.

    The PIT fundamental store materializes union-schema partitions, so window
    content cannot attribute columns (an all-NA-in-window legitimate field
    would be dropped); the raw vendor footers are the schema truth.
    """
    out: dict[str, list[str]] = {}
    for dataset in datasets:
        files = sorted((raw_dir / dataset).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"missing raw partitions for fundamental dataset: {raw_dir / dataset}")
        columns = dict.fromkeys(FUNDAMENTAL_SIDECAR_COLUMNS)
        for index in sorted({0, len(files) // 2, len(files) - 1}):
            columns.update(dict.fromkeys(pq.read_schema(files[index]).names))
        out[dataset] = list(columns)
    return out


def _apply_fundamental_exclusions(
    frame: pd.DataFrame, dataset_columns: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Apply SNAPSHOT_EXCLUDED_COLUMNS to the already-unioned fundamentals.

    events/macro drop excluded columns per dataset before their union; the PIT
    fundamental store hands over one union frame, so a column can only be
    dropped for every dataset at once. Refuse when another fundamental dataset
    still declares an excluded name rather than silently widening the removal.
    """
    kept = {
        dataset: [
            column
            for column in columns
            if column not in SNAPSHOT_EXCLUDED_COLUMNS.get(dataset, ())
        ]
        for dataset, columns in dataset_columns.items()
    }
    excluded = {
        column
        for dataset, columns in dataset_columns.items()
        for column in SNAPSHOT_EXCLUDED_COLUMNS.get(dataset, ())
        if column in columns
    }
    if not excluded:
        return frame, kept
    shared = sorted(
        column for column in excluded if any(column in columns for columns in kept.values())
    )
    if shared:
        raise ValueError(
            "excluded fundamental columns are also declared by another dataset "
            f"of the same union file: {shared}"
        )
    return frame.drop(columns=[c for c in frame.columns if c in excluded]), kept


# The raw-lake stamp also records the update transaction (host commands) and
# the config identity (host interpreter path). The snapshot manifest is
# mounted read-only into the Agent sandbox, so it keeps only the identity the
# PIT contract checks; the full stamp stays in the lake's own record.
_RAW_GENERATION_IDENTITY_KEYS = ("schema_version", "state", "generation_id", "completed_at")


def _raw_generation_identity(stamp: dict[str, object] | None) -> dict[str, object] | None:
    if stamp is None:
        return None
    return {key: stamp[key] for key in _RAW_GENERATION_IDENTITY_KEYS if key in stamp}


def _write_manifest(output_dir: Path, manifest: dict[str, object], *, trim_trade_dates: bool = True) -> None:
    """Single manifest.json writer. The builder trims bulky per-domain
    ``trade_dates`` (coverage fields remain); ``finalize_snapshot_dir`` keeps
    the caller's fields verbatim."""
    if trim_trade_dates:
        manifest = json.loads(json.dumps(manifest, ensure_ascii=False, default=str))
        for domain in manifest.get("domains", {}).values():
            domain.pop("trade_dates", None)  # keep the manifest small; coverage fields remain
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
