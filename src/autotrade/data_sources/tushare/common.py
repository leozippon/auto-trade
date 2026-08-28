#!/usr/bin/env python3
"""Shared TuShare constants, schemas, client, and utility helpers."""

from __future__ import annotations
import argparse
from bisect import bisect_right
import calendar
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd

# Raw-lake research contract values are owned by the environment's PIT layer;
# the ingest adapter imports them so producer and consumer cannot drift.
# STK_AUCTION_PRICE_ABS_TOLERANCE is re-exported: download.py reads it from
# this hub, like the .io re-exports below.
from autotrade.environment.data.contracts import (  # noqa: F401
    BOARD_TRADING_DATASETS,
    DOMAIN_STATUS_FILES,
    STK_AUCTION_PRICE_ABS_TOLERANCE,
)

from .io import CorruptSidecarError, append_jsonl_unique, frames_content_equal, migrate_partition_identity, parquet_meta, parquet_rows, read_many, write_parquet
from .io import committed_partition_intact  # noqa: F401 -- re-exported via this hub
from .io import has_pagination_probe, inherited_updater_lock_fd  # noqa: F401 -- re-exported via this hub (audit.py imports has_pagination_probe, download.py imports inherited_updater_lock_fd from common)


DEFAULT_TUSHARE_RELAY_URL = "https://fast.xiaodefa.cn"

# Status file names are owned by environment.data.contracts (one source for
# audit producers, snapshot gates, and research-release pinning).
CORE_MARKET_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["daily"]
FUNDAMENTAL_RAW_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["fundamentals_raw"]

TEXT_EVIDENCE_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["text"]

INTRADAY_MINUTES_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["intraday_1min"]

REVISION_EVENTS_PATH = "results/data_quality/revision_events.jsonl"

REVISION_SUMMARY_PATH = "results/data_quality/revision_summary.json"

# v2 drops the always-constant record_type/downstream_status and api_name
# (verbatim duplicate of dataset); schema_version stays as the discriminator.
REVISION_EVENT_SCHEMA_VERSION = 2
REVISION_EVENT_FIELDS = (
    "schema_version",
    "event_id",
    "detected_at",
    "source",
    "dataset",
    "partition",
    "path",
    "severity",
    "key_columns",
    "old_rows",
    "new_rows",
    "changed_keys",
    "added_keys",
    "removed_keys",
    "missing_key_columns_old",
    "missing_key_columns_new",
    "duplicate_key_rows_old",
    "duplicate_key_rows_new",
    "comparison_issue",
    "affected_ts_codes",
    "affected_ts_codes_sample",
    "changed_keys_sample",
    "added_keys_sample",
    "removed_keys_sample",
    "changed_columns",
    "changed_columns_sample",
    "added_rows_sample",
    "removed_rows_sample",
    "old_columns",
    "new_columns",
    "schema_changed",
    "write_id",
    "write_action",
    "allow_empty_revision_overwrite",
)
MACRO_CONTEXT_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["macro"]

EVENT_FLOW_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["events"]

BOARD_TRADING_STATUS_PATH = "results/data_quality/" + DOMAIN_STATUS_FILES["board_trading"]

# Child-process contract: validation/polling ended before any raw write began.
# The cron runner may restore the previous committed generation and retry later.
NO_MUTATION_RETRY_EXIT_CODE = 75

# Child-process contract: the lake DID mutate, but a required partition is still
# unpublished at the source. The generation must be committed (the writes are
# real), yet the run must not report success: an "ok" state lets
# skip_if_already_ok suppress every later attempt at the same resolved end date,
# which silently abandons the partition once the window moves on.
MUTATED_NOT_READY_RETRY_EXIT_CODE = 76

def resolve_revision_ledger(
    raw_dir: Path | str,
    revision_ledger: Path | str | None = REVISION_EVENTS_PATH,
    *,
    repo_root: Path | str | None = None,
) -> Path | None:
    """Resolve revision ledger path without letting temp raw dirs pollute production."""
    if revision_ledger in {None, ""}:
        return None
    root = Path(repo_root or Path.cwd()).resolve()
    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    raw_path = raw_path.resolve()
    ledger = Path(revision_ledger)
    if ledger.is_absolute():
        return ledger
    if ledger == Path(REVISION_EVENTS_PATH):
        production_raw = (root / "data/raw").resolve()
        if raw_path != production_raw:
            try:
                raw_path.relative_to(production_raw)
            except ValueError:
                return raw_path.parent / "revision_events.jsonl"
    return root / ledger

DOWNLOAD_TIER_CHOICES = ("reference", "daily", "fundamental", "intraday", "event_flow", "board_trading", "text_evidence", "macro", "global")

REFERENCE_DATASETS = [
    "stock_basic",
    "stock_company",
    "trade_cal",
    "bak_basic",
    "namechange",
    "index_classify",
    "index_member_all",
    "index_member",
    "ths_index",
    "ths_member",
    "index_basic",
    "index_weight",
]

# Generic daily download/update default. stk_auction remains in the required/audit
# gate, but it is written only by dedicated capture and evening backfill.
DAILY_DOWNLOAD_DATASETS = ["daily", "adj_factor", "daily_basic", "stk_limit", "suspend_d"]
DAILY_REQUIRED_DATASETS = [*DAILY_DOWNLOAD_DATASETS, "stk_auction"]

FUNDAMENTAL_DATASETS = [
    "income_vip",
    "balancesheet_vip",
    "cashflow_vip",
    "fina_indicator_vip",
    "forecast_vip",
    "express_vip",
    "dividend",
    "fina_audit",
    "fina_mainbz_vip",
    "disclosure_date",
]

TEXT_DATASETS = [
    "anns_d",
    "major_news",
    "cctv_news",
    "npr",
    "research_report",
    "report_rc",
    "irm_qa_sh",
    "irm_qa_sz",
    "news",
]

INTRADAY_DATASETS = ["stk_mins_1min"]

EVENT_FLOW_DATASETS = [
    "margin",
    "margin_detail",
    "margin_secs",
    "moneyflow",
    "moneyflow_dc",
    "moneyflow_ths",
    "moneyflow_ind_dc",
    "moneyflow_ind_ths",
    "moneyflow_cnt_ths",
    "cyq_perf",
    "bak_daily",
    "stk_holdernumber",
    "top10_holders",
    "top10_floatholders",
    "pledge_detail",
    "stk_surv",
    "new_share",
    "stk_holdertrade",
    "repurchase",
    "share_float",
    "block_trade",
]

BOARD_TRADING_DEFAULT_DATASETS = list(BOARD_TRADING_DATASETS)

BOARD_KPL_TAGS = ["涨停", "炸板", "跌停", "自然涨停", "竞价"]

BOARD_THS_LIMIT_TYPES = ["涨停池", "连扳池", "冲刺涨停", "炸板池", "跌停池"]

BOARD_THS_HOT_MARKETS = ["热股", "行业板块", "概念板块"]

BOARD_DC_HOT_MARKETS = ["A股市场"]

BOARD_DC_HOT_TYPES = ["人气榜", "飙升榜"]

BOARD_HOT_IS_NEW = ["N"]

SHARE_FLOAT_FIELDS = "ts_code,ann_date,float_date,float_share,float_ratio,holder_name,share_type"

SHARE_FLOAT_ROW_LIMIT = 6000

SHARE_FLOAT_UNLOCK_TITLE_PATTERN = re.compile(
    r"限售|解禁|上市流通|解除限售|限售股份上市|限售股上市|首次公开发行限售|非公开发行限售|定向增发限售"
)

# Macro range pulls (quarter_once/month_once) always retain history from at
# least this floor so the canonical range file's coverage never shrinks with
# a rolling cron window.
MACRO_RETAINED_FLOOR = "20200101"

MACRO_DATASETS = [
    "cn_schedule",
    "cn_gdp",
    "cn_cpi",
    "cn_ppi",
    "cn_pmi",
    "cn_m",
    "sf_month",
    "shibor",
    "shibor_quote",
    "shibor_lpr",
    "repo_daily",
    "us_tycr",
    "us_trycr",
    "us_tbr",
    "us_tltr",
    "index_daily",
    "broker_recommend",
    "ths_daily",
    "index_dailybasic",
    "sw_daily",
    "ci_daily",
    "daily_info",
    "sz_daily_info",
    "moneyflow_mkt_dc",
    "index_global",
    "fx_daily",
    "eco_cal",
    "fut_basic",
    "fut_mapping",
    "fut_daily",
    "opt_basic",
    "opt_daily",
    "cb_basic",
    "cb_daily",
    "cb_call",
]

MACRO_REGIME_DEFAULT_DATASETS = [
    "cn_schedule",
    "cn_gdp",
    "cn_cpi",
    "cn_ppi",
    "cn_pmi",
    "cn_m",
    "sf_month",
    "shibor",
    # Bank-level SHIBOR quotes ride the evening refresh like the other rate
    # series (historically a one-off download that silently went stale).
    "shibor_quote",
    "shibor_lpr",
    "repo_daily",
    # Core A-share benchmark indexes (see DEFAULT_CN_INDEX_CODES).
    "index_daily",
    "index_dailybasic",
    "sw_daily",
    "ci_daily",
    "daily_info",
    "sz_daily_info",
    "moneyflow_mkt_dc",
    "ths_daily",
    "broker_recommend",
    # Derivatives market context: CFFEX futures basis inputs, option chains,
    # CB premium/redemption ledger.
    "fut_basic",
    "fut_mapping",
    "fut_daily",
    "opt_basic",
    "opt_daily",
    "cb_basic",
    "cb_daily",
    "cb_call",
]

GLOBAL_CONTEXT_DEFAULT_DATASETS = [
    "eco_cal",
    "index_global",
    "fx_daily",
    "us_tycr",
    "us_trycr",
    "us_tbr",
    "us_tltr",
]

DEFAULT_GLOBAL_INDEX_CODES = ["XIN9", "HSI", "HKTECH", "DJI", "SPX", "IXIC", "FTSE", "GDAXI", "N225", "RUT"]

# Core A-share benchmark indexes for `index_daily` (上证指数、上证50、沪深300、
# 中证500、中证1000、创业板指、科创50): market timing, beta management, and
# relative-strength context for the Agent, plus the host-side CSI300 benchmark.
DEFAULT_CN_INDEX_CODES = [
    "000001.SH",
    "000016.SH",
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "399006.SZ",
    "000688.SH",
]

# The index_weight source clamps every call to 7,000 rows regardless of the
# requested limit and rejects offsets beyond ~150k, so a whole-history pull
# per index truncates silently (most-recent-first). Downloads page inside
# per-year windows per index code (largest universe ~28k rows/year).
INDEX_WEIGHT_PAGE_LIMIT = 7000
INDEX_WEIGHT_START_DATE = "20200101"
INDEX_WEIGHT_FIELDS = "index_code,con_code,trade_date,weight"

DEFAULT_FX_CODES = ["USDCNH.FXCM"]

TRADE_DATE_PAGE_LIMIT = 5000

STK_MINS_API_NAME = "stk_mins"

STK_MINS_DATASET = "stk_mins_1min"

STK_MINS_BY_DATE_DATASET = "stk_mins_1min_by_date"

STK_MINS_FREQ = "1min"

STK_MINS_FIELDS = "ts_code,trade_time,open,high,low,close,vol,amount"

STK_MINS_PAGE_LIMIT = 8000

STK_MINS_REQUIRED_COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount", "trade_date", "available_at", "available_at_rule"]

NEWS_SOURCES = [
    "sina",
    "wallstreetcn",
    "10jqka",
    "eastmoney",
    "yuncaijing",
    "fenghuang",
    "jinrongjie",
    "cls",
    "yicai",
]

BAK_BASIC_FIELDS = "trade_date,ts_code,name,industry,area,pe,float_share,total_share,total_assets,liquid_assets,fixed_assets,reserved,reserved_pershare,eps,bvps,pb,list_date,undp,per_undp,rev_yoy,profit_yoy,gpr,npr,holder_num"

SEMANTIC_DOC_REFS = {
    "stock_basic": "https://tushare.pro/document/2?doc_id=25",
    "trade_cal": "https://tushare.pro/document/2?doc_id=26",
    "daily": "https://tushare.pro/document/2?doc_id=27",
    "daily_basic": "https://tushare.pro/document/2?doc_id=32",
    "bak_daily": "https://tushare.pro/document/2?doc_id=255",
    "bak_basic": "https://tushare.pro/document/2?doc_id=262",
    "stk_limit": "https://tushare.pro/document/2?doc_id=183",
    "adj_factor": "https://tushare.pro/document/2?doc_id=28",
    "suspend_d": "https://tushare.pro/document/2?doc_id=214",
    "limit_list_d": "https://tushare.pro/document/2?doc_id=298",
}

INTEGRATED_DOC_REFS = {
    **SEMANTIC_DOC_REFS,
    "stock_company": "https://tushare.pro/document/2?doc_id=112",
    "namechange": "https://tushare.pro/document/2?doc_id=100",
    "index_classify": "https://tushare.pro/document/2?doc_id=181",
    "index_member_all": "https://tushare.pro/document/2?doc_id=335",
    "index_member": "https://tushare.pro/document/2?doc_id=182",
    "income_vip": "https://tushare.pro/document/2?doc_id=33",
    "balancesheet_vip": "https://tushare.pro/document/2?doc_id=36",
    "cashflow_vip": "https://tushare.pro/document/2?doc_id=44",
    "forecast_vip": "https://tushare.pro/document/2?doc_id=45",
    "express_vip": "https://tushare.pro/document/2?doc_id=46",
    "fina_indicator_vip": "https://tushare.pro/document/2?doc_id=79",
    "dividend": "https://tushare.pro/document/2?doc_id=103",
    "fina_audit": "https://tushare.pro/document/2?doc_id=80",
    "fina_mainbz_vip": "https://tushare.pro/document/2?doc_id=81",
    "disclosure_date": "https://tushare.pro/document/2?doc_id=162",
    "anns_d": "https://www.tushare.pro/document/2?doc_id=176",
    "major_news": "https://tushare.pro/document/2?doc_id=195",
    "cctv_news": "https://tushare.pro/document/2?doc_id=154",
    "npr": "https://tushare.pro/document/2?doc_id=406",
    "research_report": "https://tushare.pro/document/2?doc_id=415",
    "report_rc": "https://tushare.pro/document/2?doc_id=292",
    "news": "https://www.tushare.pro/document/41?doc_id=143",
    "stk_mins_1min": "https://tushare.pro/document/2?doc_id=370",
    "cn_schedule": "https://tushare.pro/document/2?doc_id=461",
    "cn_gdp": "https://tushare.pro/document/2?doc_id=227",
    "cn_cpi": "https://tushare.pro/document/2?doc_id=228",
    "cn_ppi": "https://tushare.pro/document/2?doc_id=229",
    "cn_pmi": "https://tushare.pro/document/2?doc_id=325",
    "cn_m": "https://tushare.pro/document/2?doc_id=242",
    "sf_month": "https://tushare.pro/document/2?doc_id=310",
    "shibor": "https://tushare.pro/document/2?doc_id=202",
    "shibor_quote": "https://tushare.pro/document/2?doc_id=203",
    "shibor_lpr": "https://tushare.pro/document/2?doc_id=204",
    "repo_daily": "https://tushare.pro/document/2?doc_id=256",
    "us_tycr": "https://tushare.pro/document/2?doc_id=218",
    "us_trycr": "https://tushare.pro/document/2?doc_id=219",
    "us_tbr": "https://tushare.pro/document/2?doc_id=220",
    "us_tltr": "https://tushare.pro/document/2?doc_id=222",
    "index_daily": "https://tushare.pro/document/2?doc_id=95",
    "index_dailybasic": "https://tushare.pro/document/2?doc_id=128",
    "ths_index": "https://tushare.pro/document/2?doc_id=259",
    "ths_member": "https://tushare.pro/document/2?doc_id=261",
    "index_basic": "https://tushare.pro/document/2?doc_id=94",
    "index_weight": "https://tushare.pro/document/2?doc_id=96",
    "irm_qa_sh": "https://tushare.pro/document/2?doc_id=364",
    "irm_qa_sz": "https://tushare.pro/document/2?doc_id=365",
    "ths_daily": "https://tushare.pro/document/2?doc_id=260",
    "broker_recommend": "https://tushare.pro/document/2?doc_id=337",
    "top10_holders": "https://tushare.pro/document/2?doc_id=61",
    "top10_floatholders": "https://tushare.pro/document/2?doc_id=62",
    "pledge_detail": "https://tushare.pro/document/2?doc_id=111",
    "stk_surv": "https://tushare.pro/document/2?doc_id=275",
    "new_share": "https://tushare.pro/document/2?doc_id=123",
    "kpl_concept_cons": "https://tushare.pro/document/2?doc_id=351",
    "dc_index": "https://tushare.pro/document/2?doc_id=362",
    "dc_member": "https://tushare.pro/document/2?doc_id=363",
    "sw_daily": "https://tushare.pro/document/2",
    "ci_daily": "https://tushare.pro/document/2",
    "daily_info": "https://tushare.pro/document/2?doc_id=215",
    "sz_daily_info": "https://tushare.pro/document/2?doc_id=278",
    "moneyflow_mkt_dc": "https://tushare.pro/document/2?doc_id=345",
    "index_global": "https://tushare.pro/document/2?doc_id=211",
    "fx_daily": "https://tushare.pro/document/2?doc_id=179",
    "eco_cal": "https://tushare.pro/document/2?doc_id=233",
    "moneyflow_dc": "https://tushare.pro/document/2?doc_id=339",
    "moneyflow_ths": "https://tushare.pro/document/2?doc_id=344",
    "moneyflow_ind_dc": "https://tushare.pro/document/2?doc_id=343",
    "moneyflow_ind_ths": "https://tushare.pro/document/2?doc_id=341",
    "moneyflow_cnt_ths": "https://tushare.pro/document/2?doc_id=342",
    "cyq_perf": "https://tushare.pro/document/2?doc_id=293",
    "stk_auction": "https://tushare.pro/document/2?doc_id=369",
    "margin": "https://tushare.pro/document/2?doc_id=58",
    "margin_detail": "https://tushare.pro/document/2?doc_id=59",
    "margin_secs": "https://tushare.pro/document/2?doc_id=326",
    "moneyflow": "https://tushare.pro/document/2?doc_id=170",
    "stk_holdernumber": "https://tushare.pro/document/2?doc_id=166",
    "stk_holdertrade": "https://tushare.pro/document/2?doc_id=175",
    "repurchase": "https://tushare.pro/document/2?doc_id=124",
    "share_float": "https://tushare.pro/document/2?doc_id=160",
    "block_trade": "https://tushare.pro/document/2?doc_id=161",
    "kpl_list": "https://tushare.pro/document/2?doc_id=347",
    "limit_step": "https://tushare.pro/document/2?doc_id=356",
    "limit_cpt_list": "https://tushare.pro/document/2?doc_id=357",
    "limit_list_ths": "https://tushare.pro/document/2?doc_id=355",
    "top_list": "https://tushare.pro/document/2?doc_id=106",
    "top_inst": "https://tushare.pro/document/2?doc_id=107",
    "hm_list": "https://tushare.pro/document/2?doc_id=311",
    "hm_detail": "https://tushare.pro/document/2?doc_id=312",
    "ths_hot": "https://tushare.pro/document/2?doc_id=320",
    "dc_hot": "https://tushare.pro/document/2?doc_id=321",
}

@dataclass
class ApiResult:
    fields: list[str]
    items: list[list[Any]]

@dataclass
class TradeDateDataset:
    api_name: str
    fields: str
    start_date: str = "20100101"
    zero_rows_ok: bool = False
    key_columns: tuple[str, ...] = ("trade_date", "ts_code")
    # Source-side page overlap can repeat identical rows (observed on the
    # auction endpoints); dropping EXACT duplicates is information-preserving.
    dedup_exact: bool = False

@dataclass
class FundamentalDataset:
    api_name: str
    strategy: str
    key_columns: tuple[str, ...]
    period_param: str = "period"
    fields: str = ""
    # Measured per-call cap for sources that silently truncate large requests
    # (spec_page_limit clamps the tier default to it); None = no known cap.
    page_limit: int | None = None

@dataclass
class TextDataset:
    api_name: str
    strategy: str
    key_columns: tuple[str, ...]
    fields: str
    page_limit: int | None = None
    start_date: str = "20100101"
    zero_rows_ok: bool = True
    time_column: str = ""
    date_column: str = ""

@dataclass
class MacroDataset:
    api_name: str
    strategy: str
    key_columns: tuple[str, ...]
    fields: str = ""
    page_limit: int | None = None
    date_column: str = ""
    time_column: str = ""
    start_date: str = "20100101"
    start_month: str = "201001"
    start_quarter: str = "2010Q1"
    month_param: str = "m"  # month_loop query param name (broker_recommend uses "month")
    # static_full registry pulls: one canonical file per loop value
    # (e.g. exchange=CFFEX.parquet), or a single full.parquet when empty.
    loop_param: str = ""
    loop_values: tuple[str, ...] = ()
    # Optional per-loop-value start dates (parallel to loop_values) for
    # trade_date loops whose venues listed at different times.
    loop_start_dates: tuple[str, ...] = ()

    def loop_start_date(self, value: str) -> str:
        if self.loop_start_dates:
            if len(self.loop_start_dates) != len(self.loop_values):
                raise ValueError(
                    f"{self.api_name}: loop_start_dates ({len(self.loop_start_dates)}) "
                    f"must parallel loop_values ({len(self.loop_values)})"
                )
            if value in self.loop_values:
                return self.loop_start_dates[self.loop_values.index(value)]
        return self.start_date

@dataclass(frozen=True)
class AvailabilityRule:
    """Official per-dataset PIT publication time.

    ``available_at`` is stamped as ``clock`` on the source date shifted by a
    calendar-day or trading-day offset, tagged with ``rule``.
    ``only_when_is_new`` restricts the rule to latest-snapshot pulls (request
    param is_new=Y); every other case falls back to conservative date EOD.
    """

    clock: str
    rule: str
    day_offset: int = 0
    trading_day_offset: int = 0
    only_when_is_new: bool = False

    def __post_init__(self) -> None:
        if self.day_offset < 0 or self.trading_day_offset < 0:
            raise ValueError("availability offsets cannot be negative")
        if self.day_offset and self.trading_day_offset:
            raise ValueError("availability rule cannot mix calendar-day and trading-day offsets")

OFFICIAL_NEXT_DAY_09 = AvailabilityRule("09:00:00", "official_next_day_09_from:trade_date", day_offset=1)
OFFICIAL_PREOPEN_09 = AvailabilityRule("09:00:00", "official_preopen_09_from:trade_date")
OFFICIAL_19 = AvailabilityRule("19:00:00", "official_19_from:trade_date")
OFFICIAL_NEXT_DAY_0830 = AvailabilityRule("08:30:00", "official_next_day_0830_from:trade_date", day_offset=1)
OFFICIAL_20 = AvailabilityRule("20:00:00", "official_20_from:trade_date")
OFFICIAL_LATEST_2230 = AvailabilityRule("22:30:00", "official_latest_2230_from:trade_date", only_when_is_new=True)
# Shared limit-list family rule (limit_list_d + limit_list_ths): both sources
# publish the day's full 涨跌停/炸板 list around 16:00 on the trade date.
OFFICIAL_16 = AvailabilityRule("16:00:00", "official_16_from:trade_date")

@dataclass
class EventDataset:
    api_name: str
    strategy: str
    key_columns: tuple[str, ...]
    fields: str
    page_limit: int
    date_column: str
    start_date: str = "20100101"
    fallback_date_column: str = ""
    zero_rows_ok: bool = True
    availability: AvailabilityRule | None = None

@dataclass
class BoardTradingDataset:
    api_name: str
    strategy: str
    key_columns: tuple[str, ...]
    fields: str
    page_limit: int
    date_column: str = "trade_date"
    time_column: str = ""
    start_date: str = "20200101"
    zero_rows_ok: bool = True
    availability: AvailabilityRule | None = None

FUNDAMENTAL_SPECS = {
    "income_vip": FundamentalDataset(
        api_name="income_vip",
        strategy="period",
        key_columns=("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    ),
    "balancesheet_vip": FundamentalDataset(
        api_name="balancesheet_vip",
        strategy="period",
        key_columns=("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    ),
    "cashflow_vip": FundamentalDataset(
        api_name="cashflow_vip",
        strategy="period",
        key_columns=("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
        # Measured per-call cap (probed 2026-08-10): a 7,000-row request
        # returns at most 6,400 rows, a silently truncated "short" page that
        # stops query_paged (23 periods were pinned at exactly 6,400). 6,000
        # pages safely; income_vip/balancesheet_vip paginate correctly at the
        # 7,000 tier default.
        page_limit=6000,
    ),
    "fina_indicator_vip": FundamentalDataset(
        api_name="fina_indicator_vip",
        strategy="period",
        key_columns=("ts_code", "ann_date", "end_date"),
    ),
    "forecast_vip": FundamentalDataset(
        api_name="forecast_vip",
        strategy="ann_month",
        key_columns=("ts_code", "ann_date", "end_date", "type", "first_ann_date", "update_flag"),
    ),
    "express_vip": FundamentalDataset(
        api_name="express_vip",
        strategy="ann_month",
        key_columns=("ts_code", "ann_date", "end_date"),
    ),
    "dividend": FundamentalDataset(
        api_name="dividend",
        strategy="ts_code",
        key_columns=("ts_code", "end_date", "ann_date", "div_proc", "record_date", "ex_date", "pay_date"),
    ),
    "fina_audit": FundamentalDataset(
        api_name="fina_audit",
        strategy="ts_code",
        key_columns=("ts_code", "ann_date", "end_date"),
    ),
    "fina_mainbz_vip": FundamentalDataset(
        api_name="fina_mainbz_vip",
        strategy="ts_code",
        key_columns=("ts_code", "end_date", "bz_item", "bz_code", "curr_type"),
    ),
    "disclosure_date": FundamentalDataset(
        api_name="disclosure_date",
        strategy="period",
        key_columns=("ts_code", "end_date", "ann_date", "pre_date", "actual_date"),
        period_param="end_date",
    ),
}

TEXT_SPECS = {
    "anns_d": TextDataset(
        api_name="anns_d",
        strategy="range_month",
        fields="ann_date,ts_code,name,title,url,rec_time",
        page_limit=2000,
        key_columns=("ann_date", "ts_code", "title", "url"),
        start_date="20100101",
        time_column="rec_time",
        date_column="ann_date",
    ),
    "major_news": TextDataset(
        api_name="major_news",
        strategy="time_range_month",
        fields="title,pub_time,src,content",
        page_limit=400,
        key_columns=("pub_time", "src", "title"),
        start_date="20170101",
        time_column="pub_time",
    ),
    "cctv_news": TextDataset(
        api_name="cctv_news",
        strategy="day",
        fields="date,title,content",
        key_columns=("date", "title"),
        start_date="20170101",
        date_column="date",
    ),
    "npr": TextDataset(
        api_name="npr",
        strategy="range_month",
        fields="pubtime,title,pcode,puborg,ptype,url,content_html",
        page_limit=500,
        key_columns=("pubtime", "pcode", "title"),
        start_date="20100101",
        time_column="pubtime",
    ),
    "research_report": TextDataset(
        api_name="research_report",
        strategy="range_month",
        fields="trade_date,abstr,title,report_type,author,name,ts_code,inst_csname,ind_name,url",
        page_limit=1000,
        key_columns=("trade_date", "report_type", "title", "inst_csname", "ts_code"),
        start_date="20170101",
        date_column="trade_date",
    ),
    "report_rc": TextDataset(
        api_name="report_rc",
        strategy="range_month",
        fields="ts_code,name,report_date,report_title,report_type,classify,org_name,author_name,quarter,op_rt,op_pr,tp,np,eps,pe,rd,roe,ev_ebitda,rating,max_price,min_price,imp_dg,create_time",
        page_limit=3000,
        key_columns=("ts_code", "report_date", "report_title", "org_name", "author_name", "quarter"),
        start_date="20100101",
        time_column="create_time",
        date_column="report_date",
    ),
    "irm_qa_sh": TextDataset(
        api_name="irm_qa_sh",
        strategy="day",
        fields="ts_code,name,trade_date,q,a,pub_time",
        key_columns=("trade_date", "ts_code", "q"),
        start_date="20220101",
        time_column="pub_time",
        date_column="trade_date",
    ),
    "irm_qa_sz": TextDataset(
        api_name="irm_qa_sz",
        strategy="day",
        fields="ts_code,name,trade_date,q,a,pub_time,industry",
        key_columns=("trade_date", "ts_code", "q"),
        start_date="20220101",
        time_column="pub_time",
        date_column="trade_date",
    ),
    "news": TextDataset(
        api_name="news",
        strategy="news_src_day",
        fields="datetime,content,title,channels",
        page_limit=1500,
        key_columns=("datetime", "title"),
        start_date="20180101",
        time_column="datetime",
    ),
}

MACRO_SPECS = {
    "cn_schedule": MacroDataset(
        api_name="cn_schedule",
        strategy="month_loop",
        fields="month,publish_date,title,issuing_org,data_api",
        # The source keeps no schedule rows before 2026 (observed rolling
        # retention, not a declared floor): months 202001-202512 always
        # return empty, so the floor stops re-querying and expecting them.
        start_date="20260101",
        key_columns=("month", "publish_date", "title", "issuing_org", "data_api"),
        date_column="publish_date",
    ),
    "cn_gdp": MacroDataset(
        api_name="cn_gdp",
        strategy="quarter_once",
        fields="quarter,gdp,gdp_yoy,pi,pi_yoy,si,si_yoy,ti,ti_yoy",
        key_columns=("quarter",),
        date_column="quarter",
        start_quarter="2010Q1",
    ),
    "cn_cpi": MacroDataset(
        api_name="cn_cpi",
        strategy="month_once",
        fields="month,nt_val,nt_yoy,nt_mom,nt_accu,town_val,town_yoy,town_mom,town_accu,cnt_val,cnt_yoy,cnt_mom,cnt_accu",
        key_columns=("month",),
        date_column="month",
    ),
    "cn_ppi": MacroDataset(
        api_name="cn_ppi",
        strategy="month_once",
        fields="month,ppi_yoy,ppi_mp_yoy,ppi_mp_qm_yoy,ppi_mp_rm_yoy,ppi_mp_p_yoy,ppi_cg_yoy,ppi_cg_f_yoy,ppi_cg_c_yoy,ppi_cg_adu_yoy,ppi_cg_dcg_yoy,ppi_mom,ppi_mp_mom,ppi_mp_qm_mom,ppi_mp_rm_mom,ppi_mp_p_mom,ppi_cg_mom,ppi_cg_f_mom,ppi_cg_c_mom,ppi_cg_adu_mom,ppi_cg_dcg_mom,ppi_accu,ppi_mp_accu,ppi_mp_qm_accu,ppi_mp_rm_accu,ppi_mp_p_accu,ppi_cg_accu,ppi_cg_f_accu,ppi_cg_c_accu,ppi_cg_adu_accu,ppi_cg_dcg_accu",
        key_columns=("month",),
        date_column="month",
    ),
    "cn_pmi": MacroDataset(
        api_name="cn_pmi",
        strategy="month_once",
        fields="month,pmi010000,pmi010400,pmi010500,pmi010900,pmi011000,pmi011200,pmi011300,pmi011600,pmi020100,pmi020200,pmi020300,pmi020400,pmi020600,pmi030000",
        key_columns=("month",),
        date_column="month",
    ),
    "cn_m": MacroDataset(
        api_name="cn_m",
        strategy="month_once",
        fields="month,m0,m0_yoy,m0_mom,m1,m1_yoy,m1_mom,m2,m2_yoy,m2_mom",
        key_columns=("month",),
        date_column="month",
    ),
    "sf_month": MacroDataset(
        api_name="sf_month",
        strategy="month_once",
        fields="month,inc_month,inc_cumval,stk_endval",
        key_columns=("month",),
        date_column="month",
    ),
    "shibor": MacroDataset(
        api_name="shibor",
        strategy="date_year",
        fields="date,on,1w,2w,1m,3m,6m,9m,1y",
        key_columns=("date",),
        date_column="date",
    ),
    "shibor_quote": MacroDataset(
        api_name="shibor_quote",
        strategy="date_year",
        fields="date,bank,on_b,on_a,1w_b,1w_a,2w_b,2w_a,1m_b,1m_a,3m_b,3m_a,6m_b,6m_a,9m_b,9m_a,1y_b,1y_a",
        # Measured per-call cap; a larger request returns one silently
        # truncated "short" page and query_paged stops after it.
        page_limit=4000,
        key_columns=("date", "bank"),
        date_column="date",
    ),
    "shibor_lpr": MacroDataset(
        api_name="shibor_lpr",
        strategy="date_year",
        fields="date,1y,5y",
        key_columns=("date",),
        date_column="date",
    ),
    "repo_daily": MacroDataset(
        api_name="repo_daily",
        strategy="date_year",
        fields="ts_code,trade_date,repo_maturity,pre_close,open,high,low,close,weight,weight_r,amount,num",
        # Measured per-call cap (see shibor_quote note).
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "us_tycr": MacroDataset(
        api_name="us_tycr",
        strategy="date_year",
        # m4 (the 4-month CMT) has never been returned by the vendor for any
        # year in the lake; requesting it only trips the schema audit.
        fields="date,m1,m2,m3,m6,y1,y2,y3,y5,y7,y10,y20,y30",
        key_columns=("date",),
        date_column="date",
    ),
    "us_trycr": MacroDataset(
        api_name="us_trycr",
        strategy="date_year",
        fields="date,y5,y7,y10,y20,y30",
        key_columns=("date",),
        date_column="date",
    ),
    "us_tbr": MacroDataset(
        api_name="us_tbr",
        strategy="date_year",
        # w17_bd/w17_ce (the 17-week tenor) have never been returned by the
        # vendor for any year in the lake.
        fields="date,w4_bd,w4_ce,w8_bd,w8_ce,w13_bd,w13_ce,w26_bd,w26_ce,w52_bd,w52_ce",
        key_columns=("date",),
        date_column="date",
    ),
    "us_tltr": MacroDataset(
        api_name="us_tltr",
        strategy="date_year",
        fields="date,ltc,cmt,e_factor",
        key_columns=("date",),
        date_column="date",
    ),
    "index_daily": MacroDataset(
        api_name="index_daily",
        strategy="date_year_by_ts_code",
        fields="ts_code,trade_date,open,close,high,low,pre_close,change,pct_chg,vol,amount",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "broker_recommend": MacroDataset(
        api_name="broker_recommend",
        strategy="month_loop",
        fields="month,broker,ts_code,name",
        start_date="20190101",
        key_columns=("month", "broker", "ts_code"),
        date_column="month",
        month_param="month",
    ),
    "ths_daily": MacroDataset(
        api_name="ths_daily",
        strategy="trade_date",
        fields="ts_code,trade_date,open,high,low,close,pre_close,avg_price,change,pct_change,vol,turnover_rate",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "index_dailybasic": MacroDataset(
        api_name="index_dailybasic",
        strategy="date_year_by_ts_code",
        fields="ts_code,trade_date,total_mv,float_mv,total_share,float_share,free_share,turnover_rate,turnover_rate_f,pe,pe_ttm,pb",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "sw_daily": MacroDataset(
        api_name="sw_daily",
        strategy="trade_date",
        fields="ts_code,trade_date,name,open,low,high,close,change,pct_change,vol,amount,pe,pb,float_mv,total_mv",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "ci_daily": MacroDataset(
        api_name="ci_daily",
        strategy="trade_date",
        fields="ts_code,trade_date,open,low,high,close,pre_close,change,pct_change,vol,amount",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "daily_info": MacroDataset(
        api_name="daily_info",
        strategy="date_year",
        fields="trade_date,ts_code,ts_name,com_count,total_share,float_share,total_mv,float_mv,amount,vol,trans_count,pe,tr,exchange",
        start_date="20190101",
        # Measured per-call cap (see shibor_quote note).
        page_limit=4000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "sz_daily_info": MacroDataset(
        api_name="sz_daily_info",
        strategy="date_year",
        fields="trade_date,ts_code,count,amount,vol,total_share,total_mv,float_share,float_mv",
        start_date="20190101",
        # Measured per-call cap (see shibor_quote note).
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "moneyflow_mkt_dc": MacroDataset(
        api_name="moneyflow_mkt_dc",
        strategy="date_year",
        fields="trade_date,close_sh,pct_change_sh,close_sz,pct_change_sz,net_amount,net_amount_rate,buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate",
        start_date="20190101",
        key_columns=("trade_date",),
        date_column="trade_date",
    ),
    "index_global": MacroDataset(
        api_name="index_global",
        strategy="date_year_by_ts_code",
        fields="ts_code,trade_date,open,close,high,low,pre_close,change,pct_chg,swing,vol,amount",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "fx_daily": MacroDataset(
        api_name="fx_daily",
        strategy="date_year_by_ts_code",
        fields="ts_code,trade_date,bid_open,bid_close,bid_high,bid_low,ask_open,ask_close,ask_high,ask_low,tick_qty",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "eco_cal": MacroDataset(
        api_name="eco_cal",
        strategy="eco_cal_month",
        fields="date,time,currency,country,event,value,pre_value,fore_value",
        # The API returns at most 100 rows per call regardless of the
        # requested limit; a larger page size makes query_paged stop after
        # one silently truncated page.
        page_limit=100,
        key_columns=("date", "time", "currency", "country", "event"),
        date_column="date",
        time_column="time",
    ),
    # ---- derivatives market context (non-tradable; the Agent computes signals
    # such as index-futures basis, option PCR/IV and CB conversion premium).
    # Daily tables ride the trade_date strategy and the evening Timeview node;
    # registries are small static_full re-pulls with PIT rows keyed by
    # list/announcement dates. ----
    "fut_basic": MacroDataset(
        api_name="fut_basic",
        strategy="static_full",
        fields="ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,d_month,list_date,delist_date,last_ddate,trade_time_desc",
        page_limit=10000,
        key_columns=("ts_code",),
        date_column="list_date",
        loop_param="exchange",
        loop_values=("CFFEX", "DCE", "CZCE", "SHFE", "INE", "GFEX"),
    ),
    "fut_mapping": MacroDataset(
        api_name="fut_mapping",
        strategy="trade_date",
        fields="ts_code,trade_date,mapping_ts_code",
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "fut_daily": MacroDataset(
        api_name="fut_daily",
        strategy="trade_date",
        fields="ts_code,trade_date,pre_close,pre_settle,open,high,low,close,settle,change1,change2,vol,amount,oi,oi_chg,delv_settle",
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "opt_basic": MacroDataset(
        api_name="opt_basic",
        strategy="static_full",
        fields="ts_code,symbol,exchange,name,per_unit,opt_code,opt_type,call_put,exercise_type,exercise_price,s_month,maturity_date,list_price,list_date,delist_date,last_edate,last_ddate,quote_unit,min_price_chg",
        page_limit=10000,
        key_columns=("ts_code",),
        date_column="list_date",
        # Financial exchanges only, matching the opt_daily scope: commodity
        # option registries add ~185k dead rows to every snapshot macro frame.
        loop_param="exchange",
        loop_values=("SSE", "SZSE", "CFFEX"),
    ),
    "opt_daily": MacroDataset(
        api_name="opt_daily",
        strategy="trade_date",
        fields="ts_code,trade_date,exchange,pre_settle,pre_close,open,high,low,close,settle,vol,amount,oi",
        page_limit=15000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20150209",
        # Financial options only (ETF + index): the whole-market pull is ~27k
        # rows/day of mostly commodity options with no equity signal, which
        # would put a multi-million-row year window into every snapshot.
        # SZSE 300ETF options and CFFEX index options both listed 2019-12-23.
        loop_param="exchange",
        loop_values=("SSE", "SZSE", "CFFEX"),
        loop_start_dates=("20150209", "20191223", "20191223"),
    ),
    "cb_basic": MacroDataset(
        api_name="cb_basic",
        strategy="static_full",
        # conv_price/clauses are the fields the conversion-premium and
        # redemption-event signals need; stk_code links CB -> underlying stock.
        fields="ts_code,bond_short_name,cb_code,cb_type,stk_code,stk_short_name,maturity,par,issue_price,issue_size,remain_size,value_date,maturity_date,rate_type,coupon_rate,pay_per_year,list_date,delist_date,exchange,conv_start_date,conv_end_date,first_conv_price,conv_price,put_clause,maturity_call_price,call_clause,reset_clause,guarantor,issue_rating,newest_rating,rating_comp",
        page_limit=2000,
        key_columns=("ts_code",),
        date_column="list_date",
    ),
    "cb_daily": MacroDataset(
        api_name="cb_daily",
        # bond_value/cb_value/cb_over_rate are non-default display fields and
        # must be requested explicitly.
        strategy="trade_date",
        fields="ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,bond_value,bond_over_rate,cb_value,cb_over_rate",
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20180102",
    ),
    "cb_call": MacroDataset(
        api_name="cb_call",
        strategy="static_full",
        fields="ts_code,call_type,is_call,ann_date,call_date,call_price,call_price_tax,call_vol,call_amount,payment_date,call_reg_date",
        page_limit=2000,
        # The staged is_call states of one bond are separate announcements.
        key_columns=("ts_code", "ann_date", "is_call"),
        date_column="ann_date",
    ),
}

EVENT_FLOW_SPECS = {
    "margin": EventDataset(
        api_name="margin",
        strategy="trade_date",
        fields="trade_date,exchange_id,rzye,rzmre,rzche,rqye,rqmcl,rzrqye,rqyl",
        page_limit=4000,
        key_columns=("trade_date", "exchange_id"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_NEXT_DAY_09,
    ),
    "margin_detail": EventDataset(
        api_name="margin_detail",
        strategy="trade_date",
        fields="trade_date,ts_code,name,rzye,rqye,rzmre,rqyl,rzche,rqchl,rqmcl,rzrqye",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_NEXT_DAY_09,
    ),
    "margin_secs": EventDataset(
        api_name="margin_secs",
        strategy="trade_date",
        fields="trade_date,ts_code,name,exchange",
        page_limit=6000,
        key_columns=("trade_date", "ts_code", "exchange"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_PREOPEN_09,
    ),
    "moneyflow": EventDataset(
        api_name="moneyflow",
        strategy="trade_date",
        fields="ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_19,
    ),
    "moneyflow_dc": EventDataset(
        api_name="moneyflow_dc",
        strategy="trade_date",
        fields="trade_date,ts_code,name,pct_change,close,net_amount,net_amount_rate,buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20231201",
        availability=OFFICIAL_19,
    ),
    "moneyflow_ths": EventDataset(
        api_name="moneyflow_ths",
        strategy="trade_date",
        fields="trade_date,ts_code,name,pct_change,latest,net_amount,net_d5_amount,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20250101",
        availability=OFFICIAL_19,
    ),
    "moneyflow_ind_dc": EventDataset(
        api_name="moneyflow_ind_dc",
        strategy="trade_date",
        fields="trade_date,content_type,ts_code,name,pct_change,close,net_amount,net_amount_rate,buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate,buy_sm_amount_stock,rank",
        page_limit=5000,
        key_columns=("trade_date", "content_type", "ts_code"),
        date_column="trade_date",
        start_date="20231201",
        availability=OFFICIAL_19,
    ),
    "moneyflow_ind_ths": EventDataset(
        api_name="moneyflow_ind_ths",
        strategy="trade_date",
        fields="trade_date,ts_code,industry,lead_stock,close,pct_change,company_num,pct_change_stock,close_price,net_buy_amount,net_sell_amount,net_amount",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20250101",
        availability=OFFICIAL_19,
    ),
    "moneyflow_cnt_ths": EventDataset(
        api_name="moneyflow_cnt_ths",
        strategy="trade_date",
        fields="trade_date,ts_code,name,lead_stock,close_price,pct_change,industry_index,company_num,pct_change_stock,net_buy_amount,net_sell_amount,net_amount",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20250101",
        availability=OFFICIAL_19,
    ),
    "cyq_perf": EventDataset(
        api_name="cyq_perf",
        strategy="trade_date",
        fields="ts_code,trade_date,his_low,his_high,cost_5pct,cost_15pct,cost_50pct,cost_85pct,cost_95pct,weight_avg,winner_rate",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20180102",
        availability=OFFICIAL_19,
    ),
    "bak_daily": EventDataset(
        api_name="bak_daily",
        strategy="trade_date",
        fields="ts_code,trade_date,name,pct_change,close,change,open,high,low,pre_close,vol_ratio,turn_over,swing,vol,amount,selling,buying,total_share,float_share,pe,industry,area,float_mv,total_mv,avg_price,strength,activity,avg_turnover,attack,interval_3,interval_6",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20170103",
        availability=OFFICIAL_19,
    ),
    "stk_holdernumber": EventDataset(
        api_name="stk_holdernumber",
        strategy="range_month",
        fields="ts_code,ann_date,end_date,holder_num",
        page_limit=3000,
        key_columns=("ts_code", "ann_date", "end_date"),
        date_column="ann_date",
        zero_rows_ok=True,
    ),
    "top10_holders": EventDataset(
        api_name="top10_holders",
        strategy="range_month",
        fields="ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio,hold_float_ratio,hold_change,holder_type",
        page_limit=5000,
        key_columns=("ts_code", "ann_date", "end_date", "holder_name"),
        date_column="ann_date",
    ),
    "top10_floatholders": EventDataset(
        api_name="top10_floatholders",
        strategy="range_month",
        fields="ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio,hold_float_ratio,hold_change,holder_type",
        page_limit=5000,
        key_columns=("ts_code", "ann_date", "end_date", "holder_name"),
        date_column="ann_date",
    ),
    "pledge_detail": EventDataset(
        api_name="pledge_detail",
        strategy="range_month",
        fields="ts_code,ann_date,holder_name,pledge_amount,start_date,end_date,is_release,release_date,pledgor,holding_amount,pledged_amount,p_total_ratio,h_total_ratio,is_buyback",
        # Measured per-call cap (probed 2026-08-10): a 3,000-row request
        # returns at most 1,500 rows, a silently truncated "short" page that
        # stops query_paged (42/80 month partitions were pinned at 1,500).
        page_limit=1500,
        key_columns=("ts_code", "ann_date", "holder_name", "start_date", "pledgor"),
        date_column="ann_date",
    ),
    "stk_surv": EventDataset(
        api_name="stk_surv",
        strategy="trade_date",
        fields="ts_code,name,surv_date,fund_visitors,rece_place,rece_mode,rece_org,org_type,comp_rece",
        # Measured per-call cap (probed 2026-08-10): a 3,000-row request
        # returns at most 400 rows, a silently truncated "short" page that
        # stops query_paged; offset paging past 400 works (624/1113 partitions
        # were pinned at exactly 400).
        page_limit=400,
        key_columns=("ts_code", "surv_date", "rece_org"),
        date_column="surv_date",
        start_date="20220101",
        # The source has no announcement column and rows keep landing for
        # several days after surv_date (disclosure is due within 2 trading
        # days), so surv_date EOD is lookahead. +5 calendar days covers the
        # 2-trading-day window across weekends and ordinary holidays.
        availability=AvailabilityRule("23:59:59", "conservative_plus_5d_eod_from:surv_date", day_offset=5),
    ),
    "new_share": EventDataset(
        api_name="new_share",
        strategy="range_month",
        fields="ts_code,sub_code,name,ipo_date,issue_date,amount,market_amount,price,pe,limit_amount,funds,ballot",
        page_limit=2000,
        key_columns=("ts_code",),
        date_column="ipo_date",
        # ballot (中签率) is announced 1-2 trading days after ipo_date. Use the
        # second following A-share trading day rather than a calendar-day
        # approximation, which leaks across National Day and Spring Festival.
        availability=AvailabilityRule(
            "23:59:59",
            "conservative_tplus_2_eod_from:ipo_date",
            trading_day_offset=2,
        ),
    ),
    "stk_holdertrade": EventDataset(
        api_name="stk_holdertrade",
        strategy="range_month",
        fields="ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio,after_share,after_ratio,avg_price,total_share,begin_date,close_date",
        page_limit=3000,
        key_columns=("ts_code", "ann_date", "holder_name", "holder_type", "in_de", "begin_date", "close_date"),
        date_column="ann_date",
        zero_rows_ok=True,
        availability=AvailabilityRule("19:00:00", "official_19_from:ann_date"),
    ),
    "repurchase": EventDataset(
        api_name="repurchase",
        strategy="range_month",
        fields="ts_code,ann_date,end_date,proc,exp_date,vol,amount,high_limit,low_limit",
        page_limit=2000,
        key_columns=("ts_code", "ann_date", "end_date", "proc", "exp_date"),
        date_column="ann_date",
        start_date="20110101",
        zero_rows_ok=True,
    ),
    "share_float": EventDataset(
        api_name="share_float",
        strategy="day",
        fields=SHARE_FLOAT_FIELDS,
        page_limit=6000,
        key_columns=("ts_code", "ann_date", "float_date", "holder_name", "share_type"),
        date_column="ann_date",
        fallback_date_column="float_date",
        zero_rows_ok=True,
    ),
    "block_trade": EventDataset(
        api_name="block_trade",
        strategy="trade_date",
        fields="ts_code,trade_date,price,vol,amount,buyer,seller",
        page_limit=1000,
        key_columns=("trade_date", "ts_code", "price", "vol", "amount", "buyer", "seller"),
        date_column="trade_date",
        zero_rows_ok=True,
        availability=AvailabilityRule("21:00:00", "official_21_from:trade_date"),
    ),
}

BOARD_TRADING_SPECS = {
    "kpl_list": BoardTradingDataset(
        api_name="kpl_list",
        strategy="trade_date_by_tag",
        fields="ts_code,name,trade_date,lu_time,ld_time,open_time,last_time,lu_desc,tag,theme,net_change,bid_amount,status,bid_change,bid_turnover,lu_bid_vol,pct_chg,bid_pct_chg,rt_pct_chg,limit_order,amount,turnover_rate,free_float,lu_limit_order",
        page_limit=8000,
        key_columns=("trade_date", "ts_code", "tag", "status", "lu_time", "open_time", "last_time"),
        availability=OFFICIAL_NEXT_DAY_0830,
    ),
    "kpl_concept_cons": BoardTradingDataset(
        api_name="kpl_concept_cons",
        strategy="trade_date",
        fields="ts_code,name,con_name,con_code,trade_date,desc,hot_num",
        page_limit=3000,
        key_columns=("trade_date", "con_code", "ts_code"),
        start_date="20250101",
        availability=OFFICIAL_NEXT_DAY_0830,
    ),
    "dc_index": BoardTradingDataset(
        api_name="dc_index",
        strategy="trade_date",
        fields="ts_code,trade_date,name,leading,leading_code,pct_change,leading_pct,total_mv,turnover_rate,up_num,down_num,idx_type,level",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        start_date="20250101",
        availability=OFFICIAL_20,
    ),
    "dc_member": BoardTradingDataset(
        api_name="dc_member",
        strategy="trade_date",
        fields="trade_date,ts_code,con_code,name",
        page_limit=8000,
        key_columns=("trade_date", "ts_code", "con_code"),
        start_date="20250101",
        availability=OFFICIAL_20,
    ),
    "limit_step": BoardTradingDataset(
        api_name="limit_step",
        strategy="trade_date",
        fields="ts_code,name,trade_date,nums",
        page_limit=2000,
        key_columns=("trade_date", "ts_code", "nums"),
    ),
    "limit_cpt_list": BoardTradingDataset(
        api_name="limit_cpt_list",
        strategy="trade_date",
        fields="ts_code,name,trade_date,days,up_stat,cons_nums,up_nums,pct_chg,rank",
        page_limit=2000,
        key_columns=("trade_date", "ts_code", "rank"),
    ),
    "limit_list_d": BoardTradingDataset(
        api_name="limit_list_d",
        strategy="trade_date",
        fields="trade_date,ts_code,industry,name,close,pct_chg,amount,limit_amount,float_mv,total_mv,turnover_ratio,fd_amount,first_time,last_time,open_times,up_stat,limit_times,limit",
        page_limit=5000,
        key_columns=("trade_date", "ts_code", "limit"),
        availability=OFFICIAL_16,
    ),
    "limit_list_ths": BoardTradingDataset(
        api_name="limit_list_ths",
        strategy="trade_date_by_limit_type",
        fields="trade_date,ts_code,name,price,pct_chg,open_num,lu_desc,limit_type,tag,status,limit_order,limit_amount,turnover_rate,free_float,lu_limit_order,limit_up_suc_rate,turnover,market_type,rise_rate,sum_float,first_lu_time,last_lu_time,first_ld_time,last_ld_time",
        page_limit=8000,
        key_columns=("trade_date", "ts_code", "limit_type", "tag", "status", "first_lu_time", "last_lu_time", "first_ld_time", "last_ld_time"),
        start_date="20231101",
        availability=OFFICIAL_16,
    ),
    "top_list": BoardTradingDataset(
        api_name="top_list",
        strategy="trade_date",
        fields="trade_date,ts_code,name,close,pct_change,turnover_rate,amount,l_sell,l_buy,l_amount,net_amount,net_rate,amount_rate,float_values,reason",
        page_limit=10000,
        key_columns=(
            "trade_date",
            "ts_code",
            "name",
            "close",
            "pct_change",
            "turnover_rate",
            "amount",
            "l_sell",
            "l_buy",
            "l_amount",
            "net_amount",
            "net_rate",
            "amount_rate",
            "float_values",
            "reason",
        ),
        zero_rows_ok=True,
        availability=OFFICIAL_20,
    ),
    "top_inst": BoardTradingDataset(
        api_name="top_inst",
        strategy="trade_date",
        fields="trade_date,ts_code,exalter,buy,buy_rate,sell,sell_rate,net_buy,side,reason",
        page_limit=10000,
        key_columns=("trade_date", "ts_code", "exalter", "buy", "sell", "net_buy", "side", "reason"),
        zero_rows_ok=True,
        availability=OFFICIAL_20,
    ),
    "hm_list": BoardTradingDataset(
        api_name="hm_list",
        strategy="static_full",
        fields="name,desc,orgs",
        page_limit=1000,
        key_columns=("name",),
        date_column="",
        start_date="20220801",
    ),
    "hm_detail": BoardTradingDataset(
        api_name="hm_detail",
        strategy="trade_date",
        fields="trade_date,ts_code,ts_name,buy_amount,sell_amount,net_amount,hm_name,hm_orgs",
        page_limit=10000,
        key_columns=("trade_date", "ts_code", "hm_name", "hm_orgs"),
        start_date="20220801",
    ),
    "ths_hot": BoardTradingDataset(
        api_name="ths_hot",
        strategy="trade_date_by_market",
        fields="trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,hot,concept,rank_time,rank_reason",
        page_limit=2000,
        key_columns=("trade_date", "data_type", "ts_code", "rank_time", "rank"),
        time_column="rank_time",
        availability=OFFICIAL_LATEST_2230,
    ),
    "dc_hot": BoardTradingDataset(
        api_name="dc_hot",
        strategy="trade_date_by_market_hot_type",
        fields="trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,hot,concept,rank_time",
        page_limit=2000,
        key_columns=("trade_date", "data_type", "ts_code", "rank_time", "rank"),
        time_column="rank_time",
        availability=OFFICIAL_LATEST_2230,
    ),
}

BAK_BASIC_SPEC = TradeDateDataset(
    api_name="bak_basic",
    fields=BAK_BASIC_FIELDS,
    start_date="20160101",
    zero_rows_ok=True,
    key_columns=("trade_date", "ts_code"),
)

DAILY_SPECS = {
    "daily": TradeDateDataset(
        api_name="daily",
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    ),
    "adj_factor": TradeDateDataset(
        api_name="adj_factor",
        fields="ts_code,trade_date,adj_factor",
    ),
    "daily_basic": TradeDateDataset(
        api_name="daily_basic",
        fields="ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv",
    ),
    "stk_limit": TradeDateDataset(
        api_name="stk_limit",
        fields="trade_date,ts_code,pre_close,up_limit,down_limit",
    ),
    "suspend_d": TradeDateDataset(
        api_name="suspend_d",
        fields="ts_code,trade_date,suspend_timing,suspend_type",
        zero_rows_ok=True,
        key_columns=("trade_date", "ts_code", "suspend_type", "suspend_timing"),
    ),
    # Exact opening call-auction result. TuShare's usable history begins on
    # 2025-01-16; older replays retain the explicitly labelled minute proxy.
    "stk_auction": TradeDateDataset(
        api_name="stk_auction",
        fields="ts_code,trade_date,vol,price,amount,pre_close,turnover_rate,volume_ratio,float_share",
        start_date="20250116",
        dedup_exact=True,
    ),
}

class TuShareClient:
    """TuShare SDK adapter whose HTTP transport is pinned to the configured relay."""

    def __init__(
        self,
        token: str,
        min_interval: float = 0.2,
        timeout: int = 60,
        relay_url: str | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("TuShare timeout must be a positive integer number of seconds")
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError(
                "the tushare SDK is not installed; install the project's declared dependencies"
            ) from exc
        self.pro = ts.pro_api(token, timeout=timeout)
        # Relay mode: the endpoint is pinned by writing the SDK's name-mangled
        # private attribute. Assigning it always "succeeds", so an SDK rename
        # would silently create a dead attribute and send every request to the
        # official endpoint instead. Require the attribute to exist first, and
        # refuse to start rather than route traffic out of relay mode.
        endpoint = (relay_url or load_relay_url()).rstrip("/")
        if not hasattr(self.pro, "_DataApi__http_url"):
            raise RuntimeError(
                "the tushare SDK no longer exposes _DataApi__http_url; the relay endpoint "
                f"cannot be pinned to {endpoint} and requests would leave relay mode"
            )
        self.pro._DataApi__http_url = endpoint
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.last_call = 0.0

    def query(self, api_name: str, params: dict[str, Any] | None = None, fields: str = "", retries: int = 5) -> ApiResult:
        for attempt in range(1, retries + 1):
            self._throttle()
            try:
                result_frame = self.pro.query(api_name, fields=fields, **(params or {}))
            except Exception as exc:
                if attempt == retries:
                    raise RuntimeError(f"{api_name} failed after {retries} attempts: {exc}") from exc
                time.sleep(2 * attempt)
                continue
            if not isinstance(result_frame, pd.DataFrame):
                raise TypeError(
                    f"TuShare {api_name} returned {type(result_frame).__name__}, expected DataFrame"
                )
            return ApiResult(list(result_frame.columns), result_frame.to_numpy().tolist())
        raise RuntimeError(f"{api_name} failed unexpectedly")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.monotonic()

def load_token(repo_root: Path) -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    env_file = repo_root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token:
                    return token
    raise RuntimeError("TUSHARE_TOKEN is not set in the environment or ignored .env")


def load_relay_url() -> str:
    return os.environ.get("TUSHARE_RELAY_URL", DEFAULT_TUSHARE_RELAY_URL).strip() or DEFAULT_TUSHARE_RELAY_URL

def frame(result: ApiResult) -> pd.DataFrame:
    return pd.DataFrame(result.items, columns=result.fields)

def query_paged(client: TuShareClient, api_name: str, params: dict[str, Any], fields: str = "", page_limit: int | None = 10000, allow_repeated_pages: bool = False) -> tuple[ApiResult, int]:
    page_limit = page_limit or 10000
    all_items: list[list[Any]] = []
    result_fields: list[str] = []
    page_payloads: set[str] = set()
    pages = 0
    offset = 0
    while True:
        page_params = dict(params)
        page_params.update({"limit": page_limit, "offset": offset})
        result = client.query(api_name, page_params, fields)
        if not result_fields:
            result_fields = result.fields
        elif result.fields != result_fields:
            raise RuntimeError(f"{api_name} returned inconsistent fields while paging")
        page_payload = json.dumps(
            {"fields": result.fields, "items": result.items},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if result.items and page_payload in page_payloads and not allow_repeated_pages:
            # A repeated full page normally means broken pagination. Sources
            # that serve heavily duplicated rows (eco_cal repeats each event
            # many times) legitimately produce identical pages; callers that
            # dedup by key opt out and rely on the offset safety limit below.
            raise RuntimeError(
                f"{api_name} returned a repeated page while paging params={params}; "
                f"offset={offset} page_limit={page_limit}"
            )
        page_payloads.add(page_payload)
        all_items.extend(result.items)
        pages += 1
        if len(result.items) < page_limit:
            break
        offset += page_limit
        if offset > 500000:
            raise RuntimeError(f"{api_name} pagination exceeded safety limit for params={params}")
    return ApiResult(result_fields, all_items), pages

def canonical_revision_value(value: Any) -> str:
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

REVISION_CHANGED_KEY_SAMPLE_SIZE = 5
REVISION_CHANGED_COLUMN_SAMPLE_SIZE = 12
REVISION_ADDED_REMOVED_ROW_SAMPLE_SIZE = 5


def compare_keyed_frames(old_df: pd.DataFrame, new_df: pd.DataFrame, key_columns: list[str]) -> dict[str, Any]:
    keys = list(key_columns)
    missing_old = [column for column in keys if column not in old_df.columns]
    missing_new = [column for column in keys if column not in new_df.columns]
    old_columns = sorted(str(column) for column in old_df.columns)
    new_columns = sorted(str(column) for column in new_df.columns)
    # The column SET is compared explicitly and carried on the event. A keyed
    # value diff cannot imply it: both frames are padded to the union before
    # comparison, so an added/dropped empty column reads as identical content.
    # It is also what makes two structurally identical diffs over different
    # schemas distinct records in the append-once ledger.
    schema_changed = old_columns != new_columns
    base: dict[str, Any] = {
        "key_columns": keys,
        "old_rows": int(len(old_df)),
        "new_rows": int(len(new_df)),
        "old_columns": old_columns,
        "new_columns": new_columns,
        "schema_changed": schema_changed,
        "missing_key_columns_old": missing_old,
        "missing_key_columns_new": missing_new,
        "duplicate_key_rows_old": 0,
        "duplicate_key_rows_new": 0,
        "comparison_issue": "",
    }
    if missing_old or missing_new:
        return {
            **base,
            "changed": True,
            "comparison_issue": "missing_key_columns",
            "changed_keys": [],
            "added_keys": [],
            "removed_keys": [],
            "changed_columns": {},
            "changed_columns_sample": [],
            "added_rows_sample": [],
            "removed_rows_sample": [],
        }
    duplicate_old = int(old_df.duplicated(keys).sum()) if not old_df.empty else 0
    duplicate_new = int(new_df.duplicated(keys).sum()) if not new_df.empty else 0
    base["duplicate_key_rows_old"] = duplicate_old
    base["duplicate_key_rows_new"] = duplicate_new
    if duplicate_old or duplicate_new:
        # Duplicate business keys prevent a keyed diff, but identical content
        # is still decidable — and identical content is not a revision. Without
        # this, every re-pull of a dup-key partition (dividend, news, ...)
        # logged a false "revised" event on unchanged data.
        return {
            **base,
            "changed": schema_changed or not frames_content_equal(old_df, new_df),
            "comparison_issue": "duplicate_key_rows",
            "changed_keys": [],
            "added_keys": [],
            "removed_keys": [],
            "changed_columns": {},
            "changed_columns_sample": [],
            "added_rows_sample": [],
            "removed_rows_sample": [],
        }
    value_columns = sorted((set(old_df.columns) | set(new_df.columns)) - set(keys))

    def keyed_rows(df: pd.DataFrame) -> dict[tuple[str, ...], dict[str, str]]:
        if df.empty:
            return {}
        normalized = df.copy()
        for column in keys + value_columns:
            if column not in normalized.columns:
                normalized[column] = ""
        normalized = normalized[keys + value_columns]
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        for record in normalized.to_dict("records"):
            key = tuple(canonical_revision_value(record[column]) for column in keys)
            values = {column: canonical_revision_value(record[column]) for column in value_columns}
            rows[key] = values
        return rows

    old_rows = keyed_rows(old_df)
    new_rows = keyed_rows(new_df)
    old_keys = set(old_rows)
    new_keys = set(new_rows)
    changed_keys = sorted(key for key in old_keys & new_keys if old_rows[key] != new_rows[key])
    added_keys = sorted(new_keys - old_keys)
    removed_keys = sorted(old_keys - new_keys)
    changed_columns: dict[str, int] = {}
    changed_columns_sample: list[dict[str, Any]] = []
    for key in changed_keys:
        changed_values = []
        for column in value_columns:
            old_value = old_rows[key].get(column, "")
            new_value = new_rows[key].get(column, "")
            if old_value == new_value:
                continue
            changed_columns[column] = changed_columns.get(column, 0) + 1
            if len(changed_values) < REVISION_CHANGED_COLUMN_SAMPLE_SIZE:
                changed_values.append({"column": column, "old": old_value, "new": new_value})
        if changed_values and len(changed_columns_sample) < REVISION_CHANGED_KEY_SAMPLE_SIZE:
            changed_columns_sample.append({"key": list(key), "changes": changed_values})
    added_rows_sample = [
        {"key": list(key), "values": new_rows[key]}
        for key in added_keys[:REVISION_ADDED_REMOVED_ROW_SAMPLE_SIZE]
    ]
    removed_rows_sample = [
        {"key": list(key), "values": old_rows[key]}
        for key in removed_keys[:REVISION_ADDED_REMOVED_ROW_SAMPLE_SIZE]
    ]
    return {
        **base,
        "changed": bool(schema_changed or changed_keys or added_keys or removed_keys),
        "changed_keys": changed_keys,
        "added_keys": added_keys,
        "removed_keys": removed_keys,
        "changed_columns": changed_columns,
        "changed_columns_sample": changed_columns_sample,
        "added_rows_sample": added_rows_sample,
        "removed_rows_sample": removed_rows_sample,
    }

def revision_severity(dataset: str) -> str:
    if dataset in {"daily", "stk_limit", "suspend_d"}:
        return "high"
    if dataset in {"adj_factor", "daily_basic", "limit_list_d", "share_float_complete"}:
        return "medium"
    return "low"


def normalize_revision_event(event: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a revision event to the one fixed JSONL record schema."""
    unknown = set(event) - set(REVISION_EVENT_FIELDS)
    if unknown:
        raise ValueError(f"unknown revision event fields: {sorted(unknown)}")
    missing = set(REVISION_EVENT_FIELDS) - set(event)
    if missing:
        raise ValueError(f"missing revision event fields: {sorted(missing)}")
    return {field: event[field] for field in REVISION_EVENT_FIELDS}


def revision_event_id(event: dict[str, Any]) -> str:
    return str(uuid.uuid4())

def build_revision_event(
    *,
    dataset: str,
    partition: str,
    path: Path,
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_columns: list[str],
    source: str,
) -> dict[str, Any] | None:
    comparison = compare_keyed_frames(old_df, new_df, key_columns)
    if not comparison.get("changed"):
        return None
    keys = comparison["key_columns"]
    ts_code_index = keys.index("ts_code") if "ts_code" in keys else None
    all_changed = comparison["changed_keys"] + comparison["added_keys"] + comparison["removed_keys"]
    affected_codes = sorted(
        {
            key[ts_code_index]
            for key in all_changed
            if ts_code_index is not None and len(key) > ts_code_index and key[ts_code_index]
        }
    )
    event = {
        "schema_version": REVISION_EVENT_SCHEMA_VERSION,
        "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "dataset": dataset,
        "partition": partition,
        "path": str(path),
        "severity": revision_severity(dataset),
        "key_columns": keys,
        "old_rows": comparison["old_rows"],
        "new_rows": comparison["new_rows"],
        "changed_keys": len(comparison["changed_keys"]),
        "added_keys": len(comparison["added_keys"]),
        "removed_keys": len(comparison["removed_keys"]),
        "missing_key_columns_old": comparison.get("missing_key_columns_old", []),
        "missing_key_columns_new": comparison.get("missing_key_columns_new", []),
        "duplicate_key_rows_old": comparison.get("duplicate_key_rows_old", 0),
        "duplicate_key_rows_new": comparison.get("duplicate_key_rows_new", 0),
        "comparison_issue": comparison.get("comparison_issue", ""),
        "affected_ts_codes": len(affected_codes),
        "affected_ts_codes_sample": affected_codes[:20],
        "changed_keys_sample": [list(key) for key in comparison["changed_keys"][:5]],
        "added_keys_sample": [list(key) for key in comparison["added_keys"][:5]],
        "removed_keys_sample": [list(key) for key in comparison["removed_keys"][:5]],
        "changed_columns": comparison.get("changed_columns", {}),
        "changed_columns_sample": comparison.get("changed_columns_sample", []),
        "added_rows_sample": comparison.get("added_rows_sample", []),
        "removed_rows_sample": comparison.get("removed_rows_sample", []),
        "old_columns": comparison.get("old_columns", []),
        "new_columns": comparison.get("new_columns", []),
        "schema_changed": bool(comparison.get("schema_changed")),
        "write_id": None,
        "write_action": None,
        "allow_empty_revision_overwrite": None,
    }
    event["event_id"] = revision_event_id(event)
    return normalize_revision_event(event)

def finalize_revision_event(
    event: dict[str, Any],
    *,
    write_id: str | None,
    write_action: str,
    allow_empty_revision_overwrite: bool,
) -> dict[str, Any]:
    event.update({
        "write_id": write_id,
        "write_action": write_action,
        "allow_empty_revision_overwrite": bool(allow_empty_revision_overwrite),
    })
    event["event_id"] = revision_event_id(event)
    return normalize_revision_event(event)

def write_parquet_revision_aware(
    path: Path,
    df: pd.DataFrame,
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    key_columns: list[str],
    revision_ledger: Path | str | None,
    source: str = "force_refresh",
    allow_empty_revision_overwrite: bool = False,
    allow_key_removal_overwrite: bool = False,
    revision_comparison_old_df: pd.DataFrame | None = None,
    revision_comparison_new_df: pd.DataFrame | None = None,
    existing_df: pd.DataFrame | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    if (revision_comparison_old_df is None) != (revision_comparison_new_df is None):
        raise ValueError("revision comparison frames must be provided together")
    event: dict[str, Any] | None = None
    write_action = "overwrite"
    old_meta: dict[str, Any] = {}
    if path.exists():
        old_df = existing_df if existing_df is not None else pd.read_parquet(path)
        try:
            old_meta = parquet_meta(path)
        except CorruptSidecarError as exc:
            # Torn pair from a crashed writer: this rewrite repairs both files
            # atomically, so proceed -- but loudly, never silently.
            print("REVISION_ALERT " + json.dumps(
                {"note": "corrupt_sidecar_replaced", "dataset": api_name,
                 "partition": path.with_suffix("").name, "error": str(exc)[:300]},
                ensure_ascii=False, sort_keys=True,
            ))
            old_meta = {}
        if len(old_df) > 0 and df.empty and not allow_empty_revision_overwrite:
            write_action = "skipped_empty_revision_overwrite"
        event = build_revision_event(
            dataset=api_name,
            partition=path.with_suffix("").name,
            path=path,
            old_df=revision_comparison_old_df if revision_comparison_old_df is not None else old_df,
            new_df=revision_comparison_new_df if revision_comparison_new_df is not None else df,
            key_columns=key_columns,
            source=source,
        )
        if event and revision_comparison_old_df is not None:
            # The event diff is restricted to a caller-proven affected slice,
            # while partition row counts continue to describe the durable file.
            event["old_rows"] = int(len(old_df))
            event["new_rows"] = int(len(df))
        # A re-pull that DROPS existing keys is destructive (the ann_month
        # truncated-window class silently deleted announcements). Blocked by
        # default; accepting a genuine source retraction means deleting the
        # partition file first or passing allow_key_removal_overwrite.
        if write_action == "overwrite" and event and event["removed_keys"] and not allow_key_removal_overwrite:
            write_action = "skipped_key_removal_overwrite"
        if (
            write_action == "overwrite"
            and event
            and event["removed_keys"]
            and allow_key_removal_overwrite
        ):
            # Full-partition pulls accept key removals as source corrections, but a
            # transiently truncated (non-empty) response must not wipe a partition:
            # a disproportionate shrink (>20 keys AND >20% of existing keys) blocks.
            removed = event["removed_keys"]
            removed_count = removed if isinstance(removed, int) else len(removed)
            existing_cols = [col for col in key_columns if col in old_df.columns]
            old_key_count = len(old_df.drop_duplicates(existing_cols)) if existing_cols else len(old_df)
            if removed_count > 20 and old_key_count > 0 and removed_count > 0.2 * old_key_count:
                write_action = "blocked_shrink_overwrite"
        if write_action == "skipped_empty_revision_overwrite":
            if revision_ledger and event:
                record = finalize_revision_event(
                    event,
                    write_id=None,
                    write_action=write_action,
                    allow_empty_revision_overwrite=allow_empty_revision_overwrite,
                )
                append_jsonl_unique(Path(revision_ledger), record, key="event_id")
            print(f"{api_name} {path} returned zero rows for existing nonempty partition; skipped_empty_revision_overwrite")
            return False
        if write_action == "skipped_key_removal_overwrite":
            if revision_ledger and event:
                record = finalize_revision_event(
                    event,
                    write_id=None,
                    write_action=write_action,
                    allow_empty_revision_overwrite=allow_empty_revision_overwrite,
                )
                append_jsonl_unique(Path(revision_ledger), record, key="event_id")
            print(
                f"{api_name} {path} new pull removes {event['removed_keys']} existing keys; "
                "skipped_key_removal_overwrite (delete the partition to accept a source retraction)"
            )
            return False
        if write_action == "blocked_shrink_overwrite":
            if revision_ledger and event:
                record = finalize_revision_event(
                    event,
                    write_id=None,
                    write_action=write_action,
                    allow_empty_revision_overwrite=allow_empty_revision_overwrite,
                )
                append_jsonl_unique(Path(revision_ledger), record, key="event_id")
            print(
                f"{api_name} {path} new pull would remove a disproportionate share of existing keys; "
                "blocked_shrink_overwrite (kept the old partition; delete it to accept a mass retraction)"
            )
            return False
    metadata = dict(extra_metadata or {})
    previous_availability = old_meta.get("availability")
    if previous_availability and "availability" not in metadata:
        metadata["availability"] = dict(previous_availability)
        if event:
            metadata["availability"].update(
                available_at=datetime.now(timezone.utc).isoformat(),
                rule="observed:source_revision_fetch",
                row_count=int(len(df)),
            )
    commit = write_parquet(
        path,
        df,
        api_name=api_name,
        params=params,
        fields=fields,
        extra_metadata=metadata,
    )
    if revision_ledger and event:
        record = finalize_revision_event(
            event,
            write_id=str(commit["write_id"]),
            write_action=write_action,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        )
        append_jsonl_unique(Path(revision_ledger), record, key="event_id")
        print("REVISION_ALERT " + json.dumps(
            {key: record.get(key) for key in ("event_id", "dataset", "partition", "severity", "comparison_issue", "write_action")},
            ensure_ascii=False, sort_keys=True,
        ))
    return True

def load_stock_codes(raw_dir: Path) -> list[str]:
    files = sorted((raw_dir / "stock_basic").glob("list_status=*.parquet"))
    if not files:
        raise RuntimeError("stock_basic partitions are missing")
    data = read_many(files, columns=["ts_code"])
    codes = data["ts_code"].dropna().astype(str).str.strip()
    valid = codes[codes.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")]
    return sorted(valid.unique().tolist())

def normalize_date_key(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))[:8]
    return digits if len(digits) == 8 else ""


def normalized_date_keys(values: pd.Series) -> pd.Series:
    """Vectorized twin of :func:`normalize_date_key` for whole-column passes.

    Same rule, same output for every input (a drift test pins the equivalence):
    non-digits dropped, first 8 digits kept, and anything that is not exactly 8
    digits -- missing, null, empty, malformed -- becomes ``""``. ``""`` sorts
    BELOW every real ``YYYYMMDD``, which is what makes an undated row lose the
    newest-dating comparison instead of winning it (``str(None) == "None"``
    sorted above every date and inverted the comparison)."""
    keys = values.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
    return keys.where(keys.str.len() == 8, "")


# One physical unlock (解禁) event, independent of its announcement date:
# ``ann_date`` is excluded because the provider re-dates announcements.
PHYSICAL_IDENTITY_COLUMNS = ["ts_code", "float_date", "holder_name", "share_type"]


def normalized_holder_keys(values: pd.Series) -> pd.Series:
    """Holder names with whitespace ignored (raw rows keep their source bytes).

    Restatements carry cosmetic spacing drift (observed live: "...企业 (有限合伙)"
    re-dated as "...企业(有限合伙)"), so whitespace must not split one holder
    into two identities."""
    return values.astype(str).str.replace(r"\s+", "", regex=True)


def physical_identity_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """Physical-identity index for membership tests between two frames."""
    view = frame[PHYSICAL_IDENTITY_COLUMNS].copy()
    view["holder_name"] = normalized_holder_keys(view["holder_name"])
    return pd.MultiIndex.from_frame(view)


def physical_identity_codes(frame: pd.DataFrame) -> pd.Series:
    """Dense integer identity code per row -- same identity rule as
    :func:`physical_identity_index`, but for whole-baseline grouping where
    materializing 12M index tuples would be wasteful."""
    view = frame[PHYSICAL_IDENTITY_COLUMNS].copy()
    view["holder_name"] = normalized_holder_keys(view["holder_name"])
    return view.groupby(PHYSICAL_IDENTITY_COLUMNS, sort=False, dropna=False).ngroup()


def undated_duplicate_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows carrying no announcement date whose physical identity already holds
    a dated row of the identical ``float_share``.

    Such a row is the undated copy of an announcement the lake already has: it
    contributes no observation and double-counts the lot. Deleting it cannot
    move point-in-time visibility, because the undated row's availability was
    the conservative ``float_date`` fallback while the dated twin was already
    present and announced no later.

    Shared by the union and the one-off repair so the property is an invariant
    of the data rather than a fact about the current update schedule: a rescue
    fetch that returns an undated lot must not re-introduce the duplicate.
    """
    required = [*PHYSICAL_IDENTITY_COLUMNS, "ann_date", "float_share"]
    if frame.empty or any(column not in frame.columns for column in required):
        return np.zeros(len(frame), dtype=bool)
    ann = normalized_date_keys(frame["ann_date"]).to_numpy()
    identity = physical_identity_codes(frame).to_numpy()
    share = pd.to_numeric(frame["float_share"], errors="coerce").to_numpy()
    undated = (ann == "") & ~pd.isna(share)
    dated = ann != ""
    if not undated.any() or not dated.any():
        return np.zeros(len(frame), dtype=bool)
    twins = pd.MultiIndex.from_arrays([identity[dated], share[dated]])
    candidates = pd.MultiIndex.from_arrays([identity[undated], share[undated]])
    mask = np.zeros(len(frame), dtype=bool)
    mask[np.flatnonzero(undated)[candidates.isin(twins)]] = True
    return mask


def load_sse_open_dates(raw_dir: Path, start_date: str, end_date: str, *, allow_empty: bool = False) -> list[str]:
    files = sorted((raw_dir / "trade_cal" / "exchange=SSE").glob("year=*.parquet"))
    if not files:
        raise RuntimeError("SSE trade_cal partitions are missing; run download --tier reference first")
    calendar = read_many(files, columns=["cal_date", "is_open"])
    if calendar.empty:
        raise RuntimeError("SSE trade_cal is empty; run download --tier reference first")
    calendar["cal_date"] = calendar["cal_date"].map(normalize_date_key)
    calendar = calendar[calendar["cal_date"] != ""].copy()
    if calendar.empty:
        raise RuntimeError("SSE trade_cal has no parseable cal_date values; refresh reference trade_cal first")
    available_min = str(calendar["cal_date"].min())
    available_max = str(calendar["cal_date"].max())
    if start_date < available_min or end_date > available_max:
        raise RuntimeError(
            f"SSE trade_cal covers {available_min}-{available_max}, not requested {start_date}-{end_date}; refresh reference trade_cal first"
        )
    mask = (calendar["is_open"].astype(str) == "1") & (calendar["cal_date"] >= start_date) & (calendar["cal_date"] <= end_date)
    dates = sorted(calendar.loc[mask, "cal_date"].tolist())
    if not dates and not allow_empty:
        raise RuntimeError(f"no SSE open dates found for {start_date}-{end_date}")
    return dates

def latest_sse_calendar_date(raw_dir: Path) -> str:
    files = sorted((raw_dir / "trade_cal" / "exchange=SSE").glob("year=*.parquet"))
    if not files:
        raise RuntimeError("SSE trade_cal partitions are missing; run download --tier reference first")
    calendar = read_many(files, columns=["cal_date"])
    if calendar.empty:
        raise RuntimeError("SSE trade_cal is empty; run download --tier reference first")
    dates = calendar["cal_date"].dropna().map(normalize_date_key)
    dates = dates[dates != ""]
    if dates.empty:
        raise RuntimeError("SSE trade_cal has no parseable cal_date values; refresh reference trade_cal first")
    return str(dates.max())

def select_datasets(
    values: Iterable[str] | None,
    *,
    default: Iterable[str],
    allowed: Iterable[str],
    label: str,
) -> list[str]:
    """Resolve an optional CLI dataset list: fall back to ``default`` when
    empty and fail explicitly on names missing from ``allowed``."""
    datasets = list(values or default)
    invalid = sorted(set(datasets) - set(allowed))
    if invalid:
        raise RuntimeError(f"unknown {label} datasets: {invalid}; supported={sorted(allowed)}")
    return datasets

def selected_daily_datasets(args: argparse.Namespace, *, default: Iterable[str] | None = None) -> list[str]:
    return select_datasets(
        args.datasets,
        default=DAILY_DOWNLOAD_DATASETS if default is None else default,
        allowed=DAILY_SPECS,
        label="daily market",
    )

def partition_date(path: Path) -> str:
    return path.stem.split("=", 1)[1] if "=" in path.stem else ""

def selected_fundamental_datasets(values: Iterable[str] | None) -> list[str]:
    return select_datasets(values, default=FUNDAMENTAL_DATASETS, allowed=FUNDAMENTAL_SPECS, label="fundamental")

def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()

def format_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")

def month_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    current = date(start.year, start.month, 1)
    windows: list[tuple[str, str, str]] = []
    while current <= end:
        last = date(current.year, current.month, calendar.monthrange(current.year, current.month)[1])
        window_start = max(current, start)
        window_end = min(last, end)
        windows.append((format_yyyymmdd(window_start), format_yyyymmdd(window_end), f"{current.year}{current.month:02d}"))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return windows

def quarter_periods(start_date: str, end_date: str) -> list[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    periods: list[str] = []
    for year in range(start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            period = date(year, month, day)
            if start <= period <= end:
                periods.append(format_yyyymmdd(period))
    return periods

def yyyymmdd_to_month(value: str) -> str:
    return value[:6]

def month_end_from_yyyymm(value: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError(f"invalid yyyymm value: {value}")
    year = int(text[:4])
    month = int(text[4:6])
    day = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}{day:02d}"

def yyyymmdd_to_quarter(value: str) -> str:
    parsed = parse_yyyymmdd(value)
    quarter = (parsed.month - 1) // 3 + 1
    return f"{parsed.year}Q{quarter}"

def quarter_end_date(value: str) -> str:
    match = re.fullmatch(r"(\d{4})Q([1-4])", str(value).strip())
    if not match:
        return ""
    year = int(match.group(1))
    quarter = int(match.group(2))
    month, day = ((3, 31), (6, 30), (9, 30), (12, 31))[quarter - 1]
    return f"{year}{month:02d}{day:02d}"

def local_eod(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} 23:59:59+08:00"

def period_available_at(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return local_eod(text), "conservative_date_eod"
    if re.fullmatch(r"\d{6}", text):
        month_end = parse_yyyymmdd(month_end_from_yyyymm(text))
        available = month_end + timedelta(days=31)
        return local_eod(format_yyyymmdd(available)), "conservative_month_end_plus_31d"
    if re.fullmatch(r"\d{4}Q[1-4]", text):
        quarter_end = quarter_end_date(text)
        available = parse_yyyymmdd(quarter_end) + timedelta(days=45)
        return local_eod(format_yyyymmdd(available)), "conservative_quarter_end_plus_45d"
    return "", "missing_source_date"

def combine_date_time_available(date_value: str, time_value: str) -> str:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if not re.fullmatch(r"\d{8}", date_text):
        return ""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", time_text)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return ""
    return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]} {hour:02d}:{minute:02d}:{second:02d}+08:00"

def augment_macro_frame(df: pd.DataFrame, spec: MacroDataset) -> pd.DataFrame:
    out = df.copy()
    if "available_at" not in out.columns:
        out["available_at"] = ""
    if "available_at_rule" not in out.columns:
        out["available_at_rule"] = "missing_source_date"
    if spec.date_column and spec.date_column in out.columns:
        date_values = out[spec.date_column].astype(str).str.strip()
        if spec.time_column and spec.time_column in out.columns:
            time_values = out[spec.time_column].astype(str).str.strip()
            combined = [combine_date_time_available(day, clock) for day, clock in zip(date_values, time_values, strict=False)]
            combined_series = pd.Series(combined, index=out.index)
            mask = combined_series.astype(str).str.strip().ne("")
            out.loc[mask, "available_at"] = combined_series[mask]
            out.loc[mask, "available_at_rule"] = f"source:{spec.date_column}+{spec.time_column}"
        missing = out["available_at"].astype(str).str.strip().eq("")
        derived = date_values.map(period_available_at)
        available = derived.map(lambda item: item[0])
        rules = derived.map(lambda item: item[1])
        mask = missing & available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = rules[mask]
    return out

def selected_macro_datasets(args: argparse.Namespace) -> list[str]:
    default = GLOBAL_CONTEXT_DEFAULT_DATASETS if args.tier == "global" else MACRO_REGIME_DEFAULT_DATASETS
    return select_datasets(args.datasets, default=default, allowed=MACRO_SPECS, label="macro/global")

def spec_page_limit(spec, requested: int | None) -> int:
    """Effective page size for one dataset spec: the caller's request clamped
    to the spec's documented limit; 10000 when the spec documents none."""
    if spec.page_limit is not None:
        return min(requested or spec.page_limit, spec.page_limit)
    return requested or 10000

def selected_index_codes(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "index_code", None) or [] if str(value).strip()]
    return values or list(DEFAULT_GLOBAL_INDEX_CODES)

def selected_cn_index_codes(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "cn_index_code", None) or [] if str(value).strip()]
    return values or list(DEFAULT_CN_INDEX_CODES)

def selected_fx_codes(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "fx_code", None) or [] if str(value).strip()]
    return values or list(DEFAULT_FX_CODES)

def selected_eco_filter_values(args: argparse.Namespace, attr: str) -> list[str | None]:
    values = [str(value).strip() for value in getattr(args, attr, None) or [] if str(value).strip()]
    return values or [None]

def write_macro_result(
    path: Path,
    result: ApiResult,
    spec: MacroDataset,
    params: dict[str, Any],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> int:
    df = augment_macro_frame(frame(result), spec)
    written = write_parquet_revision_aware(
        path,
        df,
        api_name=spec.api_name,
        params=params,
        fields=list(df.columns),
        key_columns=list(spec.key_columns),
        revision_ledger=revision_ledger,
        allow_empty_revision_overwrite=allow_empty_revision_overwrite,
    )
    if not written:
        return 0
    return len(df)

def selected_event_flow_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(args.datasets, default=EVENT_FLOW_DATASETS, allowed=EVENT_FLOW_SPECS, label="event/flow")

def selected_board_trading_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(getattr(args, "datasets", None), default=BOARD_TRADING_DEFAULT_DATASETS, allowed=BOARD_TRADING_SPECS, label="board-trading")

def selected_board_kpl_tags(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "kpl_tag", None) or [] if str(value).strip()]
    return values or list(BOARD_KPL_TAGS)

def selected_board_ths_limit_types(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "ths_limit_type", None) or [] if str(value).strip()]
    return values or list(BOARD_THS_LIMIT_TYPES)

def selected_board_ths_hot_markets(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "ths_hot_market", None) or [] if str(value).strip()]
    return values or list(BOARD_THS_HOT_MARKETS)

def selected_board_dc_hot_markets(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "dc_hot_market", None) or [] if str(value).strip()]
    return values or list(BOARD_DC_HOT_MARKETS)

def selected_board_dc_hot_types(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in getattr(args, "dc_hot_type", None) or [] if str(value).strip()]
    return values or list(BOARD_DC_HOT_TYPES)

def selected_board_hot_is_new(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip().upper() for value in getattr(args, "hot_is_new", None) or [] if str(value).strip()]
    values = values or list(BOARD_HOT_IS_NEW)
    invalid = sorted(set(values) - {"Y", "N"})
    if invalid:
        raise RuntimeError(f"invalid hot is_new values: {invalid}; supported=['Y', 'N']")
    return values

def local_time(value: str, clock: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {clock}+08:00"

# BSE margin trading opened 2023-02-13; before that the exchange-level margin
# table legitimately carries SSE+SZSE only. A day missing a required exchange
# poisons market-wide aggregates (measured: 2026-06 Σrzye −49% craters from
# SSE-only days), so partial days must never be committed.
MARGIN_BSE_START = "20230213"

def margin_missing_exchanges(trade_date: str, present: Iterable[str]) -> list[str]:
    required = {"SSE", "SZSE"}
    if str(trade_date) >= MARGIN_BSE_START:
        required.add("BSE")
    return sorted(required - {str(item) for item in present})

# Catalog-scale reference pulls must page: a single unpaged call silently
# truncates at whatever cap the vendor applies (index_basic sat at exactly
# 8,000 of 12,484 rows). The limit only needs to stay at or below every real
# cap; short pages end pagination naturally.
REFERENCE_PAGE_LIMIT = 5000

# Full-market per-trade-date tables whose per-day stock universe must track
# the daily table. Measured 2020-2026 per-day floors: cyq_perf 0.9109 (2020
# risk-warning/select-layer gap), moneyflow 0.9401 (the vendor feed carries no
# BSE codes at all), stk_auction 0.9486; every other day sits higher. 0.90
# therefore trips only on a genuinely dropped batch of stocks, never on the
# known structural residuals.
FULL_MARKET_COVERAGE_MIN_RATIO = 0.90
FULL_MARKET_COVERAGE_DATASETS = (
    "bak_basic",
    "bak_daily",
    "cyq_perf",
    "moneyflow",
    "moneyflow_dc",
    "moneyflow_ths",
    "stk_auction",
)

# All margin tables publish per exchange and share the requirement above, but
# they name the exchange differently: the summary and the eligibility list
# carry it as a column, the per-stock detail only through the ts_code
# listing-venue suffix. Holding them here keeps one required-exchange rule and
# lets callers stay dataset-agnostic.
MARGIN_EXCHANGE_BY_TS_CODE_SUFFIX = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}
MARGIN_EXCHANGE_COLUMNS = {"margin": "exchange_id", "margin_detail": "ts_code", "margin_secs": "exchange"}

def margin_exchange_column(api_name: str) -> str | None:
    """Column identifying the exchange, or None outside the margin family."""
    return MARGIN_EXCHANGE_COLUMNS.get(str(api_name))

def margin_family_missing_exchanges(api_name: str, trade_date: str, frame: pd.DataFrame) -> list[str]:
    """Required exchanges absent from one margin-family trade-date partition.

    Empty for every other dataset, so callers need no per-dataset branch. An
    unreadable partition fails loudly instead of passing as complete: silently
    skipping the check is what let SSE-only margin_detail days be committed.
    """
    column = margin_exchange_column(api_name)
    if column is None:
        return []
    if column not in frame.columns:
        raise RuntimeError(
            f"{api_name} trade_date={trade_date} has no {column} column; "
            "exchange completeness cannot be verified"
        )
    values = [str(item) for item in frame[column]]
    if column == "ts_code":
        values = [MARGIN_EXCHANGE_BY_TS_CODE_SUFFIX.get(value.rsplit(".", 1)[-1], "") for value in values]
    return margin_missing_exchanges(trade_date, values)

def official_available_at(
    value: str,
    availability: AvailabilityRule | None,
    params: dict[str, Any] | None = None,
    trading_dates: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Stamp the spec's official publication time, or conservative date EOD."""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{8}", text):
        return "", "missing_source_date"
    rule = availability
    if rule is not None and rule.only_when_is_new and (params or {}).get("is_new") != "Y":
        rule = None
    if rule is None:
        return local_eod(text), "conservative_date_eod"
    day = text
    if rule.trading_day_offset:
        if trading_dates is None:
            raise RuntimeError(f"{rule.rule} requires the A-share trading calendar")
        target_index = bisect_right(trading_dates, text) + rule.trading_day_offset - 1
        if target_index >= len(trading_dates):
            raise RuntimeError(
                f"A-share trading calendar has fewer than {rule.trading_day_offset} open dates after {text}"
            )
        day = str(trading_dates[target_index])
    elif rule.day_offset:
        day = format_yyyymmdd(parse_yyyymmdd(text) + timedelta(days=rule.day_offset))
    return local_time(day, rule.clock), rule.rule

def augment_event_frame(
    df: pd.DataFrame,
    spec: EventDataset,
    *,
    trading_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "available_at" not in out.columns:
        out["available_at"] = ""
    if "available_at_rule" not in out.columns:
        out["available_at_rule"] = "missing_source_date"
    if spec.date_column and spec.date_column in out.columns:
        date_values = out[spec.date_column].astype(str).str.strip()
        derived = date_values.map(
            lambda item: official_available_at(item, spec.availability, trading_dates=trading_dates)
        )
        available = derived.map(lambda item: item[0])
        rules = derived.map(lambda item: item[1])
        mask = available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = rules[mask]
    if spec.fallback_date_column and spec.fallback_date_column in out.columns:
        missing = out["available_at"].astype(str).str.strip().eq("")
        fallback_values = out[spec.fallback_date_column].astype(str).str.strip()
        derived = fallback_values.map(
            lambda item: official_available_at(item, spec.availability, trading_dates=trading_dates)
        )
        available = derived.map(lambda item: item[0])
        mask = missing & available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = f"fallback_conservative_from:{spec.fallback_date_column}"
    return out

def normalize_source_datetime(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"nan", "None"}:
        return ""
    if re.fullmatch(r"\d{8} \d{2}:\d{2}:\d{2}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]} {text[9:]}+08:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
        return f"{text}+08:00"
    return text

def augment_board_frame(df: pd.DataFrame, spec: BoardTradingDataset, params: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    if "available_at" not in out.columns:
        out["available_at"] = ""
    if "available_at_rule" not in out.columns:
        out["available_at_rule"] = "missing_source_time"
    if spec.time_column and spec.time_column in out.columns:
        source_time = out[spec.time_column].map(normalize_source_datetime)
        mask = source_time.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = source_time[mask]
        out.loc[mask, "available_at_rule"] = f"source:{spec.time_column}"
    if spec.date_column and spec.date_column in out.columns:
        missing = out["available_at"].astype(str).str.strip().eq("")
        date_values = out[spec.date_column].astype(str).str.strip()
        derived = date_values.map(lambda item: official_available_at(item, spec.availability, params))
        available = derived.map(lambda item: item[0])
        rules = derived.map(lambda item: item[1])
        mask = missing & available.astype(str).str.strip().ne("")
        out.loc[mask, "available_at"] = available[mask]
        out.loc[mask, "available_at_rule"] = rules[mask]
    if spec.strategy == "static_full" and out["available_at"].astype(str).str.strip().eq("").all():
        out["available_at_rule"] = "static_reference_no_pit_time"
    return out

def write_board_result(
    path: Path,
    result: ApiResult,
    spec: BoardTradingDataset,
    params: dict[str, Any],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> int:
    df = augment_board_frame(frame(result), spec, params)
    # Board pulls cover their partition's full scope (request param == partition
    # key), so removed keys can only be alerted source corrections.
    written = write_parquet_revision_aware(
        path,
        df,
        api_name=spec.api_name,
        params=params,
        fields=list(df.columns),
        key_columns=list(spec.key_columns),
        revision_ledger=revision_ledger,
        allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        allow_key_removal_overwrite=True,
    )
    if not written:
        return 0
    return len(df)

def selected_intraday_datasets(values: Iterable[str] | None) -> list[str]:
    aliases = {"stk_mins": STK_MINS_DATASET}
    normalized = [aliases.get(str(name), str(name)) for name in values or INTRADAY_DATASETS]
    return select_datasets(normalized, default=INTRADAY_DATASETS, allowed=INTRADAY_DATASETS, label="intraday minute")

def clean_yyyymmdd(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"\d{8}", text) else ""

def date_overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    return start_a <= end_b and start_b <= end_a


def load_minute_universe(raw_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    files = sorted((raw_dir / "stock_basic").glob("list_status=*.parquet"))
    if not files:
        raise RuntimeError("stock_basic partitions are missing; run download --tier reference first")
    cols = ["ts_code", "name", "market", "exchange", "list_status", "list_date", "delist_date"]
    df = read_many(files, columns=cols)
    if df.empty:
        raise RuntimeError("stock_basic is empty; cannot build full-A minute universe")
    df = df.dropna(subset=["ts_code"]).copy()
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df = df[df["ts_code"].ne("")].drop_duplicates("ts_code", keep="first")
    if getattr(args, "codes", None):
        wanted = {str(code).strip() for code in args.codes if str(code).strip()}
        df = df[df["ts_code"].isin(wanted)]
    df = df.sort_values("ts_code").reset_index(drop=True)
    if getattr(args, "max_codes", None):
        df = df.head(int(args.max_codes))
    if df.empty:
        raise RuntimeError("minute universe is empty after code/max-code filtering")
    return df


def active_year_windows(row: pd.Series, start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    list_date = clean_yyyymmdd(row.get("list_date")) or start_date
    delist_date = clean_yyyymmdd(row.get("delist_date")) or end_date
    active_start = max(start_date, list_date)
    active_end = min(end_date, delist_date)
    if active_start > active_end:
        return []
    windows: list[tuple[int, str, str]] = []
    for year in range(int(active_start[:4]), int(active_end[:4]) + 1):
        year_start = max(active_start, f"{year}0101")
        year_end = min(active_end, f"{year}1231")
        if date_overlaps(year_start, year_end, start_date, end_date):
            windows.append((year, year_start, year_end))
    return windows


def minute_datetime(value: str, *, end: bool = False) -> str:
    suffix = "15:00:00" if end else "09:00:00"
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {suffix}"

def augment_stk_mins_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
    out = df.copy()
    out["trade_time"] = out["trade_time"].astype(str).str.strip()
    parsed = pd.to_datetime(out["trade_time"], errors="coerce")
    out["trade_date"] = parsed.dt.strftime("%Y%m%d").fillna("")
    out["available_at"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S+08:00").fillna("")
    out["available_at_rule"] = "source:trade_time_bar_close"
    sort_cols = [col for col in ["ts_code", "trade_time"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    for col in STK_MINS_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"ts_code", "trade_time", "trade_date", "available_at", "available_at_rule"} else pd.NA
    return out[STK_MINS_REQUIRED_COLUMNS]

def stk_mins_by_date_path(raw_dir: Path, dataset: str, trade_date: str) -> Path:
    return raw_dir / dataset / f"trade_date={trade_date}.parquet"

def normalize_stk_mins_by_date_frame(df: pd.DataFrame, trade_date: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        out = pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
        return out, {"input_rows": 0, "filtered_rows": 0, "duplicate_rows_dropped": 0}
    out = df.copy()
    for col in STK_MINS_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"ts_code", "trade_time", "trade_date", "available_at", "available_at_rule"} else pd.NA
    out = out[STK_MINS_REQUIRED_COLUMNS]
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_time"] = out["trade_time"].astype(str).str.strip()
    if "trade_date" not in out or out["trade_date"].astype(str).str.strip().eq("").all():
        parsed = pd.to_datetime(out["trade_time"], errors="coerce")
        out["trade_date"] = parsed.dt.strftime("%Y%m%d").fillna("")
    else:
        out["trade_date"] = out["trade_date"].astype(str).str.strip()
    before_filter = len(out)
    out = out[out["trade_date"] == trade_date].copy()
    if out["available_at"].astype(str).str.strip().eq("").any():
        parsed = pd.to_datetime(out["trade_time"], errors="coerce")
        mask = out["available_at"].astype(str).str.strip().eq("") & parsed.notna()
        out.loc[mask, "available_at"] = parsed[mask].dt.strftime("%Y-%m-%d %H:%M:%S+08:00")
    missing_rule = out["available_at_rule"].astype(str).str.strip().eq("")
    out.loc[missing_rule, "available_at_rule"] = "source:trade_time_bar_close"
    before_dedup = len(out)
    out = out.drop_duplicates(["ts_code", "trade_time"], keep="last")
    out = out.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)
    return out, {
        "input_rows": int(before_filter),
        "filtered_rows": int(before_filter - len(out)),
        "duplicate_rows_dropped": int(before_dedup - len(out)),
    }

def intraday_day_time_details(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "trade_time" not in df.columns:
        return {"unique_times": 0, "has_0930": False, "has_1500": False, "invalid_time_rows": int(len(df))}
    time_hhmm = df["trade_time"].astype(str).str.extract(r"(\d{2}:\d{2})", expand=False)
    valid = (
        (time_hhmm == "09:30")
        | ((time_hhmm >= "09:31") & (time_hhmm <= "11:30"))
        | ((time_hhmm >= "13:00") & (time_hhmm <= "15:00"))
    )
    times = set(time_hhmm.dropna().tolist())
    return {
        "unique_times": int(time_hhmm.nunique(dropna=True)),
        "has_0930": "09:30" in times,
        "has_1500": "15:00" in times,
        "invalid_time_rows": int((~valid.fillna(False)).sum()),
    }

def validate_stk_mins_by_date_frame(
    df: pd.DataFrame,
    trade_date: str,
    *,
    expected_codes: set[str] | None = None,
    min_rows: int = 0,
    allow_missing_codes: int = 0,
) -> tuple[bool, dict[str, Any]]:
    missing_columns = sorted(set(STK_MINS_REQUIRED_COLUMNS) - set(df.columns))
    details: dict[str, Any] = {
        "trade_date": trade_date,
        "rows": int(len(df)),
        "unique_codes": int(df["ts_code"].nunique()) if "ts_code" in df.columns else 0,
        "missing_columns": missing_columns,
        "duplicate_key_rows": 0,
        "wrong_trade_date_rows": 0,
        "unparseable_trade_time_rows": 0,
        "unparseable_available_at_rows": 0,
        "min_rows": int(min_rows),
        "expected_codes": len(expected_codes or set()),
        "missing_expected_codes": 0,
        "extra_codes": 0,
        "missing_expected_sample": [],
        "extra_code_sample": [],
    }
    if missing_columns:
        return False, details
    details["duplicate_key_rows"] = int(df.duplicated(["ts_code", "trade_time"]).sum())
    details["wrong_trade_date_rows"] = int((df["trade_date"].astype(str) != trade_date).sum())
    parsed_trade = pd.to_datetime(df["trade_time"].astype(str), errors="coerce")
    details["unparseable_trade_time_rows"] = int(parsed_trade.isna().sum())
    available = df["available_at"].astype(str).str.strip()
    parsed_available = pd.to_datetime(available[available.ne("")], errors="coerce", utc=True)
    details["unparseable_available_at_rows"] = int(parsed_available.isna().sum())
    details.update(intraday_day_time_details(df))
    if expected_codes is not None:
        actual_codes = set(df["ts_code"].dropna().astype(str))
        missing = sorted(expected_codes - actual_codes)
        extra = sorted(actual_codes - expected_codes)
        details["missing_expected_codes"] = len(missing)
        details["extra_codes"] = len(extra)
        details["missing_expected_sample"] = missing[:20]
        details["extra_code_sample"] = extra[:20]
    ok = not any([
        missing_columns,
        details["duplicate_key_rows"],
        details["wrong_trade_date_rows"],
        details["unparseable_trade_time_rows"],
        details["unparseable_available_at_rows"],
        not details["has_0930"] if len(df) else False,
        not details["has_1500"] if len(df) else False,
        len(df) < min_rows,
        details["missing_expected_codes"] > allow_missing_codes,
    ])
    return bool(ok), details

def intraday_expected_codes_for_day(raw_dir: Path, args: argparse.Namespace, trade_date: str) -> set[str] | None:
    source = getattr(args, "expected_codes_source", "none")
    if source == "none":
        return None
    if getattr(args, "codes", None):
        codes = {str(code).strip() for code in args.codes if str(code).strip()}
    elif source in {"daily", "minute"}:
        if source == "minute":
            dataset = getattr(args, "output_dataset", STK_MINS_BY_DATE_DATASET)
            existing_path = stk_mins_by_date_path(raw_dir, dataset, trade_date)
            if existing_path.exists() and parquet_rows(existing_path) > 0:
                existing = pd.read_parquet(existing_path, columns=["ts_code"])
                codes = set(existing["ts_code"].dropna().astype(str))
                if getattr(args, "max_codes", None):
                    codes = set(sorted(codes)[: int(args.max_codes)])
                return codes
        daily_path = raw_dir / "daily" / f"trade_date={trade_date}.parquet"
        if not daily_path.exists():
            return set()
        if parquet_rows(daily_path) == 0:
            return set()
        daily = pd.read_parquet(daily_path, columns=["ts_code"])
        codes = set(daily["ts_code"].dropna().astype(str))
    else:
        universe = load_minute_universe(raw_dir, argparse.Namespace(codes=None, max_codes=None))
        active = []
        for _, row in universe.iterrows():
            list_date = clean_yyyymmdd(row.get("list_date")) or "00000000"
            delist_date = clean_yyyymmdd(row.get("delist_date")) or "99999999"
            if list_date <= trade_date <= delist_date:
                active.append(str(row["ts_code"]))
        codes = set(active)
    if getattr(args, "max_codes", None):
        codes = set(sorted(codes)[: int(args.max_codes)])
    return codes

def selected_text_datasets(values: Iterable[str] | None, *, news_src: list[str] | None = None) -> list[str]:
    datasets = select_datasets(values, default=TEXT_DATASETS, allowed=TEXT_SPECS, label="text")
    if news_src is not None and "news" in datasets:
        selected_news_sources(news_src)
    return datasets

def selected_news_sources(values: list[str] | None) -> list[str]:
    requested = [str(value).strip() for value in values or [] if str(value).strip()]
    if not requested or any(value.lower() == "all" for value in requested):
        return list(NEWS_SOURCES)
    invalid = sorted(set(requested) - set(NEWS_SOURCES))
    if invalid:
        raise RuntimeError(f"unknown news source(s): {invalid}; official sources are {NEWS_SOURCES}")
    return requested

def date_range_days(start_date: str, end_date: str) -> list[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    days = pd.date_range(start, end, freq="D")
    return days.strftime("%Y%m%d").tolist()

def as_datetime_window(value: str, *, end: bool = False) -> str:
    suffix = "23:59:59" if end else "00:00:00"
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {suffix}"

def safe_partition_value(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    if cleaned.strip("_"):
        return cleaned.strip("_")
    encoded = str(value).encode("utf-8").hex()
    return encoded[:96] or "empty"

# A source timestamp is a credible publication time only when it sits near the
# announcement/report date. Backfilled history carries collection timestamps
# (e.g. 2025 rec_time on 2020 announcements) that must not gate PIT visibility.
TEXT_TIME_PLAUSIBLE_BEFORE_DAYS = 1.0
TEXT_TIME_PLAUSIBLE_AFTER_DAYS = 3.0

def augment_text_frame(df: pd.DataFrame, spec: TextDataset) -> pd.DataFrame:
    df = df.copy()
    if "available_at" not in df.columns:
        df["available_at"] = ""
        df["available_at_rule"] = "missing_source_time"
    date_value = None
    if spec.date_column and spec.date_column in df.columns:
        date_value = df[spec.date_column].astype(str).str.strip()
    implausible = pd.Series(False, index=df.index)
    if spec.time_column and spec.time_column in df.columns:
        source_time = df[spec.time_column].map(normalize_source_datetime)
        mask = source_time.ne("") & source_time.ne("nan") & source_time.ne("None")
        if date_value is not None:
            base = pd.to_datetime(date_value, format="%Y%m%d", errors="coerce")
            parsed = pd.to_datetime(source_time.str.slice(0, 19), errors="coerce")
            lag_days = (parsed - base).dt.total_seconds() / 86400.0
            plausible = lag_days.between(-TEXT_TIME_PLAUSIBLE_BEFORE_DAYS, TEXT_TIME_PLAUSIBLE_AFTER_DAYS)
            implausible = mask & base.notna() & ~plausible.fillna(False)
            mask = mask & (base.isna() | plausible.fillna(False))
        df.loc[mask, "available_at"] = source_time[mask]
        df.loc[mask, "available_at_rule"] = f"source:{spec.time_column}"
        df.loc[implausible, "available_at"] = ""
    if date_value is not None:
        mask = (df["available_at"].astype(str).str.strip() == "") & date_value.str.fullmatch(r"\d{8}", na=False)
        suffix = " 22:00:00+08:00" if spec.api_name == "report_rc" else " 23:59:59+08:00"
        fallback = date_value.str.slice(0, 4) + "-" + date_value.str.slice(4, 6) + "-" + date_value.str.slice(6, 8) + suffix
        df.loc[mask, "available_at"] = fallback[mask]
        df.loc[mask, "available_at_rule"] = f"conservative_from:{spec.date_column}"
        df.loc[mask & implausible, "available_at_rule"] = (
            f"conservative_from:{spec.date_column}:implausible_{spec.time_column}"
        )
    return df

def refresh_sidecar_commit_identity(path: Path) -> None:
    """Restamp an in-place local repair with one footer/sidecar UUID."""
    migrate_partition_identity(path)


def repair_text_available_at(raw_dir: str, datasets: list[str]) -> dict[str, Any]:
    """Re-derive available_at for existing text partitions under the current rule.

    Pure local rewrite of the two derived columns; source fields and sidecars
    stay untouched.
    """
    stats: dict[str, Any] = {"datasets": {}, "files_rewritten": 0, "rows_changed": 0}
    for dataset in datasets:
        spec = TEXT_SPECS.get(dataset)
        if spec is None:
            raise RuntimeError(f"unknown text dataset: {dataset}")
        dataset_dir = Path(raw_dir) / dataset
        if not dataset_dir.exists():
            raise RuntimeError(f"missing dataset directory: {dataset_dir}")
        files = 0
        changed_rows = 0
        for path in sorted(dataset_dir.rglob("*.parquet")):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            before = frame.get("available_at", pd.Series("", index=frame.index)).astype(str)
            repaired = augment_text_frame(
                frame.drop(columns=[c for c in ("available_at", "available_at_rule") if c in frame.columns]),
                spec,
            )
            delta = int((repaired["available_at"].astype(str) != before).sum())
            if delta:
                tmp = path.with_suffix(path.suffix + ".tmp")
                repaired.to_parquet(tmp, index=False)
                tmp.replace(path)
                refresh_sidecar_commit_identity(path)
                files += 1
                changed_rows += delta
        stats["datasets"][dataset] = {"files_rewritten": files, "rows_changed": changed_rows}
        stats["files_rewritten"] += files
        stats["rows_changed"] += changed_rows
    return stats


def add_raw_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", default="data/raw")

def add_macro_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-code", action="append", default=[], help="Global index ts_code for index_global; repeatable. Defaults to a curated major-index list.")
    parser.add_argument("--cn-index-code", action="append", default=[], help="A-share index ts_code for index_daily; repeatable. Defaults to the core benchmark set (上证指数/上证50/沪深300/中证500/中证1000/创业板指/科创50).")
    parser.add_argument("--fx-code", action="append", default=[], help="FX ts_code for fx_daily; repeatable. Defaults to USDCNH.FXCM.")
    parser.add_argument("--eco-country", action="append", default=[], help="Optional eco_cal country filter; repeatable. Omit for all countries.")
    parser.add_argument("--eco-currency", action="append", default=[], help="Optional eco_cal currency filter; repeatable. Omit for all currencies.")
    parser.add_argument("--eco-event", action="append", default=[], help="Optional eco_cal event fuzzy filter; repeatable. Omit for all events.")

def add_board_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kpl-tag", action="append", default=[], help="开盘啦榜单 tag; repeatable. Defaults to 涨停/炸板/跌停/自然涨停/竞价.")
    parser.add_argument("--ths-limit-type", action="append", default=[], help="同花顺涨跌停榜单 limit_type; repeatable. Defaults to all official THS pools.")
    parser.add_argument("--ths-hot-market", action="append", default=[], help="同花顺热榜 market; repeatable. Defaults to 热股/行业板块/概念板块.")
    parser.add_argument("--dc-hot-market", action="append", default=[], help="东方财富热榜 market; repeatable. Defaults to A股市场.")
    parser.add_argument("--dc-hot-type", action="append", default=[], help="东方财富热榜 hot_type; repeatable. Defaults to 人气榜/飙升榜.")
    parser.add_argument("--hot-is-new", action="append", default=[], help="Hot-list is_new value Y or N; repeatable. Defaults to N for PIT rank_time snapshots.")

def add_runtime_args(parser: argparse.ArgumentParser, *, min_interval: float | None, timeout: int | None) -> None:
    parser.add_argument("--min-interval-seconds", type=float, default=min_interval)
    parser.add_argument("--timeout-seconds", type=int, default=timeout)

def add_intraday_by_date_common_args(
    parser: argparse.ArgumentParser,
    *,
    expected_codes_choices: list[str] | None = None,
    expected_codes_default: str = "none",
) -> None:
    expected_choices = expected_codes_choices or ["none", "daily", "active", "minute"]
    add_raw_arg(parser)
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--output-dataset", default=STK_MINS_BY_DATE_DATASET)
    parser.add_argument("--codes", nargs="+", help="Optional explicit ts_code list for window tests or targeted refreshes.")
    parser.add_argument("--max-codes", type=int, help="Optional first-N stock_basic/daily codes for window tests.")
    parser.add_argument(
        "--expected-codes-source",
        choices=expected_choices,
        default=expected_codes_default,
        help=f"Coverage universe for validation: {', '.join(expected_choices)}.",
    )
    parser.add_argument("--min-rows-per-day", type=int, default=0)
    parser.add_argument("--allow-missing-codes", type=int, default=0)
    parser.add_argument("--existing-allow-missing-codes", type=int, default=50)
