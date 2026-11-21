"""The TuShare dataset registry: every dataset's spec (how it is fetched, keyed,
stamped and audited), the family lists and the default filter values that the
downloader, the audit and the cron runner share.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Raw-lake research contract values are owned by the environment's PIT layer;
# the ingest adapter imports them so producer and consumer cannot drift.
from autotrade.environment.data.contracts import (
    BENCHMARK_INDEXES,
    BOARD_TRADING_DATASETS,
)

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

INTRADAY_DATASETS = ["stk_mins_1min"]

BOARD_TRADING_DEFAULT_DATASETS = list(BOARD_TRADING_DATASETS)

BOARD_KPL_TAGS = ["涨停", "炸板", "跌停", "自然涨停", "竞价"]

BOARD_THS_LIMIT_TYPES = ["涨停池", "连扳池", "冲刺涨停", "炸板池", "跌停池"]

BOARD_THS_HOT_MARKETS = ["热股", "行业板块", "概念板块"]

BOARD_DC_HOT_MARKETS = ["A股市场"]

BOARD_DC_HOT_TYPES = ["人气榜", "飙升榜"]

BOARD_HOT_IS_NEW = ["N"]

SHARE_FLOAT_FIELDS = "ts_code,ann_date,float_date,float_share,float_ratio,holder_name,share_type"


# The research-history floor: the earliest date the lake keeps for the
# datasets this repository, not the vendor, clamps -- index_weight, the macro
# range files (quarter_once/month_once, whose canonical file therefore never
# shrinks with a rolling cron window) and the board-trading tier. A backfill
# run lowers it for the datasets it downloads with `download --history-floor`
# (`history_floor`); the history it lands then carries itself: a range file is
# never rewritten from a later start than the one it holds
# (`retained_range_start`), and the index_weight audit expects every year from
# the earliest one on disk. A range file already on disk is never moved to an
# earlier start either (`layout.refuse_moved_range_start` refuses it).
RESEARCH_HISTORY_FLOOR = "20200101"

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

# Core A-share benchmark indexes for `index_daily` and `index_weight`: market
# timing, beta management and relative-strength context for the Agent, plus the
# host-side benchmark an experiment's `benchmark_index` selects. The environment owns
# the table (BENCHMARK_INDEXES), so what is downloaded and what a create
# request may name are the same set.
DEFAULT_CN_INDEX_CODES = list(BENCHMARK_INDEXES)

# The index_weight source clamps every call to 7,000 rows regardless of the
# requested limit and rejects offsets beyond ~150k, so a whole-history pull
# per index truncates silently (most-recent-first). Downloads page inside
# per-year windows per index code (largest universe ~28k rows/year).
INDEX_WEIGHT_PAGE_LIMIT = 7000
INDEX_WEIGHT_FIELDS = "index_code,con_code,trade_date,weight"

DEFAULT_FX_CODES = ["USDCNH.FXCM"]

TRADE_DATE_PAGE_LIMIT = 5000

# Catalog-scale reference pulls must page: a single unpaged call silently
# truncates at whatever cap the vendor applies (index_basic sat at exactly
# 8,000 of 12,484 rows). The limit only needs to stay at or below every real
# cap; short pages end pagination naturally.
REFERENCE_PAGE_LIMIT = 5000

STK_MINS_API_NAME = "stk_mins"

STK_MINS_DATASET = "stk_mins_1min"

STK_MINS_BY_DATE_DATASET = "stk_mins_1min_by_date"

STK_MINS_FREQ = "1min"

STK_MINS_FIELDS = "ts_code,trade_time,open,high,low,close,vol,amount"

STK_MINS_PAGE_LIMIT = 8000

# stk_mins carries a per-token daily request quota (first hit 2026-09-14 CST
# after ~17.5k single-code requests); the vendor's own answer asks callers to
# put several codes in one request. A request ends at 15:00, so a code has at
# most 241 bars a session and 25 codes (6,025 rows) stay inside one 8,000-row
# page.
STK_MINS_BATCH_CODES = 25

# The quota answer is permanent until the vendor's next day: retrying it only
# burns the run, so the minute loop stops on it.
STK_MINS_QUOTA_MARKER = "额定额度已用完"

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
    "shibor_lpr": "https://tushare.pro/document/2?doc_id=151",
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
    "irm_qa_sh": "https://tushare.pro/document/2?doc_id=366",
    "irm_qa_sz": "https://tushare.pro/document/2?doc_id=367",
    "ths_daily": "https://tushare.pro/document/2?doc_id=260",
    "broker_recommend": "https://tushare.pro/document/2?doc_id=337",
    "top10_holders": "https://tushare.pro/document/2?doc_id=61",
    "top10_floatholders": "https://tushare.pro/document/2?doc_id=62",
    "pledge_detail": "https://tushare.pro/document/2?doc_id=111",
    "stk_surv": "https://tushare.pro/document/2?doc_id=275",
    "new_share": "https://tushare.pro/document/2?doc_id=123",
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

# Margin summary/detail for T publish at 09:00 on the next OPEN day: on a
# non-trading day the vendor answers with SSE rows only, and the SZSE/BSE rows
# arrive with the next session's pre-open (the writer never commits a partial
# day), so a Friday or pre-holiday session is not visible over the break.
OFFICIAL_NEXT_TRADING_DAY_09 = AvailabilityRule(
    "09:00:00", "official_next_trading_day_09_from:trade_date", trading_day_offset=1
)
OFFICIAL_PREOPEN_09 = AvailabilityRule("09:00:00", "official_preopen_09_from:trade_date")
OFFICIAL_19 = AvailabilityRule("19:00:00", "official_19_from:trade_date")
OFFICIAL_NEXT_DAY_0830 = AvailabilityRule("08:30:00", "official_next_day_0830_from:trade_date", day_offset=1)
OFFICIAL_20 = AvailabilityRule("20:00:00", "official_20_from:trade_date")
OFFICIAL_LATEST_2230 = AvailabilityRule("22:30:00", "official_latest_2230_from:trade_date", only_when_is_new=True)
# Shared limit-list family rule (limit_list_d + limit_list_ths): both sources
# publish the day's full 涨跌停/炸板 list around 16:00 on the trade date.
OFFICIAL_16 = AvailabilityRule("16:00:00", "official_16_from:trade_date")


class FetchStrategy(StrEnum):
    """How a dataset's requests cover its history (one request per trade
    date, per report period, per month window, ...). The download and audit
    code of the spec's family dispatches on it, and an audit report records
    it by value."""

    # The per-trade-date daily family, whose one way needs no name.
    DAILY_FAMILY = ""
    TRADE_DATE = "trade_date"
    TRADE_DATE_BY_TAG = "trade_date_by_tag"
    TRADE_DATE_BY_LIMIT_TYPE = "trade_date_by_limit_type"
    TRADE_DATE_BY_MARKET = "trade_date_by_market"
    TRADE_DATE_BY_MARKET_HOT_TYPE = "trade_date_by_market_hot_type"
    PERIOD = "period"
    ANN_MONTH = "ann_month"
    TS_CODE = "ts_code"
    DAY = "day"
    RANGE_MONTH = "range_month"
    TIME_RANGE_MONTH = "time_range_month"
    NEWS_SRC_DAY = "news_src_day"
    MONTH_LOOP = "month_loop"
    MONTH_ONCE = "month_once"
    QUARTER_ONCE = "quarter_once"
    DATE_YEAR = "date_year"
    DATE_YEAR_BY_TS_CODE = "date_year_by_ts_code"
    STATIC_FULL = "static_full"
    ECO_CAL_MONTH = "eco_cal_month"


@dataclass(frozen=True)
class DatasetSpec:
    """How one vendor dataset is fetched, keyed and audited.

    One type for every family. A family's registry (``DAILY_SPECS``,
    ``FUNDAMENTAL_SPECS``, ``TEXT_SPECS``, ``MACRO_SPECS``,
    ``EVENT_FLOW_SPECS``, ``BOARD_TRADING_SPECS`` and their kin) decides which
    download and audit code reads a spec, and that code reads only the fields
    its family sets; the rest keep their defaults.
    """

    api_name: str
    strategy: FetchStrategy = FetchStrategy.DAILY_FAMILY
    key_columns: tuple[str, ...] = ("trade_date", "ts_code")
    fields: str = ""
    # Measured per-call cap for sources that silently truncate large requests
    # (spec_page_limit clamps the tier default to it); None = no known cap.
    page_limit: int | None = None
    start_date: str = "20100101"
    zero_rows_ok: bool = True
    date_column: str = ""
    time_column: str = ""
    # ---- daily family ----
    # Source-side page overlap can repeat identical rows (observed on the
    # auction endpoints); dropping EXACT duplicates is information-preserving.
    dedup_exact: bool = False
    # ---- fundamental family ----
    period_param: str = "period"
    # ---- text family ----
    # Fetched from the official service with OFFICIAL_TOKEN_ENV instead of the
    # relay (the relay's permission for it lapsed).
    official: bool = False
    # time_column is when the row's whole text was published, at any lag after
    # date_column: it binds without the plausibility cap, and a row without it
    # is never visible (augment_text_frame).
    source_time_binds: bool = False
    # ---- text and event families ----
    # The business key names one record, so duplicate keys in an answer are
    # artefacts and are dropped deterministically at ingest (versions.publish).
    # Off where rows legitimately share a key (block_trade repeats identical
    # trades).
    unique_key: bool = False
    # ---- macro family ----
    month_param: str = "m"  # month_loop query param name (broker_recommend uses "month")
    # static_full registry pulls: one canonical file per loop value
    # (e.g. exchange=CFFEX.parquet), or a single full.parquet when empty.
    loop_param: str = ""
    loop_values: tuple[str, ...] = ()
    # Optional per-loop-value start dates (parallel to loop_values) for
    # trade_date loops whose venues listed at different times.
    loop_start_dates: tuple[str, ...] = ()
    # ---- event and board-trading families ----
    availability: AvailabilityRule | None = None
    # ---- event family ----
    fallback_date_column: str = ""
    # The column the supplier's start_date/end_date filter applies to, when it
    # is not date_column. A range_month refresh replaces exactly the rows its
    # request re-serves, so the merge must drop by this column: dropping by
    # ann_date while the API filters on end_date silently lost every row whose
    # report period precedes a mid-month window start.
    window_date_column: str = ""
    # How far a report period's disclosures trail it, when window_date_column
    # is a report period: a run's window starts this many days earlier, so the
    # rows announced inside its lookback are requested whatever their period.
    # Without it a 30-day nightly window never re-requested a quarter once 30
    # days had passed, and the 2026-06 top-10 tables held 444 of 59,204 rows.
    report_lag_days: int = 0

    def __post_init__(self) -> None:
        # An undeclared strategy is refused when the registry is built, not
        # at the first download or audit dispatch that reaches it.
        object.__setattr__(self, "strategy", FetchStrategy(self.strategy))

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

FUNDAMENTAL_SPECS = {
    "income_vip": DatasetSpec(
        api_name="income_vip",
        strategy=FetchStrategy.PERIOD,
        key_columns=("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    ),
    "balancesheet_vip": DatasetSpec(
        api_name="balancesheet_vip",
        strategy=FetchStrategy.PERIOD,
        key_columns=("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
    ),
    "cashflow_vip": DatasetSpec(
        api_name="cashflow_vip",
        strategy=FetchStrategy.PERIOD,
        key_columns=("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"),
        # Measured per-call cap (probed 2026-08-10): a 7,000-row request
        # returns at most 6,400 rows, a silently truncated "short" page that
        # stops query_paged (23 periods were pinned at exactly 6,400). 6,000
        # pages safely; income_vip/balancesheet_vip paginate correctly at the
        # 7,000 tier default.
        page_limit=6000,
    ),
    "fina_indicator_vip": DatasetSpec(
        api_name="fina_indicator_vip",
        strategy=FetchStrategy.PERIOD,
        key_columns=("ts_code", "ann_date", "end_date"),
    ),
    "forecast_vip": DatasetSpec(
        api_name="forecast_vip",
        strategy=FetchStrategy.ANN_MONTH,
        key_columns=("ts_code", "ann_date", "end_date", "type", "first_ann_date", "update_flag"),
    ),
    "express_vip": DatasetSpec(
        api_name="express_vip",
        strategy=FetchStrategy.ANN_MONTH,
        key_columns=("ts_code", "ann_date", "end_date"),
    ),
    "dividend": DatasetSpec(
        api_name="dividend",
        strategy=FetchStrategy.TS_CODE,
        key_columns=("ts_code", "end_date", "ann_date", "div_proc", "record_date", "ex_date", "pay_date"),
    ),
    "fina_audit": DatasetSpec(
        api_name="fina_audit",
        strategy=FetchStrategy.TS_CODE,
        key_columns=("ts_code", "ann_date", "end_date"),
    ),
    "fina_mainbz_vip": DatasetSpec(
        api_name="fina_mainbz_vip",
        strategy=FetchStrategy.TS_CODE,
        key_columns=("ts_code", "end_date", "bz_item", "bz_code", "curr_type"),
    ),
    "disclosure_date": DatasetSpec(
        api_name="disclosure_date",
        strategy=FetchStrategy.PERIOD,
        key_columns=("ts_code", "end_date", "ann_date", "pre_date", "actual_date"),
        period_param="end_date",
    ),
}

FUNDAMENTAL_DATASETS = list(FUNDAMENTAL_SPECS)

# The relay token lost every text interface but ``report_rc`` on 2026-09-10
# (each answers "没有接口访问权限"; ``major_news`` only inside a
# 40-request/day trial). The others are fetched from the official service
# (``official``): ``anns_d`` since 2026-10-06, the other seven since 2026-11.
# The official answers carry the same fields and business keys as the relay
# copies they continue (logs/data/anns_official_20261006/,
# logs/research_audit_20261102/stopped_datasets/).
TEXT_SPECS = {
    "anns_d": DatasetSpec(
        api_name="anns_d",
        official=True,
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ann_date,ts_code,name,title,url,rec_time",
        page_limit=2000,
        key_columns=("ann_date", "ts_code", "title", "url"),
        start_date="20100101",
        time_column="rec_time",
        date_column="ann_date",
        unique_key=True,
    ),
    "major_news": DatasetSpec(
        api_name="major_news",
        official=True,
        strategy=FetchStrategy.TIME_RANGE_MONTH,
        fields="title,pub_time,src,content",
        page_limit=400,
        key_columns=("pub_time", "src", "title"),
        start_date="20170101",
        time_column="pub_time",
    ),
    "cctv_news": DatasetSpec(
        api_name="cctv_news",
        official=True,
        strategy=FetchStrategy.DAY,
        fields="date,title,content",
        key_columns=("date", "title"),
        start_date="20170101",
        date_column="date",
    ),
    "npr": DatasetSpec(
        api_name="npr",
        official=True,
        strategy=FetchStrategy.RANGE_MONTH,
        fields="pubtime,title,pcode,puborg,ptype,url,content_html",
        page_limit=500,
        key_columns=("pubtime", "pcode", "title"),
        start_date="20100101",
        time_column="pubtime",
    ),
    "research_report": DatasetSpec(
        api_name="research_report",
        official=True,
        strategy=FetchStrategy.RANGE_MONTH,
        fields="trade_date,abstr,title,report_type,author,name,ts_code,inst_csname,ind_name,url",
        page_limit=1000,
        key_columns=("trade_date", "report_type", "title", "inst_csname", "ts_code"),
        start_date="20170101",
        date_column="trade_date",
    ),
    "report_rc": DatasetSpec(
        api_name="report_rc",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,name,report_date,report_title,report_type,classify,org_name,author_name,quarter,op_rt,op_pr,tp,np,eps,pe,rd,roe,ev_ebitda,rating,max_price,min_price,imp_dg,create_time",
        page_limit=3000,
        key_columns=("ts_code", "report_date", "report_title", "org_name", "author_name", "quarter"),
        start_date="20100101",
        time_column="create_time",
        date_column="report_date",
    ),
    # Investor Q&A. SH: trade_date is the day the reply appeared and pub_time the
    # vendor's crawl time, so the date rule holds. SZ: trade_date is the day the
    # question was asked and pub_time when its reply was published, so it binds.
    "irm_qa_sh": DatasetSpec(
        api_name="irm_qa_sh",
        official=True,
        strategy=FetchStrategy.DAY,
        fields="ts_code,name,trade_date,q,a,pub_time",
        key_columns=("trade_date", "ts_code", "q"),
        start_date="20220101",
        time_column="pub_time",
        date_column="trade_date",
    ),
    "irm_qa_sz": DatasetSpec(
        api_name="irm_qa_sz",
        official=True,
        strategy=FetchStrategy.DAY,
        fields="ts_code,name,trade_date,q,a,pub_time,industry",
        key_columns=("trade_date", "ts_code", "q"),
        start_date="20220101",
        time_column="pub_time",
        date_column="trade_date",
        source_time_binds=True,
    ),
    "news": DatasetSpec(
        api_name="news",
        official=True,
        strategy=FetchStrategy.NEWS_SRC_DAY,
        fields="datetime,content,title,channels",
        page_limit=1500,
        key_columns=("datetime", "title"),
        start_date="20180101",
        time_column="datetime",
    ),
}

TEXT_DATASETS = list(TEXT_SPECS)

MACRO_SPECS = {
    "cn_schedule": DatasetSpec(
        api_name="cn_schedule",
        strategy=FetchStrategy.MONTH_LOOP,
        fields="month,publish_date,title,issuing_org,data_api",
        # The source keeps no schedule rows before 2026 (observed rolling
        # retention, not a declared floor): months 202001-202512 always
        # return empty, so the floor stops re-querying and expecting them.
        start_date="20260101",
        key_columns=("month", "publish_date", "title", "issuing_org", "data_api"),
        date_column="publish_date",
    ),
    "cn_gdp": DatasetSpec(
        api_name="cn_gdp",
        strategy=FetchStrategy.QUARTER_ONCE,
        fields="quarter,gdp,gdp_yoy,pi,pi_yoy,si,si_yoy,ti,ti_yoy",
        key_columns=("quarter",),
        date_column="quarter",
    ),
    "cn_cpi": DatasetSpec(
        api_name="cn_cpi",
        strategy=FetchStrategy.MONTH_ONCE,
        fields="month,nt_val,nt_yoy,nt_mom,nt_accu,town_val,town_yoy,town_mom,town_accu,cnt_val,cnt_yoy,cnt_mom,cnt_accu",
        key_columns=("month",),
        date_column="month",
    ),
    "cn_ppi": DatasetSpec(
        api_name="cn_ppi",
        strategy=FetchStrategy.MONTH_ONCE,
        fields="month,ppi_yoy,ppi_mp_yoy,ppi_mp_qm_yoy,ppi_mp_rm_yoy,ppi_mp_p_yoy,ppi_cg_yoy,ppi_cg_f_yoy,ppi_cg_c_yoy,ppi_cg_adu_yoy,ppi_cg_dcg_yoy,ppi_mom,ppi_mp_mom,ppi_mp_qm_mom,ppi_mp_rm_mom,ppi_mp_p_mom,ppi_cg_mom,ppi_cg_f_mom,ppi_cg_c_mom,ppi_cg_adu_mom,ppi_cg_dcg_mom,ppi_accu,ppi_mp_accu,ppi_mp_qm_accu,ppi_mp_rm_accu,ppi_mp_p_accu,ppi_cg_accu,ppi_cg_f_accu,ppi_cg_c_accu,ppi_cg_adu_accu,ppi_cg_dcg_accu",
        key_columns=("month",),
        date_column="month",
    ),
    "cn_pmi": DatasetSpec(
        api_name="cn_pmi",
        strategy=FetchStrategy.MONTH_ONCE,
        fields="month,pmi010000,pmi010400,pmi010500,pmi010900,pmi011000,pmi011200,pmi011300,pmi011600,pmi020100,pmi020200,pmi020300,pmi020400,pmi020600,pmi030000",
        key_columns=("month",),
        date_column="month",
    ),
    "cn_m": DatasetSpec(
        api_name="cn_m",
        strategy=FetchStrategy.MONTH_ONCE,
        fields="month,m0,m0_yoy,m0_mom,m1,m1_yoy,m1_mom,m2,m2_yoy,m2_mom",
        key_columns=("month",),
        date_column="month",
    ),
    "sf_month": DatasetSpec(
        api_name="sf_month",
        strategy=FetchStrategy.MONTH_ONCE,
        fields="month,inc_month,inc_cumval,stk_endval",
        key_columns=("month",),
        date_column="month",
    ),
    "shibor": DatasetSpec(
        api_name="shibor",
        strategy=FetchStrategy.DATE_YEAR,
        fields="date,on,1w,2w,1m,3m,6m,9m,1y",
        key_columns=("date",),
        date_column="date",
    ),
    "shibor_quote": DatasetSpec(
        api_name="shibor_quote",
        strategy=FetchStrategy.DATE_YEAR,
        fields="date,bank,on_b,on_a,1w_b,1w_a,2w_b,2w_a,1m_b,1m_a,3m_b,3m_a,6m_b,6m_a,9m_b,9m_a,1y_b,1y_a",
        # Measured per-call cap; a larger request returns one silently
        # truncated "short" page and query_paged stops after it.
        page_limit=4000,
        key_columns=("date", "bank"),
        date_column="date",
    ),
    "shibor_lpr": DatasetSpec(
        api_name="shibor_lpr",
        strategy=FetchStrategy.DATE_YEAR,
        fields="date,1y,5y",
        key_columns=("date",),
        date_column="date",
    ),
    "repo_daily": DatasetSpec(
        api_name="repo_daily",
        strategy=FetchStrategy.DATE_YEAR,
        fields="ts_code,trade_date,repo_maturity,pre_close,open,high,low,close,weight,weight_r,amount,num",
        # Measured per-call cap (see shibor_quote note).
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "us_tycr": DatasetSpec(
        api_name="us_tycr",
        strategy=FetchStrategy.DATE_YEAR,
        # m4 (the 4-month CMT) has never been returned by the vendor for any
        # year in the lake; requesting it only trips the schema audit.
        fields="date,m1,m2,m3,m6,y1,y2,y3,y5,y7,y10,y20,y30",
        key_columns=("date",),
        date_column="date",
    ),
    "us_trycr": DatasetSpec(
        api_name="us_trycr",
        strategy=FetchStrategy.DATE_YEAR,
        fields="date,y5,y7,y10,y20,y30",
        key_columns=("date",),
        date_column="date",
    ),
    "us_tbr": DatasetSpec(
        api_name="us_tbr",
        strategy=FetchStrategy.DATE_YEAR,
        # w17_bd/w17_ce (the 17-week tenor) have never been returned by the
        # vendor for any year in the lake.
        fields="date,w4_bd,w4_ce,w8_bd,w8_ce,w13_bd,w13_ce,w26_bd,w26_ce,w52_bd,w52_ce",
        key_columns=("date",),
        date_column="date",
    ),
    "us_tltr": DatasetSpec(
        api_name="us_tltr",
        strategy=FetchStrategy.DATE_YEAR,
        fields="date,ltc,cmt,e_factor",
        key_columns=("date",),
        date_column="date",
    ),
    "index_daily": DatasetSpec(
        api_name="index_daily",
        strategy=FetchStrategy.DATE_YEAR_BY_TS_CODE,
        fields="ts_code,trade_date,open,close,high,low,pre_close,change,pct_chg,vol,amount",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "broker_recommend": DatasetSpec(
        api_name="broker_recommend",
        strategy=FetchStrategy.MONTH_LOOP,
        fields="month,broker,ts_code,name",
        start_date="20190101",
        key_columns=("month", "broker", "ts_code"),
        date_column="month",
        month_param="month",
    ),
    "ths_daily": DatasetSpec(
        api_name="ths_daily",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,open,high,low,close,pre_close,avg_price,change,pct_change,vol,turnover_rate",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "index_dailybasic": DatasetSpec(
        api_name="index_dailybasic",
        strategy=FetchStrategy.DATE_YEAR_BY_TS_CODE,
        fields="ts_code,trade_date,total_mv,float_mv,total_share,float_share,free_share,turnover_rate,turnover_rate_f,pe,pe_ttm,pb",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "sw_daily": DatasetSpec(
        api_name="sw_daily",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,name,open,low,high,close,change,pct_change,vol,amount,pe,pb,float_mv,total_mv",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "ci_daily": DatasetSpec(
        api_name="ci_daily",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,open,low,high,close,pre_close,change,pct_change,vol,amount",
        start_date="20190101",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "daily_info": DatasetSpec(
        api_name="daily_info",
        strategy=FetchStrategy.DATE_YEAR,
        fields="trade_date,ts_code,ts_name,com_count,total_share,float_share,total_mv,float_mv,amount,vol,trans_count,pe,tr,exchange",
        start_date="20190101",
        # Measured per-call cap (see shibor_quote note).
        page_limit=4000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "sz_daily_info": DatasetSpec(
        api_name="sz_daily_info",
        strategy=FetchStrategy.DATE_YEAR,
        fields="trade_date,ts_code,count,amount,vol,total_share,total_mv,float_share,float_mv",
        start_date="20190101",
        # Measured per-call cap (see shibor_quote note).
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "moneyflow_mkt_dc": DatasetSpec(
        api_name="moneyflow_mkt_dc",
        strategy=FetchStrategy.DATE_YEAR,
        fields="trade_date,close_sh,pct_change_sh,close_sz,pct_change_sz,net_amount,net_amount_rate,buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate",
        start_date="20190101",
        key_columns=("trade_date",),
        date_column="trade_date",
    ),
    "index_global": DatasetSpec(
        api_name="index_global",
        strategy=FetchStrategy.DATE_YEAR_BY_TS_CODE,
        fields="ts_code,trade_date,open,close,high,low,pre_close,change,pct_chg,swing,vol,amount",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "fx_daily": DatasetSpec(
        api_name="fx_daily",
        strategy=FetchStrategy.DATE_YEAR_BY_TS_CODE,
        fields="ts_code,trade_date,bid_open,bid_close,bid_high,bid_low,ask_open,ask_close,ask_high,ask_low,tick_qty",
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "eco_cal": DatasetSpec(
        api_name="eco_cal",
        strategy=FetchStrategy.ECO_CAL_MONTH,
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
    "fut_basic": DatasetSpec(
        api_name="fut_basic",
        strategy=FetchStrategy.STATIC_FULL,
        # ``trade_time_desc`` was dropped from the request on 2026-09-10: the
        # vendor stopped returning it and nothing reads it.
        fields="ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,d_month,list_date,delist_date,last_ddate",
        page_limit=10000,
        key_columns=("ts_code",),
        date_column="list_date",
        loop_param="exchange",
        loop_values=("CFFEX", "DCE", "CZCE", "SHFE", "INE", "GFEX"),
    ),
    "fut_mapping": DatasetSpec(
        api_name="fut_mapping",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,mapping_ts_code",
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "fut_daily": DatasetSpec(
        api_name="fut_daily",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,pre_close,pre_settle,open,high,low,close,settle,change1,change2,vol,amount,oi,oi_chg,delv_settle",
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
    ),
    "opt_basic": DatasetSpec(
        api_name="opt_basic",
        strategy=FetchStrategy.STATIC_FULL,
        fields="ts_code,symbol,exchange,name,per_unit,opt_code,opt_type,call_put,exercise_type,exercise_price,s_month,maturity_date,list_price,list_date,delist_date,last_edate,last_ddate,quote_unit,min_price_chg",
        page_limit=10000,
        key_columns=("ts_code",),
        date_column="list_date",
        # Financial exchanges only, matching the opt_daily scope: commodity
        # option registries add ~185k dead rows to every snapshot macro frame.
        loop_param="exchange",
        loop_values=("SSE", "SZSE", "CFFEX"),
    ),
    "opt_daily": DatasetSpec(
        api_name="opt_daily",
        strategy=FetchStrategy.TRADE_DATE,
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
    "cb_basic": DatasetSpec(
        api_name="cb_basic",
        strategy=FetchStrategy.STATIC_FULL,
        # conv_price/clauses are the fields the conversion-premium and
        # redemption-event signals need; stk_code links CB -> underlying stock.
        fields="ts_code,bond_short_name,cb_code,cb_type,stk_code,stk_short_name,maturity,par,issue_price,issue_size,remain_size,value_date,maturity_date,rate_type,coupon_rate,pay_per_year,list_date,delist_date,exchange,conv_start_date,conv_end_date,first_conv_price,conv_price,put_clause,maturity_call_price,call_clause,reset_clause,guarantor,issue_rating,newest_rating,rating_comp",
        page_limit=2000,
        key_columns=("ts_code",),
        date_column="list_date",
    ),
    "cb_daily": DatasetSpec(
        api_name="cb_daily",
        # bond_value/cb_value/cb_over_rate are non-default display fields and
        # must be requested explicitly.
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,bond_value,bond_over_rate,cb_value,cb_over_rate",
        page_limit=2000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20180102",
    ),
    "cb_call": DatasetSpec(
        api_name="cb_call",
        strategy=FetchStrategy.STATIC_FULL,
        fields="ts_code,call_type,is_call,ann_date,call_date,call_price,call_price_tax,call_vol,call_amount,payment_date,call_reg_date",
        page_limit=2000,
        # The staged is_call states of one bond are separate announcements.
        key_columns=("ts_code", "ann_date", "is_call"),
        date_column="ann_date",
    ),
}

MACRO_DATASETS = list(MACRO_SPECS)

EVENT_FLOW_SPECS = {
    "margin": DatasetSpec(
        api_name="margin",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,exchange_id,rzye,rzmre,rzche,rqye,rqmcl,rzrqye,rqyl",
        page_limit=4000,
        key_columns=("trade_date", "exchange_id"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_NEXT_TRADING_DAY_09,
    ),
    "margin_detail": DatasetSpec(
        api_name="margin_detail",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,name,rzye,rqye,rzmre,rqyl,rzche,rqchl,rqmcl,rzrqye",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_NEXT_TRADING_DAY_09,
    ),
    "margin_secs": DatasetSpec(
        api_name="margin_secs",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,name,exchange",
        page_limit=6000,
        key_columns=("trade_date", "ts_code", "exchange"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_PREOPEN_09,
    ),
    "moneyflow": DatasetSpec(
        api_name="moneyflow",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        zero_rows_ok=False,
        availability=OFFICIAL_19,
    ),
    "moneyflow_dc": DatasetSpec(
        api_name="moneyflow_dc",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,name,pct_change,close,net_amount,net_amount_rate,buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20231201",
        availability=OFFICIAL_19,
    ),
    "moneyflow_ths": DatasetSpec(
        api_name="moneyflow_ths",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,name,pct_change,latest,net_amount,net_d5_amount,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20250101",
        availability=OFFICIAL_19,
    ),
    "moneyflow_ind_dc": DatasetSpec(
        api_name="moneyflow_ind_dc",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,content_type,ts_code,name,pct_change,close,net_amount,net_amount_rate,buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate,buy_sm_amount_stock,rank",
        page_limit=5000,
        key_columns=("trade_date", "content_type", "ts_code"),
        date_column="trade_date",
        start_date="20231201",
        availability=OFFICIAL_19,
    ),
    "moneyflow_ind_ths": DatasetSpec(
        api_name="moneyflow_ind_ths",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,industry,lead_stock,close,pct_change,company_num,pct_change_stock,close_price,net_buy_amount,net_sell_amount,net_amount",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20250101",
        availability=OFFICIAL_19,
    ),
    "moneyflow_cnt_ths": DatasetSpec(
        api_name="moneyflow_cnt_ths",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,name,lead_stock,close_price,pct_change,industry_index,company_num,pct_change_stock,net_buy_amount,net_sell_amount,net_amount",
        page_limit=5000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20250101",
        availability=OFFICIAL_19,
    ),
    "cyq_perf": DatasetSpec(
        api_name="cyq_perf",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,his_low,his_high,cost_5pct,cost_15pct,cost_50pct,cost_85pct,cost_95pct,weight_avg,winner_rate",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20180102",
        availability=OFFICIAL_19,
    ),
    "bak_daily": DatasetSpec(
        api_name="bak_daily",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,name,pct_change,close,change,open,high,low,pre_close,vol_ratio,turn_over,swing,vol,amount,selling,buying,total_share,float_share,pe,industry,area,float_mv,total_mv,avg_price,strength,activity,avg_turnover,attack,interval_3,interval_6",
        page_limit=6000,
        key_columns=("trade_date", "ts_code"),
        date_column="trade_date",
        start_date="20170103",
        availability=OFFICIAL_19,
    ),
    "stk_holdernumber": DatasetSpec(
        api_name="stk_holdernumber",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,ann_date,end_date,holder_num",
        page_limit=3000,
        key_columns=("ts_code", "ann_date", "end_date"),
        date_column="ann_date",
        zero_rows_ok=True,
        unique_key=True,
    ),
    "top10_holders": DatasetSpec(
        api_name="top10_holders",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio,hold_float_ratio,hold_change,holder_type",
        page_limit=5000,
        key_columns=("ts_code", "ann_date", "end_date", "holder_name"),
        date_column="ann_date",
        # start_date/end_date filter the report period, not the announcement;
        # annual reports are due 120 days after the period ends.
        window_date_column="end_date",
        report_lag_days=120,
        unique_key=True,
    ),
    "top10_floatholders": DatasetSpec(
        api_name="top10_floatholders",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio,hold_float_ratio,hold_change,holder_type",
        page_limit=5000,
        key_columns=("ts_code", "ann_date", "end_date", "holder_name"),
        date_column="ann_date",
        # start_date/end_date filter the report period, not the announcement;
        # annual reports are due 120 days after the period ends.
        window_date_column="end_date",
        report_lag_days=120,
        unique_key=True,
    ),
    "pledge_detail": DatasetSpec(
        api_name="pledge_detail",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,ann_date,holder_name,pledge_amount,start_date,end_date,is_release,release_date,pledgor,holding_amount,pledged_amount,p_total_ratio,h_total_ratio,is_buyback",
        # Measured per-call cap (probed 2026-08-10): a 3,000-row request
        # returns at most 1,500 rows, a silently truncated "short" page that
        # stops query_paged (42/80 month partitions were pinned at 1,500).
        page_limit=1500,
        key_columns=("ts_code", "ann_date", "holder_name", "start_date", "pledgor"),
        date_column="ann_date",
    ),
    "stk_surv": DatasetSpec(
        api_name="stk_surv",
        strategy=FetchStrategy.TRADE_DATE,
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
    "new_share": DatasetSpec(
        api_name="new_share",
        strategy=FetchStrategy.RANGE_MONTH,
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
    "stk_holdertrade": DatasetSpec(
        api_name="stk_holdertrade",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,ann_date,holder_name,holder_type,in_de,change_vol,change_ratio,after_share,after_ratio,avg_price,total_share,begin_date,close_date",
        page_limit=3000,
        key_columns=("ts_code", "ann_date", "holder_name", "holder_type", "in_de", "begin_date", "close_date"),
        date_column="ann_date",
        zero_rows_ok=True,
        availability=AvailabilityRule("19:00:00", "official_19_from:ann_date"),
        unique_key=True,
    ),
    "repurchase": DatasetSpec(
        api_name="repurchase",
        strategy=FetchStrategy.RANGE_MONTH,
        fields="ts_code,ann_date,end_date,proc,exp_date,vol,amount,high_limit,low_limit",
        page_limit=2000,
        key_columns=("ts_code", "ann_date", "end_date", "proc", "exp_date"),
        date_column="ann_date",
        start_date="20110101",
        zero_rows_ok=True,
        unique_key=True,
    ),
    "share_float": DatasetSpec(
        api_name="share_float",
        strategy=FetchStrategy.DAY,
        fields=SHARE_FLOAT_FIELDS,
        page_limit=6000,
        key_columns=("ts_code", "ann_date", "float_date", "holder_name", "share_type"),
        date_column="ann_date",
        fallback_date_column="float_date",
        zero_rows_ok=True,
    ),
    "block_trade": DatasetSpec(
        api_name="block_trade",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,trade_date,price,vol,amount,buyer,seller",
        page_limit=1000,
        key_columns=("trade_date", "ts_code", "price", "vol", "amount", "buyer", "seller"),
        date_column="trade_date",
        zero_rows_ok=True,
        availability=AvailabilityRule("21:00:00", "official_21_from:trade_date"),
    ),
}

EVENT_FLOW_DATASETS = list(EVENT_FLOW_SPECS)

BOARD_TRADING_SPECS = {
    "kpl_list": DatasetSpec(
        api_name="kpl_list",
        strategy=FetchStrategy.TRADE_DATE_BY_TAG,
        fields="ts_code,name,trade_date,lu_time,ld_time,open_time,last_time,lu_desc,tag,theme,net_change,bid_amount,status,bid_change,bid_turnover,lu_bid_vol,pct_chg,bid_pct_chg,rt_pct_chg,limit_order,amount,turnover_rate,free_float,lu_limit_order",
        page_limit=8000,
        key_columns=("trade_date", "ts_code", "tag", "status", "lu_time", "open_time", "last_time"),
        availability=OFFICIAL_NEXT_DAY_0830,
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "limit_step": DatasetSpec(
        api_name="limit_step",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,name,trade_date,nums",
        page_limit=2000,
        key_columns=("trade_date", "ts_code", "nums"),
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "limit_cpt_list": DatasetSpec(
        api_name="limit_cpt_list",
        strategy=FetchStrategy.TRADE_DATE,
        fields="ts_code,name,trade_date,days,up_stat,cons_nums,up_nums,pct_chg,rank",
        page_limit=2000,
        key_columns=("trade_date", "ts_code", "rank"),
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "limit_list_d": DatasetSpec(
        api_name="limit_list_d",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,industry,name,close,pct_chg,amount,limit_amount,float_mv,total_mv,turnover_ratio,fd_amount,first_time,last_time,open_times,up_stat,limit_times,limit",
        page_limit=5000,
        key_columns=("trade_date", "ts_code", "limit"),
        availability=OFFICIAL_16,
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "limit_list_ths": DatasetSpec(
        api_name="limit_list_ths",
        strategy=FetchStrategy.TRADE_DATE_BY_LIMIT_TYPE,
        fields="trade_date,ts_code,name,price,pct_chg,open_num,lu_desc,limit_type,tag,status,limit_order,limit_amount,turnover_rate,free_float,lu_limit_order,limit_up_suc_rate,turnover,market_type,rise_rate,sum_float,first_lu_time,last_lu_time,first_ld_time,last_ld_time",
        page_limit=8000,
        # The four ``first_lu_time``/``last_ld_time`` columns are still
        # requested and stored but are NOT part of the identity: the vendor
        # stopped returning them on 2026-08-14, and over the whole table
        # (128,998 rows, 3,475 partitions) this five-column key has zero
        # duplicates. Keeping them in the key made every post-2026-08-14
        # partition read as "missing key columns", which also blinded the
        # revision guard to real key removals.
        key_columns=("trade_date", "ts_code", "limit_type", "tag", "status"),
        start_date="20231101",
        availability=OFFICIAL_16,
        date_column="trade_date",
    ),
    "top_list": DatasetSpec(
        api_name="top_list",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,name,close,pct_change,turnover_rate,amount,l_sell,l_buy,l_amount,net_amount,net_rate,amount_rate,float_values,reason",
        page_limit=10000,
        # Every field but ``name``: a stock can list several times a day, so
        # the figures stay in the key. ``name`` is a value: the vendor restated
        # it to later names on historical rows (2026-10), which as a key field
        # read as a removed plus an added row instead of a field revision.
        key_columns=(
            "trade_date",
            "ts_code",
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
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "top_inst": DatasetSpec(
        api_name="top_inst",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,exalter,buy,buy_rate,sell,sell_rate,net_buy,side,reason",
        page_limit=10000,
        key_columns=("trade_date", "ts_code", "exalter", "buy", "sell", "net_buy", "side", "reason"),
        zero_rows_ok=True,
        availability=OFFICIAL_20,
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "hm_list": DatasetSpec(
        api_name="hm_list",
        strategy=FetchStrategy.STATIC_FULL,
        fields="name,desc,orgs",
        page_limit=1000,
        key_columns=("name",),
        date_column="",
        start_date="20220801",
    ),
    "hm_detail": DatasetSpec(
        api_name="hm_detail",
        strategy=FetchStrategy.TRADE_DATE,
        fields="trade_date,ts_code,ts_name,buy_amount,sell_amount,net_amount,hm_name,hm_orgs",
        page_limit=10000,
        key_columns=("trade_date", "ts_code", "hm_name", "hm_orgs"),
        start_date="20220801",
        date_column="trade_date",
    ),
    "ths_hot": DatasetSpec(
        api_name="ths_hot",
        strategy=FetchStrategy.TRADE_DATE_BY_MARKET,
        fields="trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,hot,concept,rank_time,rank_reason",
        page_limit=2000,
        key_columns=("trade_date", "data_type", "ts_code", "rank_time", "rank"),
        time_column="rank_time",
        availability=OFFICIAL_LATEST_2230,
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
    "dc_hot": DatasetSpec(
        api_name="dc_hot",
        strategy=FetchStrategy.TRADE_DATE_BY_MARKET_HOT_TYPE,
        fields="trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,hot,concept,rank_time",
        page_limit=2000,
        key_columns=("trade_date", "data_type", "ts_code", "rank_time", "rank"),
        time_column="rank_time",
        availability=OFFICIAL_LATEST_2230,
        start_date=RESEARCH_HISTORY_FLOOR,
        date_column="trade_date",
    ),
}

BAK_BASIC_SPEC = DatasetSpec(
    api_name="bak_basic",
    fields=BAK_BASIC_FIELDS,
    start_date="20160101",
    zero_rows_ok=True,
    key_columns=("trade_date", "ts_code"),
)

DAILY_SPECS = {
    "daily": DatasetSpec(
        api_name="daily",
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        zero_rows_ok=False,
    ),
    "adj_factor": DatasetSpec(
        api_name="adj_factor",
        fields="ts_code,trade_date,adj_factor",
        zero_rows_ok=False,
    ),
    "daily_basic": DatasetSpec(
        api_name="daily_basic",
        fields="ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv",
        zero_rows_ok=False,
    ),
    "stk_limit": DatasetSpec(
        api_name="stk_limit",
        fields="trade_date,ts_code,pre_close,up_limit,down_limit",
        zero_rows_ok=False,
    ),
    "suspend_d": DatasetSpec(
        api_name="suspend_d",
        fields="ts_code,trade_date,suspend_timing,suspend_type",
        zero_rows_ok=True,
        key_columns=("trade_date", "ts_code", "suspend_type", "suspend_timing"),
    ),
    # Exact opening call-auction result. TuShare's usable history begins on
    # 2025-01-16; older replays retain the explicitly labelled minute proxy.
    "stk_auction": DatasetSpec(
        api_name="stk_auction",
        fields="ts_code,trade_date,vol,price,amount,pre_close,turnover_rate,volume_ratio,float_share",
        start_date="20250116",
        dedup_exact=True,
        zero_rows_ok=False,
    ),
}
