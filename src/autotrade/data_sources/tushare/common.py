"""What the TuShare ingest modules share: the run's exit codes and revision
ledger, the SSE trading calendar (read, and extended when a pull needs more of
it), dataset selection, date windows, the margin family's exchange rule,
partition names and the common CLI arguments.
"""

from __future__ import annotations

import argparse
import calendar
import re
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from autotrade.common.time import cn_today
from autotrade.environment.data.pit import concat_rows

from .client import TuShareClient, frame
from .io import read_many
from .specs import (
    BOARD_DC_HOT_MARKETS,
    BOARD_DC_HOT_TYPES,
    BOARD_HOT_IS_NEW,
    BOARD_KPL_TAGS,
    BOARD_THS_HOT_MARKETS,
    BOARD_THS_LIMIT_TYPES,
    BOARD_TRADING_DEFAULT_DATASETS,
    BOARD_TRADING_SPECS,
    DAILY_DOWNLOAD_DATASETS,
    DAILY_SPECS,
    DEFAULT_CN_INDEX_CODES,
    DEFAULT_FX_CODES,
    DEFAULT_GLOBAL_INDEX_CODES,
    EVENT_FLOW_DATASETS,
    EVENT_FLOW_SPECS,
    FUNDAMENTAL_DATASETS,
    FUNDAMENTAL_SPECS,
    NEWS_SOURCES,
    STK_MINS_BY_DATE_DATASET,
    TEXT_DATASETS,
    TEXT_SPECS,
)
from .versions import publish, record_defect

REVISION_EVENTS_PATH = "results/data_quality/revision_events.jsonl"


# Partition outcomes (versions.publish) that changed the visible partition.
WRITTEN_OUTCOMES = frozenset({"appended", "replaced", "restated"})


# Child-process contract: validation/polling ended before any raw write began.
# The cron runner may restore the previous committed generation and retry later.
NO_MUTATION_RETRY_EXIT_CODE = 75

# Child-process contract: the run failed before any raw write began, on a
# dataset whose own access is gone (the official-service datasets and
# OFFICIAL_TOKEN_ENV). Nothing changed, so the runner restores the previous
# committed generation instead of fencing the whole lake, and records the job
# as an error: losing that access stops those datasets alone, loudly.
NO_MUTATION_FAILURE_EXIT_CODE = 77

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


def sse_calendar(raw_dir: Path) -> pd.DataFrame:
    """The lake's SSE trading calendar, the one reader of it in this package:
    ``cal_date`` (YYYYMMDD) and ``is_open`` (bool), rows without a parseable
    date dropped. Raises when the lake holds no usable calendar."""
    files = sorted((Path(raw_dir) / "trade_cal" / "exchange=SSE").glob("year=*.parquet"))
    if not files:
        raise RuntimeError(f"SSE trade_cal partitions are missing under {raw_dir}; run download --tier reference first")
    calendar = read_many(files, columns=["cal_date", "is_open"])
    if calendar.empty:
        raise RuntimeError(f"SSE trade_cal is empty under {raw_dir}; run download --tier reference first")
    calendar = pd.DataFrame({
        "cal_date": calendar["cal_date"].map(normalize_date_key),
        "is_open": calendar["is_open"].astype(str) == "1",
    })
    calendar = calendar[calendar["cal_date"] != ""]
    if calendar.empty:
        raise RuntimeError("SSE trade_cal has no parseable cal_date values; refresh reference trade_cal first")
    return calendar.reset_index(drop=True)

def load_sse_open_dates(raw_dir: Path, start_date: str, end_date: str, *, allow_empty: bool = False) -> list[str]:
    calendar = sse_calendar(raw_dir)
    available_min = str(calendar["cal_date"].min())
    available_max = str(calendar["cal_date"].max())
    if start_date < available_min or end_date > available_max:
        raise RuntimeError(
            f"SSE trade_cal covers {available_min}-{available_max}, not requested {start_date}-{end_date}; refresh reference trade_cal first"
        )
    mask = calendar["is_open"] & (calendar["cal_date"] >= start_date) & (calendar["cal_date"] <= end_date)
    dates = sorted(calendar.loc[mask, "cal_date"].tolist())
    if not dates and not allow_empty:
        raise RuntimeError(f"no SSE open dates found for {start_date}-{end_date}")
    return dates

def latest_sse_calendar_date(raw_dir: Path) -> str:
    return str(sse_calendar(raw_dir)["cal_date"].max())

def merge_trade_cal_partition(path: Path, refreshed: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return refreshed
    existing = pd.read_parquet(path)
    if existing.empty:
        return refreshed
    if refreshed.empty:
        return existing
    # A vendor flip of is_open/pretrade_date on an already-elapsed date would
    # retroactively shift every derived PIT stamp (T+2 visibility rules,
    # resolved cron end dates), so it must never merge silently. Refuse and
    # leave the decision to a human: verify against the exchange, then delete
    # the year partition to accept the correction.
    today = cn_today()
    checked = [column for column in ("is_open", "pretrade_date") if column in existing.columns and column in refreshed.columns]
    if checked:
        # Each partition file is single-exchange by construction (exchange=
        # directory), so cal_date alone identifies the day.
        old = existing[existing["cal_date"].astype(str) <= today].set_index("cal_date")[checked]
        new = refreshed[refreshed["cal_date"].astype(str) <= today].set_index("cal_date")[checked]
        joined = old.join(new, how="inner", lsuffix="_old", rsuffix="_new")
        changed = pd.Series(False, index=joined.index)
        for column in checked:
            changed |= joined[f"{column}_old"].fillna("").astype(str) != joined[f"{column}_new"].fillna("").astype(str)
        if bool(changed.any()):
            raise RuntimeError(
                f"trade_cal {path} revises {int(changed.sum())} already-elapsed calendar day(s) "
                f"(sample: {sorted(joined.index[changed].tolist())[:5]}); refusing to merge silently -- "
                "verify against the exchange and delete the year partition to accept the correction"
            )
    merged = concat_rows([existing, refreshed])
    return (
        merged.drop_duplicates(["exchange", "cal_date"], keep="last")
        .sort_values(["exchange", "cal_date"], ascending=[True, False])
        .reset_index(drop=True)
    )

def download_trade_cal(
    client: TuShareClient, raw_dir: Path, start_date: str, end_date: str, force: bool, *, ledger: Path | str | None
) -> list[str]:
    fields = "exchange,cal_date,is_open,pretrade_date"
    sse_open: set[str] = set()
    for exchange in ("SSE", "SZSE", "BSE"):
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
            path = raw_dir / "trade_cal" / f"exchange={exchange}" / f"year={year}.parquet"
            params = {"exchange": exchange, "start_date": max(start_date, f"{year}0101"), "end_date": min(end_date, f"{year}1231")}
            if path.exists() and not force:
                df = pd.read_parquet(path)
                dates = df["cal_date"].map(normalize_date_key) if "cal_date" in df.columns else pd.Series(dtype=str)
                dates = dates[dates != ""]
                if not dates.empty and dates.min() <= params["start_date"] and dates.max() >= params["end_date"]:
                    if exchange == "SSE" and not df.empty:
                        sse_open.update(
                            date_key
                            for date_key in df.loc[df["is_open"].astype(str) == "1", "cal_date"].map(normalize_date_key).tolist()
                            if date_key
                        )
                    continue
            merging = path.exists()
            result = client.query("trade_cal", params, fields)
            answer = frame(result)
            df = merge_trade_cal_partition(path, answer)
            if answer.empty and not df.empty:
                # The year already holds days, so an empty answer is the
                # vendor's fault: the partition keeps them.
                record_defect(path, f"trade_cal params={params} returned zero rows for an existing partition", ledger=ledger, source="force_refresh")
            else:
                publish(
                    path,
                    df,
                    api_name="trade_cal",
                    params={**params, "merge_existing": True} if merging else params,
                    fields=fields.split(",") if merging else result.fields,
                    key_columns=(),
                    ledger=ledger,
                )
            if exchange == "SSE" and not df.empty:
                sse_open.update(
                    date_key
                    for date_key in df.loc[df["is_open"].astype(str) == "1", "cal_date"].map(normalize_date_key).tolist()
                    if date_key
                )
    return sorted(sse_open)

def sse_trade_cal_covers(raw_dir: Path, start_date: str, end_date: str) -> bool:
    try:
        dates = sse_calendar(raw_dir)["cal_date"]
    except RuntimeError:
        return False
    return bool(dates.min() <= start_date and dates.max() >= end_date)

def ensure_trade_cal_coverage(
    client: TuShareClient, raw_dir: Path, start_date: str, end_date: str, *, ledger: Path | str | None
) -> bool:
    """Extend local SSE trade_cal when it lags; True means the lake was written."""
    if parse_yyyymmdd(start_date) > parse_yyyymmdd(end_date):
        return False
    if sse_trade_cal_covers(raw_dir, start_date, end_date):
        return False
    print(f"refreshing trade_cal coverage for {start_date}-{end_date}")
    download_trade_cal(client, raw_dir, start_date, end_date, force=False, ledger=ledger)
    return True

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

def selected_fundamental_datasets(values: Iterable[str] | None) -> list[str]:
    return select_datasets(values, default=FUNDAMENTAL_DATASETS, allowed=FUNDAMENTAL_SPECS, label="fundamental")

def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()  # noqa: DTZ007 - calendar date, no time of day

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

def year_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Calendar-year windows of ``start_date``..``end_date``: (start, end, year)."""
    return [
        (max(start_date, f"{year}0101"), min(end_date, f"{year}1231"), str(year))
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
    ]

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


# What a repeatable filter option (add_macro_filter_args, add_board_filter_args)
# means when it is not given; None is "no filter".
CLI_LIST_DEFAULTS: dict[str, list[str] | list[None]] = {
    "index_code": DEFAULT_GLOBAL_INDEX_CODES,
    "cn_index_code": DEFAULT_CN_INDEX_CODES,
    "fx_code": DEFAULT_FX_CODES,
    "eco_country": [None],
    "eco_currency": [None],
    "eco_event": [None],
    "kpl_tag": BOARD_KPL_TAGS,
    "ths_limit_type": BOARD_THS_LIMIT_TYPES,
    "ths_hot_market": BOARD_THS_HOT_MARKETS,
    "dc_hot_market": BOARD_DC_HOT_MARKETS,
    "dc_hot_type": BOARD_DC_HOT_TYPES,
    "hot_is_new": BOARD_HOT_IS_NEW,
}

def cli_values(args: argparse.Namespace, attr: str) -> list:
    """The non-blank values given for the repeatable option ``attr``, or its
    default (CLI_LIST_DEFAULTS) when none is given."""
    values = [str(value).strip() for value in getattr(args, attr, None) or [] if str(value).strip()]
    return values or list(CLI_LIST_DEFAULTS[attr])


def selected_event_flow_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(args.datasets, default=EVENT_FLOW_DATASETS, allowed=EVENT_FLOW_SPECS, label="event/flow")

def selected_board_trading_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(getattr(args, "datasets", None), default=BOARD_TRADING_DEFAULT_DATASETS, allowed=BOARD_TRADING_SPECS, label="board-trading")

def selected_board_hot_is_new(args: argparse.Namespace) -> list[str]:
    values = [value.upper() for value in cli_values(args, "hot_is_new")]
    invalid = sorted(set(values) - {"Y", "N"})
    if invalid:
        raise RuntimeError(f"invalid hot is_new values: {invalid}; supported=['Y', 'N']")
    return values


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


def selected_text_datasets(
    values: Iterable[str] | None,
    *,
    news_src: list[str] | None = None,
    default: Iterable[str] | None = None,
) -> list[str]:
    datasets = select_datasets(
        values,
        default=TEXT_DATASETS if default is None else default,
        allowed=TEXT_SPECS,
        label="text",
    )
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


def safe_partition_value(value: str) -> str:
    """The single authority for raw-lake partition directory names.

    Percent-encoding is reversible and injective: distinct vendor values map to
    distinct directories and every directory decodes back to its value, so the
    downloader, the skip check and the audit's expected-path builder all resolve
    the same tree. It is also what the lake already holds for the Chinese board
    labels (`market=%E7%83%AD%E8%82%A1`, `tag=%E6%B6%A8%E5%81%9C`, ...).

    ASCII values that need no escaping (ts_code, exchange, news src, `all`) keep
    their plain names, so only the CJK partitions are affected by this rule.
    """
    text = str(value).strip()
    if not text:
        return "empty"
    return quote(text, safe="")


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
    parser.add_argument("--end-date", default=cn_today())
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
