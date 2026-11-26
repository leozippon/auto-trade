#!/usr/bin/env python3
"""TuShare download, update, and raw-data maintenance CLI for AutoTrade."""

from __future__ import annotations

import argparse
import calendar
import json
import re
import time
from collections.abc import Callable
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any

import pandas as pd

from autotrade.common.records import read_json_object
from autotrade.common.time import CN_TZ, cn_today
from autotrade.environment.data.contracts import BOARD_TRADING_DATASETS
from autotrade.environment.data.pit import concat_rows, first_seen_at, key_strings

from . import versions
from .auction import (
    AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO,
    AUCTION_CAPTURE_MIN_ROWS_FLOOR,
    capture_open_auction,
    recheck_stk_auction,
)
from .client import (
    MIN_REQUEST_INTERVAL_SECONDS,
    OFFICIAL_TOKEN_ENV,
    ApiResult,
    TuShareClient,
    frame,
    load_token,
    official_client,
    query_paged,
)
from .common import (
    MUTATED_NOT_READY_RETRY_EXIT_CODE,
    NO_MUTATION_FAILURE_EXIT_CODE,
    NO_MUTATION_RETRY_EXIT_CODE,
    REVISION_EVENTS_PATH,
    WRITTEN_OUTCOMES,
    add_board_filter_args,
    add_intraday_by_date_common_args,
    add_macro_filter_args,
    add_raw_arg,
    add_runtime_args,
    date_range_days,
    download_trade_cal,
    ensure_trade_cal_coverage,
    format_yyyymmdd,
    latest_sse_calendar_date,
    load_sse_open_dates,
    load_stock_codes,
    margin_exchange_column,
    margin_family_missing_exchanges,
    month_windows,
    normalize_date_key,
    parse_yyyymmdd,
    quarter_periods,
    resolve_revision_ledger,
    safe_partition_value,
    select_datasets,
    selected_board_trading_datasets,
    selected_daily_datasets,
    selected_fundamental_datasets,
    selected_text_datasets,
    year_windows,
    yyyymmdd_to_month,
    yyyymmdd_to_quarter,
)
from .intraday import update_intraday_by_date
from .io import committed_partition_intact, parquet_rows
from .layout import (
    Partition,
    directory_name,
    partitions,
    range_partition,
    refuse_moved_range_start,
)
from .lock import manual_production_write
from .share_float import download_share_float_complete
from .specs import (
    BAK_BASIC_SPEC,
    BOARD_TRADING_SPECS,
    DAILY_DOWNLOAD_DATASETS,
    DAILY_SPECS,
    DEFAULT_CN_INDEX_CODES,
    DOWNLOAD_TIER_CHOICES,
    EVENT_FLOW_DATASETS,
    EVENT_FLOW_SPECS,
    FUNDAMENTAL_DATASETS,
    FUNDAMENTAL_SPECS,
    GLOBAL_CONTEXT_DEFAULT_DATASETS,
    INDEX_WEIGHT_FIELDS,
    INDEX_WEIGHT_PAGE_LIMIT,
    MACRO_DATASETS,
    MACRO_REGIME_DEFAULT_DATASETS,
    MACRO_SPECS,
    REFERENCE_DATASETS,
    REFERENCE_PAGE_LIMIT,
    RESEARCH_HISTORY_FLOOR,
    STK_MINS_BY_DATE_DATASET,
    TEXT_DATASETS,
    TEXT_SPECS,
    TRADE_DATE_PAGE_LIMIT,
    DatasetSpec,
    FetchStrategy,
)
from .stamps import (
    augment_board_frame,
    augment_event_frame,
    augment_macro_frame,
    augment_text_frame,
    repair_event_available_at,
    repair_text_available_at,
)
from .versions import DefectiveAnswer, answer_defects, publish, report_complete


def history_floor(floor: str, args: argparse.Namespace) -> str:
    """``floor`` lowered to a backfill run's ``--history-floor``, never raised."""

    lowered = getattr(args, "history_floor", None)
    return min(floor, lowered) if lowered else floor

def yyyymmdd_arg(value: str) -> str:
    """argparse type of a YYYYMMDD calendar date."""

    try:
        parse_yyyymmdd(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a YYYYMMDD date, got {value!r}") from None
    return value


def selected_macro_datasets(args: argparse.Namespace) -> list[str]:
    default = GLOBAL_CONTEXT_DEFAULT_DATASETS if args.tier == "global" else MACRO_REGIME_DEFAULT_DATASETS
    return select_datasets(args.datasets, default=default, allowed=MACRO_SPECS, label="macro/global")

def spec_page_limit(spec, requested: int | None) -> int:
    """Effective page size for one dataset spec: the caller's request clamped
    to the spec's documented limit; 10000 when the spec documents none."""
    if spec.page_limit is not None:
        return min(requested or spec.page_limit, spec.page_limit)
    return requested or 10000


def write_reference_query_result(
    path: Path,
    result: ApiResult,
    *,
    api_name: str,
    params: dict[str, Any],
    required_nonempty: bool,
    ledger: Path | str | None,
) -> pd.DataFrame:
    """Publish one unkeyed reference partition; an empty answer for a
    partition holding rows is a recorded defect that keeps it."""
    df = frame(result)
    if df.empty and required_nonempty and not path.exists():
        raise RuntimeError(f"{api_name} params={params} returned zero rows for required reference partition")
    if publish(path, df, api_name=api_name, params=params, fields=result.fields, key_columns=(), ledger=ledger) == "defect":
        return pd.read_parquet(path)
    return df


def download_reference(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    refresh_datasets = set(args.refresh_reference_datasets)
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    trade_cal_end_date = args.trade_cal_end_date or args.end_date

    def should_force(dataset: str) -> bool:
        return bool(args.force or dataset in refresh_datasets)

    stock_basic_fields = "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type"
    for status in ("L", "D", "P"):
        path = raw_dir / "stock_basic" / f"list_status={status}.parquet"
        if path.exists() and not should_force("stock_basic"):
            continue
        params = {"exchange": "", "list_status": status}
        result, _pages = query_paged(client, "stock_basic", params, stock_basic_fields, REFERENCE_PAGE_LIMIT)
        write_reference_query_result(
            path,
            result,
            api_name="stock_basic",
            params=params,
            required_nonempty=status == "L",
            ledger=revision_ledger,
        )

    open_dates = download_trade_cal(client, raw_dir, args.start_date, trade_cal_end_date, should_force("trade_cal"), ledger=revision_ledger)
    open_dates = [trade_date for trade_date in open_dates if trade_date <= args.end_date]
    if not args.skip_bak_basic:
        download_bak_basic(
            client,
            raw_dir,
            open_dates,
            args.bak_start_date,
            should_force("bak_basic"),
            revision_ledger,
        )
    recent_since = None
    if args.namechange_recent_days:
        recent_since = format_yyyymmdd(parse_yyyymmdd(args.end_date) - timedelta(days=args.namechange_recent_days))
    download_namechange(client, raw_dir, should_force("namechange"), ledger=revision_ledger, recent_since=recent_since)
    classify = download_index_classify(client, raw_dir, should_force("index_classify"), ledger=revision_ledger)
    download_index_member_all(client, raw_dir, classify["SW2021"], should_force("index_member_all"), ledger=revision_ledger)
    download_index_member_history(client, raw_dir, classify["SW2014"], should_force("index_member"), ledger=revision_ledger)
    download_index_reference(client, raw_dir, should_force("index_basic"), ledger=revision_ledger)
    download_index_weight(
        client,
        raw_dir,
        args.start_date,
        args.end_date,
        history_floor(RESEARCH_HISTORY_FLOOR, args),
        should_force("index_weight"),
        revision_ledger,
    )
    print(f"reference download finished under {raw_dir}")
    return 0


def download_bak_basic(
    client: TuShareClient,
    raw_dir: Path,
    trade_dates: list[str],
    start_date: str,
    force: bool,
    revision_ledger: Path,
) -> None:
    def pull(part: Partition) -> tuple[bool, int, int]:
        params = {"trade_date": part.unit}
        result = client.query("bak_basic", params, BAK_BASIC_SPEC.fields)
        df = frame(result)
        outcome = publish(
            part.path,
            df,
            api_name="bak_basic",
            params=params,
            fields=result.fields,
            key_columns=list(BAK_BASIC_SPEC.key_columns),
            ledger=revision_ledger,
        )
        written = outcome in WRITTEN_OUTCOMES
        return written, len(df) if written else 0, 0

    land(
        partitions(raw_dir, BAK_BASIC_SPEC, [d for d in trade_dates if d >= start_date]), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force),
        revision_ledger=revision_ledger, name="bak_basic", every=250,
    )

def download_namechange(
    client: TuShareClient, raw_dir: Path, force: bool, *, ledger: Path | str | None, recent_since: str | None = None
) -> None:
    """Every code's name history by one call per code (about 80 minutes),
    when the table is missing or ``force``d. An existing table otherwise stays,
    unless ``recent_since`` is given: the changes announced since that day,
    by announcement date, then replace their rows in it. That query is paged
    like the other reference tables: the vendor honours limit and offset on
    it and served 6,416 rows (every change announced since 2015) to one
    unpaged call on 2026-11-24, so a short page of REFERENCE_PAGE_LIMIT rows
    is the whole answer."""
    path = raw_dir / "namechange" / "namechange.parquet"
    if path.exists() and not force and recent_since is None:
        return
    fields = "ts_code,name,start_date,end_date,ann_date,change_reason"
    frames: list[pd.DataFrame] = []
    if path.exists() and not force:
        recent = frame(query_paged(client, "namechange", {"start_date": recent_since}, fields, REFERENCE_PAGE_LIMIT)[0])
        current = pd.read_parquet(path)
        key = ["ts_code", "name", "start_date"]
        frames = [current[~current.set_index(key).index.isin(recent.set_index(key).index)], recent]
        params = {"strategy": "announced_since", "start_date": recent_since}
    else:
        for index, code in enumerate(load_stock_codes(raw_dir), start=1):
            result = client.query("namechange", {"ts_code": code}, fields)
            df = frame(result)
            if not df.empty:
                frames.append(df)
            if index % 500 == 0:
                print(f"namechange {index}")
        params = {"strategy": "per_ts_code_all_stock_basic"}
    combined = concat_rows(frames) if frames else pd.DataFrame(columns=fields.split(","))
    deduped = combined.drop_duplicates().sort_values(["ts_code", "start_date", "name"], na_position="last").reset_index(drop=True)
    publish(
        path,
        deduped,
        api_name="namechange",
        params=params,
        fields=fields.split(","),
        key_columns=(),
        ledger=ledger,
    )

def download_index_classify(client: TuShareClient, raw_dir: Path, force: bool, *, ledger: Path | str | None) -> dict[str, pd.DataFrame]:
    """Both Shenwan classification vintages: SW2014 for pre-2021 history, SW2021 for the current scheme."""
    fields = "index_code,industry_name,level,industry_code,is_pub,parent_code,src"
    frames: dict[str, pd.DataFrame] = {}
    for src in ("SW2014", "SW2021"):
        path = raw_dir / "index_classify" / f"src={src}.parquet"
        if path.exists() and not force:
            frames[src] = pd.read_parquet(path)
            continue
        params = {"src": src}
        result, _pages = query_paged(client, "index_classify", params, fields, REFERENCE_PAGE_LIMIT)
        frames[src] = write_reference_query_result(
            path,
            result,
            api_name="index_classify",
            params=params,
            required_nonempty=True,
            ledger=ledger,
        )
    return frames

def download_index_member_all(
    client: TuShareClient, raw_dir: Path, classify: pd.DataFrame, force: bool, *, ledger: Path | str | None
) -> None:
    fields = "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new"
    l1_codes = classify.loc[classify["level"].astype(str) == "L1", "index_code"].dropna().astype(str).tolist()
    for code in sorted(l1_codes):
        path = raw_dir / "index_member_all" / f"l1_code={code}.parquet"
        if path.exists() and not force:
            continue
        # The API filters on is_new and defaults to current members only, so
        # departed members (real out_date rows) need the explicit second pull.
        frames = []
        for is_new in ("Y", "N"):
            result, _pages = query_paged(client, "index_member_all", {"l1_code": code, "is_new": is_new}, fields, REFERENCE_PAGE_LIMIT)
            frames.append(frame(result))
        combined = concat_rows(frames).drop_duplicates()
        if combined.empty:
            if not path.exists():
                # A valid SW2021 L1 always has current members; an empty Y+N pull is a vendor failure.
                raise RuntimeError(f"index_member_all l1_code={code} returned zero rows for required reference partition")
        else:
            combined = combined.sort_values(["ts_code", "in_date"], na_position="last").reset_index(drop=True)
        # An empty answer for the existing partition is a recorded defect that keeps it.
        publish(
            path,
            combined,
            api_name="index_member_all",
            params={"l1_code": code, "is_new": ["Y", "N"]},
            fields=fields.split(","),
            key_columns=(),
            ledger=ledger,
        )

def download_index_member_history(
    client: TuShareClient, raw_dir: Path, classify_sw2014: pd.DataFrame, force: bool, *, ledger: Path | str | None
) -> None:
    """SW2014 L1 membership via the legacy index_member API. The old taxonomy
    closed on 20211213 (every retired index's members carry that out_date), so
    the history is frozen and only re-downloads under an explicit force or
    where a partition lacks its commit identity."""
    fields = "index_code,con_code,in_date,out_date,is_new"
    l1_codes = classify_sw2014.loc[classify_sw2014["level"].astype(str) == "L1", "index_code"].dropna().astype(str).tolist()
    for code in sorted(l1_codes):
        path = raw_dir / "index_member" / f"l1_code={code}.parquet"
        if should_skip_existing_partition(path, force=force):
            continue
        params = {"index_code": code}
        result, _pages = query_paged(client, "index_member", params, fields, 3000)
        write_reference_query_result(
            path,
            result,
            api_name="index_member",
            params=params,
            required_nonempty=True,
            ledger=ledger,
        )

def download_index_reference(client: TuShareClient, raw_dir: Path, force: bool, *, ledger: Path | str | None) -> None:
    """Index metadata catalog. Must page: the unpaged call silently truncated
    at the vendor's 8,000-row cap (12,484 exist; the whole CI and NH families
    were missing)."""
    path = raw_dir / "index_basic" / "catalog.parquet"
    fields = "ts_code,name,fullname,market,publisher,index_type,category,base_date,base_point,list_date,weight_rule,desc,exp_date"
    if not path.exists() or force:
        result, _pages = query_paged(client, "index_basic", {}, fields, REFERENCE_PAGE_LIMIT)
        write_reference_query_result(path, result, api_name="index_basic", params={}, required_nonempty=True, ledger=ledger)


def download_index_weight(
    client: TuShareClient,
    raw_dir: Path,
    start_date: str,
    end_date: str,
    floor: str,
    force: bool,
    revision_ledger: Path | str | None,
) -> None:
    """Monthly core-index constituent weights, one partition per index per year,
    from the research-history ``floor`` on."""
    start_date = max(start_date, floor)
    if end_date < start_date:
        return

    def request(part: Partition) -> dict[str, str]:
        year_start, year_end, _ = part.unit
        return {**part.dims, "start_date": year_start, "end_date": year_end}

    def pull(part: Partition) -> tuple[bool, int, int]:
        params = request(part)
        result, pages = query_paged(client, "index_weight", params, INDEX_WEIGHT_FIELDS, INDEX_WEIGHT_PAGE_LIMIT)
        df = frame(result)
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name="index_weight",
            params={**params, "pagination": {"page_limit": INDEX_WEIGHT_PAGE_LIMIT, "pages": pages}},
            fields=list(df.columns) if len(df.columns) else result.fields,
            key_columns=["index_code", "con_code", "trade_date"],
            date_columns=["trade_date"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            revision_ledger=revision_ledger,
        )
        return did_write, rows, pages

    parts = [
        Partition(raw_dir / "index_weight" / f"index_code={safe_partition_value(code)}" / f"year={year}.parquet", (year_start, year_end, year), {"index_code": code})
        for code in DEFAULT_CN_INDEX_CODES
        for year_start, year_end, year in year_windows(start_date, end_date)
    ]
    land(
        parts, pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=request(part)),
        revision_ledger=revision_ledger, name="index_weight",
    )


def download_daily(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    ensure_trade_cal_coverage(client, raw_dir, args.start_date, args.end_date, ledger=revision_ledger)
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    refresh_datasets = set(args.refresh_daily_datasets)
    required_zero_skipped: list[str] = []
    for dataset in selected_daily_datasets(args):
        if dataset == "stk_auction":
            # The generic path has none of the auction completeness checks and
            # would stamp fetch-time availability over the observed landing.
            raise RuntimeError(
                "stk_auction is written only by the dedicated paths "
                "(capture-open-auction, recheck-stk-auction); "
                "pass an explicit --daily-datasets list without it"
            )
        spec = DAILY_SPECS[dataset]
        start_date = max(args.start_date, spec.start_date)
        dates = [d for d in trade_dates if start_date <= d <= args.end_date]
        zero_skipped = download_trade_date_dataset(
            client,
            raw_dir,
            spec,
            dates,
            args.force or dataset in refresh_datasets,
            args.page_limit,
            revision_ledger,
        )
        if zero_skipped and not spec.zero_rows_ok:
            required_zero_skipped.append(f"{spec.api_name}:{zero_skipped}")
    if required_zero_skipped:
        raise RuntimeError(f"required daily partitions returned zero rows and were not written: {required_zero_skipped}")
    print(f"daily market download finished under {raw_dir}")
    return 0


def date_series_in_window(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    keys = series.map(normalize_date_key)
    return keys.between(start_date, end_date)

def merge_existing_window_rows(
    existing: pd.DataFrame,
    refreshed: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    date_columns: list[str],
    key_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    """The partition after replacing its rows inside the window with the
    refreshed answer, and how many existing rows the window held.

    ``key_columns`` (empty to keep every row) collapses rows sharing a key,
    the refreshed row winning."""
    if existing.empty:
        return refreshed, 0
    date_column = next((column for column in date_columns if column and (column in existing.columns or column in refreshed.columns)), "")
    if not date_column:
        return refreshed, len(existing)
    existing = existing.copy()
    refreshed = refreshed.copy()
    if date_column not in existing.columns:
        existing[date_column] = ""
    if date_column not in refreshed.columns:
        refreshed[date_column] = ""
    inside = date_series_in_window(existing[date_column], start_date, end_date)
    merged = concat_rows([existing[~inside], refreshed])
    if key_columns and set(key_columns).issubset(merged.columns):
        merged = merged.drop_duplicates(key_columns, keep="last")
    return merged.reset_index(drop=True), int(inside.sum())

def write_window_merged_partition(
    path: Path,
    refreshed: pd.DataFrame,
    *,
    api_name: str,
    params: dict[str, Any],
    fields: list[str],
    key_columns: list[str],
    date_columns: list[str],
    start_date: str,
    end_date: str,
    revision_ledger: Path | str | None,
    deduplicate: bool = False,
) -> tuple[bool, int]:
    """Merge the refreshed window into the partition. Returns (did_write, rows).

    An empty answer for a window in which the partition holds rows is a
    defective answer. A dataset whose key names one record (``deduplicate``)
    leaves duplicate keys to the versioned write, which drops them
    deterministically and counts them; any other keeps the refreshed row of a
    repeated key.
    """
    df = refreshed
    meta_params = dict(params)
    current = None
    if path.exists():
        current = pd.read_parquet(path)
        df, held = merge_existing_window_rows(
            current,
            refreshed,
            start_date=start_date,
            end_date=end_date,
            date_columns=date_columns,
            key_columns=[] if deduplicate else key_columns,
        )
        if refreshed.empty and held:
            raise DefectiveAnswer(
                f"{api_name} answered no rows for {start_date}-{end_date}, where the partition holds {held}"
            )
        meta_params["merge_existing_window"] = {"start_date": start_date, "end_date": end_date}
    outcome = publish(
        path,
        df,
        api_name=api_name,
        params=meta_params,
        fields=list(df.columns) if len(df.columns) else fields,
        key_columns=key_columns,
        ledger=revision_ledger,
        deduplicate=deduplicate,
        current=current,
    )
    written = outcome in WRITTEN_OUTCOMES
    return written, len(df) if written else 0

def write_frame(
    path: Path,
    df: pd.DataFrame,
    spec: DatasetSpec,
    params: dict[str, Any],
    revision_ledger: Path | str | None = None,
) -> tuple[bool, int]:
    """Publish one augmented answer at ``path`` under ``spec``'s key. Returns
    whether the visible partition changed and the rows that change wrote (0
    when ``publish`` wrote nothing)."""
    outcome = publish(
        path,
        df,
        api_name=spec.api_name,
        params=params,
        fields=list(df.columns),
        key_columns=list(spec.key_columns),
        ledger=revision_ledger,
        deduplicate=spec.unique_key,
    )
    written = outcome in WRITTEN_OUTCOMES
    return written, len(df) if written else 0

def land(
    parts: list[Partition],
    pull: Callable[[Partition], tuple[bool | None, int, int]],
    *,
    skip: Callable[[Partition], bool],
    revision_ledger: Path | str | None,
    name: str,
    unit: str = "",
    every: int | None = None,
    pages: bool = False,
) -> tuple[int, int]:
    """Pull every partition of ``parts`` that ``skip`` does not rule out.

    Each pull runs under :func:`answer_defects`: a defective answer leaves its
    partition as it is and the loop goes on. ``pull(part)`` returns whether it
    wrote the partition (None when it declined the answer: no rows, or not
    every exchange, where the dataset requires them), the rows it wrote and
    the pages it read. Prints the tally every ``every`` partitions (counted
    with the skipped ones; a skipped partition prints nothing) and at the end,
    ``unit`` naming what is counted; returns the partitions written and the
    declined ones."""
    written = skipped = declined = rows_written = pages_read = 0

    def tally() -> str:
        return (
            f"skipped={skipped} written={written}"
            + (f" zero_skipped={declined}" if declined else "")
            + f" rows_written={rows_written}"
            + (f" pages={pages_read}" if pages else "")
        )

    for index, part in enumerate(parts, start=1):
        if skip(part):
            skipped += 1
            continue
        with answer_defects(part.path, ledger=revision_ledger):
            wrote, rows, read = pull(part)
            written += wrote is True
            declined += wrote is None
            rows_written += rows
            pages_read += read
        if every and index % every == 0:
            print(f"{name} {unit + ' ' if unit else ''}{index}/{len(parts)} {tally()}")
    print(f"{name} done {unit or 'tasks'}={len(parts)} {tally()}")
    return written, declined

def download_trade_date_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    trade_dates: list[str],
    force: bool,
    page_limit: int,
    revision_ledger: Path,
) -> int:
    """Returns the trade dates whose answer was declined or kept out as a defect."""
    page_limit = page_limit or TRADE_DATE_PAGE_LIMIT

    def pull(part: Partition) -> tuple[bool | None, int, int]:
        params = {"trade_date": part.unit}
        try:
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        except DefectiveAnswer:
            raise
        except Exception as exc:
            raise RuntimeError(f"{spec.api_name} trade_date={part.unit} failed: {exc}") from exc
        df = frame(result)
        if spec.dedup_exact and not df.empty:
            df = df.drop_duplicates().reset_index(drop=True)
        if df.empty and not spec.zero_rows_ok:
            print(f"{spec.api_name} trade_date={part.unit} returned zero rows; skipped_write")
            return None, 0, pages
        outcome = publish(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=result.fields,
            key_columns=list(spec.key_columns),
            ledger=revision_ledger,
        )
        written = outcome in WRITTEN_OUTCOMES
        return None if outcome == "defect" else written, len(df) if written else 0, pages

    # A parquet whose sidecar is missing or carries another write ID is an
    # interrupted write, not a committed partition: re-attempt it.
    _, declined = land(
        partitions(raw_dir, spec, trade_dates), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force) and (spec.zero_rows_ok or parquet_rows(part.path) > 0),
        revision_ledger=revision_ledger, name=spec.api_name, every=250,
    )
    return declined

def download_macro(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    retained_start_date = args.macro_start_date or args.start_date
    for dataset in selected_macro_datasets(args):
        spec = MACRO_SPECS[dataset]
        spec_floor = history_floor(spec.start_date, args)
        start_date = max(args.start_date, spec_floor)
        if spec.strategy == FetchStrategy.QUARTER_ONCE:
            download_macro_quarter_once(client, raw_dir, spec, max(retained_start_date, spec_floor), args.end_date, args.force, revision_ledger)
        elif spec.strategy == FetchStrategy.MONTH_ONCE:
            download_macro_month_once(client, raw_dir, spec, max(retained_start_date, spec_floor), args.end_date, args.force, revision_ledger)
        elif spec.strategy == FetchStrategy.MONTH_LOOP:
            download_macro_month_loop(client, raw_dir, spec, start_date, args.end_date, args.force, revision_ledger)
        elif spec.strategy == FetchStrategy.TRADE_DATE:
            download_macro_trade_date(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), revision_ledger)
        elif spec.strategy == FetchStrategy.DATE_YEAR:
            download_macro_year_windows(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), args, revision_ledger)
        elif spec.strategy == FetchStrategy.STATIC_FULL:
            download_macro_static_full(client, raw_dir, spec, spec_page_limit(spec, args.page_limit), revision_ledger)
        elif spec.strategy == FetchStrategy.DATE_YEAR_BY_TS_CODE:
            download_macro_year_windows(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), args, revision_ledger)
        elif spec.strategy == FetchStrategy.ECO_CAL_MONTH:
            download_macro_eco_cal_month(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), args, revision_ledger)
        else:
            raise RuntimeError(f"unsupported macro strategy {spec.strategy} for {dataset}")
    print(f"{args.tier} download finished under {raw_dir}")
    return 0

def download_macro_quarter_once(client: TuShareClient, raw_dir: Path, spec: DatasetSpec, start_date: str, end_date: str, force: bool, revision_ledger: Path | str | None = None) -> None:
    part = range_partition(raw_dir, spec, start_date)
    refuse_moved_range_start(part.path)
    path, (start_date, start_q) = part.path, part.unit
    end_q = yyyymmdd_to_quarter(end_date)
    coverage_params = {"start_date": start_date, "end_date": end_date}
    if should_skip_existing_partition(path, force=force, requested_params=coverage_params):
        return
    params = {"start_q": start_q, "end_q": end_q}
    with answer_defects(path, ledger=revision_ledger):
        result = client.query(spec.api_name, params, spec.fields)
        _, rows = write_frame(path, augment_macro_frame(frame(result), spec), spec, {**params, **coverage_params}, revision_ledger)
        print(f"{spec.api_name} quarters {start_q}-{end_q} rows={rows}")

def download_macro_month_once(client: TuShareClient, raw_dir: Path, spec: DatasetSpec, start_date: str, end_date: str, force: bool, revision_ledger: Path | str | None = None) -> None:
    part = range_partition(raw_dir, spec, start_date)
    refuse_moved_range_start(part.path)
    path, (start_date, start_m) = part.path, part.unit
    end_m = yyyymmdd_to_month(end_date)
    coverage_params = {"start_date": start_date, "end_date": end_date}
    if should_skip_existing_partition(path, force=force, requested_params=coverage_params):
        return
    params = {"start_m": start_m, "end_m": end_m}
    with answer_defects(path, ledger=revision_ledger):
        result = client.query(spec.api_name, params, spec.fields)
        _, rows = write_frame(path, augment_macro_frame(frame(result), spec), spec, {**params, **coverage_params}, revision_ledger)
        print(f"{spec.api_name} months {start_m}-{end_m} rows={rows}")

def download_macro_static_full(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    page_limit: int,
    revision_ledger: Path | str | None = None,
) -> None:
    """Full registry refresh: small reference tables (contract/bond registries,
    announcement ledgers) are re-pulled whole on every run and written through
    the versioned write path."""
    for part in partitions(raw_dir, spec):
        params, path = dict(part.dims), part.path
        name = path.name
        with answer_defects(path, ledger=revision_ledger):
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
            df = augment_macro_frame(frame(result), spec)
            meta_params = dict(params)
            meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
            outcome = publish(
                path,
                df,
                api_name=spec.api_name,
                params=meta_params,
                fields=list(df.columns),
                key_columns=list(spec.key_columns),
                ledger=revision_ledger,
            )
            print(f"{spec.api_name} {name} rows={len(df)} outcome={outcome}")


def download_macro_month_loop(client: TuShareClient, raw_dir: Path, spec: DatasetSpec, start_date: str, end_date: str, force: bool, revision_ledger: Path | str | None = None) -> None:
    def coverage(part: Partition) -> dict[str, str]:
        # Coverage params for EVERY month, not only the newest: skipping a
        # middle month on bare file existence would freeze a partition whose
        # recorded coverage stops mid-month once the window rolls past it.
        window_start, window_end, _ = part.unit
        return {"start_date": window_start, "end_date": window_end}

    def pull(part: Partition) -> tuple[bool, int, int]:
        window_start, window_end, month = part.unit
        params = {spec.month_param: month}
        result = client.query(spec.api_name, params, spec.fields)
        df = augment_macro_frame(frame(result), spec)
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, **coverage(part)},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=window_start,
            end_date=window_end,
            revision_ledger=revision_ledger,
        )
        return did_write, rows, 0

    land(
        partitions(raw_dir, spec, month_windows(start_date, end_date)), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=coverage(part)),
        revision_ledger=revision_ledger, name=spec.api_name, unit="months", every=24,
    )

def download_macro_trade_date(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    start_date: str,
    end_date: str,
    force: bool,
    page_limit: int,
    revision_ledger: Path | str | None = None,
) -> None:
    """Per-trade-date pulls for daily macro tables whose range endpoints cap the
    response server-side and ignore offset paging (ths/sw/ci index dailies:
    year-range pulls silently truncated at 3000/4000 rows). An optional venue
    loop (opt_daily per financial exchange) keeps one file per (venue, trade
    date) so partition pruning by date keeps working."""
    def pull(part: Partition) -> tuple[bool, int, int]:
        params = {"trade_date": part.unit, **part.dims}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = augment_macro_frame(frame(result), spec)
        if df.empty:
            print(f"{spec.api_name} trade_date={part.unit} returned zero rows; skipped_write")
            return False, 0, pages
        outcome = publish(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            ledger=revision_ledger,
        )
        written = outcome in WRITTEN_OUTCOMES
        return written, len(df) if written else 0, pages

    land(
        partitions(raw_dir, spec, load_sse_open_dates(raw_dir, start_date, end_date)), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force) and parquet_rows(part.path) > 0,
        revision_ledger=revision_ledger, name=spec.api_name, every=250,
    )


def download_macro_year_windows(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    start_date: str,
    end_date: str,
    force: bool,
    page_limit: int,
    args: argparse.Namespace | None,
    revision_ledger: Path | str | None = None,
) -> None:
    """One partition per calendar year, and per code for a by-ts_code table
    (``args`` names the codes)."""
    def request(part: Partition) -> dict[str, str]:
        year_start, year_end, _ = part.unit
        return {**part.dims, "start_date": year_start, "end_date": year_end}

    def pull(part: Partition) -> tuple[bool, int, int]:
        params = request(part)
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = augment_macro_frame(frame(result), spec)
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            revision_ledger=revision_ledger,
        )
        return did_write, rows, pages

    by_code = spec.strategy == FetchStrategy.DATE_YEAR_BY_TS_CODE
    land(
        partitions(raw_dir, spec, year_windows(start_date, end_date), args), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=request(part)),
        revision_ledger=revision_ledger, name=spec.api_name, unit="" if by_code else "years", every=25 if by_code else None,
    )

def download_macro_eco_cal_month(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    start_date: str,
    end_date: str,
    force: bool,
    page_limit: int,
    args: argparse.Namespace,
    revision_ledger: Path | str | None = None,
) -> None:
    def request(part: Partition) -> dict[str, Any]:
        start, end, _ = part.unit
        return {"start_date": start, "end_date": end, **{key: value for key, value in part.dims.items() if value}}

    def pull(part: Partition) -> tuple[bool, int, int]:
        # The API clamps every response to 100 rows, sorts with no stable
        # tiebreak, and serves each event many times over (measured ~8x), so
        # offset-paging a whole month is slow and repeats pages verbatim. Day
        # windows bound each paging run, repeated pages are tolerated as
        # expected source behavior, and the key dedup below collapses the
        # duplicates.
        start, end, month = part.unit
        params = request(part)
        day_frames: list[pd.DataFrame] = []
        day_windows = 0
        pages_total = 0
        day = parse_yyyymmdd(start)
        last_day = parse_yyyymmdd(end)
        while day <= last_day:
            day_params = dict(params)
            day_params["start_date"] = day_params["end_date"] = format_yyyymmdd(day)
            result, pages = query_paged(client, spec.api_name, day_params, spec.fields, page_limit, allow_repeated_pages=True)
            pages_total += pages
            day_windows += 1
            day_frames.append(frame(result))
            day += timedelta(days=1)
        combined = pd.concat(day_frames, ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(subset=list(spec.key_columns), keep="first", ignore_index=True)
        meta_params = {
            **params,
            "country_partition": directory_name(part.dims["country"]),
            "currency_partition": directory_name(part.dims["currency"]),
            "event_partition": directory_name(part.dims["event"]),
            "month": month,
            "pagination": {"page_limit": page_limit, "pages": pages_total, "day_windows": day_windows},
        }
        df = augment_macro_frame(combined, spec)
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=start,
            end_date=end,
            revision_ledger=revision_ledger,
        )
        return did_write, rows, pages_total

    land(
        partitions(raw_dir, spec, month_windows(start_date, end_date), args), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=request(part)),
        revision_ledger=revision_ledger, name=spec.api_name, every=24,
    )

def download_event_flow(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    datasets = selected_event_flow_download_datasets(args)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    trade_dates: list[str] = []
    stamping_dates: list[str] | None = None
    trade_end_date = args.end_date
    trade_cal_refreshed = False
    has_trade_date_dataset = any(EVENT_FLOW_SPECS[name].strategy == FetchStrategy.TRADE_DATE for name in datasets)
    # Rules counted in open days (margin's next open day, new_share's second)
    # stamp from the calendar after the window. A 31-day calendar request
    # comfortably spans regular exchange holidays; the stamper still fails
    # explicitly if the open date is missing, so this never approximates.
    open_day_stamped = any(
        (rule := EVENT_FLOW_SPECS[name].availability) is not None and rule.trading_day_offset
        for name in datasets
    )
    stamping_end_date = format_yyyymmdd(parse_yyyymmdd(args.end_date) + timedelta(days=31))
    if has_trade_date_dataset or open_day_stamped:
        trade_cal_refreshed = ensure_trade_cal_coverage(
            client, raw_dir, args.start_date, stamping_end_date if open_day_stamped else args.end_date, ledger=revision_ledger
        )
    if has_trade_date_dataset:
        latest_trade_calendar_date = latest_sse_calendar_date(raw_dir)
        trade_end_date = min(args.end_date, latest_trade_calendar_date)
        if trade_end_date < args.end_date:
            print(
                f"event/flow trade-date datasets capped at local SSE trade_cal end {trade_end_date}; "
                f"requested end_date={args.end_date}"
            )
        trade_dates = load_sse_open_dates(raw_dir, args.start_date, trade_end_date, allow_empty=True)
        if not trade_dates:
            print(f"event/flow trade-date datasets skipped: no SSE open dates for {args.start_date}-{trade_end_date}")
    if open_day_stamped:
        stamping_dates = load_sse_open_dates(raw_dir, args.start_date, stamping_end_date, allow_empty=True)
    required_zero_skipped: list[str] = []
    # trade_cal coverage refresh IS a lake write: it must veto the exit-75
    # no-mutation contract even when every dataset response was empty.
    total_written = 1 if trade_cal_refreshed else 0
    for dataset in datasets:
        spec = EVENT_FLOW_SPECS[dataset]
        start_date = max(args.start_date, spec.start_date)
        if spec.strategy == FetchStrategy.TRADE_DATE:
            dates = [d for d in trade_dates if start_date <= d <= trade_end_date]
            written, zero_skipped = download_event_trade_date_dataset(
                client,
                raw_dir,
                spec,
                dates,
                args.force,
                spec_page_limit(spec, args.page_limit),
                revision_ledger,
                trading_dates=stamping_dates,
            )
            total_written += written
            if zero_skipped and not spec.zero_rows_ok:
                required_zero_skipped.append(f"{spec.api_name}:{zero_skipped}")
        elif spec.strategy == FetchStrategy.RANGE_MONTH:
            window_start = format_yyyymmdd(parse_yyyymmdd(args.start_date) - timedelta(days=spec.report_lag_days))
            windows = month_windows(window_start, args.end_date)
            dataset_windows = [(s, e, m) for s, e, m in windows if e >= max(window_start, spec.start_date)]
            total_written += download_event_range_month(
                client,
                raw_dir,
                spec,
                dataset_windows,
                args.force,
                spec_page_limit(spec, args.page_limit),
                revision_ledger,
                trading_dates=stamping_dates,
            )
        else:
            raise RuntimeError(f"unsupported event/flow strategy {spec.strategy} for {dataset}")
    if required_zero_skipped:
        if args.zero_rows_not_ready:
            # Pre-open attempts hit the source before T-1 margin publication;
            # an empty response is a regular not-ready event, not a failure.
            # Exit 75 keeps the auction-capture no-mutation contract: cron
            # restores the committed generation and the retry job re-attempts.
            if total_written == 0:
                print(json.dumps({
                    "status": "not_ready_no_mutation",
                    "zero_row_partitions": required_zero_skipped,
                }, ensure_ascii=False, sort_keys=True))
                return NO_MUTATION_RETRY_EXIT_CODE
            # Other partitions committed, so the generation must be published --
            # but required coverage was NOT achieved. Reporting success here
            # would record an "ok" job state, and skip_if_already_ok would then
            # suppress every later attempt at the same resolved end date,
            # abandoning the partition once the window moves past it.
            print(json.dumps({
                "status": "not_ready_after_mutation",
                "zero_row_partitions": required_zero_skipped,
            }, ensure_ascii=False, sort_keys=True))
            return MUTATED_NOT_READY_RETRY_EXIT_CODE
        else:
            raise RuntimeError(f"required event/flow partitions returned zero or incomplete rows and were not written: {required_zero_skipped}")
    print(f"event/flow download finished under {raw_dir}")
    return 0

def selected_event_flow_download_datasets(args: argparse.Namespace) -> list[str]:
    default = [dataset for dataset in EVENT_FLOW_DATASETS if dataset != "share_float"]
    datasets = select_datasets(args.datasets, default=default, allowed=EVENT_FLOW_SPECS, label="event/flow")
    if "share_float" in datasets:
        raise RuntimeError(
            "share_float must be downloaded with download-share-float-complete; "
            "the generic event_flow path cannot provide the ann_date rescue and union guard."
        )
    return datasets

def download_event_trade_date_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    trade_dates: list[str],
    force: bool,
    page_limit: int | None,
    revision_ledger: Path | str | None = None,
    *,
    trading_dates: list[str] | None = None,
) -> tuple[int, int]:
    """Returns the trade dates written and those whose answer was declined or
    kept out as a defect."""
    page_limit = page_limit or spec.page_limit

    def complete(part: Partition) -> bool:
        # A parquet whose sidecar is missing or carries another write ID is
        # an interrupted write, not a committed partition: re-attempt it.
        if not (should_skip_existing_partition(part.path, force=force) and (spec.zero_rows_ok or parquet_rows(part.path) > 0)):
            return False
        # A committed partial day (missing exchanges) must be re-attempted by
        # any covering run, not treated as done.
        exchange_column = margin_exchange_column(spec.api_name)
        if exchange_column is None:
            return True
        present = pd.read_parquet(part.path, columns=[exchange_column])
        return not margin_family_missing_exchanges(spec.api_name, part.unit, present)

    def pull(part: Partition) -> tuple[bool | None, int, int]:
        params = {"trade_date": part.unit}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = augment_event_frame(frame(result), spec, trading_dates=trading_dates)
        if df.empty and not spec.zero_rows_ok:
            print(f"{spec.api_name} trade_date={part.unit} returned zero rows; skipped_write")
            return None, 0, pages
        missing = margin_family_missing_exchanges(spec.api_name, part.unit, df) if not df.empty else []
        if missing:
            # Exchanges publish independently; a partial response poisons
            # market-wide aggregates, so it rides the same not-ready/retry
            # contract as an empty one instead of being committed.
            print(f"{spec.api_name} trade_date={part.unit} missing exchanges {missing}; skipped_write")
            return None, 0, pages
        outcome = publish(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            ledger=revision_ledger,
        )
        written = outcome in WRITTEN_OUTCOMES
        return None if outcome == "defect" else written, len(df) if written else 0, pages

    return land(
        partitions(raw_dir, spec, trade_dates), pull,
        skip=complete,
        revision_ledger=revision_ledger, name=spec.api_name, every=250, pages=True,
    )

def query_event_window(
    client: TuShareClient,
    spec: DatasetSpec,
    start_date: str,
    end_date: str,
    page_limit: int,
) -> tuple[ApiResult, int]:
    """Every row of one date window, from requests that each fit in one page.

    Offset paging is not order-stable on every endpoint: stk_holdernumber sorts
    on a non-unique key, so consecutive pages repeat the rows at each boundary
    and never serve as many others, reproducibly (2026-10-14: 202609 repeated 8
    of 3,141 rows, 202504 2,694 of 12,588). A window whose answer fills a page
    is therefore halved by date until every half fits in one page. Only a single
    day that still fills a page is offset-paged (the top-10 holder tables at a
    quarter end, whose paging is stable), and a row served on two of its pages
    fails the run: the rows it displaced are missing from the response.
    Returns the rows and the number of requests.
    """
    params = {"start_date": start_date, "end_date": end_date}
    if start_date < end_date:
        result = client.query(spec.api_name, {**params, "limit": page_limit, "offset": 0}, spec.fields)
        if len(result.items) < page_limit:
            return result, 1
        start, end = parse_yyyymmdd(start_date), parse_yyyymmdd(end_date)
        middle = start + (end - start) // 2
        left, left_requests = query_event_window(client, spec, start_date, format_yyyymmdd(middle), page_limit)
        right, right_requests = query_event_window(client, spec, format_yyyymmdd(middle + timedelta(days=1)), end_date, page_limit)
        if left.items and right.items and left.fields != right.fields:
            raise DefectiveAnswer(f"{spec.api_name} returned inconsistent fields across {start_date}-{end_date}")
        return ApiResult(left.fields or right.fields, left.items + right.items), 1 + left_requests + right_requests
    result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
    first_page: dict[str, int] = {}
    repeated = 0
    for index, row in enumerate(result.items):
        page = first_page.setdefault(json.dumps(row, ensure_ascii=False, default=str), index // page_limit)
        repeated += page != index // page_limit
    if repeated:
        raise DefectiveAnswer(
            f"{spec.api_name} {start_date}: offset paging served {repeated} row(s) on two pages over {pages} pages, "
            "so as many rows are missing from the response; refusing to merge an incomplete day"
        )
    return result, pages

def download_event_range_month(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    windows: list[tuple[str, str, str]],
    force: bool,
    page_limit: int,
    revision_ledger: Path | str | None = None,
    *,
    trading_dates: list[str] | None = None,
) -> int:
    def pull(part: Partition) -> tuple[bool, int, int]:
        start_date, end_date, month = part.unit
        result, pages = query_event_window(client, spec, start_date, end_date, page_limit)
        df = augment_event_frame(frame(result), spec, trading_dates=trading_dates)
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name=spec.api_name,
            params={"start_date": start_date, "end_date": end_date, "month": month, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.window_date_column or spec.date_column, spec.fallback_date_column, "available_at"],
            start_date=start_date,
            end_date=end_date,
            revision_ledger=revision_ledger,
            deduplicate=spec.unique_key,
        )
        return did_write, rows, pages

    written, _ = land(
        partitions(raw_dir, spec, windows), pull,
        skip=lambda part: should_skip_existing_partition(
            part.path, force=force, requested_params={"start_date": part.unit[0], "end_date": part.unit[1]}
        ),
        revision_ledger=revision_ledger, name=spec.api_name, unit="months", every=24, pages=True,
    )
    return written

BOARD_TRADE_DATE_STRATEGIES = frozenset({
    FetchStrategy.TRADE_DATE,
    FetchStrategy.TRADE_DATE_BY_TAG,
    FetchStrategy.TRADE_DATE_BY_LIMIT_TYPE,
    FetchStrategy.TRADE_DATE_BY_MARKET,
    FetchStrategy.TRADE_DATE_BY_MARKET_HOT_TYPE,
})

def download_board_trading(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    datasets = selected_board_trading_datasets(args)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    trade_dates: list[str] = []
    if any(BOARD_TRADING_SPECS[name].strategy != FetchStrategy.STATIC_FULL for name in datasets):
        ensure_trade_cal_coverage(client, raw_dir, args.start_date, args.end_date, ledger=revision_ledger)
        latest_trade_calendar_date = latest_sse_calendar_date(raw_dir)
        trade_end_date = min(args.end_date, latest_trade_calendar_date)
        if trade_end_date < args.end_date:
            print(
                f"board-trading trade-date datasets capped at local SSE trade_cal end {trade_end_date}; "
                f"requested end_date={args.end_date}"
            )
        trade_dates = load_sse_open_dates(raw_dir, args.start_date, trade_end_date, allow_empty=True)
        if not trade_dates:
            print(f"board-trading trade-date datasets skipped: no SSE open dates for {args.start_date}-{trade_end_date}")
    for dataset in datasets:
        spec = BOARD_TRADING_SPECS[dataset]
        start_date = max(args.start_date, history_floor(spec.start_date, args))
        dates = [trade_date for trade_date in trade_dates if start_date <= trade_date <= args.end_date]
        page_limit = spec_page_limit(spec, args.page_limit)
        if spec.strategy == FetchStrategy.STATIC_FULL:
            download_board_static_dataset(client, raw_dir, spec, revision_ledger)
        elif spec.strategy in BOARD_TRADE_DATE_STRATEGIES:
            download_board_trade_date_dataset(client, raw_dir, spec, dates, args.force, page_limit, revision_ledger, args)
        else:
            raise RuntimeError(f"unsupported board-trading strategy {spec.strategy} for {dataset}")
    print(f"board-trading download finished under {raw_dir}")
    return 0

def download_board_static_dataset(client: TuShareClient, raw_dir: Path, spec: DatasetSpec, revision_ledger: Path | str | None = None) -> None:
    """Full-table refresh on every run; an empty answer for a non-empty
    partition is a defect and keeps the current version."""
    [part] = partitions(raw_dir, spec)
    path = part.path
    with answer_defects(path, ledger=revision_ledger):
        result = client.query(spec.api_name, {}, spec.fields)
        _, rows = write_frame(path, augment_board_frame(frame(result), spec, {}), spec, {}, revision_ledger)
        print(f"{spec.api_name} static rows={rows}")

def download_board_trade_date_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    trade_dates: list[str],
    force: bool,
    page_limit: int,
    revision_ledger: Path | str | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Every trade-date board table: one partition per trade date and, for
    the tag, pool and hot-list tables, per value of their dimensions
    (``args`` filters them; layout.dimensions)."""
    def pull(part: Partition) -> tuple[bool, int, int]:
        params = {"trade_date": part.unit, **part.dims}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        # The sidecar also names each percent-encoded directory (is_new is a plain Y/N).
        meta_params = {
            **params,
            **{f"{key}_partition": directory_name(value) for key, value in part.dims.items() if key != "is_new"},
            "pagination": {"page_limit": page_limit, "pages": pages},
        }
        landed, rows = write_frame(part.path, augment_board_frame(frame(result), spec, meta_params), spec, meta_params, revision_ledger)
        return landed, rows, pages

    land(
        partitions(raw_dir, spec, trade_dates, args), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force),
        revision_ledger=revision_ledger, name=spec.api_name, every=250, pages=True,
    )


def download_fundamental(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    datasets = selected_fundamental_datasets(args.datasets)
    stock_codes: list[str] = []
    if any(FUNDAMENTAL_SPECS[name].strategy == FetchStrategy.TS_CODE for name in datasets):
        explicit_codes = [str(code).strip() for code in (args.codes or []) if str(code).strip()]
        stock_codes = sorted(set(explicit_codes)) if explicit_codes else load_stock_codes(raw_dir)
        if args.max_codes:
            stock_codes = stock_codes[: args.max_codes]
    refresh_periods = set(recent_quarter_periods(args.end_date, args.fundamental_refresh_period_count))
    refresh_months = set(recent_month_keys(args.end_date, args.fundamental_refresh_ann_month_count))
    periods = sorted(set(quarter_periods(args.start_date, args.end_date)) | refresh_periods)
    windows = fundamental_ann_month_windows(args.start_date, args.end_date, refresh_months)

    unsupported = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy not in {FetchStrategy.PERIOD, FetchStrategy.ANN_MONTH, FetchStrategy.TS_CODE}]
    if unsupported:
        raise RuntimeError(f"unsupported fundamental strategy for datasets: {unsupported}")

    period_datasets = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy == FetchStrategy.PERIOD]
    ann_month_datasets = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy == FetchStrategy.ANN_MONTH]
    ts_code_datasets = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy == FetchStrategy.TS_CODE]

    for dataset in period_datasets:
        spec = FUNDAMENTAL_SPECS[dataset]
        download_fundamental_period_dataset(client, raw_dir, spec, periods, args.force, spec_page_limit(spec, args.page_limit), refresh_periods, revision_ledger)
    for dataset in ann_month_datasets:
        spec = FUNDAMENTAL_SPECS[dataset]
        download_fundamental_ann_month_dataset(client, raw_dir, spec, windows, args.force, spec_page_limit(spec, args.page_limit), refresh_months, revision_ledger)

    affected_days = args.fundamental_refresh_event_days
    affected_start = format_yyyymmdd(parse_yyyymmdd(args.end_date) - timedelta(days=max(affected_days, 1) - 1))
    affected_codes = recent_fundamental_event_codes(
        raw_dir,
        refresh_periods,
        refresh_months,
        period_datasets,
        ann_month_datasets,
        affected_start,
        args.end_date,
    )
    dividend_probe_codes: set[str] = set()
    if "dividend" in ts_code_datasets and args.fundamental_dividend_probe_days:
        dividend_probe_codes = probe_recent_dividend_codes(client, args.end_date, args.fundamental_dividend_probe_days, args.page_limit)
    refresh_ts_code_datasets = set(args.fundamental_refresh_ts_code_datasets)
    for dataset in ts_code_datasets:
        spec = FUNDAMENTAL_SPECS[dataset]
        force_codes = set()
        if dataset in refresh_ts_code_datasets:
            force_codes = set(affected_codes)
            if dataset == "dividend":
                force_codes.update(dividend_probe_codes)
        download_fundamental_ts_code_dataset(client, raw_dir, spec, stock_codes, args.force, spec_page_limit(spec, args.page_limit), force_codes, revision_ledger)
    print(f"fundamental download finished under {raw_dir}")
    return 0

def recent_quarter_periods(end_date: str, count: int) -> list[str]:
    if count <= 0:
        return []
    end = parse_yyyymmdd(end_date)
    candidates: list[date] = []
    for year in range(end.year, end.year - 4, -1):
        for month, day in ((12, 31), (9, 30), (6, 30), (3, 31)):
            value = date(year, month, day)
            if value <= end:
                candidates.append(value)
    return [format_yyyymmdd(value) for value in sorted(candidates, reverse=True)[:count]]


def recent_month_keys(end_date: str, count: int) -> list[str]:
    if count <= 0:
        return []
    end = parse_yyyymmdd(end_date)
    current = date(end.year, end.month, 1)
    months: list[str] = []
    for _ in range(count):
        months.append(f"{current.year}{current.month:02d}")
        if current.month == 1:
            current = date(current.year - 1, 12, 1)
        else:
            current = date(current.year, current.month - 1, 1)
    return months


def fundamental_ann_month_windows(start_date: str, end_date: str, refresh_months: set[str]) -> list[tuple[str, str, str]]:
    """ann_month partitions are replaced whole, so every touched month is
    pulled as the FULL natural month: a window truncated at a mid-month
    start_date deletes that month's earlier announcements on overwrite."""
    touched = {month for _, _, month in month_windows(start_date, end_date)} | set(refresh_months)
    return [month_window_for_key(month, end_date) for month in sorted(touched)]


def month_window_for_key(month: str, end_date: str) -> tuple[str, str, str]:
    year = int(month[:4])
    month_num = int(month[4:])
    start = date(year, month_num, 1)
    last = date(year, month_num, calendar.monthrange(year, month_num)[1])
    end = min(last, parse_yyyymmdd(end_date))
    return format_yyyymmdd(start), format_yyyymmdd(end), month


def recent_fundamental_event_codes(
    raw_dir: Path,
    refresh_periods: set[str],
    refresh_months: set[str],
    period_datasets: list[str],
    ann_month_datasets: list[str],
    start_date: str,
    end_date: str,
) -> set[str]:
    codes: set[str] = set()
    for dataset in period_datasets:
        for period in refresh_periods:
            codes.update(read_partition_ts_codes(raw_dir / dataset / f"period={period}.parquet", start_date, end_date, require_date_match=True, date_columns=SCHEDULE_DATASET_EVENT_COLUMNS.get(dataset)))
    for dataset in ann_month_datasets:
        for month in refresh_months:
            codes.update(read_partition_ts_codes(raw_dir / dataset / f"ann_month={month}.parquet", start_date, end_date, require_date_match=False))
    return codes


FUNDAMENTAL_EVENT_DATE_COLUMNS = ("f_ann_date", "ann_date", "first_ann_date", "imp_ann_date", "actual_date", "pre_date")


# disclosure_date is the vendor's disclosure schedule: ann_date and pre_date are
# planned/refreshed dates that exist for every listed code as soon as a quarter
# ends, so only actual_date marks a report that was really disclosed.
SCHEDULE_DATASET_EVENT_COLUMNS = {"disclosure_date": ("actual_date",)}


def read_partition_ts_codes(path: Path, start_date: str | None = None, end_date: str | None = None, require_date_match: bool = False, date_columns: tuple[str, ...] | None = None) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path)
    if "ts_code" not in frame.columns:
        return set()
    if start_date and end_date:
        present = [column for column in (date_columns or FUNDAMENTAL_EVENT_DATE_COLUMNS) if column in frame.columns]
        if present:
            mask = pd.Series(False, index=frame.index)
            for column in present:
                mask |= frame[column].map(lambda value: date_value_in_window(value, start_date, end_date))
            frame = frame[mask]
        elif require_date_match:
            return set()
    return set(frame["ts_code"].dropna().astype(str))


def date_value_in_window(value: object, start_date: str, end_date: str) -> bool:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return False
    try:
        parsed = pd.Timestamp(text).strftime("%Y%m%d")
    except Exception:  # noqa: BLE001 - an unparseable date is simply outside the window
        return False
    return start_date <= parsed <= end_date


def probe_recent_dividend_codes(client: TuShareClient, end_date: str, days: int, page_limit: int | None) -> set[str]:
    if days <= 0:
        return set()
    end = parse_yyyymmdd(end_date)
    start = end - timedelta(days=days - 1)
    codes: set[str] = set()
    for offset in range(days):
        key = format_yyyymmdd(start + timedelta(days=offset))
        for param_name in ("ann_date", "imp_ann_date", "ex_date", "record_date"):
            result, _pages = query_paged(client, "dividend", {param_name: key}, "ts_code", page_limit)
            df = frame(result)
            if "ts_code" in df.columns:
                codes.update(df["ts_code"].dropna().astype(str))
    if codes:
        print(f"dividend recent date probes found {len(codes)} candidate codes over {days} days")
    return codes


def land_fundamental(
    client: TuShareClient,
    spec: DatasetSpec,
    parts: list[Partition],
    request: Callable[[Partition], dict[str, str]],
    *,
    skip: Callable[[Partition], bool],
    page_limit: int,
    revision_ledger: Path | str | None,
    unit: str,
    every: int,
) -> None:
    """Pull each fundamental partition whole with the query ``request(part)``;
    an announcement-month partition's sidecar also names its month."""
    def pull(part: Partition) -> tuple[bool, int, int]:
        params = request(part)
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = frame(result)
        month = {"ann_month": part.unit[2]} if spec.strategy == FetchStrategy.ANN_MONTH else {}
        outcome = publish(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, **month, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=result.fields,
            key_columns=list(spec.key_columns),
            ledger=revision_ledger,
        )
        written = outcome in WRITTEN_OUTCOMES
        return written, len(df) if written else 0, pages

    land(parts, pull, skip=skip, revision_ledger=revision_ledger, name=spec.api_name, unit=unit, every=every)


def download_fundamental_period_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    periods: list[str],
    force: bool,
    page_limit: int,
    force_periods: set[str] | None = None,
    revision_ledger: Path | str | None = None,
) -> None:
    force_periods = force_periods or set()
    land_fundamental(
        client, spec, partitions(raw_dir, spec, periods), lambda part: {spec.period_param: part.unit},
        skip=lambda part: part.unit not in force_periods and should_skip_existing_partition(part.path, force=force),
        page_limit=page_limit, revision_ledger=revision_ledger, unit="periods", every=16,
    )

def download_fundamental_ann_month_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    windows: list[tuple[str, str, str]],
    force: bool,
    page_limit: int,
    force_months: set[str] | None = None,
    revision_ledger: Path | str | None = None,
) -> None:
    force_months = force_months or set()

    def window(part: Partition) -> dict[str, str]:
        start_date, end_date, _ = part.unit
        return {"start_date": start_date, "end_date": end_date}

    land_fundamental(
        client, spec, partitions(raw_dir, spec, windows), window,
        skip=lambda part: part.unit[2] not in force_months and should_skip_existing_partition(part.path, force=force, requested_params=window(part)),
        page_limit=page_limit, revision_ledger=revision_ledger, unit="months", every=24,
    )

def download_fundamental_ts_code_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    stock_codes: list[str],
    force: bool,
    page_limit: int,
    force_codes: set[str] | None = None,
    revision_ledger: Path | str | None = None,
) -> None:
    force_codes = force_codes or set()
    land_fundamental(
        client, spec, partitions(raw_dir, spec, stock_codes), lambda part: {"ts_code": part.unit},
        skip=lambda part: part.unit not in force_codes and should_skip_existing_partition(part.path, force=force),
        page_limit=page_limit, revision_ledger=revision_ledger, unit="codes", every=500,
    )


def official_text_client(datasets: list[str], repo_root: Path, args: argparse.Namespace) -> TuShareClient:
    """The official-service client for ``datasets``, proven before anything is written.

    Each dataset is asked for one row of the run's last day, so a missing,
    expired or refused OFFICIAL_TOKEN_ENV fails here, with the lake untouched,
    rather than on the first page of a long loop.
    """
    client = official_client(repo_root, args.min_interval_seconds, args.timeout_seconds)
    for dataset in datasets:
        spec = TEXT_SPECS[dataset]
        client.query(spec.api_name, {"start_date": args.end_date, "end_date": args.end_date, "limit": 1}, spec.fields)
    return client

def as_datetime_window(value: str, *, end: bool = False) -> str:
    suffix = "23:59:59" if end else "00:00:00"
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {suffix}"


def download_text(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    datasets = selected_text_datasets(args.datasets, news_src=args.news_src)
    official = [dataset for dataset in datasets if TEXT_SPECS[dataset].official]
    # One client per source, keyed by ``spec.official``; each reads only its
    # own token. A run that cannot prove official access writes nothing, the
    # relay datasets it names included, so a job's datasets land together.
    clients: dict[bool, TuShareClient] = {}
    if len(official) < len(datasets):
        clients[False] = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    if official:
        try:
            clients[True] = official_text_client(official, repo_root, args)
        except Exception as exc:  # noqa: BLE001 - every cause leaves the lake untouched
            # One line in the runner's "SomeError: ..." form, so the job state
            # names the cause; the client has already scrubbed the token from it.
            print(
                f"OfficialAccessError: {' '.join(official)} could not be fetched from the official "
                f"TuShare service with {OFFICIAL_TOKEN_ENV}; nothing was written: {exc}",
                flush=True,
            )
            return NO_MUTATION_FAILURE_EXIT_CODE
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    windows = month_windows(args.start_date, args.end_date)
    days = date_range_days(args.start_date, args.end_date)
    for dataset in datasets:
        spec = TEXT_SPECS[dataset]
        client = clients[spec.official]
        start_date = max(args.start_date, spec.start_date)
        dataset_windows = [(s, e, m) for s, e, m in windows if e >= start_date]
        dataset_days = [d for d in days if d >= start_date]
        page_limit = spec_page_limit(spec, args.page_limit)
        if spec.strategy == FetchStrategy.RANGE_MONTH:
            download_text_range_month(client, raw_dir, spec, dataset_windows, args.force, page_limit, revision_ledger)
        elif spec.strategy == FetchStrategy.TIME_RANGE_MONTH:
            download_text_time_range_month(client, raw_dir, spec, dataset_windows, args.force, page_limit, args, revision_ledger)
        elif spec.strategy == FetchStrategy.NEWS_SRC_DAY:
            download_text_news_src_day(client, raw_dir, spec, dataset_days, args.force, page_limit, args, revision_ledger)
        elif spec.strategy == FetchStrategy.DAY:
            download_text_day(client, raw_dir, spec, dataset_days, args.force, revision_ledger)
        else:
            raise RuntimeError(f"unsupported text strategy {spec.strategy} for {dataset}")
    print(f"Text download finished under {raw_dir}")
    return 0

def landing_first_seen(path: Path, answer: pd.DataFrame, key_columns: list[str] | tuple[str, ...]) -> pd.Series:
    """When the lake first held each row of a vendor answer about to be merged
    into the partition at ``path`` (CN time, NaT where it keeps no such time):
    the partition's first-seen row, or now for a key this write lands in a
    partition of the versioned era. ``publish`` records its own instant for
    such a key a moment later from the same clock, inside the same landing
    run, so the stamp a later refresh derives from the recorded row is the
    same."""
    seen = first_seen_at(path, answer)
    if seen is None:
        seen = pd.Series(pd.NaT, index=answer.index, dtype=f"datetime64[ns, {CN_TZ}]")
    if versions.records_observations(path):
        held = key_strings(pd.read_parquet(path, columns=list(key_columns)), key_columns) if path.exists() else pd.Series()
        landing = seen.isna() & ~key_strings(answer, key_columns).isin(set(held))
        seen = seen.mask(landing, pd.Timestamp(versions.utc_now("microseconds")).tz_convert(CN_TZ))
    return seen


def download_text_range_month(client: TuShareClient, raw_dir: Path, spec: DatasetSpec, windows: list[tuple[str, str, str]], force: bool, page_limit: int, revision_ledger: Path | str | None = None) -> None:
    def request(part: Partition) -> dict[str, str]:
        start_date, end_date, _ = part.unit
        return {"start_date": start_date, "end_date": end_date}

    def pull(part: Partition) -> tuple[bool, int, int]:
        params = request(part)
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        answer = frame(result)
        df = augment_text_frame(answer, spec, landing_first_seen(part.path, answer, spec.key_columns))
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, "month": part.unit[2], "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            # range_month queries window on the DATE column (ann_date/report_date);
            # retention must match it — keying on the collection-time column keeps
            # out-of-window rec_time rows forever, so source revisions/retractions
            # of anns_d/report_rc could never be purged.
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            revision_ledger=revision_ledger,
            deduplicate=spec.unique_key,
        )
        return did_write, rows, pages

    land(
        partitions(raw_dir, spec, windows), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=request(part)),
        revision_ledger=revision_ledger, name=spec.api_name, unit="months", every=24,
    )

def download_text_time_range_month(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    windows: list[tuple[str, str, str]],
    force: bool,
    page_limit: int,
    args: argparse.Namespace,
    revision_ledger: Path | str | None = None,
) -> None:
    def request(part: Partition) -> dict[str, str]:
        start_date, end_date, _ = part.unit
        params = {"start_date": as_datetime_window(start_date), "end_date": as_datetime_window(end_date, end=True)}
        if part.dims["src"]:
            params["src"] = part.dims["src"]
        return params

    def pull(part: Partition) -> tuple[bool, int, int]:
        start_date, end_date, month = part.unit
        params = request(part)
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = augment_text_frame(frame(result), spec)
        did_write, rows = write_window_merged_partition(
            part.path,
            df,
            api_name=spec.api_name,
            params={**params, "month": month, "pagination": {"page_limit": page_limit, "pages": pages}},
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.time_column, spec.date_column, "available_at"],
            start_date=start_date,
            end_date=end_date,
            revision_ledger=revision_ledger,
        )
        return did_write, rows, pages

    for source, group in groupby(partitions(raw_dir, spec, windows, args), key=lambda part: part.dims["src"]):
        land(
            list(group), pull,
            skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=request(part)),
            revision_ledger=revision_ledger, name=f"{spec.api_name}/src={directory_name(source)}", unit="months", every=24,
        )

def download_text_news_src_day(
    client: TuShareClient,
    raw_dir: Path,
    spec: DatasetSpec,
    days: list[str],
    force: bool,
    page_limit: int,
    args: argparse.Namespace,
    revision_ledger: Path | str | None = None,
) -> None:
    def request(part: Partition) -> dict[str, str]:
        return {"src": part.dims["src"], "start_date": as_datetime_window(part.unit), "end_date": as_datetime_window(part.unit, end=True)}

    def pull(part: Partition) -> tuple[bool, int, int]:
        params = request(part)
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        meta_params = {**params, "date": part.unit, "pagination": {"page_limit": page_limit, "pages": pages}}
        landed, rows = write_frame(part.path, augment_text_frame(frame(result), spec), spec, meta_params, revision_ledger)
        return landed, rows, pages

    for source, group in groupby(partitions(raw_dir, spec, days, args), key=lambda part: part.dims["src"]):
        land(
            list(group), pull,
            skip=lambda part: should_skip_existing_partition(part.path, force=force, requested_params=request(part)),
            revision_ledger=revision_ledger, name=f"{spec.api_name}/src={directory_name(source)}", unit="days", every=250,
        )

def download_text_day(client: TuShareClient, raw_dir: Path, spec: DatasetSpec, days: list[str], force: bool, revision_ledger: Path | str | None = None) -> None:
    def pull(part: Partition) -> tuple[bool, int, int]:
        params = {spec.date_column or "date": part.unit}
        result = client.query(spec.api_name, params, spec.fields)
        landed, rows = write_frame(part.path, augment_text_frame(frame(result), spec), spec, params, revision_ledger)
        return landed, rows, 0

    land(
        partitions(raw_dir, spec, days), pull,
        skip=lambda part: should_skip_existing_partition(part.path, force=force),
        revision_ledger=revision_ledger, name=spec.api_name, unit="days", every=250,
    )

def set_download_defaults(args: argparse.Namespace) -> None:
    if args.min_interval_seconds is None:
        # One rate for every tier: the vendor cooldown is global, so the old
        # per-tier table (0.12s on reference, 0.18s on daily) only meant the
        # lighter tiers tripped it first.
        args.min_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    if args.timeout_seconds is None:
        args.timeout_seconds = 90 if args.tier in {"fundamental", "event_flow", "board_trading", "text_evidence", "macro", "global"} else 60
    if args.page_limit is None:
        # None leaves each text and board dataset its spec's own limit.
        args.page_limit = {
            "text_evidence": None,
            "fundamental": 7000,
            "daily": TRADE_DATE_PAGE_LIMIT,
            "board_trading": None,
        }.get(args.tier, 10000)

def download_selected_tier(args: argparse.Namespace) -> int:
    """Run one tier; when it returns 0, each dataset it pulled is reported to
    the runner as complete (``versions.report_complete``)."""
    set_download_defaults(args)
    tiers = {
        "reference": (download_reference, lambda: [d for d in REFERENCE_DATASETS if not (d == "bak_basic" and args.skip_bak_basic)]),
        "daily": (download_daily, lambda: selected_daily_datasets(args)),
        "fundamental": (download_fundamental, lambda: selected_fundamental_datasets(args.datasets)),
        "event_flow": (download_event_flow, lambda: selected_event_flow_download_datasets(args)),
        "board_trading": (download_board_trading, lambda: selected_board_trading_datasets(args)),
        "text_evidence": (download_text, lambda: selected_text_datasets(args.datasets, news_src=args.news_src)),
        "macro": (download_macro, lambda: selected_macro_datasets(args)),
        "global": (download_macro, lambda: selected_macro_datasets(args)),
    }
    if args.tier not in tiers:
        raise RuntimeError(f"unknown download tier {args.tier}")
    run, pulled = tiers[args.tier]
    code = run(args)
    if code == 0:
        for dataset in pulled():
            report_complete(dataset)
    return code

def ns_from(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)

def sidecar_params(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return read_json_object(meta_path).get("params") or {}
    except Exception:  # noqa: BLE001 - unreadable sidecar is treated as having no params
        return {}

def normalized_coverage_bound(value: Any, *, end: bool) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if re.fullmatch(r"\d{8}", digits):
        return f"{digits}{'235959' if end else '000000'}"
    if re.fullmatch(r"\d{14}", digits):
        return digits
    if len(digits) > 14:
        return digits[:14]
    return ""

def existing_partition_covers_request(path: Path, requested_params: dict[str, Any] | None = None) -> bool:
    if not path.exists():
        return False
    requested_params = requested_params or {}
    if "start_date" in requested_params and "end_date" in requested_params:
        existing = sidecar_params(path)
        existing_start = normalized_coverage_bound(existing.get("start_date"), end=False)
        existing_end = normalized_coverage_bound(existing.get("end_date"), end=True)
        requested_start = normalized_coverage_bound(requested_params.get("start_date"), end=False)
        requested_end = normalized_coverage_bound(requested_params.get("end_date"), end=True)
        return bool(existing_start and existing_end and existing_start <= requested_start and existing_end >= requested_end)
    return True

def should_skip_existing_partition(path: Path, *, force: bool, requested_params: dict[str, Any] | None = None) -> bool:
    return not force and committed_partition_intact(path) and existing_partition_covers_request(path, requested_params)

def run_update_step(label: str, fn, args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    print(f"update step start: {label}")
    started = time.monotonic()
    code = fn(args)
    elapsed = round(time.monotonic() - started, 3)
    summary.append({"step": label, "exit_code": int(code), "elapsed_seconds": elapsed})
    if code:
        raise RuntimeError(f"update step failed: {label} exit_code={code}")
    print(f"update step done: {label} elapsed_seconds={elapsed}")

def update_share_float_complete_data(
    args: argparse.Namespace,
    summary: list[dict[str, Any]],
    *,
    start_date: str,
    force: bool,
) -> None:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    run_update_step(
        "share_float_complete",
        download_share_float_complete,
        ns_from(
            args,
            ann_start_date=start_date,
            ann_end_date=args.end_date,
            float_start_date=start_date,
            float_end_date=args.end_date,
            skip_ann_date=False,
            rescue_ann_date=[],
            float_rescue_date=[],
            rescue_code=[],
            rescue_codes_file=None,
            no_anns_candidates=False,
            no_cross_path_candidates=False,
            skip_float_date_union=False,
            force=force,
            write_union=True,
            output=args.share_float_process_output,
            revision_ledger=revision_ledger,
        ),
        summary,
    )

def update_all_dimensions(args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    """Run the update's steps in order. Each step gets the update's own
    arguments with only its differences overridden."""
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    start_date = args.start_date
    if parse_yyyymmdd(start_date) > parse_yyyymmdd(args.end_date):
        raise RuntimeError(f"start_date {start_date} is after end_date {args.end_date}")
    macro_start_date = min(start_date, args.macro_start_date, key=parse_yyyymmdd)
    refresh_open_window = args.refresh_open_window
    force_open_window = bool(args.force or refresh_open_window)
    trade_cal_end_date = format_yyyymmdd(
        parse_yyyymmdd(args.end_date) + timedelta(days=max(args.trade_cal_lookahead_days, 0))
    )

    def tier(label: str, datasets: str | None, force: bool, **overrides: Any) -> None:
        run_update_step(label, download_selected_tier, ns_from(
            args, tier=label, datasets=getattr(args, datasets) if datasets else None, force=force,
            revision_ledger=revision_ledger, **overrides,
        ), summary)

    tier(
        "reference", None, args.force,
        bak_start_date=args.bak_start_date or start_date,
        trade_cal_end_date=trade_cal_end_date,
        min_interval_seconds=args.reference_min_interval_seconds or args.min_interval_seconds,
    )
    tier("daily", "daily_datasets", args.force)
    tier("macro", "macro_datasets", force_open_window, macro_start_date=macro_start_date)
    if args.include_global:
        tier("global", "global_datasets", force_open_window, macro_start_date=macro_start_date)
    tier("event_flow", "event_datasets", force_open_window, zero_rows_not_ready=False)
    if args.include_intraday:
        intraday_force_from = None
        intraday_lookback = args.intraday_refresh_lookback_days
        if refresh_open_window and intraday_lookback > 0:
            # Scope FORCE to the tail only, but keep the full window as the
            # scan range: earlier days whose by-date file is missing or
            # invalid are re-downloaded (an outage self-heals on the next
            # evening run instead of requiring a manual backfill).
            intraday_force_from = max(
                start_date,
                format_yyyymmdd(parse_yyyymmdd(args.end_date) - timedelta(days=intraday_lookback - 1)),
            )
        run_update_step(
            "intraday_by_date",
            update_intraday_by_date,
            ns_from(args, force=args.force, force_from=intraday_force_from, revision_ledger=revision_ledger),
            summary,
        )
    if args.include_share_float_complete:
        update_share_float_complete_data(args, summary, start_date=start_date, force=force_open_window)
    if args.include_text:
        tier("text_evidence", "text_datasets", force_open_window)
    tier("fundamental", "fundamental_datasets", args.force)
    # Board datasets run last: none of them feeds Paper, and a vendor change
    # to one of them (twice in 2026-11, to a table since retired) must not
    # stop the fundamental refresh that Paper's readiness check requires.
    if args.include_board_trading:
        tier("board_trading", "board_datasets", force_open_window)

def update_data(args: argparse.Namespace) -> int:
    if args.min_interval_seconds is None:
        args.min_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    if args.timeout_seconds is None:
        args.timeout_seconds = 120
    summary: list[dict[str, Any]] = []
    update_all_dimensions(args, summary)
    print(json.dumps({"status": "ok", "start_date": args.start_date, "end_date": args.end_date, "steps": summary}, ensure_ascii=False, indent=2))
    return 0


def add_download_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("download", help="download TuShare raw data by semantic tier")
    parser.add_argument("--tier", required=True, choices=DOWNLOAD_TIER_CHOICES)
    add_raw_arg(parser)
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--bak-start-date", default="20160101")
    parser.add_argument("--end-date", default=cn_today())
    parser.add_argument("--trade-cal-end-date", help="Optional reference-tier lookahead end date used only for trade_cal coverage.")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--refresh-daily-datasets", nargs="+", choices=DAILY_DOWNLOAD_DATASETS, default=[])
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--skip-bak-basic", action="store_true")
    parser.add_argument("--refresh-reference-datasets", nargs="+", choices=REFERENCE_DATASETS, default=[])
    parser.add_argument(
        "--namechange-recent-days",
        type=int,
        default=0,
        help="Without a forced namechange: replace the name changes announced in the last N days before "
        "--end-date (and since) by one call by announcement date, over the table the full per-code loop wrote.",
    )
    parser.add_argument("--fundamental-refresh-period-count", type=int, default=0)
    parser.add_argument("--fundamental-refresh-ann-month-count", type=int, default=0)
    parser.add_argument("--fundamental-refresh-ts-code-datasets", nargs="+", choices=["dividend", "fina_audit", "fina_mainbz_vip"], default=[])
    parser.add_argument("--fundamental-refresh-event-days", type=int, default=90)
    parser.add_argument("--fundamental-dividend-probe-days", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--history-floor",
        type=yyyymmdd_arg,
        help="Backfill only: lower the research-history floor (RESEARCH_HISTORY_FLOOR, and the "
        "tier's own per-dataset floors) to this date for the datasets this run downloads, so "
        "--start-date can reach below it. Never raises a floor.",
    )
    parser.add_argument(
        "--zero-rows-not-ready",
        action="store_true",
        help="event_flow tier only: treat required zero-row responses as source-not-ready instead of failing. "
        "Exits 75 (no-mutation retry contract) when nothing was written; with partial writes it commits and "
        "defers the empty partitions to the retry job. For pre-open cron attempts that race source publication.",
    )
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--codes", nargs="+", help="Optional explicit ts_code list for targeted per-code fundamental refreshes.")
    parser.add_argument("--page-limit", type=int, help="Optional requested page size; text evidence datasets are clamped to official per-call limits.")
    parser.add_argument("--news-src", action="append", default=[], help="News short-message source; repeatable. Defaults to all official TuShare sources; use all to expand explicitly.")
    parser.add_argument("--major-news-src", action="append", default=[], help="Optional major_news source filter; repeatable.")
    add_board_filter_args(parser)
    add_macro_filter_args(parser)
    add_runtime_args(parser, min_interval=None, timeout=None)
    # Only update sets a retained macro floor; a tier download starts at --start-date.
    parser.set_defaults(macro_start_date=None)

def add_update_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("update", help="fill missing TuShare data across all retained domains")
    add_raw_arg(parser)
    parser.add_argument("--end-date", default=cn_today())
    parser.add_argument("--start-date", required=True, help="Fill missing data from this date through --end-date across all retained data domains.")
    parser.add_argument("--bak-start-date", help="Optional bak_basic lower bound. Defaults to --start-date.")
    parser.add_argument(
        "--trade-cal-lookahead-days",
        type=int,
        default=7,
        help="Extend reference trade_cal this many calendar days beyond --end-date so pre-open jobs and next-session PIT mapping have calendar coverage.",
    )
    parser.add_argument("--daily-datasets", nargs="+", choices=DAILY_DOWNLOAD_DATASETS)
    parser.add_argument(
        "--refresh-daily-datasets",
        nargs="+",
        choices=DAILY_DOWNLOAD_DATASETS,
        default=[],
        help="Daily trade-date datasets to force-refresh within the update window; cron config selects the routine recent-window revision set.",
    )
    parser.add_argument(
        "--refresh-reference-datasets",
        nargs="+",
        choices=REFERENCE_DATASETS,
        default=["stock_basic", "namechange", "index_classify", "index_member_all"],
        help="Reference datasets to force-refresh during daily update; heavier datasets should be explicit.",
    )
    parser.add_argument(
        "--refresh-open-window",
        action="store_true",
        help="Force-refresh the rolling update window for macro/global/event/board/text/share_float tiers; intended for cron, not large manual history fills.",
    )
    parser.add_argument("--reference-min-interval-seconds", type=float, help="Optional lower call frequency for the reference refresh step.")
    parser.add_argument(
        "--namechange-recent-days",
        type=int,
        default=0,
        help="Without a forced namechange: replace the name changes announced in the last N days before "
        "--end-date (and since) by one call by announcement date, over the table the full per-code loop wrote.",
    )
    parser.add_argument("--fundamental-datasets", nargs="+", choices=FUNDAMENTAL_DATASETS)
    parser.add_argument(
        "--fundamental-refresh-period-count",
        type=int,
        default=6,
        help="Force-refresh the latest N financial report periods during update, even when the rolling update window does not include quarter-end dates.",
    )
    parser.add_argument(
        "--fundamental-refresh-ann-month-count",
        type=int,
        default=3,
        help="Force-refresh the latest N announcement months for forecast/express style datasets.",
    )
    parser.add_argument(
        "--fundamental-refresh-ts-code-datasets",
        nargs="+",
        choices=["dividend", "fina_audit", "fina_mainbz_vip"],
        default=["dividend", "fina_audit", "fina_mainbz_vip"],
        help="ts_code snapshot datasets to refresh for recently affected stocks instead of forcing the whole market.",
    )
    parser.add_argument(
        "--fundamental-refresh-event-days",
        type=int,
        default=90,
        help="Select affected ts_code snapshots from recently visible financial announcement/disclosure rows in this trailing-day window.",
    )
    parser.add_argument(
        "--fundamental-dividend-probe-days",
        type=int,
        default=90,
        help="Probe recent dividend date fields to discover affected stocks for targeted ts_code snapshot refresh.",
    )
    parser.add_argument("--macro-datasets", nargs="+", choices=MACRO_DATASETS)
    parser.add_argument("--global-datasets", nargs="+", choices=MACRO_DATASETS)
    parser.add_argument(
        "--macro-start-date",
        default=RESEARCH_HISTORY_FLOOR,
        help="Retained lower bound for macro/global range-style datasets during rolling updates; prevents short-window range files from replacing full-context coverage.",
    )
    parser.add_argument("--event-datasets", nargs="+", choices=[dataset for dataset in EVENT_FLOW_DATASETS if dataset != "share_float"])
    parser.add_argument("--board-datasets", nargs="+", choices=BOARD_TRADING_DATASETS)
    parser.add_argument(
        "--include-board-trading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update 打板专题数据 by default; use --no-include-board-trading for lightweight refreshes.",
    )
    parser.add_argument("--text-datasets", nargs="+", choices=TEXT_DATASETS)
    parser.add_argument(
        "--include-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update text evidence by default; the production evening job passes --no-include-text because text is a natural-day domain with its own daily job.",
    )
    parser.add_argument(
        "--include-global",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update the global tier by default; the production evening job passes --no-include-global because overseas indices/FX/rates/economic events publish on A-share holidays and land via their own natural-day job.",
    )
    parser.add_argument(
        "--include-intraday",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update final by-date 1-minute files by default; use --no-include-intraday for lightweight metadata-only refreshes.",
    )
    parser.add_argument("--output-dataset", default=STK_MINS_BY_DATE_DATASET)
    parser.add_argument("--expected-codes-source", choices=["daily", "active", "minute"], default="minute")
    parser.add_argument("--codes", nargs="+")
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--min-rows-per-day", type=int, default=0)
    parser.add_argument("--allow-missing-codes", type=int, default=0)
    parser.add_argument("--existing-allow-missing-codes", type=int, default=50)
    parser.add_argument("--allow-validation-warnings", action="store_true")
    parser.add_argument("--intraday-refresh-lookback-days", type=int, default=0, help="When --refresh-open-window is set, force-refresh only this trailing calendar-day window for by-date minute files.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--include-share-float-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh recent share_float partitions and merge them into the durable share_float_complete baseline.",
    )
    parser.add_argument("--rescue-ann-limit-hits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rescue-universe", choices=["candidate", "explicit", "all_a"], default="candidate")
    parser.add_argument("--max-rescue-calls", type=int, default=50000)
    parser.add_argument("--union-output", default="data/raw/share_float_complete/share_float_complete.parquet")
    parser.add_argument("--share-float-process-output", help="Optional temporary process report path for share_float_complete.")
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--skip-bak-basic", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page-limit", type=int)
    parser.add_argument("--news-src", action="append", default=[])
    parser.add_argument("--major-news-src", action="append", default=[])
    add_board_filter_args(parser)
    add_macro_filter_args(parser)
    add_runtime_args(parser, min_interval=None, timeout=None)


def add_auction_recheck_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("recheck-stk-auction", help="evening auction self-heal: re-read the latest open day, capture missing days in the window")
    add_raw_arg(parser)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--landing-job", default="cn_evening_auction_backfill")
    parser.add_argument("--page-limit", type=int, default=TRADE_DATE_PAGE_LIMIT)
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--min-interval-seconds", type=float, default=MIN_REQUEST_INTERVAL_SECONDS)
    parser.add_argument("--timeout-seconds", type=int, default=60)


def add_auction_capture_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("capture-open-auction", help="poll and atomically publish today's complete stk_auction frame")
    add_raw_arg(parser)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--page-limit", type=int, default=TRADE_DATE_PAGE_LIMIT)
    parser.add_argument("--max-wait-seconds", type=float, default=170.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--stable-reads", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=AUCTION_CAPTURE_MIN_ROWS_FLOOR)
    parser.add_argument("--min-previous-day-ratio", type=float, default=AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO)
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    add_runtime_args(parser, min_interval=MIN_REQUEST_INTERVAL_SECONDS, timeout=30)

def add_intraday_parser(sub: argparse._SubParsersAction) -> None:
    update = sub.add_parser("update-intraday-by-date", help="download/retry trade dates directly into final daily minute files")
    add_intraday_by_date_common_args(update, expected_codes_choices=["daily", "active", "minute"], expected_codes_default="minute")
    update.add_argument("--force", action="store_true")
    update.add_argument("--allow-validation-warnings", action="store_true")
    update.add_argument("--max-retries", type=int, default=3)
    update.add_argument("--retry-delay-seconds", type=float, default=5.0)
    update.add_argument("--page-limit", type=int)
    update.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    add_runtime_args(update, min_interval=MIN_REQUEST_INTERVAL_SECONDS, timeout=120)
    # Only update's open-window refresh forces a trailing tail.
    update.set_defaults(force_from=None)

def add_share_float_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("download-share-float-complete", help="download share_float through ann_date and targeted ts_code rescue paths")
    add_raw_arg(parser)
    parser.add_argument("--ann-start-date", default="20100101")
    parser.add_argument("--ann-end-date", default=cn_today())
    parser.add_argument("--float-start-date", default="20200101")
    parser.add_argument("--float-end-date", default=cn_today())
    parser.add_argument("--skip-ann-date", action="store_true")
    parser.add_argument("--rescue-ann-limit-hits", action="store_true", help="Retry ann_date partitions that hit 6000 rows by ann_date + ts_code.")
    parser.add_argument("--rescue-ann-date", action="append", default=[], help="Specific ann_date to retry by ts_code; repeatable.")
    parser.add_argument("--float-rescue-date", action="append", default=[], help="Specific float_date to retry by ts_code; repeatable.")
    parser.add_argument("--rescue-universe", choices=["candidate", "explicit", "all_a"], default="candidate")
    parser.add_argument("--rescue-code", action="append", default=[])
    parser.add_argument("--rescue-codes-file")
    parser.add_argument("--no-anns-candidates", action="store_true")
    parser.add_argument("--no-cross-path-candidates", action="store_true")
    parser.add_argument("--max-rescue-calls", type=int, default=50000, help="Bound on ts_code rescue calls; a run that would exceed it fails before rescuing.")
    parser.add_argument("--skip-float-date-union", action="store_true")
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-union", action="store_true")
    parser.add_argument("--union-output", default="data/raw/share_float_complete/share_float_complete.parquet")
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    add_runtime_args(parser, min_interval=MIN_REQUEST_INTERVAL_SECONDS, timeout=90)
    parser.add_argument("--output", help="Optional process report path. No status file is written by default; event-flow audit checks the union artifact.")

def add_repair_text_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "repair-text-available-at",
        help="re-derive text available_at locally under the plausibility rule (no API calls)",
    )
    add_raw_arg(parser)
    parser.add_argument("--datasets", nargs="+", default=["anns_d", "report_rc"], choices=sorted(TEXT_SPECS))
    event = sub.add_parser(
        "repair-event-available-at",
        help="re-derive trade-date event available_at locally under the current rule (no API calls)",
    )
    add_raw_arg(event)
    event.add_argument(
        "--datasets", nargs="+", required=True,
        choices=sorted(name for name, spec in EVENT_FLOW_SPECS.items() if spec.strategy == FetchStrategy.TRADE_DATE),
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_download_parser(sub)
    add_update_parser(sub)
    add_auction_capture_parser(sub)
    add_auction_recheck_parser(sub)
    add_intraday_parser(sub)
    add_share_float_parser(sub)
    add_repair_text_parser(sub)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)

def _production_repo_root() -> Path:
    """The repo this module is installed from (src layout), independent of cwd.

    Anchoring production detection to cwd let a manual run from any other
    directory mutate the production lake with neither the flock nor a
    generation transaction; the code location is the stable anchor."""
    return Path(__file__).resolve().parents[4]


def _refuse_manual_production_write(args: argparse.Namespace) -> None:
    """A direct manual command against the production data/raw is refused with
    its runner equivalent (data docs §2.2; ``manual_production_write``);
    non-production raw dirs (tests, local lakes) are unaffected. Subtree
    membership counts: --raw-dir <prod>/daily or a --union-output file under
    the production lake are production writes just the same. Only the two
    share_float subcommands have --union-output."""
    repo_root = _production_repo_root()
    production_raw = repo_root / "data" / "raw"
    targets = (args.raw_dir, getattr(args, "union_output", None))
    if not any(manual_production_write(repo_root, production_raw, target) for target in targets):
        return
    raise SystemExit(
        "direct manual writes to the production lake are not supported; run the "
        "equivalent job through the cron runner (generation + state + alerts), e.g.:\n"
        "  scripts/data/tushare_cron_update.py --job cn_evening_full "
        "--start-date <YYYYMMDD> --end-date <YYYYMMDD>   # full window incl. minute gap self-heal\n"
        "  scripts/data/tushare_cron_update.py --job cn_open_auction_capture_0927 "
        "--end-date <YYYYMMDD> --force-run               # one auction partition\n"
        "  scripts/data/tushare_cron_update.py --job cn_preopen_margin_backfill_0905 "
        "--start-date <YYYYMMDD> --end-date <YYYYMMDD>   # margin backfill"
    )


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "download":
        return download_selected_tier(args)
    if args.command == "update":
        return update_data(args)
    if args.command == "capture-open-auction":
        return capture_open_auction(args)
    if args.command == "recheck-stk-auction":
        return recheck_stk_auction(args)
    if args.command == "update-intraday-by-date":
        return update_intraday_by_date(args)
    if args.command == "download-share-float-complete":
        return download_share_float_complete(args)
    if args.command == "repair-text-available-at":
        stats = repair_text_available_at(args.raw_dir, list(args.datasets))
        print(json.dumps({"status": "ok", **stats}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "repair-event-available-at":
        stats = repair_event_available_at(args.raw_dir, list(args.datasets))
        print(json.dumps({"status": "ok", **stats}, ensure_ascii=False, sort_keys=True))
        return 0
    raise RuntimeError(f"unknown command {args.command}")


def main() -> int:
    args = parse_args()
    _refuse_manual_production_write(args)
    return _dispatch(args)

if __name__ == "__main__":
    raise SystemExit(main())
