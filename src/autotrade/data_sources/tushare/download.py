#!/usr/bin/env python3
"""TuShare download, update, and raw-data maintenance CLI for AutoTrade."""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import common as core
from .common import (
    ApiResult,
    BAK_BASIC_SPEC,
    BOARD_TRADING_SPECS,
    DAILY_SPECS,
    DEFAULT_CN_INDEX_CODES,
    EVENT_FLOW_DATASETS,
    EVENT_FLOW_SPECS,
    FUNDAMENTAL_SPECS,
    INDEX_WEIGHT_FIELDS,
    INDEX_WEIGHT_PAGE_LIMIT,
    INDEX_WEIGHT_START_DATE,
    MACRO_RETAINED_FLOOR,
    MACRO_SPECS,
    MUTATED_NOT_READY_RETRY_EXIT_CODE,
    NO_MUTATION_RETRY_EXIT_CODE,
    PHYSICAL_IDENTITY_COLUMNS,
    REVISION_EVENTS_PATH,
    SHARE_FLOAT_FIELDS,
    SHARE_FLOAT_ROW_LIMIT,
    SHARE_FLOAT_UNLOCK_TITLE_PATTERN,
    STK_AUCTION_PRICE_ABS_TOLERANCE,
    STK_MINS_API_NAME,
    STK_MINS_BY_DATE_DATASET,
    STK_MINS_DATASET,
    STK_MINS_FIELDS,
    STK_MINS_FREQ,
    REFERENCE_PAGE_LIMIT,
    committed_partition_intact,
    STK_MINS_PAGE_LIMIT,
    STK_MINS_REQUIRED_COLUMNS,
    TEXT_SPECS,
    TRADE_DATE_PAGE_LIMIT,
    BoardTradingDataset,
    EventDataset,
    FundamentalDataset,
    MacroDataset,
    TextDataset,
    TradeDateDataset,
    TuShareClient,
    active_year_windows,
    as_datetime_window,
    augment_event_frame,
    augment_macro_frame,
    augment_stk_mins_frame,
    augment_text_frame,
    date_range_days,
    format_yyyymmdd,
    frame,
    intraday_expected_codes_for_day,
    latest_sse_calendar_date,
    load_minute_universe,
    load_sse_open_dates,
    load_stock_codes,
    load_token,
    margin_exchange_column,
    margin_family_missing_exchanges,
    minute_datetime,
    month_windows,
    normalize_date_key,
    normalize_stk_mins_by_date_frame,
    normalized_date_keys,
    normalized_holder_keys,
    parquet_rows,
    parse_yyyymmdd,
    physical_identity_index,
    quarter_periods,
    query_paged,
    read_many,
    resolve_revision_ledger,
    safe_partition_value,
    select_datasets,
    selected_board_dc_hot_markets,
    selected_board_dc_hot_types,
    selected_board_hot_is_new,
    selected_board_kpl_tags,
    selected_board_ths_hot_markets,
    selected_board_ths_limit_types,
    selected_board_trading_datasets,
    selected_cn_index_codes,
    selected_daily_datasets,
    selected_eco_filter_values,
    selected_fundamental_datasets,
    selected_fx_codes,
    selected_index_codes,
    selected_intraday_datasets,
    selected_macro_datasets,
    selected_news_sources,
    selected_text_datasets,
    spec_page_limit,
    inherited_updater_lock_fd,
    stk_mins_by_date_path,
    validate_stk_mins_by_date_frame,
    write_board_result,
    write_macro_result,
    write_parquet,
    undated_duplicate_mask,
    write_parquet_revision_aware,
    yyyymmdd_to_month,
    yyyymmdd_to_quarter,
)

from autotrade.environment.data.pit import concat_rows

def write_reference_query_result(
    path: Path,
    result: ApiResult,
    *,
    api_name: str,
    params: dict[str, Any],
    required_nonempty: bool,
) -> pd.DataFrame:
    df = frame(result)
    if df.empty and path.exists():
        print(f"{api_name} params={params} returned zero rows for existing partition; skipped_empty_reference_overwrite")
        return pd.read_parquet(path)
    if df.empty and required_nonempty:
        raise RuntimeError(f"{api_name} params={params} returned zero rows for required reference partition")
    write_parquet(path, df, api_name=api_name, params=params, fields=result.fields)
    return df


def download_reference(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    refresh_datasets = set(getattr(args, "refresh_reference_datasets", None) or [])
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    trade_cal_end_date = getattr(args, "trade_cal_end_date", None) or args.end_date

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
        )

    company_fields = "ts_code,com_name,com_id,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,introduction"
    for exchange in ("SSE", "SZSE", "BSE"):
        path = raw_dir / "stock_company" / f"exchange={exchange}.parquet"
        if path.exists() and not should_force("stock_company"):
            continue
        params = {"exchange": exchange}
        result, _pages = query_paged(client, "stock_company", params, company_fields, REFERENCE_PAGE_LIMIT)
        write_reference_query_result(
            path,
            result,
            api_name="stock_company",
            params=params,
            required_nonempty=True,
        )

    open_dates = download_trade_cal(client, raw_dir, args.start_date, trade_cal_end_date, should_force("trade_cal"))
    open_dates = [trade_date for trade_date in open_dates if trade_date <= args.end_date]
    if not args.skip_bak_basic:
        download_bak_basic(
            client,
            raw_dir,
            open_dates,
            args.bak_start_date,
            should_force("bak_basic"),
            revision_ledger,
            getattr(args, "allow_empty_revision_overwrite", False),
        )
    download_namechange(client, raw_dir, should_force("namechange"))
    classify = download_index_classify(client, raw_dir, should_force("index_classify"))
    download_index_member_all(client, raw_dir, classify["SW2021"], should_force("index_member_all"))
    download_index_member_history(client, raw_dir, classify["SW2014"], should_force("index_member"))
    download_ths_catalog(client, raw_dir, should_force("ths_index"))
    download_index_reference(client, raw_dir, should_force("index_basic"))
    download_index_weight(
        client,
        raw_dir,
        args.start_date,
        args.end_date,
        should_force("index_weight"),
        revision_ledger,
        getattr(args, "allow_empty_revision_overwrite", False),
    )
    print(f"reference download finished under {raw_dir}")
    return 0

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
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
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

def download_trade_cal(client: TuShareClient, raw_dir: Path, start_date: str, end_date: str, force: bool) -> list[str]:
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
                result = client.query("trade_cal", params, fields)
                df = merge_trade_cal_partition(path, frame(result))
                meta_params = dict(params)
                meta_params["merge_existing"] = True
                write_parquet(path, df, api_name="trade_cal", params=meta_params, fields=fields.split(","))
            else:
                result = client.query("trade_cal", params, fields)
                df = merge_trade_cal_partition(path, frame(result))
                meta_params = dict(params)
                if path.exists():
                    meta_params["merge_existing"] = True
                write_parquet(path, df, api_name="trade_cal", params=meta_params, fields=result.fields)
            if exchange == "SSE" and not df.empty:
                sse_open.update(
                    date_key
                    for date_key in df.loc[df["is_open"].astype(str) == "1", "cal_date"].map(normalize_date_key).tolist()
                    if date_key
                )
    return sorted(sse_open)

def sse_trade_cal_covers(raw_dir: Path, start_date: str, end_date: str) -> bool:
    files = sorted((raw_dir / "trade_cal" / "exchange=SSE").glob("year=*.parquet"))
    if not files:
        return False
    calendar = read_many(files, columns=["cal_date"])
    if calendar.empty or "cal_date" not in calendar.columns:
        return False
    dates = calendar["cal_date"].dropna().map(normalize_date_key)
    dates = dates[dates != ""]
    return bool(not dates.empty and dates.min() <= start_date and dates.max() >= end_date)

def ensure_trade_cal_coverage(client: TuShareClient, raw_dir: Path, start_date: str, end_date: str) -> bool:
    """Extend local SSE trade_cal when it lags; True means the lake was written."""
    if parse_yyyymmdd(start_date) > parse_yyyymmdd(end_date):
        return False
    if sse_trade_cal_covers(raw_dir, start_date, end_date):
        return False
    print(f"refreshing trade_cal coverage for {start_date}-{end_date}")
    download_trade_cal(client, raw_dir, start_date, end_date, force=False)
    return True

def download_bak_basic(
    client: TuShareClient,
    raw_dir: Path,
    trade_dates: list[str],
    start_date: str,
    force: bool,
    revision_ledger: Path,
    allow_empty_revision_overwrite: bool,
) -> None:
    fields = "trade_date,ts_code,name,industry,area,pe,float_share,total_share,total_assets,liquid_assets,fixed_assets,reserved,reserved_pershare,eps,bvps,pb,list_date,undp,per_undp,rev_yoy,profit_yoy,gpr,npr,holder_num"
    dates = [d for d in trade_dates if d >= start_date]
    for index, trade_date in enumerate(dates, start=1):
        path = raw_dir / "bak_basic" / f"trade_date={trade_date}.parquet"
        if path.exists() and not force:
            continue
        params = {"trade_date": trade_date}
        result = client.query("bak_basic", params, fields)
        write_parquet_revision_aware(
            path,
            frame(result),
            api_name="bak_basic",
            params=params,
            fields=result.fields,
            key_columns=list(BAK_BASIC_SPEC.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            allow_key_removal_overwrite=True,
        )
        if index % 250 == 0:
            print(f"bak_basic {index}/{len(dates)}")

def download_namechange(client: TuShareClient, raw_dir: Path, force: bool) -> None:
    path = raw_dir / "namechange" / "namechange.parquet"
    if path.exists() and not force:
        return
    fields = "ts_code,name,start_date,end_date,ann_date,change_reason"
    frames: list[pd.DataFrame] = []
    for index, code in enumerate(load_stock_codes(raw_dir), start=1):
        result = client.query("namechange", {"ts_code": code}, fields)
        df = frame(result)
        if not df.empty:
            frames.append(df)
        if index % 500 == 0:
            print(f"namechange {index}")
    combined = concat_rows(frames) if frames else pd.DataFrame(columns=fields.split(","))
    if combined.empty and path.exists():
        print("namechange returned zero rows for existing partition; skipped_empty_reference_overwrite")
        return
    deduped = combined.drop_duplicates().sort_values(["ts_code", "start_date", "name"], na_position="last").reset_index(drop=True)
    write_parquet(path, deduped, api_name="namechange", params={"strategy": "per_ts_code_all_stock_basic"}, fields=fields.split(","))

def download_index_classify(client: TuShareClient, raw_dir: Path, force: bool) -> dict[str, pd.DataFrame]:
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
        )
    return frames

def download_index_member_all(client: TuShareClient, raw_dir: Path, classify: pd.DataFrame, force: bool) -> None:
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
            if path.exists():
                print(f"index_member_all l1_code={code} returned zero rows for existing partition; skipped_empty_reference_overwrite")
                continue
            # A valid SW2021 L1 always has current members; an empty Y+N pull is a vendor failure.
            raise RuntimeError(f"index_member_all l1_code={code} returned zero rows for required reference partition")
        combined = combined.sort_values(["ts_code", "in_date"], na_position="last").reset_index(drop=True)
        write_parquet(
            path,
            combined,
            api_name="index_member_all",
            params={"l1_code": code, "is_new": ["Y", "N"]},
            fields=fields.split(","),
        )

def download_index_member_history(client: TuShareClient, raw_dir: Path, classify_sw2014: pd.DataFrame, force: bool) -> None:
    """SW2014 L1 membership via the legacy index_member API. The old taxonomy
    closed on 20211213 (every retired index's members carry that out_date), so
    the history is frozen and only re-downloads under an explicit force."""
    fields = "index_code,con_code,in_date,out_date,is_new"
    l1_codes = classify_sw2014.loc[classify_sw2014["level"].astype(str) == "L1", "index_code"].dropna().astype(str).tolist()
    for code in sorted(l1_codes):
        path = raw_dir / "index_member" / f"l1_code={code}.parquet"
        if path.exists() and not force:
            continue
        params = {"index_code": code}
        result, _pages = query_paged(client, "index_member", params, fields, 3000)
        write_reference_query_result(
            path,
            result,
            api_name="index_member",
            params=params,
            required_nonempty=True,
        )

def download_ths_catalog(client: TuShareClient, raw_dir: Path, force: bool) -> None:
    """THS concept/industry index catalog + per-index membership."""
    path = raw_dir / "ths_index" / "catalog.parquet"
    fields = "ts_code,name,count,exchange,list_date,type"
    if not path.exists() or force:
        result, _pages = query_paged(client, "ths_index", {"exchange": "A"}, fields, REFERENCE_PAGE_LIMIT)
        write_reference_query_result(path, result, api_name="ths_index", params={"exchange": "A"}, required_nonempty=True)
    catalog = pd.read_parquet(path)
    member_fields = "ts_code,con_code,con_name"
    # N = concept, I = industry; other types (S regional etc.) are noise for us.
    codes = catalog.loc[catalog["type"].astype(str).isin(["N", "I"]), "ts_code"].dropna().astype(str)
    for code in sorted(codes):
        member_path = raw_dir / "ths_member" / f"ts_code={code}.parquet"
        if member_path.exists() and not force:
            continue
        result = client.query("ths_member", {"ts_code": code}, member_fields)
        write_reference_query_result(
            member_path, result, api_name="ths_member", params={"ts_code": code}, required_nonempty=False
        )


def download_index_reference(client: TuShareClient, raw_dir: Path, force: bool) -> None:
    """Index metadata catalog. Must page: the unpaged call silently truncated
    at the vendor's 8,000-row cap (12,484 exist; the whole CI and NH families
    were missing)."""
    path = raw_dir / "index_basic" / "catalog.parquet"
    fields = "ts_code,name,fullname,market,publisher,index_type,category,base_date,base_point,list_date,weight_rule,desc,exp_date"
    if not path.exists() or force:
        result, _pages = query_paged(client, "index_basic", {}, fields, REFERENCE_PAGE_LIMIT)
        write_reference_query_result(path, result, api_name="index_basic", params={}, required_nonempty=True)


def download_index_weight(
    client: TuShareClient,
    raw_dir: Path,
    start_date: str,
    end_date: str,
    force: bool,
    revision_ledger: Path | str | None,
    allow_empty_revision_overwrite: bool,
) -> None:
    """Monthly core-index constituent weights, one partition per index per year."""
    start_date = max(start_date, INDEX_WEIGHT_START_DATE)
    if end_date < start_date:
        return
    written = 0
    skipped = 0
    total_rows = 0
    for code in DEFAULT_CN_INDEX_CODES:
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
            year_start = max(start_date, f"{year}0101")
            year_end = min(end_date, f"{year}1231")
            path = raw_dir / "index_weight" / f"index_code={safe_partition_value(code)}" / f"year={year}.parquet"
            params = {"index_code": code, "start_date": year_start, "end_date": year_end}
            if should_skip_existing_partition(path, force=force, requested_params=params):
                skipped += 1
                continue
            result, pages = query_paged(client, "index_weight", params, INDEX_WEIGHT_FIELDS, INDEX_WEIGHT_PAGE_LIMIT)
            meta_params = dict(params)
            meta_params["pagination"] = {"page_limit": INDEX_WEIGHT_PAGE_LIMIT, "pages": pages}
            df = frame(result)
            total_rows += write_window_merged_partition(
                path,
                df,
                api_name="index_weight",
                params=meta_params,
                fields=list(df.columns) if len(df.columns) else result.fields,
                key_columns=["index_code", "con_code", "trade_date"],
                date_columns=["trade_date"],
                start_date=year_start,
                end_date=year_end,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            )
            written += 1
    print(f"index_weight done codes={len(DEFAULT_CN_INDEX_CODES)} skipped={skipped} written={written} rows_written={total_rows}")


def download_daily(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    ensure_trade_cal_coverage(client, raw_dir, args.start_date, args.end_date)
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    refresh_datasets = set(getattr(args, "refresh_daily_datasets", None) or [])
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
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
            getattr(args, "allow_empty_revision_overwrite", False),
        )
        if zero_skipped and not spec.zero_rows_ok:
            required_zero_skipped.append(f"{spec.api_name}:{zero_skipped}")
    if required_zero_skipped:
        raise RuntimeError(f"required daily partitions returned zero rows and were not written: {required_zero_skipped}")
    print(f"daily market download finished under {raw_dir}")
    return 0


def _auction_capture_min_rows(
    raw_dir: Path,
    trade_date: str,
    *,
    floor: int,
    ratio: float,
) -> int:
    """Completeness floor for one auction partition vs the previous one.

    Percentage-only by design: an absolute max-drop bound (previously
    prev-10 rows) dominated the ratio for full-market partitions and
    rejected legitimate provider-side coverage narrowing (2026-07:
    ~1,100 fund/ETF rows removed retroactively). The stability double-read
    still guards same-day flapping; the ratio guards truncation."""
    previous = sorted(
        path
        for path in (raw_dir / "stk_auction").glob("trade_date=*.parquet")
        if path.stem.split("=", 1)[-1] < trade_date
    )
    previous_rows = pq.ParquetFile(previous[-1]).metadata.num_rows if previous else 0
    return max(int(floor), math.floor(int(previous_rows) * float(ratio)))


def _validate_auction_capture(frame_: pd.DataFrame, trade_date: str, *, min_rows: int) -> list[str]:
    required = set(DAILY_SPECS["stk_auction"].fields.split(","))
    missing = sorted(required.difference(frame_.columns))
    errors = [f"missing_columns={missing}"] if missing else []
    if frame_.empty:
        return [*errors, "empty_response"]
    dates = set(frame_["trade_date"].astype(str)) if "trade_date" in frame_.columns else set()
    if dates != {trade_date}:
        errors.append(f"unexpected_trade_dates={sorted(dates)}")
    if {"trade_date", "ts_code"}.issubset(frame_.columns):
        duplicates = int(frame_.duplicated(["trade_date", "ts_code"]).sum())
        if duplicates:
            errors.append(f"duplicate_keys={duplicates}")
        valid_codes = frame_["ts_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")
        invalid_codes = int((~valid_codes).sum())
        if invalid_codes:
            errors.append(f"invalid_codes={invalid_codes}")
    if len(frame_) < int(min_rows):
        errors.append(f"rows={len(frame_)} below_min_rows={min_rows}")
    if {"price", "vol", "amount"}.issubset(frame_.columns):
        price = pd.to_numeric(frame_["price"], errors="coerce")
        volume = pd.to_numeric(frame_["vol"], errors="coerce")
        amount = pd.to_numeric(frame_["amount"], errors="coerce")
        finite_volume = pd.Series(np.isfinite(volume), index=frame_.index)
        finite_amount = pd.Series(np.isfinite(amount), index=frame_.index)
        invalid_volume = ~finite_volume | volume.lt(0)
        invalid_amount = ~finite_amount | amount.lt(0)
        if invalid_volume.any():
            errors.append(f"invalid_vol_rows={int(invalid_volume.sum())}")
        if invalid_amount.any():
            errors.append(f"invalid_amount_rows={int(invalid_amount.sum())}")

        valid_quantities = ~invalid_volume & ~invalid_amount
        traded = valid_quantities & (volume.gt(0) | amount.gt(0))
        matched_trade = valid_quantities & volume.gt(0) & amount.gt(0)
        inconsistent = traded & ~matched_trade
        if inconsistent.any():
            errors.append(f"inconsistent_trade_rows={int(inconsistent.sum())}")

        direct_price = pd.Series(np.isfinite(price), index=frame_.index) & price.gt(0)
        recovered_price = amount.div(volume)
        recoverable = (
            volume.gt(0)
            & amount.gt(0)
            & pd.Series(np.isfinite(recovered_price), index=frame_.index)
            & recovered_price.gt(0)
        )
        bad_trade_price = matched_trade & ~recoverable
        if bad_trade_price.any():
            errors.append(f"unrecoverable_trade_price_rows={int(bad_trade_price.sum())}")
        price_mismatch = (
            matched_trade
            & direct_price
            & recoverable
            & ~pd.Series(
                np.isclose(
                    price,
                    recovered_price,
                    rtol=1e-9,
                    atol=STK_AUCTION_PRICE_ABS_TOLERANCE,
                ),
                index=frame_.index,
            )
        )
        if price_mismatch.any():
            errors.append(f"inconsistent_trade_price_rows={int(price_mismatch.sum())}")
        no_trade = valid_quantities & volume.eq(0) & amount.eq(0)
        hidden_no_trade_price = no_trade & direct_price
        if hidden_no_trade_price.any():
            errors.append(f"hidden_no_trade_price_rows={int(hidden_no_trade_price.sum())}")
    return errors


def _auction_frame_snapshot(frame_: pd.DataFrame) -> str:
    ordered = frame_.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    return ordered.fillna("").astype(str).to_json(orient="split", force_ascii=False)


# Completeness floor shared by the pre-open capture CLI defaults and the
# evening recheck (which takes no knobs).
AUCTION_CAPTURE_MIN_ROWS_FLOOR = 1000
AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO = 0.85


def _publish_auction_frame(
    raw_dir: Path,
    repo_root: Path,
    trade_date: str,
    captured: pd.DataFrame,
    result: ApiResult,
    pages: int,
    page_limit,
    landing_job: str,
    revision_ledger_arg,
) -> tuple[bool, str]:
    """Canonical-order publish with observed availability, shared by the
    pre-open capture and the evening recheck. Returns (wrote, landed_at)."""
    # Stable identity is key-based, so persist the same canonical order. A
    # source response that only reorders rows must remain byte-identical and
    # keep the first observed availability instead of looking like a revision.
    captured = captured.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    landed_at = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
    path = raw_dir / "stk_auction" / f"trade_date={trade_date}.parquet"
    availability = {
        "matched_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T09:25:00+08:00",
        "available_at": landed_at,
        "rule": "observed:cn_open_auction_capture",
        "landing_job": landing_job,
        "row_count": int(len(captured)),
    }
    revision_ledger = resolve_revision_ledger(raw_dir, revision_ledger_arg, repo_root=repo_root)
    spec = DAILY_SPECS["stk_auction"]
    wrote = write_parquet_revision_aware(
        path,
        captured,
        api_name=spec.api_name,
        params={"trade_date": trade_date, "pagination": {"page_limit": page_limit, "pages": pages}},
        fields=result.fields,
        key_columns=list(spec.key_columns),
        revision_ledger=revision_ledger,
        allow_key_removal_overwrite=False,
        extra_metadata={"availability": availability},
    )
    return bool(wrote), landed_at


def capture_open_auction(args: argparse.Namespace) -> int:
    """Poll the same-day opening-auction endpoint until a stable, complete frame lands."""
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    trade_date = normalize_date_key(args.trade_date)
    if not trade_date:
        raise ValueError(f"invalid --trade-date: {args.trade_date!r}")
    if not load_sse_open_dates(raw_dir, trade_date, trade_date, allow_empty=True):
        print(json.dumps({"status": "skipped_non_trading_day", "trade_date": trade_date}, ensure_ascii=False))
        return 0

    spec = DAILY_SPECS["stk_auction"]
    try:
        client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
        min_rows = _auction_capture_min_rows(
            raw_dir,
            trade_date,
            floor=args.min_rows,
            ratio=args.min_previous_day_ratio,
        )
        polling_started = time.monotonic()
        deadline = polling_started + max(0.0, float(args.max_wait_seconds))
        poll_interval = max(0.0, float(args.retry_delay_seconds))
        next_poll_at = polling_started
        required_stable_reads = max(1, int(args.stable_reads))
        stable_count = 0
        last_snapshot = ""
        accepted: tuple[pd.DataFrame, ApiResult, int] | None = None
        last_errors: list[str] = ["not_queried"]
        while True:
            try:
                result, pages = query_paged(
                    client,
                    spec.api_name,
                    {"trade_date": trade_date},
                    spec.fields,
                    args.page_limit,
                )
                candidate = frame(result).reset_index(drop=True)
                last_errors = _validate_auction_capture(candidate, trade_date, min_rows=min_rows)
            except Exception as exc:
                last_errors = [f"query_error={type(exc).__name__}: {exc}"]
                stable_count = 0
                last_snapshot = ""
            else:
                if not last_errors:
                    snapshot = _auction_frame_snapshot(candidate)
                    stable_count = stable_count + 1 if snapshot == last_snapshot else 1
                    last_snapshot = snapshot
                    if stable_count >= required_stable_reads:
                        accepted = (candidate, result, pages)
                        break
                else:
                    stable_count = 0
                    last_snapshot = ""
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                break
            print(
                f"stk_auction trade_date={trade_date} not ready; errors={last_errors} "
                f"stable={stable_count}/{required_stable_reads}; retrying"
            )
            if poll_interval <= 0:
                continue
            next_poll_at += poll_interval
            if next_poll_at <= now:
                skipped_slots = math.floor((now - next_poll_at) / poll_interval) + 1
                next_poll_at += skipped_slots * poll_interval
            time.sleep(min(next_poll_at - now, remaining))
    except Exception as exc:
        print(json.dumps({
            "status": "not_ready_no_mutation",
            "trade_date": trade_date,
            "errors": [f"setup_error={type(exc).__name__}: {exc}"],
        }, ensure_ascii=False, sort_keys=True))
        return NO_MUTATION_RETRY_EXIT_CODE
    if accepted is None:
        print(json.dumps({
            "status": "not_ready_no_mutation",
            "trade_date": trade_date,
            "errors": last_errors,
        }, ensure_ascii=False, sort_keys=True))
        return NO_MUTATION_RETRY_EXIT_CODE

    captured, result, pages = accepted
    wrote, landed_at = _publish_auction_frame(
        raw_dir,
        repo_root,
        trade_date,
        captured,
        result,
        pages,
        args.page_limit,
        "cn_open_auction_capture_0927",
        getattr(args, "revision_ledger", REVISION_EVENTS_PATH),
    )
    if not wrote:
        raise RuntimeError(f"stk_auction trade_date={trade_date} complete frame was not published")
    print(json.dumps({
        "status": "ok",
        "trade_date": trade_date,
        "rows": len(captured),
        "pages": pages,
        "available_at": landed_at,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def recheck_stk_auction(args: argparse.Namespace) -> int:
    """Validated evening stk_auction reconciliation inside the universal
    update, standardized with the margin family's evening self-heal shape:
    the latest open day is always re-read (late source corrections), and any
    EARLIER open day in the update window whose partition is missing or not an
    intact commit gets a validated capture attempt — a multi-day capture
    outage or torn write heals itself instead of waiting for a manual backfill. Healed days keep OBSERVED
    landing timestamps (auction visibility is deliberately not rule-stamped:
    a fixed morning stamp on an evening-healed day would show replay auction
    data at an open the live system provably lacked).

    Standard reads with the capture's completeness validation — no polling.
    An invalid or unchanged result never replaces an existing partition, and
    an invalid result never fails the whole update batch: gaps are escalated
    by the nightly audit (missing expected partition) instead of poisoning
    the batch's other datasets."""
    end_day = normalize_date_key(args.end_date)
    start_day = normalize_date_key(getattr(args, "start_date", "") or args.end_date)
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    open_days = load_sse_open_dates(raw_dir, start_day, end_day, allow_empty=True)
    if not open_days:
        print(json.dumps({"step": "stk_auction_recheck", "status": "skipped_non_trading_window", "start_date": start_day, "end_date": end_day}, ensure_ascii=False, sort_keys=True))
        return 0
    exit_code = 0
    for day in open_days:
        path = repo_root / args.raw_dir / "stk_auction" / f"trade_date={day}.parquet"
        if day != open_days[-1] and committed_partition_intact(path):
            continue  # latest day is always re-read; incomplete earlier days self-heal
        exit_code = _recheck_stk_auction_day(args, repo_root, raw_dir, day) or exit_code
    return exit_code

def _recheck_stk_auction_day(args: argparse.Namespace, repo_root: Path, raw_dir: Path, trade_date: str) -> int:
    outcome: dict[str, Any] = {"step": "stk_auction_recheck", "trade_date": trade_date}
    spec = DAILY_SPECS["stk_auction"]
    path = raw_dir / "stk_auction" / f"trade_date={trade_date}.parquet"
    existed = path.exists()
    candidate: pd.DataFrame | None = None
    result = None
    pages = 0
    try:
        client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
        min_rows = _auction_capture_min_rows(
            raw_dir,
            trade_date,
            floor=AUCTION_CAPTURE_MIN_ROWS_FLOOR,
            ratio=AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO,
        )
        result, pages = query_paged(client, spec.api_name, {"trade_date": trade_date}, spec.fields, args.page_limit)
        candidate = frame(result).reset_index(drop=True)
        errors = _validate_auction_capture(candidate, trade_date, min_rows=min_rows)
    except Exception as exc:
        errors = [f"query_error={type(exc).__name__}: {exc}"]
    if errors:
        print(json.dumps({
            **outcome, "status": "recheck_invalid", "errors": errors, "partition_exists": existed,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    snapshot = _auction_frame_snapshot(candidate)
    if committed_partition_intact(path) and _auction_frame_snapshot(pd.read_parquet(path)) == snapshot:
        # Unchanged content on an intact commit keeps the first observed
        # availability and the original file untouched. A torn pair is rewritten
        # even when the payload matches, so the commit identity can self-heal.
        print(json.dumps({
            **outcome, "status": "recheck_unchanged", "rows": int(len(candidate)),
        }, ensure_ascii=False, sort_keys=True))
        return 0
    wrote, landed_at = _publish_auction_frame(
        raw_dir,
        repo_root,
        trade_date,
        candidate,
        result,
        pages,
        args.page_limit,
        getattr(args, "landing_job", "cn_evening_auction_backfill"),
        getattr(args, "revision_ledger", REVISION_EVENTS_PATH),
    )
    if not wrote:
        # The key-removal/shrink guards kept the existing partition and have
        # already recorded the blocked revision in the ledger.
        print(json.dumps({**outcome, "status": "recheck_blocked_kept_existing"}, ensure_ascii=False, sort_keys=True))
        return 0
    print(json.dumps({
        **outcome,
        "status": "recheck_published",
        "revised_existing": existed,
        "rows": int(len(candidate)),
        "available_at": landed_at,
    }, ensure_ascii=False, sort_keys=True))
    return 0

def date_series_in_window(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    keys = series.map(normalize_date_key)
    return keys.between(start_date, end_date)

def merge_existing_window_rows(
    path: Path,
    refreshed: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    date_columns: list[str],
    key_columns: list[str],
) -> pd.DataFrame:
    if not path.exists():
        return refreshed
    existing = pd.read_parquet(path)
    if existing.empty:
        return refreshed
    date_column = next((column for column in date_columns if column and (column in existing.columns or column in refreshed.columns)), "")
    if not date_column:
        return refreshed
    if date_column not in existing.columns:
        existing[date_column] = ""
    if date_column not in refreshed.columns:
        refreshed[date_column] = ""
    retained = existing[~date_series_in_window(existing[date_column], start_date, end_date)].copy()
    merged = concat_rows([retained, refreshed])
    if key_columns and set(key_columns).issubset(merged.columns):
        merged = merged.drop_duplicates(key_columns, keep="last")
    return merged.reset_index(drop=True)

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
    allow_empty_revision_overwrite: bool,
) -> int:
    df = refreshed
    meta_params = dict(params)
    if path.exists() and (not refreshed.empty or allow_empty_revision_overwrite):
        df = merge_existing_window_rows(
            path,
            refreshed,
            start_date=start_date,
            end_date=end_date,
            date_columns=date_columns,
            key_columns=key_columns,
        )
        meta_params["merge_existing_window"] = {"start_date": start_date, "end_date": end_date}
    written = write_parquet_revision_aware(
        path,
        df,
        api_name=api_name,
        params=meta_params,
        fields=list(df.columns) if len(df.columns) else fields,
        key_columns=key_columns,
        revision_ledger=revision_ledger,
        allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        allow_key_removal_overwrite=True,
    )
    return len(df) if written else 0

def download_trade_date_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: TradeDateDataset,
    trade_dates: list[str],
    force: bool,
    page_limit: int,
    revision_ledger: Path,
    allow_empty_revision_overwrite: bool,
) -> int:
    page_limit = page_limit or TRADE_DATE_PAGE_LIMIT
    dataset_dir = raw_dir / spec.api_name
    written = 0
    skipped = 0
    zero_skipped = 0
    total_rows = 0
    for index, trade_date in enumerate(trade_dates, start=1):
        path = dataset_dir / f"trade_date={trade_date}.parquet"
        if not force:
            # A parquet whose sidecar is missing or carries another write ID is
            # an interrupted write, not a committed partition: re-attempt it.
            if committed_partition_intact(path) and (spec.zero_rows_ok or parquet_rows(path) > 0):
                skipped += 1
                if index % 250 == 0:
                    print(f"{spec.api_name} {index}/{len(trade_dates)} skipped={skipped} written={written}")
                continue
        params = {"trade_date": trade_date}
        try:
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        except Exception as exc:
            raise RuntimeError(f"{spec.api_name} trade_date={trade_date} failed: {exc}") from exc
        df = frame(result)
        if getattr(spec, "dedup_exact", False) and not df.empty:
            df = df.drop_duplicates().reset_index(drop=True)
        if df.empty and not spec.zero_rows_ok:
            zero_skipped += 1
            print(f"{spec.api_name} trade_date={trade_date} returned zero rows; skipped_write")
            continue
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=result.fields,
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        )
        if did_write:
            total_rows += len(df)
            written += 1
        else:
            zero_skipped += 1
        if index % 250 == 0:
            print(f"{spec.api_name} {index}/{len(trade_dates)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done dates={len(trade_dates)} skipped={skipped} written={written} zero_skipped={zero_skipped} rows_written={total_rows}")
    return zero_skipped

def download_macro(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    allow_empty_revision_overwrite = getattr(args, "allow_empty_revision_overwrite", False)
    retained_start_date = getattr(args, "macro_start_date", None) or args.start_date
    for dataset in selected_macro_datasets(args):
        spec = MACRO_SPECS[dataset]
        start_date = max(args.start_date, spec.start_date)
        if spec.strategy == "quarter_once":
            download_macro_quarter_once(client, raw_dir, spec, max(retained_start_date, spec.start_date), args.end_date, args.force, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "month_once":
            download_macro_month_once(client, raw_dir, spec, max(retained_start_date, spec.start_date), args.end_date, args.force, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "month_loop":
            download_macro_month_loop(client, raw_dir, spec, start_date, args.end_date, args.force, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "trade_date":
            download_macro_trade_date(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "date_year":
            download_macro_date_year(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "static_full":
            download_macro_static_full(client, raw_dir, spec, spec_page_limit(spec, args.page_limit), revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "date_year_by_ts_code":
            if dataset == "index_global":
                codes = selected_index_codes(args)
            elif dataset in ("index_daily", "index_dailybasic"):
                codes = selected_cn_index_codes(args)
            elif dataset == "fx_daily":
                codes = selected_fx_codes(args)
            else:
                raise RuntimeError(f"no ts_code universe defined for by-ts_code macro dataset {dataset!r}")
            download_macro_date_year_by_ts_code(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), codes, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "eco_cal_month":
            download_macro_eco_cal_month(client, raw_dir, spec, start_date, args.end_date, args.force, spec_page_limit(spec, args.page_limit), args, revision_ledger, allow_empty_revision_overwrite)
        else:
            raise RuntimeError(f"unsupported macro strategy {spec.strategy} for {dataset}")
    print(f"{args.tier} download finished under {raw_dir}")
    return 0

def download_macro_quarter_once(client: TuShareClient, raw_dir: Path, spec: MacroDataset, start_date: str, end_date: str, force: bool, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    start_q = max(yyyymmdd_to_quarter(start_date), spec.start_quarter)
    end_q = yyyymmdd_to_quarter(end_date)
    path = raw_dir / spec.api_name / f"range={start_q}_latest.parquet"
    coverage_params = {"start_date": start_date, "end_date": end_date}
    if should_skip_existing_partition(path, force=force, requested_params=coverage_params):
        return
    params = {"start_q": start_q, "end_q": end_q}
    result = client.query(spec.api_name, params, spec.fields)
    rows = write_macro_result(path, result, spec, {**params, **coverage_params}, revision_ledger, allow_empty_revision_overwrite)
    prune_stale_range_files(path)
    print(f"{spec.api_name} quarters {start_q}-{end_q} rows={rows}")

def download_macro_month_once(client: TuShareClient, raw_dir: Path, spec: MacroDataset, start_date: str, end_date: str, force: bool, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    start_m = max(yyyymmdd_to_month(start_date), spec.start_month)
    end_m = yyyymmdd_to_month(end_date)
    path = raw_dir / spec.api_name / f"range={start_m}_latest.parquet"
    coverage_params = {"start_date": start_date, "end_date": end_date}
    if should_skip_existing_partition(path, force=force, requested_params=coverage_params):
        return
    params = {"start_m": start_m, "end_m": end_m}
    result = client.query(spec.api_name, params, spec.fields)
    rows = write_macro_result(path, result, spec, {**params, **coverage_params}, revision_ledger, allow_empty_revision_overwrite)
    prune_stale_range_files(path)
    print(f"{spec.api_name} months {start_m}-{end_m} rows={rows}")

def prune_stale_range_files(canonical: Path) -> None:
    """A range dataset holds exactly ONE file (the canonical `range=*_latest`).
    Older end-suffixed range files from before canonical naming duplicated the
    whole series 2-3x in snapshot domain unions — remove them and their meta."""
    if not canonical.exists():
        return
    for stale in sorted(canonical.parent.glob("range=*.parquet")):
        if stale == canonical:
            continue
        meta = stale.with_suffix(stale.suffix + ".meta.json")
        stale.unlink()
        meta.unlink(missing_ok=True)
        print(f"pruned stale range partition {stale}")

def download_macro_static_full(
    client: TuShareClient,
    raw_dir: Path,
    spec: MacroDataset,
    page_limit: int,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    """Full registry refresh: small reference tables (contract/bond registries,
    announcement ledgers) are re-pulled whole on every run and written through
    the revision-aware overwrite (key-removal accepted as source correction;
    the disproportionate-shrink guard still blocks transient truncation)."""
    loops = spec.loop_values or ("",)
    for value in loops:
        params = {spec.loop_param: value} if value else {}
        name = f"{spec.loop_param}={safe_partition_value(value)}.parquet" if value else "full.parquet"
        path = raw_dir / spec.api_name / name
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = augment_macro_frame(frame(result), spec)
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            allow_key_removal_overwrite=True,
        )
        print(f"{spec.api_name} {name} rows={len(df)} written={bool(did_write)}")


def download_macro_month_loop(client: TuShareClient, raw_dir: Path, spec: MacroDataset, start_date: str, end_date: str, force: bool, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    windows = month_windows(start_date, end_date)
    written = 0
    skipped = 0
    total_rows = 0
    for index, (window_start, window_end, month) in enumerate(windows, start=1):
        path = raw_dir / spec.api_name / f"month={month}.parquet"
        # Coverage params for EVERY month, not only the newest: skipping a
        # middle month on bare file existence would freeze a partition whose
        # recorded coverage stops mid-month once the window rolls past it.
        coverage_params = {"start_date": window_start, "end_date": window_end}
        if should_skip_existing_partition(path, force=force, requested_params=coverage_params):
            skipped += 1
            continue
        params = {spec.month_param: month}
        meta_params = {**params, **coverage_params}
        result = client.query(spec.api_name, params, spec.fields)
        df = augment_macro_frame(frame(result), spec)
        total_rows += write_window_merged_partition(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=window_start,
            end_date=window_end,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        )
        written += 1
        if index % 24 == 0:
            print(f"{spec.api_name} months {index}/{len(windows)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done months={len(windows)} skipped={skipped} written={written} rows_written={total_rows}")

def download_macro_trade_date(
    client: TuShareClient,
    raw_dir: Path,
    spec: MacroDataset,
    start_date: str,
    end_date: str,
    force: bool,
    page_limit: int,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    """Per-trade-date pulls for daily macro tables whose range endpoints cap the
    response server-side and ignore offset paging (ths/sw/ci index dailies:
    year-range pulls silently truncated at 3000/4000 rows)."""
    trade_dates = [d for d in load_sse_open_dates(raw_dir, start_date, end_date)]
    dataset_dir = raw_dir / spec.api_name
    written = 0
    skipped = 0
    total_rows = 0
    # Optional scope loop (e.g. opt_daily per financial exchange): one file per
    # (loop value, trade date) so partition pruning by date keeps working; each
    # loop value honours its own venue listing date.
    loops = spec.loop_values or ("",)
    tasks = [
        (value, trade_date)
        for value in loops
        for trade_date in trade_dates
        if not value or trade_date >= spec.loop_start_date(value)
    ]
    for index, (loop_value, trade_date) in enumerate(tasks, start=1):
        directory = dataset_dir / f"{spec.loop_param}={safe_partition_value(loop_value)}" if loop_value else dataset_dir
        path = directory / f"trade_date={trade_date}.parquet"
        if path.exists() and not force and parquet_rows(path) > 0:
            skipped += 1
            continue
        params = {"trade_date": trade_date}
        if loop_value:
            params[spec.loop_param] = loop_value
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = augment_macro_frame(frame(result), spec)
        if df.empty:
            print(f"{spec.api_name} trade_date={trade_date} returned zero rows; skipped_write")
            continue
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            allow_key_removal_overwrite=True,
        )
        if did_write:
            written += 1
            total_rows += len(df)
        if index % 250 == 0:
            print(f"{spec.api_name} {index}/{len(tasks)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done tasks={len(tasks)} skipped={skipped} written={written} rows_written={total_rows}")


def download_macro_date_year(client: TuShareClient, raw_dir: Path, spec: MacroDataset, start_date: str, end_date: str, force: bool, page_limit: int, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    written = 0
    skipped = 0
    total_rows = 0
    years = range(int(start_date[:4]), int(end_date[:4]) + 1)
    for year in years:
        year_start = max(start_date, f"{year}0101")
        year_end = min(end_date, f"{year}1231")
        path = raw_dir / spec.api_name / f"year={year}.parquet"
        params = {"start_date": year_start, "end_date": year_end}
        if should_skip_existing_partition(path, force=force, requested_params=params):
            skipped += 1
            continue
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        df = augment_macro_frame(frame(result), spec)
        total_rows += write_window_merged_partition(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=year_start,
            end_date=year_end,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        )
        written += 1
    print(f"{spec.api_name} done years={int(end_date[:4]) - int(start_date[:4]) + 1} skipped={skipped} written={written} rows_written={total_rows}")

def download_macro_date_year_by_ts_code(
    client: TuShareClient,
    raw_dir: Path,
    spec: MacroDataset,
    start_date: str,
    end_date: str,
    force: bool,
    page_limit: int,
    codes: list[str],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    total_tasks = len(codes) * (int(end_date[:4]) - int(start_date[:4]) + 1)
    written = 0
    skipped = 0
    total_rows = 0
    task_index = 0
    for ts_code in codes:
        for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
            task_index += 1
            year_start = max(start_date, f"{year}0101")
            year_end = min(end_date, f"{year}1231")
            path = raw_dir / spec.api_name / f"ts_code={safe_partition_value(ts_code)}" / f"year={year}.parquet"
            params = {"ts_code": ts_code, "start_date": year_start, "end_date": year_end}
            if should_skip_existing_partition(path, force=force, requested_params=params):
                skipped += 1
                continue
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
            meta_params = dict(params)
            meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
            df = augment_macro_frame(frame(result), spec)
            total_rows += write_window_merged_partition(
                path,
                df,
                api_name=spec.api_name,
                params=meta_params,
                fields=list(df.columns),
                key_columns=list(spec.key_columns),
                date_columns=[spec.date_column, spec.time_column, "available_at"],
                start_date=year_start,
                end_date=year_end,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            )
            written += 1
            if task_index % 25 == 0:
                print(f"{spec.api_name} {task_index}/{total_tasks} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done tasks={total_tasks} skipped={skipped} written={written} rows_written={total_rows}")

def download_macro_eco_cal_month(
    client: TuShareClient,
    raw_dir: Path,
    spec: MacroDataset,
    start_date: str,
    end_date: str,
    force: bool,
    page_limit: int,
    args: argparse.Namespace,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    countries = selected_eco_filter_values(args, "eco_country")
    currencies = selected_eco_filter_values(args, "eco_currency")
    events = selected_eco_filter_values(args, "eco_event")
    windows = month_windows(start_date, end_date)
    total_tasks = len(countries) * len(currencies) * len(events) * len(windows)
    written = 0
    skipped = 0
    total_rows = 0
    task_index = 0
    for country in countries:
        for currency in currencies:
            for event in events:
                country_part = safe_partition_value(country) if country else "all"
                currency_part = safe_partition_value(currency) if currency else "all"
                event_part = safe_partition_value(event) if event else "all"
                for start, end, month in windows:
                    task_index += 1
                    path = raw_dir / spec.api_name / f"country={country_part}" / f"currency={currency_part}" / f"event={event_part}" / f"month={month}.parquet"
                    params: dict[str, Any] = {"start_date": start, "end_date": end}
                    if country:
                        params["country"] = country
                    if currency:
                        params["currency"] = currency
                    if event:
                        params["event"] = event
                    if should_skip_existing_partition(path, force=force, requested_params=params):
                        skipped += 1
                        continue
                    # The API clamps every response to 100 rows, sorts with
                    # no stable tiebreak, and serves each event many times
                    # over (measured ~8x), so offset-paging a whole month is
                    # slow and repeats pages verbatim. Day windows bound each
                    # paging run, repeated pages are tolerated as expected
                    # source behavior, and the key dedup below collapses the
                    # duplicates.
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
                    meta_params = dict(params)
                    meta_params.update({
                        "country_partition": country_part,
                        "currency_partition": currency_part,
                        "event_partition": event_part,
                        "month": month,
                        "pagination": {"page_limit": page_limit, "pages": pages_total, "day_windows": day_windows},
                    })
                    df = augment_macro_frame(combined, spec)
                    total_rows += write_window_merged_partition(
                        path,
                        df,
                        api_name=spec.api_name,
                        params=meta_params,
                        fields=list(df.columns),
                        key_columns=list(spec.key_columns),
                        date_columns=[spec.date_column, spec.time_column, "available_at"],
                        start_date=start,
                        end_date=end,
                        revision_ledger=revision_ledger,
                        allow_empty_revision_overwrite=allow_empty_revision_overwrite,
                    )
                    written += 1
                    if task_index % 24 == 0:
                        print(f"{spec.api_name} {task_index}/{total_tasks} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done tasks={total_tasks} skipped={skipped} written={written} rows_written={total_rows}")

def download_event_flow(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    datasets = selected_event_flow_download_datasets(args)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    allow_empty_revision_overwrite = getattr(args, "allow_empty_revision_overwrite", False)
    trade_dates: list[str] = []
    new_share_trading_dates: list[str] | None = None
    trade_end_date = args.end_date
    trade_cal_refreshed = False
    has_trade_date_dataset = any(EVENT_FLOW_SPECS[name].strategy == "trade_date" for name in datasets)
    if has_trade_date_dataset or "new_share" in datasets:
        # new_share needs two future open dates to place the ballot field after
        # its real disclosure window. A 31-day calendar request comfortably
        # spans regular exchange holidays; the stamper still fails explicitly
        # if no second open date exists, so this never becomes an approximation.
        calendar_end_date = args.end_date
        if "new_share" in datasets:
            calendar_end_date = format_yyyymmdd(parse_yyyymmdd(args.end_date) + timedelta(days=31))
        trade_cal_refreshed = ensure_trade_cal_coverage(
            client, raw_dir, args.start_date, calendar_end_date
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
    if "new_share" in datasets:
        new_share_trading_dates = load_sse_open_dates(
            raw_dir,
            args.start_date,
            format_yyyymmdd(parse_yyyymmdd(args.end_date) + timedelta(days=31)),
            allow_empty=True,
        )
    required_zero_skipped: list[str] = []
    blocked_overwrites: list[str] = []
    # trade_cal coverage refresh IS a lake write: it must veto the exit-75
    # no-mutation contract even when every dataset response was empty.
    total_written = 1 if trade_cal_refreshed else 0
    for dataset in datasets:
        spec = EVENT_FLOW_SPECS[dataset]
        start_date = max(args.start_date, spec.start_date)
        if spec.strategy == "trade_date":
            dates = [d for d in trade_dates if start_date <= d <= trade_end_date]
            written, zero_skipped, blocked_skipped = download_event_trade_date_dataset(
                client,
                raw_dir,
                spec,
                dates,
                args.force,
                spec_page_limit(spec, args.page_limit),
                revision_ledger,
                allow_empty_revision_overwrite,
            )
            total_written += written
            if zero_skipped and not spec.zero_rows_ok:
                required_zero_skipped.append(f"{spec.api_name}:{zero_skipped}")
            if blocked_skipped and not spec.zero_rows_ok:
                blocked_overwrites.append(f"{spec.api_name}:{blocked_skipped}")
        elif spec.strategy == "range_month":
            windows = month_windows(args.start_date, args.end_date)
            dataset_windows = [(s, e, m) for s, e, m in windows if e >= start_date]
            total_written += download_event_range_month(
                client,
                raw_dir,
                spec,
                dataset_windows,
                args.force,
                spec_page_limit(spec, args.page_limit),
                revision_ledger,
                allow_empty_revision_overwrite,
                trading_dates=new_share_trading_dates if dataset == "new_share" else None,
            )
        else:
            raise RuntimeError(f"unsupported event/flow strategy {spec.strategy} for {dataset}")
    if blocked_overwrites:
        # Non-empty responses whose overwrite the revision guard refused
        # (destructive shrink): a data-integrity alarm, never "not ready".
        raise RuntimeError(
            "required event/flow partitions had non-empty responses whose overwrite was blocked; "
            f"review the revision ledger before retrying: {blocked_overwrites}"
        )
    if required_zero_skipped:
        if getattr(args, "zero_rows_not_ready", False):
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
    spec: EventDataset,
    trade_dates: list[str],
    force: bool,
    page_limit: int | None,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> tuple[int, int, int]:
    page_limit = page_limit or spec.page_limit
    written = 0
    skipped = 0
    zero_skipped = 0
    blocked_skipped = 0
    total_rows = 0
    total_pages = 0
    for index, trade_date in enumerate(trade_dates, start=1):
        path = raw_dir / spec.api_name / f"trade_date={trade_date}.parquet"
        if path.exists() and not force:
            # A parquet whose sidecar is missing or carries another write ID is
            # an interrupted write, not a committed partition: re-attempt it.
            complete = committed_partition_intact(path) and (spec.zero_rows_ok or parquet_rows(path) > 0)
            exchange_column = margin_exchange_column(spec.api_name)
            if complete and exchange_column is not None:
                # A committed partial day (missing exchanges) must be re-attempted
                # by any covering run, not treated as done.
                present = pd.read_parquet(path, columns=[exchange_column])
                complete = not margin_family_missing_exchanges(spec.api_name, trade_date, present)
            if complete:
                skipped += 1
                if index % 250 == 0:
                    print(f"{spec.api_name} {index}/{len(trade_dates)} skipped={skipped} written={written} rows_written={total_rows}")
                continue
        params = {"trade_date": trade_date}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        df = augment_event_frame(frame(result), spec)
        if df.empty and not spec.zero_rows_ok:
            zero_skipped += 1
            total_pages += pages
            print(f"{spec.api_name} trade_date={trade_date} returned zero rows; skipped_write")
            continue
        if not df.empty:
            missing = margin_family_missing_exchanges(spec.api_name, trade_date, df)
            if missing:
                # Exchanges publish independently; a partial response poisons
                # market-wide aggregates, so it rides the same not-ready/retry
                # contract as an empty one instead of being committed.
                zero_skipped += 1
                total_pages += pages
                print(f"{spec.api_name} trade_date={trade_date} missing exchanges {missing}; skipped_write")
                continue
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            allow_key_removal_overwrite=True,
        )
        if did_write:
            total_rows += len(df)
            written += 1
        else:
            # Non-empty response refused by the revision guard (destructive
            # shrink): distinct from an empty/not-published response.
            blocked_skipped += 1
        total_pages += pages
        if index % 250 == 0:
            print(f"{spec.api_name} {index}/{len(trade_dates)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done dates={len(trade_dates)} skipped={skipped} written={written} zero_skipped={zero_skipped} blocked_skipped={blocked_skipped} rows_written={total_rows} pages={total_pages}")
    return written, zero_skipped, blocked_skipped

def download_event_range_month(
    client: TuShareClient,
    raw_dir: Path,
    spec: EventDataset,
    windows: list[tuple[str, str, str]],
    force: bool,
    page_limit: int,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
    *,
    trading_dates: list[str] | None = None,
) -> int:
    written = 0
    skipped = 0
    total_rows = 0
    total_pages = 0
    for index, (start_date, end_date, month) in enumerate(windows, start=1):
        path = raw_dir / spec.api_name / f"month={month}.parquet"
        params = {"start_date": start_date, "end_date": end_date}
        if should_skip_existing_partition(path, force=force, requested_params=params):
            skipped += 1
            continue
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        meta_params = dict(params)
        meta_params["month"] = month
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        df = augment_event_frame(frame(result), spec, trading_dates=trading_dates)
        rows = write_window_merged_partition(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            date_columns=[spec.date_column, spec.fallback_date_column, "available_at"],
            start_date=start_date,
            end_date=end_date,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        )
        total_rows += rows
        written += 1 if rows or not (path.exists() and parquet_rows(path) > 0 and df.empty) else 0
        total_pages += pages
        if index % 24 == 0:
            print(f"{spec.api_name} months {index}/{len(windows)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done months={len(windows)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    return written

def download_board_trading(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    datasets = selected_board_trading_datasets(args)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    allow_empty_revision_overwrite = getattr(args, "allow_empty_revision_overwrite", False)
    trade_dates: list[str] = []
    if any(BOARD_TRADING_SPECS[name].strategy != "static_full" for name in datasets):
        ensure_trade_cal_coverage(client, raw_dir, args.start_date, args.end_date)
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
        start_date = max(args.start_date, spec.start_date)
        dates = [trade_date for trade_date in trade_dates if start_date <= trade_date <= args.end_date]
        page_limit = spec_page_limit(spec, args.page_limit)
        if spec.strategy == "static_full":
            download_board_static_dataset(client, raw_dir, spec, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "trade_date":
            download_board_trade_date_dataset(client, raw_dir, spec, dates, args.force, page_limit, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "trade_date_by_tag":
            download_board_kpl_list(client, raw_dir, spec, dates, args.force, page_limit, selected_board_kpl_tags(args), revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "trade_date_by_limit_type":
            download_board_limit_list_ths(client, raw_dir, spec, dates, args.force, page_limit, selected_board_ths_limit_types(args), revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "trade_date_by_market":
            download_board_hot_by_market(client, raw_dir, spec, dates, args.force, page_limit, selected_board_ths_hot_markets(args), selected_board_hot_is_new(args), revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "trade_date_by_market_hot_type":
            download_board_dc_hot(client, raw_dir, spec, dates, args.force, page_limit, selected_board_dc_hot_markets(args), selected_board_dc_hot_types(args), selected_board_hot_is_new(args), revision_ledger, allow_empty_revision_overwrite)
        else:
            raise RuntimeError(f"unsupported board-trading strategy {spec.strategy} for {dataset}")
    print(f"board-trading download finished under {raw_dir}")
    return 0

def download_board_static_dataset(client: TuShareClient, raw_dir: Path, spec: BoardTradingDataset, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    """Full-table refresh on every run; the revision-aware write refuses to
    replace a non-empty partition with an empty vendor response."""
    path = raw_dir / spec.api_name / f"{spec.api_name}.parquet"
    result = client.query(spec.api_name, {}, spec.fields)
    rows = write_board_result(path, result, spec, {}, revision_ledger, allow_empty_revision_overwrite)
    print(f"{spec.api_name} static rows={rows}")

def download_board_trade_date_dataset(client: TuShareClient, raw_dir: Path, spec: BoardTradingDataset, trade_dates: list[str], force: bool, page_limit: int, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    written = 0
    skipped = 0
    total_rows = 0
    total_pages = 0
    for index, trade_date in enumerate(trade_dates, start=1):
        path = raw_dir / spec.api_name / f"trade_date={trade_date}.parquet"
        if should_skip_existing_partition(path, force=force):
            skipped += 1
            continue
        params = {"trade_date": trade_date}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        total_rows += write_board_result(path, result, spec, meta_params, revision_ledger, allow_empty_revision_overwrite)
        total_pages += pages
        written += 1
        if index % 250 == 0:
            print(f"{spec.api_name} {index}/{len(trade_dates)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done dates={len(trade_dates)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")

def download_board_kpl_list(client: TuShareClient, raw_dir: Path, spec: BoardTradingDataset, trade_dates: list[str], force: bool, page_limit: int, tags: list[str], revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    total_tasks = len(trade_dates) * len(tags)
    task_index = 0
    written = 0
    skipped = 0
    total_rows = 0
    total_pages = 0
    for tag in tags:
        tag_part = safe_partition_value(tag)
        for trade_date in trade_dates:
            task_index += 1
            path = raw_dir / spec.api_name / f"tag={tag_part}" / f"trade_date={trade_date}.parquet"
            if should_skip_existing_partition(path, force=force):
                skipped += 1
                continue
            params = {"trade_date": trade_date, "tag": tag}
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
            meta_params = dict(params)
            meta_params["tag_partition"] = tag_part
            meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
            total_rows += write_board_result(path, result, spec, meta_params, revision_ledger, allow_empty_revision_overwrite)
            total_pages += pages
            written += 1
            if task_index % 250 == 0:
                print(f"{spec.api_name} {task_index}/{total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done tasks={total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")

def download_board_limit_list_ths(client: TuShareClient, raw_dir: Path, spec: BoardTradingDataset, trade_dates: list[str], force: bool, page_limit: int, limit_types: list[str], revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    total_tasks = len(trade_dates) * len(limit_types)
    task_index = 0
    written = 0
    skipped = 0
    total_rows = 0
    total_pages = 0
    for limit_type in limit_types:
        limit_type_part = safe_partition_value(limit_type)
        for trade_date in trade_dates:
            task_index += 1
            path = raw_dir / spec.api_name / f"limit_type={limit_type_part}" / f"trade_date={trade_date}.parquet"
            if should_skip_existing_partition(path, force=force):
                skipped += 1
                continue
            params = {"trade_date": trade_date, "limit_type": limit_type}
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
            meta_params = dict(params)
            meta_params["limit_type_partition"] = limit_type_part
            meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
            total_rows += write_board_result(path, result, spec, meta_params, revision_ledger, allow_empty_revision_overwrite)
            total_pages += pages
            written += 1
            if task_index % 250 == 0:
                print(f"{spec.api_name} {task_index}/{total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done tasks={total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")

def download_board_hot_by_market(
    client: TuShareClient,
    raw_dir: Path,
    spec: BoardTradingDataset,
    trade_dates: list[str],
    force: bool,
    page_limit: int,
    markets: list[str],
    is_new_values: list[str],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    total_tasks = len(trade_dates) * len(markets) * len(is_new_values)
    task_index = 0
    written = 0
    skipped = 0
    total_rows = 0
    total_pages = 0
    for market in markets:
        market_part = safe_partition_value(market)
        for is_new in is_new_values:
            for trade_date in trade_dates:
                task_index += 1
                path = raw_dir / spec.api_name / f"market={market_part}" / f"is_new={is_new}" / f"trade_date={trade_date}.parquet"
                if should_skip_existing_partition(path, force=force):
                    skipped += 1
                    continue
                params = {"trade_date": trade_date, "market": market, "is_new": is_new}
                result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
                meta_params = dict(params)
                meta_params["market_partition"] = market_part
                meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
                total_rows += write_board_result(path, result, spec, meta_params, revision_ledger, allow_empty_revision_overwrite)
                total_pages += pages
                written += 1
                if task_index % 250 == 0:
                    print(f"{spec.api_name} {task_index}/{total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done tasks={total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")

def download_board_dc_hot(
    client: TuShareClient,
    raw_dir: Path,
    spec: BoardTradingDataset,
    trade_dates: list[str],
    force: bool,
    page_limit: int,
    markets: list[str],
    hot_types: list[str],
    is_new_values: list[str],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    total_tasks = len(trade_dates) * len(markets) * len(hot_types) * len(is_new_values)
    task_index = 0
    written = 0
    skipped = 0
    total_rows = 0
    total_pages = 0
    for market in markets:
        market_part = safe_partition_value(market)
        for hot_type in hot_types:
            hot_type_part = safe_partition_value(hot_type)
            for is_new in is_new_values:
                for trade_date in trade_dates:
                    task_index += 1
                    path = raw_dir / spec.api_name / f"market={market_part}" / f"hot_type={hot_type_part}" / f"is_new={is_new}" / f"trade_date={trade_date}.parquet"
                    if should_skip_existing_partition(path, force=force):
                        skipped += 1
                        continue
                    params = {"trade_date": trade_date, "market": market, "hot_type": hot_type, "is_new": is_new}
                    result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
                    meta_params = dict(params)
                    meta_params["market_partition"] = market_part
                    meta_params["hot_type_partition"] = hot_type_part
                    meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
                    total_rows += write_board_result(path, result, spec, meta_params, revision_ledger, allow_empty_revision_overwrite)
                    total_pages += pages
                    written += 1
                    if task_index % 250 == 0:
                        print(f"{spec.api_name} {task_index}/{total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{spec.api_name} done tasks={total_tasks} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")

def query_share_float_to_path(
    client: TuShareClient,
    path: Path,
    params: dict[str, Any],
    source: str,
    force: bool,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> dict[str, Any]:
    if not force and committed_partition_intact(path):
        rows = parquet_rows(path)
        # Every return of this function carries the same keys: the caller reads
        # guard_retained on each day, so omitting it here turned a plain resume
        # (partition present, no --force) into a KeyError.
        return {
            "path": str(path),
            "rows": rows,
            "skipped": True,
            "source_cap_risk": rows >= SHARE_FLOAT_ROW_LIMIT,
            "guard_retained": False,
        }
    result = client.query("share_float", params, SHARE_FLOAT_FIELDS)
    meta_params = dict(params)
    meta_params["download_path"] = source
    meta_params["row_limit"] = SHARE_FLOAT_ROW_LIMIT
    df = augment_event_frame(frame(result), EVENT_FLOW_SPECS["share_float"])
    did_write = write_parquet_revision_aware(
        path,
        df,
        api_name="share_float",
        params=meta_params,
        fields=list(df.columns),
        key_columns=list(EVENT_FLOW_SPECS["share_float"].key_columns),
        revision_ledger=revision_ledger,
        allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        allow_key_removal_overwrite=True,
    )
    rows = len(df) if did_write else parquet_rows(path) if path.exists() else 0
    return {
        "path": str(path),
        "rows": rows,
        "skipped": not did_write,
        # Cap risk must consider the FETCHED response, not only the rows kept
        # on disk: a guard-retained day whose new pull sits at the per-call cap
        # is the rotating-slice case the ts_code rescue exists for. Classifying
        # it by the retained old rows escalated exactly that case (a small
        # early-evening partition, then the vendor's late fill pushed the day
        # past the cap) into an "unrescuable retraction" run failure.
        "source_cap_risk": max(rows, len(df)) >= SHARE_FLOAT_ROW_LIMIT,
        # A guard decision is NOT a fetch failure: the writer deliberately kept
        # the old partition and recorded a revision event. Raising here aborted
        # the whole run mid-loop, which made the ts_code rescue below
        # unreachable -- so a capped partition (the exact case the rescue
        # exists for) could never be repaired and every attempt ended with the
        # lake left dirty. The caller decides what a retention means: capped ->
        # rescue by ts_code, otherwise a mass retraction an operator must rule on.
        "guard_retained": not did_write and not df.empty,
    }

def download_share_float_ann_dates(client: TuShareClient, raw_dir: Path, args: argparse.Namespace, report: dict[str, Any]) -> list[str]:
    days = date_range_days(args.ann_start_date, args.ann_end_date)
    limit_hits: list[str] = []
    unrescuable: list[str] = []
    written = 0
    skipped = 0
    total_rows = 0
    for index, day in enumerate(days, start=1):
        path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        result = query_share_float_to_path(
            client,
            path,
            {"ann_date": day},
            "ann_date",
            args.force,
            getattr(args, "revision_ledger", REVISION_EVENTS_PATH),
            getattr(args, "allow_empty_revision_overwrite", False),
        )
        written += 0 if result["skipped"] else 1
        skipped += 1 if result["skipped"] else 0
        total_rows += int(result["rows"])
        if result["source_cap_risk"]:
            limit_hits.append(day)
        elif result["guard_retained"]:
            # Retained but NOT capped: the source really did drop a large share
            # of the keys, which no rescue can repair because the ts_code path
            # would only confirm the same retraction. This needs an operator
            # ruling, so it must not pass silently -- it is raised after every
            # recoverable day has been processed.
            unrescuable.append(day)
        if index % 250 == 0:
            print(f"share_float ann_date {index}/{len(days)} skipped={skipped} written={written} rows_seen={total_rows} limit_hits={len(limit_hits)}")
    report["ann_date"] = {
        "start_date": args.ann_start_date,
        "end_date": args.ann_end_date,
        "days": len(days),
        "written": written,
        "skipped": skipped,
        "rows_seen": total_rows,
        "limit_hit_days": limit_hits,
        "guard_retained_days": unrescuable,
    }
    print(f"share_float ann_date done days={len(days)} skipped={skipped} written={written} rows_seen={total_rows} limit_hits={len(limit_hits)}")
    if unrescuable:
        raise RuntimeError(
            "share_float refresh was retained by the revision guard on non-capped partitions "
            f"{unrescuable}: the source retracted a disproportionate share of keys and no ts_code "
            "rescue can recover it. Inspect the revision ledger, then delete the partition to "
            "accept the retraction or keep the old data deliberately."
        )
    return limit_hits

def download_share_float_ts_code_rescue_by_date(
    client: TuShareClient,
    raw_dir: Path,
    date_codes: dict[str, list[str]],
    *,
    date_param: str,
    dataset_dir: str,
    source: str,
    force: bool,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> dict[str, Any]:
    written = 0
    skipped = 0
    total_rows = 0
    limit_hits: list[dict[str, Any]] = []
    no_candidate_dates = sorted(date for date, codes in date_codes.items() if not codes)
    total_tasks = sum(len(codes) for codes in date_codes.values())
    task_index = 0
    for day in sorted(date_codes):
        for code in date_codes[day]:
            task_index += 1
            path = raw_dir / dataset_dir / f"{date_param}={day}" / f"ts_code={code}.parquet"
            result = query_share_float_to_path(
                client,
                path,
                {date_param: day, "ts_code": code},
                source,
                force,
                revision_ledger,
                allow_empty_revision_overwrite,
            )
            written += 0 if result["skipped"] else 1
            skipped += 1 if result["skipped"] else 0
            total_rows += int(result["rows"])
            if result["source_cap_risk"]:
                limit_hits.append({"date": day, "ts_code": code, "rows": int(result["rows"])})
            if task_index % 500 == 0:
                print(
                    f"share_float {source} {task_index}/{total_tasks} skipped={skipped} "
                    f"written={written} rows_seen={total_rows} limit_hits={len(limit_hits)}"
                )
    return {
        "date_param": date_param,
        "dates": sorted(date_codes),
        "date_candidate_counts": {date: len(codes) for date, codes in sorted(date_codes.items())},
        "no_candidate_dates": no_candidate_dates,
        "tasks": total_tasks,
        "written": written,
        "skipped": skipped,
        "rows_seen": total_rows,
        "limit_hits": limit_hits,
    }

def unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result

def explicit_rescue_stock_codes(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    requested.extend(getattr(args, "rescue_code", []) or [])
    codes_file = getattr(args, "rescue_codes_file", None)
    if codes_file:
        text = Path(codes_file).read_text(encoding="utf-8")
        requested.extend(re.split(r"[\s,]+", text))
    return unique_ordered(requested)

def selected_share_float_rescue_dates(args: argparse.Namespace, ann_limit_hits: list[str]) -> tuple[list[str], list[str]]:
    explicit = unique_ordered(getattr(args, "rescue_ann_date", []) or [])
    auto: list[str] = []
    skipped_auto: list[str] = []
    if args.rescue_ann_limit_hits:
        auto = ann_limit_hits[: args.max_ann_rescue_days]
        skipped_auto = ann_limit_hits[args.max_ann_rescue_days :]
    elif ann_limit_hits:
        skipped_auto = ann_limit_hits
    return unique_ordered(explicit + auto), skipped_auto

def enforce_share_float_rescue_date_code_budget(args: argparse.Namespace, ann_date_codes: dict[str, list[str]], float_date_codes: dict[str, list[str]]) -> int:
    estimated_calls = sum(len(codes) for codes in ann_date_codes.values()) + sum(len(codes) for codes in float_date_codes.values())
    if args.max_rescue_calls is not None and estimated_calls > args.max_rescue_calls:
        raise RuntimeError(
            f"share_float rescue would make {estimated_calls} calls "
            f"({len(ann_date_codes)} ann_date dates and {len(float_date_codes)} float_date dates), "
            f"exceeding --max-rescue-calls={args.max_rescue_calls}"
        )
    return estimated_calls

def read_ts_codes_from_parquet(path: Path, *, extra_filter: tuple[str, str] | None = None) -> list[str]:
    if not path.exists() or parquet_rows(path) == 0:
        return []
    columns = ["ts_code"]
    if extra_filter:
        columns.append(extra_filter[0])
    df = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    if extra_filter and extra_filter[0] in df.columns:
        df = df[df[extra_filter[0]].astype(str).str.strip() == extra_filter[1]]
    if "ts_code" not in df.columns:
        return []
    return unique_ordered(df["ts_code"].dropna().astype(str).str.strip().tolist())

def share_float_self_candidate_codes(raw_dir: Path, date_param: str, dates: list[str]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {date: [] for date in dates}
    for day in dates:
        if date_param == "ann_date":
            path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
            candidates[day].extend(read_ts_codes_from_parquet(path))
        elif date_param == "float_date":
            # No self source: the legacy whole-day share_float/ dir was
            # archived out of the lake in 2026-05 and has no writer; float_date
            # rescue candidates come from the ann-path cross lookup instead.
            pass
        else:
            raise RuntimeError(f"unsupported share_float candidate date_param {date_param}")
        candidates[day] = unique_ordered(candidates[day])
    return candidates

def anns_unlock_candidate_codes(raw_dir: Path, ann_dates: list[str]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {date: [] for date in ann_dates}
    dates_by_month: dict[str, set[str]] = {}
    for day in ann_dates:
        dates_by_month.setdefault(day[:6], set()).add(day)
    for month, days in sorted(dates_by_month.items()):
        path = raw_dir / "anns_d" / f"month={month}.parquet"
        if not path.exists() or parquet_rows(path) == 0:
            continue
        columns = ["ann_date", "ts_code", "title"]
        df = pd.read_parquet(path, columns=columns)
        df["ann_date"] = df["ann_date"].astype(str).str.strip()
        mask = df["ann_date"].isin(days) & df["title"].fillna("").astype(str).str.contains(SHARE_FLOAT_UNLOCK_TITLE_PATTERN)
        for day, group in df.loc[mask].groupby("ann_date"):
            candidates.setdefault(str(day), []).extend(group["ts_code"].dropna().astype(str).str.strip().tolist())
    return {date: unique_ordered(codes) for date, codes in candidates.items()}

def share_float_ann_path_float_candidate_codes(raw_dir: Path, float_dates: list[str], ann_start_date: str, ann_end_date: str) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {date: [] for date in float_dates}
    target_dates = set(float_dates)
    for day in date_range_days(ann_start_date, ann_end_date):
        path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        if not path.exists() or parquet_rows(path) == 0:
            continue
        df = pd.read_parquet(path, columns=["float_date", "ts_code"])
        df["float_date"] = df["float_date"].astype(str).str.strip()
        mask = df["float_date"].isin(target_dates)
        for float_date, group in df.loc[mask].groupby("float_date"):
            candidates.setdefault(str(float_date), []).extend(group["ts_code"].dropna().astype(str).str.strip().tolist())
    return {date: unique_ordered(codes) for date, codes in candidates.items()}

def merge_candidate_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for mapping in maps:
        for date_key, codes in mapping.items():
            merged.setdefault(date_key, []).extend(codes)
    return {date_key: unique_ordered(codes) for date_key, codes in merged.items()}

def select_share_float_rescue_date_codes(
    raw_dir: Path,
    args: argparse.Namespace,
    *,
    date_param: str,
    dates: list[str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    explicit = explicit_rescue_stock_codes(args)
    all_a_codes: list[str] = []
    detail: dict[str, Any] = {"date_param": date_param, "mode": args.rescue_universe, "dates": dates}
    if args.rescue_universe == "all_a":
        all_a_codes = load_stock_codes(raw_dir)
        if args.max_codes:
            all_a_codes = all_a_codes[: args.max_codes]
        return {date: all_a_codes for date in dates}, {**detail, "all_a_codes": len(all_a_codes)}
    if args.rescue_universe == "explicit":
        if not explicit:
            raise RuntimeError("--rescue-universe explicit requires --rescue-code or --rescue-codes-file")
        codes = explicit[: args.max_codes] if args.max_codes else explicit
        return {date: codes for date in dates}, {**detail, "explicit_codes": len(codes)}

    self_candidates = share_float_self_candidate_codes(raw_dir, date_param, dates)
    explicit_map = {date: explicit for date in dates}
    if date_param == "ann_date":
        anns_candidates = anns_unlock_candidate_codes(raw_dir, dates) if not args.no_anns_candidates else {date: [] for date in dates}
        # The float-dir -> ann cross lookup died with the archived whole-day
        # share_float/ dir; ann-path candidates come from self + anns sources.
        cross_candidates = {date: [] for date in dates}
    else:
        anns_candidates = {date: [] for date in dates}
        cross_candidates = (
            share_float_ann_path_float_candidate_codes(raw_dir, dates, args.ann_start_date, args.ann_end_date)
            if not args.no_cross_path_candidates
            else {date: [] for date in dates}
        )
    candidates = merge_candidate_maps(self_candidates, anns_candidates, cross_candidates, explicit_map)
    if args.max_codes:
        candidates = {date: codes[: args.max_codes] for date, codes in candidates.items()}
    detail.update({
        "self_candidate_counts": {date: len(self_candidates.get(date, [])) for date in dates},
        "anns_candidate_counts": {date: len(anns_candidates.get(date, [])) for date in dates},
        "cross_path_candidate_counts": {date: len(cross_candidates.get(date, [])) for date in dates},
        "explicit_codes": len(explicit),
        "final_candidate_counts": {date: len(candidates.get(date, [])) for date in dates},
        "max_codes": args.max_codes,
    })
    return candidates, detail

def share_float_union_files(raw_dir: Path, args: argparse.Namespace) -> list[tuple[Path, str]]:
    ann_start_date, ann_end_date = args.ann_start_date, args.ann_end_date
    float_start_date, float_end_date = args.float_start_date, args.float_end_date
    files: list[tuple[Path, str]] = []
    for day in date_range_days(ann_start_date, ann_end_date):
        path = raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        if path.exists():
            files.append((path, "ann_date"))
        rescue_dir = raw_dir / "share_float_ann_date_ts_code" / f"ann_date={day}"
        files.extend((path, "ann_date_ts_code") for path in sorted(rescue_dir.glob("ts_code=*.parquet")))
    if not args.skip_float_date_union:
        # Only the rescue partitions are readable on the float_date path: the
        # legacy whole-day dirs (share_float/, share_float_float_date/) were
        # archived out of the lake in 2026-05 and no writer can recreate them
        # (dev-phase removal 2026-08-02).
        for day in date_range_days(float_start_date, float_end_date):
            rescue_dir = raw_dir / "share_float_float_date_ts_code" / f"float_date={day}"
            files.extend((path, "float_date_ts_code") for path in sorted(rescue_dir.glob("ts_code=*.parquet")))
    return files

def _drop_stale_active_restatements(active: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only the newest announcement of each physical event in ``active``.

    Only a NON-capped source may win the comparison: a capped (truncated)
    partition proves nothing about rows it could not return, so identities
    whose newest dating appears only in a capped file are left untouched and
    fall back to the conservative union. Distinct lots inside one winning
    ``(identity, ann_date)`` group are preserved.

    Datings are normalized first: an undated row (the float_date query path
    returns rows whose ``ann_date`` the provider left empty) ranks BELOW every
    real date, so it can never win and delete a dated announcement.
    """
    if active.empty:
        return active, 0
    keys = active[PHYSICAL_IDENTITY_COLUMNS].copy()
    keys["holder_name"] = normalized_holder_keys(keys["holder_name"])
    keys["ann"] = normalized_date_keys(active["ann_date"]).to_numpy()
    # "" marks a row whose source was capped, so it cannot win the comparison;
    # an identity with only capped sources gets newest == "" and, because no
    # date is ever "< ''", is never pruned (conservative union).
    keys["trusted_ann"] = keys["ann"].where(~active["source_cap_risk"].to_numpy(), "")
    newest = keys.groupby(PHYSICAL_IDENTITY_COLUMNS, dropna=False)["trusted_ann"].transform("max")
    stale = keys["ann"] < newest
    if not bool(stale.any()):
        return active, 0
    return active.loc[~stale.to_numpy()].reset_index(drop=True), int(stale.sum())


def write_share_float_union(raw_dir: Path, args: argparse.Namespace, report: dict[str, Any]) -> None:
    repo_root = Path.cwd().resolve()
    output = (repo_root / args.union_output).resolve()
    revision_ledger = resolve_revision_ledger(
        raw_dir,
        getattr(args, "revision_ledger", REVISION_EVENTS_PATH),
        repo_root=repo_root,
    )
    if not output.exists():
        raise RuntimeError(f"share_float_complete baseline is missing at {output}; restore it before updating")
    files = share_float_union_files(raw_dir, args)
    existing_rows = parquet_rows(output)
    active_frames: list[pd.DataFrame] = []
    for path, source in files:
        rows = parquet_rows(path)
        source_file = os.path.relpath(str(path), str(Path.cwd()))
        if rows == 0:
            continue
        df = pd.read_parquet(path)
        df["download_path"] = source
        # Repo-relative provenance: an absolute host path would flow into the
        # Agent-visible events snapshot (isolation contract, env docs §2.2).
        df["source_file"] = source_file
        df["source_cap_risk"] = rows >= SHARE_FLOAT_ROW_LIMIT
        active_frames.append(df)
    group_columns = ["ts_code", "ann_date", "float_date", "holder_name", "share_type"]
    identity_columns = SHARE_FLOAT_FIELDS.split(",")
    active = concat_rows(active_frames) if active_frames else pd.DataFrame(columns=identity_columns)
    missing_identity_columns = [column for column in identity_columns if column not in active.columns]
    if not active.empty and missing_identity_columns:
        raise RuntimeError(
            "share_float_complete active rows are missing identity columns: "
            + ", ".join(missing_identity_columns)
        )
    if not active.empty:
        active = active.sort_values("source_cap_risk", kind="stable")
        active = active.drop_duplicates(identity_columns).reset_index(drop=True)
    # A restatement's refresh window usually covers BOTH the stale and the
    # corrected ann_date partition, so the stale rows arrive in `active` too;
    # they must be dropped here or they would survive the union and make the
    # stale baseline group look "refreshed" (measured live: 301583.SZ stayed
    # at 72 rows across 3 ann_dates with baseline-only supersession).
    active, active_restatements_dropped = _drop_stale_active_restatements(active)
    safe_active_groups = (
        active.loc[~active["source_cap_risk"], group_columns].drop_duplicates()
        if not active.empty else pd.DataFrame(columns=group_columns)
    )
    # The canonical union is the durable historical baseline. A refreshed
    # coarse event group replaces the same baseline group; distinct source rows
    # inside that group remain separate. Unchanged history is inherited without
    # reopening archived process files.
    revision_old = None
    revision_new = None
    capped_groups_preserved = 0
    frames = [active] if not active.empty else []
    existing = pd.read_parquet(output)
    missing_baseline_columns = [column for column in identity_columns if column not in existing.columns]
    if missing_baseline_columns:
        raise RuntimeError(
            "share_float_complete baseline is missing identity columns: "
            + ", ".join(missing_baseline_columns)
        )
    existing_key_count = len(existing.drop_duplicates(group_columns))
    baseline = existing
    superseded_group_count = 0
    if not active.empty:
        active_groups = active[group_columns].drop_duplicates()
        refreshed_groups = pd.MultiIndex.from_frame(active_groups)
        safe_refreshed_groups = pd.MultiIndex.from_frame(safe_active_groups)
        baseline_groups = pd.MultiIndex.from_frame(baseline[group_columns])
        refreshed = baseline_groups.isin(refreshed_groups)
        safe_refreshed = baseline_groups.isin(safe_refreshed_groups)
        capped_groups_preserved = len(active_groups) - len(safe_active_groups)
        # Provider re-dating: the same physical unlock event -- identified by
        # (ts_code, float_date, holder_name, share_type), WITHOUT ann_date --
        # can reappear under a corrected announcement date (observed 2026-07:
        # 301583.SZ moved from ann_date 20260630 to 20260709). The stale-dated
        # baseline group is superseded so the unlock is never double-counted;
        # its rows are preserved in the revision event, where source revisions
        # belong.
        baseline_identity = physical_identity_index(baseline)
        # Only NON-capped identities may supersede baseline history, matching
        # the active-side rule: a truncated partition proves nothing about the
        # rows it could not return, so it must not be allowed to delete an
        # older announcement either. (Previously the active side refused capped
        # evidence while the baseline side accepted it -- inconsistent.)
        # An UNDATED active row carries no announcement date at all, so it may
        # only add: letting it supersede would delete the dated generation of
        # the same event and downgrade the baseline's PIT availability.
        dated_active_groups = safe_active_groups.loc[
            normalized_date_keys(safe_active_groups["ann_date"]) != ""
        ]
        active_identity = physical_identity_index(dated_active_groups)
        superseded = baseline_identity.isin(active_identity) & ~refreshed
        superseded_group_count = len(baseline.loc[superseded, group_columns].drop_duplicates())
        revision_old = baseline.loc[refreshed | superseded].reset_index(drop=True)
        preserved_capped = baseline.loc[refreshed & ~safe_refreshed].reset_index(drop=True)
        revision_new = concat_rows([active, preserved_capped]).drop_duplicates(identity_columns).reset_index(drop=True)
        baseline = baseline.loc[~(safe_refreshed | superseded)].reset_index(drop=True)
    else:
        revision_old = baseline.iloc[:0].copy()
        revision_new = active
    frames.append(baseline)
    union = concat_rows(frames) if frames else pd.DataFrame(columns=SHARE_FLOAT_FIELDS.split(","))
    before = len(union)
    undated_duplicates_dropped = 0
    if not union.empty:
        union = union.drop_duplicates(identity_columns).reset_index(drop=True)
        # The seven-field dedup above cannot collapse an undated row against
        # its dated twin (they differ in ann_date), so an undated rescue fetch
        # would re-add a lot the lake already carries and double the supply.
        # This keeps that class out by construction rather than relying on the
        # rescue path staying unused.
        undated_duplicates = undated_duplicate_mask(union)
        undated_duplicates_dropped = int(undated_duplicates.sum())
        if undated_duplicates_dropped:
            union = union.loc[~undated_duplicates].reset_index(drop=True)
    # Refreshed groups replace their complete baseline group, preserving
    # multiple distinct lots while removing superseded source revisions.
    new_key_count = len(union.drop_duplicates(group_columns)) if not union.empty else 0
    # Superseded re-dated groups are the only sanctioned shrink: each one is
    # replaced by the same physical event under its corrected ann_date.
    if new_key_count < existing_key_count - superseded_group_count:
        raise RuntimeError(
            "share_float_complete merge would shrink distinct event groups of "
            f"{output} from {existing_key_count} to {new_key_count} "
            f"(beyond {superseded_group_count} superseded re-dated groups) "
            f"using {len(files)} input files"
        )
    published = write_parquet_revision_aware(
        output,
        union,
        api_name="share_float_complete",
        params={
            "strategy": "canonical_incremental_merge",
            "source_api": "share_float",
            "ann_start_date": args.ann_start_date,
            "ann_end_date": args.ann_end_date,
            "float_start_date": args.float_start_date,
            "float_end_date": args.float_end_date,
            "input_files": len(files),
        },
        fields=list(union.columns),
        key_columns=identity_columns,
        revision_ledger=revision_ledger,
        source="share_float_union_merge",
        allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
        allow_key_removal_overwrite=True,
        revision_comparison_old_df=revision_old,
        revision_comparison_new_df=revision_new,
        existing_df=existing,
    )
    if not published:
        raise RuntimeError(f"share_float_complete merge was not published to {output}")
    report["union"] = {
        "output": str(output),
        "input_files": len(files),
        "rows_before_dedup": before,
        "rows_after_dedup": len(union),
        "previous_rows": existing_rows,
        "baseline_inherited": True,
        "capped_groups_preserved": capped_groups_preserved,
        "superseded_redated_groups": superseded_group_count,
        "active_stale_restatement_rows_dropped": active_restatements_dropped,
        "undated_duplicate_rows_dropped": undated_duplicates_dropped,
    }

def download_share_float_complete(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    args.revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "row_limit": SHARE_FLOAT_ROW_LIMIT,
        "scope": {
            "ann_start_date": args.ann_start_date,
            "ann_end_date": args.ann_end_date,
            "float_start_date": args.float_start_date,
            "float_end_date": args.float_end_date,
            "float_rescue_dates": args.float_rescue_date,
            "max_codes": args.max_codes,
        },
    }
    ann_limit_hits: list[str] = []
    if not args.skip_ann_date:
        ann_limit_hits = download_share_float_ann_dates(client, raw_dir, args, report)

    rescue_ann_dates, skipped_ann_rescue = selected_share_float_rescue_dates(args, ann_limit_hits)
    ann_date_codes: dict[str, list[str]] = {}
    float_date_codes: dict[str, list[str]] = {}
    if rescue_ann_dates or args.float_rescue_date:
        ann_detail: dict[str, Any] = {}
        float_detail: dict[str, Any] = {}
        if rescue_ann_dates:
            ann_date_codes, ann_detail = select_share_float_rescue_date_codes(raw_dir, args, date_param="ann_date", dates=rescue_ann_dates)
        if args.float_rescue_date:
            float_date_codes, float_detail = select_share_float_rescue_date_codes(raw_dir, args, date_param="float_date", dates=args.float_rescue_date)
        report["rescue_candidates"] = {"ann_date": ann_detail, "float_date": float_detail}
        report["rescue_estimated_calls"] = enforce_share_float_rescue_date_code_budget(args, ann_date_codes, float_date_codes)

    if rescue_ann_dates:
        report["ann_date_ts_code"] = download_share_float_ts_code_rescue_by_date(
            client,
            raw_dir,
            ann_date_codes,
            date_param="ann_date",
            dataset_dir="share_float_ann_date_ts_code",
            source="ann_date_ts_code",
            force=args.force,
            revision_ledger=getattr(args, "revision_ledger", REVISION_EVENTS_PATH),
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
        )
    report["ann_date_ts_code_skipped_limit_days"] = skipped_ann_rescue

    if args.float_rescue_date:
        report["float_date_ts_code"] = download_share_float_ts_code_rescue_by_date(
            client,
            raw_dir,
            float_date_codes,
            date_param="float_date",
            dataset_dir="share_float_float_date_ts_code",
            source="float_date_ts_code",
            force=args.force,
            revision_ledger=getattr(args, "revision_ledger", REVISION_EVENTS_PATH),
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
        )

    if args.write_union:
        write_share_float_union(raw_dir, args, report)

    if args.output:
        output = (repo_root / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"share_float complete download process_report={output}")
    else:
        union = report.get("union") or {}
        print(
            "share_float complete download finished; "
            f"ann_limit_hits={len(report.get('ann_date', {}).get('limit_hit_days', []))} "
            f"ann_rescue_tasks={report.get('ann_date_ts_code', {}).get('tasks', 0)} "
            f"union_rows={union.get('rows_after_dedup', 'not_written')}; "
            "pass --output to write a process report"
        )
    return 0

def download_fundamental(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    allow_empty_revision_overwrite = getattr(args, "allow_empty_revision_overwrite", False)
    datasets = selected_fundamental_datasets(args.datasets)
    stock_codes: list[str] = []
    if any(FUNDAMENTAL_SPECS[name].strategy == "ts_code" for name in datasets):
        explicit_codes = [str(code).strip() for code in (getattr(args, "codes", None) or []) if str(code).strip()]
        stock_codes = sorted(set(explicit_codes)) if explicit_codes else load_stock_codes(raw_dir)
        if args.max_codes:
            stock_codes = stock_codes[: args.max_codes]
    refresh_periods = set(recent_quarter_periods(args.end_date, getattr(args, "fundamental_refresh_period_count", 0)))
    refresh_months = set(recent_month_keys(args.end_date, getattr(args, "fundamental_refresh_ann_month_count", 0)))
    periods = sorted(set(quarter_periods(args.start_date, args.end_date)) | refresh_periods)
    windows = fundamental_ann_month_windows(args.start_date, args.end_date, refresh_months)

    unsupported = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy not in {"period", "ann_month", "ts_code"}]
    if unsupported:
        raise RuntimeError(f"unsupported fundamental strategy for datasets: {unsupported}")

    period_datasets = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy == "period"]
    ann_month_datasets = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy == "ann_month"]
    ts_code_datasets = [dataset for dataset in datasets if FUNDAMENTAL_SPECS[dataset].strategy == "ts_code"]

    for dataset in period_datasets:
        spec = FUNDAMENTAL_SPECS[dataset]
        download_fundamental_period_dataset(client, raw_dir, spec, periods, args.force, spec_page_limit(spec, args.page_limit), refresh_periods, revision_ledger, allow_empty_revision_overwrite)
    for dataset in ann_month_datasets:
        spec = FUNDAMENTAL_SPECS[dataset]
        download_fundamental_ann_month_dataset(client, raw_dir, spec, windows, args.force, spec_page_limit(spec, args.page_limit), refresh_months, revision_ledger, allow_empty_revision_overwrite)

    affected_days = int(getattr(args, "fundamental_refresh_event_days", 90) or 0)
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
    if "dividend" in ts_code_datasets and getattr(args, "fundamental_dividend_probe_days", 0):
        dividend_probe_codes = probe_recent_dividend_codes(client, args.end_date, args.fundamental_dividend_probe_days, args.page_limit)
    refresh_ts_code_datasets = set(getattr(args, "fundamental_refresh_ts_code_datasets", []) or [])
    for dataset in ts_code_datasets:
        spec = FUNDAMENTAL_SPECS[dataset]
        force_codes = set()
        if dataset in refresh_ts_code_datasets:
            force_codes = set(affected_codes)
            if dataset == "dividend":
                force_codes.update(dividend_probe_codes)
        download_fundamental_ts_code_dataset(client, raw_dir, spec, stock_codes, args.force, spec_page_limit(spec, args.page_limit), force_codes, revision_ledger, allow_empty_revision_overwrite)
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
            codes.update(read_partition_ts_codes(raw_dir / dataset / f"period={period}.parquet", start_date, end_date, require_date_match=True))
    for dataset in ann_month_datasets:
        for month in refresh_months:
            codes.update(read_partition_ts_codes(raw_dir / dataset / f"ann_month={month}.parquet", start_date, end_date, require_date_match=False))
    return codes


FUNDAMENTAL_EVENT_DATE_COLUMNS = ("f_ann_date", "ann_date", "first_ann_date", "imp_ann_date", "actual_date", "pre_date")


def read_partition_ts_codes(path: Path, start_date: str | None = None, end_date: str | None = None, require_date_match: bool = False) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path)
    if "ts_code" not in frame.columns:
        return set()
    if start_date and end_date:
        date_columns = [column for column in FUNDAMENTAL_EVENT_DATE_COLUMNS if column in frame.columns]
        if date_columns:
            mask = pd.Series(False, index=frame.index)
            for column in date_columns:
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
    except Exception:
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


def download_fundamental_period_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: FundamentalDataset,
    periods: list[str],
    force: bool,
    page_limit: int,
    force_periods: set[str] | None = None,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    force_periods = force_periods or set()
    written = 0
    skipped = 0
    total_rows = 0
    blocked: list[str] = []
    for index, period in enumerate(periods, start=1):
        path = raw_dir / spec.api_name / f"period={period}.parquet"
        if not force and period not in force_periods and committed_partition_intact(path):
            skipped += 1
            continue
        params = {spec.period_param: period}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = frame(result)
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=result.fields,
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            allow_key_removal_overwrite=True,
        )
        if did_write:
            written += 1
            total_rows += len(df)
        elif not df.empty:
            blocked.append(period)
        if index % 16 == 0:
            print(f"{spec.api_name} periods {index}/{len(periods)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done periods={len(periods)} skipped={skipped} written={written} rows_written={total_rows} blocked={len(blocked)}")
    raise_on_blocked_fundamental_overwrites(spec.api_name, blocked)

def download_fundamental_ann_month_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: FundamentalDataset,
    windows: list[tuple[str, str, str]],
    force: bool,
    page_limit: int,
    force_months: set[str] | None = None,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    force_months = force_months or set()
    written = 0
    skipped = 0
    total_rows = 0
    blocked: list[str] = []
    for index, (start_date, end_date, ann_month) in enumerate(windows, start=1):
        path = raw_dir / spec.api_name / f"ann_month={ann_month}.parquet"
        params = {"start_date": start_date, "end_date": end_date}
        if ann_month not in force_months and should_skip_existing_partition(path, force=force, requested_params=params):
            skipped += 1
            continue
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = frame(result)
        meta_params = dict(params)
        meta_params["ann_month"] = ann_month
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=result.fields,
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            # Full-month re-pulls legitimately drop keys (forecast update_flag
            # supersedes in place, retractions): same contract as every other
            # whole-partition path, or the nightly 3-month refresh wedges on the
            # first removed key. The disproportionate-shrink guard still blocks
            # transient truncation.
            allow_key_removal_overwrite=True,
        )
        if did_write:
            written += 1
            total_rows += len(df)
        elif not df.empty:
            blocked.append(ann_month)
        if index % 24 == 0:
            print(f"{spec.api_name} months {index}/{len(windows)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done months={len(windows)} skipped={skipped} written={written} rows_written={total_rows} blocked={len(blocked)}")
    raise_on_blocked_fundamental_overwrites(spec.api_name, blocked)

def raise_on_blocked_fundamental_overwrites(api_name: str, blocked: list[str]) -> None:
    """Non-empty responses the revision guard refused are a data-integrity
    alarm, never a quiet counter: swallowing them froze fina_mainbz_vip at a
    stale vintage for months while every job reported ok. Review the revision
    ledger; if the restatement is genuine, delete the affected partitions per
    the shrink-guard runbook and re-run the covering job."""
    if blocked:
        raise RuntimeError(
            f"{api_name}: the revision guard refused non-empty overwrites for {len(blocked)} partition(s) "
            f"(destructive shrink); review the revision ledger before retrying: {blocked[:10]}"
        )

def download_fundamental_ts_code_dataset(
    client: TuShareClient,
    raw_dir: Path,
    spec: FundamentalDataset,
    stock_codes: list[str],
    force: bool,
    page_limit: int,
    force_codes: set[str] | None = None,
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    force_codes = force_codes or set()
    written = 0
    skipped = 0
    total_rows = 0
    blocked: list[str] = []
    for index, ts_code in enumerate(stock_codes, start=1):
        path = raw_dir / spec.api_name / f"ts_code={ts_code}.parquet"
        if not force and ts_code not in force_codes and committed_partition_intact(path):
            skipped += 1
            if index % 500 == 0:
                print(f"{spec.api_name} codes {index}/{len(stock_codes)} skipped={skipped} written={written}")
            continue
        params = {"ts_code": ts_code}
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        df = frame(result)
        meta_params = dict(params)
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        did_write = write_parquet_revision_aware(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=result.fields,
            key_columns=list(spec.key_columns),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            allow_key_removal_overwrite=True,
        )
        if did_write:
            written += 1
            total_rows += len(df)
        elif not df.empty:
            blocked.append(ts_code)
        if index % 500 == 0:
            print(f"{spec.api_name} codes {index}/{len(stock_codes)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done codes={len(stock_codes)} skipped={skipped} written={written} rows_written={total_rows} blocked={len(blocked)}")
    raise_on_blocked_fundamental_overwrites(spec.api_name, blocked)

def download_intraday(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    selected_intraday_datasets(args.datasets)
    universe = load_minute_universe(raw_dir, args)
    tasks: list[tuple[str, int, str, str]] = []
    for _, row in universe.iterrows():
        ts_code = str(row["ts_code"])
        for year, year_start, year_end in active_year_windows(row, args.start_date, args.end_date):
            tasks.append((ts_code, year, year_start, year_end))
    if not tasks:
        raise RuntimeError(f"no active stock-year windows for {args.start_date}-{args.end_date}")
    page_limit = min(args.page_limit or STK_MINS_PAGE_LIMIT, STK_MINS_PAGE_LIMIT)
    dataset_dir = raw_dir / STK_MINS_DATASET
    skipped = 0
    written = 0
    total_rows = 0
    total_pages = 0
    for index, (ts_code, year, year_start, year_end) in enumerate(tasks, start=1):
        path = dataset_dir / f"ts_code={safe_partition_value(ts_code)}" / f"year={year}.parquet"
        if path.exists() and not args.force:
            skipped += 1
            if index % 250 == 0:
                print(f"{STK_MINS_DATASET} {index}/{len(tasks)} skipped={skipped} written={written} rows_written={total_rows}")
            continue
        params = {
            "ts_code": ts_code,
            "freq": STK_MINS_FREQ,
            "start_date": minute_datetime(year_start),
            "end_date": minute_datetime(year_end, end=True),
        }
        try:
            result, pages = query_paged(client, STK_MINS_API_NAME, params, STK_MINS_FIELDS, page_limit)
        except Exception as exc:
            raise RuntimeError(f"{STK_MINS_API_NAME} ts_code={ts_code} year={year} failed: {exc}") from exc
        df = augment_stk_mins_frame(frame(result))
        if path.exists() and args.force:
            # A partial-window --force must not replace the whole stock-year file:
            # rows outside the requested window are kept, the window is replaced.
            existing = pd.read_parquet(path)
            if not existing.empty and "trade_date" in existing.columns:
                dates = existing["trade_date"].astype(str)
                outside = existing[(dates < str(year_start)) | (dates > str(year_end))]
                if not outside.empty:
                    df = concat_rows([outside, df])
                    dedup_cols = [c for c in ("ts_code", "trade_time") if c in df.columns]
                    if dedup_cols:
                        df = df.drop_duplicates(dedup_cols, keep="last")
                    sort_cols = [c for c in ("trade_date", "trade_time") if c in df.columns]
                    if sort_cols:
                        df = df.sort_values(sort_cols, kind="stable")
                    df = df.reset_index(drop=True)
        meta_params = dict(params)
        meta_params.update({
            "dataset": STK_MINS_DATASET,
            "official_page_limit": STK_MINS_PAGE_LIMIT,
            "pagination": {"page_limit": page_limit, "pages": pages},
            "unit_rules": {"vol": "shares", "amount": "CNY", "trade_time": "Asia/Shanghai minute/bar timestamp; includes 09:30 and 15:00 auction bars"},
        })
        write_parquet(path, df, api_name=STK_MINS_API_NAME, params=meta_params, fields=list(df.columns))
        written += 1
        total_rows += len(df)
        total_pages += pages
        if index % 50 == 0:
            print(f"{STK_MINS_DATASET} {index}/{len(tasks)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"{STK_MINS_DATASET} done tasks={len(tasks)} skipped={skipped} written={written} rows_written={total_rows} pages={total_pages}")
    print(f"intraday minute download finished under {raw_dir}")
    return 0

def read_stk_mins_source_subset(path: Path, trade_dates: set[str]) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
    columns = [col for col in STK_MINS_REQUIRED_COLUMNS if col in pq.ParquetFile(path).schema_arrow.names]
    try:
        df = pd.read_parquet(path, columns=columns, filters=[("trade_date", "in", sorted(trade_dates))])
    except Exception:
        df = pd.read_parquet(path, columns=columns)
    if df.empty or "trade_date" not in df.columns:
        return pd.DataFrame(columns=columns)
    return df[df["trade_date"].astype(str).isin(trade_dates)].copy()

def source_stk_mins_files(raw_dir: Path, args: argparse.Namespace, years: set[str]) -> list[Path]:
    files = sorted((raw_dir / STK_MINS_DATASET).glob("ts_code=*/year=*.parquet"))
    if years:
        files = [path for path in files if path.stem.split("=", 1)[-1] in years]
    if getattr(args, "codes", None):
        wanted = {safe_partition_value(str(code).strip()) for code in args.codes if str(code).strip()}
        files = [path for path in files if path.parent.name.split("=", 1)[-1] in wanted]
    if getattr(args, "max_codes", None):
        selected_codes = sorted({path.parent.name for path in files})[: int(args.max_codes)]
        files = [path for path in files if path.parent.name in set(selected_codes)]
    return files

def write_stk_mins_by_date(
    path: Path,
    df: pd.DataFrame,
    *,
    trade_date: str,
    source: str,
    params: dict[str, Any],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    meta_params = dict(params)
    meta_params.update({
        "dataset": STK_MINS_BY_DATE_DATASET,
        "trade_date": trade_date,
        "source": source,
        "unit_rules": {"vol": "shares", "amount": "CNY", "available_at": "source trade_time bar close"},
    })
    write_parquet_revision_aware(
        path,
        df,
        api_name=STK_MINS_API_NAME,
        params=meta_params,
        fields=list(df.columns),
        key_columns=["trade_date", "ts_code", "trade_time"],
        revision_ledger=revision_ledger,
        allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        allow_key_removal_overwrite=True,
    )

def compact_intraday_by_date(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output_dataset = args.output_dataset
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    month_to_dates: dict[str, list[str]] = {}
    for trade_date in trade_dates:
        month_to_dates.setdefault(trade_date[:6], []).append(trade_date)
    years = {trade_date[:4] for trade_date in trade_dates}
    source_files = source_stk_mins_files(raw_dir, args, years)
    if not source_files:
        raise RuntimeError(f"no source {STK_MINS_DATASET} files found for years={sorted(years)}")
    written = 0
    skipped = 0
    total_rows = 0
    for month, month_dates in sorted(month_to_dates.items()):
        needed_dates = []
        for trade_date in month_dates:
            path = stk_mins_by_date_path(raw_dir, output_dataset, trade_date)
            if path.exists() and not args.force:
                skipped += 1
            else:
                needed_dates.append(trade_date)
        if not needed_dates:
            continue
        date_set = set(needed_dates)
        buffers: dict[str, list[pd.DataFrame]] = {trade_date: [] for trade_date in needed_dates}
        scanned = 0
        for source_path in source_files:
            if source_path.stem.split("=", 1)[-1] not in {date[:4] for date in date_set}:
                continue
            subset = read_stk_mins_source_subset(source_path, date_set)
            scanned += 1
            if subset.empty:
                continue
            for trade_date, group in subset.groupby("trade_date", sort=False):
                key = str(trade_date)
                if key in buffers:
                    buffers[key].append(group)
        for trade_date in needed_dates:
            if not buffers[trade_date]:
                if args.allow_empty:
                    combined = pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
                else:
                    raise RuntimeError(f"no minute rows found while compacting trade_date={trade_date}")
            else:
                combined = concat_rows(buffers[trade_date])
            normalized, normalize_details = normalize_stk_mins_by_date_frame(combined, trade_date)
            expected_codes = intraday_expected_codes_for_day(raw_dir, args, trade_date)
            ok, details = validate_stk_mins_by_date_frame(
                normalized,
                trade_date,
                expected_codes=expected_codes,
                min_rows=args.min_rows_per_day,
                allow_missing_codes=args.allow_missing_codes,
            )
            if not ok and not args.allow_validation_warnings:
                raise RuntimeError(f"compacted intraday by-date validation failed for {trade_date}: {details}")
            params = {
                "source_dataset": STK_MINS_DATASET,
                "source_layout": "ts_code/year",
                "output_layout": "trade_date",
                "month": month,
                "source_files_scanned": scanned,
                "normalize": normalize_details,
                "validation": details,
            }
            write_stk_mins_by_date(
                stk_mins_by_date_path(raw_dir, output_dataset, trade_date),
                normalized,
                trade_date=trade_date,
                source="compact_from_stock_year",
                params=params,
            )
            written += 1
            total_rows += len(normalized)
        print(f"{output_dataset} month={month} written={written} skipped={skipped} rows_written={total_rows}")
    print(f"{output_dataset} compact finished dates={len(trade_dates)} written={written} skipped={skipped} rows_written={total_rows}")
    return 0

def update_intraday_by_date(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    ensure_trade_cal_coverage(client, raw_dir, args.start_date, args.end_date)
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    page_limit = min(args.page_limit or STK_MINS_PAGE_LIMIT, STK_MINS_PAGE_LIMIT)
    written = 0
    skipped = 0
    total_rows = 0
    force_from = str(getattr(args, "force_from", None) or "")
    for trade_date in trade_dates:
        day_force = args.force or (force_from != "" and trade_date >= force_from)
        path = stk_mins_by_date_path(raw_dir, args.output_dataset, trade_date)
        if path.exists() and not day_force:
            existing = pd.read_parquet(path)
            if args.expected_codes_source == "minute" and not existing.empty:
                expected_codes = set(existing["ts_code"].dropna().astype(str))
                if getattr(args, "max_codes", None):
                    expected_codes = set(sorted(expected_codes)[: int(args.max_codes)])
            else:
                expected_codes = intraday_expected_codes_for_day(raw_dir, args, trade_date) or set()
            existing_allow_missing = max(int(args.allow_missing_codes), int(getattr(args, "existing_allow_missing_codes", 0)))
            ok, _ = validate_stk_mins_by_date_frame(
                existing,
                trade_date,
                expected_codes=expected_codes if expected_codes else None,
                min_rows=args.min_rows_per_day,
                allow_missing_codes=existing_allow_missing,
            )
            if ok:
                skipped += 1
                continue
        else:
            expected_codes = intraday_expected_codes_for_day(raw_dir, args, trade_date) or set()
        if not expected_codes:
            skipped += 1
            print(f"{args.output_dataset} trade_date={trade_date} skipped_empty_expected_codes")
            continue
        collected: dict[str, pd.DataFrame] = {}
        pending = sorted(expected_codes)
        pages_by_code: dict[str, int] = {}
        for attempt in range(1, args.max_retries + 1):
            if not pending:
                break
            failed: list[str] = []
            for index, ts_code in enumerate(pending, start=1):
                params = {
                    "ts_code": ts_code,
                    "freq": STK_MINS_FREQ,
                    "start_date": minute_datetime(trade_date),
                    "end_date": minute_datetime(trade_date, end=True),
                }
                try:
                    result, pages = query_paged(client, STK_MINS_API_NAME, params, STK_MINS_FIELDS, page_limit)
                    df = augment_stk_mins_frame(frame(result))
                    df = df[df["trade_date"].astype(str) == trade_date].copy()
                    if df.empty:
                        failed.append(ts_code)
                    else:
                        collected[ts_code] = df
                        pages_by_code[ts_code] = pages
                except Exception:
                    failed.append(ts_code)
                if index % 500 == 0:
                    print(f"{trade_date} attempt={attempt}/{args.max_retries} codes={index}/{len(pending)} collected={len(collected)} failed_current={len(failed)}")
            pending = failed
            if pending and attempt < args.max_retries:
                time.sleep(args.retry_delay_seconds)
        if len(pending) > args.allow_missing_codes:
            raise RuntimeError(f"{trade_date}: {len(pending)} minute codes still missing after retries; sample={pending[:20]}")
        combined = concat_rows(list(collected.values())) if collected else pd.DataFrame(columns=STK_MINS_REQUIRED_COLUMNS)
        normalized, normalize_details = normalize_stk_mins_by_date_frame(combined, trade_date)
        if expected_codes and normalized.empty:
            raise RuntimeError(
                f"{trade_date}: refusing to write zero-row intraday by-date file "
                f"for nonempty expected universe ({len(expected_codes)} codes)"
            )
        ok, details = validate_stk_mins_by_date_frame(
            normalized,
            trade_date,
            expected_codes=expected_codes,
            min_rows=args.min_rows_per_day,
            allow_missing_codes=args.allow_missing_codes,
        )
        if not ok and not args.allow_validation_warnings:
            raise RuntimeError(f"{trade_date}: intraday by-date update validation failed: {details}")
        params = {
            "output_layout": "trade_date",
            "expected_codes_source": args.expected_codes_source,
            "expected_codes": len(expected_codes),
            "missing_codes_after_retry": len(pending),
            "missing_code_sample": pending[:20],
            "normalize": normalize_details,
            "validation": details,
            "pagination": {"page_limit": page_limit, "pages_total": int(sum(pages_by_code.values()))},
            "max_retries": args.max_retries,
        }
        write_stk_mins_by_date(
            path,
            normalized,
            trade_date=trade_date,
            source="daily_incremental_update",
            params=params,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
        )
        written += 1
        total_rows += len(normalized)
        print(f"{args.output_dataset} trade_date={trade_date} written rows={len(normalized)} missing_codes={len(pending)}")
    print(f"{args.output_dataset} update finished dates={len(trade_dates)} written={written} skipped={skipped} rows_written={total_rows}")
    return 0

def download_text(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    allow_empty_revision_overwrite = getattr(args, "allow_empty_revision_overwrite", False)
    windows = month_windows(args.start_date, args.end_date)
    days = date_range_days(args.start_date, args.end_date)
    for dataset in selected_text_datasets(args.datasets, news_src=args.news_src):
        spec = TEXT_SPECS[dataset]
        start_date = max(args.start_date, spec.start_date)
        dataset_windows = [(s, e, m) for s, e, m in windows if e >= start_date]
        dataset_days = [d for d in days if d >= start_date]
        page_limit = spec_page_limit(spec, args.page_limit)
        if spec.strategy == "range_month":
            download_text_range_month(client, raw_dir, spec, dataset_windows, args.force, page_limit, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "time_range_month":
            download_text_time_range_month(client, raw_dir, spec, dataset_windows, args.force, page_limit, args.major_news_src, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "news_src_day":
            download_text_news_src_day(client, raw_dir, spec, dataset_days, args.force, page_limit, args.news_src, revision_ledger, allow_empty_revision_overwrite)
        elif spec.strategy == "day":
            download_text_day(client, raw_dir, spec, dataset_days, args.force, revision_ledger, allow_empty_revision_overwrite)
        else:
            raise RuntimeError(f"unsupported text strategy {spec.strategy} for {dataset}")
    print(f"Text download finished under {raw_dir}")
    return 0

def write_text_result(
    path: Path,
    result: ApiResult,
    spec: TextDataset,
    params: dict[str, Any],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> int:
    df = augment_text_frame(frame(result), spec)
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

def download_text_range_month(client: TuShareClient, raw_dir: Path, spec: TextDataset, windows: list[tuple[str, str, str]], force: bool, page_limit: int, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    written = 0
    skipped = 0
    total_rows = 0
    for index, (start_date, end_date, month) in enumerate(windows, start=1):
        path = raw_dir / spec.api_name / f"month={month}.parquet"
        params = {"start_date": start_date, "end_date": end_date}
        if should_skip_existing_partition(path, force=force, requested_params=params):
            skipped += 1
            continue
        result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
        meta_params = dict(params)
        meta_params["month"] = month
        meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
        df = augment_text_frame(frame(result), spec)
        rows = write_window_merged_partition(
            path,
            df,
            api_name=spec.api_name,
            params=meta_params,
            fields=list(df.columns),
            key_columns=list(spec.key_columns),
            # range_month queries window on the DATE column (ann_date/report_date);
            # retention must match it — keying on the collection-time column keeps
            # out-of-window rec_time rows forever, so source revisions/retractions
            # of anns_d/report_rc could never be purged.
            date_columns=[spec.date_column, spec.time_column, "available_at"],
            start_date=start_date,
            end_date=end_date,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=allow_empty_revision_overwrite,
        )
        written += 1
        total_rows += rows
        if index % 24 == 0:
            print(f"{spec.api_name} months {index}/{len(windows)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done months={len(windows)} skipped={skipped} written={written} rows_written={total_rows}")

def download_text_time_range_month(
    client: TuShareClient,
    raw_dir: Path,
    spec: TextDataset,
    windows: list[tuple[str, str, str]],
    force: bool,
    page_limit: int,
    sources: list[str],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    source_values = sources or [""]
    for source in source_values:
        source_suffix = f"src={safe_partition_value(source)}" if source else "src=all"
        written = 0
        skipped = 0
        total_rows = 0
        for index, (start_date, end_date, month) in enumerate(windows, start=1):
            path = raw_dir / spec.api_name / source_suffix / f"month={month}.parquet"
            params = {"start_date": as_datetime_window(start_date), "end_date": as_datetime_window(end_date, end=True)}
            if source:
                params["src"] = source
            if should_skip_existing_partition(path, force=force, requested_params=params):
                skipped += 1
                continue
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
            meta_params = dict(params)
            meta_params["month"] = month
            meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
            df = augment_text_frame(frame(result), spec)
            rows = write_window_merged_partition(
                path,
                df,
                api_name=spec.api_name,
                params=meta_params,
                fields=list(df.columns),
                key_columns=list(spec.key_columns),
                date_columns=[spec.time_column, spec.date_column, "available_at"],
                start_date=start_date,
                end_date=end_date,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=allow_empty_revision_overwrite,
            )
            written += 1
            total_rows += rows
            if index % 24 == 0:
                print(f"{spec.api_name}/{source_suffix} months {index}/{len(windows)} skipped={skipped} written={written} rows_written={total_rows}")
        print(f"{spec.api_name}/{source_suffix} done months={len(windows)} skipped={skipped} written={written} rows_written={total_rows}")

def download_text_news_src_day(
    client: TuShareClient,
    raw_dir: Path,
    spec: TextDataset,
    days: list[str],
    force: bool,
    page_limit: int,
    sources: list[str],
    revision_ledger: Path | str | None = None,
    allow_empty_revision_overwrite: bool = False,
) -> None:
    for source in selected_news_sources(sources):
        source_suffix = f"src={safe_partition_value(source)}"
        written = 0
        skipped = 0
        total_rows = 0
        for index, day in enumerate(days, start=1):
            path = raw_dir / spec.api_name / source_suffix / f"date={day}.parquet"
            params = {"src": source, "start_date": as_datetime_window(day), "end_date": as_datetime_window(day, end=True)}
            if should_skip_existing_partition(path, force=force, requested_params=params):
                skipped += 1
                continue
            result, pages = query_paged(client, spec.api_name, params, spec.fields, page_limit)
            meta_params = dict(params)
            meta_params["date"] = day
            meta_params["pagination"] = {"page_limit": page_limit, "pages": pages}
            rows = write_text_result(path, result, spec, meta_params, revision_ledger, allow_empty_revision_overwrite)
            written += 1
            total_rows += rows
            if index % 250 == 0:
                print(f"{spec.api_name}/{source_suffix} days {index}/{len(days)} skipped={skipped} written={written} rows_written={total_rows}")
        print(f"{spec.api_name}/{source_suffix} done days={len(days)} skipped={skipped} written={written} rows_written={total_rows}")

def download_text_day(client: TuShareClient, raw_dir: Path, spec: TextDataset, days: list[str], force: bool, revision_ledger: Path | str | None = None, allow_empty_revision_overwrite: bool = False) -> None:
    written = 0
    skipped = 0
    total_rows = 0
    for index, day in enumerate(days, start=1):
        path = raw_dir / spec.api_name / f"date={day}.parquet"
        if path.exists() and not force:
            skipped += 1
            continue
        params = {spec.date_column or "date": day}
        result = client.query(spec.api_name, params, spec.fields)
        rows = write_text_result(path, result, spec, params, revision_ledger, allow_empty_revision_overwrite)
        written += 1
        total_rows += rows
        if index % 250 == 0:
            print(f"{spec.api_name} days {index}/{len(days)} skipped={skipped} written={written} rows_written={total_rows}")
    print(f"{spec.api_name} done days={len(days)} skipped={skipped} written={written} rows_written={total_rows}")

def set_download_defaults(args: argparse.Namespace) -> None:
    if args.min_interval_seconds is None:
        args.min_interval_seconds = {
            "reference": 0.12,
            "daily": 0.18,
            "fundamental": 0.22,
            "intraday": 0.22,
            "event_flow": 0.22,
            "board_trading": 0.22,
            "text_evidence": 0.22,
            "macro": 0.22,
            "global": 0.22,
        }[args.tier]
    if args.timeout_seconds is None:
        args.timeout_seconds = 120 if args.tier == "intraday" else 90 if args.tier in {"fundamental", "event_flow", "board_trading", "text_evidence", "macro", "global"} else 60
    if args.page_limit is None:
        if args.tier == "intraday":
            args.page_limit = STK_MINS_PAGE_LIMIT
        elif args.tier == "text_evidence":
            args.page_limit = None
        elif args.tier == "fundamental":
            args.page_limit = 7000
        elif args.tier == "daily":
            args.page_limit = TRADE_DATE_PAGE_LIMIT
        elif args.tier == "board_trading":
            args.page_limit = None
        else:
            args.page_limit = 10000

def download_selected_tier(args: argparse.Namespace) -> int:
    set_download_defaults(args)
    if args.tier == "reference":
        return download_reference(args)
    if args.tier == "daily":
        return download_daily(args)
    if args.tier == "fundamental":
        return download_fundamental(args)
    if args.tier == "intraday":
        return download_intraday(args)
    if args.tier == "event_flow":
        return download_event_flow(args)
    if args.tier == "board_trading":
        return download_board_trading(args)
    if args.tier == "text_evidence":
        return download_text(args)
    if args.tier in {"macro", "global"}:
        return download_macro(args)
    raise RuntimeError(f"unknown download tier {args.tier}")

def ns_from(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)

def sidecar_params(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("params") or {}
    except Exception:
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
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
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
            rescue_ann_limit_hits=args.rescue_ann_limit_hits,
            rescue_ann_date=[],
            max_ann_rescue_days=args.max_ann_rescue_days,
            float_rescue_date=[],
            rescue_universe=args.rescue_universe,
            rescue_code=[],
            rescue_codes_file=None,
            no_anns_candidates=False,
            no_cross_path_candidates=False,
            max_rescue_calls=args.max_rescue_calls,
            skip_float_date_union=False,
            force=force,
            write_union=True,
            union_output=args.union_output,
            output=args.share_float_process_output,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            min_interval_seconds=args.min_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
        summary,
    )

def update_all_dimensions(args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    repo_root = Path.cwd().resolve()
    raw_dir = repo_root / args.raw_dir
    revision_ledger = resolve_revision_ledger(raw_dir, getattr(args, "revision_ledger", REVISION_EVENTS_PATH), repo_root=repo_root)
    start_date = args.start_date
    if parse_yyyymmdd(start_date) > parse_yyyymmdd(args.end_date):
        raise RuntimeError(f"start_date {start_date} is after end_date {args.end_date}")
    macro_floor = args.macro_start_date
    macro_start_date = min(start_date, macro_floor, key=parse_yyyymmdd)
    refresh_open_window = bool(getattr(args, "refresh_open_window", False))
    force_open_window = bool(args.force or refresh_open_window)
    trade_cal_end_date = format_yyyymmdd(
        parse_yyyymmdd(args.end_date) + timedelta(days=max(int(getattr(args, "trade_cal_lookahead_days", 0) or 0), 0))
    )
    run_update_step(
        "reference",
        download_selected_tier,
        ns_from(
            args,
            tier="reference",
            start_date=start_date,
            bak_start_date=args.bak_start_date or start_date,
            end_date=args.end_date,
            trade_cal_end_date=trade_cal_end_date,
            datasets=None,
            force=args.force,
            page_limit=args.page_limit,
            refresh_reference_datasets=getattr(args, "refresh_reference_datasets", []),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            min_interval_seconds=getattr(args, "reference_min_interval_seconds", None) or args.min_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
        summary,
    )
    run_update_step(
        "daily",
        download_selected_tier,
        ns_from(
            args,
            tier="daily",
            start_date=start_date,
            end_date=args.end_date,
            datasets=args.daily_datasets,
            refresh_daily_datasets=getattr(args, "refresh_daily_datasets", []),
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            force=args.force,
            page_limit=args.page_limit,
            min_interval_seconds=args.min_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
        summary,
    )
    run_update_step(
        "macro",
        download_selected_tier,
        ns_from(
            args,
            tier="macro",
            start_date=start_date,
            macro_start_date=macro_start_date,
            end_date=args.end_date,
            datasets=args.macro_datasets,
            force=force_open_window,
            page_limit=args.page_limit,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            min_interval_seconds=args.min_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
        summary,
    )
    if args.include_global:
        run_update_step(
            "global",
            download_selected_tier,
            ns_from(
                args,
                tier="global",
                start_date=start_date,
                macro_start_date=macro_start_date,
                end_date=args.end_date,
                datasets=args.global_datasets,
                force=force_open_window,
                page_limit=args.page_limit,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
                min_interval_seconds=args.min_interval_seconds,
                timeout_seconds=args.timeout_seconds,
            ),
            summary,
        )
    run_update_step(
        "event_flow",
        download_selected_tier,
        ns_from(
            args,
            tier="event_flow",
            start_date=start_date,
            end_date=args.end_date,
            datasets=args.event_datasets,
            force=force_open_window,
            page_limit=args.page_limit,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            min_interval_seconds=args.min_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
        summary,
    )
    if args.include_board_trading:
        run_update_step(
            "board_trading",
            download_selected_tier,
            ns_from(
                args,
                tier="board_trading",
                start_date=start_date,
                end_date=args.end_date,
                datasets=args.board_datasets,
                force=force_open_window,
                page_limit=args.page_limit,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
                min_interval_seconds=args.min_interval_seconds,
                timeout_seconds=args.timeout_seconds,
            ),
            summary,
    )
    if args.include_intraday:
        intraday_start_date = start_date
        intraday_force = args.force
        intraday_force_from = None
        intraday_lookback = int(getattr(args, "intraday_refresh_lookback_days", 0) or 0)
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
            ns_from(
                args,
                start_date=intraday_start_date,
                end_date=args.end_date,
                output_dataset=args.output_dataset,
                expected_codes_source=args.expected_codes_source,
                codes=args.codes,
                max_codes=args.max_codes,
                min_rows_per_day=args.min_rows_per_day,
                allow_missing_codes=args.allow_missing_codes,
                allow_validation_warnings=args.allow_validation_warnings,
                max_retries=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
                existing_allow_missing_codes=args.existing_allow_missing_codes,
                page_limit=args.page_limit,
                min_interval_seconds=args.min_interval_seconds,
                timeout_seconds=args.timeout_seconds,
                force=intraday_force,
                force_from=intraday_force_from,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            ),
            summary,
        )
    if args.include_share_float_complete:
        update_share_float_complete_data(args, summary, start_date=start_date, force=force_open_window)
    if args.include_text:
        run_update_step(
            "text_evidence",
            download_selected_tier,
            ns_from(
                args,
                tier="text_evidence",
                start_date=start_date,
                end_date=args.end_date,
                datasets=args.text_datasets,
                force=force_open_window,
                page_limit=args.page_limit,
                revision_ledger=revision_ledger,
                allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
                min_interval_seconds=args.min_interval_seconds,
                timeout_seconds=args.timeout_seconds,
            ),
            summary,
        )
    run_update_step(
        "fundamental",
        download_selected_tier,
        ns_from(
            args,
            tier="fundamental",
            start_date=start_date,
            end_date=args.end_date,
            datasets=args.fundamental_datasets,
            fundamental_refresh_period_count=getattr(args, "fundamental_refresh_period_count", 6),
            fundamental_refresh_ann_month_count=getattr(args, "fundamental_refresh_ann_month_count", 3),
            fundamental_refresh_ts_code_datasets=getattr(args, "fundamental_refresh_ts_code_datasets", ["dividend", "fina_audit", "fina_mainbz_vip"]),
            fundamental_refresh_event_days=getattr(args, "fundamental_refresh_event_days", 90),
            fundamental_dividend_probe_days=getattr(args, "fundamental_dividend_probe_days", 90),
            force=args.force,
            page_limit=args.page_limit,
            revision_ledger=revision_ledger,
            allow_empty_revision_overwrite=getattr(args, "allow_empty_revision_overwrite", False),
            min_interval_seconds=args.min_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
        summary,
    )

def update_data(args: argparse.Namespace) -> int:
    if args.min_interval_seconds is None:
        args.min_interval_seconds = 0.22
    if args.timeout_seconds is None:
        args.timeout_seconds = 120
    summary: list[dict[str, Any]] = []
    update_all_dimensions(args, summary)
    print(json.dumps({"status": "ok", "start_date": args.start_date, "end_date": args.end_date, "steps": summary}, ensure_ascii=False, indent=2))
    return 0


def add_download_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("download", help="download TuShare raw data by semantic tier")
    parser.add_argument("--tier", required=True, choices=core.DOWNLOAD_TIER_CHOICES)
    core.add_raw_arg(parser)
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--bak-start-date", default="20160101")
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--trade-cal-end-date", help="Optional reference-tier lookahead end date used only for trade_cal coverage.")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--refresh-daily-datasets", nargs="+", choices=core.DAILY_DOWNLOAD_DATASETS, default=[])
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--allow-empty-revision-overwrite", action="store_true")
    parser.add_argument("--skip-bak-basic", action="store_true")
    parser.add_argument("--refresh-reference-datasets", nargs="+", choices=core.REFERENCE_DATASETS, default=[])
    parser.add_argument("--fundamental-refresh-period-count", type=int, default=0)
    parser.add_argument("--fundamental-refresh-ann-month-count", type=int, default=0)
    parser.add_argument("--fundamental-refresh-ts-code-datasets", nargs="+", choices=["dividend", "fina_audit", "fina_mainbz_vip"], default=[])
    parser.add_argument("--fundamental-refresh-event-days", type=int, default=90)
    parser.add_argument("--fundamental-dividend-probe-days", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--zero-rows-not-ready",
        action="store_true",
        help="event_flow tier only: treat required zero-row responses as source-not-ready instead of failing. "
        "Exits 75 (no-mutation retry contract) when nothing was written; with partial writes it commits and "
        "defers the empty partitions to the retry job. For pre-open cron attempts that race source publication.",
    )
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--codes", nargs="+", help="Optional explicit ts_code list for intraday minute window tests or targeted refreshes.")
    parser.add_argument("--page-limit", type=int, help="Optional requested page size; text evidence datasets are clamped to official per-call limits.")
    parser.add_argument("--news-src", action="append", default=[], help="News short-message source; repeatable. Defaults to all official TuShare sources; use all to expand explicitly.")
    parser.add_argument("--major-news-src", action="append", default=[], help="Optional major_news source filter; repeatable.")
    core.add_board_filter_args(parser)
    core.add_macro_filter_args(parser)
    core.add_runtime_args(parser, min_interval=None, timeout=None)

def add_update_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("update", help="fill missing TuShare data across all retained domains")
    core.add_raw_arg(parser)
    parser.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--start-date", required=True, help="Fill missing data from this date through --end-date across all retained data domains.")
    parser.add_argument("--bak-start-date", help="Optional bak_basic lower bound. Defaults to --start-date.")
    parser.add_argument(
        "--trade-cal-lookahead-days",
        type=int,
        default=7,
        help="Extend reference trade_cal this many calendar days beyond --end-date so pre-open jobs and next-session PIT mapping have calendar coverage.",
    )
    parser.add_argument("--daily-datasets", nargs="+", choices=core.DAILY_DOWNLOAD_DATASETS)
    parser.add_argument(
        "--refresh-daily-datasets",
        nargs="+",
        choices=core.DAILY_DOWNLOAD_DATASETS,
        default=[],
        help="Daily trade-date datasets to force-refresh within the update window; cron config selects the routine recent-window revision set.",
    )
    parser.add_argument(
        "--refresh-reference-datasets",
        nargs="+",
        choices=core.REFERENCE_DATASETS,
        default=["stock_basic", "stock_company", "namechange", "index_classify", "index_member_all"],
        help="Reference datasets to force-refresh during daily update; heavier datasets should be explicit.",
    )
    parser.add_argument(
        "--refresh-open-window",
        action="store_true",
        help="Force-refresh the rolling update window for macro/global/event/board/text/share_float tiers; intended for cron, not large manual history fills.",
    )
    parser.add_argument("--reference-min-interval-seconds", type=float, help="Optional lower call frequency for the reference refresh step.")
    parser.add_argument("--fundamental-datasets", nargs="+", choices=core.FUNDAMENTAL_DATASETS)
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
    parser.add_argument("--macro-datasets", nargs="+", choices=core.MACRO_DATASETS)
    parser.add_argument("--global-datasets", nargs="+", choices=core.MACRO_DATASETS)
    parser.add_argument(
        "--macro-start-date",
        default=MACRO_RETAINED_FLOOR,
        help="Retained lower bound for macro/global range-style datasets during rolling updates; prevents short-window range files from replacing full-context coverage.",
    )
    parser.add_argument("--event-datasets", nargs="+", choices=[dataset for dataset in core.EVENT_FLOW_DATASETS if dataset != "share_float"])
    parser.add_argument("--board-datasets", nargs="+", choices=core.BOARD_TRADING_DATASETS)
    parser.add_argument(
        "--include-board-trading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update 打板专题数据 by default; use --no-include-board-trading for lightweight refreshes.",
    )
    parser.add_argument("--text-datasets", nargs="+", choices=core.TEXT_DATASETS)
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
    parser.add_argument("--output-dataset", default=core.STK_MINS_BY_DATE_DATASET)
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
    parser.add_argument("--max-ann-rescue-days", type=int, default=5)
    parser.add_argument("--rescue-universe", choices=["candidate", "explicit", "all_a"], default="candidate")
    parser.add_argument("--max-rescue-calls", type=int, default=50000)
    parser.add_argument("--union-output", default="data/raw/share_float_complete/share_float_complete.parquet")
    parser.add_argument("--share-float-process-output", help="Optional temporary process report path for share_float_complete.")
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--allow-empty-revision-overwrite", action="store_true")
    parser.add_argument("--skip-bak-basic", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page-limit", type=int)
    parser.add_argument("--news-src", action="append", default=[])
    parser.add_argument("--major-news-src", action="append", default=[])
    core.add_board_filter_args(parser)
    core.add_macro_filter_args(parser)
    core.add_runtime_args(parser, min_interval=None, timeout=None)


def add_auction_recheck_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("recheck-stk-auction", help="evening auction self-heal: re-read the latest open day, capture missing days in the window")
    core.add_raw_arg(parser)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--landing-job", default="cn_evening_auction_backfill")
    parser.add_argument("--page-limit", type=int, default=TRADE_DATE_PAGE_LIMIT)
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--min-interval-seconds", type=float, default=0.35)
    parser.add_argument("--timeout-seconds", type=int, default=60)


def add_auction_capture_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("capture-open-auction", help="poll and atomically publish today's complete stk_auction frame")
    core.add_raw_arg(parser)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--page-limit", type=int, default=TRADE_DATE_PAGE_LIMIT)
    parser.add_argument("--max-wait-seconds", type=float, default=170.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--stable-reads", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=AUCTION_CAPTURE_MIN_ROWS_FLOOR)
    parser.add_argument("--min-previous-day-ratio", type=float, default=AUCTION_CAPTURE_MIN_PREVIOUS_DAY_RATIO)
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    core.add_runtime_args(parser, min_interval=0.22, timeout=30)

def add_intraday_parsers(sub: argparse._SubParsersAction) -> None:
    compact = sub.add_parser("compact-intraday-by-date", help="build final full-market daily minute files from stock-year source partitions")
    core.add_intraday_by_date_common_args(compact)
    compact.add_argument("--force", action="store_true")
    compact.add_argument("--allow-empty", action="store_true")
    compact.add_argument("--allow-validation-warnings", action="store_true")

    update = sub.add_parser("update-intraday-by-date", help="download/retry trade dates directly into final daily minute files")
    core.add_intraday_by_date_common_args(update, expected_codes_choices=["daily", "active", "minute"], expected_codes_default="minute")
    update.add_argument("--force", action="store_true")
    update.add_argument("--allow-validation-warnings", action="store_true")
    update.add_argument("--max-retries", type=int, default=3)
    update.add_argument("--retry-delay-seconds", type=float, default=5.0)
    update.add_argument("--page-limit", type=int)
    update.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    update.add_argument("--allow-empty-revision-overwrite", action="store_true")
    core.add_runtime_args(update, min_interval=0.22, timeout=120)

def add_share_float_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("download-share-float-complete", help="download share_float through ann_date and targeted ts_code rescue paths")
    core.add_raw_arg(parser)
    parser.add_argument("--ann-start-date", default="20100101")
    parser.add_argument("--ann-end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--float-start-date", default="20200101")
    parser.add_argument("--float-end-date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--skip-ann-date", action="store_true")
    parser.add_argument("--rescue-ann-limit-hits", action="store_true", help="Retry ann_date partitions that hit 6000 rows by ann_date + ts_code.")
    parser.add_argument("--rescue-ann-date", action="append", default=[], help="Specific ann_date to retry by ts_code; repeatable.")
    parser.add_argument("--max-ann-rescue-days", type=int, default=5, help="Safety cap for automatic ann_date + ts_code rescue days.")
    parser.add_argument("--float-rescue-date", action="append", default=[], help="Specific float_date to retry by ts_code; repeatable.")
    parser.add_argument("--rescue-universe", choices=["candidate", "explicit", "all_a"], default="candidate")
    parser.add_argument("--rescue-code", action="append", default=[])
    parser.add_argument("--rescue-codes-file")
    parser.add_argument("--no-anns-candidates", action="store_true")
    parser.add_argument("--no-cross-path-candidates", action="store_true")
    parser.add_argument("--max-rescue-calls", type=int, default=50000)
    parser.add_argument("--skip-float-date-union", action="store_true")
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-union", action="store_true")
    parser.add_argument("--union-output", default="data/raw/share_float_complete/share_float_complete.parquet")
    parser.add_argument("--revision-ledger", default=REVISION_EVENTS_PATH)
    parser.add_argument("--allow-empty-revision-overwrite", action="store_true")
    core.add_runtime_args(parser, min_interval=0.22, timeout=90)
    parser.add_argument("--output", help="Optional process report path. No status file is written by default; event-flow audit checks the union artifact.")

def add_repair_text_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "repair-text-available-at",
        help="re-derive text available_at locally under the plausibility rule (no API calls)",
    )
    core.add_raw_arg(parser)
    parser.add_argument("--datasets", nargs="+", default=["anns_d", "report_rc"], choices=sorted(core.TEXT_SPECS))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_download_parser(sub)
    add_update_parser(sub)
    add_auction_capture_parser(sub)
    add_auction_recheck_parser(sub)
    add_intraday_parsers(sub)
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
    """The cron runner is the ONLY production-lake write entrance.

    It alone provides the generation fence, per-job state, failure alerting
    and the exclusive flock; a direct manual command against production
    data/raw would bypass all four, so it is refused with the runner
    equivalent (data docs §2.2). Only a genuine runner child -- proven by the
    inherited lock fd, not by an environment flag -- passes; non-production
    raw dirs (tests, local lakes) are unaffected."""
    if inherited_updater_lock_fd(_production_repo_root()) is not None:
        return  # genuine cron-runner child (inherited the held lock fd)
    production_raw = (_production_repo_root() / "data" / "raw").resolve()

    def _targets_production(value: object) -> bool:
        if not value:
            return False
        try:
            resolved = Path(str(value)).resolve()
        except OSError:
            return False
        # Subtree membership: --raw-dir <prod>/daily or a --union-output file
        # under the production lake are production writes just the same.
        return resolved == production_raw or production_raw in resolved.parents

    if not (_targets_production(getattr(args, "raw_dir", "data/raw"))
            or _targets_production(getattr(args, "union_output", None))):
        return
    raise SystemExit(
        "direct manual writes to the production lake are not supported; run the "
        "equivalent job through the cron runner (fence + state + alerts), e.g.:\n"
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
    if args.command == "compact-intraday-by-date":
        return compact_intraday_by_date(args)
    if args.command == "update-intraday-by-date":
        return update_intraday_by_date(args)
    if args.command == "download-share-float-complete":
        return download_share_float_complete(args)
    if args.command == "repair-text-available-at":
        stats = core.repair_text_available_at(args.raw_dir, list(args.datasets))
        print(json.dumps({"status": "ok", **stats}, ensure_ascii=False, sort_keys=True))
        return 0
    raise RuntimeError(f"unknown command {args.command}")


def main() -> int:
    args = parse_args()
    _refuse_manual_production_write(args)
    return _dispatch(args)

if __name__ == "__main__":
    raise SystemExit(main())
