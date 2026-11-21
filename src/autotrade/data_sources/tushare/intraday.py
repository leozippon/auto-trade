"""The one-minute bar layer: per-code yearly source files (stk_mins_1min) and
the full-market daily files built from them or pulled directly
(stk_mins_1min_by_date), with the checks a day file must pass.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from autotrade.environment.data.pit import concat_rows, partition_value

from .client import TuShareClient, frame, load_token, query_paged
from .common import (
    MUTATED_NOT_READY_RETRY_EXIT_CODE,
    ensure_trade_cal_coverage,
    load_sse_open_dates,
    resolve_revision_ledger,
    safe_partition_value,
    select_datasets,
)
from .io import parquet_rows, read_many, write_parquet
from .specs import (
    INTRADAY_DATASETS,
    STK_MINS_API_NAME,
    STK_MINS_BATCH_CODES,
    STK_MINS_BY_DATE_DATASET,
    STK_MINS_DATASET,
    STK_MINS_FIELDS,
    STK_MINS_FREQ,
    STK_MINS_PAGE_LIMIT,
    STK_MINS_QUOTA_MARKER,
    STK_MINS_REQUIRED_COLUMNS,
)
from .versions import publish, report_complete


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
        return {"unique_times": 0, "has_0930": False, "has_1500": False, "invalid_time_rows": len(df)}
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
        "rows": len(df),
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
    except Exception:  # noqa: BLE001 - filtered read failure retries unfiltered; a second failure propagates
        df = pd.read_parquet(path, columns=columns)
    if df.empty or "trade_date" not in df.columns:
        return pd.DataFrame(columns=columns)
    return df[df["trade_date"].astype(str).isin(trade_dates)].copy()

def source_stk_mins_files(raw_dir: Path, args: argparse.Namespace, years: set[str]) -> list[Path]:
    files = sorted((raw_dir / STK_MINS_DATASET).glob("ts_code=*/year=*.parquet"))
    if years:
        files = [path for path in files if partition_value(path) in years]
    if args.codes:
        wanted = {safe_partition_value(str(code).strip()) for code in args.codes if str(code).strip()}
        files = [path for path in files if path.parent.name.split("=", 1)[-1] in wanted]
    if args.max_codes:
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
) -> None:
    meta_params = dict(params)
    meta_params.update({
        "dataset": STK_MINS_BY_DATE_DATASET,
        "trade_date": trade_date,
        "source": source,
        "unit_rules": {"vol": "shares", "amount": "CNY", "available_at": "source trade_time bar close"},
    })
    publish(
        path,
        df,
        api_name=STK_MINS_API_NAME,
        params=meta_params,
        fields=list(df.columns),
        key_columns=["trade_date", "ts_code", "trade_time"],
        ledger=revision_ledger,
    )

def compact_intraday_by_date(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
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
            if partition_value(source_path) not in {date[:4] for date in date_set}:
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
                revision_ledger=revision_ledger,
            )
            written += 1
            total_rows += len(normalized)
        print(f"{output_dataset} month={month} written={written} skipped={skipped} rows_written={total_rows}")
    print(f"{output_dataset} compact finished dates={len(trade_dates)} written={written} skipped={skipped} rows_written={total_rows}")
    return 0

def update_intraday_by_date(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    revision_ledger = resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)
    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    ensure_trade_cal_coverage(client, raw_dir, args.start_date, args.end_date, ledger=revision_ledger)
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    page_limit = min(args.page_limit or STK_MINS_PAGE_LIMIT, STK_MINS_PAGE_LIMIT)
    written = 0
    skipped = 0
    total_rows = 0
    force_from = str(args.force_from or "")
    for trade_date in trade_dates:
        day_force = args.force or (force_from != "" and trade_date >= force_from)
        path = stk_mins_by_date_path(raw_dir, args.output_dataset, trade_date)
        if path.exists() and not day_force:
            existing = pd.read_parquet(path)
            if args.expected_codes_source == "minute" and not existing.empty:
                expected_codes = set(existing["ts_code"].dropna().astype(str))
                if args.max_codes:
                    expected_codes = set(sorted(expected_codes)[: int(args.max_codes)])
            else:
                expected_codes = intraday_expected_codes_for_day(raw_dir, args, trade_date) or set()
            existing_allow_missing = max(int(args.allow_missing_codes), int(args.existing_allow_missing_codes))
            ok, _ = validate_stk_mins_by_date_frame(
                existing,
                trade_date,
                expected_codes=expected_codes or None,
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
        pages_total = 0
        quota_error = ""
        for attempt in range(1, args.max_retries + 1):
            if not pending:
                break
            failed: list[str] = []
            for offset in range(0, len(pending), STK_MINS_BATCH_CODES):
                batch = pending[offset : offset + STK_MINS_BATCH_CODES]
                params = {
                    "ts_code": ",".join(batch),
                    "freq": STK_MINS_FREQ,
                    "start_date": minute_datetime(trade_date),
                    "end_date": minute_datetime(trade_date, end=True),
                }
                try:
                    result, pages = query_paged(client, STK_MINS_API_NAME, params, STK_MINS_FIELDS, page_limit)
                except Exception as exc:  # noqa: BLE001 - non-quota failures mark the batch failed for the final report
                    if STK_MINS_QUOTA_MARKER in str(exc):
                        quota_error = str(exc)
                        break
                    failed.extend(batch)
                    continue
                pages_total += pages
                df = augment_stk_mins_frame(frame(result))
                df = df[df["trade_date"].astype(str) == trade_date]
                # A code the batch answered nothing for is retried like a failed
                # request; the batch's other codes are kept.
                by_code = {str(code): rows for code, rows in df.groupby("ts_code", sort=False)}
                for ts_code in batch:
                    if ts_code in by_code:
                        collected[ts_code] = by_code[ts_code].copy()
                    else:
                        failed.append(ts_code)
                done = offset + len(batch)
                if done // 500 > offset // 500:
                    print(f"{trade_date} attempt={attempt}/{args.max_retries} codes={done}/{len(pending)} collected={len(collected)} failed_current={len(failed)}")
            if quota_error:
                break
            pending = failed
            if pending and attempt < args.max_retries:
                time.sleep(args.retry_delay_seconds)
        if quota_error:
            # The quota holds until the vendor's next day, so the run stops
            # here: every day already written stays committed (76 = mutated,
            # not ready) and the next run resumes at this day.
            print(json.dumps({
                "status": "not_ready_after_mutation",
                "note": "stk_mins_daily_quota_exhausted",
                "trade_date": trade_date,
                "days_written": written,
                "error": quota_error[:300],
            }, ensure_ascii=False, sort_keys=True))
            return MUTATED_NOT_READY_RETRY_EXIT_CODE
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
            "pagination": {"page_limit": page_limit, "pages_total": pages_total, "batch_codes": STK_MINS_BATCH_CODES},
            "max_retries": args.max_retries,
        }
        write_stk_mins_by_date(
            path,
            normalized,
            trade_date=trade_date,
            source="daily_incremental_update",
            params=params,
            revision_ledger=revision_ledger,
        )
        written += 1
        total_rows += len(normalized)
        print(f"{args.output_dataset} trade_date={trade_date} written rows={len(normalized)} missing_codes={len(pending)}")
    print(f"{args.output_dataset} update finished dates={len(trade_dates)} written={written} skipped={skipped} rows_written={total_rows}")
    report_complete(args.output_dataset)
    return 0
