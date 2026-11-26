"""The one-minute bar layer: the full-market daily files pulled from stk_mins
(stk_mins_1min_by_date), with the checks a day file must pass.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from autotrade.environment.data.pit import concat_rows

from .client import TuShareClient, frame, load_token, query_paged
from .common import (
    MUTATED_NOT_READY_RETRY_EXIT_CODE,
    ensure_trade_cal_coverage,
    load_sse_open_dates,
    resolve_revision_ledger,
)
from .io import parquet_rows, read_many
from .specs import (
    STK_MINS_API_NAME,
    STK_MINS_BATCH_CODES,
    STK_MINS_BY_DATE_DATASET,
    STK_MINS_FIELDS,
    STK_MINS_FREQ,
    STK_MINS_PAGE_LIMIT,
    STK_MINS_QUOTA_MARKER,
    STK_MINS_REQUIRED_COLUMNS,
)
from .versions import publish, report_complete


def clean_yyyymmdd(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"\d{8}", text) else ""

def load_minute_universe(raw_dir: Path) -> pd.DataFrame:
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
    df = df.sort_values("ts_code").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("minute universe is empty: stock_basic has no ts_code")
    return df


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
        universe = load_minute_universe(raw_dir)
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
