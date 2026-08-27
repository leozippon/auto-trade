#!/usr/bin/env python3
"""TuShare data-quality audit CLI for AutoTrade."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

from . import common as core
from .common import (
    BAK_BASIC_SPEC,
    CORE_MARKET_STATUS_PATH,
    FUNDAMENTAL_RAW_STATUS_PATH,
    BOARD_TRADING_DATASETS,
    BOARD_TRADING_SPECS,
    BOARD_TRADING_STATUS_PATH,
    DAILY_REQUIRED_DATASETS,
    DAILY_SPECS,
    DEFAULT_CN_INDEX_CODES,
    EVENT_FLOW_SPECS,
    EVENT_FLOW_STATUS_PATH,
    FUNDAMENTAL_DATASETS,
    FUNDAMENTAL_SPECS,
    INDEX_WEIGHT_START_DATE,
    INTEGRATED_DOC_REFS,
    INTRADAY_MINUTES_STATUS_PATH,
    MACRO_CONTEXT_STATUS_PATH,
    MACRO_DATASETS,
    MACRO_RETAINED_FLOOR,
    MACRO_SPECS,
    REFERENCE_DATASETS,
    REVISION_SUMMARY_PATH,
    SEMANTIC_DOC_REFS,
    SHARE_FLOAT_ROW_LIMIT,
    STK_MINS_API_NAME,
    STK_MINS_DATASET,
    STK_MINS_PAGE_LIMIT,
    STK_MINS_REQUIRED_COLUMNS,
    TEXT_EVIDENCE_STATUS_PATH,
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
    append_jsonl_unique,
    augment_board_frame,
    augment_event_frame,
    build_revision_event,
    committed_partition_intact,
    date_range_days,
    frame,
    has_pagination_probe,
    intraday_expected_codes_for_day,
    latest_sse_calendar_date,
    load_minute_universe,
    load_sse_open_dates,
    load_stock_codes,
    load_token,
    margin_exchange_column,
    margin_family_missing_exchanges,
    month_end_from_yyyymm,
    month_windows,
    parquet_meta,
    parquet_rows,
    partition_date,
    quarter_periods,
    query_paged,
    read_many,
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
    selected_event_flow_datasets,
    selected_fx_codes,
    selected_index_codes,
    selected_fundamental_datasets,
    selected_intraday_datasets,
    selected_news_sources,
    selected_text_datasets,
    stk_mins_by_date_path,
    validate_stk_mins_by_date_frame,
    yyyymmdd_to_month,
    yyyymmdd_to_quarter,
)

from autotrade.data_quality import build_quality_report, write_quality_report
from autotrade.environment.data.auction import AuctionCorrectionConfig, market_bucket
from autotrade.environment.data.units import column_source_units, dataset_rules_records, rules_for

# Consecutive zero-row partitions at the tail of a zero-tolerant dataset that
# suggest the feed itself stopped publishing (~3 months of trading days).
TRAILING_ZERO_RUN_WARN_PARTITIONS = 60


def audit_trade_date_dataset(raw_dir: Path, spec: TradeDateDataset, expected_dates: set[str], add) -> None:
    files = sorted((raw_dir / spec.api_name).glob("trade_date=*.parquet"))
    file_dates = {partition_date(path): path for path in files}
    row_counts = {trade_date: parquet_rows(path) for trade_date, path in file_dates.items()}
    zero_dates = sorted(d for d, count in row_counts.items() if count == 0)
    nonzero_dates = sorted(d for d, count in row_counts.items() if count > 0)
    missing = sorted(expected_dates - set(file_dates))
    extra = sorted(set(file_dates) - expected_dates)
    exact_limit_dates = sorted(d for d, count in row_counts.items() if count in {5000, 6000, 7000, 8000, 10000} and not has_pagination_probe(file_dates[d]))
    details = {
        "files": len(files),
        "rows": int(sum(row_counts.values())),
        "expected_files": len(expected_dates),
        "missing_expected_files": len(missing),
        "extra_files": len(extra),
        "zero_row_partitions": len(zero_dates),
        "first_file_date": min(file_dates) if file_dates else None,
        "last_file_date": max(file_dates) if file_dates else None,
        "first_nonzero_date": nonzero_dates[0] if nonzero_dates else None,
        "last_nonzero_date": nonzero_dates[-1] if nonzero_dates else None,
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
        "zero_sample": zero_dates[:20],
        "exact_common_limit_row_count_dates": exact_limit_dates[:20],
    }
    has_partition_error = not files or bool(missing) or (bool(zero_dates) and not spec.zero_rows_ok)
    severity = "error" if has_partition_error else "warning" if exact_limit_dates else "info"
    add(severity, f"{spec.api_name}_partitions", f"{spec.api_name} trade-date partition checks", details)

    key_details = audit_partition_keys(files, spec, row_counts)
    has_key_error = any(key_details[name] for name in ("blank_trade_date", "blank_ts_code", "duplicate_key_rows", "filename_trade_date_mismatches", "missing_key_column_files"))
    add("error" if has_key_error else "info", f"{spec.api_name}_keys", f"{spec.api_name} key checks", key_details)
    audit_business_payload(files, spec.api_name, f"{spec.api_name}_payload", add, key_columns=tuple(spec.key_columns), expected_fields=spec_fields_tuple(spec))

def audit_partition_keys(files: list[Path], spec: TradeDateDataset, row_counts: dict[str, int]) -> dict[str, Any]:
    duplicate_rows = 0
    blank_trade_date = 0
    blank_ts_code = 0
    filename_mismatches = 0
    missing_key_column_files: list[str] = []
    key_columns = list(spec.key_columns)
    for path in files:
        if row_counts[partition_date(path)] == 0:
            continue
        schema = pq.ParquetFile(path).schema_arrow.names
        missing = [col for col in key_columns if col not in schema]
        if missing:
            missing_key_column_files.append(str(path))
            continue
        df = pd.read_parquet(path, columns=key_columns)
        if "trade_date" in df:
            trade_dates = df["trade_date"].astype(str).str.strip()
            blank_trade_date += int(df["trade_date"].isna().sum() + (trade_dates == "").sum())
            filename_mismatches += int((trade_dates != partition_date(path)).sum())
        if "ts_code" in df:
            ts_codes = df["ts_code"].astype(str).str.strip()
            blank_ts_code += int(df["ts_code"].isna().sum() + (ts_codes == "").sum())
        duplicate_rows += int(df.duplicated(key_columns).sum())
    return {
        "files_checked": len(files),
        "key_columns": key_columns,
        "blank_trade_date": blank_trade_date,
        "blank_ts_code": blank_ts_code,
        "duplicate_key_rows": duplicate_rows,
        "filename_trade_date_mismatches": filename_mismatches,
        "missing_key_column_files": len(missing_key_column_files),
        "missing_key_column_sample": missing_key_column_files[:10],
    }

def select_revision_sentinel_dates(trade_dates: list[str], sample_size: int, seed: str) -> list[str]:
    if sample_size <= 0 or len(trade_dates) <= sample_size:
        return sorted(trade_dates)
    ordered = sorted(trade_dates)
    offset = sum((index + 1) * ord(char) for index, char in enumerate(seed)) % len(ordered)
    ranked = ordered[offset:] + ordered[:offset]
    return sorted(ranked[:sample_size])

# Trade dates are keyed on the exchange calendar, so "today" is a CN date --
# using the host date would shift the audit window near UTC midnight.
CN_TZ = ZoneInfo("Asia/Shanghai")

def revision_sentinel_spec(dataset: str) -> Any:
    """Resolve a sentinel-probeable spec: daily-tier, board-tier or event-tier
    datasets with a plain trade_date strategy (one request covers the whole
    partition, so a probe compares like-for-like)."""
    if dataset in DAILY_SPECS:
        return DAILY_SPECS[dataset]
    spec = BOARD_TRADING_SPECS.get(dataset) or EVENT_FLOW_SPECS.get(dataset)
    if spec is None or spec.strategy != "trade_date":
        raise RuntimeError(f"revision sentinel supports daily, board and event datasets with a plain trade_date strategy; got {dataset}")
    return spec


# A vendor rewriting a large share of a dataset's history is a different class
# from routine value recalculation (several computed feeds -- chip-distribution
# percentiles, rolling flow sums, daily_basic ratios -- churn chronically and
# stay ledgered warnings). The mass-rewrite signature is STRUCTURAL: dates
# whose probe adds/removes a disproportionate share of the partition's keys
# (the fina_mainbz restatement shape, same 20-key/20% thresholds as the shrink
# guard). Escalate to error when at least this many probed dates AND this
# share of the checked dates are structural rewrites.
SENTINEL_MASSIVE_REVISION_MIN_DATES = 5
SENTINEL_MASSIVE_REVISION_RATIO = 0.25

def sentinel_event_is_structural(event: dict) -> bool:
    changed_keys = int(event.get("added_keys") or 0) + int(event.get("removed_keys") or 0)
    old_rows = int(event.get("old_rows") or 0)
    return changed_keys > 20 and changed_keys > 0.2 * max(old_rows, 1)


def audit_revision_sentinel(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = Path(args.output or REVISION_SUMMARY_PATH)
    if not output.is_absolute():
        output = repo_root / output
    ledger = core.resolve_revision_ledger(raw_dir, args.revision_ledger, repo_root=repo_root)

    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    datasets = list(args.datasets or DAILY_REQUIRED_DATASETS)
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    events: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    page_limit = args.page_limit or TRADE_DATE_PAGE_LIMIT
    total_missing_local = 0
    total_remote_zero = 0
    total_errors = 0
    total_no_effective_checks = 0

    for dataset in datasets:
        spec = revision_sentinel_spec(dataset)
        candidate_dates = [date_value for date_value in trade_dates if max(args.start_date, spec.start_date) <= date_value <= args.end_date]
        sample_dates = select_revision_sentinel_dates(candidate_dates, args.sample_size, f"{args.seed or args.end_date}:{dataset}")
        checked = 0
        missing_local: list[str] = []
        remote_zero: list[str] = []
        errors: list[dict[str, str]] = []
        dataset_events = 0
        structural_dates = 0
        for trade_date in sample_dates:
            path = raw_dir / spec.api_name / f"trade_date={trade_date}.parquet"
            if not path.exists():
                missing_local.append(trade_date)
                continue
            try:
                result, _pages = query_paged(client, spec.api_name, {"trade_date": trade_date}, spec.fields, page_limit)
            except Exception as exc:  # pragma: no cover - defensive runtime path
                errors.append({"trade_date": trade_date, "error": str(exc)})
                continue
            # Stamp the probe exactly like the writer would: board partitions
            # carry derived availability columns (limit_list_d), so a raw API
            # frame would flag them as a false source revision. Daily-tier
            # writers add no derived columns.
            new_df = frame(result)
            if dataset in BOARD_TRADING_SPECS:
                new_df = augment_board_frame(new_df, spec, {"trade_date": trade_date})
            elif dataset in EVENT_FLOW_SPECS:
                new_df = augment_event_frame(new_df, spec)
            if new_df.empty and not spec.zero_rows_ok:
                remote_zero.append(trade_date)
                continue
            checked += 1
            event = build_revision_event(
                dataset=spec.api_name,
                partition=f"trade_date={trade_date}",
                path=path,
                old_df=pd.read_parquet(path),
                new_df=new_df,
                key_columns=list(spec.key_columns),
                source="sentinel_probe",
            )
            if event:
                append_jsonl_unique(ledger, event, key="event_id")
                print("REVISION_ALERT " + json.dumps(
                    {key: event.get(key) for key in ("event_id", "dataset", "partition", "severity", "comparison_issue")},
                    ensure_ascii=False, sort_keys=True,
                ))
                events.append(event)
                dataset_events += 1
                structural_dates += int(sentinel_event_is_structural(event))
        no_effective_checks = int(bool(sample_dates) and checked == 0)
        total_missing_local += len(missing_local)
        total_remote_zero += len(remote_zero)
        total_errors += len(errors)
        total_no_effective_checks += no_effective_checks
        massive_revision = bool(
            checked
            and structural_dates >= SENTINEL_MASSIVE_REVISION_MIN_DATES
            and structural_dates >= SENTINEL_MASSIVE_REVISION_RATIO * checked
        )
        dataset_reports.append({
            "dataset": dataset,
            "candidate_dates": len(candidate_dates),
            "sampled_dates": len(sample_dates),
            "checked_dates": checked,
            "revision_events": dataset_events,
            "structural_revision_dates": structural_dates,
            "massive_revision": massive_revision,
            "missing_local_dates": len(missing_local),
            "remote_zero_dates": len(remote_zero),
            "errors": len(errors),
            "no_effective_checks": no_effective_checks,
            "sample_dates": sample_dates[:20],
            "missing_local_sample": missing_local[:20],
            "remote_zero_sample": remote_zero[:20],
            "error_sample": errors[:10],
        })

    findings = []
    for item in dataset_reports:
        severity = (
            "error"
            if item["errors"] or item["remote_zero_dates"] or item["massive_revision"]
            else "warning"
            if item["revision_events"] or item["missing_local_dates"] or item["no_effective_checks"]
            else "info"
        )
        findings.append(
            {
                "severity": severity,
                "check": f"{item['dataset']}_revision_sentinel",
                "message": f"{item['dataset']} sampled source-revision checks",
                "details": item,
            }
        )
    report = build_quality_report(
        report_type="revision_sentinel",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": datasets,
        },
        findings=findings,
        metadata={
            "revision_ledger": str(ledger),
            "sample_size": args.sample_size,
            "seed": args.seed or args.end_date,
            "totals": {
                "revision_events": len(events),
                "missing_local_dates": total_missing_local,
                "remote_zero_dates": total_remote_zero,
                "api_errors": total_errors,
                "datasets_without_effective_checks": total_no_effective_checks,
            },
            "revision_event_sample": events[:20],
        },
    )
    status = report["status"]
    has_error = status == "error"
    write_quality_report(output, report)
    print(f"revision sentinel status={status} events={len(events)} errors={total_errors} remote_zero={total_remote_zero} no_effective_checks={total_no_effective_checks} output={output} ledger={ledger}")
    return 1 if has_error or (events and args.fail_on_revision) else 0

def audit_intraday_by_date(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = (repo_root / args.output).resolve() if args.output else (repo_root / INTRADAY_MINUTES_STATUS_PATH).resolve()
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    dataset_dir = raw_dir / args.output_dataset
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    paths = {trade_date: stk_mins_by_date_path(raw_dir, args.output_dataset, trade_date) for trade_date in trade_dates}
    missing = [str(path) for trade_date, path in paths.items() if not path.exists()]
    files = [path for path in paths.values() if path.exists()]
    meta_files = [path.with_suffix(path.suffix + ".meta.json") for path in files]
    missing_meta = [str(path) for path in meta_files if not path.exists()]
    all_meta = sorted(dataset_dir.glob("*.parquet.meta.json"))
    expected_meta = {str(path) for path in meta_files}
    orphan_meta = [str(path) for path in all_meta if str(path) not in expected_meta and not Path(str(path).removesuffix(".meta.json")).exists()]
    row_counts = {path.name: parquet_rows(path) for path in files}
    zero_files = [str(path) for path in files if row_counts.get(path.name, 0) == 0]
    schema_missing: list[str] = []
    for path in files:
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        if not set(STK_MINS_REQUIRED_COLUMNS).issubset(schema):
            schema_missing.append(str(path))
    add("error" if missing or missing_meta or orphan_meta or zero_files or schema_missing else "info", f"{args.output_dataset}_inventory", "date-organized intraday minute inventory", {
        "dataset_dir": str(dataset_dir),
        "expected_trade_dates": len(trade_dates),
        "files": len(files),
        "missing_files": len(missing),
        "missing_meta": len(missing_meta),
        "orphan_meta": len(orphan_meta),
        "schema_missing_required_columns": len(schema_missing),
        "rows": int(sum(row_counts.values())),
        "zero_row_files": len(zero_files),
        "missing_sample": missing[:20],
        "missing_meta_sample": missing_meta[:20],
        "orphan_meta_sample": orphan_meta[:20],
        "zero_file_sample": zero_files[:20],
        "schema_missing_sample": schema_missing[:10],
    })

    # The by-date lake gets the same commit-pair coverage as every raw dataset.
    audit_integrated_filesystem(raw_dir, [args.output_dataset], add)

    # Deep-check the NEWEST trade dates: those are the partitions the daily
    # pipeline just wrote. The head of the list is the window's first days,
    # which never change and would be re-validated forever.
    deep_limit = max(0, args.sample_limit)
    deep_paths = files if args.full_scan else (files[len(files) - deep_limit:] if deep_limit else [])
    bad_days: list[dict[str, Any]] = []
    for path in deep_paths:
        trade_date = path.stem.split("=", 1)[-1]
        df = pd.read_parquet(path)
        expected_codes = intraday_expected_codes_for_day(raw_dir, args, trade_date)
        ok, details = validate_stk_mins_by_date_frame(
            df,
            trade_date,
            expected_codes=expected_codes,
            min_rows=args.min_rows_per_day,
            allow_missing_codes=args.allow_missing_codes,
        )
        if not ok:
            bad_days.append(details)
    add("warning" if bad_days else "info", f"{args.output_dataset}_deep_checks", "date-organized intraday minute row/key/PIT/time checks", {
        "full_scan": bool(args.full_scan),
        "files_checked": len(deep_paths),
        "bad_days": len(bad_days),
        "bad_day_sample": bad_days[:10],
        "expected_codes_source": args.expected_codes_source,
        "min_rows_per_day": args.min_rows_per_day,
        "allow_missing_codes": args.allow_missing_codes,
    })

    report = build_quality_report(
        report_type="intraday_minutes",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": [args.output_dataset],
            "expected_codes_source": args.expected_codes_source,
        },
        findings=findings,
        metadata={
            "unit_rules": {
                args.output_dataset: {
                    "source": f"derived from {STK_MINS_DATASET} or daily incremental {STK_MINS_API_NAME}",
                    "partition": "one full-market parquet per trade_date",
                    **column_source_units("intraday_1min.parquet"),
                    "available_at": "bar close time from trade_time",
                }
            },
            "conclusions": [
                "The date-organized minute store is the preferred research/live-update layout for PIT daily replay.",
                "The stock-year source store remains the historical download and traceability layer.",
                "Rows must still be filtered by available_at <= decision_time inside PIT snapshot construction.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"intraday by-date audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def numeric_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce")
    right = pd.to_numeric(denominator, errors="coerce")
    return left.where(right.ne(0)) / right.where(right.ne(0))

def grouped_ratio_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int | None]]:
    if df.empty:
        return {}
    result: dict[str, dict[str, float | int | None]] = {}
    for bucket, group in df.groupby("bucket", dropna=False):
        item: dict[str, float | int | None] = {"rows": int(len(group))}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                item[column] = None
                continue
            item[f"{column}_median"] = float(values.median())
            item[f"{column}_p10"] = float(values.quantile(0.1))
            item[f"{column}_p90"] = float(values.quantile(0.9))
        result[str(bucket)] = item
    return result

def audit_auction_alignment(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = (repo_root / (args.output or "results/data_quality/process/auction_alignment_status.json")).resolve()
    trade_dates = load_sse_open_dates(raw_dir, args.start_date, args.end_date)
    if args.max_trade_dates > 0:
        trade_dates = trade_dates[-args.max_trade_dates :]
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    client = TuShareClient(load_token(repo_root), args.min_interval_seconds, args.timeout_seconds)
    correction = AuctionCorrectionConfig()
    auction_day_stats: list[dict[str, Any]] = []
    daily_day_stats: list[dict[str, Any]] = []
    missing_minute_dates: list[str] = []
    missing_daily_dates: list[str] = []

    for trade_date in trade_dates:
        minute_path = stk_mins_by_date_path(raw_dir, args.output_dataset, trade_date)
        if not minute_path.exists():
            missing_minute_dates.append(trade_date)
            continue
        minutes = pd.read_parquet(minute_path, columns=["ts_code", "trade_time", "vol", "amount"])
        hhmm = minutes["trade_time"].astype(str).str.slice(11, 16)
        open_bar = minutes[hhmm.eq("09:30")].copy()
        auction = api_frame(client, "stk_auction", {"trade_date": trade_date}, "ts_code,trade_date,vol,amount")
        merged = open_bar.merge(auction, on="ts_code", suffixes=("_minute", "_auction"))
        merged["bucket"] = merged["ts_code"].map(market_bucket)
        merged["vol_ratio"] = numeric_ratio(merged["vol_minute"], merged["vol_auction"])
        merged["amount_ratio"] = numeric_ratio(merged["amount_minute"], merged["amount_auction"])
        merged["vol_ratio_after_factor"] = merged["vol_ratio"] * merged["bucket"].map(
            lambda bucket: correction.volume_factors.get(bucket, 1.0)
        )
        merged["amount_ratio_after_factor"] = merged["amount_ratio"] * merged["bucket"].map(
            lambda bucket: correction.amount_factors.get(bucket, 1.0)
        )
        auction_day_stats.append({
            "trade_date": trade_date,
            "minute_open_rows": int(len(open_bar)),
            "stk_auction_rows": int(len(auction)),
            "matched_rows": int(len(merged)),
            "bucket_stats": grouped_ratio_stats(merged, ["vol_ratio", "amount_ratio", "vol_ratio_after_factor", "amount_ratio_after_factor"]),
        })

        daily_path = raw_dir / "daily" / f"trade_date={trade_date}.parquet"
        if not daily_path.exists():
            missing_daily_dates.append(trade_date)
            continue
        daily = pd.read_parquet(daily_path, columns=["ts_code", "vol", "amount"])
        minute_sum = minutes.groupby("ts_code", as_index=False)[["vol", "amount"]].sum()
        daily_merge = minute_sum.merge(daily, on="ts_code", suffixes=("_minute_sum", "_daily"))
        daily_merge["bucket"] = daily_merge["ts_code"].map(market_bucket)
        daily_merge["minute_to_daily_vol_ratio"] = numeric_ratio(daily_merge["vol_minute_sum"], daily_merge["vol_daily"])
        daily_merge["minute_to_daily_amount_ratio"] = numeric_ratio(daily_merge["amount_minute_sum"], daily_merge["amount_daily"])
        daily_day_stats.append({
            "trade_date": trade_date,
            "matched_rows": int(len(daily_merge)),
            "bucket_stats": grouped_ratio_stats(daily_merge, ["minute_to_daily_vol_ratio", "minute_to_daily_amount_ratio"]),
        })

    add("error" if missing_minute_dates else "info", "auction_alignment_inputs", "input partition availability for auction alignment", {
        "trade_dates_checked": trade_dates,
        "missing_minute_dates": missing_minute_dates,
        "missing_daily_dates": missing_daily_dates,
    })
    add("warning" if not auction_day_stats else "info", "minute_0930_vs_stk_auction", "09:30 minute bar against live stk_auction ratios by market bucket", {
        "days": auction_day_stats,
        "expected_pattern": "SH/BJ buckets should stay near 1.0; historical SZ 09:30 minute bars are adjusted in PIT snapshot construction before comparing with live stk_auction.",
        "correction_factors": {
            "volume": {**correction.volume_factors, "others": 1.0},
            "amount": {**correction.amount_factors, "others": 1.0},
        },
    })
    add("warning" if not daily_day_stats else "info", "minute_sum_vs_daily_units", "full-day minute sums against daily unit ratios", {
        "days": daily_day_stats,
        "expected_ratios": {"vol": "minute shares / daily hands ~= 100", "amount": "minute CNY / daily thousand CNY ~= 1000"},
    })

    report = build_quality_report(
        report_type="auction_alignment",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": [args.output_dataset, "stk_auction", "daily"],
            "trade_dates_checked": trade_dates,
            "output_dataset": args.output_dataset,
        },
        findings=findings,
        metadata={
            "conclusions": [
                "Raw minute files are not modified by this audit.",
                "Historical 09:30 minute auction bars should be corrected in the Environment snapshot layer when they are used as a proxy for live stk_auction.",
                "Full-day minute sums should still align with daily units after the documented share/hand and CNY/thousand-CNY conversions.",
            ]
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"auction alignment audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def expected_stk_mins_paths(raw_dir: Path, args: argparse.Namespace) -> set[Path]:
    universe_args = argparse.Namespace(codes=getattr(args, "intraday_codes", None), max_codes=getattr(args, "intraday_max_codes", None))
    universe = load_minute_universe(raw_dir, universe_args)
    expected: set[Path] = set()
    for _, row in universe.iterrows():
        ts_code = str(row["ts_code"])
        for year, _, _ in active_year_windows(row, args.intraday_start_date, args.intraday_end_date):
            expected.add(raw_dir / STK_MINS_DATASET / f"ts_code={safe_partition_value(ts_code)}" / f"year={year}.parquet")
    return expected

def audit_stk_mins_completeness(raw_dir: Path, args: argparse.Namespace, add) -> None:
    all_files = sorted((raw_dir / STK_MINS_DATASET).glob("ts_code=*/year=*.parquet"))
    all_meta_files = sorted((raw_dir / STK_MINS_DATASET).glob("ts_code=*/year=*.parquet.meta.json"))
    expected = expected_stk_mins_paths(raw_dir, args)
    scoped = bool(getattr(args, "intraday_codes", None) or getattr(args, "intraday_max_codes", None))
    if scoped:
        expected_parent_dirs = {path.parent.resolve() for path in expected}
        files = [path for path in all_files if path.parent.resolve() in expected_parent_dirs]
        meta_files = [path for path in all_meta_files if path.parent.resolve() in expected_parent_dirs]
    else:
        files = all_files
        meta_files = all_meta_files
    file_set = {path.resolve() for path in files}
    expected_set = {path.resolve() for path in expected}
    missing = sorted(str(path) for path in expected_set - file_set)
    extra = sorted(str(path) for path in file_set - expected_set)
    parquet_meta = {str(path.with_suffix(path.suffix + ".meta.json")) for path in files}
    meta_set = {str(path) for path in meta_files}
    missing_meta = sorted(parquet_meta - meta_set)
    orphan_meta = sorted(meta_set - parquet_meta)
    row_counts = {str(path): parquet_rows(path) for path in files}
    zero_files = sorted(path for path, rows in row_counts.items() if rows == 0)
    exact_limit_files = sorted(
        path
        for path, rows in row_counts.items()
        if rows in {STK_MINS_PAGE_LIMIT, 5000, 6000, 7000, 8000, 10000} and not has_pagination_probe(Path(path))
    )
    schema_missing: list[str] = []
    for path in files:
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        if not set(STK_MINS_REQUIRED_COLUMNS).issubset(schema):
            schema_missing.append(str(path))
    has_error = bool(missing or missing_meta or orphan_meta or schema_missing)
    sample_seed = str(getattr(args, "intraday_end_date", None) or args.intraday_start_date)
    add("error" if has_error else "warning" if zero_files or exact_limit_files else "info", f"{STK_MINS_DATASET}_partitions", "stk_mins 1min stock-year partition inventory", {
        "files": len(files),
        "expected_files": len(expected),
        "missing_expected_files": len(missing),
        "extra_files": len(extra),
        "meta_files": len(meta_files),
        "rows": int(sum(row_counts.values())),
        "zero_row_partitions": len(zero_files),
        "exact_common_limit_row_count_partitions": len(exact_limit_files),
        "missing_meta": len(missing_meta),
        "orphan_meta": len(orphan_meta),
        "schema_missing_required_columns": len(schema_missing),
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
        "zero_sample": zero_files[:20],
        "exact_limit_sample": exact_limit_files[:20],
        "schema_missing_sample": schema_missing[:10],
        "unit_rules": {**column_source_units("intraday_1min.parquet"), "official_page_limit": STK_MINS_PAGE_LIMIT, "doc_ref": INTEGRATED_DOC_REFS[STK_MINS_DATASET]},
    })
    audit_stk_mins_sample(files, row_counts, args.sample_limit, sample_seed, add)

def audit_stk_mins_sample(files: list[Path], row_counts: dict[str, int], sample_limit: int, seed: str, add) -> None:
    # Deterministic seed-driven rotation (the revision-sentinel pattern): a
    # fixed head slice re-read the same lowest ts_codes forever, so the rest of
    # the store was never sampled. The seed moves with the audit end date, so
    # coverage rotates daily while any given day stays reproducible.
    candidates = [path for path in files if row_counts[str(path)] > 0]
    ranked = sorted(candidates)
    if ranked:
        offset = sum((index + 1) * ord(char) for index, char in enumerate(seed)) % len(ranked)
        ranked = ranked[offset:] + ranked[:offset]
    sample = ranked[: max(0, sample_limit)]
    duplicate_key_rows = 0
    unparseable_trade_time_rows = 0
    unparseable_available_at_rows = 0
    missing_0930_files: list[str] = []
    missing_1500_files: list[str] = []
    code_partition_mismatch = 0
    year_partition_mismatch = 0
    for path in sample:
        df = pd.read_parquet(path, columns=["ts_code", "trade_time", "trade_date", "available_at"])
        duplicate_key_rows += int(df.duplicated(["ts_code", "trade_time"]).sum())
        trade_time = df["trade_time"].astype(str).str.strip()
        parsed_trade = pd.to_datetime(trade_time, errors="coerce")
        unparseable_trade_time_rows += int(parsed_trade.isna().sum())
        available = df["available_at"].astype(str).str.strip()
        parsed_available = pd.to_datetime(available[available.ne("")], errors="coerce", utc=True, format="mixed")
        unparseable_available_at_rows += int(parsed_available.isna().sum())
        times = set(trade_time.str.extract(r"(\d{2}:\d{2})", expand=False).dropna().tolist())
        if "09:30" not in times:
            missing_0930_files.append(str(path))
        if "15:00" not in times:
            missing_1500_files.append(str(path))
        expected_code = path.parent.name.split("=", 1)[1]
        expected_year = path.stem.split("=", 1)[1]
        code_partition_mismatch += int((df["ts_code"].astype(str) != expected_code).sum())
        year_partition_mismatch += int((df["trade_date"].astype(str).str[:4] != expected_year).sum())
    has_issue = any([duplicate_key_rows, unparseable_trade_time_rows, unparseable_available_at_rows, missing_0930_files, missing_1500_files, code_partition_mismatch, year_partition_mismatch])
    add("warning" if has_issue else "info", f"{STK_MINS_DATASET}_sample_keys", "stk_mins 1min sampled key/PIT/auction-bar checks", {
        "files_sampled": len(sample),
        "duplicate_key_rows": duplicate_key_rows,
        "unparseable_trade_time_rows": unparseable_trade_time_rows,
        "unparseable_available_at_rows": unparseable_available_at_rows,
        "missing_0930_files": len(missing_0930_files),
        "missing_1500_files": len(missing_1500_files),
        "code_partition_mismatch_rows": code_partition_mismatch,
        "year_partition_mismatch_rows": year_partition_mismatch,
        "missing_0930_sample": missing_0930_files[:10],
        "missing_1500_sample": missing_1500_files[:10],
    })

def fundamental_partition_value(path: Path, prefix: str) -> str:
    stem = path.stem
    expected = f"{prefix}="
    return stem[len(expected):] if stem.startswith(expected) else ""

def audit_fundamental_dataset(raw_dir: Path, spec: FundamentalDataset, expected: set[str], prefix: str, add) -> dict[str, int]:
    files = sorted((raw_dir / spec.api_name).glob(f"{prefix}=*.parquet"))
    file_values = {fundamental_partition_value(path, prefix): path for path in files}
    rows_by_path = {path: parquet_rows(path) for path in files}
    row_counts = {value: rows_by_path[path] for value, path in file_values.items()}
    zero_values = sorted(value for value, count in row_counts.items() if count == 0)
    nonzero_values = sorted(value for value, count in row_counts.items() if count > 0)
    missing = sorted(expected - set(file_values))
    extra = sorted(set(file_values) - expected)
    # 6400/9000 are measured historical caps found pinned in the lake
    # (2026-08: cashflow_vip at exactly 6,400 for 23 periods; income_vip at
    # exactly 9,000 for 6 periods from the legacy limit-10000 era).
    exact_limit_values = sorted(value for value, count in row_counts.items() if count in {5000, 6000, 6400, 7000, 8000, 9000, 10000})
    details = {
        "strategy": spec.strategy,
        "partition_prefix": prefix,
        "files": len(files),
        "rows": int(sum(row_counts.values())),
        "expected_files": len(expected),
        "missing_expected_files": len(missing),
        "extra_files": len(extra),
        "zero_row_partitions": len(zero_values),
        "nonzero_partitions": len(nonzero_values),
        "first_partition": min(file_values) if file_values else None,
        "last_partition": max(file_values) if file_values else None,
        "first_nonzero_partition": nonzero_values[0] if nonzero_values else None,
        "last_nonzero_partition": nonzero_values[-1] if nonzero_values else None,
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
        "zero_sample": zero_values[:20],
        "exact_common_limit_row_count_partitions": len(exact_limit_values),
        "exact_limit_sample": exact_limit_values[:20],
    }
    add("error" if not files or missing else "warning" if exact_limit_values else "info", f"{spec.api_name}_partitions", f"{spec.api_name} fundamental partition checks", details)
    key_details = audit_fundamental_keys(files, spec, rows_by_path)
    has_key_error = key_details["blank_ts_code"] or key_details["missing_key_column_files"]
    has_key_warning = key_details["duplicate_key_rows"] or key_details["duplicate_full_rows"]
    severity = "error" if has_key_error else "warning" if has_key_warning else "info"
    add(severity, f"{spec.api_name}_keys", f"{spec.api_name} fundamental key checks", key_details)
    # Fundamentals request all vendor fields (spec.fields is empty), so a file
    # stripped to business keys must not pass silently. disclosure_date is the
    # exact-key exception: its five source fields are all business keys.
    audit_business_payload(
        files,
        spec.api_name,
        f"{spec.api_name}_payload",
        add,
        key_columns=tuple(spec.key_columns),
        require_business_columns=spec.api_name != "disclosure_date",
    )
    return row_counts

def audit_fundamental_keys(files: list[Path], spec: FundamentalDataset, rows_by_path: dict[Path, int]) -> dict[str, Any]:
    duplicate_key_rows = 0
    duplicate_full_rows = 0
    blank_ts_code = 0
    blank_date_fields: dict[str, int] = {}
    missing_key_column_files: list[str] = []
    date_fields = [field for field in ("ann_date", "f_ann_date", "end_date", "actual_date", "pre_date") if field in spec.key_columns]
    for path in files:
        rows = rows_by_path[path]
        if rows == 0:
            continue
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow.names
        missing = [col for col in spec.key_columns if col not in schema]
        if missing:
            missing_key_column_files.append(str(path))
            continue
        key_df = pd.read_parquet(path, columns=list(spec.key_columns))
        if "ts_code" in key_df:
            ts_codes = key_df["ts_code"].astype(str).str.strip()
            blank_ts_code += int(key_df["ts_code"].isna().sum() + (ts_codes == "").sum())
        for field in date_fields:
            values = key_df[field].astype(str).str.strip()
            blank_date_fields[field] = blank_date_fields.get(field, 0) + int(key_df[field].isna().sum() + (values == "").sum())
        duplicate_key_rows += int(key_df.duplicated(list(spec.key_columns)).sum())
        if rows <= 20000:
            full_df = pd.read_parquet(path)
            duplicate_full_rows += int(full_df.duplicated().sum())
    return {
        "files_checked": len(files),
        "key_columns": list(spec.key_columns),
        "blank_ts_code": blank_ts_code,
        "blank_date_fields": blank_date_fields,
        "duplicate_key_rows": duplicate_key_rows,
        "duplicate_full_rows": duplicate_full_rows,
        "missing_key_column_files": len(missing_key_column_files),
        "missing_key_column_sample": missing_key_column_files[:10],
    }

def blank_count(series: pd.Series) -> int:
    return int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())

def spec_fields_tuple(spec: Any) -> tuple[str, ...]:
    """The request-contract column list; empty for specs that request all vendor fields."""
    return tuple(field for field in str(getattr(spec, "fields", "")).split(",") if field)

def newest_nonzero_partition(files: list[Path]) -> Path | None:
    # Newest by date-encoded file name; today's partition may legitimately
    # still be empty for zero-tolerant feeds, so walk back to data.
    for path in sorted(files, key=lambda item: item.name, reverse=True):
        if parquet_rows(path) > 0:
            return path
    return None

def audit_business_payload(files: list[Path], api_name: str, check_name: str, add, *, key_columns: tuple, expected_fields: tuple = (), require_business_columns: bool = False) -> None:
    """Key/PIT audits read key columns only, so a partition carrying nothing
    but keys and availability stamps used to pass every raw check. On the
    newest non-empty partition, verify the requested source fields all arrived
    (missing column = warning; individual sparse columns are routine vendor
    behavior) and that the non-key payload is not entirely blank (= error).

    A file stripped down to key columns is hollow too: for spec'd families
    that means every expected business field is absent; for families that
    request all vendor fields (fundamentals) the caller declares the payload
    mandatory via require_business_columns. Datasets whose whole contract is
    the key set (suspend_d, block_trade, top_list, cn_schedule) pass both
    gates by construction."""
    path = newest_nonzero_partition(files)
    if path is None:
        return  # empty/missing datasets are the inventory checks' finding
    df = pd.read_parquet(path)
    excluded = set(key_columns) | {"available_at", "available_at_rule"}
    business = [col for col in df.columns if col not in excluded]
    all_null = [col for col in business if blank_count(df[col]) == len(df)]
    missing_fields = [col for col in expected_fields if col not in df.columns]
    expected_business = [col for col in expected_fields if col not in excluded]
    hollow = bool(len(df)) and (
        (bool(business) and len(all_null) == len(business))
        or (not business and require_business_columns)
        or (bool(expected_business) and all(col in missing_fields for col in expected_business))
    )
    severity = "error" if hollow else "warning" if missing_fields else "info"
    add(severity, check_name, f"{api_name} business payload checks", {
        "checked_partition": str(path),
        "rows": int(len(df)),
        "business_columns": len(business),
        "all_null_business_columns": all_null[:20],
        "missing_expected_fields": missing_fields,
        "business_payload_empty": hollow,
    })

def audit_full_market_coverage(raw_dir: Path, api_names: list[str], add) -> None:
    """Inventory and key checks cannot see a vendor dropping a batch of
    stocks: the partition stays plausible and paging never hits a cap. Compare
    each full-market dataset's per-day universe against the daily table and
    warn when a day falls under the measured floor."""
    datasets = [api for api in api_names if api in core.FULL_MARKET_COVERAGE_DATASETS]
    days_by_dataset = {
        api: {partition_date(path): path for path in sorted((raw_dir / api).glob("trade_date=*.parquet"))}
        for api in datasets
    }
    low: dict[str, list[dict[str, Any]]] = {api: [] for api in datasets}
    checked: dict[str, int] = {api: 0 for api in datasets}
    all_days = sorted(set().union(*days_by_dataset.values())) if days_by_dataset else []
    for day in all_days:
        base_codes: set[str] | None = None
        for api in datasets:
            path = days_by_dataset[api].get(day)
            if path is None or parquet_rows(path) == 0:
                continue
            if "ts_code" not in pq.ParquetFile(path).schema_arrow.names:
                continue  # a missing key column is the key checks' error
            if base_codes is None:
                daily_path = raw_dir / "daily" / f"trade_date={day}.parquet"
                base_codes = (
                    set(pd.read_parquet(daily_path, columns=["ts_code"])["ts_code"].astype(str))
                    if daily_path.exists() and parquet_rows(daily_path) > 0
                    else set()
                )
            if not base_codes:
                continue  # a missing daily partition is the daily inventory's error
            codes = set(pd.read_parquet(path, columns=["ts_code"])["ts_code"].astype(str))
            ratio = len(codes & base_codes) / len(base_codes)
            checked[api] += 1
            if ratio < core.FULL_MARKET_COVERAGE_MIN_RATIO:
                low[api].append({
                    "trade_date": day,
                    "dataset_codes": len(codes),
                    "daily_codes": len(base_codes),
                    "coverage": round(ratio, 4),
                    "missing_sample": sorted(base_codes - codes)[:10],
                })
    for api in datasets:
        days = sorted(low[api], key=lambda item: item["coverage"])
        add("warning" if days else "info", f"{api}_daily_coverage", f"{api} per-day stock coverage against daily", {
            "days_checked": checked[api],
            "min_coverage_ratio": core.FULL_MARKET_COVERAGE_MIN_RATIO,
            "days_below_threshold": len(days),
            "low_coverage_sample": days[:20],
        })

def audit_stock_basic(raw_dir: Path, add) -> pd.DataFrame:
    files = sorted((raw_dir / "stock_basic").glob("list_status=*.parquet"))
    # The table is a union of one file per list_status; a missing slice
    # (e.g. every delisted stock) still reads as a plausible universe.
    missing_slices = sorted(
        status for status in ("L", "D", "P")
        if not (raw_dir / "stock_basic" / f"list_status={status}.parquet").exists()
    )
    df = read_many(files)
    details = {"files": len(files), "rows": len(df), "missing_list_status_slices": missing_slices}
    if df.empty:
        add("error", "stock_basic", "stock_basic is empty or missing", details)
        return df
    details.update({
        "unique_ts_code": int(df["ts_code"].nunique()),
        "duplicate_ts_code_rows": int(df.duplicated(["ts_code"]).sum()),
        "blank_required": {col: blank_count(df[col]) for col in ["ts_code", "symbol", "name", "list_status", "list_date"]},
        "status_counts": df["list_status"].value_counts(dropna=False).to_dict(),
    })
    has_error = details["duplicate_ts_code_rows"] or any(details["blank_required"].values()) or missing_slices
    add("error" if has_error else "info", "stock_basic", "stock_basic key and required-field checks", details)
    return df

def audit_stock_company(raw_dir: Path, stock_basic: pd.DataFrame, add) -> None:
    df = read_many(sorted((raw_dir / "stock_company").glob("exchange=*.parquet")))
    if df.empty:
        add("warning", "stock_company", "stock_company is empty or missing")
        return
    # One file per exchange; a missing slice silently shrinks the union.
    missing_slices = sorted(
        exchange for exchange in ("SSE", "SZSE", "BSE")
        if not (raw_dir / "stock_company" / f"exchange={exchange}.parquet").exists()
    )
    details = {
        "rows": len(df), "unique_ts_code": int(df["ts_code"].nunique()),
        "duplicate_ts_code_rows": int(df.duplicated(["ts_code"]).sum()),
        "blank_ts_code": blank_count(df["ts_code"]), "blank_com_name": blank_count(df["com_name"]),
        "missing_exchange_slices": missing_slices,
    }
    add("warning" if details["blank_com_name"] or missing_slices else "info", "stock_company", "stock_company key and name checks", details)
    if not stock_basic.empty:
        basic_codes = set(stock_basic["ts_code"].dropna().astype(str))
        company_codes = set(df["ts_code"].dropna().astype(str))
        # Structural coverage difference (company table lags/leads by design);
        # the anomaly judgment lives in stock_universe_semantics. Info keeps a
        # healthy day at status ok.
        add("info", "stock_company_vs_stock_basic", "stock_company and stock_basic coverage differs", {
            "stock_basic_missing_in_company": len(basic_codes - company_codes),
            "stock_company_missing_in_basic": len(company_codes - basic_codes),
            "stock_basic_missing_sample": sorted(basic_codes - company_codes)[:20],
            "stock_company_missing_sample": sorted(company_codes - basic_codes)[:20],
        })

def audit_trade_cal(raw_dir: Path, add) -> set[str]:
    calendars: dict[str, pd.DataFrame] = {}
    for exchange in ("SSE", "SZSE", "BSE"):
        files = sorted((raw_dir / "trade_cal" / f"exchange={exchange}").glob("year=*.parquet"))
        df = read_many(files)
        calendars[exchange] = df
        add("warning" if exchange == "BSE" and df.empty else "info", f"trade_cal_{exchange}", f"{exchange} trade calendar checks", {
            "files": len(files), "rows": len(df),
            "open_days": int((df["is_open"].astype(str) == "1").sum()) if not df.empty else 0,
            "duplicate_cal_date_rows": int(df.duplicated(["cal_date"]).sum()) if not df.empty else 0,
        })
    sse_open = set(calendars["SSE"].loc[calendars["SSE"]["is_open"].astype(str) == "1", "cal_date"].astype(str)) if not calendars["SSE"].empty else set()
    szse_open = set(calendars["SZSE"].loc[calendars["SZSE"]["is_open"].astype(str) == "1", "cal_date"].astype(str)) if not calendars["SZSE"].empty else set()
    add("error" if sse_open != szse_open else "info", "trade_cal_sse_szse", "SSE/SZSE open-day alignment", {"sse_not_szse": len(sse_open - szse_open), "szse_not_sse": len(szse_open - sse_open)})
    return sse_open

def audit_bak_basic(raw_dir: Path, sse_open: set[str], end_date: str, add) -> None:
    files = sorted((raw_dir / "bak_basic").glob("trade_date=*.parquet"))
    rows = {path.stem.split("=", 1)[1]: parquet_rows(path) for path in files}
    expected = {d for d in sse_open if "20160101" <= d <= end_date}
    scoped_rows = {d: count for d, count in rows.items() if d <= end_date}
    missing_dates = sorted(expected - set(scoped_rows))
    extra_dates = sorted(set(scoped_rows) - expected)
    zero_dates = sorted(d for d in expected if scoped_rows.get(d, 0) == 0)
    nonzero_dates = sorted(d for d in expected if scoped_rows.get(d, 0) > 0)
    details = {
        "files": len(files), "rows": int(sum(rows.values())), "end_date": end_date,
        "missing_expected_files": len(missing_dates), "extra_files": len(extra_dates),
        "zero_row_partitions": len(zero_dates), "first_nonzero_date": nonzero_dates[0] if nonzero_dates else None,
        "last_nonzero_date": nonzero_dates[-1] if nonzero_dates else None,
        "zero_after_first_nonzero": len([d for d in zero_dates if nonzero_dates and d > nonzero_dates[0]]),
        "missing_sample": missing_dates[:20], "extra_sample": extra_dates[:20], "zero_sample": zero_dates[:20],
    }
    severity = "error" if details["missing_expected_files"] else "warning" if zero_dates else "info"
    add(severity, "bak_basic_partitions", "bak_basic partition and source-empty checks", details)
    key_df = read_many(files, columns=["trade_date", "ts_code"])
    add("error" if key_df.duplicated(["trade_date", "ts_code"]).any() else "info", "bak_basic_keys", "bak_basic key checks", {
        "blank_trade_date": blank_count(key_df["trade_date"]), "blank_ts_code": blank_count(key_df["ts_code"]),
        "duplicate_trade_date_ts_code_rows": int(key_df.duplicated(["trade_date", "ts_code"]).sum()),
    })
    audit_business_payload(files, "bak_basic", "bak_basic_payload", add, key_columns=tuple(BAK_BASIC_SPEC.key_columns), expected_fields=spec_fields_tuple(BAK_BASIC_SPEC))

def audit_namechange(raw_dir: Path, stock_basic: pd.DataFrame, add) -> None:
    path = raw_dir / "namechange" / "namechange.parquet"
    if not path.exists():
        add("error", "namechange", "namechange final table is missing")
        return
    df = pd.read_parquet(path)
    details = {
        "rows": len(df), "unique_ts_code": int(df["ts_code"].nunique()),
        "blank_ts_code": blank_count(df["ts_code"]), "duplicate_full_rows": int(df.duplicated().sum()),
        "start_date_min": str(df["start_date"].dropna().astype(str).replace("", pd.NA).dropna().min()),
        "start_date_max": str(df["start_date"].dropna().astype(str).replace("", pd.NA).dropna().max()),
    }
    # The table unions one per-code pull over the whole stock_basic universe,
    # so a batch of codes whose pull returned nothing vanishes silently.
    # Nearly every code carries at least its listing name (measured 2026-08:
    # 2 of 5,878 codes absent), so >1% absent is a dropped batch.
    missing_codes: set[str] = set()
    if not stock_basic.empty:
        basic_codes = set(stock_basic["ts_code"].dropna().astype(str))
        missing_codes = basic_codes - set(df["ts_code"].dropna().astype(str))
        details.update({
            "stock_basic_codes": len(basic_codes),
            "stock_basic_codes_without_namechange": len(missing_codes),
            "missing_ratio": round(len(missing_codes) / len(basic_codes), 4) if basic_codes else 0.0,
            "missing_code_sample": sorted(missing_codes)[:20],
        })
    has_error = details["blank_ts_code"] or details["duplicate_full_rows"]
    has_warning = bool(details.get("missing_ratio", 0.0) > 0.01)
    add("error" if has_error else "warning" if has_warning else "info", "namechange", "final namechange table checks", details)

def audit_ths_membership(raw_dir: Path, add) -> None:
    catalog_path = raw_dir / "ths_index" / "catalog.parquet"
    if not catalog_path.exists():
        add("error", "ths_member_coverage", "ths_index catalog is missing", {})
        return
    catalog = pd.read_parquet(catalog_path)
    # The downloader fetches membership for concept (N) and industry (I)
    # indices only; other types are intentionally skipped, so the coverage
    # expectation must use the same filter.
    expected = set(catalog.loc[catalog["type"].astype(str).isin(["N", "I"]), "ts_code"].dropna().astype(str))
    member_files = {path.stem.split("=", 1)[1] for path in (raw_dir / "ths_member").glob("ts_code=*.parquet")}
    missing = sorted(expected - member_files)
    stale = sorted(member_files - expected)
    add("error" if missing else "info", "ths_member_coverage", "ths_member files cover the N/I catalog", {
        "catalog_n_i_codes": len(expected),
        "member_files": len(member_files),
        "missing_member_files": len(missing),
        "missing_sample": missing[:20],
        "member_files_not_in_catalog": len(stale),
        "stale_sample": stale[:10],
    })

def audit_index_classify(raw_dir: Path, add) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for src in ("SW2014", "SW2021"):
        path = raw_dir / "index_classify" / f"src={src}.parquet"
        if not path.exists():
            add("error", "index_classify", f"index_classify src={src} is missing")
            frames[src] = pd.DataFrame()
            continue
        df = pd.read_parquet(path)
        details = {"rows": len(df), "level_counts": df["level"].value_counts(dropna=False).to_dict(), "duplicate_index_code_rows": int(df.duplicated(["index_code"]).sum())}
        add("error" if details["duplicate_index_code_rows"] else "info", "index_classify", f"{src} industry classification checks", details)
        frames[src] = df
    return frames

def audit_index_member_all(raw_dir: Path, classify: pd.DataFrame, stock_basic: pd.DataFrame, add) -> None:
    files = sorted((raw_dir / "index_member_all").glob("l1_code=*.parquet"))
    df = read_many(files)
    l1_codes = set(classify.loc[classify["level"].astype(str) == "L1", "index_code"].astype(str)) if not classify.empty else set()
    file_codes = set(path.stem.split("=", 1)[1] for path in files)
    missing_in_basic: set[str] = set()
    if not df.empty and not stock_basic.empty:
        missing_in_basic = set(df["ts_code"].dropna().astype(str)) - set(stock_basic["ts_code"].dropna().astype(str))
    rows_with_out_date = 0
    if not df.empty and "out_date" in df.columns:
        rows_with_out_date = int(df["out_date"].fillna("").astype(str).replace("None", "").ne("").sum())
    details = {
        "files": len(files), "rows": len(df), "missing_l1_partitions": len(l1_codes - file_codes),
        "extra_l1_partitions": len(file_codes - l1_codes), "blank_ts_code": blank_count(df["ts_code"]) if not df.empty else 0,
        "duplicate_full_rows": int(df.duplicated().sum()) if not df.empty else 0,
        "member_codes_missing_in_stock_basic": len(missing_in_basic), "missing_code_sample": sorted(missing_in_basic)[:20],
        # Departure history comes only from the explicit is_new=N pull; losing
        # it would silently project current membership backwards again.
        "rows_with_out_date": rows_with_out_date,
    }
    missing_departure_history = bool(len(df)) and rows_with_out_date == 0
    has_error = details["missing_l1_partitions"] or details["blank_ts_code"] or missing_departure_history
    severity = "error" if has_error else "warning" if missing_in_basic else "info"
    add(severity, "index_member_all", "SW2021 member table checks", details)

def audit_index_member_history(raw_dir: Path, classify_sw2014: pd.DataFrame, add) -> None:
    files = sorted((raw_dir / "index_member").glob("l1_code=*.parquet"))
    df = read_many(files)
    l1_codes = set(classify_sw2014.loc[classify_sw2014["level"].astype(str) == "L1", "index_code"].astype(str)) if not classify_sw2014.empty else set()
    file_codes = set(path.stem.split("=", 1)[1] for path in files)
    details = {
        "files": len(files), "rows": len(df), "missing_l1_partitions": len(l1_codes - file_codes),
        "extra_l1_partitions": len(file_codes - l1_codes),
        "blank_con_code": blank_count(df["con_code"]) if not df.empty else 0,
        "duplicate_full_rows": int(df.duplicated().sum()) if not df.empty else 0,
    }
    has_error = details["missing_l1_partitions"] or details["blank_con_code"] or not files
    add("error" if has_error else "info", "index_member", "SW2014 legacy member table checks", details)

def audit_index_weight(raw_dir: Path, end_date: str, add) -> None:
    """Core-index monthly constituent weights: per-code/per-year partitions.

    The source clamps unpaginated calls to 7,000 rows (most-recent-first), so
    truncation shows up as missing year partitions or closed years with fewer
    than 12 distinct publication months — both checked here. A code's first
    year with data is exempt from the month check (index launch ramp)."""
    dataset_dir = raw_dir / "index_weight"
    end_year = int(end_date[:4])
    years = list(range(int(INDEX_WEIGHT_START_DATE[:4]), end_year + 1))
    missing: list[str] = []
    zero_rows: list[str] = []
    short_months: dict[str, int] = {}
    rows_total = 0
    for code in DEFAULT_CN_INDEX_CODES:
        first_data_year: int | None = None
        for year in years:
            path = dataset_dir / f"index_code={safe_partition_value(code)}" / f"year={year}.parquet"
            if not path.exists():
                missing.append(str(path))
                continue
            trade_dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"].astype(str)
            rows_total += len(trade_dates)
            if trade_dates.empty:
                zero_rows.append(str(path))
                continue
            if first_data_year is None:
                first_data_year = year
            if year < end_year and year > first_data_year:
                months = int(trade_dates.str[:6].nunique())
                if months < 12:
                    short_months[f"{code}/{year}"] = months
    legacy = sorted(str(path) for path in dataset_dir.glob("index_code=*.parquet"))
    details = {
        "codes": len(DEFAULT_CN_INDEX_CODES),
        "expected_files": len(DEFAULT_CN_INDEX_CODES) * len(years),
        "rows": rows_total,
        "missing_year_partitions": len(missing),
        "zero_row_year_partitions": len(zero_rows),
        "closed_years_with_missing_months": short_months,
        "legacy_flat_partitions": len(legacy),
        "missing_sample": missing[:10],
        "zero_sample": zero_rows[:10],
        "legacy_sample": legacy[:5],
    }
    severity = "error" if missing or zero_rows else "warning" if short_months or legacy else "info"
    add(severity, "index_weight", "core-index monthly weight partition checks", details)
    audit_business_payload(
        sorted(dataset_dir.rglob("year=*.parquet")), "index_weight", "index_weight_payload", add,
        key_columns=("index_code", "con_code", "trade_date"),
        expected_fields=tuple(field for field in core.INDEX_WEIGHT_FIELDS.split(",") if field),
    )

def json_count_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}

def latest_parquet_schema(raw_dir: Path, dataset: str) -> list[str]:
    """Schema of the newest partition (last in sorted order): schema checks
    guard what the CURRENT writer produces; the oldest file never changes and
    would return the same answer forever."""
    files = sorted((raw_dir / dataset).rglob("*.parquet"))
    if not files:
        return []
    return list(pq.ParquetFile(files[-1]).schema_arrow.names)

def read_partition_codes(raw_dir: Path, dataset: str, trade_date: str) -> set[str]:
    path = raw_dir / dataset / f"trade_date={trade_date}.parquet"
    if not path.exists() or parquet_rows(path) == 0:
        return set()
    schema = pq.ParquetFile(path).schema_arrow.names
    if "ts_code" not in schema:
        return set()
    df = pd.read_parquet(path, columns=["ts_code"])
    return set(df["ts_code"].dropna().astype(str).str.strip()) - {""}

def code_type_counts(codes: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code in codes:
        symbol, _, suffix = code.partition(".")
        if suffix == "BJ":
            key = "BJ"
        elif symbol.startswith(("900", "200")):
            key = "B_share_like"
        elif symbol.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "519", "520", "560", "561", "562", "563", "588", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169")):
            key = "fund_or_etf_like"
        elif suffix in {"SH", "SZ"}:
            key = "A_share_like"
        else:
            key = suffix or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))

def new_coverage_acc(left_name: str, right_name: str) -> dict[str, Any]:
    return {
        "left_dataset": left_name,
        "right_dataset": right_name,
        "dates_checked": 0,
        "left_only_rows": 0,
        "right_only_rows": 0,
        "dates_with_left_only": 0,
        "dates_with_right_only": 0,
        "max_left_only_on_date": {"trade_date": None, "rows": 0},
        "max_right_only_on_date": {"trade_date": None, "rows": 0},
        "samples": [],
        "_left_only_codes": set(),
        "_right_only_codes": set(),
    }

def update_coverage_acc(acc: dict[str, Any], trade_date: str, left_codes: set[str], right_codes: set[str], sample_limit: int, extra: dict[str, Any] | None = None) -> None:
    left_only = left_codes - right_codes
    right_only = right_codes - left_codes
    acc["dates_checked"] += 1
    acc["left_only_rows"] += len(left_only)
    acc["right_only_rows"] += len(right_only)
    acc["_left_only_codes"].update(left_only)
    acc["_right_only_codes"].update(right_only)
    if left_only:
        acc["dates_with_left_only"] += 1
        if len(left_only) > acc["max_left_only_on_date"]["rows"]:
            acc["max_left_only_on_date"] = {"trade_date": trade_date, "rows": len(left_only)}
    if right_only:
        acc["dates_with_right_only"] += 1
        if len(right_only) > acc["max_right_only_on_date"]["rows"]:
            acc["max_right_only_on_date"] = {"trade_date": trade_date, "rows": len(right_only)}
    if (left_only or right_only) and len(acc["samples"]) < sample_limit:
        sample = {
            "trade_date": trade_date,
            "left_only_count": len(left_only),
            "right_only_count": len(right_only),
            "left_only_sample": sorted(left_only)[:10],
            "right_only_sample": sorted(right_only)[:10],
        }
        if extra:
            sample.update(extra)
        acc["samples"].append(sample)

def finish_coverage_acc(acc: dict[str, Any]) -> dict[str, Any]:
    left_only_codes = acc.pop("_left_only_codes")
    right_only_codes = acc.pop("_right_only_codes")
    acc["unique_left_only_codes"] = len(left_only_codes)
    acc["unique_right_only_codes"] = len(right_only_codes)
    acc["unique_left_only_sample"] = sorted(left_only_codes)[:20]
    acc["unique_right_only_sample"] = sorted(right_only_codes)[:20]
    acc["unique_left_only_code_types"] = code_type_counts(left_only_codes)
    acc["unique_right_only_code_types"] = code_type_counts(right_only_codes)
    return acc

def audit_daily_cross_coverage(raw_dir: Path, trade_dates: set[str], args: argparse.Namespace, add) -> dict[str, set[str]]:
    dates = sorted(d for d in trade_dates if args.start_date <= d <= args.end_date)
    acc_daily_basic = new_coverage_acc("daily", "daily_basic")
    acc_adj = new_coverage_acc("adj_factor", "daily")
    acc_stk_limit = new_coverage_acc("stk_limit", "daily")
    all_codes = {"daily": set(), "daily_basic": set(), "adj_factor": set(), "stk_limit": set()}
    for trade_date in dates:
        daily_codes = read_partition_codes(raw_dir, "daily", trade_date)
        daily_basic_codes = read_partition_codes(raw_dir, "daily_basic", trade_date)
        adj_codes = read_partition_codes(raw_dir, "adj_factor", trade_date)
        stk_limit_codes = read_partition_codes(raw_dir, "stk_limit", trade_date)
        all_codes["daily"].update(daily_codes)
        all_codes["daily_basic"].update(daily_basic_codes)
        all_codes["adj_factor"].update(adj_codes)
        all_codes["stk_limit"].update(stk_limit_codes)
        update_coverage_acc(acc_daily_basic, trade_date, daily_codes, daily_basic_codes, args.sample_limit)
        adj_only = adj_codes - daily_codes
        extra = None
        if adj_only and len(acc_adj["samples"]) < args.sample_limit:
            suspend_codes = read_partition_codes(raw_dir, "suspend_d", trade_date)
            extra = {"left_only_in_suspend_d": len(adj_only & suspend_codes), "suspend_d_codes": len(suspend_codes)}
        update_coverage_acc(acc_adj, trade_date, adj_codes, daily_codes, args.sample_limit, extra)
        update_coverage_acc(acc_stk_limit, trade_date, stk_limit_codes, daily_codes, args.sample_limit)
    daily_basic_details = finish_coverage_acc(acc_daily_basic)
    adj_details = finish_coverage_acc(acc_adj)
    stk_limit_details = finish_coverage_acc(acc_stk_limit)
    add("warning" if daily_basic_details["left_only_rows"] or daily_basic_details["right_only_rows"] else "info", "daily_vs_daily_basic_coverage", "daily and daily_basic same-day code coverage", daily_basic_details)
    add("warning" if adj_details["right_only_rows"] else "info", "adj_factor_vs_daily_coverage", "adj_factor can validly exceed daily because factors may exist for non-trading/suspended names", adj_details)
    add("warning" if stk_limit_details["right_only_rows"] else "info", "stk_limit_vs_daily_coverage", "stk_limit covers A/B shares and funds, so rows can exceed daily", stk_limit_details)
    return all_codes

def audit_unit_schema(raw_dir: Path, add) -> None:
    daily_schema = latest_parquet_schema(raw_dir, "daily")
    daily_basic_schema = latest_parquet_schema(raw_dir, "daily_basic")
    bak_basic_schema = latest_parquet_schema(raw_dir, "bak_basic")
    add("info", "unit_schema_reference", "local schemas and official unit references", {
        "daily_has_vol_amount": {"vol": "vol" in daily_schema, "amount": "amount" in daily_schema},
        "daily_basic_has_vol_amount": {"vol": "vol" in daily_basic_schema, "amount": "amount" in daily_basic_schema},
        "bak_basic_has_vol_amount": {"vol": "vol" in bak_basic_schema, "amount": "amount" in bak_basic_schema},
        "unit_rules": [
            rule.to_record()
            for rule in (
                rules_for(file="daily.parquet")
                + rules_for(file="events.parquet", datasets=("bak_daily",))
                + rules_for(file="raw_only", datasets=("bak_basic",))
            )
        ],
        "doc_refs": SEMANTIC_DOC_REFS,
    })
    if "vol" not in bak_basic_schema and "amount" not in bak_basic_schema:
        add("info", "bak_basic_no_turnover_fields", "bak_basic does not contain volume or amount and must not be used for turnover-unit alignment")
    else:
        add("warning", "bak_basic_no_turnover_fields", "unexpected bak_basic volume/amount fields found; inspect schema before using it for unit alignment", {"schema": bak_basic_schema})

def api_frame(client: TuShareClient, api_name: str, params: dict[str, Any], fields: str) -> pd.DataFrame:
    return frame(client.query(api_name, params, fields))

def numeric_value(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df or pd.isna(df[column].iloc[0]):
        return None
    return float(df[column].iloc[0])

def audit_stock_universe_semantics(raw_dir: Path, all_codes: dict[str, set[str]], add) -> None:
    stock_basic = read_many(sorted((raw_dir / "stock_basic").glob("list_status=*.parquet")), columns=["ts_code", "name", "market", "exchange", "list_status", "list_date", "delist_date"])
    stock_company = read_many(sorted((raw_dir / "stock_company").glob("exchange=*.parquet")), columns=["ts_code", "exchange", "com_name"])
    index_member = read_many(sorted((raw_dir / "index_member_all").glob("l1_code=*.parquet")), columns=["ts_code", "l1_code", "l1_name", "in_date", "out_date"])
    basic_codes = set(stock_basic["ts_code"].dropna().astype(str)) if not stock_basic.empty else set()
    company_codes = set(stock_company["ts_code"].dropna().astype(str)) if not stock_company.empty else set()
    member_codes = set(index_member["ts_code"].dropna().astype(str)) if not index_member.empty else set()
    daily_codes = all_codes.get("daily", set())
    listed_codes = set(stock_basic.loc[stock_basic["list_status"].astype(str) == "L", "ts_code"].astype(str)) if not stock_basic.empty else set()
    delisted_codes = set(stock_basic.loc[stock_basic["list_status"].astype(str) == "D", "ts_code"].astype(str)) if not stock_basic.empty else set()
    bse_basic_codes = set(stock_basic.loc[stock_basic["exchange"].astype(str) == "BSE", "ts_code"].astype(str)) if not stock_basic.empty else set()
    bj_daily_codes = {code for code in daily_codes if code.endswith(".BJ")}
    details = {
        "stock_basic_rows": int(len(stock_basic)),
        "stock_basic_unique_codes": len(basic_codes),
        "stock_basic_status_counts": json_count_dict(stock_basic["list_status"]) if not stock_basic.empty else {},
        "stock_basic_exchange_counts": json_count_dict(stock_basic["exchange"]) if not stock_basic.empty else {},
        "stock_basic_market_counts": json_count_dict(stock_basic["market"]) if not stock_basic.empty else {},
        "stock_company_rows": int(len(stock_company)),
        "stock_company_missing_stock_basic_codes": len(basic_codes - company_codes),
        "stock_company_extra_codes_vs_stock_basic": len(company_codes - basic_codes),
        "stock_company_missing_sample": sorted(basic_codes - company_codes)[:20],
        "stock_company_extra_sample": sorted(company_codes - basic_codes)[:20],
        "daily_unique_codes": len(daily_codes),
        "daily_codes_missing_in_stock_basic": len(daily_codes - basic_codes),
        "daily_missing_in_stock_basic_sample": sorted(daily_codes - basic_codes)[:20],
        "stock_basic_codes_missing_in_daily": len(basic_codes - daily_codes),
        "listed_stock_basic_codes_missing_in_daily": len(listed_codes - daily_codes),
        "listed_missing_in_daily_sample": sorted(listed_codes - daily_codes)[:20],
        "delisted_stock_basic_codes": len(delisted_codes),
        "delisted_codes_with_daily": len(delisted_codes & daily_codes),
        "delisted_with_daily_sample": sorted(delisted_codes & daily_codes)[:20],
        "bse_stock_basic_codes": len(bse_basic_codes),
        "bj_daily_unique_codes": len(bj_daily_codes),
        "bj_daily_codes_missing_in_stock_basic": len(bj_daily_codes - basic_codes),
        "bse_stock_basic_codes_missing_in_daily": len(bse_basic_codes - daily_codes),
        "bj_daily_missing_in_stock_basic_sample": sorted(bj_daily_codes - basic_codes)[:20],
        "industry_member_codes": len(member_codes),
        "industry_member_codes_missing_in_stock_basic": len(member_codes - basic_codes),
        "industry_member_missing_sample": sorted(member_codes - basic_codes)[:20],
    }
    has_diff = any(details[key] for key in ("stock_company_missing_stock_basic_codes", "stock_company_extra_codes_vs_stock_basic", "daily_codes_missing_in_stock_basic", "listed_stock_basic_codes_missing_in_daily", "bj_daily_codes_missing_in_stock_basic", "industry_member_codes_missing_in_stock_basic"))
    add("warning" if has_diff else "info", "stock_universe_semantics", "North-board, delisted, stock_company, daily, and industry-member coverage differences", details)

def audit_pit_availability(raw_dir: Path, add) -> None:
    dataset_columns = {dataset: latest_parquet_schema(raw_dir, dataset) for dataset in ("daily", "daily_basic", "adj_factor", "stk_limit", "bak_basic", "namechange")}
    row_available_at = {dataset: "available_at" in columns for dataset, columns in dataset_columns.items()}
    sidecar_has_fetched_at: dict[str, bool] = {}
    for dataset in dataset_columns:
        meta_files = sorted((raw_dir / dataset).rglob("*.meta.json"))
        if not meta_files:
            sidecar_has_fetched_at[dataset] = False
            continue
        meta = json.loads(meta_files[-1].read_text(encoding="utf-8"))
        sidecar_has_fetched_at[dataset] = bool(meta.get("fetched_at"))
    # Design fact, not an anomaly: severity info so a healthy day reads ok.
    add("info", "pit_available_at", "raw files carry fetch metadata but no row-level available_at; snapshot construction must enforce PIT rules", {
        "row_level_available_at_present": row_available_at,
        "sample_sidecar_has_fetched_at": sidecar_has_fetched_at,
        "rules": {
            "daily": "officially loaded after market close around 15:00-16:00; do not use same-day values for 09:25 decisions",
            "daily_basic": "officially updated around 15:00-17:00; do not use same-day values for 09:25 decisions",
            "stk_limit": "officially around 08:40 and covers A/B shares and funds; keep explicit available_at in PIT layer",
            "adj_factor": "officially around 09:15-09:20, but raw trade_date alone is not enough for PIT-safe joins",
            "namechange": "use ann_date or a derived available_at; start_date can be a future effective date",
        },
    })

def existing_partition_values(raw_dir: Path, datasets: list[str], prefix: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for dataset in datasets:
        values[dataset] = sorted(fundamental_partition_value(path, prefix) for path in (raw_dir / dataset).glob(f"{prefix}=*.parquet"))
    return values

def _grouped_rule_records(rules) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for rule in rules:
        grouped.setdefault(rule.dataset or rule.file, []).append(rule.to_record())
    return grouped


def core_market_unit_rules() -> dict[str, list[dict[str, object]]]:
    """Projection of the shared unit registry (environment/data/units.py) over the core-market files."""
    return _grouped_rule_records(
        rules_for(file="daily.parquet")
        + rules_for(file="intraday_1min.parquet")
        + rules_for(file="auction.parquet")
        + rules_for(file="events.parquet", datasets=("bak_daily",))
        + rules_for(file="raw_only", datasets=("bak_basic",))
    )


def fundamental_unit_rules() -> dict[str, list[dict[str, object]]]:
    """Projection of the shared unit registry over the raw financial statement files."""
    return _grouped_rule_records(rules_for(file="fundamentals.parquet"))

def audit_integrated_filesystem(raw_dir: Path, datasets: list[str], add) -> None:
    parquet_files: list[Path] = []
    meta_files: list[Path] = []
    tmp_files: list[Path] = []
    missing_dirs: list[str] = []
    per_dataset: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        dataset_dir = raw_dir / dataset
        if not dataset_dir.exists():
            missing_dirs.append(dataset)
            per_dataset[dataset] = {"parquet_files": 0, "meta_files": 0, "tmp_files": 0}
            continue
        ds_parquet = sorted(dataset_dir.rglob("*.parquet"))
        ds_meta = sorted(dataset_dir.rglob("*.meta.json"))
        ds_tmp = sorted(path for path in dataset_dir.rglob("*") if ".tmp" in path.name)
        parquet_files.extend(ds_parquet)
        meta_files.extend(ds_meta)
        tmp_files.extend(ds_tmp)
        per_dataset[dataset] = {"parquet_files": len(ds_parquet), "meta_files": len(ds_meta), "tmp_files": len(ds_tmp)}
    parquet_meta_paths = {str(path.with_suffix(path.suffix + ".meta.json")) for path in parquet_files}
    meta_set = {str(path) for path in meta_files}
    missing_meta = sorted(parquet_meta_paths - meta_set)
    orphan_meta = sorted(meta_set - parquet_meta_paths)
    corrupt_sidecars: list[str] = []
    incomplete_commits: list[str] = []
    for path in parquet_files:
        try:
            parquet_meta(path)
        except core.CorruptSidecarError:
            corrupt_sidecars.append(str(path))
            continue
        if not committed_partition_intact(path):
            incomplete_commits.append(str(path))
    add("error" if missing_dirs or missing_meta or orphan_meta or tmp_files or incomplete_commits or corrupt_sidecars else "info", "integrated_filesystem", "base research parquet sidecar inventory", {
        "datasets": datasets,
        "per_dataset": per_dataset,
        "parquet_files": len(parquet_files),
        "meta_files": len(meta_files),
        "missing_meta": len(missing_meta),
        "orphan_meta": len(orphan_meta),
        "tmp_files": len(tmp_files),
        "missing_dataset_dirs": missing_dirs,
        "missing_meta_sample": missing_meta[:10],
        "orphan_meta_sample": orphan_meta[:10],
        "commit_pairs_checked": len(parquet_files),
        "incomplete_commit_pairs": len(incomplete_commits),
        "incomplete_commit_sample": incomplete_commits[:10],
        "corrupt_sidecars": len(corrupt_sidecars),
        "corrupt_sidecar_sample": corrupt_sidecars[:10],
    })

# A vendor per-call cap pins many periods at exactly the same row count -- the
# dataset maximum -- because every capped call returns the cap and query_paged
# stops on the short page (lake evidence 2026-08: cashflow_vip held 23 periods
# at exactly 6,400 rows and income_vip 6 at exactly 9,000, each the dataset
# maximum at the time, while genuine period counts vary organically).
# Cross-statement row ratios are NOT a usable signature: the repaired lake
# shows the statements legitimately diverging per period by more than 2x.
FUNDAMENTAL_STATEMENT_DATASETS = ("income_vip", "balancesheet_vip", "cashflow_vip")
CAP_PLATEAU_MIN_PERIODS = 3
CAP_PLATEAU_MIN_ROWS = 1000

def audit_fundamental_cap_plateau(period_rows: dict[str, dict[str, int]], add) -> None:
    """Warn when a statement dataset has several periods pinned at exactly its
    maximum row count -- the truncation signature of an unknown per-call cap."""
    plateaus: list[dict[str, Any]] = []
    for dataset in FUNDAMENTAL_STATEMENT_DATASETS:
        rows = period_rows.get(dataset)
        if not rows:
            continue
        peak = max(rows.values())
        pinned = sorted(period for period, count in rows.items() if count == peak)
        if peak >= CAP_PLATEAU_MIN_ROWS and len(pinned) >= CAP_PLATEAU_MIN_PERIODS:
            plateaus.append({"dataset": dataset, "max_rows": peak, "periods_at_max": len(pinned), "period_sample": pinned[:10]})
    message = (
        "statement datasets have multiple periods pinned at their maximum row count"
        if plateaus
        else "no statement dataset has multiple periods pinned at exactly its maximum row count"
    )
    add("warning" if plateaus else "info", "fundamental_statement_cap_plateau", message, {
        "datasets_checked": [dataset for dataset in FUNDAMENTAL_STATEMENT_DATASETS if period_rows.get(dataset)],
        "min_periods_at_max": CAP_PLATEAU_MIN_PERIODS,
        "min_rows": CAP_PLATEAU_MIN_ROWS,
        "plateaus": plateaus,
    })

def audit_fundamental_completeness(raw_dir: Path, args: argparse.Namespace, add) -> None:
    stock_codes = load_stock_codes(raw_dir)
    period_datasets = [name for name, spec in FUNDAMENTAL_SPECS.items() if spec.strategy == "period"]
    ann_month_datasets = [name for name, spec in FUNDAMENTAL_SPECS.items() if spec.strategy == "ann_month"]
    period_values = existing_partition_values(raw_dir, period_datasets, "period")
    ann_values = existing_partition_values(raw_dir, ann_month_datasets, "ann_month")
    all_periods = sorted({value for values in period_values.values() for value in values if value})
    all_months = sorted({value for values in ann_values.values() for value in values if value})
    period_end = args.fundamental_end_date or (all_periods[-1] if all_periods else args.fundamental_start_date)
    ann_end = args.fundamental_end_date or (month_end_from_yyyymm(all_months[-1]) if all_months else args.fundamental_start_date)
    if args.fundamental_end_date:
        # A new announcing month's partition appears on its first trading
        # evening; clamp an explicit calendar end so a weekend month boundary
        # does not expect the partition before the producing job could run.
        ann_end = min(ann_end, load_sse_open_dates(raw_dir, args.fundamental_start_date, args.fundamental_end_date)[-1])
    periods = set(quarter_periods(args.fundamental_start_date, period_end))
    months = {month for _, _, month in month_windows(args.fundamental_start_date, ann_end)}
    add("info", "fundamental_expected_ranges", "fundamental expected partition ranges inferred from local data or explicit args", {
        "fundamental_start_date": args.fundamental_start_date,
        "fundamental_end_date_arg": args.fundamental_end_date,
        "period_datasets": period_datasets,
        "ann_month_datasets": ann_month_datasets,
        "period_expected_end": period_end,
        "ann_month_expected_end": ann_end,
        "expected_period_partitions": len(periods),
        "expected_ann_month_partitions": len(months),
        "expected_ts_code_partitions": len(stock_codes),
    })
    statement_period_rows: dict[str, dict[str, int]] = {}
    for dataset in selected_fundamental_datasets(getattr(args, "fundamental_datasets", None)):
        spec = FUNDAMENTAL_SPECS[dataset]
        if spec.strategy == "period":
            row_counts = audit_fundamental_dataset(raw_dir, spec, periods, "period", add)
            if dataset in FUNDAMENTAL_STATEMENT_DATASETS:
                statement_period_rows[dataset] = row_counts
        elif spec.strategy == "ann_month":
            audit_fundamental_dataset(raw_dir, spec, months, "ann_month", add)
        else:
            audit_fundamental_dataset(raw_dir, spec, set(stock_codes), "ts_code", add)
    audit_fundamental_cap_plateau(statement_period_rows, add)

def audit_fundamental_unit_and_pit_semantics(raw_dir: Path, add) -> None:
    schemas = {dataset: latest_parquet_schema(raw_dir, dataset) for dataset in FUNDAMENTAL_DATASETS}
    tables_with_f_ann_date = sorted(dataset for dataset, columns in schemas.items() if "f_ann_date" in columns)
    tables_without_f_ann_date = sorted(dataset for dataset, columns in schemas.items() if columns and "f_ann_date" not in columns)
    # Design fact, not an anomaly: severity info so a healthy day reads ok.
    add("info", "fundamental_unit_and_pit_semantics", "fundamental units and PIT fields are mixed by interface and field family", {
        "tables_with_f_ann_date": tables_with_f_ann_date,
        "tables_without_f_ann_date": tables_without_f_ann_date,
        "unit_rules": fundamental_unit_rules(),
        "pit_rules": {
            "income_vip/balancesheet_vip/cashflow_vip": "use f_ann_date first, then ann_date; choose the latest visible version at decision time",
            "fina_indicator_vip": "no f_ann_date in local schema; use ann_date conservatively and expect duplicate same-period rows",
            "forecast_vip": "PIT visibility = each version's own ann_date (first_ann_date is a series attribute, never an availability floor); keep update_flag/type; it is an event table, not a final statement table",
            "express_vip": "use ann_date; it is a preliminary result table and may differ from later statements",
            "dividend": "use imp_ann_date, ex_date, record_date, and pay_date according to field meaning; ann_date can be blank",
            "disclosure_date": "calendar/planned disclosure table; do not treat it as a fundamental value",
        },
        "doc_refs": {
            dataset: INTEGRATED_DOC_REFS[dataset]
            for dataset in sorted(set(FUNDAMENTAL_DATASETS) & set(INTEGRATED_DOC_REFS))
        },
    })

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class DomainAuditProfile:
    """Shared partition-inventory + key/PIT auditor knobs for one raw domain.

    The four domain auditors (text/macro/event/board) were hand-written copies
    of the same shape and had drifted (uneven pagination-probe exclusion,
    text-only empty early-return, event-only zero_rows_ok). The profile encodes
    exactly what genuinely differs; the emitted check names and details keys
    stay the per-domain status-JSON contract the nightly consumers read.
    """

    domain: str                    # check-name suffix: <api>_<domain>_partitions/_keys
    exact_limit_rows: frozenset    # row counts that look like a page-limit truncation
    exact_limit_key: str           # details key name for that count
    apply_pagination_probe: bool   # exclude partitions with a recorded pagination probe
    empty_error_mode: str          # "with_expected" | "always" | "early_return"
    include_strategy: bool
    key_columns: tuple
    key_extra_columns: tuple       # extra columns joined into the keys read
    blank_mode: str                # "per_key_field" | "available_at_total" | "available_at_rows"
    pit_rules: object
    partitions_message: str
    keys_message: str
    partition_prefix: str | None = None  # event: strategy partition glob + ignored tracking
    zero_rows_ok: bool | None = None     # event: emitted and gates the zero-rows warning
    availability_required: bool = True   # board static_full: availability is not an error
    expected_fields: tuple = ()          # spec request contract for the payload check


def audit_domain_dataset(raw_dir: Path, spec, expected_paths: set[Path], add, profile: DomainAuditProfile) -> None:
    dataset_dir = raw_dir / spec.api_name
    ignored_parquet: list[str] = []
    ignored_meta: list[str] = []
    if profile.partition_prefix is not None:
        files = sorted(dataset_dir.rglob(f"{profile.partition_prefix}=*.parquet"))
        meta_files = sorted(dataset_dir.rglob(f"{profile.partition_prefix}=*.parquet.meta.json"))
        ignored_parquet = sorted(str(path.resolve()) for path in dataset_dir.rglob("*.parquet") if path not in files)
        ignored_meta = sorted(str(path.resolve()) for path in dataset_dir.rglob("*.meta.json") if path not in meta_files)
    else:
        files = sorted(dataset_dir.rglob("*.parquet"))
        meta_files = sorted(dataset_dir.rglob("*.meta.json"))
    file_set = {path.resolve() for path in files}
    expected_set = {path.resolve() for path in expected_paths}
    missing_expected = sorted(str(path) for path in expected_set - file_set)
    extra_files = sorted(str(path) for path in file_set - expected_set)
    if profile.empty_error_mode == "early_return" and not files:
        add("error", f"{spec.api_name}_{profile.domain}_partitions", f"{spec.api_name} {profile.domain} dataset is missing", {
            "strategy": spec.strategy,
            "expected_files": len(expected_set),
            "missing_expected_files": len(missing_expected),
            "missing_sample": missing_expected[:10],
        })
        return
    parquet_meta = {str(path.with_suffix(path.suffix + ".meta.json")) for path in files}
    meta_set = {str(path) for path in meta_files}
    missing_meta = sorted(parquet_meta - meta_set)
    orphan_meta = sorted(meta_set - parquet_meta)
    row_counts = {str(path): parquet_rows(path) for path in files}
    zero_files = [path for path, rows in row_counts.items() if rows == 0]
    exact_limit_files = [
        path
        for path, rows in row_counts.items()
        if rows in profile.exact_limit_rows
        and not (profile.apply_pagination_probe and has_pagination_probe(Path(path)))
    ]
    has_error = bool(missing_expected or missing_meta or orphan_meta)
    if profile.empty_error_mode == "always":
        has_error = has_error or not files
    elif profile.empty_error_mode == "with_expected":
        has_error = has_error or (not files and expected_set)
    # zero_rows_ok suppresses the zero-row warning partition by partition, so a
    # discontinued feed can die silently (slb_len_mm published its last row on
    # 2025-07-25 and stayed empty for a year without a single finding). A long
    # trailing run of consecutive zero-row partitions is the discontinuation
    # signature: warn so a human checks the official feed status. Runs are
    # counted per directory: a multi-source dataset (news src=<name>/date=*)
    # sorts whole sources consecutively, so a single whole-listing run could
    # only ever see the last-sorted source; each source subtree is its own
    # feed and dies independently.
    trailing_zero_by_dir: dict[Path, int] = {}
    for path in files:
        trailing_zero_by_dir[path.parent] = (
            trailing_zero_by_dir.get(path.parent, 0) + 1 if row_counts[str(path)] == 0 else 0
        )
    trailing_zero_files = max(trailing_zero_by_dir.values(), default=0)
    stale_feed_sources = sorted(
        parent.name
        for parent, run in trailing_zero_by_dir.items()
        if run >= TRAILING_ZERO_RUN_WARN_PARTITIONS
    )
    stale_feed = bool(profile.zero_rows_ok) and bool(stale_feed_sources)
    has_warning = bool(exact_limit_files)
    if profile.zero_rows_ok is not None:
        has_warning = has_warning or bool(zero_files and not profile.zero_rows_ok) or bool(ignored_parquet or ignored_meta)
        has_warning = has_warning or stale_feed
    details: dict[str, Any] = {}
    if profile.include_strategy:
        details["strategy"] = spec.strategy
    if profile.partition_prefix is not None:
        details["partition_prefix"] = profile.partition_prefix
    details.update({
        "files": len(files),
        "expected_files": len(expected_set),
        "missing_expected_files": len(missing_expected),
        "extra_files": len(extra_files),
    })
    if profile.partition_prefix is not None:
        details["ignored_non_strategy_parquet_files"] = len(ignored_parquet)
        details["ignored_non_strategy_meta_files"] = len(ignored_meta)
    details.update({
        "meta_files": len(meta_files),
        "rows": int(sum(row_counts.values())),
        "zero_row_partitions": len(zero_files),
    })
    if profile.zero_rows_ok is not None:
        details["zero_rows_ok"] = profile.zero_rows_ok
        details["trailing_zero_row_partitions"] = trailing_zero_files
        details["stale_feed_sources"] = stale_feed_sources
    details.update({
        "missing_meta": len(missing_meta),
        "orphan_meta": len(orphan_meta),
        profile.exact_limit_key: len(exact_limit_files),
        "missing_sample": missing_expected[:10],
        "extra_sample": extra_files[:10],
    })
    if profile.partition_prefix is not None:
        details["ignored_non_strategy_parquet_sample"] = ignored_parquet[:10]
        details["ignored_non_strategy_meta_sample"] = ignored_meta[:10]
    details.update({
        "zero_sample": zero_files[:10],
        "exact_limit_sample": exact_limit_files[:10],
    })
    add(
        "error" if has_error else "warning" if has_warning else "info",
        f"{spec.api_name}_{profile.domain}_partitions",
        f"{spec.api_name} {profile.partitions_message}",
        details,
    )
    key_details = audit_domain_keys(files, profile, row_counts)
    if profile.blank_mode == "per_key_field":
        has_blank = any(int(value) for value in key_details["blank_key_fields"].values())
    elif profile.blank_mode == "available_at_total":
        has_blank = bool(key_details["blank_time_fields"])
    else:
        has_blank = bool(key_details["blank_available_at_rows"])
    availability_error = bool(key_details["missing_available_at_files"] or key_details["unparseable_available_at_rows"])
    has_key_error = bool(key_details["missing_key_column_files"]) or (profile.availability_required and availability_error)
    has_key_warning = bool(key_details["duplicate_key_rows"]) or (
        has_blank if profile.availability_required or profile.blank_mode != "available_at_rows" else False
    )
    add(
        "error" if has_key_error else "warning" if has_key_warning else "info",
        f"{spec.api_name}_{profile.domain}_keys",
        f"{spec.api_name} {profile.keys_message}",
        key_details,
    )
    audit_business_payload(
        files, spec.api_name, f"{spec.api_name}_{profile.domain}_payload", add,
        key_columns=profile.key_columns, expected_fields=profile.expected_fields,
    )


def audit_domain_keys(files: list[Path], profile: DomainAuditProfile, row_counts: dict[str, int]) -> dict[str, Any]:
    duplicate_key_rows = 0
    blank_key_fields: dict[str, int] = {}
    blank_available_at = 0
    unparseable_available_at_rows = 0
    missing_key_column_files: list[str] = []
    missing_available_at_files: list[str] = []
    for path in files:
        if row_counts[str(path)] == 0:
            continue
        schema = pq.ParquetFile(path).schema_arrow.names
        missing = [col for col in profile.key_columns if col not in schema]
        if missing:
            missing_key_column_files.append(str(path))
            continue
        keys = list(profile.key_columns)
        columns = list(dict.fromkeys(keys + list(profile.key_extra_columns)))
        df = pd.read_parquet(path, columns=[col for col in columns if col and col in schema])
        if keys:
            duplicate_key_rows += int(df.duplicated(keys).sum())
            if profile.blank_mode == "per_key_field":
                for col in keys:
                    blank_key_fields[col] = blank_key_fields.get(col, 0) + blank_count(df[col])
        if "available_at" not in df.columns:
            missing_available_at_files.append(str(path))
            continue
        available = df["available_at"].astype(str).str.strip()
        if profile.blank_mode == "available_at_total":
            blank_available_at += blank_count(df["available_at"])
        blank = available.eq("") | available.eq("nan") | available.eq("None")
        if profile.blank_mode == "available_at_rows":
            blank_available_at += int(blank.sum())
        nonblank = available[~blank]
        if not nonblank.empty:
            parsed = pd.to_datetime(nonblank, errors="coerce", utc=True, format="mixed")
            unparseable_available_at_rows += int(parsed.isna().sum())
    details: dict[str, Any] = {
        "files_checked": len(files),
        "key_columns": list(profile.key_columns),
        "duplicate_key_rows": duplicate_key_rows,
    }
    if profile.blank_mode == "per_key_field":
        details["blank_key_fields"] = blank_key_fields
    elif profile.blank_mode == "available_at_total":
        details["blank_time_fields"] = blank_available_at
    else:
        details["blank_available_at_rows"] = blank_available_at
    details.update({
        "unparseable_available_at_rows": unparseable_available_at_rows,
        "missing_key_column_files": len(missing_key_column_files),
        "missing_available_at_files": len(missing_available_at_files),
        "missing_key_column_sample": missing_key_column_files[:10],
        "missing_available_at_sample": missing_available_at_files[:10],
        "pit_rules": profile.pit_rules,
    })
    return details


def audit_text_dataset(raw_dir: Path, spec: TextDataset, expected_paths: set[Path], add) -> None:
    audit_domain_dataset(raw_dir, spec, expected_paths, add, DomainAuditProfile(
        domain="text",
        # The generic web caps never matched any real text page limit
        # (2000/400/500/1000/3000/1500), so the truncation signature could not
        # fire for a single text dataset; include the spec's own limit.
        exact_limit_rows=frozenset({spec.page_limit, 5000, 6000, 7000, 8000, 10000}),
        exact_limit_key="exact_common_limit_row_count_partitions",
        apply_pagination_probe=False,
        empty_error_mode="early_return",
        include_strategy=False,
        key_columns=tuple(spec.key_columns),
        key_extra_columns=("available_at", spec.time_column, spec.date_column),
        blank_mode="available_at_total",
        pit_rules=text_pit_rules().get(spec.api_name, {}),
        partitions_message="text partition inventory",
        keys_message="text key and PIT checks",
        # Text feeds tolerate empty days, which is exactly why they need the
        # trailing-zero stale-feed detector (news src=fenghuang died 2026-04
        # and stayed invisible for four months without it).
        zero_rows_ok=spec.zero_rows_ok,
        expected_fields=spec_fields_tuple(spec),
    ))


def text_pit_rules() -> dict[str, dict[str, str]]:
    return {
        "anns_d": {"available_at": "rec_time; if missing, treat ann_date as visible only after close or next session", "unit": "text/url, no numeric unit"},
        "major_news": {"available_at": "pub_time", "unit": "text"},
        "news": {"available_at": "datetime", "unit": "text"},
        "cctv_news": {"available_at": "date at 23:59:59+08:00 conservative fallback", "unit": "text"},
        "npr": {"available_at": "pubtime", "unit": "HTML/text"},
        "irm_qa_sh": {"available_at": "pub_time; trade_date end-of-day fallback", "unit": "Q&A text"},
        "irm_qa_sz": {"available_at": "pub_time; trade_date end-of-day fallback", "unit": "Q&A text"},
        "research_report": {"available_at": "trade_date conservative end-of-day unless a more precise time is available", "unit": "text/summary/url"},
        "report_rc": {"available_at": "create_time if present, otherwise report_date 22:00+08 based on documented nightly update", "unit": "mixed forecast fields; do not mix directly with P2 actual statements"},
    }

def expected_text_paths(raw_dir: Path, spec: TextDataset, start_date: str, end_date: str, args: argparse.Namespace) -> set[Path]:
    start = max(start_date, spec.start_date)
    if spec.strategy in {"range_month", "time_range_month"}:
        months = [month for _, _, month in month_windows(start, end_date)]
        if spec.strategy == "time_range_month":
            sources = args.major_news_src or [""]
            return {raw_dir / spec.api_name / (f"src={safe_partition_value(source)}" if source else "src=all") / f"month={month}.parquet" for source in sources for month in months}
        return {raw_dir / spec.api_name / f"month={month}.parquet" for month in months}
    if spec.strategy == "news_src_month":
        sources = selected_news_sources(getattr(args, "news_src", []))
        months = [month for _, _, month in month_windows(start, end_date)]
        return {raw_dir / spec.api_name / f"src={safe_partition_value(source)}" / f"month={month}.parquet" for source in sources for month in months}
    if spec.strategy == "news_src_day":
        sources = selected_news_sources(getattr(args, "news_src", []))
        days = date_range_days(start, end_date)
        return {raw_dir / spec.api_name / f"src={safe_partition_value(source)}" / f"date={day}.parquet" for source in sources for day in days}
    if spec.strategy == "day":
        return {raw_dir / spec.api_name / f"date={day}.parquet" for day in date_range_days(start, end_date)}
    raise RuntimeError(f"unsupported text strategy {spec.strategy} for {spec.api_name}")

def audit_text_completeness(raw_dir: Path, args: argparse.Namespace, add) -> None:
    datasets = selected_text_datasets(getattr(args, "text_datasets", None))
    text_end = args.text_end_date or args.end_date
    # Text is a natural-day domain: news, announcements, policy documents and
    # research reports publish on weekends and holidays, and its producing job
    # runs every calendar evening. Expectations therefore run to the calendar
    # end date -- clamping them to the last SSE open day left every weekend
    # partition unchecked until the next session (data docs §4).
    add("info", "text_expected_scope", "optional TuShare text/NL datasets included in this audit", {
        "datasets": datasets,
        "start_date": args.text_start_date,
        "end_date": text_end,
        "dataset_pit_rules": text_pit_rules(),
    })
    for dataset in datasets:
        spec = TEXT_SPECS[dataset]
        expected = expected_text_paths(raw_dir, spec, args.text_start_date, text_end, args)
        audit_text_dataset(raw_dir, spec, expected, add)


def audit_text_only(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = (repo_root / (args.output or TEXT_EVIDENCE_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    datasets = selected_text_datasets(getattr(args, "text_datasets", None))
    audit_integrated_filesystem(raw_dir, datasets, add)
    audit_text_completeness(raw_dir, args, add)
    report = build_quality_report(
        report_type="text_evidence",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.text_start_date,
            "end_date": args.text_end_date,
            "datasets": datasets,
        },
        findings=findings,
        metadata={
            "pit_rules": text_pit_rules(),
            "doc_refs": {
                dataset: INTEGRATED_DOC_REFS[dataset]
                for dataset in sorted(set(datasets) & set(INTEGRATED_DOC_REFS))
            },
            "conclusions": [
                "Text rows remain raw evidence; snapshot construction must apply each source's recorded availability rule.",
                "Repeated delivery and republication are retained in raw data and deduplicated deterministically in the snapshot layer.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"text audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def selected_audit_macro_datasets(args: argparse.Namespace) -> list[str]:
    return select_datasets(getattr(args, "datasets", None), default=MACRO_DATASETS, allowed=MACRO_SPECS, label="macro/global")

def expected_macro_paths(raw_dir: Path, spec: MacroDataset, start_date: str, end_date: str, args: argparse.Namespace) -> set[Path]:
    start = max(start_date, spec.start_date)
    if spec.strategy in {"quarter_once", "month_once"}:
        # Mirrors the downloader's retained floor: range pulls always cover
        # [floor, latest] and land in ONE canonical file regardless of the
        # audit window, so the expectation must not follow start/end_date.
        retained = max(min(start_date, MACRO_RETAINED_FLOOR), spec.start_date)
        if spec.strategy == "quarter_once":
            start_q = max(yyyymmdd_to_quarter(retained), spec.start_quarter)
            return {raw_dir / spec.api_name / f"range={start_q}_latest.parquet"}
        start_m = max(yyyymmdd_to_month(retained), spec.start_month)
        return {raw_dir / spec.api_name / f"range={start_m}_latest.parquet"}
    if spec.strategy == "month_loop":
        return {raw_dir / spec.api_name / f"month={month}.parquet" for _, _, month in month_windows(start, end_date)}
    if spec.strategy == "date_year":
        return {raw_dir / spec.api_name / f"year={year}.parquet" for year in range(int(start[:4]), int(end_date[:4]) + 1)}
    if spec.strategy == "trade_date":
        open_dates = load_sse_open_dates(raw_dir, start, end_date)
        if spec.loop_values:
            return {
                raw_dir / spec.api_name / f"{spec.loop_param}={safe_partition_value(value)}" / f"trade_date={d}.parquet"
                for value in spec.loop_values
                for d in open_dates
                if d >= spec.loop_start_date(value)
            }
        return {raw_dir / spec.api_name / f"trade_date={d}.parquet" for d in open_dates}
    if spec.strategy == "static_full":
        directory = raw_dir / spec.api_name
        if spec.loop_values:
            return {directory / f"{spec.loop_param}={safe_partition_value(value)}.parquet" for value in spec.loop_values}
        return {directory / "full.parquet"}
    if spec.strategy == "date_year_by_ts_code":
        if spec.api_name == "index_global":
            codes = selected_index_codes(args)
        elif spec.api_name in ("index_daily", "index_dailybasic"):
            codes = selected_cn_index_codes(args)
        else:
            codes = selected_fx_codes(args)
        return {
            raw_dir / spec.api_name / f"ts_code={safe_partition_value(ts_code)}" / f"year={year}.parquet"
            for ts_code in codes
            for year in range(int(start[:4]), int(end_date[:4]) + 1)
        }
    if spec.strategy == "eco_cal_month":
        countries = selected_eco_filter_values(args, "eco_country")
        currencies = selected_eco_filter_values(args, "eco_currency")
        events = selected_eco_filter_values(args, "eco_event")
        return {
            raw_dir / spec.api_name / f"country={safe_partition_value(country) if country else 'all'}" / f"currency={safe_partition_value(currency) if currency else 'all'}" / f"event={safe_partition_value(event) if event else 'all'}" / f"month={month}.parquet"
            for country in countries
            for currency in currencies
            for event in events
            for _, _, month in month_windows(start, end_date)
        }
    raise RuntimeError(f"unsupported macro strategy {spec.strategy} for {spec.api_name}")

def macro_pit_rules() -> dict[str, str]:
    return {
        "cn_schedule": "publish_date is the intended release date and should refine monthly/quarterly macro visibility when snapshot construction maps data_api to realized releases.",
        "monthly_macro": "raw month-only indicators are stamped conservatively as month-end plus 31 days until cn_schedule or another release timestamp is applied.",
        "quarterly_macro": "raw quarter-only indicators are stamped conservatively as quarter-end plus 45 days until a release schedule is applied.",
        "daily_rates": "date-only rates and cross-market daily series are stamped at local end-of-day; do not use them for same-day open decisions without an explicit release time.",
        "eco_cal": "date+time events use source time when parseable; all-day or missing-time events fall back to date end-of-day.",
        "derivatives_daily": "fut_daily/fut_mapping/opt_daily/cb_daily rows are stamped at trade_date end-of-day and roll on the evening node: usable from the NEXT trading morning, never for same-day open decisions.",
        "derivatives_registry": "fut_basic/opt_basic/cb_basic rows become visible at their list_date; cb_call announcements at ann_date end-of-day (redemption events are evening disclosures). WARNING: cb_basic is a nightly CURRENT-STATE refresh — conv_price/remain_size/newest_rating/delist_date must never feed historical backtests; derive the as-of conversion price from cb_daily (100 * stock close / cb_value), use cb_over_rate for as-of premium and cb_call for redemption outcomes.",
    }

def macro_unit_rules() -> dict[str, list[dict[str, object]]]:
    """Projection of the shared unit registry (environment/data/units.py) over all MACRO_SPECS datasets."""
    return dataset_rules_records(tuple(MACRO_SPECS))

def audit_macro_dataset(raw_dir: Path, spec: MacroDataset, expected_paths: set[Path], add) -> None:
    audit_domain_dataset(raw_dir, spec, expected_paths, add, DomainAuditProfile(
        domain="macro",
        # 2000/4000 are the measured per-call caps of the date_year rate and
        # market-stat feeds (repo_daily/sz_daily_info and shibor_quote/
        # daily_info); a partition landing exactly there is a truncation
        # signature, which hid six years of two-month "annual" repo data.
        # The spec's own limit joins the set so small-cap feeds (eco_cal 100)
        # are covered too.
        exact_limit_rows=frozenset({spec.page_limit, 1000, 2000, 3000, 4000, 5000, 8000, 10000}),
        exact_limit_key="exact_common_limit_row_count_partitions",
        apply_pagination_probe=False,
        empty_error_mode="with_expected",
        include_strategy=True,
        key_columns=tuple(spec.key_columns),
        key_extra_columns=("available_at", "available_at_rule"),
        blank_mode="per_key_field",
        pit_rules=macro_pit_rules(),
        partitions_message="macro/global partition inventory",
        keys_message="key, PIT, and duplicate checks",
        expected_fields=spec_fields_tuple(spec),
    ))


def audit_macro_completeness(raw_dir: Path, args: argparse.Namespace, add) -> None:
    datasets = selected_audit_macro_datasets(args)
    # The producing jobs end on the last SSE open date on or before their
    # calendar end date, so a month/year that only weekend or holiday
    # calendar dates have reached is not expected yet (a plain calendar end
    # made the three month-partitioned datasets error every month boundary).
    # An audit without a usable trade calendar must still complete with
    # findings, so the clamp failure is itself an error finding.
    try:
        covered_end = load_sse_open_dates(raw_dir, args.start_date, args.end_date)[-1]
    except RuntimeError as exc:
        covered_end = args.end_date
        add("error", "macro_expected_window_unclamped", "SSE trade_cal unavailable; completeness expectations use the raw calendar end date", {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "error": str(exc),
        })
    add("info", "macro_expected_scope", "TuShare macro, policy, and global-context datasets included in this audit", {
        "datasets": datasets,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "covered_end_date": covered_end,
        "index_codes": selected_index_codes(args),
        "fx_codes": selected_fx_codes(args),
        "eco_country": selected_eco_filter_values(args, "eco_country"),
        "eco_currency": selected_eco_filter_values(args, "eco_currency"),
        "eco_event": selected_eco_filter_values(args, "eco_event"),
        "dataset_pit_rules": macro_pit_rules(),
        "dataset_unit_rules": macro_unit_rules(),
    })
    for dataset in datasets:
        spec = MACRO_SPECS[dataset]
        expected = expected_macro_paths(raw_dir, spec, args.start_date, covered_end, args)
        if spec.strategy in {"quarter_once", "month_once"}:
            # Extra range files duplicate the whole series in snapshot domain
            # unions; the downloader prunes them, so any survivor is an error.
            stale = sorted(
                str(path)
                for path in (raw_dir / spec.api_name).glob("range=*.parquet")
                if path.resolve() not in {p.resolve() for p in expected}
            )
            if stale:
                add("error", f"{spec.api_name}_stale_range_partitions", "non-canonical range partitions duplicate the series", {
                    "strategy": spec.strategy,
                    "stale_files": len(stale),
                    "stale_sample": stale[:10],
                })
        audit_macro_dataset(raw_dir, spec, expected, add)

def audit_macro_only(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = (repo_root / (args.output or MACRO_CONTEXT_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    datasets = selected_audit_macro_datasets(args)
    audit_integrated_filesystem(raw_dir, datasets, add)
    audit_macro_completeness(raw_dir, args, add)
    report = build_quality_report(
        report_type="macro_context",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": datasets,
            "index_codes": selected_index_codes(args),
            "fx_codes": selected_fx_codes(args),
            "eco_country": selected_eco_filter_values(args, "eco_country"),
            "eco_currency": selected_eco_filter_values(args, "eco_currency"),
            "eco_event": selected_eco_filter_values(args, "eco_event"),
        },
        findings=findings,
        metadata={
            "unit_rules": macro_unit_rules(),
            "pit_rules": macro_pit_rules(),
            "doc_refs": {
                dataset: INTEGRATED_DOC_REFS[dataset]
                for dataset in sorted(set(datasets) & set(INTEGRATED_DOC_REFS))
            },
            "conclusions": [
                "Macro/global context is stored as raw evidence and regime context; snapshot construction must still apply release-time and event-specific PIT rules.",
                "Monthly and quarterly macro tables use conservative availability fallbacks until cn_schedule or a more precise source release time is joined.",
                "Economic-calendar values are heterogeneous by event and should not be turned into numeric signals without event-specific parsing.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"macro audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def expected_event_paths(raw_dir: Path, spec: EventDataset, start_date: str, end_date: str) -> set[Path]:
    start = max(start_date, spec.start_date)
    if spec.strategy == "trade_date":
        trade_end_date = min(end_date, latest_sse_calendar_date(raw_dir))
        trade_dates = load_sse_open_dates(raw_dir, start, trade_end_date)
        return {raw_dir / spec.api_name / f"trade_date={trade_date}.parquet" for trade_date in trade_dates}
    if spec.strategy == "range_month":
        return {raw_dir / spec.api_name / f"month={month}.parquet" for _, _, month in month_windows(start, end_date)}
    if spec.strategy == "day":
        return {raw_dir / spec.api_name / f"date={day}.parquet" for day in date_range_days(start, end_date)}
    raise RuntimeError(f"unsupported event/flow strategy {spec.strategy} for {spec.api_name}")

# Raw event/flow datasets whose snapshot dataset id differs from the api name.
SNAPSHOT_EVENT_ID_BY_RAW = {"share_float": "share_float_complete"}

def event_unit_rules() -> dict[str, list[dict[str, object]]]:
    """Projection of the shared unit registry (environment/data/units.py) over all EVENT_FLOW_SPECS datasets."""
    datasets = tuple(SNAPSHOT_EVENT_ID_BY_RAW.get(name, name) for name in EVENT_FLOW_SPECS)
    return dataset_rules_records(datasets)

def event_pit_rules() -> dict[str, str]:
    return {
        "margin": "available_at uses next-day 09:00+08 from trade_date.",
        "margin_detail": "available_at uses next-day 09:00+08 from trade_date.",
        "margin_secs": "available_at uses same-day 09:00+08 from trade_date because this is a pre-open eligibility table.",
        "moneyflow": "available_at uses 19:00+08 from trade_date.",
        "moneyflow_dc": "available_at uses 19:00+08 from trade_date.",
        "moneyflow_ths": "available_at uses 19:00+08 from trade_date.",
        "moneyflow_ind_dc": "available_at uses 19:00+08 from trade_date.",
        "moneyflow_ind_ths": "available_at uses 19:00+08 from trade_date.",
        "moneyflow_cnt_ths": "available_at uses 19:00+08 from trade_date.",
        "cyq_perf": "available_at uses 19:00+08 from trade_date.",
        "bak_daily": "available_at uses 19:00+08 from trade_date.",
        "block_trade": "available_at uses 21:00+08 from trade_date.",
        "top10_holders": "available_at uses conservative end-of-day from ann_date.",
        "top10_floatholders": "available_at uses conservative end-of-day from ann_date.",
        "pledge_detail": "available_at uses conservative end-of-day from ann_date.",
        "stk_surv": "available_at uses conservative surv_date+5d end-of-day: the source has no announcement column and rows keep landing for days after surv_date (disclosure due within 2 trading days), so surv_date EOD would be lookahead.",
        "new_share": "available_at uses the second A-share trading day after ipo_date at end-of-day: ballot is announced 1-2 trading days after ipo_date, so the whole row is delayed with it.",
        "stk_holdernumber": "available_at uses ann_date end-of-day.",
        "stk_holdertrade": "available_at uses ann_date 19:00+08.",
        "repurchase": "available_at uses ann_date end-of-day.",
        "share_float": "available_at uses ann_date end-of-day; if ann_date is blank, raw layer falls back to float_date and snapshot layer must treat that as conservative event-date availability, not pre-event knowledge.",
    }

def event_partition_prefix(spec: EventDataset) -> str:
    if spec.strategy == "trade_date":
        return "trade_date"
    if spec.strategy == "range_month":
        return "month"
    if spec.strategy == "day":
        return "date"
    raise RuntimeError(f"unsupported event/flow strategy {spec.strategy} for {spec.api_name}")

def audit_margin_exchange_completeness(raw_dir: Path, api_name: str, add) -> None:
    # Every margin table must carry every publishing exchange (SSE+SZSE, plus
    # BSE from 20230213): partial days poison market-wide aggregates or hide a
    # venue's eligibility list, and the downloader refuses to commit them, so
    # any committed partial partition is a data-integrity error needing manual
    # repair. The detail table needs this as much as the summary -- an SSE-only
    # day silently drops ~55% of its rows while still looking like a complete
    # partition -- and margin_secs' BSE slice intermittently lags the vendor.
    column = margin_exchange_column(api_name)
    if column is None:
        return
    incomplete: list[dict[str, Any]] = []
    for path in sorted((raw_dir / api_name).glob("trade_date=*.parquet")):
        trade_date = path.stem.split("=", 1)[1]
        frame = pd.read_parquet(path, columns=[column])
        missing = margin_family_missing_exchanges(api_name, trade_date, frame)
        if missing:
            incomplete.append({"trade_date": trade_date, "missing_exchanges": missing})
    if incomplete:
        add(
            "error",
            f"{api_name}_exchange_completeness",
            f"{api_name} partitions missing required exchange rows: {len(incomplete)} day(s)",
            {"incomplete_days": incomplete[:20], "incomplete_day_count": len(incomplete)},
        )


def audit_event_dataset(raw_dir: Path, spec: EventDataset, expected_paths: set[Path], add) -> None:
    audit_margin_exchange_completeness(raw_dir, spec.api_name, add)
    audit_domain_dataset(raw_dir, spec, expected_paths, add, DomainAuditProfile(
        domain="event",
        exact_limit_rows=frozenset({spec.page_limit, 5000, 6000, 10000}),
        exact_limit_key="exact_common_limit_row_count_partitions",
        apply_pagination_probe=True,
        empty_error_mode="always",
        include_strategy=True,
        key_columns=tuple(spec.key_columns),
        key_extra_columns=("available_at", "available_at_rule"),
        blank_mode="per_key_field",
        pit_rules=event_pit_rules().get(spec.api_name, ""),
        partitions_message="event/flow partition inventory",
        keys_message="key, duplicate, and PIT checks",
        partition_prefix=event_partition_prefix(spec),
        zero_rows_ok=spec.zero_rows_ok,
    ))


def audit_share_float_complete_union(raw_dir: Path, add) -> None:
    union_path = raw_dir / "share_float_complete" / "share_float_complete.parquet"
    meta_path = union_path.with_suffix(union_path.suffix + ".meta.json")
    ann_rescue_files = sorted((raw_dir / "share_float_ann_date_ts_code").rglob("*.parquet"))
    float_rescue_files = sorted((raw_dir / "share_float_float_date_ts_code").rglob("*.parquet"))
    rescue_limit_files: list[str] = []
    rescue_zero_files = 0
    for path in ann_rescue_files + float_rescue_files:
        rows = parquet_rows(path)
        if rows == 0:
            rescue_zero_files += 1
        if rows >= SHARE_FLOAT_ROW_LIMIT:
            rescue_limit_files.append(str(path))

    base_details: dict[str, Any] = {
        "path": str(union_path),
        "ann_date_ts_code_files": len(ann_rescue_files),
        "float_date_ts_code_files": len(float_rescue_files),
        "rescue_zero_files": rescue_zero_files,
        "rescue_limit_files": len(rescue_limit_files),
        "rescue_limit_file_sample": rescue_limit_files[:10],
    }
    if not union_path.exists():
        add("warning", "share_float_complete_union", "share_float complete union file is missing; event_flow audit falls back to raw share_float partitions only", base_details)
        return

    rows = parquet_rows(union_path)
    schema = pq.ParquetFile(union_path).schema_arrow.names
    required_columns = ["ts_code", "ann_date", "float_date", "download_path", "source_file", "source_cap_risk"]
    missing_columns = [column for column in required_columns if column not in schema]
    meta_row_count = None
    meta_error = ""
    if meta_path.exists():
        try:
            meta_row_count = json.loads(meta_path.read_text(encoding="utf-8")).get("row_count")
        except Exception as exc:
            meta_error = str(exc)

    details: dict[str, Any] = {
        **base_details,
        "rows": rows,
        "meta_exists": meta_path.exists(),
        "meta_row_count": meta_row_count,
        "meta_error": meta_error,
        "missing_columns": missing_columns,
    }
    # Business-identity duplication: the same PHYSICAL unlock event (ts_code +
    # float_date + holder + share_type, ann_date excluded, holder whitespace
    # normalized) must not survive under several announcement dates -- the
    # provider re-dates announcements, and a retained stale generation double
    # counts supply. Row/key checks cannot see this class, so it is checked
    # explicitly here; the incremental merge only fixes what its refresh window
    # touches, so historical residue needs a dedicated repair.
    identity_columns = ["ts_code", "ann_date", "float_date", "holder_name", "share_type"]
    # An absent identity column must surface, not silently skip the check --
    # silently passing unverifiable data is the failure mode audits exist for.
    identity_columns_missing = [column for column in identity_columns if column not in schema]
    details["identity_columns_missing"] = identity_columns_missing
    has_identity_columns = not identity_columns_missing
    stat_columns = [column for column in ("download_path", "source_cap_risk", "source_file") if column in schema]
    # One physical read serves both the identity check and the provenance
    # stats; each block keeps its own error attribution.
    read_columns = list(dict.fromkeys((identity_columns if has_identity_columns else []) + stat_columns))
    union_frame: pd.DataFrame | None = None
    union_read_error: Exception | None = None
    if read_columns:
        try:
            union_frame = pd.read_parquet(union_path, columns=read_columns)
        except Exception as exc:
            union_read_error = exc
    if has_identity_columns:
        try:
            if union_frame is None:
                raise union_read_error
            frame = union_frame[identity_columns].copy()
            # Datings are normalized: an UNDATED row (float_date query path, the
            # provider left ann_date empty) must rank below every real date, or
            # str(None)=="None" sorts above them and the informative dated row
            # gets reported as the stale one.
            frame["ann_date"] = core.normalized_date_keys(frame["ann_date"])
            frame["holder_name"] = core.normalized_holder_keys(frame["holder_name"])
            physical = ["ts_code", "float_date", "holder_name", "share_type"]
            datings = frame.groupby(physical, sort=False)["ann_date"].nunique()
            multi = datings[datings > 1]
            details["cross_ann_date_identity_groups"] = int(len(multi))
            if len(multi):
                affected = frame.merge(multi.reset_index()[physical], on=physical, how="inner")
                newest = affected.groupby(physical, sort=False)["ann_date"].transform("max")
                stale = affected["ann_date"] < newest
                details["stale_redated_rows"] = int(stale.sum())
                # The two mechanisms need different repairs: an undated duplicate
                # is provable from the baseline alone, an older REAL dating is not.
                details["undated_duplicate_rows"] = int((stale & affected["ann_date"].eq("")).sum())
                details["cross_ann_date_sample"] = [
                    dict(zip(physical, values)) for values in multi.index[:5]
                ]
        except Exception as exc:
            details["identity_check_error"] = str(exc)
    try:
        if stat_columns and union_frame is None:
            raise union_read_error
        df = union_frame[stat_columns] if union_frame is not None else pd.DataFrame()
        if "download_path" in df.columns:
            details["download_path_counts"] = {str(key): int(value) for key, value in df["download_path"].value_counts(dropna=False).sort_index().items()}
        if "source_file" in df.columns:
            details["input_files_seen"] = int(df["source_file"].nunique(dropna=True))
        if "source_cap_risk" in df.columns:
            risk = df["source_cap_risk"].fillna(False)
            if risk.dtype != bool:
                risk = risk.astype(str).str.lower().isin({"true", "1", "yes"})
            details["source_cap_risk_rows"] = int(risk.sum())
    except Exception as exc:
        details["read_error"] = str(exc)

    severity = "warning" if (
        rows == 0
        or missing_columns
        or identity_columns_missing
        or not meta_path.exists()
        or meta_error
        or details.get("source_cap_risk_rows", 0)
        or rescue_limit_files
        or details.get("cross_ann_date_identity_groups", 0)
        or details.get("identity_check_error")
    ) else "info"
    add(severity, "share_float_complete_union", "share_float complete union and rescue coverage", details)

def share_float_complete_union_exists(raw_dir: Path) -> bool:
    return (raw_dir / "share_float_complete" / "share_float_complete.parquet").exists()

def audit_event_flow_only(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = (repo_root / (args.output or EVENT_FLOW_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    datasets = selected_event_flow_datasets(args)
    filesystem_datasets = [dataset for dataset in datasets if dataset != "share_float" or not share_float_complete_union_exists(raw_dir)]
    if "share_float" in datasets and share_float_complete_union_exists(raw_dir):
        # The union replaces the raw share_float dataset in this audit, but
        # its on-disk inputs and the union itself still deserve the pairing
        # and commit-pair inventory; they were previously invisible to it.
        # The rescue trees are created on first need, so an absent one is a
        # normal state, not a missing dataset (the union audit reports their
        # file counts either way).
        filesystem_datasets = filesystem_datasets + [
            name
            for name in ("share_float_ann_date", "share_float_ann_date_ts_code",
                         "share_float_float_date_ts_code", "share_float_complete")
            if (raw_dir / name).exists()
        ]
    audit_integrated_filesystem(raw_dir, filesystem_datasets, add)
    # Producing jobs close on trading evenings, so month/day expectations
    # clamp to the last SSE open date in the window (mirrors the
    # macro/fundamental clamp; the clamp failure is an error).
    try:
        covered_end = load_sse_open_dates(raw_dir, args.start_date, args.end_date)[-1]
    except RuntimeError as exc:
        covered_end = args.end_date
        add("error", "event_expected_window_unclamped", "SSE trade_cal unavailable; completeness expectations use the raw calendar end date", {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "error": str(exc),
        })
    add("info", "event_flow_expected_scope", "TuShare event/flow datasets included in this audit", {
        "datasets": datasets,
        "filesystem_datasets": filesystem_datasets,
        "share_float_retained_as_union": bool("share_float" in datasets and share_float_complete_union_exists(raw_dir)),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "covered_end_date": covered_end,
        "dataset_pit_rules": event_pit_rules(),
        "dataset_unit_rules": event_unit_rules(),
    })
    for dataset in datasets:
        if dataset == "share_float" and share_float_complete_union_exists(raw_dir):
            continue
        spec = EVENT_FLOW_SPECS[dataset]
        expected = expected_event_paths(raw_dir, spec, args.start_date, covered_end)
        audit_event_dataset(raw_dir, spec, expected, add)
    if "share_float" in datasets:
        audit_share_float_complete_union(raw_dir, add)
    audit_full_market_coverage(raw_dir, datasets, add)

    report = build_quality_report(
        report_type="event_flow",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": datasets,
        },
        findings=findings,
        metadata={
            "unit_rules": event_unit_rules(),
            "pit_rules": event_pit_rules(),
            "doc_refs": {
                dataset: INTEGRATED_DOC_REFS[dataset]
                for dataset in sorted(set(datasets) & set(INTEGRATED_DOC_REFS))
            },
            "conclusions": [
                "Event/flow raw data is sparse by design; zero-row event months or block-trade dates are expected for sparse event sources.",
                "Daily flow tables must still be joined with explicit PIT availability; same-day open decisions cannot use post-close or next-day event/flow values.",
                "share_float raw partitions and the optional share_float_complete union are audited together; exact 6000-row partitions remain source-cap risks.",
                "Raw event rows are not deduplicated; downstream evidence/snapshot layers need deterministic event-key and availability rules.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"event_flow audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def expected_board_paths(raw_dir: Path, spec: BoardTradingDataset, start_date: str, end_date: str, args: argparse.Namespace) -> set[Path]:
    start = max(start_date, spec.start_date)
    if spec.strategy == "static_full":
        return {raw_dir / spec.api_name / f"{spec.api_name}.parquet"}
    if start > end_date:
        return set()
    trade_dates = load_sse_open_dates(raw_dir, start, end_date)
    if spec.strategy == "trade_date":
        return {raw_dir / spec.api_name / f"trade_date={trade_date}.parquet" for trade_date in trade_dates}
    if spec.strategy == "trade_date_by_tag":
        return {
            raw_dir / spec.api_name / f"tag={safe_partition_value(tag)}" / f"trade_date={trade_date}.parquet"
            for tag in selected_board_kpl_tags(args)
            for trade_date in trade_dates
        }
    if spec.strategy == "trade_date_by_limit_type":
        return {
            raw_dir / spec.api_name / f"limit_type={safe_partition_value(limit_type)}" / f"trade_date={trade_date}.parquet"
            for limit_type in selected_board_ths_limit_types(args)
            for trade_date in trade_dates
        }
    if spec.strategy == "trade_date_by_market":
        return {
            raw_dir / spec.api_name / f"market={safe_partition_value(market)}" / f"is_new={is_new}" / f"trade_date={trade_date}.parquet"
            for market in selected_board_ths_hot_markets(args)
            for is_new in selected_board_hot_is_new(args)
            for trade_date in trade_dates
        }
    if spec.strategy == "trade_date_by_market_hot_type":
        return {
            raw_dir / spec.api_name / f"market={safe_partition_value(market)}" / f"hot_type={safe_partition_value(hot_type)}" / f"is_new={is_new}" / f"trade_date={trade_date}.parquet"
            for market in selected_board_dc_hot_markets(args)
            for hot_type in selected_board_dc_hot_types(args)
            for is_new in selected_board_hot_is_new(args)
            for trade_date in trade_dates
        }
    raise RuntimeError(f"unsupported board-trading strategy {spec.strategy} for {spec.api_name}")

def audit_board_dataset(raw_dir: Path, spec: BoardTradingDataset, expected_paths: set[Path], add) -> None:
    audit_domain_dataset(raw_dir, spec, expected_paths, add, DomainAuditProfile(
        domain="board",
        exact_limit_rows=frozenset({spec.page_limit}),
        exact_limit_key="exact_page_limit_row_count_partitions",
        apply_pagination_probe=True,
        empty_error_mode="always",
        include_strategy=True,
        key_columns=tuple(spec.key_columns),
        key_extra_columns=("available_at", "available_at_rule", spec.date_column, spec.time_column),
        blank_mode="available_at_rows",
        pit_rules=board_pit_rules().get(spec.api_name, {}),
        partitions_message="board-trading partition inventory",
        keys_message="board-trading key and PIT checks",
        availability_required=spec.strategy != "static_full",
        # Flat trade-date datasets get the trailing-zero stale-feed detector;
        # nested tag/market strategies stay opted out (an individual pool or
        # tag legitimately goes quiet for long stretches).
        zero_rows_ok=spec.zero_rows_ok if spec.strategy == "trade_date" else None,
    ))


def board_pit_rules() -> dict[str, dict[str, str]]:
    return {
        "kpl_list": {"available_at": "official next-day 08:30 from trade_date", "usage": "next-day board sentiment/evidence; no same-day intraday lookahead"},
        "kpl_concept_cons": {"available_at": "official next-day 08:30 from trade_date", "usage": "next-day concept membership/heat evidence; no same-day intraday lookahead"},
        "dc_index": {"available_at": "official 20:00 from trade_date", "usage": "post-close board-index rotation evidence"},
        "dc_member": {"available_at": "official 20:00 from trade_date", "usage": "post-close board membership map"},
        "limit_step": {"available_at": "conservative trade-date end-of-day", "usage": "market height and limit-up ladder after close"},
        "limit_cpt_list": {"available_at": "conservative trade-date end-of-day", "usage": "topic strength and limit-up board rotation after close"},
        "limit_list_d": {"available_at": "official 16:00 from trade_date (row-level available_at column)", "usage": "post-close official limit-up/down/broken labels; execution constraints come from stk_limit, not this table"},
        "limit_list_ths": {"available_at": "official around 16:00 from trade_date", "usage": "post-close THS limit-up/down pool evidence; no same-day intraday lookahead"},
        "top_list": {"available_at": "official 20:00 from trade_date", "usage": "next-day Dragon-Tiger list evidence"},
        "top_inst": {"available_at": "official 20:00 from trade_date", "usage": "next-day institutional seat evidence"},
        "hm_list": {"available_at": "daily-refreshed reference list without PIT timestamps; do not use as historical time-series signal without hm_detail rows"},
        "hm_detail": {"available_at": "conservative trade-date end-of-day", "usage": "next-day hot-money seat evidence"},
        "ths_hot": {"available_at": "rank_time when returned; is_new=Y falls back to 22:30", "usage": "intraday/evening hot-list evidence by observable rank_time"},
        "dc_hot": {"available_at": "rank_time when returned; is_new=Y falls back to 22:30", "usage": "intraday/evening hot-list evidence by observable rank_time"},
    }

def board_unit_rules() -> dict[str, list[dict[str, object]]]:
    """Projection of the shared unit registry (environment/data/units.py) over
    all BOARD_TRADING_DATASETS."""
    return dataset_rules_records(tuple(BOARD_TRADING_DATASETS))

def audit_board_trading_only(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    output = (repo_root / (args.output or BOARD_TRADING_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    datasets = selected_board_trading_datasets(args)
    audit_integrated_filesystem(raw_dir, datasets, add)
    add("info", "board_trading_expected_scope", "TuShare board-trading datasets included in this audit", {
        "datasets": datasets,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "kpl_tags": selected_board_kpl_tags(args),
        "ths_limit_types": selected_board_ths_limit_types(args),
        "ths_hot_markets": selected_board_ths_hot_markets(args),
        "dc_hot_markets": selected_board_dc_hot_markets(args),
        "dc_hot_types": selected_board_dc_hot_types(args),
        "hot_is_new": selected_board_hot_is_new(args),
        "dataset_pit_rules": board_pit_rules(),
        "dataset_unit_rules": board_unit_rules(),
    })
    for dataset in datasets:
        spec = BOARD_TRADING_SPECS[dataset]
        expected = expected_board_paths(raw_dir, spec, args.start_date, args.end_date, args)
        audit_board_dataset(raw_dir, spec, expected, add)

    report = build_quality_report(
        report_type="board_trading",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": datasets,
        },
        findings=findings,
        metadata={
            "unit_rules": board_unit_rules(),
            "pit_rules": board_pit_rules(),
            "doc_refs": {
                dataset: INTEGRATED_DOC_REFS[dataset]
                for dataset in sorted(set(datasets) & set(INTEGRATED_DOC_REFS))
            },
            "conclusions": [
                "Board-trading raw data is a dedicated sentiment/event evidence domain for limit-up, ladder, topic, Dragon-Tiger, hot-money, and hot-list signals.",
                "Most board-trading datasets are only valid after close or the next morning; intraday usage must rely on rank_time or a documented observable timestamp.",
                "These raw rows complement limit_list_d and minute-derived labels; they do not replace PIT execution constraints built from stk_limit and 1-minute bars.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"board_trading audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def audit_daily_direct(raw_dir: Path, args: argparse.Namespace, add) -> set[str]:
    try:
        trade_dates = set(load_sse_open_dates(raw_dir, args.start_date, args.end_date))
    except Exception as exc:
        add("error", "daily_trade_calendar", str(exc))
        return set()
    for dataset in selected_daily_datasets(args, default=DAILY_REQUIRED_DATASETS):
        spec = DAILY_SPECS[dataset]
        expected = {d for d in trade_dates if max(args.start_date, spec.start_date) <= d <= args.end_date}
        audit_trade_date_dataset(raw_dir, spec, expected, add)
    return trade_dates

def resolve_audit_end_date(raw_dir: Path, requested: str | None) -> str:
    """The audit window end: an explicit request, else the last SETTLED date.

    This audit answers "is the settled history complete?", so its window ends at
    the last trading day whose data has had its ingestion window -- the most
    recent Asia/Shanghai date strictly before today. Two earlier bounds were
    both wrong in the same direction: the SSE calendar's maximum reaches days
    that have not happened at all, and today's own date reaches a session that
    has not closed (and is not ingested until the evening cron). Either way the
    audit reported missing partitions that could not exist, so it never reached
    ok and a real gap -- an outage, a refused revision -- was indistinguishable
    from permanent noise.

    Whether TODAY's ingestion succeeded is the cron's question, and the cron
    records and alerts on its own failure; conflating the two is what made this
    audit unreadable.
    """
    if requested:
        return requested
    settled = (datetime.now(CN_TZ) - timedelta(days=1)).strftime("%Y%m%d")
    return min(latest_sse_calendar_date(raw_dir), settled)


def audit_core_market(args: argparse.Namespace) -> int:
    """Reference tables plus daily quotes/constraints: the execution-critical
    inputs every experiment consumes. Raw financial data has its own report
    (fundamental-raw) so its failures cannot block fundamentals-disabled
    experiments; limit_list_d lives entirely on the board tier."""
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    args.end_date = resolve_audit_end_date(raw_dir, args.end_date)
    output = (repo_root / (args.output or CORE_MARKET_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    daily_datasets = selected_daily_datasets(args, default=DAILY_REQUIRED_DATASETS)
    datasets = REFERENCE_DATASETS + daily_datasets
    audit_integrated_filesystem(raw_dir, datasets, add)

    stock_basic = audit_stock_basic(raw_dir, add)
    audit_stock_company(raw_dir, stock_basic, add)
    sse_open = audit_trade_cal(raw_dir, add)
    audit_bak_basic(raw_dir, sse_open, args.end_date, add)
    audit_namechange(raw_dir, stock_basic, add)
    audit_ths_membership(raw_dir, add)
    classify = audit_index_classify(raw_dir, add)
    audit_index_member_all(raw_dir, classify["SW2021"], stock_basic, add)
    audit_index_member_history(raw_dir, classify["SW2014"], add)
    audit_index_weight(raw_dir, args.end_date, add)
    audit_full_market_coverage(raw_dir, ["bak_basic"] + daily_datasets, add)

    trade_dates = audit_daily_direct(raw_dir, args, add)
    all_codes = audit_daily_cross_coverage(raw_dir, trade_dates, args, add) if trade_dates else {"daily": set(), "daily_basic": set(), "adj_factor": set(), "stk_limit": set()}
    audit_unit_schema(raw_dir, add)
    audit_stock_universe_semantics(raw_dir, all_codes, add)
    audit_pit_availability(raw_dir, add)

    report = build_quality_report(
        report_type="core_market",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": datasets,
        },
        findings=findings,
        metadata={
            "unit_rules": core_market_unit_rules(),
            "doc_refs": {
                dataset: INTEGRATED_DOC_REFS[dataset]
                for dataset in sorted(set(datasets) & set(INTEGRATED_DOC_REFS))
            },
            "conclusions": [
                "Core-market audit covers reference tables and daily market quotes/constraints: the execution-critical inputs of every experiment.",
                "Files are structurally usable when errors are zero, but source and semantic warnings require PIT-aware snapshot construction.",
                "bak_basic and bak_daily are supplemental snapshots; neither should replace daily/daily_basic as the main daily market data source.",
                "Do not compare amount, market value, or share fields across interfaces until each field is normalized to a common unit.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"core_market audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0


def audit_fundamental_raw(args: argparse.Namespace) -> int:
    """Raw financial statement/indicator/forecast/disclosure completeness and
    semantics. Enforced by the snapshot only when fundamentals are enabled;
    the derived PIT event index has its own independent report."""
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    args.end_date = resolve_audit_end_date(raw_dir, args.end_date)
    output = (repo_root / (args.output or FUNDAMENTAL_RAW_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    datasets = selected_fundamental_datasets(getattr(args, "fundamental_datasets", None))
    audit_integrated_filesystem(raw_dir, datasets, add)
    audit_fundamental_completeness(raw_dir, args, add)
    audit_fundamental_unit_and_pit_semantics(raw_dir, add)

    report = build_quality_report(
        report_type="fundamental_raw",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "datasets": datasets,
        },
        findings=findings,
        metadata={
            "unit_rules": fundamental_unit_rules(),
            "doc_refs": {
                dataset: INTEGRATED_DOC_REFS[dataset]
                for dataset in sorted(set(datasets) & set(INTEGRATED_DOC_REFS))
            },
            "windows": {
                "fundamental": {
                    "start_date": args.fundamental_start_date,
                    "end_date": args.fundamental_end_date or args.end_date,
                },
            },
            "conclusions": [
                "Raw financial records are intentionally not deduplicated; the PIT event layer selects visible versions (revisions ranked by update_flag).",
                "Raw-finance findings gate only fundamentals-enabled experiments; the derived PIT index carries its own enforced report.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"fundamental_raw audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0

def audit_intraday_only(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    if getattr(args, "intraday_end_date", None) is None:
        args.intraday_end_date = date.today().strftime("%Y%m%d")
    output = (repo_root / (args.output or INTRADAY_MINUTES_STATUS_PATH)).resolve()
    findings: list[dict[str, Any]] = []

    def add(severity: str, check: str, message: str, details: dict[str, Any] | None = None) -> None:
        findings.append({"severity": severity, "check": check, "message": message, "details": details or {}})

    intraday_datasets = selected_intraday_datasets(getattr(args, "intraday_datasets", None))
    audit_integrated_filesystem(raw_dir, intraday_datasets, add)
    if STK_MINS_DATASET in intraday_datasets:
        audit_stk_mins_completeness(raw_dir, args, add)

    report = build_quality_report(
        report_type="intraday_minutes",
        scope={
            "data_root": str(raw_dir),
            "start_date": args.intraday_start_date,
            "end_date": args.intraday_end_date,
            "datasets": intraday_datasets,
            "intraday_codes": getattr(args, "intraday_codes", None),
            "intraday_max_codes": getattr(args, "intraday_max_codes", None),
        },
        findings=findings,
        metadata={
            "unit_rules": {
                STK_MINS_DATASET: {
                    **column_source_units("intraday_1min.parquet"),
                    "available_at": "source trade_time in Asia/Shanghai; use as bar-close availability",
                    "auction_bars": "opening and closing auction are represented by 09:30 and 15:00 1-minute bars; no separate auction dataset is required for historical intraday minute",
                }
            },
            "doc_refs": {STK_MINS_DATASET: INTEGRATED_DOC_REFS[STK_MINS_DATASET]},
            "conclusions": [
                "Intraday minute data is stored as stock-year Parquet partitions and sidecar metadata under data/raw/stk_mins_1min.",
                "TuShare stk_mins uses shares for vol and CNY for amount; do not mix it with daily.amount or bak_daily.amount without unit conversion.",
                "Before stk_auction coverage begins, 09:30 minute rows remain the explicitly labelled opening-auction proxy; 15:00 close is the closing-auction clearing price.",
            ],
        },
    )
    counts = report["finding_counts"]
    status = report["status"]
    write_quality_report(output, report)
    print(f"intraday audit status={status} errors={counts['error']} warnings={counts['warning']} output={output}")
    return 1 if counts["error"] else 0


def add_core_market_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("core-market", help="audit reference tables and daily market quotes/constraints")
    core.add_raw_arg(parser)
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--end-date")
    parser.add_argument("--datasets", nargs="+", choices=core.DAILY_REQUIRED_DATASETS)
    parser.add_argument("--sample-limit", type=int, default=10)
    core.add_runtime_args(parser, min_interval=0.18, timeout=60)
    parser.add_argument("--output", help=f"Defaults to {core.CORE_MARKET_STATUS_PATH}.")


def add_fundamental_raw_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("fundamental-raw", help="audit raw financial statement/indicator/forecast/disclosure data")
    core.add_raw_arg(parser)
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--end-date")
    parser.add_argument("--fundamental-start-date", default="20100101")
    parser.add_argument("--fundamental-end-date")
    parser.add_argument("--fundamental-datasets", nargs="+", choices=core.FUNDAMENTAL_DATASETS, dest="fundamental_datasets")
    parser.add_argument("--sample-limit", type=int, default=10)
    core.add_runtime_args(parser, min_interval=0.18, timeout=60)
    parser.add_argument("--output", help=f"Defaults to {core.FUNDAMENTAL_RAW_STATUS_PATH}.")

def add_intraday_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("intraday", help="audit stock-year intraday minute raw data")
    core.add_raw_arg(parser)
    parser.add_argument("--intraday-start-date", default="20200101")
    parser.add_argument("--intraday-end-date")
    parser.add_argument("--intraday-datasets", nargs="+", choices=core.INTRADAY_DATASETS + ["stk_mins"])
    parser.add_argument("--intraday-codes", nargs="+")
    parser.add_argument("--intraday-max-codes", type=int)
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--output", help=f"Defaults to {core.INTRADAY_MINUTES_STATUS_PATH}.")

    by_date = sub.add_parser("intraday-by-date", help="audit final full-market daily minute files")
    core.add_intraday_by_date_common_args(by_date)
    by_date.add_argument("--full-scan", action="store_true")
    by_date.add_argument("--sample-limit", type=int, default=20)
    by_date.add_argument("--output", help=f"Defaults to {core.INTRADAY_MINUTES_STATUS_PATH}.")

    auction = sub.add_parser("auction-alignment", help="compare 09:30 minute auction bars with stk_auction and daily units")
    core.add_raw_arg(auction)
    auction.add_argument("--start-date", required=True)
    auction.add_argument("--end-date", required=True)
    auction.add_argument("--output-dataset", default=core.STK_MINS_BY_DATE_DATASET)
    auction.add_argument("--max-trade-dates", type=int, default=8, help="Use the latest N open dates in the requested window; <=0 means all.")
    auction.add_argument("--output", help="Defaults to results/data_quality/process/auction_alignment_status.json.")
    core.add_runtime_args(auction, min_interval=0.25, timeout=120)

def add_event_macro_parsers(sub: argparse._SubParsersAction) -> None:
    event = sub.add_parser("event-flow", help="audit only event/flow raw data")
    core.add_raw_arg(event)
    event.add_argument("--start-date", default="20200101")
    event.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    event.add_argument("--datasets", nargs="+", choices=core.EVENT_FLOW_DATASETS)
    event.add_argument("--output", help=f"Defaults to {core.EVENT_FLOW_STATUS_PATH}.")

    macro = sub.add_parser("macro", help="audit macro, policy, and global-context raw data")
    core.add_raw_arg(macro)
    macro.add_argument("--start-date", default="20100101")
    macro.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    macro.add_argument("--datasets", nargs="+", choices=core.MACRO_DATASETS)
    core.add_macro_filter_args(macro)
    macro.add_argument("--output", help=f"Defaults to {core.MACRO_CONTEXT_STATUS_PATH}.")

    text = sub.add_parser("text", help="audit only text-evidence raw data")
    core.add_raw_arg(text)
    text.add_argument("--start-date", dest="text_start_date", default="20100101")
    text.add_argument("--end-date", dest="text_end_date", default=date.today().strftime("%Y%m%d"))
    text.add_argument("--text-datasets", nargs="+", choices=core.TEXT_DATASETS, dest="text_datasets")
    text.add_argument("--news-src", action="append", default=[])
    text.add_argument("--major-news-src", action="append", default=[])
    text.add_argument("--output", help=f"Defaults to {core.TEXT_EVIDENCE_STATUS_PATH}.")

def add_board_parser(sub: argparse._SubParsersAction) -> None:
    board = sub.add_parser("board-trading", help="audit 打板专题 raw data")
    core.add_raw_arg(board)
    board.add_argument("--start-date", default="20200101")
    board.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    board.add_argument("--datasets", nargs="+", choices=core.BOARD_TRADING_DATASETS)
    core.add_board_filter_args(board)
    board.add_argument("--output", help=f"Defaults to {core.BOARD_TRADING_STATUS_PATH}.")

def add_revision_parser(sub: argparse._SubParsersAction) -> None:
    revision = sub.add_parser("revision-sentinel", help="sample TuShare source partitions and compare them with local raw data without overwriting raw files")
    core.add_raw_arg(revision)
    revision.add_argument("--start-date", default="20200101")
    revision.add_argument("--end-date", default=date.today().strftime("%Y%m%d"))
    revision.add_argument(
        "--datasets",
        nargs="+",
        choices=core.DAILY_REQUIRED_DATASETS
        + sorted(name for name, spec in core.BOARD_TRADING_SPECS.items() if spec.strategy == "trade_date")
        + sorted(name for name, spec in core.EVENT_FLOW_SPECS.items() if spec.strategy == "trade_date"),
    )
    revision.add_argument("--sample-size", type=int, default=12, help="Deterministic sample size per dataset; <=0 checks all dates.")
    revision.add_argument("--seed", help="Deterministic sampling seed. Defaults to --end-date.")
    revision.add_argument("--page-limit", type=int, default=core.TRADE_DATE_PAGE_LIMIT)
    revision.add_argument("--revision-ledger", default=core.REVISION_EVENTS_PATH)
    revision.add_argument("--output", default=core.REVISION_SUMMARY_PATH)
    revision.add_argument("--fail-on-revision", action="store_true", help="Return nonzero when source revisions are found.")
    core.add_runtime_args(revision, min_interval=0.22, timeout=120)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_core_market_parser(sub)
    add_fundamental_raw_parser(sub)
    add_intraday_parser(sub)
    add_event_macro_parsers(sub)
    add_board_parser(sub)
    add_revision_parser(sub)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.command == "core-market":
        return audit_core_market(args)
    if args.command == "fundamental-raw":
        return audit_fundamental_raw(args)
    if args.command == "intraday":
        return audit_intraday_only(args)
    if args.command == "intraday-by-date":
        return audit_intraday_by_date(args)
    if args.command == "auction-alignment":
        return audit_auction_alignment(args)
    if args.command == "event-flow":
        return audit_event_flow_only(args)
    if args.command == "macro":
        return audit_macro_only(args)
    if args.command == "text":
        return audit_text_only(args)
    if args.command == "board-trading":
        return audit_board_trading_only(args)
    if args.command == "revision-sentinel":
        return audit_revision_sentinel(args)
    raise RuntimeError(f"unknown command {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())
