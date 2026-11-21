"""Raw lake builders for the snapshot tests: the vendor tables a PIT snapshot
is built from, written the way the TuShare download leaves them."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from autotrade.environment.data.fundamental_events import FUNDAMENTAL_EVENT_DATASETS
from autotrade.environment.data.quality import build_quality_report


def write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def build_raw(raw: Path) -> None:
    calendar = pd.DataFrame({"cal_date": ["20210930", "20211008", "20211011"], "is_open": ["1", "1", "1"]})
    write(raw / "trade_cal" / "exchange=SSE" / "year=2021.parquet", calendar)
    # Vendor raw schema for the fundamentals dataset: the builder attributes
    # fundamentals unit-reference columns from these footers.
    write(
        raw / "income_vip" / "period=20210630.parquet",
        pd.DataFrame(
            [{"ts_code": "000001.SZ", "ann_date": "20210910", "end_date": "20210630", "basic_eps": 0.5, "revenue": 1_000_000.0, "n_income": 100_000.0}]
        ),
    )
    for trade_date in ("20210930", "20211008"):
        write(
            raw / "daily" / f"trade_date={trade_date}.parquet",
            pd.DataFrame(
                [{"trade_date": trade_date, "ts_code": "000001.SZ", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "pre_close": 10.0, "pct_chg": 5.0, "vol": 1000.0, "amount": 1050.0}]
            ),
        )
        write(
            raw / "daily_basic" / f"trade_date={trade_date}.parquet",
            pd.DataFrame([
                {"trade_date": trade_date, "ts_code": "000001.SZ", "turnover_rate": 2.0, "dv_ttm": 3.0, "pe": 10.0,
                 "total_share": 100.0, "total_mv": 1000.0, "close": 10.5, "circ_mv": 2_000_000.0},
                {"trade_date": trade_date, "ts_code": "000010.SZ", "turnover_rate": 1.0, "dv_ttm": 1.5, "pe": 30.0,
                 "total_share": 10.0, "total_mv": 100.0, "close": 3.2, "circ_mv": 80_000.0},
            ]),
        )
        write(
            raw / "stk_limit" / f"trade_date={trade_date}.parquet",
            pd.DataFrame([{"trade_date": trade_date, "ts_code": "000001.SZ", "up_limit": 11.55, "down_limit": 9.45}]),
        )
        write(
            raw / "adj_factor" / f"trade_date={trade_date}.parquet",
            pd.DataFrame([{"trade_date": trade_date, "ts_code": "000001.SZ", "adj_factor": 1.0}]),
        )
        write(
            raw / "suspend_d" / f"trade_date={trade_date}.parquet",
            pd.DataFrame(columns=["trade_date", "ts_code", "suspend_timing", "suspend_type"]),
        )
        write(
            raw / "stk_auction" / f"trade_date={trade_date}.parquet",
            pd.DataFrame([{
                "ts_code": "000001.SZ", "trade_date": trade_date, "price": 10.02,
                "vol": 30000.0, "amount": 300600.0, "pre_close": 10.0,
                "turnover_rate": 0.01, "volume_ratio": 1.2, "float_share": 300000.0,
            }]),
        )
    write(
        raw / "stk_mins_1min_by_date" / "trade_date=20210930.parquet",
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_time": "2021-09-30 09:30:00", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "vol": 20000.0, "amount": 200000.0, "trade_date": "20210930", "available_at": "2021-09-30T09:30:00+08:00", "available_at_rule": "bar_close"},
            ]
        ),
    )
    write(
        raw / "margin_secs" / "trade_date=20211008.parquet",
        pd.DataFrame([{"trade_date": "20211008", "ts_code": "000001.SZ", "available_at": "2021-10-08T09:00:00+08:00", "available_at_rule": "same_day_preopen"}]),
    )
    write(
        raw / "moneyflow" / "trade_date=20211008.parquet",
        pd.DataFrame([{"trade_date": "20211008", "ts_code": "000001.SZ", "net_mf_amount": 1.0, "available_at": "2021-10-08T19:00:00+08:00", "available_at_rule": "same_day_evening"}]),
    )
    write(
        raw / "cn_gdp" / "range=2020Q1_2021Q4.parquet",
        pd.DataFrame(
            [
                {"quarter": "2021Q2", "gdp": 1.0, "available_at": "2021-07-15T10:00:00+08:00", "available_at_rule": "release"},
                {"quarter": "2021Q3", "gdp": 1.1, "available_at": "2021-10-18T10:00:00+08:00", "available_at_rule": "release"},
            ]
        ),
    )
    write(
        raw / "cctv_news" / "date=20211007.parquet",
        pd.DataFrame([{"date": "20211007", "title": "新闻联播标题", "content": "正文内容", "available_at": "2021-10-07T19:30:00+08:00", "available_at_rule": "evening"}]),
    )
    write(
        raw / "stock_basic" / "list_status=L.parquet",
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "平安银行", "exchange": "SZSE", "list_date": "19910403", "delist_date": None},
                # ST at the decision day + small cap: universe-screening fixture.
                {"ts_code": "000010.SZ", "name": "ST美丽", "exchange": "SZSE", "list_date": "19951027", "delist_date": None},
            ]
        ),
    )
    write(
        raw / "stock_basic" / "list_status=D.parquet",
        pd.DataFrame(
            [
                # Delisted AFTER the decision day: must stay in the as-of universe.
                {"ts_code": "000005.SZ", "name": "世纪星源", "exchange": "SZSE", "list_date": "19901210", "delist_date": "20240426"},
                # Delisted BEFORE the decision day: must be excluded.
                {"ts_code": "000003.SZ", "name": "PT金田A", "exchange": "SZSE", "list_date": "19910703", "delist_date": "20020614"},
            ]
        ),
    )
    write(
        raw / "namechange" / "namechange.parquet",
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20120801", "end_date": "", "ann_date": "20120730", "change_reason": "改名"},
                {"ts_code": "000005.SZ", "name": "世纪星源", "start_date": "19901210", "end_date": "", "ann_date": "19901210", "change_reason": "上市"},
                # Renamed AFTER the decision day: the as-of universe must keep 世纪星源.
                {"ts_code": "000005.SZ", "name": "ST星源", "start_date": "20230601", "end_date": "", "ann_date": "20230525", "change_reason": "ST"},
                {"ts_code": "000010.SZ", "name": "ST美丽", "start_date": "20200101", "end_date": "", "ann_date": "20191230", "change_reason": "ST"},
            ]
        ),
    )


def write_index_daily(raw: Path, trade_dates: tuple[str, ...]) -> None:
    """CSI300 index bars carrying the ingest adapter's raw date-EOD placeholder."""
    write(
        raw / "index_daily" / "ts_code=000300.SH" / "year=2021.parquet",
        pd.DataFrame([
            {"ts_code": "000300.SH", "trade_date": trade_date, "close": 4800.0 + index,
             "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 23:59:59+08:00",
             "available_at_rule": "conservative_date_eod"}
            for index, trade_date in enumerate(trade_dates)
        ]),
    )


def write_dividend_history(root: Path) -> None:
    """A PIT dividend store reaching back before every fixture replay slot.

    A replay refuses a slot the store does not precede (DividendHistoryError).
    The one implemented event went ex long before the slots, so their
    corporate-action tables stay empty."""
    write(
        root / "dividend" / "available_month=202101.parquet",
        pd.DataFrame(
            [
                {
                    "dataset": "dividend", "ts_code": "000001.SZ", "end_date": "20191231",
                    "div_proc": "实施", "ex_date": "20210120", "record_date": "20210119",
                    "pay_date": "20210120", "div_listdate": None, "cash_div": 0.1,
                    "cash_div_tax": 0.1, "stk_div": None, "stk_bo_rate": None, "stk_co_rate": None,
                    "available_at": "2021-01-12T18:00:00+08:00",
                    "available_at_rule": "source:imp_ann_date_or_ann_date", "available_month": "202101",
                    "business_key": "d0", "source_path": "x", "source_write_id": "w", "source_row_id": 0,
                },
            ]
        ),
    )


def build_fundamental_events(root: Path) -> None:
    write(
        root / "income_vip" / "available_month=202109.parquet",
        pd.DataFrame(
            [
                {"dataset": "income_vip", "ts_code": "000001.SZ", "available_at": "2021-09-10T18:00:00+08:00", "available_at_rule": "source:f_ann_date_or_ann_date", "available_month": "202109", "business_key": "k1", "source_path": "x", "source_write_id": "w", "source_row_id": 0},
            ]
        ),
    )
    # Every other month of 2021 was built without income_vip events.
    write_event_month_markers(
        root, "income_vip", (f"2021{month:02d}" for month in range(1, 13) if month != 9)
    )
    write_dividend_history(root)


def write_fundamental_status(
    path: Path,
    *,
    status: str = "ok",
    errors: int = 0,
    scope_start_date: str = "20200101",
    scope_datasets: tuple[str, ...] = FUNDAMENTAL_EVENT_DATASETS,
) -> None:
    write_quality_status(
        path,
        "fundamental_events",
        status="error" if errors else status,
        scope_start_date=scope_start_date,
        scope_datasets=scope_datasets,
    )
    write_domain_statuses(path.parent)


def write_event_month_markers(root: Path, dataset: str, months: Iterable[str]) -> None:
    """Zero-row partitions, as the PIT event builder writes a month it built
    without events; a replay slot refuses a month the store never built
    (FundamentalCoverageError)."""

    columns = (
        "dataset", "ts_code", "available_at", "available_at_rule", "available_month",
        "business_key", "source_path", "source_write_id", "source_row_id",
    )
    for month in months:
        write(
            root / dataset / f"available_month={month}.parquet",
            pd.DataFrame({column: pd.Series(dtype="object") for column in columns}),
        )


def write_quality_status(
    path: Path,
    report_type: str,
    *,
    status: str = "ok",
    created_at: str = "2026-01-01T00:00:00+00:00",
    scope_start_date: str = "20200101",
    scope_datasets: tuple[str, ...] = ("fixture",),
    checks: tuple[str, ...] = ("fixture_status",),
) -> None:
    findings = []
    if status != "ok":
        findings = [
            {
                "severity": status,
                "check": check,
                "message": "fixture quality status",
                "details": {},
            }
            for check in checks
        ]
    report = build_quality_report(
        report_type=report_type,
        scope={
            "data_root": "/fixture/raw",
            "start_date": scope_start_date,
            "end_date": "20260101",
            "datasets": list(scope_datasets),
        },
        findings=findings,
        created_at=created_at,
    )
    path.write_text(json.dumps(report), encoding="utf-8")


def write_domain_statuses(quality_dir: Path, **overrides: str) -> None:
    """The per-domain audit statuses the snapshot gates read (default all ok)."""
    for domain, filename, report_type in (
        ("daily", "core_market_status.json", "core_market"),
        ("intraday_1min", "intraday_minutes_status.json", "intraday_minutes"),
        ("fundamentals_raw", "fundamental_raw_status.json", "fundamental_raw"),
        ("events", "event_flow_status.json", "event_flow"),
        ("board_trading", "board_trading_status.json", "board_trading"),
        ("macro", "macro_context_status.json", "macro_context"),
        ("text", "text_evidence_status.json", "text_evidence"),
    ):
        quality_dir.mkdir(parents=True, exist_ok=True)
        write_quality_status(
            quality_dir / filename,
            report_type,
            status=overrides.get(domain, "ok"),
        )
