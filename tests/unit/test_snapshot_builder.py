import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import pytest

from autotrade.data_quality import build_quality_report
from autotrade.environment.data import snapshot as snapshot_module
from autotrade.environment.data.fundamental_events import (
    FUNDAMENTAL_EVENT_DATASETS,
    read_fundamental_events,
)
from autotrade.environment.data.snapshot import (
    SnapshotBuilder,
    SnapshotConfig,
    finalize_snapshot_dir,
    load_snapshot_manifest,
)

CN_TZ = ZoneInfo("Asia/Shanghai")
DECISION = datetime(2021, 10, 8, 9, 25, tzinfo=CN_TZ)


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
        write(raw / "suspend_d" / f"trade_date={trade_date}.parquet", pd.DataFrame(columns=["trade_date", "ts_code"]))
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


def build_fundamental_events(root: Path) -> None:
    write(
        root / "income_vip" / "available_month=202109.parquet",
        pd.DataFrame(
            [
                {"dataset": "income_vip", "ts_code": "000001.SZ", "available_at": "2021-09-10T18:00:00+08:00", "available_at_rule": "source:f_ann_date_or_ann_date", "available_month": "202109", "business_key": "k1", "source_path": "x", "source_write_id": "w", "source_row_id": 0},
            ]
        ),
    )


def write_quality_status(
    path: Path,
    report_type: str,
    *,
    status: str = "ok",
    created_at: str = "2026-01-01T00:00:00+00:00",
    scope_start_date: str = "20200101",
    scope_datasets: tuple[str, ...] = ("fixture",),
) -> None:
    findings = []
    if status != "ok":
        findings.append(
            {
                "severity": status,
                "check": "fixture_status",
                "message": "fixture quality status",
                "details": {},
            }
        )
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


CONFIG = SnapshotConfig(
    events_datasets=("margin_secs", "moneyflow"),
    macro_datasets=("cn_gdp",),
    text_datasets=("cctv_news",),
    fundamental_datasets=("income_vip",),
    intraday_trade_days=1,
    include_industry=False,
)


class SnapshotBuilderTest(unittest.TestCase):
    def test_domain_scheduler_bounds_concurrency_and_respects_dependencies(self):
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        finished: set[str] = set()

        def task(name: str, required: str | None = None):
            def build(completed):
                nonlocal active, maximum_active
                if required is not None:
                    self.assertIn(required, completed)
                    self.assertIn(required, finished)
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                    finished.add(name)
                return {"name": name}, {}

            return build

        results = snapshot_module._run_domain_tasks(
            [
                ("daily", (), task("daily")),
                ("intraday", ("daily",), task("intraday", "daily")),
                ("events", (), task("events")),
                ("text", (), task("text")),
            ]
        )

        self.assertEqual(set(results), {"daily", "intraday", "events", "text"})
        self.assertEqual(maximum_active, snapshot_module.SNAPSHOT_DOMAIN_WORKERS)

    def test_replay_minutes_stream_daily_partitions_into_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            minute_dir = raw / "stk_mins_1min_by_date"
            rows = []
            for trade_date, code in (("20211008", "000001.SZ"), ("20211011", "600000.SH")):
                row = {
                    "ts_code": code,
                    "trade_time": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 09:30:00",
                    "open": 10.0,
                    "high": 10.1,
                    "low": 9.9,
                    "close": 10.0,
                    "vol": 1000.0,
                    "amount": 10000.0,
                    "trade_date": trade_date,
                    "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T09:30:00+08:00",
                    "available_at_rule": "bar_close",
                }
                rows.append(row)
                write(minute_dir / f"trade_date={trade_date}.parquet", pd.DataFrame([row]))
            output = Path(tmp) / "intraday_1min.parquet"

            meta, profile = SnapshotBuilder(raw, Path(tmp) / "missing_events")._write_minutes_range(
                "20211008", "20211011", output, None
            )

            actual = pd.read_parquet(output)
            self.assertEqual(actual["trade_date"].tolist(), ["20211008", "20211011"])
            self.assertEqual(actual["auction_market_bucket"].tolist(), ["sz_main_00", "sh_main_60"])
            self.assertAlmostEqual(actual.loc[0, "vol_pit"], 760.0)
            self.assertAlmostEqual(actual.loc[1, "vol_pit"], 1000.0)
            self.assertEqual(pq.ParquetFile(output).metadata.num_row_groups, 2)
            self.assertEqual(meta, {"rows": 2, "datasets": ["stk_mins_1min_by_date"], "files": 2})
            self.assertEqual(profile["rows"], 2)
            self.assertEqual(profile["date_ranges"]["trade_date"], {"min": "20211008", "max": "20211011"})

    def test_replay_minute_prefetch_is_bounded_and_writes_in_partition_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            minute_dir = raw / "stk_mins_1min_by_date"
            trade_dates = [f"202110{day:02d}" for day in range(8, 14)]
            for trade_date in trade_dates:
                write(
                    minute_dir / f"trade_date={trade_date}.parquet",
                    pd.DataFrame(
                        [
                            {
                                "ts_code": "600000.SH",
                                "trade_time": f"2021-10-{trade_date[-2:]} 09:30:00",
                                "open": 10.0,
                                "high": 10.1,
                                "low": 9.9,
                                "close": 10.0,
                                "vol": 1000.0,
                                "amount": 10000.0,
                                "trade_date": trade_date,
                                "available_at": f"2021-10-{trade_date[-2:]}T09:30:00+08:00",
                                "available_at_rule": "bar_close",
                            }
                        ]
                    ),
                )

            original_read = pd.read_parquet
            lock = threading.Lock()
            active = 0
            maximum_active = 0
            completion_order: list[str] = []

            def delayed_read(path, *args, **kwargs):
                nonlocal active, maximum_active
                trade_date = Path(path).stem.split("=", 1)[1]
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                # The earliest input deliberately finishes last; the writer
                # must still emit its row group first.
                time.sleep(0.06 if trade_date == trade_dates[0] else 0.01)
                try:
                    return original_read(path, *args, **kwargs)
                finally:
                    with lock:
                        active -= 1
                        completion_order.append(trade_date)

            output = Path(tmp) / "parallel.parquet"
            with patch.object(snapshot_module.pd, "read_parquet", side_effect=delayed_read):
                SnapshotBuilder(raw, Path(tmp) / "missing_events")._write_minutes_range(
                    trade_dates[0], trade_dates[-1], output, None
                )

            self.assertEqual(maximum_active, snapshot_module._MINUTE_PARTITION_WORKERS)
            self.assertNotEqual(completion_order[0], trade_dates[0])
            parquet = pq.ParquetFile(output)
            self.assertEqual(parquet.metadata.num_row_groups, len(trade_dates))
            self.assertEqual(
                [parquet.read_row_group(i)["trade_date"][0].as_py() for i in range(len(trade_dates))],
                trade_dates,
            )

    def test_parallel_replay_minutes_are_arrow_and_row_group_equivalent_to_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            minute_dir = raw / "stk_mins_1min_by_date"
            trade_dates = ("20211008", "20211011", "20211012")
            for index, trade_date in enumerate(trade_dates):
                write(
                    minute_dir / f"trade_date={trade_date}.parquet",
                    pd.DataFrame(
                        [
                            {
                                "ts_code": code,
                                "trade_time": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 09:30:00",
                                "open": 10.0 + row,
                                "high": 10.1 + row,
                                "low": 9.9 + row,
                                "close": 10.0 + row,
                                "vol": 1000.0 + index,
                                "amount": 10000.0 + index,
                                "trade_date": trade_date,
                                "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T09:30:00+08:00",
                                "available_at_rule": "bar_close",
                            }
                            for row, code in enumerate(("000001.SZ", "600000.SH"))
                        ]
                    ),
                )

            builder = SnapshotBuilder(raw, Path(tmp) / "missing_events")
            serial_path = Path(tmp) / "serial.parquet"
            parallel_path = Path(tmp) / "parallel.parquet"
            screened = frozenset({"000001.SZ"})
            with patch.object(snapshot_module, "_MINUTE_PARTITION_WORKERS", 1):
                serial_meta, serial_profile = builder._write_minutes_range(
                    trade_dates[0], trade_dates[-1], serial_path, screened
                )
            parallel_meta, parallel_profile = builder._write_minutes_range(
                trade_dates[0], trade_dates[-1], parallel_path, screened
            )

            serial = pq.ParquetFile(serial_path)
            parallel = pq.ParquetFile(parallel_path)
            self.assertEqual(serial_meta, parallel_meta)
            self.assertTrue(serial.schema_arrow.equals(parallel.schema_arrow, check_metadata=True))
            self.assertEqual(serial.metadata.num_row_groups, parallel.metadata.num_row_groups)
            for index in range(serial.metadata.num_row_groups):
                self.assertTrue(serial.read_row_group(index).equals(parallel.read_row_group(index)))
            self.assertTrue(serial.read().equals(parallel.read()))
            self.assertEqual(
                {
                    key: value
                    for key, value in serial_profile.items()
                    if key not in {"file", "build_seconds", "write_seconds", "size_bytes"}
                },
                {
                    key: value
                    for key, value in parallel_profile.items()
                    if key not in {"file", "build_seconds", "write_seconds", "size_bytes"}
                },
            )
            # Row-level PIT timestamps survive screening/correction unchanged.
            self.assertEqual(
                parallel.read()["available_at"].to_pylist(),
                [f"2021-10-{trade_date[-2:]}T09:30:00+08:00" for trade_date in trade_dates],
            )

    def test_replay_minute_prefetch_failure_cancels_window_and_never_publishes_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            minute_dir = raw / "stk_mins_1min_by_date"
            trade_dates = [f"202110{day:02d}" for day in range(8, 14)]
            for trade_date in trade_dates:
                write(
                    minute_dir / f"trade_date={trade_date}.parquet",
                    pd.DataFrame(
                        [
                            {
                                "ts_code": "600000.SH",
                                "trade_time": f"2021-10-{trade_date[-2:]} 09:30:00",
                                "trade_date": trade_date,
                                "available_at": f"2021-10-{trade_date[-2:]}T09:30:00+08:00",
                            }
                        ]
                    ),
                )

            original_read = pd.read_parquet
            started: list[str] = []
            lock = threading.Lock()
            active = 0
            maximum_active = 0

            def failing_read(path, *args, **kwargs):
                nonlocal active, maximum_active
                trade_date = Path(path).stem.split("=", 1)[1]
                with lock:
                    started.append(trade_date)
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    if trade_date == trade_dates[1]:
                        # Let the first partition publish one row group into
                        # the private tmp file before this prefetched failure.
                        time.sleep(0.05)
                        raise RuntimeError("fixture minute decode failure")
                    if trade_date != trade_dates[0]:
                        time.sleep(0.1)
                    return original_read(path, *args, **kwargs)
                finally:
                    with lock:
                        active -= 1

            output = Path(tmp) / "stable.parquet"
            output.write_bytes(b"existing-output")
            with (
                patch.object(snapshot_module.pd, "read_parquet", side_effect=failing_read),
                self.assertRaisesRegex(RuntimeError, "fixture minute decode failure"),
            ):
                SnapshotBuilder(raw, Path(tmp) / "missing_events")._write_minutes_range(
                    trade_dates[0], trade_dates[-1], output, None
                )

            self.assertLessEqual(maximum_active, snapshot_module._MINUTE_PARTITION_WORKERS)
            self.assertLess(len(started), len(trade_dates))
            self.assertEqual(output.read_bytes(), b"existing-output")
            self.assertFalse(output.with_suffix(".parquet.tmp").exists())

    def test_industry_membership_switches_taxonomy_at_sw2021_effective_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            (raw / "index_classify").mkdir(parents=True)
            pd.DataFrame([
                {"index_code": "801020.SI", "industry_name": "采掘", "level": "L1"},
            ]).to_parquet(raw / "index_classify" / "src=SW2014.parquet", index=False)
            (raw / "index_member").mkdir()
            pd.DataFrame([
                {"index_code": "801020.SI", "con_code": "600123.SH", "in_date": "20120101", "out_date": "20211213"},
            ]).to_parquet(raw / "index_member" / "l1_code=801020.SI.parquet", index=False)
            # The SW2021 table backfills in_date far into the past; before the
            # switch day that backfill must not leak into the classification.
            (raw / "index_member_all").mkdir()
            pd.DataFrame([
                {"l1_code": "801950.SI", "l1_name": "煤炭", "ts_code": "600123.SH", "in_date": "19980601", "out_date": None},
                # 600456.SH moved industries: coal until 2023, petro afterwards.
                {"l1_code": "801950.SI", "l1_name": "煤炭", "ts_code": "600456.SH", "in_date": "20211213", "out_date": "20230101"},
                {"l1_code": "801960.SI", "l1_name": "石油石化", "ts_code": "600456.SH", "in_date": "20230101", "out_date": None},
            ]).to_parquet(raw / "index_member_all" / "l1_code=801950.SI.parquet", index=False)

            builder = SnapshotBuilder(raw, Path(tmp) / "missing_events")
            before = builder._industry_membership("20180702")
            self.assertEqual(before.iloc[0]["l1_code"], "801020.SI")
            self.assertEqual(before.iloc[0]["l1_name"], "采掘")
            # The last pre-switch day still classifies from the SW2014 row that
            # closes on the switch day itself (out_date exclusive).
            last_old_day = builder._industry_membership("20211212")
            self.assertEqual(last_old_day.iloc[0]["l1_code"], "801020.SI")
            after = builder._industry_membership("20211213")
            self.assertEqual(dict(zip(after["ts_code"], after["l1_code"])),
                             {"600123.SH": "801950.SI", "600456.SH": "801950.SI"})
            moved = builder._industry_membership("20240101")
            self.assertEqual(dict(zip(moved["ts_code"], moved["l1_code"])),
                             {"600123.SH": "801950.SI", "600456.SH": "801960.SI"})

    def test_empty_auction_builder_writes_canonical_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            (raw / "stk_auction").mkdir(parents=True)
            auction, meta = SnapshotBuilder(raw, Path(tmp) / "missing_events")._build_auction(
                "20241001", "20241231"
            )
            path = Path(tmp) / "auction.parquet"
            auction.to_parquet(path, index=False)

            schema = pq.ParquetFile(path).schema_arrow
            field_types = {field.name: str(field.type) for field in schema}
            self.assertEqual(meta["rows"], 0)
            self.assertNotIn("coverage_start", meta)
            self.assertNotIn("coverage_end", meta)
            self.assertEqual(field_types["ts_code"], "string")
            self.assertEqual(field_types["trade_date"], "string")
            self.assertEqual(field_types["available_at"], "string")
            self.assertEqual(field_types["price"], "double")
            self.assertEqual(field_types["amount"], "double")


    def test_large_frame_profile_uses_footer_without_changing_manifest_fields(self):
        frame = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20210102",
                    "available_at": "2021-01-02T17:30:00+08:00",
                    "amount": 1.0,
                },
                {
                    "ts_code": None,
                    "trade_date": "20210101",
                    "available_at": None,
                    "amount": None,
                },
            ]
        )
        expected_ranges = snapshot_module._profile_date_ranges(frame)
        expected_nulls = snapshot_module._profile_key_nulls(frame)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.parquet"
            with (
                patch.object(snapshot_module, "PROFILE_FULL_SCAN_MAX_ROWS", 0),
                patch.object(
                    snapshot_module,
                    "_profile_date_ranges",
                    side_effect=AssertionError("DataFrame date scan"),
                ),
                patch.object(
                    snapshot_module,
                    "_profile_key_nulls",
                    side_effect=AssertionError("DataFrame null scan"),
                ),
            ):
                profile = snapshot_module._write_with_profile(path, frame, build_seconds=0.25)

        self.assertEqual(profile["rows"], len(frame))
        self.assertEqual(profile["columns"], list(frame.columns))
        self.assertEqual(profile["date_ranges"], expected_ranges)
        self.assertEqual(profile["key_nulls"], expected_nulls)

    def test_footer_profile_falls_back_for_timezone_timestamp_contract(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "available_at": pd.to_datetime(
                    ["2021-01-02 17:30:00+08:00", "2021-01-03 18:00:00+08:00"]
                ),
            }
        )
        expected = snapshot_module._profile_date_ranges(frame)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.parquet"
            with (
                patch.object(snapshot_module, "PROFILE_FULL_SCAN_MAX_ROWS", 0),
                patch.object(
                    snapshot_module,
                    "_profile_date_ranges",
                    wraps=snapshot_module._profile_date_ranges,
                ) as scanned,
            ):
                profile = snapshot_module._write_with_profile(path, frame, build_seconds=0.1)

        scanned.assert_called_once()
        self.assertEqual(profile["date_ranges"], expected)

    def test_auction_availability_is_the_official_publish_constant(self):
        # One official-publication constant for every partition and vintage:
        # 09:29 of the trade date, whatever the sidecar's landing evidence
        # says. Multi-day pipeline outages were script defects, not source
        # unavailability; backtests run against the corrected world.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "stk_auction" / "trade_date=20260713.parquet"
            write(path, pd.DataFrame([{"trade_date": "20260713", "ts_code": "000001.SZ"}]))
            path.with_suffix(".parquet.meta.json").write_text(
                json.dumps(
                    {
                        "availability": {
                            "available_at": "2026-07-20T23:20:00+08:00",
                            "rule": "observed:days_late_heal",
                        }
                    }
                ),
                encoding="utf-8",
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "missing_events")
            self.assertEqual(
                builder._auction_partition_availability("20260713"),
                ("2026-07-13T09:29:00+08:00", "official:latest_publish_time"),
            )

    def test_auction_builder_materializes_recovered_price_and_no_trade_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "stk_auction" / "trade_date=20260713.parquet"
            write(
                path,
                pd.DataFrame(
                    [
                        {
                            "trade_date": "20260713", "ts_code": "000001.SZ", "price": None,
                            "vol": 1000, "amount": 10000, "pre_close": 9.8,
                            "turnover_rate": 0.1, "volume_ratio": 1.2, "float_share": 100000,
                        },
                        {
                            "trade_date": "20260713", "ts_code": "000002.SZ", "price": 99.0,
                            "vol": 0, "amount": 0, "pre_close": 10.0,
                            "turnover_rate": 0.0, "volume_ratio": 0.0, "float_share": 200000,
                        },
                    ]
                ),
            )
            path.with_suffix(".parquet.meta.json").write_text(
                json.dumps(
                    {
                        "availability": {
                            "available_at": "2026-07-13T09:28:00+08:00",
                            "rule": "observed:test_capture",
                        }
                    }
                ),
                encoding="utf-8",
            )
            auction, meta = SnapshotBuilder(raw, Path(tmp) / "missing_events")._build_auction(
                "20260713", "20260713"
            )

            prices = auction.set_index("ts_code")["price"]
            self.assertEqual(float(prices["000001.SZ"]), 10.0)
            self.assertTrue(pd.isna(prices["000002.SZ"]))
            self.assertEqual(meta["price_quality"]["derived_price_rows"], 1)
            self.assertEqual(meta["price_quality"]["no_trade_rows"], 1)

    def test_auction_builder_drops_unobserved_rows_and_counts_them(self):
        # Suspended codes — and the retired BSE aliases around the 2025-08
        # renumbering — are listed by the source with price/vol/amount all NaN.
        # They carry no auction observation: equivalent to a missing row, so the
        # build drops and counts them instead of failing (lap-test17 crash).
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "stk_auction" / "trade_date=20250818.parquet"
            write(
                path,
                pd.DataFrame(
                    [
                        {"trade_date": "20250818", "ts_code": "832491.BJ", "price": None,
                         "vol": None, "amount": None, "pre_close": 9.8,
                         "turnover_rate": None, "volume_ratio": None, "float_share": 100000},
                        {"trade_date": "20250818", "ts_code": "000001.SZ", "price": 10.0,
                         "vol": 1000, "amount": 10000, "pre_close": 9.8,
                         "turnover_rate": 0.1, "volume_ratio": 1.2, "float_share": 100000},
                    ]
                ),
            )
            path.with_suffix(".parquet.meta.json").write_text(
                json.dumps({"availability": {
                    "available_at": "2025-08-18T09:28:00+08:00",
                    "rule": "observed:test_capture",
                }}),
                encoding="utf-8",
            )
            auction, meta = SnapshotBuilder(raw, Path(tmp) / "missing_events")._build_auction(
                "20250818", "20250818"
            )
            self.assertEqual(auction["ts_code"].tolist(), ["000001.SZ"])
            self.assertEqual(meta["price_quality"]["unobserved_rows_dropped"], 1)
            self.assertEqual(meta["price_quality"]["source_price_rows"], 1)

    def test_auction_builder_still_rejects_partial_nan_quantities(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "stk_auction" / "trade_date=20250818.parquet"
            write(path, pd.DataFrame([{
                "trade_date": "20250818", "ts_code": "000001.SZ", "price": 10.0,
                "vol": 1000, "amount": None, "pre_close": 9.8,
                "turnover_rate": 0.1, "volume_ratio": 1.2, "float_share": 100000,
            }]))
            path.with_suffix(".parquet.meta.json").write_text(
                json.dumps({"availability": {
                    "available_at": "2025-08-18T09:28:00+08:00",
                    "rule": "observed:test_capture",
                }}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid quantity combinations"):
                SnapshotBuilder(raw, Path(tmp) / "missing_events")._build_auction(
                    "20250818", "20250818"
                )

    def test_auction_builder_rejects_price_quantity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "stk_auction" / "trade_date=20260713.parquet"
            write(path, pd.DataFrame([{
                "trade_date": "20260713", "ts_code": "000001.SZ", "price": 100.0,
                "vol": 1000, "amount": 10000, "pre_close": 9.8,
                "turnover_rate": 0.1, "volume_ratio": 1.2, "float_share": 100000,
            }]))
            path.with_suffix(".parquet.meta.json").write_text(
                json.dumps({"availability": {
                    "available_at": "2026-07-13T09:28:00+08:00",
                    "rule": "observed:test_capture",
                }}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inconsistent clearing prices"):
                SnapshotBuilder(raw, Path(tmp) / "missing_events")._build_auction(
                    "20260713", "20260713"
                )

    def test_decision_snapshot_is_pit_filtered_and_unit_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            auction_path = raw / "stk_auction" / "trade_date=20211008.parquet"
            auction_path.with_suffix(".parquet.meta.json").write_text(
                json.dumps(
                    {
                        "availability": {
                            "available_at": "2021-10-08T09:28:36+08:00",
                            "rule": "observed:test_capture",
                        }
                    }
                ),
                encoding="utf-8",
            )
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            out = Path(tmp) / "snap"
            builder = SnapshotBuilder(raw, events_root, status_path)
            manifest = builder.build_decision_snapshot(DECISION, out, CONFIG)

            daily = pd.read_parquet(out / "daily.parquet")
            # Same-day 20211008 daily data is not visible at the 09:25 decision.
            self.assertEqual(sorted(daily["trade_date"].unique()), ["20210930"])
            self.assertEqual(daily.loc[0, "vol"], 100000.0)  # 手 -> 股
            self.assertEqual(daily.loc[0, "amount"], 1050000.0)  # 千元 -> 元
            self.assertAlmostEqual(daily.loc[0, "pct_chg"], 0.05)
            self.assertAlmostEqual(daily.loc[0, "turnover_rate"], 0.02)
            self.assertAlmostEqual(daily.loc[0, "dv_ttm"], 0.03)

            intraday = pd.read_parquet(out / "intraday_1min.parquet")
            self.assertIn("auction_correction_rule", intraday.columns)
            # available_at on minute bars == bar close, an internal gate, not agent info.
            self.assertNotIn("available_at", intraday.columns)

            auction = pd.read_parquet(out / "auction.parquet")
            self.assertNotIn("20211008", set(auction["trade_date"]))
            self.assertIn("20210930", set(auction["trade_date"]))

            # Auction visibility is the uniform official 09:29 constant:
            # invisible at 09:28, visible from 09:29 on, regardless of when
            # the partition actually landed.
            boundary_out = Path(tmp) / "snap_at_0928"
            builder.build_decision_snapshot(
                datetime(2021, 10, 8, 9, 28, tzinfo=CN_TZ), boundary_out, CONFIG
            )
            self.assertNotIn(
                "20211008", set(pd.read_parquet(boundary_out / "auction.parquet")["trade_date"])
            )
            after_out = Path(tmp) / "snap_after_auction"
            builder.build_decision_snapshot(
                datetime(2021, 10, 8, 9, 30, tzinfo=CN_TZ), after_out, CONFIG
            )
            after_auction = pd.read_parquet(after_out / "auction.parquet")
            self.assertIn("20211008", set(after_auction["trade_date"]))
            self.assertEqual(
                float(after_auction.loc[after_auction["trade_date"] == "20211008", "price"].iloc[0]),
                10.02,
            )
            auction_row = after_auction.loc[after_auction["trade_date"] == "20211008"].iloc[0]
            self.assertAlmostEqual(float(auction_row["turnover_rate"]), 0.0001)
            self.assertAlmostEqual(float(auction_row["volume_ratio"]), 1.2)
            self.assertEqual(float(auction_row["float_share"]), 3_000_000_000.0)
            self.assertEqual(
                auction_row["available_at_rule"],
                "official:latest_publish_time",
            )
            self.assertEqual(str(auction_row["available_at"]), "2021-10-08T09:29:00+08:00")
            after_manifest = load_snapshot_manifest(after_out)
            self.assertEqual(after_manifest["domains"]["auction"]["units"], "unit_contract")
            self.assertEqual(
                after_manifest["domains"]["auction"]["unit_conversions"],
                [
                    {"column": "turnover_rate", "factor": 0.01, "rule": "percent->decimal"},
                    {
                        "column": "float_share",
                        "factor": 10_000.0,
                        "rule": "10k_shares->shares",
                    },
                ],
            )

            events = pd.read_parquet(out / "events.parquet")
            datasets = set(events["dataset"])
            self.assertIn("margin_secs", datasets)  # same-day 09:00 visible at 09:25
            self.assertNotIn("moneyflow", datasets)  # same-day 19:00 is in the future

            macro = pd.read_parquet(out / "macro.parquet")
            self.assertEqual(list(macro["quarter"]), ["2021Q2"])

            text_index = pd.read_parquet(out / "text_index.parquet")
            self.assertEqual(len(text_index), 1)
            bodies = pd.read_parquet(out / "text_library" / text_index.loc[0, "library_file"])
            body_map = dict(zip(bodies["text_id"], bodies["body"]))
            self.assertIn("正文内容", body_map[text_index.loc[0, "text_id"]])

            fundamentals = pd.read_parquet(out / "fundamentals.parquet")
            self.assertEqual(len(fundamentals), 1)

            universe = pd.read_parquet(out / "universe.parquet")
            codes = set(universe["ts_code"])
            self.assertIn("000001.SZ", codes)
            self.assertIn("000005.SZ", codes)  # delisted 2024 -> alive at the 2021 decision
            self.assertIn("000010.SZ", codes)  # screening off by default: ST stays
            self.assertNotIn("000003.SZ", codes)  # delisted 2002 -> excluded
            named = universe.set_index("ts_code")
            self.assertEqual(named.loc["000001.SZ", "name"], "平安银行")
            # PIT name: the 2023 rename must not leak into a 2021 decision snapshot.
            self.assertEqual(named.loc["000005.SZ", "name"], "世纪星源")
            # Future information is masked: survivors' delistings are all post-decision.
            self.assertNotIn("delist_date", universe.columns)
            self.assertNotIn("name_asof", universe.columns)

            self.assertEqual(manifest["kind"], "decision_input")
            stored = load_snapshot_manifest(out)
            self.assertEqual(stored["snapshot_id"], manifest["snapshot_id"])
            self.assertIn("build_profile", stored)
            self.assertIn("data_profile", stored)
            self.assertIn("daily.parquet", stored["data_profile"]["files"])
            self.assertEqual(stored["data_profile"]["files"]["daily.parquet"]["rows"], len(daily))
            self.assertIn("build_seconds", stored["data_profile"]["files"]["fundamentals.parquet"])
            event_profiles = stored["domains"]["events"]["dataset_build_profile"]
            self.assertEqual(set(event_profiles), {"margin_secs", "moneyflow"})
            self.assertEqual(event_profiles["margin_secs"]["partition_files"], 1)
            self.assertEqual(event_profiles["margin_secs"]["source_rows"], 1)
            self.assertEqual(event_profiles["margin_secs"]["rows_after_visibility"], 1)
            self.assertEqual(event_profiles["margin_secs"]["rows_output"], 1)
            self.assertEqual(event_profiles["moneyflow"]["source_rows"], 1)
            self.assertEqual(event_profiles["moneyflow"]["rows_after_visibility"], 0)
            self.assertEqual(event_profiles["moneyflow"]["rows_output"], 0)
            for profile in event_profiles.values():
                self.assertGreaterEqual(profile["total_seconds"], 0)
                self.assertEqual(
                    set(profile["phases"]),
                    {
                        "discover_seconds",
                        "read_filter_seconds",
                        "concat_seconds",
                        "deduplicate_seconds",
                        "screen_seconds",
                    },
                )
                self.assertTrue(all(seconds >= 0 for seconds in profile["phases"].values()))

    def test_daily_join_filters_each_dataset_by_own_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            out = Path(tmp) / "snap_after_close"
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=(),
                include_intraday=False,
                include_industry=False,
            )
            decision = datetime(2021, 10, 8, 17, 45, tzinfo=CN_TZ)

            SnapshotBuilder(raw, Path(tmp) / "fund_events").build_decision_snapshot(decision, out, config)

            daily = pd.read_parquet(out / "daily.parquet").set_index(["trade_date", "ts_code"])
            same_day = daily.loc[("20211008", "000001.SZ")]
            self.assertAlmostEqual(same_day["pct_chg"], 0.05)
            self.assertAlmostEqual(same_day["up_limit"], 11.55)
            self.assertTrue(pd.isna(same_day["turnover_rate"]))
            self.assertTrue(pd.isna(same_day["total_share"]))
            manifest = load_snapshot_manifest(out)
            self.assertIn("20211008", manifest["domains"]["daily"]["visible_trade_dates_by_dataset"]["daily"])
            self.assertNotIn("20211008", manifest["domains"]["daily"]["visible_trade_dates_by_dataset"]["daily_basic"])

    def test_universe_screening_restricts_per_stock_domains(self):
        from dataclasses import replace as dc_replace

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            out = Path(tmp) / "snap"
            config = dc_replace(CONFIG, screen_exclude_st=True, screen_min_circ_mv_yi=50.0)
            builder = SnapshotBuilder(raw, events_root, status_path)
            manifest = builder.build_decision_snapshot(DECISION, out, config)

            universe = pd.read_parquet(out / "universe.parquet")
            codes = set(universe["ts_code"])
            self.assertIn("000001.SZ", codes)          # 200亿, non-ST
            self.assertNotIn("000010.SZ", codes)       # ST at the decision day
            # No daily_basic row -> cannot prove the cap floor -> fails closed.
            self.assertNotIn("000005.SZ", codes)
            screen = manifest["domains"]["universe_screen"]
            self.assertTrue(screen["active"])
            self.assertEqual(screen["codes"], 1)
            self.assertTrue(screen["config"]["exclude_st"])
            # Every per-stock domain is restricted to the screened set.
            for name in ("daily.parquet", "intraday_1min.parquet", "events.parquet", "fundamentals.parquet"):
                frame = pd.read_parquet(out / name)
                if "ts_code" in frame.columns and len(frame):
                    stock_rows = frame[frame["ts_code"].notna()]
                    self.assertTrue(set(stock_rows["ts_code"].astype(str)) <= {"000001.SZ"}, name)

            # The replay slot freezes the SAME pre-period set.
            slot = Path(tmp) / "slot"
            slot_manifest = builder.build_replay_slot("20211008", "20211011", slot, label="valid", config=config)
            self.assertTrue(slot_manifest["domains"]["universe_screen"]["active"])
            slot_daily = pd.read_parquet(slot / "daily.parquet")
            self.assertTrue(set(slot_daily["ts_code"].astype(str)) <= {"000001.SZ"})

    def test_apply_screen_passes_index_level_rows(self):
        from autotrade.environment.data.snapshot import SnapshotBuilder

        frame = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "v": 1},   # screened stock, allowed
                {"ts_code": "000002.SZ", "v": 2},   # screened stock, excluded
                {"ts_code": "881141.TI", "v": 3},   # THS industry index
                {"ts_code": "BK1752.DC", "v": 4},   # DC board index
                {"ts_code": "000242.KP", "v": 5},   # KPL concept
                {"ts_code": None, "v": 6},           # market-level row
            ]
        )
        kept = SnapshotBuilder._apply_screen(frame, frozenset({"000001.SZ"}))
        self.assertEqual(set(kept["v"]), {1, 3, 4, 5, 6})

    def test_replay_slot_available_from_floors_pre_period_rows(self):
        # A row published on the weekend between the decision anchor (Friday
        # 23:59:59) and the Monday period start must enter the replay slot.
        from datetime import datetime as dt
        from zoneinfo import ZoneInfo

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            weekend = pd.DataFrame([
                {"trade_date": "20211007", "ts_code": "000001.SZ", "net_mf_amount": 9.0,
                 "available_at": "2021-10-07T23:59:59+08:00", "available_at_rule": "same_day_evening"},
            ])
            (raw / "moneyflow").mkdir(parents=True, exist_ok=True)
            weekend.to_parquet(raw / "moneyflow" / "trade_date=20211007.parquet", index=False)
            builder = SnapshotBuilder(raw, events_root, status_path)
            anchor = dt(2021, 10, 7, 23, 59, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

            floored = builder.build_replay_slot(
                "20211008", "20211011", Path(tmp) / "with_anchor", label="valid",
                config=CONFIG, available_from=anchor,
            )
            events = pd.read_parquet(Path(tmp) / "with_anchor" / "events.parquet")
            self.assertIn(9.0, set(events.get("net_mf_amount", pd.Series(dtype=float)).dropna()))
            self.assertEqual(floored["available_from"], anchor.isoformat())

            builder.build_replay_slot(
                "20211008", "20211011", Path(tmp) / "no_anchor", label="valid", config=CONFIG,
            )
            events_plain = pd.read_parquet(Path(tmp) / "no_anchor" / "events.parquet")
            self.assertNotIn(9.0, set(events_plain.get("net_mf_amount", pd.Series(dtype=float)).dropna()))

    def test_decision_windows_are_configurable_by_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            out = Path(tmp) / "snap"
            config = SnapshotConfig(
                window_months=21,
                events_datasets=("margin_secs", "moneyflow"),
                macro_datasets=("cn_gdp",),
                text_datasets=("cctv_news",),
                fundamental_datasets=("income_vip",),
                daily_window_months=1,
                fundamentals_window_months=1,
                events_window_months=1,
                macro_window_months=2,
                text_window_months=1,
                include_intraday=False,
                include_industry=False,
            )

            manifest = SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(DECISION, out, config)

            daily = pd.read_parquet(out / "daily.parquet")
            self.assertEqual(sorted(daily["trade_date"].unique()), ["20210930"])
            fundamentals = pd.read_parquet(out / "fundamentals.parquet")
            self.assertEqual(len(fundamentals), 1)
            events = pd.read_parquet(out / "events.parquet")
            self.assertEqual(set(events["dataset"]), {"margin_secs"})
            # Datasets without visible rows in the window still contribute
            # their typed schema: the Timeview intersects replay parts to the
            # frozen schema, so a missing column would silently drop that
            # dataset's replay data for the whole fold.
            self.assertIn("net_mf_amount", events.columns)
            macro = pd.read_parquet(out / "macro.parquet")
            self.assertEqual(len(macro), 0)
            text_index = pd.read_parquet(out / "text_index.parquet")
            self.assertEqual(len(text_index), 1)
            self.assertEqual(manifest["window_config"]["daily_months"], 1)
            self.assertEqual(manifest["window_config"]["macro_months"], 2)
            self.assertEqual(manifest["domain_windows"]["macro"]["window_months"], 2)

    def test_macro_registry_datasets_are_exempt_from_the_window_floor(self):
        # Instrument registries (fut_basic/opt_basic/cb_basic) stay valid for the
        # instrument's whole life: an old list_date must survive the macro window
        # floor, while a future list_date stays PIT-hidden and a windowed daily
        # dataset keeps the floor.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            write(
                raw / "fut_basic" / "exchange=CFFEX.parquet",
                pd.DataFrame([
                    {"ts_code": "IF1005.CFX", "exchange": "CFFEX", "list_date": "20100416",
                     "available_at": "2010-04-16T23:59:59+08:00", "available_at_rule": "conservative_date_eod"},
                    {"ts_code": "IF2299.CFX", "exchange": "CFFEX", "list_date": "20220916",
                     "available_at": "2022-09-16T23:59:59+08:00", "available_at_rule": "conservative_date_eod"},
                ]),
            )
            write(
                raw / "fut_daily" / "trade_date=20200108.parquet",
                pd.DataFrame([{
                    "ts_code": "IF2001.CFX", "trade_date": "20200108", "settle": 4100.0,
                    "available_at": "2020-01-08T23:59:59+08:00", "available_at_rule": "conservative_date_eod",
                }]),
            )
            out = Path(tmp) / "snap"
            config = SnapshotConfig(
                window_months=2,
                events_datasets=(),
                macro_datasets=("fut_basic", "fut_daily"),
                text_datasets=("cctv_news",),
                fundamental_datasets=("income_vip",),
                include_intraday=False,
                include_industry=False,
            )
            SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(DECISION, out, config)
            macro = pd.read_parquet(out / "macro.parquet")
            registry = macro[macro["dataset"] == "fut_basic"]
            self.assertEqual(sorted(registry["ts_code"]), ["IF1005.CFX"])  # old kept, future hidden
            self.assertTrue(macro[macro["dataset"] == "fut_daily"].empty)  # window floor still applies
            # Replay slots must NOT apply the exemption: the Timeview unions the
            # slot with the frozen snapshot, so a second full-life registry copy
            # would duplicate every row in the agent's backtest view.
            slot = Path(tmp) / "slot"
            SnapshotBuilder(raw, events_root, status_path).build_replay_slot(
                "20211008", "20211011", slot, label="valid", config=config
            )
            slot_macro = pd.read_parquet(slot / "macro.parquet")
            self.assertTrue(slot_macro.empty or slot_macro[slot_macro["dataset"] == "fut_basic"].empty)

    def test_fundamental_event_reader_filters_partitions_by_min_available_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fund_events"
            write(
                root / "income_vip" / "available_month=202001.parquet",
                pd.DataFrame(
                    [
                        {
                            "dataset": "income_vip",
                            "ts_code": "000001.SZ",
                            "available_at": "2020-01-10T18:00:00+08:00",
                            "available_at_rule": "source:f_ann_date_or_ann_date",
                            "available_month": "202001",
                            "business_key": "old",
                            "source_path": "x",
                            "source_write_id": "w",
                            "source_row_id": 0,
                        }
                    ]
                ),
            )
            write(
                root / "income_vip" / "available_month=202109.parquet",
                pd.DataFrame(
                    [
                        {
                            "dataset": "income_vip",
                            "ts_code": "000001.SZ",
                            "available_at": "2021-09-10T18:00:00+08:00",
                            "available_at_rule": "source:f_ann_date_or_ann_date",
                            "available_month": "202109",
                            "business_key": "new",
                            "source_path": "x",
                            "source_write_id": "w",
                            "source_row_id": 1,
                        }
                    ]
                ),
            )

            events = read_fundamental_events(
                root,
                "2021-10-08T09:25:00+08:00",
                datasets=("income_vip",),
                min_available_at="2021-09-01T00:00:00+08:00",
                require_partitions=True,
            )

            self.assertEqual(events["business_key"].tolist(), ["new"])

    def test_replay_slot_includes_daily_events_text_and_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            out = Path(tmp) / "replay"
            builder = SnapshotBuilder(raw, Path(tmp) / "missing_events")
            manifest = builder.build_replay_slot("20211007", "20211011", out, label="valid", config=CONFIG)
            daily = pd.read_parquet(out / "daily.parquet")
            self.assertEqual(sorted(daily["trade_date"].unique()), ["20211008"])
            # Every replay domain now carries a row-level available_at for the Timeview;
            # the daily core stamps the trade_date's evening publish time.
            self.assertIn("available_at", daily.columns)
            self.assertEqual(set(daily["available_at"]), {"2021-10-08T17:30:00+08:00"})
            # Replay region is not PIT-filtered: the same-evening moneyflow row is included.
            events = pd.read_parquet(out / "events.parquet")
            self.assertEqual(set(events["dataset"]), {"margin_secs", "moneyflow"})
            text_index = pd.read_parquet(out / "text_index.parquet")
            self.assertEqual(len(text_index), 1)
            minutes = pd.read_parquet(out / "intraday_1min.parquet")
            self.assertEqual(len(minutes), 0)  # fixture minutes are outside the period
            # Macro and fundamentals domains are written even when empty for this period
            # (cn_gdp rows fall outside, the events root is absent), so the Timeview
            # always has a stable per-domain file to roll.
            self.assertTrue((out / "macro.parquet").exists())
            self.assertEqual(len(pd.read_parquet(out / "macro.parquet")), 0)
            self.assertTrue((out / "fundamentals.parquet").exists())
            self.assertEqual(len(pd.read_parquet(out / "fundamentals.parquet")), 0)
            self.assertEqual(manifest["kind"], "replay_slot")
            stored = load_snapshot_manifest(out)
            self.assertIn("build_profile", stored)
            self.assertIn("intraday_1min.parquet", stored["data_profile"]["files"])

    def test_replay_slot_rolls_in_period_macro_and_fundamentals(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            # A macro release and a fundamental filing published inside the period.
            write(
                raw / "cn_gdp" / "range=2021Q4.parquet",
                pd.DataFrame([{"quarter": "2021Q3", "gdp": 1.2, "available_at": "2021-10-08T10:00:00+08:00", "available_at_rule": "release"}]),
            )
            events_root = Path(tmp) / "fund_events"
            write(
                events_root / "income_vip" / "available_month=202110.parquet",
                pd.DataFrame([{"dataset": "income_vip", "ts_code": "000001.SZ", "available_at": "2021-10-08T18:00:00+08:00", "available_at_rule": "source:f_ann_date_or_ann_date", "available_month": "202110", "business_key": "k2", "source_path": "x", "source_write_id": "w", "source_row_id": 0}]),
            )
            out = Path(tmp) / "replay"
            builder = SnapshotBuilder(raw, events_root)
            builder.build_replay_slot("20211007", "20211011", out, label="valid", config=CONFIG)
            macro = pd.read_parquet(out / "macro.parquet")
            self.assertEqual(set(macro["dataset"]), {"cn_gdp"})
            self.assertIn("available_at", macro.columns)
            fundamentals = pd.read_parquet(out / "fundamentals.parquet")
            self.assertEqual(fundamentals["business_key"].tolist(), ["k2"])
            self.assertIn("available_at", fundamentals.columns)

    def test_macro_domain_dedups_overlapping_range_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            # Legacy end-suffixed range file repeating the SAME rows.
            duplicate = pd.read_parquet(raw / "cn_gdp" / "range=2020Q1_2021Q4.parquet")
            write(raw / "cn_gdp" / "range=2020Q1_2021Q2.parquet", duplicate)
            out = Path(tmp) / "snap"
            builder = SnapshotBuilder(raw, events_root, status_path)
            manifest = builder.build_decision_snapshot(DECISION, out, CONFIG)

            macro = pd.read_parquet(out / "macro.parquet")
            self.assertEqual(list(macro["quarter"]), ["2021Q2"])
            self.assertEqual(manifest["domains"]["macro"]["duplicate_rows_dropped"], {"cn_gdp": 1})
            profile = manifest["domains"]["macro"]["dataset_build_profile"]["cn_gdp"]
            self.assertEqual(profile["partition_files"], 2)
            self.assertEqual(profile["source_rows"], 4)
            self.assertEqual(profile["rows_after_visibility"], 2)
            self.assertEqual(profile["duplicate_rows_dropped"], 1)
            self.assertEqual(profile["rows_output"], 1)

    def test_stale_audit_status_warns_against_current_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            # Macro status proves an OLDER generation than the current lake.
            write_quality_status(
                Path(tmp) / "macro_context_status.json",
                "macro_context",
                created_at="2021-10-01T00:00:00+00:00",
            )
            (raw / ".raw_generation.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "state": "committed",
                    "generation_id": "g2",
                    "completed_at": "2021-10-07T00:00:00+00:00",
                }),
                encoding="utf-8",
            )
            manifest = SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                DECISION, Path(tmp) / "snap", CONFIG
            )
            warnings = manifest["data_quality_warnings"]
            self.assertIn("predates current raw generation", warnings["macro"])
            # The other fixture statuses are newer than the generation.
            self.assertNotIn("events", warnings)

    def test_warning_audit_status_recorded_in_manifest(self):
        # data docs §3.1: warning reports must be explicitly handled downstream.
        # A domain whose audit is in status=warning leaves a manifest trace.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            write_domain_statuses(Path(tmp), events="warning")
            manifest = SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                DECISION, Path(tmp) / "snap", CONFIG
            )
            warnings = manifest["data_quality_warnings"]
            self.assertIn("audit status is warning", warnings["events"])
            self.assertIn("1 warning findings", warnings["events"])
            self.assertNotIn("macro", warnings)

    def test_raw_generation_recorded_and_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            generation = {
                "schema_version": 2,
                "state": "committed",
                "generation_id": "abc123",
                "completed_at": "2021-10-08T01:00:00+00:00",
            }
            (raw / ".raw_generation.json").write_text(json.dumps(generation), encoding="utf-8")
            builder = SnapshotBuilder(raw, events_root, status_path)

            manifest = builder.build_decision_snapshot(DECISION, Path(tmp) / "snap", CONFIG)
            self.assertEqual(manifest["raw_generation"], generation)
            replay_manifest = builder.build_replay_slot("20211007", "20211011", Path(tmp) / "replay", label="valid", config=CONFIG)
            self.assertEqual(replay_manifest["raw_generation"], generation)

            # Lake mutating mid-build must fail the build.
            with self.assertRaisesRegex(RuntimeError, "generation changed"):
                with builder._raw_lake_guard():
                    (raw / ".raw_generation.json").write_text(
                        json.dumps({
                            "schema_version": 2,
                            "state": "committed",
                            "generation_id": "def456",
                            "completed_at": "2021-10-08T02:00:00+00:00",
                        }),
                        encoding="utf-8",
                    )

    def test_non_committed_raw_generation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            (raw / ".raw_generation.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "state": "dirty",
                    "generation_id": "old",
                    "transaction": {"job": "cn_evening_full"},
                }),
                encoding="utf-8",
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events")
            with self.assertRaisesRegex(RuntimeError, "generation is not committed"):
                builder.build_decision_snapshot(DECISION, Path(tmp) / "snap", CONFIG)

    def test_partial_raw_generation_stamp_fails_closed(self):
        # The updater always writes schema_version and state; a stamp missing
        # either must not be implicitly treated as committed.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            (raw / ".raw_generation.json").write_text(
                json.dumps({"generation_id": "old", "completed_at": "2021-10-08T01:00:00+00:00"}),
                encoding="utf-8",
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events")
            with self.assertRaisesRegex(RuntimeError, "not an explicit schema-v2 committed record"):
                builder.build_decision_snapshot(DECISION, Path(tmp) / "snap", CONFIG)

    def test_replay_slot_builds_corporate_actions_from_dividend_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            events_root = Path(tmp) / "fund_events"

            def dividend(**overrides):
                base = {
                    "dataset": "dividend", "ts_code": "000001.SZ", "end_date": "20210630",
                    "div_proc": "实施", "ex_date": "20211008", "record_date": "20211007",
                    "pay_date": "20211008", "div_listdate": None, "cash_div": 0.45,
                    "cash_div_tax": 0.5, "stk_div": None, "stk_bo_rate": None, "stk_co_rate": None,
                    "available_at": "2021-09-25T18:00:00+08:00",
                    "available_at_rule": "source:imp_ann_date_or_ann_date", "available_month": "202109",
                    "business_key": "d1", "source_path": "x", "source_write_id": "w", "source_row_id": 0,
                }
                return base | overrides

            write(
                events_root / "dividend" / "available_month=202109.parquet",
                pd.DataFrame(
                    [
                        # Superseded revision of the same event: the later announcement wins.
                        dividend(cash_div_tax=0.4),
                        dividend(available_at="2021-09-26T18:00:00+08:00"),
                        # Same code, second event on the same ex-date: amounts sum.
                        dividend(end_date="20201231", cash_div_tax=0.1, stk_div=0.5, business_key="d2"),
                        # Plan-stage row and an out-of-window ex-date are both excluded.
                        dividend(div_proc="预案", cash_div_tax=9.9, business_key="d3"),
                        dividend(ex_date="20211201", business_key="d4"),
                    ]
                ),
            )
            write(
                events_root / "dividend" / "available_month=202110.parquet",
                # Announced only after its own ex-date: a revision artifact, dropped.
                pd.DataFrame([dividend(ex_date="20211008", available_at="2021-10-09T18:00:00+08:00", business_key="d5")]),
            )
            out = Path(tmp) / "replay"
            manifest = SnapshotBuilder(raw, events_root).build_replay_slot(
                "20211007", "20211011", out, label="valid", config=CONFIG
            )
            actions = pd.read_parquet(out / "corporate_actions.parquet")
            self.assertEqual(len(actions), 1)
            row = actions.iloc[0]
            self.assertEqual((row["ts_code"], row["ex_date"]), ("000001.SZ", "20211008"))
            self.assertAlmostEqual(float(row["cash_per_share"]), 0.6)  # 0.5 (kept revision) + 0.1
            self.assertAlmostEqual(float(row["stock_per_share"]), 0.5)
            self.assertEqual(row["record_date"], "20211007")
            meta = manifest["domains"]["corporate_actions"]
            self.assertEqual(meta["rows"], 1)
            self.assertEqual(meta["dropped"]["announced_after_ex_date"], 1)

    def test_default_config_exposes_coverage_audit_additions(self):
        # Drift guard for the raw-coverage audit batch: board/sentiment events,
        # macro regime additions, the news wire, and the A-share index set stay
        # exposed; cn_schedule stays out (source keeps no history).
        config = SnapshotConfig()
        for dataset in ("kpl_list", "limit_step", "limit_cpt_list", "limit_list_d",
                        "limit_list_ths", "ths_hot", "dc_hot", "hm_detail"):
            self.assertIn(dataset, config.events_datasets, dataset)
        for dataset in ("repo_daily", "us_tycr", "us_trycr", "us_tbr", "us_tltr",
                        "shibor_quote", "index_daily"):
            self.assertIn(dataset, config.macro_datasets, dataset)
        self.assertIn("news", config.text_datasets)
        self.assertNotIn("cn_schedule", config.macro_datasets)
        # Deliberate default exclusions (see SnapshotConfig comment): datasets
        # with contradictory duplicate versions (pledge_detail, repurchase), no
        # PIT timestamps (hm_list), or decommissioned sources (slb_len_mm,
        # slb_len).
        for dataset in ("pledge_detail", "repurchase", "hm_list", "slb_len_mm", "slb_len"):
            self.assertNotIn(dataset, config.events_datasets, dataset)

    def test_events_limit_list_d_rows_gate_on_the_16_00_stamp(self):
        # The official limit list becomes visible at trade-date 16:00: the prior
        # day's list is decision input at 09:25, the same day's list is not.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            for trade_date in ("20210930", "20211008"):
                write(
                    raw / "limit_list_d" / f"trade_date={trade_date}.parquet",
                    pd.DataFrame([{
                        "trade_date": trade_date, "ts_code": "000001.SZ", "close": 10.5,
                        "pct_chg": 10.0, "amount": 1_050_000.0, "limit": "U",
                        "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 16:00:00+08:00",
                        "available_at_rule": "official_16_from:trade_date",
                    }]),
                )
            out = Path(tmp) / "snap"
            config = replace(
                CONFIG,
                events_datasets=("limit_list_d",),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=(),
                include_intraday=False,
            )
            SnapshotBuilder(raw, Path(tmp) / "fund_events").build_decision_snapshot(DECISION, out, config)
            events = pd.read_parquet(out / "events.parquet")
            rows = events[events["dataset"] == "limit_list_d"]
            self.assertEqual(sorted(rows["trade_date"]), ["20210930"])
            self.assertEqual(set(rows["limit"]), {"U"})

    def test_snapshot_excluded_columns_are_masked(self):
        # SNAPSHOT_EXCLUDED_COLUMNS: cb_basic current-state registry fields and
        # the source-rewritten limit_list_d.limit_amount never enter frozen
        # research inputs; sibling columns and static issuance terms stay.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            write(
                raw / "limit_list_d" / "trade_date=20211007.parquet",
                pd.DataFrame([{
                    "trade_date": "20211007", "ts_code": "000001.SZ", "limit": "U",
                    "fd_amount": 1.0e8, "limit_amount": 4.28e12,
                    "available_at": "2021-10-07 16:00:00+08:00",
                    "available_at_rule": "official_16_from:trade_date",
                }]),
            )
            write(
                raw / "cb_basic" / "full.parquet",
                pd.DataFrame([{
                    "ts_code": "128039.SZ", "stk_code": "002745.SZ",
                    "issue_size": 7.5e8, "conv_price": 12.5, "remain_size": 0.0,
                    "newest_rating": "AA", "delist_date": "20240611",
                    "list_date": "20180629",
                    "available_at": "2018-06-29 23:59:59+08:00",
                    "available_at_rule": "conservative_date_eod",
                }]),
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            window_start = pd.Timestamp("2021-09-01", tz="Asia/Shanghai")
            events, events_meta = builder._build_available_at_domain(
                ("limit_list_d",), DECISION, window_start
            )
            self.assertEqual(len(events), 1)
            self.assertNotIn("limit_amount", events.columns)
            self.assertIn("fd_amount", events.columns)
            self.assertNotIn("limit_amount", events_meta["dataset_columns"]["limit_list_d"])
            macro, _ = builder._build_available_at_domain(
                ("cb_basic",), DECISION, window_start, lifetime_registries=True
            )
            self.assertEqual(len(macro), 1)
            for column in ("conv_price", "remain_size", "newest_rating", "delist_date"):
                self.assertNotIn(column, macro.columns)
            self.assertIn("issue_size", macro.columns)
            # Zero-visible-row builds contribute footer schema: the exclusion
            # must hold there too, or frozen and replay parts would diverge.
            early = datetime(2021, 10, 6, 9, 25, tzinfo=CN_TZ)
            zero_rows, zero_meta = builder._build_available_at_domain(
                ("limit_list_d",), early, window_start
            )
            self.assertEqual(len(zero_rows), 0)
            self.assertNotIn("limit_amount", zero_meta["dataset_columns"]["limit_list_d"])
            self.assertIn("fd_amount", zero_meta["dataset_columns"]["limit_list_d"])

    def test_available_at_union_keeps_dataset_order_when_datasets_load_in_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            write(
                raw / "top_list" / "trade_date=20211007.parquet",
                pd.DataFrame([
                    {
                        "trade_date": "20211007",
                        "ts_code": "000001.SZ",
                        "name": "a",
                        "available_at": "2021-10-07 16:00:00+08:00",
                    }
                ]),
            )
            write(
                raw / "block_trade" / "trade_date=20211007.parquet",
                pd.DataFrame([
                    {
                        "trade_date": "20211007",
                        "ts_code": "000002.SZ",
                        "price": 10.0,
                        "available_at": "2021-10-07 16:00:00+08:00",
                    }
                ]),
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            window_start = pd.Timestamp("2021-09-01", tz="Asia/Shanghai")
            union, meta = builder._build_available_at_domain(
                ("block_trade", "top_list"), DECISION, window_start
            )
            self.assertEqual(list(union["dataset"]), ["block_trade", "top_list"])
            self.assertEqual(meta["datasets"], ["block_trade", "top_list"])
            self.assertIn("price", union.columns)
            self.assertIn("name", union.columns)

    def test_news_text_guards_sources_window_and_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            flash = {"title": "", "channels": "要闻"}
            def write_news(src, day, rows):
                out = raw / "news" / f"src={src}" / f"date={day}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([
                    {"datetime": f"{day[:4]}-{day[4:6]}-{day[6:]} {9 + i:02d}:00:00", "content": content,
                     "available_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T{9 + i:02d}:00:00+08:00",
                     "available_at_rule": "source:datetime", **flash}
                    for i, content in enumerate(rows)
                ]).to_parquet(out, index=False)
            # In-window day: cls carries the flash first; eastmoney repeats one
            # of them later plus a unique item; sina is NOT a configured source.
            write_news("cls", "20210920", ["A股放量上行", "两融余额创新高"])
            write_news("eastmoney", "20210920", ["A股放量上行", "北向资金净流入"])
            write_news("sina", "20210920", ["不应出现的来源"])
            # Outside the 1-month news window (though inside the text window).
            write_news("cls", "20210601", ["过期快讯"])
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            config = SnapshotConfig(
                events_datasets=(), macro_datasets=(), fundamental_datasets=(),
                text_datasets=("news",), news_sources=("cls", "eastmoney"), news_window_months=1,
                intraday_trade_days=1, include_industry=False,
            )
            out_dir = Path(tmp) / "text_out"
            out_dir.mkdir()
            window_start = pd.Timestamp("2021-01-01", tz="Asia/Shanghai")
            index, _ = builder._build_text(config, DECISION, window_start, out_dir)
            news = index[index["dataset"] == "news"]
            bodies = pd.read_parquet(out_dir / "text_library" / "news.parquet")["body"]
            joined = "\n".join(bodies)
            # Duplicate flash collapsed to the earliest copy; unique rows kept.
            self.assertEqual(len(news), 3)
            self.assertEqual(int(bodies.duplicated().sum()), 0)
            self.assertIn("北向资金净流入", joined)
            # Unconfigured source and beyond-window rows never enter.
            self.assertNotIn("不应出现的来源", joined)
            self.assertNotIn("过期快讯", joined)
            # Default (empty news_sources) discovers every source on disk.
            all_sources = SnapshotConfig(
                events_datasets=(), macro_datasets=(), fundamental_datasets=(),
                text_datasets=("news",), news_window_months=None,
                intraday_trade_days=1, include_industry=False,
            )
            out_all = Path(tmp) / "text_out_all"
            out_all.mkdir()
            index_all, _ = builder._build_text(all_sources, DECISION, window_start, out_all)
            joined_all = "\n".join(pd.read_parquet(out_all / "text_library" / "news.parquet")["body"])
            self.assertIn("不应出现的来源", joined_all)   # sina now included
            self.assertIn("过期快讯", joined_all)          # no month clamp
            self.assertEqual(len(index_all[index_all["dataset"] == "news"]), 5)
            # A configured source with no raw directory fails fast.
            bad = SnapshotConfig(
                events_datasets=(), macro_datasets=(), fundamental_datasets=(),
                text_datasets=("news",), news_sources=("cls", "missing_src"),
                intraday_trade_days=1, include_industry=False,
            )
            with self.assertRaises(FileNotFoundError):
                builder._build_text(bad, DECISION, window_start, out_dir)

    def test_announcement_and_major_news_dedup_before_text_id(self):
        # Re-ingested duplicate rows collapse to the earliest copy BEFORE
        # text_id assignment (measured 38% / 9% duplicates in production);
        # distinct filings sharing a title, and joint announcements sharing a
        # url across stocks, are preserved.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"

            def ann(ts_code, title, url, avail):
                return {
                    "ann_date": "20211007", "ts_code": ts_code, "name": "示例",
                    "title": title, "url": url, "rec_time": avail,
                    "available_at": avail, "available_at_rule": "source:rec_time",
                }

            write(
                raw / "anns_d" / "month=202110.parquet",
                pd.DataFrame([
                    ann("000001.SZ", "回购公告", "http://cninfo/U1", "2021-10-07T18:00:00+08:00"),
                    # Literal re-ingest of the same filing, later rec_time.
                    ann("000001.SZ", "回购公告", "http://cninfo/U1", "2021-10-08T09:00:00+08:00"),
                    # Joint announcement: same filing url, different stock.
                    ann("000002.SZ", "回购公告", "http://cninfo/U1", "2021-10-07T18:00:00+08:00"),
                    # Different filing with the same recurring title.
                    ann("000001.SZ", "回购公告", "http://cninfo/U2", "2021-10-07T19:00:00+08:00"),
                ]),
            )

            def news_row(title, content, avail):
                return {
                    "title": title, "content": content, "pub_time": avail, "src": "all",
                    "available_at": avail, "available_at_rule": "source:pub_time",
                }

            write(
                raw / "major_news" / "src=all" / "month=202110.parquet",
                pd.DataFrame([
                    news_row("市场综述", "沪指收涨", "2021-10-07T08:00:00+08:00"),
                    # Literal duplicate push, later timestamp.
                    news_row("市场综述", "沪指收涨", "2021-10-07T09:00:00+08:00"),
                    # Same title, different body: a distinct item.
                    news_row("市场综述", "深成指收跌", "2021-10-07T08:30:00+08:00"),
                ]),
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            config = SnapshotConfig(
                events_datasets=(), macro_datasets=(), fundamental_datasets=(),
                text_datasets=("anns_d", "major_news"),
                intraday_trade_days=1, include_industry=False,
            )
            out_dir = Path(tmp) / "text_out"
            out_dir.mkdir()
            window_start = pd.Timestamp("2021-09-01", tz="Asia/Shanghai")
            index, _ = builder._build_text(config, DECISION, window_start, out_dir)
            anns = index[index["dataset"] == "anns_d"]
            self.assertEqual(len(anns), 3)
            # The surviving copy of the re-ingested filing is the earliest one.
            self.assertNotIn("2021-10-08T09:00:00+08:00", set(anns["available_at"]))
            major = index[index["dataset"] == "major_news"]
            self.assertEqual(len(major), 2)
            self.assertNotIn("2021-10-07T09:00:00+08:00", set(major["available_at"]))
            self.assertEqual(index["text_id"].nunique(), len(index))

    def test_investor_qa_uses_question_title_and_question_answer_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            write(
                raw / "irm_qa_sh" / "date=20211007.parquet",
                pd.DataFrame([{
                    "ts_code": "600000.SH",
                    "name": "浦发银行",
                    "trade_date": "20211007",
                    "q": "公司最新股东人数是多少？",
                    "a": "截至季度末为十万户。",
                    "pub_time": "2021-10-07 18:30:00",
                    "available_at": "2021-10-07 18:30:00+08:00",
                    "available_at_rule": "source:pub_time",
                }]),
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            config = SnapshotConfig(
                events_datasets=(), macro_datasets=(), fundamental_datasets=(),
                text_datasets=("irm_qa_sh",), intraday_trade_days=1, include_industry=False,
            )
            out_dir = Path(tmp) / "text_out"
            out_dir.mkdir()

            index, _ = builder._build_text(
                config, DECISION, pd.Timestamp("2021-01-01", tz="Asia/Shanghai"), out_dir
            )
            bodies = pd.read_parquet(out_dir / "text_library" / "irm_qa_sh.parquet")

            self.assertEqual(index.loc[0, "title"], "公司最新股东人数是多少？")
            self.assertIn("公司最新股东人数是多少？", bodies.loc[0, "body"])
            self.assertIn("截至季度末为十万户。", bodies.loc[0, "body"])

    def test_missing_configured_dataset_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            config = SnapshotConfig(
                events_datasets=("margin_secs", "block_trade"),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=(),
                intraday_trade_days=1,
                include_industry=False,
            )
            with self.assertRaises(FileNotFoundError):
                builder.build_decision_snapshot(DECISION, Path(tmp) / "snap", config)

    def test_missing_fundamental_event_partitions_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            build_raw(raw)
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=("income_vip",),
                include_intraday=False,
                include_industry=False,
            )
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)

            with self.assertRaises(FileNotFoundError):
                SnapshotBuilder(raw, Path(tmp) / "missing_events", status_path).build_decision_snapshot(
                    DECISION, Path(tmp) / "snap", config
                )

    def test_missing_one_configured_fundamental_dataset_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=("income_vip", "balancesheet_vip"),
                include_intraday=False,
                include_industry=False,
            )

            with self.assertRaisesRegex(FileNotFoundError, "balancesheet_vip"):
                SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                    DECISION, Path(tmp) / "snap", config
                )

    def test_fundamental_event_status_is_required_when_fundamentals_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=("income_vip",),
                include_intraday=False,
                include_industry=False,
            )

            with self.assertRaisesRegex(ValueError, "status is required"):
                SnapshotBuilder(raw, events_root).build_decision_snapshot(DECISION, Path(tmp) / "snap", config)

    def test_critical_domain_audit_error_blocks_decision_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            write_domain_statuses(Path(tmp), daily="error")

            with self.assertRaisesRegex(ValueError, "execution-critical domain 'daily'"):
                SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                    DECISION, Path(tmp) / "snap", CONFIG
                )

    def test_fundamental_raw_audit_error_gates_only_fundamentals_enabled(self):
        # The core/finance report split's purpose: a raw-finance audit error
        # hard-fails fundamentals-enabled builds but never blocks experiments
        # that do not enable financial data (previously it rode the
        # unconditional critical daily gate and blocked everything).
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            write_domain_statuses(Path(tmp), fundamentals_raw="error")
            builder = SnapshotBuilder(raw, events_root, status_path)

            with self.assertRaisesRegex(ValueError, "execution-critical domain 'fundamentals_raw'"):
                builder.build_decision_snapshot(DECISION, Path(tmp) / "snap", CONFIG)

            no_fundamentals = replace(CONFIG, fundamental_datasets=())
            manifest = builder.build_decision_snapshot(DECISION, Path(tmp) / "snap2", no_fundamentals)
            self.assertNotIn("fundamentals_raw", manifest["data_quality_warnings"])

    def test_research_domain_audit_error_degrades_to_manifest_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            write_domain_statuses(Path(tmp), macro="error")
            (Path(tmp) / "event_flow_status.json").unlink()  # enabled but never audited

            manifest = SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                DECISION, Path(tmp) / "snap", CONFIG
            )
            warnings = manifest["data_quality_warnings"]
            self.assertIn("audit status is error", warnings["macro"])
            self.assertIn("status file missing", warnings["events"])
            self.assertNotIn("text", warnings)

    def test_board_trading_audit_is_an_independent_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            build_raw(raw)
            events_root = root / "fund_events"
            build_fundamental_events(events_root)
            status_path = root / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            write_domain_statuses(root, board_trading="error")

            warnings = SnapshotBuilder(raw, events_root, status_path)._domain_status_gates(
                replace(CONFIG, events_datasets=("kpl_list",))
            )

            self.assertIn("audit status is error", warnings["board_trading"])

    def test_fundamental_event_audit_scope_must_cover_loaded_months(self):
        # The 120-day-window regression class: a status whose audited range
        # starts AFTER the earliest loadable partition month must not pass the
        # gate — unaudited history would ride in on a narrower report.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)  # partition month 202109
            write_fundamental_status(status_path, scope_start_date="20260408")
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=("income_vip",),
                include_intraday=False,
                include_industry=False,
            )

            with self.assertRaisesRegex(ValueError, "audit window starts at 20260408"):
                SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                    DECISION, Path(tmp) / "snap", config
                )

    def test_fundamental_event_audit_scope_must_cover_enabled_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path, scope_datasets=("dividend",))
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=("income_vip",),
                include_intraday=False,
                include_industry=False,
            )

            with self.assertRaisesRegex(ValueError, "does not cover datasets \\['income_vip'\\]"):
                SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                    DECISION, Path(tmp) / "snap", config
                )

    def test_fundamental_event_audit_error_blocks_decision_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path, status="error", errors=1)
            config = SnapshotConfig(
                events_datasets=(),
                macro_datasets=(),
                text_datasets=(),
                fundamental_datasets=("income_vip",),
                include_intraday=False,
                include_industry=False,
            )

            with self.assertRaises(ValueError):
                SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                    DECISION, Path(tmp) / "snap", config
                )

    def test_disabled_events_do_not_write_an_empty_events_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            out = Path(tmp) / "snap"
            config = replace(
                CONFIG,
                events_datasets=(),
                include_intraday=False,
            )
            SnapshotBuilder(raw, events_root, status_path).build_decision_snapshot(
                DECISION, out, config
            )
            self.assertFalse((out / "events.parquet").exists())
            manifest = load_snapshot_manifest(out)
            self.assertTrue(manifest["domains"]["events"].get("skipped"))

    def test_later_decision_reuses_prior_events_and_unions_only_the_new_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            first = Path(tmp) / "first"
            builder = SnapshotBuilder(raw, events_root, status_path)
            config = replace(CONFIG, include_intraday=False)
            builder.build_decision_snapshot(DECISION, first, config)
            later = datetime(2021, 10, 11, 9, 25, tzinfo=CN_TZ)
            write(
                raw / "daily" / "trade_date=20211011.parquet",
                pd.DataFrame(
                    [
                        {
                            "trade_date": "20211011",
                            "ts_code": "000001.SZ",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.5,
                            "close": 10.6,
                            "pre_close": 10.5,
                            "pct_chg": 1.0,
                            "vol": 1000.0,
                            "amount": 1060.0,
                        }
                    ]
                ),
            )
            write(
                raw / "daily_basic" / "trade_date=20211011.parquet",
                pd.DataFrame(
                    [
                        {
                            "trade_date": "20211011",
                            "ts_code": "000001.SZ",
                            "turnover_rate": 2.0,
                            "dv_ttm": 3.0,
                            "pe": 10.0,
                            "total_share": 100.0,
                            "total_mv": 1000.0,
                            "close": 10.6,
                            "circ_mv": 2_000_000.0,
                        }
                    ]
                ),
            )
            write(
                raw / "margin_secs" / "trade_date=20211011.parquet",
                pd.DataFrame(
                    [
                        {
                            "trade_date": "20211011",
                            "ts_code": "000001.SZ",
                            "available_at": "2021-10-11T09:00:00+08:00",
                            "available_at_rule": "same_day_preopen",
                        }
                    ]
                ),
            )
            second = Path(tmp) / "second"
            manifest = builder.build_decision_snapshot(
                later,
                second,
                config,
                prior_events=(first / "events.parquet", DECISION),
            )
            self.assertTrue(manifest["domains"]["events"].get("incremental"))
            events = pd.read_parquet(second / "events.parquet")
            dates = set(events["trade_date"].astype(str))
            self.assertIn("20211008", dates)
            self.assertIn("20211011", dates)

    def test_forward_unlock_events_survive_the_decision_window(self):
        # IPO lockups announce years ahead: windowing on announcement recency
        # alone hid the largest near-term unlocks (measured 56.8% of the
        # share-weighted supply due within 90 days). The decision snapshot must
        # keep long-announced rows whose float_date is still in scope; replay
        # slots keep the announcement-window semantics (their rows union with
        # the frozen snapshot).
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            write(
                raw / "share_float_complete" / "share_float_complete.parquet",
                pd.DataFrame([
                    # Announced 2019 (far outside the window), unlocking soon.
                    {"ts_code": "688001.SH", "ann_date": "20190715", "float_date": "20211105",
                     "float_share": 1e8, "available_at": "2019-07-15 23:59:59+08:00", "available_at_rule": "r"},
                    # Announced 2019, unlocked long before the window: out.
                    {"ts_code": "600000.SH", "ann_date": "20190110", "float_date": "20200110",
                     "float_share": 1e6, "available_at": "2019-01-10 23:59:59+08:00", "available_at_rule": "r"},
                    # Announced inside the window, past float_date: stays.
                    {"ts_code": "000001.SZ", "ann_date": "20210901", "float_date": "20210910",
                     "float_share": 1e6, "available_at": "2021-09-01 23:59:59+08:00", "available_at_rule": "r"},
                ]),
            )
            builder = SnapshotBuilder(raw, Path(tmp) / "missing_events")
            window_start = pd.Timestamp("2021-04-08", tz=CN_TZ)

            decision, _ = builder._build_available_at_domain(
                ("share_float_complete",), DECISION, window_start, forward_events=True
            )
            self.assertEqual(set(decision["ts_code"]), {"688001.SH", "000001.SZ"})

            slot, _ = builder._build_available_at_domain(
                ("share_float_complete",), DECISION, window_start
            )
            self.assertEqual(set(slot["ts_code"]), {"000001.SZ"})

            # An unlock table without its event date column cannot be windowed
            # for forward events and must fail loudly, not fall through.
            write(
                raw / "share_float_complete" / "share_float_complete.parquet",
                pd.DataFrame([
                    {"ts_code": "000001.SZ", "ann_date": "20210901",
                     "available_at": "2021-09-01 23:59:59+08:00", "available_at_rule": "r"},
                ]),
            )
            with self.assertRaises(ValueError):
                builder._build_available_at_domain(
                    ("share_float_complete",), DECISION, window_start, forward_events=True
                )

    def test_forward_unlock_events_survive_incremental_prior_reuse(self):
        # Incremental reuse must not drop a long-announced unlock whose
        # float_date is still in the decision window. The delta scan cannot
        # resurrect it: available_at is years before prior_time.
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            status_path = Path(tmp) / "fundamental_events_status.json"
            build_raw(raw)
            build_fundamental_events(events_root)
            write_fundamental_status(status_path)
            write(
                raw / "share_float_complete" / "share_float_complete.parquet",
                pd.DataFrame([
                    {"ts_code": "688001.SH", "ann_date": "20190715", "float_date": "20211105",
                     "float_share": 1e8, "available_at": "2019-07-15 23:59:59+08:00", "available_at_rule": "r"},
                    {"ts_code": "600000.SH", "ann_date": "20190110", "float_date": "20200110",
                     "float_share": 1e6, "available_at": "2019-01-10 23:59:59+08:00", "available_at_rule": "r"},
                    {"ts_code": "000001.SZ", "ann_date": "20210901", "float_date": "20210910",
                     "float_share": 1e6, "available_at": "2021-09-01 23:59:59+08:00", "available_at_rule": "r"},
                ]),
            )
            prior_path = Path(tmp) / "prior_events.parquet"
            pd.DataFrame([
                {"dataset": "share_float_complete", "ts_code": "688001.SH", "ann_date": "20190715",
                 "float_date": "20211105", "float_share": 1e8,
                 "available_at": "2019-07-15 23:59:59+08:00", "available_at_rule": "r"},
                {"dataset": "share_float_complete", "ts_code": "600000.SH", "ann_date": "20190110",
                 "float_date": "20200110", "float_share": 1e6,
                 "available_at": "2019-01-10 23:59:59+08:00", "available_at_rule": "r"},
                {"dataset": "share_float_complete", "ts_code": "000001.SZ", "ann_date": "20210901",
                 "float_date": "20210910", "float_share": 1e6,
                 "available_at": "2021-09-01 23:59:59+08:00", "available_at_rule": "r"},
            ]).to_parquet(prior_path, index=False)
            config = replace(
                CONFIG,
                events_datasets=("share_float_complete",),
                events_window_months=6,
                include_intraday=False,
            )
            prior_time = datetime(2021, 10, 7, 9, 25, tzinfo=CN_TZ)
            builder = SnapshotBuilder(raw, events_root, status_path)
            manifest = builder.build_decision_snapshot(
                DECISION, Path(tmp) / "snap", config, prior_events=(prior_path, prior_time)
            )
            self.assertTrue(manifest["domains"]["events"].get("incremental"))
            events = pd.read_parquet(Path(tmp) / "snap" / "events.parquet")
            self.assertEqual(set(events["ts_code"]), {"688001.SH", "000001.SZ"})

            with self.assertRaises(ValueError):
                builder._extend_prior_events(
                    pd.DataFrame([
                        {"dataset": "share_float_complete", "ts_code": "688001.SH",
                         "available_at": "2019-07-15 23:59:59+08:00", "available_at_rule": "r"},
                    ]),
                    prior_time,
                    ("share_float_complete",),
                    DECISION,
                    pd.Timestamp("2021-04-08", tz=CN_TZ),
                    screen=None,
                )


if __name__ == "__main__":
    unittest.main()


def test_external_union_snapshot_fixture_finalizes_under_the_unit_contract(tmp_path):
    """The shared decision/replay snapshot fixtures must satisfy the same
    manifest and unit-registry gate as a builder-produced snapshot: an external
    union directory declares its dataset ownership explicitly, and every column
    must classify."""
    from .fixtures_sandbox import PRICE_ROWS, TS_CODE, FakeSnapshotProvider

    provider = FakeSnapshotProvider()
    decision = provider.decision_snapshot(
        datetime(2026, 1, 5, 8, 30, tzinfo=ZoneInfo("Asia/Shanghai")), tmp_path / "decision"
    )
    assert decision["kind"] == "decision_input"
    assert decision["decision_date"] == "20260105"
    assert decision["snapshot_id"].startswith("snap_")
    manifest = load_snapshot_manifest(tmp_path / "decision")
    assert manifest["snapshot_id"] == decision["snapshot_id"]
    assert set(manifest["domains"]) == {"events", "fundamentals"}
    universe = pd.read_parquet(tmp_path / "decision" / "universe.parquet")
    assert universe["ts_code"].tolist() == [TS_CODE]

    replay = provider.replay_slot("20260105", "20260331", tmp_path / "replay", label="valid")
    assert replay["kind"] == "replay_slot"
    assert (replay["period_start"], replay["period_end"]) == ("20260105", "20260331")
    days = pd.read_parquet(tmp_path / "replay" / "daily.parquet")["trade_date"].tolist()
    assert days == [day for day, _, _ in PRICE_ROWS if "20260105" <= day <= "20260331"]

    # Dataset ownership is never inferred from file content: a union directory
    # finalized without declaring it is refused, not silently accepted.
    stray = tmp_path / "stray"
    stray.mkdir()
    pd.DataFrame([{"dataset": "margin_secs", "trade_date": "20260105", "ts_code": TS_CODE}]).to_parquet(
        stray / "events.parquet", index=False
    )
    with pytest.raises(ValueError, match="dataset_columns"):
        finalize_snapshot_dir(stray, kind="decision_input", decision_date="20260105")
