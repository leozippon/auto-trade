# Consolidated unit tests: test_data_sources_tushare.py


# Source: test_tushare_download_update_guards.py
import argparse
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pyarrow.lib import ArrowInvalid

from autotrade.data_sources.tushare import audit, common, cron_update, download
from autotrade.data_sources.tushare import io as tushare_io
from autotrade.environment.data.snapshot import (
    SELECTABLE_DATASETS,
    SnapshotConfig,
)


class EmptyMinuteClient:
    def query(self, api_name, params=None, fields="", retries=5):
        return common.ApiResult(fields.split(",") if fields else [], [])


class CountingMacroClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        row = []
        for field in result_fields:
            if field == "month":
                row.append(params.get("m", "202605"))
            elif field in {"date", "trade_date"}:
                row.append(params.get("end_date", "20260529"))
            elif field == "publish_date":
                row.append("20260529")
            elif field == "title":
                row.append("sample")
            elif field == "issuing_org":
                row.append("sample_org")
            elif field == "data_api":
                row.append(api_name)
            else:
                row.append("")
        return common.ApiResult(result_fields, [row])


class BoardClient:
    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        result_fields = fields.split(",") if fields else []
        row = []
        for field in result_fields:
            if field == "trade_date":
                row.append(params.get("trade_date", "20200102"))
            elif field == "ts_code":
                row.append("000001.SZ")
            elif field in {"name", "ts_name"}:
                row.append("sample")
            elif field == "tag":
                row.append(params.get("tag", "涨停"))
            elif field == "limit_type":
                row.append(params.get("limit_type", "涨停池"))
            elif field == "data_type":
                row.append(params.get("market", "热股"))
            elif field == "rank_time":
                row.append("2020-01-02 10:00:00")
            elif field == "hm_name":
                row.append("sample_hot_money")
            elif field == "hm_orgs":
                row.append("sample_org")
            elif field == "exalter":
                row.append("sample_broker")
            elif field == "side":
                row.append("0")
            elif field == "reason":
                row.append("sample_reason")
            elif field == "nums":
                row.append("2")
            elif field == "rank":
                row.append(1)
            elif field == "desc":
                row.append("sample_desc")
            elif field == "orgs":
                row.append("[]")
            else:
                row.append(1.0)
        return common.ApiResult(result_fields, [row])


class NoQueryClient:
    def query(self, api_name, params=None, fields="", retries=5):
        raise AssertionError(f"unexpected TuShare query: {api_name}")


class ReferenceClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        rows = []
        if api_name == "stock_basic" and params.get("list_status") == "L":
            rows = [["000001.SZ" if field == "ts_code" else params.get("list_status", "") if field == "list_status" else "" for field in result_fields]]
        elif api_name == "stock_company":
            rows = [["000001.SZ" if field == "ts_code" else params.get("exchange", "") if field == "exchange" else "" for field in result_fields]]
        elif api_name == "namechange":
            rows = [[params.get("ts_code", ""), "sample", "20200101", "", "20200101", "name"]]
        elif api_name == "index_classify":
            rows = [["801010.SI" if field == "index_code" else "L1" if field == "level" else "sample" for field in result_fields]]
        elif api_name == "index_member_all":
            rows = [["801010.SI" if field == "l1_code" else "000001.SZ" if field == "ts_code" else "" for field in result_fields]]
        elif api_name == "index_member":
            rows = [[params.get("index_code", "") if field == "index_code" else "000001.SZ" if field == "con_code" else "20111010" if field == "in_date" else "N" if field == "is_new" else "" for field in result_fields]]
        elif api_name == "ths_index":
            rows = [["885001.TI" if field == "ts_code" else "N" if field == "type" else "sample" for field in result_fields]]
        elif api_name == "ths_member":
            rows = [[params.get("ts_code", "") if field == "ts_code" else "000001.SZ" if field == "con_code" else "" for field in result_fields]]
        elif api_name == "index_basic":
            rows = [["000300.SH" if field == "ts_code" else "sample" for field in result_fields]]
        elif api_name == "index_weight":
            rows = [[params.get("index_code", "") if field == "index_code" else "000001.SZ" if field == "con_code" else "20260630" if field == "trade_date" else "1.0" for field in result_fields]]
        return common.ApiResult(result_fields, rows)


class TradeCalClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        rows = []
        if api_name == "trade_cal":
            cal_date = params.get("end_date", "20260604")
            rows = [[
                params.get("exchange", "") if field == "exchange" else
                cal_date if field == "cal_date" else
                "1" if field == "is_open" else
                "20260603" if field == "pretrade_date" else
                ""
                for field in result_fields
            ]]
        return common.ApiResult(result_fields, rows)


class EmptyReferenceClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        return common.ApiResult(fields.split(",") if fields else [], [])


class DailyMarketClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        row = []
        for field in result_fields:
            if field == "trade_date":
                row.append(params.get("trade_date", "20200102"))
            elif field == "ts_code":
                row.append("000001.SZ")
            elif field == "adj_factor":
                row.append(1.0)
            else:
                row.append(0)
        return common.ApiResult(result_fields, [row])


class FundamentalClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        if api_name == "income_vip":
            result_fields = ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type"]
            row = ["000001.SZ", "20200430", "20200430", params.get("period", "20200331"), "1", "1", "1"]
        elif api_name in {"dividend", "fina_audit", "fina_mainbz_vip"}:
            result_fields = ["ts_code", "ann_date", "end_date"]
            row = [params.get("ts_code", "000001.SZ"), "20200430", "20200331"]
        else:
            result_fields = ["ts_code"]
            row = ["000001.SZ"]
        return common.ApiResult(result_fields, [row])


class CalendarEventClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        result_fields = fields.split(",") if fields else []
        if api_name == "trade_cal":
            row = []
            for field in result_fields:
                if field == "exchange":
                    row.append(params.get("exchange", "SSE"))
                elif field == "cal_date":
                    row.append(params.get("end_date", "20260604"))
                elif field == "is_open":
                    row.append("1")
                elif field == "pretrade_date":
                    row.append("20260603")
                else:
                    row.append("")
            return common.ApiResult(result_fields, [row])
        if api_name == "margin_secs":
            rows = []
            for ts_code, exchange in (("600000.SH", "SSE"), ("000001.SZ", "SZSE"), ("830000.BJ", "BSE")):
                row = []
                for field in result_fields:
                    if field == "trade_date":
                        row.append(params.get("trade_date", "20260604"))
                    elif field == "ts_code":
                        row.append(ts_code)
                    elif field == "exchange":
                        row.append(exchange)
                    elif field == "name":
                        row.append("sample")
                    else:
                        row.append("")
                rows.append(row)
            return common.ApiResult(result_fields, rows)
        return common.ApiResult(result_fields, [])


class EmptyTradeDateClient:
    def __init__(self):
        self.calls = []

    def query(self, api_name, params=None, fields="", retries=5):
        params = params or {}
        self.calls.append((api_name, dict(params)))
        return common.ApiResult(fields.split(",") if fields else [], [])


class ErrorTradeDateClient:
    def query(self, api_name, params=None, fields="", retries=5):
        raise RuntimeError("mock source failure")


class RepeatingPagedClient:
    def query(self, api_name, params=None, fields="", retries=5):
        result_fields = fields.split(",") if fields else ["trade_date", "ts_code"]
        row = ["20200102" if field == "trade_date" else "000001.SZ" for field in result_fields]
        return common.ApiResult(result_fields, [row])


class TuShareDownloadUpdateGuardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_dir = self.root / "raw"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_trade_cal(self, trade_date="20200102", is_open="1"):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / f"year={trade_date[:4]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"cal_date": trade_date, "is_open": is_open}]).to_parquet(path, index=False)

    def _write_daily_universe(self, trade_date="20200102"):
        path = self.raw_dir / "daily" / f"trade_date={trade_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"trade_date": trade_date, "ts_code": "000001.SZ"},
            {"trade_date": trade_date, "ts_code": "000002.SZ"},
        ]).to_parquet(path, index=False)

    def test_default_revision_ledger_for_temp_raw_stays_local(self):
        ledger = common.resolve_revision_ledger(self.raw_dir, common.REVISION_EVENTS_PATH, repo_root=Path.cwd())

        self.assertEqual(ledger, self.root / "revision_events.jsonl")

    def test_revision_ledger_appends_each_logical_event_once(self):
        from autotrade.data_sources.tushare.io import append_jsonl_unique

        # ``event_id`` and ``write_id`` are fresh UUIDs per write, so the
        # ledger dedupes on the event's stable content: re-detecting the same
        # revision appends nothing, a genuinely different one appends.
        ledger = self.root / "revision_events.jsonl"
        first = {"event_id": "a", "detected_at": "first", "write_id": "w1", "dataset": "daily", "changed_keys": 1}
        repeated = {"event_id": "b", "detected_at": "later", "write_id": "w2", "dataset": "daily", "changed_keys": 1}
        distinct = {"event_id": "c", "detected_at": "later", "write_id": "w3", "dataset": "daily", "changed_keys": 2}

        self.assertTrue(append_jsonl_unique(ledger, first, key="event_id"))
        self.assertFalse(append_jsonl_unique(ledger, repeated, key="event_id"))
        self.assertTrue(append_jsonl_unique(ledger, distinct, key="event_id"))
        records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records, [first, distinct])
        # A non-event key keeps the plain by-value contract.
        other = self.root / "by_value.jsonl"
        self.assertTrue(append_jsonl_unique(other, {"job": "a", "n": 1}, key="job"))
        self.assertFalse(append_jsonl_unique(other, {"job": "a", "n": 2}, key="job"))
        self.assertTrue(append_jsonl_unique(other, {"job": "b", "n": 1}, key="job"))
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            append_jsonl_unique(other, {"job": ""}, key="job")

    def test_load_stock_codes_keeps_only_valid_a_share_codes(self):
        path = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ"},
                {"ts_code": "T00018.SH"},
                {"ts_code": "TS0018.SH"},
                {"ts_code": "920126.BJ"},
                {"ts_code": "bad"},
            ]
        ).to_parquet(path, index=False)

        self.assertEqual(common.load_stock_codes(self.raw_dir), ["000001.SZ", "920126.BJ"])

    def test_query_paged_rejects_repeated_full_pages(self):
        with self.assertRaisesRegex(RuntimeError, "returned a repeated page"):
            common.query_paged(RepeatingPagedClient(), "daily", {"trade_date": "20200102"}, "trade_date,ts_code", page_limit=1)

    def test_trade_cal_helpers_normalize_date_strings(self):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "2026-06-03", "is_open": "1"},
                {"cal_date": "20260604", "is_open": "0"},
                {"cal_date": "2026/06/05", "is_open": "1"},
            ]
        ).to_parquet(path, index=False)

        self.assertEqual(common.load_sse_open_dates(self.raw_dir, "20260603", "20260605"), ["20260603", "20260605"])
        self.assertEqual(common.latest_sse_calendar_date(self.raw_dir), "20260605")
        self.assertTrue(download.sse_trade_cal_covers(self.raw_dir, "20260603", "20260605"))

    def test_audit_window_never_expects_future_trading_days(self):
        """The calendar is published ahead; the audit must not demand data for
        days that have not happened, or it can never reach ok and real gaps
        hide in the permanent noise."""
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2126.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        # A calendar reaching a century into the future, as the real one does
        # for the rest of the current year.
        pd.DataFrame([{"cal_date": "21260605", "is_open": "1"}]).to_parquet(path, index=False)
        self.assertEqual(common.latest_sse_calendar_date(self.raw_dir), "21260605")

        # The window ends at the last SETTLED day: today's session has not
        # closed and is not ingested until the evening cron, so expecting it
        # would keep the audit red from midnight onward -- the same
        # desensitization as expecting future days.
        today = datetime.now(audit.CN_TZ).strftime("%Y%m%d")
        resolved = audit.resolve_audit_end_date(self.raw_dir, None)
        self.assertLess(resolved, today)
        self.assertEqual(resolved, (datetime.now(audit.CN_TZ) - timedelta(days=1)).strftime("%Y%m%d"))
        # An explicit window is the operator's business and is never clamped:
        # re-auditing a historical window must stay possible.
        self.assertEqual(audit.resolve_audit_end_date(self.raw_dir, "20200101"), "20200101")

    def test_bak_basic_audit_ignores_trade_cal_lookahead_after_end_date(self):
        path = self.raw_dir / "bak_basic" / "trade_date=20260618.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20260618", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        findings = []

        audit.audit_bak_basic(self.raw_dir, {"20260618", "20260622"}, "20260618", lambda *item: findings.append(item))

        partition_finding = next(item for item in findings if item[1] == "bak_basic_partitions")
        self.assertEqual(partition_finding[0], "info")
        self.assertEqual(partition_finding[3]["missing_expected_files"], 0)
        self.assertEqual(partition_finding[3]["missing_sample"], [])

    def test_update_intraday_by_date_refuses_zero_row_write_for_nonempty_universe(self):
        self._write_trade_cal()
        self._write_daily_universe()
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            output_dataset=common.STK_MINS_BY_DATE_DATASET,
            expected_codes_source="daily",
            codes=None,
            max_codes=None,
            min_rows_per_day=0,
            allow_missing_codes=2,
            allow_validation_warnings=True,
            max_retries=1,
            retry_delay_seconds=0,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            force=False,
        )

        output = self.raw_dir / common.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet"
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyMinuteClient()):
            with self.assertRaisesRegex(RuntimeError, "refusing to write zero-row intraday"):
                download.update_intraday_by_date(args)
        self.assertFalse(output.exists())

    def test_minute_expected_universe_uses_existing_minute_store_when_present(self):
        self._write_daily_universe()
        minute_path = self.raw_dir / common.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet"
        minute_rows = pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "trade_time": "2020-01-02 09:30:00",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "vol": 100,
                "amount": 100.0,
                "trade_date": "20200102",
                "available_at": "2020-01-02 09:30:00+08:00",
                "available_at_rule": "source:trade_time_bar_close",
            },
            {
                "ts_code": "000001.SZ",
                "trade_time": "2020-01-02 15:00:00",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "vol": 100,
                "amount": 100.0,
                "trade_date": "20200102",
                "available_at": "2020-01-02 15:00:00+08:00",
                "available_at_rule": "source:trade_time_bar_close",
            },
        ])
        common.write_parquet(
            minute_path,
            minute_rows,
            api_name=common.STK_MINS_API_NAME,
            params={},
            fields=list(minute_rows.columns),
        )

        codes = common.intraday_expected_codes_for_day(
            self.raw_dir,
            argparse.Namespace(expected_codes_source="minute", output_dataset=common.STK_MINS_BY_DATE_DATASET, codes=None, max_codes=None),
            "20200102",
        )

        self.assertEqual(codes, {"000001.SZ"})

    def test_event_flow_trade_date_download_skips_non_trading_day(self):
        self._write_trade_cal("20260530", is_open="0")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260530",
            end_date="20260530",
            datasets=["margin", "margin_detail", "margin_secs"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=NoQueryClient()):
            self.assertEqual(download.download_event_flow(args), 0)

    def test_share_float_union_inherits_baseline_and_prefers_refreshed_rows(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "000002.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h2", "share_type": "t", "float_share": 2.0, "float_ratio": 0.2},
        ])
        common.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20200101",
            ann_end_date="20200102",
            float_start_date="20200101",
            float_end_date="20200102",
            skip_float_date_union=False,
        )

        with patch.object(download, "share_float_union_files", return_value=[]):
            report = {}
            download.write_share_float_union(self.raw_dir, args, report)
        self.assertEqual(common.parquet_rows(output), 2)
        self.assertTrue(report["union"]["baseline_inherited"])

        # Refreshed source rows precede the baseline and replace matching keys.
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        rows = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 10.0, "float_ratio": 1.0},
            {"ts_code": "000002.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h2", "share_type": "t", "float_share": 20.0, "float_ratio": 2.0},
        ])
        common.write_parquet(source, rows, api_name="share_float", params={}, fields=list(rows.columns))
        report = {}
        with patch.object(download, "share_float_union_files", return_value=[(source, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, report)
        self.assertEqual(report["union"]["previous_rows"], 2)
        self.assertEqual(report["union"]["rows_after_dedup"], 2)
        updated = pd.read_parquet(output).sort_values("ts_code")
        self.assertEqual(updated["float_share"].tolist(), [10.0, 20.0])
        self.assertTrue((self.root / "revision_events.jsonl").exists())

    def test_share_float_union_reads_baseline_once_and_republishes_active_content(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        baseline = pd.DataFrame([{
            "ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
            "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1,
        }])
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102", skip_float_date_union=False,
        )

        def publish(value: float) -> str:
            common.write_parquet(output, baseline, api_name="share_float", params={}, fields=list(baseline.columns))
            active = baseline.copy()
            active["float_share"] = value
            common.write_parquet(source, active, api_name="share_float", params={}, fields=list(active.columns))
            with patch.object(download, "share_float_union_files", return_value=[(source, "ann_date")]):
                with patch.object(pd, "read_parquet", wraps=pd.read_parquet) as read_parquet:
                    download.write_share_float_union(self.raw_dir, args, {})
            output_reads = [
                call for call in read_parquet.call_args_list
                if Path(call.args[0]).resolve() == output.resolve()
            ]
            self.assertEqual(len(output_reads), 1)
            return (
                str(common.parquet_meta(output)["write_id"]),
                pd.read_parquet(output)["float_share"].tolist(),
            )

        first, second = publish(10.0), publish(11.0)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual((first[1], second[1]), ([10.0], [11.0]))

    def test_share_float_union_requires_baseline_for_incremental_update(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
        )

        with patch.object(download, "share_float_union_files") as list_files:
            with self.assertRaisesRegex(RuntimeError, "baseline is missing"):
                download.write_share_float_union(self.raw_dir, args, {})
        list_files.assert_not_called()
        self.assertFalse(output.exists())

    def test_share_float_union_replaces_group_but_preserves_distinct_lots_idempotently(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "000002.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h2", "share_type": "t", "float_share": 2.0, "float_ratio": 0.2},
        ])
        common.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        refreshed = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 10.0, "float_ratio": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 20.0, "float_ratio": 2.0},
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 20.0, "float_ratio": 2.0},
        ])
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        common.write_parquet(source, refreshed, api_name="share_float", params={}, fields=list(refreshed.columns))
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
        )

        with patch.object(download, "share_float_union_files", return_value=[(source, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, {})
            first = pd.read_parquet(output)
            download.write_share_float_union(self.raw_dir, args, {})
            second = pd.read_parquet(output)

        self.assertEqual(len(first), 3)
        group = first[first["ts_code"] == "000001.SZ"].sort_values("float_share")
        self.assertEqual(group["float_share"].tolist(), [10.0, 20.0])
        pd.testing.assert_frame_equal(first, second)
        events = [
            json.loads(line)
            for line in (self.root / "revision_events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["old_rows"], events[0]["new_rows"]), (2, 3))

    def test_share_float_union_file_scope_uses_active_window(self):
        recent = self.raw_dir / "share_float_ann_date" / "ann_date=20200102.parquet"
        historical = self.raw_dir / "share_float_ann_date" / "ann_date=20190101.parquet"
        for path in (recent, historical):
            common.write_parquet(path, pd.DataFrame(), api_name="share_float", params={}, fields=[])
        args = argparse.Namespace(
            ann_start_date="20200101", ann_end_date="20200103",
            float_start_date="20200101", float_end_date="20200103",
            skip_float_date_union=False,
        )

        files = [path for path, _ in download.share_float_union_files(self.raw_dir, args)]
        self.assertEqual(files, [recent])

    def test_share_float_union_capped_group_preserves_complete_baseline(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": float(index), "float_ratio": 0.1}
            for index in range(7000)
        ])
        common.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        common.write_parquet(
            source,
            existing.iloc[:common.SHARE_FLOAT_ROW_LIMIT],
            api_name="share_float",
            params={},
            fields=list(existing.columns),
        )
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
        )

        report = {}
        with patch.object(download, "share_float_union_files", return_value=[(source, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, report)

        self.assertEqual(common.parquet_rows(output), 7000)
        self.assertEqual(report["union"]["rows_after_dedup"], 7000)
        self.assertEqual(report["union"]["capped_groups_preserved"], 1)

    def test_share_float_union_non_capped_source_makes_duplicate_group_replaceable(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1},
        ])
        common.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        refreshed = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 10.0, "float_ratio": 1.0},
        ])
        capped = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        rescue = self.raw_dir / "share_float_ann_date_ts_code" / "ann_date=20200101" / "ts_code=000001.SZ.parquet"
        common.write_parquet(
            capped,
            pd.concat([refreshed] * common.SHARE_FLOAT_ROW_LIMIT, ignore_index=True),
            api_name="share_float",
            params={},
            fields=list(refreshed.columns),
        )
        common.write_parquet(rescue, refreshed, api_name="share_float", params={}, fields=list(refreshed.columns))
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
        )

        report = {}
        with patch.object(
            download,
            "share_float_union_files",
            return_value=[(capped, "ann_date"), (rescue, "ann_date_ts_code")],
        ):
            download.write_share_float_union(self.raw_dir, args, report)

        result = pd.read_parquet(output)
        self.assertEqual(result["float_share"].tolist(), [10.0])
        self.assertEqual(result["download_path"].tolist(), ["ann_date_ts_code"])
        self.assertEqual(result["source_cap_risk"].tolist(), [False])
        self.assertEqual(report["union"]["capped_groups_preserved"], 0)

    def test_share_float_union_supersedes_redated_groups(self):
        # Provider re-dating (observed 2026-07: 301583.SZ moved from
        # ann_date 20260630 to 20260709): the same physical unlock event
        # (ts_code+float_date+holder+share_type) under a corrected ann_date
        # supersedes the stale-dated baseline group -- never double-counted;
        # removed rows land in the revision event. Distinct physical events
        # (other holders/dates) are untouched.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 2.0, "float_ratio": 0.2},
            # Whitespace variant of an incoming holder (observed live:
            # "...企业 (有限合伙)" vs "...企业(有限合伙)"): must supersede too.
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "基金 (有限合伙)", "share_type": "首发原始股", "float_share": 9.0, "float_ratio": 0.9},
            {"ts_code": "000009.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hB", "share_type": "首发原始股", "float_share": 3.0, "float_ratio": 0.3},
        ])
        common.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        redated = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260709", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 5.0, "float_ratio": 0.5},
            {"ts_code": "301583.SZ", "ann_date": "20260709", "float_date": "20270712",
             "holder_name": "基金(有限合伙)", "share_type": "首发原始股", "float_share": 9.0, "float_ratio": 0.9},
        ])
        common.write_parquet(source, redated, api_name="share_float", params={}, fields=list(redated.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260701", ann_end_date="20260710",
            float_start_date="20260701", float_end_date="20280101",
            skip_float_date_union=False,
        )
        report = {}
        with patch.object(download, "share_float_union_files", return_value=[(source, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(output)
        stale = union[(union["ts_code"] == "301583.SZ") & (union["ann_date"] == "20260630")]
        fresh = union[(union["ts_code"] == "301583.SZ") & (union["ann_date"] == "20260709")]
        other = union[union["ts_code"] == "000009.SZ"]
        self.assertTrue(stale.empty)          # superseded (incl. whitespace variant)
        self.assertEqual(len(fresh), 2)
        self.assertEqual(len(other), 1)       # distinct physical event untouched
        # Two baseline groups vanish: hA's stale-dated group and the
        # whitespace-variant holder's group.
        self.assertEqual(report["union"]["superseded_redated_groups"], 2)

        # Idempotent re-run: nothing left to supersede, content stable.
        report = {}
        with patch.object(download, "share_float_union_files", return_value=[(source, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, report)
        self.assertEqual(report["union"]["superseded_redated_groups"], 0)
        self.assertEqual(common.parquet_rows(output), 3)

    def test_share_float_active_window_holding_both_datings_heals(self):
        # The real production shape (301583.SZ): the refresh window covers the
        # stale AND the corrected ann_date partition, so the stale rows arrive
        # in `active` too. Baseline-only supersession left 72 rows across 3
        # ann_dates unchanged; active-side pruning is what makes it heal.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        rows = []
        for ann, share in (("20260630", 1.0), ("20260701", 2.0), ("20260709", 3.0)):
            rows.append({"ts_code": "301583.SZ", "ann_date": ann, "float_date": "20270712",
                         "holder_name": "hA", "share_type": "首发原始股",
                         "float_share": share, "float_ratio": share / 10})
        # An unrelated event must be untouched by the pruning.
        rows.append({"ts_code": "000009.SZ", "ann_date": "20260630", "float_date": "20270712",
                     "holder_name": "hB", "share_type": "首发原始股", "float_share": 9.0, "float_ratio": 0.9})
        existing = pd.DataFrame(rows)
        common.write_parquet(output, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        files = []
        for ann in ("20260630", "20260701", "20260709"):
            path = self.raw_dir / "share_float_ann_date" / f"ann_date={ann}.parquet"
            part = existing[existing["ann_date"] == ann]
            common.write_parquet(path, part, api_name="share_float", params={},
                                 fields=list(part.columns))
            files.append((path, "ann_date"))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20260601", float_end_date="20280101",
            skip_float_date_union=False,
        )
        report = {}
        with patch.object(download, "share_float_union_files", return_value=files):
            download.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(output)
        event = union[union["ts_code"] == "301583.SZ"]
        self.assertEqual(sorted(event["ann_date"]), ["20260709"])  # newest dating only
        self.assertEqual(event["float_share"].tolist(), [3.0])
        self.assertEqual(len(union[union["ts_code"] == "000009.SZ"]), 1)
        self.assertEqual(report["union"]["active_stale_restatement_rows_dropped"], 2)

        # Idempotent: nothing left to prune or supersede.
        report2 = {}
        with patch.object(download, "share_float_union_files", return_value=files):
            download.write_share_float_union(self.raw_dir, args, report2)
        self.assertEqual(report2["union"]["superseded_redated_groups"], 0)
        self.assertEqual(common.parquet_rows(output), 2)

    def test_share_float_capped_newer_dating_does_not_prune_active(self):
        # A capped (truncated) partition proves nothing about rows it could not
        # return, so it must not win the newest-dating comparison.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
        ])
        common.write_parquet(output, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        old_path = self.raw_dir / "share_float_ann_date" / "ann_date=20260630.parquet"
        common.write_parquet(old_path, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        capped = pd.DataFrame([
            dict(existing.iloc[0], ann_date="20260709", float_share=2.0),
            *[{"ts_code": f"9{i:05d}.SZ", "ann_date": "20260709", "float_date": "20270712",
               "holder_name": f"h{i}", "share_type": "首发原始股", "float_share": 1.0,
               "float_ratio": 0.1} for i in range(common.SHARE_FLOAT_ROW_LIMIT - 1)],
        ])
        new_path = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        common.write_parquet(new_path, capped, api_name="share_float", params={},
                             fields=list(capped.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20260601", float_end_date="20280101",
            skip_float_date_union=False,
        )
        report = {}
        with patch.object(download, "share_float_union_files",
                          return_value=[(old_path, "ann_date"), (new_path, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, report)
        union = pd.read_parquet(output)
        event = union[(union["ts_code"] == "301583.SZ")]
        self.assertEqual(sorted(event["ann_date"]), ["20260630", "20260709"])  # conservative union
        self.assertEqual(report["union"]["active_stale_restatement_rows_dropped"], 0)

    def test_share_float_capped_identity_cannot_supersede_baseline_history(self):
        # Symmetry with the active-side rule: a capped (truncated) partition
        # proves nothing about rows it could not return, so it must not delete
        # an older announcement of the same physical event either.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
        ])
        common.write_parquet(output, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        # Only a CAPPED partition carries the newer dating.
        capped = pd.DataFrame([
            dict(existing.iloc[0], ann_date="20260709", float_share=2.0),
            *[{"ts_code": f"9{i:05d}.SZ", "ann_date": "20260709", "float_date": "20270712",
               "holder_name": f"h{i}", "share_type": "首发原始股", "float_share": 1.0,
               "float_ratio": 0.1} for i in range(common.SHARE_FLOAT_ROW_LIMIT - 1)],
        ])
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        common.write_parquet(path, capped, api_name="share_float", params={},
                             fields=list(capped.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260701", ann_end_date="20260731",
            float_start_date="20260601", float_end_date="20280101",
            skip_float_date_union=False,
        )
        report = {}
        with patch.object(download, "share_float_union_files", return_value=[(path, "ann_date")]):
            download.write_share_float_union(self.raw_dir, args, report)
        union = pd.read_parquet(output)
        event = union[union["ts_code"] == "301583.SZ"]
        self.assertEqual(sorted(event["ann_date"]), ["20260630", "20260709"])  # old row kept
        self.assertEqual(report["union"]["superseded_redated_groups"], 0)

    def test_event_flow_audit_reports_cross_ann_date_identity_duplication(self):
        # The formal audit was blind to business-identity duplication: row and
        # key checks cannot see one physical unlock retained under several
        # announcement dates (measured live: >145k such groups).
        union = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        rows = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "基金 (有限合伙)", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "301583.SZ", "ann_date": "20260709", "float_date": "20270712",
             "holder_name": "基金(有限合伙)", "share_type": "首发原始股", "float_share": 2.0, "float_ratio": 0.2},
            {"ts_code": "000009.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hB", "share_type": "首发原始股", "float_share": 3.0, "float_ratio": 0.3},
        ])
        common.write_parquet(union, rows, api_name="share_float_complete", params={},
                             fields=list(rows.columns))
        findings: list[dict] = []

        def add(severity, check, message, details=None):
            findings.append({"severity": severity, "check": check, "details": details or {}})

        audit.audit_share_float_complete_union(self.raw_dir, add)
        finding = next(f for f in findings if f["check"] == "share_float_complete_union")
        self.assertEqual(finding["severity"], "warning")
        # whitespace variants of one holder count as ONE physical identity
        self.assertEqual(finding["details"]["cross_ann_date_identity_groups"], 1)
        self.assertEqual(finding["details"]["stale_redated_rows"], 1)

    def test_share_float_empty_refresh_keeps_existing_cap_risk_signal(self):
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        fields = common.SHARE_FLOAT_FIELDS.split(",")
        existing = pd.DataFrame([
            {
                field: (
                    f"{index:06d}.SZ" if field == "ts_code"
                    else "20200101" if field == "ann_date"
                    else "20200102" if field == "float_date"
                    else str(index) if field == "holder_name"
                    else "type" if field == "share_type"
                    else 1
                )
                for field in fields
            }
            for index in range(common.SHARE_FLOAT_ROW_LIMIT)
        ])
        common.write_parquet(path, existing, api_name="share_float", params={}, fields=fields)

        result = download.query_share_float_to_path(
            EmptyTradeDateClient(),
            self.raw_dir,
            path,
            {"ann_date": "20200101"},
            "ann_date",
            True,
            revision_ledger=None,
            allow_empty_revision_overwrite=False,
        )

        self.assertTrue(result["skipped"])
        self.assertEqual(result["rows"], common.SHARE_FLOAT_ROW_LIMIT)
        self.assertTrue(result["source_cap_risk"])
        self.assertEqual(common.parquet_rows(path), common.SHARE_FLOAT_ROW_LIMIT)

    def test_share_float_nonempty_rejected_refresh_fails(self):
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        fields = common.SHARE_FLOAT_FIELDS.split(",")
        existing = pd.DataFrame([
            [f"{index:06d}.SZ", "20200101", "20200102", f"h{index}", "type", 1.0, 0.1]
            for index in range(100)
        ], columns=fields)
        existing = common.augment_event_frame(existing, common.EVENT_FLOW_SPECS["share_float"])
        common.write_parquet(path, existing, api_name="share_float", params={}, fields=list(existing.columns))

        class ShrunkClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return common.ApiResult(
                    common.SHARE_FLOAT_FIELDS.split(","),
                    [["000000.SZ", "20200101", "20200102", "h0", "type", 1.0, 0.1]])

        # The guard keeps the old partition; that decision is REPORTED, not
        # raised. Raising here aborted the whole run mid-loop, which made the
        # ts_code rescue unreachable and left the lake dirty every time a
        # capped partition rotated its keys.
        result = download.query_share_float_to_path(
            ShrunkClient(), self.raw_dir, path, {"ann_date": "20200101"}, "ann_date", True,
            revision_ledger=self.root / "revision_events.jsonl",
        )
        self.assertTrue(result["guard_retained"])
        self.assertTrue(result["skipped"])
        self.assertEqual(common.parquet_rows(path), 100)
        self.assertEqual(result["rows"], 100)  # reports the RETAINED partition

    def test_existing_partition_resume_reports_every_key(self):
        """A plain resume (partition present, no --force) returns early; the
        caller reads guard_retained on every day, so omitting it there turned
        the resume path into a KeyError."""
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        fields = common.SHARE_FLOAT_FIELDS.split(",")
        frame = pd.DataFrame([["000001.SZ", "20200101", "20200102", "h", "type", 1.0, 0.1]], columns=fields)
        frame = common.augment_event_frame(frame, common.EVENT_FLOW_SPECS["share_float"])
        common.write_parquet(path, frame, api_name="share_float", params={},
                             fields=list(frame.columns))

        class Unused:
            def query(self, *a, **k):
                raise AssertionError("resume must not call the API")

        result = download.query_share_float_to_path(
            Unused(), self.raw_dir, path, {"ann_date": "20200101"}, "ann_date", False,
        )
        self.assertEqual(
            sorted(result), ["guard_retained", "path", "rows", "skipped", "source_cap_risk"])
        self.assertFalse(result["guard_retained"])
        self.assertTrue(result["skipped"])

    def test_capped_new_response_on_small_partition_routes_to_rescue(self):
        """The 2026-08-13 incident class: a small early-evening partition, then
        the vendor's late fill pushes the day past the per-call cap. The
        retained day's cap risk must come from the FETCHED response too --
        classifying by the retained old rows escalated the exact case the
        ts_code rescue exists for into an 'unrescuable retraction' that failed
        the run and left the lake dirty."""
        day = "20200101"
        fields = common.SHARE_FLOAT_FIELDS.split(",")
        path = self.raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        existing = pd.DataFrame([
            [f"{index:06d}.SZ", day, "20200102", f"h{index}", "type", 1.0, 0.1]
            for index in range(100)
        ], columns=fields)
        existing = common.augment_event_frame(existing, common.EVENT_FLOW_SPECS["share_float"])
        common.write_parquet(path, existing, api_name="share_float", params={},
                             fields=list(existing.columns))

        class CappedLateFillClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return common.ApiResult(
                    common.SHARE_FLOAT_FIELDS.split(","),
                    [[f"{900000 + index:06d}.SZ", day, "20200103", f"n{index}", "type", 1.0, 0.1]
                     for index in range(common.SHARE_FLOAT_ROW_LIMIT)])

        args = argparse.Namespace(
            ann_start_date=day, ann_end_date=day, force=True,
            revision_ledger=self.root / "revision_events.jsonl",
            allow_empty_revision_overwrite=False,
        )
        report: dict = {}
        limit_hits = download.download_share_float_ann_dates(
            CappedLateFillClient(), self.raw_dir, args, report)
        self.assertEqual(limit_hits, [day])
        self.assertEqual(report["ann_date"]["guard_retained_days"], [])
        # The guard still retained the old partition; the rescue owns the repair.
        self.assertEqual(common.parquet_rows(path), 100)

    def test_capped_partition_retention_reaches_rescue_while_a_real_retraction_fails(self):
        """A retained partition means one of two very different things.

        Capped (the source truncated its answer and rotates which keys it
        returns): recoverable by the per-ts_code rescue, so the run must carry
        on and hand the day to it. Not capped: the source genuinely retracted
        the keys, which no rescue can undo -- that needs an operator, so the
        run fails, but only AFTER every recoverable day has been processed.
        """
        fields = common.SHARE_FLOAT_FIELDS.split(",")

        def seed(day, rows):
            path = self.raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
            frame = pd.DataFrame([
                [f"{index:06d}.SZ", day, "20200102", f"h{index}", "type", 1.0, 0.1]
                for index in range(rows)
            ], columns=fields)
            frame = common.augment_event_frame(frame, common.EVENT_FLOW_SPECS["share_float"])
            common.write_parquet(path, frame, api_name="share_float", params={},
                                 fields=list(frame.columns))

        capped_day, retracted_day = "20200101", "20200102"
        seed(capped_day, common.SHARE_FLOAT_ROW_LIMIT)  # at the API cap
        seed(retracted_day, 100)

        class ShrunkClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return common.ApiResult(fields.split(",") if isinstance(fields, str) else list(fields),
                                        [["999999.SZ", params["ann_date"], "20200102", "hZ", "type", 1.0, 0.1]])

        args = argparse.Namespace(
            ann_start_date=capped_day, ann_end_date=retracted_day, force=True,
            revision_ledger=self.root / "revision_events.jsonl",
            allow_empty_revision_overwrite=False,
        )
        report: dict = {}
        with self.assertRaisesRegex(RuntimeError, "retained by the revision guard on non-capped"):
            download.download_share_float_ann_dates(ShrunkClient(), self.raw_dir, args, report)

        # The capped day still reached the rescue list, and BOTH partitions were
        # preserved -- the failure never cost data.
        self.assertEqual(report["ann_date"]["limit_hit_days"], [capped_day])
        self.assertEqual(report["ann_date"]["guard_retained_days"], [retracted_day])
        self.assertEqual(
            common.parquet_rows(self.raw_dir / "share_float_ann_date" / f"ann_date={capped_day}.parquet"),
            common.SHARE_FLOAT_ROW_LIMIT,
        )

    def test_share_float_candidate_read_failure_is_not_silent(self):
        path = self.raw_dir / "anns_d" / "month=202001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ann_date": "20200101", "title": "解禁"}]).to_parquet(path, index=False)

        with self.assertRaises(ArrowInvalid):
            download.anns_unlock_candidate_codes(self.raw_dir, ["20200101"])

    def test_generic_event_flow_download_excludes_dedicated_share_float_path(self):
        selected = download.selected_event_flow_download_datasets(argparse.Namespace(datasets=None))
        self.assertNotIn("share_float", selected)
        self.assertIn("margin_secs", selected)
        with self.assertRaisesRegex(RuntimeError, "download-share-float-complete"):
            download.selected_event_flow_download_datasets(argparse.Namespace(datasets=["share_float"]))

    def test_range_partition_skip_requires_sidecar_coverage(self):
        path = self.raw_dir / "anns_d" / "month=202605.parquet"
        existing = pd.DataFrame([{"ann_date": "20260528", "title": "old"}])
        common.write_parquet(
            path,
            existing,
            api_name="anns_d",
            params={"start_date": "20260501", "end_date": "20260528"},
            fields=list(existing.columns),
        )

        self.assertTrue(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260501", "end_date": "20260528"},
        ))
        self.assertFalse(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260501", "end_date": "20260529"},
        ))
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        sidecar.unlink()
        self.assertFalse(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260501", "end_date": "20260528"},
        ))

    def test_sidecar_coverage_normalizes_date_and_datetime_bounds(self):
        path = self.raw_dir / "major_news" / "src=all" / "month=202605.parquet"
        existing = pd.DataFrame([{"pub_time": "2026-05-29 12:00:00", "title": "old"}])
        common.write_parquet(
            path,
            existing,
            api_name="major_news",
            params={"start_date": "2026-05-29 00:00:00", "end_date": "2026-05-29 23:59:59"},
            fields=list(existing.columns),
        )
        self.assertTrue(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260529000000", "end_date": "20260529235959"},
        ))

        common.write_parquet(
            path,
            existing,
            api_name="major_news",
            params={"start_date": "20260529000000", "end_date": "20260529000000"},
            fields=list(existing.columns),
        )
        self.assertFalse(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "20260529", "end_date": "20260529"},
        ))

        common.write_parquet(
            path,
            existing,
            api_name="major_news",
            params={"start_date": "20260529", "end_date": "20260529"},
            fields=list(existing.columns),
        )
        self.assertTrue(download.should_skip_existing_partition(
            path,
            force=False,
            requested_params={"start_date": "2026-05-29 00:00:00", "end_date": "2026-05-29 23:59:59"},
        ))

    def test_macro_month_loop_backfills_months_without_recorded_coverage_once(self):
        # A closed month whose sidecar records no coverage window must be
        # re-requested (its recorded pull may have stopped mid-month), then
        # skip forever once coverage is stamped. Skipping on bare file
        # existence would freeze such a partition permanently.
        existing = pd.DataFrame([{
            "month": "202604",
            "publish_date": "20260401",
            "title": "old",
            "issuing_org": "old",
            "data_api": "cn_schedule",
        }])
        for month in ("202604", "202605"):
            path = self.raw_dir / "cn_schedule" / f"month={month}.parquet"
            frame = existing.assign(month=month)
            common.write_parquet(
                path,
                frame,
                api_name="cn_schedule",
                params={"m": month},
                fields=list(frame.columns),
            )

        client = CountingMacroClient()
        download.download_macro_month_loop(
            client,
            self.raw_dir,
            common.MACRO_SPECS["cn_schedule"],
            "20260401",
            "20260529",
            False,
        )

        self.assertEqual([params["m"] for _, params in client.calls], ["202604", "202605"])
        for month, start, end in (("202604", "20260401", "20260430"), ("202605", "20260501", "20260529")):
            meta = json.loads((self.raw_dir / "cn_schedule" / f"month={month}.parquet.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["params"]["start_date"], start)
            self.assertEqual(meta["params"]["end_date"], end)

        # Second run: recorded coverage now satisfies both windows.
        client.calls.clear()
        download.download_macro_month_loop(
            client, self.raw_dir, common.MACRO_SPECS["cn_schedule"], "20260401", "20260529", False,
        )
        self.assertEqual(client.calls, [])

    def test_window_merged_partition_preserves_rows_outside_refresh_window(self):
        path = self.raw_dir / "repurchase" / "month=202605.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20260501", "amount": 1},
            {"ts_code": "000002.SZ", "ann_date": "20260515", "amount": 2},
        ])
        common.write_parquet(
            path,
            existing,
            api_name="repurchase",
            params={"start_date": "20260501", "end_date": "20260531"},
            fields=list(existing.columns),
        )
        refreshed = pd.DataFrame([
            {"ts_code": "000002.SZ", "ann_date": "20260515", "amount": 20},
            {"ts_code": "000003.SZ", "ann_date": "20260516", "amount": 30},
        ])

        rows = download.write_window_merged_partition(
            path,
            refreshed,
            api_name="repurchase",
            params={"start_date": "20260510", "end_date": "20260520"},
            fields=list(refreshed.columns),
            key_columns=["ts_code", "ann_date"],
            date_columns=["ann_date"],
            start_date="20260510",
            end_date="20260520",
            revision_ledger=str(self.root / "revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
        )

        merged = pd.read_parquet(path).sort_values("ts_code").reset_index(drop=True)
        self.assertEqual(rows, 3)
        self.assertEqual(merged["ts_code"].tolist(), ["000001.SZ", "000002.SZ", "000003.SZ"])
        self.assertEqual(merged["amount"].tolist(), [1, 20, 30])

    def test_macro_range_once_uses_retained_start_during_rolling_update(self):
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            tier="macro",
            start_date="20260504",
            macro_start_date="20200101",
            end_date="20260603",
            datasets=["cn_cpi"],
            force=True,
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = CountingMacroClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_macro(args), 0)

        cpi_calls = [params for api_name, params in client.calls if api_name == "cn_cpi"]
        self.assertEqual(cpi_calls[0]["start_m"], "202001")
        self.assertTrue((self.raw_dir / "cn_cpi" / "range=202001_latest.parquet").exists())

    def test_macro_range_once_prunes_stale_end_suffixed_files(self):
        stale = self.raw_dir / "cn_cpi" / "range=202001_202605.parquet"
        pd.DataFrame([{"month": "202001"}]).pipe(
            lambda df: common.write_parquet(stale, df, api_name="cn_cpi", params={}, fields=list(df.columns))
        )
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            tier="macro",
            start_date="20260504",
            macro_start_date="20200101",
            end_date="20260603",
            datasets=["cn_cpi"],
            force=True,
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = CountingMacroClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_macro(args), 0)

        remaining = sorted(path.name for path in (self.raw_dir / "cn_cpi").glob("range=*.parquet"))
        self.assertEqual(remaining, ["range=202001_latest.parquet"])
        self.assertFalse(stale.with_suffix(stale.suffix + ".meta.json").exists())

    def test_fundamental_ann_month_windows_pull_full_natural_months(self):
        windows = download.fundamental_ann_month_windows("20200615", "20200705", {"202004"})
        self.assertEqual(windows, [
            ("20200401", "20200430", "202004"),
            ("20200601", "20200630", "202006"),
            ("20200701", "20200705", "202007"),
        ])

    def test_concat_rows_preserves_schema_when_all_inputs_are_empty(self):
        from autotrade.environment.data.pit import concat_rows

        left = pd.DataFrame({"ts_code": pd.Series(dtype=object), "close": pd.Series(dtype="float64")})
        right = pd.DataFrame({"ts_code": pd.Series(dtype=object), "volume": pd.Series(dtype="int64")})
        merged = concat_rows([left, right])
        self.assertEqual(list(merged.columns), ["ts_code", "close", "volume"])
        self.assertEqual(str(merged["close"].dtype), "float64")
        self.assertEqual(len(merged), 0)

        # Mixed inputs: an empty frame's extra columns still enter the schema.
        rows = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})
        widened = concat_rows([rows, right])
        self.assertIn("volume", widened.columns)
        self.assertEqual(len(widened), 1)
        self.assertTrue(widened["volume"].isna().all())

    def test_dup_key_identical_content_records_no_revision_event(self):
        path = self.raw_dir / "dividend" / "ann_month=202001.parquet"
        rows = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200105", "cash_div": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20200105", "cash_div": 1.0},  # duplicate business key
        ])
        common.write_parquet(path, rows, api_name="dividend", params={}, fields=list(rows.columns))
        ledger = self.root / "dupkey_revision_events.jsonl"

        output = io.StringIO()
        with redirect_stdout(output):
            did_write = common.write_parquet_revision_aware(
                path, rows.copy(), api_name="dividend", params={}, fields=list(rows.columns), key_columns=["ts_code", "ann_date"], revision_ledger=ledger,
            )
        # Identical content with duplicate keys is not a revision: no event, no alert.
        self.assertTrue(did_write)
        self.assertFalse(ledger.exists())
        self.assertNotIn("REVISION_ALERT", output.getvalue())

        changed = rows.copy()
        changed.loc[1, "cash_div"] = 2.0
        with redirect_stdout(io.StringIO()):
            common.write_parquet_revision_aware(
                path, changed, api_name="dividend", params={}, fields=list(changed.columns), key_columns=["ts_code", "ann_date"], revision_ledger=ledger,
            )
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["comparison_issue"], "duplicate_key_rows")

        # Re-detecting the very same revision appends nothing: the ledger keys
        # on the event's stable content, so an unchanged diff never duplicates.
        with redirect_stdout(io.StringIO()):
            common.write_parquet_revision_aware(
                path, changed.copy(), api_name="dividend", params={}, fields=list(changed.columns), key_columns=["ts_code", "ann_date"], revision_ledger=ledger,
            )
        events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 1)

        # A schema change (added column, even with empty values) is a revision
        # despite identical row content on the shared columns.
        widened = changed.copy()
        widened["note"] = ""
        with redirect_stdout(io.StringIO()):
            common.write_parquet_revision_aware(
                path, widened, api_name="dividend", params={}, fields=list(widened.columns), key_columns=["ts_code", "ann_date"], revision_ledger=ledger,
            )
        events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 2)

    def test_revision_aware_writer_blocks_key_removal_overwrite(self):
        path = self.raw_dir / "forecast_vip" / "ann_month=202001.parquet"
        original = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200105", "type": "预增"},
            {"ts_code": "000002.SZ", "ann_date": "20200110", "type": "预减"},
        ])
        common.write_parquet(path, original, api_name="forecast_vip", params={}, fields=list(original.columns))
        truncated = pd.DataFrame([
            {"ts_code": "000002.SZ", "ann_date": "20200110", "type": "预减"},
            {"ts_code": "000003.SZ", "ann_date": "20200120", "type": "预增"},
        ])
        ledger = self.root / "removal_revision_events.jsonl"

        output = io.StringIO()
        with redirect_stdout(output):
            did_write = common.write_parquet_revision_aware(
                path,
                truncated,
                api_name="forecast_vip",
                params={"start_date": "20200110", "end_date": "20200131"},
                fields=list(truncated.columns),
                key_columns=["ts_code", "ann_date", "type"],
                revision_ledger=ledger,
            )

        self.assertFalse(did_write)
        self.assertIn("skipped_key_removal_overwrite", output.getvalue())
        self.assertTrue(pd.read_parquet(path).equals(original))
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["write_action"], "skipped_key_removal_overwrite")
        self.assertEqual(event["removed_keys"], 1)

        with redirect_stdout(io.StringIO()):
            self.assertFalse(common.write_parquet_revision_aware(
                path,
                truncated,
                api_name="forecast_vip",
                params={"start_date": "20200110", "end_date": "20200131"},
                fields=list(truncated.columns),
                key_columns=["ts_code", "ann_date", "type"],
                revision_ledger=ledger,
            ))
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)

        with redirect_stdout(io.StringIO()):
            did_write = common.write_parquet_revision_aware(
                path,
                truncated,
                api_name="forecast_vip",
                params={"start_date": "20200110", "end_date": "20200131"},
                fields=list(truncated.columns),
                key_columns=["ts_code", "ann_date", "type"],
                revision_ledger=ledger,
                allow_key_removal_overwrite=True,
            )
        self.assertTrue(did_write)
        self.assertEqual(set(pd.read_parquet(path)["ts_code"]), {"000002.SZ", "000003.SZ"})

    def test_revision_aware_writer_blocks_disproportionate_shrink(self):
        path = self.raw_dir / "repurchase" / "month=202001.parquet"
        original = pd.DataFrame([
            {"ts_code": f"{index:06d}.SZ", "ann_date": "20200105", "amount": 1.0}
            for index in range(120)
        ])
        common.write_parquet(path, original, api_name="repurchase", params={}, fields=list(original.columns))
        truncated = original.head(10)  # 110 keys removed: >20 keys and >20%
        ledger = self.root / "shrink_revision_events.jsonl"

        output = io.StringIO()
        with redirect_stdout(output):
            did_write = common.write_parquet_revision_aware(
                path,
                truncated,
                api_name="repurchase",
                params={},
                fields=list(truncated.columns),
                key_columns=["ts_code", "ann_date"],
                revision_ledger=ledger,
                allow_key_removal_overwrite=True,
            )
        self.assertFalse(did_write)
        self.assertIn("blocked_shrink_overwrite", output.getvalue())
        self.assertTrue(pd.read_parquet(path).equals(original))
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["write_action"], "blocked_shrink_overwrite")

        # Paths that opt into source retractions still accept a proportionate
        # correction; the generic daily path does not use this allowance.
        small = original.head(115)
        with redirect_stdout(io.StringIO()):
            did_write = common.write_parquet_revision_aware(
                path,
                small,
                api_name="repurchase",
                params={},
                fields=list(small.columns),
                key_columns=["ts_code", "ann_date"],
                revision_ledger=ledger,
                allow_key_removal_overwrite=True,
            )
        self.assertTrue(did_write)
        self.assertEqual(len(pd.read_parquet(path)), 115)


    def test_daily_audit_warns_on_exact_limit_without_pagination_probe(self):
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "trade_date": ["20200102"] * 5000,
            "ts_code": [f"{index:06d}.SZ" for index in range(5000)],
        }).to_parquet(path, index=False)
        findings = []
        audit.audit_trade_date_dataset(self.raw_dir, common.DAILY_SPECS["daily"], {"20200102"}, lambda *item: findings.append(item))
        self.assertEqual(findings[0][0], "warning")
        self.assertEqual(findings[0][3]["exact_common_limit_row_count_dates"], ["20200102"])

    def test_trailing_zero_run_warns_for_zero_tolerant_dataset(self):
        # zero_rows_ok suppresses per-partition zero warnings, so a feed that
        # stops publishing (slb_len_mm: one year of empty partitions) never
        # surfaced. A long trailing run of zero-row partitions must warn.
        spec = common.EVENT_FLOW_SPECS["block_trade"]
        columns = spec.fields.split(",") + ["available_at", "available_at_rule"]
        dates = ["20260701", "20260702", "20260703", "20260706", "20260707"]
        for index, trade_date in enumerate(dates):
            if index < 2:
                rows = pd.DataFrame([{
                    **{column: "1.0" for column in spec.fields.split(",")},
                    "trade_date": trade_date,
                    "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T21:00:00+08:00",
                    "available_at_rule": "official_21_from:trade_date",
                }])
            else:
                rows = pd.DataFrame(columns=columns)
            common.write_parquet(
                self.raw_dir / "block_trade" / f"trade_date={trade_date}.parquet",
                rows, api_name="block_trade", params={"trade_date": trade_date},
                fields=columns,
            )
        expected = {self.raw_dir / "block_trade" / f"trade_date={d}.parquet" for d in dates}

        def run_audit():
            findings = []
            audit.audit_event_dataset(self.raw_dir, spec, expected, lambda *item: findings.append(item))
            return next(item for item in findings if item[1] == "block_trade_event_partitions")

        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 3):
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "warning")
            self.assertEqual(details["trailing_zero_row_partitions"], 3)
        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 4):
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "info")
            self.assertEqual(details["trailing_zero_row_partitions"], 3)

    def test_trailing_zero_run_warns_on_the_generic_board_path(self):
        # limit_list_d moved to the board tier; the stale-feed detector must
        # survive the move (the dedicated board audit used to enable it).
        spec = common.BOARD_TRADING_SPECS["limit_list_d"]
        columns = spec.fields.split(",") + ["available_at", "available_at_rule"]
        dates = ["20260701", "20260702", "20260703", "20260706", "20260707"]
        for index, trade_date in enumerate(dates):
            if index < 2:
                rows = common.augment_board_frame(
                    pd.DataFrame([{
                        **{column: "1.0" for column in spec.fields.split(",")},
                        "trade_date": trade_date,
                    }]),
                    spec,
                    {"trade_date": trade_date},
                )
            else:
                rows = pd.DataFrame(columns=columns)
            common.write_parquet(
                self.raw_dir / "limit_list_d" / f"trade_date={trade_date}.parquet",
                rows, api_name="limit_list_d", params={"trade_date": trade_date},
                fields=columns,
            )
        expected = {self.raw_dir / "limit_list_d" / f"trade_date={d}.parquet" for d in dates}

        findings = []
        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 3):
            audit.audit_board_dataset(self.raw_dir, spec, expected, lambda *item: findings.append(item))
        severity, _, _, details = next(item for item in findings if item[1] == "limit_list_d_board_partitions")
        self.assertEqual(severity, "warning")
        self.assertEqual(details["trailing_zero_row_partitions"], 3)

    def test_text_trailing_zero_run_is_counted_per_source_subdirectory(self):
        # news partitions sort by source first, so a whole-listing trailing run
        # could only ever see the last-sorted source. A dead feed that sorts
        # earlier (src=deadfeed < src=livefeed) must still be detected, and the
        # live sibling must not be flagged.
        spec = common.TEXT_SPECS["news"]
        columns = spec.fields.split(",") + ["available_at", "available_at_rule"]
        days = ["20260701", "20260702", "20260703", "20260704"]
        files = set()
        for source, dead in (("deadfeed", True), ("livefeed", False)):
            for index, day in enumerate(days):
                if dead and index > 0:
                    rows = pd.DataFrame(columns=columns)
                else:
                    rows = common.augment_text_frame(
                        pd.DataFrame([{
                            "datetime": f"{day[:4]}-{day[4:6]}-{day[6:]} 09:00:00",
                            "content": "内容", "title": f"标题{day}", "channels": "",
                        }]),
                        spec,
                    )
                path = self.raw_dir / "news" / f"src={source}" / f"date={day}.parquet"
                common.write_parquet(
                    path, rows, api_name="news", params={"src": source, "date": day},
                    fields=columns,
                )
                files.add(path)

        findings = []
        with patch.object(audit, "TRAILING_ZERO_RUN_WARN_PARTITIONS", 3):
            audit.audit_text_dataset(self.raw_dir, spec, files, lambda *item: findings.append(item))
        severity, _, _, details = next(item for item in findings if item[1] == "news_text_partitions")
        self.assertEqual(severity, "warning")
        self.assertEqual(details["trailing_zero_row_partitions"], 3)
        self.assertEqual(details["stale_feed_sources"], ["src=deadfeed"])

    def test_stk_surv_calendar_and_new_share_trading_day_offsets(self):
        # stk_surv rows keep landing for days after surv_date (the source has
        # no announcement column) and new_share's ballot is announced 1-2
        # trading days after ipo_date: both stamps must clear a weekend and a
        # month boundary instead of using the source date's own EOD.
        surv = common.augment_event_frame(
            pd.DataFrame([{"ts_code": "000001.SZ", "surv_date": "20260828"}]),
            common.EVENT_FLOW_SPECS["stk_surv"],
        )
        self.assertEqual(surv.loc[0, "available_at"], "2026-09-02 23:59:59+08:00")
        self.assertEqual(surv.loc[0, "available_at_rule"], "conservative_plus_5d_eod_from:surv_date")

        ipo = common.augment_event_frame(
            pd.DataFrame([{"ts_code": "301000.SZ", "ipo_date": "20260731", "ballot": "0.02"}]),
            common.EVENT_FLOW_SPECS["new_share"],
            trading_dates=["20260803", "20260804"],
        )
        self.assertEqual(ipo.loc[0, "available_at"], "2026-08-04 23:59:59+08:00")
        self.assertEqual(ipo.loc[0, "available_at_rule"], "conservative_tplus_2_eod_from:ipo_date")

        holiday = common.augment_event_frame(
            pd.DataFrame([{"ts_code": "301380.SZ", "ipo_date": "20220930", "ballot": "0.03"}]),
            common.EVENT_FLOW_SPECS["new_share"],
            trading_dates=["20220930", "20221010", "20221011"],
        )
        self.assertEqual(holiday.loc[0, "available_at"], "2022-10-11 23:59:59+08:00")

        with self.assertRaisesRegex(RuntimeError, "requires the A-share trading calendar"):
            common.augment_event_frame(
                pd.DataFrame([{"ts_code": "301380.SZ", "ipo_date": "20220930"}]),
                common.EVENT_FLOW_SPECS["new_share"],
            )

    def test_new_share_download_uses_the_local_trading_calendar(self):
        calendar_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2022.parquet"
        calendar_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "20220901", "is_open": "0"},
                {"cal_date": "20220930", "is_open": "1"},
                {"cal_date": "20221010", "is_open": "1"},
                {"cal_date": "20221011", "is_open": "1"},
                {"cal_date": "20221031", "is_open": "0"},
            ]
        ).to_parquet(calendar_path, index=False)

        class NewShareClient:
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "new_share":
                    raise AssertionError(f"unexpected TuShare query: {api_name}")
                columns = fields.split(",")
                values = {
                    "ts_code": "301380.SZ",
                    "ipo_date": "20220930",
                    "issue_date": "20221019",
                    "ballot": "0.03",
                }
                return common.ApiResult(
                    columns,
                    [[values.get(column, "") for column in columns]])

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20220901",
            end_date="20220930",
            datasets=["new_share"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        with patch.object(download, "load_token", return_value="token"), patch.object(
            download, "TuShareClient", return_value=NewShareClient()
        ):
            self.assertEqual(download.download_event_flow(args), 0)

        frame = pd.read_parquet(self.raw_dir / "new_share" / "month=202209.parquet")
        self.assertEqual(frame.loc[0, "available_at"], "2022-10-11 23:59:59+08:00")

    def test_fundamental_cap_plateau_flags_periods_pinned_at_dataset_max(self):
        # A vendor per-call cap pins many periods at exactly the dataset
        # maximum (cashflow_vip held 23 periods at 6,400); organic counts vary,
        # so several periods sharing the exact maximum is the cap signature
        # regardless of the cap value.
        findings = []
        audit.audit_fundamental_cap_plateau({
            "income_vip": {"20231231": 11771, "20240331": 5300, "20240630": 7707},
            "balancesheet_vip": {"20231231": 11771, "20240331": 5310, "20240630": 6915},
            "cashflow_vip": {"20231231": 6400, "20240331": 6400, "20240630": 6400},
        }, lambda *item: findings.append(item))
        severity, check, message, details = findings[0]
        self.assertEqual(check, "fundamental_statement_cap_plateau")
        self.assertEqual(severity, "warning")
        self.assertEqual(message, "statement datasets have multiple periods pinned at their maximum row count")
        self.assertEqual(details["plateaus"], [
            {"dataset": "cashflow_vip", "max_rows": 6400, "periods_at_max": 3,
             "period_sample": ["20231231", "20240331", "20240630"]},
        ])

    def test_fundamental_cap_plateau_stays_silent_on_organic_counts(self):
        # The repaired lake diverges across statements by more than 2x per
        # period; only exact repeated maxima may warn, never ratios.
        findings = []
        audit.audit_fundamental_cap_plateau({
            "income_vip": {"20231231": 11771, "20240331": 5300},
            "balancesheet_vip": {"20231231": 11771, "20240331": 5310},
            "cashflow_vip": {"20231231": 6400, "20240331": 12047},
        }, lambda *item: findings.append(item))
        severity, _, message, details = findings[0]
        self.assertEqual(severity, "info")
        self.assertEqual(message, "no statement dataset has multiple periods pinned at exactly its maximum row count")
        self.assertEqual(details["plateaus"], [])

        # Tiny early-history partitions below the row floor never warn, and a
        # dataset subset is checked per dataset instead of being skipped.
        small = []
        audit.audit_fundamental_cap_plateau(
            {"cashflow_vip": {"20100331": 900, "20100630": 900, "20100930": 900}},
            lambda *item: small.append(item),
        )
        severity, _, _, details = small[0]
        self.assertEqual(severity, "info")
        self.assertEqual(details["datasets_checked"], ["cashflow_vip"])
        self.assertEqual(details["plateaus"], [])

    def test_fundamental_audit_wires_cap_plateau_check_over_period_partitions(self):
        stock_basic = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        stock_basic.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}]).to_parquet(stock_basic, index=False)
        periods = ("20200331", "20200630", "20200930")
        rows_by_dataset = {
            "income_vip": (1200, 1100, 1300),
            "balancesheet_vip": (1250, 1150, 1350),
            "cashflow_vip": (1000, 1000, 1000),  # pinned at its maximum: cap signature
        }
        for dataset, row_counts in rows_by_dataset.items():
            for period, rows in zip(periods, row_counts):
                frame = pd.DataFrame([
                    {"ts_code": f"{index:06d}.SZ", "ann_date": f"{period[:6]}30", "f_ann_date": f"{period[:6]}30",
                     "end_date": period, "report_type": "1", "comp_type": "1", "end_type": "1"}
                    for index in range(rows)
                ])
                common.write_parquet(
                    self.raw_dir / dataset / f"period={period}.parquet",
                    frame, api_name=dataset, params={"period": period},
                    fields=list(frame.columns),
                )
        args = argparse.Namespace(
            fundamental_start_date="20200101",
            fundamental_end_date=None,
            fundamental_datasets=["income_vip", "balancesheet_vip", "cashflow_vip"],
        )
        findings = []
        audit.audit_fundamental_completeness(self.raw_dir, args, lambda *item: findings.append(item))
        severity, _, _, details = next(item for item in findings if item[1] == "fundamental_statement_cap_plateau")
        self.assertEqual(severity, "warning")
        self.assertEqual(details["plateaus"], [
            {"dataset": "cashflow_vip", "max_rows": 1000, "periods_at_max": 3, "period_sample": list(periods)},
        ])

    def test_index_weight_audit_checks_year_partitions_and_monthly_coverage(self):
        def write_year(code, year, months, legacy=False):
            if legacy:
                path = self.raw_dir / "index_weight" / f"index_code={code}.parquet"
            else:
                path = self.raw_dir / "index_weight" / f"index_code={code}" / f"year={year}.parquet"
            rows = pd.DataFrame([
                {"index_code": code, "con_code": f"{m:06d}.SZ", "trade_date": f"{year}{m:02d}28", "weight": 1.0}
                for m in months
            ])
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(path, index=False)

        def run_audit():
            findings = []
            audit.audit_index_weight(self.raw_dir, "20221231", lambda *item: findings.append(item))
            return findings[0]

        with patch.object(audit, "DEFAULT_CN_INDEX_CODES", ["000300.SH"]):
            write_year("000300.SH", 2020, range(1, 13))
            write_year("000300.SH", 2021, range(1, 12))  # closed year, 11 months
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "error")  # year=2022 missing
            self.assertEqual(details["missing_year_partitions"], 1)

            write_year("000300.SH", 2022, range(1, 8))  # open year: no month check
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "warning")
            self.assertEqual(details["closed_years_with_missing_months"], {"000300.SH/2021": 11})

            write_year("000300.SH", 2021, range(1, 13))
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "info")

            # Unmigrated flat legacy partitions (the pre-pagination layout) warn.
            write_year("000300.SH", 2020, range(1, 13), legacy=True)
            severity, _, _, details = run_audit()
            self.assertEqual(severity, "warning")
            self.assertEqual(details["legacy_flat_partitions"], 1)

    def test_index_weight_download_pages_per_year_and_skips_covered_years(self):
        source_rows = {
            2020: 7005,  # forces two pages at the 7,000-row source clamp
            2021: 10,
        }

        class WeightClient:
            def __init__(self):
                self.calls = []

            def query(self, api_name, params=None, fields="", retries=5):
                assert api_name == "index_weight"
                self.calls.append(dict(params))
                year = int(params["start_date"][:4])
                rows = [
                    [params["index_code"], f"{index:06d}.SZ", f"{year}0131", 1.0]
                    for index in range(source_rows.get(year, 0))
                ]
                limit = min(int(params["limit"]), 7000)  # source-side clamp
                offset = int(params["offset"])
                page = rows[offset:offset + limit]
                return common.ApiResult(
                    fields=["index_code", "con_code", "trade_date", "weight"],
                    items=page)

        client = WeightClient()
        with patch.object(download, "DEFAULT_CN_INDEX_CODES", ["000300.SH"]):
            download.download_index_weight(
                client, self.raw_dir, "20190101", "20211231", False, None, False
            )
            base = self.raw_dir / "index_weight" / "index_code=000300.SH"
            # The 2019 window is floored away; both years land as partitions.
            self.assertFalse((base / "year=2019.parquet").exists())
            self.assertEqual(len(pd.read_parquet(base / "year=2020.parquet")), 7005)
            self.assertEqual(len(pd.read_parquet(base / "year=2021.parquet")), 10)
            offsets_2020 = [c["offset"] for c in client.calls if c["start_date"] == "20200101"]
            self.assertEqual(offsets_2020, [0, 7000])

            # Covered years skip without force; a later end date re-pulls only
            # the open year.
            calls_before = len(client.calls)
            download.download_index_weight(
                client, self.raw_dir, "20190101", "20211231", False, None, False
            )
            self.assertEqual(len(client.calls), calls_before)
            source_rows[2022] = 3
            download.download_index_weight(
                client, self.raw_dir, "20190101", "20221231", False, None, False
            )
            new_calls = client.calls[calls_before:]
            self.assertEqual({c["start_date"] for c in new_calls}, {"20220101"})
            self.assertEqual(len(pd.read_parquet(base / "year=2022.parquet")), 3)

    def test_write_raw_generation_publishes_atomic_stamp(self):
        raw = self.root / "genraw"
        cron_update.write_raw_generation(raw)
        first = json.loads((raw / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(first["state"], "committed")
        self.assertTrue(first["generation_id"])
        self.assertTrue(first["completed_at"])
        cron_update.write_raw_generation(raw)
        second = json.loads((raw / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertNotEqual(first["generation_id"], second["generation_id"])
        self.assertEqual(list((raw).glob(".raw_generation.json.tmp*")), [])

    def test_raw_generation_failed_mutation_is_dirty_and_only_same_job_can_recover(self):
        raw = self.root / "genraw"
        cron_update.write_raw_generation(raw)
        transaction = {
            "job": "cn_evening_full",
            "start_date": "20260601",
            "end_date": "20260630",
            "commands": [["python", "download.py", "cn_evening_full"]],
            "config_identity": {"operation": "update"},
        }
        active = cron_update.begin_raw_generation_update(raw, transaction)
        cron_update.mark_raw_generation_dirty(raw, active, error="step 2 failed")
        dirty = json.loads((raw / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual(dirty["state"], "dirty")
        self.assertEqual(dirty["transaction"]["job"], "cn_evening_full")

        # A different job must never bless a partially-updated lake.
        with self.assertRaisesRegex(RuntimeError, "rerun the original job"):
            cron_update.begin_raw_generation_update(raw, {**transaction, "job": "another_job"})

        # An exact identity match resumes the original transaction.
        resumed = cron_update.begin_raw_generation_update(raw, transaction)
        self.assertEqual(resumed["transaction_id"], active["transaction_id"])
        cron_update.mark_raw_generation_dirty(raw, resumed, error="failed again")

        # A same-job run with a newer window/command supersedes as a fresh
        # transaction (daily-recomputed windows can never replay exactly), and
        # its success commits and clears the fence.
        superseding = {
            **transaction,
            "start_date": "20260602",
            "end_date": "20260701",
            "commands": [["python", "download.py", "cn_evening_full", "--force"]],
        }
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            adopted = cron_update.begin_raw_generation_update(raw, superseding)
        self.assertNotEqual(adopted["transaction_id"], active["transaction_id"])
        note = json.loads(stdout.getvalue().splitlines()[0])
        self.assertEqual(note["note"], "raw_generation_dirty_superseded")
        self.assertEqual(note["previous_window"], "20260601..20260630")
        self.assertEqual(note["previous_error"], "failed again")
        with redirect_stdout(io.StringIO()):
            committed = cron_update.write_raw_generation(raw, transaction=adopted)
        self.assertEqual(committed["state"], "committed")
        self.assertNotEqual(committed["generation_id"], dirty["generation_id"])

    def test_revision_sentinel_is_not_a_mutating_operation(self):
        self.assertNotIn("revision_sentinel", cron_update.MUTATING_OPERATIONS)

    def test_parquet_availability_survives_identical_refresh_but_moves_on_revision(self):
        path = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"
        first = pd.DataFrame(
            [{"trade_date": "20260713", "ts_code": "000001.SZ", "price": 10.0}]
        )
        availability = {
            "available_at": "2026-07-13T09:28:36+08:00",
            "rule": "observed:cn_open_auction_capture",
        }
        common.write_parquet(
            path,
            first,
            api_name="stk_auction",
            params={},
            fields=list(first.columns),
            extra_metadata={"availability": availability},
        )
        common.write_parquet(
            path,
            first.copy(),
            api_name="stk_auction",
            params={},
            fields=list(first.columns),
        )
        self.assertEqual(common.parquet_meta(path)["availability"], availability)

        revised = first.assign(price=10.1)
        common.write_parquet(
            path,
            revised,
            api_name="stk_auction",
            params={},
            fields=list(revised.columns),
        )
        revised_availability = common.parquet_meta(path)["availability"]
        self.assertEqual(revised_availability["rule"], "observed:content_revision_fetch")
        self.assertNotEqual(revised_availability["available_at"], availability["available_at"])

    def test_capture_open_auction_waits_for_stable_complete_frame(self):
        self._write_trade_cal("20260713")
        previous = self.raw_dir / "stk_auction" / "trade_date=20260710.parquet"
        previous.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"x": 1}, {"x": 2}]).to_parquet(previous, index=False)
        fields = common.DAILY_SPECS["stk_auction"].fields.split(",")

        def result(items):
            return common.ApiResult(fields, items)

        full = [
            ["000001.SZ", "20260713", 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
            ["600000.SH", "20260713", 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
        ]
        responses = [
            (result([]), 1),
            (result(full), 1),
            RuntimeError("transient source error"),
            (result(full), 1),
            (result(full), 1),
        ]
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            trade_date="20260713",
            page_limit=10000,
            max_wait_seconds=10.0,
            retry_delay_seconds=0.0,
            stable_reads=2,
            min_rows=2,
            min_previous_day_ratio=0.98,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient"),
            patch.object(download, "query_paged", side_effect=responses) as query,
            patch.object(download.time, "sleep", return_value=None),
        ):
            self.assertEqual(download.capture_open_auction(args), 0)

        self.assertEqual(query.call_count, 5)
        target = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"
        self.assertEqual(len(pd.read_parquet(target)), 2)
        availability = common.parquet_meta(target)["availability"]
        self.assertEqual(availability["rule"], "observed:cn_open_auction_capture")
        self.assertEqual(availability["row_count"], 2)

        # A later strict reconciliation may return the same keyed rows in a
        # different API order. Canonical persistence keeps the first landing.
        reversed_result = result(list(reversed(full)))
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient"),
            patch.object(download, "query_paged", side_effect=[(reversed_result, 1), (reversed_result, 1)]),
            patch.object(download.time, "sleep", return_value=None),
        ):
            self.assertEqual(download.capture_open_auction(args), 0)
        self.assertEqual(common.parquet_meta(target)["availability"], availability)
        self.assertEqual(pd.read_parquet(target)["ts_code"].tolist(), ["000001.SZ", "600000.SH"])

    def test_capture_open_auction_timeout_does_not_replace_partition(self):
        self._write_trade_cal("20260713")
        target = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"
        original = pd.DataFrame(
            [{"trade_date": "20260713", "ts_code": "000001.SZ", "price": 9.9}]
        )
        common.write_parquet(
            target,
            original,
            api_name="stk_auction",
            params={},
            fields=list(original.columns),
        )
        original_bytes = target.read_bytes()
        fields = common.DAILY_SPECS["stk_auction"].fields.split(",")
        partial = common.ApiResult(
            fields,
            [["000001.SZ", "20260713", 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0]])
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            trade_date="20260713",
            page_limit=10000,
            max_wait_seconds=0.0,
            retry_delay_seconds=0.0,
            stable_reads=1,
            min_rows=2,
            min_previous_day_ratio=0.98,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient"),
            patch.object(download, "query_paged", return_value=(partial, 1)),
        ):
            self.assertEqual(
                download.capture_open_auction(args),
                common.NO_MUTATION_RETRY_EXIT_CODE,
            )

        self.assertEqual(target.read_bytes(), original_bytes)

    def test_capture_open_auction_polls_on_fixed_start_times(self):
        self._write_trade_cal("20260713")
        fields = common.DAILY_SPECS["stk_auction"].fields.split(",")
        items = [
            ["000001.SZ", "20260713", 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
        ]
        result = common.ApiResult(fields, items)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            trade_date="20260713",
            page_limit=10000,
            max_wait_seconds=30.0,
            retry_delay_seconds=10.0,
            stable_reads=3,
            min_rows=1,
            min_previous_day_ratio=0.98,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
        )

        class Clock:
            def __init__(self):
                self.now = 0.0
                self.query_starts = []
                self.sleeps = []

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.sleeps.append(seconds)
                self.now += seconds

            def query(self, *_args, **_kwargs):
                self.query_starts.append(self.now)
                self.now += 3.0
                return result, 1

        clock = Clock()
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient"),
            patch.object(download, "query_paged", side_effect=clock.query),
            patch.object(download.time, "monotonic", side_effect=clock.monotonic),
            patch.object(download.time, "sleep", side_effect=clock.sleep),
        ):
            self.assertEqual(download.capture_open_auction(args), 0)

        self.assertEqual(clock.query_starts, [0.0, 10.0, 20.0])
        self.assertEqual(clock.sleeps, [7.0, 7.0])

    def test_auction_capture_rejects_duplicate_business_keys(self):
        row = {
            "ts_code": "000001.SZ",
            "trade_date": "20260713",
            "vol": 1000.0,
            "price": 10.0,
            "amount": 10000.0,
            "pre_close": 9.9,
            "turnover_rate": 0.1,
            "volume_ratio": 1.0,
            "float_share": 100000.0,
        }
        errors = download._validate_auction_capture(
            pd.DataFrame([row, row]), "20260713", min_rows=1
        )
        self.assertIn("duplicate_keys=1", errors)

    def test_auction_capture_validates_trade_and_no_trade_quantities(self):
        base = {
            "trade_date": "20260713",
            "pre_close": 9.9,
            "turnover_rate": 0.1,
            "volume_ratio": 1.0,
            "float_share": 100000.0,
        }
        valid = pd.DataFrame([
            {**base, "ts_code": "000001.SZ", "price": 10.0, "vol": 1000.0, "amount": 10000.0},
            # A missing source price is safe when the clearing price can be
            # reconstructed exactly from two positive finite quantities.
            {**base, "ts_code": "000002.SZ", "price": None, "vol": 2000.0, "amount": 16000.0},
            {**base, "ts_code": "000003.SZ", "price": None, "vol": 0.0, "amount": 0.0},
        ])
        self.assertEqual(download._validate_auction_capture(valid, "20260713", min_rows=3), [])

        invalid = pd.DataFrame([
            {**base, "ts_code": "000004.SZ", "price": 10.0, "vol": float("nan"), "amount": 10.0},
            {**base, "ts_code": "000005.SZ", "price": 10.0, "vol": -1.0, "amount": 0.0},
            {**base, "ts_code": "000006.SZ", "price": 10.0, "vol": 1.0, "amount": 0.0},
            {**base, "ts_code": "000007.SZ", "price": 10.0, "vol": 1.0, "amount": float("nan")},
            {**base, "ts_code": "000008.SZ", "price": 10.0, "vol": 1.0, "amount": -1.0},
            {**base, "ts_code": "000009.SZ", "price": None, "vol": 5e-324, "amount": 1e308},
            {**base, "ts_code": "000010.SZ", "price": 10.0, "vol": 0.0, "amount": 0.0},
            {**base, "ts_code": "000011.SZ", "price": 100.0, "vol": 1000.0, "amount": 10000.0},
        ])
        errors = download._validate_auction_capture(invalid, "20260713", min_rows=8)
        self.assertIn("invalid_vol_rows=2", errors)
        self.assertIn("invalid_amount_rows=2", errors)
        self.assertIn("inconsistent_trade_rows=1", errors)
        self.assertIn("unrecoverable_trade_price_rows=1", errors)
        self.assertIn("hidden_no_trade_price_rows=1", errors)
        self.assertIn("inconsistent_trade_price_rows=1", errors)

    def test_auction_capture_row_floor_is_percentage_only(self):
        # Percentage-only: an absolute prev-minus-N bound dominated the ratio
        # for full-market partitions and rejected legitimate provider-side
        # coverage narrowing (2026-07: ~1,100 fund/ETF rows removed
        # retroactively). 85% still blocks truncated captures.
        previous = self.raw_dir / "stk_auction" / "trade_date=20260710.parquet"
        previous.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row": range(5519)}).to_parquet(previous, index=False)

        minimum = download._auction_capture_min_rows(
            self.raw_dir, "20260713", floor=1000, ratio=0.85,
        )
        self.assertEqual(minimum, 4691)  # floor(5519 * 0.85)
        # No previous partition: the absolute floor holds.
        self.assertEqual(
            download._auction_capture_min_rows(self.raw_dir, "20260701", floor=1000, ratio=0.85),
            1000,
        )

    def test_stk_auction_recheck_single_shot_paths(self):
        self._write_trade_cal("20260713")
        fields = common.DAILY_SPECS["stk_auction"].fields.split(",")
        full = [
            ["000001.SZ", "20260713", 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
            ["600000.SH", "20260713", 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
        ]

        def result(items, tag):
            return common.ApiResult(fields, items)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            end_date="20260713",
            landing_job="cn_evening_auction_backfill",
            page_limit=10000,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
        target = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"

        def run(**response):
            output = io.StringIO()
            with (
                patch.object(download, "load_token", return_value="token"),
                patch.object(download, "TuShareClient"),
                patch.object(download, "AUCTION_CAPTURE_MIN_ROWS_FLOOR", 2),
                patch.object(download, "query_paged", **response),
                redirect_stdout(output),
            ):
                code = download.recheck_stk_auction(args)
            return code, output.getvalue()

        # Catch-up: no partition yet (the whole pre-open window failed); one
        # valid read publishes with observed availability.
        code, out = run(return_value=(result(full, "a"), 1))
        self.assertEqual(code, 0)
        self.assertIn("recheck_published", out)
        availability = common.parquet_meta(target)["availability"]
        self.assertEqual(availability["landing_job"], "cn_evening_auction_backfill")
        first_bytes = target.read_bytes()

        # Unchanged content (even reordered): no rewrite; the partition bytes
        # and the first observed availability both survive.
        code, out = run(return_value=(result(list(reversed(full)), "b"), 1))
        self.assertEqual(code, 0)
        self.assertIn("recheck_unchanged", out)
        self.assertEqual(target.read_bytes(), first_bytes)
        self.assertEqual(common.parquet_meta(target)["availability"], availability)

        # Incomplete fetch (below the completeness floor): existing file kept,
        # the failure is surfaced, the batch does not fail.
        code, out = run(return_value=(result(full[:1], "c"), 1))
        self.assertEqual(code, 0)
        self.assertIn("recheck_invalid", out)
        self.assertIn("below_min_rows", out)
        self.assertEqual(target.read_bytes(), first_bytes)

        # Query failure: same containment.
        code, out = run(side_effect=RuntimeError("source down"))
        self.assertEqual(code, 0)
        self.assertIn("recheck_invalid", out)
        self.assertIn("query_error", out)
        self.assertEqual(target.read_bytes(), first_bytes)

        # A genuine late correction (same keys, consistent new values)
        # republishes with fresh observed availability.
        revised = [row[:] for row in full]
        revised[0][2] = 2000.0
        revised[0][4] = 20000.0
        code, out = run(return_value=(result(revised, "d"), 1))
        self.assertEqual(code, 0)
        self.assertIn("recheck_published", out)
        self.assertIn('"revised_existing": true', out)
        self.assertNotEqual(target.read_bytes(), first_bytes)
        self.assertNotEqual(
            common.parquet_meta(target)["availability"]["available_at"],
            availability["available_at"],
        )

    def test_stk_auction_recheck_heals_missing_earlier_days_in_window(self):
        # Standardized with the margin evening self-heal: earlier open days
        # with a partition are skipped, a MISSING earlier day gets a validated
        # capture attempt, and the latest day is always re-read.
        cal_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cal_date": ["20260709", "20260710", "20260713"],
            "is_open": ["1", "1", "1"],
        }).to_parquet(cal_path, index=False)
        fields = common.DAILY_SPECS["stk_auction"].fields.split(",")

        def day_rows(day):
            return [
                ["000001.SZ", day, 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
                ["600000.SH", day, 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
            ]

        # 20260709 already captured as an intact commit; 20260710 missing.
        existing = pd.DataFrame(day_rows("20260709"), columns=fields)
        target_dir = self.raw_dir / "stk_auction"
        common.write_parquet(
            target_dir / "trade_date=20260709.parquet",
            existing,
            api_name="stk_auction",
            params={"trade_date": "20260709"},
            fields=fields,
        )

        queried: list[str] = []

        def fake_query(client, api_name, params, fields_, page_limit):
            day = params["trade_date"]
            queried.append(day)
            return common.ApiResult(fields, day_rows(day)), 1

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260709",
            end_date="20260713",
            page_limit=10000,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient"),
            patch.object(download, "AUCTION_CAPTURE_MIN_ROWS_FLOOR", 2),
            patch.object(download, "query_paged", side_effect=fake_query),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(download.recheck_stk_auction(args), 0)
        self.assertEqual(queried, ["20260710", "20260713"])
        self.assertTrue((target_dir / "trade_date=20260710.parquet").exists())
        self.assertTrue((target_dir / "trade_date=20260713.parquet").exists())

    def test_stk_auction_recheck_heals_incomplete_earlier_days_in_window(self):
        cal_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cal_date": ["20260709", "20260710", "20260713"],
            "is_open": ["1", "1", "1"],
        }).to_parquet(cal_path, index=False)
        fields = common.DAILY_SPECS["stk_auction"].fields.split(",")

        def day_rows(day):
            return [
                ["000001.SZ", day, 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
                ["600000.SH", day, 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
            ]

        target_dir = self.raw_dir / "stk_auction"
        target_dir.mkdir(parents=True, exist_ok=True)
        torn = target_dir / "trade_date=20260709.parquet"
        pd.DataFrame(day_rows("20260709"), columns=fields).to_parquet(torn, index=False)
        self.assertFalse(common.committed_partition_intact(torn))

        queried: list[str] = []

        def fake_query(client, api_name, params, fields_, page_limit):
            day = params["trade_date"]
            queried.append(day)
            return common.ApiResult(fields, day_rows(day)), 1

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260709",
            end_date="20260713",
            page_limit=10000,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient"),
            patch.object(download, "AUCTION_CAPTURE_MIN_ROWS_FLOOR", 2),
            patch.object(download, "query_paged", side_effect=fake_query),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(download.recheck_stk_auction(args), 0)
        self.assertEqual(queried, ["20260709", "20260710", "20260713"])
        self.assertTrue(common.committed_partition_intact(torn))

    def test_default_daily_download_excludes_stk_auction(self):
        selected = common.selected_daily_datasets(argparse.Namespace(datasets=None))
        self.assertEqual(selected, common.DAILY_DOWNLOAD_DATASETS)
        self.assertNotIn("stk_auction", selected)
        required = common.selected_daily_datasets(
            argparse.Namespace(datasets=None),
            default=common.DAILY_REQUIRED_DATASETS,
        )
        self.assertIn("stk_auction", required)
        self.assertIn("stk_auction", common.DAILY_REQUIRED_DATASETS)

    def test_generic_daily_cli_choices_exclude_stk_auction(self):
        parser = download.build_parser()
        rejected = (
            ["update", "--start-date", "20200102", "--daily-datasets", "stk_auction"],
            ["update", "--start-date", "20200102", "--refresh-daily-datasets", "stk_auction"],
            ["download", "--tier", "daily", "--refresh-daily-datasets", "stk_auction"],
        )
        for argv in rejected:
            with self.subTest(argv=argv), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parser.parse_args(argv)
        parsed = parser.parse_args([
            "update",
            "--start-date",
            "20200102",
            "--daily-datasets",
            "daily",
            "--refresh-daily-datasets",
            "adj_factor",
        ])
        self.assertEqual(parsed.daily_datasets, ["daily"])
        self.assertEqual(parsed.refresh_daily_datasets, ["adj_factor"])

    def test_generic_daily_path_refuses_stk_auction(self):
        self._write_trade_cal("20200102", is_open="1")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["stk_auction"],
            refresh_daily_datasets=[],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        with (
            patch.object(download, "load_token", return_value="token"),
            patch.object(download, "TuShareClient", return_value=NoQueryClient()),
            self.assertRaisesRegex(RuntimeError, "capture-open-auction"),
        ):
            download.download_daily(args)

    def test_non_trading_auction_job_skips_before_generation_fence(self):
        self._write_trade_cal("20260712", is_open="0")
        config_path = self.root / "auction_schedule.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "timezone": "Asia/Shanghai",
                    "repo_root": str(self.root),
                    "python": "/env/python",
                    "default_raw_dir": "raw",
                    "default_start_date": "20200101",
                    "jobs": {
                        "auction": {
                            "operation": "auction_capture",
                            "only_if_sse_open_date": True,
                            "start_date_lookback_days": 0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        generation = self.raw_dir / ".raw_generation.json"
        cron_update.write_raw_generation(self.raw_dir)
        before = generation.read_bytes()
        args = argparse.Namespace(
            config=str(config_path),
            job="auction",
            start_date=None,
            end_date="20260712",
            dry_run=False,
            force_run=False,
        )
        jobs_root = self.root / "runtime" / "jobs"
        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", side_effect=AssertionError("must not lock")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), 0)

        self.assertEqual(generation.read_bytes(), before)
        record = json.loads((jobs_root / "auction.json").read_text(encoding="utf-8"))
        self.assertTrue(record["skipped_non_trading_day"])
        # Every run outcome carries its own real log file.
        self.assertTrue(Path(record["log_path"]).is_file())

    def test_same_day_open_check_fails_when_calendar_does_not_cover_target(self):
        self._write_trade_cal("20260712", is_open="0")

        with self.assertRaisesRegex(RuntimeError, "does not cover target date 20260713"):
            cron_update.is_sse_open_date(self.root, "raw", "20260713")

    def test_not_ready_auction_job_restores_committed_generation(self):
        self._write_trade_cal("20260713", is_open="1")
        config_path = self.root / "auction_schedule.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "timezone": "Asia/Shanghai",
                    "repo_root": str(self.root),
                    "python": "/env/python",
                    "default_raw_dir": "raw",
                    "default_start_date": "20200101",
                    "jobs": {
                        "auction": {
                            "operation": "auction_capture",
                            "only_if_sse_open_date": True,
                            "start_date_lookback_days": 0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        generation = self.raw_dir / ".raw_generation.json"
        cron_update.write_raw_generation(self.raw_dir)
        before = json.loads(generation.read_text(encoding="utf-8"))
        args = argparse.Namespace(
            config=str(config_path),
            job="auction",
            start_date=None,
            end_date="20260713",
            dry_run=False,
            force_run=False,
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        jobs_root = self.root / "runtime" / "jobs"
        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(
                cron_update,
                "run_update",
                return_value=common.NO_MUTATION_RETRY_EXIT_CODE,
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), common.NO_MUTATION_RETRY_EXIT_CODE)

        self.assertEqual(json.loads(generation.read_text(encoding="utf-8")), before)
        record = json.loads((jobs_root / "auction.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "not_ready")
        self.assertTrue(Path(record["log_path"]).is_file())

    def _write_event_flow_schedule(self, name: str, operation: str = "download_event_flow") -> Path:
        job = {
            "operation": operation,
            "end_date_offset_days": 0,
            "skip_if_already_ok": True,
        }
        if operation == "download_tier":
            job["tier"] = "event_flow"
        config_path = self.root / f"{name}_schedule.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "timezone": "Asia/Shanghai",
                    "repo_root": str(self.root),
                    "python": "/env/python",
                    "default_raw_dir": "raw",
                    "default_start_date": "20200101",
                    "jobs": {name: job},
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def _run_job_once(self, config_path: Path, job: str, end_date: str, returncode: int, jobs_root: Path):
        args = argparse.Namespace(
            config=str(config_path),
            job=job,
            start_date=None,
            end_date=end_date,
            dry_run=False,
            force_run=False,
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(cron_update, "run_update", return_value=returncode) as runner,
            redirect_stdout(io.StringIO()) as out,
        ):
            result = cron_update.main()
        return result, runner, out.getvalue()

    def test_mutated_not_ready_commits_generation_and_is_retried_next_run(self):
        # Regression: a run that wrote real partitions but left a required one
        # unpublished must publish the generation (the writes are real) AND
        # record a non-ok status. Recording "ok" is what let skip_if_already_ok
        # abandon margin trade_date=20260807 once the resolved window moved on.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("margin_retry")
        generation = self.raw_dir / ".raw_generation.json"
        cron_update.write_raw_generation(self.raw_dir)
        before = json.loads(generation.read_text(encoding="utf-8"))
        jobs_root = self.root / "runtime" / "jobs"

        result, _, _ = self._run_job_once(
            config_path,
            "margin_retry",
            "20260807",
            common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
            jobs_root,
        )
        self.assertEqual(result, common.MUTATED_NOT_READY_RETRY_EXIT_CODE)

        after = json.loads(generation.read_text(encoding="utf-8"))
        self.assertEqual(after["state"], "committed")
        self.assertNotEqual(after["generation_id"], before["generation_id"])

        record = json.loads((jobs_root / "margin_retry.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "not_ready")

        # The next run at the SAME resolved end date must actually re-attempt.
        _, runner, output = self._run_job_once(
            config_path, "margin_retry", "20260807", 0, jobs_root
        )
        self.assertTrue(runner.called)
        self.assertNotIn("skipped_already_ok", output)
        self.assertEqual(
            json.loads((jobs_root / "margin_retry.json").read_text(encoding="utf-8"))["status"],
            "ok",
        )

    def test_ok_event_flow_run_is_still_skipped_on_repeat(self):
        # The mirror of the regression: a genuinely complete run must keep
        # suppressing duplicate work at the same resolved end date.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("margin_ok")
        cron_update.write_raw_generation(self.raw_dir)
        jobs_root = self.root / "runtime" / "jobs"

        self._run_job_once(config_path, "margin_ok", "20260807", 0, jobs_root)
        _, runner, output = self._run_job_once(
            config_path, "margin_ok", "20260807", 0, jobs_root
        )
        self.assertFalse(runner.called)
        self.assertIn("skipped_already_ok", output)

    def test_mutated_not_ready_from_unexpected_operation_is_an_error(self):
        # Exit 76 only carries the commit-and-retry contract for the download
        # path that enforces it; anywhere else it is an ordinary failure and
        # must fence the lake rather than quietly commit.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("tier_job", operation="download_tier")
        generation = self.raw_dir / ".raw_generation.json"
        cron_update.write_raw_generation(self.raw_dir)
        jobs_root = self.root / "runtime" / "jobs"

        result, _, _ = self._run_job_once(
            config_path,
            "tier_job",
            "20260807",
            common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
            jobs_root,
        )
        self.assertEqual(result, common.MUTATED_NOT_READY_RETRY_EXIT_CODE)
        self.assertEqual(json.loads(generation.read_text(encoding="utf-8"))["state"], "dirty")
        record = json.loads((jobs_root / "tier_job.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "error")

    def test_finishing_run_preserves_concurrently_recorded_failure(self):
        # Invariant: an outcome another process persisted while this run was
        # active (here: a job that timed out waiting on the update lock and
        # recorded its failure) must survive the finish. Per-job state files
        # make cross-job interference structurally impossible; this pins it.
        config_path = self.root / "cron_schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20260101",
            "jobs": {"job_a": {"operation": "audit_event_flow"}},
        }), encoding="utf-8")
        jobs_root = self.root / "runtime" / "jobs"
        args = argparse.Namespace(
            config=str(config_path),
            job="job_a",
            start_date=None,
            end_date="20260713",
            dry_run=False,
            force_run=False,
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        def concurrent_failure_writer(*_args, **_kwargs):
            # While job_a's (long) run is active, another cron process times
            # out on the update lock and durably records its failure.
            cron_update.record_job_state("job_b", {
                "status": "error",
                "returncode": 1,
                "error": "lock is held after waiting 900s",
                "updated_at": "2026-07-13T01:00:00+00:00",
            })
            return 0

        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(cron_update, "run_update", side_effect=concurrent_failure_writer),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), 0)

        job_a = json.loads((jobs_root / "job_a.json").read_text(encoding="utf-8"))
        self.assertEqual(job_a["status"], "ok")
        # The concurrently recorded failure stays visible.
        job_b = json.loads((jobs_root / "job_b.json").read_text(encoding="utf-8"))
        self.assertEqual(job_b["status"], "error")
        self.assertIn("lock is held", job_b["error"])

    def test_record_job_state_concurrent_writers_stay_isolated_and_atomic(self):
        jobs_root = self.root / "runtime" / "jobs"

        def write_entries(name: str) -> None:
            for seq in range(10):
                cron_update.record_job_state(name, {"status": "ok", "seq": seq})

        with patch.object(cron_update, "JOB_STATE_ROOT", jobs_root):
            workers = [
                threading.Thread(target=write_entries, args=(f"job_{index}",))
                for index in range(4)
            ]
            # Same-job concurrent writers: the file must never tear and the
            # last completed write must win.
            workers += [
                threading.Thread(target=write_entries, args=("shared",))
                for _ in range(3)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        for index in range(4):
            record = json.loads((jobs_root / f"job_{index}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["seq"], 9)
        shared = json.loads((jobs_root / "shared.json").read_text(encoding="utf-8"))
        self.assertEqual(shared["seq"], 9)

    def test_auction_cron_command_overrides_global_request_timeout(self):
        ctx = cron_update.RunContext(
            config={
                "default_raw_dir": "raw",
                "default_update_args": ["--timeout-seconds", "120"],
            },
            repo_root=self.root,
            python="/env/python",
            job_name="auction",
            job={
                "operation": "auction_capture",
                "extra_args": ["--timeout-seconds", "15"],
            },
            start_date="20260713",
            end_date="20260713",
            timezone_name="Asia/Shanghai",
        )

        command = cron_update.build_job_commands(ctx)[0]
        timeout_positions = [i for i, value in enumerate(command) if value == "--timeout-seconds"]
        self.assertEqual(command[timeout_positions[-1] + 1], "15")

    def test_cron_job_identity_ignores_unrelated_job_edits(self):
        selected_job = {"operation": "auction_capture", "skip_if_already_ok": True}
        base_config = {
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_start_date": "20200101",
            "default_raw_dir": "raw",
            "default_update_args": ["--timeout-seconds", "15"],
            "jobs": {"auction": selected_job, "unrelated": {"extra_args": ["old"]}},
        }
        edited_config = {
            **base_config,
            "jobs": {"auction": selected_job, "unrelated": {"extra_args": ["new"]}},
        }
        contexts = [
            cron_update.RunContext(
                config=config,
                repo_root=self.root,
                python="/env/python",
                job_name="auction",
                job=selected_job,
                start_date="20260713",
                end_date="20260713",
                timezone_name="Asia/Shanghai",
            )
            for config in (base_config, edited_config)
        ]
        first_identity, edited_identity = map(cron_update.job_config_identity, contexts)
        self.assertEqual(first_identity, edited_identity)
        commands = [["python", "capture.py"]]
        payload = {"commands": commands, "config_identity": edited_identity}
        state = {
            "start_date": "20260713",
            "end_date": "20260713",
            "status": "ok",
            "commands": commands,
            "config_identity": first_identity,
        }
        args = argparse.Namespace(force_run=False)
        self.assertTrue(cron_update.should_skip_completed(contexts[1], args, state, payload))

    def test_cron_full_audit_builds_all_formal_status_commands(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_full_audit",
            job={"operation": "audit_full", "event_flow_end_extra_offset_days": 1},
            start_date="20200101",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 6)
        command_text = [" ".join(command) for command in commands]
        self.assertIn("scripts/data/tushare_audit.py core-market", command_text[0])
        self.assertIn("scripts/data/tushare_audit.py fundamental-raw", command_text[1])
        self.assertIn("--fundamental-end-date 20260601", command_text[1])
        self.assertIn("scripts/data/tushare_audit.py macro", command_text[2])
        self.assertIn("scripts/data/tushare_audit.py intraday-by-date", command_text[3])
        # "daily" is an independent universe for the newest-day deep check;
        # "minute" read the audited file's own codes back and could never
        # report a dropped stock.
        self.assertIn("--expected-codes-source daily", command_text[3])
        self.assertIn("scripts/data/tushare_audit.py event-flow", command_text[4])
        self.assertIn("--end-date 20260531", command_text[4])
        self.assertIn("scripts/data/tushare_audit.py board-trading", command_text[5])
        # Text is natural-day data with its own daily audit job: this
        # trading-day job must not claim to refresh its status file.
        self.assertTrue(all(" text " not in f"{text} " for text in command_text))
        self.assertTrue(all("--start-date 20200101" in text for text in command_text))
        self.assertTrue(all("--raw-dir raw" in text for text in command_text))

    def test_cron_text_audit_uses_the_natural_day_window(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_text_audit",
            job={"operation": "audit_text"},
            start_date="20200101",
            end_date="20260621",  # a Sunday: audited as itself, never rolled back
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        command_text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_audit.py text", command_text)
        self.assertIn("--start-date 20200101", command_text)
        self.assertIn("--end-date 20260621", command_text)
        self.assertIn("--raw-dir raw", command_text)

    def test_cron_run_log_retention_preserves_state_link_and_unrelated_files(self):
        log_root = self.root / "logs" / "tushare" / "cron"
        log_root.mkdir(parents=True)
        stale = log_root / "tushare_cron_stale.log"
        referenced = log_root / "tushare_cron_referenced.log"
        fresh = log_root / "tushare_cron_fresh.log"
        unrelated = log_root / "notes.txt"
        for path in (stale, referenced, fresh, unrelated):
            path.write_text("record\n", encoding="utf-8")
        baseline = stale.stat().st_mtime
        os.utime(fresh, (baseline + 14 * 86400, baseline + 14 * 86400))
        jobs_root = self.root / "runtime" / "jobs"
        jobs_root.mkdir(parents=True)
        (jobs_root / "job.json").write_text(
            json.dumps({"status": "ok", "log_path": str(referenced)}), encoding="utf-8"
        )
        # A torn per-job file must not break retention for the others.
        (jobs_root / "corrupt.json").write_text("{not json", encoding="utf-8")

        with (
            patch.object(cron_update, "RUN_LOG_ROOT", log_root),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
        ):
            cron_update.prune_run_logs(
                now=baseline + (cron_update.RUN_LOG_RETENTION_DAYS + 1) * 86400,
            )

        self.assertFalse(stale.exists())
        self.assertTrue(referenced.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(unrelated.exists())

    def test_skip_and_lock_failure_write_their_own_run_logs(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="audit",
            job={"operation": "audit_full"},
            start_date="20200101",
            end_date="20260720",
            timezone_name="Asia/Shanghai",
        )
        args = argparse.Namespace(dry_run=False, force_run=False)
        log_root = self.root / "logs" / "tushare" / "cron"
        jobs_root = self.root / "runtime" / "jobs"

        # Already-ok skip: the invocation leaves its own log, but the state
        # file stays untouched so it keeps pointing at (and protecting) the
        # last real run.
        with (
            patch.object(cron_update, "build_context", return_value=ctx),
            patch.object(cron_update, "build_job_commands", return_value=[["audit"]]),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "RUN_LOG_ROOT", log_root),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "should_skip_completed", return_value=True),
            patch.object(cron_update, "acquire_lock", side_effect=AssertionError("must not lock")),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(cron_update._run(args), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "skipped_already_ok")
        skip_logs = list(log_root.glob("tushare_cron_audit_*.log"))
        self.assertEqual(len(skip_logs), 1)
        self.assertIn("skipped_already_ok", skip_logs[0].read_text(encoding="utf-8"))
        self.assertFalse((jobs_root / "audit.json").exists())

        # Lock failure: the error is durably recorded with its own log.
        with (
            patch.object(cron_update, "build_context", return_value=ctx),
            patch.object(cron_update, "build_job_commands", return_value=[["audit"]]),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "RUN_LOG_ROOT", log_root),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "should_skip_completed", return_value=False),
            patch.object(cron_update, "acquire_lock", side_effect=RuntimeError("lock is held")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update._run(args), 1)
        record = json.loads((jobs_root / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "error")
        self.assertIn("lock is held", record["error"])
        self.assertTrue(Path(record["log_path"]).is_file())
        self.assertIn("lock is held", Path(record["log_path"]).read_text(encoding="utf-8"))

    def test_cron_full_audit_can_use_open_date_for_event_flow(self):
        trade_cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "20260618", "is_open": "1"},
                {"cal_date": "20260619", "is_open": "0"},
                {"cal_date": "20260620", "is_open": "0"},
                {"cal_date": "20260621", "is_open": "0"},
            ]
        ).to_parquet(trade_cal, index=False)
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_full_audit",
            job={"operation": "audit_full", "event_flow_end_extra_offset_days": 3},
            start_date="20200101",
            end_date="20260621",
            timezone_name="Asia/Shanghai",
        )

        command_text = [" ".join(command) for command in cron_update.build_job_commands(ctx)]

        self.assertIn("scripts/data/tushare_audit.py event-flow", command_text[4])
        self.assertIn("--end-date 20260618", command_text[4])

    def test_cron_update_job_can_use_rolling_start_lookback(self):
        config_path = self.root / "schedule.json"
        trade_cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "20260618", "is_open": "1"},
                {"cal_date": "20260619", "is_open": "0"},
                {"cal_date": "20260620", "is_open": "0"},
                {"cal_date": "20260621", "is_open": "0"},
            ]
        ).to_parquet(trade_cal, index=False)
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {
                "cn_evening_full": {
                    "start_date_lookback_days": 30,
                    "extra_args": ["--refresh-daily-datasets", "daily", "adj_factor"],
                },
                "cn_preopen_margin_backfill_0905": {
                    "operation": "download_event_flow",
                    "end_date_offset_days": 1,
                    "end_date_mode": "sse_open_on_or_before",
                },
            },
        }), encoding="utf-8")

        args = argparse.Namespace(config=str(config_path), job="cn_evening_full", start_date=None, end_date="20260601", dry_run=False, force_run=False)
        ctx = cron_update.build_context(args)
        self.assertEqual(ctx.start_date, "20260502")
        self.assertEqual(ctx.end_date, "20260601")
        command = " ".join(cron_update.build_job_commands(ctx)[0])
        self.assertIn("--refresh-daily-datasets daily adj_factor", command)

        margin_args = argparse.Namespace(config=str(config_path), job="cn_preopen_margin_backfill_0905", start_date=None, end_date="20260621", dry_run=False, force_run=False)
        margin_ctx = cron_update.build_context(margin_args)
        self.assertEqual(margin_ctx.start_date, "20260618")
        self.assertEqual(margin_ctx.end_date, "20260618")

    def test_cron_nightly_audit_weekend_runs_resolve_to_friday_and_skip(self):
        # Saturday and Sunday invocations resolve the same trading end date as
        # Friday's completed run, so skip_if_already_ok suppresses the weekend
        # full re-scans (and no weekend calendar date leaks into the window).
        config_path = self.root / "schedule.json"
        trade_cal = self.root / "raw" / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260619", "is_open": "1"},
            {"cal_date": "20260620", "is_open": "0"},
            {"cal_date": "20260621", "is_open": "0"},
        ]).to_parquet(trade_cal, index=False)
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {
                "cn_nightly_full_audit": {
                    "operation": "audit_full",
                    "end_date_offset_days": 1,
                    "end_date_mode": "sse_open_on_or_before",
                    "event_flow_end_extra_offset_days": 1,
                    "text_end_extra_offset_days": 1,
                },
            },
        }), encoding="utf-8")

        contexts = {}
        for target in ("20260620", "20260621"):  # Sat and Sun morning targets
            args = argparse.Namespace(config=str(config_path), job="cn_nightly_full_audit", start_date=None, end_date=target, dry_run=False, force_run=False)
            contexts[target] = cron_update.build_context(args)
        self.assertEqual(contexts["20260620"].end_date, "20260619")
        self.assertEqual(contexts["20260621"].end_date, "20260619")

        ctx = contexts["20260621"]
        payload = {
            "commands": cron_update.build_job_commands(ctx),
            "config_identity": cron_update.job_config_identity(ctx),
        }
        saturday_state = {
            "start_date": ctx.start_date,
            "end_date": "20260619",
            "status": "ok",
            "commands": payload["commands"],
            "config_identity": payload["config_identity"],
        }
        skip_args = argparse.Namespace(force_run=False)
        self.assertTrue(cron_update.should_skip_completed(ctx, skip_args, saturday_state, payload))
        # --force-run (the Saturday post-sweep cron line) still runs.
        self.assertFalse(cron_update.should_skip_completed(ctx, argparse.Namespace(force_run=True), saturday_state, payload))

    def test_cron_pit_event_job_covers_full_window_even_with_existing_partitions(self):
        # The rolling 120-day window let months outside it drift against the
        # living raw lake unaudited; the pipeline now always rebuilds and
        # audits from default_start_date.
        config_path = self.root / "schedule.json"
        existing = self.root / "pit" / "fundamental_events" / "income_vip"
        existing.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"dataset": "income_vip", "available_at": "2026-01-01T00:00:00+08:00"}]).to_parquet(
            existing / "available_month=202601.parquet",
            index=False,
        )
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_start_date": "20200101",
            "default_raw_dir": "raw",
            "default_pit_root": "pit",
            "jobs": {
                "cn_nightly_pit_event_build": {
                    "operation": "pit_event_pipeline",
                    "fundamental_events_root": "pit/fundamental_events",
                },
            },
        }), encoding="utf-8")

        args = argparse.Namespace(
            config=str(config_path),
            job="cn_nightly_pit_event_build",
            start_date=None,
            end_date="20260621",
            dry_run=False,
            force_run=False,
        )
        ctx = cron_update.build_context(args)
        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(ctx.start_date, "20200101")
        self.assertIn("--start-date 20200101", " ".join(commands[0]))
        self.assertIn("--start-date 20200101", " ".join(commands[1]))

    def test_cron_download_tier_job_builds_targeted_command(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_update_args": ["--min-interval-seconds", "0.22"]},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_preopen_board_backfill_0850",
            job={"operation": "download_tier", "tier": "board_trading", "extra_args": ["--datasets", "kpl_list", "--force"]},
            start_date="20260601",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_download.py download --tier board_trading", text)
        self.assertIn("--start-date 20260601 --end-date 20260601", text)
        self.assertIn("--raw-dir raw", text)
        self.assertIn("--datasets kpl_list --force", text)

    def test_cron_revision_sentinel_job_builds_audit_command(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_update_args": ["--min-interval-seconds", "0.22"]},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_daily_revision_sentinel",
            job={"operation": "revision_sentinel", "extra_args": ["--sample-size", "12", "--datasets", "adj_factor"]},
            start_date="20200101",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_audit.py revision-sentinel", text)
        self.assertIn("--start-date 20200101 --end-date 20260601", text)
        self.assertIn("--raw-dir raw", text)
        self.assertIn("--sample-size 12 --datasets adj_factor", text)

    def test_cron_event_flow_audit_job_builds_targeted_status_refresh(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_preopen_event_flow_audit_0920",
            job={"operation": "audit_event_flow"},
            start_date="20200101",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_audit.py event-flow", text)
        self.assertIn("--start-date 20200101 --end-date 20260601", text)
        self.assertIn("--raw-dir raw", text)

    def test_cron_pit_event_pipeline_builds_and_audits_fundamental_events(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_pit_root": "pit"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_pit_event_build",
            job={
                "operation": "pit_event_pipeline",
                "fundamental_events_root": "pit/fundamental_events",
                "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
            },
            start_date="20260201",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 2)
        self.assertIn("scripts/data/build_pit_events.py build-fundamental-events", " ".join(commands[0]))
        self.assertIn("--raw-dir raw --output-root pit/fundamental_events", " ".join(commands[0]))
        self.assertIn("scripts/data/build_pit_events.py audit-fundamental-events", " ".join(commands[1]))
        self.assertIn("--events-root pit/fundamental_events", " ".join(commands[1]))
        self.assertIn("--require-partitions", " ".join(commands[1]))

    def test_cron_pit_event_pipeline_builds_and_audits_the_month_aligned_window(self):
        # A manual mid-month --start-date must month-align for BOTH commands:
        # the builder aligns its replace window internally, so an unaligned
        # audit start would flag the first half-month as outside_audit_window.
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_pit_root": "pit", "default_start_date": "20200101"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_pit_event_build",
            job={"operation": "pit_event_pipeline"},
            start_date="20260715",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertIn("--start-date 20260701 --end-date 20260601", " ".join(commands[0]))
        self.assertIn("--start-date 20260701 --end-date 20260601", " ".join(commands[1]))

    def test_cron_lock_blocks_while_held_and_releases_on_exit(self):
        # flock is per open-file-description: a second acquire in the same
        # process must block exactly like a second process would.
        runtime = self.root / ".runtime" / "tushare"
        with patch.object(cron_update, "RUNTIME_ROOT", runtime):
            held = cron_update.acquire_lock("tushare_update", wait_seconds=0)
            try:
                with self.assertRaisesRegex(RuntimeError, "lock is held"):
                    cron_update.acquire_lock("tushare_update", wait_seconds=0)
            finally:
                held.release()
            reacquired = cron_update.acquire_lock("tushare_update", wait_seconds=0)
            reacquired.release()

    def test_cron_lock_file_from_dead_process_never_blocks(self):
        # A leftover lock FILE (crash, kill -9, PID reuse) carries no kernel
        # flock, so the next run acquires immediately - no stale-lock heuristics.
        runtime = self.root / ".runtime" / "tushare"
        lock_file = runtime / "locks" / "tushare_update.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("pid=999999999\nstarted_at=2020-01-01T00:00:00+00:00\n", encoding="utf-8")

        with patch.object(cron_update, "RUNTIME_ROOT", runtime):
            acquired = cron_update.acquire_lock("tushare_update", wait_seconds=0)
            self.assertTrue(acquired.path.exists())
            self.assertIn(f"pid={cron_update.os.getpid()}", acquired.path.read_text(encoding="utf-8"))
            acquired.release()

    def test_cron_multi_command_jobs_fail_fast(self):
        ctx = cron_update.RunContext(
            config={},
            repo_root=self.root,
            python="/env/python",
            job_name="unit_fail_fast",
            job={"fail_fast": True},
            start_date="20200101",
            end_date="20200102",
            timezone_name="Asia/Shanghai",
        )
        commands = [["cmd1"], ["cmd2"], ["cmd3"]]
        calls = []

        class Result:
            def __init__(self, returncode):
                self.returncode = returncode

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result(1 if command == ["cmd2"] else 0)

        with patch.object(cron_update, "run_probe"), patch.object(cron_update.subprocess, "run", side_effect=fake_run):
            code = cron_update.run_update(ctx, commands, self.root / "cron.log")

        self.assertEqual(code, 1)
        self.assertEqual(calls, [["cmd1"], ["cmd2"]])

    def test_cron_multi_command_jobs_can_continue_after_error(self):
        ctx = cron_update.RunContext(
            config={},
            repo_root=self.root,
            python="/env/python",
            job_name="unit_continue",
            job={"fail_fast": False},
            start_date="20200101",
            end_date="20200102",
            timezone_name="Asia/Shanghai",
        )
        commands = [["cmd1"], ["cmd2"], ["cmd3"]]
        calls = []

        class Result:
            def __init__(self, returncode):
                self.returncode = returncode

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result(1 if command == ["cmd2"] else 0)

        with patch.object(cron_update, "run_probe"), patch.object(cron_update.subprocess, "run", side_effect=fake_run):
            code = cron_update.run_update(ctx, commands, self.root / "cron_continue.log")

        self.assertEqual(code, 1)
        self.assertEqual(calls, commands)

    def test_reference_refresh_datasets_force_selected_tables(self):
        self._write_trade_cal("20200102")
        for status in ("L", "D", "P"):
            path = self.raw_dir / "stock_basic" / f"list_status={status}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"ts_code": "999999.SZ", "list_status": status}]).to_parquet(path, index=False)
        for exchange in ("SSE", "SZSE", "BSE"):
            path = self.raw_dir / "stock_company" / f"exchange={exchange}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"ts_code": "999999.SZ", "exchange": exchange}]).to_parquet(path, index=False)
        namechange = self.raw_dir / "namechange" / "namechange.parquet"
        namechange.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "999999.SZ", "name": "old"}]).to_parquet(namechange, index=False)
        classify = self.raw_dir / "index_classify" / "src=SW2021.parquet"
        classify.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"index_code": "801999.SI", "level": "L1"}]).to_parquet(classify, index=False)
        member = self.raw_dir / "index_member_all" / "l1_code=801999.SI.parquet"
        member.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"l1_code": "801999.SI", "ts_code": "999999.SZ"}]).to_parquet(member, index=False)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            bak_start_date="20200102",
            skip_bak_basic=True,
            force=False,
            refresh_reference_datasets=["stock_basic", "stock_company", "namechange", "index_classify", "index_member_all"],
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = ReferenceClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_reference(args), 0)

        called_apis = [api_name for api_name, _ in client.calls]
        self.assertEqual(called_apis.count("stock_basic"), 3)
        self.assertEqual(called_apis.count("stock_company"), 3)
        self.assertIn("namechange", called_apis)
        self.assertIn("index_classify", called_apis)
        self.assertIn("index_member_all", called_apis)
        self.assertIn("index_member", called_apis)
        # Both classification vintages refresh together.
        classify_srcs = {params.get("src") for api, params in client.calls if api == "index_classify"}
        self.assertEqual(classify_srcs, {"SW2014", "SW2021"})
        self.assertTrue((self.raw_dir / "index_classify" / "src=SW2014.parquet").exists())
        # Member refresh pulls current and departed members.
        member_is_new = {params.get("is_new") for api, params in client.calls if api == "index_member_all"}
        self.assertEqual(member_is_new, {"Y", "N"})
        self.assertTrue((self.raw_dir / "index_member" / "l1_code=801010.SI.parquet").exists())
        # New reference statics download on first run (files absent).
        for api in ("ths_index", "ths_member", "index_basic", "index_weight"):
            self.assertIn(api, called_apis)

        # SW2014 membership is frozen history: a second run without force must
        # not re-request it while the classify refresh still runs.
        client.calls.clear()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_reference(args), 0)
        self.assertNotIn("index_member", [api for api, _ in client.calls])

    def test_reference_refresh_does_not_overwrite_existing_stock_company_on_empty_response(self):
        self._write_trade_cal("20200102")
        for status in ("L", "D", "P"):
            stock_path = self.raw_dir / "stock_basic" / f"list_status={status}.parquet"
            stock_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"ts_code": "000001.SZ", "list_status": status}]).to_parquet(stock_path, index=False)
        path = self.raw_dir / "stock_company" / "exchange=SSE.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"ts_code": "000001.SH", "exchange": "SSE", "com_name": "old"}])
        original.to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            bak_start_date="20200102",
            skip_bak_basic=True,
            force=False,
            refresh_reference_datasets=["stock_company"],
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyReferenceClient()):
            with self.assertRaisesRegex(RuntimeError, "required reference partition"):
                download.download_reference(args)
        self.assertTrue(pd.read_parquet(path).equals(original))

    def test_index_member_all_empty_refresh_keeps_existing_and_fails_on_missing(self):
        path = self.raw_dir / "index_member_all" / "l1_code=801010.SI.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"l1_code": "801010.SI", "ts_code": "000001.SZ", "in_date": "20200101", "out_date": ""}])
        original.to_parquet(path, index=False)
        classify = pd.DataFrame([{"index_code": "801010.SI", "level": "L1"}])
        download.download_index_member_all(EmptyReferenceClient(), self.raw_dir, classify, True)
        self.assertTrue(pd.read_parquet(path).equals(original))
        # An absent partition with an empty vendor response must fail fast.
        classify_missing = pd.DataFrame([{"index_code": "801999.SI", "level": "L1"}])
        with self.assertRaisesRegex(RuntimeError, "zero rows for required reference partition"):
            download.download_index_member_all(EmptyReferenceClient(), self.raw_dir, classify_missing, True)

    def test_trade_cal_force_refresh_merges_without_shrinking_year_partition(self):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"exchange": "SSE", "cal_date": "20260603", "is_open": "1", "pretrade_date": "20260602"},
        ]).to_parquet(path, index=False)
        client = TradeCalClient()

        open_dates = download.download_trade_cal(client, self.raw_dir, "20260604", "20260604", force=True)

        refreshed = pd.read_parquet(path)
        self.assertEqual(sorted(refreshed["cal_date"].astype(str).tolist()), ["20260603", "20260604"])
        self.assertIn("20260604", open_dates)

    def test_update_parser_force_refreshes_stock_company_by_default(self):
        argv = [
            "download.py",
            "update",
            "--start-date",
            "20260601",
            "--end-date",
            "20260601",
        ]
        with patch.object(sys, "argv", argv):
            args = download.parse_args()

        self.assertIn("stock_basic", args.refresh_reference_datasets)
        self.assertIn("stock_company", args.refresh_reference_datasets)
        self.assertIn("namechange", args.refresh_reference_datasets)
        self.assertIn("index_classify", args.refresh_reference_datasets)
        self.assertIn("index_member_all", args.refresh_reference_datasets)
        self.assertEqual(args.refresh_daily_datasets, [])
        self.assertEqual(args.macro_start_date, "20200101")

    def test_update_all_dimensions_uses_retained_start_for_macro_context(self):
        args = argparse.Namespace(
            start_date="20260504",
            end_date="20260603",
            macro_start_date="20200101",
            bak_start_date=None,
            force=False,
            refresh_open_window=True,
            trade_cal_lookahead_days=7,
            raw_dir=str(self.raw_dir),
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
            reference_min_interval_seconds=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            refresh_reference_datasets=[],
            daily_datasets=None,
            refresh_daily_datasets=[],
            macro_datasets=["cn_gdp"],
            global_datasets=["index_global"],
            event_datasets=[],
            include_board_trading=False,
            include_intraday=False,
            include_share_float_complete=False,
            include_text=True,
            include_global=True,
            text_datasets=[],
            fundamental_datasets=[],
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
        )
        seen = {}

        def capture(label, _fn, step_args, summary):
            seen[label] = step_args
            summary.append({"step": label, "exit_code": 0})

        with patch.object(download, "run_update_step", side_effect=capture):
            summary = []
            download.update_all_dimensions(args, summary)

        self.assertEqual(seen["daily"].start_date, "20260504")
        self.assertEqual(seen["event_flow"].start_date, "20260504")
        self.assertEqual(seen["macro"].start_date, "20260504")
        self.assertEqual(seen["macro"].macro_start_date, "20200101")
        self.assertEqual(seen["global"].start_date, "20260504")
        self.assertEqual(seen["global"].macro_start_date, "20200101")
        self.assertIn("text_evidence", seen)

    def test_evening_update_excludes_the_natural_day_text_tier(self):
        # Text is landed by its own calendar-day job; running it inside the
        # trading-day evening update would both duplicate the calls and tie
        # weekend text to a job that never launches on weekends.
        args = argparse.Namespace(
            start_date="20260504",
            end_date="20260603",
            macro_start_date="20200101",
            bak_start_date=None,
            force=False,
            refresh_open_window=True,
            trade_cal_lookahead_days=7,
            raw_dir=str(self.raw_dir),
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
            reference_min_interval_seconds=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            refresh_reference_datasets=[],
            daily_datasets=None,
            refresh_daily_datasets=[],
            macro_datasets=["cn_gdp"],
            global_datasets=["index_global"],
            event_datasets=[],
            include_board_trading=False,
            include_intraday=False,
            include_share_float_complete=False,
            include_text=False,
            include_global=False,
            text_datasets=[],
            fundamental_datasets=[],
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
        )
        seen = {}

        def capture(label, _fn, step_args, summary):
            seen[label] = step_args
            summary.append({"step": label, "exit_code": 0})

        with patch.object(download, "run_update_step", side_effect=capture):
            download.update_all_dimensions(args, [])

        self.assertNotIn("text_evidence", seen)
        self.assertNotIn("global", seen)
        self.assertIn("macro", seen)

    def test_production_evening_job_excludes_text_and_text_job_covers_the_tier(self):
        schedule = Path(__file__).resolve().parents[2] / "configs" / "tushare_update_schedule.json"
        config = json.loads(schedule.read_text(encoding="utf-8"))
        jobs = config["jobs"]
        self.assertIn("--no-include-text", jobs["cn_evening_full"]["extra_args"])
        text_job = jobs["cn_nightly_text_full"]
        self.assertEqual(text_job["operation"], "download_tier")
        self.assertEqual(text_job["tier"], "text_evidence")
        # Natural-day window: no trading-calendar mode, so a weekend launch
        # resolves a new end date instead of skipping as an unchanged range.
        self.assertNotIn("end_date_mode", text_job)
        self.assertNotIn("end_date_mode", jobs["cn_nightly_text_audit"])
        self.assertNotIn("--datasets", text_job["extra_args"])
        # The global tier and the ann-date disclosure tables are natural-day
        # domains too: overseas sessions and weekend announcements must not
        # wait for the next trading evening.
        self.assertIn("--no-include-global", jobs["cn_evening_full"]["extra_args"])
        global_job = jobs["cn_nightly_global_full"]
        self.assertEqual((global_job["operation"], global_job["tier"]), ("download_tier", "global"))
        self.assertNotIn("end_date_mode", global_job)
        disclosure_job = jobs["cn_nightly_disclosure_full"]
        self.assertEqual(disclosure_job["operation"], "download_event_flow")
        self.assertNotIn("end_date_mode", disclosure_job)
        disclosure_datasets = {"top10_holders", "top10_floatholders", "stk_holdernumber", "stk_holdertrade", "repurchase"}
        self.assertTrue(disclosure_datasets.issubset(set(disclosure_job["extra_args"])))
        # Single producer per dataset: the evening event list no longer
        # carries the disclosure tables.
        self.assertFalse(disclosure_datasets & set(jobs["cn_evening_full"]["extra_args"]))
        for name in ("cn_nightly_text_full", "cn_nightly_text_audit", "cn_nightly_global_full", "cn_nightly_disclosure_full"):
            # A Sunday resolves to itself: no trade-calendar rollback, so
            # skip_if_already_ok never sees an unchanged weekend range.
            self.assertEqual(
                cron_update.resolve_job_end_date(jobs[name], self.root, "raw", "20260621"),
                "20260621",
                name,
            )

    def test_daily_refresh_datasets_force_only_selected_trade_date_dataset(self):
        self._write_trade_cal("20200102")
        daily_rows = pd.DataFrame([{"trade_date": "20200102", "ts_code": "999999.SZ", "close": 1.0}])
        adj_rows = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "adj_factor": 2.0}])
        common.write_parquet(
            self.raw_dir / "daily" / "trade_date=20200102.parquet",
            daily_rows,
            api_name="daily",
            params={"trade_date": "20200102"},
            fields=list(daily_rows.columns),
        )
        common.write_parquet(
            self.raw_dir / "adj_factor" / "trade_date=20200102.parquet",
            adj_rows,
            api_name="adj_factor",
            params={"trade_date": "20200102"},
            fields=list(adj_rows.columns),
        )

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["daily", "adj_factor"],
            refresh_daily_datasets=["adj_factor"],
            revision_ledger=str(self.root / "revision_events.jsonl"),
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = DailyMarketClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_daily(args), 0)

        self.assertEqual([api_name for api_name, _ in client.calls], ["adj_factor"])
        self.assertIn("REVISION_ALERT", output.getvalue())
        # The alert is a compact one-liner; the full record lives in the ledger.
        self.assertIn('"dataset": "adj_factor"', output.getvalue())
        self.assertNotIn('"removed_keys_sample"', output.getvalue())
        ledger_lines = (self.root / "revision_events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(ledger_lines), 1)
        ledger_event = json.loads(ledger_lines[0])
        self.assertEqual(ledger_event["schema_version"], 2)
        self.assertEqual(ledger_event["write_action"], "overwrite")
        self.assertTrue(ledger_event["write_id"])
        self.assertEqual(set(pd.read_parquet(self.raw_dir / "daily" / "trade_date=20200102.parquet")["ts_code"]), {"999999.SZ"})

    def test_daily_force_refresh_does_not_drop_existing_keys(self):
        self._write_trade_cal("20200102")
        ledger = self.root / "daily_key_removal.jsonl"
        for dataset in ("daily", "adj_factor"):
            original = pd.DataFrame([
                {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.0, "adj_factor": 1.0},
                {"trade_date": "20200102", "ts_code": "000002.SZ", "close": 20.0, "adj_factor": 1.1},
            ])
            path = self.raw_dir / dataset / "trade_date=20200102.parquet"
            common.write_parquet(
                path,
                original,
                api_name=dataset,
                params={"trade_date": "20200102"},
                fields=list(original.columns),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                zero_skipped = download.download_trade_date_dataset(
                    DailyMarketClient(),
                    self.raw_dir,
                    common.DAILY_SPECS[dataset],
                    ["20200102"],
                    True,
                    5000,
                    ledger,
                    False,
                )
            self.assertEqual(zero_skipped, 1, dataset)
            self.assertIn("skipped_key_removal_overwrite", output.getvalue())
            kept = pd.read_parquet(path)
            self.assertEqual(set(kept["ts_code"]), {"000001.SZ", "000002.SZ"}, dataset)

    def test_fundamental_update_refreshes_recent_periods_and_affected_ts_code_snapshots(self):
        stock_basic = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        stock_basic.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}]).to_parquet(stock_basic, index=False)
        for dataset in ("dividend", "fina_audit", "fina_mainbz_vip"):
            path = self.raw_dir / dataset / "ts_code=000001.SZ.parquet"
            common.write_parquet(
                path,
                pd.DataFrame([{"ts_code": "000001.SZ", "ann_date": "20190101", "end_date": "20181231"}]),
                api_name=dataset,
                params={"ts_code": "000001.SZ"},
                fields=["ts_code", "ann_date", "end_date"],
            )
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200601",
            end_date="20200603",
            datasets=["dividend", "fina_audit", "fina_mainbz_vip", "income_vip"],
            force=False,
            page_limit=None,
            max_codes=None,
            fundamental_refresh_period_count=2,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=["dividend", "fina_audit", "fina_mainbz_vip"],
            fundamental_dividend_probe_days=0,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        client = FundamentalClient()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_fundamental(args), 0)

        calls = [(api, params) for api, params in client.calls if params.get("offset") == 0]
        income_periods = [params["period"] for api, params in calls if api == "income_vip"]
        self.assertEqual(income_periods, ["20191231", "20200331"])
        refreshed_ts_code_calls = [(api, params.get("ts_code")) for api, params in calls if api in {"dividend", "fina_audit", "fina_mainbz_vip"}]
        self.assertEqual(refreshed_ts_code_calls, [("dividend", "000001.SZ"), ("fina_audit", "000001.SZ"), ("fina_mainbz_vip", "000001.SZ")])

    def test_fundamental_download_clamps_cashflow_to_measured_page_limit(self):
        # cashflow_vip returns at most 6,400 rows per call (probed 2026-08-10),
        # so the tier-default 7,000 request must clamp to the spec's measured
        # 6,000 while income_vip keeps paging at 7,000.
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20191231",
            end_date="20191231",
            datasets=["income_vip", "cashflow_vip"],
            force=True,
            page_limit=7000,
            max_codes=None,
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=None,
            allow_empty_revision_overwrite=False,
        )
        client = FundamentalClient()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_fundamental(args), 0)
        limits = {(api, params["limit"]) for api, params in client.calls if api in {"income_vip", "cashflow_vip"}}
        self.assertEqual(limits, {("income_vip", 7000), ("cashflow_vip", 6000)})

    def test_fundamental_affected_codes_raise_on_corrupt_partition(self):
        """A corrupt partition must fail the targeted refresh loudly, not
        silently shrink the affected-codes set while the job publishes."""
        corrupt = self.raw_dir / "income_vip" / "period=20191231.parquet"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_bytes(b"not a parquet file")

        with self.assertRaises(ValueError):
            download.recent_fundamental_event_codes(
                self.raw_dir,
                refresh_periods={"20191231"},
                refresh_months=set(),
                period_datasets=["income_vip"],
                ann_month_datasets=[],
                start_date="20200601",
                end_date="20200603",
            )

    def test_fundamental_download_uses_explicit_codes_for_ts_code_datasets(self):
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200601",
            end_date="20200603",
            datasets=["dividend", "fina_audit", "fina_mainbz_vip"],
            force=True,
            page_limit=None,
            max_codes=None,
            codes=["920126.BJ"],
            fundamental_refresh_period_count=0,
            fundamental_refresh_ann_month_count=0,
            fundamental_refresh_ts_code_datasets=[],
            fundamental_refresh_event_days=0,
            fundamental_dividend_probe_days=0,
            min_interval_seconds=0,
            timeout_seconds=1,
            revision_ledger=None,
            allow_empty_revision_overwrite=False,
        )
        client = FundamentalClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_fundamental(args), 0)

        calls = [(api, params.get("ts_code")) for api, params in client.calls if params.get("offset") == 0]
        self.assertEqual(calls, [("dividend", "920126.BJ"), ("fina_audit", "920126.BJ"), ("fina_mainbz_vip", "920126.BJ")])

    def test_dividend_probe_uses_only_supported_date_params(self):
        client = FundamentalClient()

        codes = download.probe_recent_dividend_codes(client, "20200603", 1, page_limit=1000)

        self.assertEqual(codes, {"000001.SZ"})
        probe_params = [
            set(params) - {"limit", "offset"}
            for api_name, params in client.calls
            if api_name == "dividend"
        ]
        self.assertEqual(probe_params, [{"ann_date"}, {"imp_ann_date"}, {"ex_date"}, {"record_date"}])
        self.assertNotIn("pay_date", {key for params in probe_params for key in params})

    def test_event_flow_refreshes_trade_cal_before_same_day_margin_secs(self):
        self._write_trade_cal("20260603")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260604",
            end_date="20260604",
            datasets=["margin_secs"],
            force=True,
            page_limit=None,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = CalendarEventClient()

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_event_flow(args), 0)

        self.assertTrue((self.raw_dir / "margin_secs" / "trade_date=20260604.parquet").exists())
        self.assertIn(("trade_cal", {"exchange": "SSE", "start_date": "20260604", "end_date": "20260604"}), client.calls)
        margin_calls = [params for api_name, params in client.calls if api_name == "margin_secs"]
        self.assertEqual(margin_calls[0]["trade_date"], "20260604")

    def test_recent_fundamental_event_codes_filters_period_rows_by_visible_date(self):
        period_path = self.raw_dir / "income_vip" / "period=20200331.parquet"
        period_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200430", "end_date": "20200331"},
            {"ts_code": "000002.SZ", "ann_date": "20190430", "end_date": "20190331"},
        ]).to_parquet(period_path, index=False)

        codes = download.recent_fundamental_event_codes(
            self.raw_dir,
            {"20200331"},
            set(),
            ["income_vip"],
            [],
            "20200401",
            "20200603",
        )

        self.assertEqual(codes, {"000001.SZ"})

    def test_revision_sentinel_compares_without_overwriting_raw(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "adj_factor" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"trade_date": "20200102", "ts_code": "999999.SZ", "adj_factor": 9.9}])
        original.to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["adj_factor"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_events.jsonl"),
            output=str(self.root / "sentinel_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=DailyMarketClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(audit.audit_revision_sentinel(args), 0)

        self.assertIn("REVISION_ALERT", output.getvalue())
        self.assertTrue((self.root / "sentinel_events.jsonl").exists())
        summary = json.loads((self.root / "sentinel_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "warning")
        self.assertEqual(summary["metadata"]["totals"]["revision_events"], 1)
        self.assertTrue(pd.read_parquet(path).equals(original))

    def test_weekly_sentinel_probes_event_datasets_and_flags_massive_revision(self):
        # The weekly deep audit reuses the sentinel over the flat event tier:
        # a vendor rewriting a large share of history (the fina_mainbz class)
        # must escalate to error, not drown as routine one-off warnings.
        cal_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        days = ["20200102", "20200103", "20200106", "20200107", "20200108", "20200109"]
        pd.DataFrame({
            "exchange": ["SSE"] * 8,
            "cal_date": ["20200101"] + days + ["20200110"],
            "is_open": ["0"] + ["1"] * 6 + ["0"],
        }).to_parquet(cal_path, index=False)
        flow_dir = self.raw_dir / "moneyflow"
        flow_dir.mkdir(parents=True, exist_ok=True)
        for day in days:
            pd.DataFrame([{"trade_date": day, "ts_code": "000001.SZ", "net_mf_amount": 1.0}]).to_parquet(
                flow_dir / f"trade_date={day}.parquet", index=False
            )

        class RevisedFlowClient(EmptyTradeDateClient):
            # A STRUCTURAL rewrite: the vendor replaced every stored key with a
            # different universe (value-only recalculation must stay warning).
            def query(self, api_name, params=None, fields="", retries=5):
                params = params or {}
                self.calls.append((api_name, dict(params)))
                columns = fields.split(",") if fields else ["trade_date", "ts_code", "net_mf_amount"]
                rows = [
                    [
                        params.get("trade_date", "") if col == "trade_date"
                        else f"{300000 + i:06d}.SZ" if col == "ts_code" else "2.0"
                        for col in columns
                    ]
                    for i in range(40)
                ]
                return common.ApiResult(columns, rows)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200101",
            end_date="20200110",
            datasets=["moneyflow"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_events.jsonl"),
            output=str(self.root / "sentinel_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=RevisedFlowClient()):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.audit_revision_sentinel(args), 1)
        summary = json.loads((self.root / "sentinel_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "error")
        finding = next(f for f in summary["findings"] if f["check"] == "moneyflow_revision_sentinel")
        self.assertEqual(finding["severity"], "error")
        self.assertTrue(finding["details"]["massive_revision"])
        self.assertEqual(finding["details"]["revision_events"], 6)
        # Nested-strategy datasets stay out: a probe would not be like-for-like.
        with self.assertRaises(RuntimeError):
            audit.revision_sentinel_spec("ths_hot")

    def test_revision_comparison_flags_missing_and_duplicate_keys(self):
        missing = common.build_revision_event(
            dataset="daily",
            partition="trade_date=20200102",
            path=self.raw_dir / "daily" / "trade_date=20200102.parquet",
            old_df=pd.DataFrame([{"trade_date": "20200102", "open": 1.0}]),
            new_df=pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0}]),
            key_columns=["trade_date", "ts_code"],
            source="unit",
        )
        self.assertIsNotNone(missing)
        self.assertEqual(missing["comparison_issue"], "missing_key_columns")
        self.assertEqual(missing["missing_key_columns_old"], ["ts_code"])

        duplicate = common.build_revision_event(
            dataset="daily",
            partition="trade_date=20200102",
            path=self.raw_dir / "daily" / "trade_date=20200102.parquet",
            old_df=pd.DataFrame([
                {"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0},
                {"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0},
            ]),
            new_df=pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0}]),
            key_columns=["trade_date", "ts_code"],
            source="unit",
        )
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate["comparison_issue"], "duplicate_key_rows")
        self.assertEqual(duplicate["duplicate_key_rows_old"], 1)

    def test_revision_comparison_canonicalizes_numeric_values(self):
        old_df = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "adj_factor": 1}])
        new_df = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "adj_factor": 1.0}])
        event = common.build_revision_event(
            dataset="adj_factor",
            partition="trade_date=20200102",
            path=self.raw_dir / "adj_factor" / "trade_date=20200102.parquet",
            old_df=old_df,
            new_df=new_df,
            key_columns=["trade_date", "ts_code"],
            source="unit",
        )
        self.assertIsNone(event)

    def test_revision_event_records_changed_columns_and_row_samples(self):
        old_df = pd.DataFrame([
            {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.0, "amount": 100.0},
            {"trade_date": "20200102", "ts_code": "000002.SZ", "close": 20.0, "amount": 200.0},
            {"trade_date": "20200102", "ts_code": "000003.SZ", "close": 30.0, "amount": 300.0},
        ])
        new_df = pd.DataFrame([
            {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.5, "amount": 100.0},
            {"trade_date": "20200102", "ts_code": "000002.SZ", "close": 20.0, "amount": 201.5},
            {"trade_date": "20200102", "ts_code": "000004.SZ", "close": 40.0, "amount": 400.0},
        ])

        event = common.build_revision_event(
            dataset="daily",
            partition="trade_date=20200102",
            path=self.raw_dir / "daily" / "trade_date=20200102.parquet",
            old_df=old_df,
            new_df=new_df,
            key_columns=["trade_date", "ts_code"],
            source="unit",
        )

        self.assertIsNotNone(event)
        self.assertEqual(tuple(event), common.REVISION_EVENT_FIELDS)
        self.assertEqual(event["schema_version"], common.REVISION_EVENT_SCHEMA_VERSION)
        self.assertIsNone(event["write_action"])
        self.assertEqual(event["changed_keys"], 2)
        self.assertEqual(event["added_keys"], 1)
        self.assertEqual(event["removed_keys"], 1)
        self.assertEqual(event["changed_columns"], {"amount": 1, "close": 1})
        self.assertEqual(event["changed_columns_sample"][0]["key"], ["20200102", "000001.SZ"])
        self.assertEqual(event["changed_columns_sample"][0]["changes"], [{"column": "close", "old": "10", "new": "10.5"}])
        self.assertEqual(event["added_rows_sample"][0]["key"], ["20200102", "000004.SZ"])
        self.assertEqual(event["removed_rows_sample"][0]["key"], ["20200102", "000003.SZ"])

    def test_zero_ok_force_refresh_does_not_overwrite_existing_nonempty_partition(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "limit_list_d" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "limit": "U"}])
        original.to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["limit_list_d"],
            revision_ledger=str(self.root / "zero_ok_revision_events.jsonl"),
            allow_empty_revision_overwrite=False,
            force=True,
            page_limit=10000,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyTradeDateClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_board_trading(args), 0)

        self.assertIn("skipped_empty_revision_overwrite", output.getvalue())
        self.assertTrue(pd.read_parquet(path).equals(original))
        ledger = (self.root / "zero_ok_revision_events.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(json.loads(ledger[0])["removed_keys"], 1)

    def test_revision_aware_writer_empty_guard_does_not_depend_on_ledger(self):
        path = self.raw_dir / "limit_list_d" / "trade_date=20200102.parquet"
        original = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "limit": "U"}])
        common.write_parquet(path, original, api_name="limit_list_d", params={}, fields=list(original.columns))

        did_write = common.write_parquet_revision_aware(
            path,
            pd.DataFrame(columns=list(original.columns)),
            api_name="limit_list_d",
            params={"trade_date": "20200102"},
            fields=list(original.columns),
            key_columns=["trade_date", "ts_code", "limit"],
            revision_ledger=None,
            allow_empty_revision_overwrite=False,
        )

        self.assertFalse(did_write)
        self.assertTrue(pd.read_parquet(path).equals(original))

    def test_required_event_flow_zero_rows_raise_instead_of_cron_ok(self):
        self._write_trade_cal("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyTradeDateClient()):
            with self.assertRaisesRegex(RuntimeError, "required event/flow partitions returned zero or incomplete rows"):
                download.download_event_flow(args)

    def test_event_flow_zero_rows_not_ready_exits_75_without_mutation(self):
        self._write_trade_cal("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=EmptyTradeDateClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_event_flow(args), common.NO_MUTATION_RETRY_EXIT_CODE)

        self.assertIn("not_ready_no_mutation", output.getvalue())
        self.assertFalse((self.raw_dir / "margin" / "trade_date=20200102.parquet").exists())

    def test_margin_partial_exchange_day_is_not_ready_not_committed(self):
        # Exchanges publish independently; an SSE-only margin day poisons
        # market-wide aggregates and must ride the not-ready contract.
        class PartialMarginClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "margin":
                    return super().query(api_name, params, fields, retries)
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                row = ["20260529" if col == "trade_date" else "SSE" if col == "exchange_id" else "1.0" for col in columns]
                return common.ApiResult(columns, [row])

        self._write_trade_cal("20260529")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260529",
            end_date="20260529",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
        )
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=PartialMarginClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(download.download_event_flow(args), common.NO_MUTATION_RETRY_EXIT_CODE)
        self.assertIn("missing exchanges ['BSE', 'SZSE']", output.getvalue())
        self.assertFalse((self.raw_dir / "margin" / "trade_date=20260529.parquet").exists())

    def test_margin_committed_partial_partition_is_reattempted_and_audited(self):
        # Pre-BSE days need SSE+SZSE only; post-cutover days need all three.
        self.assertEqual(common.margin_missing_exchanges("20230210", ["SSE", "SZSE"]), [])
        self.assertEqual(common.margin_missing_exchanges("20230213", ["SSE", "SZSE"]), ["BSE"])

        margin_dir = self.raw_dir / "margin"
        margin_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": ["20260529"], "exchange_id": ["SSE"], "rzye": [1.0]}).to_parquet(
            margin_dir / "trade_date=20260529.parquet", index=False
        )
        # A covering non-force run must not skip the committed partial day.
        client = EmptyTradeDateClient()
        written, zero_skipped, blocked = download.download_event_trade_date_dataset(
            client, self.raw_dir, common.EVENT_FLOW_SPECS["margin"], ["20260529"], False, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertEqual([api for api, _ in client.calls], ["margin"])

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_margin_exchange_completeness(
            self.raw_dir, "margin", lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        )
        self.assertEqual(len(findings), 1)
        severity, code, _, details = findings[0]
        self.assertEqual((severity, code), ("error", "margin_exchange_completeness"))
        self.assertEqual(details["incomplete_days"], [{"trade_date": "20260529", "missing_exchanges": ["BSE", "SZSE"]}])

    def test_margin_detail_partial_exchange_day_is_refused_and_audited(self):
        # margin_detail names its venue only through the ts_code suffix, so an
        # SSE-only day looks like a normal partition while silently dropping
        # ~55% of the market. It must ride the same contract as the summary.
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_detail", "20260529", pd.DataFrame({"ts_code": ["600000.SH"]})
            ),
            ["BSE", "SZSE"],
        )
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_detail",
                "20260529",
                pd.DataFrame({"ts_code": ["600000.SH", "000001.SZ", "830000.BJ"]}),
            ),
            [],
        )
        # Datasets outside the margin family keep no per-dataset branch at all.
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "moneyflow", "20260529", pd.DataFrame({"ts_code": ["600000.SH"]})
            ),
            [],
        )

        class ShOnlyDetailClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "margin_detail":
                    return super().query(api_name, params, fields, retries)
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                row = ["20260529" if col == "trade_date" else "600000.SH" if col == "ts_code" else "1.0" for col in columns]
                return common.ApiResult(columns, [row])

        written, zero_skipped, _ = download.download_event_trade_date_dataset(
            ShOnlyDetailClient(), self.raw_dir, common.EVENT_FLOW_SPECS["margin_detail"], ["20260529"], True, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertFalse((self.raw_dir / "margin_detail" / "trade_date=20260529.parquet").exists())

        # A committed SH-only partition must be re-attempted, not skipped.
        detail_dir = self.raw_dir / "margin_detail"
        detail_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": ["20260529"], "ts_code": ["600000.SH"], "rzye": [1.0]}).to_parquet(
            detail_dir / "trade_date=20260529.parquet", index=False
        )
        client = EmptyTradeDateClient()
        written, zero_skipped, _ = download.download_event_trade_date_dataset(
            client, self.raw_dir, common.EVENT_FLOW_SPECS["margin_detail"], ["20260529"], False, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertEqual([api for api, _ in client.calls], ["margin_detail"])

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_margin_exchange_completeness(
            self.raw_dir, "margin_detail", lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        )
        self.assertEqual(len(findings), 1)
        severity, code, _, details = findings[0]
        self.assertEqual((severity, code), ("error", "margin_detail_exchange_completeness"))
        self.assertEqual(details["incomplete_days"], [{"trade_date": "20260529", "missing_exchanges": ["BSE", "SZSE"]}])

    def test_margin_secs_partial_exchange_day_is_refused_and_audited(self):
        # margin_secs carries its venue as a plain exchange column. The BSE
        # slice lags the vendor's SSE/SZSE feed intermittently, so a committed
        # two-exchange day must be re-attempted and audited, never frozen.
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_secs", "20260529", pd.DataFrame({"exchange": ["SSE", "SZSE"]})
            ),
            ["BSE"],
        )
        self.assertEqual(
            common.margin_family_missing_exchanges(
                "margin_secs", "20260529", pd.DataFrame({"exchange": ["SSE", "SZSE", "BSE"]})
            ),
            [],
        )

        class NoBseSecsClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name != "margin_secs":
                    return super().query(api_name, params, fields, retries)
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                rows = [
                    ["20260529" if col == "trade_date" else exchange if col == "exchange" else "600000.SH" for col in columns]
                    for exchange in ("SSE", "SZSE")
                ]
                return common.ApiResult(columns, rows)

        written, zero_skipped, _ = download.download_event_trade_date_dataset(
            NoBseSecsClient(), self.raw_dir, common.EVENT_FLOW_SPECS["margin_secs"], ["20260529"], True, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertFalse((self.raw_dir / "margin_secs" / "trade_date=20260529.parquet").exists())

        # A committed BSE-less partition must be re-attempted, not skipped.
        secs_dir = self.raw_dir / "margin_secs"
        secs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"trade_date": ["20260529"] * 2, "ts_code": ["600000.SH", "000001.SZ"], "exchange": ["SSE", "SZSE"]}
        ).to_parquet(secs_dir / "trade_date=20260529.parquet", index=False)
        client = EmptyTradeDateClient()
        written, zero_skipped, _ = download.download_event_trade_date_dataset(
            client, self.raw_dir, common.EVENT_FLOW_SPECS["margin_secs"], ["20260529"], False, None
        )
        self.assertEqual((written, zero_skipped), (0, 1))
        self.assertEqual([api for api, _ in client.calls], ["margin_secs"])

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_margin_exchange_completeness(
            self.raw_dir, "margin_secs", lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        )
        self.assertEqual(len(findings), 1)
        severity, code, _, details = findings[0]
        self.assertEqual((severity, code), ("error", "margin_secs_exchange_completeness"))
        self.assertEqual(details["incomplete_days"], [{"trade_date": "20260529", "missing_exchanges": ["BSE"]}])

    def test_business_payload_hollow_partition_is_an_error(self):
        d = self.raw_dir / "moneyflow"
        d.mkdir(parents=True, exist_ok=True)
        hollow = d / "trade_date=20260601.parquet"
        pd.DataFrame({
            "trade_date": ["20260601"], "ts_code": ["000001.SZ"],
            "buy_sm_vol": [None], "net_mf_amount": [None],
            "available_at": ["2026-06-02 09:00:00+08:00"], "available_at_rule": ["r"],
        }).to_parquet(hollow, index=False)

        findings: list[tuple[str, str, str, dict]] = []
        add = lambda sev, code, msg, details: findings.append((sev, code, msg, details))
        audit.audit_business_payload([hollow], "moneyflow", "moneyflow_payload", add, key_columns=("trade_date", "ts_code"))
        self.assertEqual(findings[0][0], "error")
        self.assertTrue(findings[0][3]["business_payload_empty"])

        # A healthy partition with one sparse column stays info.
        healthy = d / "trade_date=20260602.parquet"
        pd.DataFrame({
            "trade_date": ["20260602"], "ts_code": ["000001.SZ"],
            "buy_sm_vol": [1.0], "net_mf_amount": [None],
            "available_at": ["2026-06-03 09:00:00+08:00"], "available_at_rule": ["r"],
        }).to_parquet(healthy, index=False)
        findings.clear()
        audit.audit_business_payload([hollow, healthy], "moneyflow", "moneyflow_payload", add, key_columns=("trade_date", "ts_code"))
        self.assertEqual(findings[0][0], "info")  # newest non-empty partition wins

        # A requested source field that never arrived is a warning.
        findings.clear()
        audit.audit_business_payload(
            [healthy], "moneyflow", "moneyflow_payload", add,
            key_columns=("trade_date", "ts_code"), expected_fields=("buy_sm_vol", "sell_sm_vol"),
        )
        self.assertEqual(findings[0][0], "warning")
        self.assertEqual(findings[0][3]["missing_expected_fields"], ["sell_sm_vol"])

        # A file stripped down to keys is hollow, not merely field-incomplete:
        # every expected business field absent = error for spec'd families,
        # and require_business_columns covers fundamentals (empty spec fields).
        keys_only = d / "trade_date=20260603.parquet"
        pd.DataFrame({
            "trade_date": ["20260603"], "ts_code": ["000001.SZ"],
            "available_at": ["2026-06-04 09:00:00+08:00"], "available_at_rule": ["r"],
        }).to_parquet(keys_only, index=False)
        findings.clear()
        audit.audit_business_payload(
            [keys_only], "moneyflow", "moneyflow_payload", add,
            key_columns=("trade_date", "ts_code"), expected_fields=("trade_date", "ts_code", "buy_sm_vol", "sell_sm_vol"),
        )
        self.assertEqual(findings[0][0], "error")
        self.assertTrue(findings[0][3]["business_payload_empty"])
        findings.clear()
        audit.audit_business_payload(
            [keys_only], "income_vip", "income_vip_payload", add,
            key_columns=("trade_date", "ts_code"), require_business_columns=True,
        )
        self.assertEqual(findings[0][0], "error")
        # A genuinely key-only contract (suspend_d/top_list class) stays info.
        findings.clear()
        audit.audit_business_payload(
            [keys_only], "suspend_d", "suspend_d_payload", add,
            key_columns=("trade_date", "ts_code"), expected_fields=("trade_date", "ts_code"),
        )
        self.assertEqual(findings[0][0], "info")

    def test_full_market_coverage_flags_dropped_batch(self):
        codes = [f"{i:06d}.SZ" for i in range(100)]
        daily_dir = self.raw_dir / "daily"
        flow_dir = self.raw_dir / "moneyflow"
        daily_dir.mkdir(parents=True, exist_ok=True)
        flow_dir.mkdir(parents=True, exist_ok=True)
        for day, kept in (("20260601", 95), ("20260602", 80)):
            pd.DataFrame({"trade_date": [day] * 100, "ts_code": codes}).to_parquet(
                daily_dir / f"trade_date={day}.parquet", index=False
            )
            pd.DataFrame({"trade_date": [day] * kept, "ts_code": codes[:kept]}).to_parquet(
                flow_dir / f"trade_date={day}.parquet", index=False
            )

        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_full_market_coverage(self.raw_dir, ["moneyflow", "block_trade"], lambda *f: findings.append(f))
        # block_trade is sparse by design and must not be checked at all.
        self.assertEqual([f[1] for f in findings], ["moneyflow_daily_coverage"])
        severity, _, _, details = findings[0]
        self.assertEqual(severity, "warning")
        self.assertEqual(details["days_checked"], 2)
        self.assertEqual(details["days_below_threshold"], 1)
        self.assertEqual(details["low_coverage_sample"][0]["trade_date"], "20260602")
        self.assertAlmostEqual(details["low_coverage_sample"][0]["coverage"], 0.8)

    def test_stock_basic_missing_list_status_slice_is_an_error(self):
        d = self.raw_dir / "stock_basic"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "ts_code": ["000001.SZ"], "symbol": ["000001"], "name": ["平安银行"],
            "list_status": ["L"], "list_date": ["19910403"],
        }).to_parquet(d / "list_status=L.parquet", index=False)
        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_stock_basic(self.raw_dir, lambda *f: findings.append(f))
        severity, _, _, details = findings[-1]
        self.assertEqual(severity, "error")
        self.assertEqual(details["missing_list_status_slices"], ["D", "P"])

    def test_namechange_missing_code_batch_warns(self):
        d = self.raw_dir / "namechange"
        d.mkdir(parents=True, exist_ok=True)
        codes = [f"{i:06d}.SZ" for i in range(200)]
        pd.DataFrame({
            "ts_code": codes[:190], "name": ["x"] * 190,
            "start_date": ["20200101"] * 190, "end_date": [""] * 190,
            "ann_date": ["20200101"] * 190, "change_reason": ["上市"] * 190,
        }).to_parquet(d / "namechange.parquet", index=False)
        stock_basic = pd.DataFrame({"ts_code": codes})
        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_namechange(self.raw_dir, stock_basic, lambda *f: findings.append(f))
        severity, _, _, details = findings[0]
        self.assertEqual(severity, "warning")
        self.assertEqual(details["stock_basic_codes_without_namechange"], 10)

    def test_ths_member_coverage_error_on_missing_catalog_code(self):
        (self.raw_dir / "ths_index").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "ths_member").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "ts_code": ["885001.TI", "885002.TI", "700900.TI"],
            "name": ["a", "b", "c"], "type": ["N", "N", "S"],
        }).to_parquet(self.raw_dir / "ths_index" / "catalog.parquet", index=False)
        pd.DataFrame({"ts_code": ["885001.TI"], "con_code": ["000001.SZ"], "con_name": ["x"]}).to_parquet(
            self.raw_dir / "ths_member" / "ts_code=885001.TI.parquet", index=False
        )
        findings: list[tuple[str, str, str, dict]] = []
        audit.audit_ths_membership(self.raw_dir, lambda *f: findings.append(f))
        severity, _, _, details = findings[0]
        self.assertEqual(severity, "error")
        self.assertEqual(details["missing_sample"], ["885002.TI"])  # type S intentionally not expected

    def test_blocked_fundamental_shrink_overwrite_raises(self):
        spec = common.FUNDAMENTAL_SPECS["fina_mainbz_vip"]
        keys = list(spec.key_columns)
        d = self.raw_dir / spec.api_name
        d.mkdir(parents=True, exist_ok=True)
        old = pd.DataFrame({col: [f"{col}{i}" for i in range(40)] for col in keys})
        old["bz_sales"] = 1.0
        old.to_parquet(d / "ts_code=000001.SZ.parquet", index=False)

        class ShrunkClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = keys + ["bz_sales"]
                rows = [[f"{col}{i}" if col in keys else "2.0" for col in columns] for i in range(3)]
                return common.ApiResult(columns, rows)

        # The guard keeps the old partition AND the run must fail loudly:
        # swallowing the refusal froze fina_mainbz_vip at a stale vintage.
        with self.assertRaises(RuntimeError) as ctx:
            download.download_fundamental_ts_code_dataset(
                ShrunkClient(), self.raw_dir, spec, ["000001.SZ"], True, 5000,
            )
        self.assertIn("revision guard refused", str(ctx.exception))
        self.assertEqual(len(pd.read_parquet(d / "ts_code=000001.SZ.parquet")), 40)

    def test_event_skip_reattempts_partition_without_intact_sidecar(self):
        # An interrupted write leaves a parquet without (or with a stale)
        # sidecar; treating it as committed froze such partitions forever.
        secs_dir = self.raw_dir / "margin_secs"
        secs_dir.mkdir(parents=True, exist_ok=True)
        orphan = secs_dir / "trade_date=20260601.parquet"
        pd.DataFrame({
            "trade_date": ["20260601"] * 3, "ts_code": ["600000.SH", "000001.SZ", "830000.BJ"],
            "exchange": ["SSE", "SZSE", "BSE"],
        }).to_parquet(orphan, index=False)  # complete content, NO sidecar

        class ThreeExchangeSecsClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                rows = [
                    ["20260601" if col == "trade_date" else code if col == "ts_code" else exch if col == "exchange" else "x" for col in columns]
                    for code, exch in (("600000.SH", "SSE"), ("000001.SZ", "SZSE"), ("830000.BJ", "BSE"))
                ]
                return common.ApiResult(columns, rows)

        client = ThreeExchangeSecsClient()
        written, zero_skipped, _ = download.download_event_trade_date_dataset(
            client, self.raw_dir, common.EVENT_FLOW_SPECS["margin_secs"], ["20260601"], False, None
        )
        self.assertEqual([api for api, _ in client.calls], ["margin_secs"])
        self.assertEqual((written, zero_skipped), (1, 0))
        self.assertTrue(common.committed_partition_intact(orphan))

        # Once the pair is intact, the same covering run skips it.
        client.calls.clear()
        written, zero_skipped, _ = download.download_event_trade_date_dataset(
            client, self.raw_dir, common.EVENT_FLOW_SPECS["margin_secs"], ["20260601"], False, None
        )
        self.assertEqual(client.calls, [])
        # A replaced Parquet from an interrupted publish is also re-attempted:
        # its footer and row count no longer match the committed sidecar.
        pd.read_parquet(orphan).iloc[:2].to_parquet(orphan, index=False)
        self.assertFalse(common.committed_partition_intact(orphan))

    def test_daily_skip_reattempts_partition_without_intact_sidecar(self):
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        self.assertFalse(common.committed_partition_intact(path))
        client = DailyMarketClient()
        ledger = self.root / "daily_skip_heal.jsonl"
        with redirect_stdout(io.StringIO()):
            zero_skipped = download.download_trade_date_dataset(
                client, self.raw_dir, common.DAILY_SPECS["daily"], ["20200102"], False, 5000, ledger, False,
            )
        self.assertEqual([api for api, _ in client.calls], ["daily"])
        self.assertEqual(zero_skipped, 0)
        self.assertTrue(common.committed_partition_intact(path))
        client.calls.clear()
        with redirect_stdout(io.StringIO()):
            download.download_trade_date_dataset(
                client, self.raw_dir, common.DAILY_SPECS["daily"], ["20200102"], False, 5000, ledger, False,
            )
        self.assertEqual(client.calls, [])
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["write_id"] = "00000000-0000-0000-0000-000000000000"
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        self.assertFalse(common.committed_partition_intact(path))
        client.calls.clear()
        with redirect_stdout(io.StringIO()):
            download.download_trade_date_dataset(
                client, self.raw_dir, common.DAILY_SPECS["daily"], ["20200102"], False, 5000, ledger, False,
            )
        self.assertEqual([api for api, _ in client.calls], ["daily"])
        self.assertTrue(common.committed_partition_intact(path))

    def test_board_skip_reattempts_partition_without_intact_sidecar(self):
        spec = common.BOARD_TRADING_SPECS["limit_list_d"]
        path = self.raw_dir / spec.api_name / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "limit": "U"}]).to_parquet(path, index=False)
        self.assertFalse(download.should_skip_existing_partition(path, force=False))

        class LimitClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",") if fields else ["trade_date", "ts_code", "limit"]
                row = ["20200102" if col == "trade_date" else "000001.SZ" if col == "ts_code" else "U" if col == "limit" else "x" for col in columns]
                return common.ApiResult(columns, [row])

        client = LimitClient()
        with redirect_stdout(io.StringIO()):
            download.download_board_trade_date_dataset(client, self.raw_dir, spec, ["20200102"], False, 5000, None, False)
        self.assertEqual([api for api, _ in client.calls], ["limit_list_d"])
        self.assertTrue(common.committed_partition_intact(path))
        client.calls.clear()
        with redirect_stdout(io.StringIO()):
            download.download_board_trade_date_dataset(client, self.raw_dir, spec, ["20200102"], False, 5000, None, False)
        self.assertEqual(client.calls, [])

    def test_share_float_skip_reattempts_partition_without_intact_sidecar(self):
        fields = common.SHARE_FLOAT_FIELDS.split(",")
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        seeded = pd.DataFrame([["000001.SZ", "20200101", "20200102", 1.0, 0.1, "h1", "type"]], columns=fields)
        seeded.to_parquet(path, index=False)
        self.assertFalse(common.committed_partition_intact(path))

        class FloatClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",") if isinstance(fields, str) else list(fields)
                return common.ApiResult(columns, [["000001.SZ", "20200101", "20200102", 1.0, 0.1, "h1", "type"]])

        client = FloatClient()
        with redirect_stdout(io.StringIO()):
            report = download.query_share_float_to_path(
                client, self.raw_dir, path, {"ann_date": "20200101"}, "ann_date", False,
            )
        self.assertEqual([api for api, _ in client.calls], ["share_float"])
        self.assertFalse(report["skipped"])
        self.assertTrue(common.committed_partition_intact(path))
        client.calls.clear()
        report = download.query_share_float_to_path(
            client, self.raw_dir, path, {"ann_date": "20200101"}, "ann_date", False,
        )
        self.assertEqual(client.calls, [])
        self.assertTrue(report["skipped"])

    def test_fundamental_skip_reattempts_partition_without_intact_sidecar(self):
        spec = common.FUNDAMENTAL_SPECS["income_vip"]
        keys = list(spec.key_columns)
        path = self.raw_dir / spec.api_name / "period=20200331.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({col: ["x"] for col in keys}).to_parquet(path, index=False)
        self.assertFalse(common.committed_partition_intact(path))

        class PeriodClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                return common.ApiResult(keys, [["000001.SZ", "20200430", "20200430", "20200331", "1", "1", "1"]])

        client = PeriodClient()
        with redirect_stdout(io.StringIO()):
            download.download_fundamental_period_dataset(client, self.raw_dir, spec, ["20200331"], False, 5000)
        self.assertEqual([api for api, _ in client.calls], ["income_vip"])
        self.assertTrue(common.committed_partition_intact(path))
        client.calls.clear()
        with redirect_stdout(io.StringIO()):
            download.download_fundamental_period_dataset(client, self.raw_dir, spec, ["20200331"], False, 5000)
        self.assertEqual(client.calls, [])

    def test_trade_cal_past_day_revision_is_refused(self):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "exchange": ["SSE", "SSE"], "cal_date": ["20200102", "20200103"],
            "is_open": ["1", "1"], "pretrade_date": ["20200101", "20200102"],
        }).to_parquet(path, index=False)
        flipped = pd.DataFrame({
            "exchange": ["SSE", "SSE"], "cal_date": ["20200102", "20200103"],
            "is_open": ["0", "1"], "pretrade_date": ["20200101", "20200102"],
        })
        with self.assertRaises(RuntimeError) as ctx:
            download.merge_trade_cal_partition(path, flipped)
        self.assertIn("already-elapsed", str(ctx.exception))
        # Future-dated schedule publication merges freely.
        future = pd.DataFrame({
            "exchange": ["SSE"], "cal_date": ["29991231"], "is_open": ["0"], "pretrade_date": [""],
        })
        merged = download.merge_trade_cal_partition(path, future)
        self.assertEqual(len(merged), 3)




    def test_margin_exchange_check_fails_loudly_without_its_column(self):
        # Silently skipping an unverifiable partition is the failure mode this
        # guard exists to prevent, so an absent column must raise.
        with self.assertRaises(RuntimeError):
            common.margin_family_missing_exchanges(
                "margin_detail", "20260529", pd.DataFrame({"rzye": [1.0]})
            )

    def test_macro_static_full_writes_per_loop_registry_files(self):
        class RegistryClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",")
                exchange = (params or {}).get("exchange", "")
                if exchange != "CFFEX":
                    return common.ApiResult(columns, [])
                row = ["IF2001.CFX" if col == "ts_code" else "20191018" if col == "list_date" else exchange if col == "exchange" else "x" for col in columns]
                return common.ApiResult(columns, [row])

        client = RegistryClient()
        spec = common.MACRO_SPECS["fut_basic"]
        download.download_macro_static_full(client, self.raw_dir, spec, 10000)

        path = self.raw_dir / "fut_basic" / "exchange=CFFEX.parquet"
        self.assertTrue(path.exists())
        frame = pd.read_parquet(path)
        self.assertEqual(list(frame["ts_code"]), ["IF2001.CFX"])
        # available_at stamped from list_date so registry rows are PIT-gated.
        self.assertTrue(frame["available_at"].iloc[0].startswith("2019-10-18"))
        self.assertEqual({p.get("exchange") for _, p in client.calls}, set(spec.loop_values))

    def test_expected_macro_paths_cover_new_strategies(self):
        from autotrade.data_sources.tushare import audit
        import argparse

        args = argparse.Namespace(datasets=None)
        static_paths = audit.expected_macro_paths(self.raw_dir, common.MACRO_SPECS["fut_basic"], "20240101", "20240110", args)
        self.assertIn(self.raw_dir / "fut_basic" / "exchange=CFFEX.parquet", static_paths)
        self.assertEqual(len(static_paths), len(common.MACRO_SPECS["fut_basic"].loop_values))
        self._write_trade_cal("20240104")
        # Loop venues that listed later must not be expected before their start.
        self._write_trade_cal("20180104")
        opt_paths = audit.expected_macro_paths(self.raw_dir, common.MACRO_SPECS["opt_daily"], "20180104", "20180104", args)
        self.assertEqual({p.parent.name for p in opt_paths}, {"exchange=SSE"})

    def test_macro_completeness_month_expectation_clamps_to_open_dates(self):
        # The producing job ends on the last SSE open date; a weekend month
        # boundary must not expect the new month's partition yet, while a
        # missing month containing an elapsed trading day still errors.
        from autotrade.data_sources.tushare import audit
        import argparse

        cal_path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260701", "is_open": "1"},
            {"cal_date": "20260731", "is_open": "1"},
            {"cal_date": "20260801", "is_open": "0"},
            {"cal_date": "20260802", "is_open": "0"},
            {"cal_date": "20260803", "is_open": "1"},
        ]).to_parquet(cal_path, index=False)
        month_path = self.raw_dir / "cn_schedule" / "month=202607.parquet"
        month_frame = pd.DataFrame([{
            "month": "202607", "publish_date": "20260715", "title": "t",
            "issuing_org": "o", "data_api": "a",
            "available_at": "2026-07-15 23:59:59+08:00",
            "available_at_rule": "conservative_date_eod",
        }])
        common.write_parquet(month_path, month_frame, api_name="cn_schedule", params={}, fields=list(month_frame.columns))

        def partition_finding(end_date):
            findings = []
            args = argparse.Namespace(datasets=["cn_schedule"], start_date="20260701", end_date=end_date)
            audit.audit_macro_completeness(
                self.raw_dir, args,
                lambda sev, check, msg, details=None: findings.append((sev, check, details or {})),
            )
            found = [f for f in findings if f[1] == "cn_schedule_macro_partitions"]
            self.assertEqual(len(found), 1)
            return found[0]

        sev, _, details = partition_finding("20260802")
        self.assertEqual(details.get("missing_expected_files"), 0)
        self.assertNotEqual(sev, "error")
        sev, _, details = partition_finding("20260803")
        self.assertEqual(details.get("missing_expected_files"), 1)
        self.assertEqual(sev, "error")

    def test_event_flow_not_ready_vetoed_by_trade_cal_refresh(self):
        # A trade_cal coverage refresh IS a lake write: exit 75 asserts "no
        # mutation" and must not fire even when every dataset was empty. The
        # run still failed to reach required coverage, so it reports 76 (commit
        # the generation, retry later) rather than success.
        self._write_trade_cal("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
        )

        with patch.object(download, "load_token", return_value="token"), \
                patch.object(download, "TuShareClient", return_value=EmptyTradeDateClient()), \
                patch.object(download, "ensure_trade_cal_coverage", return_value=True):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    download.download_event_flow(args),
                    common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
                )

        self.assertIn("not_ready_after_mutation", output.getvalue())
        self.assertNotIn("not_ready_no_mutation", output.getvalue())

    def test_event_flow_blocked_shrink_raises_even_when_not_ready_enabled(self):
        # A non-empty response refused by the destructive-shrink guard is a
        # data-integrity alarm, never a "source not published yet" condition.
        self._write_trade_cal("20200102")
        path = self.raw_dir / "margin" / "trade_date=20200102.parquet"
        original = pd.DataFrame(
            [{"trade_date": "20200102", "exchange_id": f"EX{i:02d}", "rzye": 1.0} for i in range(30)]
        )
        common.write_parquet(path, original, api_name="margin", params={}, fields=list(original.columns))

        class ShrunkMarginClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name == "margin":
                    columns = fields.split(",")
                    # Exchange-complete (SSE+SZSE pre-BSE) so the completeness
                    # guard passes and the shrink guard is what fires.
                    rows = [
                        ["20200102" if col == "trade_date" else exchange if col == "exchange_id" else 1.0 for col in columns]
                        for exchange in ("SSE", "SZSE")
                    ]
                    return common.ApiResult(columns, rows)
                return super().query(api_name, params, fields, retries)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
            revision_ledger=str(self.root / "shrink_revision_events.jsonl"),
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=ShrunkMarginClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "overwrite was blocked"):
                    download.download_event_flow(args)

        self.assertTrue(pd.read_parquet(path).equals(original))

    def test_event_flow_zero_rows_not_ready_partial_write_reports_not_ready(self):
        self._write_trade_cal("20200102")

        class MarginOnlyClient(EmptyTradeDateClient):
            def query(self, api_name, params=None, fields="", retries=5):
                if api_name == "margin":
                    columns = fields.split(",")
                    # Both pre-BSE required exchanges: the day is complete and commits.
                    rows = [
                        ["20200102" if col == "trade_date" else exchange if col == "exchange_id" else 1.0 for col in columns]
                        for exchange in ("SSE", "SZSE")
                    ]
                    return common.ApiResult(columns, rows)
                return super().query(api_name, params, fields, retries)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["margin", "margin_detail"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            zero_rows_not_ready=True,
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=MarginOnlyClient()):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    download.download_event_flow(args),
                    common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
                )

        # margin committed, margin_detail still unpublished: the generation is
        # real, so this is never exit 75 -- but it is never success either, or
        # skip_if_already_ok would abandon margin_detail for this date.
        self.assertIn("not_ready_after_mutation", output.getvalue())
        self.assertTrue((self.raw_dir / "margin" / "trade_date=20200102.parquet").exists())
        self.assertFalse((self.raw_dir / "margin_detail" / "trade_date=20200102.parquet").exists())

    def test_revision_sentinel_marks_source_failures_as_error(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["daily"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_error_events.jsonl"),
            output=str(self.root / "sentinel_error_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=ErrorTradeDateClient()):
            self.assertEqual(audit.audit_revision_sentinel(args), 1)

        summary = json.loads((self.root / "sentinel_error_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["metadata"]["totals"]["api_errors"], 1)

    def test_revision_sentinel_warns_on_missing_local_partition(self):
        self._write_trade_cal("20200102")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["adj_factor"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_missing_events.jsonl"),
            output=str(self.root / "sentinel_missing_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=DailyMarketClient()):
            self.assertEqual(audit.audit_revision_sentinel(args), 0)

        summary = json.loads((self.root / "sentinel_missing_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "warning")
        self.assertEqual(summary["metadata"]["totals"]["missing_local_dates"], 1)

    def test_revision_sentinel_marks_required_remote_zero_as_error(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ"}]).to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            datasets=["daily"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=str(self.root / "sentinel_zero_events.jsonl"),
            output=str(self.root / "sentinel_zero_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with patch.object(audit, "load_token", return_value="token"), patch.object(audit, "TuShareClient", return_value=EmptyTradeDateClient()):
            self.assertEqual(audit.audit_revision_sentinel(args), 1)

        summary = json.loads((self.root / "sentinel_zero_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["metadata"]["totals"]["remote_zero_dates"], 1)

    def test_revision_sentinel_default_ledger_for_temp_raw_stays_local(self):
        self._write_trade_cal("20200102")
        path = self.raw_dir / "daily" / "trade_date=20200102.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "open": 99.0}]).to_parquet(path, index=False)
        args = argparse.Namespace(
            raw_dir="raw",
            start_date="20200102",
            end_date="20200102",
            datasets=["daily"],
            sample_size=0,
            seed=None,
            page_limit=10000,
            revision_ledger=common.REVISION_EVENTS_PATH,
            output=str(self.root / "sentinel_default_ledger_summary.json"),
            fail_on_revision=False,
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        with (
            patch.object(audit.Path, "cwd", return_value=self.root),
            patch.object(audit, "load_token", return_value="token"),
            patch.object(audit, "TuShareClient", return_value=DailyMarketClient()),
        ):
            self.assertEqual(audit.audit_revision_sentinel(args), 0)

        local_ledger = self.root / "revision_events.jsonl"
        formal_ledger = self.root / common.REVISION_EVENTS_PATH
        self.assertTrue(local_ledger.exists())
        self.assertFalse(formal_ledger.exists())
        event = json.loads(local_ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["dataset"], "daily")
        self.assertEqual(event["changed_columns"]["open"], 1)

    def test_intraday_by_date_audit_errors_on_zero_row_partition(self):
        self._write_trade_cal()
        path = self.raw_dir / common.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet"
        empty = pd.DataFrame(columns=common.STK_MINS_REQUIRED_COLUMNS)
        common.write_parquet(path, empty, api_name=common.STK_MINS_API_NAME, params={}, fields=list(empty.columns))
        status_path = self.root / "status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            output_dataset=common.STK_MINS_BY_DATE_DATASET,
            codes=None,
            max_codes=None,
            expected_codes_source="none",
            min_rows_per_day=0,
            allow_missing_codes=0,
            full_scan=False,
            sample_limit=0,
            output=str(status_path),
        )

        self.assertEqual(audit.audit_intraday_by_date(args), 1)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        inventory = next(item for item in status["findings"] if item["check"] == f"{common.STK_MINS_BY_DATE_DATASET}_inventory")
        self.assertEqual(inventory["severity"], "error")
        self.assertEqual(inventory["details"]["zero_row_files"], 1)

    def _write_by_date_minutes(self, trade_date: str, *, duplicate_first_bar: bool = False) -> None:
        rows = []
        for bar_time in ("09:30", "15:00"):
            rows.append({
                "ts_code": "000001.SZ",
                "trade_time": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {bar_time}:00",
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "vol": 100.0, "amount": 100.0,
                "trade_date": trade_date,
                "available_at": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}T{bar_time}:00+08:00",
                "available_at_rule": "bar_close",
            })
        if duplicate_first_bar:
            rows.append(dict(rows[0]))
        path = self.raw_dir / common.STK_MINS_BY_DATE_DATASET / f"trade_date={trade_date}.parquet"
        df = pd.DataFrame(rows)
        common.write_parquet(path, df, api_name=common.STK_MINS_API_NAME, params={}, fields=list(df.columns))

    def test_intraday_by_date_deep_checks_cover_the_newest_dates(self):
        # The head slice re-validated the window's FIRST trading days forever;
        # deep checks must target what the pipeline just wrote.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20200102", "is_open": "1"},
            {"cal_date": "20200103", "is_open": "1"},
        ]).to_parquet(cal, index=False)
        self._write_by_date_minutes("20200102")
        self._write_by_date_minutes("20200103", duplicate_first_bar=True)
        status_path = self.root / "status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200103",
            output_dataset=common.STK_MINS_BY_DATE_DATASET,
            codes=None,
            max_codes=None,
            expected_codes_source="none",
            min_rows_per_day=0,
            allow_missing_codes=0,
            full_scan=False,
            sample_limit=1,
            output=str(status_path),
        )

        self.assertEqual(audit.audit_intraday_by_date(args), 0)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        deep = next(item for item in status["findings"] if item["check"] == f"{common.STK_MINS_BY_DATE_DATASET}_deep_checks")
        self.assertEqual(deep["details"]["files_checked"], 1)
        self.assertEqual(deep["details"]["bad_days"], 1)
        self.assertEqual(deep["details"]["bad_day_sample"][0]["trade_date"], "20200103")

    def test_stk_mins_sample_rotates_deterministically_with_seed(self):
        base = self.raw_dir / common.STK_MINS_DATASET
        files = []
        for code in ("000001.SZ", "000002.SZ", "600000.SH"):
            path = base / f"ts_code={code}" / "year=2020.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "ts_code": code,
                "trade_time": "2020-01-02 10:00:00",
                "trade_date": "20200102",
                "available_at": "2020-01-02T10:00:00+08:00",
            }]).to_parquet(path, index=False)
            files.append(path)
        row_counts = {str(path): 1 for path in files}
        seed = "20260101"
        ranked = sorted(files)
        offset = sum((index + 1) * ord(char) for index, char in enumerate(seed)) % len(ranked)
        expected_first = (ranked[offset:] + ranked[:offset])[0]

        findings = []
        def add(severity, check, message, details=None):
            findings.append({"check": check, "details": details or {}})

        audit.audit_stk_mins_sample(files, row_counts, 1, seed, add)
        sample_finding = findings[-1]["details"]
        self.assertEqual(sample_finding["files_sampled"], 1)
        # No 09:30/15:00 bars in the fixture: the flagged file IS the sampled one.
        self.assertEqual(sample_finding["missing_0930_sample"], [str(expected_first)])

    def test_design_fact_checks_report_as_info(self):
        # Unconditional design-fact notes must not occupy the warning severity:
        # a healthy day should be able to read status=ok (round 18).
        daily = self.raw_dir / "daily"
        daily.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}]).to_parquet(daily / "trade_date=20200102.parquet", index=False)
        findings = []
        def add(severity, check, message, details=None):
            findings.append({"severity": severity, "check": check})

        audit.audit_pit_availability(self.raw_dir, add)
        audit.audit_fundamental_unit_and_pit_semantics(self.raw_dir, add)
        company = self.raw_dir / "stock_company"
        company.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ", "exchange": "SZSE", "com_name": "平安银行"}]).to_parquet(company / "exchange=SZSE.parquet", index=False)
        stock_basic = pd.DataFrame([{"ts_code": "000001.SZ"}, {"ts_code": "000002.SZ"}])
        audit.audit_stock_company(self.raw_dir, stock_basic, add)

        by_check = {item["check"]: item["severity"] for item in findings}
        self.assertEqual(by_check["pit_available_at"], "info")
        self.assertEqual(by_check["fundamental_unit_and_pit_semantics"], "info")
        self.assertEqual(by_check["stock_company_vs_stock_basic"], "info")

    def test_latest_parquet_schema_reads_the_newest_partition(self):
        dataset_dir = self.raw_dir / "daily"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"old_only": 1}]).to_parquet(dataset_dir / "trade_date=20200102.parquet", index=False)
        pd.DataFrame([{"old_only": 1, "added_later": 2}]).to_parquet(dataset_dir / "trade_date=20260101.parquet", index=False)

        self.assertEqual(audit.latest_parquet_schema(self.raw_dir, "daily"), ["old_only", "added_later"])



    def test_event_expectations_clamp_to_last_sse_open_date(self):
        # A weekend calendar end date must not expect the new month's event
        # partition before the next producing run (the month-boundary weekend
        # false-error class; macro/fundamental got the same clamp earlier).
        # Text is deliberately NOT clamped -- see the natural-day test below.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260731", "is_open": "1"},
            {"cal_date": "20260801", "is_open": "0"},
            {"cal_date": "20260802", "is_open": "0"},
        ]).to_parquet(cal, index=False)
        holders = pd.DataFrame([{
            "ts_code": "000001.SZ", "ann_date": "20260710", "end_date": "20260630",
            "holder_num": 10000, "available_at": "2026-07-10T23:59:59+08:00", "available_at_rule": "ann_date_eod",
        }])
        common.write_parquet(
            self.raw_dir / "stk_holdernumber" / "month=202607.parquet",
            holders, api_name="stk_holdernumber", params={}, fields=list(holders.columns),
        )
        status_path = self.root / "event_status.json"
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260731",
            end_date="20260801",
            datasets=["stk_holdernumber"],
            output=str(status_path),
        )
        with patch.object(audit.Path, "cwd", return_value=self.root):
            audit.audit_event_flow_only(args)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        scope_finding = next(f for f in status["findings"] if f["check"] == "event_flow_expected_scope")
        self.assertEqual(scope_finding["details"]["covered_end_date"], "20260731")
        partitions = next(f for f in status["findings"] if f["check"] == "stk_holdernumber_event_partitions")
        self.assertEqual(partitions["details"]["missing_expected_files"], 0)

    def _audit_text_window(self, start_date: str, end_date: str, output_name: str) -> dict:
        status_path = self.root / output_name
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            text_start_date=start_date,
            text_end_date=end_date,
            text_datasets=["irm_qa_sh"],
            news_src=[],
            major_news_src=[],
            output=str(status_path),
        )
        with patch.object(audit.Path, "cwd", return_value=self.root):
            audit.audit_text_only(args)
        return json.loads(status_path.read_text(encoding="utf-8"))

    def test_text_expectations_follow_natural_days_across_a_weekend(self):
        # Text publishes on weekends, and its producing job runs every calendar
        # evening, so a Saturday end date must expect Saturday's partition.
        # Clamping to the last SSE open day left weekend text unchecked until
        # the next session (data docs §4).
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260731", "is_open": "1"},
            {"cal_date": "20260801", "is_open": "0"},
        ]).to_parquet(cal, index=False)
        qa = pd.DataFrame([{
            "trade_date": "20260731", "ts_code": "000001.SZ", "q": "问题", "a": "回答",
            "pub_time": "2026-07-31 18:00:00",
        }])
        common.write_parquet(
            self.raw_dir / "irm_qa_sh" / "date=20260731.parquet",
            qa, api_name="irm_qa_sh", params={}, fields=list(qa.columns),
        )

        missing = self._audit_text_window("20260731", "20260801", "text_status_missing.json")
        scope = next(f for f in missing["findings"] if f["check"] == "text_expected_scope")
        self.assertEqual(scope["details"]["end_date"], "20260801")
        self.assertNotIn("covered_end_date", scope["details"])
        partitions = next(f for f in missing["findings"] if f["check"] == "irm_qa_sh_text_partitions")
        self.assertEqual(partitions["details"]["missing_expected_files"], 1)
        self.assertEqual(missing["status"], "error")

        # The same window is clean once the calendar-day job has landed Saturday
        # (an empty vendor response still writes the partition).
        common.write_parquet(
            self.raw_dir / "irm_qa_sh" / "date=20260801.parquet",
            qa.iloc[0:0], api_name="irm_qa_sh", params={}, fields=list(qa.columns),
        )
        landed = self._audit_text_window("20260731", "20260801", "text_status_landed.json")
        partitions = next(f for f in landed["findings"] if f["check"] == "irm_qa_sh_text_partitions")
        self.assertEqual(partitions["details"]["missing_expected_files"], 0)
        self.assertNotIn(
            "irm_qa_sh_text_partitions",
            [f["check"] for f in landed["findings"] if f["severity"] == "error"],
        )

    def test_board_trading_download_and_audit_use_dedicated_dimension(self):
        self._write_trade_cal("20231101")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["kpl_list", "limit_step", "limit_list_ths", "top_list", "hm_list", "ths_hot", "dc_hot"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            kpl_tag=["涨停"],
            ths_limit_type=["涨停池"],
            ths_hot_market=["热股"],
            dc_hot_market=["A股市场"],
            dc_hot_type=["人气榜"],
            hot_is_new=["N"],
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=BoardClient()):
            self.assertEqual(download.download_board_trading(args), 0)

        self.assertTrue((self.raw_dir / "kpl_list" / f"tag={common.safe_partition_value('涨停')}" / "trade_date=20231101.parquet").exists())
        self.assertTrue((self.raw_dir / "limit_list_ths" / f"limit_type={common.safe_partition_value('涨停池')}" / "trade_date=20231101.parquet").exists())
        hot = pd.read_parquet(self.raw_dir / "ths_hot" / f"market={common.safe_partition_value('热股')}" / "is_new=N" / "trade_date=20231101.parquet")
        self.assertEqual(hot.loc[0, "available_at"], "2020-01-02 10:00:00+08:00")

        # limit_list_d is a regular board dataset: give it a valid partition
        # with the PIT stamp columns.
        limit_rows = pd.DataFrame([{
            "trade_date": "20231101", "ts_code": "000001.SZ", "limit": "U",
            "available_at": "2023-11-01 16:00:00+08:00",
            "available_at_rule": "official_16_from:trade_date",
        }])
        common.write_parquet(
            self.raw_dir / "limit_list_d" / "trade_date=20231101.parquet",
            limit_rows,
            api_name="limit_list_d",
            params={"trade_date": "20231101"},
            fields=list(limit_rows.columns),
        )
        status_path = self.root / "board_status.json"
        audit_args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["kpl_list", "limit_step", "limit_list_d", "limit_list_ths", "top_list", "hm_list", "ths_hot", "dc_hot"],
            kpl_tag=["涨停"],
            ths_limit_type=["涨停池"],
            ths_hot_market=["热股"],
            dc_hot_market=["A股市场"],
            dc_hot_type=["人气榜"],
            hot_is_new=["N"],
            output=str(status_path),
        )
        self.assertEqual(audit.audit_board_trading_only(audit_args), 0)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "ok")
        self.assertIn("kpl_list", status["datasets"])
        self.assertIn("limit_list_d", status["datasets"])
        self.assertEqual(status["datasets"]["limit_list_d"]["status"], "ok")

    def test_board_static_hm_list_refreshes_every_run(self):
        class CountingBoardClient(BoardClient):
            def __init__(self):
                self.calls = []

            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append(api_name)
                return super().query(api_name, params, fields, retries)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20231101",
            end_date="20231101",
            datasets=["hm_list"],
            force=False,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        client = CountingBoardClient()
        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=client):
            self.assertEqual(download.download_board_trading(args), 0)
            self.assertTrue((self.raw_dir / "hm_list" / "hm_list.parquet").exists())
            # The reference table must re-pull on every run, not skip once downloaded.
            self.assertEqual(download.download_board_trading(args), 0)
        self.assertEqual(client.calls.count("hm_list"), 2)

    def test_board_trading_skips_non_trading_window(self):
        self._write_trade_cal("20260530", is_open="0")
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260530",
            end_date="20260530",
            datasets=["kpl_list", "limit_step", "limit_cpt_list"],
            force=True,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            kpl_tag=["涨停"],
            ths_limit_type=["涨停池"],
            ths_hot_market=["热股"],
            dc_hot_market=["A股市场"],
            dc_hot_type=["人气榜"],
            hot_is_new=["N"],
        )

        with patch.object(download, "load_token", return_value="token"), patch.object(download, "TuShareClient", return_value=NoQueryClient()):
            self.assertEqual(download.download_board_trading(args), 0)

    def test_text_source_time_is_normalized_to_china_timezone(self):
        frame = pd.DataFrame([{"title": "sample", "pub_time": "2020-01-02 10:00:00", "src": "x"}])
        out = common.augment_text_frame(frame, common.TEXT_SPECS["major_news"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-02 10:00:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "source:pub_time")

    def test_same_day_empty_cctv_fetch_does_not_block_the_evening_refresh(self):
        # The 08:55 pre-open window now includes day D. cctv_news airs in the
        # evening, so the morning fetch of day D legitimately returns zero
        # rows; the zero-row partition it writes must not block the evening
        # --force refresh from landing the real transcript.
        spec = common.TEXT_SPECS["cctv_news"]
        path = self.raw_dir / "cctv_news" / "date=20260809.parquet"

        class DayClient:
            def __init__(self, rows):
                self.rows = rows

            def query(self, api_name, params=None, fields="", retries=5):
                return common.ApiResult(fields.split(","), list(self.rows))

        with redirect_stdout(io.StringIO()):
            download.download_text_day(DayClient([]), self.raw_dir, spec, ["20260809"], True)
        self.assertEqual(common.parquet_rows(path), 0)

        with redirect_stdout(io.StringIO()):
            download.download_text_day(DayClient([["20260809", "标题", "内容"]]), self.raw_dir, spec, ["20260809"], True)
        frame = pd.read_parquet(path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "title"], "标题")


# Source: test_tushare_intraday_by_date.py
import types
import unittest



def load_tushare_data_module():
    return types.SimpleNamespace(
        compact_intraday_by_date=download.compact_intraday_by_date,
        audit_intraday_by_date=audit.audit_intraday_by_date,
    )


class IntradayByDateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmp.name) / "raw"
        self.module = load_tushare_data_module()
        self._write_reference_inputs()
        self._write_stock_year_inputs()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_reference_inputs(self):
        trade_cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20200102", "is_open": "1"},
        ]).to_parquet(trade_cal, index=False)

        stock_basic = self.raw_dir / "stock_basic" / "list_status=L.parquet"
        stock_basic.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "name": "A",
                "market": "主板",
                "exchange": "SZSE",
                "list_status": "L",
                "list_date": "19910403",
                "delist_date": "",
            },
            {
                "ts_code": "000002.SZ",
                "name": "B",
                "market": "主板",
                "exchange": "SZSE",
                "list_status": "L",
                "list_date": "19910129",
                "delist_date": "",
            },
        ]).to_parquet(stock_basic, index=False)

    def _write_stock_year_inputs(self):
        rows_by_code = {
            "000001.SZ": [
                ("2020-01-02 09:30:00", 10.0),
                ("2020-01-02 15:00:00", 10.5),
            ],
            "000002.SZ": [
                ("2020-01-02 09:30:00", 20.0),
                ("2020-01-02 15:00:00", 20.5),
            ],
        }
        for ts_code, bars in rows_by_code.items():
            path = self.raw_dir / "stk_mins_1min" / f"ts_code={ts_code}" / "year=2020.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                {
                    "ts_code": ts_code,
                    "trade_time": trade_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "vol": 100,
                    "amount": price * 100,
                    "trade_date": "20200102",
                    "available_at": f"{trade_time}+08:00",
                    "available_at_rule": "source:trade_time_bar_close",
                }
                for trade_time, price in bars
            ]).to_parquet(path, index=False)

    def test_compact_and_audit_intraday_by_date(self):
        compact_args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            output_dataset="stk_mins_1min_by_date",
            codes=None,
            max_codes=None,
            expected_codes_source="active",
            min_rows_per_day=4,
            allow_missing_codes=0,
            force=False,
            allow_empty=False,
            allow_validation_warnings=False,
        )
        self.assertEqual(self.module.compact_intraday_by_date(compact_args), 0)

        output = self.raw_dir / "stk_mins_1min_by_date" / "trade_date=20200102.parquet"
        self.assertTrue(output.exists())
        self.assertTrue(output.with_suffix(output.suffix + ".meta.json").exists())
        df = pd.read_parquet(output)
        self.assertEqual(len(df), 4)
        self.assertEqual(sorted(df["ts_code"].unique().tolist()), ["000001.SZ", "000002.SZ"])
        self.assertEqual(set(df["trade_date"].astype(str)), {"20200102"})

        status_path = Path(self.tmp.name) / "intraday_by_date_status.json"
        audit_args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            output_dataset="stk_mins_1min_by_date",
            codes=None,
            max_codes=None,
            expected_codes_source="active",
            min_rows_per_day=4,
            allow_missing_codes=0,
            full_scan=True,
            sample_limit=20,
            output=str(status_path),
        )
        self.assertEqual(self.module.audit_intraday_by_date(audit_args), 0)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "ok")




class AuditSubcommandExecutionTest(unittest.TestCase):
    """Every formal audit subcommand must EXECUTE, not just render as a cron
    command string: the 2026-07-25 selector-consolidation merge dropped two
    imports and the macro/text audits crashed with NameError for five nights
    while their status files stayed frozen at a pre-outage state."""

    def _run_audit(self, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["tushare_audit.py", *argv]), redirect_stdout(io.StringIO()):
            return audit.main()

    def test_text_subcommand_executes_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            output = Path(tmp) / "text_status.json"
            code = self._run_audit([
                "text", "--raw-dir", str(raw_dir), "--output", str(output),
                "--start-date", "20240101", "--end-date", "20240102",
                "--text-datasets", "cctv_news",
            ])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["report_type"], "text_evidence")
            # An empty lake is a completed audit with error findings, never a crash.
            self.assertEqual(report["status"], "error")
            self.assertEqual(code, 1)

    def test_macro_subcommand_executes_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            output = Path(tmp) / "macro_status.json"
            code = self._run_audit([
                "macro", "--raw-dir", str(raw_dir), "--output", str(output),
                "--start-date", "20240101", "--end-date", "20240102",
                "--datasets", "cn_gdp",
            ])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["report_type"], "macro_context")
            self.assertEqual(report["status"], "error")
            self.assertEqual(code, 1)


class _StubLock:
    fd = None

    def release(self) -> None:
        return None


class CronFailureRecordingTest(unittest.TestCase):
    """A cron job that aborts before run_update (e.g. at the dirty-lake fence)
    must still demote its per-job state record: during the
    2026-07-23..29 outage, 9 of 14 jobs kept a week-old ``ok`` because the
    fence RuntimeError propagated past the state writer (silent false
    success)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True)
        self.jobs_root = self.root / "runtime" / "jobs"
        self.jobs_root.mkdir(parents=True)
        (self.jobs_root / "evening.json").write_text(
            json.dumps({"status": "ok", "end_date": "20260722"}), encoding="utf-8"
        )
        config_path = self.root / "schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {"evening": {"operation": "update"}},
        }), encoding="utf-8")
        self.args = argparse.Namespace(
            config=str(config_path), job="evening",
            start_date="20260623", end_date="20260723",
            dry_run=False, force_run=False,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _job_record(self) -> dict:
        return json.loads((self.jobs_root / "evening.json").read_text(encoding="utf-8"))

    def _run_main(self):
        return (
            patch.object(cron_update, "parse_args", return_value=self.args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "prune_run_logs"),
            patch.object(cron_update, "JOB_STATE_ROOT", self.jobs_root),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=_StubLock()),
        )

    def test_dirty_lake_fence_abort_records_error_state(self) -> None:
        txn = cron_update.begin_raw_generation_update(self.raw_dir, {
            "job": "other_job", "start_date": "20260601", "end_date": "20260701",
            "commands": [["python", "download.py", "evening"]], "config_identity": {},
        })
        cron_update.mark_raw_generation_dirty(self.raw_dir, txn, error="boom")

        patches = self._run_main()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "unfinished mutation"):
            cron_update.main()

        record = self._job_record()
        self.assertEqual(record["status"], "error")
        self.assertIn("unfinished mutation", record["error"])
        self.assertEqual(record["end_date"], "20260723")

    def test_nonzero_returncode_records_error(self) -> None:
        cron_update.write_raw_generation(self.raw_dir)
        patches = self._run_main()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patch.object(cron_update, "run_update", return_value=1), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(cron_update.main(), 1)

        record = self._job_record()
        self.assertEqual(record["status"], "error")
        # No run log exists (run_update is stubbed): the summary falls back to
        # the exit code, so the state never records a content-free error.
        self.assertEqual(record["error"], "job_returncode=1")
        generation = json.loads((self.raw_dir / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual(generation["state"], "dirty")



class FailureSummaryTest(unittest.TestCase):
    """The persisted job state's error field must distinguish "the audit
    completed and found data errors" from "the tool crashed": completed audit
    domain summaries win, else the traceback's final exception line, else the
    bare exit code."""

    def test_audit_findings_traceback_and_fallback_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "started_at=x\n"
                "core_market audit status=ok errors=0 warnings=3 output=/s/core.json\n"
                "board_trading audit status=error errors=13 warnings=2 output=/s/board.json\n",
                encoding="utf-8",
            )
            summary = cron_update.summarize_failure_from_log(log, 1)
            self.assertIn("board_trading audit status=error errors=13 warnings=2", summary)
            self.assertIn("core_market audit status=ok errors=0 warnings=3", summary)
            self.assertNotIn("output=", summary)

            log.write_text(
                "Traceback (most recent call last):\n"
                '  File "download.py", line 1551, in download_event_flow\n'
                "RuntimeError: required event/flow partitions returned zero rows\n",
                encoding="utf-8",
            )
            self.assertEqual(
                cron_update.summarize_failure_from_log(log, 1),
                "RuntimeError: required event/flow partitions returned zero rows",
            )

            self.assertEqual(
                cron_update.summarize_failure_from_log(Path(tmp) / "missing.log", 7),
                "job_returncode=7",
            )


class CorruptSidecarTest(unittest.TestCase):
    """A present-but-unparseable .meta.json is torn-write evidence and must be
    distinguishable from an absent sidecar: parquet_meta used to return {} for
    both, so the audit counted corruption as 'legacy' info and the snapshot
    silently fell back to imputed availability."""

    def test_parquet_meta_distinguishes_absent_from_corrupt(self):
        from autotrade.environment.data.pit import CorruptSidecarError, parquet_meta

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trade_date=20260101.parquet"
            pd.DataFrame({"a": [1]}).to_parquet(path, index=False)
            self.assertEqual(parquet_meta(path), {})  # absent sidecar: pre-scheme file
            sidecar = Path(str(path) + ".meta.json")
            sidecar.write_text('{"fetched_at": "2026', encoding="utf-8")  # torn write
            with self.assertRaisesRegex(CorruptSidecarError, "unreadable parquet sidecar"):
                parquet_meta(path)
            sidecar.write_text('["not", "an", "object"]', encoding="utf-8")
            with self.assertRaisesRegex(CorruptSidecarError, "not a JSON object"):
                parquet_meta(path)

    def test_revision_writer_heals_corrupt_sidecar_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trade_date=20260101.parquet"
            pd.DataFrame({"ts_code": ["000001.SZ"], "v": [1]}).to_parquet(path, index=False)
            Path(str(path) + ".meta.json").write_text("{broken", encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                wrote = common.write_parquet_revision_aware(
                    path,
                    pd.DataFrame({"ts_code": ["000001.SZ"], "v": [2]}),
                    api_name="daily",
                    params={},
                    fields=["ts_code", "v"],
                    key_columns=["ts_code"],
                    revision_ledger=Path(tmp) / "ledger.jsonl",
                )
            self.assertTrue(wrote)
            alert_lines = [line for line in out.getvalue().splitlines() if "corrupt_sidecar_replaced" in line]
            self.assertEqual(len(alert_lines), 1)
            # The rewrite repaired the torn pair: the sidecar parses again.
            from autotrade.environment.data.pit import parquet_meta

            self.assertTrue(parquet_meta(path).get("write_id"))

    def test_audit_records_corrupt_sidecar_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            dataset_dir = raw_dir / "daily"
            dataset_dir.mkdir()
            path = dataset_dir / "trade_date=20260101.parquet"
            pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(path, index=False)
            Path(str(path) + ".meta.json").write_text("{broken", encoding="utf-8")
            findings: list[dict] = []

            def add(severity, check, message, details=None):
                findings.append({"severity": severity, "check": check, "details": details or {}})

            audit.audit_integrated_filesystem(raw_dir, ["daily"], add)
            inventory = next(f for f in findings if f["check"] == "integrated_filesystem")
            self.assertEqual(inventory["severity"], "error")
            self.assertEqual(inventory["details"]["corrupt_sidecars"], 1)


class ManualProductionWriteRefusalTest(unittest.TestCase):
    """The cron runner is the ONLY production-lake write entrance (fence,
    per-job state, alerting, flock). Direct manual commands against the
    production data/raw are refused with the runner equivalent; cron children
    (TUSHARE_UPDATE_LOCK_HELD=1) and non-production raw dirs pass through."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "data" / "raw").mkdir(parents=True)
        self._old_cwd = os.getcwd()
        os.chdir(self.root)
        self._env = patch.dict(os.environ)
        self._env.start()
        os.environ.pop("TUSHARE_UPDATE_LOCK_HELD", None)

    def tearDown(self) -> None:
        self._env.stop()
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _run(self, argv: list[str], impl_name: str, impl_result) -> int:
        with (
            patch.object(sys, "argv", ["tushare_download.py", *argv]),
            patch.object(download, "_production_repo_root", lambda: self.root),
            patch.object(download, impl_name, return_value=impl_result),
            redirect_stdout(io.StringIO()),
        ):
            return download.main()

    def test_manual_production_write_is_refused_with_runner_equivalent(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        self.assertFalse((self.root / "data" / "raw" / ".raw_generation.json").exists())

    def test_genuine_cron_child_with_inherited_lock_fd_passes(self) -> None:
        # Only an INHERITED, exclusively-held lock fd proves a runner child:
        # env flag + "somebody holds the lock" was forgeable by any manual run.
        import fcntl

        from autotrade.data_sources.tushare.io import UPDATER_LOCK_FD_ENV

        lock_path = self.root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        lock_path.parent.mkdir(parents=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.environ[UPDATER_LOCK_FD_ENV] = str(fd)
            code = self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
            self.assertEqual(code, 0)
        finally:
            os.close(fd)

    def test_forged_marker_without_an_inherited_lock_fd_is_refused(self) -> None:
        import fcntl

        from autotrade.data_sources.tushare.io import UPDATER_LOCK_FD_ENV

        lock_path = self.root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        lock_path.parent.mkdir(parents=True)
        # (a) no fd at all
        os.environ[UPDATER_LOCK_FD_ENV] = "1"
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        # (b) an fd for the RIGHT file but nobody holds the lock exclusively
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        try:
            os.environ[UPDATER_LOCK_FD_ENV] = str(fd)
            with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
                self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        finally:
            os.close(fd)
        # (c) an fd pointing at an unrelated file, lock genuinely held elsewhere
        other = self.root / "unrelated.bin"
        other.write_bytes(b"x")
        holder = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        decoy = os.open(str(other), os.O_RDONLY)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX)
            os.environ[UPDATER_LOCK_FD_ENV] = str(decoy)
            with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
                self._run(["update", "--start-date", "20260101", "--end-date", "20260102"], "update_data", 0)
        finally:
            os.close(decoy)
            os.close(holder)

    def test_non_production_raw_dir_and_union_output_are_unaffected(self) -> None:
        other = self.root / "elsewhere" / "raw"
        other.mkdir(parents=True)
        code = self._run(
            ["update", "--start-date", "20260101", "--end-date", "20260102",
             "--raw-dir", str(other), "--union-output", str(other / "share_float_complete.parquet")],
            "update_data", 0,
        )
        self.assertEqual(code, 0)

    def test_production_union_output_is_refused_even_with_foreign_raw_dir(self) -> None:
        # --raw-dir elsewhere but --union-output under the production lake:
        # still a production write (the previously missed bypass).
        other = self.root / "elsewhere" / "raw"
        other.mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(
                ["update", "--start-date", "20260101", "--end-date", "20260102",
                 "--raw-dir", str(other),
                 "--union-output", str(self.root / "data" / "raw" / "share_float_complete" / "x.parquet")],
                "update_data", 0,
            )

    def test_production_raw_subdirectory_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cron runner|tushare_cron_update"):
            self._run(
                ["update", "--start-date", "20260101", "--end-date", "20260102",
                 "--raw-dir", str(self.root / "data" / "raw" / "daily")],
                "update_data", 0,
            )


class ShareFloatUndatedDuplicateInvariantTest(unittest.TestCase):
    """Production union invariants around undated share_float duplicates.

    The one-off historical repair tool (repair.py) ran on 2026-07-30 and was
    retired on 2026-08-02 (dev-phase ruling); the rule it applied is now a
    union-merge invariant (``common.undated_duplicate_mask``), pinned here.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.raw_dir = self.root / "raw"
        self.baseline = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_baseline(self, rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        common.write_parquet(
            self.baseline, frame, api_name="share_float_complete", params={},
            fields=list(frame.columns),
        )
        return frame

    @staticmethod
    def _row(**overrides) -> dict:
        row = {
            "ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
            "holder_name": "基金(有限合伙)", "share_type": "首发原始股",
            "float_share": 1000.0, "float_ratio": 1.0,
        }
        row.update(overrides)
        return row

    def test_union_keeps_undated_duplicates_out_by_construction(self) -> None:
        """A rescue fetch that returns a lot undated must not re-add it.

        The seven-field dedup cannot collapse an undated row against its dated
        twin (they differ in ann_date), so without this invariant the class the
        one-off repair removed could return through a supported CLI path and
        double the float supply again.
        """
        dated = self._row(float_share=32974.0)
        undated = self._row(ann_date=None, float_share=32974.0, float_ratio=0.9)
        distinct_lot = self._row(ann_date=None, float_share=555.0)
        frame = pd.DataFrame([dated, undated, distinct_lot])

        mask = common.undated_duplicate_mask(frame)

        self.assertEqual(mask.tolist(), [False, True, False])
        kept = frame.loc[~mask]
        self.assertEqual(float(kept.loc[kept["float_share"] == 32974.0, "float_share"].sum()), 32974.0)
        # A differing lot size is never collapsed: it may be a real second lot.
        self.assertIn(555.0, kept["float_share"].tolist())

    def test_undated_active_row_cannot_supersede_a_dated_baseline_announcement(self) -> None:
        # The float_date query path returns rows without ann_date. Identity-based
        # supersession would otherwise delete the DATED baseline generation and
        # downgrade its availability to the float_date fallback.
        existing = pd.DataFrame([self._row(ann_date="20260630", float_share=1000.0)])
        common.write_parquet(self.baseline, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        source = self.raw_dir / "share_float" / "date=20270712.parquet"
        undated = pd.DataFrame([self._row(ann_date=None, float_share=2000.0)])
        common.write_parquet(source, undated, api_name="share_float", params={},
                             fields=list(undated.columns))
        args = argparse.Namespace(
            union_output=str(self.baseline), ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20270712", float_end_date="20270712", skip_float_date_union=False,
        )
        report: dict = {}

        with patch.object(download, "share_float_union_files", return_value=[(source, "float_date_existing")]):
            download.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(self.baseline)
        self.assertEqual(report["union"]["superseded_redated_groups"], 0)
        self.assertEqual(sorted(union["float_share"]), [1000.0, 2000.0])
        self.assertIn("20260630", union["ann_date"].astype(str).tolist())

    def test_undated_active_row_loses_the_newest_dating_comparison(self) -> None:
        existing = pd.DataFrame([self._row(ann_date="20260630", float_share=1000.0)])
        common.write_parquet(self.baseline, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        dated_path = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        dated = pd.DataFrame([self._row(ann_date="20260709", float_share=1000.0)])
        common.write_parquet(dated_path, dated, api_name="share_float", params={},
                             fields=list(dated.columns))
        undated_path = self.raw_dir / "share_float" / "date=20270712.parquet"
        undated = pd.DataFrame([self._row(ann_date=None, float_share=1000.0)])
        common.write_parquet(undated_path, undated, api_name="share_float", params={},
                             fields=list(undated.columns))
        args = argparse.Namespace(
            union_output=str(self.baseline), ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20270712", float_end_date="20270712", skip_float_date_union=False,
        )
        report: dict = {}

        with patch.object(
            download, "share_float_union_files",
            return_value=[(dated_path, "ann_date"), (undated_path, "float_date_existing")],
        ):
            download.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(self.baseline)
        self.assertEqual(report["union"]["active_stale_restatement_rows_dropped"], 1)
        self.assertEqual(union["ann_date"].astype(str).tolist(), ["20260709"])

    def test_audit_reports_the_undated_copy_as_the_stale_row(self) -> None:
        rows = pd.DataFrame([
            self._row(ann_date=None),
            self._row(),
            self._row(ts_code="000009.SZ", ann_date="20260630"),
            self._row(ts_code="000009.SZ", ann_date="20260709"),
        ])
        common.write_parquet(self.baseline, rows, api_name="share_float_complete", params={},
                             fields=list(rows.columns))
        findings: list[dict] = []

        audit.audit_share_float_complete_union(
            self.raw_dir,
            lambda severity, check, message, details=None: findings.append(
                {"check": check, "details": details or {}}
            ),
        )

        details = next(f for f in findings if f["check"] == "share_float_complete_union")["details"]
        self.assertEqual(details["cross_ann_date_identity_groups"], 2)
        self.assertEqual(details["stale_redated_rows"], 2)
        self.assertEqual(details["undated_duplicate_rows"], 1)

    def test_vectorized_date_normalization_matches_the_scalar_rule(self) -> None:
        values = [None, float("nan"), "", "20200102", "2020-01-02", "20200102123", "2020010", "x", 20200102]
        vector = common.normalized_date_keys(pd.Series(values, dtype=object)).tolist()

        self.assertEqual(vector, [common.normalize_date_key(value) for value in values])
        self.assertEqual(vector[0], "")
        self.assertLess(vector[0], "20200102")  # undated sorts BELOW every real date


class TuShareClientTest(unittest.TestCase):
    """Relay (中转) access is the only supported TuShare transport."""

    def test_sdk_uses_default_relay(self) -> None:
        pro = SimpleNamespace(_DataApi__http_url=None, query=lambda *args, **kwargs: pd.DataFrame())
        calls = []

        def pro_api(token, timeout):
            calls.append((token, timeout))
            return pro

        fake = SimpleNamespace(pro_api=pro_api)
        with patch.dict(sys.modules, {"tushare": fake}), patch.dict(os.environ, {}, clear=True):
            common.TuShareClient("secret", timeout=17)
        self.assertEqual(pro._DataApi__http_url, common.DEFAULT_TUSHARE_RELAY_URL)
        self.assertEqual(calls, [("secret", 17)])

    def test_environment_overrides_relay(self) -> None:
        pro = SimpleNamespace(_DataApi__http_url=None, query=lambda *args, **kwargs: pd.DataFrame())
        fake = SimpleNamespace(pro_api=lambda token, timeout: pro)
        with patch.dict(sys.modules, {"tushare": fake}), patch.dict(os.environ, {"TUSHARE_RELAY_URL": "https://relay.example/"}):
            common.TuShareClient("secret")
        self.assertEqual(pro._DataApi__http_url, "https://relay.example")

    def test_timeout_must_be_positive_integer(self) -> None:
        for timeout in (0, -1, True, 1.5):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "positive integer"):
                common.TuShareClient("secret", timeout=timeout)  # type: ignore[arg-type]

    def test_missing_sdk_fails_with_project_dependency_instruction(self) -> None:
        with (
            patch.dict(sys.modules, {"tushare": None}),
            self.assertRaisesRegex(RuntimeError, "project's declared dependencies") as caught,
        ):
            common.TuShareClient("secret")
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_download_cli_keeps_real_sdk_timeout_option(self) -> None:
        args = download.build_parser().parse_args(["download", "--tier", "daily", "--timeout-seconds", "23"])
        self.assertEqual(args.timeout_seconds, 23)


class FundamentalAuditTest(unittest.TestCase):
    """``disclosure_date``'s whole contract IS its key set, so the business
    payload gate must exempt it instead of erroring on correct data."""

    @staticmethod
    def _run_dataset(raw: Path, dataset: str, partition: str) -> list[dict]:
        findings: list[dict] = []

        def add(severity, check, message, details=None):
            findings.append(
                {
                    "severity": severity,
                    "check": check,
                    "message": message,
                    "details": details or {},
                }
            )

        audit.audit_fundamental_dataset(
            raw,
            common.FUNDAMENTAL_SPECS[dataset],
            {partition},
            "period" if dataset == "disclosure_date" else "ts_code",
            add,
        )
        return findings

    def test_disclosure_date_exact_key_schema_is_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = raw / "disclosure_date" / "period=20240630.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20240630",
                        "ann_date": "20240401",
                        "pre_date": "20240420",
                        "actual_date": "20240419",
                    }
                ]
            ).to_parquet(path, index=False)
            findings = self._run_dataset(raw, "disclosure_date", "20240630")

        payload = next(item for item in findings if item["check"] == "disclosure_date_payload")
        self.assertEqual(payload["severity"], "info")
        self.assertFalse(payload["details"]["business_payload_empty"])

    def test_other_fundamental_key_only_schema_remains_hollow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = raw / "fina_audit" / "ts_code=000001.SZ.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [{"ts_code": "000001.SZ", "ann_date": "20240401", "end_date": "20231231"}]
            ).to_parquet(path, index=False)
            findings = self._run_dataset(raw, "fina_audit", "000001.SZ")

        payload = next(item for item in findings if item["check"] == "fina_audit_payload")
        self.assertEqual(payload["severity"], "error")
        self.assertTrue(payload["details"]["business_payload_empty"])

    def test_disclosure_date_missing_key_column_remains_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            path = raw / "disclosure_date" / "period=20240630.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "end_date": "20240630",
                        "ann_date": "20240401",
                        "pre_date": "20240420",
                    }
                ]
            ).to_parquet(path, index=False)
            findings = self._run_dataset(raw, "disclosure_date", "20240630")

        keys = next(item for item in findings if item["check"] == "disclosure_date_keys")
        self.assertEqual(keys["severity"], "error")
        self.assertEqual(keys["details"]["missing_key_column_files"], 1)


class UuidCommitIdentityTest(unittest.TestCase):
    """One UUID identity shared by a Parquet footer and its sidecar."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sidecar_uses_footer_write_id(self) -> None:
        path = self.root / "daily" / "trade_date=20240102.parquet"
        meta = tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"], "close": [10.0]}),
            api_name="daily",
            params={},
            fields=["trade_date", "close"],
        )
        self.assertTrue(tushare_io.committed_partition_intact(path))
        self.assertEqual(tushare_io.parquet_write_id(path), meta["write_id"])
        self.assertEqual(meta["row_count"], 1)

    def test_commit_integrity_rejects_interrupted_mixed_and_wrong_count_pairs(self) -> None:
        path = self.root / "daily" / "trade_date=20240102.parquet"
        tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"], "close": [10.0]}),
            api_name="daily",
            params={},
            fields=["trade_date", "close"],
        )
        sidecar = path.with_suffix(".parquet.meta.json")
        original_sidecar = sidecar.read_text(encoding="utf-8")

        sidecar.unlink()
        self.assertFalse(tushare_io.committed_partition_intact(path))
        sidecar.write_text(original_sidecar, encoding="utf-8")

        other = self.root / "daily" / "trade_date=20240103.parquet"
        tushare_io.write_parquet(
            other,
            pd.DataFrame({"trade_date": ["20240103"], "close": [11.0]}),
            api_name="daily",
            params={},
            fields=["trade_date", "close"],
        )
        sidecar.write_text(
            other.with_suffix(".parquet.meta.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.assertFalse(tushare_io.committed_partition_intact(path))

        wrong_count = json.loads(original_sidecar)
        wrong_count["row_count"] = 2
        sidecar.write_text(json.dumps(wrong_count), encoding="utf-8")
        self.assertFalse(tushare_io.committed_partition_intact(path))

    def test_write_parquet_has_no_metadata_field_name_policy(self) -> None:
        path = self.root / "daily" / "trade_date=20240102.parquet"
        meta = tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"]}),
            api_name="daily",
            params={"filters": [{"opaque_marker": "source-value"}]},
            fields=["trade_date"],
            extra_metadata={"source_context": {"opaque_marker": "landing-value"}},
        )
        self.assertTrue(tushare_io.committed_partition_intact(path))
        self.assertEqual(meta["params"]["filters"][0]["opaque_marker"], "source-value")
        self.assertEqual(meta["source_context"]["opaque_marker"], "landing-value")

    def test_legacy_migration_reads_one_file_below_hive_style_directory(self) -> None:
        path = self.root / "country=CN" / "month=202601.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"country": ["CN"], "value": [1]}).to_parquet(path, index=False)
        path.with_suffix(".parquet.meta.json").write_text(
            json.dumps({"row_count": 1, "legacy_marker": "legacy"}),
            encoding="utf-8",
        )
        tushare_io.migrate_partition_identity(path)
        migrated = tushare_io.parquet_meta(path)
        self.assertTrue(tushare_io.committed_partition_intact(path))
        self.assertNotIn("legacy_marker", migrated)
        self.assertEqual(pd.read_parquet(path)["country"].tolist(), ["CN"])

    def test_legacy_migration_only_copies_supported_sidecar_schema(self) -> None:
        path = self.root / "stk_auction" / "trade_date=20240102.parquet"
        meta = tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"], "price": [10.0]}),
            api_name="stk_auction",
            params={},
            fields=["trade_date", "price"],
        )
        sidecar = path.with_suffix(".parquet.meta.json")
        meta["unknown_top_level"] = {"nested": "old-value"}
        meta["availability"] = {
            "available_at": "2024-01-02T09:29:00+08:00",
            "rule": "observed:test",
            "unknown_availability_field": {"nested": "old-value"},
        }
        sidecar.write_text(json.dumps(meta), encoding="utf-8")

        self.assertTrue(tushare_io.committed_partition_intact(path))
        tushare_io.migrate_partition_identity(path)

        migrated = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertTrue(tushare_io.committed_partition_intact(path))
        self.assertNotIn("unknown_top_level", migrated)
        self.assertEqual(
            migrated["availability"],
            {
                "available_at": "2024-01-02T09:29:00+08:00",
                "rule": "observed:test",
            },
        )
        self.assertEqual(pd.read_parquet(path)["price"].tolist(), [10.0])


class FullPortContractTest(unittest.TestCase):
    def test_schedule_retains_full_job_set_and_uuid_migration(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8"))
        self.assertEqual(len(config["jobs"]), 29)
        self.assertEqual(
            config["jobs"]["manual_commit_identity_migration"]["operation"],
            "commit_identity_migration",
        )

    def test_commit_identity_job_uses_explicit_migration_script(self) -> None:
        context = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=Path("."),
            python="python",
            job_name="migration",
            job={"operation": "commit_identity_migration"},
            start_date="20240101",
            end_date="20240102",
            timezone_name="Asia/Shanghai",
        )
        self.assertEqual(
            cron_update.build_job_commands(context),
            [["python", "scripts/data/migrate_commit_identity.py", "--raw-dir", "raw"]],
        )

    def test_generation_resume_uses_explicit_command_identity(self) -> None:
        transaction = {
            "job": "daily",
            "start_date": "20240101",
            "end_date": "20240102",
            "commands": [["python", "download.py", "daily"]],
            "config_identity": {"job": {"operation": "update"}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            first = cron_update.begin_raw_generation_update(raw, transaction)
            cron_update.mark_raw_generation_dirty(raw, first, error="interrupted")
            with self.assertRaisesRegex(RuntimeError, "rerun the original job"):
                cron_update.begin_raw_generation_update(raw, {**transaction, "job": "other"})
            resumed = cron_update.begin_raw_generation_update(raw, transaction)
            self.assertEqual(resumed["transaction_id"], first["transaction_id"])
            cron_update.mark_raw_generation_dirty(raw, resumed, error="interrupted again")
            output = io.StringIO()
            with redirect_stdout(output):
                superseded = cron_update.begin_raw_generation_update(
                    raw, {**transaction, "commands": [["python", "download.py", "daily", "--force"]]}
                )
        self.assertNotEqual(superseded["transaction_id"], first["transaction_id"])
        self.assertEqual(json.loads(output.getvalue())["note"], "raw_generation_dirty_superseded")

    def test_snapshot_inventory_retains_full_dataset_coverage(self) -> None:
        root = Path(__file__).resolve().parents[2]
        inventory = json.loads(
            (root / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"]
        self.assertIn("margin", inventory["events.parquet"])
        self.assertIn("cn_cpi", inventory["macro.parquet"])
        self.assertIn("income_vip", inventory["fundamentals.parquet"])
        self.assertIn(
            "source_write_id",
            inventory["fundamentals.parquet"]["income_vip"],
        )

    def test_every_downloaded_reference_dataset_is_registered(self) -> None:
        # interfaces[] is the operational fact source for datasets (the console
        # reads it for dataset labels): a table the downloader writes but the
        # registry omits carries no documented refresh or availability contract.
        root = Path(__file__).resolve().parents[2]
        schedule = json.loads(
            (root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8")
        )
        scheduled = {item["dataset"] for item in schedule["interfaces"]}
        self.assertEqual(set(common.REFERENCE_DATASETS) - scheduled, set())

    def test_active_macro_registries_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schedule = json.loads(
            (root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8")
        )
        inventory = json.loads(
            (root / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"]
        scheduled_datasets = {item["dataset"] for item in schedule["interfaces"]}

        self.assertEqual(set(common.MACRO_DATASETS), set(common.MACRO_SPECS))
        self.assertTrue(set(common.MACRO_DATASETS).issubset(scheduled_datasets))
        # The committed inventory covers every SELECTABLE macro dataset, not
        # just the default scope: the unit registry is fail-closed, so a
        # dataset an experiment can opt into must resolve column by column.
        self.assertEqual(
            set(SELECTABLE_DATASETS["macro"]),
            set(inventory["macro.parquet"]),
        )
        self.assertTrue(set(SELECTABLE_DATASETS["macro"]).issubset(common.MACRO_SPECS))
