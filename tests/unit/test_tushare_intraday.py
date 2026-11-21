"""The one-minute bar layer (data_sources/tushare/intraday.py)."""

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autotrade.data_sources.tushare import audit, common, intraday, specs
from autotrade.data_sources.tushare import client as tushare_client
from autotrade.data_sources.tushare import io as tushare_io

from .support.tushare_fixture import EmptyClient, LakeTestCase


class MinuteUpdateTest(LakeTestCase):
    def _write_daily_universe(self, trade_date="20200102"):
        path = self.raw_dir / "daily" / f"trade_date={trade_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"trade_date": trade_date, "ts_code": "000001.SZ"},
            {"trade_date": trade_date, "ts_code": "000002.SZ"},
        ]).to_parquet(path, index=False)

    def test_update_intraday_by_date_refuses_zero_row_write_for_nonempty_universe(self):
        self._write_trade_cal()
        self._write_daily_universe()
        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20200102",
            end_date="20200102",
            output_dataset=specs.STK_MINS_BY_DATE_DATASET,
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
            force_from=None,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        output = self.raw_dir / specs.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet"
        with patch.object(intraday, "load_token", return_value="token"), patch.object(intraday, "TuShareClient", return_value=EmptyClient()):
            with self.assertRaisesRegex(RuntimeError, "refusing to write zero-row intraday"):
                intraday.update_intraday_by_date(args)
        self.assertFalse(output.exists())

    def _minute_update_args(self, start_date, end_date, max_retries=2):
        return argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date=start_date,
            end_date=end_date,
            output_dataset=specs.STK_MINS_BY_DATE_DATASET,
            expected_codes_source="daily",
            codes=None,
            max_codes=None,
            min_rows_per_day=0,
            allow_missing_codes=0,
            allow_validation_warnings=True,
            max_retries=max_retries,
            retry_delay_seconds=0,
            page_limit=None,
            min_interval_seconds=0,
            timeout_seconds=1,
            force=False,
            force_from=None,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

    def _write_minute_calendar_and_universe(self, trade_dates, codes):
        path = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2020.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"cal_date": day, "is_open": "1"} for day in trade_dates]).to_parquet(path, index=False)
        for day in trade_dates:
            daily = self.raw_dir / "daily" / f"trade_date={day}.parquet"
            daily.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"trade_date": day, "ts_code": code} for code in codes]).to_parquet(daily, index=False)

    def test_update_intraday_by_date_batches_codes_and_retries_unanswered_ones(self):
        """The vendor's daily stk_mins quota counts requests, so one request
        carries a batch of codes; a code the batch did not answer is retried
        without refetching the rest of its batch."""
        codes = [f"{number:06d}.SZ" for number in range(1, 31)]
        self._write_minute_calendar_and_universe(["20200102"], codes)
        silent_once = {"000030.SZ"}
        requests: list[list[str]] = []

        class BatchMinuteClient:
            def query(self, api_name, params=None, fields="", retries=5):
                requested = str(params["ts_code"]).split(",")
                requests.append(requested)
                items = []
                for code in requested:
                    if code in silent_once:
                        silent_once.discard(code)
                        continue
                    for bar in ("2020-01-02 09:30:00", "2020-01-02 15:00:00"):
                        items.append([code, bar, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0])
                return tushare_client.ApiResult(fields.split(","), items)

        with patch.object(intraday, "load_token", return_value="token"), patch.object(intraday, "TuShareClient", return_value=BatchMinuteClient()):
            self.assertEqual(intraday.update_intraday_by_date(self._minute_update_args("20200102", "20200102")), 0)

        self.assertEqual([len(batch) for batch in requests], [specs.STK_MINS_BATCH_CODES, 5, 1])
        self.assertEqual(requests[-1], ["000030.SZ"])
        written = pd.read_parquet(self.raw_dir / specs.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet")
        self.assertEqual(sorted(written["ts_code"].unique()), codes)
        self.assertEqual(len(written), 60)

    def test_update_intraday_by_date_commits_written_days_when_the_quota_runs_out(self):
        """The quota answer holds until the vendor's next day: the run stops on
        it at once with the not-ready exit, keeping the days already written
        and discarding the half-collected day."""
        codes = ["000001.SZ", "000002.SZ"]
        self._write_minute_calendar_and_universe(["20200102", "20200103"], codes)
        calls: list[str] = []

        class QuotaMinuteClient:
            def query(self, api_name, params=None, fields="", retries=5):
                day = str(params["start_date"])[:10]
                calls.append(day)
                if day == "2020-01-03":
                    raise RuntimeError("stk_mins failed after 5 attempts: 您今日stk_min额定额度已用完，请明日再试。")
                items = [[code, f"{day} 09:30:00", 1.0, 1.0, 1.0, 1.0, 100.0, 100.0] for code in str(params["ts_code"]).split(",")]
                return tushare_client.ApiResult(fields.split(","), items)

        output = io.StringIO()
        with patch.object(intraday, "load_token", return_value="token"), patch.object(intraday, "TuShareClient", return_value=QuotaMinuteClient()), redirect_stdout(output):
            code = intraday.update_intraday_by_date(self._minute_update_args("20200102", "20200103", max_retries=3))

        self.assertEqual(code, common.MUTATED_NOT_READY_RETRY_EXIT_CODE)
        self.assertEqual(calls, ["2020-01-02", "2020-01-03"])
        minute_dir = self.raw_dir / specs.STK_MINS_BY_DATE_DATASET
        self.assertTrue((minute_dir / "trade_date=20200102.parquet").exists())
        self.assertFalse((minute_dir / "trade_date=20200103.parquet").exists())
        self.assertIn("stk_mins_daily_quota_exhausted", output.getvalue())

    def test_minute_expected_universe_uses_existing_minute_store_when_present(self):
        self._write_daily_universe()
        minute_path = self.raw_dir / specs.STK_MINS_BY_DATE_DATASET / "trade_date=20200102.parquet"
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
        tushare_io.write_parquet(
            minute_path,
            minute_rows,
            api_name=specs.STK_MINS_API_NAME,
            params={},
            fields=list(minute_rows.columns),
        )

        codes = intraday.intraday_expected_codes_for_day(
            self.raw_dir,
            argparse.Namespace(expected_codes_source="minute", output_dataset=specs.STK_MINS_BY_DATE_DATASET, codes=None, max_codes=None),
            "20200102",
        )

        self.assertEqual(codes, {"000001.SZ"})


class IntradayByDateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmp.name) / "raw"
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
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        self.assertEqual(intraday.compact_intraday_by_date(compact_args), 0)

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
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        self.assertEqual(audit.audit_intraday_by_date(audit_args), 0)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "ok")


if __name__ == "__main__":
    unittest.main()
