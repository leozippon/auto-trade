"""The stk_auction capture and recheck (data_sources/tushare/auction.py)."""

import argparse
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

from autotrade.data_sources.tushare import auction, common, specs
from autotrade.data_sources.tushare import client as tushare_client
from autotrade.data_sources.tushare import io as tushare_io

from .tushare_fixture import LakeTestCase


class AuctionTest(LakeTestCase):
    def test_capture_open_auction_waits_for_stable_complete_frame(self):
        self._write_trade_cal("20260713")
        previous = self.raw_dir / "stk_auction" / "trade_date=20260710.parquet"
        previous.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ts_code": "000001.SZ"}, {"ts_code": "600000.SH"}]).to_parquet(previous, index=False)
        fields = specs.DAILY_SPECS["stk_auction"].fields.split(",")

        def result(items):
            return tushare_client.ApiResult(fields, items)

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
            patch.object(auction, "load_token", return_value="token"),
            patch.object(auction, "TuShareClient"),
            patch.object(auction, "query_paged", side_effect=responses) as query,
            patch.object(auction.time, "sleep", return_value=None),
        ):
            self.assertEqual(auction.capture_open_auction(args), 0)

        self.assertEqual(query.call_count, 5)
        target = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"
        self.assertEqual(len(pd.read_parquet(target)), 2)
        availability = tushare_io.parquet_meta(target)["availability"]
        self.assertEqual(availability["rule"], "observed:cn_open_auction_capture")
        self.assertEqual(availability["row_count"], 2)

        # A later strict reconciliation may return the same keyed rows in a
        # different API order. Canonical persistence keeps the first landing.
        reversed_result = result(list(reversed(full)))
        with (
            patch.object(auction, "load_token", return_value="token"),
            patch.object(auction, "TuShareClient"),
            patch.object(auction, "query_paged", side_effect=[(reversed_result, 1), (reversed_result, 1)]),
            patch.object(auction.time, "sleep", return_value=None),
        ):
            self.assertEqual(auction.capture_open_auction(args), 0)
        self.assertEqual(tushare_io.parquet_meta(target)["availability"], availability)
        self.assertEqual(pd.read_parquet(target)["ts_code"].tolist(), ["000001.SZ", "600000.SH"])

    def test_capture_open_auction_timeout_does_not_replace_partition(self):
        self._write_trade_cal("20260713")
        target = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"
        original = pd.DataFrame(
            [{"trade_date": "20260713", "ts_code": "000001.SZ", "price": 9.9}]
        )
        tushare_io.write_parquet(
            target,
            original,
            api_name="stk_auction",
            params={},
            fields=list(original.columns),
        )
        original_bytes = target.read_bytes()
        fields = specs.DAILY_SPECS["stk_auction"].fields.split(",")
        partial = tushare_client.ApiResult(
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
            patch.object(auction, "load_token", return_value="token"),
            patch.object(auction, "TuShareClient"),
            patch.object(auction, "query_paged", return_value=(partial, 1)),
        ):
            self.assertEqual(
                auction.capture_open_auction(args),
                common.NO_MUTATION_RETRY_EXIT_CODE,
            )

        self.assertEqual(target.read_bytes(), original_bytes)

    def test_capture_open_auction_polls_on_fixed_start_times(self):
        self._write_trade_cal("20260713")
        fields = specs.DAILY_SPECS["stk_auction"].fields.split(",")
        items = [
            ["000001.SZ", "20260713", 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
        ]
        result = tushare_client.ApiResult(fields, items)
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
            patch.object(auction, "load_token", return_value="token"),
            patch.object(auction, "TuShareClient"),
            patch.object(auction, "query_paged", side_effect=clock.query),
            patch.object(auction.time, "monotonic", side_effect=clock.monotonic),
            patch.object(auction.time, "sleep", side_effect=clock.sleep),
        ):
            self.assertEqual(auction.capture_open_auction(args), 0)

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
        errors = auction._validate_auction_capture(
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
        self.assertEqual(auction._validate_auction_capture(valid, "20260713", min_rows=3), [])

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
        errors = auction._validate_auction_capture(invalid, "20260713", min_rows=8)
        self.assertIn("invalid_vol_rows=2", errors)
        self.assertIn("invalid_amount_rows=2", errors)
        self.assertIn("inconsistent_trade_rows=1", errors)
        self.assertIn("unrecoverable_trade_price_rows=1", errors)
        self.assertIn("hidden_no_trade_price_rows=1", errors)
        self.assertIn("inconsistent_trade_price_rows=1", errors)

    def test_auction_capture_row_floor_is_percentage_of_stock_rows(self):
        # Percentage-only: an absolute prev-minus-N bound dominated the ratio
        # for full-market partitions and rejected legitimate provider-side
        # coverage narrowing. The ratio counts A-share rows only: from 20260911
        # the vendor dropped all 1,879 fund/ETF/LOF/convertible rows at once
        # (7,943 -> 6,065 rows) while every A-share code stayed, and a
        # whole-frame ratio refused every capture from then on. 85% of the
        # stock rows still blocks a truncated capture.
        previous = self.raw_dir / "stk_auction" / "trade_date=20260910.parquet"
        previous.parent.mkdir(parents=True, exist_ok=True)
        stocks = [f"{600000 + index}.SH" for index in range(3000)] + [f"{index:06d}.SZ" for index in range(1, 2001)]
        funds = [f"{510000 + index}.SH" for index in range(1000)] + [f"{150000 + index}.SZ" for index in range(900)]
        pd.DataFrame({"ts_code": stocks + funds}).to_parquet(previous, index=False)

        minimum = auction._auction_capture_min_rows(
            self.raw_dir, "20260911", floor=1000, ratio=0.85,
        )
        self.assertEqual(minimum, 4250)  # floor(5000 stock rows * 0.85)
        row = {"trade_date": "20260911", "vol": 0.0, "price": None, "amount": 0.0, "pre_close": 9.9, "turnover_rate": 0.0, "volume_ratio": 1.0, "float_share": 1.0}
        without_funds = pd.DataFrame([{**row, "ts_code": code} for code in stocks])
        self.assertEqual(auction._validate_auction_capture(without_funds, "20260911", min_rows=minimum), [])
        truncated = pd.DataFrame([{**row, "ts_code": code} for code in stocks[:4000] + funds])
        self.assertIn(
            "stock_rows=4000 below_min_rows=4250",
            auction._validate_auction_capture(truncated, "20260911", min_rows=minimum),
        )
        # No previous partition: the absolute floor holds.
        self.assertEqual(
            auction._auction_capture_min_rows(self.raw_dir, "20260701", floor=1000, ratio=0.85),
            1000,
        )

    def test_stk_auction_recheck_single_shot_paths(self):
        self._write_trade_cal("20260713")
        fields = specs.DAILY_SPECS["stk_auction"].fields.split(",")
        full = [
            ["000001.SZ", "20260713", 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
            ["600000.SH", "20260713", 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
        ]

        def result(items, tag):
            return tushare_client.ApiResult(fields, items)

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260713",
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
                patch.object(auction, "load_token", return_value="token"),
                patch.object(auction, "TuShareClient"),
                patch.object(auction, "AUCTION_CAPTURE_MIN_ROWS_FLOOR", 2),
                patch.object(auction, "query_paged", **response),
                redirect_stdout(output),
            ):
                code = auction.recheck_stk_auction(args)
            return code, output.getvalue()

        # Catch-up: no partition yet (the whole pre-open window failed); one
        # valid read publishes with observed availability.
        code, out = run(return_value=(result(full, "a"), 1))
        self.assertEqual(code, 0)
        self.assertIn("recheck_published", out)
        availability = tushare_io.parquet_meta(target)["availability"]
        self.assertEqual(availability["landing_job"], "cn_evening_auction_backfill")
        first_bytes = target.read_bytes()

        # Unchanged content (even reordered): no rewrite; the partition bytes
        # and the first observed availability both survive.
        code, out = run(return_value=(result(list(reversed(full)), "b"), 1))
        self.assertEqual(code, 0)
        self.assertIn("recheck_unchanged", out)
        self.assertEqual(target.read_bytes(), first_bytes)
        self.assertEqual(tushare_io.parquet_meta(target)["availability"], availability)

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
            tushare_io.parquet_meta(target)["availability"]["available_at"],
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
        fields = specs.DAILY_SPECS["stk_auction"].fields.split(",")

        def day_rows(day):
            return [
                ["000001.SZ", day, 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
                ["600000.SH", day, 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
            ]

        # 20260709 already captured as an intact commit; 20260710 missing.
        existing = pd.DataFrame(day_rows("20260709"), columns=fields)
        target_dir = self.raw_dir / "stk_auction"
        tushare_io.write_parquet(
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
            return tushare_client.ApiResult(fields, day_rows(day)), 1

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260709",
            end_date="20260713",
            page_limit=10000,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
            landing_job="cn_evening_auction_backfill",
        )
        with (
            patch.object(auction, "load_token", return_value="token"),
            patch.object(auction, "TuShareClient"),
            patch.object(auction, "AUCTION_CAPTURE_MIN_ROWS_FLOOR", 2),
            patch.object(auction, "query_paged", side_effect=fake_query),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(auction.recheck_stk_auction(args), 0)
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
        fields = specs.DAILY_SPECS["stk_auction"].fields.split(",")

        def day_rows(day):
            return [
                ["000001.SZ", day, 1000.0, 10.0, 10000.0, 9.9, 0.1, 1.0, 100000.0],
                ["600000.SH", day, 2000.0, 8.0, 16000.0, 7.9, 0.2, 1.1, 200000.0],
            ]

        target_dir = self.raw_dir / "stk_auction"
        target_dir.mkdir(parents=True, exist_ok=True)
        torn = target_dir / "trade_date=20260709.parquet"
        pd.DataFrame(day_rows("20260709"), columns=fields).to_parquet(torn, index=False)
        self.assertFalse(tushare_io.committed_partition_intact(torn))

        queried: list[str] = []

        def fake_query(client, api_name, params, fields_, page_limit):
            day = params["trade_date"]
            queried.append(day)
            return tushare_client.ApiResult(fields, day_rows(day)), 1

        args = argparse.Namespace(
            raw_dir=str(self.raw_dir),
            start_date="20260709",
            end_date="20260713",
            page_limit=10000,
            revision_ledger=str(self.root / "revision_events.jsonl"),
            min_interval_seconds=0.0,
            timeout_seconds=1.0,
            landing_job="cn_evening_auction_backfill",
        )
        with (
            patch.object(auction, "load_token", return_value="token"),
            patch.object(auction, "TuShareClient"),
            patch.object(auction, "AUCTION_CAPTURE_MIN_ROWS_FLOOR", 2),
            patch.object(auction, "query_paged", side_effect=fake_query),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(auction.recheck_stk_auction(args), 0)
        self.assertEqual(queried, ["20260709", "20260710", "20260713"])
        self.assertTrue(tushare_io.committed_partition_intact(torn))


if __name__ == "__main__":
    unittest.main()
