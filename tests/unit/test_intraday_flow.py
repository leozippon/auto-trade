"""Derived intraday_flow dataset: the tick-rule contract and its PIT stamp."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from autotrade.environment.data.contracts import CN_TZ, INTRADAY_FLOW_CONTRACT
from autotrade.environment.data.intraday_flow import (
    BARS_PER_DAY,
    INTRADAY_FLOW_COLUMNS,
    MINUTE_COLUMNS,
    aggregate_intraday_flow,
)
from autotrade.environment.data.pit import to_cn_timestamps

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FLOW = (
    REPO_ROOT
    / "configs/workspace_refs/order_flow_ranker_20260917/starter/lib/flow.py"
)

TRADE_DATE = "20260818"
# The 241-bar A-share grid: 09:30 auction, 09:31-11:30, 13:01-15:00.
SESSION = (
    [("09", 30)]
    + [("09", minute) for minute in range(31, 60)]
    + [("10", minute) for minute in range(60)]
    + [("11", minute) for minute in range(31)]
    + [("13", minute) for minute in range(1, 60)]
    + [("14", minute) for minute in range(60)]
    + [("15", 0)]
)


def _session_frame(ts_code: str, closes, volumes) -> pd.DataFrame:
    """One synthetic stock-day on the real minute grid."""
    assert len(SESSION) == BARS_PER_DAY == len(closes) == len(volumes)
    return pd.DataFrame(
        {
            "ts_code": ts_code,
            "trade_date": TRADE_DATE,
            "trade_time": [f"2026-08-18 {hour}:{minute:02d}:00" for hour, minute in SESSION],
            "close": [float(value) for value in closes],
            "vol": [float(value) for value in volumes],
            # Turnover consistent with the bar: price * volume.
            "amount": [float(close) * float(vol) for close, vol in zip(closes, volumes)],
        }
    )


class IntradayFlowAggregationTest(unittest.TestCase):
    def test_full_session_signs_every_minute_except_the_auction_and_lunch(self):
        # A strictly rising session: every contiguous minute is an up-tick, so
        # the imbalance saturates at +1 and the day yields 239 signed minutes
        # (241 bars minus the 09:30 auction, which has no predecessor, minus
        # the 11:30 -> 13:01 lunch gap).
        closes = [10.0 + 0.01 * index for index in range(BARS_PER_DAY)]
        volumes = [1000.0] * BARS_PER_DAY
        flow = aggregate_intraday_flow(_session_frame("000001.SZ", closes, volumes))
        self.assertEqual(list(flow.columns), list(INTRADAY_FLOW_COLUMNS))
        self.assertEqual(len(flow), 1)
        row = flow.iloc[0]
        self.assertEqual(int(row["nret"]), 239)
        # 239 signed bars out of 241, all positive, equal volume each.
        self.assertAlmostEqual(float(row["ofi_1d"]), 239 / 241)
        self.assertEqual(float(row["zero_share"]), 0.0)
        self.assertFalse(bool(row["sealed_limit"]))

    def test_auction_volume_counts_in_the_denominator_but_carries_no_sign(self):
        # The 09:30 bar is the opening auction: it has no predecessor, so it can
        # never contribute a signed return, yet its volume is genuinely traded
        # and stays in the denominator. Making the auction bar huge must
        # therefore push |ofi_1d| towards zero, never change its sign.
        closes = [10.0 + 0.01 * index for index in range(BARS_PER_DAY)]
        small = [1.0] + [1.0] * (BARS_PER_DAY - 1)
        large = [239.0] + [1.0] * (BARS_PER_DAY - 1)
        base = aggregate_intraday_flow(_session_frame("000001.SZ", closes, small))
        diluted = aggregate_intraday_flow(_session_frame("000001.SZ", closes, large))
        self.assertAlmostEqual(float(base.iloc[0]["ofi_1d"]), 239 / 241)
        self.assertAlmostEqual(float(diluted.iloc[0]["ofi_1d"]), 239 / 479)

    def test_the_lunch_gap_produces_no_return(self):
        # 11:30 -> 13:01 is not a contiguous minute pair. Here the session is
        # flat on both sides and the day's ONLY price move is across lunch, so
        # a correct aggregation reports exactly zero; treating the gap as a
        # normal minute would report +1000/241000.
        closes = [10.0] * 121 + [11.0] * 120
        flow = aggregate_intraday_flow(
            _session_frame("000001.SZ", closes, [1000.0] * BARS_PER_DAY)
        )
        row = flow.iloc[0]
        self.assertEqual(float(row["ofi_1d"]), 0.0)
        self.assertEqual(int(row["nret"]), 239)

    def test_no_trade_minutes_are_neutral_by_construction(self):
        # A no-trade minute repeats the previous close with vol = 0, so it
        # weighs nothing in either sum: it is neutral without being filtered,
        # and it must not be read as a missing bar.
        closes = [10.0 + 0.01 * index for index in range(BARS_PER_DAY)]
        volumes = [1000.0] * 191 + [0.0] * 50
        flow = aggregate_intraday_flow(_session_frame("000001.SZ", closes, volumes))
        row = flow.iloc[0]
        # 191 traded bars; neither the auction nor the post-lunch bar is signed.
        self.assertAlmostEqual(float(row["ofi_1d"]), 189 / 191)
        self.assertEqual(int(row["nret"]), 239)
        self.assertAlmostEqual(float(row["zero_share"]), 50 / 241)
        self.assertFalse(bool(row["sealed_limit"]))

    def test_sealed_limit_is_flagged_and_never_filtered_away(self):
        # A sealed limit board trades in the auction and then stands still:
        # the diagnostics must travel with the row so research can apply its
        # own threshold, and the row itself must survive.
        closes = [11.0] * BARS_PER_DAY
        volumes = [500_000.0] + [0.0] * (BARS_PER_DAY - 1)
        flow = aggregate_intraday_flow(_session_frame("600000.SH", closes, volumes))
        row = flow.iloc[0]
        self.assertEqual(len(flow), 1)
        self.assertAlmostEqual(float(row["zero_share"]), 240 / 241)
        self.assertTrue(bool(row["sealed_limit"]))
        self.assertEqual(float(row["ofi_1d"]), 0.0)

    def test_untraded_stock_day_and_beijing_names_produce_no_row(self):
        # A whole-day halt has no defined imbalance (zero denominator) and no
        # `daily` bar either; .BJ carries after-hours bars that break both the
        # 241-bar denominator and the lunch-only contiguity rule.
        closes = [11.0] * BARS_PER_DAY
        halted = _session_frame("600000.SH", closes, [0.0] * BARS_PER_DAY)
        beijing = _session_frame("920627.BJ", closes, [1000.0] * BARS_PER_DAY)
        self.assertTrue(aggregate_intraday_flow(halted).empty)
        self.assertTrue(aggregate_intraday_flow(beijing).empty)
        self.assertTrue(
            aggregate_intraday_flow(pd.concat([halted, beijing], ignore_index=True)).empty
        )

    def test_rows_are_stamped_by_the_shared_close_contract(self):
        closes = [10.0 + 0.01 * index for index in range(BARS_PER_DAY)]
        flow = aggregate_intraday_flow(
            _session_frame("000001.SZ", closes, [1000.0] * BARS_PER_DAY)
        )
        expected = INTRADAY_FLOW_CONTRACT.available_at(pd.Timestamp(TRADE_DATE).date())
        self.assertEqual(expected.hour, 17)
        self.assertEqual(expected.minute, 30)
        stamped = to_cn_timestamps(flow["available_at"]).iloc[0]
        self.assertEqual(stamped, pd.Timestamp(expected).tz_convert(CN_TZ))
        self.assertEqual(
            flow["available_at_rule"].iloc[0], "contract_1730_from:trade_date"
        )

    def test_grouping_is_per_stock_day_regardless_of_call_shape(self):
        # A partition is one trade date, but the aggregate must not depend on
        # how many dates or codes share a call.
        rising = [10.0 + 0.01 * index for index in range(BARS_PER_DAY)]
        falling = list(reversed(rising))
        first = _session_frame("000001.SZ", rising, [1000.0] * BARS_PER_DAY)
        second = _session_frame("600000.SH", falling, [1000.0] * BARS_PER_DAY)
        together = aggregate_intraday_flow(pd.concat([second, first], ignore_index=True))
        apart = pd.concat(
            [aggregate_intraday_flow(first), aggregate_intraday_flow(second)],
            ignore_index=True,
        ).sort_values("ts_code", ignore_index=True)
        pd.testing.assert_frame_equal(together, apart)
        self.assertAlmostEqual(float(together.iloc[1]["ofi_1d"]), -239 / 241)

    @unittest.skipUnless(
        REFERENCE_FLOW.exists(),
        "reference research pack retired; the byte-exactness oracle is gone",
    )
    def test_matches_the_reference_study_aggregation_bit_for_bit(self):
        # The dataset exists to replace an in-strategy minute pass, so its
        # per-stock-day values must be the SAME numbers that pass computed.
        # `lib/flow.py` is the frozen reference implementation of the probe.
        spec = importlib.util.spec_from_file_location("reference_flow", REFERENCE_FLOW)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)

        rng = np.random.default_rng(20260917)
        frames = []
        for index, code in enumerate(("000001.SZ", "300750.SZ", "600000.SH", "920627.BJ")):
            steps = rng.choice([-0.01, 0.0, 0.01], size=BARS_PER_DAY)
            closes = np.maximum(10.0 + np.cumsum(steps), 0.01)
            volumes = rng.integers(0, 5000, size=BARS_PER_DAY).astype(float)
            volumes[index] = 0.0
            frames.append(_session_frame(code, closes, volumes))
        minutes = pd.concat(frames, ignore_index=True)

        mine = aggregate_intraday_flow(minutes)
        theirs = reference.aggregate_minutes(minutes[list(MINUTE_COLUMNS)])
        merged = theirs.merge(mine, on=["ts_code", "trade_date"], how="left")
        self.assertEqual(len(merged), len(theirs))
        self.assertTrue(merged["ofi_1d"].notna().all())
        for reference_column, column in (
            ("ofi", "ofi_1d"),
            ("ofi_amt", "ofi_amt_1d"),
            ("nret_x", "nret_y"),
            ("zero_share_x", "zero_share_y"),
        ):
            with self.subTest(column=column):
                self.assertTrue(
                    np.array_equal(
                        merged[reference_column].to_numpy(dtype="float64"),
                        merged[column].to_numpy(dtype="float64"),
                    )
                )


if __name__ == "__main__":
    unittest.main()
