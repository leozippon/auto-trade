"""Per-quarter sub-window breakdown of one replay window.

A Fold decides on one Validation window, and the audited folds show quarter to
quarter reversal is the normal case: a whole-window number alone cannot say
whether an edge persisted or one stretch of market carried it. Every
Validation / Test / Held-out result therefore carries the same metrics per
calendar quarter, and these tests pin the properties that make the rows
readable next to the whole-window figures — they chain back to
``total_return``, their turnover and trade counts sum to the whole-window
ones, and an absent benchmark stays absent instead of becoming zero.
"""

from __future__ import annotations

import unittest

from autotrade.environment.replay.stats import (
    ReplayResult,
    attach_sub_window_benchmark,
    compute_return_stats,
    sub_window_stats,
)

# Two calendar quarters, opened at 1000 and closed at 900. 2021Q4 round-trips
# to its opening equity (up 10%, back down); 2022Q1 spikes then falls under it.
_CURVE = (
    {"trade_date": "20211201", "initial_equity": 1000.0, "equity": 1100.0, "cash": 0.0},
    {"trade_date": "20211231", "initial_equity": 1000.0, "equity": 1000.0, "cash": 0.0},
    {"trade_date": "20220104", "initial_equity": 1000.0, "equity": 1200.0, "cash": 0.0},
    {"trade_date": "20220331", "initial_equity": 1000.0, "equity": 900.0, "cash": 0.0},
)
_EXECUTIONS = (
    {
        "status": "filled",
        "action": "buy",
        "price": 10.0,
        "quantity": 100,
        "matched_at": "2021-12-01T09:30:00+08:00",
    },
    {
        "status": "filled",
        "action": "sell",
        "price": 11.0,
        "quantity": 100,
        "realized_pnl": 100.0,
        "matched_at": "2022-01-04T09:30:00+08:00",
    },
    {
        "status": "rejected",
        "action": "buy",
        "price": 9.0,
        "quantity": 100,
        "reason": "insufficient_cash",
        "matched_at": "2022-03-31T09:30:00+08:00",
    },
)


class SubWindowStatsTest(unittest.TestCase):
    def rows(self):
        return sub_window_stats(_CURVE, _EXECUTIONS, initial=1000.0)

    def test_one_row_per_calendar_quarter_with_its_covered_span(self) -> None:
        rows = self.rows()
        self.assertEqual([row["label"] for row in rows], ["2021Q4", "2022Q1"])
        self.assertEqual(
            [(row["start"], row["end"], row["trade_days"]) for row in rows],
            [("20211201", "20211231", 2), ("20220104", "20220331", 2)],
        )
        self.assertEqual([row["kind"] for row in rows], ["quarter", "quarter"])

    def test_a_quarter_the_window_does_not_span_is_marked_partial(self) -> None:
        rows = self.rows()
        # The replay starts on 1 December, inside 2021Q4.
        self.assertTrue(rows[0]["partial"])
        # It ends on the last calendar day of 2022Q1.
        self.assertFalse(rows[1]["partial"])

    def test_partial_is_measured_against_the_requested_window(self) -> None:
        # 2022Q1 opens on a holiday, so its first trading day is the 4th: a
        # replay of the whole quarter covers it end to end all the same.
        whole = sub_window_stats(
            _CURVE[2:], _EXECUTIONS, initial=1000.0, start="20220101", end="20220331"
        )
        self.assertEqual([row["label"] for row in whole], ["2022Q1"])
        self.assertFalse(whole[0]["partial"])
        # A window that opened after the quarter began, or closed before it
        # ended, covers only part of it.
        for start, end in (("20220201", "20220331"), ("20220101", "20220330")):
            rows = sub_window_stats(
                _CURVE[2:], _EXECUTIONS, initial=1000.0, start=start, end=end
            )
            self.assertTrue(rows[0]["partial"], (start, end))

    def test_an_unparseable_requested_window_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be YYYYMMDD"):
            sub_window_stats(_CURVE, (), initial=1000.0, start="2022-01-01")

    def test_returns_chain_back_to_the_whole_window_return(self) -> None:
        rows = self.rows()
        # 2021Q4 opens at the initial equity and closes back on it; 2022Q1
        # opens there and closes 10% below.
        self.assertAlmostEqual(rows[0]["return"], 0.0, places=6)
        self.assertAlmostEqual(rows[1]["return"], -0.1, places=6)
        compounded = 1.0
        for row in rows:
            compounded *= 1.0 + float(row["return"])
        whole = compute_return_stats(
            ReplayResult(_CURVE, _EXECUTIONS, ("20211201",), ())
        )
        self.assertAlmostEqual(compounded - 1.0, whole["total_return"], places=6)

    def test_drawdown_is_measured_inside_the_quarter(self) -> None:
        rows = self.rows()
        # 1100 -> 1000 within 2021Q4; 1200 -> 900 within 2022Q1. A quarter's
        # peak is not carried in from the previous one.
        self.assertAlmostEqual(rows[0]["max_drawdown"], 100 / 1100, places=6)
        self.assertAlmostEqual(rows[1]["max_drawdown"], 0.25, places=6)

    def test_sharpe_follows_the_sign_of_the_quarter_and_needs_two_days(self) -> None:
        rows = self.rows()
        self.assertGreater(rows[0]["sharpe"], 0.0)
        self.assertLess(rows[1]["sharpe"], 0.0)
        single = sub_window_stats(
            ({"trade_date": "20220104", "initial_equity": 100.0, "equity": 120.0},),
            (),
            initial=100.0,
        )
        # One daily return has no dispersion to annualize.
        self.assertEqual(single[0]["sharpe"], 0.0)
        self.assertAlmostEqual(single[0]["return"], 0.2, places=6)

    def test_turnover_and_trade_count_sum_to_the_whole_window(self) -> None:
        rows = self.rows()
        # Rejected fills are not traded notional, and only realized exits count
        # as trades — the same rule the whole-window figures use.
        self.assertAlmostEqual(rows[0]["turnover"], 1.0, places=6)
        self.assertAlmostEqual(rows[1]["turnover"], 1.1, places=6)
        self.assertEqual([row["trade_count"] for row in rows], [0, 1])
        whole = compute_return_stats(
            ReplayResult(_CURVE, _EXECUTIONS, ("20211201",), ())
        )
        self.assertAlmostEqual(
            sum(float(row["turnover"]) for row in rows), whole["turnover"], places=6
        )
        self.assertEqual(
            sum(int(row["trade_count"]) for row in rows), whole["trade_count"]
        )

    def test_an_empty_replay_has_no_rows(self) -> None:
        self.assertEqual(sub_window_stats((), (), initial=1000.0), [])


class SubWindowBenchmarkTest(unittest.TestCase):
    def summary(self) -> dict[str, object]:
        return compute_return_stats(
            ReplayResult(_CURVE, _EXECUTIONS, ("20211201",), ())
        )

    def test_the_replay_alone_reports_no_benchmark(self) -> None:
        rows = self.summary()["sub_windows"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["benchmark_return"])
            self.assertIsNone(row["excess_return"])

    def test_excess_is_the_quarter_return_minus_the_compounded_benchmark(self) -> None:
        summary = self.summary()
        attach_sub_window_benchmark(
            summary,
            {
                "benchmark_daily": [
                    ["20211201", 0.05],
                    ["20211231", -0.05],
                    ["20220104", 0.10],
                    ["20220331", -0.10],
                ]
            },
        )
        rows = summary["sub_windows"]
        self.assertAlmostEqual(rows[0]["benchmark_return"], 1.05 * 0.95 - 1.0, places=6)
        self.assertAlmostEqual(rows[1]["benchmark_return"], 1.10 * 0.90 - 1.0, places=6)
        self.assertAlmostEqual(rows[0]["excess_return"], 0.0 - (1.05 * 0.95 - 1.0), places=6)
        self.assertAlmostEqual(rows[1]["excess_return"], -0.1 - (1.10 * 0.90 - 1.0), places=6)

    def test_a_slot_without_a_usable_benchmark_reports_nothing_not_zero(self) -> None:
        for sidecar in ({}, {"benchmark_daily": []}, {"benchmark_daily": "broken"}):
            summary = self.summary()
            attach_sub_window_benchmark(summary, sidecar)
            for row in summary["sub_windows"]:
                self.assertIsNone(row["benchmark_return"], sidecar)
                self.assertIsNone(row["excess_return"], sidecar)

    def test_the_block_rides_in_the_persisted_replay_record(self) -> None:
        record = ReplayResult(_CURVE, _EXECUTIONS, ("20211201",), ()).to_record()
        self.assertEqual(
            [row["label"] for row in record["stats"]["sub_windows"]],
            ["2021Q4", "2022Q1"],
        )


if __name__ == "__main__":
    unittest.main()
