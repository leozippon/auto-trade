"""Per July-June year sub-window breakdown of one replay window.

A whole-window number alone cannot say whether an edge persisted or one
stretch of market carried it, so every replay result carries the same metrics
per July-June year, the blocks research, forward and Held-out are laid out in.
These tests pin the properties that make the rows readable next to the
whole-window figures — they chain back to ``total_return``, their turnover and
trade counts sum to the whole-window ones, and an absent benchmark stays absent
instead of becoming zero.
"""

from __future__ import annotations

import unittest

from autotrade.environment.replay.stats import (
    ReplayResult,
    attach_sub_window_benchmark,
    compute_return_stats,
    sub_window_stats,
    window_activity,
)

# Two July-June years, opened at 1000 and closed at 900. The first
# round-trips to its opening equity (up 10%, back down); the second spikes then
# falls under it, and its first trading day is not its first calendar day.
_CURVE = (
    {"trade_date": "20220601", "initial_equity": 1000.0, "equity": 1100.0, "cash": 0.0},
    {"trade_date": "20220630", "initial_equity": 1000.0, "equity": 1000.0, "cash": 0.0},
    {"trade_date": "20220704", "initial_equity": 1000.0, "equity": 1200.0, "cash": 0.0},
    {"trade_date": "20230630", "initial_equity": 1000.0, "equity": 900.0, "cash": 0.0},
)
_EXECUTIONS = (
    {
        "status": "filled",
        "action": "buy",
        "price": 10.0,
        "quantity": 100,
        "matched_at": "2022-06-01T09:30:00+08:00",
    },
    {
        "status": "filled",
        "action": "sell",
        "price": 11.0,
        "quantity": 100,
        "realized_pnl": 100.0,
        "matched_at": "2022-07-04T09:30:00+08:00",
    },
    {
        "status": "rejected",
        "action": "buy",
        "price": 9.0,
        "quantity": 100,
        "reason": "insufficient_cash",
        "matched_at": "2023-06-30T09:30:00+08:00",
    },
)


class SubWindowStatsTest(unittest.TestCase):
    def rows(self):
        return sub_window_stats(_CURVE, _EXECUTIONS, initial=1000.0)

    def test_one_row_per_july_june_year_with_its_covered_span(self) -> None:
        rows = self.rows()
        self.assertEqual([row["label"] for row in rows], ["202107-202206", "202207-202306"])
        self.assertEqual(
            [(row["start"], row["end"], row["trade_days"]) for row in rows],
            [("20220601", "20220630", 2), ("20220704", "20230630", 2)],
        )
        self.assertEqual([row["kind"] for row in rows], ["july_june_year"] * 2)

    def test_a_year_the_window_does_not_span_is_marked_partial(self) -> None:
        rows = self.rows()
        # The replay starts on 1 June, inside the 2021-22 year.
        self.assertTrue(rows[0]["partial"])
        # It ends on the last calendar day of the 2022-23 year.
        self.assertFalse(rows[1]["partial"])

    def test_partial_is_measured_against_the_requested_window(self) -> None:
        # The year's first trading day is the 4th: a replay of the whole year
        # covers it end to end all the same.
        whole = sub_window_stats(
            _CURVE[2:], _EXECUTIONS, initial=1000.0, start="20220701", end="20230630"
        )
        self.assertEqual([row["label"] for row in whole], ["202207-202306"])
        self.assertFalse(whole[0]["partial"])
        # A window that opened after the year began, or closed before it ended,
        # covers only part of it.
        for start, end in (("20220801", "20230630"), ("20220701", "20230629")):
            rows = sub_window_stats(
                _CURVE[2:], _EXECUTIONS, initial=1000.0, start=start, end=end
            )
            self.assertTrue(rows[0]["partial"], (start, end))

    def test_an_unparseable_requested_window_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be YYYYMMDD"):
            sub_window_stats(_CURVE, (), initial=1000.0, start="2022-07-01")

    def test_returns_chain_back_to_the_whole_window_return(self) -> None:
        rows = self.rows()
        # The first year opens at the initial equity and closes back on it; the
        # second opens there and closes 10% below.
        self.assertAlmostEqual(rows[0]["return"], 0.0, places=6)
        self.assertAlmostEqual(rows[1]["return"], -0.1, places=6)
        compounded = 1.0
        for row in rows:
            compounded *= 1.0 + float(row["return"])
        whole = compute_return_stats(
            ReplayResult(_CURVE, _EXECUTIONS, ("20220601",), ())
        )
        self.assertAlmostEqual(compounded - 1.0, whole["total_return"], places=6)

    def test_drawdown_is_measured_inside_the_year(self) -> None:
        rows = self.rows()
        # 1100 -> 1000 within the first year; 1200 -> 900 within the second. A
        # year's peak is not carried in from the previous one.
        self.assertAlmostEqual(rows[0]["max_drawdown"], 100 / 1100, places=6)
        self.assertAlmostEqual(rows[1]["max_drawdown"], 0.25, places=6)

    def test_sharpe_follows_the_sign_of_the_year_and_needs_two_days(self) -> None:
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
            ReplayResult(_CURVE, _EXECUTIONS, ("20220601",), ())
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
            ReplayResult(_CURVE, _EXECUTIONS, ("20220601",), ())
        )

    def test_the_replay_alone_reports_no_benchmark(self) -> None:
        rows = self.summary()["sub_windows"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["benchmark_return"])
            self.assertIsNone(row["excess_return"])

    def test_excess_is_the_year_return_minus_the_compounded_benchmark(self) -> None:
        summary = self.summary()
        attach_sub_window_benchmark(
            summary,
            {
                "benchmark_daily": [
                    ["20220601", 0.05],
                    ["20220630", -0.05],
                    ["20220704", 0.10],
                    ["20230630", -0.10],
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

    def test_a_single_year_window_agrees_with_the_whole_window(self) -> None:
        """The whole window is the one-bucket case, so the two must not be two
        implementations that can drift apart."""

        curve = tuple(row for row in _CURVE if row["trade_date"] >= "20220701")
        executions = tuple(
            order for order in _EXECUTIONS if order["matched_at"] >= "2022-07"
        )
        summary = compute_return_stats(
            ReplayResult(curve, executions, ("20220704",), ()),
            start="20220704",
            end="20230630",
        )
        [row] = summary["sub_windows"]
        self.assertEqual(row["label"], "202207-202306")
        self.assertAlmostEqual(row["return"], summary["total_return"], places=6)
        for key in ("sharpe", "max_drawdown"):
            self.assertAlmostEqual(row[key], summary[key], places=6, msg=key)

    def test_the_block_rides_in_the_persisted_replay_record(self) -> None:
        record = ReplayResult(_CURVE, _EXECUTIONS, ("20220601",), ()).to_record()
        self.assertEqual(
            [row["label"] for row in record["stats"]["sub_windows"]],
            ["202107-202206", "202207-202306"],
        )


class WindowActivityTest(unittest.TestCase):
    """One slice of a continuous replay read on its own terms."""

    CURVE = (
        {"trade_date": "20250627", "initial_equity": 1000.0, "equity": 1000.0, "cash": 1000.0},
        {"trade_date": "20250630", "initial_equity": 1000.0, "equity": 1200.0, "cash": 200.0},
        {"trade_date": "20250701", "initial_equity": 1000.0, "equity": 1100.0, "cash": 100.0},
        {"trade_date": "20250702", "initial_equity": 1000.0, "equity": 1150.0, "cash": 1150.0},
    )
    EXECUTIONS = (
        {"status": "filled", "action": "buy", "price": 10.0, "quantity": 100,
         "matched_at": "2025-06-30T09:30:00+08:00"},
        {"status": "filled", "action": "sell", "price": 12.0, "quantity": 100, "realized_pnl": 200.0,
         "matched_at": "2025-07-02T09:30:00+08:00"},
        {"status": "rejected", "action": "buy", "price": 9.0, "quantity": 100,
         "matched_at": "2025-07-02T09:30:00+08:00"},
    )

    def test_the_slice_is_measured_from_the_equity_it_opened_at(self) -> None:
        later = window_activity(self.CURVE, self.EXECUTIONS, start="20250701", end="20250731")
        # Opened at the previous day's close; only the fill inside the slice
        # is traded notional, and it is the realised exit.
        self.assertEqual(later["opening_equity"], 1200.0)
        self.assertAlmostEqual(later["turnover"], 1200.0 / 1200.0)
        self.assertEqual((later["trade_days"], later["round_trips"]), (2, 1))
        self.assertAlmostEqual(later["mean_gross"], (1000.0 / 1100.0 + 0.0) / 2)
        first = window_activity(self.CURVE, self.EXECUTIONS, start="20250601", end="20250630")
        # A slice that opens the replay opens at the initial equity.
        self.assertEqual(first["opening_equity"], 1000.0)
        self.assertAlmostEqual(first["turnover"], 1.0)
        self.assertEqual(first["round_trips"], 0)

    def test_a_slice_without_a_replay_day_is_not_measured(self) -> None:
        with self.assertRaisesRegex(ValueError, "no day in 20250801..20250831"):
            window_activity(self.CURVE, self.EXECUTIONS, start="20250801", end="20250831")


if __name__ == "__main__":
    unittest.main()
