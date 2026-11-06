"""Per research year sub-window breakdown of one replay window.

A whole-window number alone cannot say whether an edge persisted or one
stretch of market carried it, so every replay result carries the same metrics
per research year of its window, the years the experiment's own geometry lays out
(``pipelines/calendar.py ResearchGeometry.research_years``). These tests pin the
properties that make the rows readable next to the whole-window figures — they
use the years the replay was requested over, they chain back to
``total_return``, their turnover and trade counts sum to the whole-window ones,
and an absent benchmark stays absent instead of becoming zero.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autotrade.environment.artifacts import ArtifactRevision
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.contracts import DEFAULT_BENCHMARK_INDEX
from autotrade.environment.replay.stats import (
    ReplayResult,
    attach_sub_window_benchmark,
    compute_return_stats,
    sub_window_stats,
    window_activity,
)
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import ReplaySpan, SnapshotBundle, research_span
from autotrade.pipelines.local_backend import LocalDailyEvaluationBackend

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
        later = tuple(row for row in _CURVE if row["trade_date"] >= "20220701")
        executions = tuple(
            order for order in _EXECUTIONS if order["matched_at"] >= "2022-07"
        )
        whole = sub_window_stats(
            later, executions, initial=1000.0, start="20220701", end="20230630"
        )
        self.assertEqual([row["label"] for row in whole], ["202207-202306"])
        self.assertFalse(whole[0]["partial"])
        # A window that opened after the year began, or closed before it ended,
        # covers only part of it.
        for start, end in (("20220801", "20230630"), ("20220701", "20230629")):
            rows = sub_window_stats(
                later, executions, initial=1000.0, start=start, end=end
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

    def test_each_year_carries_its_random_baseline_compounded_only_where_one_was_drawn(self) -> None:
        """The holder's year reads three returns side by side: the book's, the
        index's and zero skill's on the same skeleton. A replay without a random baseline
        reports none rather than zero."""

        benchmark = [["20220601", 0.05], ["20220630", -0.05], ["20220704", 0.10], ["20230630", -0.10]]
        strategy = [["20220601", 0.10], ["20220630", -1 / 11], ["20220704", 0.20], ["20230630", -0.25]]
        baselines = [["20220601", 0.02], ["20220630", 0.01], ["20220704", -0.03], ["20230630", 0.04]]
        summary = self.summary()
        attach_sub_window_benchmark(
            summary,
            {"benchmark_daily": benchmark, "strategy_daily": strategy, "baseline_daily": baselines},
        )
        rows = summary["sub_windows"]
        self.assertAlmostEqual(rows[0]["baseline_return"], 1.02 * 1.01 - 1.0, places=6)
        self.assertAlmostEqual(rows[1]["baseline_return"], 0.97 * 1.04 - 1.0, places=6)
        bare = self.summary()
        attach_sub_window_benchmark(bare, {"benchmark_daily": benchmark})
        self.assertEqual([row.get("baseline_return") for row in bare["sub_windows"]], [None, None])

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


# A ten-year experiment laid out from 1 August, the geometry that made the hardcoded
# July boundary wrong: 20150801..20250731 is exactly ten whole years, and its
# last one ends on 31 July instead of leaving a 23-day July tail.
AUGUST_GEOMETRY = ResearchGeometry(
    research_start="20150801",
    research_end="20250731",
    forward_end="20260731",
    heldout_end="20261031",
)


def _year_windows(geometry: ResearchGeometry) -> tuple[tuple[str, str], ...]:
    """The ``(start, end)`` bounds of a geometry's research years."""

    return tuple((slot.start, slot.end) for slot in geometry.research_years)


def _trading_days(start: str, end: str) -> list[str]:
    return [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range(start, end)]


def _path_curve(
    days: list[str], *, opening: float = 1000.0, daily: float = 0.0004
) -> tuple[dict[str, object], ...]:
    """One end-of-day equity path over ``days``, opened at ``opening``."""

    equity = opening
    rows = []
    for day in days:
        equity *= 1.0 + daily
        rows.append(
            {
                "trade_date": day,
                "initial_equity": opening,
                "equity": equity,
                "cash": equity * 0.5,
            }
        )
    return tuple(rows)


def _stamp(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}T09:30:00+08:00"


def _round_trips(
    windows: tuple[tuple[str, str], ...], days: list[str]
) -> tuple[dict[str, object], ...]:
    """One buy and one realised exit per year, on the first and last day it holds."""

    orders: list[dict[str, object]] = []
    for start, end in windows:
        first = next(day for day in days if start <= day <= end)
        last = max(day for day in days if start <= day <= end)
        orders.append(
            {
                "status": "filled",
                "action": "buy",
                "price": 10.0,
                "quantity": 100,
                "matched_at": _stamp(first),
            }
        )
        orders.append(
            {
                "status": "filled",
                "action": "sell",
                "price": 11.0,
                "quantity": 100,
                "realized_pnl": 100.0,
                "matched_at": _stamp(last),
            }
        )
    return tuple(orders)


class SubWindowResearchYearTest(unittest.TestCase):
    """The rows are the experiment's own research years, not July-June calendar blocks."""

    def setUp(self) -> None:
        self.windows = _year_windows(AUGUST_GEOMETRY)
        self.days = _trading_days(
            AUGUST_GEOMETRY.research_start, AUGUST_GEOMETRY.research_end
        )

    def rows(
        self, windows: tuple[tuple[str, str], ...], days: list[str]
    ) -> list[dict[str, object]]:
        return sub_window_stats(
            _path_curve(days),
            _round_trips(windows, days),
            initial=1000.0,
            start=windows[0][0],
            end=windows[-1][1],
            year_windows=windows,
        )

    def test_one_row_per_research_year_of_a_month_start_geometry(self) -> None:
        rows = self.rows(self.windows, self.days)
        self.assertEqual(
            [row["label"] for row in rows],
            [f"{start[:6]}-{end[:6]}" for start, end in self.windows],
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual([row["kind"] for row in rows], ["research_year"] * 10)
        # The window spans every year end to end, so none is partial, and the
        # last row ends on the geometry's own last day: no July tail block.
        self.assertEqual([row["partial"] for row in rows], [False] * 10)
        self.assertEqual((rows[0]["start"], rows[-1]["end"]), ("20150803", "20250731"))
        self.assertTrue(all(int(row["trade_days"]) > 200 for row in rows))

    def test_a_span_of_the_last_four_years_is_four_whole_rows(self) -> None:
        windows = self.windows[6:]
        days = [day for day in self.days if day >= windows[0][0]]
        rows = self.rows(windows, days)
        self.assertEqual(
            [row["label"] for row in rows],
            [f"{start[:6]}-{end[:6]}" for start, end in windows],
        )
        # Each row holds every trading day of its own year: the span is not
        # re-split into calendar blocks plus a leftover tail.
        self.assertEqual(
            [row["trade_days"] for row in rows],
            [
                len([day for day in days if start <= day <= end])
                for start, end in windows
            ],
        )
        self.assertFalse(any(row["partial"] for row in rows))

    def test_a_window_cut_inside_one_year_keeps_that_years_bounds(self) -> None:
        days = [day for day in self.days if "20160301" <= day <= "20160630"]
        rows = sub_window_stats(
            _path_curve(days),
            (),
            initial=1000.0,
            start="20160301",
            end="20160630",
            year_windows=self.windows,
        )
        [row] = rows
        # A probe inside the first research year reports that year, marked
        # partial, rather than a year re-anchored on the probe's own dates.
        self.assertEqual(row["label"], "201508-201607")
        self.assertEqual(row["kind"], "research_year")
        self.assertTrue(row["partial"])
        self.assertEqual((row["start"], row["end"]), (days[0], days[-1]))

    def test_a_leap_day_belongs_to_the_year_that_contains_it(self) -> None:
        geometry = ResearchGeometry("20160201", "20200131", "20210131", "20210430")
        windows = _year_windows(geometry)
        self.assertEqual(
            windows,
            (
                ("20160201", "20170131"),
                ("20170201", "20180131"),
                ("20180201", "20190131"),
                ("20190201", "20200131"),
            ),
        )
        days = ["20160226", "20160229", "20160301", "20170131", "20170201", "20200131"]
        rows = sub_window_stats(
            _path_curve(days),
            (),
            initial=1000.0,
            start=geometry.research_start,
            end=geometry.research_end,
            year_windows=windows,
        )
        # The empty third year reports no row; the others hold the days that
        # fall inside them, leap day included.
        self.assertEqual(
            [row["label"] for row in rows],
            ["201602-201701", "201702-201801", "201902-202001"],
        )
        self.assertEqual(
            [row["trade_days"] for row in rows],
            [4, 1, 1],
        )
        self.assertEqual([row["partial"] for row in rows], [False, False, False])

    def test_the_rows_compound_and_sum_back_to_the_whole_window(self) -> None:
        curve = _path_curve(self.days)
        executions = _round_trips(self.windows, self.days)
        whole = compute_return_stats(
            ReplayResult(curve, executions, (), ()),
            start=AUGUST_GEOMETRY.research_start,
            end=AUGUST_GEOMETRY.research_end,
            year_windows=self.windows,
        )
        rows = whole["sub_windows"]
        compounded = 1.0
        for row in rows:
            compounded *= 1.0 + float(row["return"])
        # Row returns are rounded to six decimals, and ten of them compound
        # the rounding up: the chain is checked at that tolerance, not exactly.
        self.assertAlmostEqual(compounded - 1.0, whole["total_return"], places=4)
        self.assertAlmostEqual(
            sum(float(row["turnover"]) for row in rows), whole["turnover"], places=5
        )
        self.assertEqual(
            sum(int(row["trade_count"]) for row in rows), whole["trade_count"]
        )

    def test_the_years_ride_in_the_persisted_replay_record(self) -> None:
        result = ReplayResult(
            _path_curve(self.days),
            _round_trips(self.windows, self.days),
            (),
            (),
        )
        record = result.to_record(
            start=AUGUST_GEOMETRY.research_start,
            end=AUGUST_GEOMETRY.research_end,
            year_windows=self.windows,
        )
        self.assertEqual(len(record["stats"]["sub_windows"]), 10)
        self.assertEqual(record["stats"]["sub_windows"][-1]["label"], "202408-202507")

    def test_explicit_july_june_years_read_as_the_calendar_fallback(self) -> None:
        """A July-June geometry keeps the rows it always had; only the kind says
        whether they are research years or a calendar fallback."""

        fallback = sub_window_stats(_CURVE, _EXECUTIONS, initial=1000.0)
        explicit = sub_window_stats(
            _CURVE,
            _EXECUTIONS,
            initial=1000.0,
            year_windows=(("20210701", "20220630"), ("20220701", "20230630")),
        )
        self.assertEqual(
            [{key: value for key, value in row.items() if key != "kind"} for row in explicit],
            [{key: value for key, value in row.items() if key != "kind"} for row in fallback],
        )
        self.assertEqual([row["kind"] for row in fallback], ["july_june_year"] * 2)
        self.assertEqual([row["kind"] for row in explicit], ["research_year"] * 2)

    def test_a_day_outside_the_declared_years_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the requested years"):
            sub_window_stats(
                ({"trade_date": "20220704", "equity": 100.0},),
                (),
                initial=1000.0,
                year_windows=(("20220801", "20230630"),),
            )

    def test_a_malformed_year_is_refused(self) -> None:
        curve = ({"trade_date": "20220704", "equity": 100.0},)
        with self.assertRaisesRegex(ValueError, "two ordered YYYYMMDD dates"):
            sub_window_stats(
                curve, (), initial=1000.0, year_windows=(("20230101", "20220701"),)
            )
        with self.assertRaisesRegex(ValueError, "must be YYYYMMDD"):
            sub_window_stats(
                curve, (), initial=1000.0, year_windows=(("2022-08-01", "20230630"),)
            )


IDLE_STRATEGY = """def generate_orders(context):
    return []
"""


class SubWindowRequestTest(unittest.TestCase):
    """The years travel from the span request through the backend into stats.

    The statistics cannot see the geometry; the only thing that can tell them
    what years to divide a replay by is the request the backend is handed.
    """

    GEOMETRY = ResearchGeometry(
        research_start="20150801",
        research_end="20170731",
        forward_end="20180731",
        heldout_end="20181031",
    )

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="sub_window_stats_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        days = _trading_days(self.GEOMETRY.research_start, self.GEOMETRY.research_end)
        daily = self.root / "daily.parquet"
        pd.DataFrame(
            {
                "trade_date": days,
                "symbol": ["000001.SZ"] * len(days),
                "open": [10.0] * len(days),
                "close": [10.5] * len(days),
                "pre_close": [10.5] * len(days),
            }
        ).to_parquet(daily, index=False)
        strategy = self.root / "strategy"
        strategy.mkdir()
        (strategy / "main.py").write_text(IDLE_STRATEGY, encoding="utf-8")
        self.revision = ArtifactRevision("r1", strategy, None)
        snapshot = SnapshotBundle("snap", str(daily), str(daily))
        self.years = tuple(
            ReplaySpan(slot.label, "valid", slot.start, slot.end, snapshot)
            for slot in self.GEOMETRY.research_years
        )
        self.backend = LocalDailyEvaluationBackend(
            daily,
            self.root / "results",
            execution_mode="trusted",
            benchmark_index=DEFAULT_BENCHMARK_INDEX,
        )

    def evaluate(self, **kwargs: object) -> list[dict[str, object]]:
        span = research_span(self.years, "full")
        result = self.backend.evaluate(
            span.request(
                self.revision,
                schedule=StrategySchedule("day", "09:00"),
                broker_profile=BrokerProfile(initial_cash=100_000),
            ),
            **kwargs,  # type: ignore[arg-type]
        )
        record = json.loads(Path(result.result_ref).read_text(encoding="utf-8"))
        return record["stats"]["sub_windows"]

    def test_the_research_years_reach_the_stats_through_the_request(self) -> None:
        rows = self.evaluate()
        self.assertEqual(
            [row["label"] for row in rows],
            [
                f"{slot.start[:6]}-{slot.end[:6]}"
                for slot in self.GEOMETRY.research_years
            ],
        )
        self.assertEqual([row["kind"] for row in rows], ["research_year"] * 2)
        self.assertEqual([row["partial"] for row in rows], [False, False])

    def test_a_truncated_probe_keeps_the_year_it_was_cut_from(self) -> None:
        second = self.GEOMETRY.research_years[1]
        rows = self.evaluate(max_days=5, start_day="20160901")
        [row] = rows
        self.assertEqual(row["label"], f"{second.start[:6]}-{second.end[:6]}")
        self.assertTrue(row["partial"])
        self.assertEqual(row["start"], "20160901")
        self.assertLessEqual(row["end"], second.end)


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
