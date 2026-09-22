"""Derived intraday_stats dataset: column definitions, PIT stamp, unit coverage, build."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from autotrade.data_sources.tushare import audit
from autotrade.environment.data.contracts import (
    CN_TZ,
    EVENT_DATASET_REFRESH_NODES,
    INTRADAY_FLOW_CONTRACT,
    INTRADAY_STATS_CONTRACT,
    event_dataset_visible_cutoff,
)
from autotrade.environment.data.intraday_flow import aggregate_intraday_flow
from autotrade.environment.data.intraday_stats import (
    INTRADAY_STATS_COLUMNS,
    aggregate_intraday_stats,
)
from autotrade.environment.data.snapshot import DEFAULT_DATASETS, SELECTABLE_DATASETS
from autotrade.environment.data.units import resolve_field

from .test_intraday_flow import SESSION, TRADE_DATE

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = [f"{hour}:{minute:02d}" for hour, minute in SESSION]


def _day(ts_code, closes, volumes, *, first_open=None, drop=(), extra=()):
    """One synthetic stock-day on the real 241-bar grid.

    Each bar opens at the previous close (the first at ``first_open``);
    ``drop`` removes bars by label and ``extra`` appends off-grid
    ``(label, close, vol)`` bars.
    """
    closes = [float(value) for value in closes]
    opens = [closes[0] if first_open is None else float(first_open), *closes[:-1]]
    rows = [
        (label, opened, close, float(vol))
        for label, opened, close, vol in zip(LABELS, opens, closes, volumes)
        if label not in drop
    ]
    rows += [(label, close, close, float(vol)) for label, close, vol in extra]
    return pd.DataFrame(
        {
            "ts_code": ts_code,
            "trade_date": TRADE_DATE,
            "trade_time": [f"2026-08-18 {label}:00" for label, *_ in rows],
            "open": [row[1] for row in rows],
            "close": [row[2] for row in rows],
            "vol": [row[3] for row in rows],
            # Turnover consistent with the bar: price * volume.
            "amount": [row[2] * row[3] for row in rows],
        }
    )


def _moments(returns, n):
    """Hand formulas over the day's nonzero 1-minute log returns, N returns in all."""
    rv = sum(r * r for r in returns)
    down = sum(r * r for r in returns if r < 0)
    return {
        "rv": rv,
        "rv_down_share": down / rv,
        "rsk": math.sqrt(n) * sum(r**3 for r in returns) / rv**1.5,
        "rku": n * sum(r**4 for r in returns) / rv**2,
        "sjv": (rv - down) - down,
    }


# Reference day: auction bar opens 9.9 and closes 10.0; +2% at 09:31; flat;
# the only post-lunch move is the 13:01 bar (10.2 -> 10.1); 15:00 closes 10.0.
# Bars: 09:30 (vol 100), 09:31-11:30 at 10.2 (120 bars), 13:01-14:59 at 10.1
# (119 bars), 15:00 at 10.0; every bar after the auction trades vol 10.
REFERENCE_CLOSES = [10.0] + [10.2] * 120 + [10.1] * 119 + [10.0]
REFERENCE_VOLUMES = [100.0] + [10.0] * 240
REFERENCE_RETURNS = [math.log(10.0 / 9.9), math.log(10.2 / 10.0), math.log(10.1 / 10.2), math.log(10.0 / 10.1)]


class IntradayStatsDefinitionTest(unittest.TestCase):
    def _row(self, frame: pd.DataFrame) -> pd.Series:
        stats = aggregate_intraday_stats(frame)
        self.assertEqual(list(stats.columns), list(INTRADAY_STATS_COLUMNS))
        self.assertEqual(len(stats), 1)
        return stats.iloc[0]

    def _assert_values(self, row: pd.Series, expected: dict[str, float]) -> None:
        for column, value in expected.items():
            with self.subTest(column=column):
                actual = float(row[column])
                self.assertTrue(math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-15), (actual, value))

    def test_hand_computed_session_pins_every_column(self):
        row = self._row(_day("000001.SZ", REFERENCE_CLOSES, REFERENCE_VOLUMES, first_open=9.9))
        vwap = (10.0 * 100 + 10.2 * 10 * 120 + 10.1 * 10 * 119 + 10.0 * 10) / 2500
        self._assert_values(
            row,
            {
                # N = 241: the auction bar's own move from the opening price
                # counts, and the lunch-spanning 11:30 -> 13:01 return is one
                # ordinary return.
                **_moments(REFERENCE_RETURNS, 241),
                "ret_open30": 10.2 / 9.9 - 1,
                "ret_mid": 10.1 / 10.2 - 1,
                "ret_close30": 10.0 / 10.1 - 1,
                "vwap_dev": 10.0 / vwap - 1,
                # 09:30 auction bar + 30 bars to 10:00; 30 bars 14:31-15:00.
                "vol_open30_share": 400 / 2500,
                "vol_close30_share": 300 / 2500,
            },
        )
        self.assertEqual(int(row["n_bars"]), 241)
        # The segment returns chain from the opening price to the close.
        chained = sum(math.log1p(float(row[c])) for c in ("ret_open30", "ret_mid", "ret_close30"))
        self.assertAlmostEqual(chained, math.log(10.0 / 9.9), places=12)

    def test_missing_bars_span_the_gap_and_boundaries_take_the_last_bar_before(self):
        # 09:45/10:00/14:30 are flat bars, 13:01 carries the post-lunch move;
        # dropping them moves no price (the 13:02 return spans the gap) and
        # both boundaries fall back to the bar just before. An off-grid 15:05
        # bar is ignored outright.
        frame = _day(
            "000001.SZ",
            REFERENCE_CLOSES,
            REFERENCE_VOLUMES,
            first_open=9.9,
            drop=("09:45", "10:00", "13:01", "14:30"),
            extra=(("15:05", 12.0, 1000.0),),
        )
        row = self._row(frame)
        vwap = (10.0 * 100 + 10.2 * 10 * 118 + 10.1 * 10 * 117 + 10.0 * 10) / 2460
        self._assert_values(
            row,
            {
                **_moments(REFERENCE_RETURNS, 237),
                "ret_open30": 10.2 / 9.9 - 1,
                "ret_mid": 10.1 / 10.2 - 1,
                "ret_close30": 10.0 / 10.1 - 1,
                "vwap_dev": 10.0 / vwap - 1,
                "vol_open30_share": 380 / 2460,
                "vol_close30_share": 300 / 2460,
            },
        )
        self.assertEqual(int(row["n_bars"]), 237)

    def test_limit_down_and_sealed_limit_up_days(self):
        # Opens 9.50 against a 10.00 previous close, falls 0.05 a bar to the
        # 9.00 limit by 09:40, then sits sealed there (vol 1 per bar).
        prices = [9.5] + [9.5 - 0.05 * k for k in range(1, 11)] + [9.0] * 230
        volumes = [1000.0] + [100.0] * 10 + [1.0] * 230
        row = self._row(_day("000002.SZ", prices, volumes))
        returns = [math.log(prices[k] / prices[k - 1]) for k in range(1, 11)]
        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
        self._assert_values(
            row,
            {
                **_moments(returns, 241),
                "ret_open30": 9.0 / 9.5 - 1,
                "ret_mid": 0.0,
                "ret_close30": 0.0,
                "vwap_dev": 9.0 / vwap - 1,
                "vol_open30_share": (1000 + 1000 + 20) / sum(volumes),
                "vol_close30_share": 30 / sum(volumes),
            },
        )
        self.assertEqual(float(row["rv_down_share"]), 1.0)
        self.assertLess(float(row["rsk"]), 0.0)

        # Sealed at the limit from the auction: the price never moves, so the
        # rv-normalised moments are undefined (null), never an invented 0.
        sealed = self._row(_day("600000.SH", [11.0] * 241, [5000.0] + [0.0] * 240))
        for column in ("rv", "sjv", "ret_open30", "ret_mid", "ret_close30", "vwap_dev", "vol_close30_share"):
            self.assertEqual(float(sealed[column]), 0.0, column)
        self.assertEqual(float(sealed["vol_open30_share"]), 1.0)
        for column in ("rv_down_share", "rsk", "rku"):
            self.assertTrue(math.isnan(float(sealed[column])), column)

    def test_path_starts_at_the_first_traded_bar(self):
        # No auction match: 09:30-09:34 only repeat the previous close (10.0)
        # with vol 0, and the first trade opens the 09:35 bar at 10.5. Starting
        # the path earlier would leak the overnight gap into rv as a jump.
        closes = [10.0] * 5 + [10.6] * 236
        opens_first_trade = _day("000003.SZ", closes, [0.0] * 5 + [100.0] + [10.0] * 235)
        opens_first_trade.loc[5, "open"] = 10.5
        row = self._row(opens_first_trade)
        self._assert_values(
            row,
            {
                **_moments([math.log(10.6 / 10.5)], 236),
                "ret_open30": 10.6 / 10.5 - 1,
                "ret_mid": 0.0,
                "ret_close30": 0.0,
            },
        )
        self.assertEqual(int(row["n_bars"]), 241)

    def test_rows_mirror_intraday_flow_and_do_not_depend_on_call_shape(self):
        rising = [10.0 + 0.01 * k for k in range(241)]
        first = _day("000001.SZ", REFERENCE_CLOSES, REFERENCE_VOLUMES, first_open=9.9)
        second = _day("600000.SH", list(reversed(rising)), [1000.0] * 241)
        halted = _day("600001.SH", [11.0] * 241, [0.0] * 241)
        beijing = _day("920627.BJ", rising, [1000.0] * 241, extra=(("15:05", 12.0, 10.0),))
        minutes = pd.concat([second, beijing, halted, first], ignore_index=True)
        stats = aggregate_intraday_stats(minutes)
        flow = aggregate_intraday_flow(minutes)
        # Same stock-days as intraday_flow: whole-day halts and .BJ are out.
        self.assertEqual(stats["ts_code"].tolist(), ["000001.SZ", "600000.SH"])
        pd.testing.assert_frame_equal(stats[["ts_code", "trade_date"]], flow[["ts_code", "trade_date"]])
        apart = pd.concat(
            [aggregate_intraday_stats(first), aggregate_intraday_stats(second)], ignore_index=True
        )
        pd.testing.assert_frame_equal(stats, apart)
        self.assertTrue(aggregate_intraday_stats(pd.concat([halted, beijing])).empty)

    def test_a_non_positive_price_fails_instead_of_publishing(self):
        frame = _day("000001.SZ", REFERENCE_CLOSES, REFERENCE_VOLUMES)
        frame.loc[100, "close"] = 0.0
        with self.assertRaisesRegex(ValueError, "non-positive"):
            aggregate_intraday_stats(frame)


class IntradayStatsContractTest(unittest.TestCase):
    def test_availability_is_the_intraday_flow_rule(self):
        self.assertEqual(INTRADAY_STATS_CONTRACT.rule, INTRADAY_FLOW_CONTRACT.rule)
        self.assertEqual(INTRADAY_STATS_CONTRACT.rule, "contract_1730_from:trade_date")
        day = date(2022, 1, 5)
        self.assertEqual(INTRADAY_STATS_CONTRACT.available_at(day), INTRADAY_FLOW_CONTRACT.available_at(day))
        minutes = _day("000001.SZ", REFERENCE_CLOSES, REFERENCE_VOLUMES)
        stamps = ["available_at", "available_at_rule"]
        pd.testing.assert_frame_equal(
            aggregate_intraday_stats(minutes)[stamps], aggregate_intraday_flow(minutes)[stamps]
        )
        # Same refresh node as intraday_flow (the events default): invisible
        # during day D's session, visible at the D+1 pre-open.
        self.assertNotIn("intraday_stats", EVENT_DATASET_REFRESH_NODES)
        stamp = INTRADAY_STATS_CONTRACT.available_at(day)
        for when, visible in ((datetime(2022, 1, 5, 14, 0), False), (datetime(2022, 1, 6, 8, 30), True)):
            cutoff = event_dataset_visible_cutoff("intraday_stats", when.replace(tzinfo=CN_TZ))
            self.assertEqual(stamp <= cutoff, visible, when)
            self.assertEqual(cutoff, event_dataset_visible_cutoff("intraday_flow", when.replace(tzinfo=CN_TZ)))

    def test_every_column_resolves_in_the_registry_inventory_and_nightly_audit(self):
        self.assertIn("intraday_stats", SELECTABLE_DATASETS["events"])
        self.assertNotIn("intraday_stats", DEFAULT_DATASETS["events"])
        inventory = json.loads(
            (REPO_ROOT / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"]["events.parquet"]["intraday_stats"]
        self.assertEqual(sorted(inventory), sorted({*INTRADAY_STATS_COLUMNS, "dataset"}))
        units = {column: resolve_field("events.parquet", "intraday_stats", column) for column in inventory}
        self.assertEqual(units["rv"]["source_unit"], "decimal_squared")
        self.assertEqual(units["ret_close30"]["source_unit"], "decimal")
        self.assertEqual(units["vol_open30_share"]["source_unit"], "dimensionless_ratio")
        self.assertEqual(units["n_bars"]["source_unit"], "count")

        # The evening audit scans every selectable events dataset's partition
        # schemas, so a real published partition must resolve there too.
        findings: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "intraday_stats" / f"trade_date={TRADE_DATE}.parquet"
            target.parent.mkdir(parents=True)
            aggregate_intraday_stats(_day("000001.SZ", REFERENCE_CLOSES, REFERENCE_VOLUMES)).to_parquet(target)
            unresolved = audit.audit_snapshot_unit_coverage(
                Path(tmp), "events.parquet", lambda *args: findings.append(args)
            )
        self.assertEqual(unresolved, 0)
        self.assertEqual(findings[0][3]["columns_checked"], len(INTRADAY_STATS_COLUMNS))


class MinuteDerivedBuildTest(unittest.TestCase):
    def test_builder_writes_each_missing_dataset_and_skips_existing_ones(self):
        script = REPO_ROOT / "scripts" / "data" / "build_minute_derived.py"
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            source = raw / "stk_mins_1min_by_date" / f"trade_date={TRADE_DATE}.parquet"
            source.parent.mkdir(parents=True)
            _day("000001.SZ", REFERENCE_CLOSES, REFERENCE_VOLUMES).to_parquet(source)

            def build(*extra: str) -> dict:
                done = subprocess.run(
                    [sys.executable, str(script), "--raw-dir", str(raw), *extra],
                    check=True, capture_output=True, text=True,
                )
                return json.loads(done.stdout)["datasets"]

            only_flow = build("--dataset", "intraday_flow")
            self.assertEqual(list(only_flow), ["intraday_flow"])
            self.assertFalse((raw / "intraday_stats").exists())
            both = build()
            self.assertEqual(both["intraday_flow"]["partitions_written"], 0)
            self.assertEqual(both["intraday_stats"]["partitions_written"], 1)
            written = pd.read_parquet(raw / "intraday_stats" / f"trade_date={TRADE_DATE}.parquet")
            self.assertEqual(list(written.columns), list(INTRADAY_STATS_COLUMNS))
            meta = json.loads(
                (raw / "intraday_stats" / f"trade_date={TRADE_DATE}.parquet.meta.json").read_text()
            )
            self.assertEqual(meta["api_name"], "derived:intraday_stats")
            self.assertEqual(meta["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
