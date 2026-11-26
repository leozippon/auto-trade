"""Derived intraday bar datasets: grid, exact aggregation, missing minutes, PIT stamp and view, build."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from autotrade.data_sources.tushare import audit
from autotrade.data_sources.tushare.io import write_parquet
from autotrade.environment.data.contracts import (
    CN_TZ,
    EVENT_DATASET_REFRESH_NODES,
    INTRADAY_BARS_CONTRACTS,
    INTRADAY_BARS_DATASETS,
    INTRADAY_FLOW_CONTRACT,
    LOCALLY_DERIVED_DATASETS,
    event_dataset_visible_cutoff,
)
from autotrade.environment.data.intraday_bars import (
    INTRADAY_BARS_COLUMNS,
    aggregate_intraday_bars,
)
from autotrade.environment.data.pit import to_cn_timestamps
from autotrade.environment.data.snapshot import (
    DEFAULT_DATASETS,
    SELECTABLE_DATASETS,
    SnapshotBuilder,
    SnapshotConfig,
)
from autotrade.environment.data.units import resolve_field
from autotrade.environment.replay.timeview import ReplayRows, Timeview
from autotrade.pipelines.pit_backend import HistoricalMinuteSource
from autotrade.pipelines.pit_views_seed import snapshot_timing_contract
from tests.unit.support.snapshot_fixture import (
    MINUTE_GRID,
    MINUTE_TRADE_DATE,
    build_fundamental_events,
    build_raw,
    write_fundamental_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "data" / "build_minute_derived.py"
LABELS = [f"{hour}:{minute:02d}" for hour, minute in MINUTE_GRID]
AFTERNOON = tuple(label for label in LABELS if label > "13:00")
THIRTY_MINUTE_ENDS = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]
VALUES = ["open", "high", "low", "close", "vol", "amount", "n_min"]


def _minutes(ts_code, seed, *, trade_date=MINUTE_TRADE_DATE, drop=(), extra=()):
    """One synthetic stock-day on the 241-minute grid: a seeded walk in whole
    ticks with a range around every minute and some no-trade minutes; the
    auction minutes trade most. ``drop`` removes minutes by label and
    ``extra`` appends off-grid ``(label, price, vol)`` minutes."""
    rng = np.random.default_rng(seed)
    close = np.round(10.0 + 0.01 * np.cumsum(rng.integers(-3, 4, len(LABELS))), 2)
    opened = np.concatenate(([close[0] - 0.02], close[:-1]))
    vol = 100.0 * rng.integers(0, 40, len(LABELS))
    vol[0], vol[-1] = 500_000.0, 300_000.0
    frame = pd.DataFrame(
        {
            "label": LABELS,
            "open": opened,
            "high": np.round(np.maximum(opened, close) + 0.01 * rng.integers(0, 3, len(LABELS)), 2),
            "low": np.round(np.minimum(opened, close) - 0.01 * rng.integers(0, 3, len(LABELS)), 2),
            "close": close,
            "vol": vol,
            "amount": np.round(vol * close, 2),
        }
    )
    frame = frame[~frame["label"].isin(drop)]
    if extra:
        off_grid = pd.DataFrame(
            [(label, price, price, price, price, vol, price * vol) for label, price, vol in extra],
            columns=frame.columns,
        )
        frame = pd.concat([frame, off_grid], ignore_index=True)
    day = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    frame.insert(0, "trade_time", day + " " + frame.pop("label") + ":00")
    frame.insert(0, "trade_date", trade_date)
    frame.insert(0, "ts_code", ts_code)
    return frame


def _clock(start: str, minutes: int) -> str:
    total = int(start[:2]) * 60 + int(start[3:]) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _reference(day: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """The bars of one stock-day by a plain loop over the grid."""
    labels = day["trade_time"].str.slice(11, 16)
    bars = []
    for opening in ("09:30", "13:00"):
        for index in range(120 // bar_minutes):
            after, end = _clock(opening, index * bar_minutes), _clock(opening, (index + 1) * bar_minutes)
            # (after, end], and the 09:30 auction minute with the first bar.
            rows = day[((labels > after) & (labels <= end)) | ((labels == "09:30") & (after == "09:30"))]
            bars.append(
                {
                    "trade_time": f"{day['trade_time'].iloc[0][:10]} {end}:00",
                    "open": rows["open"].iloc[0] if len(rows) else np.nan,
                    "high": rows["high"].max(),
                    "low": rows["low"].min(),
                    "close": rows["close"].iloc[-1] if len(rows) else np.nan,
                    "vol": math.fsum(rows["vol"]),
                    "amount": math.fsum(rows["amount"]),
                    "n_min": len(rows),
                }
            )
    return pd.DataFrame(bars)


def _stock_day(bars: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    return bars[bars["ts_code"] == ts_code].reset_index(drop=True)


def _write_minute_partition(raw: Path, trade_date: str, minutes: pd.DataFrame) -> pd.DataFrame:
    """A minute partition as the lake holds one: stamped at each bar's own minute, committed with its sidecar."""
    stamped = minutes.assign(
        available_at=minutes["trade_time"] + "+08:00", available_at_rule="source:trade_time_bar_close"
    )
    write_parquet(
        raw / "stk_mins_1min_by_date" / f"trade_date={trade_date}.parquet",
        stamped,
        api_name="stk_mins",
        params={"trade_date": trade_date},
        fields=list(stamped.columns),
    )
    return stamped


def _build(raw: Path, *extra: str) -> dict:
    """Run the minute-derived build for the bar datasets; its per-dataset summary."""
    datasets = [flag for dataset in INTRADAY_BARS_DATASETS for flag in ("--dataset", dataset)]
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--raw-dir", str(raw), *datasets, *extra],
        check=True, capture_output=True, text=True,
    )
    return json.loads(done.stdout)["datasets"]


class IntradayBarsDefinitionTest(unittest.TestCase):
    def _assert_bars(self, got: pd.DataFrame, want: pd.DataFrame) -> None:
        self.assertEqual(got["trade_time"].tolist(), want["trade_time"].tolist())
        for column in ("open", "high", "low", "close", "vol", "n_min"):
            np.testing.assert_array_equal(got[column].to_numpy(), want[column].to_numpy(), err_msg=column)
        np.testing.assert_allclose(got["amount"].to_numpy(), want["amount"].to_numpy(), rtol=1e-12, atol=0.0)

    def test_grid_holds_the_opening_minute_the_lunch_break_and_the_closing_auction(self):
        minutes = _minutes("000001.SZ", seed=1)
        minute = minutes.set_index(minutes["trade_time"].str.slice(11, 16))
        for dataset, bar_minutes in INTRADAY_BARS_DATASETS.items():
            with self.subTest(dataset=dataset):
                bars = aggregate_intraday_bars(minutes, dataset)
                self.assertEqual(list(bars.columns), list(INTRADAY_BARS_COLUMNS))
                ends = bars["trade_time"].str.slice(11, 16).tolist()
                per_half = 120 // bar_minutes
                self.assertEqual(len(ends), 2 * per_half)
                if bar_minutes == 30:
                    self.assertEqual(ends, THIRTY_MINUTE_ENDS)
                else:
                    self.assertEqual(ends[:2] + ends[23:25] + ends[-1:], ["09:35", "09:40", "11:30", "13:05", "15:00"])
                self.assertTrue(bars["trade_time"].str.startswith("2026-08-18 ").all())
                # The 09:30 auction minute is the first bar's extra minute: it
                # sets the bar's open and its volume is in the bar.
                self.assertEqual(bars["n_min"].tolist(), [bar_minutes + 1] + [bar_minutes] * (2 * per_half - 1))
                self.assertEqual(bars["open"].iloc[0], minute.loc["09:30", "open"])
                self.assertEqual(bars["vol"].iloc[0], minute.loc[minute.index <= ends[0], "vol"].sum())
                self.assertGreaterEqual(bars["vol"].iloc[0], 500_000.0)
                # No bar spans lunch: the morning ends on the 11:30 minute and
                # the afternoon opens on the 13:01 one.
                self.assertEqual(bars["close"].iloc[per_half - 1], minute.loc["11:30", "close"])
                self.assertEqual(bars["open"].iloc[per_half], minute.loc["13:01", "open"])
                self.assertEqual(int(bars["n_min"].iloc[:per_half].sum()), 121)
                # The 15:00 minute, with the closing auction, ends the last bar.
                self.assertEqual(bars["close"].iloc[-1], minute.loc["15:00", "close"])
                self.assertEqual(bars["vol"].iloc[-1], minute.loc[minute.index > ends[-2], "vol"].sum())
                self.assertGreaterEqual(bars["vol"].iloc[-1], 300_000.0)

    def test_bars_equal_the_minutes_they_came_from(self):
        days = [_minutes("000001.SZ", seed=1), _minutes("600000.SH", seed=2)]
        minutes = pd.concat(days, ignore_index=True)
        for dataset, bar_minutes in INTRADAY_BARS_DATASETS.items():
            bars = aggregate_intraday_bars(minutes, dataset)
            for day in days:
                with self.subTest(dataset=dataset, ts_code=day["ts_code"].iloc[0]):
                    got = _stock_day(bars, day["ts_code"].iloc[0])
                    self._assert_bars(got, _reference(day, bar_minutes))
                    # Nothing of the day is lost: its totals and extremes are the minutes'.
                    self.assertEqual(got["vol"].sum(), day["vol"].sum())
                    self.assertTrue(math.isclose(got["amount"].sum(), day["amount"].sum(), rel_tol=1e-12))
                    self.assertEqual(
                        (got["open"].iloc[0], got["high"].max(), got["low"].min(), got["close"].iloc[-1]),
                        (day["open"].iloc[0], day["high"].max(), day["low"].min(), day["close"].iloc[-1]),
                    )

    def test_thirty_minute_bars_are_six_five_minute_bars(self):
        minutes = pd.concat([_minutes("000001.SZ", seed=1), _minutes("600000.SH", seed=2)], ignore_index=True)
        five = aggregate_intraday_bars(minutes, "intraday_bars_5min")
        thirty = aggregate_intraday_bars(minutes, "intraday_bars_30min")
        folded = five.groupby(np.arange(len(five)) // 6).agg(
            ts_code=("ts_code", "first"), trade_time=("trade_time", "last"), open=("open", "first"),
            high=("high", "max"), low=("low", "min"), close=("close", "last"), vol=("vol", "sum"),
            amount=("amount", "sum"), n_min=("n_min", "sum"),
        )
        self.assertEqual(folded["ts_code"].tolist(), thirty["ts_code"].tolist())
        self._assert_bars(thirty, folded)

    def test_missing_minutes_keep_the_grid_and_invent_no_price(self):
        # The auction minute, one whole 5-minute bar and the whole afternoon
        # are missing; a 15:05 minute is off the grid.
        minutes = _minutes(
            "000001.SZ",
            seed=3,
            drop=("09:30", "10:01", "10:02", "10:03", "10:04", "10:05", *AFTERNOON),
            extra=(("15:05", 99.0, 1_000_000.0),),
        )
        minute = minutes.set_index(minutes["trade_time"].str.slice(11, 16))
        for dataset, bar_minutes in INTRADAY_BARS_DATASETS.items():
            with self.subTest(dataset=dataset):
                bars = aggregate_intraday_bars(minutes, dataset)
                per_half = 120 // bar_minutes
                self.assertEqual(len(bars), 2 * per_half)
                self._assert_bars(bars, _reference(minutes, bar_minutes))
                ends = bars["trade_time"].str.slice(11, 16).tolist()
                bar = bars.set_index(pd.Index(ends))
                # Without its auction minute the first bar opens on 09:31.
                self.assertEqual(int(bars["n_min"].iloc[0]), bar_minutes)
                self.assertEqual(bars["open"].iloc[0], minute.loc["09:31", "open"])
                # The half-day leaves the afternoon bars in place, empty.
                afternoon = bars.iloc[per_half:]
                self.assertTrue(afternoon[["open", "high", "low", "close"]].isna().all().all())
                self.assertEqual(afternoon[["vol", "amount", "n_min"]].to_numpy().tolist(), [[0.0, 0.0, 0.0]] * per_half)
                if bar_minutes == 5:
                    # The bar none of whose minutes exists is the morning's only empty one.
                    empty = [end for end, close in zip(ends[:per_half], bars["close"]) if math.isnan(close)]
                    self.assertEqual(empty, ["10:05"])
                    self.assertEqual(bar.loc["10:05", ["vol", "amount", "n_min"]].tolist(), [0.0, 0.0, 0])
                else:
                    # A bar missing some of its minutes aggregates the rest.
                    self.assertFalse(bars["close"].iloc[:per_half].isna().any())
                    self.assertEqual(int(bar.loc["10:30", "n_min"]), 25)
                    self.assertEqual(bar.loc["10:30", "open"], minute.loc["10:06", "open"])

    def test_stock_days_order_and_call_shape(self):
        first = _minutes("000001.SZ", seed=1)
        second = _minutes("600000.SH", seed=2)
        later = _minutes("000001.SZ", seed=4, trade_date="20260819")
        halted = _minutes("600001.SH", seed=5).assign(vol=0.0, amount=0.0)
        beijing = _minutes("920627.BJ", seed=6, extra=(("15:05", 12.0, 10.0),))
        minutes = pd.concat([second, beijing, later, halted, first], ignore_index=True)
        for dataset in INTRADAY_BARS_DATASETS:
            with self.subTest(dataset=dataset):
                bars = aggregate_intraday_bars(minutes, dataset)
                # Whole-day halts and .BJ are out, as in intraday_flow; rows
                # come by code, day and bar.
                self.assertEqual(
                    bars[["ts_code", "trade_date"]].drop_duplicates().to_numpy().tolist(),
                    [["000001.SZ", "20260818"], ["000001.SZ", "20260819"], ["600000.SH", "20260818"]],
                )
                apart = pd.concat(
                    [aggregate_intraday_bars(day, dataset) for day in (first, later, second)], ignore_index=True
                )
                pd.testing.assert_frame_equal(bars, apart)
                shuffled = minutes.sample(frac=1.0, random_state=7)
                pd.testing.assert_frame_equal(aggregate_intraday_bars(shuffled, dataset), bars)
                empty = aggregate_intraday_bars(pd.concat([halted, beijing]), dataset)
                self.assertTrue(empty.empty)
                self.assertEqual(list(empty.columns), list(INTRADAY_BARS_COLUMNS))

    def test_a_defective_minute_fails_instead_of_publishing(self):
        for column, value, message in (("low", 0.0, "non-positive"), ("amount", float("nan"), "volume or amount")):
            frame = _minutes("000001.SZ", seed=1)
            frame.loc[100, column] = value
            with self.assertRaisesRegex(ValueError, message):
                aggregate_intraday_bars(frame, "intraday_bars_30min")


class IntradayBarsContractTest(unittest.TestCase):
    def test_every_bar_takes_the_close_stamp_and_rolls_in_after_the_evening_job(self):
        day = date(2026, 8, 18)  # a Tuesday
        moments = (
            ((18, 15, 0), False),   # during the session
            ((18, 23, 59), False),  # stamped, but the evening job has not landed it
            ((19, 3, 4), False),
            ((19, 3, 5), True),     # the evening job is ready
            ((19, 8, 0), True),     # the next pre-open decision
        )
        for dataset, contract in INTRADAY_BARS_CONTRACTS.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(contract.dataset, dataset)
                self.assertEqual(contract.rule, INTRADAY_FLOW_CONTRACT.rule)
                self.assertIn(dataset, LOCALLY_DERIVED_DATASETS)
                self.assertNotIn(dataset, EVENT_DATASET_REFRESH_NODES)
                bars = aggregate_intraday_bars(_minutes("000001.SZ", seed=1), dataset)
                # One stamp for the whole day, whatever minute a bar ends on.
                self.assertEqual(set(bars["available_at"]), {"2026-08-18 17:30:00+08:00"})
                self.assertEqual(set(bars["available_at_rule"]), {"contract_1730_from:trade_date"})
                stamp = contract.available_at(day)
                for (dom, hour, minute), visible in moments:
                    when = datetime(2026, 8, dom, hour, minute, tzinfo=CN_TZ)
                    self.assertEqual(stamp <= event_dataset_visible_cutoff(dataset, when), visible, when)
                # A seed that mounts the dataset records the rule it was stamped by.
                timing = snapshot_timing_contract(SnapshotConfig(events_datasets=(dataset,)))
                self.assertEqual(timing["stamps"][f"events/{dataset}"], contract.rule)

    def test_every_column_resolves_in_the_registry_inventory_and_nightly_audit(self):
        inventory = json.loads(
            (REPO_ROOT / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"]["events.parquet"]
        findings: list[tuple] = []
        with tempfile.TemporaryDirectory() as tmp:
            for dataset in INTRADAY_BARS_DATASETS:
                self.assertIn(dataset, SELECTABLE_DATASETS["events"])
                self.assertNotIn(dataset, DEFAULT_DATASETS["events"])
                self.assertEqual(sorted(inventory[dataset]), sorted({*INTRADAY_BARS_COLUMNS, "dataset"}))
                units = {column: resolve_field("events.parquet", dataset, column) for column in inventory[dataset]}
                self.assertEqual(units["close"]["source_unit"], "CNY_per_share")
                self.assertEqual(units["vol"]["source_unit"], "shares")
                self.assertEqual(units["amount"]["source_unit"], "CNY")
                self.assertEqual(units["n_min"]["source_unit"], "count")
                target = Path(tmp) / dataset / f"trade_date={MINUTE_TRADE_DATE}.parquet"
                target.parent.mkdir(parents=True)
                aggregate_intraday_bars(_minutes("000001.SZ", seed=1), dataset).to_parquet(target)
            # The evening audit scans every selectable events dataset's
            # partition schemas, so a published partition must resolve there too.
            unresolved = audit.audit_snapshot_unit_coverage(
                Path(tmp), "events.parquet", lambda *args: findings.append(args)
            )
        self.assertEqual(unresolved, 0)
        self.assertEqual(findings[0][3]["columns_checked"], len(INTRADAY_BARS_DATASETS) * len(INTRADAY_BARS_COLUMNS))


class IntradayBarsViewTest(unittest.TestCase):
    def test_a_replay_reads_the_bars_it_would_aggregate_from_the_minute_view_and_none_early(self):
        # Thursday, then Friday and Monday after the holiday week.
        sessions = ("20210930", "20211008", "20211011")
        config = SnapshotConfig(
            events_datasets=tuple(INTRADAY_BARS_DATASETS),
            macro_datasets=("cn_gdp",),
            text_datasets=("cctv_news",),
            fundamental_datasets=("income_vip",),
            intraday_trade_days=1,
            include_industry=False,
            include_intraday=True,
            replay_include_minutes=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            events_root = Path(tmp) / "fund_events"
            build_raw(raw)
            build_fundamental_events(events_root)
            status_path = Path(tmp) / "fundamental_events_status.json"
            write_fundamental_status(status_path)
            for index, session in enumerate(sessions):
                _write_minute_partition(
                    raw,
                    session,
                    pd.concat(
                        [
                            _minutes("000001.SZ", seed=10 + index, trade_date=session),
                            _minutes("600000.SH", seed=20 + index, trade_date=session),
                        ],
                        ignore_index=True,
                    ),
                )
            _build(raw)
            anchor = datetime(2021, 10, 7, 23, 59, 59, tzinfo=CN_TZ)
            builder = SnapshotBuilder(raw, events_root, status_path)
            snapshot, slot = Path(tmp) / "decision", Path(tmp) / "replay"
            builder.build_decision_snapshot(anchor, snapshot, config)
            builder.build_replay_slot("20211008", "20211011", slot, label="valid", config=config, available_from=anchor)
            minute_source = HistoricalMinuteSource(slot / "intraday_1min.parquet")
            view = Timeview(
                host_dir=Path(tmp) / "asof",
                snapshot_dir=snapshot,
                replay={"events": ReplayRows(slot / "events.parquet")},
                incremental_domains={"intraday_1min"},
            )
            decisions = (
                ("2021-10-08 08:00", "20210930"),
                ("2021-10-08 15:00", "20210930"),
                ("2021-10-08 23:59", "20210930"),  # Friday is stamped 17:30; its evening job is not ready
                ("2021-10-09 03:04", "20210930"),
                ("2021-10-09 03:05", "20211008"),
                ("2021-10-11 08:00", "20211008"),  # Monday's pre-open decision
                ("2021-10-11 17:45", "20211008"),
                ("2021-10-12 08:00", "20211011"),
            )
            for when, newest in decisions:
                with self.subTest(when=when):
                    now = pd.Timestamp(when, tz="Asia/Shanghai")
                    minute_source.append_visible(view, now.to_pydatetime())
                    asof_dir, _ = view.refresh(now)
                    events = pd.read_parquet(Path(asof_dir) / "events")
                    minutes = pd.read_parquet(Path(asof_dir) / "intraday_1min")
                    # No bar is visible before its stamp, and the bars end on
                    # the session the minute view ends on.
                    self.assertTrue((to_cn_timestamps(events["available_at"]) <= now).all())
                    self.assertEqual(events["trade_date"].max(), newest)
                    self.assertEqual(minutes["trade_date"].max(), newest)
                    for dataset in INTRADAY_BARS_DATASETS:
                        through_view = events.loc[events["dataset"] == dataset, list(INTRADAY_BARS_COLUMNS)]
                        aggregated = aggregate_intraday_bars(minutes, dataset).sort_values(
                            ["trade_date", "ts_code", "trade_time"], kind="stable"
                        )
                        pd.testing.assert_frame_equal(
                            through_view.reset_index(drop=True), aggregated.reset_index(drop=True)
                        )


class IntradayBarsBuildTest(unittest.TestCase):
    def test_the_build_is_deterministic_and_skips_what_it_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            minutes = _write_minute_partition(
                raw,
                MINUTE_TRADE_DATE,
                pd.concat([_minutes("000001.SZ", seed=1), _minutes("600000.SH", seed=2)], ignore_index=True),
            )
            first = _build(raw)
            built = {}
            for dataset, bar_minutes in INTRADAY_BARS_DATASETS.items():
                path = raw / dataset / f"trade_date={MINUTE_TRADE_DATE}.parquet"
                self.assertEqual(first[dataset]["partitions_written"], 1)
                self.assertEqual(first[dataset]["rows"], 2 * 240 // bar_minutes)
                built[dataset] = pd.read_parquet(path)
                pd.testing.assert_frame_equal(built[dataset], aggregate_intraday_bars(minutes, dataset))
                meta = json.loads(path.with_suffix(".parquet.meta.json").read_text(encoding="utf-8"))
                self.assertEqual((meta["api_name"], meta["row_count"]), (f"derived:{dataset}", len(built[dataset])))
            # A committed trade date is skipped; a rebuild writes the same rows.
            self.assertEqual({summary["partitions_written"] for summary in _build(raw).values()}, {0})
            self.assertEqual({summary["partitions_written"] for summary in _build(raw, "--rebuild").values()}, {1})
            for dataset, frame in built.items():
                pd.testing.assert_frame_equal(
                    pd.read_parquet(raw / dataset / f"trade_date={MINUTE_TRADE_DATE}.parquet"), frame
                )


if __name__ == "__main__":
    unittest.main()
