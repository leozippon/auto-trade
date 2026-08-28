"""Per-tick Timeview: node-gated six-domain rolling, write-once parts, versioning."""

import json
import tempfile
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from autotrade.environment.replay.timeview import Timeview

CN_TZ = ZoneInfo("Asia/Shanghai")
TS = "000001.SZ"


class FakeExecutor:
    def map_path(self, path) -> str:
        return str(path)


def _when(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=CN_TZ)


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _frozen_snapshot(root: Path) -> Path:
    snap = root / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    _write(snap / "daily.parquet", pd.DataFrame([{"trade_date": "20211231", "ts_code": TS, "open": 9.0, "close": 9.5}]))
    _write(snap / "universe.parquet", pd.DataFrame([{"ts_code": TS, "name": "x"}]))
    _write(
        snap / "text_index.parquet",
        pd.DataFrame(
            [
                {
                    "text_id": "frozen_news",
                    "dataset": "news",
                    "ts_codes": TS,
                    "title": "frozen",
                    "available_at": "2021-12-31T08:55:00+08:00",
                    "library_file": "news.parquet",
                }
            ]
        ),
    )
    _write(snap / "text_library" / "news.parquet", pd.DataFrame([{"text_id": "frozen_news", "body": "frozen body"}]))
    # Production builders emit schema-equal frozen/replay pairs, so each empty
    # frozen domain carries the same columns its `_replay_frames` frame does.
    frozen_columns = {
        "events": ["dataset", "ts_code", "trade_date", "available_at"],
        "macro": ["dataset", "ts_code", "available_at"],
        "fundamentals": ["dataset", "ts_code", "business_key", "available_at"],
        "intraday_1min": ["dataset", "ts_code", "available_at"],
    }
    for name, columns in frozen_columns.items():
        _write(snap / f"{name}.parquet", pd.DataFrame(columns=columns))
    return snap


def _replay_frames() -> dict[str, pd.DataFrame]:
    daily = pd.DataFrame(
        [
            {"trade_date": "20220104", "ts_code": TS, "open": 10.0, "close": 10.2, "available_at": "2022-01-04T17:30:00+08:00"},
            {"trade_date": "20220105", "ts_code": TS, "open": 10.3, "close": 11.0, "available_at": "2022-01-05T17:30:00+08:00"},
        ]
    )
    events = pd.DataFrame(
        [
            {"dataset": "margin_secs", "ts_code": TS, "trade_date": "20220104", "available_at": "2022-01-04T09:00:00+08:00"},
            {"dataset": "block_trade", "ts_code": TS, "trade_date": "20220104", "available_at": "2022-01-04T21:00:00+08:00"},
        ]
    )
    fundamentals = pd.DataFrame(
        [{"dataset": "income_vip", "ts_code": TS, "business_key": "k", "available_at": "2022-01-04T18:00:00+08:00"}]
    )
    text_index = pd.DataFrame(
        [
            {
                "text_id": "news_early",
                "dataset": "news",
                "ts_codes": TS,
                "title": "early",
                "available_at": "2022-01-04T08:55:00+08:00",
                "library_file": "news.parquet",
            },
            {
                "text_id": "news_late",
                "dataset": "news",
                "ts_codes": TS,
                "title": "late",
                "available_at": "2022-01-04T09:05:00+08:00",
                "library_file": "news.parquet",
            },
        ]
    )
    return {"daily": daily, "events": events, "fundamentals": fundamentals, "text_index": text_index}


class TimeviewTest(unittest.TestCase):
    def _build(self, root: Path) -> Timeview:
        replay_library = root / "replay" / "text_library"
        _write(
            replay_library / "news.parquet",
            pd.DataFrame(
                [
                    {"text_id": "news_early", "body": "early body"},
                    {"text_id": "news_late", "body": "late body"},
                ]
            ),
        )
        return Timeview(
            host_dir=root / "asof",
            executor=FakeExecutor(),
            snapshot_dir=_frozen_snapshot(root),
            replay_frames=_replay_frames(),
            replay_text_library_dir=replay_library,
        )

    def _dates(self, asof_dir: str, domain: str) -> set[str]:
        frame = pd.read_parquet(Path(asof_dir) / domain)
        col = "trade_date" if "trade_date" in frame.columns else "available_at"
        return set(frame[col].astype(str)) if col in frame.columns else set()

    def test_frozen_base_is_part_zero_and_today_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            asof, version = tv.refresh(_when("2022-01-04 09:10:00"))
            # Intraday-session day: daily view is just the frozen history; today's bar
            # waits for that night's conservative evening boundary (~03:05 next day).
            self.assertEqual(self._dates(asof, "daily"), {"20211231"})
            self.assertTrue((Path(asof) / "daily" / "part_0000.parquet").exists())

    def test_schema_padded_null_column_carries_replay_values_without_warning(self):
        # End-to-end for the v6 schema padding: a dataset absent from the
        # decision window contributes a null-typed column to the frozen file;
        # replay rows carrying real values for it must survive the roll and no
        # schema-mismatch RuntimeWarning may fire.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            frozen = pd.DataFrame({
                "dataset": ["margin_secs"],
                "available_at": ["2021-12-31T18:00:00+08:00"],
                "ts_code": [TS],
                "hot_num": pd.Series([None], dtype=pd.ArrowDtype(pa.float64())),
            })
            _write(snapshot / "events.parquet", frozen)
            replay = pd.DataFrame([{
                "dataset": "ths_hot", "available_at": "2022-01-04T18:00:00+08:00",
                "ts_code": TS, "hot_num": 7.0,
            }])
            tv = Timeview(
                host_dir=root / "asof", executor=FakeExecutor(),
                snapshot_dir=snapshot, replay_frames={"events": replay},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                asof, _ = tv.refresh(_when("2022-01-05 09:10:00"))
            events = pd.read_parquet(Path(asof) / "events")
            self.assertIn(7.0, set(events["hot_num"].dropna().astype(float)))

    def test_frozen_only_column_is_padded_with_the_frozen_type(self):
        # The mirror of the warning below: a frozen column the replay window
        # never carries must still appear in the rolled part, typed by the
        # frozen part, or the parts directory stops reading back as one table.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            _write(
                snapshot / "daily.parquet",
                pd.DataFrame([{"trade_date": "20211231", "ts_code": TS, "close": 9.5, "note": "frozen"}]),
            )
            replay = pd.DataFrame([{
                "trade_date": "20220104", "ts_code": TS, "close": 10.2,
                "available_at": "2022-01-04T17:30:00+08:00",
            }])
            tv = Timeview(
                host_dir=root / "asof", executor=FakeExecutor(),
                snapshot_dir=snapshot, replay_frames={"daily": replay},
            )
            asof, _ = tv.refresh(_when("2022-01-05 09:10:00"))
            part = pq.ParquetFile(Path(asof) / "daily" / "part_0001.parquet")
            self.assertEqual(part.schema_arrow.field("note").type, pa.string())
            daily = pd.read_parquet(Path(asof) / "daily")
            self.assertEqual(sorted(daily["trade_date"].astype(str)), ["20211231", "20220104"])
            self.assertEqual(list(daily.loc[daily["trade_date"] == "20220104", "note"].isna()), [True])

    def test_replay_column_missing_from_frozen_schema_warns(self):
        # Designed advisory: replay-only columns are dropped by the roll's
        # projection onto the frozen schema, so a frozen/replay schema mismatch
        # must warn loudly (not fail — the financial domain allows schema ahead
        # of window data).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            _write(
                snapshot / "daily.parquet",
                pd.DataFrame([{"trade_date": "20211231", "ts_code": TS, "open": 9.0, "close": 9.5}]),
            )
            replay = pd.DataFrame([{
                "trade_date": "20220104", "ts_code": TS, "open": 10.0, "close": 10.2,
                "extra_factor": 1.0, "available_at": "2022-01-04T17:30:00+08:00",
            }])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                Timeview(
                    host_dir=root / "asof", executor=FakeExecutor(),
                    snapshot_dir=snapshot, replay_frames={"daily": replay},
                )
            runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
            self.assertEqual(len(runtime), 1, [str(w.message) for w in caught])
            message = str(runtime[0].message)
            self.assertIn("Timeview domain 'daily'", message)
            self.assertIn("extra_factor", message)
            self.assertIn("rebuild the frozen snapshot", message)

    def test_daily_rolls_after_evening_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            tv.refresh(_when("2022-01-04 09:10:00"))
            asof, _ = tv.refresh(_when("2022-01-05 09:10:00"))
            # Prior replay day visible once its evening node completed; today still not.
            self.assertEqual(self._dates(asof, "daily"), {"20211231", "20220104"})

    def test_incremental_intraday_partitions_match_eager_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            frozen = pd.DataFrame(
                [{
                    "trade_date": "20211231", "ts_code": TS,
                    "trade_time": "2021-12-31 15:00:00", "open": 9.0, "close": 9.5,
                }]
            )
            _write(snapshot / "intraday_1min.parquet", frozen)
            day1 = pd.DataFrame(
                [{
                    "trade_date": "20220104", "ts_code": TS,
                    "trade_time": "2022-01-04 15:00:00", "open": 10.0, "close": 10.2,
                    "available_at": "2022-01-04T15:00:00+08:00",
                }]
            )
            day2 = pd.DataFrame(
                [{
                    "trade_date": "20220105", "ts_code": TS,
                    "trade_time": "2022-01-05 15:00:00", "open": 10.3, "close": 11.0,
                    "available_at": "2022-01-05T15:00:00+08:00",
                }]
            )
            eager = Timeview(
                host_dir=root / "eager",
                executor=FakeExecutor(),
                snapshot_dir=snapshot,
                replay_frames={"intraday_1min": pd.concat([day1, day2], ignore_index=True)},
            )
            incremental = Timeview(
                host_dir=root / "incremental",
                executor=FakeExecutor(),
                snapshot_dir=snapshot,
                replay_frames={},
                incremental_domains={"intraday_1min"},
            )

            incremental.append_replay_partition("intraday_1min", day1)
            eager_dir, eager_version = eager.refresh(_when("2022-01-04 09:10:00"))
            incremental_dir, incremental_version = incremental.refresh(_when("2022-01-04 09:10:00"))
            self.assertEqual(eager_version, incremental_version)
            pd.testing.assert_frame_equal(
                pd.read_parquet(Path(eager_dir) / "intraday_1min"),
                pd.read_parquet(Path(incremental_dir) / "intraday_1min"),
            )
            incremental.append_replay_partition("intraday_1min", day2)
            eager_dir, eager_version = eager.refresh(_when("2022-01-05 03:30:00"))
            incremental_dir, incremental_version = incremental.refresh(_when("2022-01-05 03:30:00"))
            self.assertEqual(eager_version, incremental_version)
            pd.testing.assert_frame_equal(
                pd.read_parquet(Path(eager_dir) / "intraday_1min"),
                pd.read_parquet(Path(incremental_dir) / "intraday_1min"),
            )
            eager_dir, eager_version = eager.refresh(_when("2022-01-06 03:30:00"))
            incremental_dir, incremental_version = incremental.refresh(_when("2022-01-06 03:30:00"))
            self.assertEqual(eager_version, incremental_version)
            pd.testing.assert_frame_equal(
                pd.read_parquet(Path(eager_dir) / "intraday_1min"),
                pd.read_parquet(Path(incremental_dir) / "intraday_1min"),
            )
            self.assertEqual(incremental._domains["intraday_1min"]._pending, [])

    def test_margin_secs_visible_same_day_block_trade_waits_for_evening(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            # 09:10 on 20220104: the 09:03 margin_secs node is done, the evening node is not.
            asof, _ = tv.refresh(_when("2022-01-04 09:10:00"))
            events = pd.read_parquet(Path(asof) / "events")
            self.assertEqual(set(events["dataset"]), {"margin_secs"})
            # Block trade (evening dataset) only rolls in after its evening node completes.
            asof2, _ = tv.refresh(_when("2022-01-05 03:06:00"))
            events2 = pd.read_parquet(Path(asof2) / "events")
            self.assertEqual(set(events2["dataset"]), {"margin_secs", "block_trade"})

    def test_fundamentals_roll_on_pit_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            asof, _ = tv.refresh(_when("2022-01-04 09:10:00"))
            self.assertEqual(len(pd.read_parquet(Path(asof) / "fundamentals")), 0)  # before the PIT build
            asof2, _ = tv.refresh(_when("2022-01-05 04:10:00"))  # after the ~04:05 PIT full rebuild
            self.assertEqual(len(pd.read_parquet(Path(asof2) / "fundamentals")), 1)

    def test_text_index_and_library_roll_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            asof, _ = tv.refresh(_when("2022-01-04 08:59:00"))
            index = pd.read_parquet(Path(asof) / "text_index")
            bodies = pd.concat(pd.read_parquet(path) for path in sorted((Path(asof) / "text_library").glob("*.parquet")))
            self.assertEqual(set(index["text_id"].astype(str)), {"frozen_news"})
            self.assertEqual(set(bodies["text_id"].astype(str)), {"frozen_news"})

            asof2, _ = tv.refresh(_when("2022-01-04 09:01:00"))
            index2 = pd.read_parquet(Path(asof2) / "text_index")
            bodies2 = pd.concat(pd.read_parquet(path) for path in sorted((Path(asof2) / "text_library").glob("*.parquet")))
            self.assertEqual(set(index2["text_id"].astype(str)), {"frozen_news", "news_early"})
            self.assertEqual(set(bodies2["text_id"].astype(str)), {"frozen_news", "news_early"})
            self.assertNotIn("news_late", set(bodies2["text_id"].astype(str)))

    def test_version_bumps_on_roll_and_is_stable_in_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            _, v_open = tv.refresh(_when("2022-01-04 09:10:00"))
            # No covering node completes across the session, so the view is frozen.
            _, v_mid = tv.refresh(_when("2022-01-04 11:00:00"))
            _, v_close = tv.refresh(_when("2022-01-04 14:30:00"))
            self.assertEqual(v_open, v_mid)
            self.assertEqual(v_open, v_close)
            # The next day's evening + pre-open nodes roll new rows, advancing the version.
            _, v_next = tv.refresh(_when("2022-01-05 09:20:00"))
            self.assertNotEqual(v_open, v_next)

    def test_ticks_before_next_boundary_do_not_traverse_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            tv.refresh(_when("2022-01-04 09:10:00"))
            with ExitStack() as stack:
                rolls = [
                    stack.enter_context(mock.patch.object(view, "roll", wraps=view.roll))
                    for view in tv._domains.values()
                ]
                text_roll = stack.enter_context(mock.patch.object(tv._text, "roll", wraps=tv._text.roll))
                tv.refresh(_when("2022-01-04 11:00:00"))
                tv.refresh(_when("2022-01-04 14:30:00"))
                for roll in rolls:
                    roll.assert_not_called()
                text_roll.assert_not_called()

    def test_node_boundary_is_inclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            asof, _ = tv.refresh(_when("2022-01-04 09:04:59"))
            self.assertEqual(list((Path(asof) / "events").glob("*.parquet")), [])

            asof, _ = tv.refresh(_when("2022-01-04 09:05:00"))
            events = pd.read_parquet(Path(asof) / "events")
            self.assertEqual(events["dataset"].tolist(), ["margin_secs"])

    def test_one_refresh_catches_up_across_multiple_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            tv.refresh(_when("2022-01-04 08:59:00"))

            # One clock jump crosses the 09:00 text and 09:05 margin nodes. Both
            # cursors catch up to the latest eligible cutoff in a single refresh.
            asof, _ = tv.refresh(_when("2022-01-04 09:10:00"))
            events = pd.read_parquet(Path(asof) / "events")
            text = pd.read_parquet(Path(asof) / "text_index")
            self.assertEqual(events["dataset"].tolist(), ["margin_secs"])
            self.assertIn("news_early", set(text["text_id"].astype(str)))

    def test_parts_are_write_once_no_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            tv.refresh(_when("2022-01-05 09:10:00"))
            asof, _ = tv.refresh(_when("2022-01-05 09:20:00"))  # same signatures: no new parts
            daily = pd.read_parquet(Path(asof) / "daily")
            # 20220104 appears exactly once even after repeated refreshes.
            self.assertEqual(list(daily["trade_date"].astype(str)).count("20220104"), 1)

    def test_stash_requires_a_validated_contract_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stash = root / "stash"
            stash.mkdir()
            with self.assertRaisesRegex(RuntimeError, "no validated semantic contract"):
                Timeview(
                    host_dir=root / "asof",
                    snapshot_dir=_frozen_snapshot(root),
                    replay_frames=_replay_frames(),
                    stash_dir=stash,
                )

    def test_concurrent_stash_part_publication_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _frozen_snapshot(root)
            stash = root / "stash"
            stash.mkdir()
            # PIT backend validates and publishes this contract before Timeview
            # receives the directory; this test isolates concurrent part I/O.
            (stash / "contract.json").write_text(
                json.dumps({"validated": True}), encoding="utf-8"
            )
            barrier = Barrier(2)

            def build(index: int) -> pd.DataFrame:
                view = Timeview(
                    host_dir=root / f"asof_{index}",
                    executor=FakeExecutor(),
                    snapshot_dir=snapshot,
                    replay_frames=_replay_frames(),
                    stash_dir=stash,
                )
                barrier.wait()
                asof, _ = view.refresh(_when("2022-01-05 09:10:00"))
                return pd.read_parquet(Path(asof) / "daily")

            with ThreadPoolExecutor(max_workers=2) as pool:
                frames = list(pool.map(build, range(2)))

            for frame in frames:
                self.assertEqual(
                    set(frame["trade_date"].astype(str)), {"20211231", "20220104"}
                )
            published = stash / "daily" / "part_0001.parquet"
            self.assertTrue(published.is_file())
            self.assertEqual(len(pd.read_parquet(published)), 1)
            self.assertEqual(list(stash.rglob("*.tmp")), [])

    def test_auction_rolls_at_observed_row_time_not_evening_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            import duckdb

            from autotrade.environment.data.snapshot import SnapshotBuilder

            root = Path(tmp)
            snapshot = _frozen_snapshot(root)
            _write(
                snapshot / "auction.parquet",
                pd.DataFrame(
                    {
                        column: pd.Series(
                            dtype=(
                                "string"
                                if column in SnapshotBuilder._AUCTION_STRING_COLUMNS
                                else "float64"
                            )
                        )
                        for column in SnapshotBuilder._AUCTION_COLUMNS
                    }
                ),
            )
            frames = _replay_frames()
            frames["auction"] = pd.DataFrame(
                [{
                    "trade_date": "20220104",
                    "session": "open",
                    "ts_code": TS,
                    "price": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                    "pre_close": 9.5,
                    "turnover_rate": 0.1,
                    "volume_ratio": 1.2,
                    "float_share": 10000.0,
                    "available_at": "2022-01-04T09:28:36+08:00",
                    "available_at_rule": "observed",
                }]
            )
            tv = Timeview(
                host_dir=root / "asof",
                executor=FakeExecutor(),
                snapshot_dir=snapshot,
                replay_frames=frames,
            )

            asof, before = tv.refresh(_when("2022-01-04 09:28:30"))
            part0 = Path(asof) / "auction" / "part_0000.parquet"
            self.assertTrue(part0.exists())
            self.assertEqual(part0.stat().st_ino, (snapshot / "auction.parquet").stat().st_ino)
            self.assertTrue(pd.read_parquet(Path(asof) / "auction").empty)
            duck_empty = duckdb.execute(
                "SELECT * FROM read_parquet(?)", [str(Path(asof) / "auction" / "*.parquet")]
            ).fetchdf()
            self.assertTrue(duck_empty.empty)
            self.assertEqual(str(duck_empty["ts_code"].dtype), "object")
            # The observed boundary is second-precision; crossing it within the
            # same minute must not be hidden by a minute-rounded signature.
            _, still_before = tv.refresh(_when("2022-01-04 09:28:35"))
            self.assertEqual(still_before, before)
            asof, after = tv.refresh(_when("2022-01-04 09:28:40"))
            self.assertNotEqual(before, after)
            auction = pd.read_parquet(Path(asof) / "auction")
            self.assertEqual(auction["ts_code"].tolist(), [TS])
            part_count = len(list((Path(asof) / "auction").glob("*.parquet")))
            _, repeated = tv.refresh(_when("2022-01-04 09:29:00"))
            self.assertEqual(repeated, after)
            self.assertEqual(len(list((Path(asof) / "auction").glob("*.parquet"))), part_count)


class TimeviewMacroDatasetGatingTest(unittest.TestCase):
    def test_weekend_global_run_does_not_expose_domestic_macro_rows(self):
        # Domestic macro rows carry weekend available_at stamps (month-end
        # EODs, weekend repo dates) that only the Monday evening run ingests.
        # A domain-level cutoff union exposed them right after the weekend
        # global run — a reproduced early-visibility bias, not a theoretical
        # one. Macro must gate per dataset like the events domain.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _frozen_snapshot(root)
            saturday_eod = "2022-01-08T23:59:59+08:00"
            replay = {
                "macro": pd.DataFrame([
                    {"dataset": "cn_m", "ts_code": "monthly", "available_at": saturday_eod},
                    {"dataset": "index_global", "ts_code": "SPX", "available_at": saturday_eod},
                ]),
            }
            tv = Timeview(host_dir=root / "asof", executor=FakeExecutor(), snapshot_dir=snap, replay_frames=replay)

            # Sunday, after the Sunday-evening global landing completed: the
            # global row is visible, the domestic row must not be.
            asof, _ = tv.refresh(_when("2022-01-09 23:50:00"))
            macro = pd.read_parquet(Path(asof) / "macro")
            self.assertEqual(set(macro["dataset"]), {"index_global"})

            # After Monday's evening run completes, the domestic row lands.
            asof, _ = tv.refresh(_when("2022-01-11 03:10:00"))
            macro = pd.read_parquet(Path(asof) / "macro")
            self.assertEqual(set(macro["dataset"]), {"index_global", "cn_m"})


class TimeviewIntradaySchemaTest(unittest.TestCase):
    """The frozen and replay intraday domains share one schema: no internal
    available_at, and the auction-correction columns are never NaN-backfilled (R19-4)."""

    def _minute(self, trade_date: str, available_at: str | None) -> pd.DataFrame:
        from autotrade.environment.data.auction import apply_open_auction_correction

        row = {
            "ts_code": TS,
            "trade_time": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 09:30:00",
            "trade_date": trade_date,
            "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "vol": 20000.0, "amount": 200000.0,
        }
        if available_at is not None:
            row["available_at"] = available_at
        return apply_open_auction_correction(pd.DataFrame([row]))

    def test_intraday_view_has_no_available_at_and_keeps_auction_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = _frozen_snapshot(root)
            # Real frozen intraday: auction columns present, internal available_at dropped
            # (mirrors snapshot._build_intraday).
            _write(snap / "intraday_1min.parquet", self._minute("20211231", available_at=None))
            replay = {
                "daily": _replay_frames()["daily"],
                # Replay intraday keeps available_at as the row-level Timeview gate.
                "intraday_1min": self._minute("20220104", available_at="2022-01-04T09:30:00+08:00"),
            }
            tv = Timeview(host_dir=root / "asof", executor=FakeExecutor(), snapshot_dir=snap, replay_frames=replay)
            # After the 20220104 evening node completes (fallback ~03:05 on 0105) the replay bar rolls in.
            asof, _ = tv.refresh(_when("2022-01-05 09:10:00"))
            intraday = pd.read_parquet(Path(asof) / "intraday_1min")
            self.assertEqual(sorted(intraday["trade_date"].astype(str)), ["20211231", "20220104"])
            self.assertNotIn("available_at", intraday.columns)
            self.assertIn("auction_correction_rule", intraday.columns)
            # The replay row carries real correction columns, not NaN-backfill.
            self.assertFalse(intraday["auction_correction_rule"].isna().any())
            self.assertFalse(intraday["vol_pit"].isna().any())


if __name__ == "__main__":
    unittest.main()


def test_timeview_releases_auction_only_at_row_availability(tmp_path: Path) -> None:
    replay = pd.DataFrame({
        "ts_code": ["000001.SZ"],
        "trade_date": ["20240102"],
        "price": [10.0],
        "available_at": ["2024-01-02T09:29:00+08:00"],
    })
    view = Timeview(
        host_dir=tmp_path / "asof",
        snapshot_dir=tmp_path / "snapshot",
        replay_frames={"auction": replay},
    )
    _, before = view.refresh(pd.Timestamp("2024-01-02T09:28:00+08:00"))
    assert before == "0"
    _, after = view.refresh(pd.Timestamp("2024-01-02T09:29:00+08:00"))
    assert int(after) > 0
    assert len(pd.read_parquet(tmp_path / "asof" / "auction")) == 1
