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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotrade.environment.replay.timeview import ReplayRows, Timeview

CN_TZ = ZoneInfo("Asia/Shanghai")
TS = "000001.SZ"


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
                # Text rolls only on the 23:15 evening text node (ready 23:30):
                # this row lands that evening, the next one the evening after.
                "available_at": "2022-01-04T22:00:00+08:00",
                "library_file": "news.parquet",
            },
            {
                "text_id": "news_late",
                "dataset": "news",
                "ts_codes": TS,
                "title": "late",
                "available_at": "2022-01-05T22:00:00+08:00",
                "library_file": "news.parquet",
            },
        ]
    )
    return {"daily": daily, "events": events, "fundamentals": fundamentals, "text_index": text_index}


def _replay_rows(root: Path, frames: dict[str, pd.DataFrame] | None = None) -> dict[str, ReplayRows]:
    """The replay frames as the PIT backend hands them over: slot files opened for the roll."""
    rows: dict[str, ReplayRows] = {}
    for name, frame in (_replay_frames() if frames is None else frames).items():
        path = root / "replay" / f"{name}.parquet"
        _write(path, frame)
        rows[name] = ReplayRows(path)
    return rows


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
            snapshot_dir=_frozen_snapshot(root),
            replay=_replay_rows(root),
            replay_text_library_dir=replay_library,
        )

    def _dates(self, asof_dir: str, domain: str) -> set[str]:
        frame = pd.read_parquet(Path(asof_dir) / domain)
        col = "trade_date" if "trade_date" in frame.columns else "available_at"
        return set(frame[col].astype(str)) if col in frame.columns else set()

    def test_frozen_base_is_part_zero_and_today_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            tv = self._build(Path(tmp))
            asof, _version = tv.refresh(_when("2022-01-04 09:10:00"))
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
                host_dir=root / "asof",
                snapshot_dir=snapshot, replay=_replay_rows(root, {"events": replay}),
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
                host_dir=root / "asof",
                snapshot_dir=snapshot, replay=_replay_rows(root, {"daily": replay}),
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
                    host_dir=root / "asof",
                    snapshot_dir=snapshot, replay=_replay_rows(root, {"daily": replay}),
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
                snapshot_dir=snapshot,
                replay=_replay_rows(root, {"intraday_1min": pd.concat([day1, day2], ignore_index=True)}),
            )
            incremental = Timeview(
                host_dir=root / "incremental",
                snapshot_dir=snapshot,
                replay={},
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

            asof2, _ = tv.refresh(_when("2022-01-04 23:31:00"))
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

            # One clock jump crosses the 09:05 margin node and the 23:30
            # evening text node. Both cursors catch up to the latest eligible
            # cutoff in a single refresh, while the evening events node (next
            # day 03:05) has not completed, so block_trade stays invisible.
            asof, _ = tv.refresh(_when("2022-01-04 23:31:00"))
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
                    replay=_replay_rows(root),
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
                    snapshot_dir=snapshot,
                    replay=_replay_rows(root / f"slot_{index}"),
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

    def _text_stash(self, root: Path) -> Path:
        """A stash directory with the contract the PIT backend publishes."""

        stash = root / "stash"
        stash.mkdir(parents=True, exist_ok=True)
        (stash / "contract.json").write_text(json.dumps({"validated": True}), encoding="utf-8")
        return stash

    def _text_replay_library(self, root: Path) -> Path:
        library = root / "replay" / "text_library"
        _write(
            library / "news.parquet",
            pd.DataFrame(
                [
                    {"text_id": "news_early", "body": "early body"},
                    {"text_id": "news_late", "body": "late body"},
                ]
            ),
        )
        return library

    def _rolled_text(self, root: Path, name: str, *, stash: Path | None) -> Path:
        view = Timeview(
            host_dir=root / name,
            snapshot_dir=_frozen_snapshot(root),
            replay=_replay_rows(root / name),
            replay_text_library_dir=self._text_replay_library(root),
            stash_dir=stash,
        )
        # One refresh past every text node of the window: the whole replay
        # index becomes visible in a single roll.
        asof, _ = view.refresh(_when("2022-01-06 09:10:00"))
        return Path(asof)

    def test_text_parts_are_stashed_once_and_reused_by_the_next_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stash = self._text_stash(root)
            first = self._rolled_text(root, "asof_1", stash=stash)
            index_part = stash / "text_index" / "part_0001.parquet"
            body_part = stash / "text_library" / "news__part_0001.parquet"
            self.assertTrue(index_part.is_file() and body_part.is_file())
            self.assertEqual(len(pd.read_parquet(index_part)), 2)
            published = (index_part.stat().st_ino, body_part.stat().st_ino)

            second = self._rolled_text(root, "asof_2", stash=stash)
            # The second candidate hardlinks the very same parts: nothing in
            # the stash was rewritten, and no library read happened at all.
            self.assertEqual(
                (index_part.stat().st_ino, body_part.stat().st_ino), published
            )
            for run in (first, second):
                self.assertEqual(
                    (run / "text_index" / "part_0001.parquet").stat().st_ino,
                    index_part.stat().st_ino,
                )
                self.assertEqual(
                    (run / "text_library" / "news__part_0001.parquet").stat().st_ino,
                    body_part.stat().st_ino,
                )
            pd.testing.assert_frame_equal(
                pd.read_parquet(first / "text_index"), pd.read_parquet(second / "text_index")
            )
            self.assertEqual(list(stash.rglob("*.tmp")), [])

    def test_a_stashed_text_index_part_with_the_wrong_row_count_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stash = self._text_stash(root)
            self._rolled_text(root, "asof_1", stash=stash)
            index_part = stash / "text_index" / "part_0001.parquet"
            # A stash bound to different semantics would cut a different
            # visibility slice; the footer row count must fail, not be read.
            pd.read_parquet(index_part).head(1).to_parquet(index_part, index=False)
            with self.assertRaisesRegex(RuntimeError, "text index part_0001"):
                self._rolled_text(root, "asof_2", stash=stash)

    def test_stashed_text_parts_are_byte_identical_to_the_per_row_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asof = self._rolled_text(root, "asof", stash=None)
            index_part = asof / "text_index" / "part_0001.parquet"
            body_part = asof / "text_library" / "news__part_0001.parquet"

            # The writer this replaced: a per-row library_file relabel and two
            # pandas.to_parquet calls over the same newly-visible slice.
            rows = _replay_frames()["text_index"].reset_index(drop=True)
            self.assertEqual(len(pd.read_parquet(index_part)), len(rows))
            legacy = root / "legacy"
            legacy.mkdir()
            library_files: dict[tuple[str, str], str] = {}
            for dataset, group in rows.groupby(rows["dataset"].astype(str), sort=True):
                part_name = f"{dataset}__part_0001.parquet"
                body = pd.read_parquet(
                    self._text_replay_library(root) / f"{dataset}.parquet",
                    filters=[("text_id", "in", sorted(set(group["text_id"].astype(str))))],
                )
                body[["text_id", "body"]].to_parquet(legacy / part_name, index=False)
                for source_file in group["library_file"].astype(str).unique():
                    library_files[(dataset, source_file)] = part_name
            rows["library_file"] = [
                library_files.get(
                    (str(row.get("dataset", "")), str(row.get("library_file", ""))),
                    str(row.get("library_file", "")),
                )
                for _, row in rows.iterrows()
            ]
            rows.to_parquet(legacy / "part_0001.parquet", index=False)

            self.assertEqual(index_part.read_bytes(), (legacy / "part_0001.parquet").read_bytes())
            self.assertEqual(
                body_part.read_bytes(), (legacy / "news__part_0001.parquet").read_bytes()
            )

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
                snapshot_dir=snapshot,
                replay=_replay_rows(root, frames),
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
            tv = Timeview(host_dir=root / "asof", snapshot_dir=snap, replay=_replay_rows(root, replay))

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
            tv = Timeview(host_dir=root / "asof", snapshot_dir=snap, replay=_replay_rows(root, replay))
            # After the 20220104 evening node completes (fallback ~03:05 on 0105) the replay bar rolls in.
            asof, _ = tv.refresh(_when("2022-01-05 09:10:00"))
            intraday = pd.read_parquet(Path(asof) / "intraday_1min")
            self.assertEqual(sorted(intraday["trade_date"].astype(str)), ["20211231", "20220104"])
            self.assertNotIn("available_at", intraday.columns)
            self.assertIn("auction_correction_rule", intraday.columns)
            # The replay row carries real correction columns, not NaN-backfill.
            self.assertFalse(intraday["auction_correction_rule"].isna().any())
            self.assertFalse(intraday["vol_pit"].isna().any())


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
        replay=_replay_rows(tmp_path, {"auction": replay}),
    )
    _, before = view.refresh(pd.Timestamp("2024-01-02T09:28:00+08:00"))
    assert before == "0"
    _, after = view.refresh(pd.Timestamp("2024-01-02T09:29:00+08:00"))
    assert int(after) > 0
    assert len(pd.read_parquet(tmp_path / "asof" / "auction")) == 1


def _events_slot(days: list[str], *, scores: bool) -> pd.DataFrame:
    """One events slot laid out by dataset, as the builder writes it.

    A day's two rows therefore sit far apart in the file. ``note`` has no
    value on any morning roll (a string column the slice makes null-typed),
    ``flag`` is a bool column with nulls, and ``score`` has no value anywhere
    in a slot without ``scores`` (a null-typed file column) but values in one
    with them.
    """

    rows = []
    for dataset, clock, note in (("margin_secs", "09:00:00", None), ("block_trade", "21:00:00", "late")):
        for number, day in enumerate(days):
            rows.append(
                {
                    "dataset": dataset,
                    "ts_code": TS,
                    "trade_date": day,
                    "available_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T{clock}+08:00",
                    "note": note,
                    "flag": [True, False, None][number % 3],
                    "score": float(number) if scores else None,
                }
            )
    return pd.DataFrame(rows)


def _events_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    _write(
        snapshot / "events.parquet",
        pd.DataFrame(
            [{
                "dataset": "block_trade", "ts_code": TS, "trade_date": "20211231",
                "available_at": "2021-12-31T21:00:00+08:00", "note": "frozen", "flag": True, "score": 0.5,
            }]
        ),
    )
    return snapshot


def _events_rows(root: Path, name: str, frame: pd.DataFrame) -> dict[str, ReplayRows]:
    path = root / name / "events.parquet"
    path.parent.mkdir(parents=True)
    # Two-row row groups: every day's rows span row groups, and a part from
    # two slots spans files.
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=2)
    return {"events": ReplayRows(path)}


def _roll_events_across_a_boundary(root: Path, host: str) -> dict[str, bytes]:
    """Parts of slot A's rows, then of a part mixing A's evening rows with B's morning rows."""

    view = Timeview(
        host_dir=root / host,
        snapshot_dir=_events_snapshot(root),
        replay=_events_rows(root, f"{host}_a", _events_slot(["20220104", "20220105"], scores=False)),
    )
    for instant in ("2022-01-04 09:10:00", "2022-01-05 03:10:00", "2022-01-05 09:10:00"):
        view.refresh(_when(instant))
    view.continue_into(
        lambda: _events_rows(root, f"{host}_b", _events_slot(["20220106", "20220107"], scores=True)),
        universe_file=None,  # this snapshot family has no universe domain
        replay_text_library_dir=None,
        stash_dir=None,
    )
    for instant in ("2022-01-06 09:10:00", "2022-01-07 03:10:00"):
        view.refresh(_when(instant))
    parts = sorted((root / host / "events").glob("part_*.parquet"))
    assert [path.name for path in parts] == [f"part_{index:04d}.parquet" for index in range(6)]
    return {path.name: path.read_bytes() for path in parts}


def test_parts_are_byte_identical_to_slices_of_the_decoded_frame(tmp_path: Path, monkeypatch) -> None:
    """The row groups a roll takes encode exactly what a slice of the whole decoded frame did.

    The eager reference decodes each slot whole and slices it, as the view
    used to; the rows it never reads are the whole point of the lazy path.
    """

    lazy = _roll_events_across_a_boundary(tmp_path, "lazy")

    def eager_take(self: ReplayRows, indices):
        yield pa.Table.from_pandas(pd.read_parquet(self.path).iloc[indices], preserve_index=False)

    monkeypatch.setattr(ReplayRows, "take", eager_take)
    assert _roll_events_across_a_boundary(tmp_path, "eager") == lazy
    morning = pq.read_table(tmp_path / "lazy" / "events" / "part_0001.parquet")
    assert morning.schema.field("note").type == pa.null()  # a slice with no value stays null-typed
    boundary = pq.read_table(tmp_path / "lazy" / "events" / "part_0004.parquet")
    assert boundary["trade_date"].to_pylist() == ["20220105", "20220106"]  # A's evening row first
    assert boundary.schema.field("score").type == pa.float64()  # A's null column took B's type


def test_slices_of_two_slots_unify_as_their_decoded_frames_concatenate(tmp_path: Path) -> None:
    a = _events_rows(tmp_path, "a", _events_slot(["20220104"], scores=False))["events"]
    b = _events_rows(tmp_path, "b", _events_slot(["20220105"], scores=True))["events"]
    lazy = pa.concat_tables([*a.take(np.array([1])), *b.take(np.array([0]))], promote_options="default")
    with warnings.catch_warnings():
        # The reference is the legacy pandas concat, all-NA columns included.
        warnings.simplefilter("ignore", FutureWarning)
        eager = pa.Table.from_pandas(
            pd.concat([pd.read_parquet(a.path).iloc[[1]], pd.read_parquet(b.path).iloc[[0]]], ignore_index=True),
            preserve_index=False,
        )
    assert lazy.schema.remove_metadata() == eager.schema.remove_metadata()
    assert lazy.to_pylist() == eager.to_pylist()


def test_a_stash_hit_reads_no_replay_rows(tmp_path: Path, monkeypatch) -> None:
    stash = tmp_path / "stash"
    stash.mkdir()
    (stash / "contract.json").write_text(json.dumps({"validated": True}), encoding="utf-8")
    frames = {name: frame for name, frame in _replay_frames().items() if name != "text_index"}

    def rolled(host: str) -> dict[str, int]:
        view = Timeview(
            host_dir=tmp_path / host,
            snapshot_dir=_frozen_snapshot(tmp_path),
            replay=_replay_rows(tmp_path / host, frames),
            stash_dir=stash,
        )
        asof, _ = view.refresh(_when("2022-01-06 09:10:00"))
        return {
            str(path.relative_to(asof)): path.stat().st_ino
            for path in sorted(Path(asof).rglob("part_*.parquet"))
        }

    first = rolled("first")
    assert any(key.startswith("events/part_") for key in first)

    def refuse(self: ReplayRows, indices):
        raise AssertionError(f"a stash hit read {self.path.name}")

    monkeypatch.setattr(ReplayRows, "take", refuse)
    assert rolled("second") == first  # the same inodes: hardlinked, never re-encoded


def test_an_incremental_domain_takes_partitions_only(tmp_path: Path) -> None:
    replay = _replay_rows(
        tmp_path,
        {"intraday_1min": pd.DataFrame([{"ts_code": TS, "close": 1.0, "available_at": "2022-01-04T15:00:00+08:00"}])},
    )
    with pytest.raises(ValueError, match="incremental and takes partitions only"):
        Timeview(
            host_dir=tmp_path / "asof",
            snapshot_dir=tmp_path / "snapshot",
            replay=replay,
            incremental_domains={"intraday_1min"},
        )


# Two universe vintages a year apart, as the decision snapshots of two
# consecutive slots carry them: one rename into ST, one industry
# reclassification, one code listed inside the first slot.
_VINTAGE_A = [
    {"ts_code": TS, "name": "平安银行", "l1_name": "银行"},
    {"ts_code": "600705.SH", "name": "中航资本", "l1_name": "非银金融"},
]
_VINTAGE_B = [
    {"ts_code": TS, "name": "*ST平安", "l1_name": "银行"},
    {"ts_code": "600705.SH", "name": "中航资本", "l1_name": "综合"},
    {"ts_code": "301263.SZ", "name": "泰恩康", "l1_name": "医药生物"},
]


def _vintage(path: Path, rows: list[dict[str, object]]) -> Path:
    _write(path, pd.DataFrame(rows))
    return path


def _universe_view(root: Path, host: str = "asof") -> tuple[Timeview, Path]:
    """A view over slot A whose snapshot carries vintage A, plus vintage B on disk."""

    snapshot = _events_snapshot(root)
    _vintage(snapshot / "universe.parquet", _VINTAGE_A)
    view = Timeview(
        host_dir=root / host,
        snapshot_dir=snapshot,
        replay=_events_rows(root, f"{host}_a", _events_slot(["20220104", "20220105"], scores=False)),
    )
    return view, _vintage(root / f"{host}_b" / "universe.parquet", _VINTAGE_B)


def _continue(view: Timeview, root: Path, host: str, universe_file: Path | None) -> None:
    view.continue_into(
        lambda: _events_rows(root, f"{host}_next", _events_slot(["20220106"], scores=True)),
        universe_file=universe_file,
        replay_text_library_dir=None,
        stash_dir=None,
    )


def test_the_universe_rolls_to_the_vintage_of_the_slot_being_opened(tmp_path: Path) -> None:
    """Crossing a slot boundary shows the names, ST flags, industries and
    listings of that slot's anchor -- as one vintage, not two concatenated."""

    view, vintage_b = _universe_view(tmp_path)
    asof, before = view.refresh(_when("2022-01-05 09:10:00"))
    frozen = pd.read_parquet(Path(asof) / "universe")
    assert list(frozen["ts_code"]) == [TS, "600705.SH"]
    assert frozen.loc[frozen["ts_code"] == TS, "name"].item() == "平安银行"

    _continue(view, tmp_path, "asof", vintage_b)
    asof, after = view.refresh(_when("2022-01-06 09:10:00"))
    rolled = pd.read_parquet(Path(asof) / "universe")
    assert list(rolled["ts_code"]) == [TS, "600705.SH", "301263.SZ"]  # listed inside slot A
    assert rolled.loc[rolled["ts_code"] == TS, "name"].item() == "*ST平安"  # renamed inside slot A
    assert rolled.loc[rolled["ts_code"] == "600705.SH", "l1_name"].item() == "综合"  # reclassified
    assert len(list((Path(asof) / "universe").glob("*.parquet"))) == 1  # replaced, not appended
    assert int(after) > int(before)  # a feature cached on the old names is invalidated


def test_no_refresh_shows_a_universe_vintage_its_slot_has_not_reached(tmp_path: Path) -> None:
    """Only opening the next slot publishes that slot's vintage.

    The later vintage is already on disk here; every refresh inside slot A must
    still show slot A's anchor and nothing dated after it.
    """

    view, _vintage_b = _universe_view(tmp_path)
    for instant in ("2022-01-04 09:10:00", "2022-01-05 03:10:00", "2022-01-05 09:10:00", "2022-01-06 09:10:00"):
        asof, _version = view.refresh(_when(instant))
        visible = pd.read_parquet(Path(asof) / "universe")
        assert list(visible["ts_code"]) == [TS, "600705.SH"]
        assert set(visible["name"]) == {"平安银行", "中航资本"}
        assert set(visible["l1_name"]) == {"银行", "非银金融"}


def test_a_slot_without_a_usable_universe_vintage_is_refused(tmp_path: Path) -> None:
    """A view that exposes the domain never keeps the previous slot's vintage."""

    view, vintage_b = _universe_view(tmp_path)
    view.refresh(_when("2022-01-05 09:10:00"))
    with pytest.raises(ValueError, match="stale names"):
        _continue(view, tmp_path, "asof", None)
    with pytest.raises(FileNotFoundError, match="universe vintage is missing"):
        _continue(view, tmp_path, "asof", tmp_path / "absent" / "universe.parquet")
    thin = _vintage(tmp_path / "thin" / "universe.parquet", [{"ts_code": TS, "name": "平安银行"}])
    with pytest.raises(RuntimeError, match=r"drops published columns \['l1_name'\]"):
        _continue(view, tmp_path, "asof", thin)
    # The refused rolls left the published vintage untouched.
    asof, _version = view.refresh(_when("2022-01-05 09:20:00"))
    assert list(pd.read_parquet(Path(asof) / "universe")["ts_code"]) == [TS, "600705.SH"]
    _continue(view, tmp_path, "asof", vintage_b)  # and a good vintage still rolls
    asof, _version = view.refresh(_when("2022-01-06 09:10:00"))
    assert "301263.SZ" in set(pd.read_parquet(Path(asof) / "universe")["ts_code"])


def test_a_part_larger_than_a_row_group_streams_into_the_row_groups_one_write_would_cut(
    tmp_path: Path, monkeypatch
) -> None:
    """The streamed file is the file ``write_table`` writes for the whole part."""

    from autotrade.environment.replay import timeview as module

    monkeypatch.setattr(module, "_PART_ROW_GROUP_ROWS", 3)
    frame = _events_slot(["20220104", "20220105", "20220106", "20220107"], scores=True)
    view = Timeview(
        host_dir=tmp_path / "asof",
        snapshot_dir=_events_snapshot(tmp_path),
        replay=_events_rows(tmp_path, "slot", frame),
    )
    asof, _ = view.refresh(_when("2022-01-08 03:10:00"))  # every row of the slot in one part
    part = Path(asof) / "events" / "part_0001.parquet"
    assert pq.ParquetFile(part).metadata.num_row_groups == 3  # 8 rows as 3 + 3 + 2
    whole = pa.Table.from_pandas(pd.read_parquet(tmp_path / "slot" / "events.parquet"), preserve_index=False)
    columns = pq.ParquetFile(tmp_path / "snapshot" / "events.parquet").schema_arrow.names
    reference = tmp_path / "reference.parquet"
    pq.write_table(
        pa.Table.from_arrays([whole[name] for name in columns], schema=pa.schema([whole.schema.field(name) for name in columns])),
        reference,
        row_group_size=3,
    )
    assert part.read_bytes() == reference.read_bytes()
