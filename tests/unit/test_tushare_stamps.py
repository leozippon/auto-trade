"""The point-in-time stamping rules (data_sources/tushare/stamps.py)."""

import json
import os
import unittest
from pathlib import Path

import pandas as pd

from autotrade.data_sources.tushare import io as tushare_io
from autotrade.data_sources.tushare import specs, stamps, versions

from .tushare_fixture import LakeTestCase


class StampTest(LakeTestCase):
    def test_stk_surv_calendar_and_new_share_trading_day_offsets(self):
        # stk_surv rows keep landing for days after surv_date (the source has
        # no announcement column) and new_share's ballot is announced 1-2
        # trading days after ipo_date: both stamps must clear a weekend and a
        # month boundary instead of using the source date's own EOD.
        surv = stamps.augment_event_frame(
            pd.DataFrame([{"ts_code": "000001.SZ", "surv_date": "20260828"}]),
            specs.EVENT_FLOW_SPECS["stk_surv"],
        )
        self.assertEqual(surv.loc[0, "available_at"], "2026-09-02 23:59:59+08:00")
        self.assertEqual(surv.loc[0, "available_at_rule"], "conservative_plus_5d_eod_from:surv_date")

        ipo = stamps.augment_event_frame(
            pd.DataFrame([{"ts_code": "301000.SZ", "ipo_date": "20260731", "ballot": "0.02"}]),
            specs.EVENT_FLOW_SPECS["new_share"],
            trading_dates=["20260803", "20260804"],
        )
        self.assertEqual(ipo.loc[0, "available_at"], "2026-08-04 23:59:59+08:00")
        self.assertEqual(ipo.loc[0, "available_at_rule"], "conservative_tplus_2_eod_from:ipo_date")

        holiday = stamps.augment_event_frame(
            pd.DataFrame([{"ts_code": "301380.SZ", "ipo_date": "20220930", "ballot": "0.03"}]),
            specs.EVENT_FLOW_SPECS["new_share"],
            trading_dates=["20220930", "20221010", "20221011"],
        )
        self.assertEqual(holiday.loc[0, "available_at"], "2022-10-11 23:59:59+08:00")

        with self.assertRaisesRegex(RuntimeError, "requires the A-share trading calendar"):
            stamps.augment_event_frame(
                pd.DataFrame([{"ts_code": "301380.SZ", "ipo_date": "20220930"}]),
                specs.EVENT_FLOW_SPECS["new_share"],
            )

    def test_event_date_column_is_stored_without_padding(self):
        # A padded ann_date would be a second spelling of the business key and
        # break any strict date parse of the column; a missing one stays missing.
        rows = pd.DataFrame([
            {"ts_code": "000004.SZ", "ann_date": "20250512 ", "end_date": "20250509", "holder_num": 30282},
            {"ts_code": "000005.SZ", "ann_date": None, "end_date": "20250509", "holder_num": 1},
        ])
        out = stamps.augment_event_frame(rows, specs.EVENT_FLOW_SPECS["stk_holdernumber"])
        self.assertEqual(out.loc[0, "ann_date"], "20250512")
        self.assertIsNone(out.loc[1, "ann_date"])
        self.assertEqual(out.loc[0, "available_at"], "2025-05-12 23:59:59+08:00")
        self.assertEqual(rows.loc[0, "ann_date"], "20250512 ")

    def test_margin_rows_become_visible_at_09_on_the_next_open_day(self):
        # The vendor serves T's SZSE/BSE margin rows with the next session's
        # pre-open: a Friday stays invisible to a Monday 08:30 decision, a
        # pre-holiday session until the first morning after the holiday, and an
        # ordinary Tuesday is visible on Wednesday morning as before.
        calendar = ["20260929", "20260930", "20261008", "20261009", "20261012"]

        def stamp(dataset: str, trade_date: str) -> pd.Timestamp:
            rows = pd.DataFrame([{"trade_date": trade_date, "exchange_id": ex, "ts_code": f"60000{i}.SH"}
                                 for i, ex in enumerate(("SSE", "SZSE", "BSE"))])
            stamped = stamps.augment_event_frame(rows, specs.EVENT_FLOW_SPECS[dataset], trading_dates=calendar)
            self.assertEqual(set(stamped["available_at_rule"]), {"official_next_trading_day_09_from:trade_date"})
            self.assertEqual(stamped["available_at"].nunique(), 1)
            return pd.Timestamp(stamped.loc[0, "available_at"])

        def at(text: str) -> pd.Timestamp:
            return pd.Timestamp(f"{text}+08:00")

        friday = stamp("margin", "20261009")
        self.assertGreater(friday, at("2026-10-12 08:30"))
        self.assertLessEqual(friday, at("2026-10-12 09:00"))
        pre_holiday = stamp("margin_detail", "20260930")
        self.assertGreater(pre_holiday, at("2026-10-08 08:30"))
        self.assertLessEqual(pre_holiday, at("2026-10-08 09:00"))
        tuesday = stamp("margin", "20260929")
        self.assertEqual(tuesday, at("2026-09-30 09:00"))

        with self.assertRaisesRegex(RuntimeError, "fewer than 1 open dates after 20261012"):
            stamp("margin", "20261012")
        with self.assertRaisesRegex(RuntimeError, "requires the A-share trading calendar"):
            stamps.augment_event_frame(pd.DataFrame([{"trade_date": "20260929"}]), specs.EVENT_FLOW_SPECS["margin"])

    def test_event_available_at_repair_restamps_through_new_files(self):
        # Releases hard-link raw partitions: the repair must replace the live
        # path with a new file, never modify the shared one, and re-run clean.
        cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        cal.parent.mkdir(parents=True, exist_ok=True)
        open_days = {"20260929", "20260930", "20261008", "20261009", "20261012"}
        pd.DataFrame([
            {"cal_date": f"2026{m:02d}{d:02d}", "is_open": "1" if f"2026{m:02d}{d:02d}" in open_days else "0"}
            for m, d in [(9, 29), (9, 30)] + [(10, d) for d in range(1, 13)]
        ]).to_parquet(cal, index=False)
        spec = specs.EVENT_FLOW_SPECS["margin"]
        old_rule = "official_next_day_09_from:trade_date"

        def land_old(trade_date: str, next_day: str) -> Path:
            rows = pd.DataFrame([
                {**{col: 1.0 for col in spec.fields.split(",")}, "trade_date": trade_date, "exchange_id": ex,
                 "available_at": f"{next_day[:4]}-{next_day[4:6]}-{next_day[6:]} 09:00:00+08:00", "available_at_rule": old_rule}
                for ex in ("SSE", "SZSE", "BSE")
            ])
            path = self.raw_dir / "margin" / f"trade_date={trade_date}.parquet"
            tushare_io.write_parquet(path, rows, api_name="margin", params={"trade_date": trade_date}, fields=list(rows.columns))
            return path

        land_old("20260929", "20260930")  # same stamp under both rules, old label
        friday = land_old("20261009", "20261010")
        fetched_at = tushare_io.parquet_meta(friday)["fetched_at"]
        release_copy = self.root / "release_margin.parquet"
        os.link(friday, release_copy)

        stats = stamps.repair_event_available_at(str(self.raw_dir), ["margin"])
        self.assertEqual((stats["files_rewritten"], stats["rows_changed"]), (2, 3))
        live = pd.read_parquet(friday)
        self.assertEqual(set(live["available_at"]), {"2026-10-12 09:00:00+08:00"})
        self.assertEqual(set(live["available_at_rule"]), {"official_next_trading_day_09_from:trade_date"})
        self.assertEqual(live["rzye"].tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(tushare_io.committed_partition_intact(friday))
        # A restamp is a restatement: the replaced partition is kept as a
        # superseded version, and the release's hard link keeps the old bytes.
        self.assertNotEqual(tushare_io.parquet_meta(friday)["fetched_at"], fetched_at)
        (superseded,) = versions.list_versions(friday)
        self.assertTrue(superseded.endswith("-superseded"))
        self.assertEqual(set(pd.read_parquet(release_copy)["available_at"]), {"2026-10-10 09:00:00+08:00"})
        self.assertEqual(stamps.repair_event_available_at(str(self.raw_dir), ["margin"])["files_rewritten"], 0)

        meta_path = friday.with_suffix(".parquet.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_path.write_text(json.dumps({**meta, "write_id": "torn"}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not a complete commit"):
            stamps.repair_event_available_at(str(self.raw_dir), ["margin"])

    def test_text_source_time_is_normalized_to_china_timezone(self):
        frame = pd.DataFrame([{"title": "sample", "pub_time": "2020-01-02 10:00:00", "src": "x"}])
        out = stamps.augment_text_frame(frame, specs.TEXT_SPECS["major_news"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-02 10:00:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "source:pub_time")


if __name__ == "__main__":
    unittest.main()
