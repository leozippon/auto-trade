import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autotrade.data_sources.tushare import common as core


def anns_frame(rows):
    return pd.DataFrame(rows, columns=["ann_date", "ts_code", "name", "title", "url", "rec_time"])


class TextAvailableAtPlausibilityTest(unittest.TestCase):
    def test_plausible_rec_time_is_kept_as_publication_time(self):
        frame = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "2020-01-10 18:30:00")])
        out = core.augment_text_frame(frame, core.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 18:30:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "source:rec_time")

    def test_backfilled_collection_time_falls_back_to_ann_date_eod(self):
        frame = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "2025-07-05 06:55:56")])
        out = core.augment_text_frame(frame, core.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 23:59:59+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "conservative_from:ann_date:implausible_rec_time")

    def test_rec_time_evening_before_ann_date_is_plausible(self):
        frame = anns_frame([("20200111", "000001.SZ", "n", "t", "u", "2020-01-10 20:00:00")])
        out = core.augment_text_frame(frame, core.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at_rule"], "source:rec_time")

    def test_missing_rec_time_uses_ann_date_fallback(self):
        frame = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "")])
        out = core.augment_text_frame(frame, core.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 23:59:59+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "conservative_from:ann_date")

    def test_report_rc_fallback_uses_official_2200_update_window(self):
        frame = pd.DataFrame(
            [{"ts_code": "000001.SZ", "report_date": "20200110", "report_title": "t", "org_name": "o",
              "author_name": "a", "quarter": "2020Q1", "create_time": "2022-05-30 08:00:00"}]
        )
        out = core.augment_text_frame(frame, core.TEXT_SPECS["report_rc"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 22:00:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "conservative_from:report_date:implausible_create_time")

    def test_repair_rewrites_only_changed_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            dataset_dir = raw / "anns_d"
            dataset_dir.mkdir()
            stale = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "2025-07-05 06:55:56")])
            stale["available_at"] = "2025-07-05 06:55:56"
            stale["available_at_rule"] = "source:rec_time"
            core.write_parquet(
                dataset_dir / "month=202001.parquet", stale, api_name="anns_d", params={}, fields=list(stale.columns)
            )
            stats = core.repair_text_available_at(str(raw), ["anns_d"])
            self.assertEqual(stats["files_rewritten"], 1)
            self.assertEqual(stats["rows_changed"], 1)
            repaired = pd.read_parquet(dataset_dir / "month=202001.parquet")
            self.assertEqual(repaired.loc[0, "available_at"], "2020-01-10 23:59:59+08:00")
            again = core.repair_text_available_at(str(raw), ["anns_d"])
            self.assertEqual(again["files_rewritten"], 0)


if __name__ == "__main__":
    unittest.main()
