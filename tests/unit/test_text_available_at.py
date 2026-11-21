import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autotrade.data_sources.tushare import audit, download, specs, versions
from autotrade.data_sources.tushare import common as core
from autotrade.data_sources.tushare import io as tushare_io
from autotrade.environment.data.snapshot import SnapshotBuilder, SnapshotConfig

CN = "Asia/Shanghai"


def anns_frame(rows):
    return pd.DataFrame(rows, columns=["ann_date", "ts_code", "name", "title", "url", "rec_time"])


def irm_frame(rows):
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "q", "a", "pub_time"])


class TextAvailableAtPlausibilityTest(unittest.TestCase):
    def test_plausible_rec_time_is_kept_as_publication_time(self):
        frame = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "2020-01-10 18:30:00")])
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 18:30:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "source:rec_time")

    def test_backfilled_collection_time_falls_back_to_ann_date_eod(self):
        frame = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "2025-07-05 06:55:56")])
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 23:59:59+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "conservative_from:ann_date:implausible_rec_time")

    def test_rec_time_evening_before_ann_date_is_plausible(self):
        # The nightly title job fetches the next day's titles the evening
        # they arrive: each is visible from its receive time, never earlier.
        frame = anns_frame([("20200111", "000001.SZ", "n", "t", "u", "2020-01-10 20:00:00")])
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 20:00:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "source:rec_time")

    def test_midnight_rec_time_takes_the_lake_receipt_or_the_date(self):
        # Every 2015 rec_time and most official-service rows since 2026-09-14
        # read D 00:00:00: a date, not a receipt time. Such a row takes the
        # launch of the 23:15 landing run that first brought it into the lake
        # when that lies within -1..+3 days of ann_date (the evening before
        # its date, a 05:20 manual rerun), else the end of its date: a
        # backfill first seen years later, or a key the lake has no receipt
        # for. A midnight outside the window stays a collection time; a time
        # a second past midnight is a receipt time whatever the lake saw.
        cn = "Asia/Shanghai"
        frame = anns_frame([
            ("20261027", "601995.SH", "n", "a", "u", "2026-10-27 00:00:00"),
            ("20261030", "300608.SZ", "n", "b", "u", "2026-10-30 00:00:00"),
            ("20150811", "002285.SZ", "n", "c", "u", "2015-08-11 00:00:00"),
            ("20261027", "600276.SH", "n", "d", "u", "2026-10-27 00:00:00"),
            ("20200110", "000001.SZ", "n", "e", "u", "2025-07-05 00:00:00"),
            ("20261027", "601995.SH", "n", "f", "u", "2026-10-26 00:00:01"),
        ])
        first_seen = pd.Series(pd.to_datetime([
            "2026-10-26 23:15:31", "2026-10-31 05:20:07", "2026-10-30 23:40:00", None,
            "2026-10-26 23:15:31", "2026-10-26 23:15:31",
        ]).tz_localize(cn))
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["anns_d"], first_seen)
        self.assertEqual(out["available_at"].tolist(), [
            "2026-10-26 23:15:00+08:00", "2026-10-30 23:15:00+08:00", "2015-08-11 23:59:59+08:00",
            "2026-10-27 23:59:59+08:00", "2020-01-10 23:59:59+08:00", "2026-10-26 00:00:01+08:00",
        ])
        self.assertEqual(out["available_at_rule"].tolist(), [
            "first_seen_for_date_only_rec_time", "first_seen_for_date_only_rec_time",
            "conservative_from:ann_date:date_only_rec_time", "conservative_from:ann_date:date_only_rec_time",
            "conservative_from:ann_date:implausible_rec_time", "source:rec_time",
        ])
        # Without a receipt record every bare date takes the end of its date.
        bare = core.augment_text_frame(frame, specs.TEXT_SPECS["anns_d"])
        self.assertEqual(bare.loc[0, "available_at"], "2026-10-27 23:59:59+08:00")
        self.assertEqual(bare.loc[0, "available_at_rule"], "conservative_from:ann_date:date_only_rec_time")

    def test_missing_rec_time_uses_ann_date_fallback(self):
        frame = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "")])
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["anns_d"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 23:59:59+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "conservative_from:ann_date")

    def test_report_rc_fallback_uses_official_2200_update_window(self):
        frame = pd.DataFrame(
            [{"ts_code": "000001.SZ", "report_date": "20200110", "report_title": "t", "org_name": "o",
              "author_name": "a", "quarter": "2020Q1", "create_time": "2022-05-30 08:00:00"}]
        )
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["report_rc"])
        self.assertEqual(out.loc[0, "available_at"], "2020-01-10 22:00:00+08:00")
        self.assertEqual(out.loc[0, "available_at_rule"], "conservative_from:report_date:implausible_create_time")

    def test_repair_rewrites_only_changed_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The lake keeps replaced partitions beside the raw root
            # (``raw_versions``), so the root is a level below the directory.
            raw = Path(tmp) / "raw"
            dataset_dir = raw / "anns_d"
            dataset_dir.mkdir(parents=True)
            stale = anns_frame([("20200110", "000001.SZ", "n", "t", "u", "2025-07-05 06:55:56")])
            stale["available_at"] = "2025-07-05 06:55:56"
            stale["available_at_rule"] = "source:rec_time"
            tushare_io.write_parquet(
                dataset_dir / "month=202001.parquet", stale, api_name="anns_d", params={}, fields=list(stale.columns)
            )
            stats = core.repair_text_available_at(str(raw), ["anns_d"])
            self.assertEqual(stats["files_rewritten"], 1)
            self.assertEqual(stats["rows_changed"], 1)
            repaired = pd.read_parquet(dataset_dir / "month=202001.parquet")
            self.assertEqual(repaired.loc[0, "available_at"], "2020-01-10 23:59:59+08:00")
            # The rewrite is a restatement: the replaced partition is kept.
            self.assertEqual(len(list((Path(tmp) / "raw_versions").rglob("*-superseded.parquet"))), 1)
            again = core.repair_text_available_at(str(raw), ["anns_d"])
            self.assertEqual(again["files_rewritten"], 0)


class InvestorQaAvailableAtTest(unittest.TestCase):
    """Shenzhen Q&A rows hold the question (asked on trade_date) and its reply
    (published at pub_time); Shanghai rows are dated by the reply and carry the
    vendor's crawl time."""

    def test_sz_reply_days_after_the_question_is_stamped_at_its_pub_time(self):
        frame = irm_frame([
            ("301002.SZ", "20250303", "q1", "a1", "2025-03-11 18:26:27"),
            # A bare reply date is the end of that date.
            ("000657.SZ", "20251107", "q2", "a2", "2026-01-05 00:00:00"),
        ])
        out = core.augment_text_frame(frame, specs.TEXT_SPECS["irm_qa_sz"])
        self.assertEqual(out["available_at"].tolist(), ["2025-03-11 18:26:27+08:00", "2026-01-05 23:59:59+08:00"])
        self.assertEqual(out["available_at_rule"].tolist(), ["source:pub_time", "conservative_from:pub_time"])

    def test_sz_row_without_a_reply_time_never_reaches_a_view(self):
        frame = irm_frame([
            ("301002.SZ", "20250303", "replied", "a", "2025-03-11 18:26:27"),
            ("000001.SZ", "20250303", "unknown", "a", None),
        ])
        stamped = core.augment_text_frame(frame, specs.TEXT_SPECS["irm_qa_sz"])
        config = SnapshotConfig(
            events_datasets=(), macro_datasets=(), fundamental_datasets=(),
            text_datasets=("irm_qa_sz",), intraday_trade_days=1, include_industry=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "irm_qa_sz" / "date=20250303.parquet"
            path.parent.mkdir(parents=True)
            stamped.to_parquet(path, index=False)
            builder = SnapshotBuilder(raw, Path(tmp) / "fund_events_missing")
            titles = {}
            for decision in ("2025-03-04 08:00", "2025-03-12 08:00", "2026-11-21 08:00"):
                out_dir = Path(tmp) / decision[:10]
                out_dir.mkdir()
                index, _ = builder._build_text(
                    config, pd.Timestamp(decision, tz=CN).to_pydatetime(), pd.Timestamp("2025-01-01", tz=CN), out_dir
                )
                titles[decision] = sorted(index["title"])
            self.assertEqual(titles, {
                "2025-03-04 08:00": [], "2025-03-12 08:00": ["replied"], "2026-11-21 08:00": ["replied"],
            })
            # The text audit counts the row as information, not as a blank or
            # unparseable stamp.
            keys = audit.audit_domain_keys([path], specs.TEXT_SPECS["irm_qa_sz"].key_columns, "available_at_total", {str(path): 2})
            self.assertEqual(
                (keys["never_available_rows"], keys["blank_time_fields"], keys["unparseable_available_at_rows"]), (1, 0, 0)
            )

    def test_the_cap_and_the_date_fallback_hold_where_the_source_time_does_not_bind(self):
        # The vendor crawled the 2023-24 Shanghai history on 2025-01-15.
        sh = core.augment_text_frame(irm_frame([
            ("600000.SH", "20230612", "q1", "a1", "2025-01-15 17:06:50"),
            ("600000.SH", "20230612", "q2", "a2", None),
        ]), specs.TEXT_SPECS["irm_qa_sh"])
        self.assertEqual(sh["available_at"].tolist(), ["2023-06-12 23:59:59+08:00"] * 2)
        self.assertEqual(sh["available_at_rule"].tolist(), [
            "conservative_from:trade_date:implausible_pub_time", "conservative_from:trade_date",
        ])
        anns = core.augment_text_frame(anns_frame([
            ("20250303", "000001.SZ", "n", "t1", "u", "2025-03-11 18:26:27"),
            ("20250303", "000001.SZ", "n", "t2", "u", ""),
        ]), specs.TEXT_SPECS["anns_d"])
        self.assertEqual(anns["available_at"].tolist(), ["2025-03-03 23:59:59+08:00"] * 2)
        self.assertEqual(anns["available_at_rule"].tolist(), [
            "conservative_from:ann_date:implausible_rec_time", "conservative_from:ann_date",
        ])


class LakeReceiptStampTest(unittest.TestCase):
    def test_landing_and_repair_stamp_bare_dates_from_the_first_receipt(self):
        """On a lake of the versioned era: the repair restamps a stored bare
        date from its first-seen row; the nightly landing stamps a new key at
        the launch of its run, and the next night's refresh derives the same
        stamp from the recorded row, so it appends and never restates."""
        spec = specs.TEXT_SPECS["anns_d"]
        keys = list(spec.key_columns)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            path = raw / "anns_d" / "month=202610.parquet"
            path.parent.mkdir(parents=True)
            (raw / ".raw_generation.json").write_text(json.dumps({
                "schema_version": 3, "state": "committed", "generation_id": "g", "versioned_since": "20261001",
            }), encoding="utf-8")
            evening = anns_frame([("20261027", "601995.SH", "n", "a", "u1", "2026-10-27 00:00:00")])

            def land(answer: pd.DataFrame, instant: str) -> str:
                with patch.object(versions, "utc_now", return_value=instant), redirect_stdout(io.StringIO()):
                    stamped = core.augment_text_frame(answer, spec, download.landing_first_seen(path, answer, keys))
                    return versions.publish(path, stamped, api_name="anns_d", params={}, fields=list(stamped.columns),
                                            key_columns=keys, ledger=None, deduplicate=True)

            # The 23:15 run of 10-26 lands a title dated 10-27, stored under
            # the old rule (its midnight read as a receipt time).
            stale = evening.assign(available_at="2026-10-27 00:00:00+08:00", available_at_rule="source:rec_time")
            with patch.object(versions, "utc_now", return_value="2026-10-26T15:15:31+00:00"), redirect_stdout(io.StringIO()):
                versions.publish(path, stale, api_name="anns_d", params={}, fields=list(stale.columns),
                                 key_columns=keys, ledger=None)
            stats = core.repair_text_available_at(str(raw), ["anns_d"])
            self.assertEqual((stats["files_rewritten"], stats["rows_changed"]), (1, 1))
            stored = pd.read_parquet(path)
            self.assertEqual(stored.loc[0, "available_at"], "2026-10-26 23:15:00+08:00")
            self.assertEqual(stored.loc[0, "available_at_rule"], "first_seen_for_date_only_rec_time")
            # The 23:15 run of 10-27 refetches it and lands a second bare date.
            later = pd.concat([evening, anns_frame([("20261027", "600276.SH", "n", "b", "u2", "2026-10-27 00:00:00")])],
                              ignore_index=True)
            self.assertEqual(land(later, "2026-10-27T15:15:40+00:00"), "appended")
            stored = pd.read_parquet(path).set_index("ts_code")
            self.assertEqual(stored["available_at"].to_dict(), {
                "601995.SH": "2026-10-26 23:15:00+08:00", "600276.SH": "2026-10-27 23:15:00+08:00",
            })
            self.assertEqual(land(later, "2026-10-28T15:15:12+00:00"), "unchanged")
            self.assertEqual(core.repair_text_available_at(str(raw), ["anns_d"])["files_rewritten"], 0)


if __name__ == "__main__":
    unittest.main()
