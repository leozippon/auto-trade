"""share_float's pulls, per-code rescue and union file
(data_sources/tushare/share_float.py).
"""

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pyarrow.lib import ArrowInvalid

from autotrade.data_sources.tushare import client as tushare_client
from autotrade.data_sources.tushare import (
    common,
    download,
    share_float,
    specs,
    stamps,
    versions,
)
from autotrade.data_sources.tushare import io as tushare_io

from .support.tushare_fixture import EmptyClient, LakeTestCase


class ShareFloatTest(LakeTestCase):
    def test_share_float_union_inherits_baseline_and_prefers_refreshed_rows(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "000002.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h2", "share_type": "t", "float_share": 2.0, "float_ratio": 0.2},
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20200101",
            ann_end_date="20200102",
            float_start_date="20200101",
            float_end_date="20200102",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(share_float, "share_float_union_files", return_value=[]):
            report = {}
            share_float.write_share_float_union(self.raw_dir, args, report)
        self.assertEqual(tushare_io.parquet_rows(output), 2)
        self.assertTrue(report["union"]["baseline_inherited"])

        # Refreshed source rows precede the baseline and replace matching keys.
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        rows = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 10.0, "float_ratio": 1.0},
            {"ts_code": "000002.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h2", "share_type": "t", "float_share": 20.0, "float_ratio": 2.0},
        ])
        tushare_io.write_parquet(source, rows, api_name="share_float", params={}, fields=list(rows.columns))
        report = {}
        with patch.object(share_float, "share_float_union_files", return_value=[(source, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, report)
        self.assertEqual(report["union"]["previous_rows"], 2)
        self.assertEqual(report["union"]["rows_after_dedup"], 2)
        updated = pd.read_parquet(output).sort_values("ts_code")
        self.assertEqual(updated["float_share"].tolist(), [10.0, 20.0])
        self.assertTrue((self.root / "revision_events.jsonl").exists())

    def test_share_float_union_reads_baseline_once_and_republishes_active_content(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        baseline = pd.DataFrame([{
            "ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
            "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1,
        }])
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102", skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        def publish(value: float) -> str:
            # Each round starts from a fresh baseline landing, which has no
            # observation table yet.
            versions.observation_path(output).unlink(missing_ok=True)
            tushare_io.write_parquet(output, baseline, api_name="share_float", params={}, fields=list(baseline.columns))
            active = baseline.copy()
            active["float_share"] = value
            tushare_io.write_parquet(source, active, api_name="share_float", params={}, fields=list(active.columns))
            with patch.object(share_float, "share_float_union_files", return_value=[(source, "ann_date")]):
                with patch.object(pd, "read_parquet", wraps=pd.read_parquet) as read_parquet:
                    share_float.write_share_float_union(self.raw_dir, args, {})
            output_reads = [
                call for call in read_parquet.call_args_list
                if Path(call.args[0]).resolve() == output.resolve()
            ]
            self.assertEqual(len(output_reads), 1)
            return (
                str(tushare_io.parquet_meta(output)["write_id"]),
                pd.read_parquet(output)["float_share"].tolist(),
            )

        first, second = publish(10.0), publish(11.0)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual((first[1], second[1]), ([10.0], [11.0]))

    def test_share_float_union_requires_baseline_for_incremental_update(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(share_float, "share_float_union_files") as list_files:
            with self.assertRaisesRegex(RuntimeError, "baseline is missing"):
                share_float.write_share_float_union(self.raw_dir, args, {})
        list_files.assert_not_called()
        self.assertFalse(output.exists())

    def test_share_float_union_replaces_group_but_preserves_distinct_lots_idempotently(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "000002.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h2", "share_type": "t", "float_share": 2.0, "float_ratio": 0.2},
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        refreshed = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 10.0, "float_ratio": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 20.0, "float_ratio": 2.0},
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 20.0, "float_ratio": 2.0},
        ])
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        tushare_io.write_parquet(source, refreshed, api_name="share_float", params={}, fields=list(refreshed.columns))
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        with patch.object(share_float, "share_float_union_files", return_value=[(source, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, {})
            first = pd.read_parquet(output)
            share_float.write_share_float_union(self.raw_dir, args, {})
            second = pd.read_parquet(output)

        self.assertEqual(len(first), 3)
        group = first[first["ts_code"] == "000001.SZ"].sort_values("float_share")
        self.assertEqual(group["float_share"].tolist(), [10.0, 20.0])
        pd.testing.assert_frame_equal(first, second)
        events = [
            json.loads(line)
            for line in (self.root / "revision_events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["old_rows"], events[0]["new_rows"]), (2, 3))

    def test_share_float_union_file_scope_uses_active_window(self):
        recent = self.raw_dir / "share_float_ann_date" / "ann_date=20200102.parquet"
        historical = self.raw_dir / "share_float_ann_date" / "ann_date=20190101.parquet"
        for path in (recent, historical):
            tushare_io.write_parquet(path, pd.DataFrame(), api_name="share_float", params={}, fields=[])
        args = argparse.Namespace(
            ann_start_date="20200101", ann_end_date="20200103",
            float_start_date="20200101", float_end_date="20200103",
            skip_float_date_union=False,
        )

        files = [path for path, _ in share_float.share_float_union_files(self.raw_dir, args)]
        self.assertEqual(files, [recent])

    def test_share_float_union_capped_group_preserves_complete_baseline(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": float(index), "float_ratio": 0.1}
            for index in range(7000)
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        tushare_io.write_parquet(
            source,
            existing.iloc[:share_float.SHARE_FLOAT_ROW_LIMIT],
            api_name="share_float",
            params={},
            fields=list(existing.columns),
        )
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        report = {}
        with patch.object(share_float, "share_float_union_files", return_value=[(source, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, report)

        self.assertEqual(tushare_io.parquet_rows(output), 7000)
        self.assertEqual(report["union"]["rows_after_dedup"], 7000)
        self.assertEqual(report["union"]["capped_groups_preserved"], 1)

    def test_share_float_union_non_capped_source_makes_duplicate_group_replaceable(self):
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 1.0, "float_ratio": 0.1},
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        refreshed = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200101", "float_date": "20200102",
             "holder_name": "h1", "share_type": "t", "float_share": 10.0, "float_ratio": 1.0},
        ])
        capped = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        rescue = self.raw_dir / "share_float_ann_date_ts_code" / "ann_date=20200101" / "ts_code=000001.SZ.parquet"
        tushare_io.write_parquet(
            capped,
            pd.concat([refreshed] * share_float.SHARE_FLOAT_ROW_LIMIT, ignore_index=True),
            api_name="share_float",
            params={},
            fields=list(refreshed.columns),
        )
        tushare_io.write_parquet(rescue, refreshed, api_name="share_float", params={}, fields=list(refreshed.columns))
        args = argparse.Namespace(
            union_output=str(output), ann_start_date="20200101", ann_end_date="20200102",
            float_start_date="20200101", float_end_date="20200102",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )

        report = {}
        with patch.object(share_float, "share_float_union_files",
            return_value=[(capped, "ann_date"), (rescue, "ann_date_ts_code")],
        ):
            share_float.write_share_float_union(self.raw_dir, args, report)

        result = pd.read_parquet(output)
        self.assertEqual(result["float_share"].tolist(), [10.0])
        self.assertEqual(result["download_path"].tolist(), ["ann_date_ts_code"])
        self.assertEqual(result["source_cap_risk"].tolist(), [False])
        self.assertEqual(report["union"]["capped_groups_preserved"], 0)

    def test_share_float_union_supersedes_redated_groups(self):
        # Provider re-dating (observed 2026-07: 301583.SZ moved from
        # ann_date 20260630 to 20260709): the same physical unlock event
        # (ts_code+float_date+holder+share_type) under a corrected ann_date
        # supersedes the stale-dated baseline group -- never double-counted;
        # removed rows land in the revision event. Distinct physical events
        # (other holders/dates) are untouched.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 2.0, "float_ratio": 0.2},
            # Whitespace variant of an incoming holder (observed live:
            # "...企业 (有限合伙)" vs "...企业(有限合伙)"): must supersede too.
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "基金 (有限合伙)", "share_type": "首发原始股", "float_share": 9.0, "float_ratio": 0.9},
            {"ts_code": "000009.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hB", "share_type": "首发原始股", "float_share": 3.0, "float_ratio": 0.3},
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={}, fields=list(existing.columns))
        source = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        redated = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260709", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 5.0, "float_ratio": 0.5},
            {"ts_code": "301583.SZ", "ann_date": "20260709", "float_date": "20270712",
             "holder_name": "基金(有限合伙)", "share_type": "首发原始股", "float_share": 9.0, "float_ratio": 0.9},
        ])
        tushare_io.write_parquet(source, redated, api_name="share_float", params={}, fields=list(redated.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260701", ann_end_date="20260710",
            float_start_date="20260701", float_end_date="20280101",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        report = {}
        with patch.object(share_float, "share_float_union_files", return_value=[(source, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(output)
        stale = union[(union["ts_code"] == "301583.SZ") & (union["ann_date"] == "20260630")]
        fresh = union[(union["ts_code"] == "301583.SZ") & (union["ann_date"] == "20260709")]
        other = union[union["ts_code"] == "000009.SZ"]
        self.assertTrue(stale.empty)          # superseded (incl. whitespace variant)
        self.assertEqual(len(fresh), 2)
        self.assertEqual(len(other), 1)       # distinct physical event untouched
        # Two baseline groups vanish: hA's stale-dated group and the
        # whitespace-variant holder's group.
        self.assertEqual(report["union"]["superseded_redated_groups"], 2)

        # Idempotent re-run: nothing left to supersede, content stable.
        report = {}
        with patch.object(share_float, "share_float_union_files", return_value=[(source, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, report)
        self.assertEqual(report["union"]["superseded_redated_groups"], 0)
        self.assertEqual(tushare_io.parquet_rows(output), 3)

    def test_share_float_active_window_holding_both_datings_heals(self):
        # The real production shape (301583.SZ): the refresh window covers the
        # stale AND the corrected ann_date partition, so the stale rows arrive
        # in `active` too. Baseline-only supersession left 72 rows across 3
        # ann_dates unchanged; active-side pruning is what makes it heal.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        rows = []
        for ann, share in (("20260630", 1.0), ("20260701", 2.0), ("20260709", 3.0)):
            rows.append({"ts_code": "301583.SZ", "ann_date": ann, "float_date": "20270712",
                         "holder_name": "hA", "share_type": "首发原始股",
                         "float_share": share, "float_ratio": share / 10})
        # An unrelated event must be untouched by the pruning.
        rows.append({"ts_code": "000009.SZ", "ann_date": "20260630", "float_date": "20270712",
                     "holder_name": "hB", "share_type": "首发原始股", "float_share": 9.0, "float_ratio": 0.9})
        existing = pd.DataFrame(rows)
        tushare_io.write_parquet(output, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        files = []
        for ann in ("20260630", "20260701", "20260709"):
            path = self.raw_dir / "share_float_ann_date" / f"ann_date={ann}.parquet"
            part = existing[existing["ann_date"] == ann]
            tushare_io.write_parquet(path, part, api_name="share_float", params={},
                                 fields=list(part.columns))
            files.append((path, "ann_date"))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20260601", float_end_date="20280101",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        report = {}
        with patch.object(share_float, "share_float_union_files", return_value=files):
            share_float.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(output)
        event = union[union["ts_code"] == "301583.SZ"]
        self.assertEqual(sorted(event["ann_date"]), ["20260709"])  # newest dating only
        self.assertEqual(event["float_share"].tolist(), [3.0])
        self.assertEqual(len(union[union["ts_code"] == "000009.SZ"]), 1)
        self.assertEqual(report["union"]["active_stale_restatement_rows_dropped"], 2)

        # Idempotent: nothing left to prune or supersede.
        report2 = {}
        with patch.object(share_float, "share_float_union_files", return_value=files):
            share_float.write_share_float_union(self.raw_dir, args, report2)
        self.assertEqual(report2["union"]["superseded_redated_groups"], 0)
        self.assertEqual(tushare_io.parquet_rows(output), 2)

    def test_share_float_capped_newer_dating_does_not_prune_active(self):
        # A capped (truncated) partition proves nothing about rows it could not
        # return, so it must not win the newest-dating comparison.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        old_path = self.raw_dir / "share_float_ann_date" / "ann_date=20260630.parquet"
        tushare_io.write_parquet(old_path, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        capped = pd.DataFrame([
            dict(existing.iloc[0], ann_date="20260709", float_share=2.0),
            *[{"ts_code": f"9{i:05d}.SZ", "ann_date": "20260709", "float_date": "20270712",
               "holder_name": f"h{i}", "share_type": "首发原始股", "float_share": 1.0,
               "float_ratio": 0.1} for i in range(share_float.SHARE_FLOAT_ROW_LIMIT - 1)],
        ])
        new_path = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        tushare_io.write_parquet(new_path, capped, api_name="share_float", params={},
                             fields=list(capped.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20260601", float_end_date="20280101",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        report = {}
        with patch.object(share_float, "share_float_union_files",
                          return_value=[(old_path, "ann_date"), (new_path, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, report)
        union = pd.read_parquet(output)
        event = union[(union["ts_code"] == "301583.SZ")]
        self.assertEqual(sorted(event["ann_date"]), ["20260630", "20260709"])  # conservative union
        self.assertEqual(report["union"]["active_stale_restatement_rows_dropped"], 0)

    def test_share_float_capped_identity_cannot_supersede_baseline_history(self):
        # Symmetry with the active-side rule: a capped (truncated) partition
        # proves nothing about rows it could not return, so it must not delete
        # an older announcement of the same physical event either.
        output = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        existing = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hA", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
        ])
        tushare_io.write_parquet(output, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        # Only a CAPPED partition carries the newer dating.
        capped = pd.DataFrame([
            dict(existing.iloc[0], ann_date="20260709", float_share=2.0),
            *[{"ts_code": f"9{i:05d}.SZ", "ann_date": "20260709", "float_date": "20270712",
               "holder_name": f"h{i}", "share_type": "首发原始股", "float_share": 1.0,
               "float_ratio": 0.1} for i in range(share_float.SHARE_FLOAT_ROW_LIMIT - 1)],
        ])
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        tushare_io.write_parquet(path, capped, api_name="share_float", params={},
                             fields=list(capped.columns))
        args = argparse.Namespace(
            union_output=str(output),
            ann_start_date="20260701", ann_end_date="20260731",
            float_start_date="20260601", float_end_date="20280101",
            skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        report = {}
        with patch.object(share_float, "share_float_union_files", return_value=[(path, "ann_date")]):
            share_float.write_share_float_union(self.raw_dir, args, report)
        union = pd.read_parquet(output)
        event = union[union["ts_code"] == "301583.SZ"]
        self.assertEqual(sorted(event["ann_date"]), ["20260630", "20260709"])  # old row kept
        self.assertEqual(report["union"]["superseded_redated_groups"], 0)

    def test_event_flow_audit_reports_cross_ann_date_identity_duplication(self):
        # The formal audit was blind to business-identity duplication: row and
        # key checks cannot see one physical unlock retained under several
        # announcement dates (measured live: >145k such groups).
        union = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"
        rows = pd.DataFrame([
            {"ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "基金 (有限合伙)", "share_type": "首发原始股", "float_share": 1.0, "float_ratio": 0.1},
            {"ts_code": "301583.SZ", "ann_date": "20260709", "float_date": "20270712",
             "holder_name": "基金(有限合伙)", "share_type": "首发原始股", "float_share": 2.0, "float_ratio": 0.2},
            {"ts_code": "000009.SZ", "ann_date": "20260630", "float_date": "20270712",
             "holder_name": "hB", "share_type": "首发原始股", "float_share": 3.0, "float_ratio": 0.3},
        ])
        tushare_io.write_parquet(union, rows, api_name="share_float_complete", params={},
                             fields=list(rows.columns))
        findings: list[dict] = []

        def add(severity, check, message, details=None):
            findings.append({"severity": severity, "check": check, "details": details or {}})

        share_float.audit_share_float_complete_union(self.raw_dir, add)
        finding = next(f for f in findings if f["check"] == "share_float_complete_union")
        self.assertEqual(finding["severity"], "warning")
        # whitespace variants of one holder count as ONE physical identity
        self.assertEqual(finding["details"]["cross_ann_date_identity_groups"], 1)
        self.assertEqual(finding["details"]["stale_redated_rows"], 1)

    def test_share_float_empty_refresh_keeps_existing_cap_risk_signal(self):
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        fields = specs.SHARE_FLOAT_FIELDS.split(",")
        existing = pd.DataFrame([
            {
                field: (
                    f"{index:06d}.SZ" if field == "ts_code"
                    else "20200101" if field == "ann_date"
                    else "20200102" if field == "float_date"
                    else str(index) if field == "holder_name"
                    else "type" if field == "share_type"
                    else 1
                )
                for field in fields
            }
            for index in range(share_float.SHARE_FLOAT_ROW_LIMIT)
        ])
        tushare_io.write_parquet(path, existing, api_name="share_float", params={}, fields=fields)

        result = share_float.query_share_float_to_path(
            EmptyClient(),
            path,
            {"ann_date": "20200101"},
            "ann_date",
            True,
            revision_ledger=None,
        )

        self.assertTrue(result["skipped"])
        self.assertEqual(result["rows"], share_float.SHARE_FLOAT_ROW_LIMIT)
        self.assertTrue(result["source_cap_risk"])
        self.assertEqual(tushare_io.parquet_rows(path), share_float.SHARE_FLOAT_ROW_LIMIT)

    def test_share_float_extreme_shrink_is_staged_and_reported(self):
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        fields = specs.SHARE_FLOAT_FIELDS.split(",")
        existing = pd.DataFrame([
            [f"{index:06d}.SZ", "20200101", "20200102", f"h{index}", "type", 1.0, 0.1]
            for index in range(100)
        ], columns=fields)
        existing = stamps.augment_event_frame(existing, specs.EVENT_FLOW_SPECS["share_float"])
        tushare_io.write_parquet(path, existing, api_name="share_float", params={}, fields=list(existing.columns))

        class ShrunkClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return tushare_client.ApiResult(
                    specs.SHARE_FLOAT_FIELDS.split(","),
                    [["000000.SZ", "20200101", "20200102", "h0", "type", 1.0, 0.1]])

        # An answer removing most keys is staged and REPORTED, not raised:
        # raising aborted the whole run mid-loop and made the ts_code rescue
        # unreachable every time a capped partition rotated its keys.
        with redirect_stdout(io.StringIO()):
            result = share_float.query_share_float_to_path(
                ShrunkClient(), path, {"ann_date": "20200101"}, "ann_date", True,
                revision_ledger=self.root / "revision_events.jsonl",
            )
        self.assertTrue(result["staged"])
        self.assertTrue(result["skipped"])
        self.assertEqual(tushare_io.parquet_rows(path), 100)
        self.assertEqual(result["rows"], 100)  # reports the visible partition

    def test_existing_partition_resume_reports_every_key(self):
        """A plain resume (partition present, no --force) returns early; the
        caller reads staged on every day, so omitting it there turned the
        resume path into a KeyError."""
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        fields = specs.SHARE_FLOAT_FIELDS.split(",")
        frame = pd.DataFrame([["000001.SZ", "20200101", "20200102", "h", "type", 1.0, 0.1]], columns=fields)
        frame = stamps.augment_event_frame(frame, specs.EVENT_FLOW_SPECS["share_float"])
        tushare_io.write_parquet(path, frame, api_name="share_float", params={},
                             fields=list(frame.columns))

        class Unused:
            def query(self, *a, **k):
                raise AssertionError("resume must not call the API")

        result = share_float.query_share_float_to_path(
            Unused(), path, {"ann_date": "20200101"}, "ann_date", False,
        )
        self.assertEqual(sorted(result), ["path", "rows", "skipped", "source_cap_risk", "staged"])
        self.assertFalse(result["staged"])
        self.assertTrue(result["skipped"])

    def test_capped_answer_is_published_and_every_code_it_replaced_stays_a_candidate(self):
        """2026-10-26: new IPO lock-ups of ~6,000 holder rows pushed older codes
        out of five capped ann_date answers, and the shrink guard staged each
        one as a withdrawal. A capped answer proves nothing about the keys it
        lacks: it is published, the replaced partition is kept as a superseded
        version, and every code a kept answer for the day carried -- the new
        answer, the replaced partition, an earlier rescue -- is a candidate."""
        day = "20200101"
        fields = specs.SHARE_FLOAT_FIELDS.split(",")
        path = self.raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
        old_codes = [f"{index:06d}.SZ" for index in range(100)]
        existing = pd.DataFrame([[code, day, "20200102", 1.0, 0.1, "h", "type"] for code in old_codes], columns=fields)
        existing = stamps.augment_event_frame(existing, specs.EVENT_FLOW_SPECS["share_float"])
        tushare_io.write_parquet(path, existing, api_name="share_float", params={}, fields=list(existing.columns))
        rescued = self.raw_dir / "share_float_ann_date_ts_code" / f"ann_date={day}" / "ts_code=300227.SZ.parquet"
        tushare_io.write_parquet(rescued, existing.head(1), api_name="share_float", params={}, fields=list(existing.columns))

        class CappedNewIpoClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return tushare_client.ApiResult(
                    specs.SHARE_FLOAT_FIELDS.split(","),
                    [["301716.SZ", day, "20200103", 1.0, 0.1, f"n{index}", "type"]
                     for index in range(share_float.SHARE_FLOAT_ROW_LIMIT)])

        args = argparse.Namespace(
            ann_start_date=day, ann_end_date=day, force=True,
            revision_ledger=self.root / "revision_events.jsonl",
            rescue_universe="candidate", rescue_code=[], rescue_codes_file=None,
            no_anns_candidates=True, max_codes=None,
        )
        report: dict = {}
        with redirect_stdout(io.StringIO()):
            limit_hits = share_float.download_share_float_ann_dates(CappedNewIpoClient(), self.raw_dir, args, report)
        self.assertEqual(limit_hits, [day])
        self.assertEqual(report["ann_date"]["staged_days"], [])
        self.assertEqual(tushare_io.parquet_rows(path), share_float.SHARE_FLOAT_ROW_LIMIT)
        self.assertEqual([version.split("-")[-1] for version in versions.list_versions(path)], ["superseded"])
        events = [json.loads(line) for line in (self.root / "revision_events.jsonl").read_text().splitlines()]
        self.assertEqual([event["kind"] for event in events], ["restatement"])

        candidates, _ = share_float.select_share_float_rescue_date_codes(self.raw_dir, args, date_param="ann_date", dates=[day])
        self.assertEqual(set(candidates[day]), {"301716.SZ", "300227.SZ", *old_codes})

    def test_capped_staging_reaches_rescue_while_a_real_retraction_stays_staged(self):
        """An uncapped answer that drops most keys is staged on both days, but
        a day whose visible partition sits at the cap still reaches the rescue:
        the union reads that partition as truncated. On the uncapped day the
        source genuinely retracted the keys, which no rescue can undo -- the
        staged version waits for an operator, and the run carries on.
        """
        fields = specs.SHARE_FLOAT_FIELDS.split(",")

        def seed(day, rows):
            path = self.raw_dir / "share_float_ann_date" / f"ann_date={day}.parquet"
            frame = pd.DataFrame([
                [f"{index:06d}.SZ", day, "20200102", f"h{index}", "type", 1.0, 0.1]
                for index in range(rows)
            ], columns=fields)
            frame = stamps.augment_event_frame(frame, specs.EVENT_FLOW_SPECS["share_float"])
            tushare_io.write_parquet(path, frame, api_name="share_float", params={},
                                 fields=list(frame.columns))

        capped_day, retracted_day = "20200101", "20200102"
        seed(capped_day, share_float.SHARE_FLOAT_ROW_LIMIT)  # at the API cap
        seed(retracted_day, 100)

        class ShrunkClient:
            def query(self, api_name, params=None, fields="", retries=5):
                return tushare_client.ApiResult(fields.split(",") if isinstance(fields, str) else list(fields),
                                        [["999999.SZ", params["ann_date"], "20200102", "hZ", "type", 1.0, 0.1]])

        args = argparse.Namespace(
            ann_start_date=capped_day, ann_end_date=retracted_day, force=True,
            revision_ledger=self.root / "revision_events.jsonl",
        )
        report: dict = {}
        with redirect_stdout(io.StringIO()):
            share_float.download_share_float_ann_dates(ShrunkClient(), self.raw_dir, args, report)

        # The capped day reached the rescue list, and BOTH partitions were
        # preserved.
        self.assertEqual(report["ann_date"]["limit_hit_days"], [capped_day])
        self.assertEqual(report["ann_date"]["staged_days"], [retracted_day])
        self.assertEqual(
            tushare_io.parquet_rows(self.raw_dir / "share_float_ann_date" / f"ann_date={capped_day}.parquet"),
            share_float.SHARE_FLOAT_ROW_LIMIT,
        )

    def test_every_capped_day_is_rescued_within_the_call_limit_or_the_run_fails(self):
        """The 5-day rescue cap left four of nine capped days unrescued and
        their new codes out of the union. Every limit-hit day is rescued; only
        --max-rescue-calls bounds the cost, and a run that cannot rescue them
        all fails before any rescue call instead of merging a truncated day."""
        days = [f"202001{index:02d}" for index in range(1, 8)]

        class CappedDaysClient:
            def __init__(self):
                self.calls = []

            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append(dict(params))
                code = f"3000{params['ann_date'][-2:]}.SZ"
                rows = 1 if "ts_code" in params else share_float.SHARE_FLOAT_ROW_LIMIT
                return tushare_client.ApiResult(
                    specs.SHARE_FLOAT_FIELDS.split(","),
                    [[code, params["ann_date"], "20200301", 1.0, 0.1, f"h{index}", "type"] for index in range(rows)])

        def run(max_calls):
            client = CappedDaysClient()
            args = download.parse_args([
                "download-share-float-complete", "--raw-dir", str(self.raw_dir),
                "--ann-start-date", days[0], "--ann-end-date", days[-1],
                "--rescue-ann-limit-hits", "--no-anns-candidates", "--force",
                "--max-rescue-calls", str(max_calls), "--revision-ledger", str(self.root / "revision_events.jsonl"),
            ])
            with (
                patch.object(share_float, "TuShareClient", return_value=client),
                patch.object(share_float, "load_token", return_value="token"),
                redirect_stdout(io.StringIO()),
            ):
                share_float.download_share_float_complete(args)
            return {call["ann_date"] for call in client.calls if "ts_code" in call}

        with self.assertRaisesRegex(RuntimeError, "would make 7 calls .* exceeding --max-rescue-calls=6"):
            run(max_calls=6)
        self.assertEqual(run(max_calls=7), set(days))

    def test_share_float_candidate_read_failure_is_not_silent(self):
        path = self.raw_dir / "anns_d" / "month=202001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"ann_date": "20200101", "title": "解禁"}]).to_parquet(path, index=False)

        with self.assertRaises(ArrowInvalid):
            share_float.anns_unlock_candidate_codes(self.raw_dir, ["20200101"])

    def test_generic_event_flow_download_excludes_dedicated_share_float_path(self):
        selected = download.selected_event_flow_download_datasets(argparse.Namespace(datasets=None))
        self.assertNotIn("share_float", selected)
        self.assertIn("margin_secs", selected)
        with self.assertRaisesRegex(RuntimeError, "download-share-float-complete"):
            download.selected_event_flow_download_datasets(argparse.Namespace(datasets=["share_float"]))

    def test_share_float_skip_reattempts_partition_without_intact_sidecar(self):
        fields = specs.SHARE_FLOAT_FIELDS.split(",")
        path = self.raw_dir / "share_float_ann_date" / "ann_date=20200101.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        seeded = pd.DataFrame([["000001.SZ", "20200101", "20200102", 1.0, 0.1, "h1", "type"]], columns=fields)
        seeded.to_parquet(path, index=False)
        self.assertFalse(tushare_io.committed_partition_intact(path))

        class FloatClient(EmptyClient):
            def query(self, api_name, params=None, fields="", retries=5):
                self.calls.append((api_name, dict(params or {})))
                columns = fields.split(",") if isinstance(fields, str) else list(fields)
                return tushare_client.ApiResult(columns, [["000001.SZ", "20200101", "20200102", 1.0, 0.1, "h1", "type"]])

        client = FloatClient()
        with redirect_stdout(io.StringIO()):
            report = share_float.query_share_float_to_path(
                client, path, {"ann_date": "20200101"}, "ann_date", False,
            )
        self.assertEqual([api for api, _ in client.calls], ["share_float"])
        self.assertFalse(report["skipped"])
        self.assertTrue(tushare_io.committed_partition_intact(path))
        client.calls.clear()
        report = share_float.query_share_float_to_path(
            client, path, {"ann_date": "20200101"}, "ann_date", False,
        )
        self.assertEqual(client.calls, [])
        self.assertTrue(report["skipped"])


class ShareFloatUndatedDuplicateInvariantTest(unittest.TestCase):
    """Production union invariants around undated share_float duplicates.

    The one-off historical repair tool (repair.py) ran on 2026-07-30 and was
    retired on 2026-08-02 (dev-phase ruling); the rule it applied is now a
    union-merge invariant (``share_float.undated_duplicate_mask``), pinned here.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.raw_dir = self.root / "raw"
        self.baseline = self.raw_dir / "share_float_complete" / "share_float_complete.parquet"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_baseline(self, rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        tushare_io.write_parquet(
            self.baseline, frame, api_name="share_float_complete", params={},
            fields=list(frame.columns),
        )
        return frame

    @staticmethod
    def _row(**overrides) -> dict:
        row = {
            "ts_code": "301583.SZ", "ann_date": "20260630", "float_date": "20270712",
            "holder_name": "基金(有限合伙)", "share_type": "首发原始股",
            "float_share": 1000.0, "float_ratio": 1.0,
        }
        row.update(overrides)
        return row

    def test_union_keeps_undated_duplicates_out_by_construction(self) -> None:
        """A rescue fetch that returns a lot undated must not re-add it.

        The seven-field dedup cannot collapse an undated row against its dated
        twin (they differ in ann_date), so without this invariant the class the
        one-off repair removed could return through a supported CLI path and
        double the float supply again.
        """
        dated = self._row(float_share=32974.0)
        undated = self._row(ann_date=None, float_share=32974.0, float_ratio=0.9)
        distinct_lot = self._row(ann_date=None, float_share=555.0)
        frame = pd.DataFrame([dated, undated, distinct_lot])

        mask = share_float.undated_duplicate_mask(frame)

        self.assertEqual(mask.tolist(), [False, True, False])
        kept = frame.loc[~mask]
        self.assertEqual(float(kept.loc[kept["float_share"] == 32974.0, "float_share"].sum()), 32974.0)
        # A differing lot size is never collapsed: it may be a real second lot.
        self.assertIn(555.0, kept["float_share"].tolist())

    def test_undated_active_row_cannot_supersede_a_dated_baseline_announcement(self) -> None:
        # The float_date query path returns rows without ann_date. Identity-based
        # supersession would otherwise delete the DATED baseline generation and
        # downgrade its availability to the float_date fallback.
        existing = pd.DataFrame([self._row(ann_date="20260630", float_share=1000.0)])
        tushare_io.write_parquet(self.baseline, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        source = self.raw_dir / "share_float" / "date=20270712.parquet"
        undated = pd.DataFrame([self._row(ann_date=None, float_share=2000.0)])
        tushare_io.write_parquet(source, undated, api_name="share_float", params={},
                             fields=list(undated.columns))
        args = argparse.Namespace(
            union_output=str(self.baseline), ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20270712", float_end_date="20270712", skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        report: dict = {}

        with patch.object(share_float, "share_float_union_files", return_value=[(source, "float_date_existing")]):
            share_float.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(self.baseline)
        self.assertEqual(report["union"]["superseded_redated_groups"], 0)
        self.assertEqual(sorted(union["float_share"]), [1000.0, 2000.0])
        self.assertIn("20260630", union["ann_date"].astype(str).tolist())

    def test_undated_active_row_loses_the_newest_dating_comparison(self) -> None:
        existing = pd.DataFrame([self._row(ann_date="20260630", float_share=1000.0)])
        tushare_io.write_parquet(self.baseline, existing, api_name="share_float", params={},
                             fields=list(existing.columns))
        dated_path = self.raw_dir / "share_float_ann_date" / "ann_date=20260709.parquet"
        dated = pd.DataFrame([self._row(ann_date="20260709", float_share=1000.0)])
        tushare_io.write_parquet(dated_path, dated, api_name="share_float", params={},
                             fields=list(dated.columns))
        undated_path = self.raw_dir / "share_float" / "date=20270712.parquet"
        undated = pd.DataFrame([self._row(ann_date=None, float_share=1000.0)])
        tushare_io.write_parquet(undated_path, undated, api_name="share_float", params={},
                             fields=list(undated.columns))
        args = argparse.Namespace(
            union_output=str(self.baseline), ann_start_date="20260601", ann_end_date="20260731",
            float_start_date="20270712", float_end_date="20270712", skip_float_date_union=False,
            revision_ledger=common.REVISION_EVENTS_PATH,
        )
        report: dict = {}

        with patch.object(share_float, "share_float_union_files",
            return_value=[(dated_path, "ann_date"), (undated_path, "float_date_existing")],
        ):
            share_float.write_share_float_union(self.raw_dir, args, report)

        union = pd.read_parquet(self.baseline)
        self.assertEqual(report["union"]["active_stale_restatement_rows_dropped"], 1)
        self.assertEqual(union["ann_date"].astype(str).tolist(), ["20260709"])

    def test_audit_reports_the_undated_copy_as_the_stale_row(self) -> None:
        rows = pd.DataFrame([
            self._row(ann_date=None),
            self._row(),
            self._row(ts_code="000009.SZ", ann_date="20260630"),
            self._row(ts_code="000009.SZ", ann_date="20260709"),
        ])
        tushare_io.write_parquet(self.baseline, rows, api_name="share_float_complete", params={},
                             fields=list(rows.columns))
        findings: list[dict] = []

        share_float.audit_share_float_complete_union(
            self.raw_dir,
            lambda severity, check, message, details=None: findings.append(
                {"check": check, "details": details or {}}
            ),
        )

        details = next(f for f in findings if f["check"] == "share_float_complete_union")["details"]
        self.assertEqual(details["cross_ann_date_identity_groups"], 2)
        self.assertEqual(details["stale_redated_rows"], 2)
        self.assertEqual(details["undated_duplicate_rows"], 1)

    def test_vectorized_date_normalization_matches_the_scalar_rule(self) -> None:
        values = [None, float("nan"), "", "20200102", "2020-01-02", "20200102123", "2020010", "x", 20200102]
        vector = share_float.normalized_date_keys(pd.Series(values, dtype=object)).tolist()

        self.assertEqual(vector, [common.normalize_date_key(value) for value in values])
        self.assertEqual(vector[0], "")
        self.assertLess(vector[0], "20200102")  # undated sorts BELOW every real date


if __name__ == "__main__":
    unittest.main()
