"""The versioned write path and the commit identity of a partition
(data_sources/tushare/versions.py, io.py).
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from autotrade.data_sources.tushare import audit, cron_update, download, specs, versions
from autotrade.data_sources.tushare import io as tushare_io
from autotrade.environment.data.snapshot import SELECTABLE_DATASETS

from .support.tushare_fixture import LakeTestCase


class VersionedWriteTest(LakeTestCase):
    def test_revision_ledger_appends_each_logical_event_once(self):
        from autotrade.data_sources.tushare.io import append_jsonl_unique

        # ``event_id`` and ``write_id`` are fresh UUIDs per write, so the
        # ledger dedupes on the event's stable content: re-detecting the same
        # revision appends nothing, a genuinely different one appends.
        ledger = self.root / "revision_events.jsonl"
        first = {"event_id": "a", "detected_at": "first", "write_id": "w1", "dataset": "daily", "changed_keys": 1}
        repeated = {"event_id": "b", "detected_at": "later", "write_id": "w2", "dataset": "daily", "changed_keys": 1}
        distinct = {"event_id": "c", "detected_at": "later", "write_id": "w3", "dataset": "daily", "changed_keys": 2}

        self.assertTrue(append_jsonl_unique(ledger, first, key="event_id"))
        self.assertFalse(append_jsonl_unique(ledger, repeated, key="event_id"))
        self.assertTrue(append_jsonl_unique(ledger, distinct, key="event_id"))
        records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records, [first, distinct])
        # A non-event key keeps the plain by-value contract.
        other = self.root / "by_value.jsonl"
        self.assertTrue(append_jsonl_unique(other, {"job": "a", "n": 1}, key="job"))
        self.assertFalse(append_jsonl_unique(other, {"job": "a", "n": 2}, key="job"))
        self.assertTrue(append_jsonl_unique(other, {"job": "b", "n": 1}, key="job"))
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            append_jsonl_unique(other, {"job": ""}, key="job")

    def test_dup_key_partitions_compare_as_multisets_of_rows(self):
        path = self.raw_dir / "dividend" / "ann_month=202001.parquet"
        rows = pd.DataFrame([
            {"ts_code": "000001.SZ", "ann_date": "20200105", "cash_div": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20200105", "cash_div": 1.0},  # duplicate business key
        ])
        tushare_io.write_parquet(path, rows, api_name="dividend", params={}, fields=list(rows.columns))
        ledger = self.root / "dupkey_revision_events.jsonl"

        def write(frame):
            output = io.StringIO()
            with redirect_stdout(output):
                outcome = versions.publish(
                    path, frame, api_name="dividend", params={}, fields=list(frame.columns),
                    key_columns=["ts_code", "ann_date"], ledger=ledger,
                )
            return outcome, output.getvalue()

        # Identical content with duplicate keys is not a revision.
        self.assertEqual(write(rows.copy()), ("unchanged", ""))
        self.assertFalse(ledger.exists())

        changed = rows.copy()
        changed.loc[1, "cash_div"] = 2.0
        self.assertEqual(write(changed)[0], "restated")
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((event["kind"], event["changed_keys"], event["removed_keys"]), ("restatement", 1, 0))
        self.assertTrue(event["superseded_version"].endswith("-superseded"))

        # The same answer again is unchanged: nothing written, nothing logged.
        self.assertEqual(write(changed.copy())[0], "unchanged")
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)

        # An added column without values changes no key: it is written, but
        # nothing is lost, so no version is kept and no event logged.
        widened = changed.assign(note="")
        self.assertEqual(write(widened)[0], "appended")
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)
        self.assertIn("note", pd.read_parquet(path).columns)

    def test_vendor_backfill_of_a_key_field_is_not_a_retraction(self):
        # End of the H1-2026 reporting season: TuShare populated actual_date on
        # rows that had carried none. actual_date is part of disclosure_date's
        # business key, so the fill reads as every key removed and re-added;
        # counted on the key minus that field nothing was removed, so the
        # answer is accepted as a restatement instead of being staged.
        path = self.raw_dir / "disclosure_date" / "period=20260630.parquet"
        keys = ["ts_code", "end_date", "ann_date", "pre_date", "actual_date"]
        original = pd.DataFrame([
            {
                "ts_code": f"{index:06d}.SZ",
                "end_date": "20260630",
                "ann_date": "20260415",
                "pre_date": "20260828",
                "actual_date": "",
                "modify_date": "",
            }
            for index in range(120)
        ])
        tushare_io.write_parquet(
            path, original, api_name="disclosure_date", params={}, fields=list(original.columns)
        )
        backfilled = original.assign(actual_date="20260826")
        ledger = self.root / "backfill_revision_events.jsonl"

        with redirect_stdout(io.StringIO()):
            outcome = versions.publish(
                path, backfilled, api_name="disclosure_date", params={}, fields=list(backfilled.columns),
                key_columns=keys, ledger=ledger,
            )
        self.assertEqual(outcome, "restated")
        self.assertEqual(set(pd.read_parquet(path)["actual_date"]), {"20260826"})
        # The ledger records the full-key diff: the staging rule changed what
        # it decides on, not what the event reports.
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((event["kind"], event["removed_keys"], event["added_keys"]), ("restatement", 120, 120))

    def test_real_retraction_is_staged_on_the_immutable_key_subset(self):
        # Same dataset, same rule: when the stable identity itself disappears
        # the shrink is real, and an answer removing most keys is staged.
        path = self.raw_dir / "disclosure_date" / "period=20260331.parquet"
        keys = ["ts_code", "end_date", "ann_date", "pre_date", "actual_date"]
        original = pd.DataFrame([
            {
                "ts_code": f"{index:06d}.SZ",
                "end_date": "20260331",
                "ann_date": "20260115",
                "pre_date": "20260428",
                "actual_date": "",
                "modify_date": "",
            }
            for index in range(120)
        ])
        tushare_io.write_parquet(
            path, original, api_name="disclosure_date", params={}, fields=list(original.columns)
        )
        truncated = original.head(10).assign(actual_date="20260426")
        ledger = self.root / "retraction_revision_events.jsonl"

        with redirect_stdout(io.StringIO()):
            outcome = versions.publish(
                path, truncated, api_name="disclosure_date", params={}, fields=list(truncated.columns),
                key_columns=keys, ledger=ledger,
            )
        self.assertEqual(outcome, "staged")
        self.assertTrue(pd.read_parquet(path).equals(original))
        event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["kind"], "staged")

    def test_parquet_availability_survives_identical_refresh_but_moves_on_revision(self):
        path = self.raw_dir / "stk_auction" / "trade_date=20260713.parquet"
        keys = list(specs.DAILY_SPECS["stk_auction"].key_columns)
        first = pd.DataFrame(
            [{"trade_date": "20260713", "ts_code": "000001.SZ", "price": 10.0}]
        )
        availability = {
            "available_at": "2026-07-13T09:28:36+08:00",
            "rule": "observed:cn_open_auction_capture",
        }

        def write(frame, **extra):
            with redirect_stdout(io.StringIO()):
                return versions.publish(
                    path, frame, api_name="stk_auction", params={}, fields=list(frame.columns),
                    key_columns=keys, ledger=None, **extra,
                )

        self.assertEqual(write(first, extra_metadata={"availability": availability}), "appended")
        self.assertEqual(write(first.copy()), "unchanged")
        self.assertEqual(tushare_io.parquet_meta(path)["availability"], availability)

        self.assertEqual(write(first.assign(price=10.1)), "restated")
        revised_availability = tushare_io.parquet_meta(path)["availability"]
        self.assertEqual(revised_availability["rule"], "observed:source_revision_fetch")
        self.assertNotEqual(revised_availability["available_at"], availability["available_at"])

    def test_key_comparison_handles_missing_key_columns_and_duplicate_keys(self):
        # A missing key column reads as empty text: the old row's key differs.
        missing = versions.diff_frames(
            pd.DataFrame([{"trade_date": "20200102", "open": 1.0}]),
            pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0}]),
            ["trade_date", "ts_code"],
        )
        self.assertEqual((len(missing.removed), len(missing.added)), (1, 1))
        # A repeated key compares as the multiset of its rows.
        duplicate = versions.diff_frames(
            pd.DataFrame([
                {"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0},
                {"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0},
            ]),
            pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "open": 1.0}]),
            ["trade_date", "ts_code"],
        )
        self.assertEqual((len(duplicate.changed), len(duplicate.removed), duplicate.changed_columns), (1, 0, {}))

    def test_revision_comparison_canonicalizes_numeric_values(self):
        old_df = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "adj_factor": 1}])
        new_df = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "adj_factor": 1.0}])
        self.assertTrue(versions.diff_frames(old_df, new_df, ["trade_date", "ts_code"]).empty)

    def test_revision_event_records_counts_without_samples(self):
        old_df = pd.DataFrame([
            {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.0, "amount": 100.0},
            {"trade_date": "20200102", "ts_code": "000002.SZ", "close": 20.0, "amount": 200.0},
            {"trade_date": "20200102", "ts_code": "000003.SZ", "close": 30.0, "amount": 300.0},
        ])
        new_df = pd.DataFrame([
            {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.5, "amount": 100.0},
            {"trade_date": "20200102", "ts_code": "000002.SZ", "close": 20.0, "amount": 201.5},
            {"trade_date": "20200102", "ts_code": "000004.SZ", "close": 40.0, "amount": 400.0},
        ])
        diff = versions.diff_frames(old_df, new_df, ["trade_date", "ts_code"])
        event = versions.revision_event(
            "restatement", dataset="daily", partition="trade_date=20200102", source="unit",
            key_columns=["trade_date", "ts_code"], diff=diff, old_rows=3, new_rows=3,
            superseded_version="20261024T000000000000Z-superseded",
        )

        ledger = self.root / "revision_events.jsonl"
        with redirect_stdout(io.StringIO()):
            versions.append_event(ledger, event)

        [record] = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(record, event)
        self.assertEqual(record["schema_version"], versions.REVISION_EVENT_SCHEMA_VERSION)
        self.assertEqual((record["old_keys"], record["new_keys"]), (3, 3))
        self.assertEqual((record["changed_keys"], record["added_keys"], record["removed_keys"]), (2, 1, 1))
        self.assertEqual(record["changed_columns"], {"amount": 1, "close": 1})
        self.assertFalse(any("sample" in field for field in record))

    def test_empty_answer_defect_does_not_depend_on_a_ledger(self):
        path = self.raw_dir / "limit_list_d" / "trade_date=20200102.parquet"
        original = pd.DataFrame([{"trade_date": "20200102", "ts_code": "000001.SZ", "limit": "U"}])
        tushare_io.write_parquet(path, original, api_name="limit_list_d", params={}, fields=list(original.columns))

        with redirect_stdout(io.StringIO()):
            outcome = versions.publish(
                path,
                pd.DataFrame(columns=list(original.columns)),
                api_name="limit_list_d",
                params={"trade_date": "20200102"},
                fields=list(original.columns),
                key_columns=["trade_date", "ts_code", "limit"],
                ledger=None,
            )

        self.assertEqual(outcome, "defect")
        self.assertTrue(pd.read_parquet(path).equals(original))


class CorruptSidecarTest(unittest.TestCase):
    """A present-but-unparseable .meta.json is torn-write evidence and must be
    distinguishable from an absent sidecar: parquet_meta used to return {} for
    both, so the audit counted corruption as 'legacy' info and the snapshot
    silently fell back to imputed availability."""

    def test_parquet_meta_distinguishes_absent_from_corrupt(self):
        from autotrade.environment.data.pit import CorruptSidecarError, parquet_meta

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trade_date=20260101.parquet"
            pd.DataFrame({"a": [1]}).to_parquet(path, index=False)
            self.assertEqual(parquet_meta(path), {})  # absent sidecar: pre-scheme file
            sidecar = Path(str(path) + ".meta.json")
            sidecar.write_text('{"fetched_at": "2026', encoding="utf-8")  # torn write
            with self.assertRaisesRegex(CorruptSidecarError, "unreadable parquet sidecar"):
                parquet_meta(path)
            sidecar.write_text('["not", "an", "object"]', encoding="utf-8")
            with self.assertRaisesRegex(CorruptSidecarError, "unreadable parquet sidecar.*not an object"):
                parquet_meta(path)

    def test_revision_writer_heals_corrupt_sidecar_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw" / "daily" / "trade_date=20260101.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame({"ts_code": ["000001.SZ"], "v": [1]}).to_parquet(path, index=False)
            Path(str(path) + ".meta.json").write_text("{broken", encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                outcome = versions.publish(
                    path,
                    pd.DataFrame({"ts_code": ["000001.SZ"], "v": [2]}),
                    api_name="daily",
                    params={},
                    fields=["ts_code", "v"],
                    key_columns=["ts_code"],
                    ledger=Path(tmp) / "ledger.jsonl",
                )
            self.assertEqual(outcome, "restated")
            alert_lines = [line for line in out.getvalue().splitlines() if "corrupt_sidecar_replaced" in line]
            self.assertEqual(len(alert_lines), 1)
            # The rewrite repaired the torn pair: the sidecar parses again.
            self.assertTrue(tushare_io.committed_partition_intact(path))

    def test_audit_records_corrupt_sidecar_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            dataset_dir = raw_dir / "daily"
            dataset_dir.mkdir()
            path = dataset_dir / "trade_date=20260101.parquet"
            pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(path, index=False)
            Path(str(path) + ".meta.json").write_text("{broken", encoding="utf-8")
            findings: list[dict] = []

            def add(severity, check, message, details=None):
                findings.append({"severity": severity, "check": check, "details": details or {}})

            audit.audit_integrated_filesystem(raw_dir, ["daily"], add)
            inventory = next(f for f in findings if f["check"] == "integrated_filesystem")
            self.assertEqual(inventory["severity"], "error")
            self.assertEqual(inventory["details"]["corrupt_sidecars"], 1)


class UuidCommitIdentityTest(unittest.TestCase):
    """One UUID identity shared by a Parquet footer and its sidecar."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sidecar_uses_footer_write_id(self) -> None:
        path = self.root / "daily" / "trade_date=20240102.parquet"
        meta = tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"], "close": [10.0]}),
            api_name="daily",
            params={},
            fields=["trade_date", "close"],
        )
        self.assertTrue(tushare_io.committed_partition_intact(path))
        self.assertEqual(tushare_io.parquet_write_id(path), meta["write_id"])
        self.assertEqual(meta["row_count"], 1)

    def test_commit_integrity_rejects_interrupted_mixed_and_wrong_count_pairs(self) -> None:
        path = self.root / "daily" / "trade_date=20240102.parquet"
        tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"], "close": [10.0]}),
            api_name="daily",
            params={},
            fields=["trade_date", "close"],
        )
        sidecar = path.with_suffix(".parquet.meta.json")
        original_sidecar = sidecar.read_text(encoding="utf-8")

        sidecar.unlink()
        self.assertFalse(tushare_io.committed_partition_intact(path))
        sidecar.write_text(original_sidecar, encoding="utf-8")

        other = self.root / "daily" / "trade_date=20240103.parquet"
        tushare_io.write_parquet(
            other,
            pd.DataFrame({"trade_date": ["20240103"], "close": [11.0]}),
            api_name="daily",
            params={},
            fields=["trade_date", "close"],
        )
        sidecar.write_text(
            other.with_suffix(".parquet.meta.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.assertFalse(tushare_io.committed_partition_intact(path))

        wrong_count = json.loads(original_sidecar)
        wrong_count["row_count"] = 2
        sidecar.write_text(json.dumps(wrong_count), encoding="utf-8")
        self.assertFalse(tushare_io.committed_partition_intact(path))

    def test_write_parquet_has_no_metadata_field_name_policy(self) -> None:
        path = self.root / "daily" / "trade_date=20240102.parquet"
        meta = tushare_io.write_parquet(
            path,
            pd.DataFrame({"trade_date": ["20240102"]}),
            api_name="daily",
            params={"filters": [{"opaque_marker": "source-value"}]},
            fields=["trade_date"],
            extra_metadata={"source_context": {"opaque_marker": "landing-value"}},
        )
        self.assertTrue(tushare_io.committed_partition_intact(path))
        self.assertEqual(meta["params"]["filters"][0]["opaque_marker"], "source-value")
        self.assertEqual(meta["source_context"]["opaque_marker"], "landing-value")

    def test_schedule_retains_full_job_set(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8"))
        # 29 since cn_preopen_text_backfill_0855 was retired (2026-09-10); 30
        # since the minute layer left cn_evening_full for its own manual job;
        # 32 with the two research-history backfill jobs (2026-09-24); 31 since
        # the completed one-shot commit-identity migration job was retired; 32
        # with the margin available_at repair (2026-10-02); 33 with anns_d's
        # own official-service job (2026-10-06); 34 with the stk_holdertrade
        # restatement re-pull (2026-10-20); 35 with the top_list one; 36 with
        # the stk_limit one (2026-10-21); 37 with the top-10 holder repair; 39
        # with the stk_holdernumber and repurchase restatement re-pulls; 35
        # since the four disclosure-table re-pulls became the history_repull
        # policy and its operator command (2026-10-24); 36 with the weekly
        # research-release retention; 37 with the standalone share_float
        # repair (2026-10-27); 38 with the anns_d available_at repair
        # (2026-11-06).
        self.assertEqual(len(config["jobs"]), 38)
        for name, tier in (
            ("manual_history_backfill_reference", "reference"),
            ("manual_history_backfill_macro", "macro"),
        ):
            job = config["jobs"][name]
            self.assertEqual((job["operation"], job["tier"]), ("download_tier", tier))
            self.assertEqual(job["extra_args"][:2], ["--history-floor", "20140701"])
        self.assertIn("--no-include-intraday", config["jobs"]["cn_evening_full"]["extra_args"])
        self.assertEqual(config["jobs"]["cn_nightly_intraday_minutes"]["operation"], "intraday_by_date")
        # A history re-pull rewrites one dataset's whole history; it must never
        # widen to the other disclosure tables the nightly job shares, and
        # through a report lag its window must begin exactly at the lake's
        # floor.
        policy = cron_update.history_repull_policy(config)
        self.assertEqual(set(policy), {item["dataset"] for item in config["interfaces"]})
        on_restatement = sorted(name for name, value in policy.items() if value == "on_restatement")
        self.assertEqual(on_restatement, sorted([
            "pledge_detail", "repurchase", "stk_holdernumber", "stk_holdertrade", "top10_floatholders", "top10_holders",
        ]))
        for dataset in on_restatement:
            command = cron_update.history_repull_command(config, "/env/python", dataset, end_date="20261024")
            self.assertEqual(command[2:8], ["download", "--tier", "event_flow", "--datasets", dataset, "--force"])
            start = command[command.index("--start-date") + 1]
            lag = timedelta(days=specs.EVENT_FLOW_SPECS[dataset].report_lag_days)
            self.assertEqual(
                (datetime.strptime(start, "%Y%m%d") - lag).strftime("%Y%m%d"),  # noqa: DTZ007
                config["default_start_date"],
            )
        with self.assertRaisesRegex(ValueError, "cannot be re-pulled whole"):
            cron_update.history_repull_command(config, "/env/python", "top_list", end_date="20261024")
        toplist = config["jobs"]["manual_toplist_restatement_repair"]
        self.assertEqual((toplist["operation"], toplist["tier"]), ("download_tier", "board_trading"))
        self.assertEqual(toplist["extra_args"], ["--datasets", "top_list", "--force"])
        # top_list publishes at 20:00 on its trade date; ending on the last
        # open date before today never writes an unpublished, empty day.
        self.assertEqual(
            (toplist["end_date_offset_days"], toplist["end_date_mode"]), (1, "sse_open_on_or_before")
        )
        # stk_limit shares its tier with the core daily tables cn_evening_full
        # refreshes; the forced re-pull must touch it alone and leave today's
        # partition to the evening job.
        stk_limit = config["jobs"]["manual_stklimit_restatement_repair"]
        self.assertEqual((stk_limit["operation"], stk_limit["tier"]), ("download_tier", "daily"))
        self.assertEqual(stk_limit["extra_args"], ["--datasets", "stk_limit", "--force"])
        self.assertEqual(
            (stk_limit["end_date_offset_days"], stk_limit["end_date_mode"]), (1, "sse_open_on_or_before")
        )
        # share_float has no tier of its own: its manual job runs the evening
        # step's command alone, as a mutating job, and the CLI accepts it.
        share_float = cron_update.RunContext(
            config=config, repo_root=root, python="/env/python", job_name="manual_share_float_repair",
            job=config["jobs"]["manual_share_float_repair"], start_date="20260926", end_date="20261026",
            timezone_name="Asia/Shanghai",
        )
        command = cron_update.build_job_commands(share_float)[0]
        self.assertIn("share_float_complete", cron_update.MUTATING_OPERATIONS)
        parsed = download.parse_args(command[2:])
        self.assertEqual((parsed.command, parsed.ann_start_date, parsed.float_end_date), ("download-share-float-complete", "20260926", "20261026"))
        self.assertTrue(parsed.rescue_ann_limit_hits and parsed.force and parsed.write_union)

    def test_minute_job_runs_the_by_date_updater_as_a_mutating_job(self) -> None:
        context = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_update_args": ["--min-interval-seconds", "0.80"]},
            repo_root=Path("."),
            python="python",
            job_name="cn_nightly_intraday_minutes",
            job={"operation": "intraday_by_date"},
            start_date="20260819",
            end_date="20260911",
            timezone_name="Asia/Shanghai",
        )
        self.assertEqual(
            cron_update.build_job_commands(context),
            [[
                "python", "scripts/data/tushare_download.py", "update-intraday-by-date",
                "--start-date", "20260819", "--end-date", "20260911", "--raw-dir", "raw",
                "--min-interval-seconds", "0.80",
            ]],
        )
        self.assertIn("intraday_by_date", cron_update.MUTATING_OPERATIONS)

    def test_event_available_at_repair_runs_as_a_mutating_job(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8"))
        job = config["jobs"]["manual_event_available_at_repair"]
        context = cron_update.RunContext(
            config={"default_raw_dir": "raw"}, repo_root=Path("."), python="python",
            job_name="manual_event_available_at_repair", job=job,
            start_date="20200101", end_date="20260930", timezone_name="Asia/Shanghai",
        )
        self.assertEqual(
            cron_update.build_job_commands(context),
            [["python", "scripts/data/tushare_download.py", "repair-event-available-at", "--raw-dir", "raw",
              "--datasets", "margin", "margin_detail"]],
        )
        self.assertIn(job["operation"], cron_update.MUTATING_OPERATIONS)

    def test_text_available_at_repair_runs_as_a_mutating_job(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8"))
        job = config["jobs"]["manual_text_available_at_repair"]
        context = cron_update.RunContext(
            config={"default_raw_dir": "raw"}, repo_root=Path("."), python="python",
            job_name="manual_text_available_at_repair", job=job,
            start_date="20200101", end_date="20260930", timezone_name="Asia/Shanghai",
        )
        self.assertEqual(
            cron_update.build_job_commands(context),
            [["python", "scripts/data/tushare_download.py", "repair-text-available-at", "--raw-dir", "raw",
              "--datasets", "irm_qa_sz"]],
        )
        self.assertIn(job["operation"], cron_update.MUTATING_OPERATIONS)

    def test_snapshot_inventory_retains_full_dataset_coverage(self) -> None:
        root = Path(__file__).resolve().parents[2]
        inventory = json.loads(
            (root / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"]
        self.assertIn("margin", inventory["events.parquet"])
        self.assertIn("cn_cpi", inventory["macro.parquet"])
        self.assertIn("income_vip", inventory["fundamentals.parquet"])
        self.assertIn(
            "source_write_id",
            inventory["fundamentals.parquet"]["income_vip"],
        )

    def test_every_downloaded_reference_dataset_is_registered(self) -> None:
        # interfaces[] is the operational fact source for datasets (the console
        # reads it for dataset labels): a table the downloader writes but the
        # registry omits carries no documented refresh or availability contract.
        root = Path(__file__).resolve().parents[2]
        schedule = json.loads(
            (root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8")
        )
        scheduled = {item["dataset"] for item in schedule["interfaces"]}
        self.assertEqual(set(specs.REFERENCE_DATASETS) - scheduled, set())

    def test_every_selectable_snapshot_dataset_is_registered(self) -> None:
        # The creation form renders one chip per selectable dataset and reads
        # its Chinese label from interfaces[], so a dataset the console can
        # offer but the registry omits (a locally derived table included)
        # reaches the user as a bare table name.
        root = Path(__file__).resolve().parents[2]
        schedule = json.loads(
            (root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8")
        )
        scheduled = {item["dataset"] for item in schedule["interfaces"]}
        selectable = set().union(*SELECTABLE_DATASETS.values())
        self.assertEqual(selectable - scheduled, set())

    def test_active_macro_registries_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schedule = json.loads(
            (root / "configs/tushare_update_schedule.json").read_text(encoding="utf-8")
        )
        inventory = json.loads(
            (root / "configs/data/snapshot_columns.json").read_text(encoding="utf-8")
        )["files"]
        scheduled_datasets = {item["dataset"] for item in schedule["interfaces"]}

        self.assertEqual(set(specs.MACRO_DATASETS), set(specs.MACRO_SPECS))
        self.assertTrue(set(specs.MACRO_DATASETS).issubset(scheduled_datasets))
        # The committed inventory covers every SELECTABLE macro dataset, not
        # just the default scope: the unit registry is fail-closed, so a
        # dataset an experiment can opt into must resolve column by column.
        self.assertEqual(
            set(SELECTABLE_DATASETS["macro"]),
            set(inventory["macro.parquet"]),
        )
        # Every selectable macro dataset has a download contract. Almost all
        # are macro-tier series; `index_weight` is the exception, a reference
        # table the macro domain reads (month-end index constituents), landed
        # by the reference sweep inside the same evening job, hence the same
        # refresh node. A name in neither tier would be selectable with nothing
        # keeping it up to date.
        self.assertTrue(
            set(SELECTABLE_DATASETS["macro"])
            <= set(specs.MACRO_SPECS) | set(specs.REFERENCE_DATASETS)
        )
        self.assertEqual(
            set(SELECTABLE_DATASETS["macro"]) - set(specs.MACRO_SPECS), {"index_weight"}
        )


if __name__ == "__main__":
    unittest.main()
