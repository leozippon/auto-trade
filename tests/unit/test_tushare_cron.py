"""The cron runner and the generation record
(data_sources/tushare/cron_update.py, generation.py).
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from autotrade.data_sources.tushare import audit, common, cron_update, generation, specs
from autotrade.environment.data.fundamental_events import (
    FundamentalEventsBuilder,
    month_aligned_replace_window,
)
from autotrade.environment.data.snapshot import _require_event_months

from .support.tushare_fixture import LakeTestCase, commit_generation


class _StubLock:
    fd = None

    def release(self) -> None:
        return None


class CronJobTest(LakeTestCase):
    def test_finished_job_publishes_atomic_stamp(self):
        raw = self.root / "genraw"
        first = commit_generation(raw)
        self.assertEqual(json.loads((raw / ".raw_generation.json").read_text(encoding="utf-8")), first)
        self.assertEqual((first["schema_version"], first["state"]), (3, "committed"))
        self.assertTrue(first["generation_id"])
        self.assertTrue(first["completed_at"])
        second = commit_generation(raw)
        self.assertNotEqual(first["generation_id"], second["generation_id"])
        # A job that vouches it wrote nothing keeps the identity.
        kept = generation.finish_raw_generation_update(
            raw, second, {"job": "none"}, commit=False, datasets={}, issues=[]
        )
        self.assertEqual(
            (kept["generation_id"], kept["completed_at"], kept["transaction"]),
            (second["generation_id"], second["completed_at"], second["transaction"]),
        )
        self.assertEqual(list((raw).glob(".raw_generation.json.tmp*")), [])

    def test_no_generation_state_refuses_a_job(self):
        # A job that died mid-run leaves the record ``updating``; the next job
        # of any name begins, carries the per-dataset state and the open
        # issues over, and its end commits a new generation.
        raw = self.root / "genraw"
        datasets = {"anns_d": {"refreshed_at": "2026-10-23T15:30:00+00:00", "vintage": "abc"}}
        issues = [{"kind": "staged", "dataset": "repurchase", "partition": "month=202610", "since": "", "detail": "", "version": "v"}]
        with redirect_stdout(io.StringIO()):
            base = generation.finish_raw_generation_update(
                raw, {}, {"job": "seed"}, commit=True, datasets=datasets, issues=issues
            )
        generation.begin_raw_generation_update(raw, {"job": "cn_evening_full", "start_date": "", "end_date": ""})
        died = json.loads((raw / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual((died["state"], died["generation_id"]), ("updating", base["generation_id"]))

        previous = generation.begin_raw_generation_update(raw, {"job": "another_job", "start_date": "", "end_date": ""})
        self.assertEqual((previous["datasets"], previous["issues"]), (datasets, issues))
        with redirect_stdout(io.StringIO()):
            committed = generation.finish_raw_generation_update(
                raw, previous, {"job": "another_job"}, commit=True, datasets=previous["datasets"], issues=previous["issues"]
            )
        self.assertEqual(committed["state"], "committed")
        self.assertNotEqual(committed["generation_id"], base["generation_id"])
        self.assertEqual((committed["datasets"], committed["issues"]), (datasets, issues))

    def test_revision_sentinel_is_not_a_mutating_operation(self):
        self.assertNotIn("revision_sentinel", cron_update.MUTATING_OPERATIONS)

    def test_non_trading_auction_job_skips_before_the_generation_record(self):
        self._write_trade_cal("20260712", is_open="0")
        config_path = self.root / "auction_schedule.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "timezone": "Asia/Shanghai",
                    "repo_root": str(self.root),
                    "python": "/env/python",
                    "default_raw_dir": "raw",
                    "default_start_date": "20200101",
                    "jobs": {
                        "auction": {
                            "operation": "auction_capture",
                            "only_if_sse_open_date": True,
                            "start_date_lookback_days": 0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        generation = self.raw_dir / ".raw_generation.json"
        commit_generation(self.raw_dir)
        before = generation.read_bytes()
        args = argparse.Namespace(
            config=str(config_path),
            job="auction",
            start_date=None,
            end_date="20260712",
            dry_run=False,
            force_run=False,
            then=[],
        )
        jobs_root = self.root / "runtime" / "jobs"
        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", side_effect=AssertionError("must not lock")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), 0)

        self.assertEqual(generation.read_bytes(), before)
        record = json.loads((jobs_root / "auction.json").read_text(encoding="utf-8"))
        self.assertTrue(record["skipped_non_trading_day"])
        # Every run outcome carries its own real log file.
        self.assertTrue(Path(record["log_path"]).is_file())

    def test_same_day_open_check_fails_when_calendar_does_not_cover_target(self):
        self._write_trade_cal("20260712", is_open="0")

        with self.assertRaisesRegex(RuntimeError, "does not cover target date 20260713"):
            cron_update.is_sse_open_date(self.root, "raw", "20260713")

    def test_not_ready_auction_job_restores_committed_generation(self):
        self._write_trade_cal("20260713", is_open="1")
        config_path = self.root / "auction_schedule.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "timezone": "Asia/Shanghai",
                    "repo_root": str(self.root),
                    "python": "/env/python",
                    "default_raw_dir": "raw",
                    "default_start_date": "20200101",
                    "jobs": {
                        "auction": {
                            "operation": "auction_capture",
                            "only_if_sse_open_date": True,
                            "start_date_lookback_days": 0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        generation = self.raw_dir / ".raw_generation.json"
        commit_generation(self.raw_dir)
        before = json.loads(generation.read_text(encoding="utf-8"))
        args = argparse.Namespace(
            config=str(config_path),
            job="auction",
            start_date=None,
            end_date="20260713",
            dry_run=False,
            force_run=False,
            then=[],
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        jobs_root = self.root / "runtime" / "jobs"
        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(
                cron_update,
                "run_update",
                return_value=common.NO_MUTATION_RETRY_EXIT_CODE,
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), common.NO_MUTATION_RETRY_EXIT_CODE)

        self.assertEqual(json.loads(generation.read_text(encoding="utf-8")), before)
        record = json.loads((jobs_root / "auction.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "not_ready")
        self.assertTrue(Path(record["log_path"]).is_file())

    def _write_event_flow_schedule(self, name: str, operation: str = "download_event_flow") -> Path:
        job = {
            "operation": operation,
            "end_date_offset_days": 0,
            "skip_if_already_ok": True,
        }
        if operation == "download_tier":
            job["tier"] = "event_flow"
        config_path = self.root / f"{name}_schedule.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "timezone": "Asia/Shanghai",
                    "repo_root": str(self.root),
                    "python": "/env/python",
                    "default_raw_dir": "raw",
                    "default_start_date": "20200101",
                    "jobs": {name: job},
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def _run_job_once(self, config_path: Path, job: str, end_date: str, returncode: int, jobs_root: Path):
        args = argparse.Namespace(
            config=str(config_path),
            job=job,
            start_date=None,
            end_date=end_date,
            dry_run=False,
            force_run=False,
            then=[],
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(cron_update, "run_update", return_value=returncode) as runner,
            redirect_stdout(io.StringIO()) as out,
        ):
            result = cron_update.main()
        return result, runner, out.getvalue()

    def test_mutated_not_ready_commits_generation_and_is_retried_next_run(self):
        # Regression: a run that wrote real partitions but left a required one
        # unpublished must publish the generation (the writes are real) AND
        # record a non-ok status. Recording "ok" is what let skip_if_already_ok
        # abandon margin trade_date=20260807 once the resolved window moved on.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("margin_retry")
        generation = self.raw_dir / ".raw_generation.json"
        commit_generation(self.raw_dir)
        before = json.loads(generation.read_text(encoding="utf-8"))
        jobs_root = self.root / "runtime" / "jobs"

        result, _, _ = self._run_job_once(
            config_path,
            "margin_retry",
            "20260807",
            common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
            jobs_root,
        )
        self.assertEqual(result, common.MUTATED_NOT_READY_RETRY_EXIT_CODE)

        after = json.loads(generation.read_text(encoding="utf-8"))
        self.assertEqual(after["state"], "committed")
        self.assertNotEqual(after["generation_id"], before["generation_id"])

        record = json.loads((jobs_root / "margin_retry.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "not_ready")

        # The next run at the SAME resolved end date must actually re-attempt.
        _, runner, output = self._run_job_once(
            config_path, "margin_retry", "20260807", 0, jobs_root
        )
        self.assertTrue(runner.called)
        self.assertNotIn("skipped_already_ok", output)
        self.assertEqual(
            json.loads((jobs_root / "margin_retry.json").read_text(encoding="utf-8"))["status"],
            "ok",
        )

    def test_ok_event_flow_run_is_still_skipped_on_repeat(self):
        # The mirror of the regression: a genuinely complete run must keep
        # suppressing duplicate work at the same resolved end date.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("margin_ok")
        commit_generation(self.raw_dir)
        jobs_root = self.root / "runtime" / "jobs"

        self._run_job_once(config_path, "margin_ok", "20260807", 0, jobs_root)
        _, runner, output = self._run_job_once(
            config_path, "margin_ok", "20260807", 0, jobs_root
        )
        self.assertFalse(runner.called)
        self.assertIn("skipped_already_ok", output)

    def test_minute_quota_stop_commits_the_days_it_wrote(self):
        # A quota stop of the minute job keeps what it wrote committed; a dirty
        # lake would refuse the evening update Paper's release depends on.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("minutes", operation="intraday_by_date")
        generation = self.raw_dir / ".raw_generation.json"
        commit_generation(self.raw_dir)
        jobs_root = self.root / "runtime" / "jobs"

        result, _, _ = self._run_job_once(
            config_path, "minutes", "20260807", common.MUTATED_NOT_READY_RETRY_EXIT_CODE, jobs_root
        )
        self.assertEqual(result, common.MUTATED_NOT_READY_RETRY_EXIT_CODE)
        self.assertEqual(json.loads(generation.read_text(encoding="utf-8"))["state"], "committed")
        record = json.loads((jobs_root / "minutes.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "not_ready")

    def test_mutated_not_ready_from_unexpected_operation_is_an_error(self):
        # Exit 76 only carries the commit-and-retry contract for the download
        # path that enforces it; anywhere else it is an ordinary failure,
        # which commits what it wrote, records an error and exits 1: cron
        # reports failures by exit status and keeps the retry exits quiet.
        self._write_trade_cal("20260807", is_open="1")
        config_path = self._write_event_flow_schedule("tier_job", operation="download_tier")
        generation = self.raw_dir / ".raw_generation.json"
        commit_generation(self.raw_dir)
        jobs_root = self.root / "runtime" / "jobs"

        result, _, _ = self._run_job_once(
            config_path,
            "tier_job",
            "20260807",
            common.MUTATED_NOT_READY_RETRY_EXIT_CODE,
            jobs_root,
        )
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(generation.read_text(encoding="utf-8"))["state"], "committed")
        record = json.loads((jobs_root / "tier_job.json").read_text(encoding="utf-8"))
        self.assertEqual((record["status"], record["returncode"]), ("error", 1))

    def test_lost_official_access_leaves_the_lake_committed_and_the_job_failed(self):
        # Losing TUSHARE_TOKEN_TMP must stop only the jobs that need it: such a
        # job wrote nothing, so the generation every other job and consumer
        # needs stays committed, while the job reports an error every night.
        config_path = self.root / "anns_schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {"anns": {
                "operation": "download_tier",
                "tier": "text_evidence",
                "end_date_offset_days": 0,
                "extra_args": ["--datasets", "anns_d", "--force"],
                "skip_if_already_ok": True,
            }},
        }), encoding="utf-8")
        generation = self.raw_dir / ".raw_generation.json"
        commit_generation(self.raw_dir)
        jobs_root = self.root / "runtime" / "jobs"
        self._run_job_once(config_path, "anns", "20261005", 0, jobs_root)
        landed = json.loads((jobs_root / "anns.json").read_text(encoding="utf-8"))
        before = json.loads(generation.read_text(encoding="utf-8"))
        for _ in range(2):  # an error is never skipped: the next night tries again
            result, runner, _ = self._run_job_once(
                config_path, "anns", "20261006", common.NO_MUTATION_FAILURE_EXIT_CODE, jobs_root
            )
            self.assertTrue(runner.called)
            self.assertEqual(result, common.NO_MUTATION_FAILURE_EXIT_CODE)
            self.assertEqual(json.loads(generation.read_text(encoding="utf-8")), before)
            record = json.loads((jobs_root / "anns.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            # Every failure keeps the last success, which Paper's title check reads.
            self.assertEqual(
                (record["last_ok_at"], record["last_ok_end_date"]), (landed["updated_at"], "20261005")
            )

    def test_no_mutation_failure_elsewhere_commits_a_new_generation(self):
        # Exit 77 vouches for "nothing written" only where the text tier
        # enforces it; any other job claiming it is an ordinary failure,
        # whose writes are committed under a new generation.
        config_path = self._write_event_flow_schedule("tier_job", operation="download_tier")
        generation = self.raw_dir / ".raw_generation.json"
        before = commit_generation(self.raw_dir)
        result, _, _ = self._run_job_once(
            config_path, "tier_job", "20260807", common.NO_MUTATION_FAILURE_EXIT_CODE, self.root / "jobs"
        )
        self.assertEqual(result, common.NO_MUTATION_FAILURE_EXIT_CODE)
        after = json.loads(generation.read_text(encoding="utf-8"))
        self.assertEqual(after["state"], "committed")
        self.assertNotEqual(after["generation_id"], before["generation_id"])

    def test_only_a_lone_command_passes_the_no_mutation_failure_through(self):
        ctx = cron_update.RunContext(
            config={}, repo_root=self.root, python="/env/python", job_name="unit_77",
            job={"fail_fast": False}, start_date="20200101", end_date="20200102",
            timezone_name="Asia/Shanghai",
        )
        for commands, expected in (
            ([["anns"]], common.NO_MUTATION_FAILURE_EXIT_CODE),
            ([["other"], ["anns"]], 1),
        ):
            with (
                patch.object(cron_update, "run_probe"),
                patch.object(
                    cron_update.subprocess, "run",
                    side_effect=lambda command, **_: SimpleNamespace(
                        returncode=common.NO_MUTATION_FAILURE_EXIT_CODE if command == ["anns"] else 0
                    ),
                ),
            ):
                self.assertEqual(cron_update.run_update(ctx, commands, self.root / "cron_77.log"), expected)

    def test_finishing_run_preserves_concurrently_recorded_failure(self):
        # Invariant: an outcome another process persisted while this run was
        # active (here: a job that timed out waiting on the update lock and
        # recorded its failure) must survive the finish. Per-job state files
        # make cross-job interference structurally impossible; this pins it.
        config_path = self.root / "cron_schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20260101",
            "jobs": {"job_a": {"operation": "audit_event_flow"}},
        }), encoding="utf-8")
        jobs_root = self.root / "runtime" / "jobs"
        args = argparse.Namespace(
            config=str(config_path),
            job="job_a",
            start_date=None,
            end_date="20260713",
            dry_run=False,
            force_run=False,
            then=[],
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        def concurrent_failure_writer(*_args, **_kwargs):
            # While job_a's (long) run is active, another cron process times
            # out on the update lock and durably records its failure.
            cron_update.record_job_state("job_b", {
                "status": "error",
                "returncode": 1,
                "error": "lock is held after waiting 900s",
                "updated_at": "2026-07-13T01:00:00+00:00",
            })
            return 0

        with (
            patch.object(cron_update, "parse_args", return_value=args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(cron_update, "run_update", side_effect=concurrent_failure_writer),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), 0)

        job_a = json.loads((jobs_root / "job_a.json").read_text(encoding="utf-8"))
        self.assertEqual(job_a["status"], "ok")
        # The concurrently recorded failure stays visible.
        job_b = json.loads((jobs_root / "job_b.json").read_text(encoding="utf-8"))
        self.assertEqual(job_b["status"], "error")
        self.assertIn("lock is held", job_b["error"])

    def test_record_job_state_concurrent_writers_stay_isolated_and_atomic(self):
        jobs_root = self.root / "runtime" / "jobs"

        def write_entries(name: str) -> None:
            for seq in range(10):
                cron_update.record_job_state(name, {"status": "ok", "seq": seq})

        with patch.object(cron_update, "JOB_STATE_ROOT", jobs_root), patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"):
            workers = [
                threading.Thread(target=write_entries, args=(f"job_{index}",))
                for index in range(4)
            ]
            # Same-job concurrent writers: the file must never tear and the
            # last completed write must win.
            workers += [
                threading.Thread(target=write_entries, args=("shared",))
                for _ in range(3)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        for index in range(4):
            record = json.loads((jobs_root / f"job_{index}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["seq"], 9)
        shared = json.loads((jobs_root / "shared.json").read_text(encoding="utf-8"))
        self.assertEqual(shared["seq"], 9)

    def test_auction_cron_command_overrides_global_request_timeout(self):
        ctx = cron_update.RunContext(
            config={
                "default_raw_dir": "raw",
                "default_update_args": ["--timeout-seconds", "120"],
            },
            repo_root=self.root,
            python="/env/python",
            job_name="auction",
            job={
                "operation": "auction_capture",
                "extra_args": ["--timeout-seconds", "15"],
            },
            start_date="20260713",
            end_date="20260713",
            timezone_name="Asia/Shanghai",
        )

        command = cron_update.build_job_commands(ctx)[0]
        timeout_positions = [i for i, value in enumerate(command) if value == "--timeout-seconds"]
        self.assertEqual(command[timeout_positions[-1] + 1], "15")

    def test_cron_job_identity_ignores_unrelated_job_edits(self):
        selected_job = {"operation": "auction_capture", "skip_if_already_ok": True}
        base_config = {
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_start_date": "20200101",
            "default_raw_dir": "raw",
            "default_update_args": ["--timeout-seconds", "15"],
            "jobs": {"auction": selected_job, "unrelated": {"extra_args": ["old"]}},
        }
        edited_config = {
            **base_config,
            "jobs": {"auction": selected_job, "unrelated": {"extra_args": ["new"]}},
        }
        contexts = [
            cron_update.RunContext(
                config=config,
                repo_root=self.root,
                python="/env/python",
                job_name="auction",
                job=selected_job,
                start_date="20260713",
                end_date="20260713",
                timezone_name="Asia/Shanghai",
            )
            for config in (base_config, edited_config)
        ]
        first_identity, edited_identity = map(cron_update.job_config_identity, contexts)
        self.assertEqual(first_identity, edited_identity)
        commands = [["python", "capture.py"]]
        payload = {"commands": commands, "config_identity": edited_identity}
        state = {
            "start_date": "20260713",
            "end_date": "20260713",
            "status": "ok",
            "commands": commands,
            "config_identity": first_identity,
        }
        args = argparse.Namespace(force_run=False)
        self.assertTrue(cron_update.should_skip_completed(contexts[1], args, state, payload))

    def test_cron_full_audit_builds_all_formal_status_commands(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_full_audit",
            job={"operation": "audit_full", "event_flow_end_extra_offset_days": 1},
            start_date="20200101",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 7)
        # Last, the vintage report of the live Paper accounts, which reads no
        # date window.
        self.assertEqual(commands.pop(), ["/env/python", "scripts/paper/run_paper.py", "vintages"])
        command_text = [" ".join(command) for command in commands]
        self.assertIn("scripts/data/tushare_audit.py core-market", command_text[0])
        self.assertIn("scripts/data/tushare_audit.py fundamental-raw", command_text[1])
        self.assertIn("--fundamental-end-date 20260601", command_text[1])
        self.assertIn("scripts/data/tushare_audit.py macro", command_text[2])
        self.assertIn("scripts/data/tushare_audit.py intraday-by-date", command_text[3])
        # "daily" is an independent universe for the newest-day deep check;
        # "minute" read the audited file's own codes back and could never
        # report a dropped stock.
        self.assertIn("--expected-codes-source daily", command_text[3])
        self.assertIn("scripts/data/tushare_audit.py event-flow", command_text[4])
        self.assertIn("--end-date 20260531", command_text[4])
        self.assertIn("scripts/data/tushare_audit.py board-trading", command_text[5])
        # Text is natural-day data with its own daily audit job: this
        # trading-day job must not claim to refresh its status file.
        self.assertTrue(all(" text " not in f"{text} " for text in command_text))
        self.assertTrue(all("--start-date 20200101" in text for text in command_text))
        self.assertTrue(all("--raw-dir raw" in text for text in command_text))

    def test_cron_text_audit_uses_the_natural_day_window(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_text_audit",
            job={"operation": "audit_text"},
            start_date="20200101",
            end_date="20260621",  # a Sunday: audited as itself, never rolled back
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        command_text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_audit.py text", command_text)
        self.assertIn("--start-date 20200101", command_text)
        self.assertIn("--end-date 20260621", command_text)
        self.assertIn("--raw-dir raw", command_text)

    def test_cron_run_log_retention_preserves_state_link_and_unrelated_files(self):
        log_root = self.root / "logs" / "tushare" / "cron"
        log_root.mkdir(parents=True)
        stale = log_root / "tushare_cron_stale.log"
        referenced = log_root / "tushare_cron_referenced.log"
        fresh = log_root / "tushare_cron_fresh.log"
        unrelated = log_root / "notes.txt"
        for path in (stale, referenced, fresh, unrelated):
            path.write_text("record\n", encoding="utf-8")
        baseline = stale.stat().st_mtime
        os.utime(fresh, (baseline + 14 * 86400, baseline + 14 * 86400))
        jobs_root = self.root / "runtime" / "jobs"
        jobs_root.mkdir(parents=True)
        (jobs_root / "job.json").write_text(
            json.dumps({"status": "ok", "log_path": str(referenced)}), encoding="utf-8"
        )
        # A torn per-job file must not break retention for the others.
        (jobs_root / "corrupt.json").write_text("{not json", encoding="utf-8")

        with (
            patch.object(cron_update, "RUN_LOG_ROOT", log_root),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
        ):
            cron_update.prune_run_logs(
                now=baseline + (cron_update.RUN_LOG_RETENTION_DAYS + 1) * 86400,
            )

        self.assertFalse(stale.exists())
        self.assertTrue(referenced.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(unrelated.exists())

    def test_skip_and_lock_failure_write_their_own_run_logs(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="audit",
            job={"operation": "audit_full"},
            start_date="20200101",
            end_date="20260720",
            timezone_name="Asia/Shanghai",
        )
        args = argparse.Namespace(dry_run=False, force_run=False)
        log_root = self.root / "logs" / "tushare" / "cron"
        jobs_root = self.root / "runtime" / "jobs"

        # Already-ok skip: the invocation leaves its own log, but the state
        # file stays untouched so it keeps pointing at (and protecting) the
        # last real run.
        with (
            patch.object(cron_update, "build_context", return_value=ctx),
            patch.object(cron_update, "build_job_commands", return_value=[["audit"]]),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "RUN_LOG_ROOT", log_root),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "should_skip_completed", return_value=True),
            patch.object(cron_update, "acquire_lock", side_effect=AssertionError("must not lock")),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(cron_update._run(args), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "skipped_already_ok")
        skip_logs = list(log_root.glob("tushare_cron_audit_*.log"))
        self.assertEqual(len(skip_logs), 1)
        self.assertIn("skipped_already_ok", skip_logs[0].read_text(encoding="utf-8"))
        self.assertFalse((jobs_root / "audit.json").exists())

        # Lock failure: the error is durably recorded with its own log.
        with (
            patch.object(cron_update, "build_context", return_value=ctx),
            patch.object(cron_update, "build_job_commands", return_value=[["audit"]]),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "RUN_LOG_ROOT", log_root),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "should_skip_completed", return_value=False),
            patch.object(cron_update, "acquire_lock", side_effect=RuntimeError("lock is held")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update._run(args), 1)
        record = json.loads((jobs_root / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "error")
        self.assertIn("lock is held", record["error"])
        self.assertTrue(Path(record["log_path"]).is_file())
        self.assertIn("lock is held", Path(record["log_path"]).read_text(encoding="utf-8"))

    def test_cron_full_audit_can_use_open_date_for_event_flow(self):
        trade_cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "20260618", "is_open": "1"},
                {"cal_date": "20260619", "is_open": "0"},
                {"cal_date": "20260620", "is_open": "0"},
                {"cal_date": "20260621", "is_open": "0"},
            ]
        ).to_parquet(trade_cal, index=False)
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_full_audit",
            job={"operation": "audit_full", "event_flow_end_extra_offset_days": 3},
            start_date="20200101",
            end_date="20260621",
            timezone_name="Asia/Shanghai",
        )

        command_text = [" ".join(command) for command in cron_update.build_job_commands(ctx)]

        self.assertIn("scripts/data/tushare_audit.py event-flow", command_text[4])
        self.assertIn("--end-date 20260618", command_text[4])

    def test_cron_update_job_can_use_rolling_start_lookback(self):
        config_path = self.root / "schedule.json"
        trade_cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cal_date": "20260618", "is_open": "1"},
                {"cal_date": "20260619", "is_open": "0"},
                {"cal_date": "20260620", "is_open": "0"},
                {"cal_date": "20260621", "is_open": "0"},
            ]
        ).to_parquet(trade_cal, index=False)
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {
                "cn_evening_full": {
                    "start_date_lookback_days": 30,
                    "extra_args": ["--refresh-daily-datasets", "daily", "adj_factor"],
                },
                "cn_preopen_margin_backfill_0905": {
                    "operation": "download_event_flow",
                    "end_date_offset_days": 1,
                    "end_date_mode": "sse_open_on_or_before",
                },
            },
        }), encoding="utf-8")

        args = argparse.Namespace(config=str(config_path), job="cn_evening_full", start_date=None, end_date="20260601", dry_run=False, force_run=False)
        ctx = cron_update.build_context(args)
        self.assertEqual(ctx.start_date, "20260502")
        self.assertEqual(ctx.end_date, "20260601")
        command = " ".join(cron_update.build_job_commands(ctx)[0])
        self.assertIn("--refresh-daily-datasets daily adj_factor", command)

        margin_args = argparse.Namespace(config=str(config_path), job="cn_preopen_margin_backfill_0905", start_date=None, end_date="20260621", dry_run=False, force_run=False)
        margin_ctx = cron_update.build_context(margin_args)
        self.assertEqual(margin_ctx.start_date, "20260618")
        self.assertEqual(margin_ctx.end_date, "20260618")

    def test_cron_nightly_audit_weekend_runs_resolve_to_friday_and_skip(self):
        # Saturday and Sunday invocations resolve the same trading end date as
        # Friday's completed run, so skip_if_already_ok suppresses the weekend
        # full re-scans (and no weekend calendar date leaks into the window).
        config_path = self.root / "schedule.json"
        trade_cal = self.root / "raw" / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"cal_date": "20260619", "is_open": "1"},
            {"cal_date": "20260620", "is_open": "0"},
            {"cal_date": "20260621", "is_open": "0"},
        ]).to_parquet(trade_cal, index=False)
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {
                "cn_nightly_full_audit": {
                    "operation": "audit_full",
                    "end_date_offset_days": 1,
                    "end_date_mode": "sse_open_on_or_before",
                    "event_flow_end_extra_offset_days": 1,
                    "text_end_extra_offset_days": 1,
                },
            },
        }), encoding="utf-8")

        contexts = {}
        for target in ("20260620", "20260621"):  # Sat and Sun morning targets
            args = argparse.Namespace(config=str(config_path), job="cn_nightly_full_audit", start_date=None, end_date=target, dry_run=False, force_run=False)
            contexts[target] = cron_update.build_context(args)
        self.assertEqual(contexts["20260620"].end_date, "20260619")
        self.assertEqual(contexts["20260621"].end_date, "20260619")

        ctx = contexts["20260621"]
        payload = {
            "commands": cron_update.build_job_commands(ctx),
            "config_identity": cron_update.job_config_identity(ctx),
        }
        saturday_state = {
            "start_date": ctx.start_date,
            "end_date": "20260619",
            "status": "ok",
            "commands": payload["commands"],
            "config_identity": payload["config_identity"],
        }
        skip_args = argparse.Namespace(force_run=False)
        self.assertTrue(cron_update.should_skip_completed(ctx, skip_args, saturday_state, payload))
        # --force-run (the Saturday post-sweep cron line) still runs.
        self.assertFalse(cron_update.should_skip_completed(ctx, argparse.Namespace(force_run=True), saturday_state, payload))

    def test_cron_pit_event_job_covers_full_window_even_with_existing_partitions(self):
        # The rolling 120-day window let months outside it drift against the
        # living raw lake unaudited; the pipeline now always rebuilds and
        # audits from default_start_date.
        config_path = self.root / "schedule.json"
        existing = self.root / "pit" / "fundamental_events" / "income_vip"
        existing.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"dataset": "income_vip", "available_at": "2026-01-01T00:00:00+08:00"}]).to_parquet(
            existing / "available_month=202601.parquet",
            index=False,
        )
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_start_date": "20200101",
            "default_raw_dir": "raw",
            "default_pit_root": "pit",
            "jobs": {
                "cn_nightly_pit_event_build": {
                    "operation": "pit_event_pipeline",
                    "fundamental_events_root": "pit/fundamental_events",
                },
            },
        }), encoding="utf-8")

        args = argparse.Namespace(
            config=str(config_path),
            job="cn_nightly_pit_event_build",
            start_date=None,
            end_date="20260621",
            dry_run=False,
            force_run=False,
        )
        ctx = cron_update.build_context(args)
        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(ctx.start_date, "20200101")
        self.assertIn("--start-date 20200101", " ".join(commands[0]))
        self.assertIn("--start-date 20200101", " ".join(commands[1]))

    def test_cron_job_start_date_overrides_the_default_for_that_job_alone(self):
        # The nightly PIT audit must reach back to the first month a Paper
        # book's decision view loads without widening the jobs that share
        # default_start_date. The run's state record names the start it used,
        # so the audit status's scope traces back to the schedule.
        config_path = self.root / "schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_start_date": "20200101",
            "default_raw_dir": "raw",
            "default_pit_root": "pit",
            "jobs": {
                "cn_nightly_pit_event_build": {"operation": "pit_event_pipeline", "start_date": "20150101"},
                "cn_nightly_full_audit": {"operation": "audit_full"},
            },
        }), encoding="utf-8")

        def args(job, start_date=None):
            return argparse.Namespace(
                config=str(config_path), job=job, start_date=start_date, end_date="20260930",
                dry_run=False, force_run=False, then=[],
            )

        ctx = cron_update.build_context(args("cn_nightly_pit_event_build"))
        self.assertEqual(ctx.start_date, "20150101")
        for command in cron_update.build_job_commands(ctx):  # the build and its audit
            self.assertIn("--start-date 20150101 --end-date 20260930", " ".join(command))
        self.assertEqual(cron_update.build_context(args("cn_nightly_full_audit")).start_date, "20200101")
        # A one-off --start-date still wins over the job's own bound.
        self.assertEqual(
            cron_update.build_context(args("cn_nightly_pit_event_build", "20180101")).start_date, "20180101"
        )

        class FakeLock:
            fd = 7

            def release(self):
                return None

        jobs_root = self.root / "runtime" / "jobs"
        with (
            patch.object(cron_update, "parse_args", return_value=args("cn_nightly_pit_event_build")),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "JOB_STATE_ROOT", jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=FakeLock()),
            patch.object(cron_update, "run_update", return_value=0),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cron_update.main(), 0)
        record = json.loads((jobs_root / "cn_nightly_pit_event_build.json").read_text(encoding="utf-8"))
        self.assertEqual((record["status"], record["start_date"]), ("ok", "20150101"))
        self.assertEqual(record["config_identity"]["job"]["start_date"], "20150101")

    def test_cron_pit_event_job_builds_the_month_of_the_session_it_launches_on(self):
        # Paper decides session D at 05:40 through a replay slot that ends on D,
        # and the snapshot gateway refuses a slot month the store never built.
        # The scheduled 03:35 launch on D must therefore build through D: ending
        # on the previous session left D's month unbuilt on the first session of
        # every month and after every holiday (2026-10-08, after National Day).
        schedule_path = Path(__file__).resolve().parents[2] / "configs" / "tushare_update_schedule.json"
        job = json.loads(schedule_path.read_text(encoding="utf-8"))["jobs"]["cn_nightly_pit_event_build"]
        days = [
            day.strftime("%Y%m%d")
            for span in (("20260528", "20260602"), ("20260929", "20261012"))
            for day in pd.date_range(*span)
        ]
        closed = {"20260530", "20260531", *(f"202610{day:02d}" for day in range(1, 8)), "20261010", "20261011"}
        sessions = [day for day in days if day not in closed]
        trade_cal = self.raw_dir / "trade_cal" / "exchange=SSE" / "year=2026.parquet"
        trade_cal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cal_date": days, "is_open": ["0" if day in closed else "1" for day in days]}).to_parquet(
            trade_cal, index=False
        )
        config_path = self.root / "schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_start_date": "20200101",
            "default_raw_dir": "raw",
            "default_pit_root": "pit",
            "jobs": {"cn_nightly_pit_event_build": job},
        }), encoding="utf-8")
        for launch in days[1:]:
            launched_at = datetime.strptime(f"{launch}0335", "%Y%m%d%H%M").replace(tzinfo=ZoneInfo("Asia/Shanghai"))

            class LaunchClock(datetime):
                @classmethod
                def now(cls, tz=None, _at=launched_at):
                    return _at.astimezone(tz)

            # A short --start-date keeps the store small; only the end is under test.
            args = argparse.Namespace(
                config=str(config_path), job="cn_nightly_pit_event_build", start_date="20260501",
                end_date=None, dry_run=False, force_run=False,
            )
            with self.subTest(launch=launch), patch.object(cron_update, "datetime", LaunchClock):
                ctx = cron_update.build_context(args)
                # Weekend and holiday launches resolve to the previous session,
                # an unchanged range the skip rule suppresses.
                self.assertEqual(ctx.end_date, max(day for day in sessions if day <= launch))
                if launch not in sessions:
                    continue
                store = self.root / "pit" / launch
                _start, months = month_aligned_replace_window(ctx.start_date, ctx.end_date)
                FundamentalEventsBuilder(self.raw_dir).write_partitioned(
                    pd.DataFrame(), store, replace_months=months, replace_datasets=("income_vip",)
                )
                previous = max(day for day in sessions if day < launch)
                _require_event_months(
                    store,
                    ("income_vip",),
                    pd.Timestamp(previous, tz="Asia/Shanghai"),
                    pd.Timestamp(launch, tz="Asia/Shanghai") + pd.Timedelta(days=1, seconds=-1),
                    slot=f"{previous}..{launch}",
                )

    def test_cron_job_start_date_is_refused_unless_a_valid_bound(self):
        config = {"default_start_date": "20200101"}
        for job, message in (
            ({"start_date": "2015-01-01"}, "YYYYMMDD"),
            ({"start_date": "20151301"}, "YYYYMMDD"),
            ({"start_date": 20150101}, "YYYYMMDD"),
            ({"start_date": "20261001"}, "after its end date 20260930"),
            ({"start_date": "20150101", "start_date_lookback_days": 30}, "not both"),
        ):
            with self.subTest(job=job), self.assertRaisesRegex(ValueError, message):
                cron_update.resolve_job_start_date(job, config, "20260930")

    def test_cron_download_tier_job_builds_targeted_command(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_update_args": ["--min-interval-seconds", "0.22"]},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_preopen_board_backfill_0850",
            job={"operation": "download_tier", "tier": "board_trading", "extra_args": ["--datasets", "kpl_list", "--force"]},
            start_date="20260601",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_download.py download --tier board_trading", text)
        self.assertIn("--start-date 20260601 --end-date 20260601", text)
        self.assertIn("--raw-dir raw", text)
        self.assertIn("--datasets kpl_list --force", text)

    def test_cron_revision_sentinel_job_builds_audit_command(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_update_args": ["--min-interval-seconds", "0.22"]},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_daily_revision_sentinel",
            job={"operation": "revision_sentinel", "extra_args": ["--sample-size", "12", "--datasets", "adj_factor"]},
            start_date="20200101",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_audit.py revision-sentinel", text)
        self.assertIn("--start-date 20200101 --end-date 20260601", text)
        self.assertIn("--raw-dir raw", text)
        self.assertIn("--sample-size 12 --datasets adj_factor", text)

    def test_cron_event_flow_audit_job_builds_targeted_status_refresh(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_preopen_event_flow_audit_0920",
            job={"operation": "audit_event_flow"},
            start_date="20200101",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 1)
        text = " ".join(commands[0])
        self.assertIn("scripts/data/tushare_audit.py event-flow", text)
        self.assertIn("--start-date 20200101 --end-date 20260601", text)
        self.assertIn("--raw-dir raw", text)

    def test_cron_pit_event_pipeline_builds_and_audits_fundamental_events(self):
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_pit_root": "pit"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_pit_event_build",
            job={
                "operation": "pit_event_pipeline",
                "fundamental_events_root": "pit/fundamental_events",
                "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
            },
            start_date="20260201",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertEqual(len(commands), 2)
        self.assertIn("scripts/data/build_pit_events.py build-fundamental-events", " ".join(commands[0]))
        self.assertIn("--raw-dir raw --output-root pit/fundamental_events", " ".join(commands[0]))
        self.assertIn("scripts/data/build_pit_events.py audit-fundamental-events", " ".join(commands[1]))
        self.assertIn("--events-root pit/fundamental_events", " ".join(commands[1]))
        self.assertIn("--require-partitions", " ".join(commands[1]))

    def test_cron_pit_event_pipeline_builds_and_audits_the_month_aligned_window(self):
        # A manual mid-month --start-date must month-align for BOTH commands:
        # the builder aligns its replace window internally, so an unaligned
        # audit start would flag the first half-month as outside_audit_window.
        ctx = cron_update.RunContext(
            config={"default_raw_dir": "raw", "default_pit_root": "pit", "default_start_date": "20200101"},
            repo_root=self.root,
            python="/env/python",
            job_name="cn_nightly_pit_event_build",
            job={"operation": "pit_event_pipeline"},
            start_date="20260715",
            end_date="20260601",
            timezone_name="Asia/Shanghai",
        )

        commands = cron_update.build_job_commands(ctx)

        self.assertIn("--start-date 20260701 --end-date 20260601", " ".join(commands[0]))
        self.assertIn("--start-date 20260701 --end-date 20260601", " ".join(commands[1]))

    def test_cron_lock_blocks_while_held_and_releases_on_exit(self):
        # flock is per open-file-description: a second acquire in the same
        # process must block exactly like a second process would.
        held = cron_update.acquire_lock(self.root, wait_seconds=0)
        try:
            with self.assertRaisesRegex(RuntimeError, "lock is held"):
                cron_update.acquire_lock(self.root, wait_seconds=0)
        finally:
            held.release()
        reacquired = cron_update.acquire_lock(self.root, wait_seconds=0)
        reacquired.release()

    def test_cron_lock_file_from_dead_process_never_blocks(self):
        # A leftover lock FILE (crash, kill -9, PID reuse) carries no kernel
        # flock, so the next run acquires immediately - no stale-lock heuristics.
        lock_file = self.root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("pid=999999999\nstarted_at=2020-01-01T00:00:00+00:00\n", encoding="utf-8")

        acquired = cron_update.acquire_lock(self.root, wait_seconds=0)
        self.assertEqual(acquired.path, lock_file)
        self.assertIn(f"pid={cron_update.os.getpid()}", acquired.path.read_text(encoding="utf-8"))
        acquired.release()

    def test_cron_multi_command_jobs_fail_fast(self):
        ctx = cron_update.RunContext(
            config={},
            repo_root=self.root,
            python="/env/python",
            job_name="unit_fail_fast",
            job={"fail_fast": True},
            start_date="20200101",
            end_date="20200102",
            timezone_name="Asia/Shanghai",
        )
        commands = [["cmd1"], ["cmd2"], ["cmd3"]]
        calls = []

        class Result:
            def __init__(self, returncode):
                self.returncode = returncode

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result(1 if command == ["cmd2"] else 0)

        with patch.object(cron_update, "run_probe"), patch.object(cron_update.subprocess, "run", side_effect=fake_run):
            code = cron_update.run_update(ctx, commands, self.root / "cron.log")

        self.assertEqual(code, 1)
        self.assertEqual(calls, [["cmd1"], ["cmd2"]])

    def test_cron_multi_command_jobs_can_continue_after_error(self):
        ctx = cron_update.RunContext(
            config={},
            repo_root=self.root,
            python="/env/python",
            job_name="unit_continue",
            job={"fail_fast": False},
            start_date="20200101",
            end_date="20200102",
            timezone_name="Asia/Shanghai",
        )
        commands = [["cmd1"], ["cmd2"], ["cmd3"]]
        calls = []

        class Result:
            def __init__(self, returncode):
                self.returncode = returncode

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result(1 if command == ["cmd2"] else 0)

        with patch.object(cron_update, "run_probe"), patch.object(cron_update.subprocess, "run", side_effect=fake_run):
            code = cron_update.run_update(ctx, commands, self.root / "cron_continue.log")

        self.assertEqual(code, 1)
        self.assertEqual(calls, commands)

    def test_production_evening_job_excludes_text_and_text_job_covers_the_tier(self):
        schedule = Path(__file__).resolve().parents[2] / "configs" / "tushare_update_schedule.json"
        config = json.loads(schedule.read_text(encoding="utf-8"))
        jobs = config["jobs"]
        self.assertIn("--no-include-text", jobs["cn_evening_full"]["extra_args"])
        text_job = jobs["cn_nightly_text_full"]
        self.assertEqual(text_job["operation"], "download_tier")
        self.assertEqual(text_job["tier"], "text_evidence")
        # Natural-day window: no trading-calendar mode, so a weekend launch
        # resolves a new end date instead of skipping as an unchanged range.
        self.assertNotIn("end_date_mode", text_job)
        self.assertNotIn("end_date_mode", jobs["cn_nightly_text_audit"])
        # Every text dataset but anns_d, which has its own job.
        self.assertEqual(text_job["extra_args"][0], "--datasets")
        self.assertEqual(text_job["extra_args"][-1], "--force")
        self.assertEqual(
            sorted(text_job["extra_args"][1:-1]),
            sorted(set(specs.TEXT_DATASETS) - {"anns_d"}),
        )
        self.assertEqual(jobs["cn_nightly_anns_full"]["extra_args"], ["--datasets", "anns_d", "--force"])
        # The global tier and the ann-date disclosure tables are natural-day
        # domains too: overseas sessions and weekend announcements must not
        # wait for the next trading evening.
        self.assertIn("--no-include-global", jobs["cn_evening_full"]["extra_args"])
        global_job = jobs["cn_nightly_global_full"]
        self.assertEqual((global_job["operation"], global_job["tier"]), ("download_tier", "global"))
        self.assertNotIn("end_date_mode", global_job)
        disclosure_job = jobs["cn_nightly_disclosure_full"]
        self.assertEqual(disclosure_job["operation"], "download_event_flow")
        self.assertNotIn("end_date_mode", disclosure_job)
        disclosure_datasets = {"top10_holders", "top10_floatholders", "stk_holdernumber", "stk_holdertrade", "repurchase"}
        self.assertTrue(disclosure_datasets.issubset(set(disclosure_job["extra_args"])))
        # Single producer per dataset: the evening event list no longer
        # carries the disclosure tables.
        self.assertFalse(disclosure_datasets & set(jobs["cn_evening_full"]["extra_args"]))
        for name in ("cn_nightly_text_full", "cn_nightly_text_audit", "cn_nightly_global_full", "cn_nightly_disclosure_full"):
            # A Sunday resolves to itself: no trade-calendar rollback, so
            # skip_if_already_ok never sees an unchanged weekend range.
            self.assertEqual(
                cron_update.resolve_job_end_date(jobs[name], self.root, "raw", "20260621"),
                "20260621",
                name,
            )


class CronFailureRecordingTest(unittest.TestCase):
    """A cron job that aborts before run_update must still demote its per-job
    state record: during the 2026-07-23..29 outage, 9 of 14 jobs kept a
    week-old ``ok`` because a pre-run RuntimeError propagated past the state
    writer (silent false success)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True)
        self.jobs_root = self.root / "runtime" / "jobs"
        self.jobs_root.mkdir(parents=True)
        (self.jobs_root / "evening.json").write_text(
            json.dumps({"status": "ok", "end_date": "20260722"}), encoding="utf-8"
        )
        config_path = self.root / "schedule.json"
        config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {"evening": {"operation": "update"}},
        }), encoding="utf-8")
        self.args = argparse.Namespace(
            config=str(config_path), job="evening",
            start_date="20260623", end_date="20260723",
            dry_run=False, force_run=False,
            then=[],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _job_record(self) -> dict:
        return json.loads((self.jobs_root / "evening.json").read_text(encoding="utf-8"))

    def _run_main(self):
        return (
            patch.object(cron_update, "parse_args", return_value=self.args),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "prune_run_logs"),
            patch.object(cron_update, "JOB_STATE_ROOT", self.jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", self.jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", return_value=_StubLock()),
        )

    def test_pre_run_abort_records_error_state(self) -> None:
        commit_generation(self.raw_dir)
        patches = self._run_main()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], \
                patch.object(cron_update, "begin_raw_generation_update", side_effect=RuntimeError("record unwritable")), \
                redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "record unwritable"):
            cron_update.main()

        record = self._job_record()
        self.assertEqual(record["status"], "error")
        self.assertIn("record unwritable", record["error"])
        self.assertEqual(record["end_date"], "20260723")

    def test_nonzero_returncode_records_error(self) -> None:
        before = commit_generation(self.raw_dir)
        patches = self._run_main()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], \
                patch.object(cron_update, "run_update", return_value=1), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(cron_update.main(), 1)

        record = self._job_record()
        self.assertEqual(record["status"], "error")
        # No run log exists (run_update is stubbed): the summary falls back to
        # the exit code, so the state never records a content-free error.
        self.assertEqual(record["error"], "job_returncode=1")
        # A failed job fences nothing: what it wrote is committed as a new
        # generation, and the next job of any name runs.
        generation = json.loads((self.raw_dir / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual(generation["state"], "committed")
        self.assertNotEqual(generation["generation_id"], before["generation_id"])


class CronFollowUpChainTest(unittest.TestCase):
    """A job that mutates the lake and the jobs that must follow it run as one
    invocation.

    On 2026-09-19 the weekly fundamental sweep held the updater lock from
    06:13Z to 10:15Z and all three follow-ups, scheduled 60-90 minutes after
    it, died on lock timeout: the generation was committed with sweep-mutated
    fundamentals, no rebuilt PIT events and no audit. The chain has to hold by
    construction -- follow-ups start when the primary is done, each takes the
    lock itself, and one failure never starves the next step."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True)
        commit_generation(self.raw_dir)
        self.jobs_root = self.root / "runtime" / "jobs"
        self.config_path = self.root / "schedule.json"
        self.config_path.write_text(json.dumps({
            "schema_version": 1,
            "timezone": "Asia/Shanghai",
            "repo_root": str(self.root),
            "python": "/env/python",
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "jobs": {
                # The sweep mutates; its audits only read.
                "sweep": {"operation": "update"},
                "deep_audit": {"operation": "revision_sentinel"},
                "rebuild": {"operation": "revision_sentinel"},
            },
        }), encoding="utf-8")
        self.locks = {"acquired": 0, "released": 0}
        self.ran: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, then: list[str]) -> argparse.Namespace:
        return argparse.Namespace(
            config=str(self.config_path), job="sweep",
            start_date="20260601", end_date="20260919",
            dry_run=False, force_run=False, then=then,
        )

    def _lock(self, repo_root: Path, wait_seconds: int):
        chain = self
        self.locks["acquired"] += 1

        class _CountedLock:
            fd = 7

            def release(self) -> None:
                chain.locks["released"] += 1

        return _CountedLock()

    def _run_main(self, then: list[str], codes: dict[str, int] | None = None) -> int:
        codes = codes or {}

        def fake_run_update(ctx, commands, log_path, lock_fd=None, report=None):
            # The lock must be held while a step works and released before the
            # next one starts.
            self.assertEqual(self.locks["acquired"] - self.locks["released"], 1)
            self.ran.append(ctx.job_name)
            return codes.get(ctx.job_name, 0)

        with (
            patch.object(cron_update, "parse_args", return_value=self._args(then)),
            patch.object(cron_update.os, "chdir"),
            patch.object(cron_update, "prune_run_logs"),
            patch.object(cron_update, "JOB_STATE_ROOT", self.jobs_root),
            patch.object(cron_update, "RUN_REPORT_ROOT", self.jobs_root.parent / "runs"),
            patch.object(cron_update, "RUN_LOG_ROOT", self.root / "logs" / "cron"),
            patch.object(cron_update, "acquire_lock", side_effect=self._lock),
            patch.object(cron_update, "run_update", side_effect=fake_run_update),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = cron_update.main()
        self.output = output.getvalue()
        return code

    def test_follow_ups_run_in_order_and_ignore_their_own_already_ok_state(self) -> None:
        self.assertEqual(self._run_main(["deep_audit", "rebuild"]), 0)
        self.assertEqual(self.ran, ["sweep", "deep_audit", "rebuild"])
        # Three steps, three lock acquisitions, none held between them.
        self.assertEqual(self.locks, {"acquired": 3, "released": 3})
        for job in ("sweep", "deep_audit", "rebuild"):
            record = json.loads((self.jobs_root / f"{job}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")

        # A second invocation: the primary skips on its own ok state, but a
        # follow-up exists because the lake changed under it, so it is forced.
        self.ran.clear()
        self.assertEqual(self._run_main(["deep_audit", "rebuild"]), 0)
        self.assertEqual(self.ran, ["deep_audit", "rebuild"])

    def test_a_failing_follow_up_does_not_starve_the_ones_after_it(self) -> None:
        # An audit reports findings through a non-zero exit; the PIT rebuild
        # behind it must still run, and the invocation must still fail loudly.
        self.assertEqual(self._run_main(["deep_audit", "rebuild"], {"deep_audit": 1}), 1)
        self.assertEqual(self.ran, ["sweep", "deep_audit", "rebuild"])
        reported = [json.loads(line) for line in self.output.splitlines()]
        failed = [item for item in reported if item.get("follow_up") == "deep_audit"]
        self.assertEqual(failed[-1]["status"], "error")
        self.assertEqual(failed[-1]["after"], "sweep")

    def test_a_failed_primary_is_still_followed_by_its_follow_ups(self) -> None:
        # What a failed sweep wrote is committed, so its audit and rebuild
        # follow it; the invocation still reports the failure.
        self.assertEqual(self._run_main(["deep_audit", "rebuild"], {"sweep": 1}), 1)
        self.assertEqual(self.ran, ["sweep", "deep_audit", "rebuild"])
        generation = json.loads((self.raw_dir / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual(generation["state"], "committed")

    def test_an_unknown_follow_up_is_refused_before_the_primary_runs(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown --then job"):
            self._run_main(["deep_audit", "typo"])
        self.assertEqual(self.ran, [])


class FailureSummaryTest(unittest.TestCase):
    """The persisted job state's error field must distinguish "the audit
    completed and found data errors" from "the tool crashed": completed audit
    domain summaries win, else the traceback's final exception line, else the
    bare exit code."""

    def test_audit_findings_traceback_and_fallback_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "started_at=x\n"
                "core_market audit status=ok errors=0 warnings=3 output=/s/core.json\n"
                "board_trading audit status=error errors=13 warnings=2 output=/s/board.json\n",
                encoding="utf-8",
            )
            summary = cron_update.summarize_failure_from_log(log, 1)
            self.assertIn("board_trading audit status=error errors=13 warnings=2", summary)
            self.assertIn("core_market audit status=ok errors=0 warnings=3", summary)
            self.assertNotIn("output=", summary)

            log.write_text(
                "Traceback (most recent call last):\n"
                '  File "download.py", line 1551, in download_event_flow\n'
                "RuntimeError: required event/flow partitions returned zero rows\n",
                encoding="utf-8",
            )
            self.assertEqual(
                cron_update.summarize_failure_from_log(log, 1),
                "RuntimeError: required event/flow partitions returned zero rows",
            )

            self.assertEqual(
                cron_update.summarize_failure_from_log(Path(tmp) / "missing.log", 7),
                "job_returncode=7",
            )

    def test_every_producers_own_summary_line_is_read_back(self) -> None:
        """Written with the producers' own formatter, not look-alikes: the
        summaries that used to be dropped are the ones whose name is two words
        or whose counters are not errors/warnings, and dropping them left the
        operator with a bare return code on the two jobs that fail most."""

        lines = [
            audit.audit_summary_line(
                "core_market", "warning", errors=0, warnings=7, output="/s/core.json"
            ),
            audit.audit_summary_line(
                "intraday_by_date", "error", errors=1, warnings=0, output="/s/intraday.json"
            ),
            audit.audit_summary_line(
                "revision_sentinel",
                "error",
                events=27,
                errors=0,
                remote_zero=0,
                no_effective_checks=0,
                output="/s/revision.json",
                ledger="/s/revision.jsonl",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text("started_at=x\n" + "\n".join(lines) + "\n", encoding="utf-8")
            summary = cron_update.summarize_failure_from_log(log, 1)
        self.assertIn("core_market audit status=warning errors=0 warnings=7", summary)
        self.assertIn("intraday_by_date audit status=error errors=1 warnings=0", summary)
        self.assertIn(
            "revision_sentinel audit status=error events=27 errors=0 remote_zero=0"
            " no_effective_checks=0",
            summary,
        )
        # Paths belong in the log, not in the one line a job state persists.
        self.assertNotIn("output=", summary)
        self.assertNotIn("ledger=", summary)

    def test_an_audit_name_of_two_words_is_refused_at_the_formatter(self) -> None:
        with self.assertRaisesRegex(ValueError, "single token"):
            audit.audit_summary_line("intraday by-date", "ok", errors=0, warnings=0)

    def test_the_pit_event_entrypoint_lets_its_failure_reach_the_log(self) -> None:
        """The nightly PIT event job prints no audit summary, so its failures
        can only reach the job state as the exception line a traceback ends on.
        Catching every exception and reprinting it as one-line JSON hid that
        line and left the operator with a bare return code."""

        spec = importlib.util.spec_from_file_location(
            "build_pit_events_failure_path", Path("scripts/data/build_pit_events.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "build_pit_events.py",
                "build-fundamental-events",
                "--raw-dir",
                str(Path(tmp) / "raw"),
                "--output-root",
                str(Path(tmp) / "events"),
                "--start-date",
                "20200101",
                "--end-date",
                "20200131",
            ]
            failure = RuntimeError("refusing to replace the window with an empty build")
            with (
                patch.object(module, "run_build_fundamental_events", side_effect=failure),
                patch.object(sys, "argv", argv),
                self.assertRaises(RuntimeError),
            ):
                module.main()
            log = Path(tmp) / "run.log"
            log.write_text(
                "".join(traceback.format_exception(failure)), encoding="utf-8"
            )
            self.assertEqual(
                cron_update.summarize_failure_from_log(log, 1),
                "RuntimeError: refusing to replace the window with an empty build",
            )

    def test_no_audit_hand_formats_its_own_summary_line(self) -> None:
        """One contract, one formatter. A producer that builds the line itself
        is how the summary became unreadable in the first place."""

        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertEqual(
            [line.strip() for line in source.splitlines() if "audit status=" in line],
            ['_AUDIT_SUMMARY_HEAD = "audit status="'],
        )


if __name__ == "__main__":
    unittest.main()
