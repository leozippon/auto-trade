"""The versioned raw lake: what a partition write may never lose, what it
records, and how the runner and the operator commands build on it."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autotrade.data_sources.tushare import common, cron_update, lake, versions
from autotrade.data_sources.tushare import io as tushare_io
from autotrade.environment.data import research_release

KEYS = ["ts_code", "ann_date"]


def rows(*records: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(list(records), columns=["ts_code", "ann_date", "vol"])


class VersionedPartitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw = self.root / "raw"
        self.path = self.raw / "repurchase" / "month=202610.parquet"
        self.ledger = self.root / "revision_events.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def publish(self, frame: pd.DataFrame, path: Path | None = None, **extra) -> str:
        with redirect_stdout(io.StringIO()):
            return versions.publish(
                path or self.path, frame, api_name="repurchase", params={}, fields=list(frame.columns),
                key_columns=KEYS, ledger=self.ledger, **extra,
            )

    def events(self) -> list[dict]:
        if not self.ledger.exists():
            return []
        return [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]

    def observed(self, path: Path | None = None) -> dict[str, str]:
        table = versions.read_observations(path or self.path)
        return {row.ts_code: row.first_seen_at for row in table.itertuples()}

    def visible_since(self) -> str:
        return common.parquet_meta(self.path)["visible_since"]

    def test_a_pure_append_replaces_the_partition_without_keeping_a_copy(self) -> None:
        self.assertEqual(self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0))), "appended")
        vintage = versions.dataset_vintage(self.path.parent)
        self.assertEqual(self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0))), "unchanged")
        self.assertEqual(
            self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0), ("C", "20261003", 3.0))), "appended"
        )
        self.assertEqual(len(pd.read_parquet(self.path)), 3)
        self.assertEqual(versions.list_versions(self.path), [])
        self.assertEqual(self.events(), [])
        # Appending is not a change of history: the vintage stays.
        self.assertEqual(versions.dataset_vintage(self.path.parent), vintage)

    def test_a_restatement_keeps_the_replaced_partition_and_reverts_by_pointer(self) -> None:
        before = rows(("A", "20261001", 1.0), ("B", "20261002", 2.0))
        self.publish(before)
        base_vintage = versions.dataset_vintage(self.path.parent)
        self.assertEqual(self.publish(rows(("A", "20261001", 9.0), ("C", "20261003", 3.0))), "restated")

        (superseded,) = versions.list_versions(self.path)
        self.assertTrue(superseded.endswith("-superseded"))
        kept = versions.version_dir(self.path) / f"{superseded}.parquet"
        self.assertTrue(pd.read_parquet(kept).equals(before))
        self.assertTrue(common.committed_partition_intact(kept))
        (event,) = self.events()
        self.assertEqual(
            {key: event[key] for key in ("kind", "partition", "old_keys", "added_keys", "removed_keys", "changed_keys")},
            {"kind": "restatement", "partition": "month=202610", "old_keys": 2, "added_keys": 1, "removed_keys": 1, "changed_keys": 1},
        )
        self.assertEqual(event["changed_columns"], {"vol": 1})
        self.assertEqual(event["superseded_version"], superseded)
        restated_vintage = versions.dataset_vintage(self.path.parent)
        self.assertNotEqual(restated_vintage, base_vintage)

        with redirect_stdout(io.StringIO()):
            result = versions.restore(self.path, superseded, ledger=self.ledger)
        self.assertEqual(result["kind"], "revert")
        self.assertTrue(pd.read_parquet(self.path).equals(before))
        self.assertTrue(common.committed_partition_intact(self.path))
        # The revert swapped the pointer: the old file is visible again, the
        # content it replaced is kept, and the vintage is the old one.
        self.assertEqual(os.stat(self.path).st_ino, os.stat(kept).st_ino)
        self.assertEqual(len(versions.list_versions(self.path)), 2)
        self.assertEqual(versions.dataset_vintage(self.path.parent), base_vintage)

    def test_an_answer_removing_more_than_half_is_staged_until_promoted(self) -> None:
        current = rows(*[(code, "20261001", 1.0) for code in "ABCDE"])
        self.publish(current)
        shrunk = rows(("A", "20261001", 1.0), ("B", "20261001", 1.0), ("F", "20261001", 1.0))
        self.assertEqual(self.publish(shrunk), "staged")
        self.assertTrue(pd.read_parquet(self.path).equals(current))
        (staged,) = versions.list_versions(self.path)
        self.assertTrue(staged.endswith("-staged"))
        self.assertEqual(self.observed().keys(), set("ABCDE"))
        # The same shrunk answer the next night stages nothing new.
        self.assertEqual(self.publish(shrunk), "staged")
        self.assertEqual(versions.list_versions(self.path), [staged])
        self.assertEqual([event["kind"] for event in self.events()], ["staged"])

        with redirect_stdout(io.StringIO()):
            result = versions.restore(self.path, staged, ledger=self.ledger)
        self.assertEqual(result["kind"], "promotion")
        self.assertEqual(set(pd.read_parquet(self.path)["ts_code"]), {"A", "B", "F"})
        self.assertTrue(result["superseded_version"].endswith("-superseded"))
        # A staged key becomes observed when it becomes visible, not when pulled.
        self.assertEqual(self.observed()["F"], self.visible_since())

    def test_removing_exactly_half_is_an_accepted_restatement(self) -> None:
        self.publish(rows(*[(code, "20261001", 1.0) for code in "ABCD"]))
        self.assertEqual(self.publish(rows(("A", "20261001", 1.0), ("B", "20261001", 1.0))), "restated")

    def test_a_defective_answer_keeps_the_current_version(self) -> None:
        current = rows(("A", "20261001", 1.0))
        self.publish(current)
        identity = common.parquet_meta(self.path)["write_id"]
        report = self.root / "report.jsonl"
        with patch.dict(os.environ, {versions.RUN_REPORT_ENV: str(report)}):
            self.assertEqual(self.publish(current.iloc[:0]), "defect")
            with redirect_stdout(io.StringIO()), versions.answer_defects(self.path, ledger=self.ledger):
                raise versions.DefectiveAnswer("repurchase returned inconsistent fields while paging")
        self.assertTrue(pd.read_parquet(self.path).equals(current))
        self.assertEqual(common.parquet_meta(self.path)["write_id"], identity)
        self.assertEqual([event["kind"] for event in self.events()], ["defect", "defect"])
        outcomes = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [(item["dataset"], item["partition"], item["outcome"]) for item in outcomes],
            [("repurchase", "month=202610", "defect")] * 2,
        )

    def test_the_observation_table_records_when_each_key_first_became_visible(self) -> None:
        # Keys visible before versioning started (a sidecar without
        # visible_since) have no row.
        common.write_parquet(self.path, rows(("A", "20261001", 1.0)), api_name="repurchase", params={}, fields=[])
        self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0)))
        first = self.observed()
        self.assertEqual(first, {"B": self.visible_since()})

        self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0), ("C", "20261003", 3.0)))
        self.assertEqual(self.observed()["B"], first["B"])
        self.assertEqual(self.observed()["C"], self.visible_since())
        # A value change adds no row; a withdrawn key keeps its row, and so
        # does the same key when it comes back.
        self.publish(rows(("A", "20261001", 7.0), ("C", "20261003", 3.0)))
        self.publish(rows(("A", "20261001", 7.0), ("B", "20261002", 2.0), ("C", "20261003", 3.0)))
        self.assertEqual(self.observed()["B"], first["B"])
        self.assertNotIn("A", self.observed())
        self.assertEqual(
            list(versions.read_observations(self.path).columns), ["ts_code", "ann_date", versions.FIRST_SEEN_COLUMN]
        )

    def test_rebuilt_observations_are_never_earlier_than_the_recorded_ones(self) -> None:
        common.write_parquet(self.path, rows(("A", "20261001", 1.0)), api_name="repurchase", params={}, fields=[])
        self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0)))
        self.publish(rows(("A", "20261001", 1.0), ("B", "20261002", 2.0), ("C", "20261003", 3.0)))
        self.publish(rows(("A", "20261001", 2.0), ("C", "20261003", 3.0)))  # restatement: copy kept
        recorded = self.observed()
        with redirect_stdout(io.StringIO()):
            versions.rebuild_observations(self.path)
        rebuilt = self.observed()
        # The pre-versioning state of A was replaced by an append, which keeps
        # no copy, so the rebuild cannot know A predates versioning: it gives A
        # the time of the first retained state, a later visibility than the
        # vendor-date rule an absent row means -- never an earlier one.
        self.assertEqual(set(rebuilt), set(recorded) | {"A"})
        for key, instant in recorded.items():
            self.assertGreaterEqual(datetime.fromisoformat(rebuilt[key]), datetime.fromisoformat(instant), key)

    def test_a_crash_never_leaves_the_visible_partition_missing_or_torn(self) -> None:
        current = rows(("A", "20261001", 1.0), ("B", "20261002", 2.0))
        self.publish(current)
        restated = rows(("A", "20261001", 5.0), ("B", "20261002", 2.0), ("C", "20261003", 3.0))

        # 1. After the superseded copy, before anything else: an extra copy.
        with patch.object(versions, "_record_observations", side_effect=OSError("crash")), self.assertRaises(OSError):
            self.publish(restated)
        self.assertTrue(pd.read_parquet(self.path).equals(current))
        self.assertTrue(common.committed_partition_intact(self.path))
        self.assertEqual(len(versions.list_versions(self.path)), 1)

        # 2. After the observation table, before the partition: its new rows
        # belong to no visible commit, are not read, and the next write drops them.
        with patch.object(versions, "write_parquet", side_effect=OSError("crash")), self.assertRaises(OSError):
            self.publish(restated)
        self.assertTrue(pd.read_parquet(self.path).equals(current))
        self.assertNotIn("C", self.observed())

        # 3. Between the parquet and the sidecar rename: the visible parquet
        # is whole (the new one), and the next write rolls the commit forward.
        real_replace = os.replace

        def replace(source, target):
            if str(target).endswith(".meta.json") and Path(target).parent == self.path.parent:
                raise OSError("crash")
            return real_replace(source, target)

        with patch.object(tushare_io.os, "replace", side_effect=replace), self.assertRaises(OSError):
            self.publish(restated)
        self.assertTrue(pd.read_parquet(self.path).equals(restated))
        self.assertFalse(common.committed_partition_intact(self.path))
        versions.complete_interrupted_commit(self.path)
        self.assertTrue(common.committed_partition_intact(self.path))
        self.assertEqual(self.observed()["C"], self.visible_since())
        self.assertEqual(self.publish(restated), "unchanged")

    def test_duplicate_keys_are_dropped_the_same_way_whatever_the_order(self) -> None:
        served = rows(("A", "20261001", 2.0), ("A", "20261001", 1.0), ("B", "20261002", 1.0))
        self.publish(served, deduplicate=True)
        self.assertEqual(pd.read_parquet(self.path)["vol"].tolist(), [1.0, 1.0])
        self.assertEqual(self.publish(served.iloc[::-1], deduplicate=True), "unchanged")
        events = self.events()
        self.assertEqual({event["kind"] for event in events}, {"deduplicated"})
        self.assertEqual({event["duplicate_rows_dropped"] for event in events}, {1})

    def test_releases_carry_observations_and_never_versions(self) -> None:
        self.publish(rows(("A", "20261001", 1.0)))
        self.publish(rows(("A", "20261001", 2.0)))
        self.assertTrue(versions.version_root(self.raw).is_dir())
        release = self.root / "release" / "raw"
        research_release._clone_tree(self.raw, release, kind="raw")
        cloned = release / "repurchase"
        self.assertEqual(
            sorted(path.name for path in cloned.iterdir()),
            ["month=202610.parquet", "month=202610.parquet.first_seen", "month=202610.parquet.meta.json"],
        )
        observations = versions.observation_path(self.path)
        self.assertEqual(os.stat(cloned / observations.name).st_ino, os.stat(observations).st_ino)
        self.assertFalse(any("raw_versions" in str(path) for path in release.rglob("*")))


class RunnerFoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw = self.root / "raw"
        (self.raw / "stk_holdertrade").mkdir(parents=True)
        (self.raw / "top_list").mkdir(parents=True)
        self.config = {
            "default_raw_dir": "raw",
            "default_start_date": "20200101",
            "interfaces": [
                {"dataset": "stk_holdertrade", "history_repull": "on_restatement"},
                {"dataset": "top_list", "history_repull": "window_only"},
            ],
        }
        self.ctx = cron_update.RunContext(
            config=self.config, repo_root=self.root, python="/env/python", job_name="disclosure",
            job={"operation": "download_event_flow"}, start_date="20261001", end_date="20261024",
            timezone_name="Asia/Shanghai",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def outcome(dataset: str, partition: str, kind: str, detail: str = "") -> dict:
        return {"dataset": dataset, "partition": partition, "outcome": kind, "detail": detail, "version": "", "at": "t"}

    def test_partition_issues_open_and_close_and_refresh_needs_no_defect(self) -> None:
        datasets: dict = {}
        issues: list = []
        cron_update.fold_outcomes(datasets, issues, [
            self.outcome("anns_d", "month=202609", "appended"),
            self.outcome("anns_d", "month=202610", "defect", "empty answer"),
            self.outcome("repurchase", "month=202610", "staged"),
        ], refreshed=True, at="2026-10-24T15:30:00+00:00")
        self.assertEqual(
            [(issue["kind"], issue["dataset"], issue["partition"]) for issue in issues],
            [("defect", "anns_d", "month=202610"), ("staged", "repurchase", "month=202610")],
        )
        self.assertNotIn("anns_d", datasets)
        cron_update.fold_outcomes(datasets, issues, [
            self.outcome("anns_d", "month=202610", "unchanged"),
        ], refreshed=True, at="2026-10-25T15:30:00+00:00")
        self.assertEqual([issue["kind"] for issue in issues], ["staged"])
        self.assertEqual(datasets["anns_d"]["refreshed_at"], "2026-10-25T15:30:00+00:00")

    def _finish(self, outcomes: list[dict], repull_code: int = 0) -> tuple[dict, list[list[str]]]:
        report = self.root / "runs" / "job.jsonl"
        report.parent.mkdir(exist_ok=True)
        report.write_text("".join(json.dumps(item) + "\n" for item in outcomes), encoding="utf-8")
        commands: list[list[str]] = []

        def fake_run_update(ctx, commands_, log_path, lock_fd=None, report=None):
            commands.extend(commands_)
            report.write_text(json.dumps(self.outcome("stk_holdertrade", "month=202001", "restated")) + "\n", encoding="utf-8")
            return repull_code

        with (
            patch.object(cron_update, "RUN_REPORT_ROOT", self.root / "runs"),
            patch.object(cron_update, "run_update", side_effect=fake_run_update),
            redirect_stdout(io.StringIO()),
        ):
            record = cron_update.finish_job(
                self.ctx, self.raw, {"generation_id": "old"}, {"job": "disclosure"}, report,
                commit=True, refreshed=True, log_path=self.root / "job.log", lock_fd=7,
            )
        return record, commands

    def test_a_restatement_queues_a_history_repull_that_runs_in_the_same_run(self) -> None:
        record, commands = self._finish([
            self.outcome("stk_holdertrade", "month=202610", "restated"),
            self.outcome("top_list", "trade_date=20261023", "restated"),
        ])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--datasets") + 1], "stk_holdertrade")
        self.assertEqual(commands[0][commands[0].index("--start-date") + 1], "20200101")
        self.assertEqual(record["issues"], [])
        self.assertEqual(record["state"], "committed")
        self.assertNotEqual(record["generation_id"], "old")
        self.assertEqual(set(record["datasets"]), {"stk_holdertrade", "top_list"})
        self.assertTrue(all(entry["vintage"] for entry in record["datasets"].values()))

    def test_a_failed_history_repull_stays_pending(self) -> None:
        record, commands = self._finish([self.outcome("stk_holdertrade", "month=202610", "restated")], repull_code=1)
        self.assertEqual(len(commands), 1)
        self.assertEqual(
            [(issue["kind"], issue["dataset"]) for issue in record["issues"]], [("history_repull", "stk_holdertrade")]
        )


class MigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw = self.root / "data" / "raw"
        anns = pd.DataFrame(
            [["20261001", "000001.SZ", "n", "t1", "u1", ""], ["20261001", "000001.SZ", "n", "t1", "u1", ""],
             ["20261002", "000002.SZ", "n", "t2", "u2", ""]],
            columns=["ann_date", "ts_code", "name", "title", "url", "rec_time"],
        )
        common.write_parquet(self.raw / "anns_d" / "month=202610.parquet", anns, api_name="anns_d", params={}, fields=list(anns.columns))
        common.write_parquet(self.raw / "anns_d" / "month=202609.parquet", anns.iloc[[2]], api_name="anns_d", params={}, fields=list(anns.columns))
        repurchase = pd.DataFrame(
            [["000001.SZ", "20261001", "", "实施", "", 1.0], ["000001.SZ", "20261001", "", "实施", "", 2.0]],
            columns=["ts_code", "ann_date", "end_date", "proc", "exp_date", "vol"],
        )
        common.write_parquet(self.raw / "repurchase" / "month=202610.parquet", repurchase, api_name="repurchase", params={}, fields=list(repurchase.columns))
        (self.raw / ".raw_generation.json").write_text(json.dumps({
            "schema_version": 2, "state": "committed", "generation_id": "g2", "completed_at": "2026-10-23T00:00:00+00:00",
        }), encoding="utf-8")
        self.config = self.root / "schedule.json"
        self.config.write_text(json.dumps({
            "schema_version": 1, "repo_root": str(self.root), "default_raw_dir": "data/raw",
            "default_start_date": "20200101", "default_lock_wait_seconds": 0, "jobs": {},
            "revision_monitor": {"ledger_path": str(self.root / "ledger.jsonl")},
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def migrate(self, apply: bool) -> dict:
        # The command works from the lake's repository root; chdir restores
        # this process's directory before the temporary root is deleted.
        with (
            chdir(self.root),
            patch.object(cron_update, "RUNTIME_ROOT", self.root / ".runtime" / "tushare"),
            redirect_stdout(io.StringIO()),
        ):
            return lake.migrate(str(self.config), apply=apply)

    def test_dry_run_changes_nothing_and_apply_is_idempotent(self) -> None:
        def state() -> dict[str, tuple[int, int]]:
            return {str(path): (path.stat().st_ino, path.stat().st_size) for path in sorted(self.raw.rglob("*")) if path.is_file()}

        before = state()
        landed = max(
            json.loads(path.read_text(encoding="utf-8"))["fetched_at"]
            for path in (self.raw / "anns_d").glob("*.meta.json")
        )
        report = self.migrate(apply=False)
        self.assertEqual(report["deduplicate"]["anns_d"], {"partitions": 1, "duplicate_rows": 1, "rows_before": 3})
        self.assertEqual(report["deduplicate"]["repurchase"], {"partitions": 1, "duplicate_rows": 1, "rows_before": 2})
        self.assertEqual(state(), before)
        self.assertFalse(versions.version_root(self.raw).exists())

        self.migrate(apply=True)
        record = json.loads((self.raw / ".raw_generation.json").read_text(encoding="utf-8"))
        self.assertEqual((record["schema_version"], record["state"], record["issues"]), (3, "committed", []))
        self.assertNotEqual(record["generation_id"], "g2")
        self.assertEqual(set(record["datasets"]), {"anns_d", "repurchase"})
        # The refresh time is the last landing, not the deduplication commit.
        self.assertEqual(record["datasets"]["anns_d"]["refreshed_at"], datetime.fromisoformat(landed).isoformat())
        self.assertEqual(len(pd.read_parquet(self.raw / "anns_d" / "month=202610.parquet")), 2)
        self.assertEqual(pd.read_parquet(self.raw / "repurchase" / "month=202610.parquet")["vol"].tolist(), [1.0])
        for dataset in ("anns_d", "repurchase"):
            (version,) = versions.list_versions(self.raw / dataset / "month=202610.parquet")
            self.assertTrue(version.endswith("-superseded"))
        events = [json.loads(line) for line in (self.root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(event["kind"], event["source"], event["duplicate_rows_dropped"]) for event in events], [("restatement", "migration", 1)] * 2)
        # Keys visible before versioning have no observation rows.
        self.assertFalse(versions.observation_path(self.raw / "anns_d" / "month=202610.parquet").exists())

        again = self.migrate(apply=True)
        self.assertEqual({item["partitions"] for item in again["deduplicate"].values()}, {0})
        self.assertEqual(json.loads((self.raw / ".raw_generation.json").read_text(encoding="utf-8"))["generation_id"], record["generation_id"])
        self.assertEqual(len(versions.list_versions(self.raw / "anns_d" / "month=202610.parquet")), 1)


if __name__ == "__main__":
    unittest.main()
