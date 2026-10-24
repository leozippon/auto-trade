"""Research-release retention (``tushare_lake.py prune-releases``): a release
stays exactly while a consumer records it, the rest go oldest first under the
updater and registry locks, and the raw lake they link into is untouched."""

from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from autotrade.data_sources.tushare import cron_update, lake
from autotrade.data_sources.tushare.io import UPDATER_LOCK_FD_ENV
from autotrade.environment.data.research_release import pin_research_release
from autotrade.pipelines.hitl_state import HITL_STATE_SCHEMA_VERSION
from tests.unit.research_release_fixture import (
    FUNDAMENTAL_EVENTS_ROOT,
    FUNDAMENTAL_EVENTS_STATUS,
    RAW_DIR,
    publish_release,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATIONS = ("g1", "g2", "g3", "g4", "g5", "g6", "g7")
KEPT = {"g2", "g3", "g4", "g5", "g7"}


class ReleasePruneTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for generation in GENERATIONS:
            publish_release(self.root, generation, datasets=("daily_basic",))
        self.releases = self.root / "data" / "research_releases"
        self.lock = self.root / ".runtime" / "tushare" / "locks" / "tushare_update.lock"
        # g7 is the committed generation; one consumer of each kind records
        # each of g2..g5, and only ended consumers record g1.
        self._write(self.root / "data" / "seed_8y" / "provider.json", {"generation_id": "g2"})
        self._arm("arm_stopped", state="stopped")
        self._pin(self.root / "experiments" / "arm_stopped", "g3")
        # Not started yet, so nothing pinned: its seed, outside data/, names the release.
        self._arm("arm_created", state=None, seed="seeds/elsewhere")
        self._write(self.root / "seeds" / "elsewhere" / "provider.json", {"generation_id": "g4"})
        self._book("book_live", "g5")
        self._arm("arm_completed", state="completed")
        self._pin(self.root / "experiments" / "arm_completed", "g1")
        self._book("book_killed", "g1", killed=True)
        self.config = self.root / "schedule.json"
        self._write(self.config, {
            "schema_version": 1, "repo_root": str(self.root), "default_raw_dir": RAW_DIR,
            "default_start_date": "20200101", "default_lock_wait_seconds": 0, "jobs": {},
        })

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _pin(self, directory: Path, generation: str | None) -> None:
        pin_research_release(
            experiment_dir=directory,
            raw_dir=self.root / RAW_DIR,
            fundamental_events_root=self.root / FUNDAMENTAL_EVENTS_ROOT,
            fundamental_events_status=self.root / FUNDAMENTAL_EVENTS_STATUS,
            generation_id=generation,
        )

    def _arm(self, name: str, *, state: str | None, seed: str | None = None) -> None:
        hitl = self.root / "experiments" / name / "hitl"
        hitl.mkdir(parents=True)
        if state is not None:
            self._write(hitl / "status.json", {"schema_version": HITL_STATE_SCHEMA_VERSION, "state": state})
        self._write(hitl / "params.json", {"pit_views_seed": seed} if seed else {})

    def _book(self, name: str, generation: str, *, killed: bool = False) -> None:
        book = self.root / "data" / "trading" / "paper" / name
        self._write(book / "book.json", {})
        if killed:
            (book / "verdict.jsonl").write_text(json.dumps({"status": "killed"}) + "\n", encoding="utf-8")
        self._pin(book / "pit" / generation, generation)

    def prune(self, *, apply: bool) -> tuple[dict, list[dict]]:
        out = io.StringIO()
        with (
            chdir(self.root),
            patch.object(cron_update, "RUNTIME_ROOT", self.root / ".runtime" / "tushare"),
            redirect_stdout(out),
        ):
            result = lake.prune_releases(str(self.config), apply=apply)
        return result, [json.loads(line) for line in out.getvalue().splitlines()]

    def kept(self) -> dict[str, list[str]]:
        _, lines = self.prune(apply=False)
        return {line["generation_id"]: line["kept_for"] for line in lines if line["action"] == "keep"}

    def published(self) -> set[str]:
        return {path.name for path in self.releases.iterdir() if not path.name.startswith(".")}

    def test_a_release_stays_exactly_while_a_consumer_records_it(self) -> None:
        result, lines = self.prune(apply=False)
        self.assertEqual(self.kept(), {
            "g2": ["seed data/seed_8y"],
            "g3": ["experiment arm_stopped (stopped)"],
            "g4": ["experiment arm_created (created)"],
            "g5": ["Paper book book_live"],
            "g7": ["raw generation record"],
        })
        self.assertEqual([line["generation_id"] for line in lines if line["action"] == "remove"], ["g1", "g6"])
        self.assertEqual(result, {"mode": "dry_run", "releases": 7, "kept": 5, "unreferenced": 2, "leftovers": 0})
        # A state retention cannot read is no proof the arm ended.
        (self.root / "experiments" / "arm_stopped" / "hitl" / "status.json").write_text("{", encoding="utf-8")
        self.assertEqual(self.kept()["g3"], ["experiment arm_stopped (unreadable)"])
        # Each consumer that ends lets its release go.
        shutil.rmtree(self.root / "data" / "seed_8y")
        for arm in ("arm_stopped", "arm_created"):
            self._arm_ends(arm)
        (self.root / "data" / "trading" / "paper" / "book_live" / "verdict.jsonl").write_text(
            json.dumps({"status": "killed"}) + "\n", encoding="utf-8"
        )
        self.assertEqual(self.kept(), {"g7": ["raw generation record"]})
        publish_release(self.root, "g8", datasets=("daily_basic",))
        self.assertEqual(self.kept(), {"g8": ["raw generation record"]})

    def _arm_ends(self, name: str) -> None:
        self._write(
            self.root / "experiments" / name / "hitl" / "status.json",
            {"schema_version": HITL_STATE_SCHEMA_VERSION, "state": "completed"},
        )

    def test_the_dry_run_removes_nothing(self) -> None:
        def tree() -> dict[str, int]:
            return {str(path): path.lstat().st_ino for path in self.root.rglob("*")}

        before = tree()
        self.prune(apply=False)
        self.assertEqual(tree(), before)

    def test_apply_removes_only_unreferenced_releases_and_counts_only_their_own_blocks(self) -> None:
        # An earlier retirement interrupted mid-delete left its hidden tree.
        leftover = self.releases / ".g0.0123456789.tmp"
        (leftover / "raw").mkdir(parents=True)
        (leftover / "raw" / "part.parquet").write_bytes(b"x" * 10_000)
        raw = self.root / RAW_DIR
        lake_files = {path: (path.lstat().st_ino, path.read_bytes()) for path in raw.rglob("*") if path.is_file()}
        expected = _blocks_freed_by_removing([leftover, self.releases / "g1", self.releases / "g6"])

        result, lines = self.prune(apply=True)

        self.assertEqual(self.published(), KEPT)
        self.assertEqual(sorted(path.name for path in self.releases.iterdir()), [".registry.lock", *sorted(KEPT)])
        removed = [line for line in lines if line["action"] == "removed"]
        self.assertEqual([line.get("generation_id") or line.get("leftover") for line in removed], [leftover.name, "g1", "g6"])
        self.assertEqual(result["freed_bytes"], expected)
        self.assertEqual(sum(line["freed_bytes"] for line in removed), expected)
        # The lake the releases linked into keeps every file, byte for byte.
        self.assertEqual(
            {path: (path.lstat().st_ino, path.read_bytes()) for path in raw.rglob("*") if path.is_file()}, lake_files
        )
        # A kept release still pins; a removed one is refused, never half read.
        self._pin(self.root / "experiments" / "arm_new", "g2")
        with self.assertRaisesRegex(RuntimeError, "missing or incomplete"):
            self._pin(self.root / "experiments" / "arm_late", "g1")
        again, _ = self.prune(apply=True)
        self.assertEqual((again["unreferenced"], again["freed_bytes"]), (0, 0))

    def test_apply_reads_the_consumers_under_both_locks_and_refuses_a_held_updater_lock(self) -> None:
        held = []
        consumers = lake.release_consumers

        def probing(repo_root: Path, raw_dir: Path) -> dict[str, list[str]]:
            for path in (self.lock, self.releases / ".registry.lock"):
                with path.open("rb") as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        held.append(path.name)
            return consumers(repo_root, raw_dir)

        with self.lock.open("rb") as other:
            fcntl.flock(other, fcntl.LOCK_EX)
            with self.assertRaisesRegex(RuntimeError, "lock is held"):
                self.prune(apply=True)
        self.assertEqual(self.published(), set(GENERATIONS))
        with patch.object(lake, "release_consumers", probing):
            self.prune(apply=True)
        self.assertEqual(held, ["tushare_update.lock", ".registry.lock"])
        self.assertEqual(self.published(), KEPT)

    def test_the_weekly_job_prunes_under_the_lock_its_runner_holds(self) -> None:
        job = json.loads((REPO_ROOT / "configs/tushare_update_schedule.json").read_text(encoding="utf-8"))["jobs"][
            "cn_weekly_release_prune"
        ]
        context = cron_update.RunContext(
            config={}, repo_root=Path("."), python="python", job_name="cn_weekly_release_prune", job=job,
            start_date="", end_date="", timezone_name="Asia/Shanghai",
        )
        self.assertEqual(
            cron_update.build_job_commands(context),
            [["python", "scripts/data/tushare_lake.py", "prune-releases", "--apply"]],
        )
        self.assertNotIn(job["operation"], cron_update.MUTATING_OPERATIONS)
        # A runner child inherits the held lock; taking it again (no wait is
        # configured) would fail.
        fd = os.open(self.lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with patch.dict(os.environ, {UPDATER_LOCK_FD_ENV: str(fd)}):
                result, _ = self.prune(apply=True)
        finally:
            os.close(fd)
        self.assertEqual(result["unreferenced"], 2)
        self.assertEqual(self.published(), KEPT)

    def test_a_pin_that_retention_could_race_waits_and_never_records_a_retired_release(self) -> None:
        outcomes: dict[str, BaseException | None] = {}

        def pin(name: str, generation: str | None) -> threading.Thread:
            def run() -> None:
                try:
                    self._pin(self.root / "experiments" / name, generation)
                    outcomes[name] = None
                except RuntimeError as exc:
                    outcomes[name] = exc

            thread = threading.Thread(target=run)
            thread.start()
            return thread

        # Retention holds the updater lock, so a pin of the newest generation
        # falls back to an existing release, and the registry lock.
        with self.lock.open("rb") as updater, (self.releases / ".registry.lock").open("a+b") as registry:
            fcntl.flock(updater, fcntl.LOCK_EX)
            fcntl.flock(registry, fcntl.LOCK_EX)
            threads = [pin("arm_named", "g6"), pin("arm_fallback", None)]
            for thread in threads:
                thread.join(0.5)
                self.assertTrue(thread.is_alive())
            # Retired while both wait.
            (self.releases / "g6").rename(self.releases / ".g6.0123456789.tmp")
            fcntl.flock(registry, fcntl.LOCK_UN)
            for thread in threads:
                thread.join(30)
        self.assertRegex(str(outcomes["arm_named"]), "missing or incomplete")
        self.assertFalse((self.root / "experiments" / "arm_named" / "research_release").exists())
        self.assertIsNone(outcomes["arm_fallback"])
        manifest = self.root / "experiments" / "arm_fallback" / "research_release" / "manifest.json"
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["generation_id"], "g7")


def _blocks_freed_by_removing(trees: list[Path]) -> int:
    """Blocks of the directories, and of the files every link of which is inside ``trees``."""
    freed = 0
    files: dict[int, list[int]] = {}
    for tree in trees:
        for path in (tree, *tree.rglob("*")):
            info = path.lstat()
            if path.is_dir():
                freed += info.st_blocks * 512
            else:
                files.setdefault(info.st_ino, [info.st_nlink, info.st_blocks, 0])[2] += 1
    return freed + sum(blocks * 512 for links, blocks, inside in files.values() if inside == links)


if __name__ == "__main__":
    unittest.main()
