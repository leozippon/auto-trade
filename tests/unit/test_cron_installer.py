"""Safety contract for the managed crontab installer."""

import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path
from stat import S_IMODE

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "ops" / "cron" / "install_cron.py"
SPEC = importlib.util.spec_from_file_location("install_cron", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
TEMPLATE, BEGIN, END = installer.BLOCKS["tushare"]
HELPER = installer.REPO_ROOT / "ops" / "cron" / "locked.sh"
# A job line run through the helper: its lock, limit, label, command and log.
LOCKED = re.compile(r" && ops/cron/locked\.sh (\S+) (\S+) \"([^\"]+)\" (.+) >> (\S+) 2>&1$")
DATED = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d"


def job_lines(block: str) -> list[str]:
    template = installer.BLOCKS[block][0]
    return [line for line in template.read_text(encoding="utf-8").splitlines() if line[:1] == "*" or line[:1].isdigit()]


def stand_in_repository(root: Path) -> None:
    """The helper where a job line run with REPO_ROOT=root finds it, waiting 1 s instead of GRACE before it kills."""
    text = HELPER.read_text(encoding="utf-8")
    if text.count("\nGRACE=10\n") != 1:
        raise AssertionError("ops/cron/locked.sh no longer sets GRACE=10 once")
    helper = root / "ops" / "cron" / "locked.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text(text.replace("\nGRACE=10\n", "\nGRACE=1\n"), encoding="utf-8")
    helper.chmod(0o755)


def running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class CronInstallerTest(unittest.TestCase):
    def test_replacement_preserves_unrelated_lines_and_has_one_marker_pair(self) -> None:
        current = "MAILTO=owner@example.com\n0 1 * * * unrelated\n"
        managed = f"{BEGIN}\n5 2 * * * managed\n{END}\n"
        updated = installer.replace_managed_block(current, managed, begin=BEGIN, end=END)
        self.assertIn("0 1 * * * unrelated", updated)
        self.assertEqual(updated.count(BEGIN), 1)
        self.assertEqual(updated.count(END), 1)

    def test_replacement_rejects_unpaired_or_duplicate_markers(self) -> None:
        managed = f"{BEGIN}\nmanaged\n{END}\n"
        invalid_tables = (
            f"{BEGIN}\nunterminated\n",
            f"{END}\n",
            f"{BEGIN}\na\n{END}\n{BEGIN}\nb\n{END}\n",
            f"{END}\nreversed\n{BEGIN}\n",
        )
        for current in invalid_tables:
            with self.subTest(current=current), self.assertRaisesRegex(RuntimeError, "invalid managed cron markers"):
                installer.replace_managed_block(current, managed, begin=BEGIN, end=END)

    def test_post_install_verification_requires_exact_content(self) -> None:
        expected = f"unrelated\n{BEGIN}\nmanaged\n{END}\n"
        installer.verify_installed_crontab(expected, expected.rstrip("\n"), begin=BEGIN, end=END)
        with self.assertRaisesRegex(RuntimeError, "differs from requested content"):
            installer.verify_installed_crontab(expected, expected.replace("unrelated", "changed"), begin=BEGIN, end=END)

    def test_the_weekly_sweep_chains_the_jobs_that_must_follow_it(self) -> None:
        """The audit and the PIT rebuild that follow the weekly fundamental
        sweep are sequenced by the sweep itself, not by a clock. On 2026-09-19
        the sweep held the updater lock for four hours and all three
        follow-ups, then scheduled 60-90 minutes after it, died on lock
        timeout: a generation was committed with sweep-mutated fundamentals,
        no rebuilt PIT events and no audit."""
        follow_ups = [
            "cn_weekly_deep_audit",
            "cn_nightly_full_audit",
            "cn_nightly_pit_event_build",
            "cn_weekly_release_prune",
        ]
        lines = [
            line
            for line in TEMPLATE.read_text(encoding="utf-8").splitlines()
            if line[:1].isdigit()
        ]
        saturday = [line for line in lines if line.split()[4] == "6"]
        chained = [line for line in saturday if "--job cn_weekly_fundamental_deep" in line]
        self.assertEqual(len(chained), 1)
        fields = chained[0].split()
        self.assertEqual(
            [fields[index + 1] for index, field in enumerate(fields) if field == "--then"],
            follow_ups,
        )
        # Nothing follows the sweep on a clock of its own any more, and the
        # runner forces a follow-up because it follows a mutation.
        for line in saturday:
            if line is not chained[0]:
                for job in follow_ups:
                    self.assertNotIn(job, line)
            self.assertNotIn("--force-run", line)

    def test_the_paper_block_installs_beside_the_tushare_block(self) -> None:
        template, begin, end = installer.BLOCKS["paper"]
        tushare = installer.build_managed_block(TEMPLATE, BEGIN, END)
        paper = installer.build_managed_block(template, begin, end)
        current = f"0 1 * * * unrelated\n\n{tushare}"
        once = installer.replace_managed_block(current, paper, begin=begin, end=end)
        twice = installer.replace_managed_block(once, paper, begin=begin, end=end)
        self.assertEqual(once, twice)
        self.assertIn(tushare, twice)
        self.assertEqual(twice.count(begin), 1)
        jobs = job_lines("paper")
        self.assertEqual([job.split()[:2] for job in jobs], [["40", "5"], ["40", "6"], ["55", "7"]])
        for job in jobs:
            # Weekdays only, never overlapping (one lock), logged under logs/paper/.
            self.assertEqual(job.split()[4], "1-5")
            # Every book, in the run's own order (the hand-traded ones first):
            # the job names no book, so a deleted one cannot break the morning.
            self.assertEqual(LOCKED.search(job).group(4), "$QUANT_PYTHON scripts/paper/run_paper.py run")

    def test_the_research_fill_block_installs_beside_the_paper_block(self) -> None:
        template, begin, end = installer.BLOCKS["research"]
        paper = installer.build_managed_block(*installer.BLOCKS["paper"])
        research = installer.build_managed_block(template, begin, end)
        current = f"0 1 * * * unrelated\n\n{paper}"
        once = installer.replace_managed_block(current, research, begin=begin, end=end)
        twice = installer.replace_managed_block(once, research, begin=begin, end=end)
        self.assertEqual(once, twice)
        self.assertIn(paper, twice)
        self.assertEqual(twice.count(begin), 1)
        lines = template.read_text(encoding="utf-8").splitlines()
        jobs = job_lines("research")
        self.assertEqual(len(jobs), 1)
        # Every ten minutes, every day (its lock and log: the locked-jobs test).
        self.assertEqual(jobs[0].split()[:5], ["*/10", "*", "*", "*", "*"])
        # The queue it fills has to be open round files this repository holds
        # (none while nothing is queued): a mistyped or closed one would fail
        # every ten minutes instead of once.
        rounds = next(line.split("=", 1)[1] for line in lines if line.startswith("RESEARCH_ROUNDS="))
        self.assertIn("$RESEARCH_ROUNDS", jobs[0])
        for round_file in rounds.split():
            self.assertTrue((installer.REPO_ROOT / round_file).is_file(), round_file)
            module = importlib.import_module("scripts.experiments." + Path(round_file).stem)
            self.assertFalse(module.ROUND.closed, round_file)

    def test_the_research_fill_runs_every_round_in_order_and_fails_on_any_refusal(self) -> None:
        """The job line itself, run by bash with a stand-in interpreter: each
        round file's fill runs in the listed order under the one lock and into
        the one log, a round whose fill fails does not stop the ones after it,
        and the run's exit status still reports the failure."""
        import os
        import subprocess

        template, _, _ = installer.BLOCKS["research"]
        job = next(
            line for line in template.read_text(encoding="utf-8").splitlines()
            if "--fill" in line and not line.startswith("#")
        )
        command = job.split(None, 5)[5]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stand_in_repository(root)
            calls = root / "calls"
            python = root / "python"
            # Records its arguments and fails the round named `refused`.
            python.write_text(
                f'#!/bin/bash\necho "$@" >> {calls}\necho "fill $1"\n'
                '[ "$1" != refused ]\n',
                encoding="utf-8",
            )
            python.chmod(0o755)

            def run(rounds: str) -> subprocess.CompletedProcess[str]:
                calls.unlink(missing_ok=True)
                env = {"PATH": os.environ["PATH"], "REPO_ROOT": tmp, "QUANT_PYTHON": str(python), "RESEARCH_ROUNDS": rounds}
                return subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, timeout=60)

            # An empty queue is the steady state between rounds: nothing runs.
            self.assertEqual(run("").returncode, 0)
            self.assertFalse(calls.exists())
            self.assertEqual(run("first second").returncode, 0)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["first 38888 --fill", "second 38888 --fill"],
            )
            self.assertEqual(run("first refused last").returncode, 1)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["first 38888 --fill", "refused 38888 --fill", "last 38888 --fill"],
            )
            log = (root / "logs/research/cron.log").read_text(encoding="utf-8")
            self.assertTrue(log.endswith("fill first\nfill refused\nfill last\n"), log)

    def test_the_zgca_sync_block_runs_one_pass_over_its_rounds(self) -> None:
        """The block installs beside the others without touching them, its one job
        runs every 30 minutes (its lock and log: the locked-jobs test), every round it
        lists is a round file here (closed ones too: their experiments may still be
        on the instance), and the job line hands all of them to one sync-rounds pass,
        whose output and exit status reach the log and cron as they are."""
        import subprocess

        template, begin, end = installer.BLOCKS["zgca"]
        block = installer.build_managed_block(template, begin, end)
        others = [installer.build_managed_block(*installer.BLOCKS[name]) for name in ("paper", "housekeeping", "research")]
        once = installer.replace_managed_block("0 1 * * * unrelated\n\n" + "\n".join(others), block, begin=begin, end=end)
        self.assertEqual(installer.replace_managed_block(once, block, begin=begin, end=end), once)
        for other in others:
            self.assertIn(other, once)
        lines = template.read_text(encoding="utf-8").splitlines()
        jobs = job_lines("zgca")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].split()[:5], ["*/30", "*", "*", "*", "*"])
        rounds = next(line.split("=", 1)[1] for line in lines if line.startswith("ZGCA_ROUNDS="))
        self.assertIn("$ZGCA_ROUNDS", jobs[0])
        for round_file in rounds.split():
            self.assertTrue((installer.REPO_ROOT / round_file).is_file(), round_file)
            self.assertTrue(importlib.import_module("scripts.experiments." + Path(round_file).stem).ROUND.experiments, round_file)
        command = jobs[0].split(None, 5)[5]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stand_in_repository(root)
            python = root / "python"
            python.write_text('#!/bin/bash\necho "pass: $*"\nexit 1\n', encoding="utf-8")
            python.chmod(0o755)
            for listed, expected in (("", "pass: ops/zgca/zgca.py sync-rounds"), ("first second", "pass: ops/zgca/zgca.py sync-rounds first second")):
                with self.subTest(rounds=listed):
                    env = {"PATH": os.environ["PATH"], "REPO_ROOT": tmp, "QUANT_PYTHON": str(python), "ZGCA_ROUNDS": listed}
                    run = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(run.returncode, 1)
                    self.assertEqual((root / "logs/zgca/sync_cron.log").read_text(encoding="utf-8").splitlines()[-1], expected)

    @pytest.mark.slow  # runs each job line for real in bash, through a 1 s limit
    def test_every_locked_job_skips_and_stops_with_one_dated_line_in_its_own_log(self) -> None:
        """Each job line run through ops/cron/locked.sh, run for real by bash in a stand-in
        repository with a stand-in interpreter: the job's output and exit status reach its
        log and cron as they are, with no line of the helper's; a run that finds the lock
        held runs nothing, exits 0 and writes one dated line saying so (on 2026-11-21 a hung
        sync pass held its lock for 4 h 27 min while every later run was skipped without a
        word); a run reaching its limit (1 s here) is ended with every process it started,
        one that ignores TERM included, so the lock is free again, and writes its own dated
        line and exit status. Each block keeps its lock, limit and log."""
        expected = {
            "paper": (".runtime/paper/cron.lock", "6h", "paper run", "logs/paper/cron.log"),
            "research": (".runtime/research/cron.lock", "12h", "research fill", "logs/research/cron.log"),
            "zgca": (".runtime/zgca/sync_cron.lock", "4h", "zgca sync", "logs/zgca/sync_cron.log"),
        }
        for block, (lock, limit, label, log) in expected.items():
            # The Paper block's three lines differ in their time alone.
            commands = {job.split(None, 5)[5] for job in job_lines(block)}
            self.assertEqual(len(commands), 1, block)
            (command,) = commands
            # cron reads a % in a job line as a newline.
            self.assertNotIn("%", command)
            locked = LOCKED.search(command)
            self.assertIsNotNone(locked, command)
            self.assertEqual((locked.group(1), locked.group(2), locked.group(3), locked.group(5)), (lock, limit, label, log))
            with self.subTest(block=block):
                self._exercise_locked_job(command, lock, limit, label, log)

    def _exercise_locked_job(self, command: str, lock: str, limit: str, label: str, log: str) -> None:
        import fcntl
        import subprocess
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stand_in_repository(root)
            python, pids = root / "python", root / "pids"
            env = {"PATH": os.environ["PATH"], "REPO_ROOT": tmp, "QUANT_PYTHON": str(python), "RESEARCH_ROUNDS": "first", "ZGCA_ROUNDS": "first"}

            def run(line: str = command) -> subprocess.CompletedProcess[str]:
                return subprocess.run(["bash", "-c", line], env=env, capture_output=True, text=True, timeout=60)

            def logged() -> list[str]:
                return (root / log).read_text(encoding="utf-8").splitlines()

            python.write_text('#!/bin/bash\necho "job: $*"\nexit 3\n', encoding="utf-8")
            python.chmod(0o755)
            ran = run()
            self.assertEqual(ran.returncode, 1 if label == "research fill" else 3)  # the fill loop reports any failure as 1
            self.assertEqual(len(logged()), 1)
            self.assertTrue(logged()[0].startswith("job: "), logged())
            with (root / lock).open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                skipped = run()
            self.assertEqual(skipped.returncode, 0)
            self.assertEqual(len(logged()), 2)
            self.assertRegex(logged()[-1], rf"^{DATED} {label} skipped: the previous run still holds {re.escape(lock)}$")
            # A job that hangs with two grandchildren, as rsync hangs under dev, one of them deaf to TERM.
            python.write_text(f'#!/bin/bash\necho "job: $*"\nsleep 300 & echo $! >> {pids}\n'
                              f'(trap "" TERM; exec sleep 300) & echo $! >> {pids}\nwait\n', encoding="utf-8")
            self.assertEqual(command.count(f" {limit} "), 1)
            stopped = run(command.replace(f" {limit} ", " 1s "))
            self.assertEqual(stopped.returncode, 124)
            self.assertEqual(logged()[-2], logged()[0])
            self.assertRegex(logged()[-1], rf"^{DATED} {label} stopped: the run reached its time limit$")
            left = [int(pid) for pid in pids.read_text(encoding="utf-8").split()]
            self.assertEqual(len(left), 2)
            deadline = time.monotonic() + 10
            while any(map(running, left)) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(any(map(running, left)))
            with (root / lock).open("a") as free:
                fcntl.flock(free, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_the_housekeeping_job_compresses_only_old_conversation_logs(self) -> None:
        """The weekly job line itself, run by bash in a stand-in repository: a
        conversation log older than 14 days becomes a .zst that reads back byte
        for byte, a younger log and any other file stay as they are, and a log
        zstd cannot compress is kept and fails the run."""
        import subprocess
        import time

        template, begin, end = installer.BLOCKS["housekeeping"]
        block = installer.build_managed_block(template, begin, end)
        paper = installer.build_managed_block(*installer.BLOCKS["paper"])
        once = installer.replace_managed_block(f"0 1 * * * unrelated\n\n{paper}", block, begin=begin, end=end)
        self.assertEqual(installer.replace_managed_block(once, block, begin=begin, end=end), once)
        self.assertIn(paper, once)
        jobs = [line for line in template.read_text(encoding="utf-8").splitlines() if line[:1].isdigit()]
        self.assertEqual(len(jobs), 1)
        # Once a week, on Sunday, into its own log.
        self.assertEqual(jobs[0].split()[2:5], ["*", "*", "0"])
        self.assertTrue(jobs[0].endswith(" >> logs/housekeeping/cron.log 2>&1"), jobs[0])
        command = jobs[0].split(None, 5)[5]
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "data/llm_conversations/vllm/model"
            logs.mkdir(parents=True)
            content = {}
            for name, age_days in (("old.jsonl", 15), ("young.jsonl", 13), ("old.txt", 15), ("clash.jsonl", 15)):
                path = logs / name
                path.write_text(f'{{"file": "{name}"}}\n' * 1000, encoding="utf-8")
                stamp = time.time() - age_days * 86400
                os.utime(path, (stamp, stamp))
                content[name] = path.read_bytes()
            (logs / "clash.jsonl.zst").write_bytes(b"taken")

            def run() -> int:
                env = {"PATH": os.environ["PATH"], "REPO_ROOT": tmp}
                return subprocess.run(["bash", "-c", command], env=env, capture_output=True, timeout=120).returncode

            self.assertNotEqual(run(), 0)
            self.assertFalse((logs / "old.jsonl").exists())
            restored = subprocess.run(["zstd", "-dc", str(logs / "old.jsonl.zst")], capture_output=True, check=True)
            self.assertEqual(restored.stdout, content["old.jsonl"])
            for name in ("young.jsonl", "old.txt", "clash.jsonl"):
                self.assertEqual((logs / name).read_bytes(), content[name], name)
            self.assertEqual((logs / "clash.jsonl.zst").read_bytes(), b"taken")
            log = (Path(tmp) / "logs/housekeeping/cron.log").read_text(encoding="utf-8")
            self.assertIn("clash.jsonl.zst already exists", log)
            # Once the name is free the next run compresses it and succeeds.
            (logs / "clash.jsonl.zst").unlink()
            self.assertEqual(run(), 0)
            self.assertEqual(sorted(path.name for path in logs.iterdir()), ["clash.jsonl.zst", "old.jsonl.zst", "old.txt", "young.jsonl"])

    def test_removing_a_block_keeps_every_other_entry_and_is_idempotent(self) -> None:
        """Pausing one managed job -- the research fill, while a stop window
        rebuilds the sandbox image -- must not touch the Paper or TuShare
        jobs, and must leave nothing behind for the next install to duplicate."""

        _, begin, end = installer.BLOCKS["research"]
        paper = installer.build_managed_block(*installer.BLOCKS["paper"])
        research = installer.build_managed_block(*installer.BLOCKS["research"])
        installed = installer.replace_managed_block(
            f"0 1 * * * unrelated\n\n{paper}", research, begin=begin, end=end
        )
        removed = installer.strip_managed_block(installed, begin=begin, end=end)
        self.assertNotIn(begin, removed)
        self.assertNotIn(end, removed)
        self.assertIn("0 1 * * * unrelated", removed)
        self.assertIn(paper, removed)
        # Removing again changes nothing, and reinstalling restores exactly one pair.
        self.assertEqual(installer.strip_managed_block(removed, begin=begin, end=end), removed)
        restored = installer.replace_managed_block(removed, research, begin=begin, end=end)
        self.assertEqual(restored.count(begin), 1)
        self.assertIn(paper, restored)
        # The removal's own verification: markers gone is the expected state.
        installer.verify_installed_crontab(removed, removed, begin=begin, end=end, required=False)
        with self.assertRaisesRegex(RuntimeError, "missing managed cron markers"):
            installer.verify_installed_crontab(removed, removed, begin=begin, end=end)

    def test_backup_permissions_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cron_backups" / "crontab.bak"
            installer.write_private_backup(path, "MAILTO=private@example.com\n")
            self.assertEqual(S_IMODE(path.parent.stat().st_mode), 0o700)
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)

    def test_only_the_newest_backups_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            unrelated = directory / "notes.txt"
            unrelated.write_text("kept\n", encoding="utf-8")
            # Oldest first; alternating blocks make name order differ from age order.
            blocks = ("research", "paper") * 3 + ("research",)
            old = [
                directory / f"crontab-{block}-2026100{day}-000000-1.bak"
                for day, block in enumerate(blocks, start=1)
            ]
            for second, path in enumerate(old, start=1):
                path.write_text("old\n", encoding="utf-8")
                os.utime(path, (second, second))
            newest = directory / "crontab-tushare-20261008-000000-2.bak"
            installer.write_private_backup(newest, "current\n")
            self.assertEqual(
                {path.name for path in directory.glob("crontab-*.bak")},
                {newest.name, *(path.name for path in old[1 - installer.BACKUP_KEEP :])},
            )
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
