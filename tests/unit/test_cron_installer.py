"""Safety contract for the managed TuShare crontab installer."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from stat import S_IMODE


SCRIPT = Path(__file__).resolve().parents[2] / "ops" / "cron" / "install_tushare_cron.py"
SPEC = importlib.util.spec_from_file_location("install_tushare_cron", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class CronInstallerTest(unittest.TestCase):
    def test_replacement_preserves_unrelated_lines_and_has_one_marker_pair(self) -> None:
        current = "MAILTO=owner@example.com\n0 1 * * * unrelated\n"
        managed = f"{installer.BEGIN}\n5 2 * * * managed\n{installer.END}\n"
        updated = installer.replace_managed_block(current, managed)
        self.assertIn("0 1 * * * unrelated", updated)
        self.assertEqual(updated.count(installer.BEGIN), 1)
        self.assertEqual(updated.count(installer.END), 1)

    def test_replacement_rejects_unpaired_or_duplicate_markers(self) -> None:
        managed = f"{installer.BEGIN}\nmanaged\n{installer.END}\n"
        invalid_tables = (
            f"{installer.BEGIN}\nunterminated\n",
            f"{installer.END}\n",
            f"{installer.BEGIN}\na\n{installer.END}\n{installer.BEGIN}\nb\n{installer.END}\n",
            f"{installer.END}\nreversed\n{installer.BEGIN}\n",
        )
        for current in invalid_tables:
            with self.subTest(current=current), self.assertRaisesRegex(RuntimeError, "invalid managed cron markers"):
                installer.replace_managed_block(current, managed)

    def test_post_install_verification_requires_exact_content(self) -> None:
        expected = f"unrelated\n{installer.BEGIN}\nmanaged\n{installer.END}\n"
        installer.verify_installed_crontab(expected, expected.rstrip("\n"))
        with self.assertRaisesRegex(RuntimeError, "differs from requested content"):
            installer.verify_installed_crontab(expected, expected.replace("unrelated", "changed"))

    def test_the_paper_block_installs_beside_the_tushare_block(self) -> None:
        template, begin, end = installer.BLOCKS["paper"]
        tushare = installer.build_managed_block()
        paper = installer.build_managed_block(template, begin, end)
        current = f"0 1 * * * unrelated\n\n{tushare}"
        once = installer.replace_managed_block(current, paper, begin=begin, end=end)
        twice = installer.replace_managed_block(once, paper, begin=begin, end=end)
        self.assertEqual(once, twice)
        self.assertIn(tushare, twice)
        self.assertEqual(twice.count(begin), 1)
        jobs = [line for line in template.read_text(encoding="utf-8").splitlines() if line[:1].isdigit()]
        self.assertTrue(jobs)
        for job in jobs:
            # Weekdays only, never overlapping, logged under logs/paper/.
            self.assertEqual(job.split()[4], "1-5")
            self.assertIn("flock -n .runtime/paper/cron.lock", job)
            self.assertIn("scripts/paper/run_paper.py run >> logs/paper/cron.log 2>&1", job)

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
        jobs = [line for line in lines if "--fill" in line and not line.startswith("#")]
        self.assertEqual(len(jobs), 1)
        # Every ten minutes, every day, under its own lock and its own log.
        self.assertEqual(jobs[0].split()[:5], ["*/10", "*", "*", "*", "*"])
        self.assertIn("flock -n .runtime/research/cron.lock", jobs[0])
        self.assertIn("--fill >> logs/research/cron.log 2>&1", jobs[0])
        # The queue it fills has to be a round file this repository holds:
        # a mistyped one would fail every ten minutes instead of once.
        round_file = next(line.split("=", 1)[1] for line in lines if line.startswith("ROUND="))
        self.assertIn("$ROUND", jobs[0])
        self.assertTrue((installer.REPO_ROOT / round_file).is_file(), round_file)

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


if __name__ == "__main__":
    unittest.main()
