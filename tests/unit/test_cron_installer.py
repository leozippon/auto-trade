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

    def test_backup_permissions_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cron_backups" / "crontab.bak"
            installer.write_private_backup(path, "MAILTO=private@example.com\n")
            self.assertEqual(S_IMODE(path.parent.stat().st_mode), 0o700)
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
