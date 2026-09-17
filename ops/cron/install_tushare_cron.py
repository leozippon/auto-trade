#!/usr/bin/env python3
"""Install or refresh one managed ADM-Cube cron block (see BLOCKS)."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path


BEGIN = "# BEGIN ADM-Cube TuShare update"
END = "# END ADM-Cube TuShare update"
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "ops/cron/tushare_update.cron"
# Each block owns its own marker pair, so installing one never touches another.
BLOCKS = {
    "tushare": (TEMPLATE, BEGIN, END),
    "paper": (REPO_ROOT / "ops/cron/paper.cron", "# BEGIN ADM-Cube Paper book", "# END ADM-Cube Paper book"),
    "research": (
        REPO_ROOT / "ops/cron/research_fill.cron",
        "# BEGIN ADM-Cube research fill",
        "# END ADM-Cube research fill",
    ),
}
BACKUP_DIR = REPO_ROOT / ".runtime" / "crontab"
CRONTAB_LOCK = REPO_ROOT / ".runtime" / "crontab.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append or refresh one ADM-Cube cron block without replacing other crontab entries.")
    parser.add_argument("--block", choices=sorted(BLOCKS), default="tushare", help="Managed block to install (default: tushare).")
    parser.add_argument("--remove", action="store_true", help="Remove the managed block instead of installing it; every other crontab entry is kept.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resulting crontab without installing it.")
    return parser.parse_args()


def current_crontab() -> str:
    """The user's crontab; fail fast on anything but a clean read or a genuine
    'no crontab for user'. A permission/IO error treated as an empty table
    would silently wipe every unrelated job on install."""
    process = subprocess.run(["crontab", "-l"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode == 0:
        return process.stdout
    if "no crontab for" in process.stderr.lower():
        return ""
    raise RuntimeError(f"crontab -l failed (rc={process.returncode}): {process.stderr.strip()}")


def build_managed_block(template: Path = TEMPLATE, begin: str = BEGIN, end: str = END) -> str:
    body = template.read_text(encoding="utf-8").strip()
    return f"{begin}\n{body}\n{end}\n"


def validate_managed_markers(
    text: str, *, source: str, required: bool = False, begin: str = BEGIN, end: str = END
) -> None:
    """Reject unmatched, duplicated, reversed, or nested managed markers.

    ``required`` also rejects a *complete absence* of markers. Only the current
    crontab may legitimately have none (first install). A generated block or an
    installed table without them is unmanageable: nothing can find, replace or
    remove it again, so every later install silently appends another copy of
    every job, and post-install verification would still report success.
    """
    markers = [
        line.strip()
        for line in text.splitlines()
        if line.strip() in {begin, end}
    ]
    if not markers:
        if required:
            raise RuntimeError(
                f"missing managed cron markers in {source}: expected one paired "
                f"{begin!r}/{end!r}"
            )
        return
    if markers != [begin, end]:
        raise RuntimeError(
            f"invalid managed cron markers in {source}: expected one paired "
            f"{begin!r}/{end!r}, found {markers!r}"
        )


def strip_managed_block(current: str, *, begin: str = BEGIN, end: str = END) -> str:
    """The crontab with this managed block removed and nothing else touched.

    Pausing a managed job is the same transaction as installing one -- the
    lock, the backup and the post-write verification all still apply -- so it
    goes through the installer rather than a hand-edited `crontab -`."""
    validate_managed_markers(current, source="current crontab", begin=begin, end=end)
    kept: list[str] = []
    skipping = False
    for line in current.splitlines():
        stripped = line.strip()
        if stripped == begin:
            skipping = True
            continue
        if stripped == end:
            skipping = False
            continue
        if not skipping:
            kept.append(line)
    result = "\n".join(kept).rstrip()
    return f"{result}\n" if result else ""


def replace_managed_block(current: str, managed: str, *, begin: str = BEGIN, end: str = END) -> str:
    validate_managed_markers(managed, source="generated block", required=True, begin=begin, end=end)
    kept = strip_managed_block(current, begin=begin, end=end).rstrip()
    if kept:
        kept += "\n\n"
    return kept + managed


def verify_installed_crontab(
    expected: str, installed: str, *, begin: str = BEGIN, end: str = END, required: bool = True
) -> None:
    """``required=False`` is the removal case, where the markers must be gone;
    a leftover half of the pair is still rejected."""
    validate_managed_markers(installed, source="installed crontab", required=required, begin=begin, end=end)
    expected_text = expected.rstrip("\n") + "\n"
    installed_text = installed.rstrip("\n") + "\n"
    if installed_text != expected_text:
        raise RuntimeError("post-install verification failed: installed crontab differs from requested content")


def write_private_backup(path: Path, content: str) -> None:
    """Store the full user crontab without exposing unrelated job secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def planned_crontab(current: str, managed: str, *, remove: bool, begin: str, end: str) -> str:
    if remove:
        return strip_managed_block(current, begin=begin, end=end)
    return replace_managed_block(current, managed, begin=begin, end=end)


def main() -> int:
    args = parse_args()
    template, begin, end = BLOCKS[args.block]
    managed = ""
    if not args.remove:
        if not template.exists():
            raise FileNotFoundError(f"cron template not found: {template}")
        managed = build_managed_block(template, begin, end)
    if args.dry_run:
        print(
            planned_crontab(
                current_crontab(), managed, remove=args.remove, begin=begin, end=end
            ),
            end="",
        )
        return 0
    # One flock spans the whole read-modify-write-verify transaction: without
    # it a second installer (or a second run) would silently erase the managed
    # block the earlier writer just added.
    lock_path = CRONTAB_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = current_crontab()
        updated = planned_crontab(current, managed, remove=args.remove, begin=begin, end=end)
        if current.strip():
            # Source prefix + pid: two installers (or two runs) in the same
            # second can never silently overwrite each other's backup.
            backup = BACKUP_DIR / f"crontab-{args.block}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.bak"
            write_private_backup(backup, current)
            print(f"backed up current crontab to {backup}")
        subprocess.run(["crontab", "-"], input=updated, text=True, check=True)
        installed = subprocess.run(["crontab", "-l"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        verify_installed_crontab(
            updated, installed.stdout, begin=begin, end=end, required=not args.remove
        )
    print(f"{'removed' if args.remove else 'installed'} ADM-Cube {args.block} cron block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
