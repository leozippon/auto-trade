#!/usr/bin/env python3
"""Report, and with --apply delete, the derived PIT caches of finished arms.

An arm's ``pit_views/`` (decision views, replay slots, as-of stash) is a cache
of its pinned release and snapshot configuration. Its only reader is the PIT
snapshot provider inside that arm's worker, and a worker whose ledger already
carries a verdict republishes the completion status and exits before any
snapshot is prepared; a missing tree is rebuilt on demand (re-linked from the
seed, the rest cold-built). So once an arm is finished for good its views can
go, together with the sandbox work root a worker left behind before it learned
to release it. A worker that writes an arm's terminal status releases both
itself, through the same ``release_pit_views``; this tool is for the arms that
ended before it did, and for arms stopped after their verdict.

Finished for good means: hitl status ``completed`` or ``stopped``, no live
worker, and a ledger verdict (``graduated``, ``discarded`` or
``no_deliverable``), which rules out a pending forward replay. Anything else is
kept, an arm whose status or ledger cannot be read is refused, and nothing
outside ``pit_views/`` and ``.runtime/sandboxes/<id>`` is touched.

The default only reports, per arm and in total, the bytes deletion would free:
an inode counts only when every hard link to it lies inside the deleted tree,
so the views an arm shares with its seed count nothing.

usage: prune_pit_views.py [--apply]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.sandbox import remove_sandbox_tree
from autotrade.pipelines.hitl_state import (
    HITL_DIR_NAME,
    STATUS_NAME,
    read_status,
    status_pid_alive,
)
from autotrade.pipelines.ledger import ExperimentLedger, experiment_verdict
from autotrade.pipelines.worker import pit_views_tree, release_pit_views
from autotrade.webui.manager import ManagerDeleteError, _derived_sandbox_tree

TERMINAL_STATES = ("completed", "stopped")
GIB = 1024**3


def kept_reason(directory: Path) -> str:
    """Why this arm's caches stay, or "" when it is finished for good.

    Raises OSError or ValueError when its status or ledger cannot be read.
    """

    status = read_status(directory / HITL_DIR_NAME / STATUS_NAME)
    if status_pid_alive(status):
        return f"live worker (pid {status.get('pid')})"
    state = str(status.get("state") or "created")
    if state not in TERMINAL_STATES:
        return f"state {state!r} is not terminal"
    records = ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl").read()
    if experiment_verdict(records) is None:
        return f"state {state!r} without a verdict: research or the forward replay is unfinished"
    return ""


def derived_trees(repo_root: Path, directory: Path) -> dict[str, Path]:
    """The arm's derived trees that exist: its PIT views and its sandbox root."""

    trees: dict[str, Path] = {}
    views = pit_views_tree(directory)
    if views is not None:
        trees["pit_views"] = views
    sandbox = _derived_sandbox_tree(repo_root, directory.name)
    if sandbox is not None:
        trees["sandbox"] = sandbox
    return trees


def freed_bytes(root: Path) -> int:
    """Disk bytes deleting ``root`` releases: its directories, and every other
    inode whose hard links all lie inside it."""

    links: dict[tuple[int, int], int] = {}
    inodes: dict[tuple[int, int], tuple[int, int]] = {}
    freed = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        freed += directory.lstat().st_blocks * 512
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                info = entry.stat(follow_symlinks=False)
                key = (info.st_dev, info.st_ino)
                links[key] = links.get(key, 0) + 1
                inodes[key] = (info.st_nlink, info.st_blocks * 512)
    return freed + sum(size for key, (nlink, size) in inodes.items() if links[key] == nlink)


def _delete(label: str, path: Path) -> None:
    if label == "pit_views":
        # The worker's own release of a finished arm's views.
        release_pit_views(path.parent)
    elif not remove_sandbox_tree(path):
        raise OSError(f"sandbox tree still exists after removal: {path}")


def prune(repo_root: Path, *, apply: bool) -> int:
    experiments = (repo_root / "experiments").resolve(strict=True)
    free_before = shutil.disk_usage(experiments).free
    totals = dict.fromkeys(("pit_views", "sandbox"), 0)
    pruned: list[str] = []
    kept: list[str] = []
    refused: list[str] = []
    for directory in sorted(experiments.iterdir()):
        if directory.name.startswith(".") or directory.is_symlink() or not directory.is_dir():
            continue
        name = directory.name
        try:
            trees = derived_trees(repo_root, directory)
            if not trees:
                continue
            reason = kept_reason(directory)
            if reason:
                kept.append(name)
                print(f"{name}: kept, {reason}")
                continue
            sizes = {label: freed_bytes(path) for label, path in trees.items()}
        except (OSError, ValueError, ManagerDeleteError) as exc:
            refused.append(name)
            print(f"{name}: refused, {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        detail = ", ".join(f"{label} {size / GIB:.2f} GiB" for label, size in sizes.items())
        if apply:
            for label, path in trees.items():
                try:
                    _delete(label, path)
                except (OSError, ValueError, ManagerDeleteError) as exc:
                    print(f"{name}: deleting {label} failed, stopping: {exc}", file=sys.stderr)
                    return 1
        print(f"{name}: {'freed' if apply else 'would free'} {sum(sizes.values()) / GIB:.2f} GiB ({detail})")
        for label, size in sizes.items():
            totals[label] += size
        pruned.append(name)
    total = sum(totals.values())
    print(
        f"total: {len(pruned)} finished arms, {'freed' if apply else 'would free'} "
        f"{total / GIB:.1f} GiB (pit_views {totals['pit_views'] / GIB:.1f} GiB, "
        f"sandboxes {totals['sandbox'] / GIB:.1f} GiB); {len(kept)} kept, "
        f"{len(refused)} refused; filesystem free {free_before / GIB:.1f} GiB"
        + (
            f" -> {shutil.disk_usage(experiments).free / GIB:.1f} GiB"
            if apply
            else "; dry run, pass --apply to delete"
        )
    )
    return 1 if refused else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete what the report lists; without it nothing is changed",
    )
    args = parser.parse_args(argv)
    return prune(REPO_ROOT, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
