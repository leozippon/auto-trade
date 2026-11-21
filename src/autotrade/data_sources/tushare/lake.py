"""Operator commands over the versioned raw lake (``scripts/data/tushare_lake.py``).

Every command that changes the lake takes the updater lock, marks the
generation ``updating`` and commits a new one when it is done, exactly like a
scheduled job, so releases and Paper pins never see a half-applied command.

* ``revert DATASET PARTITION VERSION`` makes a superseded version visible
  again; ``promote DATASET PARTITION VERSION`` does the same for a staged one
  and closes its issue. The partition replaced is kept as a superseded version.
* ``repull DATASET`` re-pulls the whole history of a month-partitioned event
  table through the versioned write path (the pending re-pull of an
  ``on_restatement`` dataset runs the same command unattended).
* ``rebuild-observations DATASET PARTITION`` rebuilds a partition's
  observation table from its version history.
* ``prune-releases [--dry-run | --apply]`` removes the research releases no
  consumer reads any more, oldest first. Who reads one is a census of the
  seeds, experiments and Paper accounts (``paper.release_census``) that the
  script passes in (``main``'s ``release_consumers``), so this package
  imports neither the experiments' state nor Paper's. The consumers are read
  under the release registry lock and the updater lock, so a release a
  consumer records is never removed (``research_release``); a removal prints
  the bytes it freed, those of the files whose last link it removed. Nothing
  in the lake changes, so no generation is committed. The default is a dry
  run; the weekly chain runs ``--apply`` last (``cn_weekly_release_prune``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from autotrade.common.time import utc_now
from autotrade.environment.data.pit import complete_interrupted_commits
from autotrade.environment.data.research_release import (
    remove_release_tree,
    retire_research_releases,
)

from .common import REVISION_EVENTS_PATH, resolve_revision_ledger
from .cron_update import (
    DEFAULT_CONFIG,
    DEFAULT_LOCK_WAIT_SECONDS,
    RUN_LOG_ROOT,
    RunContext,
    history_repull_policy,
    load_config,
    run_history_repull,
)
from .generation import (
    begin_raw_generation_update,
    finish_raw_generation_update,
    issue_key,
    record_vintages,
)
from .lock import acquire_lock, inherited_updater_lock_fd
from .versions import (
    STAGED,
    SUPERSEDED,
    rebuild_observations,
    restore,
)


def _lake(config_path: str) -> tuple[dict, Path, Path]:
    config = load_config(Path(config_path))
    repo_root = Path(config.get("repo_root", ".")).resolve()
    os.chdir(repo_root)
    raw_dir = repo_root / config.get("default_raw_dir", "data/raw")
    ledger_path = config.get("revision_monitor", {}).get("ledger_path") or REVISION_EVENTS_PATH
    ledger = resolve_revision_ledger(raw_dir, ledger_path, repo_root=repo_root)
    return config, raw_dir, ledger


def _partition(raw_dir: Path, dataset: str, partition: str) -> Path:
    path = (raw_dir / dataset / f"{partition}.parquet").resolve()
    if raw_dir.resolve() not in path.parents or ".." in Path(partition).parts:
        raise ValueError(f"partition outside the lake: {dataset}/{partition}")
    return path


@contextmanager
def locked_generation(config: dict, raw_dir: Path, job: str) -> Iterator[dict]:
    """Hold the updater lock around one operator change. The yielded state's
    ``datasets`` and ``issues`` are what the new record carries; ``commit``
    says whether the lake changed (set it before changing anything)."""
    # _lake works from the repository root.
    lock = acquire_lock(Path.cwd(), int(config.get("default_lock_wait_seconds", DEFAULT_LOCK_WAIT_SECONDS)))
    try:
        complete_interrupted_commits(raw_dir)
        transaction = {"job": job, "start_date": "", "end_date": ""}
        previous = begin_raw_generation_update(raw_dir, transaction)
        state = {
            "previous": previous,
            "lock_fd": lock.fd,
            "datasets": {name: dict(entry) for name, entry in (previous.get("datasets") or {}).items()},
            "issues": [dict(issue) for issue in previous.get("issues") or []],
            "commit": False,
        }
        try:
            yield state
        finally:
            finish_raw_generation_update(
                raw_dir, previous, transaction, commit=state["commit"], datasets=state["datasets"], issues=state["issues"],
            )
    finally:
        lock.release()


def restore_partition(config_path: str, dataset: str, partition: str, version: str, *, kind: str) -> dict:
    config, raw_dir, ledger = _lake(config_path)
    if not version.endswith(f"-{kind}"):
        raise ValueError(f"{version!r} is not a {kind} version")
    path = _partition(raw_dir, dataset, partition)
    with locked_generation(config, raw_dir, f"operator:{'promote' if kind == STAGED else 'revert'}") as state:
        state["commit"] = True
        result = restore(path, version, ledger=ledger)
        state["issues"][:] = [issue for issue in state["issues"] if issue_key(issue) != ("staged", dataset, partition)]
        # A promoted restatement leaves the months outside it at the older
        # vintage, as an accepted one does.
        if (
            kind == STAGED
            and history_repull_policy(config).get(dataset) == "on_restatement"
            and not any(issue_key(issue) == ("history_repull", dataset, "") for issue in state["issues"])
        ):
            state["issues"].append({"kind": "history_repull", "dataset": dataset, "partition": "", "since": utc_now(), "detail": "", "version": ""})
        record_vintages(raw_dir, state["datasets"], {dataset})
    return {"status": "ok", "dataset": dataset, "partition": partition, **result}


def repull_dataset(config_path: str, dataset: str, *, start_date: str | None, end_date: str | None) -> int:
    config, raw_dir, _ledger = _lake(config_path)
    timezone_name = config.get("timezone", "Asia/Shanghai")
    stamp = datetime.now(ZoneInfo(timezone_name)).strftime("%Y%m%d_%H%M%S")
    job_name = f"operator_repull_{dataset}"
    ctx = RunContext(
        config, Path.cwd(), config.get("python") or sys.executable, job_name, {"operation": "download_event_flow"},
        start_date or "", end_date or "", timezone_name,
    )
    log_path = RUN_LOG_ROOT / f"tushare_cron_{job_name}_{stamp}.log"
    with locked_generation(config, raw_dir, f"operator:repull:{dataset}") as state:
        state["commit"] = True
        code, restated = run_history_repull(
            ctx, dataset, log_path, state["lock_fd"], state["datasets"], state["issues"], start_date=start_date, end_date=end_date,
        )
        record_vintages(raw_dir, state["datasets"], restated | {dataset})
    print(json.dumps({"status": "ok" if code == 0 else "error", "dataset": dataset, "returncode": code, "log_path": str(log_path)}, ensure_ascii=False))
    return code


def rebuild_partition_observations(config_path: str, dataset: str, partition: str) -> dict:
    config, raw_dir, _ledger = _lake(config_path)
    path = _partition(raw_dir, dataset, partition)
    with locked_generation(config, raw_dir, "operator:rebuild-observations") as state:
        state["commit"] = True
        stats = rebuild_observations(path)
    return {"status": "ok", "dataset": dataset, "partition": partition, **stats}


@contextmanager
def _updater_lock(config: dict, repo_root: Path) -> Iterator[None]:
    """The updater lock for a command that changes no partition: the weekly
    job's runner holds it for its child, a manual run takes it like every
    operator command."""
    if inherited_updater_lock_fd(repo_root) is not None:
        yield
        return
    lock = acquire_lock(repo_root, int(config.get("default_lock_wait_seconds", DEFAULT_LOCK_WAIT_SECONDS)))
    try:
        yield
    finally:
        lock.release()


# ``(repo_root, raw_dir) -> {generation: [why it stays, ...]}``: the census
# ``prune-releases`` keeps releases by (``paper.release_census.release_consumers``).
ReleaseConsumers = Callable[[Path, Path], Mapping[str, list[str]]]


def prune_releases(config_path: str, *, apply: bool, consumers: ReleaseConsumers) -> dict:
    """Remove the research releases no consumer reads any more (see the module docstring)."""
    config, raw_dir, _ledger = _lake(config_path)
    repo_root = Path.cwd()  # _lake works from the repository root
    freed = 0

    def removal(path: Path) -> dict:
        nonlocal freed
        if not apply:
            return {"action": "remove"}
        size = remove_release_tree(path)
        freed += size
        return {"action": "removed", "freed_bytes": size}

    with _updater_lock(config, repo_root) if apply else nullcontext():
        releases, leftovers = retire_research_releases(
            raw_dir.parent / "research_releases", lambda: consumers(repo_root, raw_dir), apply=apply
        )
        for path in leftovers:
            print(json.dumps({"leftover": path.name, **removal(path)}, ensure_ascii=False), flush=True)
        for release in releases:
            line = {"generation_id": release.generation_id, "created_at": release.created_at}
            if release.kept_for:
                line.update(action="keep", kept_for=list(release.kept_for))
            else:
                line.update(removal(release.path))
            print(json.dumps(line, ensure_ascii=False), flush=True)
    unreferenced = sum(not release.kept_for for release in releases)
    return {
        "mode": "apply" if apply else "dry_run",
        "releases": len(releases),
        "kept": len(releases) - unreferenced,
        "unreferenced": unreferenced,
        "leftovers": len(leftovers),
        **({"freed_bytes": freed} if apply else {}),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)
    for verb, kind in (("revert", SUPERSEDED), ("promote", STAGED)):
        command = sub.add_parser(verb, help=f"make a {kind} version the visible partition")
        command.add_argument("dataset")
        command.add_argument("partition", help="the partition without .parquet, e.g. month=202610")
        command.add_argument("version", help=f"a version id, e.g. 20261024T014501123456Z-{kind}")
    repull = sub.add_parser("repull", help="re-pull the whole history of a month-partitioned event table")
    repull.add_argument("dataset")
    repull.add_argument("--start-date", help="override the history floor")
    repull.add_argument("--end-date", help="override today (Asia/Shanghai)")
    rebuild = sub.add_parser("rebuild-observations", help="rebuild a partition's observation table from its versions")
    rebuild.add_argument("dataset")
    rebuild.add_argument("partition")
    prune = sub.add_parser("prune-releases", help="remove the research releases no consumer reads any more")
    prune_mode = prune.add_mutually_exclusive_group()
    prune_mode.add_argument("--dry-run", action="store_true", help="list what would be removed and why the rest stay (the default)")
    prune_mode.add_argument("--apply", action="store_true", help="remove them under the updater and registry locks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, release_consumers: ReleaseConsumers | None = None) -> int:
    """The operator CLI; ``prune-releases`` needs ``release_consumers``, which
    ``scripts/data/tushare_lake.py`` passes."""
    args = parse_args(argv)
    if args.command == "repull":
        return repull_dataset(args.config, args.dataset, start_date=args.start_date, end_date=args.end_date)
    if args.command == "prune-releases":
        if release_consumers is None:
            raise ValueError("prune-releases needs the release census (scripts/data/tushare_lake.py passes it)")
        result = prune_releases(args.config, apply=args.apply, consumers=release_consumers)
    elif args.command in {"revert", "promote"}:
        kind = SUPERSEDED if args.command == "revert" else STAGED
        result = restore_partition(args.config, args.dataset, args.partition, args.version, kind=kind)
    else:
        result = rebuild_partition_observations(args.config, args.dataset, args.partition)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
