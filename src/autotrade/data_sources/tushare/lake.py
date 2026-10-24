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
* ``migrate --dry-run | --apply`` starts versioning on a lake written before
  it: existing partitions become the current versions with no observation rows,
  the duplicate business keys of ``anns_d`` and ``repurchase`` are dropped
  (each changed partition kept as a superseded version, with a ledger event),
  and the generation record is rewritten to schema 3 with ``versioned_since``,
  the day the versioned era starts (periods ending before it never get
  observation rows). Idempotent: a second run keeps that day.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from autotrade.environment.data.contracts import (
    GENERATION_COMMITTED,
    GENERATION_SCHEMA_VERSION,
)
from autotrade.environment.data.pit import complete_interrupted_commits

from .common import (
    EVENT_FLOW_SPECS,
    REVISION_EVENTS_PATH,
    TEXT_SPECS,
    parquet_meta,
    resolve_revision_ledger,
)
from .cron_update import (
    DEFAULT_CONFIG,
    DEFAULT_LOCK_WAIT_SECONDS,
    RUN_LOG_ROOT,
    RunContext,
    _issue_key,
    _read_raw_generation_file,
    acquire_lock,
    begin_raw_generation_update,
    finish_raw_generation_update,
    history_repull_policy,
    load_config,
    record_vintages,
    run_history_repull,
    utc_now,
)
from .versions import (
    STAGED,
    SUPERSEDED,
    dataset_vintage,
    deduplicate_keys,
    partition_id,
    publish,
    rebuild_observations,
    restore,
)

# The two tables whose stored partitions still hold duplicate business keys
# from before keys were deduplicated at ingest.
MIGRATION_DEDUP_DATASETS = ("anns_d", "repurchase")


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
    lock = acquire_lock("tushare_update", int(config.get("default_lock_wait_seconds", DEFAULT_LOCK_WAIT_SECONDS)))
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
            "versioned_since": previous.get("versioned_since"),
        }
        try:
            yield state
        finally:
            finish_raw_generation_update(
                raw_dir, previous, transaction, commit=state["commit"], datasets=state["datasets"], issues=state["issues"],
                versioned_since=state["versioned_since"],
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
        state["issues"][:] = [issue for issue in state["issues"] if _issue_key(issue) != ("staged", dataset, partition)]
        # A promoted restatement leaves the months outside it at the older
        # vintage, as an accepted one does.
        if (
            kind == STAGED
            and history_repull_policy(config).get(dataset) == "on_restatement"
            and not any(_issue_key(issue) == ("history_repull", dataset, "") for issue in state["issues"])
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


def _latest_commit(dataset_dir: Path) -> str | None:
    """The latest commit time a dataset's sidecars record: the best evidence of
    its last refresh a lake written before refresh times were recorded has."""
    latest: datetime | None = None
    for meta_path in dataset_dir.rglob("*.parquet.meta.json"):
        fetched = json.loads(meta_path.read_text(encoding="utf-8")).get("fetched_at")
        try:
            instant = datetime.fromisoformat(str(fetched)) if fetched else None
        except ValueError:
            instant = None
        if instant is not None and (latest is None or instant > latest):
            latest = instant
    return latest.isoformat() if latest is not None else None


def _dedup_spec(dataset: str) -> list[str]:
    spec = TEXT_SPECS.get(dataset) or EVENT_FLOW_SPECS.get(dataset)
    if spec is None or not spec.unique_key:
        raise RuntimeError(f"{dataset} has no unique business key to deduplicate on")
    return list(spec.key_columns)


def migrate(config_path: str, *, apply: bool) -> dict:
    """Start versioning on the lake (see the module docstring)."""
    config, raw_dir, ledger = _lake(config_path)

    def run(state: dict | None) -> dict:
        report: dict = {"mode": "apply" if apply else "dry_run", "raw_dir": str(raw_dir), "deduplicate": {}}
        dataset_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
        # Read before the deduplication commits, which are no vendor refresh.
        refreshed = {path.name: _latest_commit(path) for path in dataset_dirs}
        for dataset in MIGRATION_DEDUP_DATASETS:
            keys = _dedup_spec(dataset)
            partitions = []
            for path in sorted((raw_dir / dataset).glob("*.parquet")):
                frame = pd.read_parquet(path)
                _, dropped = deduplicate_keys(frame, keys)
                if not dropped:
                    continue
                entry = {"partition": partition_id(path), "rows": len(frame), "duplicate_rows": dropped}
                if state is not None:
                    state["commit"] = True
                    meta = parquet_meta(path)
                    entry["outcome"] = publish(
                        path, frame, api_name=str(meta.get("api_name") or dataset), params=dict(meta.get("params") or {}),
                        fields=list(meta.get("fields") or frame.columns), key_columns=keys, ledger=ledger,
                        source="migration", deduplicate=True, current=frame,
                    )
                partitions.append(entry)
                print(json.dumps({"dataset": dataset, **entry}, ensure_ascii=False))
            report["deduplicate"][dataset] = {
                "partitions": len(partitions),
                "duplicate_rows": sum(item["duplicate_rows"] for item in partitions),
                "rows_before": sum(item["rows"] for item in partitions),
            }
        previous = _read_raw_generation_file(raw_dir) if state is None else state["previous"]
        known = (previous.get("datasets") or {}) if previous.get("schema_version") == GENERATION_SCHEMA_VERSION else {}
        datasets = {}
        for dataset_dir in dataset_dirs:
            # A record already at schema 3 keeps the refresh times the runner
            # recorded; the vintage always describes the lake as it now is.
            since = (known.get(dataset_dir.name) or {}).get("refreshed_at") or refreshed[dataset_dir.name]
            datasets[dataset_dir.name] = {"vintage": dataset_vintage(dataset_dir), **({"refreshed_at": since} if since else {})}
        # The versioned era starts on the day of the first migration and never
        # moves: periods ending before it keep no observation rows.
        since = previous.get("versioned_since") or datetime.now(ZoneInfo(config.get("timezone", "Asia/Shanghai"))).strftime("%Y%m%d")
        if state is not None:
            state["datasets"] = datasets
            state["issues"] = list(previous.get("issues") or []) if previous.get("schema_version") == GENERATION_SCHEMA_VERSION else []
            state["versioned_since"] = since
        report["generation"] = {
            "schema_version": {"from": previous.get("schema_version"), "to": GENERATION_SCHEMA_VERSION},
            "state_before": previous.get("state"),
            "generation_id_before": previous.get("generation_id"),
            # finish_raw_generation_update also commits over a record that was
            # not committed when the command began.
            "new_generation": any(item["partitions"] for item in report["deduplicate"].values())
            or previous.get("state") != GENERATION_COMMITTED,
            "datasets": len(datasets),
            "versioned_since": {"from": previous.get("versioned_since"), "to": since},
            "datasets_without_refresh_time": sorted(name for name, entry in datasets.items() if "refreshed_at" not in entry),
        }
        return report

    if not apply:
        return run(None)
    with locked_generation(config, raw_dir, "operator:migrate") as state:
        return run(state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)
    migrate_parser = sub.add_parser("migrate", help="start versioning on a lake written before it")
    mode = migrate_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report what the migration would change, change nothing")
    mode.add_argument("--apply", action="store_true", help="run the migration under the updater lock")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "repull":
        return repull_dataset(args.config, args.dataset, start_date=args.start_date, end_date=args.end_date)
    if args.command == "migrate":
        result = migrate(args.config, apply=args.apply)
    elif args.command in {"revert", "promote"}:
        kind = SUPERSEDED if args.command == "revert" else STAGED
        result = restore_partition(args.config, args.dataset, args.partition, args.version, kind=kind)
    else:
        result = rebuild_partition_observations(args.config, args.dataset, args.partition)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
