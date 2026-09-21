#!/usr/bin/env python3
"""Cron-safe TuShare update runner for AutoTrade."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from autotrade.environment.data.contracts import (
    GENERATION_COMMITTED,
    GENERATION_IN_PROGRESS,
    GENERATION_SCHEMA_VERSION,
    RAW_GENERATION_FILENAME,
)

from .audit import AUDIT_SUMMARY_RE
from .common import (
    MUTATED_NOT_READY_RETRY_EXIT_CODE,
    NO_MUTATION_RETRY_EXIT_CODE,
    normalize_date_key,
    read_many,
)
from .io import UPDATER_LOCK_FD_ENV


DEFAULT_CONFIG = Path("configs/tushare_update_schedule.json")
RUNTIME_ROOT = Path(".runtime/tushare")
# One outcome file per job: jobs write disjoint files, so persisting state
# needs no cross-job lock or merge protocol.
JOB_STATE_ROOT = RUNTIME_ROOT / "jobs"
RUN_LOG_ROOT = Path("logs/tushare/cron")
RUN_LOG_RETENTION_DAYS = 14
DEFAULT_LOCK_WAIT_SECONDS = 900
# Job operations that mutate the raw/PIT lake and therefore publish a new
# generation on success; audit-only jobs must not churn snapshot cache keys.
# The generation schema/state contract itself is owned by
# autotrade.environment.data.contracts so writer and PIT consumers cannot drift.
MUTATING_OPERATIONS = {"update", "download_tier", "download_event_flow", "intraday_by_date", "pit_event_pipeline", "auction_capture", "auction_recheck", "commit_identity_migration"}


@dataclass
class RunContext:
    config: dict
    repo_root: Path
    python: str
    job_name: str
    job: dict
    start_date: str
    end_date: str
    timezone_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a locked scheduled TuShare update job.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to tushare_update_schedule.json.")
    parser.add_argument("--job", required=True, help="Job name from the schedule config.")
    parser.add_argument("--start-date", help="Override update lower bound. Defaults to TUSHARE_UPDATE_START_DATE or config default_start_date.")
    parser.add_argument("--end-date", help="Override update end date. Defaults to job offset from current Asia/Shanghai date.")
    parser.add_argument("--dry-run", action="store_true", help="Print the computed command without running it.")
    parser.add_argument("--force-run", action="store_true", help="Run even if this job/date already has an ok state.")
    parser.add_argument(
        "--then",
        action="append",
        default=[],
        metavar="JOB",
        help="Job that must run once --job has finished; repeatable, run in the order given. Each one takes the updater lock on its own and runs forced. Skipped when --job did not succeed.",
    )
    return parser.parse_args()


# The schedule config's schema_version is a real producer/consumer contract:
# an edit that reshapes the config must bump both this constant and the file,
# or every runner refuses to start (fail-fast, no silent drift).
SCHEDULE_CONFIG_SCHEMA_VERSION = 1


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"schedule config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    version = config.get("schema_version")
    if version != SCHEDULE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schedule config schema_version {version!r} != expected "
            f"{SCHEDULE_CONFIG_SCHEMA_VERSION}; regenerate or migrate {path}"
        )
    return config


def resolve_sse_open_on_or_before(repo_root: Path, raw_dir: str, target_date: str) -> str:
    trade_cal_dir = repo_root / raw_dir / "trade_cal" / "exchange=SSE"
    files = sorted(trade_cal_dir.glob("year=*.parquet"))
    if not files:
        raise RuntimeError(f"SSE trade_cal partitions are missing under {trade_cal_dir}; run reference download first")
    calendar = read_many(files, columns=["cal_date", "is_open"])
    if calendar.empty:
        raise RuntimeError(f"SSE trade_cal is empty under {trade_cal_dir}; refresh reference trade_cal first")
    calendar["cal_date"] = calendar["cal_date"].map(normalize_date_key)
    open_dates = sorted(
        calendar.loc[
            (calendar["is_open"].astype(str) == "1")
            & (calendar["cal_date"] != "")
            & (calendar["cal_date"] <= target_date),
            "cal_date",
        ].tolist()
    )
    if not open_dates:
        raise RuntimeError(f"no SSE open date found on or before {target_date}")
    return str(open_dates[-1])


def resolve_job_end_date(job: dict, repo_root: Path, raw_dir: str, target_date: str) -> str:
    mode = str(job.get("end_date_mode", "calendar_date"))
    if mode == "calendar_date":
        return target_date
    if mode == "sse_open_on_or_before":
        return resolve_sse_open_on_or_before(repo_root, raw_dir, target_date)
    raise ValueError(f"unsupported end_date_mode: {mode}")


def is_sse_open_date(repo_root: Path, raw_dir: str, target_date: str) -> bool:
    """Whether ``target_date`` itself is open, without silently rolling backward."""
    files = sorted((repo_root / raw_dir / "trade_cal" / "exchange=SSE").glob("year=*.parquet"))
    if not files:
        raise RuntimeError(f"SSE trade_cal partitions are missing under {repo_root / raw_dir}")
    calendar = read_many(files, columns=["cal_date", "is_open"])
    dates = calendar["cal_date"].map(normalize_date_key)
    exact = dates == target_date
    if not exact.any():
        raise RuntimeError(f"SSE trade_cal does not cover target date {target_date}")
    return bool((exact & (calendar["is_open"].astype(str) == "1")).any())


def resolve_event_flow_audit_end_date(ctx: RunContext) -> str:
    # Only the calendar-offset form exists: no production job ever set an
    # event-flow-specific end_date_mode, and the drift-guard test pins its
    # absence from the schedule config (dev-phase removal 2026-08-02).
    event_flow_end_date = ctx.end_date
    event_extra_offset = int(ctx.job.get("event_flow_end_extra_offset_days", 0))
    if event_extra_offset:
        event_flow_end_date = (
            datetime.strptime(ctx.end_date, "%Y%m%d").date() - timedelta(days=event_extra_offset)
        ).strftime("%Y%m%d")
    return event_flow_end_date


def build_context(args: argparse.Namespace) -> RunContext:
    config = load_config(Path(args.config))
    jobs = config.get("jobs", {})
    if args.job not in jobs:
        raise KeyError(f"unknown job {args.job!r}; available={sorted(jobs)}")
    timezone_name = config.get("timezone", "Asia/Shanghai")
    tz = ZoneInfo(timezone_name)
    now = datetime.now(tz)
    job = jobs[args.job]
    repo_root = Path(config.get("repo_root", ".")).resolve()
    python = config.get("python") or sys.executable
    raw_dir = config.get("default_raw_dir", "data/raw")
    offset_days = int(job.get("end_date_offset_days", 0))
    target_date = args.end_date or (now.date() - timedelta(days=offset_days)).strftime("%Y%m%d")
    end_date = resolve_job_end_date(job, repo_root, raw_dir, target_date)
    env_start_date = os.environ.get("TUSHARE_UPDATE_START_DATE")
    if args.start_date or env_start_date:
        start_date = args.start_date or env_start_date or config["default_start_date"]
    elif "start_date_lookback_days" in job:
        # A declared lookback wins for every operation: the disclosure job is
        # download_event_flow but needs the trailing month for late vendor
        # corrections, unlike the same-day pre-open margin jobs below.
        end_day = datetime.strptime(end_date, "%Y%m%d").date()
        start_date = (end_day - timedelta(days=int(job["start_date_lookback_days"]))).strftime("%Y%m%d")
    elif job.get("operation") == "download_event_flow":
        start_date = end_date
    else:
        start_date = config["default_start_date"]
    return RunContext(config, repo_root, python, args.job, job, start_date, end_date, timezone_name)


def build_audit_full_commands(ctx: RunContext) -> list[list[str]]:
    """The six trading-day-bound raw status files.

    Text is deliberately absent: it is a natural-day domain audited by its own
    daily job, and this job's trading-day end date makes every weekend and
    holiday invocation skip as an unchanged range."""
    raw_dir = ctx.config.get("default_raw_dir", "data/raw")
    event_flow_end_date = resolve_event_flow_audit_end_date(ctx)
    return [
        [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "core-market",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ],
        [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "fundamental-raw",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--fundamental-start-date",
            ctx.start_date,
            "--fundamental-end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ],
        [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "macro",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ],
        [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "intraday-by-date",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            # "daily" is an INDEPENDENT universe for the newest-20-day deep
            # check; "minute" read the audited file's own codes back, which can
            # never report a dropped stock. Full-history scans still use
            # "minute": the 2020-2023 vendor minute era genuinely lacks up to
            # ~190 daily codes per day (accepted history, not a defect).
            "--expected-codes-source",
            "daily",
            "--min-rows-per-day",
            "1",
            "--raw-dir",
            raw_dir,
        ],
        [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "event-flow",
            "--start-date",
            ctx.start_date,
            "--end-date",
            event_flow_end_date,
            "--raw-dir",
            raw_dir,
        ],
        [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "board-trading",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ],
    ]


def build_job_commands(ctx: RunContext) -> list[list[str]]:
    raw_dir = ctx.config.get("default_raw_dir", "data/raw")
    operation = ctx.job.get("operation", "update")
    if operation == "update":
        command = [
            ctx.python,
            "scripts/data/tushare_download.py",
            "update",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.config.get("default_update_args", []))
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "download_event_flow":
        command = [
            ctx.python,
            "scripts/data/tushare_download.py",
            "download",
            "--tier",
            "event_flow",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.config.get("default_update_args", []))
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "download_tier":
        tier = ctx.job.get("tier")
        if not tier:
            raise ValueError("download_tier job requires a tier")
        command = [
            ctx.python,
            "scripts/data/tushare_download.py",
            "download",
            "--tier",
            tier,
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.config.get("default_update_args", []))
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "intraday_by_date":
        # The minute layer on its own: its vendor quota must not be able to
        # fail the evening update that Paper's morning release depends on.
        command = [
            ctx.python,
            "scripts/data/tushare_download.py",
            "update-intraday-by-date",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.config.get("default_update_args", []))
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "auction_capture":
        command = [
            ctx.python,
            "scripts/data/tushare_download.py",
            "capture-open-auction",
            "--trade-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.config.get("default_update_args", []))
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "auction_recheck":
        command = [
            ctx.python,
            "scripts/data/tushare_download.py",
            "recheck-stk-auction",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.config.get("default_update_args", []))
        command.extend(ctx.job.get("extra_args", []))
        command.extend(["--landing-job", ctx.job_name])
        return [command]
    if operation == "audit_event_flow":
        command = [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "event-flow",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "audit_text":
        command = [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "text",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "pit_event_pipeline":
        raw_dir = ctx.config.get("default_raw_dir", "data/raw")
        pit_root = ctx.config.get("default_pit_root", "data/pit")
        fundamental_root = ctx.job.get("fundamental_events_root", f"{pit_root}/fundamental_events")
        # Build and audit the FULL retained window every night. The build reads
        # the whole raw history regardless (availability joins need it), so a
        # rolling write window saved little and let months outside it drift
        # against the living raw lake unaudited (a full-window audit found
        # ~85k stale-provenance rows the 120-day window never saw).
        # Month-aligned like the builder's replace window, so a manual
        # mid-month --start-date cannot put built rows outside the audit window.
        event_start_date = f"{ctx.start_date[:6]}01"
        commands = [
            [
                ctx.python,
                "scripts/data/build_pit_events.py",
                "build-fundamental-events",
                "--raw-dir",
                raw_dir,
                "--output-root",
                fundamental_root,
                "--start-date",
                event_start_date,
                "--end-date",
                ctx.end_date,
            ],
            [
                ctx.python,
                "scripts/data/build_pit_events.py",
                "audit-fundamental-events",
                "--events-root",
                fundamental_root,
                "--start-date",
                event_start_date,
                "--end-date",
                ctx.end_date,
                "--output",
                ctx.job.get("fundamental_events_status", "results/data_quality/fundamental_events_status.json"),
                "--require-partitions",
            ],
        ]
        return commands
    if operation == "commit_identity_migration":
        # Content-preserving, but it rewrites the parquet itself to stamp the
        # footer write_id, so it takes the updater lock AND counts as a
        # MUTATING_OPERATION: a consumer must not read the lake while the
        # footer/sidecar pairs are being restamped, and a partial run leaves
        # the lake in a state the next run of the same job must recover.
        command = [
            ctx.python,
            "scripts/data/migrate_commit_identity.py",
            "--raw-dir",
            raw_dir,
        ]
        command.extend(ctx.job.get("extra_args", []))
        return [command]
    if operation == "revision_sentinel":
        revision_config = ctx.config.get("revision_monitor", {})
        command = [
            ctx.python,
            "scripts/data/tushare_audit.py",
            "revision-sentinel",
            "--start-date",
            ctx.start_date,
            "--end-date",
            ctx.end_date,
            "--raw-dir",
            raw_dir,
        ]
        if revision_config.get("ledger_path"):
            command.extend(["--revision-ledger", str(revision_config["ledger_path"])])
        if revision_config.get("summary_path"):
            command.extend(["--output", str(revision_config["summary_path"])])
        command.extend(ctx.config.get("default_update_args", []))
        extra_args = list(ctx.job.get("extra_args", []))
        if "--sample-size" not in extra_args and revision_config.get("sentinel_sample_size") is not None:
            extra_args.extend(["--sample-size", str(revision_config["sentinel_sample_size"])])
        if "--datasets" not in extra_args and revision_config.get("sentinel_datasets"):
            extra_args.append("--datasets")
            extra_args.extend(str(dataset) for dataset in revision_config["sentinel_datasets"])
        command.extend(extra_args)
        return [command]
    if operation == "audit_full":
        return build_audit_full_commands(ctx)
    raise ValueError(f"unsupported cron operation: {operation}")


def job_state_path(job_name: str) -> Path:
    return JOB_STATE_ROOT / f"{job_name}.json"


def read_job_state(job_name: str) -> dict:
    path = job_state_path(job_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A torn per-job file only costs this one job its skip record: the
        # next run re-executes (jobs are idempotent) and rewrites it.
        print(json.dumps({"note": "corrupt_job_state_ignored", "path": str(path)}, ensure_ascii=False))
        return {}


def record_job_state(job_name: str, record: dict) -> dict:
    """Persist one job's latest outcome as its own file via atomic replace.

    Jobs write disjoint files, so persisting an outcome needs no cross-job
    lock or read-merge protocol (the shared-file design required both, and
    a pre-run snapshot written back after a multi-hour run once erased a
    concurrently recorded failure). Concurrent runs of the SAME job are
    serialized by the updater flock for their work bodies; for the rare
    lock-timeout writer racing the holder, the last completed writer wins
    and the next scheduled run self-heals a stale record."""
    path = job_state_path(job_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return record


def job_config_identity(ctx: RunContext) -> dict:
    """Return only schedule inputs that can affect this job."""
    operation = str(ctx.job.get("operation", "update"))
    shared_keys = {
        "schema_version",
        "timezone",
        "repo_root",
        "python",
        "default_start_date",
        "default_raw_dir",
        "default_lock_wait_seconds",
    }
    if operation in {
        "update",
        "download_tier",
        "download_event_flow",
        "intraday_by_date",
        "auction_capture",
        "auction_recheck",
        "revision_sentinel",
    }:
        shared_keys.add("default_update_args")
    if operation == "pit_event_pipeline":
        shared_keys.add("default_pit_root")
    if operation == "revision_sentinel":
        shared_keys.add("revision_monitor")
    return {
        "shared": {key: ctx.config.get(key) for key in sorted(shared_keys)},
        "job": ctx.job,
    }


@dataclass
class FileLock:
    """A held kernel flock; the file itself is never deleted (unlinking a
    flock-backed lock file races a concurrent opener onto a dead inode)."""

    path: Path
    fd: int

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


def acquire_lock(lock_name: str, wait_seconds: int) -> FileLock:
    """Exclusive kernel flock: released automatically when the holder exits,
    so a crashed/killed run can never leave a permanently stale lock (the old
    PID-file scheme broke on PID reuse). pid/started_at are diagnostics only."""
    lock = RUNTIME_ROOT / "locks" / f"{lock_name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.close(fd)
                raise RuntimeError(
                    f"lock is held after waiting {wait_seconds}s, another run may be active: {lock}"
                ) from None
            time.sleep(min(15.0, remaining))
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()}\nstarted_at={utc_now()}\n".encode("utf-8"))
    return FileLock(lock, fd)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_raw_generation_file(raw_dir: Path) -> dict:
    """Writer-side lenient read: recovery must see updating/dirty/legacy states."""
    path = raw_dir / RAW_GENERATION_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid raw generation record: {path}: {exc}") from exc


def _write_raw_generation_file(raw_dir: Path, payload: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / RAW_GENERATION_FILENAME
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _restore_raw_generation_file(raw_dir: Path, payload: dict) -> None:
    """Undo an updating fence after a child proves it performed no mutation."""
    if payload:
        _write_raw_generation_file(raw_dir, payload)
    else:
        (raw_dir / RAW_GENERATION_FILENAME).unlink(missing_ok=True)


def write_raw_generation(raw_dir: Path, *, transaction: dict | None = None) -> dict:
    """Publish a committed generation after one fully-successful mutating job."""
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "state": GENERATION_COMMITTED,
        "generation_id": uuid.uuid4().hex,
        "completed_at": utc_now(),
    }
    if transaction:
        payload["transaction"] = dict(transaction)
    _write_raw_generation_file(raw_dir, payload)
    # Single-line JSON: everything the runner prints may be teed into the
    # dispatch log, which is a strict-JSONL contract.
    print(json.dumps({
        "note": "raw_generation_published",
        "generation_id": payload["generation_id"],
        "path": str(raw_dir / ".raw_generation.json"),
        "updated_at": utc_now(),
    }, ensure_ascii=False))
    return payload


def begin_raw_generation_update(raw_dir: Path, transaction: dict) -> dict:
    """Mark the lake unavailable before the first child process can mutate it.

    A dirty/updating transaction is recovered by the next run of the SAME JOB
    NAME. An exact identity match (window and command) resumes the original
    transaction; a same-job run with a different window or command supersedes
    it as a fresh transaction and, on success, commits and clears the fence.
    Supersession is sound because every partition write is atomic and
    sidecar-attested (a dirty lake is incomplete, never corrupt) and the
    nightly full-history audit independently verifies whatever a newer window
    no longer covers. Job windows and explicit commands are recomputed daily, so
    demanding an exact replay left a known-cause failure fencing the lake
    until manual recovery. A DIFFERENT job must still never bless a
    partially-updated lake left by an earlier failure.
    """
    previous = _read_raw_generation_file(raw_dir)
    previous_state = str(previous.get("state", GENERATION_COMMITTED))
    previous_transaction = previous.get("transaction") or {}
    unfinished = previous_state in GENERATION_IN_PROGRESS
    if unfinished and str(previous_transaction.get("job", "")) != str(transaction.get("job", "")):
        failed_job = str(previous_transaction.get("job", "unknown"))
        raise RuntimeError(
            "raw lake has an unfinished mutation; rerun the original job before any other mutation: "
            f"job={failed_job} state={previous_state}"
        )
    identity_keys = ("job", "start_date", "end_date", "commands", "config_identity")
    resuming = unfinished and all(previous_transaction.get(key) == transaction.get(key) for key in identity_keys)
    if unfinished and not resuming:
        # Single-line JSON: runner stdout may be teed into a strict-JSONL log.
        print(json.dumps({
            "note": "raw_generation_dirty_superseded",
            "previous_transaction_id": str(previous_transaction.get("transaction_id", "")),
            "previous_window": f"{previous_transaction.get('start_date', '')}..{previous_transaction.get('end_date', '')}",
            "previous_error": str(previous.get("error", ""))[:300],
        }, ensure_ascii=False, sort_keys=True))
    transaction = dict(transaction)
    transaction["transaction_id"] = str(
        previous_transaction.get("transaction_id") if resuming else uuid.uuid4().hex
    )
    transaction["started_at"] = str(
        previous_transaction.get("started_at") if resuming else utc_now()
    )
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "state": "updating",
        "generation_id": str(previous.get("generation_id", "")),
        "completed_at": str(previous.get("completed_at", "")),
        "updated_at": utc_now(),
        "transaction": transaction,
    }
    _write_raw_generation_file(raw_dir, payload)
    return transaction


def mark_raw_generation_dirty(raw_dir: Path, transaction: dict, *, error: str) -> None:
    previous = _read_raw_generation_file(raw_dir)
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "state": "dirty",
        "generation_id": str(previous.get("generation_id", "")),
        "completed_at": str(previous.get("completed_at", "")),
        "updated_at": utc_now(),
        "transaction": dict(transaction),
        "error": str(error)[:1000],
    }
    _write_raw_generation_file(raw_dir, payload)


def log_outcome(log_path: Path, record: dict) -> None:
    """Echo one outcome line to stdout and append it to this run's own log.

    Every invocation — success, failure, skip, lock timeout — leaves its own
    log file, so no run outcome depends on an earlier run's log surviving."""
    line = json.dumps(record, ensure_ascii=False)
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(line + "\n")


# One line per finished audit domain (audit.py's summary print, whose format
# and pattern are defined there so producer and parser cannot drift), and the
# final "SomeError: ..." line of a traceback. Both are matched against the run's
# own log so the persisted job state can tell "the audit completed and found
# data errors" apart from "the tool crashed". A command that prints neither is
# what leaves an operator with a bare return code, so every job's tooling must
# end on one of the two.
_EXCEPTION_LINE_RE = re.compile(r"^[\w.]*(?:Error|Exception): .*$", re.MULTILINE)


def summarize_failure_from_log(log_path: Path, returncode: int) -> str:
    """One-line failure summary for the job state's error field."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"job_returncode={returncode}"
    summaries = AUDIT_SUMMARY_RE.findall(text)
    if summaries:
        return "; ".join(summaries)[:600]
    exceptions = _EXCEPTION_LINE_RE.findall(text)
    if exceptions:
        return exceptions[-1][:600]
    return f"job_returncode={returncode}"


def prune_run_logs(*, now: float | None = None) -> None:
    """Bound cron-run logs while retaining each job's state-linked last run."""
    if not RUN_LOG_ROOT.exists():
        return
    referenced: set[Path] = set()
    if JOB_STATE_ROOT.exists():
        for state_file in JOB_STATE_ROOT.glob("*.json"):
            try:
                record = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and record.get("log_path"):
                referenced.add(Path(str(record["log_path"])).resolve())
    cutoff = (time.time() if now is None else now) - RUN_LOG_RETENTION_DAYS * 86400
    for path in RUN_LOG_ROOT.glob("tushare_cron_*.log"):
        try:
            if path.is_file() and path.resolve() not in referenced and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            # Retention is best-effort and must never block a market-data job.
            continue


def run_probe(command: list[str], log_handle) -> None:
    log_handle.write(f"\n$ {' '.join(command)}\n")
    log_handle.flush()
    subprocess.run(command, cwd=Path.cwd(), stdout=log_handle, stderr=subprocess.STDOUT, check=False)


def run_update(ctx: RunContext, commands: list[list[str]], log_path: Path, *, lock_fd: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    returncodes: list[int] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"started_at={started}\njob={ctx.job_name}\nstart_date={ctx.start_date}\nend_date={ctx.end_date}\ntimezone={ctx.timezone_name}\n")
        run_probe(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            log,
        )
        run_probe(["free", "-h"], log)
        for index, command in enumerate(commands, start=1):
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            env = os.environ.copy()
            src_path = str(ctx.repo_root / "src")
            env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
            env["PYTHONUNBUFFERED"] = "1"
            if lock_fd is not None:
                # The child inherits the held updater flock via pass_fds; the
                # marker stops download.py from re-acquiring it (deadlock-safe).
                env[UPDATER_LOCK_FD_ENV] = str(lock_fd)
            process = subprocess.run(
                command,
                cwd=ctx.repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=env,
                pass_fds=(lock_fd,) if lock_fd is not None else (),
            )
            returncodes.append(process.returncode)
            log.write(f"\ncommand_index={index}\nreturncode={process.returncode}\n")
            if process.returncode != 0 and ctx.job.get("fail_fast", True):
                log.write(f"fail_fast=true; skipped_remaining_commands={len(commands) - index}\n")
                break
        run_probe(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            log,
        )
        run_probe(["free", "-h"], log)
        log.write(f"returncodes={returncodes}\n")
        log.write(f"finished_at={utc_now()}\n")
    retry_codes = (MUTATED_NOT_READY_RETRY_EXIT_CODE, NO_MUTATION_RETRY_EXIT_CODE)
    if returncodes and all(code in {0, *retry_codes} for code in returncodes):
        # A mutation that left required coverage unmet outranks a clean
        # no-mutation retry: the generation must still be committed.
        for code in retry_codes:
            if code in returncodes:
                return code
        return 0
    return 1


def should_skip_completed(ctx: RunContext, args: argparse.Namespace, job_state: dict, payload: dict) -> bool:
    return bool(
        ctx.job.get("skip_if_already_ok", True)
        and not args.force_run
        and job_state.get("start_date") == ctx.start_date
        and job_state.get("end_date") == ctx.end_date
        and job_state.get("status") == "ok"
        and job_state.get("commands") == payload["commands"]
        and job_state.get("config_identity") == payload["config_identity"]
    )


def _run(args: argparse.Namespace) -> int:
    ctx = build_context(args)
    commands = build_job_commands(ctx)
    timestamp = datetime.now(ZoneInfo(ctx.timezone_name)).strftime("%Y%m%d_%H%M%S")
    log_path = RUN_LOG_ROOT / f"tushare_cron_{ctx.job_name}_{ctx.end_date}_{timestamp}.log"
    payload = {
        "job": ctx.job_name,
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "commands": commands,
        "config_identity": job_config_identity(ctx),
        "timezone": ctx.timezone_name,
    }
    mutates_lake = ctx.job.get("operation", "update") in MUTATING_OPERATIONS
    raw_dir = ctx.repo_root / ctx.config.get("default_raw_dir", "data/raw")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **payload}, ensure_ascii=False))
        return 0

    os.chdir(ctx.repo_root)
    prune_run_logs()
    job_state = read_job_state(ctx.job_name)
    if ctx.job.get("only_if_sse_open_date") and not is_sse_open_date(
        ctx.repo_root,
        ctx.config.get("default_raw_dir", "data/raw"),
        ctx.end_date,
    ):
        record = record_job_state(ctx.job_name, {
            "status": "ok",
            "returncode": 0,
            "start_date": ctx.start_date,
            "end_date": ctx.end_date,
            "commands": payload["commands"],
            "config_identity": payload["config_identity"],
            "skipped_non_trading_day": True,
            "log_path": str(log_path),
            "updated_at": utc_now(),
        })
        log_outcome(log_path, {**record, "job": ctx.job_name})
        return 0
    generation = _read_raw_generation_file(raw_dir) if mutates_lake else {}
    generation_committed = str(generation.get("state", GENERATION_COMMITTED)) == GENERATION_COMMITTED
    if should_skip_completed(ctx, args, job_state, payload) and generation_committed:
        # A skip confirms the previous ok outcome: the state file keeps
        # pointing at the real run (and protecting its log); only this run's
        # own log records that the invocation happened.
        log_outcome(log_path, {"status": "skipped_already_ok", **payload})
        return 0

    try:
        lock = acquire_lock(
            "tushare_update",
            int(ctx.job.get("lock_wait_seconds", ctx.config.get("default_lock_wait_seconds", DEFAULT_LOCK_WAIT_SECONDS))),
        )
    except RuntimeError as exc:
        record = record_job_state(ctx.job_name, {
            "status": "error",
            "returncode": 1,
            "start_date": ctx.start_date,
            "end_date": ctx.end_date,
            "commands": payload["commands"],
            "config_identity": payload["config_identity"],
            "error": str(exc),
            "log_path": str(log_path),
            "updated_at": utc_now(),
        })
        log_outcome(log_path, {**record, "job": ctx.job_name})
        return 1

    returncode = 1
    state_recorded = False
    try:
        job_state = read_job_state(ctx.job_name)
        generation = _read_raw_generation_file(raw_dir) if mutates_lake else {}
        generation_committed = str(generation.get("state", GENERATION_COMMITTED)) == GENERATION_COMMITTED
        if should_skip_completed(ctx, args, job_state, payload) and generation_committed:
            log_outcome(log_path, {"status": "skipped_already_ok_after_lock", **payload})
            return 0
        transaction = None
        generation_before = generation
        if mutates_lake:
            transaction = begin_raw_generation_update(
                raw_dir,
                {
                    "job": ctx.job_name,
                    "start_date": ctx.start_date,
                    "end_date": ctx.end_date,
                    "commands": payload["commands"],
                    "config_identity": payload["config_identity"],
                },
            )
        try:
            returncode = run_update(ctx, commands, log_path, lock_fd=lock.fd)
        except Exception as exc:
            if transaction is not None:
                mark_raw_generation_dirty(raw_dir, transaction, error=f"runner_exception: {exc}")
            raise
        # Exit 75 asserts "no lake mutation happened"; only operations whose
        # download paths enforce that contract may restore the prior generation
        # (auction polling, and event_flow runs with --zero-rows-not-ready).
        no_mutation_retry = bool(
            returncode == NO_MUTATION_RETRY_EXIT_CODE
            and ctx.job.get("operation") in {"auction_capture", "download_event_flow"}
            and len(commands) == 1
        )
        # Exit 76 asserts the opposite half of the same contract: real writes
        # landed, so the generation is committed, but a required partition is
        # still unpublished. Recording "not_ready" (never "ok") is what keeps
        # should_skip_completed from suppressing the next attempt.
        mutated_not_ready = bool(
            returncode == MUTATED_NOT_READY_RETRY_EXIT_CODE
            and ctx.job.get("operation") in {"download_event_flow", "intraday_by_date"}
            and len(commands) == 1
        )
        if transaction is not None:
            if no_mutation_retry:
                _restore_raw_generation_file(raw_dir, generation_before)
            elif returncode == 0 or mutated_not_ready:
                write_raw_generation(raw_dir, transaction=transaction)
            else:
                mark_raw_generation_dirty(raw_dir, transaction, error=f"job_returncode={returncode}")
        status = "ok" if returncode == 0 else (
            "not_ready" if no_mutation_retry or mutated_not_ready else "error"
        )
        state: dict = {
            "status": status,
            "returncode": returncode,
            "start_date": ctx.start_date,
            "end_date": ctx.end_date,
            "commands": payload["commands"],
            "config_identity": payload["config_identity"],
            "log_path": str(log_path),
            "updated_at": utc_now(),
        }
        if status == "error":
            state["error"] = summarize_failure_from_log(log_path, returncode)
        record = record_job_state(ctx.job_name, state)
        state_recorded = True
        log_outcome(log_path, {**record, "job": ctx.job_name})
        return returncode
    except Exception as exc:
        if state_recorded:
            # The outcome is already durably recorded (e.g. the final print
            # hit a broken pipe): never demote a recorded result.
            raise
        # A pre-run abort (e.g. the dirty-lake fence in
        # begin_raw_generation_update) used to propagate before the state
        # writer ran, so the job state silently preserved a stale ``ok``
        # while the job failed on every run. Record the failure and re-raise
        # -- explicit failure over false success.
        record = record_job_state(ctx.job_name, {
            "status": "error",
            "returncode": 1,
            "start_date": ctx.start_date,
            "end_date": ctx.end_date,
            "commands": payload["commands"],
            "config_identity": payload["config_identity"],
            "error": str(exc)[:1000],
            "log_path": str(log_path),
            "updated_at": utc_now(),
        })
        log_outcome(log_path, {**record, "job": ctx.job_name})
        raise
    finally:
        lock.release()


def run_follow_ups(args: argparse.Namespace) -> int:
    """Run every ``--then`` job in order, each as its own locked run.

    A job that mutates the lake and the audit and PIT rebuild that must follow
    it used to be separate cron lines 60-90 minutes apart. On 2026-09-19 the
    weekly fundamental sweep held the updater lock for four hours and all
    three of its follow-ups died waiting for it, so a generation was committed
    with sweep-mutated fundamentals, no rebuilt PIT events and no audit.
    Sequencing them here makes "audit follows mutation" hold by construction
    rather than by clock arithmetic: the follow-up starts when the primary job
    has finished and released the lock, and takes the lock itself, so nothing
    is held between steps.

    A follow-up is forced because it exists in answer to a mutation: its own
    same-day ``ok`` state describes the lake as it was before. One that fails
    -- an audit reports its findings through a non-zero exit -- must not
    starve the ones after it, which is the failure this whole chain exists to
    prevent, so each step is run and reported and the chain returns the first
    failure.
    """
    outcome = 0
    for job_name in args.then:
        step = argparse.Namespace(**{**vars(args), "job": job_name, "then": [], "force_run": True})
        error = ""
        try:
            code = _run(step)
        except Exception as exc:  # noqa: BLE001 - one step must not starve the rest
            code, error = 1, str(exc)[:1000]
        record = {
            "status": "ok" if code == 0 else "error",
            "follow_up": job_name,
            "after": args.job,
            "returncode": code,
        }
        if error:
            record["error"] = error
        print(json.dumps(record, ensure_ascii=False))
        outcome = outcome or code
    return outcome


def main() -> int:
    args = parse_args()
    if args.then:
        # A mistyped follow-up must fail now, not after a four-hour primary.
        jobs = load_config(Path(args.config)).get("jobs", {})
        unknown = sorted(name for name in args.then if name not in jobs)
        if unknown:
            raise KeyError(f"unknown --then job(s) {unknown}; available={sorted(jobs)}")
    returncode = _run(args)
    if not args.then:
        return returncode
    if returncode != 0:
        # Nothing was committed, so nothing has to follow: the failure already
        # fenced the lake, and the recovery is to rerun this same job.
        print(json.dumps({
            "status": "follow_ups_skipped",
            "after": args.job,
            "returncode": returncode,
            "follow_ups": args.then,
        }, ensure_ascii=False))
        return returncode
    return run_follow_ups(args)


if __name__ == "__main__":
    raise SystemExit(main())
