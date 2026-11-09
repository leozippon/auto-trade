"""Persistent HITL control, status, and deterministic session plans."""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from autotrade.common.records import read_json_object, write_json_atomic
from autotrade.environment.gpu import (
    GpuUnavailableError,
    gpus_with_free_memory,
    host_min_free_gpu_memory_mib,
)

from .experiment_params import console_defaults, managed_values
from .ledger import SessionKind

HITL_STATE_SCHEMA_VERSION = 1
HITL_DIR_NAME = "hitl"
PARAMS_NAME = "params.json"
CONTROL_NAME = "control.json"
STATUS_NAME = "status.json"
SCHEDULE_NAME = "schedule.json"
# The GPUs the console handed an experiment when it last started its worker
# (webui.manager.start_worker). That worker attaches exactly these to every
# container it starts, and they are held while it lives (``gpu_claims``).
GPU_CLAIM_NAME = "gpu_claim.json"
# The marker the operator's archive (scripts/experiments/archive_experiment.py) writes
# in an experiment's hitl/ directory, recording when and why; removing it restores the
# experiment. Archiving is the operator's final word that the experiment is over: the console
# lists it apart and never starts its worker again, and the derived-cache prunes
# treat it as finished for good (``experiment_is_archived``).
ARCHIVED_NAME = "archived.json"


class ExperimentState(StrEnum):
    """An experiment's ``state`` in ``status.json``, as its worker, the console and
    the operator's tools write it, plus the two the console derives when it
    reads one: ``interrupted`` (an active state whose worker died) and
    ``unreadable`` (a status file it cannot read). The values are persisted."""

    CREATED = "created"
    LAUNCHING = "launching"
    PREPARING = "preparing"
    INITIALIZING = "initializing"
    RUNNING_SESSION = "running_session"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    INTERRUPTED = "interrupted"
    UNREADABLE = "unreadable"


LIVE_RUN_STATES = {ExperimentState.RUNNING_SESSION}
# The states a worker ends in: an experiment in one of them runs nothing.
ENDED_STATES = frozenset({ExperimentState.COMPLETED, ExperimentState.FAILED, ExperimentState.STOPPED})


# The console's views of the experiment parameters (``experiment_params.ExperimentParams``): what a
# create request may name and what each defaults to, and the values the
# console stamps for the parameters it manages. Both are read off the record's
# fields.
WEB_CREATE_DEFAULTS: dict[str, object] = console_defaults()
WEB_INTERNAL_PARAMS: dict[str, object] = managed_values()


@dataclass
class ControlState:
    request: str | None = None
    # One-shot: the console asked for a code swap at the next session
    # boundary. The worker consumes it and re-executes itself in place, so an
    # in-flight session is finished instead of replayed.
    restart_pending: bool = False
    directives: dict[str, str] = field(default_factory=dict)
    resource_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    # Per-session sandbox GPU allocation set before the session starts: that
    # many of the cards the experiment holds.
    gpu_counts: dict[str, int] = field(default_factory=dict)
    # When this state was last written, as the file recorded it; None for a
    # state that has not been written yet. Only ``write_control`` stamps it,
    # so a projection of the file dates the control from the file and not
    # from the moment it was read.
    updated_at: str | None = None

    def to_record(self) -> dict[str, object]:
        if self.request not in (None, "pause", "stop"):
            raise ValueError(f"invalid HITL request: {self.request}")
        return {
            "schema_version": HITL_STATE_SCHEMA_VERSION,
            "request": self.request,
            "restart_pending": self.restart_pending,
            "directives": dict(self.directives),
            "resource_overrides": dict(self.resource_overrides),
            "gpu_counts": dict(self.gpu_counts),
            "updated_at": self.updated_at,
        }


def read_control(path: str | Path) -> ControlState:
    payload = read_json_object(path, missing_ok=True)
    _require_version(payload, Path(path))
    if not payload:
        return ControlState()
    request = payload.get("request")
    # Keys no longer in the contract are ignored, so a file written before a
    # control was removed still reads.
    return ControlState(
        request=str(request) if request in ("pause", "stop") else None,
        restart_pending=bool(payload.get("restart_pending")),
        directives=_string_map(payload.get("directives")),
        resource_overrides=_object_map(payload.get("resource_overrides")),
        gpu_counts=_int_map(payload.get("gpu_counts")),
        updated_at=str(payload["updated_at"]) if payload.get("updated_at") else None,
    )


def write_control(path: str | Path, state: ControlState) -> None:
    """Write the control file, stamping the state with this write's time."""

    state.updated_at = _now()
    write_json_atomic(Path(path), state.to_record())


@contextmanager
def control_lock(path: str | Path):
    """Serialize controller writes and worker one-shot consumption."""

    target = Path(path)
    lock_path = target.with_name(f".{target.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def consume_session_controls(
    path: str | Path,
    session_key: str,
) -> ControlState:
    """Remove one completed session's consumed control values."""

    with control_lock(path):
        state = read_control(path)
        for mapping in (
            state.directives,
            state.resource_overrides,
            state.gpu_counts,
        ):
            mapping.pop(session_key, None)
        write_control(path, state)
        return state


def consume_restart_request(path: str | Path) -> bool:
    """Consume a pending session-boundary restart; True when one was set."""

    with control_lock(path):
        state = read_control(path)
        if not state.restart_pending:
            return False
        state.restart_pending = False
        write_control(path, state)
        return True


def read_status(path: str | Path) -> dict[str, object]:
    payload = read_json_object(path, missing_ok=True)
    _require_version(payload, Path(path))
    return payload


def write_gpu_claim(hitl: str | Path, devices: list[int]) -> None:
    write_json_atomic(
        Path(hitl) / GPU_CLAIM_NAME,
        {"devices": [int(device) for device in devices], "claimed_at": _now()},
    )


def read_gpu_claim(hitl: str | Path) -> tuple[int, ...] | None:
    """The devices of an experiment's GPU claim, or ``None`` when it has none."""
    path = Path(hitl) / GPU_CLAIM_NAME
    payload = read_json_object(path, missing_ok=True)
    if not payload:
        return None
    devices = payload.get("devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(type(device) is not int or device < 0 for device in devices)
    ):
        raise ValueError(f"malformed GPU claim in {path}")
    return tuple(devices)


def experiment_is_archived(directory: str | Path) -> bool:
    """Whether the operator archived this experiment: it carries ``ARCHIVED_NAME`` and
    is finished for good, whatever its status and ledger say."""

    return (Path(directory) / HITL_DIR_NAME / ARCHIVED_NAME).is_file()


def running_experiments(experiments_root: str | Path) -> list[str]:
    """The experiments under ``experiments_root`` whose worker is alive and has not
    ended, by id. An experiment whose status cannot be read runs nothing."""

    root = Path(experiments_root)
    if not root.is_dir():
        return []
    running: list[str] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            status = read_json_object(directory / HITL_DIR_NAME / STATUS_NAME, missing_ok=True)
        except (OSError, ValueError):
            continue
        if status_pid_alive(status) and status.get("state") not in ENDED_STATES:
            running.append(directory.name)
    return running


def gpu_claims(experiments_root: str | Path) -> dict[str, list[int]]:
    """The devices every running experiment holds, by experiment.

    An experiment holds the cards claimed for its worker for as long as that worker
    lives, whether or not they show memory in use: between batches, while its
    model thinks, they show none. An experiment that asked for no GPU has no claim,
    and the claim an ended worker left behind holds nothing.
    """

    root = Path(experiments_root)
    claims: dict[str, list[int]] = {}
    for name in running_experiments(root):
        claim = read_gpu_claim(root / name / HITL_DIR_NAME)
        if claim is not None:
            claims[name] = list(claim)
    return claims


def free_gpus(
    experiments_root: str | Path,
    *,
    require_name: str | None = None,
    min_free_memory_mib: int | None = None,
) -> list[int]:
    """The GPUs eligible to take, the most free memory first: the devices
    matching ``require_name`` that keep at least ``min_free_memory_mib`` free
    (the host's floor, ``gpu.host_min_free_gpu_memory_mib``, when it is not
    given) and that no running experiment under ``experiments_root`` claims
    (:func:`gpu_claims`). Somebody else's memory on a card does not keep it
    out; only too little free memory or a running experiment's claim does.
    """

    floor = host_min_free_gpu_memory_mib() if min_free_memory_mib is None else min_free_memory_mib
    claimed = {device for devices in gpu_claims(experiments_root).values() for device in devices}
    return [
        device
        for device in gpus_with_free_memory(floor, require_name=require_name)
        if device not in claimed
    ]


def select_gpus(
    experiments_root: str | Path,
    count: int,
    *,
    require_name: str | None = None,
    min_free_memory_mib: int | None = None,
) -> list[int]:
    """The one way a GPU is chosen: the first ``count`` of :func:`free_gpus`.

    The console takes an experiment's cards here when it starts the worker and
    records them as its claim. Whatever else needs a card -- a Paper account's
    morning run, a round's smoke, a session audited from the command line, an
    operator's replay -- takes its cards here for that run and pins them on its
    containers, so it is never given a card a running experiment holds. Every caller
    gets the same rule; ``min_free_memory_mib`` defaults to the host's floor.
    ``GpuUnavailableError`` names the eligible cards and who holds the rest
    when the request cannot be met, or says why the devices could not be read.
    """

    floor = host_min_free_gpu_memory_mib() if min_free_memory_mib is None else min_free_memory_mib
    free = free_gpus(experiments_root, require_name=require_name, min_free_memory_mib=floor)
    if len(free) >= count:
        return free[:count]
    holders = "; ".join(f"{name} {devices}" for name, devices in gpu_claims(experiments_root).items())
    raise GpuUnavailableError(
        f"requested {count} GPU(s), {len(free)} free ({', '.join(map(str, free)) or 'none'}); "
        f"a card is free only when it keeps {floor} MiB free and no running experiment claims it; "
        f"claimed by running experiments: {holders or 'none'}"
    )


class StatusReporter:
    def __init__(self, path: str | Path, *, heartbeat_seconds: float = 3.0) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.path = Path(path)
        self.heartbeat_seconds = heartbeat_seconds
        pid = os.getpid()
        self._state: dict[str, object] = {
            "state": ExperimentState.INITIALIZING,
            "pid": pid,
            "pid_start_ticks": proc_start_ticks(pid),
            "session_key": None,
            "run_id": None,
            "session_started_at": None,
            "environment_stage": None,
            "environment_stage_started_at": None,
            "environment_progress": None,
            "completed_sessions": 0,
            "total_sessions": None,
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._write()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.heartbeat_seconds * 2)

    def set(self, **values: object) -> None:
        with self._lock:
            new_session = "session_started_at" in values and values.get(
                "session_started_at"
            ) != self._state.get("session_started_at")
            if new_session:
                values.setdefault("run_id", None)
                values.setdefault("environment_stage", None)
                values.setdefault("environment_stage_started_at", None)
                values.setdefault("environment_progress", None)
            if "environment_stage" in values:
                stage = values.get("environment_stage")
                # Every stage event restarts the clock, not just a changed
                # stage name: the console reads this as "how long has the
                # CURRENT step been running". Keeping it across repeated
                # llm_call events showed one 38-minute llm_call for what was
                # really a series of short calls plus idle waiting.
                values.setdefault(
                    "environment_stage_started_at",
                    _now() if stage else None,
                )
                if stage != self._state.get("environment_stage"):
                    # Progress counters belong to one stage instance.
                    values.setdefault("environment_progress", None)
            self._state.update(values)
            self._write_locked()

    def _loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._write()

    def _write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        write_json_atomic(
            self.path,
            {
                **self._state,
                "schema_version": HITL_STATE_SCHEMA_VERSION,
                "heartbeat_at": _now(),
            },
        )


@dataclass(frozen=True)
class PlannedSession:
    """One entry of the plan of record: the research session or the forward replay."""

    session_key: SessionKind
    kind: SessionKind


def planned_sessions() -> tuple[PlannedSession, ...]:
    """The experiment's plan: its one research session, then the forward replay."""

    return (
        PlannedSession(SessionKind.RESEARCH, SessionKind.RESEARCH),
        PlannedSession(SessionKind.FORWARD, SessionKind.FORWARD),
    )


def build_session_plan(*, forward: Mapping[str, object]) -> dict[str, object]:
    """The plan of record (``schedule.json``): the research session and the
    forward replay with its span (``forward`` as the pipeline states it)."""

    plan: list[dict[str, object]] = [
        {"session_key": session.session_key, "kind": session.kind}
        for session in planned_sessions()
    ]
    plan[-1] = {**plan[-1], "replay": dict(forward)}
    return {
        "schema_version": HITL_STATE_SCHEMA_VERSION,
        "sessions": plan,
    }


def proc_start_ticks(pid: int) -> int | None:
    """Return the kernel start tick that distinguishes recycled process IDs."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(
            encoding="ascii",
            errors="replace",
        )
        return int(stat.rpartition(")")[2].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def assert_no_live_writer(experiment_dir: str | Path) -> None:
    """Rolling-upgrade write isolation for maintenance rewrites.

    A running experiment worker keeps its launch-time code in memory and keeps
    writing that code's formats regardless of what is deployed or migrated on
    disk: rewriting experiment-owned files (ledger, hitl state) under it loses
    in-flight appends and reintroduces the pre-migration format afterwards.
    Every migration/maintenance rewrite must call this first and stop the
    worker if it raises. This is process coexistence, not format compatibility.
    """
    status_path = Path(experiment_dir) / HITL_DIR_NAME / STATUS_NAME
    if not status_path.exists():
        return
    status = read_status(status_path)
    if status_pid_alive(status):
        raise RuntimeError(
            f"experiment {Path(experiment_dir).name!r} has a live worker "
            f"(pid {status.get('pid')}); stop it before rewriting experiment-owned files"
        )


def status_pid_alive(status: Mapping[str, object]) -> bool:
    pid = status.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(
            encoding="ascii",
            errors="replace",
        )
        if stat.rpartition(")")[2].split()[:1] == ["Z"]:
            return False
    except OSError:
        pass
    recorded_ticks = status.get("pid_start_ticks")
    return (
        isinstance(recorded_ticks, int)
        and not isinstance(recorded_ticks, bool)
        and proc_start_ticks(pid) == recorded_ticks
    )


def _require_version(payload: dict[str, object], path: Path) -> None:
    if payload and (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != HITL_STATE_SCHEMA_VERSION
    ):
        raise ValueError(f"incompatible HITL state in {path}")


def _string_map(value: object) -> dict[str, str]:
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def _int_map(value: object) -> dict[str, int]:
    """Non-negative integer control map (the per-session GPU count).

    A GPU count of 0 is a real request -- the CPU-only session the console offers
    -- and must survive the round trip, or the worker silently runs the
    session on the experiment default.
    """

    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[str(key)] = item
    return result


def _object_map(value: object) -> dict[str, dict[str, object]]:
    return (
        {str(k): dict(v) for k, v in value.items() if isinstance(v, dict)}
        if isinstance(value, dict)
        else {}
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
