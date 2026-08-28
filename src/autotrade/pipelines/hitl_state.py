"""Persistent HITL control, status, and deterministic session plans."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm.model_profiles import MODEL_CHOICES
from autotrade.environment.sandbox import SandboxSpec

from .folds import FoldSpec
from .meta_schedule import meta_learning_trigger_counts, meta_session_key

HITL_STATE_SCHEMA_VERSION = 1
CONTROL_MODES = ("auto", "manual", "step")
CONTROL_REQUESTS = (None, "pause", "stop")
HITL_DIR_NAME = "hitl"
PARAMS_NAME = "params.json"
CONTROL_NAME = "control.json"
STATUS_NAME = "status.json"
SCHEDULE_NAME = "schedule.json"
ANALYSIS_DIR_NAME = "analysis"
HELDOUT_SESSION_KEY = "heldout"
LIVE_RUN_STATES = {"running_session", "waiting_step_user", "waiting_user_reply"}

# The persistent WebUI creation contract.  The form and manager both read
# these defaults, while the worker retains its broader file-based/CLI contract.
# Paths and capability-bearing switches are intentionally absent: they are
# console-managed values below and can never be supplied by an HTTP client.
WEB_CREATE_DEFAULTS: dict[str, object] = {
    "experiment_id": None,
    "fold_period": "quarter",
    "first_test_period": "2022Q1",
    "last_test_period": "2025Q4",
    "heldout_first_period": "2026Q1",
    "heldout_last_period": "2026Q2",
    "epochs": 3,
    "meta_learning_fold_interval": 2,
    "meta_memory_max_epochs": 3,
    "fold_exploration_directive": "",
    "workspace_reference": "",
    "inherit_from": "",
    "strategy_period": "day",
    "inference_time": "08:30",
    "initial_control_mode": "auto",
    "analysis_enabled": False,
    "analysis_model": MODEL_CHOICES[0],
    "analysis_max_tokens": 6000,
    "window_months": 21,
    "daily_window_months": None,
    "fundamentals_window_months": None,
    "events_window_months": None,
    "macro_window_months": None,
    "text_window_months": None,
    "intraday_trade_days": SnapshotConfig().intraday_trade_days,
    "include_fundamentals": True,
    "include_macro": True,
    "include_events": True,
    "include_text": True,
    "include_intraday": True,
    "fundamental_datasets": (),
    "macro_datasets": (),
    "events_datasets": (),
    "text_datasets": (),
    "screen_exclude_st": True,
    "screen_exclude_new_listed_days": 180,
    "screen_min_circ_mv_yi": None,
    "screen_max_circ_mv_yi": None,
    "screen_min_price": None,
    "screen_max_price": None,
    "screen_boards": ("main",),
    "min_region_trade_days": 2,
    "max_steps_per_fold": 10,
    "max_backtests_per_fold": 15,
    "max_llm_calls": 400,
    "session_max_attempts": 3,
    "max_fold_minutes": 240,
    "convergence_start_epoch": 3,
    "nl_failure_policy": "return_error_with_audit",
    "finalize_before_deadline_seconds": 300,
    "per_call_timeout_seconds": 3600,
    "disable_step_tree": False,
    "record_failed_attempts": True,
    "min_return": 0.0,
    "min_sharpe": 0.0,
    "max_drawdown": 0.25,
    "initial_cash": 1_000_000.0,
    "max_total_holdings": None,
    "max_single_name_weight": None,
    "commission_bps": BrokerProfile().commission_bps,
    "slippage_bps": BrokerProfile().slippage_bps,
    "model": MODEL_CHOICES[0],
    "meta_model": MODEL_CHOICES[0],
    "nl_model": MODEL_CHOICES[0],
    "compact_model": MODEL_CHOICES[0],
    "reasoning_effort": "xhigh",
    "no_thinking": False,
    "disable_context_compact": False,
    "compact_token_threshold": 200_000,
    "compact_keep_recent_messages": 10,
    "compact_max_tokens": 10_000,
    "compact_max_calls": 10,
    "max_intraday_row_group_rows": 2_000_000,
    "gpu_count": SandboxSpec().gpu_count,
    "disable_meta_sandbox_rebuild": False,
    "meta_sandbox_rebuild_timeout_seconds": 1800,
    "meta_sandbox_image_keep": 3,
}

# These values describe the only supported WebUI research environment.  The
# manager stamps them into params.json and rejects attempts to submit them.
WEB_INTERNAL_PARAMS: dict[str, object] = {
    "strategy_path": "configs/agent_output_template/main.py",
    "data_backend": "pit",
    "raw_dir": "data/raw",
    "fundamental_events_root": "data/pit/fundamental_events",
    "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
    "execution_mode": "sandbox",
    "developer_mode": "llm",
}

WEB_CLOSED_PARAMS = frozenset(
    {
        *WEB_INTERNAL_PARAMS,
        "baseline_strategy_path",
        "daily_path",
        "pit_cache_root",
        "experiments_root",
        "work_root",
        "llm_api_key_env",
        "llm_env_file",
        "llm_base_url",
        "llm_timeout_seconds",
        "llm_max_retries",
        "llm_retry_backoff_seconds",
        "llm_model",
        "llm_temperature",
        "llm_max_response_tokens",
        "nl_max_results",
        "nl_max_calls_per_decision",
        "nl_max_total_calls",
        "nl_deadline_seconds",
        "agent_sandbox_image",
        "agent_sandbox_cpus",
        "agent_sandbox_memory",
        "agent_sandbox_pids",
        "agent_sandbox_tmpfs",
    }
)

WEB_REQUIRED_PARAMS = frozenset(
    {
        "experiment_id",
        "first_test_period",
        "last_test_period",
        "heldout_first_period",
        "heldout_last_period",
    }
)


@dataclass
class ControlState:
    mode: str = "manual"
    request: str | None = None
    approved_sessions: tuple[str, ...] = ()
    directives: dict[str, str] = field(default_factory=dict)
    prompt_overrides: dict[str, str] = field(default_factory=dict)
    skip_to_heldout: bool = False
    step_gate: dict[str, bool] = field(default_factory=dict)
    step_go: dict[str, int] = field(default_factory=dict)
    step_directives: dict[str, str] = field(default_factory=dict)
    user_replies: dict[str, str] = field(default_factory=dict)
    resource_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    # Per-fold sandbox GPU allocation set at the approval gate; the sandbox's
    # "auto" selector still picks which devices by free memory at start.
    gpu_counts: dict[str, int] = field(default_factory=dict)
    # A pending re-run token per fold session: the worker re-runs the session
    # whose latest ledger record has not absorbed this id yet.
    rerun_sessions: dict[str, str] = field(default_factory=dict)
    # Step-tree node that replaces the inherited frozen chain as one fold
    # session's parent (user-side step rollback).
    parent_overrides: dict[str, str] = field(default_factory=dict)
    test_revealed: bool = False

    def to_record(self) -> dict[str, object]:
        if self.mode not in CONTROL_MODES:
            raise ValueError(f"invalid HITL mode: {self.mode}")
        if self.request not in (None, "pause", "stop"):
            raise ValueError(f"invalid HITL request: {self.request}")
        return {
            "schema_version": HITL_STATE_SCHEMA_VERSION,
            "mode": self.mode,
            "request": self.request,
            "approved_sessions": sorted(set(self.approved_sessions)),
            "directives": dict(self.directives),
            "prompt_overrides": dict(self.prompt_overrides),
            "skip_to_heldout": self.skip_to_heldout,
            "step_gate": dict(self.step_gate),
            "step_go": dict(self.step_go),
            "step_directives": dict(self.step_directives),
            "user_replies": dict(self.user_replies),
            "resource_overrides": dict(self.resource_overrides),
            "gpu_counts": dict(self.gpu_counts),
            "rerun_sessions": dict(self.rerun_sessions),
            "parent_overrides": dict(self.parent_overrides),
            "test_revealed": self.test_revealed,
            "updated_at": _now(),
        }


def read_control(path: str | Path) -> ControlState:
    payload = read_json(path)
    _require_version(payload, Path(path))
    if not payload:
        return ControlState()
    mode = str(payload.get("mode") or "manual")
    if mode not in CONTROL_MODES:
        mode = "manual"
    request = payload.get("request")
    raw_approved = payload.get("approved_sessions")
    approved_sessions = raw_approved if isinstance(raw_approved, list) else []
    return ControlState(
        mode=mode,
        request=str(request) if request in ("pause", "stop") else None,
        approved_sessions=tuple(
            str(item) for item in approved_sessions if isinstance(item, str)
        ),
        directives=_string_map(payload.get("directives")),
        prompt_overrides=_string_map(payload.get("prompt_overrides")),
        skip_to_heldout=bool(payload.get("skip_to_heldout")),
        step_gate=_bool_map(payload.get("step_gate")),
        step_go=_positive_int_map(payload.get("step_go")),
        step_directives=_string_map(payload.get("step_directives")),
        user_replies=_string_map(payload.get("user_replies")),
        resource_overrides=_object_map(payload.get("resource_overrides")),
        gpu_counts=_positive_int_map(payload.get("gpu_counts")),
        rerun_sessions=_string_map(payload.get("rerun_sessions")),
        parent_overrides=_string_map(payload.get("parent_overrides")),
        test_revealed=bool(payload.get("test_revealed")),
    )


def write_control(path: str | Path, state: ControlState) -> None:
    _write_json(Path(path), state.to_record())


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
        state.approved_sessions = tuple(
            item for item in state.approved_sessions if item != session_key
        )
        for mapping in (
            state.directives,
            state.prompt_overrides,
            state.step_gate,
            state.step_go,
            state.step_directives,
            state.user_replies,
            state.resource_overrides,
            state.gpu_counts,
        ):
            for key in list(mapping):
                if key == session_key or key.startswith(f"{session_key}#"):
                    mapping.pop(key, None)
        write_control(path, state)
        return state


def consume_step_approval(
    path: str | Path,
    session_key: str,
    step_index: int,
) -> tuple[bool, str]:
    """Consume the one-shot directive for an approved current Step."""

    directive_key = f"{session_key}#{step_index}"
    with control_lock(path):
        state = read_control(path)
        if state.step_go.get(session_key, 0) < step_index:
            return False, ""
        state.step_go.pop(session_key, None)
        directive = state.step_directives.pop(directive_key, "")
        write_control(path, state)
        return True, directive


def consume_user_reply(path: str | Path, question_key: str) -> tuple[bool, str]:
    """Consume one exact Agent-question reply, including an empty reply."""

    with control_lock(path):
        state = read_control(path)
        if question_key not in state.user_replies:
            return False, ""
        reply = state.user_replies.pop(question_key)
        write_control(path, state)
        return True, reply


def read_json(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {target}")
    return payload


def read_status(path: str | Path) -> dict[str, object]:
    payload = read_json(path)
    _require_version(payload, Path(path))
    return payload


class StatusReporter:
    def __init__(self, path: str | Path, *, heartbeat_seconds: float = 3.0) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.path = Path(path)
        self.heartbeat_seconds = heartbeat_seconds
        pid = os.getpid()
        self._state: dict[str, object] = {
            "state": "initializing",
            "pid": pid,
            "pid_start_ticks": proc_start_ticks(pid),
            "session_key": None,
            "run_id": None,
            "session_started_at": None,
            "researcher_wait_seconds": 0.0,
            "wait_started_at": None,
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
                if stage != self._state.get("environment_stage"):
                    values.setdefault(
                        "environment_stage_started_at",
                        _now() if stage else None,
                    )
                    values.setdefault("environment_progress", None)
            if "state" in values:
                state = values["state"]
                if state != "waiting_step_user":
                    self._state.pop("step_index", None)
                    self._state.pop("step_summary", None)
                if state != "waiting_user_reply":
                    self._state.pop("question_key", None)
                    self._state.pop("question", None)
                    self._state.pop("question_summary", None)
            self._state.update(values)
            self._write_locked()

    def _loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._write()

    def _write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        _write_json(
            self.path,
            {
                **self._state,
                "schema_version": HITL_STATE_SCHEMA_VERSION,
                "heartbeat_at": _now(),
            },
        )


def fold_session_key(epoch_id: str, fold_id: str) -> str:
    return f"{epoch_id}/{fold_id}"


def epoch_ids(epochs: int) -> list[str]:
    return [f"epoch_{index:03d}" for index in range(1, epochs + 1)]


@dataclass(frozen=True)
class DevelopmentSession:
    session_key: str
    kind: str
    epoch_id: str
    fold: FoldSpec | None
    fold_index: int = 0


def iter_development_sessions(
    epochs: int,
    folds: list[FoldSpec],
    *,
    meta_enabled: bool,
    meta_learning_fold_interval: int = 0,
) -> tuple[DevelopmentSession, ...]:
    result: list[DevelopmentSession] = []
    for epoch_id in epoch_ids(epochs):
        triggers = (
            set(meta_learning_trigger_counts(len(folds), meta_learning_fold_interval))
            if meta_enabled
            else set()
        )
        for fold_index, fold in enumerate(folds):
            if fold_index in triggers:
                result.append(
                    DevelopmentSession(
                        meta_session_key(epoch_id, fold_index),
                        "meta",
                        epoch_id,
                        fold,
                        fold_index,
                    )
                )
            result.append(
                DevelopmentSession(
                    fold_session_key(epoch_id, fold.fold_id),
                    "fold",
                    epoch_id,
                    fold,
                    fold_index,
                )
            )
    return tuple(result)


def build_session_plan(
    epochs: int,
    folds: list[FoldSpec],
    heldout: list[Mapping[str, object]],
    *,
    meta_enabled: bool,
    meta_learning_fold_interval: int = 0,
) -> dict[str, object]:
    sessions = iter_development_sessions(
        epochs,
        folds,
        meta_enabled=meta_enabled,
        meta_learning_fold_interval=meta_learning_fold_interval,
    )
    plan: list[dict[str, object]] = [
        {
            "session_key": session.session_key,
            "kind": session.kind,
            "epoch_id": session.epoch_id,
            "fold_id": session.fold.fold_id if session.fold else None,
            "fold_index": session.fold_index,
        }
        for session in sessions
    ]
    plan.append(
        {
            "key": HELDOUT_SESSION_KEY,
            "kind": "heldout",
            "epoch_id": epoch_ids(epochs)[-1],
            "periods": [
                {
                    "label": period["label"],
                    "start": period["start"],
                    "end": period["end"],
                }
                for period in heldout
            ],
        }
    )
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


def assert_node_not_from_later_fold(
    node: dict[str, object],
    session_key: str,
    fold_keys: list[str],
    *,
    ref_store: AgentRefStore,
) -> None:
    """Reject a step-tree parent override recorded by a LATER fold session.

    Shared by the console manager (at set time) and the worker (at consume
    time): a node validated on a later period embodies future-fitted strategy
    content, so both writers of the decision must enforce the same wall."""
    try:
        node_fold = ref_store.resolve("fold", str(node.get("fold_id") or ""))
    except (KeyError, ValueError):
        node_fold = None
    node_key = f"{node.get('epoch_id')}/{node_fold}" if node_fold else None
    if node_key not in fold_keys:
        raise ValueError(
            f"cannot locate the fold session of step node {node.get('node_id')!r}; refusing it as a parent"
        )
    if session_key not in fold_keys:
        raise ValueError(f"{session_key!r} is not a fold session")
    if fold_keys.index(node_key) > fold_keys.index(session_key):
        raise ValueError(
            "不能把更晚 Fold 会话的节点设为更早会话的起点（未来验证信息泄漏）"
        )


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _string_map(value: object) -> dict[str, str]:
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def _bool_map(value: object) -> dict[str, bool]:
    return (
        {str(k): bool(v) for k, v in value.items()} if isinstance(value, dict) else {}
    )


def _positive_int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, int) and not isinstance(item, bool) and item > 0:
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
