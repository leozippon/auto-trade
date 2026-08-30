"""Meta-only projections of completed Fold strategies and Agent traces."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import sanitize_for_log
from autotrade.pipelines.agent_views import agent_visible_metrics
from autotrade.pipelines.fold_analysis import read_strategy_files

_AGENT_TRACE_EVENT_TYPES = frozenset(
    {
        "subagent",
        "subagent_attempt",
        "subagent_llm",
        "subagent_task",
        "subagent_tool",
        "llm_call",
        "session_end",
        "session_start",
        "tool_call",
        "trace_limit_reached",
        "user_message",
        "wrap_up_started",
    }
)
_SUMMARY_FAILURE_LIMIT = 8
_SUMMARY_ERROR_CHARS = 80
_AGENT_TRACE_MAX_EVENTS = 80
_SUMMARY_CHARS = 400
_ARG_CHARS = 120
_BODY_KEYS = frozenset(
    {
        "body",
        "content",
        "description",
        "input",
        "new_text",
        "old_text",
        "source",
        "task",
        "text",
    }
)
_DROP_EVENT_KEYS = frozenset(
    {
        "instruction",
        "system_prompt",
    }
)
_LEAK_KEYS = frozenset(
    {
        "daily_returns",
        "heldout",
        "per_stock",
        "weekly_returns",
    }
)
_COMPLETED_FOLD_STATUSES = frozenset(
    {
        "baseline_missing",
        "frozen",
        "no_update",
        "no_valid_backtest",
    }
)
_SANDBOX_PREFIXES = ("/mnt/agent", "/mnt/artifacts")
# Embedded Unix absolute host paths; keep /mnt sandbox mounts and a/b division.
_HOST_PATH_RE = re.compile(
    r"(?<![\w./])/(?!mnt\b)[A-Za-z_.][\w.-]*(?:/[\w.-]*)*"
)
AGENT_TRACE_FULL_MAX_FILE_BYTES = 8 * 1024 * 1024
AGENT_TRACE_FULL_MAX_WINDOW_BYTES = 16 * 1024 * 1024
AGENT_TRACE_FULL_RELATIVE_DIR = "inputs/agent_traces"


class AgentTraceSourceError(ValueError):
    """Fold Agent Trace source is required but missing or corrupt."""


@dataclass(frozen=True)
class AgentTraceFullSidecar:
    """Internal byte-exact raw AgentTraceWriter JSONL for one review Fold."""

    fold_ref: str
    relative_path: str
    events: int
    bytes: int
    source_truncated: bool
    available: bool
    payload: bytes | None

    def metadata(self) -> dict[str, object]:
        return {
            "path": self.relative_path if self.available else None,
            "events": self.events,
            "bytes": self.bytes,
            "source_truncated": self.source_truncated,
            "available": self.available,
            "raw_jsonl": True,
            "byte_exact": True,
        }


def select_meta_review_folds(
    records: Sequence[Mapping[str, object]],
    *,
    ref_store: AgentRefStore,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Completed regular Folds after the latest ``meta_learning`` ledger record.

    The current Meta record is not yet appended, so the last ``meta_learning``
    row is the previous Meta. No previous Meta yields an empty window. Held-out,
    ``attempt_failed``, and in-progress rows are excluded. Duplicate Fold ids
    keep the latest record in window order.
    """

    last_meta_index = -1
    last_meta: Mapping[str, object] | None = None
    for index, record in enumerate(records):
        if record.get("record_type") == "meta_learning":
            last_meta_index = index
            last_meta = record
    window_source: Sequence[Mapping[str, object]] = (
        records[last_meta_index + 1 :] if last_meta is not None else ()
    )
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for record in window_source:
        if record.get("record_type") != "fold":
            continue
        if str(record.get("fold_status") or "") not in _COMPLETED_FOLD_STATUSES:
            continue
        key = (str(record.get("epoch_id") or ""), str(record.get("fold_id") or ""))
        latest[key] = dict(record)
    folds = list(latest.values())
    identity = ""
    if last_meta is not None:
        identity = str(
            last_meta.get("meta_learning_id") or last_meta.get("run_id") or ""
        )
    return folds, {
        "previous_meta_ref": (
            ref_store.get_or_create("meta", identity) if identity else None
        ),
        "fold_run_refs": [
            ref_store.get_or_create("run", str(record["run_id"]))
            for record in folds
            if record.get("run_id")
        ],
        "fold_count": len(folds),
    }


def compact_agent_trace(
    events: Sequence[Mapping[str, object]],
    *,
    max_events: int = _AGENT_TRACE_MAX_EVENTS,
) -> list[dict[str, object]]:
    """Bounded recent main-session and sub-agent events for Meta."""

    selected = [
        event
        for event in events
        if str(event.get("event_type") or "") in _AGENT_TRACE_EVENT_TYPES
    ]
    return [
        _compact_agent_event(event)
        for event in _recent_complete_trace_groups(selected, max_events=max_events)
    ]


def write_meta_agent_trace_sidecars(
    workspace: str | Path,
    sidecars: Sequence[AgentTraceFullSidecar],
) -> None:
    """Write available byte-exact raw JSONL under ``inputs/agent_traces/``."""

    root = Path(workspace) / "inputs" / "agent_traces"
    root.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    for sidecar in sidecars:
        if not sidecar.available or sidecar.payload is None:
            continue
        dest = (Path(workspace) / sidecar.relative_path).resolve()
        if dest.parent != root.resolve():
            raise ValueError("full agent-trace sidecar escaped inputs/agent_traces")
        if dest.name in written or dest.exists():
            raise ValueError("duplicate opaque fold_ref for agent-trace sidecar")
        temp = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with temp.open("xb") as handle:
                handle.write(sidecar.payload)
                handle.flush()
                os.fsync(handle.fileno())
            temp.chmod(0o444)
            temp.replace(dest)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        written.add(dest.name)


def _recent_complete_trace_groups(
    events: Sequence[Mapping[str, object]],
    *,
    max_events: int,
) -> list[Mapping[str, object]]:
    """Keep the newest complete task_id groups that fit in max_events.

    Events without ``task_id`` are each their own group. Selection walks from
    the tail so later work stays visible; events inside a kept group stay in
    their original order. If the newest group alone exceeds the bound, keep
    its trailing events.
    """
    if max_events <= 0 or not events:
        return []
    keys: list[str] = []
    key_of: list[str] = []
    counts: dict[str, int] = {}
    for index, event in enumerate(events):
        task_id = str(event.get("task_id") or "").strip()
        key = task_id or f"#{index}"
        key_of.append(key)
        if key not in counts:
            keys.append(key)
            counts[key] = 0
        counts[key] += 1
    chosen: set[str] = set()
    used = 0
    for key in reversed(keys):
        size = counts[key]
        if used + size <= max_events:
            chosen.add(key)
            used += size
            continue
        if not chosen:
            trailing: list[Mapping[str, object]] = []
            for event, event_key in zip(reversed(events), reversed(key_of)):
                if event_key != key:
                    continue
                trailing.append(event)
                if len(trailing) >= max_events:
                    break
            trailing.reverse()
            return trailing
        break
    return [event for event, key in zip(events, key_of) if key in chosen]


def _compact_agent_event(event: Mapping[str, object]) -> dict[str, object]:
    event_type = str(event.get("event_type") or "")
    item: dict[str, object] = {
        "event_type": event_type,
        "task_id": event.get("task_id"),
        "parent_call_id": event.get("parent_call_id"),
        "round": event.get("round"),
        "status": event.get("status"),
        "tool": event.get("tool"),
        "tool_names": event.get("tool_names"),
        "tool_call_id": event.get("tool_call_id"),
        "call_index": event.get("call_index"),
        "model": event.get("model"),
    }
    if event_type in {"subagent_task", "subagent"}:
        if event.get("role"):
            item["role"] = event.get("role")
        if event.get("summary"):
            item["summary"] = _redact_text(event.get("summary"), limit=_SUMMARY_CHARS)
        for key in ("rounds", "tool_calls", "llm_calls"):
            if event.get(key) is not None:
                item[key] = event.get(key)
    elif event_type == "subagent_llm":
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            item["usage"] = {
                key: usage.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in usage
            }
    elif event_type == "subagent_tool":
        _attach_result_status(item, event.get("result"), event)
    elif event_type == "subagent_attempt":
        if event.get("attempt") is not None:
            item["attempt"] = event.get("attempt")
        if event.get("role"):
            item["role"] = event.get("role")
        if event.get("ok") is not None:
            item["ok"] = event.get("ok")
    elif event_type == "llm_call":
        if event.get("content"):
            item["content"] = _redact_text(event.get("content"), limit=_SUMMARY_CHARS)
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            item["usage"] = {
                key: usage.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in usage
            }
    elif event_type == "tool_call":
        args = _summarize_args(event.get("arguments"))
        if args:
            item["args"] = args
        _attach_result_status(item, event.get("result"), event)
    elif event_type == "user_message":
        if event.get("content"):
            item["content"] = _redact_text(event.get("content"), limit=_SUMMARY_CHARS)
        if event.get("interrupt") is not None:
            item["interrupt"] = event.get("interrupt")
        if event.get("safe_point"):
            item["safe_point"] = event.get("safe_point")
    elif event_type == "session_start":
        item["mode"] = event.get("mode")
    elif event_type == "session_end":
        if event.get("llm_calls") is not None:
            item["llm_calls"] = event.get("llm_calls")
        if event.get("subagent_attempts") is not None:
            item["subagent_attempts"] = event.get("subagent_attempts")
        if event.get("subagent_roles"):
            item["subagent_roles"] = event.get("subagent_roles")
    elif event_type == "wrap_up_started":
        if event.get("remaining_seconds") is not None:
            item["remaining_seconds"] = event.get("remaining_seconds")
        if event.get("grace_seconds") is not None:
            item["grace_seconds"] = event.get("grace_seconds")
    elif event_type == "trace_limit_reached":
        if event.get("max_bytes") is not None:
            item["max_bytes"] = event.get("max_bytes")
    if event.get("error") and "error" not in item:
        item["error"] = _redact_text(event.get("error"), limit=_SUMMARY_CHARS)
    compacted = {
        key: value
        for key, value in item.items()
        if key not in _DROP_EVENT_KEYS and value not in (None, "", [])
    }
    sanitized = sanitize_for_log(compacted)
    return sanitized if isinstance(sanitized, dict) else compacted


def _attach_result_status(
    item: dict[str, object], result: object, event: Mapping[str, object]
) -> None:
    if isinstance(result, Mapping):
        item["ok"] = result.get("ok")
        if result.get("error"):
            item["error"] = _redact_text(result.get("error"), limit=_SUMMARY_CHARS)
        if result.get("status") not in (None, ""):
            item.setdefault("status", result.get("status"))
    elif event.get("error"):
        item["error"] = _redact_text(event.get("error"), limit=_SUMMARY_CHARS)


def _summarize_args(arguments: object) -> dict[str, object] | None:
    if not isinstance(arguments, Mapping):
        return None
    summary: dict[str, object] = {}
    for key, value in arguments.items():
        name = str(key)
        if name in _LEAK_KEYS or name in _DROP_EVENT_KEYS:
            continue
        if name in _BODY_KEYS:
            summary[name] = {"omitted": True, "chars": len(str(value))}
            continue
        if name == "argv" and isinstance(value, list):
            summary[name] = [
                _redact_text(item, limit=_ARG_CHARS) for item in value[:12]
            ]
            continue
        if isinstance(value, bool) or isinstance(value, (int, float)):
            summary[name] = value
        elif isinstance(value, str):
            summary[name] = _redact_text(value, limit=_ARG_CHARS)
    return summary or None


def _redact_host_paths(value: object) -> str:
    text = str(value or "")
    stripped = text.strip()
    if stripped.startswith("/") and not stripped.startswith(_SANDBOX_PREFIXES):
        return "[host_path]"
    return _HOST_PATH_RE.sub("[host_path]", text)


def _redact_text(value: object, *, limit: int) -> str:
    return _redact_host_paths(value)[:limit]


def build_agent_process_summary(
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bounded deterministic process counts from a compact Agent Trace.

    Counts only; no task body, paths, Test, or Held-out values.
    """
    llm_calls = 0
    subagent_attempts = 0
    subagent_completed = 0
    subagent_failed = 0
    daily_backtest = 0
    tool_failures = 0
    failure_counts: dict[tuple[str, str], int] = {}

    def add_failure(tool: str, error: object) -> None:
        nonlocal tool_failures
        tool_failures += 1
        key = (tool or "unknown", _redact_text(error, limit=_SUMMARY_ERROR_CHARS))
        failure_counts[key] = failure_counts.get(key, 0) + 1

    session_end_attempts: int | None = None
    session_end_llm: int | None = None
    attempt_events = 0
    task_events = 0
    tool_agent = 0
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event_type == "llm_call":
            llm_calls += 1
        elif event_type == "session_end":
            raw_calls = event.get("llm_calls")
            if isinstance(raw_calls, int) and not isinstance(raw_calls, bool):
                session_end_llm = raw_calls
            raw_attempts = event.get("subagent_attempts")
            if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool):
                session_end_attempts = raw_attempts
        elif event_type == "subagent_attempt":
            attempt_events += 1
            if event.get("ok") == False and not event.get("status"):
                subagent_failed += 1
        elif event_type == "subagent_task":
            task_events += 1
        elif event_type == "subagent":
            status = str(event.get("status") or "")
            if status == "completed":
                subagent_completed += 1
            elif status in {"error", "timeout"}:
                subagent_failed += 1
        elif event_type == "tool_call":
            tool = str(event.get("tool") or "")
            if tool == "agent":
                tool_agent += 1
            elif tool == "daily_backtest":
                daily_backtest += 1
            if event.get("ok") == False:
                add_failure(tool, event.get("error"))
        elif event_type == "subagent_tool" and event.get("ok") == False:
            add_failure(str(event.get("tool") or "agent"), event.get("error"))

    if session_end_llm is not None:
        llm_calls = session_end_llm
    if session_end_attempts is not None:
        subagent_attempts = session_end_attempts
    elif attempt_events:
        subagent_attempts = attempt_events
    elif task_events:
        subagent_attempts = task_events
    else:
        subagent_attempts = tool_agent
    repeated = [
        {"tool": tool, "count": count, "error": error}
        for (tool, error), count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:_SUMMARY_FAILURE_LIMIT]
        if count > 1
    ]
    return {
        "llm_calls": llm_calls,
        "subagent": {
            "attempts": subagent_attempts,
            "completed": subagent_completed,
            "failed": subagent_failed,
        },
        "tool_failures": tool_failures,
        "daily_backtest": daily_backtest,
        "repeated_failures": repeated,
    }


def build_meta_fold_review_bundle(
    records: Sequence[Mapping[str, object]],
    *,
    ref_store: AgentRefStore,
    artifacts_root: str | Path | None = None,
    max_file_bytes: int = AGENT_TRACE_FULL_MAX_FILE_BYTES,
    max_window_bytes: int = AGENT_TRACE_FULL_MAX_WINDOW_BYTES,
) -> tuple[list[dict[str, object]], list[AgentTraceFullSidecar]]:
    """Public Fold indexes plus internal byte-exact raw trace sidecars."""

    reviews: list[dict[str, object]] = []
    sidecars: list[AgentTraceFullSidecar] = []
    fold_count = 0
    for record in records:
        if record.get("record_type") not in {None, "fold"}:
            continue
        fold_count += 1
        path = record.get("frozen_strategy_artifact_path")
        artifact_id = record.get("frozen_strategy_artifact_id")
        validation = record.get("validation_result")
        test_result = record.get("test_result")
        strategy_files: list[dict[str, object]] = []
        if isinstance(path, str) and path:
            strategy_dir = Path(path)
            if strategy_dir.is_dir():
                strategy_files = read_strategy_files(strategy_dir)
        source_events, source_payload, available, source_truncated = (
            load_fold_agent_trace_source(record, artifacts_root=artifacts_root)
        )
        agent_trace = compact_agent_trace(source_events)
        fold_ref = ref_store.get_or_create("fold", str(record.get("fold_id")))
        sidecar_ref = ref_store.get_or_create(
            "trace", f"{record.get('epoch_id')}:{record.get('fold_id')}"
        )
        sidecar = _build_full_sidecar(
            fold_ref=fold_ref,
            sidecar_ref=sidecar_ref,
            source_events=source_events,
            source_payload=source_payload,
            available=available,
            source_truncated=source_truncated,
        )
        reviews.append(
            {
                "epoch_id": record.get("epoch_id"),
                "fold_id": fold_ref,
                "fold_status": record.get("fold_status"),
                "frozen_strategy_artifact_id": (
                    ref_store.get_or_create("strategy", str(artifact_id))
                    if artifact_id
                    else None
                ),
                "validation_result": agent_visible_metrics(
                    validation if isinstance(validation, dict) else None
                ),
                "test_result": agent_visible_metrics(
                    test_result if isinstance(test_result, dict) else None
                ),
                "strategy_files": strategy_files,
                "agent_trace": agent_trace,
                "agent_process_summary": build_agent_process_summary(agent_trace),
                "agent_trace_full": sidecar.metadata(),
            }
        )
        sidecars.append(sidecar)
    _assert_sidecar_budget(
        sidecars,
        fold_count=fold_count,
        max_file_bytes=max_file_bytes,
        max_window_bytes=max_window_bytes,
    )
    return reviews, sidecars


def _build_full_sidecar(
    *,
    fold_ref: str,
    sidecar_ref: str,
    source_events: Sequence[Mapping[str, object]],
    source_payload: bytes | None,
    available: bool,
    source_truncated: bool,
) -> AgentTraceFullSidecar:
    relative_path = f"{AGENT_TRACE_FULL_RELATIVE_DIR}/{sidecar_ref}.jsonl"
    if not available:
        return AgentTraceFullSidecar(
            fold_ref=fold_ref,
            relative_path=relative_path,
            events=0,
            bytes=0,
            source_truncated=False,
            available=False,
            payload=None,
        )
    if source_payload is None:
        raise AgentTraceSourceError("available agent_trace_ref has no source bytes")
    return AgentTraceFullSidecar(
        fold_ref=fold_ref,
        relative_path=relative_path,
        events=len(source_events),
        bytes=len(source_payload),
        source_truncated=source_truncated,
        available=True,
        payload=source_payload,
    )


def _assert_sidecar_budget(
    sidecars: Sequence[AgentTraceFullSidecar],
    *,
    fold_count: int,
    max_file_bytes: int,
    max_window_bytes: int,
) -> None:
    written = [sidecar for sidecar in sidecars if sidecar.available]
    if len(written) > fold_count:
        raise ValueError(
            "full agent-trace sidecar count "
            f"{len(written)} exceeds review fold_count {fold_count}"
        )
    total = 0
    for sidecar in written:
        if sidecar.bytes > max_file_bytes:
            raise ValueError(
                f"full agent-trace sidecar exceeds {max_file_bytes} bytes"
            )
        total += sidecar.bytes
    if total > max_window_bytes:
        raise ValueError(
            f"full agent-trace sidecar window exceeds {max_window_bytes} bytes"
        )


def load_fold_agent_trace_source(
    record: Mapping[str, object],
    *,
    artifacts_root: str | Path | None = None,
) -> tuple[list[Mapping[str, object]], bytes | None, bool, bool]:
    """Validate one raw Fold Agent Trace while retaining its exact bytes."""

    ref = record.get("agent_trace_ref")
    if ref is None or ref == "":
        return [], None, False, False
    if not isinstance(ref, str):
        raise AgentTraceSourceError("agent_trace_ref must be a string path")
    path = _resolve_existing_trace_path(ref, artifacts_root=artifacts_root)
    try:
        raw_jsonl = path.read_bytes()
        lines = raw_jsonl.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AgentTraceSourceError("agent_trace_ref cannot be read as UTF-8") from exc
    events: list[Mapping[str, object]] = []
    source_truncated = False
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentTraceSourceError(
                f"agent_trace_ref has invalid JSON at line {index}"
            ) from exc
        if not isinstance(payload, dict):
            raise AgentTraceSourceError(
                f"agent_trace_ref line {index} is not a JSON object"
            )
        events.append(payload)
        if payload.get("event_type") == "trace_limit_reached":
            source_truncated = True
    return events, raw_jsonl, True, source_truncated


def _resolve_existing_trace_path(
    ref: str,
    *,
    artifacts_root: str | Path | None = None,
) -> Path:
    candidate = Path(ref)
    if candidate.is_file():
        return candidate
    if artifacts_root is not None:
        nested = Path(artifacts_root) / ref
        if nested.is_file():
            return nested
    raise AgentTraceSourceError("agent_trace_ref is missing on disk")
