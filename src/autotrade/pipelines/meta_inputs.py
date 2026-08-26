"""Meta-only projections of completed Fold strategies and Agent traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from autotrade.environment.identity import agent_visible_ref
from autotrade.environment.runtime import sanitize_for_log
from autotrade.pipelines.agent_views import agent_visible_metrics
from autotrade.pipelines.fold_analysis import read_strategy_files

_AGENT_TRACE_EVENT_TYPES = frozenset(
    {
        "explore",
        "explore_attempt",
        "explore_llm",
        "explore_task",
        "explore_tool",
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
_DIGEST_CHARS = 400
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
        "subject",
        "task",
        "text",
    }
)
_FULL_OMIT_KEYS = _BODY_KEYS | {
    "content_preview",
    "new",
    "old",
    "question",
    "reply",
}
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
_SUMMARY_CONTENT_EVENT_TYPES = frozenset({"explore", "explore_llm", "llm_call"})
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
AGENT_TRACE_FULL_CONTENT_CHARS = 8 * 1024
AGENT_TRACE_FULL_MAX_FILE_BYTES = 8 * 1024 * 1024
AGENT_TRACE_FULL_MAX_WINDOW_BYTES = 16 * 1024 * 1024
AGENT_TRACE_FULL_RELATIVE_DIR = "inputs/agent_traces"


class AgentTraceSourceError(ValueError):
    """Fold Agent Trace source is required but missing or corrupt."""


@dataclass(frozen=True)
class AgentTraceFullSidecar:
    """Internal full safe-projection payload for one review Fold."""

    fold_ref: str
    relative_path: str
    sha256: str
    events: int
    bytes: int
    source_truncated: bool
    available: bool
    payload: bytes | None

    def metadata(self) -> dict[str, object]:
        return {
            "path": self.relative_path if self.available else None,
            "sha256": self.sha256 or None,
            "events": self.events,
            "bytes": self.bytes,
            "source_truncated": self.source_truncated,
            "available": self.available,
        }


def select_meta_review_folds(
    records: Sequence[Mapping[str, object]],
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
            agent_visible_ref(identity, prefix="meta_ref") if identity else None
        ),
        "fold_run_refs": [
            agent_visible_ref(record.get("run_id"), prefix="run_ref")
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


def project_full_agent_trace(
    events: Sequence[Mapping[str, object]],
    *,
    content_chars: int = AGENT_TRACE_FULL_CONTENT_CHARS,
) -> list[dict[str, object]]:
    """Safe projection of every source event, in original order.

    Unknown event types are included. System prompts, leak keys, and tool or
    ask_user bodies are stripped; LLM/explore summary content is kept with a
    per-event character cap.
    """

    return [
        project_full_agent_event(event, content_chars=content_chars) for event in events
    ]


def project_full_agent_event(
    event: Mapping[str, object],
    *,
    content_chars: int = AGENT_TRACE_FULL_CONTENT_CHARS,
) -> dict[str, object]:
    event_type = str(event.get("event_type") or "")
    projected = _project_full_value(
        event,
        event_type=event_type,
        at_root=True,
        content_chars=content_chars,
    )
    if not isinstance(projected, dict):
        projected = {"event_type": event_type}
    sanitized = sanitize_for_log(projected)
    result: dict[str, object]
    if isinstance(sanitized, dict):
        result = {str(key): item for key, item in sanitized.items()}
    else:
        result = {str(key): item for key, item in projected.items()}
    return result


def serialize_full_agent_trace(events: Sequence[Mapping[str, object]]) -> bytes:
    """Canonical JSONL bytes for a full safe-projection sidecar."""

    lines = [
        json.dumps(
            dict(event),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        for event in events
    ]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    return text.encode("utf-8")


def write_meta_agent_trace_sidecars(
    workspace: str | Path,
    sidecars: Sequence[AgentTraceFullSidecar],
) -> None:
    """Write available full projections under ``inputs/agent_traces/``."""

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
    if event_type in {"explore_task", "explore"}:
        if event.get("role"):
            item["role"] = event.get("role")
        if event.get("digest"):
            item["digest"] = _redact_text(event.get("digest"), limit=_DIGEST_CHARS)
        for key in ("rounds", "tool_calls", "llm_calls"):
            if event.get(key) is not None:
                item[key] = event.get(key)
    elif event_type == "explore_llm":
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            item["usage"] = {
                key: usage.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in usage
            }
    elif event_type == "explore_tool":
        _attach_result_status(item, event.get("result"), event)
    elif event_type == "explore_attempt":
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
        if event.get("explore_attempts") is not None:
            item["explore_attempts"] = event.get("explore_attempts")
        if event.get("explored_roles"):
            item["explored_roles"] = event.get("explored_roles")
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


def _project_full_value(
    value: object,
    *,
    event_type: str,
    at_root: bool,
    content_chars: int,
) -> object:
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if name in _DROP_EVENT_KEYS or name in _LEAK_KEYS:
                continue
            if name == "content" and event_type == "user_message" and at_root:
                continue
            if (
                name in {"content", "digest"}
                and at_root
                and event_type in _SUMMARY_CONTENT_EVENT_TYPES
            ):
                projected[name] = _redact_text(item, limit=content_chars)
                continue
            if name == "summary" and event_type == "context_compaction":
                projected[name] = _redact_text(item, limit=content_chars)
                continue
            if name == "summary":
                projected[name] = {"omitted": True, "chars": len(str(item))}
                continue
            if name in _FULL_OMIT_KEYS:
                projected[name] = {"omitted": True, "chars": len(str(item))}
                continue
            projected[name] = _project_full_value(
                item,
                event_type=event_type,
                at_root=False,
                content_chars=content_chars,
            )
        return projected
    if isinstance(value, (list, tuple)):
        return [
            _project_full_value(
                item,
                event_type=event_type,
                at_root=False,
                content_chars=content_chars,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _redact_host_paths(value)
    return value


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
    explore_attempts = 0
    explore_completed = 0
    explore_failed = 0
    todo_calls = 0
    todo_completed = 0
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
    tool_explore = 0
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event_type == "llm_call":
            llm_calls += 1
        elif event_type == "session_end":
            raw_calls = event.get("llm_calls")
            if isinstance(raw_calls, int) and not isinstance(raw_calls, bool):
                session_end_llm = raw_calls
            raw_attempts = event.get("explore_attempts")
            if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool):
                session_end_attempts = raw_attempts
        elif event_type == "explore_attempt":
            attempt_events += 1
            if event.get("ok") == False and not event.get("status"):
                explore_failed += 1
        elif event_type == "explore_task":
            task_events += 1
        elif event_type == "explore":
            status = str(event.get("status") or "")
            if status == "completed":
                explore_completed += 1
            elif status in {"error", "timeout"}:
                explore_failed += 1
        elif event_type == "tool_call":
            tool = str(event.get("tool") or "")
            if tool == "explore":
                tool_explore += 1
            if tool == "todo":
                todo_calls += 1
                args = event.get("args")
                if isinstance(args, Mapping) and args.get("status") == "completed":
                    todo_completed += 1
            elif tool == "daily_backtest":
                daily_backtest += 1
            if event.get("ok") == False:
                add_failure(tool, event.get("error"))
        elif event_type == "explore_tool" and event.get("ok") == False:
            add_failure(str(event.get("tool") or "explore"), event.get("error"))

    if session_end_llm is not None:
        llm_calls = session_end_llm
    if session_end_attempts is not None:
        explore_attempts = session_end_attempts
    elif attempt_events:
        explore_attempts = attempt_events
    elif task_events:
        explore_attempts = task_events
    else:
        explore_attempts = tool_explore
    repeated = [
        {"tool": tool, "count": count, "error": error}
        for (tool, error), count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:_SUMMARY_FAILURE_LIMIT]
        if count > 1
    ]
    return {
        "llm_calls": llm_calls,
        "explore": {
            "attempts": explore_attempts,
            "completed": explore_completed,
            "failed": explore_failed,
        },
        "todo": {"calls": todo_calls, "completed": todo_completed},
        "tool_failures": tool_failures,
        "daily_backtest": daily_backtest,
        "repeated_failures": repeated,
    }


def build_meta_fold_reviews(
    records: Sequence[Mapping[str, object]],
    *,
    artifacts_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Per already-selected Fold: frozen artifact/ref, strategy source, Agent Trace."""

    reviews, _sidecars = build_meta_fold_review_bundle(
        records, artifacts_root=artifacts_root
    )
    return reviews


def build_meta_fold_review_bundle(
    records: Sequence[Mapping[str, object]],
    *,
    artifacts_root: str | Path | None = None,
    max_file_bytes: int = AGENT_TRACE_FULL_MAX_FILE_BYTES,
    max_window_bytes: int = AGENT_TRACE_FULL_MAX_WINDOW_BYTES,
    content_chars: int = AGENT_TRACE_FULL_CONTENT_CHARS,
) -> tuple[list[dict[str, object]], list[AgentTraceFullSidecar]]:
    """Public fold reviews plus internal full safe-projection sidecars."""

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
        source_events, available, source_truncated = load_fold_agent_trace_source(
            record, artifacts_root=artifacts_root
        )
        agent_trace = compact_agent_trace(source_events)
        fold_ref = agent_visible_ref(record.get("fold_id"), prefix="fold_ref")
        sidecar_ref = agent_visible_ref(
            f"{record.get('epoch_id')}:{record.get('fold_id')}",
            prefix="trace_ref",
        )
        sidecar = _build_full_sidecar(
            fold_ref=fold_ref,
            sidecar_ref=sidecar_ref,
            source_events=source_events,
            available=available,
            source_truncated=source_truncated,
            content_chars=content_chars,
        )
        reviews.append(
            {
                "epoch_id": record.get("epoch_id"),
                "fold_id": fold_ref,
                "fold_status": record.get("fold_status"),
                "frozen_strategy_artifact_id": (
                    agent_visible_ref(artifact_id, prefix="strategy_ref")
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
    available: bool,
    source_truncated: bool,
    content_chars: int,
) -> AgentTraceFullSidecar:
    relative_path = f"{AGENT_TRACE_FULL_RELATIVE_DIR}/{sidecar_ref}.jsonl"
    if not available:
        return AgentTraceFullSidecar(
            fold_ref=fold_ref,
            relative_path=relative_path,
            sha256="",
            events=0,
            bytes=0,
            source_truncated=False,
            available=False,
            payload=None,
        )
    projected = project_full_agent_trace(source_events, content_chars=content_chars)
    payload = serialize_full_agent_trace(projected)
    return AgentTraceFullSidecar(
        fold_ref=fold_ref,
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        events=len(projected),
        bytes=len(payload),
        source_truncated=source_truncated,
        available=True,
        payload=payload,
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
) -> tuple[list[Mapping[str, object]], bool, bool]:
    """Load one Fold Agent Trace. Missing ref means no Agent, not an error."""

    ref = record.get("agent_trace_ref")
    if ref is None or ref == "":
        return [], False, False
    if not isinstance(ref, str):
        raise AgentTraceSourceError("agent_trace_ref must be a string path")
    path = _resolve_existing_trace_path(ref, artifacts_root=artifacts_root)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
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
    return events, True, source_truncated


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
