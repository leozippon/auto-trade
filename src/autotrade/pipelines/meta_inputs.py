"""Meta-only projections of completed Fold strategies and Agent traces."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from autotrade.environment.identity import agent_visible_ref
from autotrade.environment.runtime import agent_trace_path, sanitize_for_log
from autotrade.pipelines.agent_views import agent_visible_metrics
from autotrade.pipelines.fold_analysis import read_strategy_files

_AGENT_TRACE_EVENT_TYPES = frozenset(
    {
        "explore",
        "explore_llm",
        "explore_task",
        "explore_tool",
        "llm_call",
        "session_end",
        "session_start",
        "tool_call",
        "trace_limit_reached",
        "wrap_up_started",
    }
)
_AGENT_TRACE_MAX_EVENTS = 80
_TASK_CHARS = 500
_DIGEST_CHARS = 400
_SUMMARY_CHARS = 400
_ARG_CHARS = 120
_BODY_KEYS = frozenset(
    {
        "body",
        "content",
        "input",
        "new_text",
        "old_text",
        "source",
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
        item["task"] = _redact_text(event.get("task"), limit=_TASK_CHARS)
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
    elif event_type == "session_start":
        item["mode"] = event.get("mode")
    elif event_type == "session_end":
        if event.get("llm_calls") is not None:
            item["llm_calls"] = event.get("llm_calls")
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


def _redact_text(value: object, *, limit: int) -> str:
    text = str(value or "")
    stripped = text.strip()
    if stripped.startswith("/") and not stripped.startswith(_SANDBOX_PREFIXES):
        return "[host_path]"
    return _HOST_PATH_RE.sub("[host_path]", text)[:limit]


def build_meta_fold_reviews(
    records: Sequence[Mapping[str, object]],
    *,
    artifacts_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Per already-selected Fold: frozen artifact/ref, strategy source, Agent Trace."""

    reviews: list[dict[str, object]] = []
    for record in records:
        if record.get("record_type") not in {None, "fold"}:
            continue
        path = record.get("frozen_strategy_artifact_path")
        artifact_id = record.get("frozen_strategy_artifact_id")
        validation = record.get("validation_result")
        test_result = record.get("test_result")
        strategy_files: list[dict[str, object]] = []
        if isinstance(path, str) and path:
            strategy_dir = Path(path)
            if strategy_dir.is_dir():
                strategy_files = read_strategy_files(strategy_dir)
        reviews.append(
            {
                "epoch_id": record.get("epoch_id"),
                "fold_id": agent_visible_ref(record.get("fold_id"), prefix="fold_ref"),
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
                "agent_trace": compact_agent_trace(
                    _read_trace_events(record, artifacts_root=artifacts_root)
                ),
            }
        )
    return reviews


def _read_trace_events(
    record: Mapping[str, object],
    *,
    artifacts_root: str | Path | None = None,
) -> list[Mapping[str, object]]:
    path = _resolve_trace_path(record, artifacts_root=artifacts_root)
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[Mapping[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _resolve_trace_path(
    record: Mapping[str, object],
    *,
    artifacts_root: str | Path | None = None,
) -> Path | None:
    ref = record.get("agent_trace_ref")
    if isinstance(ref, str) and ref:
        candidate = Path(ref)
        if candidate.is_file():
            return candidate
        if artifacts_root is not None:
            nested = Path(artifacts_root) / ref
            if nested.is_file():
                return nested
    if artifacts_root is not None and record.get("run_id"):
        candidate = agent_trace_path(artifacts_root, str(record.get("run_id")))
        if candidate.is_file():
            return candidate
    return None
