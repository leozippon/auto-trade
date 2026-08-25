"""Meta-only projections of completed Fold strategies and Explore traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from autotrade.environment.identity import agent_visible_ref
from autotrade.environment.runtime import agent_trace_path
from autotrade.pipelines.agent_views import agent_visible_metrics
from autotrade.pipelines.fold_analysis import read_strategy_files

_EXPLORE_EVENT_TYPES = frozenset(
    {"explore_task", "explore_llm", "explore_tool", "explore"}
)
_EXPLORE_TRACE_MAX_EVENTS = 80
_EXPLORE_TASK_CHARS = 500
_EXPLORE_DIGEST_CHARS = 400


def compact_explore_trace(
    events: Sequence[Mapping[str, object]],
    *,
    max_events: int = _EXPLORE_TRACE_MAX_EVENTS,
) -> list[dict[str, object]]:
    """Bounded recent Explore sub-agent tasks for Meta — not tool counts only."""

    selected = [
        event
        for event in events
        if str(event.get("event_type") or "") in _EXPLORE_EVENT_TYPES
    ]
    return [
        _compact_explore_event(event)
        for event in _recent_complete_explore_tasks(selected, max_events=max_events)
    ]


def _recent_complete_explore_tasks(
    events: Sequence[Mapping[str, object]],
    *,
    max_events: int,
) -> list[Mapping[str, object]]:
    """Keep the newest complete task_id groups that fit in max_events.

    Tasks are ordered by first appearance. Selection walks from the tail so
    later sub-agent work stays visible; events inside a kept task stay in
    their original order. If the newest task alone exceeds the bound, keep
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


def _compact_explore_event(event: Mapping[str, object]) -> dict[str, object]:
    item: dict[str, object] = {
        "event_type": event.get("event_type"),
        "task_id": event.get("task_id"),
        "parent_call_id": event.get("parent_call_id"),
        "round": event.get("round"),
        "status": event.get("status"),
    }
    event_type = str(event.get("event_type") or "")
    if event_type in {"explore_task", "explore"}:
        item["task"] = str(event.get("task") or "")[:_EXPLORE_TASK_CHARS]
        if event.get("digest"):
            item["digest"] = str(event.get("digest"))[:_EXPLORE_DIGEST_CHARS]
        for key in ("rounds", "tool_calls", "llm_calls"):
            if event.get(key) is not None:
                item[key] = event.get(key)
    elif event_type == "explore_llm":
        item["model"] = event.get("model")
        item["tool_names"] = event.get("tool_names")
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            item["usage"] = {
                key: usage.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in usage
            }
    elif event_type == "explore_tool":
        item["tool"] = event.get("tool")
        result = event.get("result")
        if isinstance(result, Mapping):
            item["ok"] = result.get("ok")
            if result.get("error"):
                item["error"] = str(result.get("error"))[:400]
        elif event.get("error"):
            item["error"] = str(event.get("error"))[:400]
    if event.get("error") and "error" not in item:
        item["error"] = str(event.get("error"))[:400]
    return {key: value for key, value in item.items() if value not in (None, "", [])}


def build_meta_fold_reviews(
    records: list[Mapping[str, object]],
    *,
    artifacts_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Per completed Fold: frozen artifact/ref, strategy source, projections, Explore trace."""

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
                "explore_trace": compact_explore_trace(
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
