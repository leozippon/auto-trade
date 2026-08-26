"""Bounded access to redacted AgentTrace JSONL artifacts."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path

from autotrade.pipelines.hitl_state import read_status, status_pid_alive

DEFAULT_PAGE_BYTES = 512 * 1024
MAX_TAIL_BYTES = 4 * 1024 * 1024
MAX_BLOCK_READ_BYTES = 32 * 1024 * 1024
STREAM_POLL_SECONDS = 1.0
STREAM_IDLE_HEARTBEAT_EVERY = 15
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_BLOCK_TEXT_CHARS = 4_000
_BLOCK_SUMMARY_CHARS = 160
_BLOCK_TASK_CHARS = 400
_BLOCK_SUBAGENT_SUMMARY_CHARS = 400
_BLOCK_ERROR_CHARS = 240
_TERMINAL_SUBAGENT = frozenset({"completed", "timeout", "error", "cancelled"})


def resolve_trace_path(experiment_dir: Path, run_id: str | None) -> Path | None:
    """Resolve one canonical trace without accepting paths from client/status data."""

    directory = Path(experiment_dir).resolve()
    if run_id is None:
        status = read_status(directory / "hitl/status.json")
        value = status.get("run_id")
        run_id = str(value) if isinstance(value, str) and value else None
    if run_id is None or not _RUN_ID.fullmatch(run_id):
        return None
    path = (directory / "artifacts/traces" / f"{run_id}.jsonl").resolve()
    trace_root = (directory / "artifacts/traces").resolve()
    return path if path.is_relative_to(trace_root) and path.is_file() else None


def read_initial_prompt(path: Path) -> dict[str, object]:
    """Return the redacted system/user messages recorded at Fold session start."""

    with Path(path).open("rb") as handle:
        for raw in handle:
            event = _decode_event(raw)
            if event.get("event_type") != "session_start":
                continue
            system = event.get("system_prompt")
            instruction = event.get("instruction")
            if not isinstance(system, str) or not isinstance(instruction, str):
                break
            return {
                "run_id": event.get("run_id"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ],
            }
    raise KeyError("trace contains no Fold initial prompt")


def read_trace_page(
    path: Path,
    *,
    offset: int = 0,
    max_bytes: int = DEFAULT_PAGE_BYTES,
) -> dict[str, object]:
    """Read complete events from a byte offset; leave a partial live tail unread."""

    path = Path(path)
    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(max(1, int(max_bytes)))
        consumed = chunk.rfind(b"\n") + 1
        if consumed <= 0 and chunk and offset + len(chunk) < size:
            while True:
                more = handle.read(max(1, int(max_bytes)))
                if not more:
                    break
                newline = more.find(b"\n")
                if newline >= 0:
                    next_offset = handle.tell() - len(more) + newline + 1
                    return {
                        "events": [{"raw": f"<oversized event skipped: {next_offset - offset} bytes>"}],
                        "next_offset": next_offset,
                        "eof": next_offset >= size,
                    }
    if consumed <= 0:
        return {
            "events": [],
            "next_offset": offset,
            "eof": offset + len(chunk) >= size and not chunk,
        }
    events = [_decode_event(line) for line in chunk[:consumed].splitlines() if line.strip()]
    next_offset = offset + consumed
    return {"events": events, "next_offset": next_offset, "eof": next_offset >= size}


def read_trace_tail(
    path: Path,
    *,
    max_events: int,
    max_bytes: int = MAX_TAIL_BYTES,
) -> dict[str, object]:
    """Return a bounded tail plus the byte offset where live tailing can resume."""

    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        return {"events": [], "next_offset": 0, "eof": True, "history_truncated": False}
    read_size = min(size, max(1, int(max_bytes)))
    start = size - read_size
    with path.open("rb") as handle:
        handle.seek(start)
        blob = handle.read(read_size)
    if start:
        newline = blob.find(b"\n")
        if newline < 0:
            return {"events": [], "next_offset": size, "eof": True, "history_truncated": True}
        start += newline + 1
        blob = blob[newline + 1 :]
    complete_bytes = blob.rfind(b"\n") + 1
    lines = blob[:complete_bytes].splitlines(keepends=True)
    selected = lines[-max(1, int(max_events)) :]
    events = [_decode_event(line) for line in selected if line.strip()]
    next_offset = start + complete_bytes
    return {
        "events": events,
        "next_offset": next_offset,
        "eof": next_offset >= size,
        "history_truncated": len(selected) < len(lines) or start > 0,
    }


def read_trace_blocks(
    path: Path,
    *,
    offset: int = 0,
    max_bytes: int | None = None,
    tail_events: int | None = None,
) -> dict[str, object]:
    """Project a page or tail of raw events into display blocks. JSONL is unchanged."""

    if tail_events is not None:
        page = read_trace_tail(path, max_events=tail_events)
    else:
        size = Path(path).stat().st_size
        remaining = max(size - max(0, int(offset)), 0)
        if max_bytes is None:
            window = remaining or 1
        else:
            window = max(1, int(max_bytes))
        window = min(window, MAX_BLOCK_READ_BYTES)
        page = read_trace_page(path, offset=offset, max_bytes=window)
    events = page.get("events")
    if not isinstance(events, list):
        events = []
    projected = {
        "blocks": project_trace_blocks(events),
        "next_offset": page.get("next_offset"),
        "eof": page.get("eof"),
        "event_count": len(events),
    }
    if "history_truncated" in page:
        projected["history_truncated"] = page["history_truncated"]
    return projected


def project_trace_blocks(events: object) -> list[dict[str, object]]:
    """Project redacted trace events into stable Agent-visible display blocks."""

    if not isinstance(events, list):
        return []
    blocks: list[dict[str, object]] = []
    interval = _Interval()
    subagents: dict[str, _SubagentState] = {}
    seq = 0
    for item in events:
        seq += 1
        if not isinstance(item, dict):
            continue
        event: dict[str, object] = item
        kind = str(event.get("event_type") or "")
        if kind == "user_message":
            blocks.extend(interval.flush())
            blocks.append(
                {
                    "kind": "user",
                    "ts": _event_ts(event),
                    "text": _clip(
                        _first_text(event, "content", "text", "message"),
                        _BLOCK_TEXT_CHARS,
                    ),
                }
            )
            continue
        output = _agent_output_text(event)
        if output is not None:
            blocks.extend(interval.flush())
            blocks.append(
                {
                    "kind": "agent_output",
                    "ts": _event_ts(event),
                    "text": _clip(output, _BLOCK_TEXT_CHARS),
                    "reasoning_chars": _reasoning_chars(event),
                }
            )
            continue
        task_id = _explore_event_task_id(event)
        if task_id is not None:
            _observe_subagent(interval, subagents, event, task_id, seq)
            if kind in {"explore_tool", "explore_tool_started"}:
                subagents[task_id].tools.add(event, seq)
                _sync_subagent_fields(subagents[task_id])
            continue
        if kind in {"tool_call_started", "tool_call"}:
            interval.tools.add(event, seq)
    blocks.extend(interval.flush())
    return blocks


_STATS_CACHE: dict[str, dict[str, object]] = {}
_STATS_LOCK = threading.Lock()


def trace_stats(path: Path) -> dict[str, object]:
    """Incrementally aggregate event/tool/token counts and unique Explore tasks."""

    with _STATS_LOCK:
        path = Path(path)
        size = path.stat().st_size
        key = str(path.resolve())
        cached = _STATS_CACHE.get(key)
        if (
            cached is None
            or size < _as_int(cached.get("offset"))
            or "subagent_task_ids" not in cached
        ):
            cached = {
                "offset": 0,
                "counts": {},
                "tool_counts": {},
                "llm_total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "active_tool": None,
                "last_event_ts": None,
                "subagent_task_ids": set(),
            }
        offset = _as_int(cached.get("offset"))
        with path.open("rb") as handle:
            handle.seek(offset)
            blob = handle.read(size - offset)
        tail = blob.rfind(b"\n") + 1
        counts = _as_mapping(cached.get("counts"))
        tool_counts = _as_mapping(cached.get("tool_counts"))
        total = _as_int(cached.get("llm_total_tokens"))
        prompt = _as_int(cached.get("prompt_tokens"))
        completion = _as_int(cached.get("completion_tokens"))
        active_tool = cached.get("active_tool")
        last_ts = cached.get("last_event_ts")
        task_ids = set(_as_str_list(cached.get("subagent_task_ids")))
        for raw in blob[:tail].splitlines():
            event = _decode_event(raw)
            kind = str(event.get("event_type") or "event")
            counts[kind] = _as_int(counts.get(kind)) + 1
            last_ts = event.get("ts") or last_ts
            task_id = _explore_event_task_id(event)
            if task_id is not None:
                task_ids.add(task_id)
            if kind == "tool_call_started":
                active_tool = event.get("tool")
            elif kind == "tool_call":
                tool = str(event.get("tool") or "unknown")
                tool_counts[tool] = _as_int(tool_counts.get(tool)) + 1
                active_tool = None
            elif kind == "llm_call":
                usage = _as_mapping(event.get("usage"))
                if usage:
                    total += _as_int(usage.get("total_tokens"))
                    prompt += _as_int(usage.get("prompt_tokens"))
                    completion += _as_int(usage.get("completion_tokens"))
        cached = {
            "offset": offset + tail,
            "counts": counts,
            "tool_counts": tool_counts,
            "llm_total_tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "active_tool": active_tool,
            "last_event_ts": last_ts,
            "subagent_task_ids": task_ids,
        }
        if len(_STATS_CACHE) >= 32 and key not in _STATS_CACHE:
            _STATS_CACHE.pop(next(iter(_STATS_CACHE)))
        _STATS_CACHE[key] = cached
        return {
            "counts": counts,
            "tool_counts": tool_counts,
            "total_events": sum(_as_int(value) for value in counts.values()),
            "llm_total_tokens": total,
            "llm_prompt_tokens": prompt,
            "llm_completion_tokens": completion,
            "active_tool": active_tool,
            "last_event_ts": last_ts,
            "trace_bytes": size,
            "subagent_tasks": len(task_ids),
        }


async def stream_trace(
    experiment_dir: Path,
    run_id: str | None,
    *,
    offset: int = 0,
    project_event: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> AsyncIterator[str]:
    """Replay then tail a trace over SSE without retaining a server-side history."""

    directory = Path(experiment_dir)
    position = max(0, int(offset))
    idle = 0
    yield "retry: 5000\n\n"
    while True:
        path = resolve_trace_path(directory, run_id)
        if path is not None:
            page = read_trace_page(path, offset=position)
            raw_events = page.get("events")
            events = [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else []
            if events:
                next_offset = page.get("next_offset")
                position = int(next_offset) if isinstance(next_offset, int) else position
                projected = [project_event(event) for event in events] if project_event else events
                for event in projected[:-1]:
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                yield f"id: {position}\ndata: {json.dumps(projected[-1], ensure_ascii=False, default=str)}\n\n"
                idle = 0
                continue
        status = read_status(directory / "hitl/status.json")
        current = str(status.get("run_id") or "")
        if not status_pid_alive(status) or (run_id is not None and current not in {"", run_id}):
            yield f'event: eof\ndata: {{"offset": {position}}}\n\n'
            return
        if path is None:
            yield 'event: waiting\ndata: {"reason": "trace not started"}\n\n'
        idle += 1
        if idle % STREAM_IDLE_HEARTBEAT_EVERY == 0:
            yield ": keep-alive\n\n"
        await asyncio.sleep(STREAM_POLL_SECONDS)


def _decode_event(raw: bytes) -> dict[str, object]:
    """An unparseable line surfaces verbatim rather than silently vanishing —
    a live trace's partial tail and a corrupted record must both stay visible."""
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return value if isinstance(value, dict) else {"raw": text}


def _explore_event_task_id(event: dict[str, object]) -> str | None:
    """Return a unique Explore sub-agent task id, or None.

    The count is launched tasks, not sub-agent LLM or tool calls. Prefer
    ``explore_task_started`` when present; older traces may omit the start
    record, so any ``explore`` / ``explore_*`` event with a non-empty
    ``task_id`` joins the same set. Prompt text is never parsed.
    """
    kind = str(event.get("event_type") or "")
    if kind != "explore" and not kind.startswith("explore_"):
        return None
    value = event.get("task_id")
    if not isinstance(value, str):
        return None
    task_id = value.strip()
    return task_id or None


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset)
    ):
        return []
    return [item for item in value if isinstance(item, str)]


class _Call:
    __slots__ = ("name", "key", "status", "summary", "ts")

    def __init__(self, name: str, key: str, status: str, summary: str, ts: object) -> None:
        self.name = name
        self.key = key
        self.status = status
        self.summary = summary
        self.ts = ts


class _ToolAcc:
    def __init__(self) -> None:
        self.calls: dict[str, _Call] = {}
        self.first_ts: object = None
        self.first_seq = 0
        self._serial = 0

    def __bool__(self) -> bool:
        return bool(self.calls)

    def add(self, event: dict[str, object], seq: int) -> None:
        kind = str(event.get("event_type") or "")
        if kind.endswith("_started"):
            self._started(event, seq)
        else:
            self._finished(event, seq)

    def as_rows(self) -> list[dict[str, object]]:
        grouped: dict[str, dict[str, object]] = {}
        order: list[str] = []
        for call in self.calls.values():
            row = grouped.get(call.name)
            if row is None:
                row = {
                    "name": call.name,
                    "count": 0,
                    "ok": 0,
                    "failed": 0,
                    "running": 0,
                    "summary": "",
                }
                grouped[call.name] = row
                order.append(call.name)
            row["count"] = _as_int(row.get("count")) + 1
            bucket = "ok" if call.status == "ok" else call.status
            if bucket not in {"ok", "failed", "running"}:
                bucket = "ok"
            row[bucket] = _as_int(row.get(bucket)) + 1
            if call.summary and (not row["summary"] or call.status == "failed"):
                row["summary"] = call.summary
        return [grouped[name] for name in order]

    def to_block(self) -> dict[str, object]:
        tools = self.as_rows()
        return {
            "kind": "tool_group",
            "ts": self.first_ts,
            "tools": tools,
            "count": sum(_as_int(row.get("count")) for row in tools),
            "ok": sum(_as_int(row.get("ok")) for row in tools),
            "failed": sum(_as_int(row.get("failed")) for row in tools),
            "running": sum(_as_int(row.get("running")) for row in tools),
        }

    def _started(self, event: dict[str, object], seq: int) -> None:
        key = _call_key(event) or f"open:{self._next_serial()}"
        if key in self.calls:
            return
        self._new_call(event, seq, key, "running")

    def _finished(self, event: dict[str, object], seq: int) -> None:
        key = _call_key(event)
        if key and key in self.calls:
            self._complete(self.calls[key], event)
            return
        name = _tool_name(event)
        if key is None:
            for call in self.calls.values():
                if call.status == "running" and call.name == name:
                    self._complete(call, event)
                    return
        self._new_call(event, seq, key or f"done:{self._next_serial()}", _tool_outcome(event))

    def _new_call(self, event: dict[str, object], seq: int, key: str, status: str) -> None:
        if self.first_ts is None:
            self.first_ts = _event_ts(event)
            self.first_seq = seq
        self.calls[key] = _Call(
            _tool_name(event),
            key,
            status,
            _tool_summary(event),
            _event_ts(event),
        )

    def _complete(self, call: _Call, event: dict[str, object]) -> None:
        call.status = _tool_outcome(event)
        summary = _tool_summary(event)
        if summary:
            call.summary = summary

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial


class _Interval:
    def __init__(self) -> None:
        self.tools = _ToolAcc()
        self.extras: list[tuple[object, int, dict[str, object]]] = []

    def add_extra(self, event: dict[str, object], seq: int, block: dict[str, object]) -> None:
        self.extras.append((_event_ts(event), seq, block))

    def flush(self) -> list[dict[str, object]]:
        items = list(self.extras)
        if self.tools:
            items.append((self.tools.first_ts, self.tools.first_seq, self.tools.to_block()))
        items.sort(key=_interval_sort_key)
        self.tools = _ToolAcc()
        self.extras = []
        return [block for _ts, _seq, block in items]


class _SubagentState:
    def __init__(self) -> None:
        self.started_block: dict[str, object] | None = None
        self.ended_block: dict[str, object] | None = None
        self.ended = False
        self.tools = _ToolAcc()
        self.task = ""
        self.summary = ""
        self.error = ""


def _observe_subagent(
    interval: _Interval,
    subagents: dict[str, _SubagentState],
    event: dict[str, object],
    task_id: str,
    seq: int,
) -> None:
    state = subagents.get(task_id)
    if state is None:
        state = _SubagentState()
        subagents[task_id] = state
    phase = _subagent_phase(event)
    _absorb_subagent_text(state, event)
    if state.started_block is None:
        started = _subagent_block(task_id, "started", "started", event, state)
        state.started_block = started
        interval.add_extra(event, seq, started)
    elif phase == "progress" and not state.ended:
        state.started_block["status"] = "running"
    if phase == "ended" and not state.ended:
        status = _terminal_status(event)
        ended = _subagent_block(task_id, "ended", status, event, state)
        state.ended_block = ended
        state.ended = True
        interval.add_extra(event, seq, ended)
    _sync_subagent_fields(state)


def _subagent_block(
    task_id: str,
    phase: str,
    status: str,
    event: dict[str, object],
    state: _SubagentState,
) -> dict[str, object]:
    return {
        "kind": "subagent",
        "ts": _event_ts(event),
        "task_id": task_id,
        "phase": phase,
        "status": status,
        "task": state.task,
        "summary": state.summary,
        "error": state.error,
        "tools": state.tools.as_rows(),
    }


def _absorb_subagent_text(state: _SubagentState, event: dict[str, object]) -> None:
    task = _clip(event.get("task"), _BLOCK_TASK_CHARS)
    if task:
        state.task = task
    summary = _clip(event.get("summary"), _BLOCK_SUBAGENT_SUMMARY_CHARS)
    if summary:
        state.summary = summary
    error = _clip(event.get("error"), _BLOCK_ERROR_CHARS)
    if error:
        state.error = error


def _sync_subagent_fields(state: _SubagentState) -> None:
    rows = state.tools.as_rows()
    for block in (state.started_block, state.ended_block):
        if block is None:
            continue
        block["task"] = state.task
        block["summary"] = state.summary
        block["error"] = state.error
        block["tools"] = rows


def _subagent_phase(event: dict[str, object]) -> str:
    kind = str(event.get("event_type") or "")
    status = str(event.get("status") or "").strip().lower()
    if kind == "explore_task_started":
        return "started"
    if kind == "explore":
        return "ended"
    if kind == "explore_task":
        return "ended" if status in _TERMINAL_SUBAGENT else "started"
    if status in _TERMINAL_SUBAGENT:
        return "ended"
    return "progress"


def _terminal_status(event: dict[str, object]) -> str:
    status = str(event.get("status") or "").strip().lower()
    return status if status in _TERMINAL_SUBAGENT else "completed"


def _agent_output_text(event: dict[str, object]) -> str | None:
    if str(event.get("event_type") or "") != "llm_call":
        return None
    text = event.get("content")
    if isinstance(text, str) and text.strip():
        return text.strip()
    if event.get("truncated"):
        preview = event.get("content_preview")
        if isinstance(preview, str) and preview.strip():
            return preview.strip()
    return None


def _reasoning_chars(event: dict[str, object]) -> int:
    for field in ("reasoning_content", "reasoning"):
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            return len(value.strip())
    return 0


def _tool_name(event: dict[str, object]) -> str:
    name = event.get("tool")
    return name.strip() if isinstance(name, str) and name.strip() else "unknown"


def _call_key(event: dict[str, object]) -> str | None:
    for field in ("tool_call_id", "call_id"):
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _tool_outcome(event: dict[str, object]) -> str:
    if str(event.get("event_type") or "").endswith("_started"):
        return "running"
    result = event.get("result")
    if isinstance(result, dict):
        if result.get("ok") is False or result.get("error"):
            return "failed"
        if result.get("ok") is True:
            return "ok"
    status = str(event.get("status") or "").strip().lower()
    if status in {"error", "failed", "timeout"}:
        return "failed"
    if status == "running":
        return "running"
    return "ok"


def _tool_summary(event: dict[str, object]) -> str:
    result = event.get("result")
    if isinstance(result, dict):
        error = _clip(result.get("error"), _BLOCK_SUMMARY_CHARS)
        if error:
            return error
    return _clip(event.get("error"), _BLOCK_SUMMARY_CHARS)


def _event_ts(event: dict[str, object]) -> object:
    return event.get("ts")


def _first_text(event: dict[str, object], *fields: str) -> object:
    for field in fields:
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _clip(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]


def _interval_sort_key(item: tuple[object, int, dict[str, object]]) -> tuple[int, str, int]:
    ts, seq, _block = item
    if isinstance(ts, str) and ts:
        return (0, ts, seq)
    return (1, "", seq)

