"""Bounded access to redacted AgentTrace JSONL artifacts."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator, Mapping
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
_BLOCK_SUBAGENT_SUMMARY_CHARS = 400
_BLOCK_DESCRIPTION_CHARS = 80
_BLOCK_ERROR_CHARS = 240
_BLOCK_ARGUMENT_CHARS = 600
_BLOCK_RESULT_CHARS = 1_200
SUBAGENT_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
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
        if not kind and isinstance(event.get("raw"), str):
            blocks.extend(interval.flush())
            blocks.append(
                {
                    "kind": "raw",
                    "ts": _event_ts(event),
                    "text": _clip(event.get("raw"), _BLOCK_TEXT_CHARS),
                }
            )
            continue
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
            block: dict[str, object] = {
                "kind": "agent_output",
                "ts": _event_ts(event),
                "text": _clip(output, _BLOCK_TEXT_CHARS),
                "reasoning_chars": _reasoning_chars(event),
            }
            model = _event_model(event)
            if model:
                block["model"] = model
            blocks.append(block)
            continue
        task_id = _subagent_event_task_id(event)
        if task_id is not None:
            state = subagents.get(task_id)
            if state is None:
                state = subagents[task_id] = _SubagentState()
            if _event_kind(event) in {"subagent_tool", "subagent_tool_started"}:
                state.tools.add(event, seq)
            _observe_subagent(interval, state, event, task_id, seq)
            continue
        if kind in {"tool_call_started", "tool_call"}:
            interval.tools.add(event, seq)
    blocks.extend(interval.flush())
    return blocks


def read_subagent_trace(path: Path, task_id: str) -> dict[str, object]:
    """Project one sub-agent task's own events out of the parent trace."""

    size = Path(path).stat().st_size
    page = read_trace_page(path, offset=0, max_bytes=min(size or 1, MAX_BLOCK_READ_BYTES))
    events = page.get("events")
    projected = project_subagent_trace(events if isinstance(events, list) else [], task_id)
    projected["truncated_window"] = not bool(page.get("eof"))
    return projected


def project_subagent_trace(events: object, task_id: str) -> dict[str, object]:
    """One child's rounds in order, in the block model the parent view renders.

    ``agent_output`` per model round and ``tool_group`` per batch of tool calls
    reuse the parent renderers; ``marker`` and ``summary`` carry the wrap-up,
    truncation and final report. Meta traces arrive already reduced to counts,
    which the ``reduced`` flag reports rather than silently showing nothing.
    """

    if not isinstance(events, list):
        events = []
    header_interval = _Interval()
    state = _SubagentState()
    blocks: list[dict[str, object]] = []
    tools = _ToolAcc()
    calls: list[dict[str, object]] = []
    matched = 0
    reduced = False
    seq = 0

    def flush_tools() -> None:
        nonlocal tools, calls
        if tools:
            blocks.append({**tools.to_block(), "calls": calls})
        tools = _ToolAcc()
        calls = []

    for item in events:
        seq += 1
        if not isinstance(item, dict):
            continue
        event: dict[str, object] = item
        if _subagent_event_task_id(event) != task_id:
            continue
        matched += 1
        kind = _event_kind(event)
        if kind in {"subagent_tool", "subagent_tool_started"}:
            state.tools.add(event, seq)
            tools.add(event, seq)
            calls.append(_subagent_call_row(event))
        _observe_subagent(header_interval, state, event, task_id, seq)
        if kind == "subagent_llm":
            flush_tools()
            blocks.append(_subagent_round_block(event))
            reduced = reduced or _is_reduced_round(event)
        elif kind == "subagent_wrap_up":
            flush_tools()
            blocks.append(
                _marker_block(
                    event,
                    "收尾提示",
                    f"第 {_as_int(event.get('round'))} 轮，上限 "
                    f"{_as_int(event.get('rounds_limit'))}：已要求子代理立即收尾。",
                )
            )
        elif kind == "subagent_steer":
            flush_tools()
            chars = _as_int(event.get("chars"))
            blocks.append(
                _marker_block(
                    event,
                    "父代理指令",
                    f"第 {_as_int(event.get('round'))} 轮前送达（{chars} 字符）。"
                    if event.get("delivery") == "delivered"
                    else f"已排队（{chars} 字符），子代理下一轮前读取。",
                )
            )
        elif kind == "subagent_context_compaction":
            flush_tools()
            compaction = _as_mapping(event.get("compaction"))
            blocks.append(
                _marker_block(
                    event,
                    "上下文压缩",
                    f"第 {_as_int(event.get('round'))} 轮："
                    f"{compaction.get('status') or 'unknown'}，消息 "
                    f"{_as_int(compaction.get('messages_before'))}→"
                    f"{_as_int(compaction.get('messages_after'))}，估算 "
                    f"{_as_int(compaction.get('estimated_tokens'))} tokens。",
                )
            )
        elif _subagent_phase(event) == "ended":
            flush_tools()
            if event.get("truncated") is True:
                blocks.append(
                    _marker_block(event, "输出截断", "子代理的模型输出触及长度上限。")
                )
            summary = _clip(event.get("summary"), _BLOCK_TEXT_CHARS)
            summary_chars = _as_int(event.get("summary_chars"))
            if summary or summary_chars:
                blocks.append(
                    {
                        "kind": "summary",
                        "ts": _event_ts(event),
                        "status": _terminal_status(event),
                        "text": summary,
                        "text_chars": len(summary) if summary else summary_chars,
                    }
                )
            reduced = reduced or (summary_chars > 0 and not summary)
    flush_tools()
    return {
        "task_id": task_id,
        "found": matched > 0,
        "header": state.block,
        "blocks": blocks,
        "reduced": reduced,
        "event_count": matched,
    }


def _subagent_round_block(event: dict[str, object]) -> dict[str, object]:
    block: dict[str, object] = {
        "kind": "agent_output",
        "ts": _event_ts(event),
        "round": _as_int(event.get("round")),
        "text": _clip(event.get("content"), _BLOCK_TEXT_CHARS),
        "reasoning_chars": _reasoning_chars(event),
    }
    chars = _as_int(event.get("content_chars"))
    if chars and not block["text"]:
        block["content_chars"] = chars
    model = _event_model(event)
    if model:
        block["model"] = model
    return block


def _is_reduced_round(event: dict[str, object]) -> bool:
    """Meta payloads keep the round's shape but drop its text."""

    return "content" not in event and _as_int(event.get("content_chars")) > 0


def _marker_block(event: dict[str, object], label: str, text: str) -> dict[str, object]:
    return {"kind": "marker", "ts": _event_ts(event), "label": label, "text": text}


def _subagent_call_row(event: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "name": _tool_name(event),
        "status": _tool_outcome(event),
        "ts": _event_ts(event),
        "round": _as_int(event.get("round")),
    }
    arguments = _as_mapping(event.get("arguments"))
    if arguments:
        row["arguments"] = {
            key: _clip(_as_text(value), _BLOCK_ARGUMENT_CHARS)
            for key, value in arguments.items()
        }
    result = event.get("result")
    if result is not None:
        row["result"] = _clip(_as_text(result), _BLOCK_RESULT_CHARS)
    summary = _tool_summary(event)
    if summary:
        row["error"] = summary
    return row


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


_STATS_CACHE: dict[str, dict[str, object]] = {}
_STATS_LOCK = threading.Lock()


def trace_stats(path: Path) -> dict[str, object]:
    """Incrementally aggregate event/tool/token counts and unique sub-agent tasks."""

    with _STATS_LOCK:
        path = Path(path)
        size = path.stat().st_size
        key = str(path.resolve())
        cached = _STATS_CACHE.get(key)
        if (
            cached is None
            or size < _as_int(cached.get("offset"))
            or "subagent_task_ids" not in cached
            or "subagent_ended_ids" not in cached
            or "subagent_usage" not in cached
            or "last_main_prompt_tokens" not in cached
        ):
            cached = {
                "offset": 0,
                "counts": {},
                "tool_counts": {},
                "llm_total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "last_main_prompt_tokens": 0,
                "subagent_task_ids": set(),
                "subagent_ended_ids": set(),
                "subagent_usage": {},
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
        last_main_prompt = _as_int(cached.get("last_main_prompt_tokens"))
        task_ids = set(_as_str_list(cached.get("subagent_task_ids")))
        ended_ids = set(_as_str_list(cached.get("subagent_ended_ids")))
        subagent_usage = {
            task: _usage_row(_as_mapping(row))
            for task, row in _as_mapping(cached.get("subagent_usage")).items()
        }
        for raw in blob[:tail].splitlines():
            event = _decode_event(raw)
            kind = str(event.get("event_type") or "event")
            counts[kind] = _as_int(counts.get(kind)) + 1
            task_id = _subagent_event_task_id(event)
            if task_id is not None:
                task_ids.add(task_id)
                # A finished task reports authoritative totals; a live one is
                # summed from the per-round records seen so far.
                if _subagent_phase(event) == "ended":
                    ended_ids.add(task_id)
                    totals = _as_mapping(event.get("usage_totals"))
                    if totals:
                        subagent_usage[task_id] = _usage_row(totals)
                elif _event_kind(event) == "subagent_llm" and task_id not in ended_ids:
                    _add_usage(
                        subagent_usage.setdefault(task_id, _new_usage()),
                        _as_mapping(event.get("usage")),
                    )
            if kind == "tool_call":
                tool = str(event.get("tool") or "unknown")
                tool_counts[tool] = _as_int(tool_counts.get(tool)) + 1
            elif kind == "llm_call":
                usage = _as_mapping(event.get("usage"))
                if usage:
                    total += _as_int(usage.get("total_tokens"))
                    prompt += _as_int(usage.get("prompt_tokens"))
                    completion += _as_int(usage.get("completion_tokens"))
                    if _is_main_agent_llm_call(event):
                        last_main_prompt = _as_int(usage.get("prompt_tokens"))
        cached = {
            "offset": offset + tail,
            "counts": counts,
            "tool_counts": tool_counts,
            "llm_total_tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "last_main_prompt_tokens": last_main_prompt,
            "subagent_task_ids": task_ids,
            "subagent_ended_ids": ended_ids,
            "subagent_usage": subagent_usage,
        }
        if len(_STATS_CACHE) >= 32 and key not in _STATS_CACHE:
            _STATS_CACHE.pop(next(iter(_STATS_CACHE)))
        _STATS_CACHE[key] = cached
        return {
            "counts": counts,
            "tool_counts": tool_counts,
            "llm_total_tokens": total,
            "llm_prompt_tokens": prompt,
            "llm_completion_tokens": completion,
            "last_llm_prompt_tokens": last_main_prompt,
            "subagent_tasks": len(task_ids),
            "subagent_running": len(task_ids - ended_ids),
            "subagent_prompt_tokens": _usage_sum(subagent_usage, "prompt_tokens"),
            "subagent_completion_tokens": _usage_sum(subagent_usage, "completion_tokens"),
            "subagent_total_tokens": sum(
                _usage_total(row) for row in subagent_usage.values()
            ),
            "compact_ops": _as_int(counts.get("context_compaction")),
        }


async def stream_trace(
    experiment_dir: Path,
    run_id: str | None,
    *,
    offset: int = 0,
) -> AsyncIterator[str]:
    """Tail a trace over SSE as an offset-only nudge.

    The payload carries no event text: the console always re-reads the block
    projection, so streaming redacted events would duplicate that work and
    widen the surface that has to stay redacted.
    """

    directory = Path(experiment_dir)
    position = max(0, int(offset))
    idle = 0
    yield "retry: 5000\n\n"
    while True:
        path = resolve_trace_path(directory, run_id)
        if path is not None:
            appended = _appended_offset(path, position)
            if appended > position:
                position = appended
                yield f'id: {position}\ndata: {{"offset": {position}}}\n\n'
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


def _appended_offset(path: Path, offset: int) -> int:
    """End of the last complete line at or after ``offset``, without decoding.

    Bounded by one page per pass; the writer caps a single event well below
    that, so a complete line always fits and the tail cannot stall.
    """

    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    if size <= offset:
        return offset
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(DEFAULT_PAGE_BYTES)
    return offset + chunk.rfind(b"\n") + 1


def _decode_event(raw: bytes) -> dict[str, object]:
    """An unparseable line surfaces verbatim rather than silently vanishing —
    a live trace's partial tail and a corrupted record must both stay visible."""
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return value if isinstance(value, dict) else {"raw": text}


def _is_main_agent_llm_call(event: dict[str, object]) -> bool:
    """Parent-session ``llm_call`` rows; sub-agent child calls carry a task id."""

    if str(event.get("event_type") or "") != "llm_call":
        return False
    if _subagent_event_task_id(event) is not None:
        return False
    task_id = event.get("task_id")
    return not (isinstance(task_id, str) and task_id.strip())


_LEGACY_SUBAGENT_KINDS = {
    "explore": "subagent",
    "explore_task": "subagent_task",
    "explore_task_started": "subagent_task_started",
    "explore_llm": "subagent_llm",
    "explore_tool": "subagent_tool",
    "explore_tool_started": "subagent_tool_started",
    "explore_attempt": "subagent_attempt",
}


def _event_kind(event: dict[str, object]) -> str:
    """The event type, with the pre-rename ``explore_*`` names of traces
    written by earlier sessions mapped onto the current ``subagent_*`` names."""

    kind = str(event.get("event_type") or "")
    return _LEGACY_SUBAGENT_KINDS.get(kind, kind)


def _subagent_event_task_id(event: dict[str, object]) -> str | None:
    """Return a unique sub-agent task id, or None.

    The count is launched tasks, not sub-agent LLM or tool calls. Prefer
    ``subagent_task_started`` when present; older traces may omit the start
    record, so any ``subagent`` / ``subagent_*`` event with a non-empty
    ``task_id`` joins the same set. Prompt text is never parsed.
    """
    kind = _event_kind(event)
    if kind != "subagent" and not kind.startswith("subagent_"):
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


_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _new_usage() -> dict[str, int]:
    return dict.fromkeys(_USAGE_FIELDS, 0)


def _usage_row(usage: Mapping[str, object]) -> dict[str, int]:
    return {field: _as_int(usage.get(field)) for field in _USAGE_FIELDS}


def _add_usage(target: dict[str, int], usage: Mapping[str, object]) -> None:
    for field in _USAGE_FIELDS:
        target[field] += _as_int(usage.get(field))


def _usage_sum(usages: Mapping[str, Mapping[str, int]], field: str) -> int:
    return sum(_as_int(row.get(field)) for row in usages.values())


def _usage_total(usage: Mapping[str, int]) -> int:
    """Providers that report only the two halves still get a truthful total."""

    total = _as_int(usage.get("total_tokens"))
    return total or _as_int(usage.get("prompt_tokens")) + _as_int(
        usage.get("completion_tokens")
    )


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
        self.last_key = ""
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

    def last_call(self) -> dict[str, object] | None:
        call = self.calls.get(self.last_key)
        return {"name": call.name, "status": call.status} if call else None

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
        self.last_key = key

    def _complete(self, call: _Call, event: dict[str, object]) -> None:
        call.status = _tool_outcome(event)
        summary = _tool_summary(event)
        if summary:
            call.summary = summary
        self.last_key = call.key

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
            items.append(
                (self.tools.first_ts, self.tools.first_seq, self.tools.to_block())
            )
        items.sort(key=_interval_sort_key)
        self.tools = _ToolAcc()
        self.extras = []
        return [block for _ts, _seq, block in items]


class _SubagentState:
    """One card per sub-agent task: launch facts, live progress, final totals."""

    def __init__(self) -> None:
        self.block: dict[str, object] | None = None
        self.ended = False
        self.tools = _ToolAcc()
        self.summary = ""
        self.error = ""
        self.role = ""
        self.model = ""
        self.thinking = ""
        self.rounds_limit = 0
        self.inherit_context: bool | None = None
        self.description = ""
        self.resumed_from = ""
        self.started_at: object = None
        self.ended_at: object = None
        self.rounds = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.usage = _new_usage()


def _observe_subagent(
    interval: _Interval,
    state: _SubagentState,
    event: dict[str, object],
    task_id: str,
    seq: int,
) -> None:
    """Fold one sub-agent event into the task's single display block.

    A task emits exactly one block, created where it was launched and updated
    in place, so a finished sub-agent's report is rendered once.
    """

    phase = _subagent_phase(event)
    _absorb_subagent_text(state, event)
    if _event_kind(event) == "subagent_llm":
        state.llm_calls += 1
        state.rounds = max(state.rounds, _as_int(event.get("round")), state.llm_calls)
        _add_usage(state.usage, _as_mapping(event.get("usage")))
    if state.block is None:
        state.started_at = _event_ts(event)
        state.block = {
            "kind": "subagent",
            "ts": state.started_at,
            "task_id": task_id,
            "phase": "started",
            "status": "started",
        }
        interval.add_extra(event, seq, state.block)
    elif phase == "progress" and not state.ended:
        state.block["status"] = "running"
    if phase == "ended" and not state.ended:
        _absorb_subagent_totals(state, event)
        state.ended = True
        state.ended_at = _event_ts(event)
        state.block["phase"] = "ended"
        state.block["status"] = _terminal_status(event)
    _refresh_subagent_block(state)


def _absorb_subagent_text(state: _SubagentState, event: dict[str, object]) -> None:
    summary = _clip(event.get("summary"), _BLOCK_SUBAGENT_SUMMARY_CHARS)
    if summary:
        state.summary = summary
    error = _clip(event.get("error"), _BLOCK_ERROR_CHARS)
    if error:
        state.error = error
    role = event.get("role")
    if isinstance(role, str) and role.strip():
        state.role = role.strip()
    model = _event_model(event)
    if model:
        state.model = model
    thinking = event.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        state.thinking = thinking.strip()
    if _as_int(event.get("rounds_limit")) > 0:
        state.rounds_limit = _as_int(event.get("rounds_limit"))
    if "inherit_context" in event:
        state.inherit_context = bool(event.get("inherit_context"))
    description = _clip(event.get("description"), _BLOCK_DESCRIPTION_CHARS)
    if description:
        state.description = description
    resumed_from = _clip(event.get("resumed_from"), _BLOCK_DESCRIPTION_CHARS)
    if resumed_from:
        state.resumed_from = resumed_from


def _absorb_subagent_totals(state: _SubagentState, event: dict[str, object]) -> None:
    """The terminal event's own totals win: they also cover rounds whose
    events fell outside the read window."""

    for field in ("rounds", "llm_calls", "tool_calls"):
        value = event.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(state, field, value)
    totals = _as_mapping(event.get("usage_totals"))
    if totals:
        state.usage = _usage_row(totals)


def _refresh_subagent_block(state: _SubagentState) -> None:
    block = state.block
    if block is None:
        return
    rows = state.tools.as_rows()
    block["summary"] = state.summary
    block["error"] = state.error
    block["tools"] = rows
    block["tool_calls"] = state.tool_calls or sum(
        _as_int(row.get("count")) for row in rows
    )
    block["rounds"] = state.rounds
    block["llm_calls"] = state.llm_calls
    block["usage"] = {**state.usage, "total_tokens": _usage_total(state.usage)}
    if state.started_at is not None:
        block["started_at"] = state.started_at
    if state.ended_at is not None:
        block["ended_at"] = state.ended_at
    last_tool = state.tools.last_call()
    if last_tool is not None:
        block["last_tool"] = last_tool
    if state.role:
        block["role"] = state.role
    if state.model:
        block["model"] = state.model
    if state.thinking:
        block["thinking"] = state.thinking
    if state.rounds_limit:
        block["rounds_limit"] = state.rounds_limit
    if state.inherit_context is not None:
        block["inherit_context"] = state.inherit_context
    if state.description:
        block["description"] = state.description
    if state.resumed_from:
        block["resumed_from"] = state.resumed_from


def _subagent_phase(event: dict[str, object]) -> str:
    kind = _event_kind(event)
    status = str(event.get("status") or "").strip().lower()
    if kind == "subagent_task_started":
        return "started"
    if kind == "subagent":
        return "ended"
    if kind == "subagent_task":
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


def _event_model(event: dict[str, object]) -> str:
    model = event.get("model")
    return model.strip() if isinstance(model, str) and model.strip() else ""


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
    """Bound a text field once, here; the console renders what it receives."""

    if not isinstance(value, str):
        return ""
    text = value.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…" if limit > 1 else text[:limit]


def _interval_sort_key(item: tuple[object, int, dict[str, object]]) -> tuple[int, str, int]:
    ts, seq, _block = item
    if isinstance(ts, str) and ts:
        return (0, ts, seq)
    return (1, "", seq)
