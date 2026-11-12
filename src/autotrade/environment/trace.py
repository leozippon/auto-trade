"""The AgentTrace of one research attempt.

Its bounded, redacted JSONL event stream stays host-side under the
experiment's ``artifacts/traces/<run_id>.jsonl``; the transcript rendered from
the same events, with host paths redacted, is the session's ``trace`` read
root.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path

from autotrade.common.records import redact_host_paths
from autotrade.environment.runtime import new_id, sanitize_for_log, utc_now_iso
from autotrade.environment.tools.base import SessionInterrupt


class TraceEvent(StrEnum):
    """Every ``event_type`` an AgentTrace carries; the writer refuses any
    other name. The console's trace views also group them by name: a
    ``*_started`` event is a call or phase still running, and ``subagent``
    and ``subagent_*`` events are a sub-agent's."""

    # The session loop (``agent.runner``).
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL = "llm_call"
    OUTPUT_TRUNCATED = "output_truncated"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL = "tool_call"
    TOOL_SKIPPED = "tool_skipped"
    USER_MESSAGE = "user_message"
    BUDGET_NOTICE = "budget_notice"
    WRAP_UP_STARTED = "wrap_up_started"
    CONTEXT_COMPACTION = "context_compaction"
    CONTEXT_EDIT = "context_edit"
    CONTEXT_NOTICE = "context_notice"
    CONTEXT_OUTPUT_CLAMPED = "context_output_clamped"
    SUBAGENT_ATTEMPT = "subagent_attempt"
    SUBAGENT_WAIT_STARTED = "subagent_wait_started"
    # Sub-agents (``agent.subagent``); a steer is emitted by both sides.
    SUBAGENT = "subagent"
    SUBAGENT_TASK = "subagent_task"
    SUBAGENT_STEER = "subagent_steer"
    SUBAGENT_WRAP_UP = "subagent_wrap_up"
    SUBAGENT_LLM = "subagent_llm"
    SUBAGENT_LLM_ERROR = "subagent_llm_error"
    SUBAGENT_OUTPUT_TRUNCATED = "subagent_output_truncated"
    SUBAGENT_TOOL = "subagent_tool"
    SUBAGENT_CONTEXT_COMPACTION = "subagent_context_compaction"
    SUBAGENT_CONTEXT_EDIT = "subagent_context_edit"
    # The research session, its tools and the worker around it.
    SESSION_ERROR = "session_error"
    ENVIRONMENT_STAGE = "environment_stage"
    ENVIRONMENT_BLOCKED = "environment_blocked"
    BATCH_REJECTION_ESCALATED = "batch_rejection_escalated"
    # The writer's own marker: the trace is full and stops here.
    TRACE_LIMIT_REACHED = "trace_limit_reached"


class TraceLimitReached(SessionInterrupt):
    """The session's trace reached its size cap: the session stops here
    instead of spending on with no record, no transcript and no resume.

    A ``SessionInterrupt`` so tool dispatch re-raises it instead of handing
    it to the Agent as a tool's failure."""


# The session cap guards against a runaway writer, not against research. The
# 2026-11 traces measured 6.7-8.9 KB per model call, sub-agent calls
# included, so a session that spends those rounds' 19,200-call budget writes
# about 170 MB; the cap is six times that, about 120,000 calls at the highest
# rate measured.
TRACE_MAX_BYTES = 1024 * 1024 * 1024
TRACE_MAX_EVENT_BYTES = 256 * 1024
# The Agent-readable transcript beside the JSONL trace: one text block per
# event, every field clipped to the content preview size, rotated into parts
# under ``read_file``'s 10 MiB cap so every part stays readable.
TRANSCRIPT_PART_MAX_BYTES = 8 * 1024 * 1024
_TRANSCRIPT_TEXT_FIELDS = ("instruction", "content", "message", "summary", "task", "text", "error")
_TRANSCRIPT_JSON_FIELDS = ("arguments", "result", "value", "report", "compaction", "context_edit", "usage")
_TRANSCRIPT_HEADER_FIELDS = (
    ("call", "call_index"),
    ("tool", "tool"),
    ("id", "tool_call_id"),
    ("status", "status"),
    ("task", "task_id"),
    ("turn", "turn"),
)
# Fields the transcript leaves out: the system prompt is the Agent's own
# context already, the event ids are host-side bookkeeping, and a reply's
# reasoning is recorded for host-side review, not handed back to the Agent.
_TRANSCRIPT_OMITTED = frozenset(
    {"system_prompt", "event_type", "ts", "event_id", "reasoning_content"}
)
# What an over-sized event keeps. Identity plus these named fields survive
# verbatim; ``content`` (a model reply) keeps its own preview because the trace
# views read it back, and everything else — a ``tool_call``'s ``arguments`` and
# ``result`` above all — survives as one clipped JSON head, so an event with no
# ``content`` field still says what it did.
_TRACE_STUB_KEYS = ("status", "call_index", "tool", "tool_names")
TRACE_CONTENT_PREVIEW_CHARS = 32_000
TRACE_PAYLOAD_HEAD_CHARS = 8_000
_TRACE_LOCKS: dict[str, threading.Lock] = {}
_TRACE_LOCKS_GUARD = threading.Lock()


def agent_trace_path(artifacts_root: str | Path, run_id: str) -> Path:
    """Host-side per-run Agent trace file under an experiment's artifacts root.

    Single source for the writer and for the ledger's ``agent_trace_ref``, so a
    recorded reference always names the file the session actually wrote.
    """
    return Path(artifacts_root) / "traces" / f"{run_id}.jsonl"


def agent_transcript_dir(artifacts_root: str | Path) -> Path:
    """The Agent-readable transcripts of an experiment's research attempts.

    One text file per attempt, named by the run's opaque ref, appended by the
    same writer as the JSONL trace; the session's ``trace`` read root.
    """
    return Path(artifacts_root) / "transcripts"


def _transcript_target(directory: Path, run_ref: str) -> Path:
    """The transcript part to append to: a new part once the last one is full."""

    part = 1
    while True:
        name = f"{run_ref}.txt" if part == 1 else f"{run_ref}.part{part}.txt"
        candidate = directory / name
        try:
            size = candidate.stat().st_size
        except FileNotFoundError:
            return candidate
        if size < TRANSCRIPT_PART_MAX_BYTES:
            return candidate
        part += 1


def trace_reasoning(text: str | None) -> dict[str, str]:
    """A model call event's ``reasoning_content``: the reasoning text the
    provider returned, cut at the content preview size with the cut marked so
    one long chain of thought cannot push the event past
    ``TRACE_MAX_EVENT_BYTES``; no field when the provider returned none."""

    if not text or not text.strip():
        return {}
    if len(text) > TRACE_CONTENT_PREVIEW_CHARS:
        omitted = len(text) - TRACE_CONTENT_PREVIEW_CHARS
        text = f"{text[:TRACE_CONTENT_PREVIEW_CHARS]}\n[... {omitted} more characters not recorded]"
    return {"reasoning_content": text}


def _clip_transcript_text(text: str) -> str:
    if len(text) <= TRACE_CONTENT_PREVIEW_CHARS:
        return text
    omitted = len(text) - TRACE_CONTENT_PREVIEW_CHARS
    return f"{text[:TRACE_CONTENT_PREVIEW_CHARS]}\n[... {omitted} more characters in the host trace]"


def render_transcript_block(record: Mapping[str, object], *, omit: Collection[str] = ()) -> str:
    """One event as the Agent reads it: a header line, then its fields.

    Text fields are printed verbatim, structured fields as indented JSON so
    ``read_file``'s line paging and ``grep -C`` land on individual values, and
    whatever is left travels on one ``meta`` line. Every field is clipped at
    the trace's content preview size, and the block is the Agent-visible
    boundary: absolute host paths (an exception text, a host-side result
    reference) are redacted here, while the host JSONL keeps them.
    """

    header = f"=== {record.get('ts', '')} {record.get('event_type', '')}"
    for label, key in _TRANSCRIPT_HEADER_FIELDS:
        value = record.get(key)
        if value not in (None, ""):
            header += f" {label}={value}"
    lines = [header]
    skipped = set(_TRANSCRIPT_OMITTED) | set(omit) | {key for _label, key in _TRANSCRIPT_HEADER_FIELDS}
    meta: dict[str, object] = {}
    for key, value in record.items():
        if key in skipped:
            continue
        if key in _TRANSCRIPT_TEXT_FIELDS and isinstance(value, str):
            text = _clip_transcript_text(value)
            lines.append(f"{key}:\n{text}" if "\n" in text else f"{key}: {text}")
        elif key in _TRANSCRIPT_JSON_FIELDS and value is not None:
            rendered = json.dumps(value, ensure_ascii=False, indent=1, default=str)
            lines.append(f"{key}:\n{_clip_transcript_text(rendered)}")
        else:
            meta[key] = value
    if meta:
        lines.append(
            "meta: "
            + _clip_transcript_text(
                json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str)
            )
        )
    return redact_host_paths("\n".join(lines)) + "\n\n"


def _trace_payload_head(payload: dict[str, object]) -> str | None:
    """Clipped JSON of the payload fields the truncation stub does not keep."""

    body = {
        key: value
        for key, value in payload.items()
        if key not in _TRACE_STUB_KEYS and key not in ("content", "error")
    }
    if not body:
        return None
    head = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
    return head[:TRACE_PAYLOAD_HEAD_CHARS]


class AgentTraceWriter:
    """Bounded, redacted JSONL event stream for one Agent session.

    The session cap bounds one record, not two: once the JSONL is full it
    writes the ``trace_limit_reached`` marker, the Agent-readable transcript
    ends on the same marker, and the event that crossed the cap and every one
    after it raise :class:`TraceLimitReached`, which ends the session. The
    crossing event is not recorded, so a resume refuses to seed from such a
    trace (:func:`autotrade.pipelines.session_resume.resume_state`).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ids: dict[str, str],
        transcript_dir: str | Path | None = None,
        max_bytes: int = TRACE_MAX_BYTES,
        max_event_bytes: int = TRACE_MAX_EVENT_BYTES,
    ) -> None:
        if max_bytes <= 0 or max_event_bytes <= 0 or max_event_bytes > max_bytes:
            raise ValueError("trace size limits are invalid")
        self.path = Path(path)
        self.ids = dict(ids)
        # Where the Agent-readable transcript of this run goes, if anywhere;
        # its file is named by the run's opaque ref carried in ``ids``.
        self.transcript_dir = Path(transcript_dir) if transcript_dir is not None else None
        if self.transcript_dir is not None and not self.ids.get("run_id"):
            raise ValueError("a transcript needs the run's opaque ref in ids['run_id']")
        self.max_bytes = max_bytes
        self.max_event_bytes = max_event_bytes
        key = str(self.path.resolve())
        with _TRACE_LOCKS_GUARD:
            self._lock = _TRACE_LOCKS.setdefault(key, threading.Lock())
        self._full = False

    def emit(self, event_type: TraceEvent, payload: dict[str, object]) -> dict[str, object]:
        # A name the trace does not declare is refused before anything is
        # written: every reader of the trace reads the declared set.
        event_type = TraceEvent(event_type)
        sanitized = sanitize_for_log(payload)
        if not isinstance(sanitized, dict):
            raise TypeError("trace payload must sanitize to an object")
        record = {
            **sanitized,
            "event_type": str(event_type),
            "ts": utc_now_iso(),
            "event_id": new_id("event"),
            **self.ids,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)
        encoded = (line + "\n").encode("utf-8")
        if len(encoded) > self.max_event_bytes:
            record = {
                "event_type": str(event_type),
                "ts": record["ts"],
                "event_id": record["event_id"],
                **self.ids,
                "truncated": True,
                "original_bytes": len(encoded),
                **{key: record.get(key) for key in _TRACE_STUB_KEYS},
                "content_preview": str(record.get("content") or "")[:TRACE_CONTENT_PREVIEW_CHARS],
                "payload_head": _trace_payload_head(sanitized),
                "error": str(record.get("error") or "")[:4_000] or None,
            }
            encoded = (
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            if len(encoded) > self.max_event_bytes:
                record.pop("content_preview", None)
                record.pop("payload_head", None)
                record.pop("error", None)
                encoded = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)
                    + "\n"
                ).encode("utf-8")
            if len(encoded) > self.max_event_bytes:
                raise ValueError("trace identifiers exceed the per-event size limit")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.parent.chmod(0o700)
            if not self._full:
                current = self.path.stat().st_size if self.path.exists() else 0
                marker = {
                    "event_type": str(TraceEvent.TRACE_LIMIT_REACHED),
                    "ts": utc_now_iso(),
                    "event_id": new_id("event"),
                    **self.ids,
                    "max_bytes": self.max_bytes,
                }
                raw = (json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                # Every event leaves room for the marker, so a trace that
                # reached the cap always says so to the resume that refuses it.
                if current + len(encoded) + len(raw) > self.max_bytes:
                    if current + len(raw) <= self.max_bytes:
                        with self.path.open("ab") as handle:
                            handle.write(raw)
                            handle.flush()
                            os.fsync(handle.fileno())
                    # The transcript ends on the same marker: one bounded
                    # record of the attempt, not a host half and an Agent half
                    # that stop at different events.
                    self._append_transcript(marker)
                    self._full = True
            if self._full:
                raise TraceLimitReached(
                    f"the session trace reached its {self.max_bytes}-byte cap; the session "
                    "stops here rather than run on unrecorded"
                )
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self.path.chmod(0o600)
            self._append_transcript(record)
        return record

    def _append_transcript(self, record: dict[str, object]) -> None:
        if self.transcript_dir is None:
            return
        # Private like the JSONL beside it: the session reads it host-side
        # through the ``trace`` root, nothing else needs to.
        self.transcript_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.transcript_dir.chmod(0o700)
        target = _transcript_target(self.transcript_dir, str(self.ids["run_id"]))
        block = render_transcript_block(record, omit=self.ids)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(block)
        target.chmod(0o600)

