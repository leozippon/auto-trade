"""Conversation compaction for long Agent sessions.

The runner keeps deterministic observation summaries as a free fallback, but
long sessions need a semantic summary before expensive main-model calls.  This
module provides a small Claude-Code-inspired compaction layer: estimate the
current context window, call a cheap no-thinking model when the window is large,
replace old messages with a structured continuation state, and let the runner
record the audit event.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from autotrade.environment.llm import (
    ChatMessage,
    LLMProxy,
    ProviderResponse,
    context_request_fits,
)
from autotrade.environment.llm.extraction import ExtractionError, extract_json_object
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
)

COMPACT_SYSTEM_PROMPT = (
    "You are an anchored context compaction sub-agent. Return exactly one JSON "
    "object matching the requested schema. Do not call tools. Do not use "
    "markdown or commentary. Preserve exact file paths, commands, error "
    "strings, artifact ids, user constraints, and next steps. Avoid vague "
    "phrases and omit obsolete details. Do not mention that messages were "
    "compacted."
)
_TOOL_CONTEXT_EXCERPT_CHARS = 500
_TOOL_CONTEXT_MIN_CHARS = 512


@dataclass(frozen=True)
class ContextCompactionConfig:
    token_threshold: int = 200_000
    min_messages: int = 20
    keep_recent_messages: int = 12
    max_response_tokens: int = 1_600
    max_failures: int = 3
    max_calls: int = 8
    min_remaining_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.token_threshold <= 0:
            raise ValueError("token_threshold must be positive")
        if self.min_messages < 2:
            raise ValueError("min_messages must be at least 2")
        if self.keep_recent_messages < 1:
            raise ValueError("keep_recent_messages must be positive")
        if self.max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if self.max_failures < 0:
            raise ValueError("max_failures cannot be negative")
        if self.max_calls < 0:
            raise ValueError("max_calls cannot be negative")
        if self.min_remaining_seconds < 0:
            raise ValueError("min_remaining_seconds cannot be negative")


@dataclass(frozen=True)
class ContextCompactionResult:
    messages: tuple[ChatMessage, ...]
    event: dict[str, object]


class ContextCompactor(SessionTimeBudgetAware):
    """Semantic compactor that uses a dedicated low-cost LLM proxy."""

    def __init__(
        self, llm: LLMProxy, config: ContextCompactionConfig | None = None
    ) -> None:
        self.llm = llm
        self.config = config or ContextCompactionConfig()
        self._consecutive_failures = 0
        self.compaction_count = 0
        self.compaction_attempts = 0

    @property
    def session_time_budget(self) -> InferenceTimeBudget | None:
        if isinstance(self.llm, SessionTimeBudgetAware):
            return self.llm.session_time_budget
        return None

    def should_compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        remaining_seconds: float = float("inf"),
        force: bool = False,
    ) -> tuple[bool, dict[str, object]]:
        """Return the decision and its auditable reason.

        ``force`` bypasses only size heuristics. Structural guards and the
        failure/call circuits remain active.
        """

        estimated_tokens = estimate_messages_tokens(messages)
        non_summary_count = len(
            [message for message in messages[1:] if not is_compaction_message(message)]
        )
        reason = {
            "estimated_tokens": estimated_tokens,
            "token_threshold": self.config.token_threshold,
            "message_count": len(messages),
            "non_summary_message_count": non_summary_count,
            "keep_recent_messages": self.config.keep_recent_messages,
            "consecutive_failures": self._consecutive_failures,
            "compaction_attempts": self.compaction_attempts,
            "max_calls": self.config.max_calls,
        }
        if self.compaction_attempts >= self.config.max_calls:
            return False, {**reason, "skip_reason": "call_limit_reached"}
        if (
            self.config.max_failures
            and self._consecutive_failures >= self.config.max_failures
        ):
            return False, {**reason, "skip_reason": "failure_circuit_open"}
        if remaining_seconds < self.config.min_remaining_seconds:
            return False, {**reason, "skip_reason": "insufficient_remaining_time"}
        if len(messages) < self.config.min_messages:
            return False, {**reason, "skip_reason": "not_enough_messages"}
        if non_summary_count <= self.config.keep_recent_messages:
            return False, {**reason, "skip_reason": "nothing_to_compact"}
        if not force and estimated_tokens < self.config.token_threshold:
            return False, {**reason, "skip_reason": "below_token_threshold"}
        return True, {
            **reason,
            "trigger_reason": "forced_context_overflow"
            if force
            else "estimated_tokens",
        }

    def compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        remaining_seconds: float = float("inf"),
        step_id: str | None = None,
        force: bool = False,
    ) -> ContextCompactionResult | None:
        should_compact, decision = self.should_compact(
            messages, remaining_seconds=remaining_seconds, force=force
        )
        if not should_compact:
            return None

        started_at = datetime.now(UTC).isoformat()
        self.compaction_attempts += 1
        compact_messages, request_context_edit = self._fit_compact_request(messages)
        try:
            response = self.llm.complete(
                self._build_compact_request(compact_messages),
                tools=(),
                tool_choice="none",
                max_tokens=self.config.max_response_tokens,
            )
            summary_payload = _extract_summary_payload(response)
        except Exception as exc:  # noqa: BLE001 - compaction failure must not kill a Fold
            self._consecutive_failures += 1
            event = {
                **decision,
                "status": "error",
                "provider": getattr(self.llm, "provider", ""),
                "model": getattr(self.llm, "model", ""),
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "error": safe_error_summary(exc),
                "step_id_at_compaction": step_id,
            }
            if request_context_edit:
                event["request_context_edit"] = request_context_edit
            return ContextCompactionResult(
                messages=tuple(messages),
                event=event,
            )

        self._consecutive_failures = 0
        self.compaction_count += 1
        summary_message = _build_compaction_summary_message(
            summary_payload, self.compaction_count
        )
        non_summary = [
            message
            for message in compact_messages[1:]
            if not is_compaction_message(message)
        ]
        recent_messages = drop_leading_orphan_tools(
            non_summary[-self.config.keep_recent_messages :]
        )
        compacted_messages = (messages[0], summary_message, *recent_messages)
        event = {
            **decision,
            "status": "ok",
            "provider": getattr(self.llm, "provider", ""),
            "model": response.model,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "usage": dict(response.usage),
            "messages_before": len(messages),
            "messages_after": len(compacted_messages),
            "dropped_messages": max(len(messages) - len(compacted_messages), 0),
            "summary_chars": len(summary_message.content or ""),
            "summary": summary_payload,
            "compaction_index": self.compaction_count,
            "step_id_at_compaction": step_id,
        }
        if request_context_edit:
            event["request_context_edit"] = request_context_edit
        return ContextCompactionResult(messages=compacted_messages, event=event)

    def _fit_compact_request(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[ChatMessage], dict[str, object]]:
        """Bound a local compactor request without breaking tool-call turns."""

        prepared = list(messages)
        aggregate: dict[str, object] = {}
        while True:
            request = self._build_compact_request(prepared)
            fits, prompt_tokens, context_window = context_request_fits(
                self.llm,
                request,
                max_tokens=self.config.max_response_tokens,
            )
            if fits:
                if aggregate:
                    aggregate.update(
                        estimated_prompt_tokens_after=prompt_tokens,
                        context_window=context_window,
                    )
                return prepared, aggregate
            prepared, edit = fit_tool_results_to_context(
                self.llm,
                prepared,
                max_tokens=self.config.max_response_tokens,
                force=True,
            )
            if not edit:
                # The Provider performs the final fail-fast check. Returning the
                # unfit request here retains one authoritative error contract.
                return prepared, aggregate
            aggregate["summarized_tool_results"] = int(
                aggregate.get("summarized_tool_results", 0)
            ) + int(edit["summarized_tool_results"])
            aggregate["chars_freed"] = int(aggregate.get("chars_freed", 0)) + int(
                edit["chars_freed"]
            )

    def _build_compact_request(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[ChatMessage, ChatMessage]:
        previous_summary = _latest_compaction_summary(messages)
        messages_since_summary = [
            message for message in messages[1:] if not is_compaction_message(message)
        ]
        compact_input = {
            "instructions": (
                "Update the continuation state for a coding/trading Agent. Treat "
                "previous_summary as the current anchor when present, merge in only "
                "new information from messages_since_previous_summary, remove stale "
                "or superseded details, and do not invent facts."
            ),
            "previous_summary": sanitize_for_log(previous_summary),
            "output_schema": {
                "goal": "string",
                "constraints_and_preferences": ["string"],
                "progress": {
                    "done": ["string"],
                    "in_progress": ["string"],
                    "blocked": ["string"],
                },
                "key_decisions": ["string"],
                "errors_and_fixes": ["string"],
                "next_steps": ["string"],
                "critical_context": ["string"],
                "relevant_files": ["string"],
                "recent_user_feedback": ["string"],
            },
            "messages_since_previous_summary": sanitize_for_log(
                [_message_record(message) for message in messages_since_summary]
            ),
        }
        return (
            ChatMessage("system", COMPACT_SYSTEM_PROMPT),
            ChatMessage(
                "user",
                json.dumps(
                    compact_input, ensure_ascii=False, default=str, allow_nan=False
                ),
            ),
        )


def drop_leading_orphan_tools(seq: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Drop leading ``tool`` messages left without their ``assistant`` turn."""

    index = 0
    while index < len(seq) and seq[index].role == "tool":
        index += 1
    return list(seq[index:])


def estimate_messages_tokens(messages: Sequence[ChatMessage]) -> int:
    """Conservative rough token estimate based on serialized message content."""

    total_chars = 0
    for message in messages:
        total_chars += len(message.role)
        total_chars += len(message.content or "")
        total_chars += len(message.reasoning_content or "")
        if message.tool_calls:
            total_chars += len(
                json.dumps(
                    [call.to_record() for call in message.tool_calls],
                    ensure_ascii=False,
                    default=str,
                )
            )
        total_chars += 8
    return max(1, int((total_chars / 4.0) * (4.0 / 3.0)))


def fit_tool_results_to_context(
    llm: object,
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[Mapping[str, object]] = (),
    max_tokens: int,
    force: bool = False,
) -> tuple[list[ChatMessage], dict[str, object]]:
    """Summarize the largest tool results until one request fits.

    Assistant tool calls and tool message ids remain untouched, so the native
    function-call protocol stays valid.  The deterministic replacement keeps
    provenance and bounded evidence excerpts while explicitly marking that the
    source was omitted. ``force`` guarantees one edit when possible and is
    used only for recovery after an authoritative provider overflow.
    """

    edited = list(messages)
    fits_before, prompt_before, context_window = context_request_fits(
        llm, edited, tools=tools, max_tokens=max_tokens
    )
    candidates = sorted(
        (
            (len(message.content or ""), index)
            for index, message in enumerate(edited)
            if message.role == "tool"
            and len(message.content or "") >= _TOOL_CONTEXT_MIN_CHARS
        ),
        reverse=True,
    )
    summarized = 0
    chars_freed = 0
    fits_after = fits_before
    prompt_after = prompt_before
    for _size, index in candidates:
        if fits_after and (not force or summarized > 0):
            break
        original = edited[index]
        replacement = summarize_tool_result_for_context(original)
        original_chars = len(original.content or "")
        replacement_chars = len(replacement.content or "")
        if replacement_chars >= original_chars:
            continue
        edited[index] = replacement
        summarized += 1
        chars_freed += original_chars - replacement_chars
        fits_after, prompt_after, context_window = context_request_fits(
            llm, edited, tools=tools, max_tokens=max_tokens
        )
    if not summarized:
        return edited, {}
    return edited, {
        "reason": "context_window_budget",
        "summarized_tool_results": summarized,
        "chars_freed": chars_freed,
        "estimated_prompt_tokens_before": prompt_before,
        "estimated_prompt_tokens_after": prompt_after,
        "requested_max_output_tokens": max_tokens,
        "context_window": context_window,
        "fits_after": fits_after,
    }


def summarize_tool_result_for_context(message: ChatMessage) -> ChatMessage:
    if message.role != "tool" or message.tool_call_id is None:
        raise ValueError("only a protocol-bound tool result can be summarized")
    content = message.content or ""
    retained_fields = _retained_tool_fields(content)
    payload: dict[str, object] = {
        "observation": "context_tool_result_summary",
        "note": (
            "The exact tool result was omitted from model context to stay within the "
            "model window. Re-run a narrower paginated query when more detail is needed."
        ),
        "original_chars": len(content),
        "source_omitted": True,
        "head": content[:_TOOL_CONTEXT_EXCERPT_CHARS],
        "tail": content[-_TOOL_CONTEXT_EXCERPT_CHARS:],
    }
    if retained_fields:
        payload["retained_fields"] = retained_fields
    return ChatMessage(
        "tool",
        json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False),
        tool_call_id=message.tool_call_id,
    )


def _retained_tool_fields(content: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    retained: dict[str, object] = {}
    allowed = {
        "command_kind",
        "complete",
        "error",
        "node_id",
        "offset",
        "path",
        "revision_id",
        "rows",
        "status",
        "stderr",
        "timed_out",
        "total",
    }

    def visit(value: object, prefix: str = "") -> None:
        if len(retained) >= 20 or not isinstance(value, dict):
            return
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in allowed and item not in (None, "", [], {}):
                retained[path] = _bounded_context_value(item)
                if len(retained) >= 20:
                    return
            if isinstance(item, dict):
                visit(item, path)

    visit(payload)
    return retained


def _bounded_context_value(value: object) -> object:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 300 else text[:297] + "..."


def is_compaction_message(message: ChatMessage) -> bool:
    payload = _compaction_payload(message)
    return payload is not None and payload.get("observation") in {
        "context_summary",
        "context_compaction",
    }


def is_llm_compaction_message(message: ChatMessage) -> bool:
    payload = _compaction_payload(message)
    return payload is not None and payload.get("observation") == "context_compaction"


def _compaction_payload(message: ChatMessage) -> dict[str, Any] | None:
    if message.role != "user":
        return None
    try:
        payload = json.loads(message.content or "")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_compaction_summary(messages: Sequence[ChatMessage]) -> object | None:
    for message in reversed(messages[1:]):
        payload = _compaction_payload(message)
        if payload is None or payload.get("observation") not in {
            "context_summary",
            "context_compaction",
        }:
            continue
        return payload.get("summary", payload)
    return None


def _message_record(message: ChatMessage) -> dict[str, object]:
    return message.to_record()


def _extract_summary_payload(response: ProviderResponse) -> dict[str, object]:
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError:
        try:
            payload = extract_json_object(response.content).payload
        except ExtractionError as exc:
            raise ValueError("compaction response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("compaction response must be a JSON object")
    return _normalize_summary_payload(payload)


def _normalize_summary_payload(payload: dict[str, Any]) -> dict[str, object]:
    if not any(
        field in payload
        for field in (
            "goal",
            "progress",
            "key_decisions",
            "next_steps",
            "critical_context",
        )
    ):
        raise ValueError(
            "compaction response carries none of the requested summary fields"
        )
    progress = (
        payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    )
    return {
        "goal": _as_text(payload.get("goal")),
        "constraints_and_preferences": _as_list(
            payload.get("constraints_and_preferences")
        ),
        "progress": {
            "done": _as_list(progress.get("done")),
            "in_progress": _as_list(progress.get("in_progress")),
            "blocked": _as_list(progress.get("blocked")),
        },
        "key_decisions": _as_list(payload.get("key_decisions")),
        "errors_and_fixes": _as_list(payload.get("errors_and_fixes")),
        "next_steps": _as_list(payload.get("next_steps")),
        "critical_context": _as_list(payload.get("critical_context")),
        "relevant_files": _as_list(payload.get("relevant_files")),
        "recent_user_feedback": _as_list(payload.get("recent_user_feedback")),
    }


def _build_compaction_summary_message(
    summary_payload: dict[str, object], compaction_index: int
) -> ChatMessage:
    payload = {
        "observation": "context_compaction",
        "summary_kind": "llm_compact_summary",
        "compaction_index": compaction_index,
        "note": "Older raw messages were compacted; the summary is the retained context.",
        "summary": summary_payload,
    }
    return ChatMessage(
        "user", json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)
    )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_text(item) for item in value]
    return [_as_text(value)]


def safe_error_summary(exc: Exception, max_chars: int = 500) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", text)
    sanitized = sanitize_for_log(text)
    if not isinstance(sanitized, str):
        sanitized = str(sanitized)
    return (
        sanitized if len(sanitized) <= max_chars else sanitized[: max_chars - 3] + "..."
    )
