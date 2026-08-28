"""Semantic conversation compaction for long Agent sessions.

The runner does not proactively trim history or clear tool results. When the
estimated window crosses a threshold, or a request is forced because it does
not fit, this module asks a cheap no-thinking model for a Markdown
continuation summary (Pi's compaction shape), replaces older messages with it,
and keeps recent raw turns. One attempt per trigger: a failed compaction is
recorded and the runner continues with the emergency in-place tool-result
fitting, which is the fail-closed overflow recovery.
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
    estimate_chat_request_tokens,
)
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
)

COMPACT_SUMMARY_HEADINGS = (
    "## 目标",
    "## 约束与偏好",
    "## 进展",
    "### 已完成",
    "### 进行中",
    "### 受阻",
    "## 关键决定",
    "## 下一步",
    "## 关键上下文",
)
COMPACT_SYSTEM_PROMPT = (
    "You are a context compaction assistant for a quantitative-strategy coding "
    "Agent. Write a Markdown continuation summary with exactly these headings, in "
    "this order: " + " / ".join(COMPACT_SUMMARY_HEADINGS) + ". Keep exact file "
    "paths, commands, error strings, artifact ids, node ids, user constraints, "
    "numbers, and next steps; drop obsolete details; do not invent facts. When a "
    "previous summary is given, update it: keep everything still relevant, move "
    "finished items under 已完成, and add only what the new messages establish. "
    "Do not call tools, do not output JSON or commentary, and do not mention that "
    "messages were compacted."
)
_TOOL_CONTEXT_EXCERPT_CHARS = 500
_TOOL_CONTEXT_MIN_CHARS = 512
_FILES_TRAIL_LIMIT = 40
_READ_TOOLS = frozenset({"read_file", "grep", "glob"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "write_skill", "delete_skill"})
_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)


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

    def fresh(self) -> ContextCompactor:
        """A compactor over the same gateway and config with zero counters.

        One conversation, one compactor: the call cap and failure circuit are
        per conversation and the counters are not thread-safe, so a child
        session derives its own instance instead of sharing the parent's.
        """

        return ContextCompactor(self.llm, self.config)

    def should_compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        remaining_seconds: float = float("inf"),
        force: bool = False,
    ) -> tuple[bool, dict[str, object]]:
        """Return the decision and its auditable reason.

        ``force`` bypasses only size heuristics. Structural guards and the
        failure/call circuits remain active.
        """

        # The request the provider will see: messages plus the tool schemas,
        # estimated by the same bounded estimator as the send-time fit check.
        estimated_tokens = estimate_chat_request_tokens(messages, tools=tools)
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
        tools: Sequence[Mapping[str, object]] = (),
        remaining_seconds: float = float("inf"),
        step_id: str | None = None,
        force: bool = False,
    ) -> ContextCompactionResult | None:
        should_compact, decision = self.should_compact(
            messages, tools=tools, remaining_seconds=remaining_seconds, force=force
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
            summary_text = _extract_summary_text(response)
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
        non_summary = [
            message
            for message in compact_messages[1:]
            if not is_compaction_message(message)
        ]
        files = _merge_touched_files(
            _latest_compaction_files(compact_messages), _touched_files(non_summary)
        )
        summary_message = _build_compaction_summary_message(
            summary_text, self.compaction_count, files
        )
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
            "summary": summary_text,
            "files": files,
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
        transcript = json.dumps(
            sanitize_for_log([message.to_record() for message in messages_since_summary]),
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )
        previous_block = (
            f"## 上一份摘要\n{sanitize_for_log(previous_summary)}\n\n"
            if previous_summary
            else ""
        )
        return (
            ChatMessage("system", COMPACT_SYSTEM_PROMPT),
            ChatMessage(
                "user",
                f"{previous_block}## 此后的新消息（JSON 记录）\n{transcript}",
            ),
        )


def drop_leading_orphan_tools(seq: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Drop leading ``tool`` messages left without their ``assistant`` turn."""

    index = 0
    while index < len(seq) and seq[index].role == "tool":
        index += 1
    return list(seq[index:])


def drop_trailing_unanswered_tool_calls(seq: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Drop a trailing ``assistant`` turn whose tool calls have no results yet.

    A conversation snapshot taken while a batch is still running ends with the
    assistant's tool calls; forking it verbatim would hand a child a protocol
    invalid history. Tool results that follow their assistant turn are kept.
    """

    kept = list(seq)
    while kept and kept[-1].role == "assistant" and kept[-1].tool_calls:
        kept.pop()
    return kept


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
    return payload is not None and payload.get("observation") == "context_compaction"


def _compaction_payload(message: ChatMessage) -> dict[str, Any] | None:
    if message.role != "user":
        return None
    try:
        payload = json.loads(message.content or "")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_compaction_payload(messages: Sequence[ChatMessage]) -> dict[str, Any] | None:
    for message in reversed(messages[1:]):
        payload = _compaction_payload(message)
        if payload is not None and payload.get("observation") == "context_compaction":
            return payload
    return None


def _latest_compaction_summary(messages: Sequence[ChatMessage]) -> str:
    payload = _latest_compaction_payload(messages)
    if payload is None:
        return ""
    summary = payload.get("summary")
    return summary if isinstance(summary, str) else json.dumps(summary, ensure_ascii=False)


def _latest_compaction_files(messages: Sequence[ChatMessage]) -> dict[str, list[str]]:
    payload = _latest_compaction_payload(messages)
    files = payload.get("files") if payload is not None else None
    if not isinstance(files, dict):
        return {"read": [], "modified": []}
    return {
        key: [str(item) for item in files.get(key, []) if isinstance(item, str)]
        for key in ("read", "modified")
    }


def _touched_files(messages: Sequence[ChatMessage]) -> dict[str, list[str]]:
    """Files the summarized turns read or modified, from their tool calls.

    Survives across repeated compactions as a deterministic audit trail even
    though the summarized messages themselves are discarded.
    """

    read: list[str] = []
    modified: list[str] = []
    for message in messages:
        for call in message.tool_calls:
            path = call.arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            root = call.arguments.get("root")
            # Root and relative path, in the shape the tools accept (no colon syntax).
            label = f"[{root}] {path}" if isinstance(root, str) and root else path
            if call.name in _READ_TOOLS and label not in read:
                read.append(label)
            elif call.name in _WRITE_TOOLS and label not in modified:
                modified.append(label)
    return {"read": read, "modified": modified}


def _merge_touched_files(
    previous: dict[str, list[str]], current: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for key in ("read", "modified"):
        seen: list[str] = []
        for item in [*previous.get(key, []), *current.get(key, [])]:
            if item not in seen:
                seen.append(item)
        merged[key] = seen[-_FILES_TRAIL_LIMIT:]
    return merged


def _extract_summary_text(response: ProviderResponse) -> str:
    text = _THINK_BLOCK.sub("", response.content or "", count=1).strip()
    if not text:
        raise ValueError("compaction response is empty")
    return text


def _build_compaction_summary_message(
    summary_text: str, compaction_index: int, files: dict[str, list[str]]
) -> ChatMessage:
    payload = {
        "observation": "context_compaction",
        "summary_kind": "markdown",
        "compaction_index": compaction_index,
        "note": "Older raw messages were compacted; the summary is the retained context.",
        "summary": summary_text,
        "files": files,
    }
    return ChatMessage(
        "user", json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)
    )


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
