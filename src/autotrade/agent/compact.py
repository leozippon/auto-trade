"""Conversation compaction for the long research session.

Two triggers rebuild the conversation as system prompt + one summary message
+ the most recent messages. The Agent's own ``compact`` tool hands in the
summary it wrote (``compact_with_summary``); the runtime safety net asks a
cheap no-thinking model for a Markdown continuation summary (Pi's compaction
shape) when the estimated window crosses the threshold, or when a request is
forced because it does not fit (``ContextCompactor``). One attempt per
runtime trigger: a failed compaction is recorded and the conversation
continues with the emergency in-place tool-result fitting, the fail-closed
overflow recovery. ``ContextFitter`` is that discipline around each model
call, one implementation for the session loop and every sub-agent.

Compaction rewrites history only: the conversation's system prompt is carried
over verbatim, so the session (or child) keeps the exact contract it was
composed with and the provider's prefix cache still covers it.

What compaction drops is not lost: every event of the session is in the
Agent-readable transcript under the ``trace`` root, and the summary message
says so, so the Agent recovers exact earlier content with the same
``read_file``/``grep`` calls it uses for any other file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from autotrade.environment.llm import (
    AGENT_MAX_OUTPUT_TOKENS,
    CONTEXT_OUTPUT_MIN_TOKENS,
    CONTEXT_OUTPUT_TOKEN_MARGIN,
    MIN_CONTEXT_WINDOW_TOKENS,
    ChatMessage,
    LLMProxy,
    ProviderResponse,
    clamp_requested_max_tokens,
    context_overflow_error,
    context_request_fits,
    estimate_chat_request_tokens,
    output_shares_context_window,
)
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
)
from autotrade.environment.tools.compact import COMPACT_SUMMARY_MAX_CHARS
from autotrade.environment.trace import TraceEvent

if TYPE_CHECKING:
    from autotrade.environment.tools.base import ToolResultStore

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
    "The system prompt (contract, run facts, directive) is never dropped by "
    "compaction; do not restate it. Record only session state: rounds, node ids, "
    "results, decisions, open work. Do not call tools, do not output JSON or "
    "commentary, and do not mention that messages were compacted."
)
_TOOL_CONTEXT_EXCERPT_CHARS = 500
_TOOL_CONTEXT_MIN_CHARS = 512
_FILES_TRAIL_LIMIT = 40
_READ_TOOLS = frozenset({"read_file", "grep", "glob"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "write_skill", "delete_skill"})
_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
# Fallback threshold for a session assembled without the worker's per-role
# derivation: the smallest catalog window minus the agent output ceiling and
# the gateway's tokenizer slack, so proactive compaction still triggers on the
# narrowest supported model instead of never firing.
DEFAULT_COMPACTION_TOKEN_THRESHOLD = (
    MIN_CONTEXT_WINDOW_TOKENS - AGENT_MAX_OUTPUT_TOKENS - CONTEXT_OUTPUT_TOKEN_MARGIN
)


@dataclass(frozen=True)
class ContextCompactionConfig:
    # The experiment worker always derives the operational threshold from this
    # run's model windows (``window − role output budget − margin``); the
    # default only serves a session assembled without that derivation.
    token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD
    # ``min_messages`` and ``max_calls`` bound threshold-triggered compaction
    # only; a forced compaction (the next request does not fit) proceeds past
    # both while messages older than the retained tail remain.
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
        self,
        llm: LLMProxy,
        config: ContextCompactionConfig | None = None,
        *,
        result_store: ToolResultStore | None = None,
        trace_ref: str | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or ContextCompactionConfig()
        # Where an oversized tool body the compactor's own request cannot hold
        # goes so the session can read it back: the search tools' spill store.
        self.result_store = result_store
        # The run's opaque ref: names the transcript the summary points at.
        self.trace_ref = trace_ref
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

        return ContextCompactor(
            self.llm,
            self.config,
            result_store=self.result_store,
            trace_ref=self.trace_ref,
        )

    def should_compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        remaining_seconds: float = float("inf"),
        force: bool = False,
    ) -> tuple[bool, dict[str, object]]:
        """Return the decision and its auditable reason.

        ``force`` is the send-side request: the next request does not fit, so
        compaction is the only way forward. It bypasses the size heuristics —
        the token threshold, the call cap and the minimum message count. The
        structural guard (nothing older than the retained tail) and the
        failure and remaining-time circuits stay: they say compaction cannot
        help, not that it is not yet warranted.
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
        if not force and self.compaction_attempts >= self.config.max_calls:
            return False, {**reason, "skip_reason": "call_limit_reached"}
        if (
            self.config.max_failures
            and self._consecutive_failures >= self.config.max_failures
        ):
            return False, {**reason, "skip_reason": "failure_circuit_open"}
        if remaining_seconds < self.config.min_remaining_seconds:
            return False, {**reason, "skip_reason": "insufficient_remaining_time"}
        if not force and len(messages) < self.config.min_messages:
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
        force: bool = False,
    ) -> ContextCompactionResult | None:
        # Every caller starts its conversation with the system prompt and the
        # whole module treats ``messages[0]`` as that prompt: it is the one
        # message compaction keeps verbatim, and ``messages[1:]`` is what gets
        # summarized. A conversation shaped otherwise would silently summarize
        # the session's contract away, so it is refused here.
        if not messages or messages[0].role != "system":
            raise ValueError(
                "compaction requires a conversation whose first message is the system prompt"
            )
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
            limit = self.config.max_response_tokens
            if output_truncated(response.usage, limit):
                # Stopped at the output budget, its last line unfinished.
                summary_text = _cut_summary(
                    summary_text,
                    len(summary_text) - 1,
                    f"[Cut here at the compaction model's {limit}-token output "
                    "limit: the rest of this summary was never written.]",
                )
        except Exception as exc:  # noqa: BLE001 - compaction failure must not kill a session
            self._consecutive_failures += 1
            event = {
                **decision,
                "status": "error",
                "provider": self.llm.provider,
                "model": self.llm.model,
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "error": safe_error_summary(exc),
            }
            if request_context_edit:
                event["request_context_edit"] = request_context_edit
            return ContextCompactionResult(
                messages=tuple(messages),
                event=event,
            )

        self._consecutive_failures = 0
        self.compaction_count += 1
        compacted_messages, files = _rebuild(
            messages,
            summary_text,
            keep_recent_messages=self.config.keep_recent_messages,
            kind="model",
            trace_ref=self.trace_ref,
        )
        event = {
            **decision,
            "status": "ok",
            "provider": self.llm.provider,
            "model": response.model,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "usage": dict(response.usage),
            "messages_before": len(messages),
            "messages_after": len(compacted_messages),
            "dropped_messages": max(len(messages) - len(compacted_messages), 0),
            "summary_chars": len(summary_text),
            "summary": summary_text,
            "files": files,
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
                result_store=self.result_store,
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
                f"## 篇幅上限\n{self.config.max_response_tokens} 个输出 token，"
                "超出的部分会被截掉。\n\n"
                f"{previous_block}## 此后的新消息（JSON 记录）\n{transcript}",
            ),
        )


@dataclass(frozen=True)
class ContextFitter:
    """One conversation's context-window discipline around each model call:
    the session loop and every sub-agent run this one implementation.

    ``record`` receives what was done to the conversation under the session
    loop's event names (``context_compaction``, ``context_edit``,
    ``context_output_clamped``); a sub-agent files them under its identity.
    """

    llm: object
    # The configured output ceiling every fit is measured against.
    max_tokens: int
    compactor: ContextCompactor | None
    result_store: ToolResultStore | None
    record: Callable[[TraceEvent, dict[str, object]], None]

    def compact(
        self,
        messages: list[ChatMessage],
        tools: Sequence[Mapping[str, object]],
        remaining_seconds: float,
        *,
        force: bool = False,
    ) -> tuple[list[ChatMessage], bool]:
        """Runtime compaction at the threshold, or forced when the next request
        does not fit; returns the conversation and whether it was compacted."""

        if self.compactor is None:
            return messages, False
        result = self.compactor.compact(
            messages, tools=tools, remaining_seconds=remaining_seconds, force=force
        )
        if result is None:
            return messages, False
        self.record(TraceEvent.CONTEXT_COMPACTION, dict(result.event))
        # An ``ok`` compaction always drops messages older than the retained
        # tail; a failed one hands the history back unchanged.
        return list(result.messages), result.event.get("status") == "ok"

    def fit(
        self,
        messages: list[ChatMessage],
        tools: Sequence[Mapping[str, object]],
        remaining_seconds: float,
    ) -> tuple[list[ChatMessage], int]:
        """Fit the next request into the window; return it with its output cap.

        A request whose full output budget does not fit is compacted (forced),
        then its largest tool results are summarized in place. What still does
        not fit follows the gateway's own line: it goes out with a smaller
        output ceiling while ``CONTEXT_OUTPUT_MIN_TOKENS`` remain after the
        margin, and otherwise fails here, before the network, with the
        gateway's overflow error — a request known not to fit is never sent.
        The cap is the gateway's clamp of the configured ceiling, so the caller
        sends it as is and a reply cut at a lowered cap still reads as
        truncated.
        """

        requested = self.max_tokens
        fits, prompt_tokens, context_window = context_request_fits(
            self.llm, messages, tools=tools, max_tokens=requested
        )
        if not fits:
            messages, _ = self.compact(messages, tools, remaining_seconds, force=True)
            fits, prompt_tokens, context_window = context_request_fits(
                self.llm, messages, tools=tools, max_tokens=requested
            )
        if not fits:
            messages, edit = fit_tool_results_to_context(
                self.llm,
                messages,
                tools=tools,
                max_tokens=requested,
                result_store=self.result_store,
            )
            if edit:
                self.record(TraceEvent.CONTEXT_EDIT, edit)
            fits, prompt_tokens, context_window = context_request_fits(
                self.llm, messages, tools=tools, max_tokens=requested
            )
        available, prompt_fits = clamp_requested_max_tokens(
            requested_max_tokens=requested,
            estimated_prompt_tokens=prompt_tokens,
            context_window=context_window,
            output_shares_window=output_shares_context_window(self.llm),
        )
        if fits:
            return messages, available
        # Nothing more can be freed. The gateway refuses only a prompt that
        # leaves no room, so the send/fail line is its clamp, with a usable
        # minimum output on top.
        if prompt_fits and available >= CONTEXT_OUTPUT_MIN_TOKENS:
            self.record(
                TraceEvent.CONTEXT_OUTPUT_CLAMPED,
                {
                    "estimated_prompt_tokens": prompt_tokens,
                    "context_window": context_window,
                    "requested_max_output_tokens": requested,
                    "available_output_tokens": available,
                },
            )
            return messages, available
        raise context_overflow_error(
            estimated_prompt_tokens=prompt_tokens,
            max_tokens=requested,
            context_window=context_window,
        )

    def recover(
        self,
        messages: list[ChatMessage],
        tools: Sequence[Mapping[str, object]],
        remaining_seconds: float,
    ) -> tuple[list[ChatMessage], bool]:
        """The recovery after an overflow error, the provider's or ``fit``'s
        own: forced compaction, then at least one tool-result summary unless
        the compaction already changed the request. Returns whether anything
        changed; the caller allows it once per conversation."""

        recovered, compacted = self.compact(
            messages, tools, remaining_seconds, force=True
        )
        recovered, edit = fit_tool_results_to_context(
            self.llm,
            recovered,
            tools=tools,
            max_tokens=self.max_tokens,
            force=not compacted,
            result_store=self.result_store,
        )
        if edit:
            self.record(
                TraceEvent.CONTEXT_EDIT,
                {**edit, "reason": "provider_context_overflow_recovery"},
            )
        return recovered, compacted or bool(edit)


def compact_with_summary(
    messages: Sequence[ChatMessage],
    summary: str,
    *,
    keep_recent_messages: int,
    trace_ref: str | None = None,
) -> ContextCompactionResult:
    """The Agent's own compaction: its summary replaces everything but the
    system prompt and the most recent messages, exactly as the runtime
    compactor rebuilds a conversation, without a model call."""

    if not messages or messages[0].role != "system":
        raise ValueError(
            "compaction requires a conversation whose first message is the system prompt"
        )
    text = summary.strip()
    if not text:
        raise ValueError("an Agent compaction needs a non-empty summary")
    if len(text) > COMPACT_SUMMARY_MAX_CHARS:
        text = _cut_summary(
            text,
            COMPACT_SUMMARY_MAX_CHARS,
            f"[Cut here at the {COMPACT_SUMMARY_MAX_CHARS}-character limit: the rest "
            "of this summary is only in the trace, in this compact call's arguments.]",
        )
    compacted, files = _rebuild(
        messages,
        text,
        keep_recent_messages=keep_recent_messages,
        kind="agent",
        trace_ref=trace_ref,
    )
    return ContextCompactionResult(
        messages=compacted,
        event={
            "status": "ok",
            "messages_before": len(messages),
            "messages_after": len(compacted),
            "dropped_messages": max(len(messages) - len(compacted), 0),
            "summary_chars": len(text),
            "summary": text,
            "files": files,
        },
    )


def _cut_summary(text: str, max_chars: int, notice: str) -> str:
    """``text`` cut back to its last line ending within ``max_chars``, then
    ``notice``: a summary stopped at its writer's limit must not read as the
    whole of it."""

    line_end = text.rfind("\n", 0, max_chars + 1)
    kept = text[:line_end] if line_end > 0 else text[:max_chars]
    return f"{kept.rstrip()}\n{notice}"


def output_truncated(usage: object, max_tokens: int) -> bool:
    """True when a reply used its whole completion budget (finish_reason=length
    is not surfaced by the transport, so the usage count is the signal).

    Shared by the session runner, its sub-agents and the runtime compactor.
    """

    if not isinstance(usage, Mapping):
        return False
    completion = usage.get("completion_tokens")
    return (
        isinstance(completion, (int, float))
        and not isinstance(completion, bool)
        and completion >= max_tokens
    )


def _rebuild(
    messages: Sequence[ChatMessage],
    summary_text: str,
    *,
    keep_recent_messages: int,
    kind: str,
    trace_ref: str | None,
) -> tuple[tuple[ChatMessage, ...], dict[str, list[str]]]:
    """System prompt, one summary message, then the retained tail.

    An earlier summary message is superseded, never retained; the files trail
    it carried is merged into the new one.
    """

    non_summary = [message for message in messages[1:] if not is_compaction_message(message)]
    files = _merge_touched_files(_latest_compaction_files(messages), _touched_files(non_summary))
    recent = drop_leading_orphan_tools(non_summary[-keep_recent_messages:])
    summary_message = compaction_summary_message(
        summary_text, files, kind=kind, trace_ref=trace_ref
    )
    # The system prompt is reused as the same object, never re-rendered.
    return (messages[0], summary_message, *recent), files


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
    result_store: ToolResultStore | None = None,
) -> tuple[list[ChatMessage], dict[str, object]]:
    """Summarize the largest tool results until one request fits.

    Assistant tool calls and tool message ids remain untouched, so the native
    function-call protocol stays valid.  The deterministic replacement keeps
    provenance and bounded evidence excerpts while explicitly marking that the
    source was omitted. ``force`` guarantees one edit when possible and is
    used only for recovery after an authoritative provider overflow.
    ``result_store`` is the session's spill store: given one, the replaced body
    is persisted and referenced instead of being dropped.
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
        original_chars = len(original.content or "")
        replacement = summarize_tool_result_for_context(original)
        replacement_chars = len(replacement.content or "")
        if replacement_chars >= original_chars:
            continue
        if result_store is not None:
            # Spill only a body this edit actually removes, and only while the
            # reference still leaves a saving: a replacement decided first
            # means no unreferenced file is written for one that is discarded.
            spilled = summarize_tool_result_for_context(
                original, result_store=result_store
            )
            spilled_chars = len(spilled.content or "")
            if spilled_chars < original_chars:
                replacement, replacement_chars = spilled, spilled_chars
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


def summarize_tool_result_for_context(
    message: ChatMessage, *, result_store: ToolResultStore | None = None
) -> ChatMessage:
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
    if result_store is not None:
        # The body leaves the conversation, not the session: spilled through
        # the same store the tools themselves use, it is read back by root and
        # path rather than re-produced by a narrower query.
        payload.update(
            result_store.store_tool_result(
                tool="context_edit", kind="tool_result", content=content
            )
        )
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
    """The continuation summary a compaction response carries, or a failure.

    A compaction replaces the conversation, so what comes back has to be the
    summary that was asked for. A gateway that answers an over-sized request
    with its own error envelope as a normal completion (observed in a live
    session) would otherwise become the whole retained context under a
    ``status: "ok"`` record; an answer that carries none of the requested
    headings is not a summary, and the compaction fails instead — the caller
    keeps the history and falls back to in-place tool-result fitting.
    """

    text = _THINK_BLOCK.sub("", response.content or "", count=1).strip()
    if not text:
        raise ValueError("compaction response is empty")
    if not any(heading in text for heading in COMPACT_SUMMARY_HEADINGS):
        raise ValueError(
            "compaction response is not a continuation summary: " + text[:200]
        )
    return text


def compaction_summary_message(
    summary_text: str,
    files: dict[str, list[str]] | None = None,
    *,
    kind: str,
    trace_ref: str | None,
) -> ChatMessage:
    """The one user message a compaction leaves in the conversation.

    ``kind`` says who wrote the summary: ``agent`` (its own ``compact`` call),
    ``model`` (the runtime compactor) or ``resume`` (an interrupted attempt's
    last summary, handed to the attempt that continues it).
    """

    payload: dict[str, object] = {
        "observation": "context_compaction",
        "summary_kind": kind,
        "note": (
            "Older messages were replaced by this summary; the summary and the most "
            "recent messages are the retained context."
        ),
        "summary": summary_text,
        "files": files if files is not None else {"read": [], "modified": []},
        "trace": {
            "root": "trace",
            "path": f"{trace_ref}.txt" if trace_ref else "<run_ref>.txt",
            "hint": (
                "The whole history of this session, every attempt and sub-agent included, "
                "is readable there (`.partN.txt` once long); grep or read_file it to recover "
                "exact tool output, code you wrote or numbers instead of working from the "
                "summary alone."
            ),
        },
    }
    return ChatMessage(
        "user", json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)
    )


def safe_error_summary(exc: BaseException, max_chars: int = 500) -> str:
    sanitized = sanitize_for_log(f"{type(exc).__name__}: {exc}")
    if not isinstance(sanitized, str):
        sanitized = str(sanitized)
    return (
        sanitized if len(sanitized) <= max_chars else sanitized[: max_chars - 3] + "..."
    )
