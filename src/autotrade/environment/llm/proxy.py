"""Host-side LLM Proxy boundary (docs/environment-design.md §2.4).

All provider calls go through an :class:`LLMProxy`. API keys live only on the
host side inside provider adapters; sandbox code never sees them. Callers are
responsible for writing call details to the documented locations (main
conversation -> agent_trace.jsonl ``llm_call`` events; NL Sub Agent ->
``nl_tool/nl_llm_calls.jsonl``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

_ASCII_CHARS_PER_TOKEN = 3
_CHAT_REQUEST_FRAMING_TOKENS = 64
CONTEXT_OUTPUT_TOKEN_MARGIN = 2_048
CONTEXT_OUTPUT_MIN_TOKENS = 256
CONTEXT_OUTPUT_MAX_SHRINKS = 8
_OPAQUE_ASCII_RUN_MIN_CHARS = 256
_OPAQUE_ASCII_RUN = re.compile(
    rf"[A-Za-z0-9+/=_-]{{{_OPAQUE_ASCII_RUN_MIN_CHARS},}}"
)
_OPAQUE_ASCII_WRAPPED_CANDIDATE = re.compile(
    r"(?:[A-Za-z0-9+/=_-]|[ \t\f\v\r\n]|\\r\\n|\\[nr])+"
)
_OPAQUE_ASCII_SEPARATOR = re.compile(r"\\r\\n|\\[nr]|\r\n|[ \t\f\v\r\n]")
_OPAQUE_ASCII_MIN_ALPHABET_PERCENT = 93
_OPAQUE_ASCII_MIN_CHARS_PER_SEPARATOR = 16


class LLMProxyError(RuntimeError):
    def __init__(
        self, message: str, *, retryable: bool = False, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class MalformedToolCallError(LLMProxyError):
    """A tool call the model itself emitted unparseable.

    Model output, not transport: replaying the request repeats the whole
    generation for the same defect, so it is never retryable. ``content`` and
    ``reasoning_content`` carry the assistant text that did arrive, so the
    caller can keep the analysis in the conversation and ask only for the call
    to be re-issued.
    """

    def __init__(
        self,
        message: str,
        *,
        content: str = "",
        reasoning_content: str | None = None,
    ) -> None:
        super().__init__(message, retryable=False)
        self.content = content
        self.reasoning_content = reasoning_content


def _json_object(value: Mapping[str, object]) -> dict[str, object]:
    try:
        normalized = json.loads(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible") from exc
    if not isinstance(normalized, dict):
        raise TypeError("value must be a JSON object")
    return normalized


def _iter_json_strings(value: object):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_json_strings(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_json_strings(item)


def _opaque_ascii_chars(value: str) -> int:
    spans = [match.span() for match in _OPAQUE_ASCII_RUN.finditer(value)]
    for match in _OPAQUE_ASCII_WRAPPED_CANDIDATE.finditer(value):
        candidate = match.group(0)
        separators = tuple(_OPAQUE_ASCII_SEPARATOR.finditer(candidate))
        if not separators:
            continue
        payload_chars = len(_OPAQUE_ASCII_SEPARATOR.sub("", candidate))
        if payload_chars < _OPAQUE_ASCII_RUN_MIN_CHARS:
            continue
        if (
            payload_chars * 100
            < len(candidate) * _OPAQUE_ASCII_MIN_ALPHABET_PERCENT
        ):
            continue
        if (
            payload_chars
            < len(separators) * _OPAQUE_ASCII_MIN_CHARS_PER_SEPARATOR
        ):
            continue
        spans.append(match.span())
    if not spans:
        return 0
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum(
        len(_OPAQUE_ASCII_SEPARATOR.sub("", value[start:end]))
        for start, end in merged
    )


def _opaque_ascii_adjustment(values: object) -> int:
    adjustment = 0
    for value in _iter_json_strings(values):
        opaque_chars = _opaque_ascii_chars(value)
        adjustment += opaque_chars - (
            opaque_chars + _ASCII_CHARS_PER_TOKEN - 1
        ) // _ASCII_CHARS_PER_TOKEN
    return adjustment


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError("tool call id and name must be non-empty")
        object.__setattr__(self, "arguments", _json_object(self.arguments))

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role: {self.role}")
        if self.role == "tool" and (not self.tool_call_id or self.content is None):
            raise ValueError("tool messages require tool_call_id and content")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")
        if self.role != "assistant" and self.reasoning_content is not None:
            raise ValueError("only assistant messages may contain reasoning content")
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("reasoning content must be a string")
        if self.content is None and not self.tool_calls:
            raise ValueError("message must contain content or tool calls")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            record["tool_calls"] = [call.to_record() for call in self.tool_calls]
        if self.tool_call_id:
            record["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            record["reasoning_content"] = self.reasoning_content
        return record


MALFORMED_TOOL_CALL_HINT = (
    "Re-issue only that tool call, with a valid name and valid JSON arguments. "
    "The analysis you already wrote is kept above; do not repeat it."
)


def malformed_tool_call_messages(
    exc: MalformedToolCallError, *, error: str
) -> list[ChatMessage]:
    """The conversation turn that recovers one malformed tool call.

    The assistant text that did arrive is kept as its own message (a rejected
    trailing tool call must not cost a whole analysis), followed by the
    observation naming the defect and asking for the call again.
    """

    messages: list[ChatMessage] = []
    if exc.content or exc.reasoning_content:
        messages.append(
            ChatMessage(
                "assistant", exc.content, reasoning_content=exc.reasoning_content
            )
        )
    messages.append(
        ChatMessage(
            "user",
            json.dumps(
                {
                    "observation": "malformed_tool_call",
                    "error": error,
                    "retry_hint": MALFORMED_TOOL_CALL_HINT,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
    )
    return messages


@dataclass(frozen=True)
class ProviderResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    usage: Mapping[str, object] = field(default_factory=dict)
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _json_object(self.usage))
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("reasoning content must be a string")
        if not self.content and not self.tool_calls:
            if isinstance(self.reasoning_content, str) and self.reasoning_content.strip():
                object.__setattr__(self, "content", "")
            else:
                raise ValueError("provider response must contain content or tool calls")


class LLMProxy(Protocol):
    """The only interface through which host code may call a model provider."""

    provider: str
    model: str
    context_window_tokens: int | None

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = "auto",
        max_tokens: int | None = None,
    ) -> ProviderResponse: ...


class ScriptedLLM:
    """Deterministic injected proxy for tests and offline orchestration."""

    provider = "scripted"
    model = "scripted"

    def __init__(
        self,
        responses: Sequence[ProviderResponse],
        *,
        context_window_tokens: int | None = None,
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.context_window_tokens = context_window_tokens

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = "auto",
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "messages": tuple(messages),
                "tools": tuple(dict(item) for item in tools),
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("ScriptedLLM has no response remaining")
        return self._responses.pop(0)


def estimate_chat_request_tokens(
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[Mapping[str, object]] = (),
) -> int:
    """Estimate an OpenAI-compatible chat prompt with bounded safety guards.

    The host environment deliberately has no model-tokenizer dependency.  The
    estimate includes tool schemas (unlike the older history-only heuristic),
    counts each non-ASCII code point as at least one token, and uses three ASCII
    characters per token for ordinary prose/code. Long contiguous or
    whitespace-wrapped alphanumeric/base-encoding-like regions are opaque to
    that heuristic and are therefore charged near worst case at one character
    per token. Provider-side tokenization remains authoritative.
    """

    message_records = [message.to_record() for message in messages]
    tool_records = [dict(item) for item in tools]
    serialized = json.dumps(
        {
            "messages": message_records,
            "tools": tool_records,
        },
        ensure_ascii=False,
        default=str,
        allow_nan=False,
        separators=(",", ":"),
    )
    non_ascii = sum(ord(character) > 127 for character in serialized)
    ascii_chars = len(serialized) - non_ascii
    opaque_ascii_adjustment = _opaque_ascii_adjustment(
        {"messages": message_records, "tools": tool_records}
    )
    structural = 8 * (len(messages) + len(tools))
    return max(
        1,
        (ascii_chars + _ASCII_CHARS_PER_TOKEN - 1) // _ASCII_CHARS_PER_TOKEN
        + non_ascii
        + opaque_ascii_adjustment
        + structural
        + _CHAT_REQUEST_FRAMING_TOKENS,
    )


def context_window_tokens(proxy: object) -> int | None:
    value = getattr(proxy, "context_window_tokens", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("LLM context_window_tokens must be a positive integer or None")
    return value


def context_request_fits(
    proxy: object,
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[Mapping[str, object]] = (),
    max_tokens: int,
) -> tuple[bool, int, int | None]:
    """Return whether prompt estimate plus requested output fits the model."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    prompt_tokens = estimate_chat_request_tokens(messages, tools=tools)
    window = context_window_tokens(proxy)
    return (
        window is None or prompt_tokens + max_tokens <= window,
        prompt_tokens,
        window,
    )


def clamp_requested_max_tokens(
    *,
    requested_max_tokens: int,
    estimated_prompt_tokens: int,
    context_window: int | None,
    margin: int = CONTEXT_OUTPUT_TOKEN_MARGIN,
) -> tuple[int, bool]:
    """Fit the output budget into the remaining context window.

    Returns ``(max_tokens, prompt_fits)``. Unknown windows leave the requested
    budget unchanged. The margin covers host-estimate undercount so a request
    that the estimator calls an exact fit cannot 400 the provider by one token.
    """

    if (
        isinstance(requested_max_tokens, bool)
        or not isinstance(requested_max_tokens, int)
        or requested_max_tokens <= 0
    ):
        raise ValueError("requested_max_tokens must be a positive integer")
    if (
        isinstance(estimated_prompt_tokens, bool)
        or not isinstance(estimated_prompt_tokens, int)
        or estimated_prompt_tokens <= 0
    ):
        raise ValueError("estimated_prompt_tokens must be a positive integer")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin cannot be negative")
    if context_window is None:
        return requested_max_tokens, True
    remaining = context_window - estimated_prompt_tokens - margin
    if remaining < 1:
        return requested_max_tokens, False
    return min(requested_max_tokens, remaining), True


def context_overflow_error(
    *, estimated_prompt_tokens: int, max_tokens: int, context_window: int
) -> LLMProxyError:
    return LLMProxyError(
        "context window would be exceeded before provider request: "
        f"estimated prompt tokens {estimated_prompt_tokens} + requested max output "
        f"tokens {max_tokens} > context window {context_window}",
        retryable=False,
    )


def is_context_overflow_error(exc: Exception) -> bool:
    """Classify local preflight and standard provider overflow messages."""

    if not isinstance(exc, LLMProxyError):
        return False
    text = str(exc).lower()
    if "context" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "length",
            "maximum",
            "exceed",
            "too long",
            "token limit",
        )
    )


_PROVIDER_OUTPUT_OVERFLOW = re.compile(
    r"maximum context length is (?P<window>\d+) tokens\."
    r".*?requested (?P<output>\d+) output tokens"
    r".*?prompt contains at least (?P<prompt>\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


def max_tokens_after_provider_overflow(
    exc: Exception,
    *,
    requested_max_tokens: int,
    margin: int = CONTEXT_OUTPUT_TOKEN_MARGIN,
) -> int | None:
    """If the provider named a prompt that still leaves output room, shrink max_tokens."""

    if (
        isinstance(requested_max_tokens, bool)
        or not isinstance(requested_max_tokens, int)
        or requested_max_tokens <= 0
    ):
        raise ValueError("requested_max_tokens must be a positive integer")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin cannot be negative")
    match = _PROVIDER_OUTPUT_OVERFLOW.search(str(exc))
    if match is not None:
        window = int(match["window"])
        prompt = int(match["prompt"])
        output = int(match["output"])
        # vLLM's "prompt contains at least" is often window + 1 - requested
        # output, not a measured prompt size. Only trust it when it is not
        # that identity.
        if prompt + output != window + 1:
            remaining = window - prompt - margin
            if remaining >= CONTEXT_OUTPUT_MIN_TOKENS and remaining < requested_max_tokens:
                return remaining
    halved = requested_max_tokens // 2
    if halved >= CONTEXT_OUTPUT_MIN_TOKENS:
        return halved
    return None
