"""One model turn as both agent loops record it: the session runner's and
each sub-agent's.

Each loop keeps its own control flow and event names; the messages it
appends and the trace fields it writes for a completed, failed or truncated
call are built here, so the two conversations and their traces stay alike.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from autotrade.environment.llm import (
    ChatMessage,
    LLMProxyError,
    MalformedToolCallError,
    ProviderRefusalError,
    ProviderResponse,
    cache_write_tokens,
    cached_prompt_tokens,
)
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.trace import trace_reasoning

OUTPUT_TRUNCATED_CONTINUATION = (
    "上一回合的输出在 {limit} token 上限被截断且没有工具调用。"
    "请把已有结论压缩成几句话，然后直接调用下一步工具；不要重新展开完整推理。"
)
# The llm_error note after a call whose transient provider failures (the
# retryable class) outlasted the gateway's deadline: the request was not at
# fault, so the usual advice to change the action does not apply.
PROVIDER_TRANSIENT_FAILURE_HINT = (
    "The provider failed transiently; nothing in your request needs to change."
)
LLM_ERROR_RETRY_HINT = (
    "Proceed with a different bounded action; do not repeat the same failing request verbatim."
)


def observation(payload: Mapping[str, object]) -> ChatMessage:
    """A JSON observation handed to the model as a user turn."""

    return ChatMessage("user", json.dumps(dict(payload), ensure_ascii=False, allow_nan=False))


def assistant_message(response: ProviderResponse) -> ChatMessage:
    """The reply as the conversation keeps it: a tool-calling reply without
    text carries no content."""

    return ChatMessage(
        "assistant",
        response.content or None if response.tool_calls else response.content,
        response.tool_calls,
        reasoning_content=response.reasoning_content,
    )


def tool_message(call_id: str, record: object) -> ChatMessage:
    """One tool call's result record as the conversation keeps it."""

    return ChatMessage(
        "tool",
        json.dumps(sanitize_for_log(record), ensure_ascii=False, default=str, allow_nan=False),
        tool_call_id=call_id,
    )


def llm_call_fields(provider: str, response: ProviderResponse) -> dict[str, object]:
    """A completed call's trace fields, after the loop's own identity keys."""

    return {
        "provider": provider,
        "model": response.model,
        "usage": dict(response.usage),
        "content": response.content,
        **trace_reasoning(response.reasoning_content or response.reasoning_summary),
        "tool_names": [call.name for call in response.tool_calls],
    }


def llm_failure_fields(exc: Exception) -> dict[str, object]:
    """A failed call's class where an audit counts it: a malformed tool call,
    which costs one turn and keeps the reasoning that arrived, and a provider
    refusal."""

    if isinstance(exc, MalformedToolCallError):
        return {"error_type": "malformed_tool_call", **trace_reasoning(exc.reasoning_content)}
    if isinstance(exc, ProviderRefusalError):
        return {"error_type": "provider_refusal"}
    return {}


def llm_error_observation(exc: Exception, error: str) -> ChatMessage:
    """The note after a failed call the loop goes on from."""

    transient = isinstance(exc, LLMProxyError) and exc.retryable
    return observation(
        {
            "observation": "llm_error",
            "error": error,
            "retry_hint": PROVIDER_TRANSIENT_FAILURE_HINT if transient else LLM_ERROR_RETRY_HINT,
        }
    )


def completion_tokens(usage: Mapping[str, object]) -> int:
    return int(dict(usage).get("completion_tokens") or 0)


def truncation_observation(usage: Mapping[str, object], max_tokens: int) -> ChatMessage:
    """The note after a reply that spent its whole output cap without a tool
    call: it asks for a concise continuation."""

    return observation(
        {
            "observation": "output_truncated",
            "completion_tokens": completion_tokens(usage),
            "max_tokens": max_tokens,
            "message": OUTPUT_TRUNCATED_CONTINUATION.format(limit=max_tokens),
        }
    )


def new_token_totals() -> dict[str, int]:
    """One conversation's token counters, the session's and each child's."""

    return dict.fromkeys(
        (
            "llm_calls_with_usage",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "cache_write_tokens",
        ),
        0,
    )


def accumulate_usage(total: dict[str, int], usage: object) -> None:
    """Add one call's usage to ``new_token_totals`` counters.

    A miss is a prompt token not read from the cache, the call's cache writes
    included; hits and misses are summed only from a usage that reports a
    cache count, because one without (vLLM) leaves its hit rate unknown, not
    zero. Reasoning comes from ``completion_tokens_details`` or, in the
    responses dialect, ``output_tokens_details``.

    Cache hits are only realized while the request prefix (system prompt +
    tool schemas + early history) stays byte-stable; semantic compaction and
    emergency tool-result fitting rewrite history and reset that prefix.
    """

    if not isinstance(usage, dict):
        return
    total["llm_calls_with_usage"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] += int(value)
    cached = cached_prompt_tokens(usage)
    prompt = usage.get("prompt_tokens")
    if cached is not None:
        total["cache_hit_tokens"] += cached
        if isinstance(prompt, (int, float)) and not isinstance(prompt, bool):
            total["cache_miss_tokens"] += max(int(prompt) - cached, 0)
    total["cache_write_tokens"] += cache_write_tokens(usage)
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if isinstance(reasoning, (int, float)) and not isinstance(reasoning, bool):
            total["reasoning_tokens"] += int(reasoning)
            break
