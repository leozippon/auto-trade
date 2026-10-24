"""What one model call costs: list prices and the billed-token normaliser.

The research session's API spend cap (``pipelines.research_session``) charges
every response its role gateways return at the configured catalog model's
list prices (``ModelProfile.prices``). The providers report usage in four
shapes, read here into one ``(uncached prompt, cached prompt, completion,
cache write)`` count; a count a response did not report is estimated, never
taken as zero. Completion tokens include reasoning tokens for every catalog
provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .proxy import ChatMessage, ProviderResponse, estimate_chat_request_tokens


class TokenPrices(NamedTuple):
    """List prices in USD per million tokens, in ``billed_tokens`` order.

    ``cache_write`` prices the prompt tokens a call wrote to the provider's
    cache. A provider that reports no write lists its uncached prompt price
    there, which its write count of zero never reaches.
    """

    uncached_prompt: float
    cached_prompt: float
    completion: float
    cache_write: float

    def cost_usd(self, tokens: tuple[int, int, int, int]) -> float:
        """The cost of ``billed_tokens`` at these prices."""

        return sum(count * price for count, price in zip(tokens, self, strict=True)) / 1_000_000


# A locally served model and the injected test double cost nothing.
FREE = TokenPrices(0.0, 0.0, 0.0, 0.0)


class ApiSpendExhausted(RuntimeError):
    """A session's API spend reached its cap: no further model call is made."""


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def cached_prompt_tokens(usage: Mapping[str, object]) -> int | None:
    """Prompt tokens the provider served from its cache, ``None`` if unreported.

    DeepSeek reports ``prompt_cache_hit_tokens``; Chat Completions providers
    (MiMo, the relay's chat ids) ``prompt_tokens_details.cached_tokens``; the
    relay's responses dialect ``input_tokens_details.cached_tokens``. A usage
    with none of them (vLLM) served nothing from a priced cache, but whether
    it hit an unpriced one is unknown, not zero.
    """

    hit = _count(usage.get("prompt_cache_hit_tokens"))
    if hit is not None:
        return hit
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if isinstance(details, Mapping):
            cached = _count(details.get("cached_tokens"))
            if cached is not None:
                return cached
    return None


def cache_write_tokens(usage: Mapping[str, object]) -> int:
    """Prompt tokens the call wrote to the provider's cache.

    Writes are part of ``prompt_tokens`` and not of the cached count. The
    relay reports them for its Claude ids as Anthropic's own counter,
    ``billing_usage.claude_usage.cache_creation_input_tokens``, and for its
    responses dialect as ``input_tokens_details.cache_write_tokens``.
    DeepSeek, MiMo and vLLM report no write.
    """

    billing = usage.get("billing_usage")
    claude = billing.get("claude_usage") if isinstance(billing, Mapping) else None
    if isinstance(claude, Mapping):
        written = _count(claude.get("cache_creation_input_tokens"))
        if written is not None:
            return written
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        written = _count(details.get("cache_write_tokens"))
        if written is not None:
            return written
    return 0


def billed_tokens(
    response: ProviderResponse,
    *,
    messages: Sequence[ChatMessage],
    tools: Sequence[Mapping[str, object]] = (),
) -> tuple[int, int, int, int]:
    """``(uncached prompt, cached prompt, completion, cache write)`` tokens one
    response bills; the uncached count excludes the writes.

    Every catalog shape carries ``prompt_tokens`` and ``completion_tokens``
    (the relay's responses usage is normalized to them on parse). A stream
    can end without usage: the missing count is then estimated from the
    request and the reply with the host's conservative estimator, and an
    estimated prompt is charged as uncached.
    """

    usage = response.usage
    prompt = _count(usage.get("prompt_tokens"))
    if prompt is None:
        prompt, cached, written = estimate_chat_request_tokens(messages, tools=tools), 0, 0
    else:
        cached = min(cached_prompt_tokens(usage) or 0, prompt)
        written = min(cache_write_tokens(usage), prompt - cached)
    completion = _count(usage.get("completion_tokens"))
    if completion is None:
        completion = estimate_chat_request_tokens(
            [
                ChatMessage(
                    "assistant",
                    response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
            ]
        )
    return prompt - cached - written, cached, completion, written


__all__ = [
    "FREE",
    "ApiSpendExhausted",
    "TokenPrices",
    "billed_tokens",
    "cache_write_tokens",
    "cached_prompt_tokens",
]
