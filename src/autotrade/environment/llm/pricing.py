"""What one model call costs: list prices and the billed-token normaliser.

The research session's API spend cap (``pipelines.research_session``) charges
every response its role gateways return at the configured catalog model's
list prices (``ModelProfile.prices``), at the provider's long-context tier
when the model has one and the call's prompt passes it. The providers report
usage in four shapes, read here into one ``(uncached prompt, cached prompt,
completion, cache write)`` count; a count a response did not report is
estimated, never taken as zero. Completion tokens include reasoning tokens for every catalog
provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .proxy import ChatMessage, ProviderResponse, estimate_chat_request_tokens


class LongContextTier(NamedTuple):
    """A provider's request-length tier: a call whose whole prompt (uncached,
    cached and written) exceeds ``above_prompt_tokens`` is billed entirely at
    ``prompt_multiplier`` times every prompt price and
    ``completion_multiplier`` times the completion price."""

    above_prompt_tokens: int
    prompt_multiplier: float
    completion_multiplier: float


@dataclass(frozen=True)
class TokenPrices:
    """List prices in USD per million tokens, in ``billed_tokens`` order.

    ``cache_write`` prices the prompt tokens a call wrote to the provider's
    cache. A provider that reports no write lists its uncached prompt price
    there, which its write count of zero never reaches. ``long_context`` is
    the provider's length tier, None where one price holds at any length.
    """

    uncached_prompt: float
    cached_prompt: float
    completion: float
    cache_write: float
    long_context: LongContextTier | None = None

    def cost_usd(self, tokens: tuple[int, int, int, int]) -> float:
        """The cost of ``billed_tokens`` at these prices, at the long-context
        tier when the call's prompt is past it."""

        uncached, cached, completion, written = tokens
        prompt_rate = completion_rate = 1.0
        tier = self.long_context
        if tier is not None and uncached + cached + written > tier.above_prompt_tokens:
            prompt_rate, completion_rate = tier.prompt_multiplier, tier.completion_multiplier
        prompt = (
            uncached * self.uncached_prompt
            + cached * self.cached_prompt
            + written * self.cache_write
        )
        return (prompt_rate * prompt + completion_rate * completion * self.completion) / 1_000_000


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
    relay reports them for its responses dialect as
    ``input_tokens_details.cache_write_tokens``; its chat ids, DeepSeek, MiMo
    and vLLM report no write.
    """

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
    "LongContextTier",
    "TokenPrices",
    "billed_tokens",
    "cache_write_tokens",
    "cached_prompt_tokens",
]
