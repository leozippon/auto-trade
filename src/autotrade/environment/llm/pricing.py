"""What one model call costs: list prices and the billed-token normaliser.

The research session's API spend cap (``pipelines.research_session``) charges
every response its role gateways return at the configured catalog model's
list prices (``ModelProfile.prices``). The providers report usage in four
shapes, read here into one ``(uncached prompt, cached prompt, completion)``
triple; a count a response did not report is estimated, never taken as zero.
Completion tokens include reasoning tokens for every catalog provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .proxy import ChatMessage, ProviderResponse, estimate_chat_request_tokens


class TokenPrices(NamedTuple):
    """List prices in USD per million tokens."""

    uncached_prompt: float
    cached_prompt: float
    completion: float

    def cost_usd(self, tokens: tuple[int, int, int]) -> float:
        """The cost of ``billed_tokens`` at these prices."""

        return sum(count * price for count, price in zip(tokens, self)) / 1_000_000


# A locally served model and the injected test double cost nothing.
FREE = TokenPrices(0.0, 0.0, 0.0)


class ApiSpendExhausted(RuntimeError):
    """A session's API spend reached its cap: no further model call is made."""


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def cached_prompt_tokens(usage: Mapping[str, object]) -> int:
    """Prompt tokens the provider served from its cache.

    DeepSeek reports ``prompt_cache_hit_tokens``; Chat Completions providers
    (MiMo, the relay's chat ids) ``prompt_tokens_details.cached_tokens``; the
    relay's responses dialect ``input_tokens_details.cached_tokens``. A usage
    with none of them (vLLM) served nothing from a priced cache.
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
    return 0


def billed_tokens(
    response: ProviderResponse,
    *,
    messages: Sequence[ChatMessage],
    tools: Sequence[Mapping[str, object]] = (),
) -> tuple[int, int, int]:
    """``(uncached prompt, cached prompt, completion)`` tokens one response bills.

    Every catalog shape carries ``prompt_tokens`` and ``completion_tokens``
    (the relay's responses usage is normalized to them on parse). A stream
    can end without usage: the missing count is then estimated from the
    request and the reply with the host's conservative estimator, and an
    estimated prompt is charged as uncached.
    """

    usage = response.usage
    prompt = _count(usage.get("prompt_tokens"))
    if prompt is None:
        prompt, cached = estimate_chat_request_tokens(messages, tools=tools), 0
    else:
        cached = min(cached_prompt_tokens(usage), prompt)
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
    return prompt - cached, cached, completion


__all__ = [
    "FREE",
    "ApiSpendExhausted",
    "TokenPrices",
    "billed_tokens",
    "cached_prompt_tokens",
]
