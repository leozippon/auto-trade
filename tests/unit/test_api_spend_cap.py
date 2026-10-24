"""The research session's API spend cap: catalog prices, the billed-token
normaliser over every provider's usage shape, and the shared meter."""

from __future__ import annotations

import time

import pytest

from autotrade.environment.llm import (
    LOCAL_QWEN_MODEL,
    MIMO_FLASH_MODEL,
    MODEL_CHOICES,
    ApiSpendExhausted,
    ChatMessage,
    ProviderResponse,
    ScriptedLLM,
    TokenPrices,
    ToolCall,
    billed_tokens,
    estimate_chat_request_tokens,
    model_profile,
    token_prices,
)
from autotrade.environment.llm.pricing import FREE
from autotrade.pipelines.config import BudgetUsed
from autotrade.pipelines.research_session import SessionBudgetLLM, SessionCallBudget

MESSAGES = (ChatMessage("system", "rules"), ChatMessage("user", "quote 600519.SH"))


def _response(usage: dict[str, object], content: str = "ok") -> ProviderResponse:
    return ProviderResponse(content=content, usage=usage)


@pytest.mark.parametrize(
    "usage",
    [
        # DeepSeek (and the relay's DeepSeek id): hit/miss split.
        {
            "prompt_tokens": 1_000,
            "completion_tokens": 300,
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
        },
        # MiMo and the relay's chat ids: OpenAI prompt details.
        {
            "prompt_tokens": 1_000,
            "completion_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 600},
            "completion_tokens_details": {"reasoning_tokens": 200},
        },
        # The relay's responses dialect, after its parse adds the shared names.
        {
            "input_tokens": 1_000,
            "output_tokens": 300,
            "input_tokens_details": {"cached_tokens": 600},
            "output_tokens_details": {"reasoning_tokens": 200},
            "prompt_tokens": 1_000,
            "completion_tokens": 300,
        },
    ],
    ids=["deepseek", "chat-completions", "responses"],
)
def test_every_cached_usage_shape_bills_the_same_split(usage):
    # None of these shapes reports a cache write, so none is billed.
    assert billed_tokens(_response(usage), messages=MESSAGES) == (400, 600, 300, 0)


# The two shapes that report writes, each part of ``prompt_tokens`` beside the
# read: 100 uncached + 600 read + 300 written.
CACHE_WRITE_USAGES = {
    # The relay's Claude ids: the OpenAI fields plus Anthropic's own counters.
    "claude-opus-5-5": {
        "prompt_tokens": 1_000,
        "completion_tokens": 300,
        "prompt_tokens_details": {"cached_tokens": 600},
        "billing_usage": {
            "claude_usage": {
                "cache_creation_input_tokens": 300,
                "cache_read_input_tokens": 600,
                "input_tokens": 100,
                "output_tokens": 300,
            }
        },
    },
    # The relay's responses dialect, after its parse adds the shared names.
    "gpt-6.1-sol": {
        "input_tokens": 1_000,
        "output_tokens": 300,
        "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
        "prompt_tokens": 1_000,
        "completion_tokens": 300,
    },
}


@pytest.mark.parametrize(
    ("model", "spend"),
    [
        # 100 x 4 + 600 x 0.2 + 300 x 20 + 300 x 5.0 (1.25 x 4), per million.
        ("claude-opus-5-5", 0.008020),
        # 100 x 2 + 600 x 0.1 + 300 x 10 + 300 x 2.5, per million.
        ("gpt-6.1-sol", 0.004010),
    ],
)
def test_cache_writes_are_billed_apart_at_the_write_price(model, spend):
    usage = CACHE_WRITE_USAGES[model]
    assert billed_tokens(_response(usage), messages=MESSAGES) == (100, 600, 300, 300)

    budget = SessionCallBudget(
        max_calls=4, max_api_spend_usd=1.0, deadline=time.monotonic() + 60
    )
    SessionBudgetLLM(
        _priced(model, "lighting", [_response(usage)]), budget=budget
    ).complete(MESSAGES)

    assert budget.api_spend_usd == pytest.approx(spend)


def test_plain_usage_bills_the_whole_prompt_uncached():
    usage = {"prompt_tokens": 1_000, "completion_tokens": 300, "total_tokens": 1_300}

    assert billed_tokens(_response(usage), messages=MESSAGES) == (1_000, 0, 300, 0)


def test_a_reply_without_usage_is_estimated_never_free():
    tools = ({"type": "function", "function": {"name": "get_quote", "parameters": {}}},)
    reply = ProviderResponse(
        content="",
        tool_calls=(ToolCall("c1", "get_quote", {"symbol": "600519.SH"}),),
        reasoning_content="think " * 200,
    )

    uncached, cached, completion, written = billed_tokens(
        reply, messages=MESSAGES, tools=tools
    )

    assert uncached == estimate_chat_request_tokens(MESSAGES, tools=tools)
    assert cached == written == 0
    # The reasoning alone is ~1,200 characters: the estimate counts it.
    assert completion > 300


def test_costs_follow_the_catalog_price_of_the_configured_model():
    prices = model_profile(MIMO_FLASH_MODEL).prices

    assert prices == TokenPrices(0.141, 0.0028, 0.282, 0.141)
    assert prices.cost_usd((1_000_000, 1_000_000, 1_000_000, 0)) == pytest.approx(0.4258)


def test_the_local_model_is_free_and_every_hosted_model_is_priced():
    for model in MODEL_CHOICES:
        profile = model_profile(model)
        if profile.provider == "vllm":
            assert profile.prices == FREE, model
        else:
            assert all(price > 0 for price in profile.prices), model
            # A write is never billed below an uncached prompt token.
            assert profile.prices.cache_write >= profile.prices.uncached_prompt, model


def test_a_gateway_outside_the_catalog_is_refused_before_any_call():
    class UnknownHosted(ScriptedLLM):
        provider = "openai"
        model = "gpt-unlisted"

    with pytest.raises(ValueError, match="unsupported model 'gpt-unlisted'"):
        SessionBudgetLLM(UnknownHosted([]), max_calls=4, deadline=time.monotonic() + 60)
    assert token_prices(ScriptedLLM.provider, ScriptedLLM.model) == FREE


def _priced(model: str, provider: str, responses):
    class Gateway(ScriptedLLM):
        pass

    Gateway.provider, Gateway.model = provider, model
    return Gateway(responses)


def test_the_local_model_meters_zero():
    budget = SessionCallBudget(
        max_calls=4, max_api_spend_usd=0.01, deadline=time.monotonic() + 60
    )
    usage = {"prompt_tokens": 500_000, "completion_tokens": 100_000}
    llm = SessionBudgetLLM(
        _priced(LOCAL_QWEN_MODEL, "vllm", [_response(usage)] * 3), budget=budget
    )

    for _ in range(3):
        llm.complete(MESSAGES)

    assert budget.api_spend_usd == 0.0 and budget.calls == 3


def test_the_cap_refuses_every_role_once_the_charged_spend_reaches_it():
    budget = SessionCallBudget(
        max_calls=40, max_api_spend_usd=0.002, deadline=time.monotonic() + 60
    )
    usage = {"prompt_tokens": 10_000, "completion_tokens": 1_000}
    gateway = _priced(MIMO_FLASH_MODEL, "mimo", [_response(usage)] * 3)
    main = SessionBudgetLLM(gateway, budget=budget, role="main")
    child = SessionBudgetLLM(gateway, budget=budget, role="subagent")

    main.complete(MESSAGES)
    child.complete(MESSAGES)

    assert budget.api_spend_usd == pytest.approx(2 * 0.001692)
    for role in (main, child):
        with pytest.raises(ApiSpendExhausted, match="API spend budget exhausted"):
            role.complete(MESSAGES)
    # Refused before the provider: the third scripted reply was never asked for.
    assert len(gateway.calls) == 2 and budget.calls == 2


def test_a_resumed_attempt_continues_the_recorded_spend():
    # A block an earlier attempt wrote round-trips through the trace record.
    recorded = BudgetUsed(llm_calls=7, main_calls=7, api_spend_usd=12.34567).to_record()
    assert recorded["api_spend_usd"] == 12.3457
    # A block written before the cap existed reads as nothing spent.
    assert BudgetUsed.from_record({"llm_calls": 3}).api_spend_usd == 0.0

    budget = SessionCallBudget(
        max_calls=40, max_api_spend_usd=12.0, deadline=time.monotonic() + 60
    )
    budget.seed(BudgetUsed.from_record(recorded))

    assert budget.api_spend_usd == 12.3457
    with pytest.raises(ApiSpendExhausted):
        budget.claim("main")


@pytest.mark.parametrize("cap", [0, -1.0, float("inf"), True, "50"])
def test_a_cap_must_be_a_positive_finite_number(cap):
    with pytest.raises(ValueError, match="max_api_spend_usd"):
        SessionCallBudget(max_calls=4, max_api_spend_usd=cap, deadline=time.monotonic() + 60)
