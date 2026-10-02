"""MiMo profile and request dialect; no network (recorded-shape fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.environment.llm import (
    MIMO_FLASH_MODEL,
    MODEL_CHOICES,
    ChatMessage,
    LLMProxyError,
    OpenAICompatibleConfig,
    OpenAICompatibleProxy,
    build_model_gateway,
    model_profile,
)

TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Return the last close of one A-share symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
)


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    full = {"content": None, "role": None, "tool_calls": None, "reasoning_content": None}
    full.update(delta)
    return {
        "id": "rec-1",
        "choices": [{"delta": full, "finish_reason": finish_reason, "index": 0}],
        "created": 1790911565,
        "model": MIMO_FLASH_MODEL,
        "object": "chat.completion.chunk",
    }


def _call_delta(arguments: str, *, call_id: str | None = None, name: str | None = None):
    return {
        "tool_calls": [
            {
                "index": 0,
                "id": call_id,
                "function": {"arguments": arguments, "name": name},
                "type": "function",
            }
        ]
    }


# The stream mimo-v2.6-flash returned to the 2026-10-01 smoke (thinking on, one
# tool call), with the response id shortened: null fields in every delta, the
# usage only in a trailing chunk with empty choices.
RECORDED_TOOL_CALL_STREAM = (
    "\n".join(
        "data: " + json.dumps(chunk)
        for chunk in (
            _chunk({"content": "", "role": "assistant"}),
            _chunk({"reasoning_content": "Need"}),
            _chunk({"reasoning_content": " to call get_quote."}),
            _chunk(_call_delta("", call_id="call_ec6a", name="get_quote")),
            _chunk(_call_delta('{"symbol": ')),
            _chunk(_call_delta('"600519.SH"}')),
            {**_chunk({}, "tool_calls"), "usage": None},
            {
                "id": "rec-1",
                "choices": [],
                "model": MIMO_FLASH_MODEL,
                "object": "chat.completion.chunk",
                "usage": {
                    "completion_tokens": 32,
                    "prompt_tokens": 118,
                    "total_tokens": 150,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )
    )
    + "\ndata: [DONE]\n"
).encode()


def _json_response(content: str) -> bytes:
    return json.dumps(
        {
            "model": MIMO_FLASH_MODEL,
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"completion_tokens": 2, "prompt_tokens": 26, "total_tokens": 28},
        }
    ).encode()


@pytest.fixture(autouse=True)
def _no_process_credentials(monkeypatch):
    # The env file under test must be the only source; a real key exported in
    # the shell would otherwise win and could surface in an assertion diff.
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_BASE_URL", raising=False)


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.bodies: list[dict[str, object]] = []
        self.headers: list[dict[str, str]] = []

    def post(self, url, headers, body, timeout):
        del url, timeout
        self.headers.append(dict(headers))
        self.bodies.append(json.loads(body))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _env(tmp_path: Path, text: str = "MIMO_API_KEY=mimo-test-key\n") -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def _proxy(tmp_path: Path, outcomes, **kwargs) -> tuple[OpenAICompatibleProxy, FakeTransport]:
    gateway = build_model_gateway(
        MIMO_FLASH_MODEL,
        env_file=_env(tmp_path),
        conversation_log_dir=None,
        **kwargs,
    )
    assert isinstance(gateway, OpenAICompatibleProxy)
    transport = FakeTransport(outcomes)
    return OpenAICompatibleProxy(gateway.config, transport=transport), transport


def test_profile_reads_fixed_env_keys_and_publishes_limits(tmp_path: Path):
    assert MIMO_FLASH_MODEL in MODEL_CHOICES
    profile = model_profile(MIMO_FLASH_MODEL)
    assert (profile.provider, profile.api_key_env, profile.base_url_env) == (
        "mimo",
        "MIMO_API_KEY",
        "MIMO_BASE_URL",
    )
    assert profile.context_window_tokens == 1_000_000
    assert profile.max_output_tokens == 131_072

    default = build_model_gateway(MIMO_FLASH_MODEL, env_file=_env(tmp_path))
    assert default.config.base_url == "https://api.xiaomimimo.com/v1"
    assert default.config.max_tokens == 1_200
    plan = build_model_gateway(
        MIMO_FLASH_MODEL,
        env_file=_env(
            tmp_path,
            "MIMO_API_KEY=tp-key\nMIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1\n",
        ),
        max_tokens=500_000,
    )
    assert plan.config.base_url == "https://token-plan-cn.xiaomimimo.com/v1"
    assert plan.config.max_tokens == 131_072


def test_missing_key_fails_naming_the_variable(tmp_path: Path):
    with pytest.raises(ValueError, match="MIMO_API_KEY"):
        build_model_gateway(MIMO_FLASH_MODEL, env_file=_env(tmp_path, ""))


def test_thinking_request_and_recorded_stream_with_reasoning_replay(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [RECORDED_TOOL_CALL_STREAM, _json_response("1432.5")],
        max_tokens=32_768,
        thinking_enabled=True,
        reasoning_effort="xhigh",
    )
    ask = [ChatMessage("user", "last close of 600519.SH?")]

    response = proxy.complete(ask, tools=TOOLS)

    body = transport.bodies[0]
    assert transport.headers[0]["Authorization"] == "Bearer mimo-test-key"
    assert body["max_completion_tokens"] == 32_768
    assert body["thinking"] == {"type": "enabled"}
    assert body["tool_choice"] == "auto"
    assert body["stream"] is True
    assert not {"max_tokens", "reasoning_effort", "user_id", "chat_template_kwargs"} & body.keys()
    assert proxy.config.reasoning_effort is None
    assert response.reasoning_content == "Need to call get_quote."
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_ec6a", "get_quote", {"symbol": "600519.SH"})
    ]
    assert response.usage["completion_tokens_details"] == {"reasoning_tokens": 7}

    history = [
        *ask,
        ChatMessage(
            "assistant",
            None,
            tool_calls=response.tool_calls,
            reasoning_content=response.reasoning_content,
        ),
        ChatMessage("tool", '{"close": 1432.5}', tool_call_id="call_ec6a"),
    ]
    assert proxy.complete(history, tools=TOOLS).content == "1432.5"
    replayed = transport.bodies[1]["messages"][1]
    assert replayed["reasoning_content"] == "Need to call get_quote."


def test_tool_choice_none_withholds_the_tools_and_other_choices_fail(tmp_path: Path):
    proxy, transport = _proxy(tmp_path, [_json_response("final")])

    assert proxy.complete([ChatMessage("user", "sum up")], tools=TOOLS, tool_choice="none").content == "final"
    assert not {"tools", "tool_choice", "stream_options"} & transport.bodies[0].keys()
    assert transport.bodies[0]["thinking"] == {"type": "disabled"}

    for choice in ("required", {"type": "function", "function": {"name": "get_quote"}}):
        with pytest.raises(ValueError, match="tool_choice auto or none"):
            proxy.complete([ChatMessage("user", "x")], tools=TOOLS, tool_choice=choice)
    assert len(transport.bodies) == 1


def test_sub_agent_thinking_level_becomes_on_off(tmp_path: Path):
    proxy, transport = _proxy(tmp_path, [_json_response("no")])

    child = proxy.with_thinking(enabled=True, reasoning_effort="medium")
    child.complete([ChatMessage("user", "is 91 prime?")])

    assert child.config.thinking_enabled is True
    assert child.config.reasoning_effort is None
    assert transport.bodies[0]["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in transport.bodies[0]


def test_provider_overflow_shrinks_the_documented_output_field(tmp_path: Path):
    overflow = LLMProxyError(
        "provider HTTP error 400: This model's maximum context length is 1000000 "
        "tokens. However, you requested 32768 output tokens and your prompt "
        "contains at least 967233 input tokens, for a total of at least 1000001 tokens.",
        status_code=400,
    )
    proxy, transport = _proxy(
        tmp_path, [overflow, _json_response("ok")], max_tokens=32_768, max_retries=0
    )

    assert proxy.complete([ChatMessage("user", "meta")]).content == "ok"
    assert [body["max_completion_tokens"] for body in transport.bodies] == [32_768, 16_384]
    assert all("max_tokens" not in body for body in transport.bodies)


@pytest.mark.parametrize(
    ("update", "message"),
    [({"reasoning_effort": "xhigh"}, "no reasoning_effort"), ({"temperature": 1.6}, r"\[0, 1.5\]")],
)
def test_config_rejects_what_mimo_cannot_honour(update: dict[str, object], message: str):
    values: dict[str, object] = {
        "api_key": "k",
        "provider": "mimo",
        "model": MIMO_FLASH_MODEL,
        "base_url": "https://api.xiaomimimo.com/v1",
        "request_dialect": "mimo",
        "thinking_enabled": True,
    }
    values.update(update)
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleConfig(**values)
