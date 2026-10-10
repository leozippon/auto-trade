"""Lighting relay profile, responses dialect and credential fallback; no network.

Fixtures follow the shapes the relay returned to the 2026-10-10 probe (see the
``lighting`` module docstring): a JSON body with ``output`` items, and the SSE
event sequence ``output_item.added`` / ``function_call_arguments.delta`` /
``output_item.done`` / ``completed`` with usage on that last event.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autotrade.environment.llm import (
    MODEL_CHOICES,
    ChatMessage,
    LLMProxyError,
    MalformedToolCallError,
    ToolCall,
    build_model_gateway,
    model_profile,
)
from autotrade.environment.llm.lighting import (
    API_KEY_ALT_ENV,
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    MODEL_PROFILES,
    LightingProxy,
)
from autotrade.environment.llm.openai_compatible import load_env_value

MODEL = "gpt-6-astra"
CLAUDE_MODEL = "claude-sonnet-5-5"
RELAY_DEEPSEEK_MODEL = "DeepSeek-V4.1-Flash"

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
FLAT_TOOL = {
    "type": "function",
    "name": "get_quote",
    "description": "Return the last close of one A-share symbol.",
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.bodies: list[dict[str, object]] = []

    def post(self, url, headers, body, timeout):
        del timeout
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.bodies.append(json.loads(body))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _env(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        text
        if text is not None
        else f"{API_KEY_ENV}=relay-primary\n{API_KEY_ALT_ENV}=relay-alt\n",
        encoding="utf-8",
    )
    return path


def _proxy(
    tmp_path: Path,
    outcomes,
    *,
    model: str = MODEL,
    env_text=None,
    stream_tool_calls: bool | None = None,
    **kwargs,
):
    env_file = _env(tmp_path, env_text)
    gateway = build_model_gateway(
        model,
        env_file=env_file,
        conversation_log_dir=None,
        **kwargs,
    )
    assert isinstance(gateway, LightingProxy)
    transport = FakeTransport(outcomes)
    config = gateway.config
    if stream_tool_calls is not None:
        config = replace(config, stream_tool_calls=stream_tool_calls)
    return (
        LightingProxy(
            config,
            fallback_api_key=gateway.fallback_api_key,
            transport=transport,
        ),
        transport,
    )


def _authorization(transport: FakeTransport) -> list[str]:
    return [header["Authorization"] for header in transport.headers]


def _responses_json(*, content=None, call=None, model: str = MODEL) -> bytes:
    output: list[dict[str, object]] = []
    if call is not None:
        output.append(
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": call[0],
                "name": call[1],
                "arguments": call[2],
            }
        )
    if content is not None:
        output.append(
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    return json.dumps(
        {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": model,
            "output": output,
            "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
        }
    ).encode()


def _sse(*events: dict[str, object]) -> bytes:
    return (
        "\n\n".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}" for event in events
        )
        + "\n"
    ).encode()


def _call_item(arguments: str) -> dict[str, object]:
    return {
        "id": "fc_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "get_quote",
        "arguments": arguments,
    }


def _streamed_call(arguments: str = '{"symbol":"600519.SH"}') -> bytes:
    return _sse(
        {"type": "response.created", "response": {"status": "in_progress"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": _call_item(""),
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": '{"symbol":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": '"600519.SH"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "arguments": arguments,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": _call_item(arguments),
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "model": MODEL,
                "incomplete_details": None,
                "usage": {
                    "input_tokens": 62,
                    "output_tokens": 20,
                    "total_tokens": 82,
                },
                "output": [_call_item(arguments)],
            },
        },
    )


def _streamed_text(text: str = "OK") -> bytes:
    return _sse(
        {"type": "response.created", "response": {"status": "in_progress"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "role": "assistant"},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "delta": text,
        },
        {"type": "response.output_text.done", "output_index": 0, "text": text},
        {"type": "response.output_item.done", "output_index": 0, "item": {"id": "msg_1", "type": "message"}},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "model": MODEL,
                "incomplete_details": None,
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 5,
                    "total_tokens": 14,
                },
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
            },
        },
    )


def test_catalog_registers_every_relay_id_with_its_dialect_and_window():
    assert len(MODEL_PROFILES) == 7
    for model, dialect, window, output in MODEL_PROFILES:
        assert model in MODEL_CHOICES
        profile = model_profile(model)
        assert (
            profile.provider,
            profile.api_key_env,
            profile.base_url_env,
            profile.default_base_url,
            profile.request_dialect,
            profile.context_window_tokens,
            profile.max_output_tokens,
        ) == (
            "lighting",
            API_KEY_ENV,
            BASE_URL_ENV,
            DEFAULT_BASE_URL,
            dialect,
            window,
            output,
        )


def test_gateway_reads_the_alternate_key_from_the_same_fixed_variables(tmp_path: Path):
    gateway = build_model_gateway(
        MODEL,
        env_file=_env(tmp_path),
        conversation_log_dir=None,
        require_credentials=False,
    )
    assert isinstance(gateway, LightingProxy)
    assert gateway.config.base_url == DEFAULT_BASE_URL
    assert gateway.fallback_api_key == "relay-alt"
    without_alt = build_model_gateway(
        MODEL,
        env_file=_env(tmp_path, f"{API_KEY_ENV}=relay-primary\n"),
        conversation_log_dir=None,
    )
    assert isinstance(without_alt, LightingProxy)
    assert without_alt.fallback_api_key == ""


def test_gateway_clamps_the_output_budget_to_the_relay_ceiling(tmp_path: Path):
    responses_model = build_model_gateway(
        MODEL, env_file=_env(tmp_path), conversation_log_dir=None, max_tokens=500_000
    )
    relay_deepseek = build_model_gateway(
        RELAY_DEEPSEEK_MODEL,
        env_file=_env(tmp_path),
        conversation_log_dir=None,
        max_tokens=500_000,
    )
    assert isinstance(responses_model, LightingProxy)
    assert isinstance(relay_deepseek, LightingProxy)
    assert responses_model.config.max_tokens == 128_000
    assert relay_deepseek.config.max_tokens == 384_000


def test_missing_primary_key_fails_naming_the_variable(tmp_path: Path):
    with pytest.raises(ValueError, match=API_KEY_ENV):
        build_model_gateway(
            MODEL,
            env_file=_env(tmp_path, f"{API_KEY_ALT_ENV}=relay-alt\n"),
            conversation_log_dir=None,
        )


def test_responses_payload_maps_history_tools_and_reasoning(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [_responses_json(content="done")],
        max_tokens=900,
        thinking_enabled=True,
        reasoning_effort="xhigh",
    )
    history = [
        ChatMessage("system", "contract"),
        ChatMessage("user", "quote 600519.SH"),
        ChatMessage(
            "assistant",
            None,
            tool_calls=(ToolCall("call_1", "get_quote", {"symbol": "600519.SH"}),),
        ),
        ChatMessage("tool", '{"close":1432.5}', tool_call_id="call_1"),
        ChatMessage("assistant", "Fetching the quote."),
        ChatMessage("user", "continue"),
    ]

    response = proxy.complete(history, tools=TOOLS)

    body = transport.bodies[0]
    assert transport.urls[0] == f"{DEFAULT_BASE_URL}/responses"
    assert _authorization(transport) == ["Bearer relay-primary"]
    assert body["model"] == MODEL
    assert body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "contract"}]},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "quote 600519.SH"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_quote",
            "arguments": json.dumps({"symbol": "600519.SH"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"close":1432.5}',
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Fetching the quote."}],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
    ]
    assert body["tools"] == [FLAT_TOOL]
    assert body["tool_choice"] == "auto"
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["max_output_tokens"] == 900
    assert body["stream"] is True
    assert not {
        "messages",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "stream_options",
        "thinking",
        "user_id",
    } & body.keys()
    assert response.content == "done"
    assert response.usage == {
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
        "prompt_tokens": 11,
        "completion_tokens": 3,
    }


def test_thinking_off_omits_reasoning_and_the_clone_keeps_the_dialect(
    tmp_path: Path,
):
    proxy, transport = _proxy(
        tmp_path,
        [_responses_json(content="done"), _responses_json(content="again")],
        thinking_enabled=False,
        reasoning_effort=None,
    )

    proxy.complete([ChatMessage("user", "summarize")])
    child = proxy.with_thinking(enabled=False, reasoning_effort=None)
    assert isinstance(child, LightingProxy)
    child.complete([ChatMessage("user", "summarize")])

    for body in transport.bodies:
        assert "reasoning" not in body
        assert "input" in body and "messages" not in body
        assert child.config.request_dialect == "openai-responses"


def test_responses_stream_assembles_the_call_and_its_usage(tmp_path: Path):
    proxy, transport = _proxy(tmp_path, [_streamed_call()])

    response = proxy.complete([ChatMessage("user", "quote")], tools=TOOLS)

    assert transport.bodies[0]["stream"] is True
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_1", "get_quote", {"symbol": "600519.SH"})
    ]
    assert response.model == MODEL
    assert response.usage["total_tokens"] == 82
    assert response.content == ""


def test_responses_stream_assembles_text_content(tmp_path: Path):
    proxy, _transport = _proxy(tmp_path, [_streamed_text("OK")])

    # A text-only stream is still streamed when tools were offered.
    response = proxy.complete([ChatMessage("user", "hi")], tools=TOOLS)

    assert response.content == "OK"
    assert response.tool_calls == ()
    assert response.usage["output_tokens"] == 5


def test_responses_usage_is_normalized_to_the_shared_names(tmp_path: Path):
    # The responses contract counts input/output tokens; the host's accounting
    # and truncation checks read the Chat Completions prompt/completion names.
    proxy, _transport = _proxy(
        tmp_path, [_streamed_call(), _responses_json(content="ok")]
    )

    streamed = proxy.complete([ChatMessage("user", "quote")], tools=TOOLS)
    plain = proxy.complete([ChatMessage("user", "hi")])

    assert streamed.usage["prompt_tokens"] == streamed.usage["input_tokens"] == 62
    assert (
        streamed.usage["completion_tokens"]
        == streamed.usage["output_tokens"]
        == 20
    )
    assert plain.usage["prompt_tokens"] == 11
    assert plain.usage["completion_tokens"] == 3


def test_responses_non_stream_parses_a_message_and_a_function_call(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [
            _responses_json(
                content="600519.SH closed at 1432.50.",
                call=("call_1", "get_quote", '{"symbol":"600519.SH"}'),
            )
        ],
        stream_tool_calls=False,
    )

    response = proxy.complete([ChatMessage("user", "quote")], tools=TOOLS)

    assert transport.bodies[0]["stream"] is False
    assert response.content == "600519.SH closed at 1432.50."
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_1", "get_quote", {"symbol": "600519.SH"})
    ]


def test_truncated_streamed_call_is_rejected_without_a_retry(tmp_path: Path):
    cut = _streamed_call('{"symbol": "600519')
    proxy, transport = _proxy(tmp_path, [cut], max_retries=2)

    with pytest.raises(MalformedToolCallError, match="malformed tool call") as caught:
        proxy.complete([ChatMessage("user", "quote")], tools=TOOLS)

    assert caught.value.retryable is False
    assert len(transport.bodies) == 1


def test_exhausted_output_status_marks_the_call_as_cut(tmp_path: Path):
    """A call truncated at the output ceiling is named as such, not as a defect."""
    truncated = _sse(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": _call_item(""),
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": '{"symbol":',
        },
        {
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "model": MODEL,
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 128, "total_tokens": 138},
                "output": [_call_item('{"symbol":')],
            },
        },
    )
    proxy, _transport = _proxy(tmp_path, [truncated])

    with pytest.raises(MalformedToolCallError, match="finish_reason=length"):
        proxy.complete([ChatMessage("user", "quote")], tools=TOOLS)


def test_refusal_reply_is_never_an_answer(tmp_path: Path):
    refusal = _sse(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "role": "assistant"},
        },
        {
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "delta": "The request was rejected because it was considered high risk",
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "model": MODEL,
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "The request was rejected as high risk",
                            }
                        ],
                    }
                ],
            },
        },
    )
    proxy, transport = _proxy(tmp_path, [refusal], max_retries=2)

    with pytest.raises(Exception, match="refused the request") as caught:
        proxy.complete([ChatMessage("user", "summarize")], tools=TOOLS)

    assert type(caught.value).__name__ == "ProviderRefusalError"
    assert len(transport.bodies) == 1


def test_alternate_key_completes_after_a_rejected_primary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    proxy, transport = _proxy(
        tmp_path,
        [
            LLMProxyError("provider HTTP error 401", retryable=False, status_code=401),
            _responses_json(content="done"),
        ],
    )

    response = proxy.complete([ChatMessage("user", "hi")])

    assert response.content == "done"
    assert _authorization(transport) == ["Bearer relay-primary", "Bearer relay-alt"]
    assert len(transport.bodies) == 2
    # The switch is logged by name, never by value.
    stderr = capsys.readouterr().err
    assert API_KEY_ALT_ENV in stderr
    assert "relay-alt" not in stderr
    assert "relay-primary" not in stderr


def test_quota_exhausted_400_also_switches_to_the_alternate_key(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [
            LLMProxyError(
                "provider HTTP error 400: insufficient quota for this account",
                retryable=False,
                status_code=400,
            ),
            _responses_json(content="done"),
        ],
    )

    assert proxy.complete([ChatMessage("user", "hi")]).content == "done"
    assert _authorization(transport) == ["Bearer relay-primary", "Bearer relay-alt"]


def test_alternate_failure_still_reports_the_original_error(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [
            LLMProxyError(
                "provider HTTP error 401: invalid token",
                retryable=False,
                status_code=401,
            ),
            LLMProxyError("provider HTTP error 403", retryable=False, status_code=403),
        ],
    )

    with pytest.raises(LLMProxyError, match="invalid token") as caught:
        proxy.complete([ChatMessage("user", "hi")])

    assert caught.value.status_code == 401
    assert _authorization(transport) == ["Bearer relay-primary", "Bearer relay-alt"]


def test_without_the_alternate_key_a_rejected_primary_is_not_retried(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [LLMProxyError("provider HTTP error 401", retryable=False, status_code=401)],
        env_text=f"{API_KEY_ENV}=relay-primary\n",
    )

    with pytest.raises(LLMProxyError, match="401") as caught:
        proxy.complete([ChatMessage("user", "hi")])

    assert caught.value.status_code == 401
    assert _authorization(transport) == ["Bearer relay-primary"]


@pytest.mark.parametrize(
    "failure",
    [
        LLMProxyError("provider HTTP error 500", retryable=True, status_code=500),
        LLMProxyError(
            "provider HTTP error 400: invalid parameter", retryable=False, status_code=400
        ),
        LLMProxyError("connection reset", retryable=False),
    ],
)
def test_other_failures_never_move_to_the_alternate_key(
    tmp_path: Path, failure: LLMProxyError
):
    proxy, transport = _proxy(
        tmp_path, [failure, failure], max_retries=1, retry_backoff_seconds=0
    )

    with pytest.raises(LLMProxyError):
        proxy.complete([ChatMessage("user", "hi")])

    assert set(_authorization(transport)) == {"Bearer relay-primary"}


def test_hosted_claude_ids_reuse_the_chat_completions_contract(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [
            json.dumps(
                {
                    "model": CLAUDE_MODEL,
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "toolu_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_quote",
                                            "arguments": '{"symbol":"600519.SH"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"total_tokens": 442},
                }
            ).encode()
        ],
        model=CLAUDE_MODEL,
        thinking_enabled=True,
        reasoning_effort="xhigh",
    )

    response = proxy.complete([ChatMessage("user", "quote")], tools=TOOLS)

    body = transport.bodies[0]
    assert transport.urls[0] == f"{DEFAULT_BASE_URL}/chat/completions"
    assert body["messages"] == [{"role": "user", "content": "quote"}]
    assert body["tools"] == [dict(TOOLS[0])]
    # No thinking control for these ids: the level is never sent.
    assert proxy.config.reasoning_effort is None
    assert not {"input", "reasoning", "thinking", "max_output_tokens"} & body.keys()
    assert response.tool_calls[0].name == "get_quote"
    assert response.usage == {"total_tokens": 442}


def test_relay_deepseek_id_uses_the_deepseek_dialect(tmp_path: Path):
    proxy, transport = _proxy(
        tmp_path,
        [
            json.dumps(
                {
                    "model": RELAY_DEEPSEEK_MODEL,
                    "choices": [
                        {
                            "message": {
                                "content": "2",
                                "reasoning_content": "1+1",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 26},
                }
            ).encode()
        ],
        model=RELAY_DEEPSEEK_MODEL,
        thinking_enabled=True,
        reasoning_effort="xhigh",
    )

    response = proxy.complete([ChatMessage("user", "1+1?")])

    body = transport.bodies[0]
    assert transport.urls[0] == f"{DEFAULT_BASE_URL}/chat/completions"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "max"
    assert body["messages"] == [{"role": "user", "content": "1+1?"}]
    assert "max_tokens" in body
    assert "user_id" not in body
    assert response.content == "2"
    assert response.reasoning_content == "1+1"


def test_tools_that_are_not_chat_completions_shaped_fail_explicitly(tmp_path: Path):
    proxy, transport = _proxy(tmp_path, [_responses_json(content="done")])

    with pytest.raises(ValueError, match="chat-completions function shape"):
        proxy.complete(
            [ChatMessage("user", "quote")],
            tools=({"type": "function", "name": "get_quote"},),
        )

    assert transport.bodies == []


def test_alternate_key_is_read_from_the_environment_of_the_gateway(tmp_path: Path):
    path = _env(tmp_path)
    assert load_env_value(API_KEY_ALT_ENV, path) == "relay-alt"
