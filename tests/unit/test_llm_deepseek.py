from __future__ import annotations

import json
import random
import string
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from autotrade.environment.llm import (
    CONTEXT_OUTPUT_TOKEN_MARGIN,
    LEGACY_LOCAL_QWEN_MODEL,
    LOCAL_QWEN_MODEL,
    MODEL_CHOICES,
    ChatMessage,
    DeepSeekConfig,
    DeepSeekProxy,
    LLMProxyError,
    OpenAICompatibleConfig,
    OpenAICompatibleProxy,
    build_model_gateway,
    estimate_chat_request_tokens,
    load_api_key,
    load_env_value,
    model_profile,
)


def make_config(**kwargs) -> DeepSeekConfig:
    values = {"api_key": "secret", "conversation_log_dir": None}
    values.update(kwargs)
    return DeepSeekConfig(**values)


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def post(self, url, headers, body, timeout):
        self.requests.append((url, dict(headers), json.loads(body), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode()


class _QuietHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        del request, client_address


class _DeadlineHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *args):
        del args

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path.startswith("/credential-boundary/"):
            credential = self.headers.get("Authorization", "").removeprefix("Bearer ")
            endpoint = f"http://{self.headers['Host']}{self.path}"
            payload = json.dumps(
                {
                    "error": {
                        "message": (
                            credential
                            + " maximum context length exceeded at "
                            + endpoint
                        )
                    }
                }
            ).encode()
            self._write_error(payload)
            return
        if self.path.startswith("/endpoint-boundary/"):
            endpoint = f"http://{self.headers['Host']}{self.path}"
            payload = json.dumps(
                {"error": {"message": endpoint + " maximum context length exceeded"}}
            ).encode()
            self._write_error(payload)
            return
        if self.path.startswith("/oversized-error/"):
            payload = json.dumps(
                {"error": {"message": "maximum context length " + "x" * 100_000}}
            ).encode()
            self._write_error(payload)
            return
        if self.path.startswith("/malformed-error/"):
            self._write_error(b'{"error":{"message":')
            return
        if self.path.startswith("/openai-error/"):
            endpoint = f"http://{self.headers['Host']}{self.path}"
            payload = json.dumps(
                {
                    "error": {
                        "message": (
                            "maximum context length exceeded for local-test-key at "
                            + endpoint
                            + "; "
                            + "x" * 2_000
                        )
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = json.dumps(_response({"content": "ok"})).encode()
        if self.path.startswith("/slow-headers/"):
            time.sleep(0.2)
            self._write_content_length(payload)
            return
        if self.path.startswith("/nonstream-drip/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self._write_pieces(_split_four(payload), delay=0.08)
            return
        if self.path.startswith("/sse-drip/"):
            stream = _stream_response(
                {"choices": [{"delta": {"content": "ok"}}]},
            )
            self._write_chunked(_split_four(stream), delay=0.08)
            return
        if self.path.startswith("/normal-chunked/"):
            self._write_chunked(_split_four(payload), delay=0.0)
            return
        self.send_error(404)

    def _write_error(self, payload: bytes) -> None:
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass

    def _write_content_length(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write_pieces((payload,), delay=0.0)

    def _write_chunked(self, pieces: tuple[bytes, ...], *, delay: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunks = tuple(
            f"{len(piece):x}\r\n".encode() + piece + b"\r\n" for piece in pieces
        )
        self._write_pieces((*chunks, b"0\r\n\r\n"), delay=delay)

    def _write_pieces(self, pieces: tuple[bytes, ...], *, delay: float) -> None:
        try:
            for index, piece in enumerate(pieces):
                self.wfile.write(piece)
                self.wfile.flush()
                if delay and index + 1 < len(pieces):
                    time.sleep(delay)
        except OSError:
            pass


def _split_four(payload: bytes) -> tuple[bytes, ...]:
    width = max((len(payload) + 3) // 4, 1)
    return tuple(
        payload[offset : offset + width] for offset in range(0, len(payload), width)
    )


@contextmanager
def _provider_server():
    server = _QuietHTTPServer(("127.0.0.1", 0), _DeadlineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _response(message):
    return {
        "model": "deepseek-chat",
        "choices": [{"message": message}],
        "usage": {"total_tokens": 3},
    }


def _stream_response(*chunks):
    lines = [*(f"data: {json.dumps(chunk)}" for chunk in chunks), "data: [DONE]"]
    return ("\n".join(lines) + "\n").encode()


RETRYABLE_PROTOCOL_FAILURES = (
    (
        _stream_response(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        ({"type": "function"},),
        "provider returned an invalid stream tool call",
    ),
    (
        _stream_response({"choices": [{"delta": {"content": None}}]}),
        ({"type": "function"},),
        "provider response must contain content or tool calls",
    ),
    (
        _response({"content": None}),
        (),
        "provider response must contain content or tool calls",
    ),
    (
        _response({"content": ""}),
        (),
        "provider response must contain content or tool calls",
    ),
)

INVALID_STREAM_STRUCTURES = (
    {"choices": [{"delta": {"tool_calls": {}}}]},
    {"choices": [{"delta": {"tool_calls": ["bad"]}}]},
    {"choices": [{"delta": []}]},
    {"choices": {}},
    {"choices": ["bad"]},
    {"choices": None},
    {"choices": [{"delta": {"tool_calls": [{"function": []}]}}]},
)

INVALID_COMPLETE_TOOL_CALLS = (
    "not-an-object",
    {
        "id": 7,
        "function": {"name": "read_file", "arguments": '{"path":"main.py"}'},
    },
    {"id": "call-1", "function": []},
    {
        "id": "call-1",
        "function": {"name": ["read_file"], "arguments": '{"path":"main.py"}'},
    },
    {
        "id": "call-1",
        "function": {"name": "read_file", "arguments": {"path": "main.py"}},
    },
    {
        "id": "call-1",
        "function": {"name": "read_file", "arguments": '["main.py"]'},
    },
)


def test_load_api_key_from_env_file_without_printing_secret(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY='secret-value'\n", encoding="utf-8")
    assert load_api_key(env_file=path) == "secret-value"
    assert load_api_key(env_file=tmp_path / "absent.env") == ""


def test_env_loader_accepts_export_without_evaluating_shell(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "export HF_TOKEN='hf-test-value'\nIGNORED=$(touch /tmp/must-not-run)\n",
        encoding="utf-8",
    )
    assert load_env_value("HF_TOKEN", path) == "hf-test-value"
    assert load_env_value("ABSENT", path) == ""


def test_config_repr_redacts_api_key():
    fake_secret = "sk-" + "testsecret123456"
    config = make_config(api_key=fake_secret, model="deepseek-v4-flash")
    assert fake_secret not in repr(config)
    assert fake_secret not in json.dumps(config.safe_metadata())


def test_config_rejects_unsupported_values():
    for update, message in (
        ({"api_key": ""}, "api_key"),
        ({"base_url": "http://api.deepseek.com"}, "https"),
        ({"model": "gpt-4"}, "unsupported DeepSeek model"),
        ({"max_tokens": 0}, "positive"),
        ({"max_retries": -1}, "negative"),
        ({"temperature": 3.0}, "temperature"),
        ({"reasoning_effort": "turbo"}, "reasoning_effort"),
        ({"user_id": "bad id"}, "user_id"),
        ({"conversation_log_dir": ""}, "conversation_log_dir"),
    ):
        with pytest.raises(ValueError, match=message):
            make_config(**update)


def test_proxy_retries_timeout_and_parses_native_tool_call():
    transport = FakeTransport(
        [
            TimeoutError("slow"),
            _response(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"main.py"}',
                            },
                        }
                    ],
                }
            ),
        ]
    )
    proxy = DeepSeekProxy(
        make_config(max_retries=1, retry_backoff_seconds=0, api_key="top-secret"),
        transport=transport,
        sleep=lambda _seconds: None,
    )
    result = proxy.complete(
        [ChatMessage("user", "write a strategy")], tools=[{"type": "function"}]
    )
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "main.py"}
    assert len(transport.requests) == 2
    assert transport.requests[0][1]["Authorization"] == "Bearer top-secret"
    assert transport.requests[1][2]["stream"] is True


@pytest.mark.parametrize("raw_call", INVALID_COMPLETE_TOOL_CALLS)
def test_json_fallback_rejects_malformed_tool_call_without_retry(raw_call: object):
    transport = FakeTransport([_response({"content": None, "tool_calls": [raw_call]})])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="invalid tool call") as caught:
        proxy.complete(
            [ChatMessage("user", "inspect")],
            tools=[{"type": "function"}],
        )

    assert caught.value.retryable is False
    assert len(transport.requests) == 1
    assert transport.requests[0][2]["stream"] is True


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_nonstream_reasoning_is_normalized_and_replayed_with_assistant_message(
    field: str,
):
    transport = FakeTransport(
        [
            _response({"content": "done", field: "internal plan"}),
            _response({"content": "continued"}),
        ]
    )
    proxy = DeepSeekProxy(make_config(), transport=transport)
    initial = ChatMessage("user", "research")

    response = proxy.complete([initial])
    assistant = ChatMessage(
        "assistant",
        response.content,
        reasoning_content=response.reasoning_content,
    )
    proxy.complete([initial, assistant, ChatMessage("user", "continue")])

    assert response.reasoning_content == "internal plan"
    assistant_record = transport.requests[1][2]["messages"][1]
    assert assistant_record["reasoning_content"] == "internal plan"
    assert "reasoning" not in assistant_record


def test_response_without_reasoning_keeps_legacy_message_shape():
    transport = FakeTransport([_response({"content": "done"})])
    response = DeepSeekProxy(make_config(), transport=transport).complete(
        [ChatMessage("user", "research")]
    )

    assert response.reasoning_content is None
    assert (
        "reasoning_content"
        not in ChatMessage("assistant", response.content).to_record()
    )


def test_proxy_sends_model_thinking_reasoning_user_id_and_response_budget():
    transport = FakeTransport([_response({"content": "done"})])
    proxy = DeepSeekProxy(
        make_config(
            model="deepseek-v4-pro",
            thinking_enabled=True,
            reasoning_effort="xhigh",
            max_tokens=900,
            temperature=0.25,
            user_id="autotrade_user-1",
        ),
        transport=transport,
    )
    proxy.complete([ChatMessage("user", "research")], max_tokens=700)
    body = transport.requests[0][2]
    assert body["model"] == "deepseek-v4-pro"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "xhigh"
    assert body["max_tokens"] == 700
    assert body["temperature"] == 0.25
    assert body["user_id"] == "autotrade_user-1"
    assert "api_key" not in body


def test_thinking_disabled_omits_reasoning_effort():
    transport = FakeTransport([_response({"content": "done"})])
    proxy = DeepSeekProxy(make_config(), transport=transport)
    proxy.complete([ChatMessage("user", "hi")])
    body = transport.requests[0][2]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert body["stream"] is False


def test_proxy_reassembles_streamed_native_tool_call():
    chunks = [
        {},
        {
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "read_", "arguments": '{"path":'},
                            }
                        ]
                    }
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "file", "arguments": '"main.py"}'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [], "usage": {"total_tokens": 9}},
    ]
    raw = (
        ": keepalive\n\n"
        + "\n\n".join(f"data: {json.dumps(chunk)}" for chunk in chunks)
        + "\n\ndata: [DONE]\n"
    ).encode()
    transport = FakeTransport([raw])
    proxy = DeepSeekProxy(make_config(model="deepseek-v4-pro"), transport=transport)
    result = proxy.complete(
        [ChatMessage("user", "inspect")],
        tools=[{"type": "function"}],
    )
    assert transport.requests[0][2]["stream"] is True
    assert transport.requests[0][2]["stream_options"] == {"include_usage": True}
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "main.py"}
    assert result.usage == {"total_tokens": 9}


def test_proxy_reassembles_streamed_reasoning_with_native_tool_call():
    raw = _stream_response(
        {"choices": [{"delta": {"reasoning_content": "inspect "}}]},
        {"choices": [{"delta": {"reasoning": "the file"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    proxy = DeepSeekProxy(make_config(), transport=FakeTransport([raw]))

    response = proxy.complete(
        [ChatMessage("user", "inspect")], tools=[{"type": "function"}]
    )

    assert response.reasoning_content == "inspect the file"
    assert response.tool_calls[0].name == "read_file"


def _loopback_proxy(base_url: str, *, timeout: float) -> OpenAICompatibleProxy:
    return OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-test-key",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url=base_url,
            timeout_seconds=timeout,
            max_retries=0,
            conversation_log_dir=None,
        )
    )


def test_hard_deadline_stops_sse_drip_that_never_hits_idle_timeout():
    with _provider_server() as root:
        proxy = _loopback_proxy(f"{root}/sse-drip/v1", timeout=0.1)
        started = time.monotonic()
        with pytest.raises(LLMProxyError, match="hard deadline"):
            proxy.complete(
                [ChatMessage("user", "stream")],
                tools=[{"type": "function"}],
            )
        assert time.monotonic() - started < 0.25


@pytest.mark.parametrize("route", ["slow-headers", "nonstream-drip"])
def test_hard_deadline_covers_headers_and_full_nonstream_read(route: str):
    with _provider_server() as root:
        proxy = _loopback_proxy(f"{root}/{route}/v1", timeout=0.05)
        started = time.monotonic()
        with pytest.raises(LLMProxyError, match="hard deadline"):
            proxy.complete([ChatMessage("user", "complete")])
        assert time.monotonic() - started < 0.18


def test_normal_chunked_response_completes_within_hard_deadline():
    with _provider_server() as root:
        proxy = _loopback_proxy(f"{root}/normal-chunked/v1", timeout=1.0)
        assert proxy.complete([ChatMessage("user", "complete")]).content == "ok"


def test_standard_openai_http_error_retains_bounded_redacted_message():
    with _provider_server() as root:
        proxy = _loopback_proxy(f"{root}/openai-error/v1", timeout=1.0)
        with pytest.raises(LLMProxyError, match="maximum context length") as caught:
            proxy.complete([ChatMessage("user", "complete")])
    message = str(caught.value)
    assert caught.value.status_code == 400
    assert caught.value.retryable is False
    assert "local-test-key" not in message
    assert root not in message
    assert "[redacted]" in message
    assert len(message) <= 1_040


def test_http_error_redacts_long_credential_before_final_truncation():
    credential = "SENSITIVE_" + "K" * 5_000
    with _provider_server() as root:
        proxy = OpenAICompatibleProxy(
            OpenAICompatibleConfig(
                api_key=credential,
                provider="vllm",
                model=LOCAL_QWEN_MODEL,
                base_url=f"{root}/credential-boundary/v1",
                max_retries=2,
                conversation_log_dir=None,
            )
        )
        with pytest.raises(LLMProxyError, match="maximum context length") as caught:
            proxy.complete([ChatMessage("user", "complete")])
    message = str(caught.value)
    assert caught.value.status_code == 400
    assert caught.value.retryable is False
    assert "SENSITIVE_" not in message
    assert credential not in message
    assert root not in message
    assert message.startswith("provider HTTP error 400: [redacted]")


def test_http_error_redacts_long_endpoint_before_final_truncation():
    long_path = "p" * 5_000
    with _provider_server() as root:
        base_url = f"{root}/endpoint-boundary/{long_path}/v1"
        proxy = OpenAICompatibleProxy(
            OpenAICompatibleConfig(
                api_key="local-test-key",
                provider="vllm",
                model=LOCAL_QWEN_MODEL,
                base_url=base_url,
                max_retries=2,
                conversation_log_dir=None,
            )
        )
        with pytest.raises(LLMProxyError, match="maximum context length") as caught:
            proxy.complete([ChatMessage("user", "complete")])
    message = str(caught.value)
    assert caught.value.status_code == 400
    assert caught.value.retryable is False
    assert root not in message
    assert long_path[:100] not in message
    assert message.startswith("provider HTTP error 400: [redacted]")


@pytest.mark.parametrize("route", ["oversized-error", "malformed-error"])
def test_http_error_oversized_or_malformed_body_stays_generic(route: str):
    with _provider_server() as root:
        proxy = _loopback_proxy(f"{root}/{route}/v1", timeout=1.0)
        with pytest.raises(LLMProxyError) as caught:
            proxy.complete([ChatMessage("user", "complete")])
    assert str(caught.value) == "provider HTTP error 400"
    assert caught.value.status_code == 400
    assert caught.value.retryable is False


def test_local_context_preflight_rejects_without_transport_or_retry(tmp_path: Path):
    transport = FakeTransport([_response({"content": "must remain unused"})])
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=32_768,
            max_retries=2,
            conversation_log_dir=tmp_path,
        ),
        transport=transport,
    )
    with pytest.raises(LLMProxyError, match="before provider request") as caught:
        proxy.complete(
            [ChatMessage("user", "x" * 100_000)],
            max_tokens=8_000,
        )
    assert caught.value.retryable is False
    assert transport.requests == []
    records = _log_records(tmp_path)
    assert [record["status"] for record in records] == ["started", "error"]
    assert records[-1]["attempt"] == 1
    assert records[-1]["error"]["retryable"] is False


def test_local_context_preflight_rejects_fixed_seed_high_entropy_ascii():
    rng = random.Random(20260815)
    opaque = "".join(rng.choices(string.ascii_letters + string.digits, k=50_000))
    message = ChatMessage("user", opaque)
    assert estimate_chat_request_tokens([message]) >= len(opaque)
    transport = FakeTransport([_response({"content": "must remain unused"})])
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=32_768,
            conversation_log_dir=None,
        ),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="before provider request"):
        proxy.complete([message], max_tokens=8_000)

    assert transport.requests == []


@pytest.mark.parametrize(
    "separator",
    ("\n", "\r\n", " ", "\t", r"\n", r"\r\n"),
    ids=("lf", "crlf", "space", "tab", "escaped-lf", "escaped-crlf"),
)
def test_local_context_preflight_rejects_wrapped_high_entropy_ascii(separator):
    rng = random.Random(20260815)
    opaque = "".join(rng.choices(string.ascii_letters + string.digits, k=50_000))
    wrapped = separator.join(
        opaque[offset : offset + 76] for offset in range(0, len(opaque), 76)
    )
    message = ChatMessage("user", wrapped)
    assert estimate_chat_request_tokens([message]) >= len(opaque)
    transport = FakeTransport([_response({"content": "must remain unused"})])
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=32_768,
            conversation_log_dir=None,
        ),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="before provider request"):
        proxy.complete([message], max_tokens=8_000)

    assert transport.requests == []


def test_local_context_estimate_keeps_long_ordinary_ascii_prompt_usable():
    prose = (
        "Inspect strategy metrics, retain bounded evidence, and run the next "
        "validation step only after checking the current artifact.\n"
    ) * 320
    message = ChatMessage("user", prose)
    transport = FakeTransport([_response({"content": "ok"})])
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=32_768,
            conversation_log_dir=None,
        ),
        transport=transport,
    )

    assert proxy.complete([message], max_tokens=8_000).content == "ok"
    assert len(transport.requests) == 1


def test_gateway_clamps_output_when_estimate_exactly_fills_the_window():
    transport = FakeTransport([_response({"content": "ok"})])
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=32_768,
            conversation_log_dir=None,
        ),
        transport=transport,
    )
    message = ChatMessage("user", "continue after backtest")
    prompt = estimate_chat_request_tokens([message])
    requested = 32_768 - prompt
    assert requested > CONTEXT_OUTPUT_TOKEN_MARGIN
    assert proxy.complete([message], max_tokens=requested).content == "ok"
    assert transport.requests[0][2]["max_tokens"] == (
        requested - CONTEXT_OUTPUT_TOKEN_MARGIN
    )


def _tautology_overflow(output: int, *, window: int = 262_144) -> LLMProxyError:
    prompt = window + 1 - output
    return LLMProxyError(
        "provider HTTP error 400: This model's maximum context length is "
        f"{window} tokens. However, you requested {output} output tokens and "
        f"your prompt contains at least {prompt} input tokens, for a total of "
        f"at least {window + 1} tokens.",
        retryable=False,
        status_code=400,
    )


def test_gateway_halves_output_when_provider_overflow_is_tautological():
    transport = FakeTransport(
        [_tautology_overflow(32_768), _response({"content": "ok"})]
    )
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=262_144,
            max_retries=0,
            conversation_log_dir=None,
        ),
        transport=transport,
    )
    assert (
        proxy.complete([ChatMessage("user", "meta")], max_tokens=32_768).content
        == "ok"
    )
    assert [request[2]["max_tokens"] for request in transport.requests] == [
        32_768,
        16_384,
    ]


def test_gateway_keeps_halving_output_on_repeated_tautological_overflow():
    transport = FakeTransport(
        [
            _tautology_overflow(32_768),
            _tautology_overflow(16_384),
            _response({"content": "ok"}),
        ]
    )
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=262_144,
            max_retries=0,
            conversation_log_dir=None,
        ),
        transport=transport,
    )
    assert (
        proxy.complete([ChatMessage("user", "meta")], max_tokens=32_768).content
        == "ok"
    )
    assert [request[2]["max_tokens"] for request in transport.requests] == [
        32_768,
        16_384,
        8_192,
    ]


def test_local_context_estimate_keeps_structured_json_prompts_usable():
    payload = {
        "records": [
            {
                "symbol": f"{index % 1_000_000:06d}",
                "score": index / 1000,
                "note": "bounded ordinary metric record",
            }
            for index in range(400)
        ]
    }
    contents = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        json.dumps(payload, ensure_ascii=True, indent=2),
    )
    transport = FakeTransport(
        [_response({"content": "ok-minified"}), _response({"content": "ok-pretty"})]
    )
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            context_window_tokens=32_768,
            conversation_log_dir=None,
        ),
        transport=transport,
    )

    for content in contents:
        assert (
            estimate_chat_request_tokens([ChatMessage("user", content)])
            < len(content) // 2 + 1_000
        )
        proxy.complete([ChatMessage("user", content)], max_tokens=8_000)

    assert len(transport.requests) == 2


def test_retry_backoff_cannot_start_after_logical_call_budget_is_exhausted():
    failure = LLMProxyError("provider HTTP error 503", retryable=True, status_code=503)
    transport = FakeTransport([failure, _response({"content": "too late"})])
    proxy = DeepSeekProxy(
        make_config(
            timeout_seconds=0.05,
            max_retries=2,
            retry_backoff_seconds=0.1,
        ),
        transport=transport,
    )
    with pytest.raises(LLMProxyError, match="hard deadline exhausted before retry"):
        proxy.complete([ChatMessage("user", "hello")])
    assert len(transport.requests) == 1
    assert 0 < transport.requests[0][3] <= 0.05


def test_proxy_redacts_key_from_transport_error():
    transport = FakeTransport([RuntimeError("connection failed for top-secret")])
    proxy = DeepSeekProxy(make_config(api_key="top-secret"), transport=transport)
    with pytest.raises(LLMProxyError) as caught:
        proxy.complete([ChatMessage("user", "hello")])
    assert "top-secret" not in str(caught.value)
    assert "[redacted]" in str(caught.value)


def test_proxy_rejects_invalid_provider_payload_without_retry():
    transport = FakeTransport([{"choices": []}])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
    )
    with pytest.raises(LLMProxyError, match="invalid response"):
        proxy.complete([ChatMessage("user", "hello")])
    assert len(transport.requests) == 1


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_provider_http_5xx_and_429_are_retryable(status_code: int):
    failure = LLMProxyError(
        f"provider HTTP error {status_code}", retryable=True, status_code=status_code
    )
    transport = FakeTransport([failure])
    proxy = DeepSeekProxy(make_config(max_retries=0), transport=transport)
    with pytest.raises(LLMProxyError) as caught:
        proxy.complete([ChatMessage("user", "hello")])
    assert caught.value.status_code == status_code
    assert caught.value.retryable is True
    assert len(transport.requests) == 1


def test_provider_http_500_retries_then_succeeds():
    failure = LLMProxyError("provider HTTP error 500", retryable=True, status_code=500)
    transport = FakeTransport([failure, _response({"content": "ok"})])
    proxy = DeepSeekProxy(
        make_config(max_retries=1, retry_backoff_seconds=0),
        transport=transport,
        sleep=lambda _seconds: None,
    )
    assert proxy.complete([ChatMessage("user", "hello")]).content == "ok"
    assert len(transport.requests) == 2


def test_non_retryable_provider_error_stops_after_one_attempt():
    failure = LLMProxyError("provider HTTP error 401", retryable=False, status_code=401)
    transport = FakeTransport([failure])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0), transport=transport
    )
    with pytest.raises(LLMProxyError) as caught:
        proxy.complete([ChatMessage("user", "hello")])
    assert caught.value.retryable is False
    assert len(transport.requests) == 1


def _log_records(root: Path) -> list[dict[str, object]]:
    files = list(root.rglob("*.jsonl"))
    assert len(files) == 1
    return [
        json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()
    ]


def test_complete_writes_conversation_log(tmp_path: Path):
    fake_secret = "sk-" + "testsecret123456"
    transport = FakeTransport(
        [
            {
                "id": "resp",
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": '{"action":"hold"}'}}],
                "usage": {"total_tokens": 12},
            }
        ]
    )
    proxy = DeepSeekProxy(
        make_config(api_key=fake_secret, conversation_log_dir=tmp_path),
        transport=transport,
    )
    proxy.complete(
        [ChatMessage("system", "Return JSON only."), ChatMessage("user", "json please")]
    )
    records = _log_records(tmp_path)
    started, record = records[0], records[-1]
    assert [item["status"] for item in records] == ["started", "ok"]
    assert record["provider"] == "deepseek"
    # The payload is stored once (the first attempt's started record); the
    # terminal record joins it via call_id, not a duplicate.
    assert started["payload"]["messages"][1]["content"] == "json please"
    assert "payload" not in record
    assert record["call_id"] and record["call_id"] == started["call_id"]
    assert (
        record["raw_response"]["choices"][0]["message"]["content"]
        == '{"action":"hold"}'
    )
    assert record["usage"]["total_tokens"] == 12
    # Boundary item 5: no hash field survives anywhere in the audit record.
    assert not [key for key in record if "hash" in key]
    assert fake_secret not in json.dumps(records)


def test_concurrent_callers_append_whole_conversation_log_records(tmp_path: Path):
    """The parent conversation and its sub-agent threads share one per-pid
    log file; every record must land as one intact JSON line."""
    workers = 8
    transport = FakeTransport(
        [_response({"content": "x" * 20_000}) for _ in range(workers)]
    )
    proxy = DeepSeekProxy(
        make_config(max_retries=0, conversation_log_dir=tmp_path), transport=transport
    )
    prompt = [ChatMessage("user", "y" * 50_000)]
    errors: list[BaseException] = []

    def call() -> None:
        try:
            proxy.complete(prompt)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == []
    records = _log_records(tmp_path)
    assert sorted(record["status"] for record in records) == ["ok"] * workers + [
        "started"
    ] * workers
    assert len({record["call_id"] for record in records}) == workers


def test_failed_call_writes_conversation_log(tmp_path: Path):
    fake_secret = "sk-" + "secretvalue123456"
    failure = LLMProxyError(
        "provider HTTP error 401 for " + fake_secret, retryable=False, status_code=401
    )
    proxy = DeepSeekProxy(
        make_config(api_key=fake_secret, max_retries=0, conversation_log_dir=tmp_path),
        transport=FakeTransport([failure]),
    )
    with pytest.raises(LLMProxyError) as raised:
        proxy.complete([ChatMessage("user", "json please")])
    records = _log_records(tmp_path)
    record = records[-1]
    assert [item["status"] for item in records] == ["started", "error"]
    assert record["error"]["status_code"] == 401
    assert record["http_status_code"] == 401
    assert "[redacted]" in record["error"]["message"]
    assert "[redacted]" in str(raised.value)
    assert fake_secret not in json.dumps(records)


def test_every_attempt_of_a_retried_call_is_logged_under_one_call_id(tmp_path: Path):
    failure = LLMProxyError("provider HTTP error 503", retryable=True, status_code=503)
    proxy = DeepSeekProxy(
        make_config(
            max_retries=1, retry_backoff_seconds=0, conversation_log_dir=tmp_path
        ),
        transport=FakeTransport([failure, _response({"content": "ok"})]),
        sleep=lambda _seconds: None,
    )
    proxy.complete([ChatMessage("user", "hello")])
    records = _log_records(tmp_path)
    assert [(item["status"], item["attempt"]) for item in records] == [
        ("started", 1),
        ("error", 1),
        ("started", 2),
        ("ok", 2),
    ]
    assert len({item["call_id"] for item in records}) == 1
    assert [item for item in records if "payload" in item] == [records[0]]


@pytest.mark.parametrize(
    ("failure", "tools", "error_message"), RETRYABLE_PROTOCOL_FAILURES
)
def test_retryable_protocol_error_succeeds_on_third_attempt(
    failure: bytes | dict[str, object],
    tools: tuple[dict[str, str], ...],
    error_message: str,
    tmp_path: Path,
):
    delays = []
    transport = FakeTransport([failure, failure, _response({"content": "ok"})])
    proxy = DeepSeekProxy(
        make_config(
            max_retries=2,
            retry_backoff_seconds=0.25,
            conversation_log_dir=tmp_path,
        ),
        transport=transport,
        sleep=delays.append,
    )

    result = proxy.complete(
        [ChatMessage("user", "hello")],
        tools=tools,
    )

    assert result.content == "ok"
    assert len(transport.requests) == 3
    assert delays == [0.25, 0.5]
    records = _log_records(tmp_path)
    assert [(item["status"], item["attempt"]) for item in records] == [
        ("started", 1),
        ("error", 1),
        ("started", 2),
        ("error", 2),
        ("started", 3),
        ("ok", 3),
    ]
    errors = [item["error"] for item in records if item["status"] == "error"]
    assert [item["message"] for item in errors] == [error_message, error_message]
    assert all(item["retryable"] is True for item in errors)
    assert len({item["call_id"] for item in records}) == 1


@pytest.mark.parametrize(
    ("failure", "tools", "error_message"), RETRYABLE_PROTOCOL_FAILURES
)
def test_retryable_protocol_error_is_explicit_after_three_failures(
    failure: bytes | dict[str, object],
    tools: tuple[dict[str, str], ...],
    error_message: str,
):
    transport = FakeTransport([failure, failure, failure])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(LLMProxyError, match=error_message) as caught:
        proxy.complete(
            [ChatMessage("user", "hello")],
            tools=tools,
        )

    assert caught.value.retryable is True
    assert len(transport.requests) == 3


@pytest.mark.parametrize("invalid_content", [[], False, 0, {}])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_message_content_type_is_not_retried(
    invalid_content: object,
    stream: bool,
):
    if stream:
        failure = _stream_response(
            {"choices": [{"delta": {"content": invalid_content}}]}
        )
        tools = ({"type": "function"},)
    else:
        failure = _response({"content": invalid_content})
        tools = ()
    transport = FakeTransport([failure])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="invalid message content") as caught:
        proxy.complete([ChatMessage("user", "hello")], tools=tools)

    assert caught.value.retryable is False
    assert len(transport.requests) == 1


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_reasoning_type_is_not_retried(field: str, stream: bool):
    invalid_reasoning = ["not", "text"]
    if stream:
        failure = _stream_response({"choices": [{"delta": {field: invalid_reasoning}}]})
        tools = ({"type": "function"},)
    else:
        failure = _response({"content": "done", field: invalid_reasoning})
        tools = ()
    transport = FakeTransport([failure])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="invalid reasoning content") as caught:
        proxy.complete([ChatMessage("user", "hello")], tools=tools)

    assert caught.value.retryable is False
    assert len(transport.requests) == 1


def test_missing_message_content_without_tool_calls_is_not_retried():
    transport = FakeTransport([_response({})])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="invalid message content") as caught:
        proxy.complete([ChatMessage("user", "hello")])

    assert caught.value.retryable is False
    assert len(transport.requests) == 1


@pytest.mark.parametrize("chunk", INVALID_STREAM_STRUCTURES)
def test_invalid_stream_structure_is_not_retried(chunk: object):
    transport = FakeTransport([_stream_response(chunk)])
    proxy = DeepSeekProxy(
        make_config(max_retries=2, retry_backoff_seconds=0),
        transport=transport,
    )

    with pytest.raises(LLMProxyError, match="invalid stream") as caught:
        proxy.complete(
            [ChatMessage("user", "hello")],
            tools=[{"type": "function"}],
        )

    assert caught.value.retryable is False
    assert len(transport.requests) == 1


def test_conversation_log_redacts_sensitive_dict_keys(tmp_path: Path):
    payload = {
        "id": "resp-sk-secretvalue123456",
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": '{"action":"hold"}'}}],
        "usage": {"total_tokens": 12, "secret": "usage-secret"},
        "api_key": "plain-secret",
        "authorization": "Bearer plain-token",
        "notes": "provider echoed Authorization: Bearer plain-bearer-token",
        "nested": {"token": "plain-token", "total_tokens": 12},
    }
    proxy = DeepSeekProxy(
        make_config(conversation_log_dir=tmp_path), transport=FakeTransport([payload])
    )
    proxy.complete(
        [ChatMessage("system", "Return JSON only."), ChatMessage("user", "json please")]
    )
    records = _log_records(tmp_path)
    record = records[-1]
    assert record["raw_response"]["api_key"] == "[REDACTED]"
    assert record["raw_response"]["authorization"] == "[REDACTED]"
    assert record["raw_response"]["nested"]["token"] == "[REDACTED]"
    assert record["raw_response"]["nested"]["total_tokens"] == 12
    assert "redacted" in str(record["response_id"]).lower()
    assert record["usage"]["total_tokens"] == 12
    assert record["usage"]["[redacted-credential]"] == "[REDACTED]"
    dumped = json.dumps(records)
    for leaked in ("plain-secret", "plain-token", "plain-bearer-token", "usage-secret"):
        assert leaked not in dumped


def test_conversation_log_redacts_configured_credential_value_everywhere(
    tmp_path: Path,
):
    credential = "opaque-configured-value-4815162342"
    returned_content = f"prefix {credential} suffix"
    payload = {
        "model": "deepseek-chat",
        "choices": [{"message": {"content": returned_content}}],
        "usage": {
            "total_tokens": 3,
            "nested": {credential: [f"embedded-{credential}-value"]},
        },
    }
    proxy = DeepSeekProxy(
        make_config(api_key=credential, conversation_log_dir=tmp_path),
        transport=FakeTransport([payload]),
    )
    response = proxy.complete([ChatMessage("user", returned_content)])

    memory_response_is_unchanged = response.content == returned_content
    assert memory_response_is_unchanged
    persisted = "\n".join(path.read_text() for path in tmp_path.rglob("*.jsonl"))
    configured_value_leaked = credential in persisted
    assert configured_value_leaked is False
    records = _log_records(tmp_path)
    assert "[redacted-credential]" in records[-1]["raw_response"]["usage"]["nested"]


def test_conversation_log_failure_stops_before_provider_call(tmp_path: Path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    transport = FakeTransport([_response({"content": "unreachable"})])
    proxy = DeepSeekProxy(
        make_config(conversation_log_dir=blocked), transport=transport
    )
    with pytest.raises(LLMProxyError, match="conversation log"):
        proxy.complete([ChatMessage("user", "json please")])
    assert transport.requests == []


def test_conversation_logging_is_disabled_by_an_explicit_none(tmp_path: Path):
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmpdir:
        proxy = DeepSeekProxy(
            make_config(), transport=FakeTransport([_response({"content": "ok"})])
        )
        proxy.complete([ChatMessage("user", "hello")])
        assert list(Path(tmpdir).rglob("*.jsonl")) == []
    assert make_config().safe_metadata()["conversation_logging_enabled"] is False


def test_configured_provider_url_never_enters_persistent_audit(tmp_path: Path):
    custom_url = "https://private-provider.example.test/custom/v1"
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url=custom_url,
            conversation_log_dir=tmp_path,
        ),
        transport=FakeTransport([_response({"content": custom_url})]),
    )
    response = proxy.complete([ChatMessage("user", f"endpoint is {custom_url}")])
    assert response.content == custom_url
    records = _log_records(tmp_path)
    persisted = json.dumps(records)
    assert custom_url not in persisted
    assert "private-provider.example.test" not in persisted
    assert "base_url" not in proxy.config.safe_metadata()


def test_local_vllm_uses_shared_stream_parser_without_deepseek_fields():
    transport = FakeTransport(
        [
            _stream_response(
                {
                    "model": LOCAL_QWEN_MODEL,
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-local",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"main.py"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ],
                },
                {"choices": [], "usage": {"total_tokens": 17}},
            )
        ]
    )
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            thinking_enabled=True,
            reasoning_effort="xhigh",
            max_tokens=8_000,
            max_output_tokens=16_384,
            user_id="",
            conversation_log_dir=None,
        ),
        transport=transport,
    )
    response = proxy.complete(
        [ChatMessage("user", "inspect")],
        tools=[{"type": "function"}],
        max_tokens=8_000,
    )
    request_url, _headers, body, _timeout = transport.requests[0]
    assert request_url == "http://127.0.0.1:8010/v1/chat/completions"
    assert body["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert not {"thinking", "reasoning_effort", "user_id"} & body.keys()
    assert body["max_tokens"] == 8_000
    assert response.tool_calls[0].arguments == {"path": "main.py"}
    assert response.usage == {"total_tokens": 17}


def test_local_vllm_rejects_public_plain_http():
    with pytest.raises(ValueError, match="private-network HTTP"):
        OpenAICompatibleConfig(
            api_key="secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://8.8.8.8:8010/v1",
            request_dialect="vllm-qwen",
        )


def test_openai_compatible_config_rejects_invalid_output_cap():
    with pytest.raises(ValueError, match="max_output_tokens"):
        OpenAICompatibleConfig(
            api_key="secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            max_output_tokens=0,
        )


def test_generic_provider_rejects_model_path_traversal():
    with pytest.raises(ValueError, match="safe identifier"):
        OpenAICompatibleConfig(
            api_key="secret",
            provider="vllm",
            model="../../escaped",
            base_url="http://127.0.0.1:8010/v1",
        )


def test_model_factory_reads_local_endpoint_and_key_from_export_env(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "export VLLM_API_KEY='local-secret'\n"
        "export VLLM_BASE_URL='http://127.0.0.1:8011/v1'\n",
        encoding="utf-8",
    )
    proxy = build_model_gateway(LOCAL_QWEN_MODEL, env_file=path, max_tokens=8_000)
    assert isinstance(proxy, OpenAICompatibleProxy)
    assert proxy.provider == "vllm"
    assert proxy.model == LOCAL_QWEN_MODEL
    assert proxy.config.base_url == "http://127.0.0.1:8011/v1"
    assert proxy.context_window_tokens == 262_144
    assert model_profile(LOCAL_QWEN_MODEL).max_output_tokens == 262_144
    assert proxy.config.max_tokens == 8_000
    assert proxy.config.max_output_tokens == 262_144
    assert "local-secret" not in repr(proxy.config)

    transport = FakeTransport([_response({"content": "bounded"})])
    bounded = OpenAICompatibleProxy(proxy.config, transport=transport)
    assert (
        bounded.complete(
            [ChatMessage("user", "continue after backtest")], max_tokens=8_000
        ).content
        == "bounded"
    )
    assert transport.requests[0][2]["max_tokens"] == 8_000


def test_legacy_local_model_alias_builds_canonical_gateway(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("VLLM_API_KEY=local-secret\n", encoding="utf-8")

    proxy = build_model_gateway(
        LEGACY_LOCAL_QWEN_MODEL,
        env_file=path,
        conversation_log_dir=tmp_path / "logs",
    )

    assert LEGACY_LOCAL_QWEN_MODEL not in MODEL_CHOICES
    assert model_profile(LEGACY_LOCAL_QWEN_MODEL) == model_profile(LOCAL_QWEN_MODEL)
    assert proxy.model == LOCAL_QWEN_MODEL
    assert proxy.config.model == LOCAL_QWEN_MODEL
    assert proxy._conversation_log_path("2026-08-27T00:00:00+00:00").parent == (
        tmp_path / "logs" / "vllm" / LOCAL_QWEN_MODEL
    )


def test_model_factory_reads_deepseek_endpoint_from_fixed_env_key(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "export DEEPSEEK_API_KEY='provider-key'\n"
        "export DEEPSEEK_BASE_URL='https://trusted-deepseek.example.test/v1'\n",
        encoding="utf-8",
    )
    proxy = build_model_gateway("deepseek-v4-flash", env_file=path, max_tokens=8_000)
    assert isinstance(proxy, DeepSeekProxy)
    assert proxy.config.base_url == "https://trusted-deepseek.example.test/v1"
    assert proxy.config.max_tokens == 8_000
    assert proxy.config.max_output_tokens is None


def test_local_conversation_log_uses_truthful_provider_path(tmp_path: Path):
    proxy = OpenAICompatibleProxy(
        OpenAICompatibleConfig(
            api_key="local-secret",
            provider="vllm",
            model=LOCAL_QWEN_MODEL,
            base_url="http://127.0.0.1:8010/v1",
            request_dialect="vllm-qwen",
            conversation_log_dir=tmp_path,
        ),
        transport=FakeTransport([_response({"content": "ok"})]),
    )
    proxy.complete([ChatMessage("user", "hello")])
    files = list((tmp_path / f"vllm/{LOCAL_QWEN_MODEL}").glob("*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert {record["provider"] for record in records} == {"vllm"}
    assert "local-secret" not in json.dumps(records)


def test_reasoning_only_response_is_accepted_without_retry() -> None:
    from autotrade.environment.llm.proxy import ProviderResponse

    response = ProviderResponse(content="", reasoning_content="internal plan")
    assert response.content == ""
    assert response.reasoning_content == "internal plan"
    transport = FakeTransport(
        [_response({"content": "", "reasoning_content": "internal plan"})]
    )
    proxy = DeepSeekProxy(make_config(max_retries=0), transport=transport)
    parsed = proxy.complete([ChatMessage("user", "hello")])
    assert parsed.content == ""
    assert parsed.reasoning_content == "internal plan"
    assert len(transport.requests) == 1
