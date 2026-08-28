"""Shared OpenAI-compatible Chat Completions gateway.

DeepSeek remains the default public service.  The generic implementation also
supports an authenticated local vLLM endpoint without duplicating transport,
stream assembly, retry, or audit behavior.
"""

from __future__ import annotations

import ipaddress
import json
import os
import queue
import re
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from autotrade.environment.runtime import new_id, utc_now_iso

from .conversation_log import (
    ConversationLogConfig,
    _conversation_log_record,
    _ensure_log_parent,
)
from .proxy import (
    ChatMessage,
    LLMProxyError,
    ProviderResponse,
    ToolCall,
    CONTEXT_OUTPUT_MAX_SHRINKS,
    clamp_requested_max_tokens,
    context_overflow_error,
    context_request_fits,
    is_context_overflow_error,
    max_tokens_after_provider_overflow,
)

MODEL_CHOICES = ("deepseek-v4-pro", "deepseek-v4-flash")
SUPPORTED_MODELS = frozenset((*MODEL_CHOICES, "deepseek-chat", "deepseek-reasoner"))
SUPPORTED_REASONING_EFFORTS = frozenset({"low", "medium", "high", "max", "xhigh"})
# The local gateway contract accepts only these tiers; shared entry points
# map the wider UI scale onto them, and the transport rejects the rest.
_QWEN_REASONING_EFFORTS = frozenset({"low", "medium", "xhigh"})
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,512}$")
_EMPTY_PROVIDER_RESPONSE_ERROR = "provider response must contain content or tool calls"
_HTTP_ERROR_BODY_MAX_BYTES = 64 * 1024
_RUNTIME_ERROR_MAX_CHARS = 1_025
# The local Qwen service owns sampling in thinking mode; these values are the
# official non-thinking recommendation and are sent verbatim otherwise.
_QWEN_NON_THINKING_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
}


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_private_host(host: str) -> bool:
    """Loopback or RFC1918-style private address: the trusted local network."""

    if _is_loopback_host(host):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def _validate_base_url(value: str) -> None:
    target = urlsplit(value)
    if not target.hostname or target.username or target.password:
        raise ValueError("base_url must be an absolute URL without credentials")
    if target.query or target.fragment:
        raise ValueError("base_url cannot contain a query or fragment")
    if target.scheme == "https":
        return
    if target.scheme == "http" and _is_private_host(target.hostname):
        return
    raise ValueError("base_url must use HTTPS or private-network HTTP")


class HTTPTransport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> bytes: ...


class _DirectHTTPTransport:
    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> bytes:
        target = urlsplit(url)
        if not target.hostname:
            raise ValueError("provider URL must include a hostname")
        if target.scheme == "https":
            connection_type = HTTPSConnection
        elif target.scheme == "http" and _is_private_host(target.hostname):
            connection_type = HTTPConnection
        else:
            raise ValueError("provider URL must use HTTPS or private-network HTTP")
        path = target.path or "/"
        if target.query:
            path = f"{path}?{target.query}"
        hostname = target.hostname
        port = target.port
        deadline = time.monotonic() + timeout
        cancelled = threading.Event()
        connections: list[HTTPConnection] = []
        outcome: queue.SimpleQueue[tuple[bool, bytes | Exception]] = queue.SimpleQueue()

        def request() -> None:
            connection = connection_type(hostname, port, timeout=timeout)
            connections.append(connection)
            try:
                if cancelled.is_set():
                    raise TimeoutError("provider call exceeded hard deadline")
                self._set_remaining_timeout(connection, deadline)
                connection.connect()
                if cancelled.is_set():
                    raise TimeoutError("provider call exceeded hard deadline")
                self._set_remaining_timeout(connection, deadline)
                connection.request("POST", path, body=body, headers=dict(headers))
                if cancelled.is_set():
                    raise TimeoutError("provider call exceeded hard deadline")
                self._set_remaining_timeout(connection, deadline)
                response = connection.getresponse()
                chunks: list[bytes] = []
                response_bytes = 0
                error_response = not 200 <= response.status < 300
                while True:
                    if cancelled.is_set():
                        raise TimeoutError("provider call exceeded hard deadline")
                    self._set_remaining_timeout(connection, deadline)
                    read_size = 64 * 1024
                    if error_response:
                        remaining_error_bytes = (
                            _HTTP_ERROR_BODY_MAX_BYTES - response_bytes
                        )
                        if remaining_error_bytes <= 0:
                            break
                        read_size = min(read_size, remaining_error_bytes)
                    chunk = response.read1(read_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    response_bytes += len(chunk)
                payload = b"".join(chunks)
                if error_response:
                    retryable = response.status in {429, 500, 503}
                    raise LLMProxyError(
                        _http_error_message(response.status, payload),
                        retryable=retryable,
                        status_code=response.status,
                    )
                outcome.put((True, payload))
            except Exception as exc:  # noqa: BLE001 - propagate on the caller thread
                outcome.put((False, exc))
            finally:
                connection.close()

        worker = threading.Thread(
            target=request,
            name="llm-http-request",
            daemon=True,
        )
        worker.start()
        worker.join(max(deadline - time.monotonic(), 0.0))
        if worker.is_alive():
            cancelled.set()
            for connection in connections:
                connection.close()
            raise TimeoutError("provider call exceeded hard deadline")
        try:
            succeeded, value = outcome.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError("provider transport finished without a result") from exc
        if succeeded:
            if not isinstance(value, bytes):
                raise TypeError("provider transport returned a non-bytes payload")
            return value
        if isinstance(value, Exception):
            raise value
        raise RuntimeError("provider transport returned an invalid result")

    @staticmethod
    def _set_remaining_timeout(connection: HTTPConnection, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("provider call exceeded hard deadline")
        connection.timeout = remaining
        if connection.sock is not None:
            connection.sock.settimeout(remaining)


def load_env_value(env_var: str, env_file: str | Path = ".env") -> str:
    """Read one explicit environment value without evaluating the env file.

    Both ``NAME=value`` and the common ``export NAME=value`` spelling are
    accepted.  Shell expansion and command execution are deliberately absent.
    """

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_var):
        raise ValueError("env_var must be an environment variable name")
    direct = os.environ.get(env_var, "").strip()
    if direct:
        return direct
    path = Path(env_file)
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if key == env_var:
            return value.strip().strip("'\"")
    return ""


def load_api_key(
    env_var: str = "DEEPSEEK_API_KEY", env_file: str | Path = ".env"
) -> str:
    """Backward-compatible DeepSeek credential loader."""

    return load_env_value(env_var, env_file)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str = field(repr=False)
    provider: str
    model: str
    base_url: str
    request_dialect: str = "openai"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    max_tokens: int = 1200
    max_output_tokens: int | None = None
    temperature: float = 0.0
    thinking_enabled: bool = False
    reasoning_effort: str | None = None
    stream_tool_calls: bool = True
    user_id: str = "autotrade-hl"
    conversation_log_dir: str | Path | None = "data/llm_conversations"
    context_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.api_key or "\r" in self.api_key or "\n" in self.api_key:
            raise ValueError("api_key must be non-empty and single-line")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.provider):
            raise ValueError("provider must be a lowercase identifier")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", self.model):
            raise ValueError("model must be a safe identifier")
        _validate_base_url(self.base_url)
        if self.request_dialect not in {"openai", "deepseek", "vllm-qwen"}:
            raise ValueError("unsupported request dialect")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError("timeout_seconds and max_tokens must be positive")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        if self.max_retries < 0 or self.retry_backoff_seconds < 0:
            raise ValueError("retry settings cannot be negative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")
        if (
            self.reasoning_effort is not None
            and self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS
        ):
            raise ValueError(
                "reasoning_effort must be one of low, medium, high, max, xhigh"
            )
        if (
            self.request_dialect == "vllm-qwen"
            and self.reasoning_effort is not None
            and self.reasoning_effort not in _QWEN_REASONING_EFFORTS
        ):
            raise ValueError(
                "vllm-qwen reasoning_effort must be one of low, medium, xhigh"
            )
        if self.user_id and not USER_ID_PATTERN.fullmatch(self.user_id):
            raise ValueError(
                "user_id must match [A-Za-z0-9_-] and be at most 512 chars"
            )
        if self.conversation_log_dir == "":
            raise ValueError(
                "conversation_log_dir cannot be empty; use None to disable logging"
            )
        if self.context_window_tokens is not None and (
            isinstance(self.context_window_tokens, bool)
            or not isinstance(self.context_window_tokens, int)
            or self.context_window_tokens <= 0
        ):
            raise ValueError("context_window_tokens must be a positive integer or None")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def safe_metadata(self) -> dict[str, Any]:
        metadata = {
            "provider": self.provider,
            "model": self.model,
            "request_dialect": self.request_dialect,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "user_id": self.user_id,
            "conversation_logging_enabled": self.conversation_log_dir is not None,
            "context_window_tokens": self.context_window_tokens,
        }
        if self.max_output_tokens is not None:
            metadata["max_output_tokens"] = self.max_output_tokens
        return metadata


class OpenAICompatibleProxy:
    config_type = OpenAICompatibleConfig

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: HTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._transport = transport or _DirectHTTPTransport()
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def context_window_tokens(self) -> int | None:
        return self.config.context_window_tokens

    def with_thinking(
        self, *, enabled: bool, reasoning_effort: str | None
    ) -> OpenAICompatibleProxy:
        """Clone this proxy with another thinking setting.

        Shares the transport and clock; the config is rebuilt from its field
        values so a compatibility facade config (DeepSeekConfig) clones too.
        """

        values = {
            field.name: getattr(self.config, field.name)
            for field in dataclass_fields(self.config)
            if field.init
        }
        values.update(thinking_enabled=enabled, reasoning_effort=reasoning_effort)
        return OpenAICompatibleProxy(
            OpenAICompatibleConfig(**values),
            transport=self._transport,
            sleep=self._sleep,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        env_var: str = "DEEPSEEK_API_KEY",
        env_file: str | Path = ".env",
        **config: Any,
    ) -> OpenAICompatibleProxy:
        key = load_env_value(env_var, env_file)
        if not key:
            raise ValueError(f"missing API key in {env_var} or {env_file}")
        return cls(cls.config_type(api_key=key, **config))

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = "auto",
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if not messages:
            raise ValueError("messages cannot be empty")
        stream = bool(tools and self.config.stream_tool_calls)
        requested_max_tokens = max_tokens or self.config.max_tokens
        if self.config.max_output_tokens is not None:
            requested_max_tokens = min(
                requested_max_tokens, self.config.max_output_tokens
            )
        # Vendor-like behaviour: never reject a request merely because the
        # requested output budget does not fit. Clamp max_tokens to the
        # remaining window minus tokenizer slack. Only a prompt that leaves
        # no room after that slack is a true overflow.
        _fits, estimated_prompt_tokens, context_window = context_request_fits(
            self,
            messages,
            tools=tools,
            max_tokens=requested_max_tokens,
        )
        requested_max_tokens, fits = clamp_requested_max_tokens(
            requested_max_tokens=requested_max_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_window=context_window,
        )
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [message.to_record() for message in messages],
            "stream": stream,
            "max_tokens": requested_max_tokens,
        }
        if self.config.request_dialect == "vllm-qwen":
            # Thinking mode keeps the server-side sampling defaults; the
            # non-thinking mode sends the official recommendation verbatim.
            if not self.config.thinking_enabled:
                body.update(_QWEN_NON_THINKING_SAMPLING)
        else:
            body["temperature"] = self.config.temperature
        if self.config.request_dialect == "deepseek":
            body["thinking"] = {
                "type": "enabled" if self.config.thinking_enabled else "disabled"
            }
            if self.config.reasoning_effort is not None:
                body["reasoning_effort"] = self.config.reasoning_effort
            if self.config.user_id:
                body["user_id"] = self.config.user_id
        elif self.config.request_dialect == "vllm-qwen":
            chat_template_kwargs: dict[str, object] = {
                "enable_thinking": self.config.thinking_enabled
            }
            if (
                self.config.thinking_enabled
                and self.config.reasoning_effort is not None
            ):
                chat_template_kwargs["reasoning_effort"] = self.config.reasoning_effort
            body["chat_template_kwargs"] = chat_template_kwargs
        if stream:
            # Ask the compatible endpoint to include final usage in the SSE
            # stream; the transport still enforces the logical call deadline.
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = [dict(item) for item in tools]
            body["tool_choice"] = tool_choice
        raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        attempts = self.config.max_retries + 1
        call_deadline = time.monotonic() + self.config.timeout_seconds
        # One call_id per logical call: every attempt's log records reference
        # it, and the full payload is written once (the first attempt's
        # "started" record); terminal/retry records join via call_id.
        call_id = new_id("call")
        include_next_payload = True
        network_attempt = 0
        shrinks = 0
        post_index = 0
        while True:
            post_index += 1
            started_at = utc_now_iso()
            started_perf = time.perf_counter()
            log_path = self._conversation_log_path(started_at)
            include_payload = include_next_payload
            include_next_payload = False
            self._write_conversation_log(
                log_path,
                _conversation_log_record(
                    config=cast(ConversationLogConfig, self.config),
                    payload=body,
                    started_at=started_at,
                    completed_at=None,
                    duration_seconds=0.0,
                    attempt=post_index,
                    max_attempts=attempts,
                    status="started",
                    call_id=call_id,
                    include_payload=include_payload,
                ),
            )
            try:
                if not fits:
                    assert context_window is not None
                    raise context_overflow_error(
                        estimated_prompt_tokens=estimated_prompt_tokens,
                        max_tokens=requested_max_tokens,
                        context_window=context_window,
                    )
                payload = self._transport.post(
                    self.config.endpoint,
                    headers,
                    raw,
                    _remaining_call_seconds(call_deadline),
                )
                response = (
                    _parse_stream_response(payload, expected_model=self.config.model)
                    if stream
                    else _parse_response(payload, expected_model=self.config.model)
                )
                _remaining_call_seconds(call_deadline)
            # pi-lens-ignore: ast-grep:no-boolean-in-except
            except (LLMProxyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                error = self._normalize_error(exc)
                self._log_attempt(
                    log_path,
                    body,
                    started_at=started_at,
                    started_perf=started_perf,
                    attempt=post_index,
                    max_attempts=attempts,
                    status="error",
                    call_id=call_id,
                    http_status_code=error.status_code,
                    error=error,
                )
                if (
                    fits
                    and shrinks < CONTEXT_OUTPUT_MAX_SHRINKS
                    and is_context_overflow_error(error)
                ):
                    shrunk = max_tokens_after_provider_overflow(
                        error, requested_max_tokens=requested_max_tokens
                    )
                    if shrunk is not None:
                        requested_max_tokens = shrunk
                        body["max_tokens"] = requested_max_tokens
                        raw = json.dumps(
                            body, ensure_ascii=False, allow_nan=False
                        ).encode("utf-8")
                        shrinks += 1
                        include_next_payload = True
                        continue
                retry_delay = self.config.retry_backoff_seconds * (2**network_attempt)
                can_retry = error.retryable and network_attempt + 1 < attempts
                if can_retry and call_deadline - time.monotonic() <= retry_delay:
                    error = LLMProxyError(
                        "provider call hard deadline exhausted before retry",
                        retryable=True,
                        status_code=error.status_code,
                    )
                    can_retry = False
                if not can_retry:
                    raise error from None
                network_attempt += 1
                if retry_delay:
                    self._sleep(retry_delay)
                if time.monotonic() >= call_deadline:
                    raise LLMProxyError(
                        "provider call hard deadline exhausted during retry backoff",
                        retryable=True,
                    ) from None
                continue
            self._log_attempt(
                log_path,
                body,
                started_at=started_at,
                started_perf=started_perf,
                attempt=post_index,
                max_attempts=attempts,
                status="ok",
                call_id=call_id,
                raw_response=_decoded_response(payload) if not stream else None,
                response_body=None
                if not stream
                else payload.decode("utf-8", "replace"),
            )
            return response

    def _log_attempt(
        self,
        log_path: Path | None,
        payload: dict[str, object],
        *,
        started_at: str,
        started_perf: float,
        **extra: Any,
    ) -> None:
        # One emitter for every outcome of this attempt; the record shape (and
        # its credential redaction) is the same for success and failure.
        self._write_conversation_log(
            log_path,
            _conversation_log_record(
                config=cast(ConversationLogConfig, self.config),
                payload=payload,
                started_at=started_at,
                completed_at=utc_now_iso(),
                duration_seconds=time.perf_counter() - started_perf,
                **extra,
            ),
        )

    def _conversation_log_path(self, started_at: str) -> Path | None:
        if self.config.conversation_log_dir is None:
            return None
        date_key = started_at[:10].replace("-", "")
        # Per-process file: concurrent HITL workers share the log dir, and an
        # append over PIPE_BUF is not atomic across writers — isolating by pid
        # makes interleaving impossible without any locking.
        name = f"{date_key}-p{os.getpid()}.jsonl"
        return (
            Path(self.config.conversation_log_dir)
            / self.config.provider
            / self.config.model
            / name
        )

    def _write_conversation_log(
        self, path: Path | None, record: dict[str, Any]
    ) -> None:
        if path is None:
            return
        try:
            _ensure_log_parent(path)
            # Within one process every caller (Runner, NL/sub-agent host service)
            # drives the provider serially, so per-pid appends never interleave.
            with path.open("a", encoding="utf-8") as handle:
                safe_record = _redact_audit_details(
                    record,
                    base_url=self.config.base_url,
                    endpoint=self.config.endpoint,
                    api_key=self.config.api_key,
                )
                handle.write(
                    json.dumps(safe_record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        except OSError as exc:
            raise LLMProxyError(
                f"failed to write provider conversation log: {path}"
            ) from exc

    def _normalize_error(self, exc: Exception) -> LLMProxyError:
        if isinstance(exc, LLMProxyError):
            message = self._bounded_runtime_error(
                self._redact_runtime_details(str(exc))
            )
            return LLMProxyError(
                message,
                retryable=exc.retryable,
                status_code=exc.status_code,
            )
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return LLMProxyError("provider call exceeded hard deadline", retryable=True)
        return LLMProxyError(
            self._bounded_runtime_error(
                self._redact_runtime_details(f"provider response failed: {exc}")
            ),
            retryable=False,
        )

    def _redact_runtime_details(self, message: str) -> str:
        for value in (self.config.api_key, self.config.endpoint, self.config.base_url):
            if value:
                message = message.replace(value, "[redacted]")
        return message

    @staticmethod
    def _bounded_runtime_error(message: str) -> str:
        if len(message) <= _RUNTIME_ERROR_MAX_CHARS:
            return message
        return message[: _RUNTIME_ERROR_MAX_CHARS - 3] + "..."


def _http_error_message(status: int, payload: bytes) -> str:
    """Retain only a bounded standard OpenAI error message for classification."""

    detail = ""
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, Mapping):
        error = decoded.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            detail = str(error["message"]).strip()
    if detail:
        return f"provider HTTP error {status}: {detail}"
    return f"provider HTTP error {status}"


def _remaining_call_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("provider call exceeded hard deadline")
    return remaining


def _redact_audit_details(
    value: Any,
    *,
    base_url: str,
    endpoint: str,
    api_key: str,
) -> Any:
    if isinstance(value, str):
        replacements = (
            (api_key, "[redacted-credential]"),
            (endpoint, "[redacted-provider-url]"),
            (base_url, "[redacted-provider-url]"),
            (base_url.rstrip("/"), "[redacted-provider-url]"),
        )
        for configured_value, replacement in sorted(
            set(replacements), key=lambda item: len(item[0]), reverse=True
        ):
            if configured_value:
                value = value.replace(configured_value, replacement)
        return value
    if isinstance(value, dict):
        return {
            _redact_audit_details(
                key,
                base_url=base_url,
                endpoint=endpoint,
                api_key=api_key,
            ): _redact_audit_details(
                item,
                base_url=base_url,
                endpoint=endpoint,
                api_key=api_key,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_audit_details(
                item,
                base_url=base_url,
                endpoint=endpoint,
                api_key=api_key,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_audit_details(
                item,
                base_url=base_url,
                endpoint=endpoint,
                api_key=api_key,
            )
            for item in value
        )
    return value


def _decoded_response(raw: bytes) -> dict[str, Any] | None:
    """Best-effort JSON body for the conversation log; never fails the call."""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _provider_response_validation_error(exc: ValueError) -> LLMProxyError:
    # An empty response has not exposed any assistant action to the Runner, so
    # replay is safe. Other response-validation errors remain permanent.
    return LLMProxyError(str(exc), retryable=str(exc) == _EMPTY_PROVIDER_RESPONSE_ERROR)


def _parse_reasoning_content(record: Mapping[str, object]) -> tuple[bool, str | None]:
    """Normalize one compatible reasoning field to the gateway contract."""

    if "reasoning_content" in record:
        value = record["reasoning_content"]
    elif "reasoning" in record:
        value = record["reasoning"]
    else:
        return False, None
    if value is None:
        return True, None
    if not isinstance(value, str):
        raise LLMProxyError(
            "provider returned invalid reasoning content", retryable=False
        )
    return True, value


def _parse_complete_tool_call(
    raw_call: object,
    *,
    error_message: str,
    retryable: bool,
) -> ToolCall:
    """Validate one complete OpenAI-compatible tool call without coercion."""

    try:
        if not isinstance(raw_call, Mapping):
            raise TypeError("tool call is not an object")
        call_id = raw_call["id"]
        function = raw_call["function"]
        if not isinstance(call_id, str):
            raise TypeError("tool call id is not a string")
        if not isinstance(function, Mapping):
            raise TypeError("tool function is not an object")
        name = function["name"]
        arguments_json = function.get("arguments", "")
        if not isinstance(name, str):
            raise TypeError("tool function name is not a string")
        if not isinstance(arguments_json, str):
            raise TypeError("tool arguments are not a string")
        arguments = json.loads(arguments_json or "{}")
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments are not an object")
        return ToolCall(call_id, name, arguments)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LLMProxyError(error_message, retryable=retryable) from exc


def _parse_response(raw: bytes, *, expected_model: str) -> ProviderResponse:
    try:
        payload = json.loads(raw.decode("utf-8"))
        choice = payload["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict):
            raise TypeError("message is not an object")
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise LLMProxyError(
            "provider returned an invalid response", retryable=False
        ) from exc
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    elif not isinstance(raw_calls, list):
        raise LLMProxyError("provider returned an invalid tool call", retryable=False)
    for raw_call in raw_calls:
        calls.append(
            _parse_complete_tool_call(
                raw_call,
                error_message="provider returned an invalid tool call",
                retryable=False,
            )
        )
    _reasoning_present, reasoning_content = _parse_reasoning_content(message)
    if "content" not in message:
        if not calls and not (
            _reasoning_present
            and isinstance(reasoning_content, str)
            and reasoning_content.strip()
        ):
            raise LLMProxyError(
                "provider returned invalid message content", retryable=False
            )
        content = ""
    elif message["content"] is None:
        content = ""
    elif not isinstance(message["content"], str):
        raise LLMProxyError(
            "provider returned invalid message content", retryable=False
        )
    else:
        content = message["content"]
    model = payload.get("model") or expected_model
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    try:
        return ProviderResponse(
            content=content,
            tool_calls=tuple(calls),
            model=str(model),
            usage=usage,
            reasoning_content=reasoning_content,
        )
    except ValueError as exc:
        raise _provider_response_validation_error(exc) from exc


def _parse_stream_response(raw: bytes, *, expected_model: str) -> ProviderResponse:
    """Reassemble OpenAI-compatible SSE deltas into one normal response."""

    # Some compatible transports ignore ``stream=true`` and return JSON.
    if raw.lstrip().startswith(b"{"):
        return _parse_response(raw, expected_model=expected_model)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LLMProxyError(
            "provider returned an invalid stream", retryable=False
        ) from exc
    content: list[str] = []
    reasoning_content: list[str] = []
    reasoning_content_seen = False
    tool_order: list[int] = []
    tools: dict[int, dict[str, str]] = {}
    model = expected_model
    usage: dict[str, object] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        item = stripped[5:].strip()
        if item == "[DONE]":
            break
        try:
            chunk = json.loads(item)
        except json.JSONDecodeError as exc:
            raise LLMProxyError(
                "provider returned an invalid stream", retryable=False
            ) from exc
        if not isinstance(chunk, Mapping):
            raise LLMProxyError("provider returned an invalid stream", retryable=False)
        if chunk.get("model"):
            model = str(chunk["model"])
        if isinstance(chunk.get("usage"), dict):
            usage = dict(chunk["usage"])
        if "choices" not in chunk:
            continue
        choices = chunk["choices"]
        if not isinstance(choices, list) or not all(
            isinstance(choice, Mapping) for choice in choices
        ):
            raise LLMProxyError("provider returned an invalid stream", retryable=False)
        if not choices or "delta" not in choices[0]:
            continue
        delta = choices[0]["delta"]
        if not isinstance(delta, Mapping):
            raise LLMProxyError("provider returned an invalid stream", retryable=False)
        if "content" in delta and delta["content"] is not None:
            if not isinstance(delta["content"], str):
                raise LLMProxyError(
                    "provider returned invalid message content", retryable=False
                )
            content.append(delta["content"])
        reasoning_present, reasoning_delta = _parse_reasoning_content(delta)
        if reasoning_present and reasoning_delta is not None:
            reasoning_content_seen = True
            reasoning_content.append(reasoning_delta)
        if "tool_calls" not in delta or delta["tool_calls"] is None:
            continue
        raw_calls = delta["tool_calls"]
        if not isinstance(raw_calls, list):
            raise LLMProxyError("provider returned an invalid stream", retryable=False)
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise LLMProxyError(
                    "provider returned an invalid stream", retryable=False
                )
            index = raw_call.get("index", 0)
            if not isinstance(index, int) or isinstance(index, bool):
                raise LLMProxyError("provider returned an invalid stream tool index")
            if index not in tools:
                tools[index] = {"id": "", "name": "", "arguments": ""}
                tool_order.append(index)
            assembled = tools[index]
            call_id = raw_call.get("id")
            if call_id is not None and not isinstance(call_id, str):
                raise LLMProxyError(
                    "provider returned an invalid stream", retryable=False
                )
            if call_id:
                assembled["id"] += call_id
            function = raw_call.get("function")
            if function is not None and not isinstance(function, Mapping):
                raise LLMProxyError(
                    "provider returned an invalid stream", retryable=False
                )
            if function is not None:
                name = function.get("name")
                arguments = function.get("arguments")
                if name is not None and not isinstance(name, str):
                    raise LLMProxyError(
                        "provider returned an invalid stream", retryable=False
                    )
                if arguments is not None and not isinstance(arguments, str):
                    raise LLMProxyError(
                        "provider returned an invalid stream", retryable=False
                    )
                if name:
                    assembled["name"] += name
                if arguments:
                    assembled["arguments"] += arguments
    calls: list[ToolCall] = []
    for index in tool_order:
        assembled = tools[index]
        # Assembly finishes before complete() returns, so no partial tool call
        # can reach the Runner or produce an environment side effect.
        calls.append(
            _parse_complete_tool_call(
                {
                    "id": assembled["id"],
                    "function": {
                        "name": assembled["name"],
                        "arguments": assembled["arguments"],
                    },
                },
                error_message="provider returned an invalid stream tool call",
                retryable=True,
            )
        )
    try:
        return ProviderResponse(
            content="".join(content),
            tool_calls=tuple(calls),
            model=model,
            usage=usage,
            reasoning_content=(
                "".join(reasoning_content) if reasoning_content_seen else None
            ),
        )
    except ValueError as exc:
        raise _provider_response_validation_error(exc) from exc


class DeepSeekConfig(OpenAICompatibleConfig):
    """Compatibility facade retaining the original DeepSeek constructor."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        max_tokens: int = 1200,
        temperature: float = 0.0,
        thinking_enabled: bool = False,
        reasoning_effort: str | None = None,
        stream_tool_calls: bool = True,
        user_id: str = "autotrade-hl",
        conversation_log_dir: str | Path | None = "data/llm_conversations",
        context_window_tokens: int | None = None,
    ) -> None:
        if not str(base_url).startswith("https://"):
            raise ValueError("base_url must use https")
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported DeepSeek model: {model}")
        super().__init__(
            api_key=api_key,
            provider="deepseek",
            model=model,
            base_url=base_url,
            request_dialect="deepseek",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            stream_tool_calls=stream_tool_calls,
            user_id=user_id,
            conversation_log_dir=conversation_log_dir,
            context_window_tokens=context_window_tokens,
        )


class DeepSeekProxy(OpenAICompatibleProxy):
    """Compatibility facade over the shared OpenAI-compatible proxy."""

    config_type = DeepSeekConfig
