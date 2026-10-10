"""LightingTheWord (New API relay) gateway.

The relay serves its gpt-6 family over ``/v1/responses`` — that family refuses
function tools on Chat Completions — and its Anthropic ids and
DeepSeek-V4.1-Flash over ``/v1/chat/completions``, with a primary credential
and an alternate one.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProxy,
    _parse_complete_tool_call,
    _provider_response_validation_error,
    load_env_value,
)
from .proxy import (
    ChatMessage,
    LLMProxy,
    LLMProxyError,
    ProviderRefusalError,
    ProviderResponse,
    ToolCall,
)

if TYPE_CHECKING:  # annotation-only: model_profiles imports this module
    from .model_profiles import ModelProfile

API_KEY_ENV = "LIGHTING_API_KEY"
API_KEY_ALT_ENV = "LIGHTING_API_KEY_ALT"
BASE_URL_ENV = "LIGHTING_BASE_URL"
DEFAULT_BASE_URL = "https://lightingtheword.com/v1"

# The relay's gpt-6 family needs the responses request contract; every other
# relay id speaks a dialect the shared gateway already implements.
RESPONSES_DIALECT = "openai-responses"
CHAT_DIALECT = "openai"
DEEPSEEK_DIALECT = "deepseek"

# (relay model id, request dialect, context window, max output tokens). The
# hosted gpt-6 ids declare a 272k window; the Anthropic ids and
# DeepSeek-V4.1-Flash declare 1M, with the relay's own completion ceilings.
MODEL_PROFILES: tuple[tuple[str, str, int, int], ...] = (
    ("gpt-6.1-sol", RESPONSES_DIALECT, 272_000, 128_000),
    ("gpt-6-astra", RESPONSES_DIALECT, 272_000, 128_000),
    ("gpt-6-luna", RESPONSES_DIALECT, 272_000, 128_000),
    ("claude-fable-5-1", CHAT_DIALECT, 1_000_000, 128_000),
    ("claude-sonnet-5-5", CHAT_DIALECT, 1_000_000, 128_000),
    ("claude-opus-5-5", CHAT_DIALECT, 1_000_000, 128_000),
    ("DeepSeek-V4.1-Flash", DEEPSEEK_DIALECT, 1_000_000, 384_000),
)
MODEL_CHOICES = tuple(name for name, _dialect, _window, _output in MODEL_PROFILES)

# The responses endpoint takes the shared effort scale narrowed to its own
# tiers; the shared scale's top tier is xhigh.
_RESPONSES_REASONING_EFFORTS = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
    "xhigh": "xhigh",
}

# A rejected or exhausted relay key answers with 401/402/403, or with a
# 400/429 whose message names the account state ("insufficient quota",
# "余额不足"). Only those attempts move to the alternate key; every other
# failure keeps the shared provider semantics.
_CREDENTIAL_STATUS_CODES = frozenset({401, 402, 403})
_EXHAUSTED_ACCOUNT_MARKERS = (
    "quota",
    "balance",
    "credit",
    "billing",
    "余额",
    "额度",
)


def is_credential_failure(error: LLMProxyError) -> bool:
    """Whether one provider error means the credential was rejected or empty."""

    if error.status_code in _CREDENTIAL_STATUS_CODES:
        return True
    if error.status_code in {400, 429}:
        message = str(error).lower()
        return any(marker in message for marker in _EXHAUSTED_ACCOUNT_MARKERS)
    return False


def build_gateway(
    *,
    profile: ModelProfile,
    model: str,
    api_key: str,
    base_url: str,
    env_file: str | Path,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
    reasoning_effort: str | None,
    conversation_log_dir: str | Path | None,
) -> LLMProxy:
    """Build one relay role gateway; called only by ``model_profiles``.

    The endpoint and both credentials come from the profile's fixed
    environment keys, so no experiment parameter can point them elsewhere.
    """

    return cast(
        LLMProxy,
        LightingProxy(
            OpenAICompatibleConfig(
                api_key=api_key,
                provider=profile.provider,
                model=model,
                base_url=base_url,
                request_dialect=profile.request_dialect,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                max_tokens=max_tokens,
                max_output_tokens=profile.max_output_tokens,
                temperature=temperature,
                thinking_enabled=thinking_enabled,
                # Only the relay's two dialects that carry a level get one: the
                # hosted Claude ids accept no reasoning control at all, and the
                # responses dialect sends no field when thinking is off (the
                # gpt-6 family has no "none" tier except gpt-6-luna).
                reasoning_effort=(
                    reasoning_effort
                    if thinking_enabled
                    and profile.request_dialect
                    in {RESPONSES_DIALECT, DEEPSEEK_DIALECT}
                    else None
                ),
                user_id="",
                conversation_log_dir=conversation_log_dir,
                context_window_tokens=profile.context_window_tokens,
            ),
            fallback_api_key=load_env_value(API_KEY_ALT_ENV, env_file),
        ),
    )


class LightingProxy(OpenAICompatibleProxy):
    """The relay's own dialect, credential fallback and nothing else.

    Its Anthropic and DeepSeek ids use the shared Chat Completions dialects
    unchanged; only the gpt-6 family replaces the request payload and the
    response parser with the responses contract.
    """

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        fallback_api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.fallback_api_key = fallback_api_key

    def with_thinking(
        self, *, enabled: bool, reasoning_effort: str | None
    ) -> OpenAICompatibleProxy:
        """Clone with another thinking setting, keeping the relay behavior.

        The shared clone would return a plain Chat Completions proxy, which
        would lose both the responses dialect and the alternate key.
        """

        return LightingProxy(
            replace(
                self.config,
                thinking_enabled=enabled,
                reasoning_effort=reasoning_effort,
            ),
            fallback_api_key=self.fallback_api_key,
            transport=self._transport,
            sleep=self._sleep,
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = "auto",
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        try:
            return super().complete(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
        except LLMProxyError as error:
            if (
                not self.fallback_api_key
                or self.fallback_api_key == self.config.api_key
                or not is_credential_failure(error)
            ):
                raise
            _log_credential_switch(self.config.model, error)
            # The alternate key is one more logical call: it logs its own
            # attempts, and a failure there still reports the original
            # credential error. The clone carries no alternate key, so a call
            # can switch at most once.
            retry = type(self)(
                replace(self.config, api_key=self.fallback_api_key),
                transport=self._transport,
                sleep=self._sleep,
            )
            try:
                return retry.complete(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                )
            except LLMProxyError:
                raise error from None

    def _output_token_field(self) -> str:
        if self.config.request_dialect != RESPONSES_DIALECT:
            return super()._output_token_field()
        return "max_output_tokens"

    def _build_request_body(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]],
        tool_choice: str | Mapping[str, object],
        stream: bool,
        max_tokens: int,
    ) -> dict[str, object]:
        if self.config.request_dialect != RESPONSES_DIALECT:
            return super()._build_request_body(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                max_tokens=max_tokens,
            )
        return _responses_body(
            self.config,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            max_output_tokens=max_tokens,
        )

    def _parse_response_payload(
        self, payload: bytes, *, stream: bool
    ) -> ProviderResponse:
        if self.config.request_dialect != RESPONSES_DIALECT:
            return super()._parse_response_payload(payload, stream=stream)
        if stream:
            return _parse_responses_stream(payload, expected_model=self.config.model)
        return _parse_responses(payload, expected_model=self.config.model)


def _log_credential_switch(model: str, error: LLMProxyError) -> None:
    """One operator line naming the switch; it never carries a credential."""

    print(
        f"lighting: {model} rejected the primary credential "
        f"(HTTP {error.status_code}); retrying with {API_KEY_ALT_ENV}",
        file=sys.stderr,
    )


def _responses_body(
    config: OpenAICompatibleConfig,
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[Mapping[str, object]],
    tool_choice: str | Mapping[str, object],
    stream: bool,
    max_output_tokens: int,
) -> dict[str, object]:
    """One responses request payload.

    Sampling is left to the relay: the responses contract takes no temperature
    next to a reasoning effort, and the shared non-thinking sampling values
    are the local Qwen gateway's own recommendation.
    """

    body: dict[str, object] = {
        "model": config.model,
        "input": _responses_input(messages),
        "stream": stream,
        "max_output_tokens": max_output_tokens,
    }
    if config.thinking_enabled and config.reasoning_effort is not None:
        body["reasoning"] = {
            "effort": _RESPONSES_REASONING_EFFORTS[config.reasoning_effort]
        }
    if tools:
        body["tools"] = [_responses_tool(tool) for tool in tools]
        body["tool_choice"] = _responses_tool_choice(tool_choice)
    return body


def _responses_input(messages: Sequence[ChatMessage]) -> list[dict[str, object]]:
    """Map the conversation onto responses input items.

    Assistant reasoning is deliberately absent: the relay's reasoning models
    keep it server-side, and an empty assistant text carries nothing to send.
    """

    items: list[dict[str, object]] = []
    for message in messages:
        if message.role in {"system", "user"}:
            items.append(
                {
                    "role": message.role,
                    "content": [
                        {"type": "input_text", "text": message.content or ""}
                    ],
                }
            )
        elif message.role == "assistant":
            if message.content:
                items.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": message.content}
                        ],
                    }
                )
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False, allow_nan=False
                        ),
                    }
                )
        else:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content or "",
                }
            )
    return items


def _responses_tool(tool: Mapping[str, object]) -> dict[str, object]:
    """One Chat Completions tool schema in the responses flat form."""

    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("relay tools must use the chat-completions function shape")
    converted: dict[str, object] = {"type": "function", "name": function.get("name")}
    for key in ("description", "parameters"):
        if function.get(key) is not None:
            converted[key] = function[key]
    return converted


def _responses_tool_choice(
    tool_choice: str | Mapping[str, object],
) -> str | dict[str, object]:
    """The responses form of one caller's tool_choice."""

    if not isinstance(tool_choice, Mapping):
        return tool_choice
    function = tool_choice.get("function")
    if isinstance(function, Mapping):
        return {"type": "function", "name": function.get("name")}
    return dict(tool_choice)


def _invalid_response() -> LLMProxyError:
    return LLMProxyError("provider returned an invalid response", retryable=False)


def _invalid_stream() -> LLMProxyError:
    return LLMProxyError("provider returned an invalid stream", retryable=False)


def _refusal(message: str) -> ProviderRefusalError:
    return ProviderRefusalError(
        "provider refused the request (responses refusal); "
        f"nothing from this reply was used. Provider message: {message.strip() or '(none)'}"
    )


def _responses_usage(value: object) -> dict[str, object]:
    """The responses usage in the shared prompt/completion shape.

    The responses contract counts ``input_tokens``/``output_tokens``, but the
    host's accounting and truncation checks read the Chat Completions names.
    The original keys are kept and the shared names added next to them.
    """

    if not isinstance(value, Mapping):
        return {}
    usage = dict(value)
    for shared, own in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
    ):
        count = usage.get(own)
        if shared not in usage and isinstance(count, int) and not isinstance(count, bool):
            usage[shared] = count
    return usage


def _responses_failure(decoded: Mapping[str, object]) -> LLMProxyError | None:
    error = decoded.get("error")
    if not isinstance(error, Mapping):
        return None
    detail = error.get("message") or error.get("code") or "no detail"
    return LLMProxyError(f"provider response failed: {detail}", retryable=False)


def _tool_call_from_item(
    item: Mapping[str, object], *, output_limited: bool, content: str
) -> ToolCall:
    return _parse_complete_tool_call(
        {
            "id": item.get("call_id"),
            "function": {
                "name": item.get("name"),
                "arguments": item.get("arguments"),
            },
        },
        output_limited=output_limited,
        content=content,
    )


def _parse_responses(payload: bytes, *, expected_model: str) -> ProviderResponse:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid_response() from exc
    if not isinstance(decoded, Mapping):
        raise _invalid_response()
    return _responses_provider_response(decoded, expected_model=expected_model)


def _responses_provider_response(
    decoded: Mapping[str, object], *, expected_model: str
) -> ProviderResponse:
    """Assemble one responses object: a JSON body or a terminal SSE payload."""

    failure = _responses_failure(decoded)
    if failure is not None:
        raise failure
    incomplete = decoded.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, Mapping) else None
    output = decoded.get("output")
    if not isinstance(output, list):
        raise _invalid_response()
    text: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise _invalid_response()
        kind = item.get("type")
        if kind == "message":
            parts = item.get("content")
            if not isinstance(parts, list):
                raise _invalid_response()
            for part in parts:
                if not isinstance(part, Mapping):
                    raise _invalid_response()
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise _invalid_response()
                    text.append(part["text"])
                elif part.get("type") == "refusal":
                    value = part.get("refusal")
                    raise _refusal(value if isinstance(value, str) else "")
        elif kind == "function_call":
            # A generation cut at the output ceiling can leave arguments that
            # still parse; the status says the call was truncated.
            calls.append(
                _tool_call_from_item(
                    item,
                    output_limited=reason == "max_output_tokens",
                    content="".join(text),
                )
            )
    if reason == "content_filter":
        raise _refusal("")
    model = decoded.get("model") or expected_model
    try:
        return ProviderResponse(
            content="".join(text),
            tool_calls=tuple(calls),
            model=str(model),
            usage=_responses_usage(decoded.get("usage")),
        )
    except ValueError as exc:
        raise _provider_response_validation_error(exc) from exc


def _parse_responses_stream(payload: bytes, *, expected_model: str) -> ProviderResponse:
    """Reassemble the responses SSE events into one normal response."""

    # A transport that ignores ``stream=true`` answers with the JSON body.
    if payload.lstrip().startswith(b"{"):
        return _parse_responses(payload, expected_model=expected_model)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _invalid_stream() from exc
    content: list[str] = []
    refusal: list[str] = []
    calls: dict[str, dict[str, str]] = {}
    order: list[str] = []
    terminal: Mapping[str, object] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        item = stripped[5:].strip()
        if not item or item == "[DONE]":
            continue
        try:
            event = json.loads(item)
        except json.JSONDecodeError as exc:
            raise _invalid_stream() from exc
        if not isinstance(event, Mapping):
            raise _invalid_stream()
        kind = event.get("type")
        if kind in {"response.completed", "response.incomplete", "response.failed"}:
            response = event.get("response")
            if not isinstance(response, Mapping):
                raise _invalid_stream()
            terminal = response
        elif kind == "error":
            detail = event.get("message") or event.get("code") or "no detail"
            raise LLMProxyError(f"provider stream failed: {detail}", retryable=False)
        elif kind == "response.output_item.added":
            _record_function_call(event, calls, order, final=False)
        elif kind == "response.output_item.done":
            _record_function_call(event, calls, order, final=True)
        elif kind == "response.function_call_arguments.delta":
            record = calls.get(_event_call_key(event, None))
            delta = event.get("delta")
            if record is None or not isinstance(delta, str):
                raise _invalid_stream()
            record["arguments"] += delta
        elif kind == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise _invalid_stream()
            content.append(delta)
        elif kind == "response.refusal.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                refusal.append(delta)
    if refusal:
        raise _refusal("".join(refusal))
    if terminal is None and not content and not calls:
        # Nothing assembled and no terminal event: the stream was cut off or
        # empty, which the shared retry policy treats as transient.
        return _responses_provider_response(
            {"output": []}, expected_model=expected_model
        )
    if terminal is not None and not content and not calls:
        # A stream whose deltas carried nothing still ends with the whole
        # response on the terminal event.
        return _responses_provider_response(terminal, expected_model=expected_model)
    usage: Mapping[str, object] = {}
    model = expected_model
    output_limited = False
    if terminal is not None:
        failure = _responses_failure(terminal)
        if failure is not None:
            raise failure
        usage = _responses_usage(terminal.get("usage"))
        if terminal.get("model"):
            model = str(terminal["model"])
        incomplete = terminal.get("incomplete_details")
        output_limited = (
            isinstance(incomplete, Mapping)
            and incomplete.get("reason") == "max_output_tokens"
        )
    assembled = "".join(content)
    built: list[ToolCall] = []
    for key in order:
        record = calls[key]
        built.append(
            _tool_call_from_item(
                {
                    "call_id": record["call_id"],
                    "name": record["name"],
                    "arguments": record["arguments"],
                },
                output_limited=output_limited,
                content=assembled,
            )
        )
    try:
        return ProviderResponse(
            content=assembled,
            tool_calls=tuple(built),
            model=model,
            usage=usage,
        )
    except ValueError as exc:
        raise _provider_response_validation_error(exc) from exc


def _event_call_key(event: Mapping[str, object], item: object) -> str:
    index = event.get("output_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return f"index:{index}"
    item_id = item.get("id") if isinstance(item, Mapping) else None
    if not isinstance(item_id, str) or not item_id:
        item_id = event.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise _invalid_stream()
    return f"item:{item_id}"


def _record_function_call(
    event: Mapping[str, object],
    calls: dict[str, dict[str, str]],
    order: list[str],
    *,
    final: bool,
) -> None:
    item = event.get("item")
    if not isinstance(item, Mapping) or item.get("type") != "function_call":
        return
    key = _event_call_key(event, item)
    record = calls.get(key)
    if record is None:
        record = {"call_id": "", "name": "", "arguments": ""}
        calls[key] = record
        order.append(key)
    for field in ("call_id", "name"):
        value = item.get(field)
        if isinstance(value, str) and value:
            record[field] = value
    arguments = item.get("arguments")
    if final:
        # The completed item carries the whole argument JSON; a non-empty
        # value supersedes the assembled deltas.
        if not isinstance(arguments, str):
            raise _invalid_stream()
        if arguments:
            record["arguments"] = arguments


__all__ = [
    "API_KEY_ALT_ENV",
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "DEFAULT_BASE_URL",
    "MODEL_CHOICES",
    "MODEL_PROFILES",
    "LightingProxy",
    "build_gateway",
    "is_credential_failure",
]
