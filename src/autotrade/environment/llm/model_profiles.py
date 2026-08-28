"""Single model catalog and secure gateway factory for host-side LLM roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from . import deepseek as _deepseek
from .deepseek import (
    DeepSeekConfig,
    DeepSeekProxy,
    OpenAICompatibleConfig,
    OpenAICompatibleProxy,
    load_env_value,
)
from .proxy import LLMProxy

LOCAL_QWEN_MODEL = "qwen-3.8-27b-fp8"
LEGACY_LOCAL_QWEN_MODEL = "qwen3.8-27b-local"
MODEL_CHOICES = (LOCAL_QWEN_MODEL, *_deepseek.MODEL_CHOICES)


def canonicalize_model_name(model: str) -> str:
    """Map read-compatible legacy aliases to the current catalog name."""

    if model == LEGACY_LOCAL_QWEN_MODEL:
        return LOCAL_QWEN_MODEL
    return model


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    api_key_env: str
    base_url_env: str | None
    default_base_url: str
    request_dialect: str
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None


_DEEPSEEK_PROFILE = ModelProfile(
    provider="deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url_env="DEEPSEEK_BASE_URL",
    default_base_url="https://api.deepseek.com",
    request_dialect="deepseek",
    # DeepSeek chat/reasoner models serve a 128K context; the per-generation
    # output ceiling stays with the provider.
    context_window_tokens=128_000,
)
_VLLM_PROFILE = ModelProfile(
    provider="vllm",
    api_key_env="VLLM_API_KEY",
    base_url_env="VLLM_BASE_URL",
    default_base_url="http://127.0.0.1:8011/v1",
    request_dialect="vllm-qwen",
    context_window_tokens=262_144,
    max_output_tokens=262_144,
)

# Completion-token safety ceiling for the Fold/Meta parent conversation and
# its sub-agents (thinking tokens included). Observed calls pool around 1.4k
# tokens with p90 ≈ 4k; only a runaway outlier class (20k+) is cut. Do not
# lower it further: a truncated thinking block loses the whole answer.
# compact, NL and analysis keep their own budgets.
AGENT_MAX_OUTPUT_TOKENS = 12_000


def _qwen_reasoning_effort(value: str | None) -> str:
    """Map the shared UI scale onto the gateway's low/medium/xhigh contract."""

    return {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "max": "xhigh",
        "xhigh": "xhigh",
    }.get(value or "", "xhigh")


def _gateway_api_key(env_file: str | Path, *, required: bool) -> str:
    """Resolve the local gateway key from the environment variable or env file."""

    key = load_env_value(_VLLM_PROFILE.api_key_env, env_file)
    if key or not required:
        return key
    raise ValueError(
        f"model {LOCAL_QWEN_MODEL} requires the gateway API key: set "
        f"{_VLLM_PROFILE.api_key_env} in the environment or {env_file}"
    )


def model_profile(model: str) -> ModelProfile:
    model = canonicalize_model_name(model)
    if model in _deepseek.SUPPORTED_MODELS:
        return _DEEPSEEK_PROFILE
    if model == LOCAL_QWEN_MODEL:
        return _VLLM_PROFILE
    raise ValueError(f"unsupported DeepSeek model or local model: {model}")


def effective_max_output_tokens(model: str, requested: int) -> int:
    """Apply the trusted model profile's per-generation output ceiling."""

    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("requested output tokens must be a positive integer")
    limit = model_profile(model).max_output_tokens
    return min(requested, limit) if limit is not None else requested


def build_model_gateway(
    model: str,
    *,
    env_file: str | Path = ".env",
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY",
    timeout_seconds: float = 600.0,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.5,
    max_tokens: int = 1_200,
    temperature: float = 0.0,
    thinking_enabled: bool = False,
    reasoning_effort: str | None = None,
    conversation_log_dir: str | Path | None = "data/llm_conversations",
    require_credentials: bool = True,
) -> LLMProxy:
    """Build one role gateway from a catalog model name.

    Endpoint environment keys and local credential names are fixed by the
    trusted model profile; neither can be supplied by experiment parameters.
    """

    model = canonicalize_model_name(model)
    profile = model_profile(model)
    effective_max_tokens = effective_max_output_tokens(model, max_tokens)
    if profile.provider == "deepseek":
        key = load_env_value(deepseek_api_key_env, env_file)
        if not key and require_credentials:
            raise ValueError(
                f"model {model} requires an API key in "
                f"{deepseek_api_key_env} or {env_file}"
            )
        key = key or "preflight"
    else:
        key = _gateway_api_key(env_file, required=require_credentials) or "preflight"
    base_url = (
        load_env_value(profile.base_url_env, env_file)
        if profile.base_url_env is not None
        else ""
    ) or profile.default_base_url
    if profile.provider == "deepseek":
        return cast(
            LLMProxy,
            DeepSeekProxy(
                DeepSeekConfig(
                    api_key=key,
                    model=model,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    thinking_enabled=thinking_enabled,
                    reasoning_effort=reasoning_effort,
                    conversation_log_dir=conversation_log_dir,
                    context_window_tokens=profile.context_window_tokens,
                )
            ),
        )
    return cast(
        LLMProxy,
        OpenAICompatibleProxy(
            OpenAICompatibleConfig(
                api_key=key,
                provider=profile.provider,
                model=model,
                base_url=base_url,
                request_dialect=profile.request_dialect,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                max_tokens=effective_max_tokens,
                max_output_tokens=profile.max_output_tokens,
                temperature=temperature,
                thinking_enabled=thinking_enabled,
                reasoning_effort=(
                    _qwen_reasoning_effort(reasoning_effort)
                    if thinking_enabled
                    else None
                ),
                user_id="",
                conversation_log_dir=conversation_log_dir,
                context_window_tokens=profile.context_window_tokens,
            )
        ),
    )


__all__ = [
    "AGENT_MAX_OUTPUT_TOKENS",
    "LEGACY_LOCAL_QWEN_MODEL",
    "LOCAL_QWEN_MODEL",
    "MODEL_CHOICES",
    "ModelProfile",
    "build_model_gateway",
    "canonicalize_model_name",
    "effective_max_output_tokens",
    "model_profile",
]
