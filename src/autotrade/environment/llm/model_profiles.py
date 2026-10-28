"""Single model catalog and secure gateway factory for host-side LLM roles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from . import deepseek as _deepseek
from . import lighting as _lighting
from .deepseek import DeepSeekConfig, DeepSeekProxy
from .openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProxy,
    load_env_value,
)
from .pricing import FREE, TokenPrices
from .proxy import LLMProxy, ScriptedLLM

LOCAL_QWEN_MODEL = "qwen-3.8-27b-fp8"
# Xiaomi's hosted models; the API takes ids in lower case only.
MIMO_FLASH_MODEL = "mimo-v2.6-flash"
MIMO_PRO_MODEL = "mimo-v2.6-pro"
MIMO_MODELS = (MIMO_FLASH_MODEL, MIMO_PRO_MODEL)
MODEL_CHOICES = (
    LOCAL_QWEN_MODEL,
    *_deepseek.MODEL_CHOICES,
    *MIMO_MODELS,
    *_lighting.MODEL_CHOICES,
)


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    api_key_env: str
    base_url_env: str | None
    default_base_url: str
    request_dialect: str
    # List prices the session's API spend cap charges (``pricing``); FREE
    # only for the locally served model.
    prices: TokenPrices
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None


_DEEPSEEK_PROFILE = ModelProfile(
    provider="deepseek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url_env="DEEPSEEK_BASE_URL",
    default_base_url="https://api.deepseek.com",
    request_dialect="deepseek",
    # The weekday-peak list rate as of 2026-10-14 (docs/parameters-reference.md
    # §3); off-peak hours bill half, which the cap ignores. DeepSeek reports no
    # cache write.
    prices=TokenPrices(0.30, 0.006, 1.20, 0.30),
    # What the account's GET /models reported for deepseek-flash on 2026-10-02.
    context_window_tokens=1_048_576,
    max_output_tokens=393_216,
)
_VLLM_PROFILE = ModelProfile(
    provider="vllm",
    api_key_env="VLLM_API_KEY",
    base_url_env="VLLM_BASE_URL",
    default_base_url="http://127.0.0.1:8011/v1",
    request_dialect="vllm-qwen",
    prices=FREE,
    context_window_tokens=262_144,
    max_output_tokens=262_144,
)
_MIMO_FLASH_PROFILE = ModelProfile(
    provider="mimo",
    api_key_env="MIMO_API_KEY",
    base_url_env="MIMO_BASE_URL",
    # Pay-as-you-go endpoint; Token Plan keys and overseas accounts get their
    # own base URL from the MiMo console, set through MIMO_BASE_URL.
    default_base_url="https://api.xiaomimimo.com/v1",
    request_dialect="mimo",
    # The domestic real-time list prices as of 2026-10-14
    # (docs/parameters-reference.md §3), CNY 1.00 / 0.02 / 2.00 converted at
    # 7.1. MiMo reports no cache write.
    prices=TokenPrices(0.141, 0.0028, 0.282, 0.141),
    # Flash and Pro published the same request contract (as of 2026-10-02), a
    # "1M" window and a 131,072-token completion ceiling (thinking tokens
    # included); the decimal reading of "1M" is the safe one.
    context_window_tokens=1_000_000,
    max_output_tokens=131_072,
)
_MIMO_PROFILES = {
    MIMO_FLASH_MODEL: _MIMO_FLASH_PROFILE,
    # Only the price differs: CNY 3.00 / 0.025 / 6.00 at 7.1.
    MIMO_PRO_MODEL: replace(
        _MIMO_FLASH_PROFILE, prices=TokenPrices(0.423, 0.0035, 0.845, 0.423)
    ),
}
_LIGHTING_PROFILES = {
    model: ModelProfile(
        provider="lighting",
        api_key_env=_lighting.API_KEY_ENV,
        base_url_env=_lighting.BASE_URL_ENV,
        default_base_url=_lighting.DEFAULT_BASE_URL,
        request_dialect=dialect,
        prices=prices,
        context_window_tokens=context_window,
        max_output_tokens=max_output,
    )
    for model, dialect, context_window, max_output, prices in _lighting.MODEL_PROFILES
}

_PROFILES = (
    _DEEPSEEK_PROFILE,
    _VLLM_PROFILE,
    *_MIMO_PROFILES.values(),
    *_LIGHTING_PROFILES.values(),
)

# Smallest declared window in the catalog. A component that must pick a
# window-dependent default before the run's model roles are known (the
# compaction threshold) bounds itself here; the worker replaces it with the
# exact per-role bound once the roles are resolved.
MIN_CONTEXT_WINDOW_TOKENS = min(
    profile.context_window_tokens
    for profile in _PROFILES
    if profile.context_window_tokens is not None
)

# Transient provider failures (HTTP 429/500/503, transport timeouts, an empty
# or cut-off stream) are retried with exponential backoff from this base:
# 2 s, 4 s, 8 s, the same schedule pi uses, inside the logical call deadline.
# The context-overflow shrink path is not a transient error and never waits.
DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 2.0

# Completion-token ceiling for the research-session parent conversation and its
# sub-agents (thinking tokens included): the single default for both; the
# console-managed ``llm_max_response_tokens`` is the only override. Observed
# calls pool around 1.4k tokens with p90 ≈ 4k, and the earlier 12k cap still cut
# 1-3 rounds per session; a truncated thinking block loses the whole answer, so
# the ceiling is the practical per-round maximum the window leaves next to
# the compaction threshold. A round that spends it all on reasoning costs
# ~15 min at 25-40 tok/s — the accepted price of not cutting a long answer.
# compact and analysis keep their own budgets.
AGENT_MAX_OUTPUT_TOKENS = 32_768


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


# An arm's ``llm_key``: which relay credential its calls start on
# (``lighting.LLM_KEY_ENVS``). Every other provider has one key, so only
# ``primary`` applies to it.
LLM_KEYS = tuple(_lighting.LLM_KEY_ENVS)


def _api_key(key_env: str, model: str, env_file: str | Path, *, required: bool) -> str:
    """Resolve a credential from a name a trusted profile fixes."""

    key = load_env_value(key_env, env_file)
    if key or not required:
        return key
    raise ValueError(
        f"model {model} requires an API key: set {key_env} in the "
        f"environment or {env_file}"
    )


def gateway_env_names() -> tuple[str, ...]:
    """Every environment key a role gateway reads, each once: the profiles'
    credentials and endpoints and the relay's alternate key."""

    names = [_lighting.API_KEY_ALT_ENV]
    for profile in _PROFILES:
        names += [profile.api_key_env, profile.base_url_env or ""]
    return tuple(dict.fromkeys(name for name in names if name))


def model_profile(model: str) -> ModelProfile:
    if model in _deepseek.MODEL_CHOICES:
        return _DEEPSEEK_PROFILE
    if model == LOCAL_QWEN_MODEL:
        return _VLLM_PROFILE
    if model in _MIMO_PROFILES:
        return _MIMO_PROFILES[model]
    if model in _LIGHTING_PROFILES:
        return _LIGHTING_PROFILES[model]
    raise ValueError(
        f"unsupported model {model!r}; the catalog has {', '.join(MODEL_CHOICES)}"
    )


def token_prices(provider: str, model: str) -> TokenPrices:
    """The list prices one role gateway's calls are charged at.

    Keyed on the gateway's configured catalog model, never on the model id a
    response reports. The injected test double is free; a gateway outside
    the catalog has no price and is refused, so no model call runs unmetered.
    """

    if provider == ScriptedLLM.provider:
        return FREE
    return model_profile(model).prices


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
    timeout_seconds: float = 600.0,
    max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
    max_tokens: int = 1_200,
    temperature: float = 0.0,
    thinking_enabled: bool = False,
    reasoning_effort: str | None = None,
    conversation_log_dir: str | Path | None = "data/llm_conversations",
    require_credentials: bool = True,
    llm_key: str = "primary",
) -> LLMProxy:
    """Build one role gateway from a catalog model name.

    Endpoint and credential environment keys are fixed by the trusted model
    profile for every provider; neither can be supplied by experiment
    parameters, so no configuration can point the Authorization header at a
    secret the profile did not name. ``llm_key`` only chooses which of the
    relay's two fixed keys a relay gateway starts on; any other provider
    refuses ``alternate``, having one key.
    """

    profile = model_profile(model)
    if llm_key not in LLM_KEYS:
        raise ValueError(f"llm_key must be one of {', '.join(LLM_KEYS)}, got {llm_key!r}")
    if llm_key != "primary" and profile.provider != "lighting":
        raise ValueError(
            f"llm_key {llm_key} chooses the relay's key, and model {model} is "
            f"served by {profile.provider}, which has one"
        )
    effective_max_tokens = effective_max_output_tokens(model, max_tokens)
    key_env = (
        _lighting.LLM_KEY_ENVS[llm_key]
        if profile.provider == "lighting"
        else profile.api_key_env
    )
    key = _api_key(key_env, model, env_file, required=require_credentials) or "preflight"
    base_url = (
        load_env_value(profile.base_url_env, env_file)
        if profile.base_url_env is not None
        else ""
    ) or profile.default_base_url
    if profile.provider == "lighting":
        return _lighting.build_gateway(
            profile=profile,
            model=model,
            api_key=key,
            base_url=base_url,
            env_file=env_file,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            conversation_log_dir=conversation_log_dir,
            llm_key=llm_key,
        )
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
                    max_output_tokens=profile.max_output_tokens,
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
                # MiMo's thinking is on/off only: no effort level to send.
                reasoning_effort=(
                    _qwen_reasoning_effort(reasoning_effort)
                    if thinking_enabled and profile.request_dialect == "vllm-qwen"
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
    "DEFAULT_LLM_MAX_RETRIES",
    "DEFAULT_LLM_RETRY_BACKOFF_SECONDS",
    "LLM_KEYS",
    "LOCAL_QWEN_MODEL",
    "MIMO_FLASH_MODEL",
    "MIMO_MODELS",
    "MIMO_PRO_MODEL",
    "MIN_CONTEXT_WINDOW_TOKENS",
    "MODEL_CHOICES",
    "ModelProfile",
    "build_model_gateway",
    "effective_max_output_tokens",
    "gateway_env_names",
    "model_profile",
    "token_prices",
]
