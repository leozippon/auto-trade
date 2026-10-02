"""DeepSeek adapter over the shared OpenAI-compatible gateway.

One of the hosted providers in the catalog (``model_profiles.py``). Everything
but the model catalog and the two constructors below is provider-neutral and
lives in ``openai_compatible.py``.
"""

from __future__ import annotations

from pathlib import Path

from .openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProxy

# The one model DeepSeek serves under its own name (DeepSeek-V4.1-Flash). The
# legacy names deepseek-v4-flash, deepseek-chat and deepseek-reasoner, and
# deepseek-v4-pro since 2026-09-14, are routed to it server-side, so an arm
# naming any of them would run on Flash under another label; they are refused.
MODEL_CHOICES = ("deepseek-flash",)


class DeepSeekConfig(OpenAICompatibleConfig):
    """Compatibility facade retaining the original DeepSeek constructor."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
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
        max_output_tokens: int | None = None,
    ) -> None:
        if not str(base_url).startswith("https://"):
            raise ValueError("base_url must use https")
        if model not in MODEL_CHOICES:
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
            max_output_tokens=max_output_tokens,
        )


class DeepSeekProxy(OpenAICompatibleProxy):
    """Compatibility facade over the shared OpenAI-compatible proxy."""
