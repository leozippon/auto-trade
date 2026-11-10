"""Provider conversation-log records.

Security-sensitive, pure helpers shared by OpenAI-compatible clients: build one
audit record per provider attempt (payload stored once per logical call — see
the shared proxy retry loop), every logged object redacted by the same
``sanitize_for_log`` that cleans traces. No transport logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from autotrade.environment.runtime import sanitize_for_log

if TYPE_CHECKING:  # annotation-only: importing at runtime would be a cycle
    from autotrade.environment.llm.proxy import LLMProxyError


class ConversationLogConfig(Protocol):
    provider: str
    model: str

    def safe_metadata(self) -> dict[str, Any]: ...



def ensure_log_parent(path: Path) -> None:
    # OSError propagates; the client's _write_conversation_log boundary wraps it
    # into the provider error type (importing it here would be a cycle).
    path.parent.mkdir(parents=True, exist_ok=True)


def conversation_log_record(
    *,
    config: ConversationLogConfig,
    payload: dict[str, Any],
    started_at: str,
    completed_at: str | None,
    duration_seconds: float,
    attempt: int,
    max_attempts: int,
    status: str,
    call_id: str = "",
    include_payload: bool = False,
    http_status_code: int | None = None,
    raw_response: dict[str, Any] | None = None,
    response_body: str | None = None,
    error: LLMProxyError | None = None,
) -> dict[str, Any]:
    """One log line. The full (redacted) payload is stored once per logical
    call — the first attempt's "started" record; every other record joins it
    via ``call_id`` instead of duplicating the history."""
    safe_payload = sanitize_for_log(payload)
    safe_response = sanitize_for_log(raw_response) if raw_response is not None else None
    safe_body = sanitize_for_log(response_body) if response_body is not None else None
    record: dict[str, Any] = {
        "schema_version": 2,
        "provider": config.provider,
        "model": config.model,
        "call_id": call_id,
        "request_started_at": started_at,
        "request_completed_at": completed_at,
        "duration_seconds": round(max(duration_seconds, 0.0), 6),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "status": status,
        "http_status_code": http_status_code,
        "metadata": config.safe_metadata(),
    }
    if include_payload:
        record["payload"] = safe_payload
    if safe_response is not None:
        record["raw_response"] = safe_response
        record["response_id"] = sanitize_for_log(str(raw_response.get("id", "")))
        record["usage"] = sanitize_for_log(dict(raw_response.get("usage") or {}))
    if safe_body is not None:
        record["response_body"] = safe_body
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "message": sanitize_for_log(str(error)),
            "status_code": error.status_code,
            "retryable": error.retryable,
        }
    return record
