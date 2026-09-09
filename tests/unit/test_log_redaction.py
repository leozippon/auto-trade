"""The shared log-redaction primitive behind every trace and tool result.

``sanitize_for_log`` runs over every tool result the Runner records and over
the text the Agent reads back from it, so it has two conditions to hold at
once: a credential must never survive it, and ordinary content must come
through untouched. The second condition is not cosmetic — an over-eager
pattern rewrote file, skill and note names in the observations the Agent
acts on, which is a silent corruption of its own workspace.
"""

from __future__ import annotations

import json

import pytest

from autotrade.environment.runtime import sanitize_for_log


@pytest.mark.parametrize(
    "text",
    [
        "risk-controls",
        "notes/risk-controls/plan.md",
        "task-20260909-abc",
        "disk-usage",
        "subagent-task-contract-2026",
        "sk-short",
        # The same left boundary on the Hugging Face pattern: a word that
        # merely contains "hf_" is not a token prefix.
        "shf_abcdefgh",
        "notes/shf_abcdefgh/plan.md",
    ],
)
def test_ordinary_words_that_merely_contain_a_key_prefix_are_left_alone(
    text: str,
) -> None:
    assert sanitize_for_log(text) == text


def test_a_real_key_or_bearer_token_is_still_redacted() -> None:
    key = "sk-proj-" + "a1B2c3D4e5F6g7H8"
    assert key not in str(sanitize_for_log(f"export OPENAI_API_KEY={key}"))
    assert "sk-[redacted]" in str(sanitize_for_log(f"export OPENAI_API_KEY={key}"))
    assert "should-not-appear" not in str(
        sanitize_for_log("Authorization: Bearer should-not-appear")
    )
    assert "hunter2" not in str(sanitize_for_log("https://user:hunter2@example.com/x"))
    token = "hf_" + "AbCdEfGh12345678"
    assert token not in str(sanitize_for_log(f"HF_TOKEN={token}"))
    assert "hf_[redacted]" in str(sanitize_for_log(f"HF_TOKEN={token}"))


def test_a_key_embedded_in_a_json_tool_result_is_still_redacted() -> None:
    key = "sk-" + "0123456789abcdefgh"
    payload = {
        "ok": True,
        "stdout": json.dumps({"config": {"api_key": key}}),
        "skill": "subagent-task-contract",
    }
    sanitized = sanitize_for_log(payload)
    assert key not in json.dumps(sanitized, ensure_ascii=False)
    assert "sk-[redacted]" in sanitized["stdout"]
    # The same result carries an ordinary name through unchanged.
    assert sanitized["skill"] == "subagent-task-contract"


def test_a_sensitive_key_hides_even_a_short_value() -> None:
    assert sanitize_for_log({"api_key": "sk-short", "note": "risk-controls"}) == {
        "api_key": "[redacted]",
        "note": "risk-controls",
    }
