"""The shared log-redaction primitives behind every trace and tool result.

``sanitize_for_log`` runs over every tool result the Runner records and over
the text the Agent reads back from it, and ``redact_host_paths`` runs over
every rendered transcript block, every error summary and every tool traceback.
Both have the same two conditions to hold at once: what must never reach a
model must not survive, and ordinary content must come through untouched. The
second condition is not cosmetic — an over-eager key pattern rewrote file,
skill and note names in the observations the Agent acts on, and an over-eager
path pattern rewrote division, backquoted names and globs in the transcript the
Agent reads its own work back from. Both are silent corruption of its workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrade.environment.runtime import (
    HOST_PATH_PREFIXES,
    redact_host_paths,
    sanitize_for_log,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize("prefix", HOST_PATH_PREFIXES)
def test_an_absolute_path_under_each_known_host_root_is_redacted(prefix: str) -> None:
    text = f"FileNotFoundError: {prefix}/lzp/ADMCubeQuant/experiments/arm/result.json"
    assert redact_host_paths(text) == "FileNotFoundError: [host_path]"


def test_this_checkout_is_a_known_host_root_and_the_sandbox_mounts_are_not() -> None:
    """The roots are derived from where the host really keeps its files. The
    container's own mounts are the Agent's view of its sandbox, so none of them
    may become a prefix — redacting those is what corrupts the transcript."""

    assert f"/{REPO_ROOT.parts[1]}" in HOST_PATH_PREFIXES
    assert not {"/mnt", "/tmp", "/opt", "/usr"}.intersection(HOST_PATH_PREFIXES)
    traceback = f'  File "{REPO_ROOT}/src/autotrade/environment/runtime.py", line 12'
    assert redact_host_paths(traceback) == '  File "[host_path]", line 12'


@pytest.mark.parametrize(
    "text",
    [
        # Division, which is what the Agent's own factor code is full of.
        "score = (2.0 * C - H - L) / O",
        "x / y",
        "ocf_ta = n_cashflow_act / total_assets",
        # Relative paths and string fragments: never a host path.
        '"a/b"',
        "refs/starter/lib/*.py",
        "candidates/*/lib/*.py",
        "daily/part_0000.parquet",
        'np.save(context.state_dir + "/ranker.npz", w)',
        'context.asof_dir + "/"',
        "shell ... 2>/dev/null",
        # A root on its own is not a path; it needs a second segment.
        "/home",
        "/Data2",
        # URLs and dates are not paths either.
        "https://example.com/home/lzp/x",
        "https://arxiv.org/abs/2312.15235",
        "2026/09/16",
        # The sandbox mounts the Agent must keep reading.
        "/mnt/agent/workspace/output/main.py",
        "/mnt/snapshot/manifest.json",
        "/tmp/cache/mpl",
        "/opt/autotrade",
    ],
)
def test_ordinary_content_that_merely_contains_a_slash_is_left_alone(text: str) -> None:
    assert redact_host_paths(text) == text



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
