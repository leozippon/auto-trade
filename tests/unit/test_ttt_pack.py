"""The 20261028 test-time-training pack and its round: the owner's direction, the facts a session cannot discover, and the papers and code it reads offline.

What must hold. The pack is its README, the record of closed families and two
resource trees, papers and code, each with an index that names exactly what
the tree holds; there is no starter, so the session's `output/` begins as the
contract template. Every vendored repository carries its licence and only
text. The round's directive starts with the README's direction paragraph word
for word and adds the call to explore and the pointer to the closed families;
nothing before the README's facts names a model, and neither does the
directive. Both arms are round 20261027's arm on this pack, with their own main
model, the relay's alternate key, four replays per batch, three times every
per-arm budget and a strategy container of twice the default CPUs and memory. The README ends
with guidance the session may ignore. No file the pack's authors wrote cites a
reading after the research period; the vendored papers and code are other
people's text, left as published.
"""

from __future__ import annotations

import re
from pathlib import Path

from autotrade.agent.prompts import PROTOCOL_INSTRUCTION
from autotrade.environment.llm import MODEL_CHOICES
from scripts.experiments import create_round_20261027
from scripts.experiments.create_round_20261028 import ARMS, DIRECTIVE, PACK, ROUND
from tests.unit.test_retrieval_pack import AFTER_RESEARCH

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / PACK
MODEL_NAMES = (*MODEL_CHOICES, "Qwen", "gemini", "gpt", "claude", "DeepSeek")
# The files the pack's authors wrote, which the retrieval pack's guard against
# quoting a reading after the research period covers; the rest of papers/ and
# code/ is vendored text, where "held-out" is ordinary vocabulary.
AUTHORED = ("README.md", "families.md", "papers/INDEX.md", "code/INDEX.md")


def _head_and_direction() -> tuple[str, str]:
    """The README before its facts, and the paragraph under 「方向」."""
    readme = (PACK_DIR / "README.md").read_text(encoding="utf-8")
    head = readme.split("\n## 卡\n", 1)[0]
    return head, head.split("\n## 方向\n\n", 1)[1].split("\n\n", 1)[0]


def test_the_pack_is_a_direction_its_facts_and_two_indexed_resource_trees() -> None:
    assert sorted(path.name for path in PACK_DIR.iterdir()) == [
        "README.md",
        "code",
        "families.md",
        "papers",
    ]
    papers = PACK_DIR / "papers"
    index = (papers / "INDEX.md").read_text(encoding="utf-8")
    held = sorted(path.name for path in papers.iterdir() if path.name != "INDEX.md")
    assert held == sorted(set(re.findall(r"`(\d{4}_[a-z0-9_]+\.md)`", index)))
    code = PACK_DIR / "code"
    index = (code / "INDEX.md").read_text(encoding="utf-8")
    repositories = sorted(path.name for path in code.iterdir() if path.is_dir())
    assert repositories == sorted(re.findall(r"^\| `([^`]+)` \|", index, re.MULTILINE))
    for name in repositories:
        files = [path for path in (code / name).rglob("*") if path.is_file()]
        assert any(path.name.upper().startswith("LICENSE") for path in files), name
        # Text only: no binary, data dump or checkpoint.
        for path in files:
            path.read_text(encoding="utf-8")


def test_the_directive_starts_with_the_packs_direction_and_names_no_model() -> None:
    head, direction = _head_and_direction()
    assert DIRECTIVE.startswith(direction)
    assert "`refs/families.md`" in DIRECTIVE.removeprefix(direction)
    for name in MODEL_NAMES:
        assert name.lower() not in head.lower(), name
        assert name.lower() not in DIRECTIVE.lower(), name
    for word in AFTER_RESEARCH:
        assert word not in DIRECTIVE, word


# The per-arm budgets the owner's 2026-10-27 directive tripled for this round.
TRIPLED = (
    "max_replay_years",
    "max_null_controls",
    "max_api_spend_usd",
    "max_llm_calls",
    "max_research_minutes",
    "strategy_fit_timeout_seconds",
    "strategy_inference_timeout_seconds",
)


def test_both_arms_are_round_20261027s_arm_on_this_pack_and_the_alternate_key() -> None:
    """Their own model, this pack, the alternate key, four legs per batch,
    three times every per-arm budget and a container of 16 CPUs and 64 GiB;
    nothing else differs."""
    (reference,) = create_round_20261027.ARMS
    base = create_round_20261027.ROUND.request_params(reference)
    tripled = {key: 3 * base[key] for key in TRIPLED}
    assert tripled == {
        "max_replay_years": 288,
        "max_null_controls": 36,
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 7_200,
        "strategy_fit_timeout_seconds": 10_800,
        "strategy_inference_timeout_seconds": 1_080,
    }
    assert (base["strategy_cpus"], base["strategy_memory"]) == (8.0, "32g")
    models = {
        "ttt_100k_8y_sol_zgca_20261028": "gpt-6.1-sol",
        "ttt_100k_8y_argon_zgca_20261028": "gemini-4-argon",
    }
    assert list(ARMS) == list(models)
    for arm, model in models.items():
        assert ROUND.request_params(arm) == {
            **base,
            "experiment_id": arm,
            "model": model,
            "llm_key": "alternate",
            "workspace_reference": PACK,
            "research_directive": DIRECTIVE,
            "batch_validate_max_concurrency": 4,
            **tripled,
            "strategy_cpus": 16,
            "strategy_memory": "64g",
        }


def test_the_readme_ends_with_guidance_the_session_may_ignore() -> None:
    readme = (PACK_DIR / "README.md").read_text(encoding="utf-8")
    _, guidance = readme.split("\n## 建议（可以不采纳）\n", 1)
    assert "\n## " not in guidance
    assert "不是要求" in guidance and "不采纳不影响任何评判" in guidance
    assert "种子复制的个数在第一次复制之前定下" in guidance
    # What left the system prompt lives here once.
    for advice in ("向量化", "`smoke_backtest` 的 `start`", "`vllm serve`", "`run_null_control`"):
        assert advice in guidance, advice
        assert advice not in PROTOCOL_INSTRUCTION, advice


def test_no_authored_pack_file_cites_a_reading_after_the_research_period() -> None:
    for name in AUTHORED:
        text = (PACK_DIR / name).read_text(encoding="utf-8")
        for word in AFTER_RESEARCH:
            assert word not in text, (name, word)
