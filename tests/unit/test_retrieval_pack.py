"""The 20261025b retrieval pack: the owner's direction and the facts a session cannot discover, and no strategy.

What must hold. The pack is its README and the record of closed families,
nothing a session would start from: no starter, so the session's `output/`
begins as the contract template, and no rules file. The round's directive is
the README's direction paragraph word for word, so the direction reaches the
Agent in one wording, and nothing before the facts names a model, so neither
copy steers the session to one; round 20261026's directive starts with the
same paragraph and adds only its call to explore, which names no model
either, and its astra arm on the A100 instance sends what the local astra arm
sent but for one card instead of two. Rounds 20261027 and 20261027b send
round 20261026's directive plus one sentence pointing at the record of closed
families, word for word alike and naming no model, on the instance's card at
six replays per batch. No pack file cites a reading after the research period.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.llm import MODEL_CHOICES
from scripts.experiments import (
    create_round_20261026,
    create_round_20261027,
    create_round_20261027b,
)
from scripts.experiments.create_round_20261025b import DIRECTIVE, PACK

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / PACK
# How a reading after the research period is quoted; the other pack tests
# refuse the same words.
AFTER_RESEARCH = ("前推", "Held-out", "held-out", "forward IR", "F2 下界", "样本外读数")


def test_the_pack_is_a_direction_and_its_facts_only() -> None:
    assert sorted(path.name for path in PACK_DIR.iterdir()) == ["README.md", "families.md"]


def _head_and_direction() -> tuple[str, str]:
    """The README before its facts, and the paragraph under 「方向」."""
    readme = (PACK_DIR / "README.md").read_text(encoding="utf-8")
    head = readme.split("\n## 卡\n", 1)[0]
    return head, head.split("\n## 方向\n\n", 1)[1].split("\n\n", 1)[0]


def test_the_directive_is_the_packs_direction_and_names_no_model() -> None:
    head, direction = _head_and_direction()
    assert direction == DIRECTIVE
    assert "Qwen" not in head


def test_round_20261026_starts_from_the_direction_and_names_no_model() -> None:
    _, direction = _head_and_direction()
    directive = create_round_20261026.DIRECTIVE
    assert create_round_20261026.PACK == PACK
    assert directive.startswith(direction) and len(directive) > len(direction)
    for name in (*MODEL_CHOICES, "Qwen", "gemini", "gpt", "claude", "DeepSeek"):
        assert name.lower() not in directive.lower(), name
    arms = create_round_20261026.ARMS.values()
    assert len(arms) == 4
    for arm in arms:
        assert (arm["workspace_reference"], arm["research_directive"], arm["llm_key"]) == (
            PACK,
            directive,
            "alternate",
        )


def test_round_20261026_moves_astra_to_the_instance_unchanged_but_for_one_card() -> None:
    round_ = create_round_20261026.ROUND
    local, moved = "retrieval_100k_8y_astra_20261026", "retrieval_100k_8y_astra_zgca_20261026"
    assert round_.request_params(local)["gpu_count"] == 2
    assert round_.request_params(moved) == {
        **round_.request_params(local),
        "gpu_count": 1,
        "experiment_id": moved,
    }


def test_rounds_20261027_add_the_closed_families_and_name_no_model() -> None:
    directive = create_round_20261027.DIRECTIVE
    assert create_round_20261027b.DIRECTIVE == directive
    extra = directive.removeprefix(create_round_20261026.DIRECTIVE)
    assert extra != directive and "`refs/families.md`" in extra
    for name in (*MODEL_CHOICES, "Qwen", "gemini", "gpt", "claude", "DeepSeek"):
        assert name.lower() not in directive.lower(), name
    for module in (create_round_20261027, create_round_20261027b):
        for arm in module.ARMS:
            params = module.ROUND.request_params(arm)
            assert (
                params["workspace_reference"],
                params["research_directive"],
                params["llm_key"],
                params["gpu_count"],
                params["batch_validate_max_concurrency"],
            ) == (PACK, directive, "primary", 1, 6)


def test_no_pack_file_cites_a_reading_after_the_research_period() -> None:
    for path in sorted(PACK_DIR.rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for word in AFTER_RESEARCH:
                assert word not in text, (path.name, word)
