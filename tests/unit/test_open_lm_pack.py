"""The 20261025 open pack: the owner's question and the facts a session cannot discover, and no strategy.

What must hold. The pack is its README and the record of closed families,
nothing a session would start from or be bound by: no starter, so the
session's `output/` begins as the contract template, and no rules file. No
pack file cites a reading after the research period.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK = REPO_ROOT / "configs" / "workspace_refs" / "open_lm_100k_8y_20261025"
# How a reading after the research period is quoted; the other pack tests
# refuse the same words.
AFTER_RESEARCH = ("前推", "Held-out", "held-out", "forward IR", "F2 下界", "样本外读数")


def test_the_pack_is_a_question_and_its_facts_only() -> None:
    assert sorted(path.name for path in PACK.iterdir()) == ["README.md", "families.md"]


def test_no_pack_file_cites_a_reading_after_the_research_period() -> None:
    for path in sorted(PACK.rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for word in AFTER_RESEARCH:
                assert word not in text, (path.name, word)
