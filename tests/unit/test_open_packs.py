"""The reference packs a research session mounts.

What must hold. A pack mounts into a research session, which never sees Forward
or Held-out results, so no file a pack's authors wrote quotes a reading after
the research period.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS = sorted(path for path in (REPO_ROOT / "configs" / "workspace_refs").iterdir() if path.is_dir())
# How a reading after the research period is quoted.
AFTER_RESEARCH = ("前推", "Held-out", "held-out", "forward IR", "F2 下界", "样本外读数")


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_no_pack_cites_a_reading_after_the_research_period(pack: Path) -> None:
    assert (pack / "README.md").is_file()
    for path in pack.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".py"):
            text = path.read_text(encoding="utf-8")
            for word in AFTER_RESEARCH:
                assert word not in text, (pack.name, path.name, word)
