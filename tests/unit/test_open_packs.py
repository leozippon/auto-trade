"""The reference packs a research session mounts.

What must hold. A pack mounts into a research session, which never sees Forward
or Held-out results, so no file a pack's authors wrote quotes a reading after
the research period. A paper pack (one with a payload manifest) names where
each paper came from -- a repository at an exact commit, a licence -- and no
paper whose financial evaluation data runs past the data a session may see.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.pack_payload import PAYLOAD_MANIFEST, read_sources

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS = sorted(path for path in (REPO_ROOT / "configs" / "workspace_refs").iterdir() if path.is_dir())
PAPER_PACKS = [pack for pack in PACKS if (pack / PAYLOAD_MANIFEST).is_file()]
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


@pytest.mark.parametrize("pack", PAPER_PACKS, ids=lambda path: path.name)
def test_a_paper_pack_names_its_sources_and_no_data_after_the_research_period(pack: Path) -> None:
    """read_sources refuses an entry without its fields, a 40-hex commit for a
    named repository or a licence; the research end is the console's default
    geometry, which every round sends unless it names another."""
    research_end = date.fromisoformat(DEFAULT_RESEARCH_GEOMETRY.research_end)
    for entry in read_sources(pack):
        if entry["data_end"] is not None:
            assert date.fromisoformat(str(entry["data_end"])) <= research_end, (pack.name, entry["slug"])
