"""Source-version stamps used by long-lived services and experiment workers."""

from __future__ import annotations

import unittest
from pathlib import Path


class RepoGuidelineSyncTest(unittest.TestCase):
    def test_agents_md_matches_claude_md(self) -> None:
        # AGENTS.md and CLAUDE.md are the same guideline document maintained in
        # lockstep for two agent CLIs; they have drifted before and needed an
        # explicit reconcile commit — pin byte equality (same pattern as the
        # PROMPTS.md and UNITS_REFERENCE.md freshness regressions).
        root = Path(__file__).resolve().parents[2]
        agents = (root / "AGENTS.md").read_text(encoding="utf-8")
        claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
        self.assertEqual(agents, claude)


if __name__ == "__main__":
    unittest.main()
