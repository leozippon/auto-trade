"""The required acceptance rules a test arm states.

``config.AcceptanceRules`` gives its required rules
(``config.REQUIRED_ACCEPTANCE_KEYS``) no default, so every test that builds
rules, an arm's configuration or a session request names them; this is the
one place their test values live.
"""

from __future__ import annotations

from autotrade.pipelines.config import AcceptanceRules

REQUIRED_RULES: dict[str, object] = {"alpha": 0.05, "sealed_years": 1}
RULES = AcceptanceRules(**REQUIRED_RULES)
# The sealed research year a test session's request names after its visible
# years, which every fixture ends on 2025-06-30: dates the session is told,
# never a span it replays.
SEALED_PERIOD = ("20250701", "20260630")
# What a console create request must state beside the geometry: the rules the
# console has no value for (``hitl_state.WEB_REQUIRED_PARAMS``).
CREATE_RULES: dict[str, object] = {"sealed_years": REQUIRED_RULES["sealed_years"]}
