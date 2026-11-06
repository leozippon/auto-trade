"""The required acceptance rules a test experiment states.

``config.AcceptanceRules`` gives its required rules
(``config.REQUIRED_ACCEPTANCE_KEYS``) no default, so every test that builds
rules, an experiment's configuration or a session request names them; this is the
one place their test values live.
"""

from __future__ import annotations

from autotrade.pipelines.config import AcceptanceRules

# One research year and one validation year on the two-year fixture geometries.
REQUIRED_RULES: dict[str, object] = {"alpha": 0.05, "research_years": 1}
RULES = AcceptanceRules(**REQUIRED_RULES)
# What a console create request must state beside the geometry: the rules the
# console has no value for (``hitl_state.WEB_REQUIRED_PARAMS``).
CREATE_RULES: dict[str, object] = {key: REQUIRED_RULES[key] for key in ("research_years",)}
