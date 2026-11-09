"""The required acceptance rules a test experiment states, and the geometry
the synthetic releases are built around.

``config.AcceptanceRules`` gives its required rules
(``config.REQUIRED_ACCEPTANCE_KEYS``) no default, so every test that builds
rules, an experiment's configuration or a session request names them; this is the
one place their test values live.
"""

from __future__ import annotations

from autotrade.pipelines.calendar import ResearchGeometry
from autotrade.pipelines.config import AcceptanceRules

# Four July-June research years, the twelve forward months after them and a
# Held-out quarter: the span the synthetic releases cover, stated by a test
# that builds one rather than the console default (``config.DEFAULT_RESEARCH_GEOMETRY``).
TEST_GEOMETRY = ResearchGeometry(
    research_start="20210701",
    research_end="20250630",
    forward_end="20260630",
    heldout_end="20260930",
)

# One research year and one validation year on the two-year fixture geometries.
REQUIRED_RULES: dict[str, object] = {"alpha": 0.05, "research_years": 1}
RULES = AcceptanceRules(**REQUIRED_RULES)
# What a console create request must state beside the geometry: the rules the
# console has no value for (the required ``experiment_params.PARAMETERS``).
CREATE_RULES: dict[str, object] = {key: REQUIRED_RULES[key] for key in ("research_years",)}
# The prebuilt view cache a create request in a fixture repository names: none,
# since the repository holds no seed and a named one, the console's default
# included, must exist.
NO_SEED: dict[str, object] = {"pit_views_seed": ""}
