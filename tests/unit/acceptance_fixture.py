"""The required acceptance rules a test arm states.

``config.AcceptanceRules`` gives its required rules
(``config.REQUIRED_ACCEPTANCE_KEYS``) no default, so every test that builds
rules, an arm's configuration or a session request names them; this is the
one place their test values live.
"""

from __future__ import annotations

from autotrade.pipelines.config import AcceptanceRules

REQUIRED_RULES: dict[str, object] = {"alpha": 0.05}
RULES = AcceptanceRules(**REQUIRED_RULES)
