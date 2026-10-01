"""Agent-visible projections of host-computed blocks.

No forward or Held-out evidence passes through these projections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# The null-control fields a tool result carries. ``rejects_mean`` rides along because a null whose orders are mostly rejected
# is a weaker comparison, ``status``/``reason`` because a failed or unavailable
# null (a result with no filled trade) must not read as a missing one, and the
# null's own centre and spread over its ``k`` draws because a percentile alone
# does not say how far the observed excess sits from them. Informational: the
# pipeline gates on the active series against the panel each validation draws,
# never on this percentile.
NULL_CONTROL_KEYS = (
    "status",
    "reason",
    "observed_excess",
    "excess_percentile",
    "null_excess_mean",
    "null_excess_p05",
    "null_excess_p95",
    "k",
    "rejects_mean",
    "dropped_trips_mean",
    "step",
)


def allowed_keys(block: object, keys: Sequence[str]) -> dict[str, object] | None:
    """Whitelisted projection of one host-computed block; None when absent."""

    if not isinstance(block, Mapping):
        return None
    return {key: block.get(key) for key in keys if key in block}
