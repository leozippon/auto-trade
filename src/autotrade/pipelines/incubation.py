"""The two screens of incubating one node of a finished arm.

Incubation defers exactly one question to Paper: whether the node is merely
the luckiest of the arm's many tries. The entry screen therefore lets through
only a freeze-gate reading whose failures are that question -- the deflated
Sharpe, and the seed mean when its own active IR still reaches the arm's
undeflated floor -- and the forward screen refuses a book whose forward and
Held-out replay already shows it should not trade. Neither changes the freeze
gate, the graduation verdict or what "graduated" means
(``scripts/experiments/incubate.py``, ``ledger.incubation_candidate``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from .verdict import CONDITIONS, NOT_A_SESSION_STEP, UNMEASURABLE


def _reasons(stages: Iterable[str]) -> frozenset[str]:
    return frozenset(condition.reason for condition in CONDITIONS if condition.stage in stages)


# Every reason a freeze-gate reading (``experiment.freeze_gate_for``) carries:
# its conditions' and the two it records without reading the nominee.
GATE_REASONS = _reasons(("nomination", "registration", "freeze", "seeds")) | {
    NOT_A_SESSION_STEP,
    UNMEASURABLE,
}
_VERDICT_REASONS = _reasons(("replay", "forward", "heldout"))


def _known(reasons: Iterable[str], among: frozenset[str]) -> tuple[str, ...]:
    """``reasons`` as given, once each is checked to be a code of ``among``:
    a renamed condition fails here, at import, rather than never matching."""

    reasons = tuple(reasons)
    unknown = [reason for reason in reasons if reason not in among]
    if unknown:
        raise ValueError(f"no such reason code: {unknown}")
    return reasons


# What the entry screen defers to Paper. The seed mean only while its active
# IR reaches the arm's ``min_active_ir``: its bar is the deflated one, so a
# trained-model book could otherwise never be incubated.
DEFLATED_SHARPE, SEED_MEAN = _known(
    (
        "freeze_deflated_sharpe_below_threshold",
        "freeze_seed_mean_information_ratio_below_threshold",
    ),
    GATE_REASONS,
)
DEFERRABLE = frozenset({DEFLATED_SHARPE, SEED_MEAN})
# Every other gate reason blocks. Listed rather than derived, so a new gate
# condition is classified on purpose (the tests hold the two sets to
# ``GATE_REASONS``); an unlisted one blocks all the same.
BLOCKING = frozenset(
    _known(
        (
            "non_finite_total_return",
            "non_finite_max_drawdown",
            "non_finite_sharpe",
            "max_drawdown_above_limit",
            "freeze_needs_full_span_validation",
            "freeze_nominee_is_control",
            "freeze_too_few_full_span_validations",
            "freeze_information_ratio_below_threshold",
            "freeze_deflated_sharpe_unavailable",
            "freeze_too_few_positive_years",
            "freeze_active_drawdown_exceeded",
            "freeze_raw_excess_not_positive_at_cost_stress",
            "freeze_tracking_error_above_cap",
            "freeze_beta_outside_band",
            "freeze_seed_replicate_invalid",
            "freeze_too_few_seed_replicates",
            UNMEASURABLE,
            NOT_A_SESSION_STEP,
        ),
        GATE_REASONS,
    )
)

# The verdict reasons the forward screen blocks on, recorded with this prefix:
# the strategy raised (F1/H1), or a drawdown limit broke (F4/H3).
PREFIX = "incubation_"
FORWARD_BLOCKING = _known(
    (
        "forward_strategy_error",
        "heldout_strategy_error",
        "forward_max_drawdown_exceeded",
        "forward_active_drawdown_exceeded",
        "heldout_max_drawdown_exceeded",
        "heldout_active_drawdown_exceeded",
    ),
    _VERDICT_REASONS,
)
NEUTRALIZED_EXCESS_NOT_POSITIVE = f"{PREFIX}forward_neutralized_excess_not_positive"
PLAIN_EXCESS_NOT_POSITIVE = f"{PREFIX}forward_plain_excess_not_positive"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def entry_screen(gate: Mapping[str, object]) -> dict[str, object]:
    """Pass iff every reason ``gate`` carries is deferrable. A gate with no
    reason passes too: a node that passed the whole gate but that the Agent
    did not nominate (the operator's reason then says why it is incubated).
    ``deferred`` and ``blocking`` split the gate's reasons in its order."""

    seeds = gate.get("seed_replicates")
    mean = _number(seeds.get("mean_information_ratio")) if isinstance(seeds, Mapping) else None
    thresholds = gate.get("thresholds")
    floor = _number(thresholds.get("min_information_ratio")) if isinstance(thresholds, Mapping) else None
    seed_mean_deferrable = mean is not None and floor is not None and mean >= floor
    reasons = [str(reason) for reason in gate["reasons"]]  # type: ignore[attr-defined]
    deferred = [
        reason
        for reason in reasons
        if reason == DEFLATED_SHARPE or (reason == SEED_MEAN and seed_mean_deferrable)
    ]
    blocking = [reason for reason in reasons if reason not in deferred]
    return {"passed": not blocking, "deferred": deferred, "blocking": blocking}


def forward_screen(reading: Mapping[str, object]) -> dict[str, object]:
    """Pass unless the forward and Held-out reading (``experiment.replay_frozen``)
    shows a strategy error, a broken drawdown limit, or a forward slice whose
    neutralised or plain excess is not positive -- the seed mean's plain
    excess where the book has seed replicates. A cheap screen on a year the
    book was not selected on; the confirmation is Paper's."""

    verdict_reasons = set(reading["verdict"]["reasons"])  # type: ignore[index]
    reasons = [f"{PREFIX}{reason}" for reason in FORWARD_BLOCKING if reason in verdict_reasons]
    slices = reading.get("slices")
    if isinstance(slices, Mapping):
        forward: Mapping[str, object] = slices["forward"]
        if (_number(forward["neutralized_excess"]) or 0.0) <= 0:
            reasons.append(NEUTRALIZED_EXCESS_NOT_POSITIVE)
        seed_mean = forward.get("seed_mean")
        plain = seed_mean["plain_excess"] if isinstance(seed_mean, Mapping) else forward["plain_excess"]
        if (_number(plain) or 0.0) <= 0:
            reasons.append(PLAIN_EXCESS_NOT_POSITIVE)
    return {"passed": not reasons, "reasons": reasons}
