"""Verdict statistics: freeze gate, forward verdict, Held-out rule (PL1 §4)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.replay.style import window_neutralized_excess
from autotrade.pipelines import verdict

SCALE = math.sqrt(TRADING_DAYS_PER_YEAR)
FORWARD_START, FORWARD_END = "20250701", "20260630"
HELDOUT_START, HELDOUT_END = "20260701", "20260911"


def _weekdays(start: str, end: str) -> list[str]:
    return [day.strftime("%Y%m%d") for day in pd.bdate_range(start, end)]


FORWARD_DAYS = _weekdays(FORWARD_START, FORWARD_END)
HELDOUT_DAYS = _weekdays(HELDOUT_START, HELDOUT_END)


def _segment(days, alpha, rng, *, te=0.13, exact=False):
    """A book with annualised neutralised excess ``alpha`` on CSI 300 and size.

    ``exact`` projects the noise off the design, so the segment's own OLS
    intercept is ``alpha`` to floating precision.
    """

    n = len(days)
    benchmark = rng.normal(0.0, 0.20 / SCALE, n)
    size = rng.normal(0.0, 0.10 / SCALE, n)
    noise = rng.normal(0.0, te / SCALE, n)
    if exact:
        design = np.column_stack([np.ones(n), benchmark, size])
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    strategy = alpha / TRADING_DAYS_PER_YEAR + 0.8 * benchmark + 0.3 * size + noise
    return list(days), strategy, benchmark, size


def _analysis(*segments):
    rows = [
        (day, value, bench, size)
        for days, strategy, benchmark, factor in segments
        for day, value, bench, size in zip(days, strategy, benchmark, factor)
    ]
    return {
        "strategy_daily": [[day, float(value)] for day, value, _, _ in rows],
        "benchmark_daily": [[day, float(bench)] for day, _, bench, _ in rows],
        "size_factor_daily": [[day, float(size)] for day, _, _, size in rows],
    }


def _forward(analysis, **overrides):
    arguments = {
        "start": FORWARD_START,
        "end": FORWARD_END,
        "seed_key": "artifact-1",
        "max_drawdown": 1.0,
        "cost_stress_multiplier": 2.0,
        "slippage_bps": 10.0,
        "turnover": 0.0,
        "round_trips": 12,
        "mean_gross": 1.0,
    }
    arguments.update(overrides)
    return verdict.forward_slice(analysis, **arguments)


def _split(days, boundary="20260101"):
    return [day for day in days if day < boundary], [
        day for day in days if day >= boundary
    ]


def test_statistics_match_the_style_regression():
    rng = np.random.default_rng(11)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.07, rng))
    stats = verdict.neutralized_statistics(
        analysis, start=FORWARD_START, end=FORWARD_END
    )

    strategy = np.array([value for _, value in analysis["strategy_daily"]])
    design = np.column_stack(
        [
            np.ones(len(strategy)),
            [value for _, value in analysis["benchmark_daily"]],
            [value for _, value in analysis["size_factor_daily"]],
        ]
    )
    coefficients, residual_ss, _, _ = np.linalg.lstsq(design, strategy, rcond=None)
    rows = verdict._regression_rows(analysis, FORWARD_START, FORWARD_END)

    assert stats["neutralized_excess"] == window_neutralized_excess(analysis)
    # The vectorised refit the bootstrap uses is the style regression.
    assert (
        round(float(verdict._fit(rows)[0]) * TRADING_DAYS_PER_YEAR, 4)
        == stats["neutralized_excess"]
    )
    assert coefficients[0] * TRADING_DAYS_PER_YEAR == pytest.approx(
        stats["neutralized_excess"], abs=1e-4
    )
    expected_te = math.sqrt(residual_ss[0] / (len(strategy) - 3)) * SCALE
    assert stats["tracking_error"] == pytest.approx(expected_te, rel=1e-9)
    assert stats["information_ratio"] == pytest.approx(
        stats["neutralized_excess"] / expected_te
    )


def test_bootstrap_bound_is_fixed_by_the_artifact_id():
    rng = np.random.default_rng(12)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.05, rng))

    first = _forward(analysis, seed_key="artifact-1")["lower_bound"]
    assert _forward(analysis, seed_key="artifact-1")["lower_bound"] == first
    assert _forward(analysis, seed_key="artifact-2")["lower_bound"] != first
    assert first < window_neutralized_excess(analysis)


def test_forward_pass_rates_match_the_design_simulation():
    """PL1 §4.2 at TE 13 %: nulls pass ≈ 0.18, a steady 8 %/yr edge ≈ 0.38.

    F4–F6 are held passing so only the bootstrap bound and the recency check
    decide; the bounds are loose around the design figures, the seeds fixed.
    """

    def pass_rate(alpha, seed, draws=150):
        rng = np.random.default_rng(seed)
        passed = 0
        for index in range(draws):
            analysis = _analysis(_segment(FORWARD_DAYS, alpha, rng))
            passed += not _forward(analysis, seed_key=f"artifact-{index}")["reasons"]
        return passed / draws

    null = pass_rate(0.0, seed=21)
    edge = pass_rate(0.08, seed=22)
    assert 0.10 <= null <= 0.26
    assert 0.28 <= edge <= 0.50


@pytest.mark.parametrize(
    ("recent_alpha", "recency_passes"),
    [(-0.15, False), (-0.001, False), (0.0, True)],
)
def test_recency_rejects_an_edge_that_collapsed_in_the_last_six_months(
    recent_alpha, recency_passes
):
    rng = np.random.default_rng(31)
    early, late = _split(FORWARD_DAYS)
    analysis = _analysis(
        _segment(early, 0.40, rng, te=0.03),
        _segment(late, recent_alpha, rng, te=0.03, exact=True),
    )
    result = _forward(analysis)

    assert result["recency_start"] == "20260101"
    assert result["recency_neutralized_excess"] == pytest.approx(recent_alpha, abs=1e-4)
    assert result["lower_bound"] > 0
    assert result["reasons"] == ([] if recency_passes else ["forward_recency_negative"])


def test_each_forward_condition_fails_at_its_boundary():
    rng = np.random.default_rng(41)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.30, rng, te=0.05))
    base = _forward(analysis)
    assert base["reasons"] == []

    returns = np.array([value for _, value in analysis["strategy_daily"]])
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = float(np.max(1.0 - equity / peaks))
    assert base["max_drawdown"] == pytest.approx(drawdown)
    assert _forward(analysis, max_drawdown=base["max_drawdown"])["reasons"] == []
    assert _forward(analysis, max_drawdown=base["max_drawdown"] - 1e-9)["reasons"] == [
        "forward_max_drawdown_exceeded"
    ]

    years = base["days"] / TRADING_DAYS_PER_YEAR
    breakeven = base["neutralized_excess"] * years / (10.0 * 1e-4)
    below = _forward(analysis, turnover=breakeven * (1 - 1e-6))
    assert below["reasons"] == []
    assert below["excess_at_cost_stress"] == pytest.approx(0.0, abs=1e-6)
    assert _forward(analysis, turnover=breakeven * (1 + 1e-6))["reasons"] == [
        "forward_not_positive_at_cost_stress"
    ]
    # No stress beyond the modelled slippage when the multiplier is one.
    assert (
        _forward(analysis, turnover=breakeven * 2, cost_stress_multiplier=1.0)[
            "reasons"
        ]
        == []
    )

    assert _forward(analysis, round_trips=11)["reasons"] == [
        "forward_too_few_round_trips"
    ]
    assert base["thresholds"]["min_round_trips"] == 12
    assert _forward(analysis, mean_gross=0.5)["reasons"] == []
    assert _forward(analysis, mean_gross=0.4999)["reasons"] == [
        "forward_exposure_below_floor"
    ]

    null = _analysis(_segment(FORWARD_DAYS, 0.0, rng, exact=True))
    assert "forward_lower_bound_not_positive" in _forward(null)["reasons"]


def test_heldout_tolerates_noise_but_not_a_collapse():
    rng = np.random.default_rng(51)
    forward_te = 0.13
    tolerance = (
        -verdict.HELDOUT_TOLERANCE_Z
        * forward_te
        / math.sqrt(len(HELDOUT_DAYS) / TRADING_DAYS_PER_YEAR)
    )
    assert tolerance == pytest.approx(-0.357, abs=0.005)  # ≈ −37 %/yr over 53 days

    def heldout(alpha, **overrides):
        analysis = _analysis(
            _segment(FORWARD_DAYS, 0.10, rng),
            _segment(HELDOUT_DAYS, alpha, rng, te=0.05, exact=True),
        )
        arguments = {
            "start": HELDOUT_START,
            "end": HELDOUT_END,
            "forward_tracking_error": forward_te,
            "max_drawdown": 1.0,
            "mean_gross": 1.0,
        }
        arguments.update(overrides)
        return verdict.heldout_slice(analysis, **arguments)

    inside = heldout(tolerance + 0.0003)
    assert inside["days"] == len(HELDOUT_DAYS) == 53
    assert inside["tolerance"] == pytest.approx(tolerance)
    assert inside["reasons"] == []
    assert heldout(tolerance - 0.0003)["reasons"] == ["heldout_excess_below_tolerance"]
    assert heldout(0.0, max_drawdown=0.0)["reasons"] == [
        "heldout_max_drawdown_exceeded"
    ]
    assert heldout(0.0, mean_gross=0.49)["reasons"] == ["heldout_exposure_below_floor"]


def test_deflated_sharpe_reproduces_the_design_example():
    """PL1 §3.3: N = 20, IR s.d. 0.3 → SR* ≈ 0.57; a 4-year IR of 0.6 → ≈ 0.52."""

    returns = np.random.default_rng(61).normal(0.0, 0.01, 968)
    block = verdict.deflated_sharpe(
        observed_sharpe=0.6, trials=20, trial_sharpe_std=0.3, returns=returns
    )

    assert block["sharpe_star"] == pytest.approx(0.57, abs=0.005)
    assert block["deflated_sharpe_probability"] == pytest.approx(0.52, abs=0.01)
    assert verdict.forward_mde(0.13, TRADING_DAYS_PER_YEAR) == pytest.approx(0.2756)


def test_freeze_gate_needs_the_deflated_probability_its_threshold_names():
    """The threshold is deliberately not 0.5.

    At 0.5 the deflated Sharpe's ``√(T−1)/√(variance_term)`` factor cancels and
    the gate degenerates into ``IR > SR*`` — a point comparison the two arms
    that were actually frozen cleared at a research IR of 0.11–0.13, while the
    forward verdict they then failed needed about 0.9.
    """

    rng = np.random.default_rng(71)
    research = _weekdays("20210701", "20250630")
    # A research IR near 1.2: below roughly 0.7 no deflation at all clears the
    # threshold over four years, which is the point of raising it.
    analysis = _analysis(_segment(research, 0.16, rng, exact=True))
    information_ratio = verdict.neutralized_statistics(analysis)["information_ratio"]
    trials = 10
    gamma, normal = 0.5772156649015329, NormalDist()
    expected_max = (1 - gamma) * normal.inv_cdf(
        1 - 1 / trials
    ) + gamma * normal.inv_cdf(1 - 1 / (trials * math.e))

    def gate(sharpe_star, irs=None):
        if irs is None:
            spread = sharpe_star / expected_max
            irs = [information_ratio, information_ratio - math.sqrt(2) * spread]
        return verdict.freeze_gate(analysis, trials=trials, full_span_irs=irs)

    # The SR* that puts the probability exactly at the threshold, solved from
    # the formula on this analysis' own skew, kurtosis and measured days.
    reading = gate(0.0)["deflated_sharpe"]
    sharpe = information_ratio / SCALE
    variance_term = (
        1.0
        - reading["return_skew"] * sharpe
        + (reading["return_kurtosis"] - 1.0) / 4.0 * sharpe**2
    )
    boundary = information_ratio - normal.inv_cdf(
        verdict.FREEZE_MIN_DSR_PROBABILITY
    ) * math.sqrt(variance_term) * SCALE / math.sqrt(reading["return_days"] - 1)

    passing = gate(0.98 * boundary)
    assert passing["passed"] and passing["reasons"] == []
    assert passing["information_ratio"] == pytest.approx(information_ratio)
    assert passing["deflated_sharpe"]["sharpe_star"] == pytest.approx(0.98 * boundary)
    assert (
        passing["deflated_sharpe"]["deflated_sharpe_probability"]
        > verdict.FREEZE_MIN_DSR_PROBABILITY
    )

    failing = gate(1.02 * boundary)
    assert not failing["passed"]
    assert failing["reasons"] == ["freeze_deflated_sharpe_below_threshold"]
    assert (
        failing["deflated_sharpe"]["deflated_sharpe_probability"]
        < verdict.FREEZE_MIN_DSR_PROBABILITY
    )

    # Merely beating the deflation is not enough any more: an IR a hair above
    # SR* still reads above 0.5 and the gate refuses it.
    barely = gate(0.98 * information_ratio)
    assert barely["deflated_sharpe"]["deflated_sharpe_probability"] > 0.5
    assert barely["reasons"] == ["freeze_deflated_sharpe_below_threshold"]

    alone = gate(0.0, irs=[information_ratio])
    assert alone["reasons"] == [
        "freeze_too_few_full_span_validations",
        "freeze_deflated_sharpe_unavailable",
    ]
    assert (
        alone["deflated_sharpe"]["unavailable_reason"]
        == "fewer_than_two_full_span_validations"
    )


def test_graduation_lists_every_failed_condition_and_a_strategy_error_discards():
    rng = np.random.default_rng(81)
    analysis = _analysis(
        _segment(FORWARD_DAYS, 0.30, rng, te=0.05),
        _segment(HELDOUT_DAYS, 0.0, rng, te=0.05, exact=True),
    )
    forward = _forward(analysis)

    def heldout(mean_gross):
        return verdict.heldout_slice(
            analysis,
            start=HELDOUT_START,
            end=HELDOUT_END,
            forward_tracking_error=forward["tracking_error"],
            max_drawdown=1.0,
            mean_gross=mean_gross,
        )

    graduated = verdict.graduation_verdict(forward=forward, heldout=heldout(1.0))
    assert graduated["status"] == "graduated" and graduated["reasons"] == []
    assert graduated["thresholds"]["forward_confidence"] == 0.8
    assert graduated["thresholds"]["heldout_tolerance_z"] == 1.28

    weak_forward = _forward(analysis, round_trips=0, mean_gross=0.1)
    discarded = verdict.graduation_verdict(forward=weak_forward, heldout=heldout(0.2))
    assert discarded["status"] == "discarded"
    assert discarded["reasons"] == [
        "forward_too_few_round_trips",
        "forward_exposure_below_floor",
        "heldout_exposure_below_floor",
    ]

    crashed = verdict.graduation_verdict(
        forward=None, heldout=None, strategy_error="forward"
    )
    assert crashed == {
        "status": "discarded",
        "reasons": ["forward_strategy_error"],
        "thresholds": {},
    }
    late = verdict.graduation_verdict(
        forward=None, heldout=None, strategy_error="heldout"
    )
    assert late["status"] == "discarded" and late["reasons"] == [
        "heldout_strategy_error"
    ]
    # A replay that raised produced no result: no slice rides with the error.
    with pytest.raises(ValueError, match="do not match"):
        verdict.graduation_verdict(
            forward=forward, heldout=None, strategy_error="heldout"
        )
    with pytest.raises(ValueError, match="do not match"):
        verdict.graduation_verdict(forward=forward, heldout=None)


def test_an_unmeasurable_slice_raises_instead_of_judging():
    rng = np.random.default_rng(91)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.10, rng))

    with pytest.raises(ValueError, match="not measurable"):
        _forward(analysis, start="20250701", end="20250708")
    with pytest.raises(ValueError, match="bootstrap block"):
        _forward(analysis, start="20250701", end="20250718")
    with pytest.raises(ValueError, match="not measurable"):
        _forward({**analysis, "size_factor_daily": []})
    with pytest.raises(ValueError, match="YYYYMMDD"):
        _forward(analysis, end="2026-06-30")
