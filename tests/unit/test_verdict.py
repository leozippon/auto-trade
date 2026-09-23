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


def _segment(days, alpha, rng, *, te=0.13, exact=False, beta=0.8):
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
    strategy = alpha / TRADING_DAYS_PER_YEAR + beta * benchmark + 0.3 * size + noise
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
        "active_max_drawdown": 1.0,
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


def test_bootstrap_bound_is_fixed_by_the_artifact_id(monkeypatch):
    rng = np.random.default_rng(12)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.05, rng))

    first = _forward(analysis, seed_key="artifact-1")["lower_bound"]
    assert _forward(analysis, seed_key="artifact-1")["lower_bound"] == first
    assert _forward(analysis, seed_key="artifact-2")["lower_bound"] != first
    assert first < window_neutralized_excess(analysis)

    # The refits are batched only to bound the resample's footprint, and every
    # reduction stays inside one draw, so the batch size must not move a frozen
    # artifact's bound by even one ULP. One draw per batch is the extreme.
    monkeypatch.setattr(verdict, "_BOOTSTRAP_BATCH_BYTES", 1)
    assert _forward(analysis, seed_key="artifact-1")["lower_bound"] == first


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
            "active_max_drawdown": 1.0,
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
    assert heldout(0.0, active_max_drawdown=0.0)["reasons"] == [
        "heldout_active_drawdown_exceeded"
    ]
    assert heldout(0.0, mean_gross=0.49)["reasons"] == ["heldout_exposure_below_floor"]


def test_deflated_sharpe_reproduces_the_design_example():
    """PL1 §3.3: N = 20, IR s.d. 0.3 → SR* ≈ 0.57; a 4-year IR of 0.6 → ≈ 0.52."""

    returns = np.random.default_rng(61).normal(0.0, 0.01, 968)
    block = verdict.deflated_sharpe(
        observed_sharpe=0.6, effective_trials=20, trial_sharpe_std=0.3, returns=returns
    )

    assert block["sharpe_star"] == pytest.approx(0.57, abs=0.005)
    assert block["deflated_sharpe_probability"] == pytest.approx(0.52, abs=0.01)
    assert verdict.forward_mde(0.13, TRADING_DAYS_PER_YEAR) == pytest.approx(0.2756)


def _expected_max(trials):
    """The Bailey-López de Prado expected maximum, written out independently."""

    gamma, normal = 0.5772156649015329, NormalDist()
    return (1 - gamma) * normal.inv_cdf(1 - 1 / trials) + gamma * normal.inv_cdf(
        1 - 1 / (trials * math.e)
    )


def test_the_expected_maximum_is_the_formula_floored_at_one_trial():
    for trials in (2, 3, 5, 10, 15, 100):
        assert verdict.expected_max_sharpe(trials) == pytest.approx(_expected_max(trials))
    # Below N ≈ 1.3 the approximation turns negative; one trial's maximum is 0.
    assert _expected_max(1.1) < 0
    assert verdict.expected_max_sharpe(1.1) == verdict.expected_max_sharpe(1) == 0.0
    counts = [1, 1.1, 1.3, 1.5, 2, 2.5, 3, 5, 10]
    readings = [verdict.expected_max_sharpe(count) for count in counts]
    assert readings == sorted(readings)


def test_the_bar_over_four_research_years_is_the_documented_table():
    """√V = √(244/T): the IR the default 0.975 threshold asks for at N_eff
    1/2/3/5 over four research years (976 days), for normal returns -- and the
    former 0.90 threshold's, for comparison."""

    null = verdict.null_sharpe_std(4 * TRADING_DAYS_PER_YEAR)
    assert null == pytest.approx(0.5)
    assert verdict.FREEZE_MIN_DSR_PROBABILITY == 0.975
    for threshold, expected in (
        (verdict.FREEZE_MIN_DSR_PROBABILITY, [0.98, 1.24, 1.41, 1.58]),
        (0.90, [0.64, 0.90, 1.07, 1.24]),
    ):
        z = NormalDist().inv_cdf(threshold)
        bars = [null * (verdict.expected_max_sharpe(n) + z) for n in (1, 2, 3, 5)]
        assert bars == pytest.approx(expected, abs=0.005)
    with pytest.raises(ValueError, match="days must be an integer >= 2"):
        verdict.null_sharpe_std(1)


def test_the_freeze_gate_deflates_by_the_null_sampling_error_at_the_trial_count():
    """SR* is the zero-skill sampling error of an IR over the nominee's own
    days times the expected maximum of the trial count; 0.90 then asks the IR
    to clear SR* by 1.28 of those errors, not merely to beat it (at a
    threshold of 0.5 the ``√(T−1)/√(variance_term)`` factor cancels and the
    gate degenerates into a point comparison)."""

    rng = np.random.default_rng(71)
    research = _weekdays("20210701", "20250630")
    analysis = _analysis(_segment(research, 0.16, rng, exact=True))
    statistics = verdict.neutralized_statistics(analysis)
    information_ratio = statistics["information_ratio"]
    null = math.sqrt(TRADING_DAYS_PER_YEAR / statistics["days"])

    def gate(trials, offline=0):
        return verdict.freeze_gate(
            analysis, trials=trials, offline_trials=offline, full_span_validations=2
        )

    for trials in (1, 2, 5, 20):
        block = gate(trials)["deflated_sharpe"]
        # No trial series given: nothing to correlate, so N_eff is M itself.
        assert (block["trial_correlation"], block["trial_correlation_pairs"]) == (0.0, 0)
        assert block["effective_trials"] == block["trials"] == trials
        assert block["trial_sharpe_std"] == pytest.approx(null)
        assert block["sharpe_star"] == pytest.approx(
            null * (_expected_max(trials) if trials > 1 else 0.0)
        )
        sharpe = information_ratio / SCALE
        variance_term = (
            1.0
            - block["return_skew"] * sharpe
            + (block["return_kurtosis"] - 1.0) / 4.0 * sharpe**2
        )
        assert block["deflated_sharpe_probability"] == pytest.approx(
            NormalDist().cdf(
                (sharpe - block["sharpe_star"] / SCALE)
                * math.sqrt(block["return_days"] - 1)
                / math.sqrt(variance_term)
            )
        )

    passing = gate(1)
    assert passing["passed"] and passing["reasons"] == []
    assert passing["deflated_sharpe"]["information_ratio_bar"] < information_ratio
    # Declared offline screens are trials: the same nominee at 2 host trials
    # and 18 screened offline is judged as at 20.
    failing = gate(2, offline=18)
    assert failing["deflated_sharpe"]["deflated_sharpe_probability"] == pytest.approx(
        gate(20)["deflated_sharpe"]["deflated_sharpe_probability"]
    )
    assert (
        failing["deflated_sharpe"]["host_trials"],
        failing["deflated_sharpe"]["offline_trials"],
        failing["deflated_sharpe"]["trials"],
    ) == (2, 18, 20)
    assert failing["reasons"] == ["freeze_deflated_sharpe_below_threshold"]
    # Beating SR* is not enough: the probability reads above 0.5 and fails.
    assert failing["deflated_sharpe"]["sharpe_star"] < information_ratio
    assert failing["deflated_sharpe"]["deflated_sharpe_probability"] > 0.5
    assert failing["deflated_sharpe"]["information_ratio_bar"] > information_ratio

    alone = verdict.freeze_gate(analysis, trials=1, full_span_validations=1)
    assert alone["reasons"] == ["freeze_too_few_full_span_validations"]
    for arguments in (
        {"trials": 0, "full_span_validations": 2},
        {"trials": 1, "offline_trials": -1, "full_span_validations": 2},
        {"trials": 1, "full_span_validations": True},
    ):
        with pytest.raises(ValueError, match="must be an integer"):
            verdict.freeze_gate(analysis, **arguments)


def test_the_dispersion_does_not_depend_on_what_else_the_arm_validated():
    """Two controls far below the nominee and two near-copies of it: the
    trials' IR spreads differ tenfold, their series correlate identically, and
    the gate reads the nominee the same -- controls no longer tax a candidate
    and padding no longer helps it."""

    research = _weekdays("20210701", "20250630")
    nominee = _analysis(_segment(research, 0.13, np.random.default_rng(81), exact=True))
    # Noise projected off the design: each trial's IR is exactly its drift / TE.
    noise = [
        _segment(research, 0.0, np.random.default_rng(82 + k), exact=True) for k in range(2)
    ]

    def family(alphas):
        return [
            _analysis(
                (days, strategy + alpha / TRADING_DAYS_PER_YEAR, benchmark, size)
            )
            for alpha, (days, strategy, benchmark, size) in zip(alphas, noise, strict=True)
        ]

    controls, copies = family([0.02, 0.0]), family([0.125, 0.12])
    spreads = [
        np.std(
            [verdict.neutralized_statistics(item)["information_ratio"] for item in (nominee, *group)],
            ddof=1,
        )
        for group in (controls, copies)
    ]
    assert spreads[0] > 10 * spreads[1]
    readings = [
        verdict.freeze_gate(
            nominee, trials=3, trial_analyses=[nominee, *group], full_span_validations=3
        )["deflated_sharpe"]
        for group in (controls, copies)
    ]
    assert readings[0] == pytest.approx(readings[1])


def test_the_effective_trial_count_reads_the_correlation_of_the_trials_series():
    research = _weekdays("20210701", "20250630")
    rng = np.random.default_rng(91)
    days, _own, benchmark, size = _segment(research, 0.0, rng)
    common = rng.normal(0.0, 0.10 / SCALE, len(days))

    def trial(loading, seed, own_days=days):
        noise = np.random.default_rng(seed).normal(0.0, 0.10 / SCALE, len(days))
        series = 0.1 / TRADING_DAYS_PER_YEAR + 0.8 * benchmark + loading * common
        series = series + math.sqrt(1 - loading**2) * noise
        keep = [index for index, day in enumerate(days) if day in set(own_days)]
        return _analysis(
            (own_days, series[keep], benchmark[keep], size[keep])
        )

    # Pairwise correlation 0.6 through one shared component.
    shared = [trial(math.sqrt(0.6), seed) for seed in (1, 2, 3)]
    correlation, pairs = verdict.trial_correlation(shared)
    assert pairs == 3 and correlation == pytest.approx(0.6, abs=0.08)
    block = verdict.freeze_gate(
        shared[0], trials=3, trial_analyses=shared, full_span_validations=3
    )["deflated_sharpe"]
    assert block["trial_correlation"] == correlation
    assert block["effective_trials"] == pytest.approx(correlation + (1 - correlation) * 3)
    assert block["sharpe_star"] == pytest.approx(
        block["trial_sharpe_std"] * _expected_max(block["effective_trials"])
    )

    # Identical trials are one trial; opposite ones are not fewer than two.
    assert verdict.trial_correlation([shared[0], shared[0]]) == (1.0, 1)
    assert verdict.effective_trials(4, 1.0) == 1.0
    mirrored = {
        **shared[0],
        "strategy_daily": [[day, -value] for day, value in shared[0]["strategy_daily"]],
    }
    assert verdict.trial_correlation([shared[0], mirrored]) == (0.0, 1)
    assert verdict.effective_trials(2, 0.0) == 2.0
    # A sub-span trial is correlated over the years it shares; trials that
    # share no day add no pair, and one trial alone leaves N_eff = M.
    first_year = [day for day in days if day <= "20220630"]
    last_year = [day for day in days if day >= "20240701"]
    year_one = trial(math.sqrt(0.6), 1, first_year)
    correlation, pairs = verdict.trial_correlation([shared[0], year_one])
    assert pairs == 1 and correlation > 0.9
    assert verdict.trial_correlation([year_one, trial(0.0, 5, last_year)]) == (0.0, 0)
    assert verdict.trial_correlation([shared[0]]) == (0.0, 0)


RESEARCH_YEARS = [
    ("20210701", "20220630"),
    ("20220701", "20230630"),
    ("20230701", "20240630"),
    ("20240701", "20250630"),
]


def _with_panel(analysis, panel):
    return {
        **analysis,
        "panel_daily": [
            [day, float(value)]
            for (day, _own), value in zip(analysis["strategy_daily"], panel, strict=True)
        ],
    }


def test_a_zero_panel_reproduces_the_ungraded_figures_exactly():
    """Grading subtracts the panel and changes nothing else: against a panel of
    zeros every statistic, bound and reason is the float it was without one,
    and only ``series`` tells the two records apart."""
    rng = np.random.default_rng(101)
    research = _weekdays("20210701", "20250630")
    analysis = _analysis(_segment(research, 0.16, rng))
    forward = _analysis(
        _segment(FORWARD_DAYS, 0.10, rng), _segment(HELDOUT_DAYS, 0.02, rng, te=0.05)
    )

    def without_series(block):
        return {key: value for key, value in block.items() if key != "series"}

    def heldout(sidecar):
        return verdict.heldout_slice(
            sidecar,
            start=HELDOUT_START,
            end=HELDOUT_END,
            forward_tracking_error=0.1,
            max_drawdown=0.2,
            active_max_drawdown=0.2,
            mean_gross=1.0,
        )

    def gate(sidecar):
        return verdict.freeze_gate(
            sidecar,
            trials=6,
            full_span_validations=3,
            years=RESEARCH_YEARS,
            active_max_drawdown=0.3,
        )

    for read in (gate, _forward, heldout, verdict.neutralized_statistics):
        sidecar = analysis if read is gate else forward
        zeros = _with_panel(sidecar, np.zeros(len(sidecar["strategy_daily"])))
        own, graded = read(sidecar), read(zeros)
        assert (own["series"], graded["series"]) == ("absolute", "active")
        assert without_series(own) == without_series(graded)


def test_the_graded_series_is_the_strategy_minus_its_panel():
    """The return statistics are those of the active series, measured by the
    code that measures any series; the mandate's tracking error and beta stay
    the strategy's own, which the panel must not move."""
    rng = np.random.default_rng(102)
    research = _weekdays("20210701", "20250630")
    days, active, benchmark, size = _segment(research, 0.06, rng, te=0.05, beta=0.05)
    _days, panel, _b, _s = _segment(research, 0.04, np.random.default_rng(103), beta=0.9)
    # The panel rides the same factors as the book it was drawn from.
    panel = panel - _b * 0.9 - _s * 0.3 + benchmark * 0.9 + size * 0.3
    book = _with_panel(_analysis((days, active + panel, benchmark, size)), panel)
    alone = _analysis((days, active, benchmark, size))

    graded = verdict.neutralized_statistics(book)
    assert graded["series"] == "active"
    for key in ("days", "neutralized_excess"):
        assert graded[key] == verdict.neutralized_statistics(alone)[key]
    assert graded["tracking_error"] == pytest.approx(
        verdict.neutralized_statistics(alone)["tracking_error"], rel=1e-9
    )

    gate = verdict.freeze_gate(
        book, trials=2, full_span_validations=2, years=RESEARCH_YEARS, active_max_drawdown=1.0
    )
    own = verdict.neutralized_statistics({**book, "panel_daily": []})
    assert gate["mandate"] == {
        "tracking_error": own["tracking_error"],
        "market_beta": own["market_beta"],
    }
    assert gate["mandate"]["market_beta"] == pytest.approx(0.95, abs=0.03)
    assert gate["active_max_drawdown"] == pytest.approx(
        verdict._max_slice_drawdown(alone, "", ""), rel=1e-9
    )

    # A day the panel does not cover would be graded against nothing.
    short = {**book, "panel_daily": book["panel_daily"][:-1]}
    with pytest.raises(ValueError, match="zero-skill panel has no return"):
        verdict.neutralized_statistics(short)


def test_the_freeze_gate_refuses_zero_skill_an_uneven_edge_and_a_broken_mandate():
    research = _weekdays("20210701", "20250630")

    def year_days(index):
        start, end = RESEARCH_YEARS[index]
        return [day for day in research if start <= day <= end]

    def book(alphas, *, seed, te=0.05, beta=1.0, panel_alpha=0.03):
        rng = np.random.default_rng(seed)
        segments = [
            _segment(year_days(index), alpha, rng, te=te, exact=True, beta=0.0)
            for index, alpha in enumerate(alphas)
        ]
        active = np.concatenate([segment[1] - 0.3 * segment[3] for segment in segments])
        benchmark = np.concatenate([segment[2] for segment in segments])
        size = np.concatenate([segment[3] for segment in segments])
        panel = panel_alpha / TRADING_DAYS_PER_YEAR + beta * benchmark
        return _with_panel(_analysis((research, active + panel, benchmark, size)), panel)

    def gate(sidecar, **limits):
        arguments = {"years": RESEARCH_YEARS, "active_max_drawdown": 0.30, **limits}
        return verdict.freeze_gate(sidecar, trials=4, full_span_validations=4, **arguments)

    skilled = gate(book([0.10, 0.10, 0.10, 0.10], seed=111))
    assert skilled["passed"] and skilled["series"] == "active"
    assert skilled["information_ratio"] == pytest.approx(2.0, abs=0.05)
    assert skilled["positive_years"] == 4
    assert skilled["thresholds"] == {
        "min_information_ratio": 0.75,
        "min_deflated_sharpe_probability": 0.975,
        "min_full_span_validations": 2,
        "min_positive_years": 3,
        "research_years": 4,
        "active_max_drawdown": 0.30,
        "tracking_error_cap": None,
        "beta_min": None,
        "beta_max": None,
        "panel_draws": 20,
    }

    # Zero skill: the whole +3 %/yr of the book is what its panel earned.
    lottery = gate(book([0.0, 0.0, 0.0, 0.0], seed=112))
    assert "freeze_information_ratio_below_threshold" in lottery["reasons"]
    assert "freeze_too_few_positive_years" in lottery["reasons"]
    assert verdict.neutralized_statistics({**book([0.0] * 4, seed=112), "panel_daily": []})[
        "neutralized_excess"
    ] == pytest.approx(0.03, abs=1e-4)

    # The same total edge earned in two years of four.
    uneven = gate(book([0.30, 0.22, -0.06, -0.06], seed=113))
    assert uneven["information_ratio"] > 0.75 and uneven["positive_years"] == 2
    assert uneven["reasons"] == ["freeze_too_few_positive_years"]
    assert gate(book([0.30, 0.10, 0.06, -0.06], seed=113))["positive_years"] == 3

    steady = book([0.10, 0.10, 0.10, 0.10], seed=114)
    drawdown = gate(steady)["active_max_drawdown"]
    assert gate(steady, active_max_drawdown=drawdown)["passed"]
    assert gate(steady, active_max_drawdown=drawdown - 1e-9)["reasons"] == [
        "freeze_active_drawdown_exceeded"
    ]

    # The tracking mandate reads the strategy's own series: a skilled book that
    # carries the frozen d2's shape (tracking error 11.4 %, beta 0.71) is refused.
    mandate = {"tracking_error_cap": 0.08, "beta_min": 0.85, "beta_max": 1.15}
    assert gate(steady, **mandate)["passed"]
    like_d2 = gate(book([0.20] * 4, seed=115, te=0.114, beta=0.71), **mandate)
    assert like_d2["mandate"]["tracking_error"] == pytest.approx(0.114, abs=0.002)
    assert like_d2["mandate"]["market_beta"] == pytest.approx(0.71, abs=0.01)
    assert like_d2["reasons"] == [
        "freeze_tracking_error_above_cap",
        "freeze_beta_outside_band",
    ]
    assert gate(steady, tracking_error_cap=0.04, beta_min=0.85, beta_max=1.15)["reasons"] == [
        "freeze_tracking_error_above_cap"
    ]
    with pytest.raises(ValueError, match="beta band"):
        gate(steady, tracking_error_cap=0.08)

    # Create-time bars move the gate: a missing key is today's default, an
    # override is the arm's own.
    assert gate(book([0.10, 0.10, 0.10, 0.10], seed=111))["passed"]
    assert "freeze_information_ratio_below_threshold" in gate(
        book([0.10, 0.10, 0.10, 0.10], seed=111), min_active_ir=3.0
    )["reasons"]
    two_of_four = book([0.30, 0.22, -0.06, -0.06], seed=113)
    assert gate(two_of_four)["reasons"] == ["freeze_too_few_positive_years"]
    assert gate(two_of_four, min_positive_year_share=0.5)["passed"]


def test_the_freeze_gate_names_a_deflated_sharpe_it_cannot_compute():
    """A book identical to its zero-skill panel grades an all-zero active
    series: no tracking error, so no IR and no deflated Sharpe probability. The
    gate says the probability is unavailable rather than judging a ``None``."""

    research = _weekdays("20210701", "20250630")
    days, strategy, benchmark, size = _segment(research, 0.10, np.random.default_rng(116))
    copy = _with_panel(_analysis((days, strategy, benchmark, size)), strategy)

    gate = verdict.freeze_gate(copy, trials=2, full_span_validations=2)

    assert gate["series"] == "active" and gate["information_ratio"] is None
    assert gate["deflated_sharpe"]["deflated_sharpe_probability"] is None
    assert gate["deflated_sharpe"]["unavailable_reason"] == "no_observed_sharpe"
    assert not gate["passed"]
    assert gate["reasons"] == [
        "freeze_information_ratio_below_threshold",
        "freeze_deflated_sharpe_unavailable",
    ]


def test_the_forward_mandate_and_active_drawdown_fail_at_their_boundaries():
    rng = np.random.default_rng(121)
    days, active, benchmark, size = _segment(FORWARD_DAYS, 0.30, rng, te=0.05, beta=0.0)
    panel = 1.0 * benchmark
    book = _with_panel(_analysis((days, active + panel, benchmark, size)), panel)
    base = _forward(book)

    assert base["reasons"] == [] and base["series"] == "active"
    assert base["max_drawdown"] != base["active_max_drawdown"]
    assert _forward(book, active_max_drawdown=base["active_max_drawdown"])["reasons"] == []
    assert _forward(book, active_max_drawdown=base["active_max_drawdown"] - 1e-9)[
        "reasons"
    ] == ["forward_active_drawdown_exceeded"]
    band = {"beta_min": 0.85, "beta_max": 1.15}
    own_te = base["mandate"]["tracking_error"]
    assert _forward(book, tracking_error_cap=own_te, **band)["reasons"] == []
    assert _forward(book, tracking_error_cap=own_te - 1e-9, **band)["reasons"] == [
        "forward_tracking_error_above_cap"
    ]
    assert _forward(book, tracking_error_cap=own_te, beta_min=1.05, beta_max=1.15)[
        "reasons"
    ] == ["forward_beta_outside_band"]
    assert base["thresholds"]["tracking_error_cap"] is None
    assert _forward(book, mean_gross=0.6)["reasons"] == []
    assert _forward(book, mean_gross=0.6, min_mean_gross=0.7)["reasons"] == [
        "forward_exposure_below_floor"
    ]


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
            active_max_drawdown=1.0,
            mean_gross=mean_gross,
        )

    graduated = verdict.graduation_verdict(forward=forward, heldout=heldout(1.0))
    assert graduated["status"] == "graduated" and graduated["reasons"] == []
    assert graduated["thresholds"]["forward_confidence"] == 0.8
    assert graduated["thresholds"]["heldout_tolerance_z"] == 1.28
    tight_heldout = verdict.heldout_slice(
        analysis,
        start=HELDOUT_START,
        end=HELDOUT_END,
        forward_tracking_error=forward["tracking_error"],
        max_drawdown=1.0,
        active_max_drawdown=1.0,
        mean_gross=0.6,
        min_mean_gross=0.7,
    )
    assert "heldout_exposure_below_floor" in tight_heldout["reasons"]

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
