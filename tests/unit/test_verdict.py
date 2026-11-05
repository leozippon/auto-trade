"""Verdict statistics: freeze gate, forward verdict, Held-out rule
(docs/PIPELINE_DESIGN.md)."""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.replay.style import factor_ols, window_plain_excess
from autotrade.pipelines import verdict
from autotrade.pipelines.config import ACCEPTANCE_KEYS, AcceptanceRules
from tests.unit.acceptance_fixture import REQUIRED_RULES

SCALE = math.sqrt(TRADING_DAYS_PER_YEAR)
FORWARD_START, FORWARD_END = "20250701", "20260630"
HELDOUT_START, HELDOUT_END = "20260701", "20260911"


def _weekdays(start: str, end: str) -> list[str]:
    return [day.strftime("%Y%m%d") for day in pd.bdate_range(start, end)]


FORWARD_DAYS = _weekdays(FORWARD_START, FORWARD_END)
HELDOUT_DAYS = _weekdays(HELDOUT_START, HELDOUT_END)


def _segment(days, alpha, rng, *, te=0.13, exact=False, beta=0.0, size_beta=0.0, value_beta=0.0):
    """A book whose daily series has annualised mean ``alpha`` and standard
    deviation about ``te`` beside its loadings: ``beta`` on CSI 300,
    ``size_beta`` on size and ``value_beta`` on value, which add their own
    variance.

    ``exact`` demeans the market and size legs and projects the noise off
    them, so the segment's own annualised mean is ``alpha`` to floating
    precision. The value leg comes from a generator of its own, so ``rng``
    draws what it drew before the leg existed.
    """

    n = len(days)
    benchmark = rng.normal(0.0, 0.20 / SCALE, n)
    size = rng.normal(0.0, 0.10 / SCALE, n)
    noise = rng.normal(0.0, te / SCALE, n)
    value = np.random.default_rng(n).normal(0.0, 0.08 / SCALE, n)
    if exact:
        benchmark -= benchmark.mean()
        size -= size.mean()
        design = np.column_stack([np.ones(n), benchmark, size])
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    strategy = (
        alpha / TRADING_DAYS_PER_YEAR + beta * benchmark + size_beta * size + value_beta * value + noise
    )
    return list(days), strategy, benchmark, size, value


def _analysis(*segments):
    """A sidecar of ``segments``, each ``(days, strategy, benchmark, size)``
    with ``value`` as a fifth item where the test names one; a segment that
    names none gets a value leg of independent noise the book does not load
    on, so the three-factor fit is defined."""

    def legs(segment):
        if len(segment) == 5:
            return segment
        days = segment[0]
        noise = np.random.default_rng(len(days) + 1).normal(0.0, 0.08 / SCALE, len(days))
        return (*segment, noise)

    rows = [
        row
        for days, strategy, benchmark, size, value in map(legs, segments)
        for row in zip(days, strategy, benchmark, size, value)
    ]
    return {
        "strategy_daily": [[row[0], float(row[1])] for row in rows],
        "benchmark_daily": [[row[0], float(row[2])] for row in rows],
        "size_factor_daily": [[row[0], float(row[3])] for row in rows],
        "value_factor_daily": [[row[0], float(row[4])] for row in rows],
    }


def _ruled(arguments):
    """``arguments`` with the acceptance rules among them gathered into the
    ``rules`` a stage function takes. Neither drawdown limits unless named,
    so each test judges the condition it is about."""

    named = {key: arguments.pop(key) for key in list(arguments) if key in ACCEPTANCE_KEYS}
    rules = AcceptanceRules(**REQUIRED_RULES, **{"max_drawdown": 1.0, "active_max_drawdown": 1.0, **named})
    return {**arguments, "rules": rules}


def _heldout(analysis, **overrides):
    arguments = {"start": HELDOUT_START, "end": HELDOUT_END, "mean_gross": 1.0, **overrides}
    return verdict.heldout_slice(analysis, **_ruled(arguments))


def _forward(analysis, **overrides):
    """The forward slice of ``analysis``. A forward replay always carries its
    zero-skill panel: a sidecar built without one gets a panel of zeros, which
    leaves every graded figure the book's own
    (:func:`test_a_zero_panel_reproduces_the_ungraded_figures_exactly`)."""

    if "panel_daily" not in analysis:
        analysis = _with_panel(analysis, np.zeros(len(analysis["strategy_daily"])))
    arguments = {
        "start": FORWARD_START,
        "end": FORWARD_END,
        "seed_key": "artifact-1",
        "slippage_bps": 10.0,
        "turnover": 0.0,
        "round_trips": 12,
        "mean_gross": 1.0,
    }
    arguments.update(overrides)
    return verdict.forward_slice(analysis, **_ruled(arguments))


def _split(days, boundary="20260101"):
    return [day for day in days if day < boundary], [
        day for day in days if day >= boundary
    ]


def test_statistics_are_the_series_as_it_is_and_its_loadings_beside_it():
    """The graded figures are the series' own mean and standard deviation,
    with no regression; the factor regression only reports its loadings, and
    the strategy's own residual tracking error and beta form the mandate."""

    rng = np.random.default_rng(11)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.07, rng, beta=0.8, size_beta=0.3, value_beta=-0.2))
    stats = verdict.graded_statistics(analysis, start=FORWARD_START, end=FORWARD_END)

    strategy = np.array([value for _, value in analysis["strategy_daily"]])
    design = np.column_stack(
        [
            np.ones(len(strategy)),
            [value for _, value in analysis["benchmark_daily"]],
            [value for _, value in analysis["size_factor_daily"]],
            [value for _, value in analysis["value_factor_daily"]],
        ]
    )
    coefficients, residual_ss, _, _ = np.linalg.lstsq(design, strategy, rcond=None)

    assert stats["series"] == "absolute" and stats["days"] == len(strategy)
    assert stats["plain_excess"] == pytest.approx(strategy.mean() * TRADING_DAYS_PER_YEAR, rel=1e-12)
    assert stats["plain_excess"] == window_plain_excess(analysis)
    assert stats["tracking_error"] == pytest.approx(strategy.std(ddof=1) * SCALE, rel=1e-12)
    assert stats["information_ratio"] == pytest.approx(stats["plain_excess"] / stats["tracking_error"])
    # The loadings ride as a diagnostic; their variance stays in the series.
    assert (stats["market_beta"], stats["size_beta"], stats["value_beta"]) == pytest.approx(
        tuple(coefficients[1:]), rel=1e-9
    )
    assert stats["tracking_error"] > math.sqrt(residual_ss[0] / (len(strategy) - 4)) * SCALE * 1.5
    mandate = verdict._mandate(analysis, FORWARD_START, FORWARD_END)
    assert mandate == pytest.approx(
        {
            "tracking_error": math.sqrt(residual_ss[0] / (len(strategy) - 4)) * SCALE,
            "market_beta": coefficients[1],
        },
        rel=1e-9,
    )


def test_the_residual_statistic_recovers_a_known_alpha_and_betas():
    """The one statistic the sealed gate grades: OLS of a daily series on the
    three legs, its intercept annualised over its residual standard deviation
    annualised, with the bar z(1 - alpha) / sqrt(years) beside it."""

    rng = np.random.default_rng(5)
    days = 5 * TRADING_DAYS_PER_YEAR
    market = rng.normal(0.0, 0.20 / SCALE, days)
    size = rng.normal(0.0, 0.10 / SCALE, days)
    value = rng.normal(0.0, 0.08 / SCALE, days)
    noise = rng.normal(0.0, 0.10 / SCALE, days)
    design = np.column_stack([np.ones(days), market, size, value])
    # Noise orthogonal to the design: the fit returns the coefficients exactly.
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    active = 0.06 / TRADING_DAYS_PER_YEAR + 0.3 * market - 0.4 * size + 0.25 * value + noise

    reading = verdict.residual_statistics(active, market, size, value, alpha=0.05)

    sigma = math.sqrt(float(noise @ noise) / (days - 4))
    assert reading["days"] == days and reading["years"] == pytest.approx(5.0)
    assert reading["market_beta"] == pytest.approx(0.3, abs=1e-12)
    assert reading["size_beta"] == pytest.approx(-0.4, abs=1e-12)
    assert reading["value_beta"] == pytest.approx(0.25, abs=1e-12)
    assert reading["residual_excess"] == pytest.approx(0.06, abs=1e-12)
    assert reading["residual_tracking_error"] == pytest.approx(sigma * SCALE, rel=1e-12)
    assert reading["residual_information_ratio"] == pytest.approx(0.06 / (sigma * SCALE), rel=1e-9)
    assert reading["bar"] == pytest.approx(NormalDist().inv_cdf(0.95) / math.sqrt(5.0))
    assert reading["bar"] == pytest.approx(0.7356, abs=1e-4)

    # The bar follows the pinned alpha and the span's length, and nothing else.
    assert verdict.residual_bar(1.0, 0.05) == pytest.approx(1.6449, abs=1e-4)
    assert verdict.residual_bar(5.0, 0.025) == pytest.approx(NormalDist().inv_cdf(0.975) / math.sqrt(5.0))
    one_year = verdict.residual_statistics(
        active[:TRADING_DAYS_PER_YEAR],
        market[:TRADING_DAYS_PER_YEAR],
        size[:TRADING_DAYS_PER_YEAR],
        value[:TRADING_DAYS_PER_YEAR],
        alpha=0.05,
    )
    assert one_year["years"] == 1.0 and one_year["bar"] == pytest.approx(1.6449, abs=1e-4)

    # A sidecar's graded series over a slice reads the same statistic.
    analysis = {
        "strategy_daily": [[f"d{index:05d}", float(item)] for index, item in enumerate(active)],
        "benchmark_daily": [[f"d{index:05d}", float(item)] for index, item in enumerate(market)],
        "size_factor_daily": [[f"d{index:05d}", float(item)] for index, item in enumerate(size)],
        "value_factor_daily": [[f"d{index:05d}", float(item)] for index, item in enumerate(value)],
    }
    assert verdict.residual_reading(analysis, alpha=0.05) == {"series": "absolute", **reading, "dropped_days": 0}
    # A non-finite day is dropped and counted, never read as a figure: a NaN
    # does not pass for a series the legs explain exactly.
    holed = {**analysis, "strategy_daily": [[day, math.nan if index == 7 else value] for index, (day, value) in enumerate(analysis["strategy_daily"])]}
    patched = verdict.residual_reading(holed, alpha=0.05)
    assert (patched["days"], patched["dropped_days"]) == (days - 1, 1)
    assert math.isfinite(patched["residual_information_ratio"])

    with pytest.raises(ValueError, match="collinear"):
        verdict.residual_statistics(active, market, market * 2.0, value, alpha=0.05)
    with pytest.raises(ValueError, match="fewer than"):
        verdict.residual_statistics(active[:5], market[:5], size[:5], value[:5], alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        verdict.residual_bar(5.0, 0.0)


def test_bootstrap_bound_is_fixed_by_the_artifact_id(monkeypatch):
    rng = np.random.default_rng(12)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.05, rng))

    first = _forward(analysis, seed_key="artifact-1")["lower_bound"]
    assert _forward(analysis, seed_key="artifact-1")["lower_bound"] == first
    assert _forward(analysis, seed_key="artifact-2")["lower_bound"] != first
    assert first < window_plain_excess(analysis)

    # The draws are batched only to bound the resample's footprint, and every
    # reduction stays inside one draw, so the batch size must not move a frozen
    # artifact's bound by even one ULP. One draw per batch is the extreme.
    monkeypatch.setattr(verdict, "_BOOTSTRAP_BATCH_BYTES", 1)
    assert _forward(analysis, seed_key="artifact-1")["lower_bound"] == first
def test_the_forward_slice_reports_its_readings_and_judges_none():
    """F2-F7 are read and recorded, never judged: the slice carries no
    reasons, and the figures each retired condition read are the slice's
    own, with what they were taken at."""

    rng = np.random.default_rng(41)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.30, rng, te=0.05))
    base = _forward(analysis, round_trips=7, mean_gross=0.4)
    assert "reasons" not in base and "thresholds" not in base
    assert (base["round_trips"], base["mean_gross"]) == (7, 0.4)
    assert base["parameters"] == {
        "forward_confidence": 0.8,
        "bootstrap_block_days": verdict.BOOTSTRAP_BLOCK_DAYS,
        "bootstrap_draws": verdict.BOOTSTRAP_DRAWS,
        "panel_draws": verdict.PANEL_DRAWS,
        "cost_stress_multiplier": 2.0,
    }

    returns = np.array([value for _, value in analysis["strategy_daily"]])
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    assert base["max_drawdown"] == pytest.approx(float(np.max(1.0 - equity / peaks)))

    years = base["days"] / TRADING_DAYS_PER_YEAR
    breakeven = base["plain_excess"] * years / (10.0 * 1e-4)
    assert _forward(analysis, turnover=breakeven)["excess_at_cost_stress"] == pytest.approx(0.0, abs=1e-9)
    # No stress beyond the modelled slippage when the multiplier is one.
    assert _forward(analysis, turnover=breakeven * 2, cost_stress_multiplier=1.0)[
        "excess_at_cost_stress"
    ] == pytest.approx(base["plain_excess"])
    # The sealed gate's statistic over the forward months, for the pool.
    assert base["residual_reading"] == verdict.residual_reading(
        _with_panel(analysis, np.zeros(len(FORWARD_DAYS))), start=FORWARD_START, end=FORWARD_END, alpha=0.05
    )
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
    and only ``series`` and the panel's own reading tell the two records
    apart: a book without a panel has no plain selection, one against a zero
    panel selects exactly its own return. A forward slice reads the same with
    or without it."""
    rng = np.random.default_rng(101)
    research = _weekdays("20210701", "20250630")
    _analysis(_segment(research, 0.16, rng))  # draws the research years first: the stream the forward slice reads
    forward = _analysis(
        _segment(FORWARD_DAYS, 0.10, rng), _segment(HELDOUT_DAYS, 0.02, rng, te=0.05)
    )

    panel_readings = ("panel_return", "plain_selection")

    def without_series(block):
        kept = {key: value for key, value in block.items() if key != "series"}
        if "residual_reading" in kept:
            kept["residual_reading"] = {
                key: value for key, value in kept["residual_reading"].items() if key != "series"
            }
        if "raw_readings" in kept:
            kept["raw_readings"] = {
                key: value
                for key, value in kept["raw_readings"].items()
                if key not in panel_readings
            }
        return kept

    def heldout(sidecar):
        return _heldout(
            sidecar, max_drawdown=0.2, active_max_drawdown=0.2
        )

    def forward_slice(sidecar):
        arguments = {
            "start": FORWARD_START,
            "end": FORWARD_END,
            "seed_key": "artifact-1",
            "slippage_bps": 10.0,
            "turnover": 5.0,
            "round_trips": 12,
            "mean_gross": 1.0,
        }
        return verdict.forward_slice(sidecar, **_ruled(arguments))

    for read in (heldout, forward_slice, verdict.graded_statistics):
        sidecar = forward
        zeros = _with_panel(sidecar, np.zeros(len(sidecar["strategy_daily"])))
        own, graded = read(sidecar), read(zeros)
        assert (own["series"], graded["series"]) == ("absolute", "active")
        assert without_series(own) == without_series(graded)
        if "raw_readings" in own:
            assert {key: own["raw_readings"][key] for key in panel_readings} == {
                "panel_return": None,
                "plain_selection": None,
            }
            assert graded["raw_readings"]["panel_return"] == 0.0
            assert graded["raw_readings"]["plain_selection"] == own["raw_readings"]["strategy_return"]


def test_the_graded_series_is_the_strategy_minus_its_panel():
    """The return statistics are those of the active series, measured by the
    code that measures any series; the mandate's tracking error and beta stay
    the strategy's own, which the panel must not move."""
    rng = np.random.default_rng(102)
    research = _weekdays("20210701", "20250630")
    days, active, benchmark, size, _value = _segment(research, 0.06, rng, te=0.05, beta=0.05)
    _days, panel, _b, _s, _v = _segment(
        research, 0.04, np.random.default_rng(103), beta=0.9, size_beta=0.3
    )
    # The panel rides the same factors as the book it was drawn from.
    panel = panel - _b * 0.9 - _s * 0.3 + benchmark * 0.9 + size * 0.3
    book = _with_panel(_analysis((days, active + panel, benchmark, size)), panel)
    alone = _analysis((days, active, benchmark, size))

    graded = verdict.graded_statistics(book)
    assert graded["series"] == "active"
    assert graded["days"] == verdict.graded_statistics(alone)["days"]
    for key in ("plain_excess", "tracking_error", "market_beta"):
        assert graded[key] == pytest.approx(verdict.graded_statistics(alone)[key], rel=1e-9)

    registered = verdict.registered_mandate(book, rules=AcceptanceRules(**REQUIRED_RULES))
    own = verdict.graded_statistics({**book, "panel_daily": []})
    assert registered["mandate"]["market_beta"] == pytest.approx(own["market_beta"], rel=1e-12)
    assert registered["mandate"]["market_beta"] == pytest.approx(0.95, abs=0.03)
    # The mandate's tracking error is the residual one, below the series' own.
    assert registered["mandate"]["tracking_error"] < own["tracking_error"]
    # No mandate set, none judged; one set and broken, refused.
    assert registered["reasons"] == []
    tight = AcceptanceRules(**REQUIRED_RULES, tracking_error_cap=0.01, beta_min=0.2, beta_max=0.5)
    assert verdict.registered_mandate(book, rules=tight)["reasons"] == [
        "freeze_tracking_error_above_cap",
        "freeze_beta_outside_band",
    ]

    # A day the panel does not cover would be graded against nothing.
    short = {**book, "panel_daily": book["panel_daily"][:-1]}
    with pytest.raises(ValueError, match="zero-skill panel has no return"):
        verdict.graded_statistics(short)


def test_the_forward_mandate_and_active_drawdown_are_reported_whatever_the_rules():
    rng = np.random.default_rng(121)
    days, active, benchmark, size, _value = _segment(FORWARD_DAYS, 0.30, rng, te=0.05, beta=0.0)
    panel = 1.0 * benchmark
    book = _with_panel(_analysis((days, active + panel, benchmark, size)), panel)
    base = _forward(book)

    assert base["series"] == "active"
    assert base["max_drawdown"] != base["active_max_drawdown"]
    # A tight mandate and drawdown limit move no reading and judge nothing.
    tight = _forward(
        book,
        active_max_drawdown=base["active_max_drawdown"] / 2,
        tracking_error_cap=base["mandate"]["tracking_error"] / 2,
        beta_min=1.05,
        beta_max=1.15,
    )
    assert tight == base


def test_an_unmeasurable_slice_raises_instead_of_judging():
    rng = np.random.default_rng(91)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.10, rng))

    with pytest.raises(ValueError, match="fewer than two days"):
        _forward(analysis, start="20250701", end="20250701")
    with pytest.raises(ValueError, match="bootstrap block"):
        _forward(analysis, start="20250701", end="20250718")
    with pytest.raises(ValueError, match="YYYYMMDD"):
        _forward(analysis, end="2026-06-30")
    # The factors are diagnostics: without them the slice is read all the
    # same, its loadings and its residual reading unmeasured.
    bare = _forward({**analysis, "size_factor_daily": []})
    assert bare["lower_bound"] == _forward(analysis)["lower_bound"]
    assert (bare["market_beta"], bare["size_beta"]) == (None, None)
    assert bare["mandate"] == {"tracking_error": None, "market_beta": None}
    assert "unmeasured" in bare["residual_reading"]


def _summary(*, slippage_bps=5.0, turnover=40.0):
    """The two summary fields the freeze gate's cost stress reads."""

    return {"turnover": turnover, "cost_sensitivity": {"slippage_bps": slippage_bps}}


def test_the_graduations_cost_stress_is_one_formula_for_a_row_and_a_forward_slice():
    """The forward slice's F5 reading, the freeze gate's and a research row's
    report are the same function of the graded series, the turnover and the
    slippage."""

    rng = np.random.default_rng(117)
    analysis = _analysis(_segment(FORWARD_DAYS, 0.10, rng, exact=True))
    block = _forward(analysis, turnover=30.0, slippage_bps=5.0, cost_stress_multiplier=2.0)
    statistics = verdict.graded_statistics(_with_panel(analysis, np.zeros(len(FORWARD_DAYS))))
    assert block["excess_at_cost_stress"] == verdict.excess_at_cost_stress(
        statistics["plain_excess"],
        statistics["days"],
        cost_stress_multiplier=2.0,
        slippage_bps=5.0,
        turnover=30.0,
    )
    assert block["excess_at_cost_stress"] == verdict.summary_excess_at_cost_stress(
        statistics, _summary(turnover=30.0), cost_stress_multiplier=2.0
    )
    assert block["excess_at_cost_stress"] == pytest.approx(
        0.10 - 5.0 * 30.0 * 1e-4 / (statistics["days"] / TRADING_DAYS_PER_YEAR)
    )


def test_the_panel_return_is_what_the_panel_composite_compounded_to():
    days = _weekdays("20210701", "20210730")
    panel = [-0.001] * len(days)
    sidecar = {"panel_daily": [[day, value] for day, value in zip(days, panel)]}
    assert verdict.panel_return(sidecar) == pytest.approx((1 - 0.001) ** len(days) - 1)
    assert verdict.panel_return({}) is None


def _rally_book(rng, *, alpha, active_beta, days=FORWARD_DAYS):
    """A book whose selection earns ``alpha`` a year after the regression but
    holds lower-beta names than its random copies (``active_beta`` < 0) while
    the benchmark rallies about 45 % a year: a regression intercept would
    credit the lag back, the holder does not get it, and neither does the
    series the gates grade."""

    n = len(days)
    benchmark = rng.normal(0.0015, 0.01, n)
    size = rng.normal(0.0, 0.004, n)
    panel = 0.9 * benchmark + rng.normal(0.0, 0.003, n)
    active = alpha / TRADING_DAYS_PER_YEAR + active_beta * benchmark + rng.normal(0.0, 0.003, n)
    return _with_panel(_analysis((list(days), panel + active, benchmark, size)), panel)


def _selection_book(rng, mean, *, days=FORWARD_DAYS):
    """A book whose daily return minus its panel's averages exactly ``mean`` a
    year over ``days``, around noise of about 4.8 % a year."""

    n = len(days)
    benchmark = rng.normal(0.0, 0.01, n)
    size = rng.normal(0.0, 0.004, n)
    panel = 0.9 * benchmark + rng.normal(0.0, 0.003, n)
    noise = rng.normal(0.0, 0.003, n)
    active = noise - noise.mean() + mean / TRADING_DAYS_PER_YEAR
    return _with_panel(_analysis((list(days), panel + active, benchmark, size)), panel)


def _intercept(analysis, start="", end=""):
    """The annualised intercept of the graded series on the three factor
    legs: what a regressed grade would have read."""

    graded, _series = verdict._graded(analysis)
    rows, _dropped = verdict._regression_rows(graded, start, end)
    return float(factor_ols(rows[:, 0], rows[:, 1:])[0][0]) * TRADING_DAYS_PER_YEAR


def test_f2_grades_a_negative_market_loading_on_what_it_earned():
    """A book that lost to its own panel in a rally because it held
    lower-beta names: a regressed reading would pass it, F2 reads the series
    the holder earned and refuses it, and the holder's readings say why."""

    rng = np.random.default_rng(118)
    book = _rally_book(rng, alpha=0.12, active_beta=-0.5)
    judged = _forward(book)
    raw = judged["raw_readings"]
    assert _intercept(book, FORWARD_START, FORWARD_END) > 0
    assert judged["market_beta"] == pytest.approx(-0.5, abs=0.05)
    assert raw["plain_selection"] < 0
    assert judged["lower_bound"] < judged["plain_excess"] < 0


def test_heldout_reads_a_negative_market_loading_on_what_it_earned():
    """Held-out reads the plain series too: a book that lagged its panel
    through a Held-out rally on its lower-beta names reads below zero where
    its regression intercept would not."""

    book = _rally_book(
        np.random.default_rng(123), alpha=0.0, active_beta=-0.5, days=FORWARD_DAYS + HELDOUT_DAYS
    )
    heldout = _heldout(book)
    assert heldout["plain_excess"] < 0 < _intercept(book, HELDOUT_START, HELDOUT_END)
    assert heldout["market_beta"] < 0


def test_f2_is_the_lower_bound_of_the_plain_mean():
    """F2 reports more than a positive mean: the one-sided bound of the
    graded series' mean. A book frozen with no replicate is read on its own
    series."""

    rng = np.random.default_rng(122)

    def judged(mean, **overrides):
        return _forward(_selection_book(rng, mean), **overrides)

    # A mean above zero inside its own noise reads a negative bound.
    thin = judged(0.01)
    assert thin["plain_excess"] == pytest.approx(0.01, rel=1e-9)
    assert thin["lower_bound"] < 0 < thin["plain_excess"]
    # One clear of its noise reads a positive one.
    clear = judged(0.15)
    assert 0 < clear["lower_bound"] < clear["plain_excess"]

    # The bound is the documented procedure on the plain series: blocks of
    # BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_DRAWS draws, the artifact seed.
    book = _selection_book(rng, 0.05)
    series = np.array(
        [
            own - drawn
            for (_day, own), (_same, drawn) in zip(book["strategy_daily"], book["panel_daily"], strict=True)
        ]
    )
    bound = _forward(book, forward_confidence=0.9)
    seed = int.from_bytes(hashlib.sha256(b"artifact-1").digest()[:8], "big")
    blocks = -(-len(series) // verdict.BOOTSTRAP_BLOCK_DAYS)
    starts = np.random.default_rng(seed).integers(
        0, len(series) - verdict.BOOTSTRAP_BLOCK_DAYS + 1, size=(verdict.BOOTSTRAP_DRAWS, blocks)
    )
    index = (starts[:, :, None] + np.arange(verdict.BOOTSTRAP_BLOCK_DAYS)).reshape(len(starts), -1)
    means = series[index[:, : len(series)]].mean(axis=1)
    assert bound["lower_bound"] == pytest.approx(
        float(np.quantile(means, 0.1)) * TRADING_DAYS_PER_YEAR, rel=1e-12
    )
    assert _forward(book, seed_key="artifact-2")["lower_bound"] != pytest.approx(
        _forward(book)["lower_bound"]
    )


def test_the_holders_readings_are_the_stored_series_compounded():
    """Every reading is the stored daily series compounded over the slice's own
    days, the differences are cumulative, and the plain excess is the graded
    series' mean over the slice's days; the verdict states them in words for
    both slices."""

    rng = np.random.default_rng(119)
    book = _rally_book(rng, alpha=0.05, active_beta=-0.3, days=FORWARD_DAYS + HELDOUT_DAYS)

    def compound(key, start, end):
        values = [value for day, value in book[key] if start <= day <= end]
        return float(np.prod(1.0 + np.array(values)) - 1.0)

    forward = _forward(book)
    raw = forward["raw_readings"]
    strategy = compound("strategy_daily", FORWARD_START, FORWARD_END)
    benchmark = compound("benchmark_daily", FORWARD_START, FORWARD_END)
    panel = compound("panel_daily", FORWARD_START, FORWARD_END)
    assert raw == pytest.approx(
        {
            "strategy_return": strategy,
            "benchmark_return": benchmark,
            "panel_return": panel,
            "raw_excess": strategy - benchmark,
            "plain_selection": strategy - panel,
        },
        rel=1e-12,
    )
    active = [
        own - drawn
        for (day, own), (_day, drawn) in zip(book["strategy_daily"], book["panel_daily"], strict=True)
        if FORWARD_START <= day <= FORWARD_END
    ]
    assert forward["plain_excess"] == pytest.approx(np.mean(active) * TRADING_DAYS_PER_YEAR, rel=1e-12)
    # The whole-span reading of a sidecar is the panel_return of before.
    assert verdict.holder_readings(book)["panel_return"] == pytest.approx(verdict.panel_return(book))
    # A span with no stored day reads nothing rather than zero.
    assert verdict.holder_readings(book, start="20300101", end="20301231") == dict.fromkeys(raw)

    heldout = _heldout(book)
    assert heldout["raw_readings"]["strategy_return"] == pytest.approx(
        compound("strategy_daily", HELDOUT_START, HELDOUT_END), rel=1e-12
    )


def test_f2_reads_the_book_and_its_seed_replicates_as_one_daily_series():
    """With seed replicates the forward slice carries each one's readings and
    their mean with the book, and F2 bounds the day-by-day mean of their
    active series, whatever the frozen seed read alone; a replicate that
    cannot be read alike is never dropped from the mean, the slice is not
    read. Held-out reports the replicates too."""

    rng = np.random.default_rng(121)
    book = _selection_book(rng, 0.12)

    def replicate(analysis, name="replicate"):
        return ({"artifact_id": name, "source_step_id": f"step_{name}"}, analysis)

    alone = _forward(book)
    assert "seed_replicates" not in alone and "seed_mean" not in alone
    assert alone["lower_bound"] > 0

    # The frozen seed clears the bound alone; with the other two seeds the
    # mean is still above zero, and no longer clear of its noise.
    others = [replicate(_selection_book(rng, -0.03), "b"), replicate(_selection_book(rng, -0.07), "c")]
    sunk = _forward(book, seed_replicates=others)
    mean = sunk["seed_mean"]
    assert mean["members"] == 3
    assert mean["plain_excess"] == pytest.approx((0.12 - 0.03 - 0.07) / 3, rel=1e-9)
    assert sunk["lower_bound"] < 0 < mean["plain_excess"]
    assert [block["artifact_id"] for block in sunk["seed_replicates"]] == ["b", "c"]
    assert sunk["seed_replicates"][0]["plain_excess"] == pytest.approx(-0.03, rel=1e-9)
    # The bound is drawn on the day-by-day mean of the three active series.
    members = [book, *(sidecar for _identity, sidecar in others)]
    daily = np.mean(
        [
            [own - drawn for (_day, own), (_same, drawn) in zip(m["strategy_daily"], m["panel_daily"], strict=True)]
            for m in members
        ],
        axis=0,
    )
    assert sunk["lower_bound"] == verdict.bootstrap_lower_bound(daily, "artifact-1", confidence=0.8)
    # Every other reading stays the book's own.
    added = ("seed_replicates", "seed_mean", "lower_bound")
    assert {key: value for key, value in sunk.items() if key not in added} == {
        key: value for key, value in alone.items() if key not in added
    }

    # Seeds that agree pass together.
    agreed = _forward(
        book,
        seed_replicates=[replicate(_selection_book(rng, 0.10)), replicate(_selection_book(rng, 0.08))],
    )
    assert agreed["lower_bound"] > 0

    # A replicate without a panel, or one that measured other days.
    with pytest.raises(ValueError, match="another series"):
        _forward(book, seed_replicates=[replicate({**others[0][1], "panel_daily": []})])
    short = _selection_book(rng, 0.10, days=FORWARD_DAYS[:-1])
    with pytest.raises(ValueError, match="measured other days than the frozen book"):
        _forward(book, seed_replicates=[replicate(short)])

    span = FORWARD_DAYS + HELDOUT_DAYS
    long_book, long_other = _selection_book(rng, 0.10, days=span), _selection_book(rng, -0.50, days=span)
    heldout = _heldout(long_book, seed_replicates=[replicate(long_other)])
    assert heldout["seed_mean"]["members"] == 2
    assert "lower_bound" not in heldout


def test_the_condition_table_is_the_one_list_of_what_a_record_can_carry():
    """Every condition has its own reason token, every condition is a
    registration's (nothing after the registration judges), and the only
    optional conditions are the tracking mandate's, which its cap turns on."""

    reasons = [condition.reason for condition in verdict.CONDITIONS]
    assert len(reasons) == len(set(reasons))
    stages: dict[str, list[verdict.Condition]] = {}
    for condition in verdict.CONDITIONS:
        stages.setdefault(condition.stage, []).append(condition)
    assert list(stages) == ["nomination", "registration", "freeze", "seeds"]
    assert {condition.requires for condition in verdict.CONDITIONS} == {"", "tracking_error_cap"}
    assert {condition.reason for condition in stages["seeds"]} == {
        "freeze_seed_replicate_invalid",
        "freeze_too_few_seed_replicates",
    }
    retired = {"sealed", "validity", "strategy_error", "graduat"}
    assert not any(word in reason for reason in reasons for word in retired)


def test_a_reading_is_weighed_by_its_null_p_value_and_readings_pool_by_their_years():
    """Under the null a residual IR over ``years`` is normal with standard
    deviation 1/sqrt(years): its one-sided p-value is 1 - Phi(IR sqrt(years)),
    at the bar exactly alpha. Readings of independent strategies pool by
    their years; one that states no IR is counted, not pooled."""

    assert verdict.null_p_value(verdict.residual_bar(5.0, 0.05), 5.0) == pytest.approx(0.05)
    assert verdict.null_p_value(0.0, 3.0) == pytest.approx(0.5)
    readings = [
        {"residual_information_ratio": 1.0, "years": 1.0},
        {"residual_information_ratio": -0.5, "years": 1.0},
        {"residual_information_ratio": 0.4, "years": 2.0},
        {"unmeasured": "fewer than 60 days"},
    ]
    pooled = verdict.pooled_reading(readings)
    assert (pooled["count"], pooled["unmeasured"], pooled["years"]) == (3, 1, 4.0)
    assert pooled["residual_information_ratio"] == pytest.approx((1.0 - 0.5 + 0.8) / 4.0)
    assert pooled["z"] == pytest.approx(pooled["residual_information_ratio"] * 2.0)
    assert pooled["p_value"] == pytest.approx(1.0 - NormalDist().cdf(pooled["z"]))
    assert verdict.pooled_reading([]) == {"count": 0, "unmeasured": 0}
