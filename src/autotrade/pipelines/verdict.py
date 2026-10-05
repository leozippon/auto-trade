"""Verdict statistics of one arm: freeze gate, forward verdict, Held-out rule.

Pure functions over frozen replay data (docs/pipeline-design.md: the freeze
gate and the graduation verdict). The statistical
return figures are the neutralised excess of ``environment/replay/style.py``: a
daily return series regressed on the arm's benchmark and the replay size factor,
the intercept annualised over ``TRADING_DAYS_PER_YEAR``. The holder's readings
beside them are not regressed: :func:`holder_readings` compounds the book, the
benchmark and the panel as stored, :func:`raw_excess_at_cost_stress` reads the
replay's own excess and :func:`panel_return` the panel's. The inputs are the
daily series a ``style_analysis.json`` sidecar stores, sliced by ``YYYYMMDD`` dates, so the
forward and Held-out slices of one continuous replay are read from one sidecar.

The graded series is the ACTIVE one: the strategy's daily return minus the
zero-skill panel composite the host drew from the strategy's own realized book
(``environment/replay/null_control.py``). What zero skill earns in an account's
own shape is a property of the window and the pool -- between -4 and +8 %/yr
across the measured shapes -- so grading the strategy's own series grades that
and not the strategy. Tracking error against that benchmark and the market beta are
read off the strategy's own series, where a tracking mandate limits them. A
sidecar written before replays carried a panel has none, and its statistics
keep their former meaning: they are the strategy's own, and ``series`` says so.

Point estimates come from ``style.window_neutralized_excess`` itself, so every
neutralised excess here equals the figure the replay reports for the same span.
The bootstrap needs thousands of refits, so the same normal equations run
vectorised in :func:`_fit`; one test pins it to the style figure.

The statistical definitions live here (DSR, the block bootstrap, the panel).
The freeze, forward and Held-out bars are the arm's create-time rules, which
every stage function takes whole (``rules``: a ``config.AcceptanceRules``);
the module constants below are the defaults those fields take when a record
omits them, and nothing here falls back to them. A slice that cannot be
measured (too few days with both factors, collinear factors, a slice shorter
than one bootstrap block) raises ``ValueError``: a verdict must never be read
off a number that was not measured.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from autotrade.environment.replay.null_control import PANEL_DRAWS
from autotrade.environment.replay.stats import (
    TRADING_DAYS_PER_YEAR,
    compounded_drawdown,
    compounded_path,
    compounded_return,
)
from autotrade.environment.replay.style import (
    _series_pairs,
    active_analysis,
    window_neutralized_excess,
)

from .calendar import FULL_SPAN

if TYPE_CHECKING:
    # ``config`` reads this module's constants for its defaults.
    from .config import AcceptanceRules

# Freeze gate (docs/pipeline-design.md), calibrated on its zero-skill pass rate
# and its power, not on the record of earlier freezes
# (``scripts/dev/dsr_recalibration.py``). The forward test alone passes zero
# skill about 15 % of the time, so an arm's protection against a false
# graduate when several arms share one forward window is this gate.
FREEZE_MIN_ACTIVE_IR = 0.75
# Active neutralised excess positive in three of four research years; another
# research length keeps the share, rounded up.
FREEZE_MIN_POSITIVE_YEAR_SHARE = 0.75
# At a threshold of 0.5 the deflated Sharpe's
# ``√(T−1)/√(variance_term)`` factor cancels and the gate degenerates into
# ``SR >= SR*`` -- a point comparison that knows nothing about the estimate's
# sampling error. 0.975 asks the nominee's IR to clear SR* by 1.96 of its
# zero-skill standard errors: over four research years (standard error ≈ 0.5)
# an IR of about 0.98 at one effective trial, 1.24 at two, 1.41 at three and
# 1.58 at five. Recalibrated on 42 arms' own series (DSR1): with the arm's
# best non-control trial nominated, zero skill passes the gate's statistical
# conditions 2.6 % of the time and a true active IR of 1.0 about 61 %, against
# 4.0 % and 44 % for the former rule at 0.90 and 7.5 % and 73 % for this rule
# at 0.90; the record's failure mode is false positives (six freezes, all
# discarded forward).
FREEZE_MIN_DSR_PROBABILITY = 0.975
FREEZE_MIN_FULL_SPAN_VALIDATIONS = 2
# Forward verdict (docs/pipeline-design.md, the graduation verdict).
FORWARD_CONFIDENCE = 0.80
BOOTSTRAP_BLOCK_DAYS = 20
BOOTSTRAP_DRAWS = 2_000
RECENCY_MONTHS = 6
MIN_ROUND_TRIPS_PER_MONTH = 1
MIN_MEAN_GROSS = 0.5
# Held-out: neutralised excess no worse than this many standard errors, the
# standard error taken from the forward slice's tracking error.
HELDOUT_TOLERANCE_Z = 1.28
# Minimum detectable annualised neutralised excess at 80 % power and one-sided
# 10 % (the frozen block's ``forward_mde``), in standard errors:
# Φ⁻¹(0.90) + Φ⁻¹(0.80) = 2.12. F2 itself is a one-sided 20 % bound, so this
# is a deliberately conservative planning figure (about 1.26 × the 1.68 the
# applied gate would give), not F2's own bar.
FORWARD_MDE_Z = 2.12

_EULER_MASCHERONI = 0.5772156649015329
# The columns of one validation's daily graded series (:func:`active_daily`).
ACTIVE_DAILY_COLUMNS = (
    "trade_date",
    "strategy_return",
    "panel_return",
    "active_return",
    "active_cumulative",
    "active_drawdown",
    "benchmark_return",
    "size_factor_return",
    "active_neutralized",
)


def _date(value: str, name: str) -> str:
    text = str(value)
    if not (len(text) == 8 and text.isdigit()):
        raise ValueError(f"{name} must be YYYYMMDD, got {value!r}")
    return text


def _span(start: str, end: str) -> tuple[str, str]:
    start, end = _date(start, "start"), _date(end, "end")
    if start > end:
        raise ValueError(f"slice start {start} is after its end {end}")
    return start, end


def _non_negative(value: object, name: str) -> float:
    number = (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else math.nan
    )
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return number


def _strategy_returns(
    analysis: Mapping[str, object], start: str, end: str
) -> list[tuple[str, float]]:
    return [
        (date, value)
        for date, value in _series_pairs(analysis.get("strategy_daily"))
        if (not start or date >= start) and (not end or date <= end)
    ]


def _regression_join(
    analysis: Mapping[str, object], start: str, end: str
) -> list[tuple[str, float, float, float]]:
    """``(date, strategy, benchmark, size)`` of every day of the slice, joined
    exactly as ``style.window_neutralized_excess`` joins them."""

    benchmark = dict(_series_pairs(analysis.get("benchmark_daily")))
    size = dict(_series_pairs(analysis.get("size_factor_daily")))
    return [
        (date, value, benchmark[date], size[date])
        for date, value in _strategy_returns(analysis, start, end)
        if date in benchmark and date in size
    ]


def _regression_rows(
    analysis: Mapping[str, object], start: str, end: str
) -> np.ndarray:
    """``(strategy, benchmark, size)`` rows of the slice (:func:`_regression_join`)."""

    rows = [row[1:] for row in _regression_join(analysis, start, end)]
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _fit(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily OLS intercept, market beta and size beta over the last two axes.

    The two-regressor normal equations of ``style._neutralized_excess``,
    batched over any leading axes so a bootstrap refits every draw at once.
    """

    means = rows.mean(axis=-2)
    centered = rows - means[..., None, :]
    y, market, size = centered[..., 0], centered[..., 1], centered[..., 2]
    s11 = (market * market).sum(axis=-1)
    s22 = (size * size).sum(axis=-1)
    s12 = (market * size).sum(axis=-1)
    s1y = (market * y).sum(axis=-1)
    s2y = (size * y).sum(axis=-1)
    determinant = s11 * s22 - s12 * s12
    if not np.all(determinant > 0):
        raise ValueError("benchmark and size factor are collinear in the slice")
    market_beta = (s22 * s1y - s12 * s2y) / determinant
    size_beta = (s11 * s2y - s12 * s1y) / determinant
    intercept = means[..., 0] - market_beta * means[..., 1] - size_beta * means[..., 2]
    return intercept, market_beta, size_beta


def _measured(
    analysis: Mapping[str, object], start: str, end: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """:func:`neutralized_statistics` of a span, with its regression rows and
    daily neutralised series (strategy − β·benchmark − s·size, intercept included)."""

    excess = window_neutralized_excess(analysis, start=start, end=end)
    if excess is None:
        raise ValueError(
            f"neutralised excess of {start or 'start'}..{end or 'end'} is not measurable"
        )
    rows = _regression_rows(analysis, start, end)
    _intercept, market_beta, size_beta = _fit(rows)
    neutral = rows[:, 0] - market_beta * rows[:, 1] - size_beta * rows[:, 2]
    residual = neutral - neutral.mean()
    tracking_error = math.sqrt(
        float(residual @ residual) / (len(rows) - 3) * TRADING_DAYS_PER_YEAR
    )
    statistics = {
        "days": len(rows),
        "neutralized_excess": excess,
        "tracking_error": tracking_error,
        "information_ratio": excess / tracking_error if tracking_error > 0 else None,
        "market_beta": float(market_beta),
    }
    return statistics, rows, neutral


def _graded(analysis: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    """The analysis of the series a verdict grades, and that series' name."""

    active = active_analysis(analysis)
    return (analysis, "absolute") if active is None else (active, "active")


def neutralized_statistics(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> dict[str, object]:
    """Neutralised excess, residual tracking error, IR and beta of one span.

    Of the graded series: the active one when the sidecar carries a zero-skill
    panel, else the strategy's own, and ``series`` names which. An empty
    ``start``/``end`` takes the whole analysis, as in
    ``style.window_neutralized_excess``. ``tracking_error`` is the residual
    standard deviation (n − 3 degrees of freedom) annualised by
    √``TRADING_DAYS_PER_YEAR``; ``information_ratio`` is ``None`` when it is 0.
    """

    start = _date(start, "start") if start else ""
    end = _date(end, "end") if end else ""
    graded, series = _graded(analysis)
    return {**_measured(graded, start, end)[0], "series": series}


def _mandate(analysis: Mapping[str, object], start: str, end: str) -> dict[str, object]:
    """The strategy's own tracking error and beta against its benchmark over a
    span: what a tracking mandate limits, and what is reported, not graded,
    where the rules set none."""

    own = _measured(analysis, start, end)[0]
    return {"tracking_error": own["tracking_error"], "market_beta": own["market_beta"]}


def forward_mde(tracking_error: float, forward_days: int) -> float:
    """Smallest annualised neutralised excess a forward test of
    ``forward_days`` detects at 80 % power (``FORWARD_MDE_Z``)."""

    return (
        FORWARD_MDE_Z * tracking_error / math.sqrt(forward_days / TRADING_DAYS_PER_YEAR)
    )


def expected_max_sharpe(trials: float) -> float:
    """Expected maximum of ``trials`` independent standard normal draws.

    Bailey & López de Prado (2014)'s approximation
    ``(1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))`` with γ the Euler-Mascheroni
    constant, for a real N: an effective trial count need not be whole. The
    approximation turns negative below N ≈ 1.3, where it is floored at 0, the
    expectation of one draw -- the maximum of a few near-identical trials is no
    lower than that of one.
    """

    if trials <= 1.0:
        return 0.0
    normal = NormalDist()
    value = (1.0 - _EULER_MASCHERONI) * normal.inv_cdf(
        1.0 - 1.0 / trials
    ) + _EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return max(value, 0.0)


def null_sharpe_std(days: int) -> float:
    """Sampling standard deviation of an annualised Sharpe (IR) estimated from
    ``days`` daily returns of zero skill: √(``TRADING_DAYS_PER_YEAR`` / days).

    The dispersion √V of the False-Strategy null the deflated Sharpe tests:
    under zero skill the trials' IRs scatter by their estimation error alone,
    so √V is a property of the span, not of which trials an arm happened to
    run (≈ 0.5 over four research years).
    """

    if isinstance(days, bool) or not isinstance(days, int) or days < 2:
        raise ValueError(f"days must be an integer >= 2, got {days!r}")
    return math.sqrt(TRADING_DAYS_PER_YEAR / days)


def neutral_daily(analysis: Mapping[str, object]) -> dict[str, float]:
    """The daily neutralised graded series of a whole sidecar, by date: the
    series whose annualised mean over its residual risk is the trial's IR.
    ``ValueError`` when the sidecar's span is not measurable."""

    graded, _series = _graded(analysis)
    dates = [row[0] for row in _regression_join(graded, "", "")]
    _statistics, _rows, neutral = _measured(graded, "", "")
    return dict(zip(dates, (float(value) for value in neutral), strict=True))


def active_daily(analysis: Mapping[str, object]) -> pd.DataFrame:
    """One validation's graded series day by day, as the session reads it.

    One row per replayed day (``ACTIVE_DAILY_COLUMNS``), every column read off
    the computations the gate itself runs on this sidecar: the strategy's
    return and the zero-skill panel composite as stored, the active return
    :func:`style.active_analysis` grades, its :func:`stats.compounded_path`
    (``active_cumulative`` is the equity minus 1, ``active_drawdown`` the loss
    below the running peak, whose largest value is the gate's
    ``active_max_drawdown``), the two regressors, and the
    :func:`neutral_daily` series, whose annualised mean over its residual
    volatility is the active IR. Only the composite of the panel is stored, so
    no draw's names can be read back. Without a panel every ``active_*``
    column is empty; ``active_neutralized`` is also empty on a day missing a
    regressor and on a span the gate cannot measure.
    """

    frame = pd.DataFrame(
        _series_pairs(analysis.get("strategy_daily")),
        columns=["trade_date", "strategy_return"],
    )
    for column, key in (
        ("panel_return", "panel_daily"),
        ("benchmark_return", "benchmark_daily"),
        ("size_factor_return", "size_factor_daily"),
    ):
        frame[column] = frame["trade_date"].map(dict(_series_pairs(analysis.get(key))))
    active = active_analysis(analysis)
    if active is not None:
        # Day for day the strategy's own rows: active_analysis keeps their order.
        graded = [value for _date, value in _series_pairs(active["strategy_daily"])]
        path = compounded_path(graded)
        try:
            neutral = neutral_daily(analysis)
        except ValueError:
            # Unmeasurable: the gate reads no IR off this span either.
            neutral = {}
        frame["active_return"] = graded
        frame["active_cumulative"] = [equity - 1.0 for equity, _drawdown in path]
        frame["active_drawdown"] = [drawdown for _equity, drawdown in path]
        frame["active_neutralized"] = frame["trade_date"].map(neutral)
    return frame.reindex(columns=list(ACTIVE_DAILY_COLUMNS))


def trial_correlation(
    analyses: Sequence[Mapping[str, object]],
    series: Sequence[Mapping[str, float]] = (),
) -> tuple[float, int]:
    """Mean pairwise correlation ρ̄ of the trials' daily graded series, and the
    number of pairs it averages.

    One analysis per trial (its validation sidecar), plus ``series``: trials
    already reduced to that daily series (an arm's lineage, extracted at its
    creation), which pair with the analyses and with each other alike. Each
    series is the daily neutralised graded series (:func:`neutral_daily`), so
    ρ̄ is the correlation of the very estimates the trials' IRs are. A pair is
    correlated over the days both measured -- a sub-span trial against a
    full-span one over its own years; a pair sharing fewer than three days (two
    points correlate ±1 by construction) or flat over its overlap, and a trial
    whose span is unmeasurable, add no pair. The mean is clipped into [0, 1]: M
    trials that hedge each other are not more than M independent ones.
    ``(0.0, 0)`` when no pair is measured (one trial, or none measurable), which
    makes the effective count the raw one.
    """

    measured: list[Mapping[str, float]] = []
    for analysis in analyses:
        try:
            measured.append(neutral_daily(analysis))
        except ValueError:
            continue
    measured.extend(series)
    correlations: list[float] = []
    for first, second in itertools.combinations(measured, 2):
        common = sorted(first.keys() & second.keys())
        if len(common) < 3:
            continue
        x = np.fromiter((first[date] for date in common), dtype=float)
        y = np.fromiter((second[date] for date in common), dtype=float)
        if not (x.std() > 0 and y.std() > 0):
            continue
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    if not correlations:
        return 0.0, 0
    return min(max(float(np.mean(correlations)), 0.0), 1.0), len(correlations)


def effective_trials(trials: int, correlation: float) -> float:
    """Bailey & López de Prado (2014, App. A.3, eq. 9): ``trials`` trials with
    mean pairwise correlation ρ̄ count as ρ̄ + (1 − ρ̄)·``trials`` independent
    ones -- one when they are identical, all of them when independent."""

    return correlation + (1.0 - correlation) * trials


def deflated_sharpe(
    *,
    observed_sharpe: float | None,
    effective_trials: float,
    trial_sharpe_std: float,
    returns: np.ndarray,
) -> dict[str, object]:
    """Deflated Sharpe ratio of one selected series among ``effective_trials``
    independent trials.

    Bailey & López de Prado (2014). Sharpes are annualised; the formula runs
    per period. With E[max_N] of :func:`expected_max_sharpe`,

        SR* = √V · E[max_N]
        P   = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ]

    with γ₃ the skew and γ₄ the (normal = 3) kurtosis of the T returns.
    ``deflated_sharpe_probability`` is ``None`` whenever it is undefined, and
    ``unavailable_reason`` names why.
    """

    scale = math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_star = trial_sharpe_std * expected_max_sharpe(effective_trials)
    block: dict[str, object] = {
        "deflated_sharpe_probability": None,
        "effective_trials": effective_trials,
        "trial_sharpe_std": trial_sharpe_std,
        "sharpe_star": sharpe_star,
        "observed_sharpe": observed_sharpe,
        "return_days": len(returns),
        "return_skew": None,
        "return_kurtosis": None,
        "unavailable_reason": None,
    }
    if observed_sharpe is None:
        block["unavailable_reason"] = "no_observed_sharpe"
        return block
    centered = returns - returns.mean()
    second = float(np.mean(centered**2))
    if second <= 0:
        block["unavailable_reason"] = "zero_return_variance"
        return block
    skew = float(np.mean(centered**3)) / second**1.5
    kurtosis = float(np.mean(centered**4)) / second**2
    block["return_skew"] = skew
    block["return_kurtosis"] = kurtosis
    sharpe = observed_sharpe / scale
    variance_term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0:
        block["unavailable_reason"] = "undefined_sharpe_variance"
        return block
    statistic = (
        (sharpe - sharpe_star / scale)
        * math.sqrt(len(returns) - 1)
        / math.sqrt(variance_term)
    )
    block["deflated_sharpe_probability"] = NormalDist().cdf(statistic)
    return block


def information_ratio_bar(effective: float, days: int, probability: float) -> float | None:
    """The research IR at which the deflated Sharpe probability reaches
    ``probability`` at ``effective`` trials over ``days``, for normal returns:
    √V·(E[max] + Φ⁻¹(p)). ``None`` at p = 1, which no IR reaches."""

    if probability >= 1.0:
        return None
    return null_sharpe_std(days) * (
        expected_max_sharpe(effective) + NormalDist().inv_cdf(probability)
    )


def excess_at_cost_stress(
    neutralized_excess: float,
    days: int,
    *,
    cost_stress_multiplier: float,
    slippage_bps: float,
    turnover: float,
) -> float:
    """The graduation's cost-stress reading (F5) of one span: its annualised
    neutralised excess (of the graded series) less ``(cost_stress_multiplier −
    1) × slippage_bps`` more per side on ``turnover``, the span's traded
    notional over its opening equity, annualised over its ``days`` measured
    days. The forward slice judges it; a research row reports it."""

    years = days / TRADING_DAYS_PER_YEAR
    return (
        neutralized_excess
        - (cost_stress_multiplier - 1.0) * slippage_bps * turnover * 1e-4 / years
    )


def raw_excess_at_cost_stress(
    summary: Mapping[str, object], *, cost_stress_multiplier: float
) -> float | None:
    """The holder's money against the benchmark at the cost stress, over one
    replay's span: its own equity return after every cost minus the arm's
    ``benchmark_index`` price return over the same days
    (``benchmark.excess_return``), less ``(cost_stress_multiplier − 1)`` times
    the modelled slippage on its turnover (``cost_sensitivity``: the profile's
    slippage and what one bp per side costs as a fraction of the opening
    equity). Cumulative and not compounded, like
    ``cost_sensitivity.excess_at_2x_slippage``, which it equals at a
    multiplier of 2. ``None`` when the summary carries no benchmark excess or
    no cost block."""

    benchmark = summary.get("benchmark")
    costs = summary.get("cost_sensitivity")
    if not isinstance(benchmark, Mapping) or not isinstance(costs, Mapping):
        return None
    excess = benchmark.get("excess_return")
    slippage = costs.get("slippage_bps")
    per_bp = costs.get("cost_per_bp_per_side")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (excess, slippage, per_bp)
    ):
        return None
    return float(excess) - (cost_stress_multiplier - 1.0) * float(slippage) * float(per_bp)  # type: ignore[arg-type]


def _compounded(series: object, start: str, end: str) -> float | None:
    """One stored daily series compounded over its own days in
    ``start``..``end`` (empty bounds take the whole series); ``None`` when it
    has no day there."""

    values = [
        value
        for date, value in _series_pairs(series)
        if (not start or date >= start) and (not end or date <= end)
    ]
    return compounded_return(values) if values else None


def panel_return(analysis: Mapping[str, object]) -> float | None:
    """What the zero-skill panel composite itself earned over the sidecar's
    span, after its own costs: the return of the random-name copies of the
    book's trade skeleton. ``None`` on a sidecar without a panel."""

    return _compounded(analysis.get("panel_daily"), "", "")


def holder_readings(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> dict[str, float | None]:
    """What a long-only holder of the book saw over one span, unregressed.

    The book's own return after every cost (``strategy_return``), the arm's
    benchmark price return (``benchmark_return``) and the zero-skill panel's
    return after its costs (``panel_return``), each compounded from the stored
    daily series over its days in ``start``..``end``; ``raw_excess`` is book
    minus benchmark and ``plain_selection`` book minus panel, cumulative
    differences like ``benchmark.excess_return``. ``plain_selection`` is what
    the active series says without its regression: whether the names the book
    picked beat random names on its own skeleton. A series the sidecar does not
    carry over the span reads ``None``, and so does every difference built on
    it. Empty bounds take the whole sidecar.

    One structure wherever the holder is read: the ``raw_readings`` of a
    forward or Held-out slice (:func:`slice_readings`), and of a research
    row over its span and its last two years
    (``session_tools.SessionValidations.raw_readings``).
    """

    strategy = _compounded(analysis.get("strategy_daily"), start, end)
    benchmark = _compounded(analysis.get("benchmark_daily"), start, end)
    panel = _compounded(analysis.get("panel_daily"), start, end)
    return {
        "strategy_return": strategy,
        "benchmark_return": benchmark,
        "panel_return": panel,
        "raw_excess": None if strategy is None or benchmark is None else strategy - benchmark,
        "plain_selection": None if strategy is None or panel is None else strategy - panel,
    }


# The holder's readings by name, in order: what :func:`holder_readings` returns.
HOLDER_READINGS = tuple(holder_readings({}))


def slice_readings(
    analysis: Mapping[str, object], *, start: str, end: str
) -> dict[str, object]:
    """The readings a forward or Held-out slice carries beside its statistics.

    ``plain_excess`` is the graded series' annualised mean over the days the
    regression reads, before the regression: ``neutralized_excess`` minus it
    is what the market and size loadings were credited or charged, which an
    unhedged holder never receives. ``raw_readings`` is
    :func:`holder_readings` of the slice. Both are recomputable from a stored
    sidecar, so a slice recorded before they existed reads the same numbers
    on demand. ``ValueError`` when the slice has no regression day.
    """

    start, end = _span(start, end)
    graded, _series = _graded(analysis)
    rows = _regression_rows(graded, start, end)
    if not len(rows):
        raise ValueError(f"slice {start}..{end} has no day with both regressors")
    return {
        "plain_excess": float(rows[:, 0].mean()) * TRADING_DAYS_PER_YEAR,
        "raw_readings": holder_readings(analysis, start=start, end=end),
    }


def seed_replicate_slice(
    analysis: Mapping[str, object], *, start: str, end: str
) -> dict[str, object]:
    """One seed replicate of the frozen book over one slice, unjudged.

    A replicate is the book's strategy on another training seed, replayed
    exactly as the book is. Its slice carries what the book's own slice
    reads -- the graded series' neutralised statistics and
    :func:`slice_readings` -- and no condition: forward judges the seed mean
    (:func:`seed_mean`), never a replicate on its own.
    """

    start, end = _span(start, end)
    return {
        "start": start,
        "end": end,
        **neutralized_statistics(analysis, start=start, end=end),
        **slice_readings(analysis, start=start, end=end),
    }


def seed_mean(blocks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The mean over the frozen book's slice and its replicates' slices.

    Each holder reading (``raw_readings``) and ``plain_excess`` averaged over
    the members, ``None`` where any member lacks it: a seed the replay could
    not read is never dropped from the mean. ``members`` counts the book and
    its replicates.
    """

    def mean(values: Sequence[object]) -> float | None:
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            return None
        return float(sum(values)) / len(values)  # type: ignore[arg-type]

    raw = [block.get("raw_readings") or {} for block in blocks]
    return {
        "members": len(blocks),
        "plain_excess": mean([block.get("plain_excess") for block in blocks]),
        "raw_readings": {
            name: mean([reading.get(name) for reading in raw])  # type: ignore[union-attr]
            for name in HOLDER_READINGS
        },
    }


def _with_seed_replicates(
    readings: Mapping[str, object], seed_replicates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """What a slice carries of its seed replicates: their own slices and the
    :func:`seed_mean` with the book; nothing where there are none."""

    if not seed_replicates:
        return {}
    return {
        "seed_replicates": [dict(block) for block in seed_replicates],
        "seed_mean": seed_mean([readings, *seed_replicates]),
    }


def _count(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def trial_family_statistics(
    *,
    trials: int,
    offline_trials: int = 0,
    trial_analyses: Sequence[Mapping[str, object]] = (),
    lineage_trials: int = 0,
    lineage_series: Sequence[Mapping[str, float]] = (),
) -> dict[str, object]:
    """M, its parts, ρ̄ and N_eff of an arm's trial family (:func:`freeze_gate`
    names the parts); what the gate deflates over and what the IR bar a
    full-span nominee faces is read at."""

    trials = _count(trials, "trials", 1)
    offline_trials = _count(offline_trials, "offline_trials", 0)
    lineage_trials = _count(lineage_trials, "lineage_trials", 0)
    correlation, pairs = trial_correlation(trial_analyses, lineage_series)
    total = trials + offline_trials + lineage_trials
    return {
        "trials": total,
        "host_trials": trials,
        "offline_trials": offline_trials,
        "lineage_trials": lineage_trials,
        "trial_correlation": correlation,
        "trial_correlation_pairs": pairs,
        "effective_trials": effective_trials(total, correlation),
    }


class Condition(NamedTuple):
    """One condition a stage judges on what it measured.

    ``holds(measured, thresholds)`` reads the stage's own block and the bars
    its record states; a condition that does not hold records ``reason``.
    ``requires`` names the rule that turns an optional condition on (empty:
    every arm is held to it). An optional condition that is off is neither
    judged nor stated; one that is on states the rule ``stamp`` names in the
    record's thresholds. ``code`` is the graduation criterion a forward or
    Held-out condition belongs to (docs/pipeline-design.md).
    """

    stage: str
    reason: str
    holds: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    code: str = ""
    requires: str = ""
    stamp: str = ""


def _finite(value: object) -> bool:
    """Whether one replay metric is a finite number. ``bool`` is an ``int`` in
    Python, so ``True`` would otherwise read as a total return of 1.0."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: object) -> bool:
    """A reading above zero. One that was not measured never passes."""

    return value is not None and value > 0  # type: ignore[operator]


def _probability(measured: Mapping[str, Any]) -> object:
    return measured["deflated_sharpe"]["deflated_sharpe_probability"]


def _within_cap(measured: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    return measured["mandate"]["tracking_error"] <= thresholds["tracking_error_cap"]


def _within_band(measured: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    return thresholds["beta_min"] <= measured["mandate"]["market_beta"] <= thresholds["beta_max"]


def _refused_replicate(measured: Mapping[str, Any]) -> bool:
    return any("problem" in entry for entry in measured["replicates"])


def _seed_mean_reaches_bar(measured: Mapping[str, Any]) -> bool:
    mean, bar = measured["mean_information_ratio"], measured["information_ratio_bar"]
    return mean is not None and isinstance(bar, (int, float)) and mean >= bar


_RAW = "require_raw_excess_at_cost_stress"
_PLAIN = "require_forward_plain_selection"
_SEEDS = "require_seed_replicates"
_CAP = "tracking_error_cap"
# Not a condition's: what a gate that was never read off a nominee records.
# The session reads a node that is not one of its Steps; the nominee's
# statistics could not be measured.
NOT_A_SESSION_STEP = "freeze_needs_a_step_of_this_session"
UNMEASURABLE = "freeze_unmeasurable"

# Every condition the freeze gate and the graduation verdict judge, by stage
# and, within a stage, in the order its failures are recorded. The one list of
# the reason tokens a record can carry: each stage reads its failures off it
# (:func:`judge`), and the console's criterion codes and labels, the Agent's
# facts and the tests are checked against it.
CONDITIONS: tuple[Condition, ...] = (
    # The hard nomination rules, on the nominee's replay summary
    # (``config.AcceptanceRules.evaluate``). A non-finite metric, because every
    # IEEE comparison against NaN is False, so it would otherwise pass every
    # threshold; and a research-period drawdown over the limit F4/H3 enforce
    # forward: freezing a book that already breached it spends a forward test
    # on a candidate the verdict must reject. ``sharpe`` is read only for
    # finiteness: how much edge is enough is the deflated Sharpe's question.
    Condition("nomination", "non_finite_total_return", lambda m, t: _finite(m.get("total_return"))),
    Condition("nomination", "non_finite_max_drawdown", lambda m, t: _finite(m.get("max_drawdown"))),
    Condition("nomination", "non_finite_sharpe", lambda m, t: m.get("sharpe") is None or _finite(m["sharpe"])),
    Condition("nomination", "max_drawdown_above_limit", lambda m, t: not (_finite(m.get("max_drawdown")) and abs(m["max_drawdown"]) > t["max_drawdown"])),
    # How the nominee was registered (``experiment.freeze_gate_for``).
    Condition("registration", "freeze_needs_full_span_validation", lambda m, t: m.get("span") == FULL_SPAN),
    Condition("registration", "freeze_nominee_is_control", lambda m, t: m.get("control") is not True),
    # The freeze gate, on the nominee's research period (:func:`freeze_gate`).
    Condition("freeze", "freeze_too_few_full_span_validations", lambda m, t: m["full_span_validations"] >= t["min_full_span_validations"]),
    Condition("freeze", "freeze_information_ratio_below_threshold", lambda m, t: m["information_ratio"] is not None and m["information_ratio"] >= t["min_information_ratio"]),
    Condition("freeze", "freeze_deflated_sharpe_unavailable", lambda m, t: isinstance(_probability(m), float)),
    Condition("freeze", "freeze_deflated_sharpe_below_threshold", lambda m, t: not isinstance(_probability(m), float) or not _probability(m) < t["min_deflated_sharpe_probability"]),
    Condition("freeze", "freeze_too_few_positive_years", lambda m, t: m["positive_years"] >= t["min_positive_years"]),
    Condition("freeze", "freeze_active_drawdown_exceeded", lambda m, t: m["active_max_drawdown"] <= t["active_max_drawdown"]),
    Condition("freeze", "freeze_raw_excess_not_positive_at_cost_stress", lambda m, t: _positive(m["raw_excess_at_cost_stress"]), requires=_RAW, stamp="cost_stress_multiplier"),
    Condition("freeze", "freeze_tracking_error_above_cap", _within_cap, requires=_CAP),
    Condition("freeze", "freeze_beta_outside_band", _within_band, requires=_CAP),
    # The nominee's seed replicates (``experiment._seed_replicate_gate``): a
    # named replicate that is not one, a nominee that trains a model and names
    # none, and a mean active IR over the nominee and its replicates below
    # the nominee's own bar.
    Condition("seeds", "freeze_seed_replicate_invalid", lambda m, t: not _refused_replicate(m), requires=_SEEDS, stamp=_SEEDS),
    Condition("seeds", "freeze_too_few_seed_replicates", lambda m, t: bool(m["replicates"]) or not m["trains_a_model"], requires=_SEEDS, stamp=_SEEDS),
    Condition("seeds", "freeze_seed_mean_information_ratio_below_threshold", lambda m, t: not m["replicates"] or _refused_replicate(m) or _seed_mean_reaches_bar(m), requires=_SEEDS, stamp=_SEEDS),
    # F1/H1: the frozen strategy raised during the replay, which then has no
    # slice to judge (:func:`graduation_verdict`).
    Condition("replay", "forward_strategy_error", lambda m, t: m["strategy_error"] != "forward", "F1"),
    Condition("replay", "heldout_strategy_error", lambda m, t: m["strategy_error"] != "heldout", "H1"),
    # The forward slice (:func:`forward_slice`). F8: no panel, no plain
    # selection, and an unmeasured reading never passes. F9: a book frozen
    # with nothing to replicate has no mean to judge.
    Condition("forward", "forward_lower_bound_not_positive", lambda m, t: m["lower_bound"] > 0, "F2"),
    Condition("forward", "forward_recency_negative", lambda m, t: m["recency_neutralized_excess"] >= 0, "F3"),
    Condition("forward", "forward_max_drawdown_exceeded", lambda m, t: m["max_drawdown"] <= t["max_drawdown"], "F4"),
    Condition("forward", "forward_active_drawdown_exceeded", lambda m, t: m["active_max_drawdown"] <= t["active_max_drawdown"], "F4"),
    Condition("forward", "forward_not_positive_at_cost_stress", lambda m, t: m["excess_at_cost_stress"] > 0, "F5"),
    Condition("forward", "forward_too_few_round_trips", lambda m, t: m["round_trips"] >= t["min_round_trips"], "F6"),
    Condition("forward", "forward_exposure_below_floor", lambda m, t: m["mean_gross"] >= t["min_mean_gross"], "F6"),
    Condition("forward", "forward_tracking_error_above_cap", _within_cap, "F7", requires=_CAP),
    Condition("forward", "forward_beta_outside_band", _within_band, "F7", requires=_CAP),
    Condition("forward", "forward_plain_selection_not_positive", lambda m, t: _positive(m["raw_readings"]["plain_selection"]), "F8", requires=_PLAIN, stamp=_PLAIN),
    Condition("forward", "forward_seed_mean_plain_selection_not_positive", lambda m, t: "seed_mean" not in m or _positive(m["seed_mean"]["raw_readings"]["plain_selection"]), "F9", requires=_SEEDS, stamp=_SEEDS),
    # The Held-out slice (:func:`heldout_slice`).
    Condition("heldout", "heldout_excess_below_tolerance", lambda m, t: m["neutralized_excess"] >= m["tolerance"], "H2"),
    Condition("heldout", "heldout_max_drawdown_exceeded", lambda m, t: m["max_drawdown"] <= t["max_drawdown"], "H3"),
    Condition("heldout", "heldout_active_drawdown_exceeded", lambda m, t: m["active_max_drawdown"] <= t["active_max_drawdown"], "H3"),
    Condition("heldout", "heldout_exposure_below_floor", lambda m, t: m["mean_gross"] >= t["min_mean_gross"], "H4"),
)


def _on(condition: Condition, rules: AcceptanceRules | None) -> bool:
    """Whether ``rules`` hold ``condition``: always, a switch that is on, or
    a tracking mandate that is set."""

    if not condition.requires:
        return True
    rule = getattr(rules, condition.requires)
    return rule is not None and rule is not False


def stamps(stage: str, rules: AcceptanceRules) -> dict[str, object]:
    """What the optional conditions of ``stage`` that ``rules`` turn on state
    in a record's thresholds, so a record judged without one reads as it did
    before the condition existed."""

    return {
        condition.stamp: getattr(rules, condition.stamp)
        for condition in CONDITIONS
        if condition.stage == stage and condition.stamp and _on(condition, rules)
    }


def judge(
    stage: str,
    measured: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    rules: AcceptanceRules | None = None,
) -> list[str]:
    """The reasons of the conditions of ``stage`` that ``measured`` fails, in
    the table's order: the one path every stage's failures are read by.
    ``rules`` decide which optional conditions are on; a stage that has none
    is judged without them."""

    return [
        condition.reason
        for condition in CONDITIONS
        if condition.stage == stage
        and _on(condition, rules)
        and not condition.holds(measured, thresholds)
    ]


def freeze_gate(
    analysis: Mapping[str, object],
    *,
    rules: AcceptanceRules,
    trials: int,
    full_span_validations: int,
    offline_trials: int = 0,
    trial_analyses: Sequence[Mapping[str, object]] = (),
    lineage_trials: int = 0,
    lineage_series: Sequence[Mapping[str, float]] = (),
    years: Sequence[tuple[str, str]] = (),
    summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze gate of one nominee under the arm's ``rules``
    (docs/pipeline-design.md).

    ``analysis`` is the nominee's full-span validation sidecar, read whole.
    The deflated Sharpe deflates over the arm's trial family
    (:func:`trial_family_statistics`): ``trials`` distinct non-control
    strategies validated anywhere in the arm (the nominee among them), the
    ``offline_trials`` its batches declared screening offline and the
    ``lineage_trials`` of the earlier arms it was created to inherit, M in all,
    counted at their effective number ρ̄ + (1 − ρ̄)·M, where ρ̄ is
    :func:`trial_correlation` over ``trial_analyses`` (one sidecar per
    trial) and ``lineage_series`` (one reduced series per measurable lineage
    trial). The dispersion √V is the zero-skill sampling error
    of an IR over the nominee's own measured days (:func:`null_sharpe_std`), so
    neither controls nor near-copies of the nominee move the bar through it.
    ``information_ratio_bar`` is the research IR at which the probability
    reaches ``rules.min_dsr_probability`` for normal returns,
    √V·(E[max] + Φ⁻¹(p)); the nominee's own skew and kurtosis move the applied
    bar by hundredths.

    ``full_span_validations`` counts the arm's measurable full-span
    validations, controls included; ``years`` are the research years'
    ``(start, end)`` bounds. On the graded series the gate asks for an IR of at
    least ``min_active_ir``, a deflated Sharpe probability of at least
    ``min_dsr_probability``, at least ``min_full_span_validations`` full-span
    validations, a positive neutralised excess in ``min_positive_year_share``
    of the research years, and a drawdown within ``active_max_drawdown``; on
    the strategy's own series, for the tracking mandate when one is set. With
    ``require_raw_excess_at_cost_stress`` it also asks the holder's money to
    beat the benchmark: :func:`raw_excess_at_cost_stress` of the nominee's
    ``summary`` at ``cost_stress_multiplier`` above zero; only then does the
    record carry that reading and the multiplier, so the gate of an arm whose
    rules lack the condition reads exactly as before. The
    equity drawdown is the caller's hard nomination rule
    (``config.AcceptanceRules.evaluate``).
    """

    family = trial_family_statistics(
        trials=trials,
        offline_trials=offline_trials,
        trial_analyses=trial_analyses,
        lineage_trials=lineage_trials,
        lineage_series=lineage_series,
    )
    full_span_validations = _count(full_span_validations, "full_span_validations", 0)
    graded, series = _graded(analysis)
    statistics, _rows, neutral = _measured(graded, "", "")
    effective = float(family.pop("effective_trials"))  # type: ignore[arg-type]
    dispersion = null_sharpe_std(int(statistics["days"]))
    dsr = {
        **family,
        **deflated_sharpe(
            observed_sharpe=statistics["information_ratio"],
            effective_trials=effective,
            trial_sharpe_std=dispersion,
            returns=neutral,
        ),
        "information_ratio_bar": information_ratio_bar(
            effective, int(statistics["days"]), rules.min_dsr_probability
        ),
    }
    year_excess = [
        window_neutralized_excess(graded, start=start, end=end)
        for start, end in (_span(*year) for year in years)
    ]
    raw: dict[str, object] = {}
    if rules.require_raw_excess_at_cost_stress:
        if summary is None:
            raise ValueError("the raw cost-stress condition needs the nominee's summary")
        raw["raw_excess_at_cost_stress"] = raw_excess_at_cost_stress(
            summary, cost_stress_multiplier=rules.cost_stress_multiplier
        )
    measured = {
        "series": series,
        **statistics,
        "year_neutralized_excess": year_excess,
        "positive_years": sum(1 for value in year_excess if value is not None and value > 0),
        "active_max_drawdown": _max_slice_drawdown(graded, "", ""),
        "mandate": _mandate(analysis, "", ""),
        "full_span_validations": full_span_validations,
        "deflated_sharpe": dsr,
        **raw,
    }
    thresholds = {
        "min_information_ratio": rules.min_active_ir,
        "min_deflated_sharpe_probability": rules.min_dsr_probability,
        "min_full_span_validations": rules.min_full_span_validations,
        "min_positive_years": math.ceil(rules.min_positive_year_share * len(year_excess)),
        "research_years": len(year_excess),
        "active_max_drawdown": rules.active_max_drawdown,
        **rules.mandate,
        "panel_draws": PANEL_DRAWS,
        **stamps("freeze", rules),
    }
    reasons = judge("freeze", measured, thresholds, rules)
    return {"passed": not reasons, "reasons": reasons, **measured, "thresholds": thresholds}


# Refit the draws in batches holding at most this many bytes of resampled rows.
# Every reduction in ``_fit`` runs within one draw, so the batched bound is the
# same float as a single batch; only the peak footprint differs. It has to: one
# batch of all 2,000 draws is a ~12 MB resample plus ~25 MB of elementwise
# temporaries, and blocks that large are handed back to the kernel when freed,
# so every verdict faults them in again. On a loaded host that fault path costs
# far more than the arithmetic it feeds -- 30 forward bootstraps measured 0.9 s
# of user time against 108 s of system time. Staying under glibc's default
# 128 KiB mmap/trim threshold keeps the batch inside the allocator's arena,
# where it is reused instead: the same 30 bootstraps then take 0.7 s in total.
_BOOTSTRAP_BATCH_BYTES = 128 * 1024


def _bootstrap_lower_bound(rows: np.ndarray, seed_key: str, *, confidence: float) -> float:
    """One-sided ``confidence`` lower bound of the annualised intercept.

    Moving-block bootstrap: ``BOOTSTRAP_DRAWS`` resamples of whole rows in
    blocks of ``BOOTSTRAP_BLOCK_DAYS`` consecutive days, the regression refit
    on each, the bound read as the percentile. The generator is seeded from a
    SHA-256 of ``seed_key`` (the frozen artifact id), so one artifact always
    gets the same bound. The refits run in batches of ``_BOOTSTRAP_BATCH_BYTES``
    worth of draws, which does not move the bound.
    """

    days = len(rows)
    if days < BOOTSTRAP_BLOCK_DAYS:
        raise ValueError(
            f"forward slice has {days} measured days, fewer than one bootstrap block"
        )
    if not seed_key:
        raise ValueError("seed_key must be a non-empty artifact id")
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    blocks = -(-days // BOOTSTRAP_BLOCK_DAYS)
    starts = generator.integers(
        0, days - BOOTSTRAP_BLOCK_DAYS + 1, size=(BOOTSTRAP_DRAWS, blocks)
    )
    offsets = np.arange(BOOTSTRAP_BLOCK_DAYS)
    batch = max(1, _BOOTSTRAP_BATCH_BYTES // (days * rows.shape[1] * rows.itemsize))
    intercepts = np.empty(BOOTSTRAP_DRAWS)
    for first in range(0, BOOTSTRAP_DRAWS, batch):
        drawn = starts[first : first + batch]
        index = (drawn[:, :, None] + offsets).reshape(len(drawn), -1)[:, :days]
        intercepts[first : first + len(drawn)] = _fit(rows[index])[0]
    return (
        float(np.quantile(intercepts, 1.0 - confidence)) * TRADING_DAYS_PER_YEAR
    )


def _max_slice_drawdown(analysis: Mapping[str, object], start: str, end: str) -> float:
    """Peak-to-trough loss inside the slice, measured from the equity it opened at."""

    return compounded_drawdown(
        value for _date_text, value in _strategy_returns(analysis, start, end)
    )


def _month_index(date: str) -> int:
    return int(date[:4]) * 12 + int(date[4:6]) - 1


def forward_slice(
    analysis: Mapping[str, object],
    *,
    rules: AcceptanceRules,
    start: str,
    end: str,
    seed_key: str,
    slippage_bps: float,
    turnover: float,
    round_trips: int,
    mean_gross: float,
    seed_replicates: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Statistics and failed conditions F2–F9 of the forward slice under the
    arm's ``rules`` (docs/pipeline-design.md, the graduation verdict).

    The lower bound, the recency window, the cost stress and
    ``active_max_drawdown`` are read off the graded series; ``max_drawdown``
    limits the equity itself and the tracking mandate, when one is set, the
    strategy's own tracking error and beta over the slice. With
    ``require_forward_plain_selection`` (F8) the book's compounded return over
    the slice, after every cost, must beat its zero-skill panel's with no
    regression (``raw_readings.plain_selection`` above zero); only then do the
    thresholds name the condition, so a slice judged without it reads its
    conditions as before. The holder's readings (:func:`slice_readings`) ride
    on every slice either way. ``require_seed_replicates`` (F9) is stamped
    the same way, by the arm's rule alone: ``seed_replicates`` are the slices
    (:func:`seed_replicate_slice`) of the seed replicates the freeze
    registered, replayed like the book, and with any the slice carries them
    and their :func:`seed_mean` with the book, whose ``plain_selection`` must
    be above zero; a book frozen with nothing to replicate has no mean to
    judge. Replicates under rules without the condition are refused. Every
    other condition reads the book alone.
    ``start``/``end`` are the slice's calendar bounds; the recency window is
    the last ``recency_months`` calendar months ending in ``end``'s month.
    ``turnover`` (traded notional over the slice's opening equity),
    ``round_trips`` and ``mean_gross`` are the slice's own figures from the
    replay; ``slippage_bps`` is the Broker profile's. The cost stress charges
    ``(cost_stress_multiplier − 1) × slippage_bps × turnover × 1e−4``,
    annualised over the slice's measured days, against the neutralised excess.
    F1 (strategy error) is :func:`graduation_verdict`'s.
    """

    start, end = _span(start, end)
    turnover = _non_negative(turnover, "turnover")
    mean_gross = _non_negative(mean_gross, "mean_gross")
    slippage_bps = _non_negative(slippage_bps, "slippage_bps")
    if (
        isinstance(round_trips, bool)
        or not isinstance(round_trips, int)
        or round_trips < 0
    ):
        raise ValueError(
            f"round_trips must be a non-negative integer, got {round_trips!r}"
        )
    graded, series = _graded(analysis)
    statistics, rows, _neutral = _measured(graded, start, end)
    lower_bound = _bootstrap_lower_bound(rows, seed_key, confidence=rules.forward_confidence)
    recency_month = _month_index(end) - (rules.recency_months - 1)
    recency_start = f"{recency_month // 12:04d}{recency_month % 12 + 1:02d}01"
    recency_excess = window_neutralized_excess(
        graded, start=max(start, recency_start), end=end
    )
    if recency_excess is None:
        raise ValueError(
            f"neutralised excess of the recency window from {recency_start} is not measurable"
        )
    drawdown = _max_slice_drawdown(analysis, start, end)
    active_drawdown = _max_slice_drawdown(graded, start, end)
    mandate = _mandate(analysis, start, end)
    stressed_excess = excess_at_cost_stress(
        statistics["neutralized_excess"],
        len(rows),
        cost_stress_multiplier=rules.cost_stress_multiplier,
        slippage_bps=slippage_bps,
        turnover=turnover,
    )
    months = _month_index(end) - _month_index(start) + 1
    readings = slice_readings(analysis, start=start, end=end)
    if seed_replicates and not rules.require_seed_replicates:
        raise ValueError("seed replicates were given under rules that hold no seed condition")
    measured = {
        "start": start,
        "end": end,
        "series": series,
        **statistics,
        **readings,
        **_with_seed_replicates(readings, seed_replicates),
        "lower_bound": lower_bound,
        "recency_start": recency_start,
        "recency_neutralized_excess": recency_excess,
        "max_drawdown": drawdown,
        "active_max_drawdown": active_drawdown,
        "mandate": mandate,
        "excess_at_cost_stress": stressed_excess,
        "turnover": turnover,
        "round_trips": round_trips,
        "mean_gross": mean_gross,
    }
    thresholds = {
        "forward_confidence": rules.forward_confidence,
        "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "recency_months": rules.recency_months,
        "max_drawdown": rules.max_drawdown,
        "active_max_drawdown": rules.active_max_drawdown,
        **rules.mandate,
        "panel_draws": PANEL_DRAWS,
        "cost_stress_multiplier": rules.cost_stress_multiplier,
        "min_round_trips": rules.min_round_trips_per_month * months,
        "min_mean_gross": rules.min_mean_gross,
        **stamps("forward", rules),
    }
    return {
        **measured,
        "reasons": judge("forward", measured, thresholds, rules),
        "thresholds": thresholds,
    }


def heldout_slice(
    analysis: Mapping[str, object],
    *,
    rules: AcceptanceRules,
    start: str,
    end: str,
    forward_tracking_error: float,
    mean_gross: float,
    seed_replicates: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Statistics and failed conditions H2–H4 of the Held-out slice under the
    arm's ``rules`` (docs/pipeline-design.md, the graduation verdict).

    Non-catastrophic only: the graded neutralised excess must be at least
    −``heldout_tolerance_z`` × ``forward_tracking_error`` / √(measured years),
    the equity drawdown within ``max_drawdown``, the graded series' drawdown
    within ``active_max_drawdown`` and the mean gross exposure at least
    ``min_mean_gross``. H1 (strategy error) is :func:`graduation_verdict`'s.
    The holder's readings
    (:func:`slice_readings`) are reported, not judged; so are the seed
    replicates' slices and their :func:`seed_mean` with the book, which the
    slice carries as :func:`forward_slice` does.
    """

    start, end = _span(start, end)
    forward_tracking_error = _non_negative(
        forward_tracking_error, "forward_tracking_error"
    )
    mean_gross = _non_negative(mean_gross, "mean_gross")
    graded, series = _graded(analysis)
    statistics, rows, _neutral = _measured(graded, start, end)
    tolerance = (
        -rules.heldout_tolerance_z
        * forward_tracking_error
        / math.sqrt(len(rows) / TRADING_DAYS_PER_YEAR)
    )
    drawdown = _max_slice_drawdown(analysis, start, end)
    active_drawdown = _max_slice_drawdown(graded, start, end)
    readings = slice_readings(analysis, start=start, end=end)
    measured = {
        "start": start,
        "end": end,
        "series": series,
        **statistics,
        **readings,
        **_with_seed_replicates(readings, seed_replicates),
        "tolerance": tolerance,
        "max_drawdown": drawdown,
        "active_max_drawdown": active_drawdown,
        "mean_gross": mean_gross,
    }
    thresholds = {
        "heldout_tolerance_z": rules.heldout_tolerance_z,
        "max_drawdown": rules.max_drawdown,
        "active_max_drawdown": rules.active_max_drawdown,
        "min_mean_gross": rules.min_mean_gross,
    }
    return {
        **measured,
        "reasons": judge("heldout", measured, thresholds, rules),
        "thresholds": thresholds,
    }


def _signed(value: object, unit: str, *, scale: float = 100.0, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value) * scale:+.{digits}f}{unit}"


def holder_line(block: Mapping[str, Any]) -> str:
    """One slice in words, the holder's reading first: the book against the
    benchmark, then against its zero-skill panel, then the graded series
    before and after the regression that neutralises it."""

    raw = block.get("raw_readings") or {}
    return (
        f"{block['start']}..{block['end']}: book {_signed(raw.get('strategy_return'), '%')} "
        f"against benchmark {_signed(raw.get('benchmark_return'), '%')} "
        f"(raw excess {_signed(raw.get('raw_excess'), ' points')}); "
        f"zero-skill panel {_signed(raw.get('panel_return'), '%')} "
        f"(plain selection {_signed(raw.get('plain_selection'), ' points')}); "
        f"{block.get('series')} series {_signed(block.get('plain_excess'), '%/yr')} unregressed, "
        f"{_signed(block.get('neutralized_excess'), '%/yr')} neutralised "
        f"(market loading {_signed(block.get('market_beta'), '', scale=1.0, digits=2)}, "
        f"IR {_signed(block.get('information_ratio'), '', scale=1.0, digits=2)})"
    )


def graduation_verdict(
    *,
    forward: Mapping[str, Any] | None,
    heldout: Mapping[str, Any] | None,
    strategy_error: Literal["forward", "heldout"] | None = None,
) -> dict[str, object]:
    """``graduated`` iff F1–F9 and H1–H4 all hold, else ``discarded``.

    ``strategy_error`` names the slice in which the strategy raised (F1/H1).
    A replay that raised produced no result, so it comes with no slice at all.
    ``reasons`` lists every failed condition in F-then-H order; ``thresholds``
    merges the slices' own. A measured verdict also states each slice's
    :func:`holder_line`, so the ledger says in words what the holder's account
    did beside the codes that decided it.
    """

    if strategy_error not in (None, "forward", "heldout"):
        raise ValueError(
            f"strategy_error must be forward, heldout or None, got {strategy_error!r}"
        )
    measured = strategy_error is None
    if (forward is not None, heldout is not None) != (measured, measured):
        raise ValueError(f"slices given do not match strategy_error={strategy_error!r}")
    reasons = judge("replay", {"strategy_error": strategy_error}, {})
    thresholds: dict[str, object] = {}
    for block in (forward, heldout):
        if block is not None:
            reasons.extend(block["reasons"])
            thresholds.update(block["thresholds"])
    return {
        "status": "discarded" if reasons else "graduated",
        "reasons": reasons,
        "thresholds": thresholds,
        **(
            {
                "holder_line": {
                    name: holder_line(block)
                    for name, block in (("forward", forward), ("heldout", heldout))
                    if block is not None
                }
            }
            if measured
            else {}
        ),
    }
