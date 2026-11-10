"""Readout statistics of one experiment: the residual-IR significance test, the forward
and Held-out readings, and the registration conditions.

Pure functions over frozen replay data (docs/PIPELINE_DESIGN.md). Every
statistic grades one series, the ACTIVE one: the strategy's daily return
minus the random-baseline composite the host drew from the strategy's own
realized book (``environment/replay/null_control.py``). What zero skill earns
in an account's own shape is a property of the window and the pool, so the
random baseline removes what the universe construction hands any zero-skill book (lot
affordability, boards, equal weighting, the book's own schedule and cash).

The significance test reads the factor residual of that series: OLS on the experiment's
benchmark (market), the size and the value legs of the replay's own
cross-section (``style.FACTOR_SERIES``) removes style, and the residual
information ratio is read against ``z(1 − alpha) / √years`` for the
series' length (:func:`residual_statistics`, :func:`residual_threshold`), with
its one-sided null p-value (:func:`null_p_value`). The validation record is
read this way (``validation_record.record_significance``), and so is every
forward and Held-out slice. Nothing is judged: the slices read the
series as it is -- its annualised mean is its
plain excess, its annualised standard deviation its tracking error and their
ratio its IR (``stats.plain_statistics``) -- and its residual reading, which
:func:`pooled_reading` pools across registered strategies. The holder's readings
are not regressed either: :func:`holder_readings` compounds the book, the benchmark
and the random baseline as stored. The inputs are the daily series a
``style_analysis.json`` sidecar stores, sliced by ``YYYYMMDD`` dates, so the
forward and Held-out slices of one continuous replay are read from one
sidecar.

The rules are the experiment's create-time rules, which every stage function takes
whole (``rules``: a ``config.AcceptanceRules``); the module constants below
are the defaults those fields take, and nothing here falls back to them. A
slice that cannot be measured (fewer than two days, a slice shorter than one
bootstrap block) raises ``ValueError``: a reading is never recorded off a
number that was not measured. The registration conditions
(:data:`CONDITIONS`, :func:`judge`) are the one place something is judged:
whether a Step may be tested and registered at all.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from autotrade.environment.replay.null_control import BASELINE_DRAWS
from autotrade.environment.replay.stats import (
    TRADING_DAYS_PER_YEAR,
    compounded_drawdown,
    compounded_path,
    compounded_return,
    plain_statistics,
)
from autotrade.environment.replay.style import (
    FACTOR_SERIES,
    MIN_REGRESSION_DAYS,
    active_analysis,
    factor_ols,
    series_pairs,
)

from .calendar import FULL_SPAN

if TYPE_CHECKING:
    # ``config`` reads this module's constants for its defaults.
    from .config import AcceptanceRules

# The forward slice's block-bootstrap lower bound (reported, never judged).
FORWARD_CONFIDENCE = 0.80
BOOTSTRAP_BLOCK_DAYS = 20
BOOTSTRAP_DRAWS = 2_000
# Minimum detectable annualised plain excess at 80 % power and one-sided
# 10 % (the registered block's ``forward_mde``), in standard errors:
# Φ⁻¹(0.90) + Φ⁻¹(0.80) = 2.12. The reported F2 is a one-sided 20 % bound, so
# this is a deliberately conservative planning figure (about 1.26 × the 1.68
# that bound would give).
FORWARD_MDE_Z = 2.12

# The columns of one validation's daily graded series (:func:`active_daily`).
ACTIVE_DAILY_COLUMNS = (
    "trade_date",
    "strategy_return",
    "baseline_return",
    "active_return",
    "active_cumulative",
    "active_drawdown",
    "benchmark_return",
    "size_factor_return",
    "value_factor_return",
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
        for date, value in series_pairs(analysis.get("strategy_daily"))
        if (not start or date >= start) and (not end or date <= end)
    ]


def _regression_rows(
    analysis: Mapping[str, object], start: str, end: str
) -> tuple[np.ndarray, int]:
    """``(strategy, market, size, value)`` rows of every day of the slice that
    carries all three legs (``style.FACTOR_SERIES``), joined exactly as
    ``style._factor_regression`` joins them, and how many such days were
    dropped for a non-finite value: a NaN is never read as a figure. A day
    missing a leg yields no row."""

    legs = [dict(series_pairs(analysis.get(series))) for _name, series in FACTOR_SERIES]
    rows = np.asarray(
        [
            (value, *(leg[date] for leg in legs))
            for date, value in _strategy_returns(analysis, start, end)
            if all(date in leg for leg in legs)
        ],
        dtype=float,
    ).reshape(-1, 1 + len(FACTOR_SERIES))
    finite = np.isfinite(rows).all(axis=1)
    return rows[finite], int((~finite).sum())


class ExactFit(ValueError):
    """The factor legs explain a series exactly: no residual to read. The
    series itself is degenerate -- a book that stayed in cash against an
    empty random baseline -- rather than its data missing."""


def residual_threshold(years: float, alpha: float) -> float:
    """The residual IR a reading over ``years`` must reach to be significant
    at the one-sided level ``alpha`` under the null of zero residual skill:
    ``z(1 − alpha) / √years``. A null reading's residual IR is normal with
    standard deviation ``1 / √years`` (an annualised IR read off ``years ×
    TRADING_DAYS_PER_YEAR`` days): 1.645 for one year and 0.735 for five at
    alpha 0.05."""

    if not (isinstance(years, (int, float)) and years > 0):
        raise ValueError(f"years must be positive, got {years!r}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return NormalDist().inv_cdf(1.0 - alpha) / math.sqrt(years)


def residual_statistics(
    active: Sequence[float],
    market: Sequence[float],
    size: Sequence[float],
    value: Sequence[float],
    *,
    alpha: float,
) -> dict[str, float]:
    """The factor-residual reading of one daily series: the one statistic
    the validation record, every validation and every forward slice is read with.

    ``active`` is the daily active series (the strategy's return minus its
    random-baseline composite, ``series: active``) and ``market``, ``size``
    and ``value`` the three legs over the same days. OLS of the series on an
    intercept and the legs (``style.factor_ols``) gives the three betas and
    the residuals;
    the residual information ratio is the intercept annualised over the
    residual standard deviation annualised, ``intercept × √244 / σ̂`` with σ̂
    on n − 4 degrees of freedom. ``years`` is days / 244 and ``threshold`` the
    :func:`residual_threshold` at ``alpha`` for that length. ``ValueError`` on
    series of unequal length, fewer than ``MIN_REGRESSION_DAYS`` days or
    collinear legs, and :class:`ExactFit` on a series the legs explain
    exactly.
    """

    y = np.asarray(active, dtype=float)
    legs = np.column_stack(
        [np.asarray(market, dtype=float), np.asarray(size, dtype=float), np.asarray(value, dtype=float)]
    )
    if legs.shape[0] != len(y):
        raise ValueError("the active series and the three legs cover different days")
    days = len(y)
    if days < MIN_REGRESSION_DAYS:
        raise ValueError(f"{days} day(s) carry every factor leg, fewer than {MIN_REGRESSION_DAYS}")
    coefficients, residual = factor_ols(y, legs)
    sigma = math.sqrt(float(residual @ residual) / (days - len(coefficients)))
    if not sigma > 0:
        raise ExactFit("the factor legs explain the series exactly: no residual to read")
    scale = math.sqrt(TRADING_DAYS_PER_YEAR)
    years = days / TRADING_DAYS_PER_YEAR
    return {
        "days": days,
        "years": years,
        "residual_excess": float(coefficients[0]) * TRADING_DAYS_PER_YEAR,
        "residual_tracking_error": sigma * scale,
        "residual_information_ratio": float(coefficients[0]) * scale / sigma,
        **{name: float(beta) for (name, _series), beta in zip(FACTOR_SERIES, coefficients[1:], strict=True)},
        "threshold": residual_threshold(years, alpha),
    }


def residual_reading(
    analysis: Mapping[str, object], *, start: str = "", end: str = "", alpha: float
) -> dict[str, object]:
    """:func:`residual_statistics` of one sidecar's active series over a
    slice (empty bounds take the whole sidecar), with ``series`` naming it
    and ``dropped_days``, the days left out for a non-finite value.
    ``ValueError`` where it cannot be read (see :func:`residual_statistics`
    and :func:`_graded`)."""

    rows, dropped = _regression_rows(_graded(analysis), start, end)
    return {"series": GRADED_SERIES, **residual_statistics(*rows.T, alpha=alpha), "dropped_days": dropped}


def _reported_residual(
    analysis: Mapping[str, object], *, start: str, end: str, alpha: float
) -> dict[str, object]:
    """:func:`residual_reading` of a reported slice, or why it could not be
    read under ``unmeasured``: a reading nothing judges never fails a stage."""

    try:
        return residual_reading(analysis, start=start, end=end, alpha=alpha)
    except ValueError as exc:
        return {"unmeasured": str(exc)}


def null_p_value(residual_information_ratio: float, years: float) -> float:
    """The one-sided p-value of a residual reading under the null of no
    residual skill, whose residual IR is normal with standard deviation
    ``1 / √years`` (:func:`residual_threshold`): ``1 − Φ(IR × √years)``."""

    return 1.0 - NormalDist().cdf(residual_information_ratio * math.sqrt(years))


def pooled_reading(readings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Residual readings of independent registered strategies pooled into one.

    Each reading's residual IR is, under the null, normal with standard
    deviation ``1 / √years``, so the pool weighs each by its years: the
    pooled IR is ``Σ years × IR / Σ years``, its z-score the pooled IR ×
    √(Σ years) and its one-sided p-value :func:`null_p_value`. A reading
    that states no IR (``unmeasured``) is left out and counted. One forward
    year cannot certify a strategy; the pool of every registered strategy's
    forward year is what the forward stage reads.
    """

    measured = [
        (float(item["residual_information_ratio"]), float(item["years"]))  # type: ignore[arg-type]
        for item in readings
        if item.get("residual_information_ratio") is not None
    ]
    years = sum(length for _ratio, length in measured)
    pooled: dict[str, object] = {"count": len(measured), "unmeasured": len(readings) - len(measured)}
    if not measured:
        return pooled
    ratio = sum(value * length for value, length in measured) / years
    return {
        **pooled,
        "years": years,
        "residual_information_ratio": ratio,
        "z": ratio * math.sqrt(years),
        "p_value": null_p_value(ratio, years),
    }


def _loadings(analysis: Mapping[str, object], start: str, end: str) -> dict[str, float] | None:
    """The factor regression of a span's ``strategy_daily`` on the market,
    size and value legs (``style.factor_ols``): its three loadings and its
    residual tracking error (n − 4 degrees of freedom, annualised by
    √``TRADING_DAYS_PER_YEAR``). ``None`` when fewer than
    ``MIN_REGRESSION_DAYS`` days carry every leg or they are collinear: the
    graded series' loadings are a diagnostic, the strategy's own what a
    tracking mandate limits."""

    rows, _dropped = _regression_rows(analysis, start, end)
    if len(rows) < MIN_REGRESSION_DAYS:
        return None
    try:
        coefficients, residual = factor_ols(rows[:, 0], rows[:, 1:])
    except ValueError:
        return None
    return {
        "tracking_error": math.sqrt(
            float(residual @ residual) / (len(rows) - len(coefficients)) * TRADING_DAYS_PER_YEAR
        ),
        **{name: float(beta) for (name, _series), beta in zip(FACTOR_SERIES, coefficients[1:], strict=True)},
    }


def _measured(
    analysis: Mapping[str, object], start: str, end: str
) -> tuple[dict[str, Any], np.ndarray]:
    """``stats.plain_statistics`` of a span's ``strategy_daily`` with its
    loadings beside them (:func:`_loadings`, ``None`` each where they cannot
    be fitted), and the daily series they read. ``ValueError`` when the span
    holds fewer than two days."""

    daily = np.asarray([value for _date, value in _strategy_returns(analysis, start, end)])
    statistics = plain_statistics(daily.tolist())
    if statistics["plain_excess"] is None:
        raise ValueError(
            f"{start or 'start'}..{end or 'end'} holds fewer than two days to measure"
        )
    loadings = _loadings(analysis, start, end) or {}
    return {
        **statistics,
        **{name: loadings.get(name) for name, _series in FACTOR_SERIES},
    }, daily


# The series every reading grades, named in each record: the strategy's daily
# return minus its random-baseline composite (``style.active_analysis``).
GRADED_SERIES = "active"


def _graded(analysis: Mapping[str, object]) -> Mapping[str, object]:
    """The analysis of the active series a reading grades. A sidecar without a
    random baseline has none, and raises ``ValueError``."""

    active = active_analysis(analysis)
    if active is None:
        raise ValueError("the replay carries no random baseline, so its active series cannot be read")
    return active


def graded_statistics(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> dict[str, object]:
    """What every gate reads of one span's graded series.

    The graded series is the active one (:func:`_graded`), and ``series``
    names it. ``plain_excess`` is its annualised mean, ``tracking_error`` its
    annualised standard deviation and ``information_ratio`` their ratio
    (``None`` on a flat series), as ``stats.plain_statistics`` reads them;
    ``market_beta``, ``size_beta`` and ``value_beta`` are its loadings
    (:func:`_loadings`). An empty ``start``/``end`` takes the whole analysis.
    """

    start = _date(start, "start") if start else ""
    end = _date(end, "end") if end else ""
    return {**_measured(_graded(analysis), start, end)[0], "series": GRADED_SERIES}


def _mandate(analysis: Mapping[str, object], start: str, end: str) -> dict[str, object]:
    """The strategy's own residual tracking error and beta against its
    benchmark over a span (:func:`_loadings`): what a tracking mandate
    limits, and what is reported, not graded, where the rules set none.
    ``None`` each where the regression cannot be fitted, which no mandate
    passes."""

    own = _loadings(analysis, start, end) or {}
    return {"tracking_error": own.get("tracking_error"), "market_beta": own.get("market_beta")}


def registered_mandate(analysis: Mapping[str, object], *, rules: AcceptanceRules) -> dict[str, object]:
    """The tracking mandate over a nominee's whole span (:func:`_mandate`)
    and the ``mandate`` conditions it fails, judged only where the rules set a
    mandate: the registration's one reading of the strategy's own series."""

    measured = {"mandate": _mandate(analysis, "", "")}
    return {**measured, "reasons": judge(ConditionStage.MANDATE, measured, rules.mandate, rules)}


def forward_mde(tracking_error: float, forward_days: int) -> float:
    """Smallest annualised plain excess a forward test of ``forward_days``
    detects at 80 % power (``FORWARD_MDE_Z``), at the research period's
    tracking error (the graded series' standard deviation)."""

    return (
        FORWARD_MDE_Z * tracking_error / math.sqrt(forward_days / TRADING_DAYS_PER_YEAR)
    )


def active_daily(analysis: Mapping[str, object]) -> pd.DataFrame:
    """One validation's graded series day by day, as the session reads it.

    One row per replayed day (``ACTIVE_DAILY_COLUMNS``), every column read off
    the computations every reading runs on this sidecar: the strategy's
    return and the random-baseline composite as stored, the active return
    :func:`style.active_analysis` grades -- its annualised mean over its
    annualised standard deviation is the active IR -- its
    :func:`stats.compounded_path` (``active_cumulative`` is the equity minus
    1, ``active_drawdown`` the loss below the running peak, whose largest
    value is the ``active_max_drawdown``), and the three factor legs its
    loadings are read against. Only the composite of the random baseline is stored, so
    no draw's names can be read back. Without a random baseline every ``active_*``
    column is empty.
    """

    frame = pd.DataFrame(
        series_pairs(analysis.get("strategy_daily")),
        columns=["trade_date", "strategy_return"],
    )
    for column, key in (
        ("baseline_return", "baseline_daily"),
        ("benchmark_return", "benchmark_daily"),
        ("size_factor_return", "size_factor_daily"),
        ("value_factor_return", "value_factor_daily"),
    ):
        frame[column] = frame["trade_date"].map(dict(series_pairs(analysis.get(key))))
    active = active_analysis(analysis)
    if active is not None:
        # Day for day the strategy's own rows: active_analysis keeps their order.
        graded = [value for _date, value in series_pairs(active["strategy_daily"])]
        path = compounded_path(graded)
        frame["active_return"] = graded
        frame["active_cumulative"] = [equity - 1.0 for equity, _drawdown in path]
        frame["active_drawdown"] = [drawdown for _equity, drawdown in path]
    return frame.reindex(columns=list(ACTIVE_DAILY_COLUMNS))


def excess_at_cost_stress(
    plain_excess: float,
    days: int,
    *,
    cost_stress_multiplier: float,
    slippage_bps: float,
    turnover: float,
) -> float:
    """The cost-stress reading of one span, which a research row reports and
    F5 reads on the forward months: the graded series'
    annualised plain excess less ``(cost_stress_multiplier − 1) ×
    slippage_bps`` more per side on ``turnover``, the span's traded notional
    over its opening equity, annualised over its ``days`` days. A research
    row reports it."""

    years = days / TRADING_DAYS_PER_YEAR
    return (
        plain_excess
        - (cost_stress_multiplier - 1.0) * slippage_bps * turnover * 1e-4 / years
    )


def summary_excess_at_cost_stress(
    statistics: Mapping[str, object],
    summary: Mapping[str, object],
    *,
    cost_stress_multiplier: float,
) -> float | None:
    """:func:`excess_at_cost_stress` of one research replay: its graded
    ``statistics`` (:func:`graded_statistics` of its span) at the slippage its
    cost block records (``cost_sensitivity.slippage_bps``) on the replay's own
    ``turnover``. ``None`` where the summary carries neither."""

    costs = summary.get("cost_sensitivity")
    slippage = costs.get("slippage_bps") if isinstance(costs, Mapping) else None
    turnover = summary.get("turnover")
    if not (_finite(slippage) and _finite(turnover)):
        return None
    return excess_at_cost_stress(
        float(statistics["plain_excess"]),  # type: ignore[arg-type]
        int(statistics["days"]),  # type: ignore[call-overload]
        cost_stress_multiplier=cost_stress_multiplier,
        slippage_bps=float(slippage),  # type: ignore[arg-type]
        turnover=float(turnover),  # type: ignore[arg-type]
    )


def _compounded(series: object, start: str, end: str) -> float | None:
    """One stored daily series compounded over its own days in
    ``start``..``end`` (empty bounds take the whole series); ``None`` when it
    has no day there."""

    values = [
        value
        for date, value in series_pairs(series)
        if (not start or date >= start) and (not end or date <= end)
    ]
    return compounded_return(values) if values else None


def baseline_return(analysis: Mapping[str, object]) -> float | None:
    """What the random-baseline composite itself earned over the sidecar's
    span, after its own costs: the return of the random-name copies of the
    book's trade skeleton. ``None`` on a sidecar without a random baseline."""

    return _compounded(analysis.get("baseline_daily"), "", "")


def holder_readings(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> dict[str, float | None]:
    """What a long-only holder of the book saw over one span, unregressed.

    The book's own return after every cost (``strategy_return``), the experiment's
    benchmark price return (``benchmark_return``) and the random baseline's
    return after its costs (``baseline_return``), each compounded from the stored
    daily series over its days in ``start``..``end``; ``raw_excess`` is book
    minus benchmark and ``plain_selection`` book minus random baseline, cumulative
    differences like ``benchmark.excess_return``. ``plain_selection`` is the
    active series compounded: whether the names the book picked beat random
    names on its own skeleton. A series the sidecar does not carry over the
    span reads ``None``, and so does every difference built on it. Empty bounds
    take the whole sidecar.

    One structure wherever the holder is read: the ``raw_readings`` of a
    forward or Held-out slice and of a seed replicate's, and of a research
    row over its span and its last two years
    (``session_tools.SessionValidations.raw_readings``).
    """

    strategy = _compounded(analysis.get("strategy_daily"), start, end)
    benchmark = _compounded(analysis.get("benchmark_daily"), start, end)
    baseline = _compounded(analysis.get("baseline_daily"), start, end)
    return {
        "strategy_return": strategy,
        "benchmark_return": benchmark,
        "baseline_return": baseline,
        "raw_excess": None if strategy is None or benchmark is None else strategy - benchmark,
        "plain_selection": None if strategy is None or baseline is None else strategy - baseline,
    }


# The holder's readings by name, in order: what :func:`holder_readings` returns.
HOLDER_READINGS = tuple(holder_readings({}))


def seed_replicate_slice(
    analysis: Mapping[str, object], *, start: str, end: str
) -> dict[str, object]:
    """One seed replicate of the registered book over one slice, unjudged.

    A replicate is the book's strategy on another training seed, replayed
    exactly as the book is. Its slice carries what the book's own slice
    reads -- the graded series' statistics and the holder's readings -- and
    no condition: F2 judges the book and its replicates as one series
    (:func:`forward_lower_bound`), never a replicate on its own.
    """

    start, end = _span(start, end)
    return {
        "start": start,
        "end": end,
        **graded_statistics(analysis, start=start, end=end),
        "raw_readings": holder_readings(analysis, start=start, end=end),
    }


def seed_mean(blocks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The mean over the registered book's slice and its replicates' slices.

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
    readings: Mapping[str, object],
    seed_replicates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    start: str,
    end: str,
) -> dict[str, object]:
    """What a slice carries of its seed replicates, each given as what names
    it and its replay's sidecar: their own slices (:func:`seed_replicate_slice`)
    and the :func:`seed_mean` with the book; nothing where there are none."""

    if not seed_replicates:
        return {}
    blocks = [
        {**identity, **seed_replicate_slice(sidecar, start=start, end=end)}
        for identity, sidecar in seed_replicates
    ]
    return {"seed_replicates": blocks, "seed_mean": seed_mean([readings, *blocks])}


def forward_lower_bound(
    analyses: Sequence[Mapping[str, object]],
    *,
    start: str,
    end: str,
    seed_key: str,
    confidence: float,
) -> float:
    """What F2 judges: the one-sided ``confidence`` lower bound of the plain
    excess of ``analyses`` together.

    The series is each sidecar's daily graded series over the slice, averaged
    day by day over ``analyses``: the registered book's sidecar, then one per seed
    replicate the registration accepted, so a book registered with none is judged on
    its own series. Its annualised mean is the slice's ``plain_excess``, or
    with replicates ``seed_mean.plain_excess``, so the record carries the
    bound alone (:func:`bootstrap_lower_bound`, seeded by ``seed_key``). It is
    not ``raw_readings.plain_selection``, which compounds the book and the
    random baseline each over the span before taking their difference. A member whose
    replay carries no random baseline or measured other days than the book's
    raises ``ValueError``: a seed is never dropped from the mean, nor averaged
    on another footing.
    """

    start, end = _span(start, end)
    members = [dict(_strategy_returns(_graded(analysis), start, end)) for analysis in analyses]
    dates = list(members[0])
    if any(list(member) != dates for member in members[1:]):
        raise ValueError(
            f"a seed replicate measured other days than the registered book over {start}..{end}"
        )
    daily = np.mean([list(member.values()) for member in members], axis=0)
    return bootstrap_lower_bound(daily, seed_key, confidence=confidence)


class ConditionStage(StrEnum):
    """The stage that judges a condition (:func:`judge`)."""

    NOMINATION = "nomination"
    REGISTRATION = "registration"
    MANDATE = "mandate"
    SEEDS = "seeds"


class ConditionReason(StrEnum):
    """Every reason token a registration records: one per condition of
    :data:`CONDITIONS`, and the two a registration records when it never read
    a nominee. The values are persisted in ledger records."""

    NON_FINITE_TOTAL_RETURN = "non_finite_total_return"
    NON_FINITE_MAX_DRAWDOWN = "non_finite_max_drawdown"
    NON_FINITE_SHARPE = "non_finite_sharpe"
    MAX_DRAWDOWN_ABOVE_LIMIT = "max_drawdown_above_limit"
    REGISTRATION_NEEDS_FULL_SPAN_VALIDATION = "registration_needs_full_span_validation"
    REGISTRATION_NOMINEE_IS_CONTROL = "registration_nominee_is_control"
    REGISTRATION_TRACKING_ERROR_ABOVE_CAP = "registration_tracking_error_above_cap"
    REGISTRATION_BETA_OUTSIDE_BAND = "registration_beta_outside_band"
    REGISTRATION_SEED_REPLICATE_INVALID = "registration_seed_replicate_invalid"
    # Not a condition's: what a registration that was never read off a
    # nominee records. The session read a node that is not one of its Steps;
    # the nominee's statistics could not be measured.
    NOT_A_SESSION_STEP = "registration_needs_a_step_of_this_session"
    UNMEASURABLE = "registration_unmeasurable"


class Condition(NamedTuple):
    """One condition a stage judges on what it measured.

    ``holds(measured, thresholds)`` reads the stage's own block and the thresholds
    its record states; a condition that does not hold records ``reason``.
    ``requires`` names the rule that turns a condition on, the tracking
    mandate's cap (empty: every experiment is held to it); an experiment without a mandate
    is not judged on it.
    """

    stage: ConditionStage
    reason: ConditionReason
    holds: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    requires: str = ""


def _finite(value: object) -> bool:
    """Whether one replay metric is a finite number. ``bool`` is an ``int`` in
    Python, so ``True`` would otherwise read as a total return of 1.0."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _within_cap(measured: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    value = measured["mandate"]["tracking_error"]
    return value is not None and value <= thresholds["tracking_error_cap"]


def _within_band(measured: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    value = measured["mandate"]["market_beta"]
    return value is not None and thresholds["beta_min"] <= value <= thresholds["beta_max"]


def _refused_replicate(measured: Mapping[str, Any]) -> bool:
    return any("problem" in entry for entry in measured["replicates"])


_CAP = "tracking_error_cap"

# Every condition the registration judges (``experiment.check_registration``,
# read at every test), by stage and, within a stage, in the order its
# failures are recorded. The one list of the reason tokens a record can
# carry: each stage reads its failures off it (:func:`judge`), and the
# console's labels, the Agent's facts and the tests are checked against it.
CONDITIONS: tuple[Condition, ...] = (
    # The hard nomination rules, on the nominee's replay summary
    # (``config.AcceptanceRules.evaluate``). A non-finite metric, because every
    # IEEE comparison against NaN is False, so it would otherwise pass every
    # threshold; and an equity drawdown over ``max_drawdown`` across the
    # nominee's span, the account limit Paper also kills a running book at.
    # It is the only drawdown this table judges: the validation record's and
    # the forward slices' drawdowns are recorded, never judged.
    # ``sharpe`` is read only for finiteness.
    Condition(ConditionStage.NOMINATION, ConditionReason.NON_FINITE_TOTAL_RETURN, lambda m, t: _finite(m.get("total_return"))),
    Condition(ConditionStage.NOMINATION, ConditionReason.NON_FINITE_MAX_DRAWDOWN, lambda m, t: _finite(m.get("max_drawdown"))),
    Condition(ConditionStage.NOMINATION, ConditionReason.NON_FINITE_SHARPE, lambda m, t: m.get("sharpe") is None or _finite(m["sharpe"])),
    Condition(ConditionStage.NOMINATION, ConditionReason.MAX_DRAWDOWN_ABOVE_LIMIT, lambda m, t: not (_finite(m.get("max_drawdown")) and abs(m["max_drawdown"]) > t["max_drawdown"])),
    # How the nominee was registered (``experiment.check_registration``).
    Condition(ConditionStage.REGISTRATION, ConditionReason.REGISTRATION_NEEDS_FULL_SPAN_VALIDATION, lambda m, t: m.get("span") == FULL_SPAN),
    Condition(ConditionStage.REGISTRATION, ConditionReason.REGISTRATION_NOMINEE_IS_CONTROL, lambda m, t: m.get("control") is not True),
    # The tracking mandate over the nominee's span, when the rules set one.
    Condition(ConditionStage.MANDATE, ConditionReason.REGISTRATION_TRACKING_ERROR_ABOVE_CAP, _within_cap, requires=_CAP),
    Condition(ConditionStage.MANDATE, ConditionReason.REGISTRATION_BETA_OUTSIDE_BAND, _within_band, requires=_CAP),
    # The nominee's seed replicates (``experiment._seed_replicates``), which
    # are optional: a named replicate that is not one.
    Condition(ConditionStage.SEEDS, ConditionReason.REGISTRATION_SEED_REPLICATE_INVALID, lambda m, t: not _refused_replicate(m)),
)


def _on(condition: Condition, rules: AcceptanceRules | None) -> bool:
    """Whether ``rules`` hold ``condition``: always, or a tracking mandate
    that is set."""

    return not condition.requires or getattr(rules, condition.requires) is not None


def judge(
    stage: ConditionStage,
    measured: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    rules: AcceptanceRules | None = None,
) -> list[str]:
    """The reasons of the conditions of ``stage`` that ``measured`` fails, in
    the table's order: the one path every stage's failures are read by.
    ``rules`` decide whether the tracking mandate's conditions are on; a stage
    that has none is judged without them."""

    return [
        condition.reason
        for condition in CONDITIONS
        if condition.stage == stage
        and _on(condition, rules)
        and not condition.holds(measured, thresholds)
    ]


# Read the draws in batches holding at most this many bytes of resampled days.
# Every mean runs within one draw, so the batched bound is the same float as a
# single batch; only the peak footprint differs. It has to: one batch of all
# 2,000 draws is megabytes of resample and index, and blocks that large are
# handed back to the kernel when freed, so every verdict faults them in again.
# On a loaded host that fault path costs far more than the arithmetic it feeds
# -- 30 forward bootstraps once measured 0.9 s of user time against 108 s of
# system time. Staying under glibc's default 128 KiB mmap/trim threshold keeps
# the batch inside the allocator's arena, where it is reused instead.
_BOOTSTRAP_BATCH_BYTES = 128 * 1024


def bootstrap_lower_bound(series: np.ndarray, seed_key: str, *, confidence: float) -> float:
    """One-sided ``confidence`` lower bound of a daily series' annualised mean.

    Moving-block bootstrap: ``BOOTSTRAP_DRAWS`` resamples of whole days in
    blocks of ``BOOTSTRAP_BLOCK_DAYS`` consecutive days, the mean read on
    each, the bound read as the percentile. The generator is seeded from a
    SHA-256 of ``seed_key`` (the registered artifact id), so one artifact always
    gets the same bound, and two series over the same days resample the same
    days. The draws run in batches of ``_BOOTSTRAP_BATCH_BYTES``, which does
    not move the bound.
    """

    days = len(series)
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
    batch = max(1, _BOOTSTRAP_BATCH_BYTES // series.nbytes)
    estimates = np.empty(BOOTSTRAP_DRAWS)
    for first in range(0, BOOTSTRAP_DRAWS, batch):
        drawn = starts[first : first + batch]
        index = (drawn[:, :, None] + offsets).reshape(len(drawn), -1)[:, :days]
        estimates[first : first + len(drawn)] = series[index].mean(axis=-1)
    return float(np.quantile(estimates, 1.0 - confidence)) * TRADING_DAYS_PER_YEAR


def _max_slice_drawdown(analysis: Mapping[str, object], start: str, end: str) -> float:
    """Peak-to-trough loss inside the slice, measured from the equity it opened at."""

    return compounded_drawdown(
        value for _date_text, value in _strategy_returns(analysis, start, end)
    )


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
    seed_replicates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]] = (),
) -> dict[str, object]:
    """The forward slice's readings under the experiment's ``rules``: reported,
    never judged (docs/PIPELINE_DESIGN.md, the readout).

    The graded series is read as it is (:func:`graded_statistics`) and
    through the residual-IR significance test (``residual_reading``, what
    :func:`pooled_reading` pools across registered strategies). F2 is
    ``lower_bound``, the ``forward_confidence`` lower bound of its plain
    excess over the book and the seed replicates the registration accepted
    together (:func:`forward_lower_bound`). ``seed_replicates`` are those
    replicates, replayed like the book, each as what names it and its
    replay's sidecar; with any, the slice also carries their own slices and
    the :func:`seed_mean` with the book. F5, the cost stress, and the active
    drawdown read the book's own graded series; the equity drawdown, the
    strategy's own tracking error and beta (``mandate``), the round trips and
    the mean gross exposure are read beside them. The holder's readings
    (:func:`holder_readings`) ride on every slice. ``start``/``end`` are the
    slice's calendar bounds. ``turnover`` (traded notional over the slice's
    opening equity), ``round_trips`` and ``mean_gross`` are the slice's own
    figures from the replay; ``slippage_bps`` is the Broker profile's. The
    cost stress charges ``(cost_stress_multiplier − 1) × slippage_bps ×
    turnover × 1e−4``, annualised over the slice's days, against the plain
    excess (:func:`excess_at_cost_stress`). ``parameters`` states what the
    readings were taken at.
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
    graded = _graded(analysis)
    statistics, _daily = _measured(graded, start, end)
    drawdown = _max_slice_drawdown(analysis, start, end)
    active_drawdown = _max_slice_drawdown(graded, start, end)
    mandate = _mandate(analysis, start, end)
    stressed_excess = excess_at_cost_stress(
        statistics["plain_excess"],
        statistics["days"],
        cost_stress_multiplier=rules.cost_stress_multiplier,
        slippage_bps=slippage_bps,
        turnover=turnover,
    )
    readings = {
        "plain_excess": statistics["plain_excess"],
        "raw_readings": holder_readings(analysis, start=start, end=end),
    }
    return {
        "start": start,
        "end": end,
        "series": GRADED_SERIES,
        **statistics,
        **readings,
        **_with_seed_replicates(readings, seed_replicates, start, end),
        "residual_reading": _reported_residual(analysis, start=start, end=end, alpha=rules.alpha),
        "lower_bound": forward_lower_bound(
            [analysis, *(sidecar for _identity, sidecar in seed_replicates)],
            start=start,
            end=end,
            seed_key=seed_key,
            confidence=rules.forward_confidence,
        ),
        "max_drawdown": drawdown,
        "active_max_drawdown": active_drawdown,
        "mandate": mandate,
        "excess_at_cost_stress": stressed_excess,
        "turnover": turnover,
        "round_trips": round_trips,
        "mean_gross": mean_gross,
        "parameters": {
            "forward_confidence": rules.forward_confidence,
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "baseline_draws": BASELINE_DRAWS,
            "cost_stress_multiplier": rules.cost_stress_multiplier,
        },
    }


def heldout_slice(
    analysis: Mapping[str, object],
    *,
    rules: AcceptanceRules,
    start: str,
    end: str,
    mean_gross: float,
    seed_replicates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]] = (),
) -> dict[str, object]:
    """The Held-out slice's readings under the experiment's ``rules``: reported,
    never judged (docs/PIPELINE_DESIGN.md, the readout).

    The graded series' statistics and its residual reading, the equity and
    the graded series' drawdowns and the mean gross exposure, the holder's
    readings (:func:`holder_readings`), and the seed replicates' slices and
    their :func:`seed_mean` with the book, which the slice carries as
    :func:`forward_slice` does.
    """

    start, end = _span(start, end)
    mean_gross = _non_negative(mean_gross, "mean_gross")
    graded = _graded(analysis)
    statistics, _daily = _measured(graded, start, end)
    drawdown = _max_slice_drawdown(analysis, start, end)
    active_drawdown = _max_slice_drawdown(graded, start, end)
    readings = {
        "plain_excess": statistics["plain_excess"],
        "raw_readings": holder_readings(analysis, start=start, end=end),
    }
    return {
        "start": start,
        "end": end,
        "series": GRADED_SERIES,
        **statistics,
        **readings,
        **_with_seed_replicates(readings, seed_replicates, start, end),
        "residual_reading": _reported_residual(analysis, start=start, end=end, alpha=rules.alpha),
        "max_drawdown": drawdown,
        "active_max_drawdown": active_drawdown,
        "mean_gross": mean_gross,
    }


