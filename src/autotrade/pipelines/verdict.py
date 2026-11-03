"""Verdict statistics of one arm: the sealed gate, and the forward and
Held-out readings reported beside it.

Pure functions over frozen replay data (docs/pipeline-design.md). Every
statistic grades one series, the ACTIVE one: the strategy's daily return
minus the zero-skill panel composite the host drew from the strategy's own
realized book (``environment/replay/null_control.py``). What zero skill earns
in an account's own shape is a property of the window and the pool, so the
panel removes what the universe construction hands any zero-skill book (lot
affordability, boards, equal weighting, the book's own schedule and cash).

The sealed gate reads the factor residual of that series: OLS on the arm's
benchmark (market), the size and the value legs of the replay's own
cross-section (``style.FACTOR_SERIES``) removes style, and the residual
information ratio over the sealed research years is held to
``z(1 − alpha) / √years`` (:func:`residual_statistics`, :func:`residual_bar`).
It is the verdict's one performance condition (:func:`sealed_slice`,
:func:`graduation_verdict`); the other is that no frozen book raised in its
own code during the forward stage. The forward and Held-out slices are reported,
never judged: they read the series as it is -- its annualised mean is its
plain excess, its annualised standard deviation its tracking error and their
ratio its IR (``stats.plain_statistics``) -- and its residual reading, which
:func:`pooled_reading` pools across frozen strategies. The holder's readings
are not regressed either: :func:`holder_readings` compounds the book, the benchmark
and the panel as stored. The inputs are the daily series a
``style_analysis.json`` sidecar stores, sliced by ``YYYYMMDD`` dates, so the
forward and Held-out slices of one continuous replay are read from one
sidecar.

The rules are the arm's create-time rules, which every stage function takes
whole (``rules``: a ``config.AcceptanceRules``); the module constants below
are the defaults those fields take, and nothing here falls back to them. A
slice that cannot be measured (fewer than two days, a slice shorter than one
bootstrap block) raises ``ValueError``: a verdict must never be read off a
number that was not measured.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from autotrade.environment.replay.null_control import PANEL_DRAWS
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
    _series_pairs,
    active_analysis,
    factor_ols,
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
# 10 % (the frozen block's ``forward_mde``), in standard errors:
# Φ⁻¹(0.90) + Φ⁻¹(0.80) = 2.12. The reported F2 is a one-sided 20 % bound, so
# this is a deliberately conservative planning figure (about 1.26 × the 1.68
# that bound would give).
FORWARD_MDE_Z = 2.12

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
        for date, value in _series_pairs(analysis.get("strategy_daily"))
        if (not start or date >= start) and (not end or date <= end)
    ]


def _regression_rows(
    analysis: Mapping[str, object], start: str, end: str
) -> np.ndarray:
    """``(strategy, market, size, value)`` rows of every day of the slice that
    carries all three legs (``style.FACTOR_SERIES``), joined exactly as
    ``style._factor_regression`` joins them. A sidecar written before the
    value leg existed carries none and yields no row."""

    legs = [dict(_series_pairs(analysis.get(series))) for _name, series in FACTOR_SERIES]
    rows = [
        (value, *(leg[date] for leg in legs))
        for date, value in _strategy_returns(analysis, start, end)
        if all(date in leg for leg in legs)
    ]
    return np.asarray(rows, dtype=float).reshape(-1, 1 + len(FACTOR_SERIES))


def residual_bar(years: float, alpha: float) -> float:
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
    the sealed gate grades, every validation reports and the ablation
    increment is read with.

    ``active`` is the daily active series (the strategy's return minus its
    zero-skill panel composite, ``series: active``; for an increment, the
    nominee's minus its ablation's) and ``market``, ``size`` and ``value``
    the three legs over the same days. OLS of the series on an intercept and
    the legs (``style.factor_ols``) gives the three betas and the residuals;
    the residual information ratio is the intercept annualised over the
    residual standard deviation annualised, ``intercept × √244 / σ̂`` with σ̂
    on n − 4 degrees of freedom. ``years`` is days / 244 and ``bar`` the
    :func:`residual_bar` at ``alpha`` for that length. ``ValueError`` on
    series of unequal length, fewer than ``MIN_REGRESSION_DAYS`` days,
    collinear legs or a series the legs explain exactly.
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
        raise ValueError("the factor legs explain the series exactly: no residual to read")
    scale = math.sqrt(TRADING_DAYS_PER_YEAR)
    years = days / TRADING_DAYS_PER_YEAR
    return {
        "days": days,
        "years": years,
        "residual_excess": float(coefficients[0]) * TRADING_DAYS_PER_YEAR,
        "residual_tracking_error": sigma * scale,
        "residual_information_ratio": float(coefficients[0]) * scale / sigma,
        **{name: float(beta) for (name, _series), beta in zip(FACTOR_SERIES, coefficients[1:], strict=True)},
        "bar": residual_bar(years, alpha),
    }


def residual_reading(
    analysis: Mapping[str, object], *, start: str = "", end: str = "", alpha: float
) -> dict[str, object]:
    """:func:`residual_statistics` of one sidecar's graded series over a
    slice (empty bounds take the whole sidecar), with ``series`` naming it:
    the active series where the sidecar carries a panel. ``ValueError`` where
    it cannot be read (see :func:`residual_statistics`)."""

    graded, series = _graded(analysis)
    rows = _regression_rows(graded, start, end)
    return {"series": series, **residual_statistics(*rows.T, alpha=alpha)}


def ablation_increment(
    nominee: Mapping[str, object],
    ablation: Mapping[str, object],
    *,
    start: str,
    end: str,
    alpha: float,
) -> dict[str, object]:
    """What the line an ablation changes adds, over a slice of two replays.

    :func:`residual_statistics` of the frozen book's graded series minus its
    ablation's, day by day, on the three legs of the book's sidecar, and its
    t-statistic, the residual information ratio × √years. Reported, never
    judged: no condition reads it, so it carries no bar. ``ValueError`` when
    the two grade different series or replayed different days, or as
    :func:`residual_statistics` raises.
    """

    start, end = _span(start, end)
    (book, series), (control, other) = _graded(nominee), _graded(ablation)
    if series != other:
        raise ValueError(f"the book grades its {series} series and its ablation its {other} series")
    mine = dict(_strategy_returns(book, start, end))
    theirs = dict(_strategy_returns(control, start, end))
    if mine.keys() != theirs.keys():
        raise ValueError("the book and its ablation replayed different days")
    difference = {**book, "strategy_daily": [[day, value - theirs[day]] for day, value in mine.items()]}
    reading = residual_statistics(*_regression_rows(difference, start, end).T, alpha=alpha)
    del reading["bar"]
    return {
        "series": series,
        **reading,
        "t_statistic": reading["residual_information_ratio"] * math.sqrt(reading["years"]),
    }


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
    ``1 / √years`` (:func:`residual_bar`): ``1 − Φ(IR × √years)``."""

    return 1.0 - NormalDist().cdf(residual_information_ratio * math.sqrt(years))


def pooled_reading(readings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Residual readings of independent frozen strategies pooled into one.

    Each reading's residual IR is, under the null, normal with standard
    deviation ``1 / √years``, so the pool weighs each by its years: the
    pooled IR is ``Σ years × IR / Σ years``, its z-score the pooled IR ×
    √(Σ years) and its one-sided p-value :func:`null_p_value`. A reading
    that states no IR (``unmeasured``) is left out and counted. One forward
    year cannot certify a strategy; the pool of every frozen strategy's
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

    rows = _regression_rows(analysis, start, end)
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


def _graded(analysis: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    """The analysis of the series a verdict grades, and that series' name."""

    active = active_analysis(analysis)
    return (analysis, "absolute") if active is None else (active, "active")


def graded_statistics(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> dict[str, object]:
    """What every gate reads of one span's graded series.

    The graded series is the active one when the sidecar carries a zero-skill
    panel, else the strategy's own, and ``series`` names which.
    ``plain_excess`` is its annualised mean, ``tracking_error`` its
    annualised standard deviation and ``information_ratio`` their ratio
    (``None`` on a flat series), as ``stats.plain_statistics`` reads them;
    ``market_beta``, ``size_beta`` and ``value_beta`` are its loadings
    (:func:`_loadings`). An empty ``start``/``end`` takes the whole analysis.
    """

    start = _date(start, "start") if start else ""
    end = _date(end, "end") if end else ""
    graded, series = _graded(analysis)
    return {**_measured(graded, start, end)[0], "series": series}


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
    and the ``freeze`` conditions it fails, judged only where the rules set a
    mandate: the registration's one reading of the strategy's own series."""

    measured = {"mandate": _mandate(analysis, "", "")}
    return {**measured, "reasons": judge("freeze", measured, rules.mandate, rules)}


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
    the computations the gate itself runs on this sidecar: the strategy's
    return and the zero-skill panel composite as stored, the active return
    :func:`style.active_analysis` grades -- its annualised mean over its
    annualised standard deviation is the active IR -- its
    :func:`stats.compounded_path` (``active_cumulative`` is the equity minus
    1, ``active_drawdown`` the loss below the running peak, whose largest
    value is the ``active_max_drawdown``), and the three factor legs its
    loadings are read against. Only the composite of the panel is stored, so
    no draw's names can be read back. Without a panel every ``active_*``
    column is empty.
    """

    frame = pd.DataFrame(
        _series_pairs(analysis.get("strategy_daily")),
        columns=["trade_date", "strategy_return"],
    )
    for column, key in (
        ("panel_return", "panel_daily"),
        ("benchmark_return", "benchmark_daily"),
        ("size_factor_return", "size_factor_daily"),
        ("value_factor_return", "value_factor_daily"),
    ):
        frame[column] = frame["trade_date"].map(dict(_series_pairs(analysis.get(key))))
    active = active_analysis(analysis)
    if active is not None:
        # Day for day the strategy's own rows: active_analysis keeps their order.
        graded = [value for _date, value in _series_pairs(active["strategy_daily"])]
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


def seed_replicate_slice(
    analysis: Mapping[str, object], *, start: str, end: str
) -> dict[str, object]:
    """One seed replicate of the frozen book over one slice, unjudged.

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
    day by day over ``analyses``: the frozen book's sidecar, then one per seed
    replicate its freeze registered, so a book frozen with none is judged on
    its own series. Its annualised mean is the slice's ``plain_excess``, or
    with replicates ``seed_mean.plain_excess``, so the record carries the
    bound alone (:func:`bootstrap_lower_bound`, seeded by ``seed_key``). It is
    not ``raw_readings.plain_selection``, which compounds the book and the
    panel each over the span before taking their difference. A member whose
    replay graded another series (it carries no panel) or measured other days
    than the book's raises ``ValueError``: a seed is never dropped from the
    mean, nor averaged on another footing.
    """

    start, end = _span(start, end)
    graded = [_graded(analysis) for analysis in analyses]
    if any(series != graded[0][1] for _analysis, series in graded[1:]):
        raise ValueError(
            f"a seed replicate's replay graded another series than the frozen book's over {start}..{end}"
        )
    members = [dict(_strategy_returns(analysis, start, end)) for analysis, _series in graded]
    dates = list(members[0])
    if any(list(member) != dates for member in members[1:]):
        raise ValueError(
            f"a seed replicate measured other days than the frozen book over {start}..{end}"
        )
    daily = np.mean([list(member.values()) for member in members], axis=0)
    return bootstrap_lower_bound(daily, seed_key, confidence=confidence)


class Condition(NamedTuple):
    """One condition a stage judges on what it measured.

    ``holds(measured, thresholds)`` reads the stage's own block and the bars
    its record states; a condition that does not hold records ``reason``.
    ``requires`` names the rule that turns a condition on, the tracking
    mandate's cap (empty: every arm is held to it); an arm without a mandate
    is not judged on it. ``code`` is the graduation criterion a forward or
    Held-out condition belongs to (docs/pipeline-design.md).
    """

    stage: str
    reason: str
    holds: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    code: str = ""
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
# Not a condition's: what a registration that was never read off a nominee
# records. The session reads a node that is not one of its Steps; the
# nominee's statistics could not be measured.
NOT_A_SESSION_STEP = "freeze_needs_a_step_of_this_session"
UNMEASURABLE = "freeze_unmeasurable"

# Every condition the freeze registration and the graduation verdict judge,
# by stage and, within a stage, in the order its failures are recorded. The
# one list of the reason tokens a record can carry: each stage reads its
# failures off it (:func:`judge`), and the console's criterion codes and
# labels, the Agent's facts and the tests are checked against it.
CONDITIONS: tuple[Condition, ...] = (
    # The hard nomination rules, on the nominee's replay summary
    # (``config.AcceptanceRules.evaluate``). A non-finite metric, because every
    # IEEE comparison against NaN is False, so it would otherwise pass every
    # threshold; and a drawdown over the limit F4/H3 enforce forward:
    # freezing a book that already breached it spends the sealed reading on a
    # candidate the verdict must reject. ``sharpe`` is read only for
    # finiteness.
    Condition("nomination", "non_finite_total_return", lambda m, t: _finite(m.get("total_return"))),
    Condition("nomination", "non_finite_max_drawdown", lambda m, t: _finite(m.get("max_drawdown"))),
    Condition("nomination", "non_finite_sharpe", lambda m, t: m.get("sharpe") is None or _finite(m["sharpe"])),
    Condition("nomination", "max_drawdown_above_limit", lambda m, t: not (_finite(m.get("max_drawdown")) and abs(m["max_drawdown"]) > t["max_drawdown"])),
    # How the nominee was registered (``experiment.freeze_registration``).
    Condition("registration", "freeze_needs_full_span_validation", lambda m, t: m.get("span") == FULL_SPAN),
    Condition("registration", "freeze_nominee_is_control", lambda m, t: m.get("control") is not True),
    # The tracking mandate over the nominee's span, when the rules set one.
    Condition("freeze", "freeze_tracking_error_above_cap", _within_cap, requires=_CAP),
    Condition("freeze", "freeze_beta_outside_band", _within_band, requires=_CAP),
    # The nominee's seed replicates (``experiment._seed_replicates``): a named
    # replicate that is not one, and a nominee that trains a model and names
    # none, unless the nomination declared its package deterministic.
    Condition("seeds", "freeze_seed_replicate_invalid", lambda m, t: not _refused_replicate(m)),
    Condition("seeds", "freeze_too_few_seed_replicates", lambda m, t: bool(m["replicates"]) or not m["trains_a_model"] or m.get("deterministic", False)),
    # The nominee's ablation (``experiment._ablation``): named exactly where
    # the rules require one, and then the nominee with one line changed.
    Condition("ablation", "freeze_ablation_invalid", lambda m, t: "problem" not in m),
    Condition("ablation", "freeze_ablation_missing", lambda m, t: "step_id" in m or not m["required"]),
    # The sealed gate (:func:`sealed_slice`): the frozen book's residual IR
    # over the sealed years against its bar, unread when the book raised.
    Condition("sealed", "sealed_strategy_error", lambda m, t: m.get("strategy_error") is not True, "S"),
    Condition("sealed", "sealed_residual_ir_below_bar", lambda m, t: m.get("strategy_error") is True or m["residual_information_ratio"] >= m["bar"], "S"),
    # A frozen book that raised in its own code during a forward-stage replay
    # (:func:`graduation_verdict`): the frozen book's forward replay, in the
    # slice it raised in, or a seed replicate's sealed or forward replay. No
    # reading of such a book is complete, so it is never admitted.
    Condition("validity", "forward_strategy_error", lambda m, t: "forward" not in m["strategy_errors"], "F1"),
    Condition("validity", "heldout_strategy_error", lambda m, t: "heldout" not in m["strategy_errors"], "H1"),
    Condition("validity", "replicate_strategy_error", lambda m, t: "replicate" not in m["strategy_errors"], "R1"),
)


def _on(condition: Condition, rules: AcceptanceRules | None) -> bool:
    """Whether ``rules`` hold ``condition``: always, or a tracking mandate
    that is set."""

    return not condition.requires or getattr(rules, condition.requires) is not None


def judge(
    stage: str,
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
    SHA-256 of ``seed_key`` (the frozen artifact id), so one artifact always
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
    """The forward slice's readings under the arm's ``rules``: reported,
    never judged (docs/pipeline-design.md, the graduation verdict).

    The graded series is read as it is (:func:`graded_statistics`) and
    through the sealed gate's statistic (``residual_reading``, what
    :func:`pooled_reading` pools across frozen strategies). F2 is
    ``lower_bound``, the ``forward_confidence`` lower bound of its plain
    excess over the book and the seed replicates the freeze registered
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
    graded, series = _graded(analysis)
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
        "series": series,
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
            "panel_draws": PANEL_DRAWS,
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
    """The Held-out slice's readings under the arm's ``rules``: reported,
    never judged (docs/pipeline-design.md, the graduation verdict).

    The graded series' statistics and its residual reading, the equity and
    the graded series' drawdowns and the mean gross exposure, the holder's
    readings (:func:`holder_readings`), and the seed replicates' slices and
    their :func:`seed_mean` with the book, which the slice carries as
    :func:`forward_slice` does.
    """

    start, end = _span(start, end)
    mean_gross = _non_negative(mean_gross, "mean_gross")
    graded, series = _graded(analysis)
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
        "series": series,
        **statistics,
        **readings,
        **_with_seed_replicates(readings, seed_replicates, start, end),
        "residual_reading": _reported_residual(analysis, start=start, end=end, alpha=rules.alpha),
        "max_drawdown": drawdown,
        "active_max_drawdown": active_drawdown,
        "mean_gross": mean_gross,
    }


def _signed(value: object, unit: str, *, scale: float = 100.0, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value) * scale:+.{digits}f}{unit}"


def holder_line(block: Mapping[str, Any]) -> str:
    """One slice in words, the holder's reading first: the book against the
    benchmark, then against its zero-skill panel, then the graded series the
    verdict read, with its IR and its market loading."""

    raw = block.get("raw_readings") or {}
    return (
        f"{block['start']}..{block['end']}: book {_signed(raw.get('strategy_return'), '%')} "
        f"against benchmark {_signed(raw.get('benchmark_return'), '%')} "
        f"(raw excess {_signed(raw.get('raw_excess'), ' points')}); "
        f"zero-skill panel {_signed(raw.get('panel_return'), '%')} "
        f"(plain selection {_signed(raw.get('plain_selection'), ' points')}); "
        f"{block.get('series')} series {_signed(block.get('plain_excess'), '%/yr')} "
        f"(IR {_signed(block.get('information_ratio'), '', scale=1.0, digits=2)}, "
        f"market loading {_signed(block.get('market_beta'), '', scale=1.0, digits=2)})"
    )


def sealed_slice(
    analysis: Mapping[str, object],
    *,
    rules: AcceptanceRules,
    start: str,
    end: str,
    seed_replicates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]] = (),
) -> dict[str, object]:
    """The sealed gate on the frozen book's replay over the sealed years.

    The book's graded series over ``start``..``end`` read through
    :func:`residual_reading` at the rules' ``alpha``: its residual
    information ratio against the bar for the slice's length, the one
    performance condition of the verdict. The holder's readings ride beside
    it; ``seed_replicates`` (what names each and its replay's sidecar) are
    read the same way and reported, never judged.
    """

    start, end = _span(start, end)
    reading = residual_reading(analysis, start=start, end=end, alpha=rules.alpha)
    measured = {
        "start": start,
        "end": end,
        **reading,
        "raw_readings": holder_readings(analysis, start=start, end=end),
        **(
            {
                "seed_replicates": [
                    {
                        **identity,
                        **residual_reading(sidecar, start=start, end=end, alpha=rules.alpha),
                    }
                    for identity, sidecar in seed_replicates
                ]
            }
            if seed_replicates
            else {}
        ),
    }
    thresholds = {"alpha": rules.alpha, "bar": reading["bar"], "sealed_years": rules.sealed_years}
    return {**measured, "reasons": judge("sealed", measured, thresholds), "thresholds": thresholds}


def sealed_strategy_error(*, start: str, end: str, error: str) -> dict[str, object]:
    """The sealed gate on a frozen book that raised during its replay over
    the sealed years: no reading, and the gate fails."""

    start, end = _span(start, end)
    measured = {"start": start, "end": end, "strategy_error": True, "error": error}
    return {**measured, "reasons": judge("sealed", measured, {}), "thresholds": {}}


# Where a frozen book can raise in its own code during the forward stage.
STRATEGY_ERROR_PLACES = ("forward", "heldout", "replicate")


def graduation_verdict(
    *,
    sealed: Mapping[str, Any],
    forward: Mapping[str, Any] | None = None,
    heldout: Mapping[str, Any] | None = None,
    strategy_errors: Sequence[str] = (),
) -> dict[str, object]:
    """``graduated`` exactly when the sealed gate passes and no frozen book
    raised, else ``discarded``.

    ``sealed`` is the sealed gate's block (:func:`sealed_slice`, or
    :func:`sealed_strategy_error`), whose reasons come first.
    ``strategy_errors`` names where a frozen book raised in its own code
    (``STRATEGY_ERROR_PLACES``): the frozen book's forward replay in its
    ``forward`` or ``heldout`` slice, or a seed ``replicate``'s sealed or
    forward replay; each adds its ``validity`` reason. The forward and
    Held-out slices (:func:`forward_slice`, :func:`heldout_slice`) are
    reported, never judged -- a forward replay in which the frozen book
    raised has none -- and a measured verdict states each slice's
    :func:`holder_line`, so the ledger says in words what the holder's
    account did beside the codes that decided it.
    """

    unknown = sorted(set(strategy_errors) - set(STRATEGY_ERROR_PLACES))
    if unknown:
        raise ValueError(f"strategy_errors must name {STRATEGY_ERROR_PLACES}, got {unknown}")
    reasons = [
        *sealed["reasons"],
        *judge("validity", {"strategy_errors": frozenset(strategy_errors)}, {}),
    ]
    lines = {
        name: holder_line(block)
        for name, block in (("forward", forward), ("heldout", heldout))
        if block is not None
    }
    return {
        "status": "discarded" if reasons else "graduated",
        "reasons": reasons,
        "thresholds": dict(sealed["thresholds"]),
        **({"holder_line": lines} if lines else {}),
    }
