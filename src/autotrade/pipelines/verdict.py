"""Verdict statistics of one arm: freeze gate, forward verdict, Held-out rule.

Pure functions over frozen replay data (PL1 §3.3 and §4). Every return figure
is the neutralised excess of ``environment/replay/style.py``: the daily strategy
return regressed on CSI 300 and the replay size factor, the intercept annualised
over ``TRADING_DAYS_PER_YEAR``. The inputs are the three daily series a
``style_analysis.json`` sidecar stores, sliced by ``YYYYMMDD`` dates, so the
forward and Held-out slices of one continuous replay are read from one sidecar.

Point estimates come from ``style.window_neutralized_excess`` itself, so every
neutralised excess here equals the figure the replay reports for the same span.
The bootstrap needs thousands of refits, so the same normal equations run
vectorised in :func:`_fit`; one test pins it to the style figure.

Thresholds are module constants, stated once here; ``max_drawdown`` and the
cost-stress multiplier are round parameters the caller passes. A slice that
cannot be measured (too few days with both factors, collinear factors, a slice
shorter than one bootstrap block) raises ``ValueError``: a verdict must never
be read off a number that was not measured.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR, _max_drawdown
from autotrade.environment.replay.style import _series_pairs, window_neutralized_excess

# Freeze gate (PL1 §4.1).
FREEZE_MIN_DSR_PROBABILITY = 0.5
FREEZE_MIN_FULL_SPAN_VALIDATIONS = 2
# Forward verdict (PL1 §4.2).
FORWARD_CONFIDENCE = 0.80
BOOTSTRAP_BLOCK_DAYS = 20
BOOTSTRAP_DRAWS = 2_000
RECENCY_MONTHS = 6
MIN_ROUND_TRIPS_PER_MONTH = 1
MIN_MEAN_GROSS = 0.5
# Held-out (PL1 §4.3): neutralised excess no worse than this many standard
# errors, the standard error taken from the forward slice's tracking error.
HELDOUT_TOLERANCE_Z = 1.28
# Minimum detectable annualised neutralised excess at 80 % power and one-sided
# 10 % (PL1 §2.1), in standard errors.
FORWARD_MDE_Z = 2.12

_EULER_MASCHERONI = 0.5772156649015329


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
    """``(strategy, CSI 300, size)`` rows of the slice, joined exactly as
    ``style.window_neutralized_excess`` joins them."""

    benchmark = dict(_series_pairs(analysis.get("benchmark_daily")))
    size = dict(_series_pairs(analysis.get("size_factor_daily")))
    rows = [
        (value, benchmark[date], size[date])
        for date, value in _strategy_returns(analysis, start, end)
        if date in benchmark and date in size
    ]
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
        raise ValueError("CSI 300 and size factor are collinear in the slice")
    market_beta = (s22 * s1y - s12 * s2y) / determinant
    size_beta = (s11 * s2y - s12 * s1y) / determinant
    intercept = means[..., 0] - market_beta * means[..., 1] - size_beta * means[..., 2]
    return intercept, market_beta, size_beta


def _measured(
    analysis: Mapping[str, object], start: str, end: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """:func:`neutralized_statistics` of a span, with its regression rows and
    daily neutralised series (strategy − β·CSI 300 − s·size, intercept included)."""

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
    }
    return statistics, rows, neutral


def neutralized_statistics(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> dict[str, object]:
    """Neutralised excess, residual tracking error and IR of one span.

    An empty ``start``/``end`` takes the whole analysis, as in
    ``style.window_neutralized_excess``. ``tracking_error`` is the residual
    standard deviation (n − 3 degrees of freedom) annualised by
    √``TRADING_DAYS_PER_YEAR``; ``information_ratio`` is ``None`` when it is 0.
    """

    start = _date(start, "start") if start else ""
    end = _date(end, "end") if end else ""
    return _measured(analysis, start, end)[0]


def forward_mde(tracking_error: float, forward_days: int) -> float:
    """Smallest annualised neutralised excess a forward test of
    ``forward_days`` detects at 80 % power (PL1 §2.1)."""

    return (
        FORWARD_MDE_Z * tracking_error / math.sqrt(forward_days / TRADING_DAYS_PER_YEAR)
    )


def deflated_sharpe(
    *,
    observed_sharpe: float | None,
    trials: int,
    trial_sharpe_std: float | None,
    returns: np.ndarray,
) -> dict[str, object]:
    """Deflated Sharpe ratio of one selected series out of ``trials`` trials.

    Bailey & López de Prado (2014), with the trial count N and the dispersion
    √V of the trial Sharpes given separately (PL1 §3.3: N counts every revision
    validated in the arm, V is measured on full-span validations only).
    Sharpes are annualised; the formula runs per period. With γ the
    Euler-Mascheroni constant,

        SR* = √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
        P   = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ]

    with γ₃ the skew and γ₄ the (normal = 3) kurtosis of the T returns.
    ``deflated_sharpe_probability`` is ``None`` whenever it is undefined, and
    ``unavailable_reason`` names why.
    """

    scale = math.sqrt(TRADING_DAYS_PER_YEAR)
    block: dict[str, object] = {
        "deflated_sharpe_probability": None,
        "trials": trials,
        "trial_sharpe_std": trial_sharpe_std,
        "sharpe_star": None,
        "observed_sharpe": observed_sharpe,
        "return_days": len(returns),
        "return_skew": None,
        "return_kurtosis": None,
        "unavailable_reason": None,
    }
    if observed_sharpe is None:
        block["unavailable_reason"] = "no_observed_sharpe"
        return block
    if trials < 2:
        block["unavailable_reason"] = "fewer_than_two_trials"
        return block
    if trial_sharpe_std is None:
        block["unavailable_reason"] = "fewer_than_two_full_span_validations"
        return block
    normal = NormalDist()
    sharpe_star = trial_sharpe_std * (
        (1.0 - _EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / trials)
        + _EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
    )
    block["sharpe_star"] = sharpe_star
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
    block["deflated_sharpe_probability"] = normal.cdf(statistic)
    return block


def freeze_gate(
    analysis: Mapping[str, object],
    *,
    trials: int,
    full_span_irs: Sequence[float],
) -> dict[str, object]:
    """Freeze gate of one nominee (PL1 §4.1).

    ``analysis`` is the nominee's full-span validation sidecar, read whole;
    ``trials`` counts the distinct revisions with a completed validation
    anywhere in the arm; ``full_span_irs`` holds the neutralised IR of every
    full-span validation in the arm, the nominee's included. The gate passes
    when there are at least two of them and the deflated Sharpe probability of
    the nominee's IR, over its daily neutralised series, is at least 0.5.
    """

    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError(f"trials must be a positive integer, got {trials!r}")
    irs = [float(value) for value in full_span_irs]
    if not all(math.isfinite(value) for value in irs):
        raise ValueError("full_span_irs must all be finite")
    statistics, _rows, neutral = _measured(analysis, "", "")
    dsr = deflated_sharpe(
        observed_sharpe=statistics["information_ratio"],
        trials=trials,
        trial_sharpe_std=float(np.std(irs, ddof=1)) if len(irs) >= 2 else None,
        returns=neutral,
    )
    reasons: list[str] = []
    if len(irs) < FREEZE_MIN_FULL_SPAN_VALIDATIONS:
        reasons.append("freeze_too_few_full_span_validations")
    probability = dsr["deflated_sharpe_probability"]
    if not isinstance(probability, float):
        reasons.append("freeze_deflated_sharpe_unavailable")
    elif probability < FREEZE_MIN_DSR_PROBABILITY:
        reasons.append("freeze_deflated_sharpe_below_threshold")
    return {
        "passed": not reasons,
        "reasons": reasons,
        **statistics,
        "full_span_validations": len(irs),
        "deflated_sharpe": dsr,
        "thresholds": {
            "min_deflated_sharpe_probability": FREEZE_MIN_DSR_PROBABILITY,
            "min_full_span_validations": FREEZE_MIN_FULL_SPAN_VALIDATIONS,
        },
    }


def _bootstrap_lower_bound(rows: np.ndarray, seed_key: str) -> float:
    """One-sided ``FORWARD_CONFIDENCE`` lower bound of the annualised intercept.

    Moving-block bootstrap: ``BOOTSTRAP_DRAWS`` resamples of whole rows in
    blocks of ``BOOTSTRAP_BLOCK_DAYS`` consecutive days, the regression refit
    on each, the bound read as the percentile. The generator is seeded from a
    SHA-256 of ``seed_key`` (the frozen artifact id), so one artifact always
    gets the same bound.
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
    index = (starts[:, :, None] + np.arange(BOOTSTRAP_BLOCK_DAYS)).reshape(
        BOOTSTRAP_DRAWS, -1
    )[:, :days]
    intercepts, _market, _size = _fit(rows[index])
    return (
        float(np.quantile(intercepts, 1.0 - FORWARD_CONFIDENCE)) * TRADING_DAYS_PER_YEAR
    )


def _max_slice_drawdown(analysis: Mapping[str, object], start: str, end: str) -> float:
    """Peak-to-trough loss inside the slice, measured from the equity it opened at."""

    returns = [value for _date_text, value in _strategy_returns(analysis, start, end)]
    return _max_drawdown(
        1.0, np.cumprod(1.0 + np.asarray(returns, dtype=float)).tolist()
    )


def _month_index(date: str) -> int:
    return int(date[:4]) * 12 + int(date[4:6]) - 1


def forward_slice(
    analysis: Mapping[str, object],
    *,
    start: str,
    end: str,
    seed_key: str,
    max_drawdown: float,
    cost_stress_multiplier: float,
    slippage_bps: float,
    turnover: float,
    round_trips: int,
    mean_gross: float,
) -> dict[str, object]:
    """Statistics and failed conditions F2–F6 of the forward slice (PL1 §4.2).

    ``start``/``end`` are the slice's calendar bounds; the recency window is
    the last ``RECENCY_MONTHS`` calendar months ending in ``end``'s month.
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
    statistics, rows, _neutral = _measured(analysis, start, end)
    lower_bound = _bootstrap_lower_bound(rows, seed_key)
    recency_month = _month_index(end) - (RECENCY_MONTHS - 1)
    recency_start = f"{recency_month // 12:04d}{recency_month % 12 + 1:02d}01"
    recency_excess = window_neutralized_excess(
        analysis, start=max(start, recency_start), end=end
    )
    if recency_excess is None:
        raise ValueError(
            f"neutralised excess of the recency window from {recency_start} is not measurable"
        )
    drawdown = _max_slice_drawdown(analysis, start, end)
    years = len(rows) / TRADING_DAYS_PER_YEAR
    stressed_excess = (
        statistics["neutralized_excess"]
        - (cost_stress_multiplier - 1.0) * slippage_bps * turnover * 1e-4 / years
    )
    months = _month_index(end) - _month_index(start) + 1
    min_round_trips = MIN_ROUND_TRIPS_PER_MONTH * months

    reasons: list[str] = []
    if not lower_bound > 0:
        reasons.append("forward_lower_bound_not_positive")
    if not recency_excess >= 0:
        reasons.append("forward_recency_negative")
    if not drawdown <= max_drawdown:
        reasons.append("forward_max_drawdown_exceeded")
    if not stressed_excess > 0:
        reasons.append("forward_not_positive_at_cost_stress")
    if round_trips < min_round_trips:
        reasons.append("forward_too_few_round_trips")
    if not mean_gross >= MIN_MEAN_GROSS:
        reasons.append("forward_exposure_below_floor")
    return {
        "start": start,
        "end": end,
        **statistics,
        "lower_bound": lower_bound,
        "recency_start": recency_start,
        "recency_neutralized_excess": recency_excess,
        "max_drawdown": drawdown,
        "excess_at_cost_stress": stressed_excess,
        "turnover": turnover,
        "round_trips": round_trips,
        "mean_gross": mean_gross,
        "reasons": reasons,
        "thresholds": {
            "forward_confidence": FORWARD_CONFIDENCE,
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "recency_months": RECENCY_MONTHS,
            "max_drawdown": max_drawdown,
            "cost_stress_multiplier": cost_stress_multiplier,
            "min_round_trips": min_round_trips,
            "min_mean_gross": MIN_MEAN_GROSS,
        },
    }


def heldout_slice(
    analysis: Mapping[str, object],
    *,
    start: str,
    end: str,
    forward_tracking_error: float,
    max_drawdown: float,
    mean_gross: float,
) -> dict[str, object]:
    """Statistics and failed conditions H2–H4 of the Held-out slice (PL1 §4.3).

    Non-catastrophic only: the neutralised excess must be at least
    −``HELDOUT_TOLERANCE_Z`` × ``forward_tracking_error`` / √(measured years),
    the drawdown within ``max_drawdown`` and the mean gross exposure at least
    ``MIN_MEAN_GROSS``. H1 (strategy error) is :func:`graduation_verdict`'s.
    """

    start, end = _span(start, end)
    forward_tracking_error = _non_negative(
        forward_tracking_error, "forward_tracking_error"
    )
    mean_gross = _non_negative(mean_gross, "mean_gross")
    statistics, rows, _neutral = _measured(analysis, start, end)
    tolerance = (
        -HELDOUT_TOLERANCE_Z
        * forward_tracking_error
        / math.sqrt(len(rows) / TRADING_DAYS_PER_YEAR)
    )
    drawdown = _max_slice_drawdown(analysis, start, end)

    reasons: list[str] = []
    if not statistics["neutralized_excess"] >= tolerance:
        reasons.append("heldout_excess_below_tolerance")
    if not drawdown <= max_drawdown:
        reasons.append("heldout_max_drawdown_exceeded")
    if not mean_gross >= MIN_MEAN_GROSS:
        reasons.append("heldout_exposure_below_floor")
    return {
        "start": start,
        "end": end,
        **statistics,
        "tolerance": tolerance,
        "max_drawdown": drawdown,
        "mean_gross": mean_gross,
        "reasons": reasons,
        "thresholds": {
            "heldout_tolerance_z": HELDOUT_TOLERANCE_Z,
            "max_drawdown": max_drawdown,
            "min_mean_gross": MIN_MEAN_GROSS,
        },
    }


def graduation_verdict(
    *,
    forward: Mapping[str, Any] | None,
    heldout: Mapping[str, Any] | None,
    strategy_error: Literal["forward", "heldout"] | None = None,
) -> dict[str, object]:
    """``graduated`` iff F1–F6 and H1–H4 all hold, else ``discarded``.

    ``strategy_error`` names the slice in which the strategy raised: the
    replay stopped there, so a forward error comes with no slice at all and a
    Held-out error with the forward slice only. ``reasons`` lists every failed
    condition in F-then-H order; ``thresholds`` merges the slices' own.
    """

    expected = {
        None: (True, True),
        "forward": (False, False),
        "heldout": (True, False),
    }
    if strategy_error not in expected:
        raise ValueError(
            f"strategy_error must be forward, heldout or None, got {strategy_error!r}"
        )
    if (forward is not None, heldout is not None) != expected[strategy_error]:
        raise ValueError(f"slices given do not match strategy_error={strategy_error!r}")
    reasons: list[str] = []
    thresholds: dict[str, object] = {}
    if strategy_error == "forward":
        reasons.append("forward_strategy_error")
    for block in (forward, heldout):
        if block is not None:
            reasons.extend(block["reasons"])
            thresholds.update(block["thresholds"])
    if strategy_error == "heldout":
        reasons.append("heldout_strategy_error")
    return {
        "status": "discarded" if reasons else "graduated",
        "reasons": reasons,
        "thresholds": thresholds,
    }
