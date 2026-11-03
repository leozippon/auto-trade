"""Barra-lite style / benchmark attribution — single computation point.

All attribution math runs HOST-SIDE at replay completion, and every input is
frozen run data — never the mutable raw lake (whose history gets revised, see
the revision ledger; recomputing later from raw could disagree with what the
Agent actually saw):

- strategy daily returns: the window's own ``equity_curve``;
- holdings: the replay result's end-of-day position snapshots;
- cross-sectional style ranks: the replay slot's ``daily.parquet``;
- benchmark: the arm's ``benchmark_index`` rows of ``index_daily`` inside
  the replay slot's ``macro.parquet``;
- SW L1 industry: each replay slot's as-of ``universe.parquet``, the vintage
  the strategy's view publishes while that slot runs, so a span classifies each
  day's holdings under the membership in force on that day.

Every replay writes one ``style_analysis.json`` beside its result
(validation and forward replays alike; the forward replay runs after research
ends), which is what the console serves — the web layer performs no
attribution computation and touches no raw data.

A formal replay's sidecar also carries its zero-skill panel
(``null_control.run_null_control``): the composite's daily return beside the
strategy's own, and the active series (strategy minus composite) read as it
is (``stats.plain_statistics``), which is what ``pipelines/verdict.py``
grades.

Everything degrades to None/empty blocks when inputs are missing —
attribution is advisory and must never fail a backtest.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from autotrade.common.records import write_json_atomic
from autotrade.environment.data.contracts import (
    DEFAULT_BENCHMARK_INDEX,
    benchmark_index_label,
)
from autotrade.environment.runtime import utc_now_iso

from .stats import (
    TRADING_DAYS_PER_YEAR,
    ReplayResult,
    compounded_drawdown,
    plain_statistics,
    total_return_from_curve,
)

# The benchmark an arm takes when its create request names none, and the one a
# legacy sidecar or book that records none is read under. Every call below takes
# the arm's ``benchmark_index`` explicitly, so none can grade against this one
# by omission.
BENCHMARK_TS_CODE = DEFAULT_BENCHMARK_INDEX
BENCHMARK_LABEL = benchmark_index_label(BENCHMARK_TS_CODE)
# The fewest overlapping days the factor regression runs on: a span shorter
# than this has no loadings and no residual tracking error. The Held-out slot
# is refused below it (``pipelines.calendar.ResearchGeometry.heldout``), so a
# verdict never waits on a replay it cannot read.
MIN_REGRESSION_DAYS = 8
STYLE_ARTIFACT_NAME = "style_analysis.json"
# 2 (2026-10-28): the strategy's ``factor_regression`` block and the compact
# ``active_excess`` figures replace the regressed ``neutralized_excess`` blocks.
# 3 (2026-11-03): the value leg (``value_factor_daily``) joins the market and
# size legs, and the regression is three-factor. Every version stores the
# strategy, benchmark, size and panel series, so a reader takes any
# (``READABLE_STYLE_SCHEMA_VERSIONS``); a three-factor reading of a version
# 1 or 2 sidecar finds no value leg and is not measured.
STYLE_SCHEMA_VERSION = 3
READABLE_STYLE_SCHEMA_VERSIONS = (1, 2, STYLE_SCHEMA_VERSION)
_STYLE_COLUMNS = ("circ_mv", "pb", "turnover_rate")
# The size and value legs of the factor regression: daily spreads of the replay
# slot's OWN cross-section, equal-weighted inside each leg, the legs formed on
# the previous trading day's value -- small minus big float cap, and high
# minus low book-to-price (1 / pb, names with pb <= 0 left out). The audited
# edge read as a small-cap / low-beta tilt rather than proven alpha, and a raw
# excess return cannot tell those apart. These are not Barra or Fama-French
# factors and the result says so in ``method``.
_FACTOR_QUANTILE = 0.3
_FACTOR_MIN_NAMES = 30
# The three legs every factor regression reads, by the sidecar series that
# stores each, in the order of their betas.
FACTOR_SERIES = (
    ("market_beta", "benchmark_daily"),
    ("size_beta", "size_factor_daily"),
    ("value_beta", "value_factor_daily"),
)


def regression_method(benchmark_index: str) -> str:
    """The caliber sentence every factor loading and residual tracking error
    is computed under.

    Names the arm's own benchmark: the market leg is index-specific, the size
    leg is built from the replay's own cross-section whatever the benchmark is.
    """

    return (
        f"日度收益对{benchmark_index_label(benchmark_index)}收益、规模因子与价值因子"
        "（后两者取本回放槽全市场截面、不是本实验股票池，按前一交易日分组的 30% 等权两端之差："
        "规模为流通市值最小减最大，价值为账面市值比 1/pb 最高减最低，pb 不为正的股票不入组）"
        "的三元 OLS，给出三个载荷与残差跟踪误差"
        f"（按 {TRADING_DAYS_PER_YEAR} 个交易日年化）"
    )


def _date_text(value: object) -> str:
    return _parsed_date_text(str(value))


# Every stored series repeats the same trading days, so each distinct text is
# parsed once: a verdict or a console listing reads millions of points.
@functools.cache
def _parsed_date_text(text: str) -> str:
    try:
        return pd.Timestamp(text).strftime("%Y%m%d")
    except (TypeError, ValueError):
        return text


def slot_benchmark(
    replay_dir: Path | Sequence[Path] | None,
    *,
    benchmark_index: str,
) -> dict[str, float]:
    """Benchmark daily returns of one replay slot, or of every slot of a span.

    ``benchmark_index`` is the arm's own benchmark. Consecutive slots partition
    their rows, so a span's series is the union.
    """
    if replay_dir is None:
        return {}
    if not isinstance(replay_dir, (str, Path)):
        return {
            day: value
            for slot in replay_dir
            for day, value in slot_benchmark(slot, benchmark_index=benchmark_index).items()
        }
    path = Path(replay_dir) / "macro.parquet"
    if not path.is_file():
        return {}
    required = {"dataset", "ts_code", "trade_date", "pct_chg"}
    # A macro file without these columns carries no index rows: no benchmark
    # for this slot. Any other read failure is a damaged file and surfaces.
    if not required.issubset(pq.read_schema(path).names):
        return {}
    frame = pd.read_parquet(
        path,
        columns=list(required),
        filters=[
            ("dataset", "==", "index_daily"),
            ("ts_code", "==", benchmark_index),
        ],
    )
    rows = frame[
        frame["dataset"].astype(str).eq("index_daily")
        & frame["ts_code"].astype(str).eq(benchmark_index)
    ]
    result: dict[str, float] = {}
    for date, raw in zip(rows["trade_date"], rows["pct_chg"]):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            # macro.parquet's index_daily keeps pct_chg in percent (no snapshot
            # factor), unlike daily.parquet's already-decimal pct_chg.
            result[_date_text(date)] = value / 100.0
    return result


def slot_membership(
    slots: Sequence[Path], *, benchmark_index: str
) -> dict[str, frozenset[str]] | None:
    """The benchmark's constituents by cross-section date, over the given slots.

    ``index_weight`` rides in ``macro.parquet``: a replay slot carries the
    month-end cross-sections dated inside it and the decision view the history
    before its anchor, so a span's table is the union over both. None when no
    slot mounts the dataset -- it is a per-arm selection, and a mounted one
    writes its columns even where its window holds no row -- so an arm that
    mounts it but whose release has no section of this index in the span gets
    an empty mapping, never the unmounted answer.
    """

    columns = ["dataset", "index_code", "trade_date", "con_code"]
    result: dict[str, frozenset[str]] | None = None
    for slot in slots:
        path = Path(slot) / "macro.parquet"
        if not path.is_file() or not set(columns).issubset(pq.read_schema(path).names):
            continue
        result = {} if result is None else result
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("dataset", "==", "index_weight"),
                ("index_code", "==", benchmark_index),
            ],
        )
        for date, group in frame.groupby("trade_date"):
            result[_date_text(date)] = frozenset(group["con_code"].astype(str))
    return result


def _industry_vintages(
    universes: Sequence[tuple[str, Path]],
) -> list[tuple[str, dict[str, str]]]:
    """Each slot's first day with the SW L1 membership of its universe vintage, in day order."""

    vintages: list[tuple[str, dict[str, str]]] = []
    for day, path in universes:
        path = Path(path)
        industry: dict[str, str] = {}
        if path.is_file() and {"ts_code", "l1_name"}.issubset(pq.read_schema(path).names):
            frame = pd.read_parquet(path, columns=["ts_code", "l1_name"])
            industry = {
                str(code): str(name)
                for code, name in zip(frame["ts_code"], frame["l1_name"])
                if isinstance(name, str) and name
            }
        vintages.append((_date_text(day), industry))
    return sorted(vintages, key=lambda item: item[0])


def daily_returns_from_curve(
    curve: Sequence[Mapping[str, object]],
) -> list[tuple[str, float]]:
    rows = sorted(
        (row for row in curve if row.get("trade_date") is not None),
        key=lambda row: _date_text(row["trade_date"]),
    )
    if not rows:
        return []
    try:
        previous = float(rows[0].get("initial_equity") or 0.0)
    except (TypeError, ValueError):
        previous = 0.0
    result: list[tuple[str, float]] = []
    for row in rows:
        try:
            equity = float(row.get("equity"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(equity):
            continue
        if previous > 0:
            result.append((_date_text(row["trade_date"]), equity / previous - 1.0))
        previous = equity
    return result


def _benchmark_regression(
    strategy: list[tuple[str, float]],
    benchmark: Mapping[str, float],
) -> dict[str, object]:
    paired = [(strategy_return, benchmark[date]) for date, strategy_return in strategy if date in benchmark]
    days = len(paired)
    benchmark_total = 1.0
    for _strategy_return, benchmark_return in paired:
        benchmark_total *= 1.0 + benchmark_return
    result: dict[str, object] = {
        "available": False,
        "reason": "benchmark_unavailable" if not days else "insufficient_overlapping_days",
        "n_days": days,
        "benchmark_return": round(benchmark_total - 1.0, 6) if days else None,
        "beta": None,
        "alpha_annualized": None,
        "r2": None,
    }
    if days < MIN_REGRESSION_DAYS:
        return result
    mean_strategy = sum(value for value, _ in paired) / days
    mean_benchmark = sum(value for _, value in paired) / days
    covariance = sum(
        (strategy_return - mean_strategy) * (benchmark_return - mean_benchmark)
        for strategy_return, benchmark_return in paired
    ) / days
    variance_benchmark = sum((value - mean_benchmark) ** 2 for _, value in paired) / days
    variance_strategy = sum((value - mean_strategy) ** 2 for value, _ in paired) / days
    if variance_benchmark <= 0:
        result["reason"] = "benchmark_variance_zero"
        return result
    beta = covariance / variance_benchmark
    alpha_daily = mean_strategy - beta * mean_benchmark
    result.update(
        available=True,
        reason=None,
        beta=round(beta, 3),
        alpha_annualized=round(alpha_daily * TRADING_DAYS_PER_YEAR, 4),
        r2=(
            round((covariance * covariance) / (variance_benchmark * variance_strategy), 3)
            if variance_strategy > 0
            else None
        ),
    )
    return result


def _symbol_column(frame: pd.DataFrame) -> str:
    return "ts_code" if "ts_code" in frame.columns else "symbol"


def _prior_day_spread(
    replay_daily: pd.DataFrame,
    column: str,
    key: Callable[[pd.Series], pd.Series],
) -> dict[str, float]:
    """Daily high-minus-low spread of the replay slot's own universe, the
    names sorted on ``key`` of the PREVIOUS trading day's ``column``.

    Frozen replay data only, like every other input here: the cross-section is
    the slot's ``daily`` frame, so a leg cannot disagree with what the
    strategy actually traded against. ``daily``'s ``pct_chg`` is already a
    decimal fraction (the unit registry applies the percent->decimal factor at
    snapshot load), matching the strategy and benchmark return scale, so the
    spread is used as-is. The high and low legs are the top and bottom
    ``_FACTOR_QUANTILE`` of the day's sort, equal-weighted; a day with fewer
    than ``_FACTOR_MIN_NAMES`` sorted names has no reading.

    A name joins a leg by the value it had at the previous trading day's
    close. The same day's float cap is the end-of-day cap and already embeds
    the day's return, so sorting on it moves the day's winners into the big
    leg and its losers into the small leg and biases the spread negative
    whatever the true size premium (2022 replay: -17% sorted on the same day,
    +8% on the prior day); the same holds for a price-scaled ratio such as
    pb. A name's first row in the slot -- the window's first day, or an IPO's
    listing day -- has no prior value and sits out that day, and so does one
    whose ``key`` is NaN.
    """

    symbol = _symbol_column(replay_daily)
    required = {symbol, "trade_date", column, "pct_chg"}
    if not required.issubset(replay_daily.columns):
        return {}
    frame = replay_daily[[symbol, "trade_date", column, "pct_chg"]].copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
    frame = frame.sort_values([symbol, "trade_date"], kind="stable")
    frame["prior"] = key(frame.groupby(symbol, sort=False)[column].shift(1))
    frame = frame.dropna(subset=["prior", "pct_chg"])
    result: dict[str, float] = {}
    for date, group in frame.groupby("trade_date"):
        if len(group) < _FACTOR_MIN_NAMES:
            continue
        high = group.loc[
            group["prior"] >= group["prior"].quantile(1.0 - _FACTOR_QUANTILE), "pct_chg"
        ]
        low = group.loc[group["prior"] <= group["prior"].quantile(_FACTOR_QUANTILE), "pct_chg"]
        if high.empty or low.empty:
            continue
        value = float(high.mean() - low.mean())
        if math.isfinite(value):
            result[_date_text(date)] = value
    return result


def _size_factor(replay_daily: pd.DataFrame) -> dict[str, float]:
    """Daily small-minus-big float-cap spread (:func:`_prior_day_spread`)."""

    return {
        date: -value
        for date, value in _prior_day_spread(replay_daily, "circ_mv", lambda cap: cap).items()
    }


def _value_factor(replay_daily: pd.DataFrame) -> dict[str, float]:
    """Daily high-minus-low book-to-price spread (:func:`_prior_day_spread`):
    book-to-price is 1 / pb, and a name whose previous pb is not positive has
    no book-to-price that day."""

    return _prior_day_spread(replay_daily, "pb", lambda pb: (1.0 / pb).where(pb > 0))


def factor_ols(y: np.ndarray, legs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS of the daily series ``y`` on an intercept and the ``legs`` columns
    (market, size, value): the coefficients ``[intercept, *betas]`` and the
    residuals. ``ValueError`` when the legs are collinear over the days.

    The one fit every factor reading runs: the sidecar's own
    ``factor_regression`` block and every verdict statistic
    (``pipelines/verdict.py``).
    """

    design = np.column_stack([np.ones(len(y)), legs])
    coefficients, _residual, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("the factor legs are collinear over the span")
    return coefficients, y - design @ coefficients


def _factor_regression(
    strategy: list[tuple[str, float]],
    benchmark: Mapping[str, float],
    size: Mapping[str, float],
    value: Mapping[str, float],
) -> dict[str, object]:
    """The strategy's loadings on the benchmark, the size and the value legs.

    Three-regressor OLS (:func:`factor_ols`) over the days that carry every
    leg; a collinear set is reported as a reason instead of raising:
    attribution is advisory and must never fail a backtest. It reports the
    three loadings, the residual standard deviation annualised (what a
    tracking mandate caps, n - 4 degrees of freedom) and R².
    """

    rows = [
        (daily, benchmark[date], size[date], value[date])
        for date, daily in strategy
        if date in benchmark and date in size and date in value
    ]
    days = len(rows)
    result: dict[str, object] = {
        "available": False,
        "reason": "factors_unavailable" if not days else "insufficient_overlapping_days",
        "n_days": days,
        "tracking_error": None,
        **dict.fromkeys(name for name, _series in FACTOR_SERIES),
        "r2": None,
    }
    if days < MIN_REGRESSION_DAYS:
        return result
    table = np.asarray(rows, dtype=float)
    try:
        coefficients, residual = factor_ols(table[:, 0], table[:, 1:])
    except ValueError:
        result["reason"] = "factors_collinear"
        return result
    squares = float(residual @ residual)
    total = float(((table[:, 0] - table[:, 0].mean()) ** 2).sum())
    result.update(
        available=True,
        reason=None,
        tracking_error=round(
            math.sqrt(squares / (days - len(coefficients)) * TRADING_DAYS_PER_YEAR), 4
        ),
        **{
            name: round(float(beta), 3)
            for (name, _series), beta in zip(FACTOR_SERIES, coefficients[1:], strict=True)
        },
        r2=round(1.0 - squares / total, 3) if total > 0 else None,
    )
    return result


def _positions_from_curve(
    curve: Sequence[Mapping[str, object]],
) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = {}
    for row in curve:
        date = row.get("trade_date")
        positions = row.get("positions")
        if date is None or not isinstance(positions, Mapping):
            continue
        holdings: list[tuple[str, float]] = []
        for code, raw in positions.items():
            try:
                quantity = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(quantity) and quantity > 0:
                holdings.append((str(code), quantity))
        result[_date_text(date)] = holdings
    return result


def _rank_cross_section(frame: pd.DataFrame, symbol_column: str) -> dict[str, tuple[float, float, float, float]]:
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    codes = frame[symbol_column].astype(str).to_numpy()
    ranks = {
        column: pd.to_numeric(frame[column].rank(pct=True), errors="coerce")
        .fillna(0.5)
        .replace([np.inf, -np.inf], 0.5)
        .to_numpy(dtype=float)
        for column in _STYLE_COLUMNS
    }
    return {
        code: (float(close), float(size), float(pb), float(turnover))
        for code, close, size, pb, turnover in zip(
            codes,
            closes,
            ranks["circ_mv"],
            ranks["pb"],
            ranks["turnover_rate"],
        )
        if math.isfinite(float(close)) and float(close) > 0
    }


def _empty_style(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "days": 0,
        "tilts": None,
        "industries": [],
        "top_industry_weight": None,
        "avg_names": None,
        "avg_long_gross": None,
        "avg_short_gross": None,
    }


def _style_exposures(
    replay_daily: pd.DataFrame,
    curve: Sequence[Mapping[str, object]],
    industry_vintages: Sequence[tuple[str, Mapping[str, str]]],
) -> dict[str, object]:
    symbol_column = _symbol_column(replay_daily)
    required = {symbol_column, "trade_date", "close", *_STYLE_COLUMNS}
    if not required.issubset(replay_daily.columns):
        return _empty_style("style_columns_unavailable")
    positions = _positions_from_curve(curve)
    if not any(positions.values()):
        return _empty_style("no_holdings")
    wanted = set(positions)
    basics = {
        _date_text(date): _rank_cross_section(group, symbol_column)
        for date, group in replay_daily.groupby("trade_date")
        if _date_text(date) in wanted
    }
    tilt_sums = {"size": 0.0, "pb": 0.0, "turnover": 0.0}
    industry_sums: dict[str, float] = {}
    days = 0
    names = 0.0
    long_gross = 0.0
    for date, holdings in sorted(positions.items()):
        # The vintage of the slot this day belongs to: the last one started by it.
        industry_by_code = next(
            (industry for day, industry in reversed(industry_vintages) if day <= date), {}
        )
        valued: list[tuple[str, float, float, float, float]] = []
        for code, quantity in holdings:
            item = basics.get(date, {}).get(code)
            if item is None:
                continue
            close, size_rank, pb_rank, turnover_rank = item
            valued.append((code, quantity * close, size_rank, pb_rank, turnover_rank))
        gross = sum(value for _, value, *_ in valued)
        if gross <= 0:
            continue
        days += 1
        names += len(valued)
        long_gross += gross
        for code, value, size_rank, pb_rank, turnover_rank in valued:
            weight = value / gross
            tilt_sums["size"] += weight * (size_rank - 0.5) * 2
            tilt_sums["pb"] += weight * (pb_rank - 0.5) * 2
            tilt_sums["turnover"] += weight * (turnover_rank - 0.5) * 2
            industry = industry_by_code.get(code) or "未分类"
            industry_sums[industry] = industry_sums.get(industry, 0.0) + weight
    if not days:
        return _empty_style("no_valued_holdings")
    return {
        "available": True,
        "reason": None,
        "days": days,
        "tilts": {key: round(total / days, 3) for key, total in tilt_sums.items()},
        # The largest single SW L1 industry's share of the book, averaged over
        # decision days. The zero-skill panel draws its names without regard
        # to industry, so a single-sector book's sector move scores as
        # selection; the only frozen artifact that cleared every gate was
        # 82.5 % banks. ``industries`` stays host-side, so this one scalar is
        # what the compact block carries to the Agent.
        "top_industry_weight": round(max(industry_sums.values()) / days, 3),
        "industries": sorted(
            (
                {"name": name, "weight": round(total / days, 3)}
                for name, total in industry_sums.items()
            ),
            key=lambda item: -abs(float(item["weight"])),
        )[:8],
        "avg_names": round(names / days, 1),
        "avg_long_gross": round(long_gross / days, 2),
        "avg_short_gross": 0.0,
    }


def replay_style_analysis(
    replay: ReplayResult,
    replay_daily: pd.DataFrame,
    *,
    replay_dir: Path | Sequence[Path] | None,
    universes: Sequence[tuple[str, Path]],
    mode: str,
    panel: Mapping[str, object] | None = None,
    benchmark_index: str,
) -> dict[str, object]:
    """Compute one result sidecar from the just-finished daily replay.

    ``replay_dir`` is the replay's slot, or the slots of a span in order.
    ``universes`` pairs each slot's first day with the as-of universe file the
    strategy's view published while that slot ran; a day's holdings are
    classified under the industry membership of the vintage in force that day.
    ``benchmark_index`` is the arm's benchmark: the sidecar records it beside
    the series it was measured on, so a stored result can never be read without
    knowing what it was graded against.
    ``panel`` is the replay's zero-skill panel block
    (``null_control.run_null_control(..., panel=True)``): its two daily series
    are stored beside the strategy's, and the active series (strategy minus
    composite), the series the verdict grades, is read as it is. A truncated
    smoke replay passes none.
    """

    label = benchmark_index_label(benchmark_index)
    # The caliber of the factor regression below.
    method = regression_method(benchmark_index)
    strategy = daily_returns_from_curve(replay.equity_curve)
    benchmark = slot_benchmark(replay_dir, benchmark_index=benchmark_index)
    regression = _benchmark_regression(strategy, benchmark)
    size = _size_factor(replay_daily)
    value = _value_factor(replay_daily)
    factors = {"method": method, **_factor_regression(strategy, benchmark, size, value)}
    panel_blocks: dict[str, object] = {}
    panel_compact: dict[str, object] = {}
    if panel is not None:
        graded = active_analysis(
            {"strategy_daily": strategy, "panel_daily": panel.get("panel_daily")}
        )
        active = [value for _date, value in _series_pairs(graded["strategy_daily"])] if graded else []
        plain = plain_statistics(active)

        def rounded(key: str) -> float | None:
            value = plain[key]
            return None if value is None else round(float(value), 4)

        panel_blocks = {
            "panel": {
                key: value
                for key, value in panel.items()
                if key not in ("panel_daily", "panel_draw_sd")
            },
            "panel_daily": panel.get("panel_daily"),
            "panel_draw_sd": panel.get("panel_draw_sd"),
        }
        panel_compact = {
            # The graded series: the strategy's daily return minus the
            # zero-skill panel composite of its own book, read as it is; the
            # sealed gate regresses it on the factor legs. Its drawdown is
            # reported beside it and judged by neither the freeze registration
            # (which holds the equity drawdown to max_drawdown) nor the
            # verdict: the Agent reads the series it is graded on.
            "active_excess": rounded("plain_excess"),
            "active_tracking_error": rounded("tracking_error"),
            "active_information_ratio": rounded("information_ratio"),
            "active_max_drawdown": compounded_drawdown(active) if graded else None,
            "panel_draws": panel.get("k"),
        }
    style = _style_exposures(
        replay_daily, replay.equity_curve, _industry_vintages(universes)
    )
    total_return = total_return_from_curve(replay.equity_curve)
    benchmark_return = regression.get("benchmark_return")
    excess_return = (
        round(float(total_return) - float(benchmark_return), 6)
        if isinstance(total_return, (int, float))
        and not isinstance(total_return, bool)
        and isinstance(benchmark_return, (int, float))
        and not isinstance(benchmark_return, bool)
        else None
    )
    tilts = style.get("tilts")
    return {
        "schema_version": STYLE_SCHEMA_VERSION,
        "mode": mode,
        "benchmark": {"ts_code": benchmark_index, "label": label},
        "benchmark_regression": regression,
        "factor_regression": factors,
        "style": style,
        "strategy_daily": [[date, value] for date, value in strategy],
        "benchmark_daily": [[date, benchmark[date]] for date, _ in strategy if date in benchmark],
        "size_factor_daily": [[date, size[date]] for date, _ in strategy if date in size],
        "value_factor_daily": [[date, value[date]] for date, _ in strategy if date in value],
        **panel_blocks,
        "compact": {
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            # The two figures a tracking mandate limits: the residual standard
            # deviation of the strategy's own factor regression and its
            # loading on the benchmark.
            "tracking_error": factors.get("tracking_error"),
            "market_beta": factors.get("market_beta"),
            **panel_compact,
            "beta": regression.get("beta"),
            "n_days": regression.get("n_days"),
            "size_tilt": tilts.get("size") if isinstance(tilts, Mapping) else None,
            # ``size_tilt`` says which rung of the cap ladder the book stands
            # on; ``size_beta`` is the book's loading on the size factor.
            # Across measured books the two move in opposite
            # directions, so a budget written on the tilt cannot reach the
            # loading that produces a size-driven drawdown.
            "size_beta": factors.get("size_beta"),
            "value_beta": factors.get("value_beta"),
            "top_industry_weight": style.get("top_industry_weight"),
        },
        "created_at": utc_now_iso(),
    }


def _series_pairs(value: object) -> list[tuple[str, float]]:
    """``[[date, value], ...]`` from a stored analysis series, skipping junk."""

    rows: list[tuple[str, float]] = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return rows
    for item in value:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            continue
        try:
            rows.append((_date_text(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return rows


def active_analysis(analysis: Mapping[str, object]) -> dict[str, object] | None:
    """The analysis with the strategy's series replaced by its ACTIVE series.

    Active is the strategy's daily return minus the zero-skill panel composite
    of the same day, so every reader of ``strategy_daily`` -- the sub-span
    figure below, the verdict's statistics, bootstrap and drawdown -- measures
    the active series through the code that measures the strategy's own. ``None``
    when the sidecar carries no panel (a truncated smoke replay, or a replay
    recorded before panels existed). A panel that misses one of the strategy's
    days raises: a day graded against nothing would read as pure skill.
    """

    composite = dict(_series_pairs(analysis.get("panel_daily")))
    if not composite:
        return None
    strategy = _series_pairs(analysis.get("strategy_daily"))
    missing = [date for date, _value in strategy if date not in composite]
    if missing:
        raise ValueError(f"zero-skill panel has no return for {missing[0]}")
    return {
        **analysis,
        "strategy_daily": [[date, value - composite[date]] for date, value in strategy],
    }


def window_plain_excess(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> float | None:
    """The annualised mean of one span of a stored analysis's ``strategy_daily``,
    with no regression (``stats.plain_statistics``).

    Of an active analysis (:func:`active_analysis`) it is the plain active
    excess every gate grades: the one computation point for a span narrower
    than the whole window -- the per-year figure ``attach_sub_window_benchmark``
    writes into a result, and the verdict's research years and recency window.
    ``None`` when the span holds fewer than two days. An empty
    ``start``/``end`` takes the whole window.
    """

    value = plain_statistics(
        value
        for date, value in _series_pairs(analysis.get("strategy_daily"))
        if (not start or date >= start) and (not end or date <= end)
    )["plain_excess"]
    return None if value is None else float(value)


def benchmark_summary_block(analysis: Mapping[str, object]) -> dict[str, object] | None:
    """The compact benchmark projection an evaluation summary carries.

    The sidecar is the single computation point; this is the same numbers in the
    shape the ledger and the Agent-visible metric projection read
    (``label`` + ``benchmark_return`` at minimum), and the index is the one the
    sidecar itself recorded rather than a restated default. Returns None when
    the slot had no usable benchmark, so a missing block stays a truthful "not
    measured" instead of a fabricated zero — the sidecar beside the result still
    names the benchmark the replay ran against.
    """

    compact = analysis.get("compact")
    benchmark = analysis.get("benchmark")
    if not isinstance(compact, Mapping) or not isinstance(benchmark, Mapping):
        return None
    benchmark_return = compact.get("benchmark_return")
    if not isinstance(benchmark_return, (int, float)) or isinstance(benchmark_return, bool):
        return None
    return {
        "ts_code": benchmark.get("ts_code"),
        "label": benchmark.get("label"),
        **compact,
    }


def write_style_rollup(result_dir: Path, payload: Mapping[str, object]) -> Path:
    """Write the one canonical style sidecar for an evaluation result."""

    if payload.get("schema_version") != STYLE_SCHEMA_VERSION:
        raise ValueError("unsupported style analysis schema")
    target = Path(result_dir) / STYLE_ARTIFACT_NAME
    write_json_atomic(target, dict(payload))
    return target


__all__ = [
    "BENCHMARK_LABEL",
    "BENCHMARK_TS_CODE",
    "FACTOR_SERIES",
    "MIN_REGRESSION_DAYS",
    "READABLE_STYLE_SCHEMA_VERSIONS",
    "STYLE_ARTIFACT_NAME",
    "STYLE_SCHEMA_VERSION",
    "active_analysis",
    "benchmark_summary_block",
    "daily_returns_from_curve",
    "factor_ols",
    "regression_method",
    "replay_style_analysis",
    "slot_benchmark",
    "slot_membership",
    "window_plain_excess",
    "write_style_rollup",
]
