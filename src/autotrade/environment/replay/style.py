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
strategy's own, and the same attribution run on the active series (strategy
minus composite), which is what ``pipelines/verdict.py`` grades.

Everything degrades to None/empty blocks when inputs are missing —
attribution is advisory and must never fail a backtest.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from autotrade.environment.data.contracts import (
    DEFAULT_BENCHMARK_INDEX,
    benchmark_index_label,
)
from autotrade.environment.runtime import utc_now_iso, write_json_atomic

from .stats import TRADING_DAYS_PER_YEAR, ReplayResult, total_return_from_curve

# The benchmark an arm takes when its create request names none, and the one a
# legacy sidecar or book that records none is read under. Every call below takes
# the arm's ``benchmark_index`` explicitly, so none can grade against this one
# by omission.
BENCHMARK_TS_CODE = DEFAULT_BENCHMARK_INDEX
BENCHMARK_LABEL = benchmark_index_label(BENCHMARK_TS_CODE)
_MIN_REGRESSION_DAYS = 8
STYLE_ARTIFACT_NAME = "style_analysis.json"
STYLE_SCHEMA_VERSION = 1
_STYLE_COLUMNS = ("circ_mv", "pb", "turnover_rate")
# Size-factor proxy for the neutralized excess return: the small-minus-big
# daily spread of the replay slot's OWN cross-section, equal-weighted inside
# each leg, the legs formed on the previous trading day's float cap. The
# audited edge read as a small-cap / low-beta tilt rather than proven alpha,
# and a raw excess return cannot tell those apart. This is not a Barra or
# Fama-French factor and the result says so in ``method``.
_SIZE_FACTOR_QUANTILE = 0.3
_SIZE_FACTOR_MIN_NAMES = 30


def neutralization_method(benchmark_index: str) -> str:
    """The caliber sentence every ``neutralized_excess_return`` is computed under.

    Names the arm's own benchmark: the market leg is index-specific, the size
    leg is built from the replay's own cross-section whatever the benchmark is.
    """

    return (
        f"日度策略收益对{benchmark_index_label(benchmark_index)}收益与规模因子"
        "（本回放槽全市场截面、不是本臂股票池，"
        "按前一交易日流通市值分组：最小 30% 等权减最大 30% 等权）"
        f"的二元 OLS，截距按 {TRADING_DAYS_PER_YEAR} 个交易日年化"
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
    try:
        frame = pd.read_parquet(
            path,
            columns=list(required),
            filters=[
                ("dataset", "==", "index_daily"),
                ("ts_code", "==", benchmark_index),
            ],
        )
    except Exception:
        frame = pd.read_parquet(path)
    if not required.issubset(frame.columns):
        return {}
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
    if days < _MIN_REGRESSION_DAYS:
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


def _size_factor(replay_daily: pd.DataFrame) -> dict[str, float]:
    """Daily small-minus-big spread built from the replay slot's own universe.

    Frozen replay data only, like every other input here: the cross-section is
    the slot's ``daily`` frame, so the factor cannot disagree with what the
    strategy actually traded against. ``daily``'s ``pct_chg`` is already a
    decimal fraction (the unit registry applies the percent->decimal factor at
    snapshot load), matching the strategy and benchmark return scale, so the
    spread is used as-is.

    A name joins a leg by the float cap it had at the PREVIOUS trading day's
    close. The same day's ``circ_mv`` is the end-of-day cap and already embeds
    the day's return, so sorting on it moves the day's winners into the big
    leg and its losers into the small leg and biases the spread negative
    whatever the true size premium (2022 replay: -17% sorted on the same day,
    +8% on the prior day). A name's first row in the slot -- the window's first
    day, or an IPO's listing day -- has no prior cap and sits out that day.
    """

    symbol = _symbol_column(replay_daily)
    required = {symbol, "trade_date", "circ_mv", "pct_chg"}
    if not required.issubset(replay_daily.columns):
        return {}
    frame = replay_daily[[symbol, "trade_date", "circ_mv", "pct_chg"]].copy()
    frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce")
    frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
    frame = frame.sort_values([symbol, "trade_date"], kind="stable")
    frame["prior_cap"] = frame.groupby(symbol, sort=False)["circ_mv"].shift(1)
    frame = frame.dropna(subset=["prior_cap", "pct_chg"])
    result: dict[str, float] = {}
    for date, group in frame.groupby("trade_date"):
        if len(group) < _SIZE_FACTOR_MIN_NAMES:
            continue
        small = group.loc[
            group["prior_cap"] <= group["prior_cap"].quantile(_SIZE_FACTOR_QUANTILE),
            "pct_chg",
        ]
        big = group.loc[
            group["prior_cap"] >= group["prior_cap"].quantile(1.0 - _SIZE_FACTOR_QUANTILE),
            "pct_chg",
        ]
        if small.empty or big.empty:
            continue
        value = float(small.mean() - big.mean())
        if math.isfinite(value):
            result[_date_text(date)] = value
    return result


def _neutralized_excess(
    strategy: list[tuple[str, float]],
    benchmark: Mapping[str, float],
    size: Mapping[str, float],
) -> dict[str, object]:
    """Excess return left after the market-beta and size contributions.

    Two-regressor OLS through the normal equations, so a collinear or degenerate
    pair is reported as a reason instead of raising: attribution is advisory and
    must never fail a backtest. The figure is the daily intercept times the
    trading days of a year -- an arithmetic, annualized number, not the
    compounded window excess beside it: a book with a large loading on a factor
    that moved a lot in the window shows a large intercept the raw excess does
    not, and on a short window the annualization scales the intercept up.
    """

    rows = [
        (value, benchmark[date], size[date])
        for date, value in strategy
        if date in benchmark and date in size
    ]
    days = len(rows)
    result: dict[str, object] = {
        "available": False,
        "reason": "factors_unavailable" if not days else "insufficient_overlapping_days",
        "n_days": days,
        "neutralized_excess_return": None,
        "tracking_error": None,
        "information_ratio": None,
        "market_beta": None,
        "size_beta": None,
        "r2": None,
    }
    if days < _MIN_REGRESSION_DAYS:
        return result
    mean_y = sum(row[0] for row in rows) / days
    mean_1 = sum(row[1] for row in rows) / days
    mean_2 = sum(row[2] for row in rows) / days
    s11 = sum((row[1] - mean_1) ** 2 for row in rows)
    s22 = sum((row[2] - mean_2) ** 2 for row in rows)
    s12 = sum((row[1] - mean_1) * (row[2] - mean_2) for row in rows)
    s1y = sum((row[1] - mean_1) * (row[0] - mean_y) for row in rows)
    s2y = sum((row[2] - mean_2) * (row[0] - mean_y) for row in rows)
    syy = sum((row[0] - mean_y) ** 2 for row in rows)
    determinant = s11 * s22 - s12 * s12
    if determinant <= 0:
        result["reason"] = "factors_collinear"
        return result
    market_beta = (s22 * s1y - s12 * s2y) / determinant
    size_beta = (s11 * s2y - s12 * s1y) / determinant
    alpha_daily = mean_y - market_beta * mean_1 - size_beta * mean_2
    residual = syy - market_beta * s1y - size_beta * s2y
    excess = round(alpha_daily * TRADING_DAYS_PER_YEAR, 4)
    # The residual standard deviation the verdict calls tracking error: three
    # fitted coefficients, annualized by the square root of the trading year.
    tracking_error = math.sqrt(max(residual, 0.0) / (days - 3) * TRADING_DAYS_PER_YEAR)
    result.update(
        available=True,
        reason=None,
        neutralized_excess_return=excess,
        tracking_error=round(tracking_error, 4),
        information_ratio=round(excess / tracking_error, 4) if tracking_error > 0 else None,
        market_beta=round(market_beta, 3),
        size_beta=round(size_beta, 3),
        r2=round(1.0 - residual / syy, 3) if syy > 0 else None,
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
        # decision days. The neutralization regresses on the benchmark and size only,
        # so a single-sector book scores as alpha; the only frozen artifact that
        # cleared every gate was 82.5 % banks. ``industries`` stays host-side,
        # so this one scalar is what the compact block carries to the Agent.
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
    are stored beside the strategy's, and the same attribution is run on the
    composite and on the active series (strategy minus composite), which is
    the series the verdict grades. A truncated smoke replay passes none.
    """

    label = benchmark_index_label(benchmark_index)
    # The caliber every neutralized block below is labelled with.
    method = neutralization_method(benchmark_index)
    strategy = daily_returns_from_curve(replay.equity_curve)
    benchmark = slot_benchmark(replay_dir, benchmark_index=benchmark_index)
    regression = _benchmark_regression(strategy, benchmark)
    size = _size_factor(replay_daily)
    neutralized = {"method": method, **_neutralized_excess(strategy, benchmark, size)}
    panel_blocks: dict[str, object] = {}
    panel_compact: dict[str, object] = {}
    if panel is not None:
        composite = _series_pairs(panel.get("panel_daily"))
        graded = active_analysis(
            {"strategy_daily": strategy, "panel_daily": panel.get("panel_daily")}
        )
        active = {
            "method": method,
            **_neutralized_excess(
                _series_pairs(graded["strategy_daily"]) if graded else [],
                benchmark,
                size,
            ),
        }
        zero_skill = {"method": method, **_neutralized_excess(composite, benchmark, size)}
        panel_blocks = {
            "panel": {
                key: value
                for key, value in panel.items()
                if key not in ("panel_daily", "panel_draw_sd")
            },
            "panel_daily": panel.get("panel_daily"),
            "panel_draw_sd": panel.get("panel_draw_sd"),
            "active_neutralized_excess": active,
            "panel_neutralized_excess": zero_skill,
        }
        panel_compact = {
            # What the freeze gate and the verdict grade: the strategy's daily
            # return minus the zero-skill panel composite of its own book.
            "active_neutralized_excess": active.get("neutralized_excess_return"),
            "active_tracking_error": active.get("tracking_error"),
            "active_information_ratio": active.get("information_ratio"),
            "panel_neutralized_excess": zero_skill.get("neutralized_excess_return"),
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
        "neutralized_excess": neutralized,
        "style": style,
        "strategy_daily": [[date, value] for date, value in strategy],
        "benchmark_daily": [[date, benchmark[date]] for date, _ in strategy if date in benchmark],
        "size_factor_daily": [[date, size[date]] for date, _ in strategy if date in size],
        **panel_blocks,
        "compact": {
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            # The raw excess cannot separate an edge from a small-cap or
            # high-beta tilt, so the neutralized figure rides beside it
            # wherever the raw one is read.
            "neutralized_excess_return": neutralized.get("neutralized_excess_return"),
            "neutralized_excess_method": method,
            # The two figures a tracking mandate limits: the residual standard
            # deviation of that regression and its loading on the benchmark.
            "tracking_error": neutralized.get("tracking_error"),
            "market_beta": neutralized.get("market_beta"),
            **panel_compact,
            "beta": regression.get("beta"),
            "n_days": regression.get("n_days"),
            "size_tilt": tilts.get("size") if isinstance(tilts, Mapping) else None,
            # ``size_tilt`` says which rung of the cap ladder the book stands
            # on; ``size_beta`` is the loading the neutralization actually
            # divides out. Across measured books the two move in opposite
            # directions, so a budget written on the tilt cannot reach the
            # loading that produces a size-driven drawdown.
            "size_beta": neutralized.get("size_beta"),
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
    regression below, the verdict's bootstrap and drawdown -- measures the
    active series through the code that measures the strategy's own. ``None``
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


def window_neutralized_excess(
    analysis: Mapping[str, object], *, start: str = "", end: str = ""
) -> float | None:
    """The neutralized excess of one span of an existing style analysis.

    The sidecar stores the three daily series the attribution was run on, so
    any sub-span of it can be re-regressed without replaying anything: the same
    two-regressor OLS, the same annualization, just fewer days. This is the one
    computation point for a span narrower than the whole window -- the
    per-quarter figure ``attach_sub_window_benchmark`` writes into a result, and
    the same figure the ledger derives for a transition recorded before results
    carried it. ``None`` when the span has too few overlapping days to regress
    (a transition graded on an unmeasurable number must not silently fall back
    to the raw excess). An empty ``start``/``end`` takes the whole window.
    """

    strategy = [
        (date, value)
        for date, value in _series_pairs(analysis.get("strategy_daily"))
        if (not start or date >= start) and (not end or date <= end)
    ]
    benchmark = dict(_series_pairs(analysis.get("benchmark_daily")))
    size = dict(_series_pairs(analysis.get("size_factor_daily")))
    if not strategy or not benchmark or not size:
        return None
    value = _neutralized_excess(strategy, benchmark, size).get("neutralized_excess_return")
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else None
    )


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
        **{key: value for key, value in compact.items()},
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
    "STYLE_ARTIFACT_NAME",
    "STYLE_SCHEMA_VERSION",
    "active_analysis",
    "benchmark_summary_block",
    "daily_returns_from_curve",
    "neutralization_method",
    "replay_style_analysis",
    "slot_benchmark",
    "slot_membership",
    "window_neutralized_excess",
    "write_style_rollup",
]
