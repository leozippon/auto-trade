"""Barra-lite style / benchmark attribution — single computation point.

All attribution math runs HOST-SIDE at replay completion, and every input is
frozen run data — never the mutable raw lake (whose history gets revised, see
the revision ledger; recomputing later from raw could disagree with what the
Agent actually saw):

- strategy daily returns: the window's own ``equity_curve``;
- holdings: the replay result's end-of-day position snapshots;
- cross-sectional style ranks: the replay slot's ``daily.parquet``;
- CSI 300 benchmark: ``index_daily`` rows inside the replay slot's
  ``macro.parquet``;
- SW L1 industry: the decision snapshot's ``universe.parquet`` (as-of the
  decision day — membership drift within a replay window is negligible).

The backtest tool writes one ``style_analysis.json`` per result window (valid,
test and held-out alike; test/held-out replays run after the Agent session).
The pipeline then writes one ``style_<prefix>.json`` rollup per window chain
under ``results/``, which is what the console serves — the web layer performs
no attribution computation and touches no raw data.

Everything degrades to None/empty blocks when inputs are missing —
attribution is advisory and must never fail a backtest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from autotrade.environment.runtime import utc_now_iso, write_json_atomic

from .stats import TRADING_DAYS_PER_YEAR, ReplayResult, compute_return_stats

BENCHMARK_LABEL = "沪深300"
BENCHMARK_TS_CODE = "000300.SH"
_MIN_REGRESSION_DAYS = 8
STYLE_ARTIFACT_NAME = "style_analysis.json"
STYLE_SCHEMA_VERSION = 1
_STYLE_COLUMNS = ("circ_mv", "pb", "turnover_rate")


def _date_text(value: object) -> str:
    try:
        return pd.Timestamp(str(value)).strftime("%Y%m%d")
    except (TypeError, ValueError):
        return str(value)


def _slot_benchmark(replay_dir: Path | None) -> dict[str, float]:
    if replay_dir is None:
        return {}
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
                ("ts_code", "==", BENCHMARK_TS_CODE),
            ],
        )
    except Exception:
        frame = pd.read_parquet(path)
    if not required.issubset(frame.columns):
        return {}
    rows = frame[
        frame["dataset"].astype(str).eq("index_daily")
        & frame["ts_code"].astype(str).eq(BENCHMARK_TS_CODE)
    ]
    result: dict[str, float] = {}
    for date, raw in zip(rows["trade_date"], rows["pct_chg"]):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[_date_text(date)] = value / 100.0
    return result


def _snapshot_industry(snapshot_dir: Path | None) -> dict[str, str]:
    if snapshot_dir is None:
        return {}
    path = Path(snapshot_dir) / "universe.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    if not {"ts_code", "l1_name"}.issubset(frame.columns):
        return {}
    return {
        str(code): str(name)
        for code, name in zip(frame["ts_code"], frame["l1_name"])
        if isinstance(name, str) and name
    }


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
        "avg_names": None,
        "avg_long_gross": None,
        "avg_short_gross": None,
    }


def _style_exposures(
    replay_daily: pd.DataFrame,
    curve: Sequence[Mapping[str, object]],
    industry_by_code: Mapping[str, str],
) -> dict[str, object]:
    symbol_column = "ts_code" if "ts_code" in replay_daily.columns else "symbol"
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
    replay_dir: Path | None,
    snapshot_dir: Path | None,
    mode: str,
) -> dict[str, object]:
    """Compute one result sidecar from the just-finished daily replay."""

    strategy = daily_returns_from_curve(replay.equity_curve)
    benchmark = _slot_benchmark(replay_dir)
    regression = _benchmark_regression(strategy, benchmark)
    style = _style_exposures(replay_daily, replay.equity_curve, _snapshot_industry(snapshot_dir))
    total_return = compute_return_stats(replay).get("total_return")
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
        "benchmark": {"ts_code": BENCHMARK_TS_CODE, "label": BENCHMARK_LABEL},
        "benchmark_regression": regression,
        "style": style,
        "strategy_daily": [[date, value] for date, value in strategy],
        "benchmark_daily": [[date, benchmark[date]] for date, _ in strategy if date in benchmark],
        "compact": {
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "beta": regression.get("beta"),
            "n_days": regression.get("n_days"),
            "size_tilt": tilts.get("size") if isinstance(tilts, Mapping) else None,
        },
        "created_at": utc_now_iso(),
    }


def benchmark_summary_block(analysis: Mapping[str, object]) -> dict[str, object] | None:
    """The compact benchmark projection an evaluation summary carries.

    The sidecar is the single computation point; this is the same numbers in the
    shape the ledger, the experiment report and the Meta metric projection read
    (``label`` + ``benchmark_return`` at minimum). Returns None when the slot had
    no usable benchmark, so a missing block stays a truthful "not measured"
    instead of a fabricated zero.
    """

    compact = analysis.get("compact")
    if not isinstance(compact, Mapping):
        return None
    benchmark_return = compact.get("benchmark_return")
    if not isinstance(benchmark_return, (int, float)) or isinstance(benchmark_return, bool):
        return None
    return {
        "ts_code": BENCHMARK_TS_CODE,
        "label": BENCHMARK_LABEL,
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
    "benchmark_summary_block",
    "daily_returns_from_curve",
    "replay_style_analysis",
    "write_style_rollup",
]
