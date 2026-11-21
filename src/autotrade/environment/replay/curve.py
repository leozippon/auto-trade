"""One replay result as a daily curve: cumulative return, benchmark, exposure.

The research console draws an experiment's result through these functions, and
a Paper account copies its source experiment's curve through the same ones when it
is created, so the book's own copy and the console's read the same numbers.
Nothing here is chained or recomputed across results: one result file in, one
curve out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from autotrade.common.records import finite_number, read_json_object_or_empty

from .market import date_text
from .stats import (
    compounded_path,
    daily_returns_from_curve,
    gross_exposure,
    series_pairs,
)
from .style import STYLE_ARTIFACT_NAME

STRATEGY_LABEL = "策略"
BASELINE_LABEL = "随机对照组合"


def _exposures(curve: Sequence[Mapping[str, object]]) -> list[tuple[str, float]]:
    """Daily position weight (``stats.gross_exposure``, to four places); the
    long-only book carries no short leg."""

    rows = sorted(
        (row for row in curve if row.get("trade_date") and finite_number(row.get("equity")) is not None),
        key=lambda row: date_text(row["trade_date"]),
    )
    return [
        (date_text(row["trade_date"]), round(value, 4))
        for row in rows
        if (value := gross_exposure(row)) is not None
    ]


def _benchmark_label(sidecar: Mapping[str, object], path: Path) -> str:
    """Which benchmark this result was graded against: the label of the
    experiment's ``benchmark_index`` its own sidecar records. A sidecar
    without one is refused."""

    benchmark = sidecar.get("benchmark")
    label = benchmark.get("label") if isinstance(benchmark, Mapping) else None
    if not isinstance(label, str) or not label:
        raise ValueError(f"{path} names no benchmark label")
    return label


def curve_entry(key: str, label: str, rows: list[tuple[str, float]]) -> dict[str, object]:
    path = compounded_path(value for _day, value in rows)
    cumulative = [round(equity - 1.0, 6) for equity, _loss in path]
    # Drawn as a signed fall below the peak; ``0.0 -`` keeps a peak day at +0.0.
    drawdown = [round(0.0 - loss, 6) for _equity, loss in path]
    return {
        "key": key,
        "label": label,
        "dates": [day for day, _value in rows],
        "cum": cumulative,
        "drawdown": drawdown,
        "final": cumulative[-1] if cumulative else None,
    }


def result_curve(result_file: Path) -> dict[str, object]:
    """The strategy's cumulative curve, its benchmark and its random baseline
    (the mean of the random-name copies of its own trades; ``None`` for a
    result without one) on the strategy's own days so all start at zero
    together, and the daily position weight."""

    curve = read_json_object_or_empty(result_file).get("equity_curve") or []
    returns = daily_returns_from_curve(curve)
    days = {day for day, _value in returns}
    sidecar_path = Path(result_file).parent / STYLE_ARTIFACT_NAME
    sidecar = read_json_object_or_empty(sidecar_path)
    benchmark = [(day, value) for day, value in series_pairs(sidecar.get("benchmark_daily")) if day in days]
    baseline = [(day, value) for day, value in series_pairs(sidecar.get("baseline_daily")) if day in days]
    exposure = _exposures(curve)
    return {
        "series": [curve_entry("strategy", STRATEGY_LABEL, returns)] if returns else [],
        "benchmark": (
            curve_entry("benchmark", _benchmark_label(sidecar, sidecar_path), benchmark)
            if benchmark
            else None
        ),
        "baseline": curve_entry("baseline", BASELINE_LABEL, baseline) if baseline else None,
        "exposure": {
            "strategy": {
                "dates": [day for day, _value in exposure],
                "long": [value for _day, value in exposure],
            }
        }
        if exposure
        else {},
    }


__all__ = [
    "STRATEGY_LABEL",
    "curve_entry",
    "result_curve",
]
