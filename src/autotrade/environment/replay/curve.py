"""One replay result as a daily curve: cumulative return, benchmark, exposure.

The research console draws an experiment's result through these functions, and
a Paper book copies its source experiment's curve through the same ones when it
is created, so the book's own copy and the console's read the same numbers.
Nothing here is chained or recomputed across results: one result file in, one
curve out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .style import BENCHMARK_LABEL, STYLE_ARTIFACT_NAME

STRATEGY_LABEL = "策略"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_result(path: Path) -> dict[str, object]:
    """One ``result.json``; an unreadable or non-object file reads as empty."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _curve_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    curve = payload.get("equity_curve")
    rows = [
        row
        for row in (curve if isinstance(curve, list) else ())
        if isinstance(row, dict) and row.get("trade_date") and _finite(row.get("equity"))
    ]
    return sorted(rows, key=lambda row: str(row["trade_date"]))


def result_returns(payload: dict[str, object]) -> list[tuple[str, float]]:
    rows = _curve_rows(payload)
    if not rows:
        return []
    initial = _finite(payload.get("initial_cash")) or _finite(rows[0].get("initial_equity"))
    previous = initial if initial and initial > 0 else float(rows[0]["equity"])  # type: ignore[arg-type]
    result: list[tuple[str, float]] = []
    for row in rows:
        equity = float(row["equity"])  # type: ignore[arg-type]
        if previous > 0:
            result.append((str(row["trade_date"]), equity / previous - 1.0))
        previous = equity
    return result


def result_exposures(payload: dict[str, object]) -> list[tuple[str, float]]:
    """Daily position weight (EOD gross market value / equity); the long-only
    book carries no short leg."""
    rows: list[tuple[str, float]] = []
    for row in _curve_rows(payload):
        equity = float(row["equity"])  # type: ignore[arg-type]
        cash = _finite(row.get("cash"))
        if cash is not None and equity > 0:
            rows.append((str(row["trade_date"]), round((equity - cash) / equity, 4)))
    return rows


def benchmark_returns(result_file: Path) -> list[tuple[str, float]]:
    """Benchmark daily returns from the result's own style sidecar."""

    sidecar = read_result(Path(result_file).parent / STYLE_ARTIFACT_NAME)
    rows: dict[str, float] = {}
    for item in sidecar.get("benchmark_daily") or ():
        if isinstance(item, list) and len(item) == 2 and item[0]:
            value = _finite(item[1])
            if value is not None:
                rows.setdefault(str(item[0]), value)
    return sorted(rows.items())


def benchmark_label(result_file: Path) -> str:
    """Which benchmark this result was graded against, from its own sidecar.

    The sidecar records the arm's ``benchmark_index``; a result written before
    the parameter existed carries the default, which is what those arms ran on.
    """

    sidecar = read_result(Path(result_file).parent / STYLE_ARTIFACT_NAME)
    benchmark = sidecar.get("benchmark")
    label = benchmark.get("label") if isinstance(benchmark, dict) else None
    return str(label) if label else BENCHMARK_LABEL


def curve_entry(key: str, label: str, rows: list[tuple[str, float]]) -> dict[str, object]:
    dates: list[str] = []
    cumulative: list[float] = []
    drawdown: list[float] = []
    equity = peak = 1.0
    for day, value in rows:
        equity *= 1.0 + value
        peak = max(peak, equity)
        dates.append(day)
        cumulative.append(round(equity - 1.0, 6))
        drawdown.append(round(equity / peak - 1.0, 6))
    return {
        "key": key,
        "label": label,
        "dates": dates,
        "cum": cumulative,
        "drawdown": drawdown,
        "final": cumulative[-1] if cumulative else None,
    }


def result_curve(result_file: Path) -> dict[str, object]:
    """The strategy's cumulative curve, its benchmark on the strategy's own days
    so both start at zero together, and the daily position weight."""

    payload = read_result(result_file)
    returns = result_returns(payload)
    days = {day for day, _value in returns}
    benchmark = [(day, value) for day, value in benchmark_returns(result_file) if day in days]
    exposure = result_exposures(payload)
    return {
        "series": [curve_entry("strategy", STRATEGY_LABEL, returns)] if returns else [],
        "benchmark": (
            curve_entry("benchmark", benchmark_label(result_file), benchmark)
            if benchmark
            else None
        ),
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
    "benchmark_label",
    "benchmark_returns",
    "curve_entry",
    "read_result",
    "result_curve",
    "result_exposures",
    "result_returns",
]
