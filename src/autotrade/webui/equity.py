"""Daily equity series of one ledger-named replay result.

The curve is read from the result's own ``result.json`` and the CSI 300
benchmark from its style sidecar; nothing is chained or recomputed across
results. Which results exist at all is ``registry.ledger_result``'s answer, so
the forward replay has no curve until its verdict is recorded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from autotrade.environment.replay.style import BENCHMARK_LABEL, STYLE_ARTIFACT_NAME

from . import registry


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _returns(payload: dict[str, object]) -> list[tuple[str, float]]:
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


def _exposures(payload: dict[str, object]) -> list[tuple[str, float]]:
    """Daily position weight (EOD gross market value / equity); the long-only
    book carries no short leg."""
    rows: list[tuple[str, float]] = []
    for row in _curve_rows(payload):
        equity = float(row["equity"])  # type: ignore[arg-type]
        cash = _finite(row.get("cash"))
        if cash is not None and equity > 0:
            rows.append((str(row["trade_date"]), round((equity - cash) / equity, 4)))
    return rows


def _benchmark_returns(result_file: Path) -> list[tuple[str, float]]:
    sidecar = _read(result_file.parent / STYLE_ARTIFACT_NAME)
    rows: dict[str, float] = {}
    for item in sidecar.get("benchmark_daily") or ():
        if isinstance(item, list) and len(item) == 2 and item[0]:
            value = _finite(item[1])
            if value is not None:
                rows.setdefault(str(item[0]), value)
    return sorted(rows.items())


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


def result_equity_payload(root: Path, experiment_id: str, name: str) -> dict[str, object]:
    result_file = registry.ledger_result(root, experiment_id, name)
    payload = _read(result_file)
    returns = _returns(payload)
    days = {day for day, _value in returns}
    benchmark = [(day, value) for day, value in _benchmark_returns(result_file) if day in days]
    exposure = _exposures(payload)
    return {
        "experiment_id": experiment_id,
        "result": name,
        "series": [curve_entry("strategy", "策略", returns)] if returns else [],
        "benchmark": curve_entry("benchmark", BENCHMARK_LABEL, benchmark) if benchmark else None,
        "exposure": {
            "strategy": {
                "dates": [day for day, _value in exposure],
                "long": [value for _day, value in exposure],
            }
        }
        if exposure
        else {},
    }
