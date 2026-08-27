"""Equity series assembled only from immutable experiment result artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.replay.style import (
    BENCHMARK_LABEL,
    STYLE_ARTIFACT_NAME,
    STYLE_SCHEMA_VERSION,
    _slot_benchmark,
)
from autotrade.pipelines.ledger import latest_fold_records, latest_heldout_records

from . import registry

SERIES_LABELS = {"valid": "策略（验证）", "test": "策略（测试）", "heldout": "策略（Held-out）"}
_LABELS = {"benchmark": BENCHMARK_LABEL, **SERIES_LABELS}


def _result_file(experiment_dir: Path, reference: object) -> Path | None:
    if not isinstance(reference, str) or not reference:
        return None
    raw = Path(reference)
    candidate = raw.resolve() if raw.is_absolute() else (experiment_dir / raw).resolve()
    if not candidate.is_relative_to(experiment_dir.resolve()):
        return None
    if candidate.is_dir():
        for name in ("result.json", "detailed_return.json"):
            if (candidate / name).is_file():
                return candidate / name
        return None
    return candidate if candidate.is_file() else None


def _equities(experiment_dir: Path, reference: object) -> list[tuple[str, float, float]]:
    path = _result_file(experiment_dir, reference)
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    curve = payload.get("equity_curve")
    initial = _finite(payload.get("initial_cash"))
    rows: list[tuple[str, float]] = []
    if isinstance(curve, dict):
        rows = [(str(day), value) for day, raw in curve.items() if (value := _finite(raw)) is not None]
    elif isinstance(curve, list):
        for item in curve:
            if not isinstance(item, dict):
                continue
            day = item.get("trade_date") or item.get("date")
            value = _finite(item.get("equity"))
            if day and value is not None:
                rows.append((str(day), value))
                if initial is None:
                    initial = _finite(item.get("initial_equity"))
    rows.sort()
    if not rows:
        return []
    initial = initial if initial is not None and initial > 0 else rows[0][1]
    return [(day, value, initial) for day, value in rows if value > 0]


def _returns(experiment_dir: Path, reference: object) -> list[tuple[str, float]]:
    rows = _equities(experiment_dir, reference)
    result: list[tuple[str, float]] = []
    previous = rows[0][2] if rows else 0.0
    for day, value, _initial in rows:
        if previous > 0:
            result.append((day, value / previous - 1.0))
        previous = value
    return result


def _exposures(experiment_dir: Path, reference: object) -> list[tuple[str, float]]:
    """Daily position weight (EOD gross market value / equity) from the replay
    curve. The long-only book carries no short leg, so the pane renders one
    series per return curve."""
    path = _result_file(experiment_dir, reference)
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    curve = payload.get("equity_curve") if isinstance(payload, dict) else None
    if not isinstance(curve, list):
        return []
    rows: list[tuple[str, float]] = []
    for item in curve:
        if not isinstance(item, dict):
            continue
        day = item.get("trade_date") or item.get("date")
        equity = _finite(item.get("equity"))
        cash = _finite(item.get("cash"))
        if not day or equity is None or cash is None or equity <= 0:
            continue
        rows.append((str(day), round((equity - cash) / equity, 4)))
    rows.sort()
    return rows


def _exposure_entry(rows: list[tuple[str, float]]) -> dict[str, object]:
    return {"dates": [row[0] for row in rows], "long": [row[1] for row in rows]}


_STYLE_MODES = frozenset({"valid", "frozen_test", "heldout"})


def _benchmark_returns(experiment_dir: Path, reference: object) -> list[tuple[str, float]]:
    result_file = _result_file(experiment_dir, reference)
    if result_file is None:
        return []
    sidecar = (result_file.parent / STYLE_ARTIFACT_NAME).resolve()
    if sidecar.is_relative_to(experiment_dir.resolve()) and sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if (
            isinstance(payload, Mapping)
            and payload.get("schema_version") == STYLE_SCHEMA_VERSION
            and payload.get("mode") in _STYLE_MODES
        ):
            rows = payload.get("benchmark_daily")
            result: dict[str, float] = {}
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, list) or len(item) != 2:
                        continue
                    value = _finite(item[1])
                    if item[0] and value is not None:
                        result.setdefault(str(item[0]), value)
            if result:
                return sorted(result.items())
    return _benchmark_from_replay_slot(experiment_dir, result_file)


def _benchmark_from_replay_slot(
    experiment_dir: Path, result_file: Path
) -> list[tuple[str, float]]:
    """Older frozen-test / Held-out dirs have no style sidecar; recover CSI 300
    from the replay slot named in the result PIT block."""
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    replay_ref = payload.get("pit") if isinstance(payload, Mapping) else None
    replay_dir = replay_ref.get("replay_ref") if isinstance(replay_ref, Mapping) else None
    if not isinstance(replay_dir, str) or not replay_dir:
        return []
    raw = Path(replay_dir)
    slot = raw.resolve() if raw.is_absolute() else (experiment_dir / raw).resolve()
    if not slot.is_relative_to(experiment_dir.resolve()):
        return []
    bench = _slot_benchmark(slot)
    if not bench:
        return []
    days = [day for day, _value, _initial in _equities(experiment_dir, str(result_file))]
    wanted = days or sorted(bench)
    return [(day, bench[day]) for day in wanted if day in bench]


def _run_result_ref(experiment_dir: Path, record: Mapping[str, object], prefix: str) -> object:
    explicit = record.get(f"{prefix}_result_ref")
    if explicit:
        return explicit
    run_id = str(record.get("run_id") or "")
    if run_id and Path(run_id).name == run_id:
        results = experiment_dir / "artifacts" / run_id / "results"
        candidates = sorted(path for path in results.glob(f"{prefix}*") if path.is_dir()) if results.is_dir() else []
        if candidates:
            return str(candidates[-1])
    return None


def _local_result_refs(experiment_dir: Path, prefix: str) -> list[str]:
    root = experiment_dir / "artifacts/results"
    if not root.is_dir():
        return []
    paths = [path for path in root.glob(f"{prefix}_*") if path.is_dir()]
    paths.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return [str(path) for path in paths]


def _chain(parts: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    by_day: dict[str, float] = {}
    for part in parts:
        for day, value in part:
            by_day.setdefault(day, value)
    return sorted(by_day.items())


def _curve_entry(key: str, rows: list[tuple[str, float]]) -> dict[str, object]:
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
        "label": _LABELS[key],
        "dates": dates,
        "cum": cumulative,
        "drawdown": drawdown,
        "final": cumulative[-1] if cumulative else None,
    }


def _cycle_stats(
    series: list[tuple[str, float]], bench: dict[str, float]
) -> dict[str, object] | None:
    """Full-cycle statistics over one chained daily-return series.

    Computed server-side like the curves. Return/vol/Sharpe/drawdown/win-rate
    use every strategy day; the benchmark-relative block (β, excess, tracking
    error, information ratio) uses date-matched days only, and both legs of the
    excess are compounded over that same matched set so they stay comparable.
    """
    if not series:
        return None
    values = [value for _day, value in series]
    n = len(values)
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
    cum = equity - 1.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1) if n > 1 else 0.0
    vol = math.sqrt(variance)
    stats: dict[str, object] = {
        "n_days": n,
        "cum_return": round(cum, 6),
        "annualized_return": round((1.0 + cum) ** (TRADING_DAYS_PER_YEAR / n) - 1.0, 6) if cum > -1.0 else -1.0,
        "annualized_vol": round(vol * math.sqrt(TRADING_DAYS_PER_YEAR), 6),
        "sharpe": round(mean / vol * math.sqrt(TRADING_DAYS_PER_YEAR), 4) if vol > 0 else 0.0,
        "max_drawdown": round(max_drawdown, 6),
        "daily_win_rate": round(sum(1 for value in values if value > 0) / n, 4),
    }
    paired = [(value, bench[day]) for day, value in series if day in bench]
    if len(paired) >= 2:
        strategy_leg = [a for a, _ in paired]
        bench_leg = [b for _, b in paired]
        strategy_cum = math.prod(1.0 + value for value in strategy_leg) - 1.0
        bench_cum = math.prod(1.0 + value for value in bench_leg) - 1.0
        bench_mean = sum(bench_leg) / len(bench_leg)
        strategy_mean = sum(strategy_leg) / len(strategy_leg)
        bench_var = sum((b - bench_mean) ** 2 for b in bench_leg)
        active = [a - b for a, b in paired]
        active_mean = sum(active) / len(active)
        active_var = sum((x - active_mean) ** 2 for x in active) / (len(active) - 1)
        stats.update(
            {
                "benchmark_days": len(paired),
                "benchmark_return": round(bench_cum, 6),
                "excess_return": round(strategy_cum - bench_cum, 6),
                "beta": (
                    round(sum((a - strategy_mean) * (b - bench_mean) for a, b in paired) / bench_var, 4)
                    if bench_var > 0
                    else None
                ),
                "tracking_error": round(math.sqrt(active_var * TRADING_DAYS_PER_YEAR), 6),
                "information_ratio": (
                    round(active_mean / math.sqrt(active_var) * math.sqrt(TRADING_DAYS_PER_YEAR), 4)
                    if active_var > 0
                    else None
                ),
            }
        )
    return stats


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def fold_equity_payload(root: Path, experiment_id: str, epoch_id: str, fold_ref: str) -> dict[str, object]:
    experiment_dir, _identity, records, record = registry.resolve_fold_record(
        root, experiment_id, epoch_id, fold_ref
    )
    validation_ref = registry.selected_validation_ref(record)
    valid_rows = _returns(experiment_dir, validation_ref)
    bench_parts = [_benchmark_returns(experiment_dir, validation_ref)]
    series = [_curve_entry("valid", valid_rows)] if valid_rows else []
    exposure_rows: dict[str, list[tuple[str, float]]] = {}
    if valid_rows:
        exposure_rows["valid"] = _exposures(experiment_dir, validation_ref)
    # P1-7: test curves stay hidden until the researcher reveals (seals) the
    # experiment; the UI's collapsed test section never renders without them.
    if registry.test_results_revealed(experiment_dir, records):
        test_reference = _run_result_ref(experiment_dir, record, "test")
        test_rows = _returns(experiment_dir, test_reference)
        if not test_rows:
            candidates = _local_result_refs(experiment_dir, "frozen_test")
            ordered = sorted(latest_fold_records(records).values(), key=lambda row: str(row.get("recorded_at") or ""))
            try:
                test_reference = candidates[ordered.index(record)]
                test_rows = _returns(experiment_dir, test_reference)
            except (ValueError, IndexError):
                pass
        if test_rows:
            series.append(_curve_entry("test", test_rows))
            exposure_rows["test"] = _exposures(experiment_dir, test_reference)
            bench_parts.append(_benchmark_returns(experiment_dir, test_reference))
    benchmark = _chain(bench_parts)
    return {
        "experiment_id": experiment_id,
        "epoch_id": epoch_id,
        "fold_ref": fold_ref,
        "series": series,
        "benchmark": _curve_entry("benchmark", benchmark) if benchmark else None,
        # Daily position weight (EOD gross market value / equity) per series,
        # rendered as a linked pane under the return curves.
        "exposure": {key: _exposure_entry(rows) for key, rows in exposure_rows.items() if rows},
    }


def experiment_equity_payload(root: Path, experiment_id: str, *, epoch_id: str | None = None) -> dict[str, object]:
    experiment_dir = registry.resolve_experiment_dir(root, experiment_id)
    records = registry.read_ledger_records(experiment_dir)
    folds = list(latest_fold_records(records).values())
    epochs = sorted({str(record.get("epoch_id")) for record in folds if record.get("epoch_id")})
    selected_epoch = epoch_id or (epochs[-1] if epochs else None)
    if selected_epoch is not None and selected_epoch not in epochs:
        raise KeyError(f"unknown epoch: {selected_epoch}")
    ordered_folds = sorted(folds, key=lambda row: str(row.get("recorded_at") or row.get("fold_id") or ""))
    selected = [record for record in ordered_folds if str(record.get("epoch_id")) == selected_epoch]
    validation_refs = [registry.selected_validation_ref(record) for record in selected]
    valid_rows = _chain([_returns(experiment_dir, reference) for reference in validation_refs])
    bench_parts = [_benchmark_returns(experiment_dir, reference) for reference in validation_refs]
    rows_by_key: dict[str, list[tuple[str, float]]] = {"valid": valid_rows}
    exposure_by_key: dict[str, list[tuple[str, float]]] = {
        "valid": _chain([_exposures(experiment_dir, reference) for reference in validation_refs])
    }
    if registry.test_results_revealed(experiment_dir, records):
        test_refs = _local_result_refs(experiment_dir, "frozen_test")
        test_parts: list[list[tuple[str, float]]] = []
        test_exposure_parts: list[list[tuple[str, float]]] = []
        for record in selected:
            result_index = ordered_folds.index(record)
            reference = _run_result_ref(experiment_dir, record, "test") or (test_refs[result_index] if result_index < len(test_refs) else None)
            test_parts.append(_returns(experiment_dir, reference))
            test_exposure_parts.append(_exposures(experiment_dir, reference))
            bench_parts.append(_benchmark_returns(experiment_dir, reference))
        rows_by_key["test"] = _chain(test_parts)
        exposure_by_key["test"] = _chain(test_exposure_parts)
        heldout_refs = [record.get("result_ref") for record in latest_heldout_records(records)]
        heldout_rows = _chain([_returns(experiment_dir, reference) for reference in heldout_refs])
        rows_by_key["heldout"] = heldout_rows
        exposure_by_key["heldout"] = _chain(
            [_exposures(experiment_dir, reference) for reference in heldout_refs]
        )
        for reference in heldout_refs:
            bench_parts.append(_benchmark_returns(experiment_dir, reference))
    series = [_curve_entry(key, rows) for key, rows in rows_by_key.items() if rows]
    benchmark = _chain(bench_parts)
    return {
        "experiment_id": experiment_id,
        "epoch_id": selected_epoch,
        "epochs": epochs,
        "series": series,
        "benchmark": _curve_entry("benchmark", benchmark) if benchmark else None,
        # Daily position weight (EOD gross market value / equity) per series,
        # rendered as a linked pane under the return curves.
        "exposure": {
            key: _exposure_entry(rows)
            for key, rows in exposure_by_key.items()
            if rows and rows_by_key.get(key)
        },
        # Full-cycle statistics per chained series (Barra-lite regression core
        # plus risk/consistency metrics); fold-level tilts stay on the fold view.
        "stats": {
            key: value
            for key, rows in rows_by_key.items()
            if (value := _cycle_stats(rows, dict(benchmark))) is not None
        },
    }
