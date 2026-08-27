"""Replay result container and the return-statistics reducer.

``ReplayResult`` is what one full daily replay produces; ``compute_return_stats``
reduces it to the ``detailed_return.json`` payload (docs/environment-design.md
§3.8). Pure result→dict math — no replay logic.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from autotrade.environment.runtime import utc_now_iso

# A-share trading calendar (~244 sessions/year); style_analysis annualizes with
# the same constant so detailed_return and the Barra-lite sidecar agree.
TRADING_DAYS_PER_YEAR = 244

# Position-reducing verbs; fills here are strategy-initiated exits.
_EXIT_ACTIONS = frozenset({"sell"})


class PhaseTimer:
    """Wall-clock seconds accumulated per coarse backtest phase.

    A backtest has no other timing instrumentation: this is the whole of it.
    Phases are named by the caller, added to rather than nested, and reported
    once at the end, so a phase that never ran simply has no entry.
    """

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed

    def to_record(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in sorted(self._seconds.items())}


@dataclass(frozen=True)
class ReplayResult:
    equity_curve: tuple[dict[str, object], ...]
    executions: tuple[dict[str, object], ...]
    inference_dates: tuple[str, ...]
    pending_orders: tuple[dict[str, object], ...]
    # Replay-loop wall clock. Backend setup phases are merged in by the
    # evaluation backend, which is the only component that sees them.
    wall_seconds: float = 0.0
    phase_seconds: Mapping[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        return {
            "equity_curve": list(self.equity_curve),
            "executions": list(self.executions),
            "inference_dates": list(self.inference_dates),
            "pending_orders": list(self.pending_orders),
            "stats": compute_return_stats(self),
        }


def compute_return_stats(result: ReplayResult) -> dict[str, object]:
    """The minimum return statistics from docs/environment-design.md §3.8."""
    curve = result.equity_curve
    orders = result.executions
    initial = float(curve[0]["initial_equity"]) if curve else 0.0
    values = [float(row["equity"]) for row in curve]
    total_return = (values[-1] / initial - 1.0) if values and initial > 0 else 0.0
    # Day-0 baseline: daily returns and drawdown are measured against the initial
    # equity, so the first day's return (initial -> day-1 close) and a peak below
    # the initial level are never dropped. The persisted equity_curve and the
    # trade-day count stay end-of-day-based; style_analysis.daily_returns_from_curve
    # seeds the same baseline for attribution.
    baselined = [initial, *values]
    daily_returns = [
        later / earlier - 1.0
        for earlier, later in zip(baselined, baselined[1:])
        if earlier > 0
    ]
    sharpe = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        stdev = math.sqrt(
            sum((value - mean) ** 2 for value in daily_returns) / (len(daily_returns) - 1)
        )
        if stdev > 0:
            sharpe = mean / stdev * math.sqrt(TRADING_DAYS_PER_YEAR)
    peak = initial
    max_drawdown = 0.0
    for value in baselined:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    years = max(len(values), 1) / TRADING_DAYS_PER_YEAR
    annualized = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1.0 else -1.0

    realized = [
        order
        for order in orders
        if order.get("status") == "filled" and order.get("realized_pnl") is not None
    ]
    long_pnl = sum(float(order["realized_pnl"]) for order in realized)
    wins = sum(1 for order in realized if float(order["realized_pnl"]) > 0)
    per_stock = [
        {
            "symbol": order.get("symbol"),
            "exit_at": order.get("matched_at"),
            "exit_price": order.get("price"),
            "quantity": order.get("quantity"),
            "realized_pnl": order.get("realized_pnl"),
        }
        for order in realized
    ]

    status_counts: dict[str, int] = {}
    reject_counts: dict[str, int] = {}
    for order in orders:
        status = str(order.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = order.get("reason")
        if reason:
            reject_counts[str(reason)] = reject_counts.get(str(reason), 0) + 1
    # Entry/exit order lifecycle: the audited failure mode is a strategy whose
    # exit leg never fires, so splitting submissions and fills by direction
    # makes that visible in one probe.
    order_lifecycle = {group: {"total": 0, "filled": 0, "rejected": 0} for group in ("entry", "exit")}
    traded_notional = 0.0
    fees_paid = 0.0
    stamp_duty_paid = 0.0
    for order in orders:
        group = "exit" if str(order.get("action") or "") in _EXIT_ACTIONS else "entry"
        bucket = order_lifecycle[group]
        bucket["total"] += 1
        status = str(order.get("status") or "")
        # "total" is deliberately not a status name: a still-pending order must
        # not double-count if stats ever run before the book drains.
        if status in ("filled", "rejected"):
            bucket[status] += 1
        if status != "filled":
            continue
        price = order.get("price")
        if isinstance(price, (int, float)):
            traded_notional += float(price) * int(order.get("quantity") or 0)
        fees_paid += float(order.get("commission") or 0.0)
        stamp_duty_paid += float(order.get("stamp_duty") or 0.0)

    # Exposure diagnostics: gross = Σ EOD market value / same-day equity, read
    # off the cash/equity split the engine records for each replay day. Primitive
    # facts only — the audited failure mode is a structurally low-exposure
    # strategy whose returns get read as stock-picking quality while it mostly
    # held cash; the agent combines these with the benchmark block itself. Days
    # whose curve row carries no cash split are not measurable and are excluded
    # rather than reported as zero exposure.
    measured = [row for row in curve if isinstance(row.get("cash"), (int, float))]
    exposure_series = [
        (float(row["equity"]) - float(row["cash"])) / float(row["equity"])
        if float(row["equity"]) > 0
        else 0.0
        for row in measured
    ]
    exposure = {
        "avg_gross": float(sum(exposure_series) / len(exposure_series)) if exposure_series else 0.0,
        "max_gross": float(max(exposure_series)) if exposure_series else 0.0,
        "zero_position_days": int(sum(1 for value in exposure_series if value == 0.0)),
        "replay_days": len(exposure_series),
    }

    return {
        "initial_cash": initial,
        "final_equity": values[-1] if values else initial,
        "total_return": total_return,
        "long_return": float(long_pnl / initial) if initial > 0 else 0.0,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float(wins / len(realized)) if realized else 0.0,
        "exposure": exposure,
        "weekly_returns": _weekly_returns(curve, initial),
        "trade_count": len(realized),
        "turnover": float(traded_notional / initial) if initial > 0 else 0.0,
        "order_count": len(orders),
        "order_status_counts": status_counts,
        "order_lifecycle": order_lifecycle,
        "strategy_exit_fill_count": order_lifecycle["exit"]["filled"],
        "reject_counts": reject_counts,
        "fees_paid": fees_paid,
        "stamp_duty_paid": stamp_duty_paid,
        "decision_calls": len(result.inference_dates),
        "per_stock": per_stock,
        # Timing. ``replay_wall_seconds`` is the replay loop alone; the
        # evaluation backend adds its own setup phases to ``phase_seconds`` and
        # brackets the whole evaluation with started_at/finished_at.
        "replay_wall_seconds": round(float(result.wall_seconds), 3),
        "replayed_trade_days": len(curve),
        "phase_seconds": dict(result.phase_seconds),
    }


def finalize_summary_timing(
    summary: dict[str, object],
    *,
    started_at: str,
    setup_phases: Mapping[str, float] | None = None,
    nl_counters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Close one evaluation's timing block in its summary, in place.

    ``compute_return_stats`` only sees the replay loop. The evaluation backend
    is the component that knows when the whole evaluation started, what it did
    before and after the loop, and which NL service served it, so it closes the
    block here rather than each backend assembling its own field names.
    """
    phases = dict(summary.get("phase_seconds") or {})
    for name, seconds in (setup_phases or {}).items():
        phases[name] = round(phases.get(name, 0.0) + float(seconds), 3)
    if nl_counters:
        nl_wall = nl_counters.get("nl_wall_seconds")
        if isinstance(nl_wall, (int, float)) and not isinstance(nl_wall, bool):
            phases["nl"] = round(float(nl_wall), 3)
        summary.update(nl_counters)
    summary["phase_seconds"] = {name: phases[name] for name in sorted(phases)}
    summary["started_at"] = started_at
    summary["finished_at"] = utc_now_iso()
    return summary


def _weekly_returns(
    curve: tuple[dict[str, object], ...],
    initial: float,
) -> list[dict[str, object]]:
    """ISO-week return decomposition on the initial-equity baseline.

    Sub-period stability at a glance instead of one whole-window number.
    """
    week_end: dict[tuple[int, int], tuple[str, float]] = {}
    for row in curve:
        trade_date = str(row["trade_date"])
        day = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))
        iso = day.isocalendar()
        week_end[(iso.year, iso.week)] = (trade_date, float(row["equity"]))
    ordered = [week_end[key] for key in sorted(week_end)]
    starts = [initial, *(equity for _, equity in ordered[:-1])]
    return [
        {"week_end": trade_date, "return": float(equity / start - 1.0)}
        for (trade_date, equity), start in zip(ordered, starts)
        if start > 0
    ]
