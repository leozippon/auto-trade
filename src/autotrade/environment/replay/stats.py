"""Replay result container and the return-statistics reducer.

``ReplayResult`` is what one full daily replay produces; ``compute_return_stats``
reduces it to the ``detailed_return.json`` payload (docs/environment-design.md
§3.8). Pure result→dict math — no replay logic.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from autotrade.environment.runtime import utc_now_iso

# A-share trading calendar (~244 sessions/year); style_analysis annualizes with
# the same constant so detailed_return and the Barra-lite sidecar agree.
TRADING_DAYS_PER_YEAR = 244

# Position-reducing verbs; fills here are strategy-initiated exits.
_EXIT_ACTIONS = frozenset({"sell"})

# Sub-window granularity of every Validation / Test / Held-out result. One
# whole-window number cannot separate a persistent edge from one good month,
# and the audited folds reverse from quarter to quarter, so every result also
# carries the same metrics per calendar quarter of its replay window.
SUB_WINDOW_KIND = "quarter"
# Sub-window figures are read as a table, not compounded further: six decimals
# keep the block small enough to ride inline in an Agent observation.
_SUB_WINDOW_DIGITS = 6


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

    def to_record(self, *, start: str = "", end: str = "") -> dict[str, object]:
        return {
            "equity_curve": list(self.equity_curve),
            "executions": list(self.executions),
            "inference_dates": list(self.inference_dates),
            "pending_orders": list(self.pending_orders),
            "stats": compute_return_stats(self, start=start, end=end),
        }


def compute_return_stats(
    result: ReplayResult, *, start: str = "", end: str = ""
) -> dict[str, object]:
    """The minimum return statistics from docs/environment-design.md §3.8.

    ``start``/``end`` are the replay window that was requested (``YYYYMMDD``),
    which only the evaluation backend knows; they decide which quarters
    ``sub_windows`` reports as partial.
    """
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
    pnl_concentration = _pnl_concentration(realized)

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
        "pnl_concentration": pnl_concentration,
        "exposure": exposure,
        # Fixed-cost (four rows per replayed year) and therefore inline, unlike
        # weekly_returns / per_stock which scale with the window and the book.
        "sub_windows": sub_window_stats(curve, orders, initial=initial, start=start, end=end),
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


def _pnl_concentration(realized: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """How much of the realized gain rests on a few trades and one name.

    A window whose gains come from five trades, or from a single name, has not
    demonstrated a repeatable edge however good its total return looks (audited
    2026 Held-out: the top five trades carried 37.8% of the gross gains and one
    name nearly all of them). ``gross_losses`` is signed, so gains + losses is
    exactly ``net_realized``; the shares stay ``None`` rather than 0 when the
    window realized no gain at all.
    """

    values = [float(order["realized_pnl"]) for order in realized]
    by_symbol: dict[str, float] = {}
    for order in realized:
        symbol = str(order.get("symbol"))
        by_symbol[symbol] = by_symbol.get(symbol, 0.0) + float(order["realized_pnl"])
    gains = sum(value for value in values if value > 0)
    losses = sum(value for value in values if value < 0)
    top5 = sum(sorted(values, reverse=True)[:5])
    top_name = max(by_symbol.values(), default=0.0)
    return {
        "gross_gains": gains,
        "gross_losses": losses,
        "net_realized": gains + losses,
        "top5_share_of_gross_gains": top5 / gains if gains > 0 else None,
        "top_name_share_of_gross_gains": top_name / gains if gains > 0 else None,
    }


def sub_window_stats(
    curve: Sequence[Mapping[str, object]],
    executions: Sequence[Mapping[str, object]],
    *,
    initial: float,
    start: str = "",
    end: str = "",
) -> list[dict[str, object]]:
    """Per calendar quarter of the replay window, the same metrics as the whole.

    Each row compounds from the equity the quarter opened at (the previous
    quarter's close, or the initial equity), so the rows chain back to
    ``total_return``; ``turnover`` and ``trade_count`` keep the whole-window
    denominators and therefore sum to the whole-window figures.

    ``partial`` marks a quarter the requested replay window ``start..end``
    (``YYYYMMDD`` calendar dates) does not span end to end — interior quarters
    never are. It must be measured against those calendar bounds, not against
    the first and last trading day: a window opening on the first trading day
    of a quarter covers that quarter completely even though that day is rarely
    the first of the month. A caller that cannot state the requested window
    leaves both empty and gets the replayed span instead, which reports the two
    end quarters partial unless the replay itself reaches the calendar bounds.

    ``benchmark_return`` / ``excess_return`` stay ``None`` until the evaluation
    backend joins the benchmark series in (``attach_sub_window_benchmark``):
    the replay itself never reads an index.
    """

    rows = sorted(
        (row for row in curve if row.get("trade_date") is not None),
        key=lambda row: str(row["trade_date"]),
    )
    if not rows:
        return []
    buckets: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        buckets.setdefault(_quarter_key(str(row["trade_date"])), []).append(row)
    traded, exits = _quarter_order_totals(executions)
    keys = sorted(buckets)
    window_start = _window_bound(start, "start") or str(rows[0]["trade_date"])
    window_end = _window_bound(end, "end") or str(rows[-1]["trade_date"])
    out: list[dict[str, object]] = []
    opening = initial
    for key in keys:
        quarter_rows = buckets[key]
        quarter_start = str(quarter_rows[0]["trade_date"])
        quarter_end = str(quarter_rows[-1]["trade_date"])
        equities = [float(row["equity"]) for row in quarter_rows]
        closing = equities[-1]
        calendar_start, calendar_end = _quarter_bounds(*key)
        # Every quarter is compared against the window itself; an interior one
        # is inside both bounds and therefore never partial.
        partial = window_start > calendar_start or window_end < calendar_end
        out.append(
            {
                "kind": SUB_WINDOW_KIND,
                "label": f"{key[0]}Q{key[1]}",
                "start": quarter_start,
                "end": quarter_end,
                "trade_days": len(quarter_rows),
                "partial": partial,
                "return": _round(closing / opening - 1.0 if opening > 0 else 0.0),
                "benchmark_return": None,
                "excess_return": None,
                "sharpe": _round(_annualized_sharpe(opening, equities)),
                "max_drawdown": _round(_max_drawdown(opening, equities)),
                "turnover": _round(
                    traded.get(key, 0.0) / initial if initial > 0 else 0.0
                ),
                "trade_count": exits.get(key, 0),
            }
        )
        opening = closing
    return out


def attach_sub_window_benchmark(
    summary: dict[str, object], style_analysis: Mapping[str, object]
) -> dict[str, object]:
    """Fill each sub-window's benchmark and excess return, in place.

    ``compute_return_stats`` sees only the replay; the benchmark series is read
    once, host-side, by ``replay_style_analysis``. The evaluation backend owns
    both, so it closes the sub-window block here rather than each backend — or
    each reader — re-deriving the join. A slot without a usable benchmark
    leaves both fields ``None`` rather than reporting a fabricated zero.
    """

    rows = summary.get("sub_windows")
    if not isinstance(rows, list):
        return summary
    daily: dict[str, float] = {}
    series = style_analysis.get("benchmark_daily")
    if isinstance(series, Sequence) and not isinstance(series, (str, bytes)):
        for item in series:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                continue
            try:
                daily[str(item[0])] = float(item[1])
            except (TypeError, ValueError):
                continue
    if not daily:
        return summary
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = str(row.get("start") or "")
        end = str(row.get("end") or "")
        covered = [
            value for day, value in daily.items() if start <= day <= end
        ]
        if not covered:
            continue
        compounded = 1.0
        for value in covered:
            compounded *= 1.0 + value
        benchmark = compounded - 1.0
        own = row.get("return")
        row["benchmark_return"] = _round(benchmark)
        if isinstance(own, (int, float)) and not isinstance(own, bool):
            row["excess_return"] = _round(float(own) - benchmark)
    return summary


def attach_cost_sensitivity(
    summary: dict[str, object], slippage_bps: float
) -> dict[str, object]:
    """State what one more basis point of slippage per side costs, in place.

    A backtest is priced at one assumed slippage; the audited failure mode is an
    edge that only exists at that assumption. ``cost_per_bp_per_side`` is the
    equity fraction one extra bp per side costs at this window's turnover, so
    ``breakeven_extra_slippage_bps`` is how much worse execution the excess
    return survives and ``excess_at_2x_slippage`` is the excess left if the
    modelled slippage doubles. The block is always written — turnover and the
    profile's slippage are always known — but the two excess-derived fields
    stay ``None`` with a ``reason`` when they cannot be stated, rather than
    reporting a fabricated zero. The evaluation backend owns the broker profile,
    so it closes this block here, beside the benchmark join.
    """

    turnover = summary.get("turnover")
    cost_per_bp = (
        float(turnover) * 1e-4
        if isinstance(turnover, (int, float)) and not isinstance(turnover, bool)
        else 0.0
    )
    benchmark = summary.get("benchmark")
    excess = benchmark.get("excess_return") if isinstance(benchmark, Mapping) else None
    if not isinstance(excess, (int, float)) or isinstance(excess, bool):
        block: dict[str, object] = {
            "breakeven_extra_slippage_bps": None,
            "excess_at_2x_slippage": None,
            "reason": "no_benchmark_excess",
        }
    elif cost_per_bp == 0.0:
        block = {
            "breakeven_extra_slippage_bps": None,
            "excess_at_2x_slippage": float(excess),
            "reason": "no_turnover",
        }
    else:
        block = {
            "breakeven_extra_slippage_bps": (
                float(excess) / cost_per_bp if float(excess) > 0 else None
            ),
            "excess_at_2x_slippage": float(excess) - float(slippage_bps) * cost_per_bp,
        }
        if block["breakeven_extra_slippage_bps"] is None:
            block["reason"] = "excess_not_positive"
    summary["cost_sensitivity"] = {
        "slippage_bps": float(slippage_bps),
        "cost_per_bp_per_side": cost_per_bp,
        **block,
    }
    return summary


def _window_bound(value: str, name: str) -> str:
    """A requested replay bound as ``YYYYMMDD``, or empty when unstated."""

    text = str(value or "")
    if text and not (len(text) == 8 and text.isdigit()):
        raise ValueError(f"replay window {name} must be YYYYMMDD, got {value!r}")
    return text


def _quarter_key(trade_date: str) -> tuple[int, int]:
    return int(trade_date[:4]), (int(trade_date[4:6]) - 1) // 3 + 1


def _quarter_bounds(year: int, quarter: int) -> tuple[str, str]:
    first_month = 3 * (quarter - 1) + 1
    last_month = first_month + 2
    last_day = 31 if last_month in (3, 12) else 30
    return f"{year}{first_month:02d}01", f"{year}{last_month:02d}{last_day:02d}"


def _quarter_order_totals(
    executions: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Filled notional and realized exits per quarter, keyed by the fill date."""

    traded: dict[tuple[int, int], float] = {}
    exits: dict[tuple[int, int], int] = {}
    for order in executions:
        if str(order.get("status") or "") != "filled":
            continue
        day = _fill_date(order)
        if not day:
            continue
        key = _quarter_key(day)
        price = order.get("price")
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            traded[key] = traded.get(key, 0.0) + float(price) * int(
                order.get("quantity") or 0
            )
        if order.get("realized_pnl") is not None:
            exits[key] = exits.get(key, 0) + 1
    return traded, exits


def _fill_date(order: Mapping[str, object]) -> str:
    """``YYYYMMDD`` of one fill, from the broker's ISO ``matched_at``."""

    digits = "".join(char for char in str(order.get("matched_at") or "") if char.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _annualized_sharpe(opening: float, equities: Sequence[float]) -> float:
    previous = opening
    returns: list[float] = []
    for equity in equities:
        if previous > 0:
            returns.append(equity / previous - 1.0)
        previous = equity
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    stdev = math.sqrt(sum((value - mean) ** 2 for value in returns) / (len(returns) - 1))
    return mean / stdev * math.sqrt(TRADING_DAYS_PER_YEAR) if stdev > 0 else 0.0


def _max_drawdown(opening: float, equities: Sequence[float]) -> float:
    peak = opening
    worst = 0.0
    for equity in (opening, *equities):
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def _round(value: float) -> float:
    return round(float(value), _SUB_WINDOW_DIGITS)


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
