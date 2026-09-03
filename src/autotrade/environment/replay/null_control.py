"""Random-portfolio null control for one replay result.

The question: is a window's excess return due to WHICH names the strategy
picked, or only to its timing, sizing and exposure? The null keeps the trade
skeleton — the same entry and exit instants, the same money per round trip —
and replaces every name with a random one of comparable size, then replays the
script through the same Broker. If the observed excess sits inside the null
distribution, the name selection carried no information the timing did not
already carry.

Everything here is host-side and pure: the caller supplies the result, the
replay slot's daily frame, the benchmark series and the Broker profile; no
sandbox, no I/O, no raw data lake. The null is informational and gates nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.broker_core import (
    LOT_SIZE,
    STAR_MIN_LOT_SIZE,
    is_bse_market,
    is_star_market,
)
from autotrade.environment.strategy import CN_TZ, StrategyContext, StrategySchedule

from .engine import run_daily_replay
from .market import DailyMarketData
from .stats import ReplayResult, compute_return_stats

# How a replacement name is drawn: uniformly inside the original's circulating
# market-cap decile of the entry day's own cross-section.
MATCHING = "circ_mv_decile"
_DECILES = 10
_DIGITS = 6


@dataclass(frozen=True)
class RoundTrip:
    """One FIFO-paired entry and its exit; ``exit_at`` is None when still open."""

    symbol: str
    quantity: int
    price: float
    entry_at: datetime
    exit_at: datetime | None

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def entry_date(self) -> str:
        return self.entry_at.strftime("%Y%m%d")

    @property
    def exit_date(self) -> str | None:
        return None if self.exit_at is None else self.exit_at.strftime("%Y%m%d")


def trade_skeleton(executions: Sequence[Mapping[str, object]]) -> list[RoundTrip]:
    """Filled buys and sells paired FIFO per symbol, splitting partial exits."""

    fills = [
        (
            _cn_datetime(record["matched_at"]),
            str(record["symbol"]),
            str(record["action"]),
            int(record["quantity"]),
            float(record["price"]),
        )
        for record in executions
        if record.get("status") == "filled"
    ]
    # Stable: fills that share an instant keep the Broker's own execution order.
    fills.sort(key=lambda fill: fill[0])
    open_lots: dict[str, list[list[object]]] = {}
    trips: list[RoundTrip] = []
    for matched_at, symbol, action, quantity, price in fills:
        if action == "buy":
            open_lots.setdefault(symbol, []).append([quantity, price, matched_at])
            continue
        lots = open_lots.get(symbol, [])
        remaining = quantity
        while remaining > 0:
            if not lots:
                raise ValueError(
                    f"sell of {symbol} at {matched_at.isoformat()} exceeds its filled buys"
                )
            lot = lots[0]
            taken = min(remaining, int(lot[0]))
            trips.append(
                RoundTrip(
                    symbol=symbol,
                    quantity=taken,
                    price=float(lot[1]),
                    entry_at=lot[2],  # type: ignore[arg-type]
                    exit_at=matched_at,
                )
            )
            lot[0] = int(lot[0]) - taken
            remaining -= taken
            if lot[0] == 0:
                lots.pop(0)
    for symbol, lots in open_lots.items():
        for quantity, price, entry_at in lots:
            trips.append(
                RoundTrip(
                    symbol=symbol,
                    quantity=int(quantity),
                    price=float(price),
                    entry_at=entry_at,  # type: ignore[arg-type]
                    exit_at=None,
                )
            )
    trips.sort(key=lambda trip: (trip.entry_at, trip.symbol))
    return trips


def sample_null_orders(
    skeleton: Sequence[RoundTrip],
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, list[dict[str, object]]]:
    """One null draw of the whole skeleton, as orders keyed by execution date."""

    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe) for trip in skeleton]
    return _orders_from_pools(skeleton, pools, rng)


def run_null_control(
    result: ReplayResult,
    frame: pd.DataFrame,
    benchmark: Mapping[str, float],
    profile: BrokerProfile,
    schedule: StrategySchedule,
    *,
    k: int = 500,
    seed: int,
    step: tuple[str, str] | None = None,
) -> dict[str, object]:
    """Replay ``k`` random-name copies of ``result``'s skeleton and rank it.

    ``benchmark`` maps ``YYYYMMDD`` to the benchmark's daily return (see
    ``style._slot_benchmark``); the window's benchmark return is compounded over
    the trading days the replay actually covers, so it is the same constant for
    the observed run and every null run. ``schedule`` contributes the decision
    clock the replayed strategy used — the null's orders are drawn up front, so
    its cadence is not reused. ``step`` is an inclusive ``(start, end)``
    sub-window measured from the equity curves.
    """

    if k < 1:
        raise ValueError("k must be a positive integer")
    dates = [str(row["trade_date"]) for row in result.equity_curve]
    market = DailyMarketData(frame)
    if tuple(dates) != market.trade_dates:
        raise ValueError("result equity curve and replay frame cover different trading days")
    skeleton = trade_skeleton(result.executions)
    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe) for trip in skeleton]
    rng = np.random.default_rng(seed)

    window_benchmark = _benchmark_return(dates, benchmark)
    observed = _total_return(result) - window_benchmark
    bounds = None if step is None else _step_bounds(step)
    if bounds is not None:
        start, end = bounds
        step_benchmark = _benchmark_return(
            [date for date in dates if start <= date <= end], benchmark
        )
        observed_step = _sub_window_return(result.equity_curve, start, end) - step_benchmark

    # The draw is fixed before the replay starts, so the null needs exactly one
    # decision point. Keeping the replayed strategy's own cadence would rebuild
    # the point-in-time bar view the script never reads on every trading day,
    # which is ~85% of a null replay's wall clock.
    null_schedule = StrategySchedule("year", schedule.inference_time)
    excesses: list[float] = []
    step_excesses: list[float] = []
    rejects: list[int] = []
    for _ in range(k):
        orders = _orders_from_pools(skeleton, pools, rng)
        run = run_daily_replay(
            daily=market,
            strategy=_scripted_strategy(orders),
            schedule=null_schedule,
            profile=profile,
        )
        excesses.append(_total_return(run) - window_benchmark)
        rejects.append(sum(1 for record in run.executions if record["status"] != "filled"))
        if bounds is not None:
            step_excesses.append(
                _sub_window_return(run.equity_curve, start, end) - step_benchmark
            )

    block: dict[str, object] = {
        "k": k,
        "seed": seed,
        "matched": MATCHING,
        **_distribution(observed, excesses),
        "rejects_mean": round(sum(rejects) / k, 3),
    }
    if bounds is not None:
        block["step"] = {
            "start": start,
            "end": end,
            **_distribution(observed_step, step_excesses),
        }
    return block


class _Universe:
    """The replay slot's daily cross-section, by trading day."""

    def __init__(self, frame: pd.DataFrame) -> None:
        symbol_column = "symbol" if "symbol" in frame.columns else "ts_code"
        missing = [
            name for name in (symbol_column, "trade_date", "open") if name not in frame.columns
        ]
        if missing:
            raise ValueError(f"replay frame missing columns: {missing}")
        opens = pd.to_numeric(frame["open"], errors="coerce")
        rows = pd.DataFrame(
            {
                "symbol": frame[symbol_column].astype(str).str.strip(),
                "trade_date": [_date_text(value) for value in frame["trade_date"]],
                "open": opens,
                "circ_mv": (
                    pd.to_numeric(frame["circ_mv"], errors="coerce")
                    if "circ_mv" in frame.columns
                    else np.nan
                ),
            }
        )
        # A name without a usable opening price cannot be sized, so it is not a
        # candidate on that day at all.
        rows = rows[np.isfinite(rows["open"]) & (rows["open"] > 0)]
        self._days = {
            date: group.set_index("symbol")
            for date, group in rows.groupby("trade_date", sort=False)
        }
        self._deciles: dict[str, pd.Series] = {}

    def day(self, date: str) -> pd.DataFrame:
        day = self._days.get(date)
        if day is None:
            raise ValueError(f"replay frame has no tradable bar on {date}")
        return day

    def deciles(self, date: str) -> pd.Series:
        """Circulating market-cap decile of that day's own cross-section.

        Ranked with ties shared, so equal ``circ_mv`` names always land in the
        same bucket and a degenerate cross-section stays one bucket instead of
        raising.
        """

        cached = self._deciles.get(date)
        if cached is None:
            values = self.day(date)["circ_mv"].dropna()
            ranks = values.rank(method="min", pct=True).to_numpy()
            codes = np.minimum((ranks * _DECILES).astype(int), _DECILES - 1)
            cached = pd.Series(codes, index=values.index, dtype=int)
            self._deciles[date] = cached
        return cached


def _candidate_pool(trip: RoundTrip, universe: _Universe) -> tuple[tuple[str, float], ...]:
    """Replacement names for one round trip, with their entry-day open price."""

    entry = universe.day(trip.entry_date)
    deciles = universe.deciles(trip.entry_date)
    code = deciles.get(trip.symbol)
    names = entry.index if code is None else deciles.index[deciles.to_numpy() == int(code)]
    if trip.exit_date is not None:
        names = names.intersection(universe.day(trip.exit_date).index)
    names = names.drop(trip.symbol, errors="ignore")
    if names.empty:
        raise ValueError(
            f"no replacement name for {trip.symbol} entered {trip.entry_date}"
            f" and exited {trip.exit_date}"
        )
    return tuple(zip(names.tolist(), entry.loc[names, "open"].tolist(), strict=True))


def _orders_from_pools(
    skeleton: Sequence[RoundTrip],
    pools: Sequence[tuple[tuple[str, float], ...]],
    rng: np.random.Generator,
) -> dict[str, list[dict[str, object]]]:
    orders: dict[str, list[dict[str, object]]] = {}
    for trip, pool in zip(skeleton, pools, strict=True):
        symbol, entry_open = pool[int(rng.integers(len(pool)))]
        # Sized from the entry day's open only: the null learns nothing the
        # original position did not already know when it was opened.
        quantity = _lot_quantity(symbol, int(trip.notional // entry_open))
        if quantity <= 0:
            continue
        _queue(orders, symbol, "buy", quantity, trip.entry_at)
        if trip.exit_at is not None:
            _queue(orders, symbol, "sell", quantity, trip.exit_at)
    for day in orders.values():
        # Exits before entries at one instant, so a cash-bound null is not
        # rejected for money its own sales are about to release.
        day.sort(key=lambda order: (str(order["execute_at"]), order["action"] == "buy"))
    return orders


def _queue(
    orders: dict[str, list[dict[str, object]]],
    symbol: str,
    action: str,
    quantity: int,
    when: datetime,
) -> None:
    orders.setdefault(when.strftime("%Y%m%d"), []).append(
        {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "execute_at": when.isoformat(),
        }
    )


def _lot_quantity(symbol: str, shares: int) -> int:
    """``shares`` rounded down to what that board lets a buy declare."""

    if is_star_market(symbol):
        return shares if shares >= STAR_MIN_LOT_SIZE else 0
    if is_bse_market(symbol):
        return shares if shares >= LOT_SIZE else 0
    return shares - shares % LOT_SIZE


def _scripted_strategy(
    orders: Mapping[str, list[dict[str, object]]],
) -> Callable[[StrategyContext], list[dict[str, object]]]:
    """Hand the whole draw to the environment at the first decision.

    The engine's inbox releases each order at its own ``execute_at`` and the
    Broker applies every gate there, so submitting the script up front changes
    no fill — it only frees the null from the replayed strategy's inference
    cadence, which would otherwise silently drop orders on non-decision days.
    """

    queued = [order for date in sorted(orders) for order in orders[date]]
    sent = False

    def strategy(_context: StrategyContext) -> list[dict[str, object]]:
        nonlocal sent
        if sent:
            return []
        sent = True
        return queued

    return strategy


def _distribution(observed: float, values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "observed_excess": round(observed, _DIGITS),
        "null_excess_mean": round(float(array.mean()), _DIGITS),
        "null_excess_p05": round(float(np.percentile(array, 5)), _DIGITS),
        "null_excess_p95": round(float(np.percentile(array, 95)), _DIGITS),
        "excess_percentile": round(float((array <= observed).mean()), _DIGITS),
    }


def _total_return(result: ReplayResult) -> float:
    return float(compute_return_stats(result)["total_return"])


def _sub_window_return(
    curve: Sequence[Mapping[str, object]], start: str, end: str
) -> float:
    """Equity return over ``start..end``, based on the last close before it."""

    base: float | None = None
    final: float | None = None
    for row in curve:
        date = str(row["trade_date"])
        if date < start:
            base = float(row["equity"])
        elif date <= end:
            final = float(row["equity"])
    if final is None:
        raise ValueError(f"step window {start}..{end} contains no replayed trading day")
    if base is None:
        base = float(curve[0]["initial_equity"])
    return final / base - 1.0


def _benchmark_return(dates: Sequence[str], benchmark: Mapping[str, float]) -> float:
    total = 1.0
    for date in dates:
        daily = benchmark.get(date)
        if daily is not None:
            total *= 1.0 + float(daily)
    return total - 1.0


def _step_bounds(step: tuple[str, str]) -> tuple[str, str]:
    start, end = (_date_text(value) for value in step)
    if start > end:
        raise ValueError(f"step window start {start} is after its end {end}")
    return start, end


def _date_text(value: object) -> str:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return text
    return pd.Timestamp(text).strftime("%Y%m%d")


def _cn_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value)).astimezone(CN_TZ)


__all__ = [
    "MATCHING",
    "RoundTrip",
    "run_null_control",
    "sample_null_orders",
    "trade_skeleton",
]
