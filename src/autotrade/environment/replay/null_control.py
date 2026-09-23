"""Random-portfolio null control for one replay result, and its zero-skill panel.

The question: is a window's excess return due to WHICH names the strategy
picked, or only to its timing, sizing and exposure? The null keeps the trade
skeleton — the same entry and exit instants, the same money per round trip —
and replaces every name with a random one the same money could have bought on
the same side of the benchmark's membership, then replays the script through
the same Broker. If the observed excess sits inside the null distribution, the
name selection carried no information the timing did not already carry.

The draws' equal-weight mean daily return is the zero-skill panel composite:
what the account earns in the candidate's own shape — its review calendar, its
seats, its money per seat, its costs — with no skill in the names. The verdict
grades a strategy on its daily return minus that composite
(``pipelines/verdict.py``); the percentile beside it stays descriptive.

Everything here is host-side and pure: the caller supplies the result, the
replay slot's daily frame, the benchmark series, the benchmark membership and
the Broker profile; no sandbox, no I/O, no raw data lake. Every draw replays
through a Broker of its own after the graded replay has finished, so nothing
here can reach the result it measures.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Collection, Mapping, Sequence
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
from .stats import ReplayResult, total_return_from_curve
from .style import daily_returns_from_curve

# Draws of one null control. Twenty put the composite's standard error at
# 1.4-3.2 pp/yr against the 4-11 pp/yr active returns it is subtracted from,
# at a twenty-fifth of what a percentile-grade 500 would cost on every replay.
PANEL_DRAWS = 20
# How a replacement name is drawn: uniformly among the entry day's names that
# one board lot of this round trip's own money can buy and that sit on the
# original's side of the benchmark's membership. Without a membership table the
# side is the original's circulating market-cap decile of that day instead.
MATCHED_MEMBERSHIP = "index_membership+affordability"
MATCHED_DECILE = "circ_mv_decile+affordability"
_DECILES = 10
_DIGITS = 6


class NullControlSetupError(RuntimeError):
    """The null control could not be set up, so it drew and measured nothing.

    Raised strictly before the first draw: the result's replay slots could not
    be resolved, the tables they hold could not be read, or the benchmark
    membership they mount has no section before a round trip's entry. A caller that
    charges a budget for the host compute a null control spends gives that
    charge back on this error, because none of it was spent.
    """


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


def trade_skeleton(
    executions: Sequence[Mapping[str, object]],
) -> tuple[list[RoundTrip], int]:
    """Filled buys and sells paired FIFO per symbol, splitting partial exits.

    Also returns the sold shares no filled buy accounts for: shares an ex-date
    settlement created (bonus or transfer issues) are not a round trip of their
    own — the null's replacement names receive their own settlements in the
    replay — so they are left unpaired and only counted, never raised on.
    """

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
    unpaired_sell_shares = 0
    for matched_at, symbol, action, quantity, price in fills:
        if action == "buy":
            open_lots.setdefault(symbol, []).append([quantity, price, matched_at])
            continue
        lots = open_lots.get(symbol, [])
        remaining = quantity
        while remaining > 0:
            if not lots:
                unpaired_sell_shares += remaining
                break
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
    return trips, unpaired_sell_shares


def run_null_control(
    result: ReplayResult,
    frame: pd.DataFrame,
    benchmark: Mapping[str, float],
    profile: BrokerProfile,
    schedule: StrategySchedule,
    *,
    k: int = PANEL_DRAWS,
    seed: int,
    step: tuple[str, str] | None = None,
    corporate_actions: pd.DataFrame | None = None,
    membership: Mapping[str, Collection[str]] | None = None,
    panel: bool = False,
) -> dict[str, object]:
    """Replay ``k`` random-name copies of ``result``'s skeleton and rank it.

    ``benchmark`` maps ``YYYYMMDD`` to the benchmark's daily return (see
    ``style.slot_benchmark``); the window's benchmark return is compounded over
    the trading days the replay actually covers, so it is the same constant for
    the observed run and every null run. ``schedule`` contributes the decision
    clock the replayed strategy used — the null's orders are drawn up front, so
    its cadence is not reused. ``step`` is an inclusive ``(start, end)``
    sub-window measured from the equity curves. ``corporate_actions`` is the
    slot's ex-date table, so a null holding is settled through the same
    ex-dates as the observed one; a formal slot always passes it, and None is
    only for synthetic frames in unit tests.

    ``membership`` maps a cross-section date (``YYYYMMDD``) to the benchmark's
    constituents published on it; a round trip is matched on the latest one
    dated before its entry day, which is the one its decision could read. None
    means the table is not mounted, and the draw falls back to the float-cap
    decile; ``matched`` records which rule applied. A mounted table -- even an
    empty one -- is never replaced by the decile: a round trip with no section
    dated before its entry fails the whole call before the first draw
    (``NullControlSetupError``).

    ``panel`` adds the draws' daily return series to the block: ``panel_daily``
    is their equal-weight mean per trading day and ``panel_draw_sd`` the
    cross-draw standard deviation. A result that never filled a trade has an
    idle cash account as its zero-skill copy, so its panel is zero every day.
    """

    if k < 1:
        raise ValueError("k must be a positive integer")
    dates = [str(row["trade_date"]) for row in result.equity_curve]
    market = DailyMarketData(frame, corporate_actions)
    if tuple(dates) != market.trade_dates:
        raise ValueError("result equity curve and replay frame cover different trading days")
    members = None if membership is None else _Membership(membership)
    matched = MATCHED_DECILE if members is None else MATCHED_MEMBERSHIP
    skeleton, unpaired_sell_shares = trade_skeleton(result.executions)
    if not skeleton:
        # Nothing was ever filled: every random-name replay would be the same
        # idle cash account, and the right-inclusive percentile would read 1.0
        # for a result that picked no names at all.
        idle: dict[str, object] = {
            "status": "unavailable",
            "reason": "no_filled_trades",
            "k": 0,
            "seed": seed,
            "matched": matched,
            "excess_percentile": None,
        }
        if panel:
            returns = daily_returns_from_curve(result.equity_curve)
            idle["panel_daily"] = [[date, 0.0] for date, _value in returns]
            idle["panel_draw_sd"] = [[date, 0.0] for date, _value in returns]
        return idle
    universe = _Universe(frame)
    pools = [_candidate_pool(trip, universe, members) for trip in skeleton]
    rng = np.random.default_rng(seed)

    window_benchmark = _benchmark_return(dates, benchmark)
    observed = total_return_from_curve(result.equity_curve) - window_benchmark
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
    dropped: list[int] = []
    series: list[list[tuple[str, float]]] = []
    for _ in range(k):
        orders, dropped_trips = _orders_from_pools(skeleton, pools, rng)
        dropped.append(dropped_trips)
        run = run_daily_replay(
            daily=market,
            strategy=_scripted_strategy(orders),
            schedule=null_schedule,
            profile=profile,
        )
        excesses.append(total_return_from_curve(run.equity_curve) - window_benchmark)
        rejects.append(sum(1 for record in run.executions if record["status"] != "filled"))
        if panel:
            series.append(daily_returns_from_curve(run.equity_curve))
        if bounds is not None:
            step_excesses.append(
                _sub_window_return(run.equity_curve, start, end) - step_benchmark
            )

    block: dict[str, object] = {
        "k": k,
        "seed": seed,
        "matched": matched,
        **_distribution(observed, excesses),
        "rejects_mean": round(sum(rejects) / k, 3),
        "round_trips": len(skeleton),
        # Round trips no draw could deploy: not one matched name had a board
        # lot this trip's money could buy, so the null ran with less capital
        # than the result it is compared against.
        "dropped_trips_mean": round(sum(dropped) / k, 3),
        # Sold shares no filled buy accounts for: created by the observed
        # run's ex-date settlements, so the skeleton carries only the bought
        # shares and each null name earns its own settlements instead.
        "unpaired_sell_shares": unpaired_sell_shares,
    }
    if bounds is not None:
        block["step"] = {
            "start": start,
            "end": end,
            **_distribution(observed_step, step_excesses),
        }
    if panel:
        days = [date for date, _value in series[0]]
        if any([date for date, _value in draw] != days for draw in series):
            raise ValueError("null draws cover different trading days")
        values = np.asarray([[value for _date, value in draw] for draw in series])
        spread = values.std(axis=0, ddof=1) if k > 1 else np.zeros(len(days))
        block["panel_daily"] = [
            [date, float(value)] for date, value in zip(days, values.mean(axis=0), strict=True)
        ]
        block["panel_draw_sd"] = [
            [date, float(value)] for date, value in zip(days, spread, strict=True)
        ]
    return block


class _Membership:
    """The benchmark's constituent cross-sections, read by entry day."""

    def __init__(self, membership: Mapping[str, Collection[str]]) -> None:
        self._dates = sorted(_date_text(date) for date in membership)
        self._members = {
            _date_text(date): frozenset(str(code) for code in codes)
            for date, codes in membership.items()
        }

    def before(self, date: str) -> frozenset[str]:
        """The latest cross-section dated before ``date``.

        A cross-section is published after its own close, so the one a
        decision could read is dated strictly before the day it trades on.
        """

        position = bisect_left(self._dates, date)
        if position == 0:
            # Drawing on the float-cap decile instead would grade this span
            # against another panel than the arm's other spans, so nothing is
            # drawn: the pools are built before the first draw.
            raise NullControlSetupError(
                f"no benchmark membership is dated before {date}: the arm mounts index_weight, "
                + (
                    f"but its first section of the benchmark is {self._dates[0]}"
                    if self._dates
                    else "but no section of the benchmark is in the replayed slots"
                )
                + "; the pinned release does not reach back far enough for this span"
            )
        return self._members[self._dates[position - 1]]


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
        boards = {symbol: _board(symbol) for symbol in rows["symbol"].unique()}
        rows = rows.assign(board=rows["symbol"].map(boards))
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


def _candidate_pool(
    trip: RoundTrip, universe: _Universe, members: _Membership | None = None
) -> tuple[tuple[str, int], ...]:
    """Replacement names for one round trip, with the shares its money buys.

    Affordability belongs to the pool, not to the draw: a name whose board lot
    this trip's money cannot buy was never a candidate, so drawing it and then
    dropping the trip would run the null with less capital than the result.
    Sized from the entry day's open only: the null learns nothing the original
    position did not already know when it was opened.
    """

    entry = universe.day(trip.entry_date)
    if members is None:
        deciles = universe.deciles(trip.entry_date)
        code = deciles.get(trip.symbol)
        names = entry.index if code is None else deciles.index[deciles.to_numpy() == int(code)]
    else:
        constituents = members.before(trip.entry_date)
        inside = entry.index.isin(constituents)
        names = entry.index[inside == (trip.symbol in constituents)]
    if trip.exit_date is not None:
        names = names.intersection(universe.day(trip.exit_date).index)
    names = names.drop(trip.symbol, errors="ignore")
    bars = entry.loc[names]
    shares = np.floor_divide(trip.notional, bars["open"].to_numpy()).astype(int)
    quantities = _lot_quantities(shares, bars["board"].to_numpy())
    return tuple(
        (symbol, int(quantity))
        for symbol, quantity in zip(names.tolist(), quantities, strict=True)
        if quantity > 0
    )


def _orders_from_pools(
    skeleton: Sequence[RoundTrip],
    pools: Sequence[tuple[tuple[str, int], ...]],
    rng: np.random.Generator,
) -> tuple[dict[str, list[dict[str, object]]], int]:
    """One draw, and how many round trips it could not deploy.

    A trip with an empty pool is dropped; the count travels with the draw so
    an under-deployed null is reported instead of silently shrinking the null
    distribution.
    """

    orders: dict[str, list[dict[str, object]]] = {}
    dropped = 0
    for trip, pool in zip(skeleton, pools, strict=True):
        if not pool:
            dropped += 1
            continue
        symbol, quantity = pool[int(rng.integers(len(pool)))]
        _queue(orders, symbol, "buy", quantity, trip.entry_at)
        if trip.exit_at is not None:
            _queue(orders, symbol, "sell", quantity, trip.exit_at)
    for day in orders.values():
        # Exits before entries at one instant, so a cash-bound null is not
        # rejected for money its own sales are about to release.
        day.sort(key=lambda order: (str(order["execute_at"]), order["action"] == "buy"))
    return orders, dropped


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


_STAR, _BSE, _MAIN = "star", "bse", "main"


def _board(symbol: str) -> str:
    if is_star_market(symbol):
        return _STAR
    return _BSE if is_bse_market(symbol) else _MAIN


def _lot_quantities(shares: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """``shares`` rounded down to what each name's board lets a buy declare.

    STAR takes any size from 200 shares and BSE any size from 100; every other
    board takes whole lots of 100. Zero where not one declaration fits.
    """

    declared = np.where(boards == _MAIN, shares - shares % LOT_SIZE, shares)
    minimum = np.where(boards == _STAR, STAR_MIN_LOT_SIZE, LOT_SIZE)
    return np.where(shares >= minimum, declared, 0)


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
    "MATCHED_DECILE",
    "MATCHED_MEMBERSHIP",
    "PANEL_DRAWS",
    "NullControlSetupError",
    "RoundTrip",
    "run_null_control",
    "trade_skeleton",
]
