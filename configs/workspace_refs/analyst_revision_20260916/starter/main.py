"""Anchor strategy for family 1: consensus current-fiscal-year EPS revision.

Not a submittable artifact.  It is the smallest contract-compliant implementation
of `families.md` family 1, written so a fold agent can read the shape and then
rewrite it against the run's own data contract.

Shape: at 08:30 on the first trading day of a month, rank analyst-covered names
by their size/momentum/turnover/volatility-neutral 20d-vs-40d consensus EPS
revision, hold the top 15 equally weighted for roughly 20 trading days, and
rotate the whole basket at the next rebalance.  Sells go out at 09:30 and buys
at 15:00 of the same day so that the sale proceeds have settled inside the
Broker before the buys price; the buy budget uses a declared, haircut estimate
of those proceeds because the account snapshot does not move during the call.

Column names follow the landed `report_rc` events slice (commit 65edfae).  That
slice is opt-in, so confirm `report_rc` is in this run's `events.datasets` and
that the units match `unit_reference.json` before any backtest; abstain if it is
absent rather than falling back to the title-only text domain (plan step 1).
"""

from __future__ import annotations

import pandas as pd

from lib.revision import (
    consensus_revision,
    controls,
    eligible,
    load_daily,
    load_reports,
    load_universe,
    neutralize,
)

# Declared invariants (README "执行假设"), not fitted parameters.
BASKET_SIZE = 15
MIN_ADV20 = 30_000_000.0        # CNY, from normalized daily.amount
PRICE_CAP = 40.0                # CNY; the pack requires this to be chosen once per fold
MIN_LISTED_DAYS = 250
COST_BUFFER = 0.97
PROCEED_HAIRCUT = 0.95          # declared discount on estimated sale proceeds
LOT = 100
CONTROL_COLUMNS = ["log_size", "mom20", "mom60", "turn20", "vol20"]


def _is_rebalance_day(inference_at, latest_visible):
    """First trading day of a new month: ~20 trading days between rebalances.

    Derived from the visible PIT calendar alone, so a cold worker reproduces it.
    """
    return (inference_at.year, inference_at.month) != (latest_visible.year, latest_visible.month)


def _target_basket(context):
    """Return (targets, buyable prices, last visible close for every listed name).

    The third value covers held names that have since dropped out of the
    eligible pool: their exit proceeds still have to be estimated, so the price
    map is the full cross-section, not the filtered pool.
    """
    empty = pd.Series(dtype="float64")
    daily = load_daily(context)
    if daily.empty:
        return None, empty, empty
    trading_days = pd.DatetimeIndex(daily["trade_date"].drop_duplicates().sort_values().to_numpy())
    latest = trading_days[-1]
    if not _is_rebalance_day(context.inference_at, latest):
        return None, empty, empty

    control_frame = controls(daily, trading_days)
    if control_frame.empty:
        return None, empty, empty
    all_prices = control_frame["close"].astype(float)
    pool = eligible(
        control_frame, load_universe(context), MIN_ADV20, PRICE_CAP, MIN_LISTED_DAYS, latest
    )
    if pool.empty:
        return None, empty, all_prices

    revision = consensus_revision(load_reports(context), trading_days)
    if revision.empty:
        return None, empty, all_prices
    score = neutralize(revision, pool, CONTROL_COLUMNS)
    if score.empty:
        return None, empty, all_prices
    return list(score.index[:BASKET_SIZE]), pool["close"].astype(float), all_prices


def generate_orders(context):
    open_at = context.inference_at.replace(hour=9, minute=30, second=0, microsecond=0)
    close_at = context.inference_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if context.inference_at > open_at:
        return []

    targets, prices, all_prices = _target_basket(context)
    if targets is None:
        return []

    held = dict(context.account.positions)
    orders = []

    proceeds = 0.0
    for symbol in sorted(held):
        if symbol in targets:
            continue
        quantity = int(held[symbol])
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "execute_at": open_at.isoformat(),
                "reason": "revision_rebalance_exit",
            }
        )
        if symbol in all_prices.index:
            proceeds += quantity * float(all_prices[symbol]) * PROCEED_HAIRCUT

    buys = [s for s in targets if s not in held and s in prices.index]
    if not buys:
        return orders

    remaining = (float(context.account.cash) + proceeds) * COST_BUFFER
    for index, symbol in enumerate(buys):
        price = float(prices[symbol])
        if price <= 0:
            continue
        quantity = int(remaining / (len(buys) - index) / price // LOT * LOT)
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "execute_at": close_at.isoformat(),
                "reason": "consensus_eps_revision_top15",
            }
        )
        remaining -= quantity * price
    return orders
