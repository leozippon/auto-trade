"""Anchor strategy for family 1: seasoned margin-roster inclusion, 40-day hold.

This is the smallest artifact that expresses the family under the formal
contract. It is not a tuned candidate and carries no backtest result; it exists
so the first fold starts from a package whose PIT alignment is already right:

* visibility comes only from ``available_at <= context.inference_at``;
* the newest usable roster is ``T-1``, so an inclusion is ``R_{T-1} \\ R_{T-2}``;
* ``margin_detail`` is read at ``T-2`` and earlier only, which is exactly the
  window the F4 confirmation needs;
* the 40-day holding lifecycle is recomputed from the visible roster history,
  so the strategy needs neither ``fit`` nor a state directory.

What it deliberately does not do: no scoring, no neutralisation, no matched
control, no batch stratification, no exclusion mirror. Those are the
candidate's job and every one of them is specified in ``families.md``.
"""

from __future__ import annotations

import roster

LOOKBACK_DAYS = 400        # calendar days of events/daily needed for a 40-day book
CASH_BUFFER = 0.97         # room for fees and the move to the open price
LOT_SIZE = 100
OPEN_MINUTE = 9 * 60 + 30
CLOSE_MINUTE = 15 * 60


def _execute_at(context):
    """Next same-day daily price timestamp at or after the decision time."""
    now = context.inference_at
    minutes = now.hour * 60 + now.minute
    if minutes <= OPEN_MINUTE:
        return now.replace(hour=9, minute=30, second=0, microsecond=0)
    if minutes <= CLOSE_MINUTE:
        return now.replace(hour=15, minute=0, second=0, microsecond=0)
    return None


def generate_orders(context):
    execute_at = _execute_at(context)
    if execute_at is None:
        return []

    events = roster.visible_events(context, LOOKBACK_DAYS)
    dates = roster.roster_calendar(events)
    if len(dates) < roster.HOLD_DAYS + roster.CONFIRM_DAYS + 2:
        return []

    daily = roster.visible_daily(context, LOOKBACK_DAYS)
    universe = roster.visible_universe(context)
    target = roster.target_book(roster.inclusion_cohort(events, universe, daily, dates))

    positions = {symbol: int(quantity) for symbol, quantity in context.account.positions.items()}
    held = {symbol for symbol, quantity in positions.items() if quantity > 0}
    stamp = execute_at.isoformat()
    orders = []

    for symbol in sorted(held - set(target)):
        orders.append(
            {
                "symbol": symbol,
                "action": "sell",
                "quantity": positions[symbol],
                "execute_at": stamp,
                "reason": "margin_secs inclusion hold of 40 trading days elapsed",
            }
        )

    buys = [symbol for symbol in target if symbol not in held]
    if not buys:
        return orders

    prices = roster.last_close(daily, set(buys) | held)
    equity = float(context.account.cash)
    for symbol in held:
        price = prices.get(symbol)
        if price is not None:
            equity += positions[symbol] * price
    budget = equity * CASH_BUFFER / roster.BASKET_CAP
    remaining = float(context.account.cash) * CASH_BUFFER

    for symbol in buys:
        price = prices.get(symbol)
        if price is None:
            continue
        lots = int(min(budget, remaining) // (price * LOT_SIZE))
        if lots < 1:
            continue
        quantity = lots * LOT_SIZE
        remaining -= quantity * price
        orders.append(
            {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "execute_at": stamp,
                "reason": "seasoned margin_secs inclusion confirmed on the T-1 roster",
            }
        )
    return orders
