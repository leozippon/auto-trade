"""Anchor for family 1: the 20-day amount-weighted block-trade discount.

Reference implementation only. It shows the PIT reads, the unit conversions,
the two-level aggregation and the order shape that `families.md` requires; it
has never been backtested and is not a baseline. Do not copy this directory
into `output/` -- write the formal artifact yourself.

MODE switches the same book between the signal leg and the mandatory
within-eligible control leg of family 6: `"signal"` buys the top-ranked
discounts, `"control"` buys the names sitting at the cross-sectional median of
the very same eligible set. Run both and read the difference.
"""

import numpy as np
import pandas as pd

from lib.blocks import (
    cross_section,
    neutral_score,
    read_blocks,
    read_daily,
    read_universe,
    select,
    window_discount,
)

MODE = "signal"  # "signal" | "control"

BASKET_SIZE = 15
DISCOUNT_WINDOW = 20  # trading days of visible block-trade history
DAILY_LOOKBACK_DAYS = 150  # calendar days; covers 20 trading days of ADV plus the window
MIN_WINDOW_AMOUNT_CNY = 2_000_000.0
MIN_CROSS_SECTION = 30  # below this a cross-sectional residual is noise; report, do not trade
CASH_FRACTION = 0.97  # 3% fee buffer
SELL_PROCEEDS_HAIRCUT = 0.95  # conservative estimate of what today's sells will raise
LOT = 100

SCREEN = {
    "min_adv_cny": 30_000_000.0,
    "max_price": 40.0,
    "min_listed_calendar_days": 400,
    "excluded_markets": ("科创板", "北交所"),
}


def _execution_times(context):
    """Sells at the open, buys at the close, so the proceeds are credited first."""

    open_at = context.inference_at.replace(hour=9, minute=30, second=0, microsecond=0)
    close_at = context.inference_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if context.inference_at > close_at:
        return None, None
    if context.inference_at > open_at:
        open_at = close_at
    return open_at, close_at


def _target(context):
    """The rebalance target set, or None when today is not a rebalance day."""

    daily = read_daily(context, DAILY_LOOKBACK_DAYS)
    if daily.empty:
        return None, None
    last_date = daily["trade_date"].max()
    latest = pd.Timestamp(last_date)
    decision = pd.Timestamp(context.inference_at.date())
    # Hold 20 trading days: rebalance on the first decision of a new month,
    # which is exactly the decision whose newest visible bar is last month's.
    if (latest.year, latest.month) == (decision.year, decision.month):
        return None, daily
    dates = sorted(daily["trade_date"].unique())[-DISCOUNT_WINDOW:]
    blocks = read_blocks(context, dates[0])
    signal = window_discount(blocks, daily, dates)
    signal = signal[signal["amount_cny"] >= MIN_WINDOW_AMOUNT_CNY]
    if signal.empty:
        return [], daily
    frame = cross_section(daily, read_universe(context), last_date, SCREEN)
    frame = frame.merge(signal, on="ts_code", how="inner").dropna(
        subset=["discount", "circ_mv", "turn20"]
    )
    if len(frame) < MIN_CROSS_SECTION:
        return [], daily
    frame = frame.assign(score=neutral_score(frame, "discount"))
    return select(frame, "score", BASKET_SIZE, MODE), daily


def generate_orders(context):
    open_at, close_at = _execution_times(context)
    if open_at is None:
        return []
    target, daily = _target(context)
    if target is None:
        return []

    held = dict(context.account.positions)
    wanted = [] if len(target) == 0 else list(target["ts_code"])
    last_date = daily["trade_date"].max()
    prices = daily[daily["trade_date"] == last_date].set_index("ts_code")["close"]

    orders = []
    proceeds = 0.0
    for symbol in sorted(held):
        quantity = int(held[symbol])
        if symbol in wanted or quantity <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "execute_at": open_at.isoformat(),
                "reason": "block_discount_exit",
            }
        )
        if symbol in prices.index:
            proceeds += quantity * float(prices[symbol]) * SELL_PROCEEDS_HAIRCUT

    buys = [symbol for symbol in wanted if symbol not in held]
    if not buys:
        return orders
    remaining = (float(context.account.cash) + proceeds) * CASH_FRACTION
    for index, symbol in enumerate(buys):
        if symbol not in prices.index:
            continue
        price = float(prices[symbol])
        if not np.isfinite(price) or price <= 0:
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
                "reason": "block_discount_" + MODE,
            }
        )
        remaining -= quantity * price
    return orders
