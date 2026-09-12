"""Neutralized top-15 equal-cash book on a monthly cadence with a keep band, no fit.

Cadence: the book is reviewed on the first decision day of each calendar month
and at the replay's first decision. The rule is stateless -- it compares the
latest visible trading day's month with the decision day's -- so a cold worker
and a warm one produce identical orders.

Sells are timed at 09:30 and buys at 15:00 of the same day: the Broker
processes pending orders in timestamp order, so the proceeds are credited
before the buys are sized. `context.account` is the pre-call snapshot and never
changes mid-call, so the buy budget is decremented locally.

Keep band: a holding still ranked inside TOP_N * KEEP_BAND is not sold, which
is what keeps the turnover of a slow characteristic at a few names a month; a
holding is sold when it left the eligible pool (halted, price cap, ADV floor,
stale statement, no volatility window) or fell outside the band and a fresher
top name can replace it. At most MAX_REPLACE swaps per review.
"""

import numpy as np
import pandas as pd

from lib import face

TOP_N = 15
MAX_REPLACE = 8              # at most this many swaps per review
KEEP_BAND = 2.0              # a holding ranked inside TOP_N * KEEP_BAND is not sold
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98


def run(context, candidate):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    section, latest_day, vol_basis = face.build_cross_section(context, candidate)
    if section is None:
        return []
    if not _is_review(decision, latest_day, positions):
        return []
    section["score"] = face.neutralize(section, candidate)
    ranked = section.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    target = list(ranked["ts_code"][:TOP_N])
    sells = _sell_orders(positions, rank, target, sell_at, candidate)
    proceeds = sum(float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at, candidate, vol_basis)


def _is_review(decision, latest_day, positions):
    """First decision of a calendar month, or the first decision of the replay."""
    if not positions:
        return True
    latest = pd.Timestamp(latest_day)
    return (latest.year, latest.month) != (decision.year, decision.month)


def _sell_orders(positions, rank, target, execute_at, candidate):
    """Exit names that left the pool, then the worst-ranked holdings outside the
    keep band -- but only as many as fresh top names can replace."""
    forced = [code for code in positions if code not in rank]
    held = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    held.sort(key=lambda code: (-rank[code], code))
    fresh = [code for code in target if code not in positions]
    replaceable = max(0, min(MAX_REPLACE - len(forced), len(fresh)))
    drop = forced + held[:replaceable]
    return [
        {
            "symbol": code,
            "action": "sell",
            "quantity": int(positions[code]),
            "execute_at": execute_at.isoformat(),
            "reason": candidate + "_exit",
        }
        for code in sorted(drop)
    ]


def _buy_orders(target, keep, prices, budget, execute_at, candidate, vol_basis):
    """Equal-cash top-up to TOP_N names, whole 100-share lots, locally decremented budget."""
    buys = [code for code in target if code not in keep][: max(0, TOP_N - len(keep))]
    orders = []
    remaining = budget
    for index, code in enumerate(buys):
        price = float(prices.get(code, float("nan")))
        if not np.isfinite(price) or price <= 0:
            continue
        share = remaining / max(1, len(buys) - index)
        quantity = int(share / price // 100 * 100)
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": code,
                "action": "buy",
                "quantity": quantity,
                "execute_at": execute_at.isoformat(),
                "reason": candidate + "_entry",
                "vol_basis": vol_basis,
            }
        )
        remaining -= quantity * price
    return orders
