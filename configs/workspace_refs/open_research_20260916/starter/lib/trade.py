"""Top-20 equal-cash book on a monthly cadence with a keep band, no fit.

Review: the first decision of each calendar month, and any decision while the account
is flat. The rule compares the newest visible trading day's month with the decision
day's, so it is stateless: a cold worker and a warm one produce the same orders. A
non-review decision reads only a few days of trade dates.

Keep band: a holding still ranked inside TOP_N * KEEP_BAND is not sold; a holding is
sold when it left the pool or fell outside the band and a fresh top name can replace
it, at most MAX_REPLACE swaps a review (exits from the pool come first and are always
sold). At CNY 100k a 20-name position is about CNY 4,800, so the CNY 5 minimum
commission is ~10 bp a side.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited before
buys are sized; `context.account` never changes mid-call, so the buy budget is
decremented locally and sized at the T-1 close in 100-share lots.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import composite

TOP_N = 20
KEEP_BAND = 2.0              # a holding ranked inside TOP_N * KEEP_BAND is kept
MAX_REPLACE = 10             # swaps per monthly review
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
REASON = "composite"


def run(context):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    if positions and not _new_month(context, decision):
        return []
    section, prices = composite.build(context)
    if section is None:
        raise RuntimeError("fewer than 30 scorable names in the pool")
    ranked = section.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    target = list(ranked["ts_code"][:TOP_N])
    sells = _sell_orders(positions, rank, target, sell_at)
    proceeds = sum(float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at)


def _new_month(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision.year, decision.month)


def _sell_orders(positions, rank, target, execute_at):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    fresh = [code for code in target if code not in positions]
    replaceable = max(0, min(MAX_REPLACE - len(forced), len(fresh)))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": REASON + "_exit"}
        for code in sorted(drop)
    ]


def _buy_orders(target, keep, prices, budget, execute_at):
    buys = [code for code in target if code not in keep][: max(0, TOP_N - len(keep))]
    orders = []
    remaining = budget
    for index, code in enumerate(buys):
        price = float(prices.get(code, float("nan")))
        if not np.isfinite(price) or price <= 0:
            continue
        quantity = int(remaining / max(1, len(buys) - index) / price // 100 * 100)
        if quantity <= 0:
            continue
        orders.append({"symbol": code, "action": "buy", "quantity": quantity,
                       "execute_at": execute_at.isoformat(), "reason": REASON + "_entry"})
        remaining -= quantity * price
    return orders
