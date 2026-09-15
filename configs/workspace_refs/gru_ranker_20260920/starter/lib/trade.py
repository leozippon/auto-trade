"""Top-15 equal-cash book reviewed once a week, with a keep band and a swap cap.

Review: the first decision of each ISO week, and any decision while the account
is flat. The rule compares the decision day's ISO week with that of the newest
visible trading day, so it is stateless: a cold worker and a warm one produce the
same orders. A non-review decision reads only a few days of trade dates.

Costs: at CNY 100k a 15-name position is about CNY 6,300, so the CNY 5 minimum
commission alone is ~8 bp a side and a round trip ~25-35 bp. The book therefore
sells a holding only when it left the pool, or ranks outside TOP_N * KEEP_BAND
and a fresh top name can replace it, and makes at most MAX_SWAPS replacements a
review (forced exits included): about 6 turnovers a year.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited
before buys are sized; `context.account` never changes mid-call, so the buy
budget is decremented locally and sized at the T-1 close in 100-share lots.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import model, panel, tree

TOP_N = 15
KEEP_BAND = 2.0              # a holding ranked inside TOP_N * KEEP_BAND is kept
MAX_SWAPS = 2                # replacements per weekly review, forced exits included
REVIEW_CALENDAR_DAYS = 130   # daily rows read on a review: SEQ_LEN bars plus ADV and holidays
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
    if positions and not _new_week(context, decision):
        return []
    data = panel.build(context, REVIEW_CALENDAR_DAYS, labels=False)
    if len(data["dates"]) < panel.SEQ_LEN:
        return []
    scores = score(context, data, candidate)
    pool = panel.pool_mask(context, data) & np.isfinite(scores)
    ranked = pd.DataFrame({"ts_code": data["codes"][pool], "score": scores[pool],
                           "close": data["close"][-1][pool]})
    ranked = ranked.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    target = list(ranked["ts_code"][:TOP_N])
    sells = _sell_orders(positions, rank, target, sell_at, candidate)
    proceeds = sum(float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at, candidate)


def score(context, data, candidate):
    """Scores in [0, 1] on the newest row for the candidate `main.CANDIDATE` names."""

    if candidate == "g1":
        return model.score(context, data)
    if candidate == "c_lgbm":
        return tree.score(context, data)
    if candidate == "v_ens":
        return (model.score(context, data) + tree.score(context, data)) / 2.0
    raise ValueError(f"unknown candidate: {candidate}")


def _new_week(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return latest.isocalendar()[:2] != decision.isocalendar()[:2]


def _sell_orders(positions, rank, target, execute_at, candidate):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    fresh = [code for code in target if code not in positions]
    replaceable = max(0, min(MAX_SWAPS - len(forced), len(fresh)))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drop)
    ]


def _buy_orders(target, keep, prices, budget, execute_at, candidate):
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
                       "execute_at": execute_at.isoformat(), "reason": candidate + "_entry"})
        remaining -= quantity * price
    return orders
