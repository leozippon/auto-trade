"""Top-20 equal-cash book on a monthly cadence with a keep band, no fit.

Review: the first decision of each calendar month, and any decision while the account
is flat. The rule compares the newest visible trading day's month with the decision
day's, so it is stateless: a cold worker and a warm one produce the same orders. A
non-review decision reads only a few days of trade dates.

Keep band: a holding still ranked inside TOP_N * KEEP_BAND is not sold; a holding is
sold when it left the pool or fell outside the band, at most MAX_REPLACE swaps a
review (exits from the pool come first and are always sold). At CNY 100k a 20-name
position is about CNY 4,800, so the CNY 5 minimum commission is ~10 bp a side.

INDUSTRY_CAP is off by default and the book is then simply the top TOP_N of the
ranking. Set it to an integer and the free seats are filled under a per-industry
counter seeded with the holdings the keep band retained -- the cap is counted
against the book this review will actually hold, never against a freshly ranked
top TOP_N. That is what makes a cap hold at all: the keep band retains holdings
ranked behind that list, and capping the list alone lets their industries through
uncounted. The constant is here so a candidate that wants an exposure budget by
construction does not have to re-derive the keep-band interaction; it is not a
recommendation, and a candidate may bound its exposure any other way it likes.

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
INDUSTRY_CAP = None          # names of one SW L1 industry the book may hold; None turns the cap off
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
    sells = _sell_orders(positions, rank, len(ranked), sell_at)
    proceeds = sum(float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    target = _book(ranked, keep)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at)


def _new_month(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision.year, decision.month)


def _sell_orders(positions, rank, pool_size, execute_at):
    """Exit names that left the pool, then the worst-ranked holdings outside the keep
    band -- but only as many as the pool has unheld names to replace them with."""
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= TOP_N * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(MAX_REPLACE - len(forced), available))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": REASON + "_exit"}
        for code in sorted(drop)
    ]


def _book(ranked, keep):
    """The TOP_N names this review holds, in ranking order.

    Without INDUSTRY_CAP this is the top TOP_N. With it, the retained holdings
    take their seats first and seed the industry counter, and the free seats go
    to the best-scored names whose industry still has room -- so the cap holds by
    induction, the seats carried over having come from a book that already
    respected it.
    """
    codes = list(ranked["ts_code"])
    if INDUSTRY_CAP is None:
        return codes[:TOP_N]
    industry = dict(zip(codes, ranked["industry"]))
    retained = set(keep)
    counts = {}
    for code in codes:
        if code in retained:
            counts[industry[code]] = counts.get(industry[code], 0) + 1
    seats = TOP_N - len(retained)
    added = set()
    for code in codes:
        if len(added) >= seats:
            break
        if code in retained or counts.get(industry[code], 0) >= INDUSTRY_CAP:
            continue
        counts[industry[code]] = counts.get(industry[code], 0) + 1
        added.add(code)
    return [code for code in codes if code in retained or code in added]


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
