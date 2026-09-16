"""Risk-shaped top-15 equal-cash book on a quarterly cadence, no fit.

Construction, in this order at every review:

1. pool floor -- keep the larger CAP_FLOOR fraction of the eligible pool by
   float market cap;
2. rank the survivors by the leg's neutralized score;
3. second sort -- take the top SECOND_SORT * TOP_N by that rank, re-sort them
   ascending by `ivol60`, and put them in front of the rest;
4. hold the first TOP_N, equal cash, whole 100-share lots.

`SHAPES` says which of steps 1 and 3 each pre-registered leg runs; the score
itself comes from `lib.face`. `c_ivol` is the candidate with step 2's quality
rank replaced by the residual-volatility rank, which makes step 3 a no-op: it
is the "same construction, claimed mechanism removed" control.

Cadence: the book is reviewed on the first decision day of each calendar
quarter and at the replay's first decision. The rule is stateless -- it
compares the latest visible trading day's quarter with the decision day's -- so
a cold worker and a warm one produce identical orders.

Sells are timed at 09:30 and buys at 15:00 of the same day: the Broker
processes pending orders in timestamp order, so the proceeds are credited
before the buys are sized. `context.account` is the pre-call snapshot and never
changes mid-call, so the buy budget is decremented locally.

Keep band: a holding still ranked inside TOP_N * KEEP_BAND is not sold. A
holding is sold when it left the floored pool (halted, price cap, ADV floor,
stale statement, no volatility window, or no longer in the larger half by float
cap) or fell outside the band and a fresher top name can replace it. At most
MAX_REPLACE swaps per review.
"""

import numpy as np
import pandas as pd

from lib import face

TOP_N = face.BOOK_SIZE
MAX_REPLACE = 8              # at most this many swaps per review
KEEP_BAND = 2.0              # a holding ranked inside TOP_N * KEEP_BAND is not sold
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98

CAP_FLOOR = 0.5              # keep this fraction of the pool, largest float cap first
SECOND_SORT = 2              # re-sort the top SECOND_SORT * TOP_N by ivol60, ascending

# leg -> (does it apply the pool floor, does it apply the ivol60 second sort)
SHAPES = {
    "s_ql": (True, True),
    "c_q15": (False, False),
    "c_ivol": (True, False),
    "c_pool": (False, False),
}


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
    ranked = _shape(section, candidate)
    rank = {code: index for index, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    target = list(ranked["ts_code"][:TOP_N])
    sells = _sell_orders(positions, rank, target, sell_at, candidate)
    proceeds = sum(float(prices.get(order["symbol"], 0.0)) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    return sells + _buy_orders(target, keep, prices, budget, buy_at, candidate, vol_basis)


def _shape(section, candidate):
    """The pool floor and the ivol60 second sort, in the registered order."""
    floored, second = SHAPES[candidate]
    if floored:
        keep = int(len(section) * CAP_FLOOR)
        section = section.nlargest(max(keep, TOP_N), "circ_mv")
    ranked = section.sort_values(["score", "ts_code"], ascending=[False, True])
    if second:
        head = ranked.head(SECOND_SORT * TOP_N).sort_values(["ivol60", "ts_code"])
        ranked = pd.concat([head, ranked.iloc[SECOND_SORT * TOP_N:]])
    return ranked.reset_index(drop=True)


def _is_review(decision, latest_day, positions):
    """First decision of a calendar quarter, or the first decision of the replay."""
    if not positions:
        return True
    latest = pd.Timestamp(latest_day)
    return (latest.year, (latest.month - 1) // 3) != (decision.year, (decision.month - 1) // 3)


def _sell_orders(positions, rank, target, execute_at, candidate):
    """Exit names that left the shaped pool, then the worst-ranked holdings
    outside the keep band -- but only as many as fresh top names can replace."""
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
