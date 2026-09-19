"""An equal-cash book held INSIDE the CSI 300, on a monthly cadence, with two zero-skill controls.

`BASKET_N` is the one constant this arm sets from its directive's range; every
other seat count in the package derives from it or from `PROXY_N`. `LEGS` says
what each pre-registered leg is, and the three differ in `main.CANDIDATE` alone:

    "s_idx"    the composite score's top BASKET_N constituents      (main candidate)
    "c_rand"   BASKET_N constituents drawn by a permanent lottery   (control, zero skill)
    "c_index"  PROXY_N constituents drawn by the same lottery       (control, universe proxy)

Why these two controls and not a pool baseline. Equal-weighting inside the CSI
300 is itself worth about +2.5 %/yr of neutralized excess at IR 0.29-0.57 with
no skill at all, so measuring a candidate against zero measures the universe,
not the candidate. `c_rand` is that zero-skill book at the candidate's own seat
count, so the difference between them is skill and nothing else. `c_index` is
the same draw at the largest seat count this account can carry, so it says what
equal-weighting the index is worth here and separates "the candidate picked
well" from "more names is better". The two nest by construction -- the same
seed, the same key per code -- so `c_rand` is exactly the first BASKET_N names
of `c_index`'s ranking and the size effect is isolated with no second source of
noise. It is a SAMPLE of the index, not the index: the sampling error against a
true all-constituent equal-weight book is real and is reported, not assumed away.

Review: the first decision of each calendar month, and any decision while the
account is flat. The rule compares the newest visible trading day's month with
the decision day's, so it is stateless: a cold worker and a warm one produce the
same orders. A non-review decision reads only a few days of trade dates. The
monthly cadence is also when a new month-end constituent section becomes
visible, so index additions and deletions enter the book at a review and never
between two.

Membership, keep band and forced exits. A holding that is no longer a tradable
constituent -- deleted from the index, or newly ST/halted -- is a forced exit
and is always sold: this book is defined by the index, so leaving the index
leaves the book. A holding still ranked inside `seats * KEEP_BAND` is kept;
one that fell outside the band is sold, at most MAX_REPLACE swaps a review with
forced exits counted first.

Affordability, which is where the account decides the book. A seat is
`equity / seats`, and one 100-share lot of a constituent can cost more than
that: at CNY 100k and 15 seats a seat is CNY 6,667 and 11-23 % of the index is
unbuyable; at 30 seats a seat is CNY 3,333 and 28-43 % is. So the ranking runs
over every tradable constituent -- an unaffordable HOLDING is never force-sold
-- while only affordable names may fill a free seat, and `unbuyable_share` is
reported on every buy. STAR constituents are out of the pool entirely
(lib/composite.py): their 200-share minimum lot does not fit a seat at either
account size.

Sizing. A name's cash is `book value / seats`, in 100-share lots rounded to the
NEAREST lot and then clipped to the cash on hand -- never rounded down. At
CNY 100k and 30 seats a seat is CNY 3,333 while a lot of a constituent costs
CNY 100 to 3,300, so rounding down would leave a seat up to half empty and the
realised weights would not be the equal weights this book registers. Rounding
to the nearest lot makes the error two-sided instead of one-sided; it does not
make it small, so every buy reports `realized_weight` beside `target_weight`
and the realised total position is a reading this arm owes on every round.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited
before buys are sized; `context.account` never changes mid-call, so the buy
budget is decremented locally and sized at the T-1 close.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import composite

BASKET_N = 15                # THE constant this arm sets, inside its directive's range
PROXY_N = 30                 # seats of the universe-proxy control: the most this account can carry
KEEP_BAND = 2.0              # a holding ranked inside seats * KEEP_BAND is kept
MAX_REPLACE = 10             # discretionary swaps per monthly review
LOTTERY_SEED = 20260919      # fixed: the controls must be the same draw in every replay
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100

# leg -> (ranking rule, seats). The three legs differ in this line alone.
LEGS = {
    "s_idx": ("score", BASKET_N),
    "c_rand": ("lottery", BASKET_N),
    "c_index": ("lottery", PROXY_N),
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
    if positions and not _new_month(context, decision):
        return []
    rule, seats = LEGS[candidate]
    frame, prices, section = composite.build(context)
    if frame is None:
        raise RuntimeError(f"fewer than {composite.MIN_POOL} scorable constituents in the {section} section")

    ranked = _ranked(frame, rule)
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    equity = float(context.account.cash) + sum(_price(prices, code) * qty for code, qty in positions.items())
    seat_cash = equity / seats
    affordable = ranked[ranked["close"] * LOT <= seat_cash]
    unbuyable = 1.0 - len(affordable) / max(1, len(ranked))

    sells = _sell_orders(positions, rank, seats, len(ranked), sell_at, candidate)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    book = keep + [code for code in affordable["ts_code"] if code not in keep][: max(0, seats - len(keep))]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    index_weight = frame.set_index("ts_code")["weight"].reindex(book).fillna(0.0)
    meta = {
        "section": section,
        "unbuyable_share": round(unbuyable, 4),
        "book_index_weight_pct": round(float(index_weight.sum()), 3),
    }
    buys = _buy_orders(book, keep, prices, budget, book_value, seats, buy_at, candidate, meta,
                       frame.set_index("ts_code")["weight"])
    return sells + buys


def _new_month(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision.year, decision.month)


def _ranked(frame, rule):
    """The tradable constituents of this decision, best first.

    `score` is the composite; `lottery` is a permanent uniform draw keyed on the
    code itself, so a name's place in the control's queue never changes and the
    control turns over only when the index does.
    """
    if rule == "score":
        return frame.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    draw = frame.assign(draw=_lottery(list(frame["ts_code"])))
    return draw.sort_values(["draw", "ts_code"]).reset_index(drop=True)


def _lottery(codes):
    """A permanent uniform key per code, from one fixed seed.

    Seeded per name rather than per cross-section on purpose: a draw taken over
    today's pool would be redrawn every month and the control would churn for a
    reason that has nothing to do with the index.
    """
    return np.array([
        float(np.random.default_rng([LOTTERY_SEED, int(code[:6]), sum(ord(ch) for ch in code[7:])]).random())
        for code in codes
    ])


def _sell_orders(positions, rank, seats, pool_size, execute_at, candidate):
    """Exit names that are no longer tradable constituents, then the worst-ranked
    holdings outside the keep band -- but only as many as the pool has unheld
    names to replace them with."""
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= seats * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(MAX_REPLACE - len(forced), available))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drop)
    ]


def _price(prices, code):
    """The T-1 close of one name, or 0 when it has none: a holding halted all of T-1
    has no price on its last bar, and a NaN would poison the whole buy budget
    rather than just its own sale."""
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(book, keep, prices, budget, book_value, seats, execute_at, candidate, meta, weights):
    """Equal-cash whole-lot buys of the seats this review adds."""
    target_weight = 1.0 / seats
    orders = []
    remaining = budget
    for code in [name for name in book if name not in keep]:
        price = _price(prices, code)
        if price <= 0:
            continue
        quantity = int(min(remaining, book_value * target_weight) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": candidate + "_entry",
            "target_weight": round(target_weight, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            "index_weight_pct": round(float(weights.get(code, 0.0)), 4),
            **meta,
        })
        remaining -= quantity * price
    return orders
