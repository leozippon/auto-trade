"""One book, two shapes, sized by the account this arm runs.

The account arrives as `context.account` -- cash plus the T-1 value of the
holdings -- and two things follow from it: how many seats the book has, and
which names one lot of a seat can buy. What the book IS does not follow from
it: `BOOK` says that, and the session sets it after reading its run facts.

Seats, the one rule.
    seats = clip(equity x CASH_BUFFER / SEAT_CASH_MIN, SEATS_MIN, SEATS_MAX)
Two opposite pressures meet in it. More seats lower the residual tracking error
of a book and dilute any single name's accident; fewer seats leave more cash in
a seat, which is what decides how much of the market one lot can reach, how
large the 5 CNY commission floor is in basis points, and how coarse whole-lot
rounding is. `SEAT_CASH_MIN` is 6,900 CNY because that is the seat cash at which
one lot still reaches about 70 % of the CSI 300 by index weight (48.5 % at
3,333, 64.5 % at 5,000, 76.7 % at 10,000) and the floor is 7.2 bp a side.
`SEATS_MAX` is 100 because that is where a zero-skill benchmark-weight book
stops breaching the 35 % absolute drawdown a tracking mandate travels with:
40 % of such books breach it at 50 seats, 15 % at 80 and 0 % at 100, while
their own tracking error reads 5.5 / 4.7 / 4.6 %. Past 100 the commission floor
keeps climbing for tracking error that has stopped falling. The readings are in
`references/capital-arithmetic.md`. Set `SEATS` to a positive integer to pin the
count instead: the rule is a default, not a constraint.

Shape, stated not derived.
    BOOK = "pool"   the affordable pool of the whole market, ranked by the
                    composite score, equal cash. The default.
    BOOK = "index"  the as-of CSI 300 constituents, ranked by index weight and
                    held at target weights proportional to it, kept nearly fully
                    invested, so the book's beta sits near 1 and its tracking
                    error starts from the index's own dispersion.
The capital and whether a tracking mandate is in force are two independent
settings of this arm, and both are in its run facts; the strategy context
carries neither. So nothing here reads a mandate off the account size. An arm
whose run facts carry a tracking mandate sets `BOOK = "index"` -- a book drawn
from the whole market starts from the gap between that pool and the CSI 300 and
does not get under a tracking-error cap -- and an arm without one is free to use
either.

Review: the first decision of each calendar month, and any decision while the
account is flat. The rule compares the newest visible trading day's month with
the decision day's, so it is stateless: a cold worker and a warm one produce the
same orders. A non-review decision reads only a few days of trade dates. Monthly
is also when a new month-end constituent section becomes visible, so in the
index book additions and deletions enter at a review and never between two.

Forced exits, keep band and replacement. A holding that is no longer rankable --
delisted, newly ST, halted through T-1, or, in the index book, dropped from the
index -- is always sold. A holding still ranked inside `seats * KEEP_BAND` is
kept; one outside the band is sold, at most MAX_REPLACE swaps a review with
forced exits counted first. On a whole-market pool that band is a ~1 % slice of
the ranking, so the pool book turns over nearly fully every month; that is a
measured reading to fix, not a recommendation (`sources.md`).

Affordability, which is where the account decides the book. A seat is
`book value / seats` and one 100-share lot can cost more than that: at 100k and
14 seats a seat is about 6,900 CNY, so nothing priced above 69 CNY can fill one,
and at 1M and 100 seats the limit is about 97 CNY. Ranking runs over every
tradable name -- an unaffordable HOLDING is never force-sold -- while only
affordable names may fill a free seat, and `unbuyable_share` is reported on
every buy. In the index book a name's target money is its index weight rather
than a whole seat, so the lightest names in a large book can pass the seat test
and still not take a lot; the seat is then left empty and `book_size` beside the
buy count says so.

Sizing. A name's money is `book value x its target weight`, in 100-share lots
rounded to the NEAREST lot and then clipped to the cash on hand -- never rounded
down. At 100k a seat is a few thousand CNY while a lot costs 100 to 7,000, so
rounding down would leave a seat up to half empty and the realised weights would
not be the weights the book registers. Rounding to the nearest lot makes the
error two-sided rather than one-sided; it does not make it small, so every buy
reports `realized_weight` beside `target_weight`.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited
before buys are sized; `context.account` never changes mid-call, so the buy
budget is decremented locally and sized at the T-1 close.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import composite, index

BOOK = "pool"                # "pool" (affordable whole-market pool) | "index" (CSI 300 constituents)
SEATS = 0                    # 0 derives the seat count from equity; a positive int pins it
SEAT_CASH_MIN = 6_900.0      # CNY per seat: one lot still reaches ~70 % of the index by weight
SEATS_MIN = 12
SEATS_MAX = 100

KEEP_BAND = 2.0              # a holding ranked inside seats * KEEP_BAND is kept
MAX_REPLACE = 10             # discretionary swaps per monthly review
MIN_ADV = 3.0e7              # CNY, 20-day mean turnover; the free shape's only pool screen
MIN_POOL = 60                # rankable names below which the decision is refused
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100


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

    cross, prices, daily, trading_days = composite.market(context)
    equity = float(context.account.cash) + sum(_price(prices, code) * qty for code, qty in positions.items())
    seats = _seats(equity)
    shape = _shape()
    ranked, section = _ranked(context, cross, daily, trading_days, shape)
    if len(ranked) < max(MIN_POOL, seats):
        raise RuntimeError(f"only {len(ranked)} rankable names for a {seats}-seat {shape} book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}

    sells = _sell_orders(positions, rank, seats, len(ranked), sell_at, shape)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / seats if seats else 0.0

    weights = ranked.set_index("ts_code")["w"]
    affordable = ranked[ranked["close"] * LOT <= seat_cash]
    unbuyable = 1.0 - len(affordable) / max(1, len(ranked))
    book = keep + [code for code in affordable["ts_code"] if code not in keep][: max(0, seats - len(keep))]
    share = weights.reindex(book).fillna(0.0)
    share = share / share.sum() if float(share.sum()) > 0 else share
    meta = {
        "shape": shape,
        "seats": seats,
        "book_size": len(book),
        "unbuyable_share": round(unbuyable, 4),
    }
    if shape == "index":
        index_weight = ranked.set_index("ts_code")["index_weight_pct"].reindex(book).fillna(0.0)
        meta["section"] = section
        meta["book_index_weight_pct"] = round(float(index_weight.sum()), 3)
    buys = _buy_orders(book, keep, prices, budget, book_value, share, buy_at, shape, meta)
    return sells + buys


def _seats(equity):
    """The seat count this equity carries, by the one rule in the module docstring."""
    if SEATS:
        return int(SEATS)
    derived = int(equity * CASH_BUFFER // SEAT_CASH_MIN)
    return int(min(max(derived, SEATS_MIN), SEATS_MAX))


def _shape():
    """The book this package is, stated by BOOK and never inferred."""
    if BOOK not in ("pool", "index"):
        raise RuntimeError(f"BOOK must be 'pool' or 'index', not {BOOK!r}")
    return BOOK


def _ranked(context, cross, daily, trading_days, shape):
    """(the rankable names best first with a `w` target-weight key, the section date).

    `w` is the index weight in the index book and 1.0 in the pool book, so one
    sizing path serves both shapes: the target weight of a name is its `w` over
    the `w` of the book actually held.
    """
    pool = composite.tradable(cross)
    if shape == "index":
        members, section = index.constituents(context)
        frame = pool.merge(members, on="ts_code")
        frame = frame[frame["weight"] > 0].copy()
        frame["w"] = frame["weight"].astype("float64")
        frame["index_weight_pct"] = frame["weight"].astype("float64")
        return frame.sort_values(["w", "ts_code"], ascending=[False, True]).reset_index(drop=True), section
    frame = composite.score(context, pool[pool["adv"] >= MIN_ADV], daily, trading_days)
    frame = frame.assign(w=1.0)
    return frame.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True), ""


def _new_month(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision.year, decision.month)


def _sell_orders(positions, rank, seats, pool_size, execute_at, shape):
    """Exit names that are no longer rankable, then the worst-ranked holdings
    outside the keep band -- but only as many as the pool has unheld names to
    replace them with."""
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= seats * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(MAX_REPLACE - len(forced), available))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": shape + "_exit"}
        for code in sorted(drop)
    ]


def _price(prices, code):
    """The T-1 close of one name, or 0 when it has none: a holding halted all of T-1
    has no price on its last bar, and a NaN would poison the whole buy budget
    rather than just its own sale."""
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(book, keep, prices, budget, book_value, share, execute_at, shape, meta):
    """Whole-lot buys of the seats this review adds, at each name's target weight."""
    orders = []
    remaining = budget
    for code in [name for name in book if name not in keep]:
        price = _price(prices, code)
        target_weight = float(share.get(code, 0.0))
        if price <= 0 or target_weight <= 0:
            continue
        quantity = int(min(remaining, book_value * target_weight) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": shape + "_entry",
            "target_weight": round(target_weight, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
