"""One book, four shapes, sized by the account this arm runs.

The account arrives as `context.account` -- cash plus the T-1 value of the
holdings -- and two things follow from it: how many seats the book has, and
which names one lot of a seat can buy. What the book IS does not follow from
it: `BOOK` says that, and the session sets it after reading its run facts.

    BOOK = "index_overlay"      default. CSI 300 constituents, benchmark
                                weights times an unbounded overlay, nearly
                                fully invested, 80-100 seats, monthly review
    BOOK = "index_indneutral"   same universe and seat band; industry weights
                                matched to the index, EP/BP inside each
                                industry. Construction, not a post-hoc cap
    BOOK = "pool_momentum"      affordable whole-market pool, 8-12 seats,
                                60-120d residual momentum, weekly review
    BOOK = "pool_eyield"        same pool and seat band, earnings / cash-flow
                                yield (no PB), weekly review

Seats, when SEATS is 0: clip(equity x CASH_BUFFER / SEAT_CASH_MIN) into the
shape's own band. Pin SEATS to a positive integer to override. Capital and
whether a tracking mandate is in force are two independent settings; nothing
here reads a mandate off the account size.

Review is stateless: the newest visible trading day's calendar month or ISO
week is compared with the decision day's. A non-review decision reads only a
few days of trade dates. Monthly is also when a new month-end constituent
section becomes visible.

Forced exits, keep band and replacement. A holding that is no longer rankable
is always sold. A holding still ranked inside `seats * KEEP_BAND` is kept;
one outside the band is sold, at most the shape's MAX_REPLACE swaps a review
with forced exits counted first.

Affordability. A seat is `book value / seats`. Ranking runs over every
tradable name -- an unaffordable HOLDING is never force-sold -- while only
affordable names may fill a free seat. In an index book a name's target money
is its `w`, so the lightest names can pass the seat test and still not take a
lot; `book_size` beside the buy count says so.

Sizing. A name's money is `book value x its target weight`, in 100-share lots
rounded to the NEAREST lot and then clipped to the cash on hand -- never
rounded down.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited
before buys are sized; `context.account` never changes mid-call, so the buy
budget is decremented locally and sized at the T-1 close.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

# Absolute import of the sibling package; relative imports are refused.
from lib import composite, index

BOOK = "index_overlay"
SEATS = 0
SEAT_CASH_MIN = 6_900.0
KEEP_BAND = 2.0
MAX_REPLACE = 0                    # 0 uses the shape default in SHAPES
MIN_ADV = 3.0e7
MIN_POOL = 60
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100

SHAPES = {
    "index_overlay": {"cadence": "month", "seats_min": 80, "seats_max": 100, "max_replace": 10},
    "index_indneutral": {"cadence": "month", "seats_min": 80, "seats_max": 100, "max_replace": 10},
    "pool_momentum": {"cadence": "week", "seats_min": 8, "seats_max": 12, "max_replace": 2},
    "pool_eyield": {"cadence": "week", "seats_min": 8, "seats_max": 12, "max_replace": 2},
}


def run(context):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    spec = SHAPES[_shape()]
    if positions and not _due(context, decision, spec["cadence"]):
        return []

    cross, prices, daily, trading_days = composite.market(context)
    equity = float(context.account.cash) + sum(_price(prices, code) * qty for code, qty in positions.items())
    seats = _seats(equity, spec)
    shape = _shape()
    ranked, section = _ranked(context, cross, daily, trading_days, shape, seats)
    need = seats if shape.startswith("index") else max(MIN_POOL, seats)
    if len(ranked) < need:
        raise RuntimeError(f"only {len(ranked)} rankable names for a {seats}-seat {shape} book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}

    sells = _sell_orders(positions, rank, seats, len(ranked), sell_at, shape, MAX_REPLACE or spec["max_replace"])
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
    if shape.startswith("index"):
        index_weight = ranked.set_index("ts_code")["index_weight_pct"].reindex(book).fillna(0.0)
        meta["section"] = section
        meta["book_index_weight_pct"] = round(float(index_weight.sum()), 3)
    buys = _buy_orders(book, keep, prices, budget, book_value, share, buy_at, shape, meta)
    return sells + buys


def _seats(equity, spec):
    low, high = spec["seats_min"], spec["seats_max"]
    if SEATS:
        return int(min(max(int(SEATS), low), high))
    derived = int(equity * CASH_BUFFER // SEAT_CASH_MIN)
    return int(min(max(derived, low), high))


def _shape():
    if BOOK not in SHAPES:
        raise RuntimeError(f"BOOK must be one of {sorted(SHAPES)}, not {BOOK!r}")
    return BOOK


def _ranked(context, cross, daily, trading_days, shape, seats):
    pool = composite.tradable(cross)
    if shape in ("index_overlay", "index_indneutral"):
        members, section = index.constituents(context)
        frame = pool.merge(members, on="ts_code")
        frame = frame[frame["weight"] > 0].copy()
        if frame.empty:
            raise RuntimeError("no tradable CSI 300 constituents visible")
        if shape == "index_overlay":
            return composite.overlay_weights(context, frame, daily, trading_days), section
        return composite.indneutral_value(frame, seats), section
    liquid = pool[pool["adv"] >= MIN_ADV]
    if shape == "pool_momentum":
        frame = composite.momentum_score(context, liquid, daily, trading_days)
    elif shape == "pool_eyield":
        frame = composite.eyield_score(context, liquid, trading_days)
    else:
        raise RuntimeError(f"unhandled BOOK {shape!r}")
    return frame.assign(w=1.0).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True), ""


def _due(context, decision, cadence):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    if cadence == "week":
        return latest.isocalendar()[:2] != decision.isocalendar()[:2]
    return (latest.year, latest.month) != (decision.year, decision.month)


def _sell_orders(positions, rank, seats, pool_size, execute_at, shape, max_replace):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= seats * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(max_replace - len(forced), available))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": shape + "_exit"}
        for code in sorted(drop)
    ]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(book, keep, prices, budget, book_value, share, execute_at, shape, meta):
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
