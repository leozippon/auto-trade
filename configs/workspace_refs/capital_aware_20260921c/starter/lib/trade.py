"""One book, six shapes, sized by the account this arm runs.

The account arrives as `context.account` -- cash plus the T-1 value of the
holdings -- and two things follow from it: how many seats the book has, and
which names one lot of a seat can buy. What the book IS does not follow from
it: `BOOK` says that, and the session sets it after reading its run facts.

    BOOK = "quality_overlay"    default. CSI 300, 80-100 seats, monthly,
                                nearly fully invested. Quality tilt on
                                benchmark weights. No industry / name box.
    BOOK = "high52_overlay"     same shape; 52-week near-high, SW L1-centred
    BOOK = "rmax_overlay"       same shape; −max of the last 20 daily pct_chg
    BOOK = "pacc_index"         CSI 300, 12 seats, monthly. Seats follow
                                index L1 weights; −percent-accrual inside.
    BOOK = "net_issuance"       whole-market affordable pool, 12 seats,
                                quarterly, −net issuance, at most 2 names
                                per SW L1
    BOOK = "lottery_reverse"    same pool and seat band, monthly,
                                −within-L1 lottery rank, at most 2 per L1.
                                A name at the T-1 limit-up is not a new buy.

Seats, when SEATS is 0: clip(equity x CASH_BUFFER / SEAT_CASH_MIN) into the
shape's own min/max. Pin SEATS to a positive integer to override.

Review is stateless: the newest visible trading day's calendar month, quarter
or ISO week is compared with the decision day's. A non-review decision reads
only a few days of trade dates.

Forced exits, keep band and replacement. A holding that is no longer rankable
is always sold. A holding still ranked inside `seats * KEEP_BAND` is kept;
one outside the band is sold, at most the shape's MAX_REPLACE swaps a review
with forced exits counted first.

Affordability. A seat is `book value / seats`. Overlay books do not apply a
price cap -- a name's money is its tilted weight. The 12-seat books apply
`close * LOT <= seat cash` (price cap = seat / 100). An unaffordable HOLDING
is never force-sold; only affordable names may fill a free seat.

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

from lib import composite, index

BOOK = "quality_overlay"
SEATS = 0
SEAT_CASH_MIN = 6_900.0
KEEP_BAND = 2.0
MAX_REPLACE = 0
MIN_POOL = 60
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100
INDUSTRY_CAP = 2

OVERLAYS = ("quality_overlay", "high52_overlay", "rmax_overlay")
INDEX_BOOKS = OVERLAYS + ("pacc_index",)

SHAPES = {
    "quality_overlay": {"cadence": "month", "seats_min": 80, "seats_max": 100, "max_replace": 10},
    "high52_overlay": {"cadence": "month", "seats_min": 80, "seats_max": 100, "max_replace": 10},
    "rmax_overlay": {"cadence": "month", "seats_min": 80, "seats_max": 100, "max_replace": 10},
    "pacc_index": {"cadence": "month", "seats_min": 12, "seats_max": 12, "max_replace": 2},
    "net_issuance": {"cadence": "quarter", "seats_min": 12, "seats_max": 12, "max_replace": 2},
    "lottery_reverse": {"cadence": "month", "seats_min": 12, "seats_max": 12, "max_replace": 2},
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
    seat_cash = equity * CASH_BUFFER / seats if seats else 0.0
    ranked, section = _ranked(context, cross, daily, trading_days, shape, seats, seat_cash)
    need = seats if shape in INDEX_BOOKS else max(MIN_POOL, seats)
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
    book = _book(ranked, keep, seats, seat_cash, shape)
    share = weights.reindex(book).fillna(0.0)
    share = share / share.sum() if float(share.sum()) > 0 else share
    meta = {
        "shape": shape,
        "seats": seats,
        "book_size": len(book),
        "unbuyable_share": round(1.0 - len(_affordable(ranked, seat_cash, shape)) / max(1, len(ranked)), 4),
    }
    if shape in INDEX_BOOKS:
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


def _ranked(context, cross, daily, trading_days, shape, seats, seat_cash):
    pool = composite.tradable(cross)
    if shape in INDEX_BOOKS:
        members, section = index.constituents(context)
        if shape == "quality_overlay":
            return composite.quality_overlay(context, pool, members, trading_days), section
        if shape == "high52_overlay":
            return composite.high52_overlay(pool, members, daily, trading_days), section
        if shape == "rmax_overlay":
            return composite.rmax_overlay(pool, members, daily, trading_days), section
        cheap = pool[pool["close"] * LOT <= seat_cash] if seat_cash > 0 else pool
        return composite.pacc_index(context, cheap, members, trading_days, seats), section
    cheap = pool[pool["close"] * LOT <= seat_cash] if seat_cash > 0 else pool
    if cheap.empty:
        raise RuntimeError("no affordable tradable names for the 12-seat pool")
    composite.require_industry(cheap, "affordable pool")
    if shape == "net_issuance":
        frame = composite.net_issuance_score(cheap, daily, trading_days)
    elif shape == "lottery_reverse":
        frame = composite.lottery_reverse_score(cheap, daily, trading_days)
    else:
        raise RuntimeError(f"unhandled BOOK {shape!r}")
    return frame.assign(w=1.0).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True), ""


def _affordable(ranked, seat_cash, shape):
    if shape in OVERLAYS or seat_cash <= 0:
        return ranked
    return ranked[ranked["close"] * LOT <= seat_cash]


def _book(ranked, keep, seats, seat_cash, shape):
    if shape in OVERLAYS:
        names = list(ranked["ts_code"])
        return keep + [code for code in names if code not in keep][: max(0, seats - len(keep))]
    if shape == "pacc_index":
        names = list(ranked["ts_code"])
        return keep + [code for code in names if code not in keep][: max(0, seats - len(keep))]
    industry = ranked.set_index("ts_code")["industry"]
    counts = {}
    book = []
    for code in keep:
        book.append(code)
        label = str(industry.get(code, "未分类"))
        counts[label] = counts.get(label, 0) + 1
    candidates = _affordable(ranked, seat_cash, shape)
    skip_limit = shape == "lottery_reverse" and "at_limit_up" in ranked.columns
    blocked = set(ranked.loc[ranked["at_limit_up"].astype(bool), "ts_code"]) if skip_limit else set()
    for code in candidates["ts_code"]:
        if code in book or code in blocked:
            continue
        label = str(industry.get(code, "未分类"))
        if counts.get(label, 0) >= INDUSTRY_CAP:
            continue
        book.append(code)
        counts[label] = counts.get(label, 0) + 1
        if len(book) >= seats:
            break
    return book


def _due(context, decision, cadence):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    if cadence == "week":
        return latest.isocalendar()[:2] != decision.isocalendar()[:2]
    if cadence == "quarter":
        return (latest.year, (latest.month - 1) // 3) != (decision.year, (decision.month - 1) // 3)
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
