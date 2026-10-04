"""The recipe's equal-cash book, fitted to a CNY 100k account by a one-lot pool filter.

This is the book every [FEAT-14] arm ran (equal cash, review, keep band, swap
cap, sell 09:30 / buy 15:00), not the round's canonical rule-score book: it
caps band exits per review and does not refill seats between reviews. What a
100k account changes is which names can be held at all, so affordability is
a rule of the POOL, applied before anything is ranked:

    equity     cash + every holding at its T-1 close (the account the
               decision sees, before today's orders)
    seat cash  CASH_BUFFER x equity / knobs.SEATS
    eligible   a tradable name whose one lot (100 shares at its T-1 close)
               costs no more than the seat cash

The ranking -- and with it the keep band and the buys -- runs over the
eligible names plus every tradable holding: a holding whose lot outgrew a
seat is ranked and kept like any other, never force-sold for its price.

The pool lane narrows what may be bought the same way (`lib/pool.py`): a
name outside the restricted pool is not bought, a holding outside it is kept
until it leaves by rank.

    seats        knobs.SEATS, equal cash
    review       knobs.REVIEW: the first decision of each ISO week, or of each
                 half calendar month (days 1-15, 16-end); any decision while flat
    keep band    a holding ranked inside knobs.KEEP_BAND x SEATS is kept
    swaps        at most knobs.MAX_SWAPS band exits a review, worst first;
                 forced exits (no score, no T-1 bar, ST / 退) always go on top

Sizing: the book value is (cash + 0.98 x sale proceeds) x CASH_BUFFER plus
the kept holdings at T-1; each buy is book value / book size in 100-share
lots rounded to the nearest lot and cut to the cash left, because
`context.account` does not change mid-call. Sells at 09:30 and buys at 15:00
of the same day. Ties in the ranking are broken by code. The review is
stateless: the period of the newest visible trading day is compared with the
decision day's, so a cold worker and a warm one emit the same orders.

The score is the bag's mean rank on the newest row (`lib/model.py`). With
`knobs.SMOOTH_HALFLIFE` = H weeks it is instead the exponential average of the
bag's rank on the newest row and on the rows SMOOTH_STRIDE, 2 x SMOOTH_STRIDE,
... 3H x SMOOTH_STRIDE trading days earlier, weight 0.5 ** (k / H) on the k-th
week back, each name averaged over the weeks it was scorable. The earlier rows
are scored by today's members from their own past windows, so the average
needs no state between reviews; they fall after the members' last training
date, so none of them is a row the members were fitted on. Orders carry
`knobs.leg()`.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import knobs, model, panel, pool

CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100
REVIEW_CALENDAR_DAYS = 200   # SEQ_LEN bars, the smoothing weeks, plus holidays
SMOOTH_STRIDE = 5            # trading days between two smoothed rows: a week


def run(context):
    leg = knobs.leg()
    seats = knobs.SEATS
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    if positions and not _new_period(context, decision):
        return []

    data = panel.build(context, REVIEW_CALENDAR_DAYS, labels=False)
    if len(data["dates"]) < panel.SEQ_LEN:
        return []
    scores = score(context, data)
    last = len(data["dates"]) - 1
    closes = pd.Series(data["close"][last], index=data["codes"])
    equity = float(context.account.cash) + sum(_price(closes, code) * qty for code, qty in positions.items())
    eligible_cash = CASH_BUFFER * equity / seats
    held = np.isin(data["codes"], list(positions))
    affordable = data["close"][last] * LOT <= eligible_cash
    buyable = pool.eligible(context, data, last)
    keep_mask = panel.tradable(data, last) & np.isfinite(scores) & ((affordable & buyable) | held)
    ranked = pd.DataFrame({
        "ts_code": data["codes"][keep_mask],
        "score": scores[keep_mask],
        "close": data["close"][last][keep_mask],
    }).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < seats:
        raise RuntimeError(f"only {len(ranked)} eligible names for a {seats}-seat book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]

    sells = _sell_orders(positions, rank, sell_at, leg, seats)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    book = keep + [code for code in ranked["ts_code"] if code not in positions][: max(0, seats - len(keep))]
    meta = {
        "leg": leg,
        "seats": seats,
        "book_size": len(book),
        "equity": round(equity, 2),
        "seat_cash": round(eligible_cash, 2),
        "eligible": int(len(ranked)),
        "pool": int(buyable.sum()),
        "unaffordable_share": round(1.0 - float(affordable[panel.tradable(data, last)].mean()), 4),
    }
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, leg, meta)


def score(context, data):
    """Scores in [0, 1] on the newest row: the bag's mean rank, or its smoothed average."""

    last = len(data["dates"]) - 1
    if not knobs.SMOOTH_HALFLIFE:
        return model.score(context, data, [last])[0]
    rows = [last - SMOOTH_STRIDE * k for k in range(3 * knobs.SMOOTH_HALFLIFE + 1)]
    rows = [t for t in rows if t >= panel.SEQ_LEN - 1]
    return smooth(model.score(context, data, rows), knobs.SMOOTH_HALFLIFE)


def smooth(scores, halflife):
    """Weighted mean down the rows of `scores` (newest first), weight 0.5 ** (k / halflife) on row k.

    Each name is averaged over the rows where it has a score; NaN where the
    newest row has none.
    """

    weight = 0.5 ** (np.arange(len(scores)) / halflife)
    seen = np.isfinite(scores)
    total = np.where(seen, scores, 0.0).T @ weight
    mass = seen.T @ weight
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / mass
    return np.where(seen[0], out, np.nan)


def _period(day):
    if knobs.REVIEW == "week":
        return tuple(day.isocalendar())[:2]
    if knobs.REVIEW == "biweek":
        return day.year, day.month, day.day > 15
    raise ValueError(f"knobs.REVIEW must be 'week' or 'biweek', got {knobs.REVIEW!r}")


def _new_period(context, decision):
    start = (context.inference_at - timedelta(days=20)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return _period(latest) != _period(decision)


def _sell_orders(positions, rank, execute_at, leg, seats):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= seats * knobs.KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    replaceable = max(0, knobs.MAX_SWAPS - len(forced))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": leg + "_exit"}
        for code in sorted(drop)
    ]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(book, keep, prices, budget, book_value, execute_at, leg, meta):
    orders = []
    remaining = budget
    target_weight = 1.0 / max(1, len(book))
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
            "execute_at": execute_at.isoformat(), "reason": leg + "_entry",
            "target_weight": round(target_weight, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
