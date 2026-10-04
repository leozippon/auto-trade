"""The recipe's equal-cash book, fitted to a CNY 100k account by a one-lot pool filter.

This is the book every [FEAT-14] arm ran (equal cash, review, keep band, swap
cap), not the round's canonical rule-score book: it caps band exits per
review and does not refill seats between reviews. What a 100k account changes
is which names can be held at all, so affordability is a rule of the POOL,
applied before anything is ranked:

    equity     cash + every holding at its T-1 close (the account the
               decision sees, before today's orders)
    seat cash  CASH_BUFFER x equity / knobs.SEATS
    eligible   a tradable name whose one lot (100 shares at its T-1 close)
               costs no more than the seat cash

The ranking -- and with it the keep band and the buys -- runs over the
eligible names plus every tradable holding: a holding whose lot outgrew a
seat is ranked and kept like any other, never force-sold for its price. The
beta lane narrows what may be bought the same way (`lib/beta.py`): a name
under the floor is not bought, a holding under it is kept until it leaves by
rank.

    seats        knobs.SEATS, equal cash
    review       the first decision of each ISO week; any decision while flat
    keep band    a holding ranked inside knobs.KEEP_BAND x SEATS is kept
    swaps        at most knobs.MAX_SWAPS band exits a review, worst first;
                 forced exits (no score, no T-1 bar, ST / 退) always go on top
    clock        when the sells and the buys execute (`lib/clock.py`): a
                 review sells at 09:30 and buys at 15:00 (c_base), or sells at
                 15:00 and the next trading day buys the empty seats at 09:30

Sizing: the book value is (cash + 0.98 x proceeds of the sells that execute
before the buys) x CASH_BUFFER plus the kept holdings at T-1; each buy is
book value / book size in 100-share lots rounded to the nearest lot and cut to
the cash left, because `context.account` does not change mid-call. Ties in
the ranking are broken by code. Every decision is stateless: the review and
fill days are read from the newest visible trading days, so a cold worker and
a warm one emit the same orders.

The score is the bag's mean rank on the newest row (`lib/model.py`), or, for
the beta lane's placebo, a date-seeded random score (`lib/beta.py`). Orders
carry `knobs.leg()`.
"""

import numpy as np
import pandas as pd

from lib import beta, clock, knobs, model, panel

CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100
REVIEW_CALENDAR_DAYS = 200   # SEQ_LEN bars and the beta window, plus holidays


def run(context):
    leg = knobs.leg()
    seats = knobs.SEATS
    decision = pd.Timestamp(context.inference_at)
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    review_today, review_yesterday = clock.reviews(clock.visible_days(context), decision)
    sell_at, buy_at = clock.instants(decision, not positions, review_today, review_yesterday)
    if sell_at is None and buy_at is None:
        return []

    data = panel.build(context, REVIEW_CALENDAR_DAYS, labels=False)
    if len(data["dates"]) < panel.SEQ_LEN:
        return []
    last = len(data["dates"]) - 1
    scores = beta.placebo_score(context, data, last) if knobs.BETA_PLACEBO else score(context, data)
    closes = pd.Series(data["close"][last], index=data["codes"])
    equity = float(context.account.cash) + sum(_price(closes, code) * qty for code, qty in positions.items())
    eligible_cash = CASH_BUFFER * equity / seats
    held = np.isin(data["codes"], list(positions))
    affordable = data["close"][last] * LOT <= eligible_cash
    buyable, betas = beta.buyable(context, data, REVIEW_CALENDAR_DAYS)
    keep_mask = panel.tradable(data, last) & np.isfinite(scores) & ((affordable & buyable) | held)
    ranked = pd.DataFrame({
        "ts_code": data["codes"][keep_mask],
        "score": scores[keep_mask],
        "close": data["close"][last][keep_mask],
        "beta": betas[keep_mask],
    }).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < seats:
        raise RuntimeError(f"only {len(ranked)} eligible names for a {seats}-seat book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]

    sells = _sell_orders(positions, rank, sell_at, leg, seats) if sell_at is not None else []
    if buy_at is None:
        return sells
    # Only sells that execute no later than the buys fund them (open_close, and
    # the frozen book's same-instant case); a close_open 15:00 sell funds the
    # next morning's buys, never this one's.
    funding = sells if sell_at is not None and sell_at <= buy_at else []
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in funding)
    keep = [code for code in positions if code not in {order["symbol"] for order in funding}]
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
        "unaffordable_share": round(1.0 - float(affordable[panel.tradable(data, last)].mean()), 4),
    }
    betas_by_code = ranked.set_index("ts_code")["beta"] if knobs.BETA_FLOOR else None
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, leg, meta, betas_by_code)


def score(context, data):
    """Scores in [0, 1] on the newest row: the bag's mean rank."""

    last = len(data["dates"]) - 1
    return model.score(context, data, [last])[0]


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


def _buy_orders(book, keep, prices, budget, book_value, execute_at, leg, meta, betas=None):
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
            **({"beta_exante": round(float(betas[code]), 4)} if betas is not None else {}),
        })
        remaining -= quantity * price
    return orders
