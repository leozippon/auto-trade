"""The recipe's book: equal cash, weekly review, keep band, swap cap, at a CNY 1M account.

This is the book every [FEAT-14] single-head arm ran (their shared `trade.py`),
kept under its own name because it is not the round's canonical rule-score
book: it caps band exits per review and does not refill seats between reviews.
Two changes only: the score comes from the bag (`score` below), and the
constituent-section order metadata is gone because the pool is not an index.

    seats        knobs.SEATS, equal cash -- the host panel redraws names, not
                 weights, so it prices this book exactly
    review       the first decision of each ISO week (and any decision while flat)
    keep band    a holding still ranked inside SEATS * KEEP_BAND is kept
    swaps        at most MAX_SWAPS band exits a review; forced exits come first
                 and are always sold

Forced exits: a holding that lost its score, its bar or its tradability (ST or
退 in the decision's `universe` name). Affordability is a buy-side rule only:
a seat is `book value / SEATS`, and a holding whose lot outgrew a seat is never
force-sold. Sizing is equal cash in 100-share lots rounded to the nearest lot
and clipped to the cash on hand. Sells are timed 09:30 and buys 15:00 of the
same day, so proceeds are credited before buys; `context.account` never changes
mid-call, so the buy budget is decremented locally at the T-1 close. Ties in
the ranking are broken by code.

The review is stateless: the ISO week of the newest visible trading day is
compared with the decision day's, so a cold worker and a warm one emit the same
orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import knobs, model, panel, tree

KEEP_BAND = 2.0
MAX_SWAPS = 4
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100
REVIEW_CALENDAR_DAYS = 200   # SEQ_LEN bars plus holidays


def run(context):
    candidate = knobs.CANDIDATE
    seats = knobs.SEATS
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
    last = len(data["dates"]) - 1
    keep_mask = panel.tradable(data, last) & np.isfinite(scores)
    ranked = pd.DataFrame({
        "ts_code": data["codes"][keep_mask],
        "score": scores[keep_mask],
        "close": data["close"][last][keep_mask],
    }).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < seats:
        raise RuntimeError(f"only {len(ranked)} scorable names for a {seats}-seat book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]

    equity = float(context.account.cash) + sum(_price(prices, code) * qty for code, qty in positions.items())
    sells = _sell_orders(positions, rank, sell_at, candidate, seats)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / seats
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    book = keep + [code for code in affordable["ts_code"] if code not in keep][: max(0, seats - len(keep))]
    meta = {
        "candidate": candidate,
        "residualised": bool(knobs.RESIDUALISE),
        "seats": seats,
        "book_size": len(book),
        "equity": round(equity, 2),
        "unbuyable_share": round(1.0 - len(affordable) / max(1, len(ranked)), 4),
    }
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, candidate, meta)


def score(context, data, candidate):
    """Scores in [0, 1] on the newest row: the tree's rank, the bag's mean rank, or its tree residual."""

    if candidate == "c_lgbm":
        return tree.score(context, data)
    network = model.score(context, data, candidate)
    if knobs.RESIDUALISE:
        return _orthogonal(network, tree.score(context, data))
    return network


def _orthogonal(network_score, control_score):
    """The network score with the tree's score regressed out (cross-sectional OLS), re-ranked."""

    both = np.isfinite(network_score) & np.isfinite(control_score)
    out = np.full(len(network_score), np.nan)
    if both.sum() < 3:
        return out
    x = control_score[both]
    y = network_score[both]
    variance = float(((x - x.mean()) ** 2).mean())
    slope = float(((x - x.mean()) * (y - y.mean())).mean() / variance) if variance > 0 else 0.0
    residual = y - slope * (x - x.mean())
    order = np.argsort(np.argsort(residual, kind="stable"), kind="stable")
    out[both] = order / max(1, both.sum() - 1)
    return out


def _new_week(context, decision):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return latest.isocalendar()[:2] != decision.isocalendar()[:2]


def _sell_orders(positions, rank, execute_at, candidate, seats):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= seats * KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    replaceable = max(0, MAX_SWAPS - len(forced))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drop)
    ]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(book, keep, prices, budget, book_value, execute_at, candidate, meta):
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
            "execute_at": execute_at.isoformat(), "reason": candidate + "_entry",
            "target_weight": round(target_weight, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
