"""A 30-seat equal-cash book inside the CSI 300, reviewed weekly at a CNY 1M account.

Shape, all four numbers registered before the first batch:

    seats        SEATS, equal cash, no weighting scheme -- so the host panel,
                 which redraws names but not weights, prices this book exactly
                 and the pack owes no same-score-equal-cash attribution leg
    review       the first decision of each ISO week (and any decision while
                 flat). The label horizon is 10 trading days; a monthly review
                 of a 10-day learned score reads near zero in this repository
    keep band    a holding still ranked inside SEATS * KEEP_BAND is kept
    swaps        at most MAX_SWAPS replacements a review, forced exits first

Forced exits, in this order and unconditionally: a holding that left the newest
visible constituent section is sold -- the section IS the universe, and "it is
probably still in the index" has no data behind it -- and so is a holding that
lost its score or its bar.

Affordability is a buy-side rule only. A seat is `book value / seats`; at CNY 1M
and 30 seats that is about CNY 32,000, which one lot of 98-99 % of the
constituents fits inside, and the minimum commission is about 1.5 bp a side. A
HOLDING whose lot outgrew a seat is never force-sold: selling a name because it
rose is a reverse-momentum trade the account did not ask for.

Sizing: equal cash, in 100-share lots rounded to the NEAREST lot and then
clipped to the cash on hand. Sells are timed 09:30 and buys 15:00 of the same
day, so proceeds are credited before buys are sized; `context.account` never
changes mid-call, so the buy budget is decremented locally at the T-1 close.

The review is stateless: the ISO week of the newest visible trading day is
compared with the decision day's, so a cold worker and a warm one emit the same
orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index, model, panel, tree

SEATS = 30
KEEP_BAND = 2.0
MAX_SWAPS = 4
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100
REVIEW_CALENDAR_DAYS = 200   # SEQ_LEN bars plus holidays, plus one constituent-section gap


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
    last = len(data["dates"]) - 1
    keep_mask = panel.tradable(data, last) & np.isfinite(scores)
    ranked = pd.DataFrame({
        "ts_code": data["codes"][keep_mask],
        "score": scores[keep_mask],
        "close": data["close"][last][keep_mask],
    }).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < SEATS:
        raise RuntimeError(f"only {len(ranked)} scorable constituents for a {SEATS}-seat book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    members, section = index.latest(data["sections"])
    weights = members.set_index("ts_code")["weight"]

    equity = float(context.account.cash) + sum(_price(prices, code) * qty for code, qty in positions.items())
    sells = _sell_orders(positions, rank, sell_at, candidate)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / SEATS if SEATS else 0.0
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    book = keep + [code for code in affordable["ts_code"] if code not in keep][: max(0, SEATS - len(keep))]
    meta = {
        "candidate": candidate,
        "seats": SEATS,
        "book_size": len(book),
        "section": section,
        "equity": round(equity, 2),
        "unbuyable_share": round(1.0 - len(affordable) / max(1, len(ranked)), 4),
        "book_index_weight_pct": round(float(weights.reindex(book).fillna(0.0).sum()), 3),
    }
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, candidate, meta)


def score(context, data, candidate):
    """Scores in [0, 1] on the newest row for the candidate `main.CANDIDATE` names."""

    if candidate == "g1":
        return model.score(context, data)
    if candidate == "c_lgbm":
        return tree.score(context, data)
    if candidate == "g1r":
        return _orthogonal(model.score(context, data), tree.score(context, data))
    raise ValueError(f"unknown candidate: {candidate}")


def _orthogonal(sequence_score, control_score):
    """The sequence score with the control's score regressed out, re-ranked.

    The registered variant of `families.md`: it answers "what is left of the
    sequence model once the tree has had everything it can take", and it is the
    only shape in which the arm may keep going after the incremental-information
    probe reads a high rank correlation.
    """

    both = np.isfinite(sequence_score) & np.isfinite(control_score)
    out = np.full(len(sequence_score), np.nan)
    if both.sum() < 3:
        return out
    x = control_score[both]
    y = sequence_score[both]
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


def _sell_orders(positions, rank, execute_at, candidate):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= SEATS * KEEP_BAND]
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
