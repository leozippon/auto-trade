"""A 12-seat book inside the CSI 300, reviewed weekly at a CNY 100k account.

The account decides the shape, and at CNY 100k the arithmetic is tight
(`references/capital-arithmetic.md`): 12 seats leave about CNY 8,100 a seat, so
one lot fits any constituent under about CNY 81 -- four fifths of the index --
and the CNY 5 minimum commission is about 6.2 bp a side. More seats would cut
the affordable share and raise the commission floor; fewer would put the book
over the industry-concentration requirement on three names.

    seats        SEATS, at most INDUSTRY_CAP per SW level-1 industry, which
                 holds the time-weighted single-industry weight under 0.30 by
                 construction rather than by hoping
    review       the first decision of each ISO week (and any decision while
                 flat). The label horizon is 10 trading days; a monthly review
                 of a 10-day learned score reads near zero in this repository.
                 Membership still only refreshes when a new month-end section
                 becomes visible -- reviewing weekly trades the same section
                 more often, it does not make the index move faster
    keep band    a holding still ranked inside SEATS * KEEP_BAND is kept
    swaps        at most MAX_SWAPS replacements a review, forced exits first

How the meta stage is used is `main.CANDIDATE`, and only this file reads it:

    m1          filter. The primary's top `SEATS * PICK_POOL` names are the
                pool; the `GATE_SHARE` of that pool with the LOWEST meta
                probability is passed over and each seat goes to the next
                primary rank that survived. Equal cash, so the host panel --
                which redraws names but not weights -- prices this book exactly
                and no same-score-equal-cash attribution leg is owed
    c_primary   the control: the same code with the gate removed, seats filled
                straight down the primary's ranking. Everything else is
                identical, which is what makes it a control
    m_size      the gate used to SIZE rather than to filter, tilting weights by
                the meta probability's RANK inside the book. NOT equal cash, so
                `families.md` P5 makes a same-score-equal-cash leg in the same
                batch mandatory
    m_stack     m1 with the horizon twin's score added to the meta features

The gate is relative, not an absolute probability floor, and that is a measured
choice rather than a stylistic one: on the probed decision the shipped
classifier's output spans 0.537 to 0.564 across the whole cross-section
(`sources.md`), so any fixed floor either passes everything or blocks
everything. A relative gate keeps its strength when the classifier's calibration
moves, and `GATE_SHARE = 0` turns it off, which is the same book as the control.

Forced exits, unconditional and first: a holding that left the newest visible
section is sold -- the section IS the universe -- and so is one that lost its
score or its bar.

Affordability is a buy-side rule only. A HOLDING whose lot outgrew a seat is
never force-sold: selling a name because it rose is a reverse-momentum trade
the account did not ask for.

Sizing: 100-share lots rounded to the NEAREST lot and then clipped to the cash
on hand, never rounded down -- rounding down puts a systematic quarter of a seat
back in cash. Sells are timed 09:30 and buys 15:00 of the same day, so proceeds
are credited before buys are sized; `context.account` never changes mid-call, so
the buy budget is decremented locally at the T-1 close.

The review is stateless: the ISO week of the newest visible trading day is
compared with the decision day's, so a cold worker and a warm one emit the same
orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, index, meta, primary

SEATS = 12
INDUSTRY_CAP = 3
KEEP_BAND = 2.0
MAX_SWAPS = 2
PICK_POOL = 3                # the primary's top SEATS * PICK_POOL names the gate may choose from
GATE_SHARE = 1.0 / 3.0       # the share of that pool the gate passes over; 0 disables the gate
SIZE_TILT = 0.5              # m_size only: weight spans (1 - SIZE_TILT) to (1 + SIZE_TILT)
CASH_BUFFER = 0.97
SELL_HAIRCUT = 0.98
LOT = 100


def run(context, candidate):
    decision_at = pd.Timestamp(context.inference_at)
    sell_at = decision_at.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision_at > buy_at:
        return []
    if decision_at > sell_at:
        sell_at = buy_at
    positions = {str(k): int(v) for k, v in dict(context.account.positions).items() if int(v) > 0}
    if positions and not _new_week(context, decision_at):
        return []

    panel = data.wide(context, data.DECISION_LOOKBACK_DAYS)
    last = len(panel["dates"]) - 1
    score, probability = meta.decision(context, panel, candidate)
    keep_mask = data.tradable(panel, last) & np.isfinite(score)
    ranked = pd.DataFrame({
        "ts_code": [str(code) for code in panel["symbols"][keep_mask]],
        "rank_score": primary.pct_rank(score[keep_mask]),
        "probability": probability[keep_mask],
        "close": panel["raw_close"][keep_mask, last],
        "industry": panel["industry"][keep_mask],
    }).sort_values(["rank_score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < SEATS:
        raise RuntimeError(f"only {len(ranked)} scorable constituents for a {SEATS}-seat book")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    members, section = index.latest(panel["sections"])
    weights = members.set_index("ts_code")["weight"]

    sells = _sell_orders(positions, rank, sell_at, candidate)
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in {order["symbol"] for order in sells}]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / SEATS if SEATS else 0.0
    book, gated = _book(ranked, keep, seat_cash, candidate)
    share = _weights(ranked, book, candidate)
    meta_info = {
        "candidate": candidate,
        "seats": SEATS,
        "book_size": len(book),
        "section": section,
        "gated_out": gated,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4) if seat_cash > 0 else 0.0,
        "book_index_weight_pct": round(float(weights.reindex(book).fillna(0.0).sum()), 3),
    }
    return sells + _buy_orders(book, keep, prices, budget, book_value, share, buy_at, candidate, meta_info)


def _gated_out(ranked, candidate):
    """The pool names the relative gate passes over, in primary-rank order."""

    if candidate not in ("m1", "m_stack") or GATE_SHARE <= 0:
        return set()
    pool = ranked.head(SEATS * PICK_POOL)
    usable = pool[np.isfinite(pool["probability"].to_numpy())]
    if usable.empty:
        raise RuntimeError("the meta stage produced no probability for the primary's pool")
    drop = int(np.ceil(GATE_SHARE * len(usable)))
    if drop <= 0:
        return set()
    worst = usable.sort_values(["probability", "ts_code"], ascending=[True, True]).head(drop)
    return set(worst["ts_code"])


def _book(ranked, keep, seat_cash, candidate):
    """(the names this review wants to hold, how many picks the gate passed over).

    Two passes. The first fills seats in primary-rank order from the names the
    gate let through; the second, only if seats are still open, falls back to
    the gated ones in the same order rather than leaving the account in cash --
    a gate is a preference over a ranking, not a reason to stop investing.
    """

    industry = ranked.set_index("ts_code")["industry"]
    counts = {}
    book = []
    for code in keep:
        book.append(code)
        label = str(industry.get(code, "未分类"))
        counts[label] = counts.get(label, 0) + 1
    blocked = _gated_out(ranked, candidate)
    pool = ranked.head(SEATS * PICK_POOL) if candidate in ("m1", "m_stack") else ranked
    affordable = pool[pool["close"] * LOT <= seat_cash] if seat_cash > 0 else pool
    for allow_blocked in (False, True):
        for code in affordable["ts_code"]:
            if len(book) >= SEATS:
                break
            if code in book or (code in blocked and not allow_blocked):
                continue
            label = str(industry.get(code, "未分类"))
            if counts.get(label, 0) >= INDUSTRY_CAP:
                continue
            book.append(code)
            counts[label] = counts.get(label, 0) + 1
        if len(book) >= SEATS or not blocked:
            break
    return book, len(blocked)


def _weights(ranked, book, candidate):
    """Target weight per name: equal cash, or tilted by the meta probability's rank for m_size."""

    if candidate != "m_size" or not book:
        return pd.Series(1.0 / max(1, len(book)), index=pd.Index(book))
    probability = ranked.set_index("ts_code")["probability"].reindex(pd.Index(book))
    position = probability.rank(pct=True).fillna(0.5).to_numpy()
    weight = pd.Series(1.0 + SIZE_TILT * (2.0 * position - 1.0), index=pd.Index(book))
    return weight / float(weight.sum())


def _new_week(context, decision_at):
    start = (context.inference_at - timedelta(days=10)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return latest.isocalendar()[:2] != decision_at.isocalendar()[:2]


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


def _buy_orders(book, keep, prices, budget, book_value, share, execute_at, candidate, meta_info):
    orders = []
    remaining = budget
    for code in [name for name in book if name not in keep]:
        price = _price(prices, code)
        target = float(share.get(code, 0.0))
        if price <= 0 or target <= 0:
            continue
        quantity = int(min(remaining, book_value * target) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": candidate + "_entry",
            "target_weight": round(target, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta_info,
        })
        remaining -= quantity * price
    return orders
