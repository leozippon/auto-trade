"""A 12-seat equal-cash book inside the CSI 300, reviewed MONTHLY at a CNY 100k account.

The construction is deliberately the carrier pack's, minus its meta gate and
with one knob moved, because the arm is about the label and not about the book.
Anything else that changed would show up inside the candidate-minus-control
difference the whole arm is read on.

    seats        SEATS, at most INDUSTRY_CAP per SW level-1 industry, which
                 holds the time-weighted single-industry weight under 0.30 by
                 construction rather than by hoping
    review       the first decision of each calendar month, and any decision
                 while flat
    keep band    a holding still ranked inside SEATS * KEEP_BAND is kept
    swaps        at most MAX_SWAPS replacements a review, forced exits first

Why monthly, and why it is not free. The identical 12-seat book reviewed WEEKLY
measured 14.06x/yr turnover on the host (14.2x with the gate its successor
dropped), breaching its own pack's 12x ceiling,
and at CNY 100k the CNY 5 commission floor prices that: 17-18x/yr costs
1.05-2.32 % of equity a year, against 0.15-0.41 % at 1.5-2.8x. Cadence is the
only lever that moves turnover without touching the score, so this arm pulls it
once and holds it: 12 reviews a year replacing at most MAX_SWAPS of SEATS is
2 * 12 * MAX_SWAPS / SEATS = 6.0x a year of two-sided turnover plus the opening
build, against a ceiling of 12x. Membership only refreshes when a new month-end
section becomes visible anyway, so a monthly review is also the cadence at which
the index itself actually moves.

The cost of pulling it is on the other side and is registered, not hidden: the
FORWARD test requires at least one round trip a month, and an explicit-risk-model
arm in this repository died on exactly that condition at 0.1 round trips a month
while passing everything else. MAX_SWAPS is 3 rather than
the carrier pack's 2 for exactly this reason: at 2 this book measured 1.27
round trips a month on the host, only 27 % above the floor. The swap cap is a
ceiling on activity, not a guarantee of it, and the keep band can suppress
swaps for months at a time. Both
the ceiling and the floor are nomination conditions in `families.md`, and the
measured round trips a month is a required reading, not an afterthought.

The second cost is the one this arm exists to measure. A 10-trading-day label
reviewed every 20-odd trading days is scoring a horizon the book does not hold,
which is why horizon is the first component of the label family and why `l_h20`
and `l_h40` are registered. `c_base` keeps the 10-day label at the monthly
cadence on purpose: the mismatch is then held constant across the batch, so a
horizon variant's difference is the horizon and not the cadence.

Forced exits, unconditional and first: a holding that left the newest visible
section is sold -- the section IS the universe -- and so is one that lost its
score or its bar.

Affordability is a buy-side rule only. A HOLDING whose lot outgrew a seat is
never force-sold: selling a name because it rose is a reverse-momentum trade the
account did not ask for.

Sizing: 100-share lots rounded to the NEAREST lot and then clipped to the cash
on hand, never rounded down -- rounding down puts a systematic quarter of a seat
back in cash. Sells are timed 09:30 and buys 15:00 of the same day, so proceeds
are credited before buys are sized; `context.account` never changes mid-call, so
the buy budget is decremented locally at the T-1 close. That pair of timestamps
is also what `lib/labels.py` measures its execution-matched timing between.

The review is stateless: the calendar month of the newest visible trading day is
compared with the decision day's, so a cold worker and a warm one emit the same
orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, index, labels, primary, state

SEATS = 12
INDUSTRY_CAP = 3
KEEP_BAND = 2.0
MAX_SWAPS = 3
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
    if positions and not _new_month(context, decision_at):
        return []

    panel = data.wide(context, data.DECISION_LOOKBACK_DAYS)
    last = len(panel["dates"]) - 1
    score = primary.score(context, panel)
    keep_mask = data.tradable(panel, last) & np.isfinite(score)
    ranked = pd.DataFrame({
        "ts_code": [str(code) for code in panel["symbols"][keep_mask]],
        "rank_score": primary.pct_rank(score[keep_mask]),
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
    book = _book(ranked, keep, seat_cash)
    label = labels.spec(candidate)
    meta_info = {
        "candidate": candidate,
        "index_code": index.INDEX_CODE,
        "horizon": label.horizon,
        "timing": label.timing,
        "seats": SEATS,
        "book_size": len(book),
        "section": section,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4) if seat_cash > 0 else 0.0,
        "book_index_weight_pct": round(float(weights.reindex(book).fillna(0.0).sum()), 3),
        **state.report(context),
    }
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, candidate, meta_info)


def _book(ranked, keep, seat_cash):
    """The names this review wants to hold: the kept ones, then the best affordable newcomers."""

    industry = ranked.set_index("ts_code")["industry"]
    counts = {}
    book = []
    for code in keep:
        book.append(code)
        label = str(industry.get(code, "未分类"))
        counts[label] = counts.get(label, 0) + 1
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    for code in affordable["ts_code"]:
        if len(book) >= SEATS:
            break
        if code in book:
            continue
        label = str(industry.get(code, "未分类"))
        if counts.get(label, 0) >= INDUSTRY_CAP:
            continue
        book.append(code)
        counts[label] = counts.get(label, 0) + 1
    return book


def _new_month(context, decision_at):
    start = (context.inference_at - timedelta(days=20)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision_at.year, decision_at.month)


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


def _buy_orders(book, keep, prices, budget, book_value, execute_at, candidate, meta_info):
    orders = []
    remaining = budget
    target = 1.0 / max(1, len(book))
    for code in [name for name in book if name not in keep]:
        price = _price(prices, code)
        if price <= 0:
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
