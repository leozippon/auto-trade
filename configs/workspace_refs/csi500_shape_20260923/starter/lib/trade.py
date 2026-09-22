"""SEATS equal-cash CSI 500 constituents at CNY 1m, reviewed MONTHLY.

The shape is deliberately plain, because the arm is about the pool and not about
the construction. Equal cash is the one weighting scheme the host's zero-skill
panel already prices for free: the panel replaces each round trip's name among
that entry day's names on the original's side of INDEX membership that this
money can buy, so it aligns the pool, the calendar, the money per seat and the
costs -- but it does not align a weighting scheme. Anything other than equal
cash would owe its own same-batch, same-score equal-cash control before its
reading meant anything, and this arm has no spare legs for that.

Why 50 seats at CNY 1m. A seat is book value / SEATS, around CNY 19k at full
investment, so one 100-share lot fits any constituent under about CNY 194 a
share -- affordability stops being the price factor it is on a small account,
which is the whole reason this arm is at 1m and not at 100k. The affordable
share is still a required reading, because a CSI 500 section is not the same
price distribution as a CSI 300 one and the arithmetic has never been measured
on it.

Why monthly. Membership only refreshes when a new month-end section becomes
visible, so a faster review re-ranks the same names more often without making
the index move faster, and it multiplies turnover for nothing. At MAX_SWAPS
replacements over SEATS seats, 12 reviews a year is
2 * 12 * MAX_SWAPS / SEATS = 2.4x a year of two-sided turnover plus the opening
build. The binding constraint at this cadence is the OTHER end: the forward test
requires at least one round trip a month, and a repository arm with an otherwise
passing book died on exactly that at 0.1 a month. MAX_SWAPS is sized for margin
above that floor, and the measured round trips a month is a nomination reading
in `families.md`, not an afterthought.

No per-industry cap. At 50 equal seats a single SW level-1 industry would need
15 of them to reach the 0.30 exposure requirement, and this repository has
already measured an ex-post per-industry cap costing 8 pp/yr on a book that did
not need one. So the exposure requirement is a READING here with a registered
response, not a construction rule -- `families.md` says what happens if it
breaches.

Forced exits, keep band and replacement. A holding that is no longer a rankable
constituent is always sold -- the section IS the universe. A holding still
ranked inside SEATS * KEEP_BAND is kept; one outside the band is sold, at most
MAX_SWAPS names changing a review with forced exits counted first.

Affordability is a buy-side rule only. A HOLDING whose lot outgrew a seat is
never force-sold: selling a name because it rose is a reverse-momentum trade the
account did not ask for.

Sizing: 100-share lots rounded to the NEAREST lot and then clipped to the cash
on hand, never rounded down -- rounding down puts a systematic fraction of a
seat back in cash. Sells are timed 09:30 and buys 15:00 of the same day, so
proceeds are credited before buys are sized; `context.account` never changes
mid-call, so the buy budget is decremented locally at the T-1 close.

The review is stateless: the calendar month of the newest visible trading day is
compared with the decision day's, so a cold worker and a warm one emit the same
orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, index, labels, primary, state

SEATS = 50
KEEP_BAND = 2.0
MAX_SWAPS = 5
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
    meta_info = {
        "candidate": candidate,
        "index_code": index.INDEX_CODE,
        "horizon": labels.HOLD,
        "seats": SEATS,
        "book_size": len(book),
        "pool": int(len(ranked)),
        "section": section,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4) if seat_cash > 0 else 0.0,
        "book_index_weight_pct": round(float(weights.reindex(book).fillna(0.0).sum()), 3),
        **state.report(context),
    }
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, candidate, meta_info)


def _book(ranked, keep, seat_cash):
    """The names this review wants to hold: the kept ones, then the best affordable newcomers."""

    book = list(keep)
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    for code in affordable["ts_code"]:
        if len(book) >= SEATS:
            break
        if code not in book:
            book.append(code)
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
