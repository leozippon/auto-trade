"""A SEATS-seat equal-cash book inside the benchmark, reviewed MONTHLY, with refill.

The book is the lineage's book because this arm is about a score, not about
construction. Equal cash is the one weighting scheme the host's zero-skill
panel prices for free: the panel replaces each round trip's name among that
entry day's names on the original's side of constituent membership that this
money can buy, so it aligns the pool, the calendar and the money per seat --
but not a weighting scheme, which is why every leg here is equal cash.

Review. On the first decision of each calendar month (and on any decision while
the book is empty) forced exits are sold -- a holding that left the newest
section, lost its bar or its score, or turned ST -- and so is every holding
ranked outside SEATS * KEEP_BAND; there is no swap cap. The empty seats are then
filled in rank order among names whose 100-share lot at the T-1 close fits a
seat, with at most floor(INDUSTRY_SHARE * SEATS) names per SW L1 industry
counting the names kept (the quota binds entries only; a kept name is never
sold for it). Affordability is a BUY-side rule too: a holding whose lot outgrew
a seat is never force-sold.

Refill. Between reviews a book holding fewer than SEATS names -- an entry the
Broker rejected at the limit or for a missing price, or one cut short by cash --
looks again on every decision and buys the best-ranked eligible names it does
not hold into the empty seats; it sells nothing on those days.

Sizing. A name's money is `book value / seats`, in 100-share lots rounded to
the NEAREST lot and then clipped to the cash on hand. Sells are timed 09:30 and
buys 15:00 of the same day, so proceeds are credited before buys are sized;
`context.account` never changes mid-call, so the buy budget is decremented
locally at the T-1 close.

The review is stateless: the calendar month of the newest visible session is
compared with the decision day's, so a cold worker and a warm one emit the
same orders. Ties are broken by a fixed hash of the code: stable across
decisions and unrelated to board or exchange.
"""

from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

from lib import candidates, data, knobs

LOT = 100
SELL_HAIRCUT = 0.98


def run(context, candidate):
    decision_at = pd.Timestamp(context.inference_at)
    sell_at = decision_at.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision_at > buy_at:
        return []
    if decision_at > sell_at:
        sell_at = buy_at
    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    review = not positions or _due(context, decision_at)
    if not review and len(positions) >= knobs.SEATS:
        return []

    panel = data.panel(context, sorted(positions))
    score = candidates.score_of(context, panel, candidate)
    close = panel["close"][:, -1]
    prices = pd.Series(close, index=[str(code) for code in panel["symbols"]])
    industry = dict(zip(prices.index, panel["industry"]))
    scored = np.isfinite(score) & np.isfinite(close) & (close > 0)
    ranked = pd.DataFrame({"ts_code": prices.index[scored], "score": score[scored], "close": close[scored]})
    ranked["tie"] = pd.util.hash_pandas_object(ranked["ts_code"], index=False).to_numpy()
    ranked = ranked.sort_values(["score", "tie"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} scorable constituents for {knobs.SEATS} seats")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}

    drops = _exits(positions, rank) if review else []
    keep, budget, book_value, seat_cash = _money(context, positions, drops, prices)
    affordable = ranked.loc[ranked["close"] * LOT <= seat_cash, "ts_code"]
    buys = _entrants(keep, positions, affordable, industry)
    book = keep + buys
    weight = pd.Series(panel["index_weight"], index=prices.index)
    meta = {
        "candidate": candidate,
        "review": bool(review),
        "seats": knobs.SEATS,
        "book_size": len(book),
        "exits": len(drops),
        "pool": int(data.tradable(panel).sum()),
        "scored": int(len(ranked)),
        "section": panel["section"],
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4),
        "book_index_weight_pct": round(float(weight.reindex(book).fillna(0.0).sum()), 3),
        "top_industry_names": max(Counter(industry[code] for code in book).values()) if book else 0,
    }
    return _sell_orders(positions, drops, sell_at, candidate) + _buy_orders(
        buys, prices, budget, seat_cash, buy_at, book_value, candidate, meta)


def _due(context, decision_at):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision_at.year, decision_at.month)


def _exits(positions, rank):
    """Held names to sell at a review: off the ranked pool, or ranked outside the keep band."""

    return sorted(code for code in positions
                  if code not in rank or rank[code] >= knobs.SEATS * knobs.KEEP_BAND)


def _money(context, positions, drops, prices):
    """(kept codes, buy budget, book value, cash per seat) once `drops` are sold."""

    proceeds = sum(_price(prices, code) * positions[code] for code in drops)
    keep = sorted(code for code in positions if code not in drops)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * knobs.CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    return keep, budget, book_value, book_value / knobs.SEATS


def _entrants(keep, positions, affordable, industry):
    """Best-ranked affordable names not held, within the industry quota, into the empty seats."""

    cap = int(np.floor(knobs.INDUSTRY_SHARE * knobs.SEATS))
    count = Counter(industry[code] for code in keep)
    chosen = []
    for code in affordable:
        if len(keep) + len(chosen) >= knobs.SEATS:
            break
        if code in positions or count[industry[code]] >= cap:
            continue
        chosen.append(code)
        count[industry[code]] += 1
    return chosen


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _sell_orders(positions, drops, execute_at, candidate):
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in drops
    ]


def _buy_orders(buys, prices, budget, seat_cash, execute_at, book_value, candidate, meta):
    orders = []
    remaining = budget
    for code in buys:
        price = _price(prices, code)
        if price <= 0 or seat_cash <= 0:
            continue
        quantity = int(min(remaining, seat_cash) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": candidate + "_entry",
            "target_weight": round(1.0 / knobs.SEATS, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
