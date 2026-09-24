"""A SEATS-seat equal-cash book inside the benchmark, reviewed monthly, refilled daily.

Equal cash is the one weighting scheme the host's zero-skill panel prices for
free: the panel replaces each round trip's name among that entry day's names
on the original's side of constituent membership that this money can buy, so
it aligns the pool, the calendar and the money per seat -- not a weighting
scheme. Any non-equal weighting would owe its own same-batch equal-cash leg.

Review (the first decision of each calendar month, or any decision while the
account is flat). Exits, all unconditional and sold at 09:30: a holding that
is no longer in the pool (left the section, lost its T-1 bar or its score,
became ST); a holding ranked at or beyond SEATS * KEEP_BAND; and, if a
reclassification left an industry above the cap, its worst-ranked excess.
There is no swap cap. Entries, bought at 15:00: walking the ranking from the
top, names not held whose one 100-share lot at the T-1 close fits a seat and
whose industry is under floor(INDUSTRY_CAP * SEATS) names, kept names
included, until SEATS names. A holding whose lot outgrew a seat is never sold
for it: affordability is a buy-side rule.

Refill (any other decision while fewer than SEATS names are held -- a buy
rejected at the limit or for a missing price, or cut short by cash): the same
entry walk into the empty seats only, nothing sold, nothing re-ranked out.

Sizing. A seat is `book value / SEATS`; a name's money is a seat, rounded to
the NEAREST 100-share lot at the T-1 close and then clipped to the cash left,
so realised weights scatter around the target instead of sitting a quarter of
a lot below it. Sells are timed 09:30 and buys 15:00 of the same day, so the
proceeds are credited before the buys; `context.account` never changes within
a call, so the buy budget is decremented locally. A limit-up open, a limit-up
close or a suspension on the day is the Broker's to reject; it is reported,
not retried, and the next decision's refill takes the empty seat.

The review is stateless: the calendar month of the newest visible trading day
is compared with the decision day's, so a cold worker and a warm one emit the
same orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, knobs, score

LOT = 100
SELL_HAIRCUT = 0.98


def run(context):
    decision_at = pd.Timestamp(context.inference_at)
    sell_at = decision_at.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision_at > buy_at:
        return []
    if decision_at > sell_at:
        sell_at = buy_at
    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    cash = float(context.account.cash)
    if not positions and not knobs.CAPITAL / 2 <= cash <= knobs.CAPITAL * 2:
        raise RuntimeError(f"a flat account holds {cash:.0f} in cash, but this shape is registered for "
                           f"knobs.CAPITAL = {knobs.CAPITAL}: align it to broker_replay.initial_cash")
    review = not positions or _due(context, decision_at)
    if not review and len(positions) >= knobs.SEATS:
        return []

    pool, prices, industry, section, weights = data.read(context, positions)
    ranked = score.ranked(context, pool)
    if len(ranked) < 2 * knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} scorable {knobs.INDEX} constituents for {knobs.SEATS} seats")
    rank = dict(zip(ranked["ts_code"], ranked["rank"]))
    cap = int(np.floor(knobs.INDUSTRY_CAP * knobs.SEATS))

    forced = [code for code in positions if code not in rank] if review else []
    band = [code for code in positions
            if review and code in rank and rank[code] >= knobs.SEATS * knobs.KEEP_BAND]
    keep = [code for code in positions if code not in forced and code not in band]
    excess = _over_cap(keep, rank, industry, cap) if review else []
    keep = [code for code in keep if code not in excess]
    drops = forced + band + excess

    proceeds = sum(_price(prices, code) * positions[code] for code in drops)
    budget = (cash + proceeds * SELL_HAIRCUT) * knobs.CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / knobs.SEATS
    counts = {}
    for code in keep:
        counts[industry.get(code, "未分类")] = counts.get(industry.get(code, "未分类"), 0) + 1
    entries = []
    for code, close, name in zip(ranked["ts_code"], ranked["close"], ranked["industry"]):
        if len(keep) + len(entries) >= knobs.SEATS:
            break
        if code in positions or close * LOT > seat_cash or counts.get(name, 0) >= cap:
            continue
        entries.append(code)
        counts[name] = counts.get(name, 0) + 1
    book = keep + entries
    meta = {
        "candidate": score.label(),
        "index": knobs.INDEX,
        "review": review,
        "seats": knobs.SEATS,
        "keep_band": knobs.KEEP_BAND,
        "range_days": knobs.RANGE_DAYS,
        "pool": int(len(ranked)),
        "section": section,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4),
        "book_size": len(book),
        "exits_forced": len(forced),
        "exits_band": len(band),
        "exits_cap": len(excess),
        "top_industry_names": max(counts.values()) if counts else 0,
        "book_index_weight_pct": round(float(weights.reindex(book).fillna(0.0).sum()), 3),
    }
    return _sell_orders(positions, drops, sell_at) + _buy_orders(entries, prices, budget, seat_cash, book_value,
                                                                 buy_at, meta)


def _due(context, decision_at):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision_at.year, decision_at.month)


def _over_cap(keep, rank, industry, cap):
    """The worst-ranked kept names of any industry holding more than `cap` of them."""

    groups = {}
    for code in keep:
        groups.setdefault(industry.get(code, "未分类"), []).append(code)
    excess = []
    for members in groups.values():
        if len(members) > cap:
            excess += sorted(members, key=lambda code: -rank[code])[:len(members) - cap]
    return excess


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _sell_orders(positions, drops, execute_at):
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": score.label() + "_exit"}
        for code in sorted(drops)
    ]


def _buy_orders(entries, prices, budget, seat_cash, book_value, execute_at, meta):
    orders = []
    remaining = budget
    for code in entries:
        price = _price(prices, code)
        if price <= 0:
            continue
        quantity = int(min(remaining, seat_cash) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": score.label() + "_entry",
            "target_weight": round(1.0 / knobs.SEATS, 6),
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
