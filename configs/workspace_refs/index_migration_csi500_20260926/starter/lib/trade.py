"""A SEATS-seat equal-cash book inside the benchmark, reviewed monthly.

Equal cash is the one weighting scheme the host's zero-skill panel prices for
free: the panel replaces each round trip's name among the entry day's names on
the original's side of benchmark membership that this money can buy, so it
aligns the pool, the calendar and the money per seat -- not a weighting scheme.

Review. The first decision of each calendar month (and any decision while the
book is empty): forced exits first -- a holding that left the newest visible
section (for this arm: also one promoted into the target index), lost its score
or its bar -- then holdings ranked at or beyond SEATS * KEEP_BAND; no swap cap.
The emptied seats are filled with the best-ranked names a seat can buy that are
not held. The review is stateless: the calendar month of the newest visible
trading day is compared with the decision day's, so a cold worker and a warm
one emit the same orders.

Refill. On any later decision of the month that finds fewer than SEATS
holdings -- an entry rejected at the limit or for a missing price, or cut short
by cash -- the book buys the best-ranked affordable names it does not hold into
the empty seats, and sells nothing.

Sizing. Seat cash is book value / SEATS, the book value counting cash (plus the
day's sell proceeds at a haircut) times `knobs.CAPITAL` and the kept holdings at
the T-1 close. Quantities are rounded to the NEAREST 100-share lot and then
clipped to the cash left, since `context.account` never changes mid-call. Sells
09:30, buys 15:00 of the same day. Affordability is a BUY-side rule only: a
holding whose lot outgrew a seat is never force-sold. A limit-up open or a
suspension on the day is the Broker's to reject, and it is reported, not
retried within the call.

Ties are broken by a fixed hash of the code, unrelated to board or exchange.
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
    review = not positions or _due(context, decision_at)
    if not review and len(positions) >= knobs.SEATS:
        return []

    frame, last_day, section = data.pool(context, positions)
    tradable = frame[frame["tradable"]]
    raw, meta = score.score(context, list(tradable.index))
    leg = "c_shuf" if knobs.SHUFFLE else score.NAME
    values = score.shuffle(raw, decision_at) if knobs.SHUFFLE else raw
    ranked = tradable.loc[values.dropna().index].assign(value=values.dropna())
    ranked["tie"] = pd.util.hash_pandas_object(ranked.index.to_series(), index=False).to_numpy()
    ranked = ranked.sort_values(["value", "tie"], ascending=[False, True])
    if len(ranked) < knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} scored names for {knobs.SEATS} seats")
    rank = {code: position for position, code in enumerate(ranked.index)}
    prices = frame["close"]

    drops = _exits(positions, rank) if review else []
    keep = [code for code in positions if code not in drops]
    proceeds = sum(_price(prices, code) * positions[code] for code in drops)
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * knobs.CAPITAL
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / knobs.SEATS
    affordable = [code for code in ranked.index if ranked.at[code, "close"] * LOT <= seat_cash]
    buys = [code for code in affordable if code not in positions][:max(0, knobs.SEATS - len(keep))]
    target = keep + buys
    meta.update({
        "leg": leg,
        "review": bool(review),
        "seats": knobs.SEATS,
        "book_size": len(target),
        "exits": len(drops),
        "pool": int(len(tradable)),
        "scored": int(len(ranked)),
        "section": section,
        "last_bar": last_day,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4),
        "book_index_weight_pct": round(float(frame["weight"].reindex(target).fillna(0.0).sum()), 3),
        **score.describe(raw, target),
    })
    sells = [{"symbol": code, "action": "sell", "quantity": int(positions[code]),
              "execute_at": sell_at.isoformat(), "reason": leg + "_exit"} for code in sorted(drops)]
    return sells + _buy_orders(buys, prices, budget, seat_cash, book_value, buy_at, leg, meta)


def _due(context, decision_at):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision_at.year, decision_at.month)


def _exits(positions, rank):
    """Holdings to sell at a review: forced exits, then those ranked at or beyond the keep band."""

    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions if code in rank and rank[code] >= knobs.SEATS * knobs.KEEP_BAND]
    return forced + outside


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(buys, prices, budget, seat_cash, book_value, execute_at, leg, meta):
    orders = []
    remaining = budget
    for code in buys:
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
            "execute_at": execute_at.isoformat(), "reason": leg + "_entry",
            "target_weight": round(seat_cash / book_value, 6) if book_value > 0 else 0.0,
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
