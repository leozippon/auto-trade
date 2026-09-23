"""The carrier's monthly SEATS-seat long-only book inside the benchmark, for any leg's score.

What the legs share, and why it is fixed. The pool, SEATS equal-cash seats,
the monthly review with at most MAX_SWAPS names changed and the keep band, lot
sizing and the 09:30 sell / 15:00 buy timing are the head packs' `c_base`,
order for order; a leg differs only in the boosters its score averages
(`knobs.SEEDS`). Equal cash is the one weighting scheme the host's zero-skill
panel prices for free, so an active difference between two legs of one batch
is their scores' alone.

Sizing. A name's money is `book value / seats`, in 100-share lots rounded to
the NEAREST lot and then clipped to the cash on hand; rounding down would put a
systematic quarter of a seat back in cash. Sells are timed 09:30 and buys 15:00
of the same day, so proceeds are credited before buys are sized;
`context.account` never changes mid-call, so the buy budget is decremented
locally at the T-1 close.

Forced exits are unconditional and first: a holding that left the newest
visible section is sold -- the section IS the universe -- and so is one that
lost its score or its bar. Affordability is a BUY-side rule only: a holding
whose lot outgrew a seat is never force-sold, because selling a name for
rising is a reverse-momentum trade the account did not ask for. A seat left
empty by a rejected or cash-short entry stays empty until the next review.

The review is stateless: the calendar month of the newest visible trading day
is compared with the decision day's, so a cold worker and a warm one emit the
same orders.

Ties are broken by a fixed hash of the code: stable across decisions, and
unrelated to board or exchange. A single booster's score has no ties; a bag's
rank average can, and the hash settles them the same way every time.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import book, data, index, knobs, primary

LOT = 100
SELL_HAIRCUT = 0.98


def run(context, candidate):
    seeds = knobs.SEEDS[candidate]
    decision_at = pd.Timestamp(context.inference_at)
    sell_at = decision_at.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision_at > buy_at:
        return []
    if decision_at > sell_at:
        sell_at = buy_at
    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    if positions and not _due(context, decision_at):
        return []

    panel = data.wide(context, data.DECISION_LOOKBACK_DAYS)
    score = primary.score(context, panel, seeds)
    last = len(panel["dates"]) - 1
    keep_mask = data.tradable(panel, last) & np.isfinite(score)
    ranked = pd.DataFrame({
        "ts_code": [str(code) for code in panel["symbols"][keep_mask]],
        "rank_score": primary.pct_rank(score[keep_mask]),
        "close": panel["raw_close"][keep_mask, last],
    })
    ranked = ranked[np.isfinite(ranked["close"].to_numpy()) & (ranked["close"] > 0)]
    ranked["tie"] = pd.util.hash_pandas_object(ranked["ts_code"], index=False).to_numpy()
    ranked = ranked.sort_values(["rank_score", "tie"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} scorable constituents for {knobs.SEATS} seats")
    codes = list(ranked["ts_code"])
    rank = {code: position for position, code in enumerate(codes)}
    prices = ranked.set_index("ts_code")["close"]
    members, section = index.latest(panel["sections"])
    weights = members.set_index("ts_code")["weight"]

    drops = book.exits(positions, rank, len(ranked))
    keep, budget, book_value, seat_cash = _money(context, positions, drops, prices)
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    target = keep + book.entrants(keep, positions, list(affordable["ts_code"]))
    buys = [code for code in target if code not in keep]
    meta = {
        "candidate": candidate,
        "seeds": [int(seed) for seed in seeds],
        "max_swaps": knobs.MAX_SWAPS,
        "keep_band": knobs.KEEP_BAND,
        "seats": knobs.SEATS,
        "book_size": len(target),
        "exits": len(drops),
        "pool": int(len(ranked)),
        "section": section,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4) if seat_cash > 0 else 0.0,
        "book_index_weight_pct": round(float(weights.reindex(target).fillna(0.0).sum()), 3),
        "book_tc": _reading(book.transfer(target, codes, ranked["rank_score"])),
        **primary.report(context),
    }
    return _sell_orders(positions, drops, sell_at, candidate) + _buy_orders(
        buys, seat_cash, prices, budget, buy_at, book_value, candidate, meta)


def _money(context, positions, drops, prices):
    """(kept codes, buy budget, book value, cash per seat) once `drops` are sold."""

    proceeds = sum(_price(prices, code) * positions[code] for code in sorted(drops))
    keep = [code for code in positions if code not in drops]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * knobs.CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    return keep, budget, book_value, book_value / knobs.SEATS


def _reading(value):
    """A float reading for the order metadata; None where it is undefined (strict JSON has no NaN)."""

    return round(float(value), 4) if np.isfinite(value) else None


def _due(context, decision_at):
    """True on the first decision of a calendar month: the newest visible trading day is in an earlier month."""

    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return (latest.year, latest.month) != (decision_at.year, decision_at.month)


def _sell_orders(positions, drops, execute_at, candidate):
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drops)
    ]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(buys, seat_cash, prices, budget, execute_at, book_value, candidate, meta):
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
            "target_weight": round(seat_cash / book_value, 6) if book_value > 0 else 0.0,
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
