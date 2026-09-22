"""A SEATS-seat equal-cash book inside the benchmark, reviewed MONTHLY.

The shape is deliberately plain, because this arm is about a block of
information and not about the construction. Equal cash is the one weighting
scheme the host's zero-skill panel already prices for free: the panel replaces
each round trip's name among that entry day's names on the original's side of
constituent membership that this money can buy, so it aligns the pool, the
calendar and the money per seat -- but it does not align a weighting scheme.
Anything other than equal cash would owe its own same-batch, same-score
equal-cash control before its reading means anything.

Why monthly, and what it costs. The last account this lineage traded reviewed
WEEKLY at 12 seats and measured 14x/yr turnover, over its own 12x ceiling, and
that pack's own note says the review cadence -- not the seat count, not the
keep band -- is the lever nobody had pulled. This arm pulls it: 12 reviews a
year instead of ~52. The cost is registered and real: the label is a 10-day
horizon, so a monthly review acts on a score whose horizon has already expired
twice over, and the same lineage's note warns that a monthly review of a
10-day score can read near zero. That tension is this construction's main
risk, it is written into `families.md` as a named kill condition rather than
discovered later, and `MAX_SWAPS` is sized so the book can still move: 8 of 50
seats a review is about 1.9 book turnovers a year one-way. Round 0 recomputes
both turnover and cost from a measured replay instead of from this paragraph.

Exposure is not a function of the market state. The book is SEATS names at
equal cash and `knobs.CASH_BUFFER` of the account, whatever the state columns
read; the market-state block can only change WHICH names fill the seats. A
variant that scales gross, holds cash or skips a review because of the state
is the closed portfolio-timing family, not a variant of this arm.

Sizing. At CNY 1m over 50 seats a seat is around CNY 19,400, so one 100-share
lot of 95-96 % of the constituents fits and affordability stops being the price
factor it is on a small account -- which is exactly why an information block
should be priced here and not at 100k. A name's money is `book value / seats`,
in 100-share lots rounded to the NEAREST lot and then clipped to the cash on
hand; rounding down would put a systematic quarter of a seat back in cash.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited
before buys are sized; `context.account` never changes mid-call, so the buy
budget is decremented locally at the T-1 close.

Forced exits are unconditional and first: a holding that left the newest
visible section is sold -- the section IS the universe -- and so is one that
lost its score or its bar. Affordability is a BUY-side rule only: a holding
whose lot outgrew a seat is never force-sold, because selling a name for
rising is a reverse-momentum trade the account did not ask for.

The review is stateless: the calendar month (or ISO week) of the newest visible
trading day is compared with the decision day's, so a cold worker and a warm
one emit the same orders.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, index, knobs, primary, state

LOT = 100
SELL_HAIRCUT = 0.98


def run(context, candidate, values_of):
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
    values = values_of(context, panel)
    score = primary.score(context, panel, values, candidate)
    last = len(panel["dates"]) - 1
    keep_mask = data.tradable(panel, last) & np.isfinite(score)
    ranked = pd.DataFrame({
        "ts_code": [str(code) for code in panel["symbols"][keep_mask]],
        "rank_score": primary.pct_rank(score[keep_mask]),
        "close": panel["raw_close"][keep_mask, last],
        "industry": panel["industry"][keep_mask],
    })
    ranked = ranked[np.isfinite(ranked["close"].to_numpy()) & (ranked["close"] > 0)]
    ranked = ranked.sort_values(["rank_score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} scorable constituents for {knobs.SEATS} seats")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    members, section = index.latest(panel["sections"])
    weights = members.set_index("ts_code")["weight"]

    sells = _sell_orders(positions, rank, len(ranked), sell_at, candidate)
    sold = {order["symbol"] for order in sells}
    proceeds = sum(_price(prices, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in sold]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * knobs.CASH_BUFFER
    book_value = budget + sum(_price(prices, code) * positions[code] for code in keep)
    seat_cash = book_value / knobs.SEATS if knobs.SEATS else 0.0
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    book = keep + [code for code in affordable["ts_code"] if code not in positions][
        : max(0, knobs.SEATS - len(keep))]
    meta = {
        "candidate": candidate,
        "seats": knobs.SEATS,
        "book_size": len(book),
        "pool": int(len(ranked)),
        "horizon": knobs.HORIZON,
        "section": section,
        "refits": state.count(context),
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4) if seat_cash > 0 else 0.0,
        "block_coverage": _recent_coverage(panel, values),
        "book_index_weight_pct": round(float(weights.reindex(book).fillna(0.0).sum()), 3),
    }
    meta.update(primary.report(context))
    return sells + _buy_orders(book, keep, prices, budget, seat_cash, buy_at, book_value, candidate, meta)


def _recent_coverage(panel, values):
    """Mean block coverage of the last 20 bars, rounded; None when there is no block."""

    recent = data.block_coverage(panel, values)[-20:]
    finite = recent[np.isfinite(recent)]
    return round(float(finite.mean()), 4) if finite.size else None


def _due(context, decision_at):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    if knobs.REVIEW == "month":
        return (latest.year, latest.month) != (decision_at.year, decision_at.month)
    return latest.isocalendar()[:2] != decision_at.isocalendar()[:2]


def _sell_orders(positions, rank, pool_size, execute_at, candidate):
    forced = [code for code in positions if code not in rank]
    outside = [code for code in positions
               if code in rank and rank[code] >= knobs.SEATS * knobs.KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(positions))
    replaceable = max(0, min(knobs.MAX_SWAPS - len(forced), available))
    drop = forced + outside[:replaceable]
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drop)
    ]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(book, keep, prices, budget, seat_cash, execute_at, book_value, candidate, meta):
    orders = []
    remaining = budget
    for code in [name for name in book if name not in keep]:
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
