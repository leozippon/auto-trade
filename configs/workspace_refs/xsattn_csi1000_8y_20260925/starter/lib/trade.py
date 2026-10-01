"""SEATS equal-cash constituents, reviewed on REVIEW, with a keep band and a swap cap.

The shape is deliberately plain, because the arm is about the score and not
about the construction. Equal cash is the one weighting scheme the host's
zero-skill panel already prices for free: the panel replaces each round trip's
name among that entry day's names on the original's side of CSI 300
membership that this money can buy, so it aligns the pool, the calendar and
the money per seat -- but it does not align a weighting scheme. Anything other
than equal cash would need its own same-batch, same-score equal-cash control
before its reading means anything.

Review is stateless: the newest visible trading day's ISO week (or calendar
month) is compared with the decision day's, so a cold worker and a warm one
produce the same orders. A non-review decision reads only a few days of trade
dates. Membership itself only moves at a month-end section, so a weekly review
re-ranks the same names more often -- it does not refresh the index.

Forced exits, keep band and replacement. A holding that is no longer a
rankable constituent is always sold. A holding still ranked inside
SEATS * KEEP_BAND is kept; one outside the band is sold, at most MAX_SWAPS
names changing a review with forced exits counted first.

Sizing. At CNY 1m over SEATS seats a seat is around CNY 19k, so one 100-share
lot of essentially every constituent fits and affordability is not the binding
constraint it is on a small account -- which is exactly why this arm is the
one to put a learner on. A name's money is `book value / seats`, in 100-share
lots rounded to the NEAREST lot and then clipped to the cash on hand.

Sells are timed 09:30 and buys 15:00 of the same day, so proceeds are credited
before buys are sized; `context.account` never changes mid-call, so the buy
budget is decremented locally and sized at the T-1 close.

Tradability is applied HERE and only on the decision row: `universe` is one
decision-day vintage, so pushing today's ST labels back across a training
window would silently drop the names that became ST later.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import control, knobs, model, panel

LOT = 100
SELL_HAIRCUT = 0.98
EXCLUDED_PREFIX = ("688", "689")
EXCLUDED_SUFFIX = ".BJ"


def run(context, candidate):
    decision = pd.Timestamp(context.inference_at)
    sell_at = decision.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision > buy_at:
        return []
    if decision > sell_at:
        sell_at = buy_at
    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    if positions and not _due(context, decision):
        return []

    data = panel.review_window(context)
    scores = _score(context, data, candidate)
    ranked = _ranked(data, scores)
    if len(ranked) < knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} rankable constituents for {knobs.SEATS} seats")
    rank = {code: position for position, code in enumerate(ranked["ts_code"])}
    prices = ranked.set_index("ts_code")["close"]
    last_close = pd.Series(data["close"][-1], index=data["codes"])

    sells = _sell_orders(positions, rank, len(ranked), sell_at, candidate)
    sold = {order["symbol"] for order in sells}
    proceeds = sum(_price(last_close, order["symbol"]) * order["quantity"] for order in sells)
    keep = [code for code in positions if code not in sold]
    budget = (float(context.account.cash) + proceeds * SELL_HAIRCUT) * knobs.CASH_BUFFER
    book_value = budget + sum(_price(last_close, code) * positions[code] for code in keep)
    book = keep + [code for code in ranked["ts_code"] if code not in positions][
        : max(0, knobs.SEATS - len(keep))]
    meta = {"candidate": candidate, "seats": knobs.SEATS, "book_size": len(book),
            "pool": int(len(ranked)), "horizon": knobs.HORIZON}
    meta.update(_report(context, candidate))
    return sells + _buy_orders(book, keep, prices, budget, book_value, buy_at, candidate, meta)


def _score(context, data, candidate):
    if candidate == "x1":
        return model.score(context, data, True)
    if candidate == "c_temporal":
        return model.score(context, data, False)
    if candidate == "c_lgbm":
        return control.score(context, data)
    raise ValueError(f"unknown candidate: {candidate}")


def _report(context, candidate):
    return control.report(context) if candidate == "c_lgbm" else model.report(context)


def _tradable(data):
    codes = pd.Series(data["codes"])
    names = pd.Series(data["names"])
    return (
        ~codes.str.startswith(EXCLUDED_PREFIX).to_numpy()
        & ~codes.str.endswith(EXCLUDED_SUFFIX).to_numpy()
        & ~names.str.contains("ST|退").to_numpy()
    )


def _ranked(data, scores):
    last = len(data["dates"]) - 1
    pool = data["member"][last] & np.isfinite(scores) & _tradable(data)
    frame = pd.DataFrame({"ts_code": data["codes"][pool], "score": scores[pool],
                          "close": data["close"][last][pool]})
    frame = frame[np.isfinite(frame["close"].to_numpy()) & (frame["close"] > 0)]
    return frame.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)


def _due(context, decision):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    if knobs.REVIEW == "month":
        return (latest.year, latest.month) != (decision.year, decision.month)
    return latest.isocalendar()[:2] != decision.isocalendar()[:2]


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


def _buy_orders(book, keep, prices, budget, book_value, execute_at, candidate, meta):
    orders = []
    remaining = budget
    seat_cash = book_value / knobs.SEATS if knobs.SEATS else 0.0
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
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
