"""A SEATS-seat equal-cash book over the pool, reviewed on `knobs.REVIEW`, with a keep band and a swap cap.

Equal cash is the one weighting the host's zero-skill panel prices for free (it
redraws names on the book's own skeleton, not weights); any other weighting
owes a same-batch equal-cash leg.

Review: the first decision of each review period (`knobs.REVIEW`: ISO week,
half of the calendar month -- days 1-15 and 16-end -- or calendar month), and
any decision while the account is flat. The period of the newest visible
trading day is compared with the decision day's, so a cold worker and a warm
one emit the same orders. Between reviews nothing is traded: a seat a rejected
buy left empty waits for the next review.

Exits, sold at 09:30: every holding with a T-1 bar that is no longer scored in
the pool (left it, became ST); holdings ranked at or beyond SEATS * KEEP_BAND,
worst first, at most round(SEATS * SWAP_SHARE) minus the forced exits; with
INDUSTRY_CAP set, the worst-ranked kept names of an industry above
floor(INDUSTRY_CAP * SEATS). A holding without a T-1 bar is suspended: it
cannot be sold, so it is kept, keeps its seat and adds no proceeds; it is
reviewed again once it trades. Entries, bought at 15:00: walking
the ranking from the top, names not held whose 100-share lot at the T-1 close
fits a seat (and whose industry is under the cap), until SEATS names.
Affordability is a buy-side rule only: a holding whose lot outgrew a seat is
never sold for it.

Sizing: a seat is book value / SEATS; each buy is a seat rounded to the
NEAREST lot at the T-1 close, then clipped to the cash left. Sells at 09:30 and
buys at 15:00 of the same day, so proceeds are credited before the buys;
`context.account` never changes within a call, so the buy budget is
decremented locally.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import census_features as cf, knobs, learn

LOT = 100
SELL_HAIRCUT = 0.98


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
    cash = float(context.account.cash)
    if not positions and not knobs.CAPITAL / 2 <= cash <= knobs.CAPITAL * 2:
        raise RuntimeError(f"a flat account holds {cash:.0f} in cash but this shape is registered for "
                           f"knobs.CAPITAL = {knobs.CAPITAL}: align it to broker_replay.initial_cash")
    if positions and not _due(context, decision):
        return []

    panel = cf.read_panel(context, cf.DECISION_DAYS)
    scores = learn.score(context, panel, candidate)
    last = len(panel["dates"]) - 1
    closes = pd.Series(pd.DataFrame(panel["d"]["close"]).ffill().to_numpy()[last], index=panel["codes"])
    scored = np.isfinite(scores)
    ranked = pd.DataFrame({
        "ts_code": panel["codes"][scored],
        "score": scores[scored],
        "close": closes.to_numpy()[scored],
        "industry": np.where(panel["l1_name"][scored] == "", "未分类", panel["l1_name"][scored]),
    }).sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    if len(ranked) < 2 * knobs.SEATS:
        raise RuntimeError(f"only {len(ranked)} scored names for a {knobs.SEATS}-seat book")
    rank = dict(zip(ranked["ts_code"], ranked.index))
    industry = dict(zip(panel["codes"], np.where(panel["l1_name"] == "", "未分类", panel["l1_name"])))
    cap = int(np.floor(knobs.INDUSTRY_CAP * knobs.SEATS)) if knobs.INDUSTRY_CAP else knobs.SEATS

    traded = set(panel["codes"][panel["has_bar"][last]])
    forced = sorted(code for code in positions if code not in rank and code in traded)
    outside = sorted((code for code in positions if code in rank and rank[code] >= knobs.SEATS * knobs.KEEP_BAND),
                     key=lambda code: (-rank[code], code))
    band = outside[:max(0, round(knobs.SEATS * knobs.SWAP_SHARE) - len(forced))]
    keep = [code for code in positions if code not in forced and code not in band]
    excess = _over_cap([code for code in keep if code in traded], rank, industry, cap)
    keep = [code for code in keep if code not in excess]
    drops = forced + band + excess

    def price(code):
        value = float(closes.get(code, float("nan")))
        return value if np.isfinite(value) and value > 0 else 0.0

    proceeds = sum(price(code) * positions[code] for code in drops)
    budget = (cash + proceeds * SELL_HAIRCUT) * knobs.CASH_BUFFER
    book_value = budget + sum(price(code) * positions[code] for code in keep)
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
    meta = {
        "candidate": candidate,
        "review": knobs.REVIEW,
        "seats": knobs.SEATS,
        "pool": int(len(ranked)),
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4),
        "book_size": len(keep) + len(entries),
        "exits_forced": len(forced),
        "exits_band": len(band),
        "exits_cap": len(excess),
        "top_industry_names": max(counts.values()) if counts else 0,
    }
    return _sell_orders(positions, drops, sell_at, candidate) + _buy_orders(
        entries, price, budget, seat_cash, book_value, buy_at, meta)


def _period(day):
    if knobs.REVIEW == "weekly":
        return tuple(day.isocalendar()[:2])
    if knobs.REVIEW == "biweekly":
        return day.year, day.month, day.day > 15
    if knobs.REVIEW == "monthly":
        return day.year, day.month
    raise ValueError(f"unknown REVIEW {knobs.REVIEW!r}")


def _due(context, decision):
    start = (decision - timedelta(days=20)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    return _period(pd.Timestamp(str(days["trade_date"].max()))) != _period(decision)


def _over_cap(keep, rank, industry, cap):
    groups = {}
    for code in keep:
        groups.setdefault(industry.get(code, "未分类"), []).append(code)
    excess = []
    for members in groups.values():
        if len(members) > cap:
            excess += sorted(members, key=lambda code: (-rank.get(code, len(rank)), code))[:len(members) - cap]
    return excess


def _sell_orders(positions, drops, execute_at, candidate):
    return [{"symbol": code, "action": "sell", "quantity": int(positions[code]),
             "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"} for code in sorted(drops)]


def _buy_orders(entries, price, budget, seat_cash, book_value, execute_at, meta):
    orders = []
    remaining = budget
    for code in entries:
        value = price(code)
        if value <= 0:
            continue
        quantity = int(min(remaining, seat_cash) / value / LOT + 0.5) * LOT
        while quantity > 0 and quantity * value > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({"symbol": code, "action": "buy", "quantity": quantity,
                       "execute_at": execute_at.isoformat(), "reason": meta["candidate"] + "_entry",
                       "target_weight": round(1.0 / knobs.SEATS, 6),
                       "realized_weight": round(quantity * value / book_value, 6) if book_value > 0 else 0.0,
                       **meta})
        remaining -= quantity * value
    return orders
