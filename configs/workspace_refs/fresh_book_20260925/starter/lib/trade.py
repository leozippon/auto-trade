"""A SEATS-seat long-only book inside the benchmark, built the way the leg says.

What the legs share, and why it is fixed. The score, the pool, SEATS equal-cash
seats, lot sizing and the 09:30 sell / 15:00 buy timing are the carrier's; only
the review cadence, the swap cap, the keep band and the two conditional
mechanisms differ (`knobs.LEGS`, applied by `lib/book.py`). Equal cash is the
one weighting scheme the host's zero-skill panel prices for free: the panel
replaces each round trip's name among that entry day's names on the original's
side of constituent membership that this money can buy, so it aligns the pool,
the calendar and the money per seat -- but not a weighting scheme, which is why
`f4` is only ever read against `f1` in the same batch.

Why the construction is the lever. The monthly book with 8 swaps and a keep
band of 2.0 hits its swap cap at almost every review, holds a name for months
against a 10-day label, and keeps a slow sleeve of large, low-volatility,
low-beta names that the score once ranked highly (`sources.md`). `f1` removes
the cap, narrows the band and reviews on the cadence the score's own IC decay
supports; whether a fresher book earns more of the score's rank IC is what the
arm measures.

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
rising is a reverse-momentum trade the account did not ask for.

The review is stateless: the calendar month (or k-week bucket) of the newest
visible trading day is compared with the decision day's, so a cold worker and a
warm one emit the same orders. Between reviews a `refill` leg (f1 and its
variants) looks again whenever it holds fewer than SEATS names -- an entry
rejected at the limit or for a missing price, or cut short by cash -- and buys
the best-ranked affordable names it does not hold into the empty seats, sells
nothing and pulls nothing back; `c_base` keeps the template's behaviour and
leaves such a seat empty until its next review. Week buckets count Monday-to-Sunday weeks from
the proleptic calendar's first Monday, so a two-week bucket does not reset at
a year end.

Ties are broken by a fixed hash of the code: stable across decisions, and
unrelated to board or exchange. The carrier's score has no ties.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import book, data, index, knobs, primary

LOT = 100
SELL_HAIRCUT = 0.98


def run(context, candidate):
    leg = knobs.LEGS[candidate]
    decision_at = pd.Timestamp(context.inference_at)
    sell_at = decision_at.replace(hour=9, minute=30, second=0, microsecond=0)
    buy_at = decision_at.replace(hour=15, minute=0, second=0, microsecond=0)
    if decision_at > buy_at:
        return []
    if decision_at > sell_at:
        sell_at = buy_at
    positions = {str(code): int(quantity)
                 for code, quantity in dict(context.account.positions).items() if int(quantity) > 0}
    review = not positions or _due(context, decision_at, leg["review"])
    if not review and not (leg["refill"] and len(positions) < knobs.SEATS):
        return []

    panel = data.wide(context, data.DECISION_LOOKBACK_DAYS)
    score = primary.score(context, panel)
    last = len(panel["dates"]) - 1
    keep_mask = data.tradable(panel, last) & np.isfinite(score)
    ranked = pd.DataFrame({
        "ts_code": [str(code) for code in panel["symbols"][keep_mask]],
        "rank_score": primary.pct_rank(score[keep_mask]),
        "close": panel["raw_close"][keep_mask, last],
        "industry": panel["industry"][keep_mask],
        "row": np.flatnonzero(keep_mask),
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

    drops = book.exits(positions, rank, len(ranked), leg) if review else []
    keep, budget, book_value, seat_cash = _money(context, positions, drops, prices)
    affordable = ranked[ranked["close"] * LOT <= seat_cash] if seat_cash > 0 else ranked
    target = keep + book.entrants(keep, positions, list(affordable["ts_code"]), leg)
    meta = {}
    if leg["pull"] and review:
        z = dict(zip(codes, book.standardise(book.exposures(panel, ranked["row"].to_numpy()))))
        industry = dict(zip(codes, ranked["industry"]))
        share = ranked["industry"].value_counts(normalize=True).to_dict()
        reserve = [code for code in affordable["ts_code"] if code not in positions and code not in target]
        target = book.pull_back(target, reserve, rank, z, industry, share)
        drops = [code for code in positions if code not in target]
        keep, budget, book_value, seat_cash = _money(context, positions, drops, prices)
        meta["pull_deviation"] = book.deviation(target, z, industry, share)
    buys = [code for code in target if code not in keep]
    cash = {code: seat_cash for code in buys}
    if leg["weights"] == "epo" and buys:
        rows = ranked.set_index("ts_code").loc[target, "row"].to_numpy()
        closes = panel["C"][rows, -(knobs.EPO_DAYS + 1):]
        with np.errstate(invalid="ignore", divide="ignore"):
            returns = (closes[:, 1:] / closes[:, :-1] - 1.0).T
        signal = np.array([ranked["rank_score"].iloc[rank[code]] - 0.5 for code in target])
        epo = dict(zip(target, book.epo(returns, np.clip(signal, 1e-6, None))))
        total = sum(epo[code] for code in buys)
        cash = {code: min(seat_cash * len(buys) * epo[code] / total, knobs.EPO_CAP * seat_cash)
                for code in buys}
    meta.update({
        "candidate": candidate,
        "refill": not review,
        "review": leg["review"],
        "max_swaps": leg["max_swaps"],
        "keep_band": leg["keep_band"],
        "horizon": leg["horizon"],
        "seats": knobs.SEATS,
        "book_size": len(target),
        "exits": len(drops),
        "pool": int(len(ranked)),
        "section": section,
        "seat_cash": round(seat_cash, 2),
        "unbuyable_share": round(float((ranked["close"] * LOT > seat_cash).mean()), 4) if seat_cash > 0 else 0.0,
        "book_index_weight_pct": round(float(weights.reindex(target).fillna(0.0).sum()), 3),
        "book_tc": _reading(book.transfer(target, codes, ranked["rank_score"])),
    })
    meta.update(primary.report(context))
    return _sell_orders(positions, drops, sell_at, candidate) + _buy_orders(
        buys, cash, prices, budget, buy_at, book_value, candidate, meta)


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


def _due(context, decision_at, review):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    if review == "month":
        return (latest.year, latest.month) != (decision_at.year, decision_at.month)
    return _bucket(latest, review) != _bucket(decision_at, review)


def _bucket(day, weeks):
    """Index of the `weeks`-week bucket holding `day`; date(1, 1, 1) is a Monday."""

    return (day.date().toordinal() - 1) // 7 // weeks


def _sell_orders(positions, drops, execute_at, candidate):
    return [
        {"symbol": code, "action": "sell", "quantity": int(positions[code]),
         "execute_at": execute_at.isoformat(), "reason": candidate + "_exit"}
        for code in sorted(drops)
    ]


def _price(prices, code):
    price = float(prices.get(code, float("nan")))
    return price if np.isfinite(price) and price > 0 else 0.0


def _buy_orders(buys, cash, prices, budget, execute_at, book_value, candidate, meta):
    orders = []
    remaining = budget
    for code in buys:
        price = _price(prices, code)
        if price <= 0 or cash[code] <= 0:
            continue
        quantity = int(min(remaining, cash[code]) / price / LOT + 0.5) * LOT
        while quantity > 0 and quantity * price > remaining:
            quantity -= LOT
        if quantity <= 0:
            continue
        orders.append({
            "symbol": code, "action": "buy", "quantity": quantity,
            "execute_at": execute_at.isoformat(), "reason": candidate + "_entry",
            "target_weight": round(cash[code] / book_value, 6) if book_value > 0 else 0.0,
            "realized_weight": round(quantity * price / book_value, 6) if book_value > 0 else 0.0,
            **meta,
        })
        remaining -= quantity * price
    return orders
