"""The cycle lane's score: industries ranked by their reported cycle, the first names of each.

What the canonical book (`lib/trade.py`) relies on:

    score(context, codes) -> (values, meta)
        values: float Series indexed by ts_code over `codes` (the day's
        tradable names); higher is bought first, NaN is not scored.
    shuffle(values, decision_at) -> the control c_shuf
    describe(values, book) -> extra metadata about the chosen book
    NAME -> the leg label written on the orders

The industries are ranked by `lib/industry.py` under `knobs.SIGNAL`. Inside
each ranked industry the first NAMES_PER_INDUSTRY names that a seat can buy
(or that are already held) are scored, by `knobs.WITHIN`: largest float cap
first ("cap"), or largest own revenue acceleration first, cap breaking ties
and names without a reading last ("accel"). The value of the p-th name of the
r-th industry is -(r x NAMES_PER_INDUSTRY + p), so the book's ranking walks
the industries in order and the canonical industry cap (`knobs.INDUSTRY_CAP`,
four) fills SEATS / 4 industries. Every other name is not scored: a holding
that left its industry's first names, or whose industry lost its reading,
is a forced exit at the review.

Seat cash for the affordability test is the canonical book's estimate before
the day's sells: knobs.CAPITAL x (cash + holdings at the T-1 close) / SEATS.

c_shuf keeps every industry block (its names and their order) and redraws the
order of the blocks from a generator seeded by the decision's calendar
quarter: same names per industry, random industries. The draw is a fixed
permutation of rank positions for the quarter, so the random book is not
re-drawn at every review: it holds the industries at a few random positions of
the ranking, which change when the ranking moves there.
"""

import numpy as np
import pandas as pd

from lib import data, industry, knobs

NAME = knobs.leg()
NAMES_PER_INDUSTRY = 6
LOT = 100


def score(context, codes):
    codes = list(codes)
    label = industry.labels(context, codes)
    needs_firms = knobs.SIGNAL != "px" or knobs.WITHIN == "accel"
    firm, audit = industry.firm_readings(context, codes) if needs_firms else (None, {})
    order = industry.ranking(context, codes, knobs.SIGNAL, label, firm)
    last = data.panel(context, ["close", "circ_mv"], 1)
    close = last["close"].iloc[-1].reindex(codes)
    cap = last["circ_mv"].iloc[-1].reindex(codes)
    positions = {str(code): int(quantity) for code, quantity in dict(context.account.positions).items()
                 if int(quantity) > 0}
    held_value = sum(float(close.get(code, np.nan)) * quantity for code, quantity in positions.items()
                     if np.isfinite(float(close.get(code, np.nan))))
    seat_cash = knobs.CAPITAL * (float(context.account.cash) + held_value) / knobs.SEATS
    eligible = (close * LOT <= seat_cash) | pd.Series(close.index.isin(list(positions)), index=close.index)
    key = pd.DataFrame({"cap": cap, "accel": firm["sales_accel"] if firm is not None else np.nan, "label": label})
    values = pd.Series(np.nan, index=pd.Index(codes, dtype="object"), dtype="float64")
    for rank, name in enumerate(order):
        members = key[(key["label"] == name) & eligible.reindex(key.index, fill_value=False)]
        if knobs.WITHIN == "cap":
            members = members.sort_values(["cap"], ascending=False, kind="stable")
        else:
            members = members.sort_values(["accel", "cap"], ascending=[False, False], na_position="last", kind="stable")
        for position, code in enumerate(members.index[:NAMES_PER_INDUSTRY]):
            values[code] = -float(rank * NAMES_PER_INDUSTRY + position)
    meta = {
        "signal": knobs.SIGNAL,
        "within": knobs.WITHIN,
        "industries_ranked": len(order),
        "first_industries": order[: knobs.SEATS // knobs.INDUSTRY_CAP + 1],
        **audit,
    }
    return values, meta


def shuffle(values, decision_at):
    scored = values.dropna()
    block = np.floor(-scored / NAMES_PER_INDUSTRY)
    position = -scored - block * NAMES_PER_INDUSTRY
    blocks = np.sort(block.unique())
    when = pd.Timestamp(decision_at)
    quarter = when.year * 10 + (when.month - 1) // 3 + 1
    drawn = dict(zip(blocks, blocks[np.random.default_rng(quarter).permutation(len(blocks))]))
    return -(block.map(drawn) * NAMES_PER_INDUSTRY + position)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_industry_ranks": sorted({int(v) for v in np.floor(-picked / NAMES_PER_INDUSTRY)})}
