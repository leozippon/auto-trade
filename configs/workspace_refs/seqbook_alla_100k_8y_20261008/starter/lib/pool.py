"""The pool lane's restrictions on what the book may buy, point in time.

Three registered filters (`lib/knobs.py`), each judged on a review day against
the names tradable on the newest visible row (T-1) and intersected:

    LIQ_DROP    not among the LIQ_DROP share of tradable names with the lowest
                median daily turnover value (`amount`, CNY) over the last
                LIQ_DAYS rows, counting only days with a bar
    MV_DROP     not among the MV_DROP share with the lowest circulating market
                value (`circ_mv`) on T-1
    INDEX_POOL  a constituent of the newest visible section of 000852.SH
                ("csi1000"), or of 000852.SH or 000905.SH ("csi1500")

They gate entries only: a holding outside the restricted pool is ranked and
kept like any other and leaves by rank, the rule the one-lot filter already
follows. Nothing here reads a row after T-1, and index sections are taken only
when their `available_at` precedes the decision (`lib/index.py`). The heads
train and score on the whole pool regardless: these filters belong to the book.
"""

import numpy as np
import pandas as pd

from lib import index, knobs, panel as P

LIQ_DAYS = 20
INDEX_CODES = {"csi1000": ("000852.SH",), "csi1500": ("000852.SH", "000905.SH")}


def median_amount(data, t, days=LIQ_DAYS):
    """(names,) median turnover value over the last `days` rows up to date index t; NaN without a bar in them."""

    window = data["amount"][max(0, t - days + 1): t + 1]
    return pd.DataFrame(window).median(axis=0, skipna=True).to_numpy()


def floor(values, candidates, drop):
    """`candidates` without the `drop` share of them that have the lowest `values` (missing counts as lowest)."""

    out = candidates.copy()
    pick = np.nonzero(candidates)[0]
    cut = int(round(drop * len(pick)))
    if cut:
        lowest = np.argsort(np.where(np.isfinite(values[pick]), values[pick], -np.inf), kind="stable")[:cut]
        out[pick[lowest]] = False
    return out


def members(context, codes):
    """(names,) bool: constituents of the newest visible section of every index of `knobs.INDEX_POOL`."""

    names = set()
    for code in INDEX_CODES[knobs.INDEX_POOL]:
        names.update(index.latest(context, code)[0])
    return np.isin(codes, sorted(names))


def eligible(context, data, t):
    """(names,) bool: the names the book may buy at date index t, before the one-lot filter."""

    tradable = P.tradable(data, t)
    out = tradable.copy()
    if knobs.LIQ_DROP:
        out &= floor(median_amount(data, t), tradable, knobs.LIQ_DROP)
    if knobs.MV_DROP:
        out &= floor(data["circ_mv"][t], tradable, knobs.MV_DROP)
    if knobs.INDEX_POOL != "all":
        out &= members(context, data["codes"])
    return out
