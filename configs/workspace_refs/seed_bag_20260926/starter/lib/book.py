"""How a ranked pool becomes the monthly book -- the head packs' `c_base` rules, unchanged.

Pure functions over codes and ranks: nothing here reads a file or touches
`context`, so an offline census can walk exactly these rules over a panel of
scores and `lib/trade.py` turns their answer into orders.

    exits       forced exits (left the section, lost the score or the bar)
                first, then holdings ranked outside SEATS * KEEP_BAND, worst
                first, at most MAX_SWAPS in all
    entrants    the best-ranked names this money can buy that are not held,
                until the seats are full

`transfer` is a reading, not a rule: the correlation between the book's
equal-cash active weights over the pool and the rank-normal score -- about
0.68 for a freshly bought top 50 of about 283 names.
"""

import numpy as np
from scipy.stats import norm

from lib import knobs


def exits(held, rank, pool_size):
    """Holdings to sell at this review, forced exits first."""

    forced = [code for code in held if code not in rank]
    outside = [code for code in held
               if code in rank and rank[code] >= knobs.SEATS * knobs.KEEP_BAND]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(held))
    room = min(knobs.MAX_SWAPS - len(forced), available)
    return forced + outside[:max(0, room)]


def entrants(keep, held, affordable):
    """Best-ranked affordable names not already held, until SEATS are filled.

    `affordable` is in rank order. A name sold at this review is still held
    when the review is decided, so it is never bought back the same day.
    """

    return [code for code in affordable if code not in held][:max(0, knobs.SEATS - len(keep))]


def transfer(book, codes, pct):
    """Correlation of the equal-cash book's active weights with the rank-normal score over the pool."""

    held = np.isin(np.asarray(codes), list(book)).astype(np.float64)
    if held.all() or not held.any():
        return float("nan")
    return float(np.corrcoef(held, norm.ppf(np.asarray(pct, dtype=np.float64)))[0, 1])
