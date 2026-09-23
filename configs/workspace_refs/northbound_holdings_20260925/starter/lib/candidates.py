"""The one place each leg is defined: what score a candidate hands the book.

    "n1"      the candidate. Mean of the percentile ranks of the northbound
              level and its change (`lib/holdings.py`) over the decision bar's
              tradable constituents, `knobs.LEVEL_WEIGHT` on the level; a
              missing component takes the neutral 0.5. Not learned: no fit
    "c_shuf"  a control. n1's own values, permuted among the same tradable
              names with a generator seeded by the decision day, so the book
              has n1's score distribution and no information. Deterministic:
              a cold and a warm worker emit the same orders
    "c_base"  a control for the register only: the Alpha158 + LightGBM carrier
              (`lib/primary.py`), traded as is. The arm's verdict never depends
              on beating it
    "n2"      Family 2, allowed only after n1 passed the offline screen: mean
              of the percentile ranks of the carrier score and of the holdings
              change, `knobs.N2_CARRIER_WEIGHT` on the carrier; a missing change
              takes 0.5. A rank average, not a feature block

Every leg trades the same book (`lib/trade.py`); only the score differs.
"""

import numpy as np
import pandas as pd

from lib import data, holdings, knobs, primary

CARRIER = ("c_base", "n2")
LEGS = ("n1", "c_shuf", "c_base", "n2")


def needs_fit(candidate):
    if candidate not in LEGS:
        raise ValueError(f"unknown candidate: {candidate}")
    return candidate in CARRIER


def score(context, panel, candidate):
    """((names,) score aligned to panel["symbols"], NaN off the tradable pool; order metadata)."""

    if candidate not in LEGS:
        raise ValueError(f"unknown candidate: {candidate}")
    keep = data.tradable(panel, len(panel["dates"]) - 1)
    lvl, chg, age = holdings.components(context, panel, keep)
    meta = holdings.report(keep, lvl, chg, age)
    if candidate == "c_base":
        return primary.score(context, panel), meta
    if candidate == "n2":
        carrier = holdings.neutral_rank(primary.score(context, panel), keep)
        change = holdings.neutral_rank(chg, keep)
        return knobs.N2_CARRIER_WEIGHT * carrier + (1.0 - knobs.N2_CARRIER_WEIGHT) * change, meta
    n1 = (knobs.LEVEL_WEIGHT * holdings.neutral_rank(lvl, keep)
          + (1.0 - knobs.LEVEL_WEIGHT) * holdings.neutral_rank(chg, keep))
    if candidate == "n1":
        return n1, meta
    day = int(pd.Timestamp(context.inference_at).strftime("%Y%m%d"))
    shuffled = n1.copy()
    positions = np.nonzero(keep)[0]
    shuffled[positions] = n1[positions][np.random.default_rng(day).permutation(positions.size)]
    return shuffled, meta
