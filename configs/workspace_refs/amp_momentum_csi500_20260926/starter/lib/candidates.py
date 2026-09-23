"""The one place each leg is defined: what score a candidate hands the book.

    "a1"      the candidate: the amplitude split with `knobs.LOW_SHARE`
    "c_shuf"  a control: a1's own values, permuted among the same pool names
              with a generator seeded by the decision day, so the book has a1's
              score distribution and no information. Deterministic: a cold and
              a warm worker emit the same orders
    "c_mom"   a control: the same mean over every valid session of the window
              (share 1), plain momentum -- what the split is claimed to beat
    "a1i"     the registered variant: a1's mid-rank percentile within each SW
              L1 industry of the pool, so the book picks names, not industries

Every leg returns NaN off the pool (`data.tradable`) and trades the same book
(`lib/trade.py`); only the score differs.
"""

import numpy as np
import pandas as pd

from lib import data, knobs, score

LEGS = ("a1", "c_shuf", "c_mom", "a1i")


def score_of(context, panel, candidate):
    if candidate not in LEGS:
        raise ValueError(f"unknown candidate: {candidate}")
    share = 1.0 if candidate == "c_mom" else knobs.LOW_SHARE
    raw = np.where(data.tradable(panel), score.amplitude_split(panel, share), np.nan)
    if candidate in ("a1", "c_mom"):
        return raw
    scored = np.isfinite(raw)
    if candidate == "a1i":
        frame = pd.DataFrame({"s": raw[scored], "industry": panel["industry"][scored]})
        grouped = frame.groupby("industry")["s"]
        out = np.full(raw.shape, np.nan)
        out[scored] = ((grouped.rank(method="average") - 0.5) / grouped.transform("size")).to_numpy()
        return out
    day = int(pd.Timestamp(context.inference_at).strftime("%Y%m%d"))
    positions = np.flatnonzero(scored)
    shuffled = raw.copy()
    shuffled[positions] = raw[positions][np.random.default_rng(day).permutation(positions.size)]
    return shuffled
