"""The one place each leg is defined: what score a candidate hands the book.

    "t1"      the candidate: abnormal turnover (`lib/score.py`)
    "c_shuf"  a control: t1's own values, permuted among the same pool names
              with a generator seeded by the decision day, so the book has t1's
              score distribution and no information. Deterministic: a cold and
              a warm worker emit the same orders
    "t1i"     the registered variant: t1's mid-rank percentile within each SW
              L1 industry of the pool, so the book picks names, not industries

Every leg returns NaN off the pool (`data.tradable`) and trades the same book
(`lib/trade.py`); only the score differs.
"""

import numpy as np
import pandas as pd

from lib import data, score

LEGS = ("t1", "c_shuf", "t1i")


def score_of(context, panel, candidate):
    if candidate not in LEGS:
        raise ValueError(f"unknown candidate: {candidate}")
    raw = np.where(data.tradable(panel), score.abnormal_turnover(panel), np.nan)
    if candidate == "t1":
        return raw
    scored = np.isfinite(raw)
    if candidate == "t1i":
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
