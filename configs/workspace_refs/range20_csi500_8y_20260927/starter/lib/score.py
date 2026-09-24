"""The one place the score is defined: r1 (and r2), or its shuffle c_shuf.

    r1 / r2   score = -range: minus the mean of (high - low) / pre_close over
              the name's own last knobs.RANGE_DAYS bars (20 for r1, 60 for r2).
              The calmest names rank first. No learner, no fitted parameter.
    c_shuf    (knobs.SHUFFLE) r1's own values permuted among the same pool
              with a generator seeded by the decision day: the same score
              distribution and the same book, with the information removed.
              Deterministic, so a cold and a warm worker emit the same orders.

Ties are broken by a fixed hash of the code: stable across decisions and
unrelated to board or exchange (sorting by the code itself would put SZ main
board and ChiNext names first). The range is continuous, so ties are rare.
"""

import numpy as np
import pandas as pd

from lib import knobs


def label():
    return "c_shuf" if knobs.SHUFFLE else f"range{knobs.RANGE_DAYS}"


def ranked(context, pool):
    """The pool sorted best first, with its score and a 0-based `rank` column."""

    frame = pool.sort_values("ts_code").reset_index(drop=True)
    score = -frame["range"].to_numpy(dtype=np.float64)
    if knobs.SHUFFLE:
        day = int(pd.Timestamp(context.inference_at).strftime("%Y%m%d"))
        score = score[np.random.default_rng(day).permutation(score.size)]
    frame["score"] = score
    frame["tie"] = pd.util.hash_pandas_object(frame["ts_code"], index=False).to_numpy()
    frame = frame.sort_values(["score", "tie"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(len(frame))
    return frame
