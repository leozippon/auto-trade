"""The one place the score is defined: `knobs.SCORE`, or its shuffle c_shuf.

Every score is built from per-member inputs where higher is better; NaN means
the member cannot be scored and leaves the pool (nothing is imputed), -inf
ranks last but stays in the pool.

    ep       1 / pe (`knobs.EP_COLUMN`, T-1) where pe > 0. The vendor leaves pe
             empty for a trailing loss: such a member ranks last (-inf). A
             member whose T-1 row lacks daily_basic cannot be scored.
    dvy      dv, the first positive value among `knobs.DV_COLUMNS` (T-1): the
             trailing-twelve-month cash yield, or the last calendar year's when
             the trailing window has lapsed between two annual payments. No
             positive value (no cash dividend): last (-inf). A member whose T-1
             row lacks daily_basic cannot be scored.
    abturn   -turn_ratio: minus the mean turnover over the last TURN_SHORT
             sessions over the mean over the last TURN_LONG (Liu, Stambaugh
             and Yuan's one-month abnormal turnover: low ranks first).
             Undefined (too few valid bars) cannot be scored.
    ch3      the two inputs ep and abturn; the score is the mean of the
             member's two percentile ranks inside the pool, equal weights,
             ties averaged (every trailing loss shares the bottom ep rank).
             A member needs both inputs scorable.
    range20  -range (the other eight-year arms' score): a labelled
             diagnostic only, never a leg of this arm.
    c_shuf   (knobs.SHUFFLE) the chosen score's own values permuted among the
             same pool with a generator seeded by the decision day: the same
             score distribution and the same book, with the information
             removed. Deterministic, so a cold and a warm worker emit the same
             orders.

Ties are broken by a fixed hash of the code: stable across decisions and
unrelated to board or exchange.
"""

import numpy as np
import pandas as pd

from lib import knobs


def label():
    if knobs.SHUFFLE:
        return "c_shuf"
    return {
        "ep": f"ep_{knobs.EP_COLUMN}",
        "dvy": "dvy_" + "_".join(knobs.DV_COLUMNS),
        "abturn": f"abturn{knobs.TURN_SHORT}_{knobs.TURN_LONG}",
        "ch3": f"ch3_{knobs.EP_COLUMN}_{knobs.TURN_SHORT}_{knobs.TURN_LONG}",
        "range20": f"range{knobs.RANGE_DAYS}_diagnostic",
    }[knobs.SCORE]


def ranked(context, pool):
    """(the scorable pool sorted best first with `score` and a 0-based `rank`, the score coverage).

    Coverage is the share of the tradable members whose score inputs are all
    non-null (neither NaN nor the -inf of a loss or of no dividend).
    """

    frame = pool.sort_values("ts_code").reset_index(drop=True)
    inputs = _inputs(frame)
    coverage = float(np.logical_and.reduce([np.isfinite(x) for x in inputs]).mean()) if len(frame) else 0.0
    scorable = np.logical_and.reduce([~np.isnan(x) for x in inputs])
    frame = frame[scorable].reset_index(drop=True)
    inputs = [x[scorable] for x in inputs]
    score = inputs[0] if len(inputs) == 1 else np.mean([_pct_rank(x) for x in inputs], axis=0)
    if knobs.SHUFFLE:
        day = int(pd.Timestamp(context.inference_at).strftime("%Y%m%d"))
        score = score[np.random.default_rng(day).permutation(score.size)]
    frame["score"] = score
    frame["tie"] = pd.util.hash_pandas_object(frame["ts_code"], index=False).to_numpy()
    frame = frame.sort_values(["score", "tie"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(len(frame))
    return frame, coverage


def _inputs(frame):
    """The score's per-member inputs (lists of arrays aligned with `frame`)."""

    has_basic = frame["has_basic"].to_numpy()
    turn = -frame["turn_ratio"].to_numpy(dtype=np.float64)
    if knobs.SCORE == "ep":
        return [_ep(frame, has_basic)]
    if knobs.SCORE == "dvy":
        dv = frame["dv"].to_numpy(dtype=np.float64)
        return [np.where(has_basic, np.where(dv > 0, dv, -np.inf), np.nan)]
    if knobs.SCORE == "abturn":
        return [turn]
    if knobs.SCORE == "ch3":
        return [_ep(frame, has_basic), turn]
    if knobs.SCORE == "range20":
        return [-frame["range"].to_numpy(dtype=np.float64)]
    raise ValueError(f"unknown knobs.SCORE {knobs.SCORE!r}")


def _ep(frame, has_basic):
    pe = frame["pe"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ep = np.where(pe > 0, 1.0 / pe, -np.inf)
    return np.where(has_basic, ep, np.nan)


def _pct_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()
