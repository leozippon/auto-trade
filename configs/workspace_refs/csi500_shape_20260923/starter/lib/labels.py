"""The round's default label, unchanged: a benchmark-residual N-day open-to-open rank.

The horizon `HOLD` is registered below, before the first batch. The label is NOT
this arm's axis and is not a variant knob here: a sibling arm of this round is
moving the label on the confirmed carrier, and two arms moving the same lever at
once cannot tell which one moved a reading. `families.md` forbids changing it.

What the label is. A signal on bar t is acted on at the decision of the next
trading day, so the return is measured from the open after the signal to the
open N trading days later, and the benchmark's own move over the SAME two opens
is subtracted, scaled by the name's trailing beta:

    r_i(t) = (O_i[t+N+1] / O_i[t+1] - 1) - b_i(t) * (B[t+N+1] / B[t+1] - 1)

`b_i(t)` is a trailing OLS beta of daily adjusted-close returns on the benchmark
over BETA_DAYS, shrunk toward 1 and clipped -- trailing, so nothing after t
enters it, and shrunk because an unshrunk beta carries enough estimation noise
to re-import the exposure it is meant to remove. The target is the
cross-sectional rank of `r_i(t)` over that bar's CONSTITUENTS, mapped to
(rank - 1) / n - 0.5.

The benchmark here is the CSI 500, not the CSI 300: `B` comes from `index_daily`
filtered on `lib/index.py`'s one index code, and the constituents the rank is
taken over are that index's. Residualising a CSI 500 book against the CSI 300
would leave the whole mid-cap-minus-large-cap spread in the label, which is the
one exposure this arm must not accidentally be selecting on.
"""

import numpy as np
import pandas as pd

HOLD = 10                    # label horizon in trading days, registered up front
BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)


def trailing_beta(panel):
    """(names, dates) trailing beta against the benchmark, shrunk toward 1 and clipped."""

    closes = pd.DataFrame(panel["C"].T, index=pd.Index(panel["dates"]))
    bench = pd.Series(panel["bench_close"], index=pd.Index(panel["dates"]))
    if bench.notna().sum() < BETA_MIN_DAYS + 1:
        raise RuntimeError("the benchmark series does not cover the panel window")
    stock_return = closes.pct_change(fill_method=None)
    bench_return = bench.pct_change(fill_method=None)
    covariance = stock_return.rolling(BETA_DAYS, min_periods=BETA_MIN_DAYS).cov(bench_return)
    variance = bench_return.rolling(BETA_DAYS, min_periods=BETA_MIN_DAYS).var()
    raw = covariance.div(variance.replace(0.0, np.nan), axis=0)
    beta = BETA_SHRINK * raw + (1.0 - BETA_SHRINK)
    return beta.clip(*BETA_CLIP).fillna(1.0).to_numpy().T


def residual(panel, hold):
    """(names, dates) benchmark-residual forward return; NaN off the constituent set."""

    opens = panel["O"]
    total = opens.shape[1]
    stock = np.full_like(opens, np.nan)
    bench = np.full(total, np.nan)
    if total > hold + 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            stock[:, : total - hold - 1] = opens[:, hold + 1:] / opens[:, 1: total - hold] - 1.0
            bench_open = panel["bench_open"]
            bench[: total - hold - 1] = bench_open[hold + 1:] / bench_open[1: total - hold] - 1.0
    out = stock - trailing_beta(panel) * bench[None, :]
    return np.where(panel["member"] & np.isfinite(out), out, np.nan)


def rank_target(values):
    """(names, dates) cross-sectional rank of a residual in [-0.5, 0.5], and where it exists."""

    frame = pd.DataFrame(values)
    ranks = frame.rank(axis=0, pct=True).to_numpy()
    count = np.isfinite(values).sum(axis=0, keepdims=True)
    target = ranks - 1.0 / np.maximum(count, 1) - 0.5
    realized = np.isfinite(values)
    return np.where(realized, target, np.nan), realized


def target(panel):
    """(target, realised) for the registered label: the single entry point `main.fit` calls."""

    return rank_target(residual(panel, HOLD))
