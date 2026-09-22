"""The registered label: the benchmark-residual forward return, ranked in the bar.

Horizon N = `knobs.HORIZON` trading days. A signal on bar t is bought at the
open of t+1 and sold at the open of t+1+N, which is what the trade layer
executes.

The graduated lineage ranked the RAW forward open-to-open return. Inside a
benchmark-relative universe that teaches the model to prefer whatever carried
the most market, which is the exposure that killed every book in the forward
window. So the raw leg is first stripped of the name's own beta times the
benchmark's move over the SAME two opens

    r_i(t) = (O_i[t+N+1] / O_i[t+1] - 1) - b_i(t) * (B[t+N+1] / B[t+1] - 1)

with `b_i(t)` a trailing OLS beta of daily adjusted-close returns on the
benchmark over `knobs.BETA_DAYS`, shrunk toward 1 and clipped -- trailing, so
nothing after t enters it, and shrunk because an unshrunk beta on 300 large
caps carries enough estimation noise to re-import the exposure it removes.
The target is the cross-sectional rank of `r_i(t)` over that bar's
CONSTITUENTS, mapped to `(rank - 1) / n - 0.5`.

This label is fixed round-wide. It is not a variant axis of this arm: the arm
prices an information block on a fixed carrier, and moving the label at the
same time would make "what changed" unanswerable.
"""

import numpy as np
import pandas as pd

from lib import knobs


def trailing_beta(panel):
    """(names, dates) trailing beta against the benchmark, shrunk toward 1 and clipped."""

    closes = pd.DataFrame(panel["C"].T, index=pd.Index(panel["dates"]))
    bench = pd.Series(panel["bench_close"], index=pd.Index(panel["dates"]))
    if bench.notna().sum() < knobs.BETA_MIN_DAYS + 1:
        raise RuntimeError("the benchmark series does not cover the panel window")
    stock_return = closes.pct_change(fill_method=None)
    bench_return = bench.pct_change(fill_method=None)
    covariance = stock_return.rolling(knobs.BETA_DAYS, min_periods=knobs.BETA_MIN_DAYS).cov(bench_return)
    variance = bench_return.rolling(knobs.BETA_DAYS, min_periods=knobs.BETA_MIN_DAYS).var()
    raw = covariance.div(variance.replace(0.0, np.nan), axis=0)
    beta = knobs.BETA_SHRINK * raw + (1.0 - knobs.BETA_SHRINK)
    return beta.clip(*knobs.BETA_CLIP).fillna(1.0).to_numpy().T


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
