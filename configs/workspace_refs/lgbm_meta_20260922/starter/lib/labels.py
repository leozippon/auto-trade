"""Two registered labels: the primary's residual rank, and the meta stage's binary outcome.

Horizon N = `data.HOLD` = 10 trading days for both, registered before the first
batch and not a variant axis. A signal on bar t is bought at the open of t+1 and
sold at the open of t+1+N, which is what the trade layer executes.

Primary label, the residual rank. The frozen lineage ranked the RAW forward
open-to-open return. Inside a benchmark-relative universe that teaches the model
to prefer whatever carried the most market, which is the exposure that killed
every book in the forward window. So the raw leg is first stripped of the name's
own beta times the benchmark's move over the SAME two opens

    r_i(t) = (O_i[t+N+1] / O_i[t+1] - 1) - b_i(t) * (B[t+N+1] / B[t+1] - 1)

with `b_i(t)` a trailing OLS beta of daily adjusted-close returns on the CSI 300
over `BETA_DAYS`, shrunk toward 1 and clipped -- trailing, so nothing after t
enters it, and shrunk because an unshrunk beta on 300 large caps carries enough
estimation noise to re-import the exposure it is meant to remove. The target is
the cross-sectional rank of `r_i(t)` over that bar's CONSTITUENTS, mapped to
`(rank - 1) / n - 0.5`.

Meta label, the second stage's question. For a bar the primary has an
out-of-fold score on, each of its top-decile picks is labelled

    1 if r_i(t) > median over that bar's constituents of r(t), else 0

i.e. "did this pick beat a name drawn with no skill from the same shape over the
same horizon". The cross-sectional median of the same residual is the offline
stand-in for the host's zero-skill panel: the panel redraws names with the same
index membership and the same affordability and replays the candidate's own
trade skeleton through the Broker, which this cannot reproduce inside `fit`.
The stand-in is what the classifier is trained on; the panel is what the arm is
graded on, and `sources.md` keeps the two apart.
"""

import numpy as np
import pandas as pd

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


def beats_median(values):
    """(names, dates) 1.0 where the residual beats that bar's cross-sectional median, else 0.0."""

    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    columns = finite.any(axis=0)
    if not columns.any():
        return out
    block = np.where(finite[:, columns], values[:, columns], np.nan)
    median = np.nanmedian(block, axis=0, keepdims=True)
    out[:, columns] = np.where(finite[:, columns], (block > median).astype(np.float64), np.nan)
    return out
