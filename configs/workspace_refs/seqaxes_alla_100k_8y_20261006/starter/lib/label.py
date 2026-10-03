"""The training label: the baseline 10-day beta residual and the label lane's three variants.

`knobs.INDEX` is the label's benchmark leg and the only index code in this
package (`lib/panel.py` reads its `index_daily` rows). It must equal the arm's
`benchmark_index` (000852.SH): the host neutralises the graded series against
that index.

A signal on date t is bought at the open of t+1 and sold at the open of
t+1+N, the convention the book executes:

    base   y_i(t) = (O_i[t+N+1] / O_i[t+1] - 1) - b_i(t) * (B[t+N+1] / B[t+1] - 1),  N = 10

`O_i` are adjusted opens and `b_i(t)` is a trailing OLS beta of daily ADJUSTED
close returns on the index's daily return over `BETA_DAYS` (at least
`BETA_MIN_DAYS` observations), shrunk toward 1 and clipped; without enough
observations it is 1.0. Nothing after t enters it. This is the baseline label,
the 100k bag's, unchanged. `knobs.LABEL` picks one change to it:

    bench  b_i(t) = 1 for every name: the forward return minus the index's.
           The index leg is one number per date, so the per-date rank below is
           exactly the rank of the raw forward return; this variant asks what
           the beta leg contributes.
    size   the baseline residual minus its mean within each size quintile of
           the date (circulating market value `circ_mv` on date t, quintiles of
           the names that have a residual and a value): the heads learn which
           names beat their own size group, the exposure the host's size
           spread neutralises away.
    hold   the baseline residual with N = `knobs.HOLD_ALIGNED`, the book's
           holding period; the training embargo widens with it (`HOLD`).

The target is the per-date rank of the label over every name with a finite
label, mapped to `(rank - 1) / n - 0.5`; the loss later uses only the
tradable names of that date.
"""

import numpy as np
import pandas as pd

from lib import knobs

BASE_HOLD = 10
# The label horizon of this package's runs and the embargo before validation.
HOLD = knobs.HOLD_ALIGNED if knobs.LABEL == "hold" else BASE_HOLD
BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)
SIZE_GROUPS = 5


def _trailing_beta(data, benchmark):
    """(dates, names) trailing beta of each name against the index, shrunk and clipped."""

    closes = pd.DataFrame(data["close_adj"], index=pd.Index(data["dates"]))
    bench = pd.to_numeric(benchmark["close"], errors="coerce").reindex(pd.Index(data["dates"]))
    if bench.notna().sum() < BETA_MIN_DAYS + 1:
        raise RuntimeError("the benchmark series does not cover the panel window")
    bench = bench.ffill()
    stock_return = closes.pct_change(fill_method=None)
    bench_return = bench.pct_change(fill_method=None)
    covariance = stock_return.rolling(BETA_DAYS, min_periods=BETA_MIN_DAYS).cov(bench_return)
    variance = bench_return.rolling(BETA_DAYS, min_periods=BETA_MIN_DAYS).var()
    raw = covariance.div(variance.replace(0.0, np.nan), axis=0)
    beta = BETA_SHRINK * raw + (1.0 - BETA_SHRINK)
    return beta.clip(*BETA_CLIP).fillna(1.0).to_numpy()


def forward_returns(data, benchmark, hold):
    """(stock forward open-to-open return, index forward return) aligned on the signal date."""

    opens = data["open_adj"]
    rows = opens.shape[0]
    stock = np.full_like(opens, np.nan)
    if rows > hold + 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            stock[: rows - hold - 1] = opens[hold + 1:] / opens[1: rows - hold] - 1.0
    bench_open = pd.to_numeric(benchmark["open"], errors="coerce").reindex(
        pd.Index(data["dates"])).ffill().to_numpy()
    bench = np.full(rows, np.nan)
    if rows > hold + 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            bench[: rows - hold - 1] = bench_open[hold + 1:] / bench_open[1: rows - hold] - 1.0
    return stock, bench


def group_demean(values, size, groups=SIZE_GROUPS):
    """`values` minus their mean within each date's size quantile group; NaN where either is missing."""

    known = np.isfinite(values) & np.isfinite(size)
    ranks = pd.DataFrame(np.where(known, size, np.nan)).rank(axis=1, pct=True).to_numpy()
    group = np.ceil(np.nan_to_num(ranks) * groups) - 1
    out = np.full(values.shape, np.nan)
    for g in range(groups):
        member = known & (group == g)
        count = member.sum(axis=1, keepdims=True)
        mean = np.where(member, values, 0.0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
        out = np.where(member, values - mean, out)
    return out


def rank_target(values):
    """(dates, names) float32: per-date rank of the finite values mapped to [-0.5, 0.5)."""

    values = np.where(np.isfinite(values), values, np.nan)
    ranks = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy()
    count = np.isfinite(values).sum(axis=1, keepdims=True)
    target = ranks - 1.0 / np.maximum(count, 1) - 0.5
    return np.where(np.isfinite(values), target, np.nan).astype(np.float32)


def target(data, benchmark):
    """The training target of `knobs.LABEL` over the panel."""

    knobs.leg()
    stock, bench = forward_returns(data, benchmark, HOLD)
    if knobs.LABEL == "bench":
        return rank_target(stock - bench[:, None])
    values = stock - _trailing_beta(data, benchmark) * bench[:, None]
    if knobs.LABEL == "size":
        values = group_demean(values, data["circ_mv"])
    return rank_target(values)
