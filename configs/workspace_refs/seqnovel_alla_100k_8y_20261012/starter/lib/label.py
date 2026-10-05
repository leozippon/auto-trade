"""The training label: the 100k bag's 10-day open-to-open beta residual, ranked per date.

`knobs.INDEX` is the label's benchmark leg (`lib/panel.py` reads its
`index_daily` rows). It must equal the arm's `benchmark_index` (000852.SH):
the host neutralises the graded series against that index. The label is
undefined on a date before the index's first visible row (its forward return
is missing), so such a date never enters a training window (`lib/model.py`).

A signal on date t is bought at the open of t+1 and sold at the open of
t+1+N, the convention the book executes:

    y_i(t) = (O_i[t+N+1] / O_i[t+1] - 1) - b_i(t) * (B[t+N+1] / B[t+1] - 1),  N = 10

`O_i` are adjusted opens and `b_i(t)` is a trailing OLS beta of daily ADJUSTED
close returns on the index's daily return over `BETA_DAYS` (at least
`BETA_MIN_DAYS` observations), shrunk toward 1 and clipped; without enough
observations it is 1.0. Nothing after t enters it. This is the 100k bag's
label unchanged, and fixed in this package.

The target is the per-date rank of the label over every name with a finite
label, mapped to `(rank - 1) / n - 0.5`; the loss later uses only the
tradable names of that date.
"""

import numpy as np
import pandas as pd

HOLD = 10                    # label horizon in trading days, and the embargo before validation
BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)


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


def rank_target(values):
    """(dates, names) float32: per-date rank of the finite values mapped to [-0.5, 0.5)."""

    values = np.where(np.isfinite(values), values, np.nan)
    ranks = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy()
    count = np.isfinite(values).sum(axis=1, keepdims=True)
    target = ranks - 1.0 / np.maximum(count, 1) - 0.5
    return np.where(np.isfinite(values), target, np.nan).astype(np.float32)


def target(data, benchmark):
    """(dates, names) float32 training target over the panel."""

    stock, bench = forward_returns(data, benchmark, HOLD)
    return rank_target(stock - _trailing_beta(data, benchmark) * bench[:, None])
