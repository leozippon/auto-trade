"""The registered label: a benchmark-residual forward return, ranked inside the index.

Horizon N = `panel.HOLD` = 10 trading days, registered before the first batch and
not a variant axis. A signal on date t is bought at the open of t+1 and sold at
the open of t+1+N, so the raw leg is `O[t+N+1] / O[t+1] - 1` -- the same
open-to-open convention the trade layer executes.

Residual, not raw. The raw leg carries the market: in a benchmark-relative arm
it teaches the model to prefer whatever had the highest beta over the window,
which is exactly the exposure the earlier sequence arm was killed by when the
size and beta regime turned. So the label subtracts the name's own beta times
the benchmark's forward return over the SAME two opens:

    y_i(t) = (O_i[t+N+1] / O_i[t+1] - 1) - b_i(t) * (B[t+N+1] / B[t+1] - 1)

`b_i(t)` is a trailing OLS beta of daily adjusted close returns on the CSI 300's
own daily return over `BETA_DAYS`, shrunk toward 1 and clipped -- a trailing
estimate, so nothing after t enters it. Shrinkage is not decoration: an
unshrunk 120-day beta on 300 large caps has a standard error large enough that
the residual would import estimation noise rather than remove exposure.

`INDUSTRY_DEMEAN` additionally removes the SW level-1 mean of the residual on
each date. It is a registered switch, not a free choice: the adjudicating
neutralisation cannot see industry, so a label that already is industry-neutral
and a label that is not answer different questions, and the arm must say which
one it asked before it spends a batch.

The target is then the cross-sectional rank of the residual over the names that
were CONSTITUENTS on that date and have a finite residual, mapped to
`(rank - 1) / n - 0.5`. Ranking inside the index is what makes the label match
the universe the book is drawn from; ranking over the whole market would train
the model on a cross-section it is never allowed to trade.
"""

import numpy as np
import pandas as pd

BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)
INDUSTRY_DEMEAN = False


def _trailing_beta(data, benchmark):
    """(dates, names) trailing beta of each name against the benchmark, shrunk and clipped."""

    closes = pd.DataFrame(data["close"], index=pd.Index(data["dates"]))
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
    """(stock forward open-to-open return, benchmark forward return) aligned on the signal date."""

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


def residual(data, benchmark, hold):
    """(dates, names) benchmark-residual forward return; NaN off the constituent set."""

    stock, bench = forward_returns(data, benchmark, hold)
    beta = _trailing_beta(data, benchmark)
    out = stock - beta * bench[:, None]
    out = np.where(data["member"] & np.isfinite(out), out, np.nan)
    if INDUSTRY_DEMEAN:
        industry = np.asarray(data["industry"])
        for label_name in np.unique(industry):
            columns = np.nonzero(industry == label_name)[0]
            block = out[:, columns]
            total = np.nansum(block, axis=1, keepdims=True)
            count = np.isfinite(block).sum(axis=1, keepdims=True)
            out[:, columns] = block - np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return out


def residual_rank(data, benchmark, hold):
    """(dates, names) float32 label: cross-sectional rank of the residual in [-0.5, 0.5]."""

    values = residual(data, benchmark, hold)
    frame = pd.DataFrame(values)
    ranks = frame.rank(axis=1, pct=True).to_numpy()
    count = np.isfinite(values).sum(axis=1, keepdims=True)
    label = ranks - 1.0 / np.maximum(count, 1) - 0.5
    return np.where(np.isfinite(values), label, np.nan).astype(np.float32)
