"""Autocorrelation of traded amount.

The score is the correlation of amount with its previous session, over the
last 20 sessions. Stickier amount scores higher. This is not a return and
not a turnover ratio.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
SESSIONS = 20
MIN_PAIRS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "amount"]


def _autocorr(frame):
    ordered = frame.sort_values("trade_date")
    amount = pd.to_numeric(ordered["amount"], errors="coerce")
    pair = pd.DataFrame({"now": amount.to_numpy()[1:], "prev": amount.to_numpy()[:-1]}).dropna()
    if len(pair) < MIN_PAIRS:
        return float("nan")
    return float(pair["now"].corr(pair["prev"]))


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < MIN_PAIRS + 1:
        raise RuntimeError("not enough sessions to autocorrelate amount")
    window = bars[bars["trade_date"].isin(set(dates[-SESSIONS:]))]
    corr = window.groupby("ts_code")[["trade_date", "amount"]].apply(_autocorr, include_groups=False)
    values = corr.reindex(list(codes))
    return values, {"scored": int(values.notna().sum())}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
