"""How steady recent traded amount is.

The score is minus the coefficient of variation of amount over the last 20
sessions. Steadier amount scores higher. This is not a return, not turnover,
and not a limit-touch count.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
SESSIONS = 20
MIN_SESSIONS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "amount"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < MIN_SESSIONS:
        raise RuntimeError("not enough sessions to measure amount variation")
    window = bars[bars["trade_date"].isin(set(dates[-SESSIONS:]))].copy()
    window["amount"] = pd.to_numeric(window["amount"], errors="coerce")
    window = window[window["amount"] > 0]
    grouped = window.groupby("ts_code")["amount"]
    count = grouped.count()
    mean = grouped.mean()
    std = grouped.std()
    cv = std / mean.where(mean > 0)
    cv = cv.where(count >= MIN_SESSIONS)
    values = (-cv).reindex(list(codes))
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
