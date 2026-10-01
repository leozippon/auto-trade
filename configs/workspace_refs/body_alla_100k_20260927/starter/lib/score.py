"""How much of the day's range is the open-to-close move.

The score is the mean, over 20 sessions, of abs(close - open) / (high - low).
A larger body scores higher. This is not the size of the range and not a return.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
SESSIONS = 20
MIN_SESSIONS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < MIN_SESSIONS:
        raise RuntimeError("not enough sessions to measure the daily body")
    window = bars[bars["trade_date"].isin(set(dates[-SESSIONS:]))].copy()
    for name in ("open", "high", "low", "close"):
        window[name] = pd.to_numeric(window[name], errors="coerce")
    span = window["high"] - window["low"]
    window["body"] = (window["close"] - window["open"]).abs() / span.where(span > 0)
    mean = window.groupby("ts_code")["body"].mean()
    count = window.groupby("ts_code")["body"].count()
    values = mean.where(count >= MIN_SESSIONS).reindex(list(codes))
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
