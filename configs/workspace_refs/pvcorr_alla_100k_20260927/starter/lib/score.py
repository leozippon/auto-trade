"""Correlation of the daily return and traded amount.

The score is that correlation over 20 sessions. Amount that rises with the
return scores higher. This is not the return itself and not a turnover ratio.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
SESSIONS = 20
MIN_SESSIONS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg", "amount"]


def _corr(frame):
    if len(frame) < MIN_SESSIONS:
        return float("nan")
    return float(frame["pct_chg"].corr(frame["amount"]))


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < MIN_SESSIONS:
        raise RuntimeError("not enough sessions to correlate return and amount")
    window = bars[bars["trade_date"].isin(set(dates[-SESSIONS:]))].copy()
    window["pct_chg"] = pd.to_numeric(window["pct_chg"], errors="coerce")
    window["amount"] = pd.to_numeric(window["amount"], errors="coerce")
    window = window.dropna(subset=["pct_chg", "amount"])
    corr = window.groupby("ts_code")[["pct_chg", "amount"]].apply(_corr, include_groups=False)
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
