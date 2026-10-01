"""Traded amount versus the Shenwan level-1 median.

The score is the name's 20-session median amount divided by its industry's
median of those medians. More traded than its peers scores higher. This is
not market value and not a return.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
SESSIONS = 20
MIN_SESSIONS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "amount"]
UNI_COLUMNS = ["ts_code", "l1_name"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < MIN_SESSIONS:
        raise RuntimeError("not enough sessions to measure amount")
    window = bars[bars["trade_date"].isin(set(dates[-SESSIONS:]))].copy()
    window["amount"] = pd.to_numeric(window["amount"], errors="coerce")
    window = window[window["amount"] > 0]
    med = window.groupby("ts_code")["amount"].median()
    count = window.groupby("ts_code")["amount"].count()
    med = med.where(count >= MIN_SESSIONS)
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_name"].reindex(med.index).fillna("未分类").astype(str)
    base = med.groupby(industry).median()
    ratio = med / industry.map(base).where(lambda s: s > 0)
    values = ratio.reindex(list(codes))
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
