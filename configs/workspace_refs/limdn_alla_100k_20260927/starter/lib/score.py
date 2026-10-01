"""How often the last 60 sessions touched the down limit.

The score is minus that fraction. Names that hit the down limit less often
score higher. This is a count of limit events, not a return reversal.
"""

from datetime import timedelta

import pandas as pd

LOOKBACK_DAYS = 100
DAILY_COLUMNS = ["ts_code", "trade_date", "low", "down_limit"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < 40:
        raise RuntimeError("not enough sessions to count down-limit touches")
    use = set(dates[-60:])
    window = bars[bars["trade_date"].isin(use)].copy()
    low = pd.to_numeric(window["low"], errors="coerce")
    limit = pd.to_numeric(window["down_limit"], errors="coerce")
    window["hit"] = low.notna() & limit.notna() & (low <= limit + 0.01)
    count = window.groupby("ts_code")["hit"].sum()
    n = window.groupby("ts_code")["trade_date"].nunique()
    fraction = count / n.where(n >= 40)
    values = (-fraction).reindex(list(codes))
    return values, {"names": int(fraction.notna().sum())}

def shuffle(values, decision_at):
    import numpy as np
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
