"""Change in free shares over 60 sessions.

The score is minus that growth. Names whose free share count rose less
score higher. This is not the level free_share / float_share.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 120
SESSIONS = 60
DAILY_COLUMNS = ["ts_code", "trade_date", "free_share"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < SESSIONS:
        raise RuntimeError("not enough sessions to measure free-share growth")
    early_day, late_day = dates[-SESSIONS], dates[-1]
    use = bars[bars["trade_date"].isin((early_day, late_day))].copy()
    use["free_share"] = pd.to_numeric(use["free_share"], errors="coerce")
    early = use[use["trade_date"] == early_day].drop_duplicates("ts_code").set_index("ts_code")["free_share"]
    late = use[use["trade_date"] == late_day].drop_duplicates("ts_code").set_index("ts_code")["free_share"]
    growth = late / early.where(early > 0) - 1.0
    growth = growth.where(late > 0)
    values = (-growth).reindex(list(codes))
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
