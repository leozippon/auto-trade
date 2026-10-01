"""Book multiple on the newest bar.

The score is minus pb. A cheaper price-to-book scores higher. This is not
ps_ttm and not pe_ttm.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "pb"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    pb = pd.to_numeric(last["pb"], errors="coerce")
    values = (-pb.where(pb > 0)).reindex(list(codes))
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
