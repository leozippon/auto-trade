"""Sales multiple versus the Shenwan level-1 median.

The score is the industry median ps_ttm minus the name's own ps_ttm. Cheaper
than its industry scores higher. This is not the raw multiple.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "ps_ttm"]
UNI_COLUMNS = ["ts_code", "l1_name"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    ps = pd.to_numeric(last["ps_ttm"], errors="coerce")
    ps = ps.where(ps > 0)
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_name"].reindex(ps.index).fillna("未分类").astype(str)
    median = ps.groupby(industry).transform("median")
    values = (median - ps).reindex(list(codes))
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
