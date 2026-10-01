"""Asset turnover from static pb and ps.

pb / ps cancels price and equals sales over book. A higher ratio ranks higher.
The level of either multiple is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
DAILY_COLUMNS = ["ts_code", "trade_date", "pb", "ps"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    book = pd.to_numeric(last["pb"], errors="coerce")
    sales = pd.to_numeric(last["ps"], errors="coerce")
    values = {}
    for code in codes:
        pb = book.get(code, float("nan"))
        ps = sales.get(code, float("nan"))
        if not (pb == pb) or not (ps == ps) or pb <= 0 or ps <= 0:
            continue
        values[code] = float(pb) / float(ps)
    if len(values) < 12:
        raise RuntimeError("fewer than 12 names have positive pb and ps")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values)}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
