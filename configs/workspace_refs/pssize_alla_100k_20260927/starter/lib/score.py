"""Sales multiple after a same-day size regression.

The score is minus the residual of ps_ttm on log circulating market value.
Cheaper than the size line scores higher. This is not the raw multiple.
"""

from datetime import timedelta
from math import log

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 15
DAILY_COLUMNS = ["ts_code", "trade_date", "ps_ttm", "circ_mv"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    ps = pd.to_numeric(last["ps_ttm"], errors="coerce")
    size = pd.to_numeric(last["circ_mv"], errors="coerce")
    frame = pd.DataFrame({"ps": ps, "size": size}).dropna()
    frame = frame[(frame["ps"] > 0) & (frame["size"] > 0)]
    if len(frame) < 30:
        raise RuntimeError("not enough names to residualize ps_ttm on size")
    log_size = frame["size"].map(log)
    x = log_size - log_size.mean()
    y = frame["ps"] - frame["ps"].mean()
    denom = float((x * x).sum())
    if denom <= 0:
        raise RuntimeError("size has no cross-sectional variance")
    slope = float((x * y).sum()) / denom
    fitted = frame["ps"].mean() + slope * x
    values = (-(frame["ps"] - fitted)).reindex(list(codes))
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
