"""Cheaper than the size line inside CSI 1000.

Current members only. The score is minus the residual of ps_ttm on log
circulating market value, fit on those members. The raw multiple is not the score.
"""

from datetime import timedelta
from math import log

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 80
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "ps_ttm", "circ_mv"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")) & frame["con_code"].notna()]
    frame = frame.assign(con_code=frame["con_code"].astype(str), trade_date=frame["trade_date"].astype(str))
    frame = frame[pd.to_numeric(frame["weight"], errors="coerce") > 0]
    if frame.empty:
        raise RuntimeError("no visible CSI 1000 weights")
    latest = str(frame["trade_date"].max())
    current = set(frame.loc[frame["trade_date"] == latest, "con_code"])
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    ps = pd.to_numeric(last["ps_ttm"], errors="coerce")
    size = pd.to_numeric(last["circ_mv"], errors="coerce")
    kept = []
    for code in current:
        multiple = ps.get(code)
        mv = size.get(code)
        if multiple is None or mv is None or not (multiple == multiple) or not (mv == mv):
            continue
        if multiple <= 0 or mv <= 0:
            continue
        kept.append((code, float(multiple), float(mv)))
    if len(kept) < 30:
        raise RuntimeError("not enough CSI 1000 names to residualize ps_ttm on size")
    table = pd.DataFrame(kept, columns=["code", "ps", "size"]).set_index("code")
    log_size = table["size"].map(log)
    centered = log_size - log_size.mean()
    denom = float((centered * centered).sum())
    if denom <= 0:
        raise RuntimeError("size has no cross-sectional variance")
    slope = float((centered * (table["ps"] - table["ps"].mean())).sum()) / denom
    fitted = table["ps"].mean() + slope * centered
    values = (-(table["ps"] - fitted)).reindex(list(codes))
    return values, {"scored": int(values.notna().sum()), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
