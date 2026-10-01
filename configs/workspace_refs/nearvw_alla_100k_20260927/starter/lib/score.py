"""Close versus the day's average trade, inside the current CSI 1000.

The score is minus the mean absolute gap between close and amount/vol over
about twenty sessions. Amount is yuan and volume is shares, so their ratio is
a price. A smaller gap ranks higher. The signed gap is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 40
MIN_DAYS = 10
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "close", "vol", "amount"]


def _members(context, start):
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
    return set(frame.loc[frame["trade_date"] == latest, "con_code"]), latest


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    members, latest = _members(context, start)
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        close=pd.to_numeric(bars["close"], errors="coerce"),
        vol=pd.to_numeric(bars["vol"], errors="coerce"),
        amount=pd.to_numeric(bars["amount"], errors="coerce"),
    )
    use = bars[bars["ts_code"].isin(members) & (bars["close"] > 0) & (bars["vol"] > 0) & (bars["amount"] > 0)]
    values = {}
    for code, group in use.groupby("ts_code"):
        if len(group) < MIN_DAYS:
            continue
        gap = (group["close"] - group["amount"] / group["vol"]).abs() / group["close"]
        values[code] = -float(gap.mean())
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have a close-to-vwap gap")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
