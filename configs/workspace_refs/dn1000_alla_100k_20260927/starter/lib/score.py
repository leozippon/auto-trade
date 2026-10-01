"""Down-day amount share inside current CSI 1000, over about twenty sessions.

The score is that share. A larger share ranks higher.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 80
MIN_DAYS = 10
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg", "amount"]


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
    members = set(frame.loc[frame["trade_date"] == latest, "con_code"])
    day_start = (context.inference_at - timedelta(days=40)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", day_start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
        amt=pd.to_numeric(bars["amount"], errors="coerce"),
    )
    use = bars[bars["ts_code"].isin(members) & bars["ret"].notna() & bars["amt"].notna() & (bars["amt"] > 0)]
    if use.empty:
        raise RuntimeError("no recent CSI 1000 amount rows")
    counts = use.groupby("ts_code").size()
    total = use.groupby("ts_code")["amt"].sum()
    down = use.loc[use["ret"] < 0].groupby("ts_code")["amt"].sum()
    share = (down / total).reindex(counts.index)
    share = share[counts >= MIN_DAYS].dropna()
    if int(share.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have a down-day amount share")
    return share.reindex(list(codes)), {"scored": int(share.shape[0]), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
