"""Share of traded amount on down days, over about twenty sessions.

The score is that share. A larger share ranks higher. It is not a return.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
LOOKBACK_DAYS = 40
MIN_DAYS = 10
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg", "amount"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
        amt=pd.to_numeric(bars["amount"], errors="coerce"),
    )
    use = bars[bars["ret"].notna() & bars["amt"].notna() & (bars["amt"] > 0)]
    if use.empty:
        raise RuntimeError("no recent amount rows")
    counts = use.groupby("ts_code").size()
    total = use.groupby("ts_code")["amt"].sum()
    down = use.loc[use["ret"] < 0].groupby("ts_code")["amt"].sum()
    share = (down / total).reindex(counts.index)
    share = share[counts >= MIN_DAYS].dropna()
    if int(share.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 names have a down-day amount share")
    return share.reindex(list(codes)), {"scored": int(share.shape[0])}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
