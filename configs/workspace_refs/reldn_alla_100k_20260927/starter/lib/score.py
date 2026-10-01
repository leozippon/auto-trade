"""Share of amount on days the stock lagged CSI 1000, over about twenty sessions.

The score is that share. A larger share ranks higher.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 40
MIN_DAYS = 10
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg", "amount"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "pct_chg"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
        amt=pd.to_numeric(bars["amount"], errors="coerce"),
    )
    index = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=INDEX_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    index = index.assign(trade_date=index["trade_date"].astype(str))
    market = pd.to_numeric(index.set_index("trade_date")["pct_chg"], errors="coerce")
    if market.dropna().empty:
        raise RuntimeError("no visible CSI 1000 index returns")
    bars["mkt"] = bars["trade_date"].map(market)
    use = bars[bars["ret"].notna() & bars["mkt"].notna() & bars["amt"].notna() & (bars["amt"] > 0)]
    if use.empty:
        raise RuntimeError("no rows to compare with CSI 1000")
    counts = use.groupby("ts_code").size()
    total = use.groupby("ts_code")["amt"].sum()
    down = use.loc[use["ret"] < use["mkt"]].groupby("ts_code")["amt"].sum()
    share = (down / total).reindex(counts.index)
    share = share[counts >= MIN_DAYS].dropna()
    if int(share.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 names have a relative down-day amount share")
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
