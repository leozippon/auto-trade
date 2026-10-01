"""Negative coskewness to CSI 1000 inside the current index.

The score is minus the slope of the stock's daily return on the square of the
demeaned index return. Index returns are percent and are divided by 100.
Stocks that do worse when the index move is large rank higher. Same-day beta
is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 100
MIN_DAYS = 30
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "pct_chg"]


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
    index = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=INDEX_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    market = pd.to_numeric(
        index.assign(trade_date=index["trade_date"].astype(str)).drop_duplicates("trade_date").set_index("trade_date")["pct_chg"],
        errors="coerce",
    ).sort_index() / 100.0
    centered = market - market.mean()
    squared = centered * centered
    if squared.dropna().empty:
        raise RuntimeError("no CSI 1000 index returns")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
    )
    bars = bars[bars["ts_code"].isin(current)].copy()
    bars["x"] = bars["trade_date"].map(squared)
    values = {}
    for code, group in bars.groupby("ts_code"):
        paired = group[group["ret"].notna() & group["x"].notna()]
        if len(paired) < MIN_DAYS:
            continue
        x = paired["x"] - paired["x"].mean()
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        slope = float((x * (paired["ret"] - paired["ret"].mean())).sum()) / denom
        values[code] = -slope
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have a coskewness slope")
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
