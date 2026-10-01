"""Lagged response to CSI 1000.

The score is the beta of today's stock return on yesterday's CSI 1000 return
over about sixty sessions. Index daily returns are percent; stock daily
returns are decimals. A higher lagged beta ranks higher. Same-day beta is not
the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 100
MIN_DAYS = 30
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "pct_chg"]


def _lagged_market(context, start):
    index = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=INDEX_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    market = pd.to_numeric(
        index.assign(trade_date=index["trade_date"].astype(str)).drop_duplicates("trade_date").set_index("trade_date")["pct_chg"],
        errors="coerce",
    ).sort_index() / 100.0
    lagged = market.shift(1)
    if lagged.dropna().empty:
        raise RuntimeError("no lagged CSI 1000 index returns")
    return lagged


def _betas(bars, lagged):
    bars = bars.copy()
    bars["mkt"] = bars["trade_date"].map(lagged)
    use = bars[bars["ret"].notna() & bars["mkt"].notna()]
    values = {}
    for code, group in use.groupby("ts_code"):
        if len(group) < MIN_DAYS:
            continue
        x = group["mkt"] - group["mkt"].mean()
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        beta = float((x * (group["ret"] - group["ret"].mean())).sum()) / denom
        values[code] = beta
    return values


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
    )
    values = _betas(bars, _lagged_market(context, start))
    if len(values) < 12:
        raise RuntimeError("fewer than 12 names have a lagged beta")
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
