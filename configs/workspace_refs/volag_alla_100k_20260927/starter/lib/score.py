"""Lagged amount response inside the current CSI 1000.

The score is the beta of today's percent change in the stock's amount on
yesterday's percent change in CSI 1000 amount. Percent changes cancel units.
A higher beta ranks higher. A return beta is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 100
MIN_DAYS = 30
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "amount"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "amount"]


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
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    bars = bars[bars["ts_code"].isin(members)]
    bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce")
    index = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=INDEX_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    market = pd.to_numeric(
        index.assign(trade_date=index["trade_date"].astype(str)).drop_duplicates("trade_date").set_index("trade_date")["amount"],
        errors="coerce",
    ).sort_index()
    market = market.where(market > 0)
    lagged = (market / market.shift(1) - 1.0).shift(1)
    if lagged.dropna().empty:
        raise RuntimeError("no lagged CSI 1000 amount changes")
    values = {}
    for code, group in bars.groupby("ts_code"):
        ordered = group.sort_values("trade_date")
        amount = ordered["amount"]
        change = amount / amount.shift(1) - 1.0
        paired = pd.DataFrame({"chg": change.to_numpy(), "mkt": ordered["trade_date"].map(lagged).to_numpy()})
        paired = paired[paired["chg"].notna() & paired["mkt"].notna() & paired["chg"].abs().lt(10) & paired["mkt"].abs().lt(10)]
        if len(paired) < MIN_DAYS:
            continue
        x = paired["mkt"] - paired["mkt"].mean()
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        beta = float((x * (paired["chg"] - paired["chg"].mean())).sum()) / denom
        values[code] = beta
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have a lagged amount beta")
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
