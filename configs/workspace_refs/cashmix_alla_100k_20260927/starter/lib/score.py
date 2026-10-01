"""Share of the latest distribution that was cash, not a stock dividend.

Reads corporate_actions. The score is cash per share divided by cash plus the
stock dividend marked at the last close. Names with no visible distribution in
the trailing year are left unscored. This is not a dividend-yield rank.
"""

from datetime import timedelta

import pandas as pd

LOOKBACK_DAYS = 20
ACTION_COLUMNS = ["ts_code", "ex_date", "cash_per_share", "stock_per_share"]
DAILY_COLUMNS = ["ts_code", "trade_date", "close"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=365)).strftime("%Y%m%d")
    end = context.inference_at.strftime("%Y%m%d")
    actions = pd.read_parquet(context.asof_dir + "/corporate_actions", columns=ACTION_COLUMNS)
    actions = actions.assign(ex_date=actions["ex_date"].astype(str), ts_code=actions["ts_code"].astype(str))
    actions = actions[(actions["ex_date"] >= start) & (actions["ex_date"] < end)]
    if actions.empty:
        raise RuntimeError("no corporate action in the trailing year")
    actions = actions.sort_values("ex_date").drop_duplicates("ts_code", keep="last")
    bar_start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", bar_start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    close = pd.to_numeric(last["close"], errors="coerce")
    actions = actions.set_index("ts_code")
    cash = pd.to_numeric(actions["cash_per_share"], errors="coerce")
    stock = pd.to_numeric(actions["stock_per_share"], errors="coerce").fillna(0.0)
    marked = close.reindex(actions.index) * stock
    denom = cash.fillna(0.0) + marked.fillna(0.0)
    share = cash / denom.where(denom > 0)
    values = share.reindex(list(codes))
    return values, {"payers": int(share.notna().sum())}

def shuffle(values, decision_at):
    import numpy as np
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
