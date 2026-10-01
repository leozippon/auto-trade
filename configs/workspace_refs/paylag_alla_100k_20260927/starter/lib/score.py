"""How quickly cash from the latest dividend is paid after the ex-date.

Shorter gaps score higher. Names without a cash dividend in the trailing year
are left unscored. The cash amount itself is not the score.
"""

from datetime import timedelta

import pandas as pd

ACTION_COLUMNS = ["ts_code", "ex_date", "pay_date", "cash_per_share"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=365)).strftime("%Y%m%d")
    end = context.inference_at.strftime("%Y%m%d")
    actions = pd.read_parquet(context.asof_dir + "/corporate_actions", columns=ACTION_COLUMNS)
    actions = actions.assign(
        ex_date=actions["ex_date"].astype(str),
        pay_date=actions["pay_date"].astype(str),
        ts_code=actions["ts_code"].astype(str),
    )
    actions = actions[(actions["ex_date"] >= start) & (actions["ex_date"] < end) & (actions["cash_per_share"] > 0)]
    actions = actions[actions["pay_date"].str.fullmatch(r"\d{8}") & actions["ex_date"].str.fullmatch(r"\d{8}")]
    if actions.empty:
        raise RuntimeError("no cash dividend with both dates in the trailing year")
    actions = actions.sort_values("ex_date").drop_duplicates("ts_code", keep="last")
    ex = pd.to_datetime(actions["ex_date"], format="%Y%m%d")
    pay = pd.to_datetime(actions["pay_date"], format="%Y%m%d")
    lag = (pay - ex).dt.days
    actions = actions.assign(lag=lag.to_numpy())
    actions = actions[actions["lag"] >= 0]
    values = (-actions.set_index("ts_code")["lag"]).reindex(list(codes))
    return values, {"payers": int(values.notna().sum())}

def shuffle(values, decision_at):
    import numpy as np
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
