"""How stable a name's CSI 1000 weight has been.

The score is minus the standard deviation of its month-end weights over the
visible year. Steadier weights score higher. Names without four sections are
left unscored. This is not the latest change in weight.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 400
MIN_SECTIONS = 4
COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")) & frame["con_code"].notna()]
    frame = frame.assign(
        con_code=frame["con_code"].astype(str),
        trade_date=frame["trade_date"].astype(str),
        weight=pd.to_numeric(frame["weight"], errors="coerce"),
    )
    frame = frame[frame["weight"] > 0]
    if frame.empty:
        raise RuntimeError("no visible CSI 1000 weights")
    sections = frame.drop_duplicates(["con_code", "trade_date"])
    grouped = sections.groupby("con_code")["weight"]
    std = grouped.std()
    count = grouped.count()
    values = (-std.where(count >= MIN_SECTIONS)).reindex(list(codes))
    return values, {"scored": int(values.notna().sum())}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
