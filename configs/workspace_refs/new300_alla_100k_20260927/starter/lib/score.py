"""How newly a name joined CSI 300.

The score is minus the number of month-end sections it appears in over the
visible year, and only names in the newest section are scored. Fewer sections
means a newer member and a higher score. This is not the change in weight.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000300.SH"
LOOKBACK_DAYS = 400
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
    frame = frame.assign(con_code=frame["con_code"].astype(str), trade_date=frame["trade_date"].astype(str))
    frame = frame[pd.to_numeric(frame["weight"], errors="coerce") > 0]
    if frame.empty:
        raise RuntimeError("no visible CSI 300 weights")
    sections = frame.drop_duplicates(["con_code", "trade_date"])
    latest = str(sections["trade_date"].max())
    current = set(sections.loc[sections["trade_date"] == latest, "con_code"])
    count = sections.groupby("con_code")["trade_date"].nunique()
    values = (-count.reindex(list(current))).reindex(list(codes))
    if int(values.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 300 members are scored")
    return values, {"scored": int(values.notna().sum()), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
