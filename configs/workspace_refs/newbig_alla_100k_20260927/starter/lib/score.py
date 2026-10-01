"""Newer CSI 500 members, larger ones first inside a tenure.

The score is minus the month-end section count, plus log circulating market
value on a smaller scale, so a newer member always outranks an older one and
size only orders names with the same count. Only the newest section is scored.
"""

from datetime import timedelta
from math import log

import pandas as pd

NAME = "m1"
INDEX = "000905.SH"
LOOKBACK_DAYS = 400
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "circ_mv"]


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
        raise RuntimeError("no visible CSI 500 weights")
    sections = frame.drop_duplicates(["con_code", "trade_date"])
    latest = str(sections["trade_date"].max())
    current = set(sections.loc[sections["trade_date"] == latest, "con_code"])
    count = sections.groupby("con_code")["trade_date"].nunique()
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    size = pd.to_numeric(last["circ_mv"], errors="coerce")
    values = {}
    for code in current:
        seen = count.get(code)
        mv = size.get(code)
        if seen is None or not (mv == mv) or mv <= 0:
            continue
        values[code] = -float(seen) * 1000.0 + log(float(mv))
    scored = pd.Series(values, dtype="float64").reindex(list(codes))
    if int(scored.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 500 members are scored")
    return scored, {"scored": int(scored.notna().sum()), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
