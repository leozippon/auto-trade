"""Names that just left CSI 500, larger ones first.

A name in an earlier month-end section but not the newest one was removed.
Removal one section ago outranks removal two sections ago. Size only orders
names removed at the same time. Current members are not scored.
"""

from datetime import timedelta
from math import log

import pandas as pd

NAME = "m1"
INDEX = "000905.SH"
LOOKBACK_DAYS = 400
MAX_AGO = 2
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
    dates = sorted(sections["trade_date"].unique())
    if len(dates) < 2:
        raise RuntimeError("CSI 500 has fewer than two sections")
    current = set(sections.loc[sections["trade_date"] == dates[-1], "con_code"])
    removed: dict[str, int] = {}
    for ago, day in enumerate(reversed(dates[:-1]), start=1):
        if ago > MAX_AGO:
            break
        for code in sections.loc[sections["trade_date"] == day, "con_code"]:
            if code not in current and code not in removed:
                removed[code] = ago
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    size = pd.to_numeric(last["circ_mv"], errors="coerce")
    values = {}
    for code, ago in removed.items():
        mv = size.get(code)
        if not (mv == mv) or mv <= 0:
            continue
        values[code] = -float(ago) * 1000.0 + log(float(mv))
    scored = pd.Series(values, dtype="float64").reindex(list(codes))
    if int(scored.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 recent CSI 500 deletions are scored")
    return scored, {"scored": int(scored.notna().sum()), "section": dates[-1]}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
