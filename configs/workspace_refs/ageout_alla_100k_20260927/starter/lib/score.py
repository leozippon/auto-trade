"""Older listings outside the major indexes.

Names absent from the latest CSI 300, CSI 500 and CSI 1000 sections. The score
is minus the listing date, so an earlier listing ranks higher. Size is not used.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEXES = ("000300.SH", "000905.SH", "000852.SH")
LOOKBACK_DAYS = 80
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
UNI_COLUMNS = ["ts_code", "list_date"]


def _members(context, index, start):
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", index), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")) & frame["con_code"].notna()]
    frame = frame.assign(con_code=frame["con_code"].astype(str), trade_date=frame["trade_date"].astype(str))
    frame = frame[pd.to_numeric(frame["weight"], errors="coerce") > 0]
    if frame.empty:
        raise RuntimeError(f"no visible {index} weights")
    latest = str(frame["trade_date"].max())
    return set(frame.loc[frame["trade_date"] == latest, "con_code"])


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    inside = set()
    for index in INDEXES:
        inside |= _members(context, index, start)
    day = pd.Timestamp(context.inference_at).tz_convert("Asia/Shanghai").strftime("%Y%m%d")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code")
    info = info.assign(ts_code=info["ts_code"].astype(str)).set_index("ts_code")
    values = {}
    for code, listed_raw in info["list_date"].items():
        if code in inside:
            continue
        listed = "" if listed_raw is None or (isinstance(listed_raw, float) and listed_raw != listed_raw) else str(listed_raw).strip()
        if len(listed) != 8 or not listed.isdigit() or listed > day:
            continue
        values[code] = -float(listed)
    scored = pd.Series(values, dtype="float64").reindex(list(codes))
    if int(scored.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 names outside the major indexes have a listing date")
    return scored, {"scored": int(scored.notna().sum())}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
