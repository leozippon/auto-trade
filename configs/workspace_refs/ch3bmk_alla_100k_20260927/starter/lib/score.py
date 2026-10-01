"""A CSI 300 name's share of its Shenwan industry's index weight.

Current members only. The score is this name's weight divided by the total
weight of current CSI 300 members in the same industry. Raw weight is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000300.SH"
LOOKBACK_DAYS = 80
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
UNI_COLUMNS = ["ts_code", "l1_code"]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    if text.endswith(".SI"):
        return text
    if text.isdigit():
        return text + ".SI"
    return text


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
        raise RuntimeError("no visible CSI 300 weights")
    latest = str(frame["trade_date"].max())
    current = frame[frame["trade_date"] == latest].drop_duplicates("con_code").set_index("con_code")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    current["l1"] = info["l1_code"].map(_norm).reindex(current.index)
    current = current[current["l1"].astype(str).str.len() > 0]
    size = current.groupby("l1")["weight"].transform("size")
    current = current[size >= 4]
    total = current.groupby("l1")["weight"].transform("sum")
    current["value"] = current["weight"] / total
    values = current["value"].reindex(list(codes))
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
