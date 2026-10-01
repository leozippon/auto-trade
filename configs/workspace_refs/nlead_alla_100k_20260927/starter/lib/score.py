"""How often a CSI 500 name was its industry's heaviest weight.

The last six month-end sections. Count dominates. The latest industry weight
share only orders names with the same count. Industries with fewer than four
names are ignored on that section.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000905.SH"
LOOKBACK_DAYS = 220
SECTIONS = 6
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


def _section(frame, day, info):
    current = frame[frame["trade_date"] == day].drop_duplicates("con_code").set_index("con_code")
    current["l1"] = info["l1_code"].map(_norm).reindex(current.index)
    current = current[current["l1"].astype(str).str.len() > 0]
    size = current.groupby("l1")["weight"].transform("size")
    current = current[size >= 4]
    if current.empty:
        return current
    total = current.groupby("l1")["weight"].transform("sum")
    current = current.assign(share=current["weight"] / total)
    return current


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
    days = sorted(frame["trade_date"].unique())
    use = days[-SECTIONS:]
    if len(use) < 2:
        raise RuntimeError("need at least two CSI 500 weight sections")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    counts = {}
    for day in use:
        current = _section(frame, day, info)
        if current.empty:
            continue
        for code in current.groupby("l1")["share"].idxmax():
            counts[code] = counts.get(code, 0) + 1
    latest = _section(frame, use[-1], info)
    if latest.empty:
        raise RuntimeError("the latest CSI 500 section has no scored industry")
    values = {}
    for code, row in latest.iterrows():
        values[code] = float(counts.get(code, 0)) * 1000.0 + float(row["share"])
    scored = pd.Series(values, dtype="float64").reindex(list(codes))
    if int(scored.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 500 members are scored")
    return scored, {"scored": int(scored.notna().sum()), "section": use[-1], "sections": len(use)}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
