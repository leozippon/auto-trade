"""Change in a CSI 500 name's share of its industry's index weight.

Current members of industries with at least four names. The score is this
section's industry weight share minus the previous section's share. A name
absent from the previous section has previous share 0. The level is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000905.SH"
LOOKBACK_DAYS = 140
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


def _shares(section, info):
    current = section.drop_duplicates("con_code").set_index("con_code")
    current["l1"] = info["l1_code"].map(_norm).reindex(current.index)
    current = current[current["l1"].astype(str).str.len() > 0]
    size = current.groupby("l1")["weight"].transform("size")
    current = current[size >= 4]
    total = current.groupby("l1")["weight"].transform("sum")
    return current["weight"] / total


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
    if len(days) < 2:
        raise RuntimeError("need two CSI 500 weight sections")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    now = _shares(frame[frame["trade_date"] == days[-1]], info)
    prev = _shares(frame[frame["trade_date"] == days[-2]], info)
    values = (now - prev.reindex(now.index).fillna(0.0)).reindex(list(codes))
    if int(values.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 500 members are scored")
    return values, {"scored": int(values.notna().sum()), "section": days[-1], "previous": days[-2]}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
