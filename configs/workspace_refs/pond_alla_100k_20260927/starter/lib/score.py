"""CSI 500 names in the thinnest industries, larger ones first.

Current members only. Fewer fellow CSI 500 names in the same Shenwan industry
rank higher. Circulating market value only orders names with the same count.
Industry weight share is not the score.
"""

from datetime import timedelta
from math import log

import pandas as pd

NAME = "m1"
INDEX = "000905.SH"
LOOKBACK_DAYS = 80
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "circ_mv"]
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
        raise RuntimeError("no visible CSI 500 weights")
    latest = str(frame["trade_date"].max())
    current = frame[frame["trade_date"] == latest].drop_duplicates("con_code").set_index("con_code")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    size = pd.to_numeric(last["circ_mv"], errors="coerce")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    current["l1"] = info["l1_code"].map(_norm).reindex(current.index)
    current["mv"] = size.reindex(current.index)
    current = current[current["l1"].astype(str).str.len() > 0]
    current = current[(current["mv"] > 0)]
    counts = current.groupby("l1")["mv"].transform("size")
    current["value"] = -counts.astype(float) * 1000.0 + current["mv"].map(log)
    values = current["value"].reindex(list(codes))
    if int(values.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 500 members are scored")
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
