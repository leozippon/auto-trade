"""CSI 500 weight not explained by size, inside the industry.

Current members of industries with at least four scored names. The score is
the residual of index weight on log circulating market value, fit inside the
industry. Raw weight and raw size are not the score.
"""

from datetime import timedelta

import numpy as np
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
    current["mv"] = size.reindex(current.index)
    current["l1"] = info["l1_code"].map(_norm).reindex(current.index)
    current = current[(current["mv"] > 0) & current["l1"].astype(str).str.len().gt(0)]
    parts = []
    for _, group in current.groupby("l1"):
        if len(group) < 4:
            continue
        x = np.log(group["mv"].to_numpy(dtype="float64"))
        y = group["weight"].to_numpy(dtype="float64")
        if np.unique(np.round(x, 6)).size < 2:
            continue
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            continue
        slope = float(np.dot(xc, y - y.mean()) / denom)
        fitted = y.mean() + slope * xc
        parts.append(pd.Series(y - fitted, index=group.index))
    if not parts:
        raise RuntimeError("no CSI 500 industry has four scored names")
    values = pd.concat(parts).reindex(list(codes))
    if int(values.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 500 members are scored")
    return values, {"scored": int(values.notna().sum()), "section": latest}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
