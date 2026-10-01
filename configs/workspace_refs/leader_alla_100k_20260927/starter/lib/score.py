"""One name's share of its Shenwan industry's traded amount on the last session.

The score is that share. It is not a return, a dispersion, or an industry index move.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 10
UNI_COLUMNS = ["ts_code", "l1_code"]
DAILY_COLUMNS = ["ts_code", "trade_date", "amount"]


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
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    bars = bars.assign(amount=pd.to_numeric(bars["amount"], errors="coerce")).dropna(subset=["amount"])
    if bars.empty:
        raise RuntimeError("no recent amount rows")
    last_day = bars["trade_date"].max()
    last = bars[bars["trade_date"] == last_day].drop_duplicates("ts_code").set_index("ts_code")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    last["l1"] = info["l1_code"].map(_norm).reindex(last.index)
    last = last[last["l1"].astype(str).str.len() > 0]
    total = last.groupby("l1")["amount"].transform("sum")
    last["value"] = last["amount"] / total.replace(0, np.nan)
    values = last["value"].reindex(list(codes))
    return values, {"industries": int(last["l1"].nunique()), "last_bar": last_day}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    ranks = (-values.reindex(book)).dropna()
    return {"book_score_span": [float(ranks.min()), float(ranks.max())] if len(ranks) else None}
