"""A name's 21-session return minus its Shenwan level-1 return.

High scores are names that led their own industry. The industry move itself
is removed. At most two names per industry are held by the book.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 50
UNI_COLUMNS = ["ts_code", "l1_code"]
DAILY_COLUMNS = ["ts_code", "trade_date", "close"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "close"]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    if text.endswith(".SI"):
        return text
    if text.isdigit():
        return text + ".SI"
    return text


def _industry_return(context):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(context.asof_dir + "/macro", columns=SW_COLUMNS,
                            filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)])
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.assign(ts_code=frame["ts_code"].map(_norm), trade_date=frame["trade_date"].astype(str))
    frame = frame[frame["ts_code"].str.fullmatch(r"801\d\d0\.SI")]
    frame = frame.assign(close=pd.to_numeric(frame["close"], errors="coerce")).dropna(subset=["close"])
    frame = frame.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")
    out = {}
    for code, group in frame.groupby("ts_code"):
        closes = group["close"].to_numpy()
        if len(closes) >= 21 and closes[-21] > 0:
            out[code] = float(closes[-1] / closes[-21] - 1.0)
    if not out:
        raise RuntimeError("no Shenwan industry has 21 closes")
    return out


def score(context, codes):
    industry_return = _industry_return(context)
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    bars = bars[bars["close"] > 0]
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < 22:
        raise RuntimeError("not enough sessions for a 21-session stock return")
    use = dates[-22:]
    last = bars[bars["trade_date"] == use[-1]].drop_duplicates("ts_code").set_index("ts_code")
    first = bars[bars["trade_date"] == use[0]].drop_duplicates("ts_code").set_index("ts_code")
    both = last[["close"]].join(first[["close"]].rename(columns={"close": "close0"}), how="inner")
    both["ret"] = both["close"] / both["close0"] - 1.0
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    both["l1"] = info["l1_code"].map(_norm).reindex(both.index)
    both["ind"] = both["l1"].map(industry_return)
    both = both[both["ind"].notna()]
    both["value"] = both["ret"] - both["ind"]
    values = both["value"].reindex(list(codes))
    return values, {"industries": int(both["l1"].nunique())}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    ranks = (-values.reindex(book)).dropna()
    return {"book_score_span": [float(ranks.min()), float(ranks.max())] if len(ranks) else None}
