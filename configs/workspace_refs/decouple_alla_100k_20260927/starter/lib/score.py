"""Minus the correlation of a stock with its Shenwan level-1 index.

High scores are names whose daily returns did not track their industry over
the last 21 sessions. This is not the return gap versus the industry.
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


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    sw = pd.read_parquet(context.asof_dir + "/macro", columns=SW_COLUMNS,
                         filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)])
    stamp = pd.to_datetime(sw["available_at"], utc=True)
    sw = sw[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    sw = sw.assign(ts_code=sw["ts_code"].map(_norm), trade_date=sw["trade_date"].astype(str))
    sw = sw[sw["ts_code"].str.fullmatch(r"801\d\d0\.SI")]
    sw = sw.assign(close=pd.to_numeric(sw["close"], errors="coerce")).dropna(subset=["close"])
    sw = sw.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")
    wide = sw.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    if len(wide) < 22:
        raise RuntimeError("not enough Shenwan sessions")
    wide = wide.iloc[-22:]
    ind_ret = wide.pct_change().iloc[1:]
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    bars = bars[bars["close"] > 0]
    prices = bars.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    prices = prices.reindex(wide.index).iloc[-22:]
    stock_ret = prices.pct_change().iloc[1:]
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_code"].map(_norm)
    values = {}
    for code in codes:
        ind = industry.get(code, "")
        if ind not in ind_ret.columns or code not in stock_ret.columns:
            values[code] = float("nan")
            continue
        pair = pd.concat([stock_ret[code], ind_ret[ind]], axis=1).dropna()
        if len(pair) < 15 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
            values[code] = float("nan")
            continue
        values[code] = -float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return pd.Series(values, dtype="float64"), {"sessions": int(len(stock_ret))}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    ranks = (-values.reindex(book)).dropna()
    return {"book_score_span": [float(ranks.min()), float(ranks.max())] if len(ranks) else None}
