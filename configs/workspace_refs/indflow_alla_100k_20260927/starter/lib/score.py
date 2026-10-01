"""Change in Shenwan level-1 traded amount over 21 sessions.

Every name in an industry shares the industry's amount growth. This is not
the industry price return. At most two names per industry are held.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 50
UNI_COLUMNS = ["ts_code", "l1_code"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "amount"]


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
    frame = pd.read_parquet(context.asof_dir + "/macro", columns=SW_COLUMNS,
                            filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", start)])
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.assign(ts_code=frame["ts_code"].map(_norm), trade_date=frame["trade_date"].astype(str))
    frame = frame[frame["ts_code"].str.fullmatch(r"801\d\d0\.SI")]
    frame = frame.assign(amount=pd.to_numeric(frame["amount"], errors="coerce")).dropna(subset=["amount"])
    frame = frame.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")
    growth = {}
    for code, group in frame.groupby("ts_code"):
        amounts = group["amount"].to_numpy()
        if len(amounts) >= 21 and amounts[-21] > 0:
            growth[code] = float(amounts[-1] / amounts[-21] - 1.0)
    if not growth:
        raise RuntimeError("no Shenwan industry has 21 amount rows")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    industry = info["l1_code"].map(_norm)
    values = pd.Series({code: growth.get(industry.get(code, ""), float("nan")) for code in codes}, dtype="float64")
    return values, {"industries": len(growth)}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    ranks = (-values.reindex(book)).dropna()
    return {"book_score_span": [float(ranks.min()), float(ranks.max())] if len(ranks) else None}
