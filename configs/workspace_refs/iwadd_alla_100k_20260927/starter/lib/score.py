"""Change in a Shenwan industry's total CSI 1000 weight.

Current CSI 1000 members only. The score is the industry's weight sum on the
newest section minus the sum about six months earlier. A larger increase ranks
higher. Names in one industry share the score. The stock's own weight is not
the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 220
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
    frame = frame.assign(
        con_code=frame["con_code"].astype(str),
        trade_date=frame["trade_date"].astype(str),
        weight=pd.to_numeric(frame["weight"], errors="coerce"),
    )
    frame = frame[frame["weight"] > 0]
    if frame.empty:
        raise RuntimeError("no visible CSI 1000 weights")
    dates = sorted(frame["trade_date"].unique())
    latest = dates[-1]
    target = (pd.Timestamp(latest) - pd.Timedelta(days=180)).strftime("%Y%m%d")
    older = [day for day in dates if day <= target]
    old = older[-1] if older else dates[0]
    if old == latest:
        raise RuntimeError("no earlier CSI 1000 section")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").assign(ts_code=lambda raw: raw["ts_code"].astype(str)).set_index("ts_code")
    home = info["l1_code"].map(_norm)
    frame = frame[frame["trade_date"].isin((old, latest))].copy()
    frame["industry"] = frame["con_code"].map(home)
    frame = frame[frame["industry"].astype(str).str.len() > 0]
    sums = frame.groupby(["trade_date", "industry"])["weight"].sum()
    current = set(frame.loc[frame["trade_date"] == latest, "con_code"])
    values = {}
    for code in current:
        industry = home.get(code, "")
        if not industry:
            continue
        now = sums.get((latest, industry), 0.0)
        then = sums.get((old, industry), 0.0)
        values[code] = float(now) - float(then)
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have an industry-weight change")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {
        "scored": len(values),
        "section": latest,
        "prior_section": old,
    }


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
