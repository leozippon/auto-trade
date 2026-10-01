"""Industry median asset turnover inside the current CSI 1000.

The score is the industry median of pb / ps. Names in one industry share it.
A higher median ranks higher. The stock's own ratio is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 40
MIN_NAMES = 8
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "pb", "ps"]
UNI_COLUMNS = ["ts_code", "l1_name"]


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
        raise RuntimeError("no visible CSI 1000 weights")
    latest = str(frame["trade_date"].max())
    current = set(frame.loc[frame["trade_date"] == latest, "con_code"])
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    book = pd.to_numeric(last["pb"], errors="coerce")
    sales = pd.to_numeric(last["ps"], errors="coerce")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").assign(ts_code=lambda raw: raw["ts_code"].astype(str)).set_index("ts_code")
    home = info["l1_name"].fillna("").astype(str)
    ratios = {}
    for code in current:
        industry = home.get(code, "")
        if not str(industry).strip():
            continue
        pb = book.get(code, float("nan"))
        ps = sales.get(code, float("nan"))
        if pb is None or ps is None or not (pb == pb) or not (ps == ps) or pb <= 0 or ps <= 0:
            continue
        ratios.setdefault(str(industry), []).append(float(pb) / float(ps))
    medians = {name: float(pd.Series(rows).median()) for name, rows in ratios.items() if len(rows) >= MIN_NAMES}
    values = {}
    for code in current:
        industry = str(home.get(code, "")).strip()
        if industry in medians:
            values[code] = medians[industry]
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have an industry asset-turnover median")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values), "industries": len(medians), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
