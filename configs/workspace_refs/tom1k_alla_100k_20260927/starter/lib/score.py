"""Who rose on month turns, inside the current CSI 1000.

The score is the mean daily return on the first three and last three trading
days of each month, over about one year. A higher mean ranks higher. The book
stays on the monthly review; this is not a calendar overlay.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 400
MIN_DAYS = 30
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg"]


def _turn_days(days):
    by_month = {}
    for day in days:
        by_month.setdefault(day[:6], []).append(day)
    chosen = set()
    for month_days in by_month.values():
        ordered = sorted(month_days)
        chosen.update(ordered[:3])
        chosen.update(ordered[-3:])
    return chosen


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    weight_start = (context.inference_at - timedelta(days=80)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX), ("trade_date", ">=", weight_start)],
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
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
    )
    turns = _turn_days(sorted(bars["trade_date"].unique()))
    use = bars[bars["ts_code"].isin(current) & bars["trade_date"].isin(turns) & bars["ret"].notna()]
    values = {}
    for code, group in use.groupby("ts_code"):
        if len(group) < MIN_DAYS:
            continue
        values[code] = float(group["ret"].mean())
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have a month-turn return")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values), "section": latest}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
