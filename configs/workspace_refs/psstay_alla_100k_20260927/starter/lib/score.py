"""How often a CSI 1000 name was in the cheap fifth of ps_ttm.

The last six month-end sections. Count dominates. The newest ps_ttm only
orders names with the same count. The newest level alone is not the score.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 220
SECTIONS = 6
CHEAP_SHARE = 0.2
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "ps_ttm"]


def _cheap(bars, day, members):
    day_bars = bars[bars["trade_date"] <= day].sort_values("trade_date").drop_duplicates("ts_code", keep="last")
    ps = pd.to_numeric(day_bars.set_index("ts_code")["ps_ttm"], errors="coerce")
    kept = ps.reindex(list(members))
    kept = kept[kept > 0]
    if len(kept) < 12:
        return set()
    cut = float(kept.quantile(CHEAP_SHARE))
    return set(kept.index[kept <= cut])


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
    days = sorted(frame["trade_date"].unique())
    use = days[-SECTIONS:]
    if len(use) < 2:
        raise RuntimeError("need at least two CSI 1000 weight sections")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    counts = {}
    for day in use:
        members = set(frame.loc[frame["trade_date"] == day, "con_code"])
        for code in _cheap(bars, day, members):
            counts[code] = counts.get(code, 0) + 1
    current_members = set(frame.loc[frame["trade_date"] == use[-1], "con_code"])
    current_ps = _cheap(bars, use[-1], current_members)
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    ps = pd.to_numeric(last["ps_ttm"], errors="coerce")
    values = {}
    for code in current_members:
        multiple = ps.get(code)
        if multiple is None or not (multiple == multiple) or multiple <= 0:
            continue
        values[code] = float(counts.get(code, 0)) * 1000.0 - float(multiple)
    scored = pd.Series(values, dtype="float64").reindex(list(codes))
    if int(scored.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 current CSI 1000 members have ps_ttm")
    return scored, {"scored": int(scored.notna().sum()), "section": use[-1], "cheap_now": len(current_ps)}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
