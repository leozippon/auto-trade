"""Lagged cheap ps_ttm among main-board names already in CSI 1000 then.

The index section is the latest one on or before the lagged session, not the
newest section. The score is minus ps_ttm on that lagged bar.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 140
LAG_SESSIONS = 21
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "ps_ttm"]


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
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    days = sorted(bars["trade_date"].unique())
    if len(days) <= LAG_SESSIONS:
        raise RuntimeError("not enough sessions to lag ps_ttm")
    lag_day = days[-1 - LAG_SESSIONS]
    sections = sorted(frame["trade_date"].unique())
    eligible = [day for day in sections if day <= lag_day]
    if not eligible:
        raise RuntimeError("no CSI 1000 section on or before the lagged session")
    section = eligible[-1]
    current = set(frame.loc[frame["trade_date"] == section, "con_code"])
    lagged = bars[bars["trade_date"] <= lag_day]
    last = lagged.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    ps = pd.to_numeric(last["ps_ttm"], errors="coerce")
    values = {}
    for code in current:
        multiple = ps.get(code)
        if multiple is None or not (multiple == multiple) or multiple <= 0:
            continue
        if str(code).startswith(("300", "301")):
            continue
        values[code] = -float(multiple)
    scored = pd.Series(values, dtype="float64").reindex(list(codes))
    if int(scored.notna().sum()) < 12:
        raise RuntimeError("fewer than 12 lagged-section CSI 1000 members have lagged ps_ttm")
    return scored, {"scored": int(scored.notna().sum()), "section": section, "lag_session": lag_day}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
