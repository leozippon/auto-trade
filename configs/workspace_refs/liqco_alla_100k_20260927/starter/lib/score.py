"""Return correlation with changes in CSI 1000 turnover.

Current CSI 1000 members only. The score is the correlation of the stock's
daily return with the daily change in the index turnover rate, over about
sixty sessions. A higher correlation ranks higher. It is not a beta on the
index return, and it is not the stock's own turnover.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEX = "000852.SH"
LOOKBACK_DAYS = 100
MIN_DAYS = 30
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "pct_chg"]
TURN_COLUMNS = ["dataset", "ts_code", "trade_date", "turnover_rate"]


def _corr(left, right):
    paired = pd.DataFrame({"y": left, "x": right}).dropna()
    if len(paired) < MIN_DAYS:
        return None
    y = paired["y"] - paired["y"].mean()
    x = paired["x"] - paired["x"].mean()
    denom = float((x * x).sum() * (y * y).sum()) ** 0.5
    if denom <= 0:
        return None
    return float((x * y).sum()) / denom


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
    members = set(frame.loc[frame["trade_date"] == latest, "con_code"])
    turn = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=TURN_COLUMNS,
        filters=[("dataset", "=", "index_dailybasic"), ("ts_code", "=", INDEX), ("trade_date", ">=", start)],
    )
    series = pd.to_numeric(
        turn.assign(trade_date=turn["trade_date"].astype(str)).drop_duplicates("trade_date").set_index("trade_date")["turnover_rate"],
        errors="coerce",
    ).sort_index().diff()
    if series.dropna().empty:
        raise RuntimeError("no visible CSI 1000 turnover changes")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(
        trade_date=bars["trade_date"].astype(str),
        ts_code=bars["ts_code"].astype(str),
        ret=pd.to_numeric(bars["pct_chg"], errors="coerce"),
    )
    bars = bars[bars["ts_code"].isin(members)]
    values = {}
    for code, group in bars.groupby("ts_code"):
        corr = _corr(group["ret"], group["trade_date"].map(series))
        if corr is None:
            continue
        values[code] = corr
    if len(values) < 12:
        raise RuntimeError("fewer than 12 CSI 1000 names have a turnover correlation")
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
