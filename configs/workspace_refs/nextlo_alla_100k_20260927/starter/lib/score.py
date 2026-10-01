"""Largest float caps just below the current CSI 1000.

Names inside CSI 300, CSI 500 or CSI 1000 are not scored. Names at or above
the smallest current CSI 1000 float cap are not scored. Among the rest, a
larger circ_mv ranks higher.
"""

from datetime import timedelta

import pandas as pd

NAME = "m1"
INDEXES = ("000300.SH", "000905.SH", "000852.SH")
FLOOR_INDEX = "000852.SH"
LOOKBACK_DAYS = 80
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
DAILY_COLUMNS = ["ts_code", "trade_date", "circ_mv"]


def _section(context, start, index):
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", index), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")) & frame["con_code"].notna()]
    frame = frame.assign(con_code=frame["con_code"].astype(str), trade_date=frame["trade_date"].astype(str))
    frame = frame[pd.to_numeric(frame["weight"], errors="coerce") > 0]
    if frame.empty:
        raise RuntimeError(f"no visible weights for {index}")
    latest = str(frame["trade_date"].max())
    return set(frame.loc[frame["trade_date"] == latest, "con_code"])


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    members = set()
    for index in INDEXES:
        members.update(_section(context, start, index))
    inside = _section(context, start, FLOOR_INDEX)
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS, filters=[("trade_date", ">=", start)])
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    mv = pd.to_numeric(last["circ_mv"], errors="coerce")
    floor_values = [float(mv[code]) for code in inside if code in mv.index and mv[code] == mv[code] and mv[code] > 0]
    if not floor_values:
        raise RuntimeError("no CSI 1000 member has a float cap")
    floor = min(floor_values)
    values = {}
    for code in codes:
        if code in members:
            continue
        size = mv.get(code, float("nan"))
        if size is None or not (size == size) or size <= 0 or size >= floor:
            continue
        values[code] = float(size)
    if len(values) < 12:
        raise RuntimeError("fewer than 12 names sit below the CSI 1000 float-cap floor")
    return pd.Series(values, dtype="float64").reindex(list(codes)), {"scored": len(values), "floor": floor}


def shuffle(values, decision_at):
    import numpy as np

    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
