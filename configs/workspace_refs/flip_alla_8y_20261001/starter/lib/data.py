"""The decision day's pool, point in time.

`daily` has no `available_at` column: the newest row at an 08:30 decision is
the previous trading day (T-1). Every read projects its columns and bounds its
dates; a full-span replay's as-of `daily` grows from 2010 to the decision day.

The pool is every non-Beijing, non-STAR A-share with a fresh T-1 bar whose
name in the replay year's `universe` carries no ST / 退 marker; with
`knobs.POOL == "csi1000"` it is further cut to the newest visible INDEX
section. `universe` is one table per replay year, anchored the day before the
year starts: names, ST status and Shenwan L1 labels do not move inside a year,
and a name listed inside the year has no row (empty name, industry "未分类").
Held names are priced too, so a forced exit's proceeds are counted.
"""

from datetime import timedelta

import pandas as pd

from lib import index, knobs

DAILY_COLUMNS = ["ts_code", "trade_date", "close", "is_suspended"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
BAR_LOOKBACK_DAYS = 20
POOLS = ("alla", "csi1000")


def pool(context, held):
    """(frame indexed by ts_code: close, member, tradable, weight, industry; T-1 date; section date)."""

    if knobs.POOL not in POOLS:
        raise ValueError(f"knobs.POOL must be one of {POOLS}, got {knobs.POOL!r}")
    start = (context.inference_at - timedelta(days=BAR_LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    missing = [name for name in DAILY_COLUMNS if name not in bars.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    if bars.empty:
        raise RuntimeError("the as-of view holds no recent bar")
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    last_day = str(bars["trade_date"].max())
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    codes = sorted(set(last.index) | {str(code) for code in held})
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(codes)
    weights, section = index.latest(context, knobs.INDEX)
    frame = pd.DataFrame(index=pd.Index(codes, name="ts_code"))
    frame["close"] = pd.to_numeric(last["close"], errors="coerce").reindex(codes)
    frame["weight"] = pd.Series(weights, dtype="float64").reindex(codes).fillna(0.0)
    frame["member"] = frame["weight"] > 0
    frame["industry"] = info["l1_name"].fillna("未分类").astype(str)
    fresh = last["trade_date"].reindex(codes).eq(last_day) & ~last["is_suspended"].reindex(codes).eq(True)
    name = info["name"].fillna("").astype(str)
    code = frame.index.to_series()
    frame["tradable"] = (
        fresh.fillna(False) & (frame["close"] > 0)
        & ~code.str.startswith(("688", "689")) & ~code.str.endswith(".BJ")
        & ~name.str.contains("ST|退")
    )
    if knobs.POOL == "csi1000":
        frame["tradable"] = frame["tradable"] & frame["member"]
    return frame, last_day, section
