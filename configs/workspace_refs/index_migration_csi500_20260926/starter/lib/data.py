"""The decision day's pool and the prices the book is sized at.

`daily` has no `available_at` column: the as-of view holds exactly the rows
visible at the decision, so the newest row at an 08:30 decision is the previous
trading day (T-1). The pool is the newest visible `knobs.INDEX` section minus
what a 100-share-lot account cannot or should not trade there: STAR (688/689,
200-share lots), Beijing (.BJ), names carrying ST or 退, and names with no bar
or a suspension on T-1. Held names are priced too, so a forced exit's proceeds
are counted.
"""

from datetime import timedelta

import pandas as pd

from lib import index, knobs

DAILY_COLUMNS = ["ts_code", "trade_date", "close", "is_suspended"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
BAR_LOOKBACK_DAYS = 20


def pool(context, held):
    """(frame indexed by ts_code: close, member, tradable, weight, industry; T-1 date; section date)."""

    weights, section = index.latest(context, knobs.INDEX)
    codes = sorted(set(weights) | set(held))
    start = (context.inference_at - timedelta(days=BAR_LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS,
                           filters=[("trade_date", ">=", start), ("ts_code", "in", codes)])
    missing = [name for name in DAILY_COLUMNS if name not in bars.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    if bars.empty:
        raise RuntimeError("the as-of view holds no recent bar for the section")
    bars = bars.assign(trade_date=bars["trade_date"].astype(str))
    last_day = str(bars["trade_date"].max())
    last = bars.sort_values("trade_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(codes)
    frame = pd.DataFrame(index=pd.Index(codes, name="ts_code"))
    frame["close"] = pd.to_numeric(last["close"], errors="coerce").reindex(codes)
    frame["member"] = frame.index.isin(list(weights))
    frame["weight"] = pd.Series(weights).reindex(codes).fillna(0.0)
    frame["industry"] = info["l1_name"].fillna("未分类").astype(str)
    code = frame.index.to_series()
    fresh = last["trade_date"].reindex(codes).eq(last_day) & ~last["is_suspended"].reindex(codes).eq(True)
    frame["tradable"] = (
        frame["member"] & fresh & (frame["close"] > 0)
        & ~code.str.startswith(("688", "689")) & ~code.str.endswith(".BJ")
        & ~info["name"].fillna("").astype(str).str.contains("ST|退")
    )
    return frame, last_day, section
