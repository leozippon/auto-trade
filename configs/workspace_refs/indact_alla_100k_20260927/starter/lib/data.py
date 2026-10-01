"""The decision day's pool and the prices the book is sized at.

The pool is every non-Beijing, non-STAR A-share with a fresh bar, not an index
section. `daily` has no `available_at` column: the newest row at an 08:30
decision is the previous trading day (T-1). Held names are priced too, so a
forced exit's proceeds are counted.
"""

from datetime import timedelta

import pandas as pd

DAILY_COLUMNS = ["ts_code", "trade_date", "close", "is_suspended"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
BAR_LOOKBACK_DAYS = 20


def pool(context, held):
    """(frame indexed by ts_code: close, member, tradable, weight, industry; T-1 date; section date)."""

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
    frame = pd.DataFrame(index=pd.Index(codes, name="ts_code"))
    frame["close"] = pd.to_numeric(last["close"], errors="coerce").reindex(codes)
    frame["member"] = True
    frame["weight"] = 0.0
    frame["industry"] = info["l1_name"].fillna("未分类").astype(str)
    fresh = last["trade_date"].reindex(codes).eq(last_day) & ~last["is_suspended"].reindex(codes).eq(True)
    name = info["name"].fillna("").astype(str)
    code = frame.index.to_series()
    frame["tradable"] = (
        fresh.fillna(False) & (frame["close"] > 0)
        & ~code.str.startswith(("688", "689")) & ~code.str.endswith(".BJ")
        & ~name.str.contains("ST|退")
    )
    return frame, last_day, last_day
