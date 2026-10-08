"""Constituent sections from the MACRO domain, point in time.

`index_weight` carries one section per index per month, dated the month's last
trading day and stamped 17:30 on it, so an 08:30 decision reads the newest
section dated BEFORE its own day; nothing here ever reads a later one. On these
rows the index is in `index_code` and the constituent in `con_code` (`ts_code`
is empty) -- the opposite of `index_daily`, which this package does not read.

Every row is filtered by its own `available_at`; a decision with no visible
section fails instead of trading a different universe.
"""

from datetime import timedelta

import pandas as pd

COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
LOOKBACK_DAYS = 70


def latest(context, code):
    """({constituent: weight} of the newest visible section of `code`, that section's date)."""

    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", code), ("trade_date", ">=", start)],
    )
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_weight is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC"))
                  & frame["con_code"].notna() & (frame["weight"] > 0)]
    if frame.empty:
        raise RuntimeError(f"no visible {code} section in the macro domain")
    dates = frame["trade_date"].astype(str)
    section = str(dates.max())
    rows = frame[dates == section]
    return {str(c): float(w) for c, w in zip(rows["con_code"], rows["weight"])}, section
