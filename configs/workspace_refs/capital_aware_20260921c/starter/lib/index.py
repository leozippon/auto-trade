"""The as-of constituents of one index, read from the macro domain.

A book asked to hold its tracking error under a cap has to be built inside the
benchmark the cap is measured against: a book drawn from the whole market starts
from the gap between that pool and the CSI 300 and never gets under it.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), keyed on
`con_code`, not `ts_code`. Every row carries `available_at`, stamped 17:30 of
the cross-section's own trade date, so an 08:30 decision never sees the section
dated today. Between two month-end sections the membership a decision reads is
the earlier one.

`weight` is a PERCENT (4.64 = 4.64 %); a section sums to 100 within 0.05.

Missing `index_weight` is a hard failure. Silently scoring the whole market
under an index book's name would be a different strategy.
"""

from datetime import timedelta

import pandas as pd

INDEX_CODE = "000300.SH"
SECTION_LOOKBACK_DAYS = 120


def constituents(context):
    """(frame[ts_code, weight], the section's trade_date) visible at this decision.

    Raises when no section is visible: a book under a tracking mandate has no
    universe without one, and silently falling back to the whole market would
    be a different strategy wearing this one's name.
    """
    start = (context.inference_at - timedelta(days=SECTION_LOOKBACK_DAYS)).strftime("%Y%m%d")
    needed = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=needed,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_weight is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.dropna(subset=["weight"])
    if frame.empty:
        raise RuntimeError(f"no visible {INDEX_CODE} cross-section in the macro domain")
    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"].astype(str) == section].drop_duplicates("con_code", keep="last")
    return (rows[["con_code", "weight"]].rename(columns={"con_code": "ts_code"}).reset_index(drop=True),
            section)
