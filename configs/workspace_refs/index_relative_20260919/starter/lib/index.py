"""The as-of constituents of one index, read from the macro domain.

This is the whole point of the pack: the book is built INSIDE the benchmark the
verdict measures it against, so its residual tracking error starts from the
index's own dispersion instead of from the gap between a small/mid pool and the
CSI 300.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), keyed on
`con_code`, not `ts_code`, so the research pool screens never touch it. Every
row carries `available_at`, stamped 17:30 of the cross-section's own trade date,
so an 08:30 decision never sees the section dated today and reads the newest one
dated strictly before it. Between two month-end sections the membership a
decision reads is the earlier one -- the caveat is in `pit-field-map.md` and it
is the price of the only membership face the lake has.

`weight` is a PERCENT (4.64 = 4.64 %); a section sums to 100 within 0.05. This
package uses it only to report how much of the index a book covers, never as a
portfolio weight, but anything that does use it as a weight must divide by 100.
"""

from datetime import timedelta

import pandas as pd

INDEX_CODE = "000300.SH"
# Calendar days read per decision: always spans at least two month-end sections,
# so the newest visible one is in the window even after a long holiday.
SECTION_LOOKBACK_DAYS = 120


def constituents(context):
    """(frame[con_code, weight], the section's trade_date) visible at this decision.

    Raises when no section is visible: a benchmark-relative book has no universe
    without one, and silently falling back to the whole market would be a
    different strategy wearing this one's name.
    """
    start = (context.inference_at - timedelta(days=SECTION_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"],
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.dropna(subset=["weight"])
    if frame.empty:
        raise RuntimeError(f"no visible {INDEX_CODE} cross-section in the macro domain")
    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"].astype(str) == section].drop_duplicates("con_code", keep="last")
    return (rows[["con_code", "weight"]].rename(columns={"con_code": "ts_code"}).reset_index(drop=True),
            section)
