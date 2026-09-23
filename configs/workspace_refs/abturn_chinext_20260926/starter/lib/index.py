"""The benchmark's newest visible constituent section.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), keyed on
`con_code`, not `ts_code`, with the index in `index_code`. Every row carries
`available_at`, stamped 17:30 of the section's own trade date, so an 08:30
decision never sees the section dated today; between two month-end sections
the pool is the earlier one. `weight` is a PERCENT; the book is equal cash and
reads it only to report how much of the index a book covers.

A missing section is a hard failure: scoring the whole market, or a different
index, under this book's name would be a different strategy.
"""

from datetime import timedelta

import pandas as pd

# The one place the benchmark is named. The host grades beta, tracking error,
# the neutralisation and the zero-skill panel's constituent matching against
# ONE index, published as the run fact `benchmark_index`. It is a create-time
# parameter, so it can differ from the literal below: round 0 reads that key
# FIRST and, if it names another index, changes this constant to match it --
# the run fact overrides this literal, never the other way round. A strategy
# cannot read the fact at decision time, which is why this is a constant.
INDEX_CODE = "399006.SZ"

COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
# Calendar days searched for the newest section: a month-end section is at
# most about five weeks old at a decision.
SECTION_DAYS = 70


def latest(context):
    """(members [ts_code, weight] of the newest visible section, that section's trade_date)."""

    start = (context.inference_at - timedelta(days=SECTION_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_weight is missing columns {missing}")
    frame = frame.dropna(subset=["weight", "con_code"])
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")) & (frame["weight"] > 0)]
    if frame.empty:
        raise RuntimeError(f"no visible {INDEX_CODE} constituent section in the last {SECTION_DAYS} days")
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"] == section].drop_duplicates("con_code", keep="last")
    members = rows[["con_code", "weight"]].rename(columns={"con_code": "ts_code"}).reset_index(drop=True)
    return members, section
