"""The benchmark's constituent sections: the universe this book lives in.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), the index
in `index_code` and the stock in `con_code` (`ts_code` is empty on these rows;
`index_daily` is the opposite and is not read here). Every row carries
`available_at`, stamped 17:30 of the section's own trade date, so an 08:30
decision never sees the section dated today: between two month-end sections
the membership a decision reads is the earlier one.

`weight` is a PERCENT (a section sums to 100 within 0.05). The book is equal
cash, so the weight never enters a score or a size; it is read so every buy
reports how much of the index the book covers.

A missing section is a hard failure: scoring the whole market, or another
index, under this book's name would be a different strategy.
"""

from datetime import timedelta

import pandas as pd

from lib import knobs

SECTION_LOOKBACK_DAYS = 70   # calendar days: always holds the newest month-end section
COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]


def newest_section(context):
    """([ts_code, weight] of the newest visible section of knobs.INDEX, its trade_date)."""

    start = (context.inference_at - timedelta(days=SECTION_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", knobs.INDEX),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_weight is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[(stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC"))
                  & frame["con_code"].notna() & (frame["weight"] > 0)]
    if frame.empty:
        raise RuntimeError(f"no visible {knobs.INDEX} constituent section in the last "
                           f"{SECTION_LOOKBACK_DAYS} days of the macro domain")
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"] == section].rename(columns={"con_code": "ts_code"})
    return rows[["ts_code", "weight"]].drop_duplicates("ts_code", keep="last").reset_index(drop=True), section
