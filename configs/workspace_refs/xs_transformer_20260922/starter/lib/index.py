"""The CSI 300 constituent sections and the benchmark's own daily series.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), keyed on
`con_code`, not `ts_code`. Every row carries `available_at`, stamped 17:30 of
the cross-section's own trade date, so an 08:30 decision never sees the
section dated today. Between two month-end sections the membership a decision
reads is the earlier one, and the same rule gives every training date its own
membership: the newest section dated at or before that date.

`weight` is a PERCENT (4.64 = 4.64 %); a section sums to 100 within 0.05.
This starter uses membership only, never the weight, because the book is
equal cash -- but the column is read so a variant that tilts on benchmark
weights does not have to change the read.

`index_daily` carries the benchmark's own open and close, stamped 17:30 of
its trade date like the sections. The label's benchmark leg and the market
state both come from it, so the strategy never reconstructs an index from its
members.

A missing section or a missing benchmark series is a hard failure. Scoring
the whole market, or a different index, under this book's name would be a
different strategy.
"""

from datetime import timedelta

import pandas as pd

INDEX_CODE = "000300.SH"
SECTION_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
BENCHMARK_COLUMNS = ["dataset", "available_at", "ts_code", "trade_date", "open", "close"]


def _visible(frame, context, what):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    if frame.empty:
        raise RuntimeError(f"no visible {INDEX_CODE} {what} in the macro domain")
    return frame


def sections(context, calendar_days):
    """[ts_code, trade_date, weight] of every visible section in the window."""

    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=SECTION_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in SECTION_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_weight is missing columns {missing}")
    frame = _visible(frame.dropna(subset=["weight", "con_code"]), context, "constituent section")
    out = frame[["con_code", "trade_date", "weight"]].rename(columns={"con_code": "ts_code"}).copy()
    out["trade_date"] = out["trade_date"].astype(str)
    return out.drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def benchmark(context, calendar_days):
    """[trade_date, open, close] of the benchmark itself, ascending."""

    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=BENCHMARK_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in BENCHMARK_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_daily is missing columns {missing}")
    frame = _visible(frame.dropna(subset=["close"]), context, "benchmark daily series")
    out = frame[["trade_date", "open", "close"]].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    return out.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
