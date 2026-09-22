"""The benchmark: its constituent sections, and its own daily series.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), keyed on
`con_code`, not `ts_code`. Every row carries `available_at`, stamped 17:30 of
the cross-section's own trade date, so an 08:30 decision never sees the
section dated today. Between two month-end sections the membership a decision
reads is the earlier one, and the same rule gives every training date its own
membership: the newest section dated at or before that date.

`weight` is a PERCENT (4.64 = 4.64 %); a section sums to 100 within 0.05.
This starter uses membership only, never the weight, because the book is equal
cash -- the column is read so the round-0 census can report how much of the
index a book covers without changing this read.

`index_daily` carries the benchmark's own open, close and traded amount,
stamped 17:30 of its trade date like the sections. The label's benchmark leg
and the market-state block's benchmark columns come from it, so the strategy
never reconstructs an index from its members. Its `amount` is in THOUSANDS of
CNY; the block only takes its log inside a z-score, where the unit cancels.

A missing section or a missing benchmark series is a hard failure. Scoring the
whole market, or a different index, under this book's name would be a
different strategy.
"""

from datetime import timedelta

import pandas as pd

# The one place the benchmark is named. The host grades beta, tracking error,
# the neutralisation and the zero-skill panel's own constituent matching
# against ONE index, and the pipeline publishes which one as the top-level
# experiment fact `benchmark_index` -- a plain ts_code string, with
# `neutralized_excess_method` beside it, and repeated in every result's
# `stats.benchmark`. It is a create-time parameter, so it can differ from the
# literal below. Round 0 reads that key FIRST and, if it names another index,
# changes this constant to match it: the run fact overrides this literal,
# never the other way round. A strategy cannot read the fact at decision time,
# which is exactly why this is a constant the session aligns once rather than
# a lookup. Building a book on one index while the host grades it against
# another is a `families.md` termination condition.
INDEX_CODE = "000300.SH"

SECTION_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
BENCHMARK_COLUMNS = ["dataset", "available_at", "ts_code", "trade_date", "open", "close", "amount"]


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
    frame = frame[frame["weight"] > 0]
    out = frame[["con_code", "trade_date", "weight"]].rename(columns={"con_code": "ts_code"}).copy()
    out["trade_date"] = out["trade_date"].astype(str)
    return out.drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def benchmark(context, calendar_days):
    """[trade_date, open, close, amount] of the benchmark itself, ascending."""

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
    out = frame[["trade_date", "open", "close", "amount"]].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    return out.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)


def latest(frame):
    """(members of the newest visible section, that section's trade_date)."""

    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"] == section]
    return rows[["ts_code", "weight"]].reset_index(drop=True), section


def membership(frame, dates, codes):
    """(len(dates), len(codes)) bool: was this code in the section in force on this date.

    The section in force on a panel date is the newest section dated at or
    before it -- the same "not later than the decision day" rule the decision
    path uses, applied to every training date. Dates before the window's first
    visible section carry no membership at all: back-filling them with the
    first section would hand a future constituent list to a past training day.
    """

    import numpy as np

    sect_dates = np.array(sorted(frame["trade_date"].unique()))
    dates = np.asarray(dates)
    slot = np.searchsorted(sect_dates, dates, side="right") - 1
    member = np.zeros((len(dates), len(codes)), dtype=bool)
    position = {code: index for index, code in enumerate(codes)}
    by_section = {name: group["ts_code"].to_numpy() for name, group in frame.groupby("trade_date", sort=False)}
    rows = {}
    for index, name in enumerate(sect_dates):
        found = [position[code] for code in by_section[name] if code in position]
        rows[index] = np.array(sorted(found), dtype=np.int64)
    for t, which in enumerate(slot):
        if which >= 0:
            member[t, rows[int(which)]] = True
    return member
