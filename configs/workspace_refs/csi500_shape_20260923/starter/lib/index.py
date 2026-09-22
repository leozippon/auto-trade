"""The as-of CSI 500 constituent sections, read from the macro domain.

THE ONE PLACE AN INDEX CODE IS WRITTEN DOWN IN THIS PACKAGE. `INDEX_CODE` below
is the only literal; every other module refers to this name. The authority for
its value is NOT this file -- it is the arm's own run facts, where
`experiment_facts.benchmark_index` is a create-time parameter. The host keys three things to that
parameter at once: the benchmark leg of the verdict's neutralisation regression,
the constituent side the zero-skill panel redraws names on, and the tracking
statistics. So an arm created with a different `benchmark_index` than the value
below would build a CSI 500 book and be graded against another index's return
and another index's membership -- the exact mismatch every pack of the previous
round banned by name. `exploration-plan.md` round 0 checks the two against each
other BEFORE any replay, and `families.md` makes a disagreement a stop, not a
thing to work around. This module cannot read the run facts itself -- `context`
carries directories and a decision time, and the loader allows no JSON read --
so the check is the session's, and it is the first item of the plan. Putting
`benchmark_index` on `context` would change the baked strategy contract, so it
belongs to the next image rebuild rather than to this round.

`index_weight` is the one macro dataset that is per-stock rather than
per-series: one row per (index, constituent, month-end trade date), keyed on
`con_code`, not `ts_code`. Every row carries `available_at`, stamped 17:30 of
the cross-section's own trade date, so an 08:30 decision never sees the section
dated today. Between two month-end sections the membership a decision reads is
the earlier one.

`weight` is a PERCENT (4.64 = 4.64 %); a section sums to 100 within 0.05. This
arm never uses the weights as portfolio weights -- the book is equal cash -- so
they are read only to drop zero-weight rows and to report the index weight a
book covers. On this index a section holds 500 names, so a single name's weight
is a fraction of a percent and the "index weight covered" reading is small by
construction; it is a coverage reading, not a quality one.

A learned arm needs more than the newest section: every training date in the fit
window is scored on the membership that was in force on THAT date, so `sections`
returns every visible section of the window and `membership` maps them onto the
panel's date axis. A missing section is a hard failure -- silently training on
the whole market, or on another index, under this book's name would be a
different strategy and would be graded as this one.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

# 000905.SH = 中证500. Must equal the arm's `benchmark_index` run fact; see the
# module docstring and `exploration-plan.md` round 0, item 1.
INDEX_CODE = "000905.SH"
NEEDED = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]


def sections(context, lookback_days):
    """Every visible `INDEX_CODE` section of the last `lookback_days`, as (trade_date, ts_code, weight)."""

    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=NEEDED,
        filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in NEEDED if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_weight is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame = frame.dropna(subset=["weight"])
    frame = frame[frame["weight"] > 0]
    if frame.empty:
        raise RuntimeError(f"no visible {INDEX_CODE} cross-section in the macro domain")
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    frame = frame.rename(columns={"con_code": "ts_code"})
    return frame[["trade_date", "ts_code", "weight"]].drop_duplicates(
        ["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def latest(frame):
    """(members of the newest visible section, that section's trade_date)."""

    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"] == section]
    return rows[["ts_code", "weight"]].reset_index(drop=True), section


def membership(frame, dates, codes):
    """(len(dates), len(codes)) bool: was this code in the section in force on this date.

    The section in force on a panel date is the newest section dated at or
    before it -- the same "not later than the decision day" rule the decision
    path uses, applied to every training date.
    """

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
