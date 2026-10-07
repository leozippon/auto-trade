"""Event score for the esop60 leg.

Mechanism: a commitment-type announcement of family `Employee shareholding plan DRAFT ` starts an
event. T0 = the first trading day on or after the title's local day + 2
calendar days (the nightly 23:15 text node makes a day-D title visible at the
08:30 decision of D+2, so T0 is never acted on before it is visible). The
entry day is T0 + LATE_SHIFT trading days (the skeleton buys at 15:00 of the
first decision whose T-1 bar is on or after the entry day, i.e. entry+1).
A name is scored on every decision day D with 0 <= pos(T-1) - pos(entry) <
HOLD_DAYS, i.e. the name stays in the book for HOLD_DAYS decision days and is
force-exited at the next weekly review after the window closes. Among active
names the newest entry ranks first, so a full book holds the freshest events
and lets the oldest expire.

No fit: the rule is a fixed announcement pattern, 90-day quiet dedupe and the
knobs below. `LATE_SHIFT > 0` makes the control leg of the same event set
enter the event late.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from sleeves.esop60 import knobs, titles

NAME = "esop60"
TITLE_PATTERN = '员工持股计划.*草案|草案.*员工持股计划'
TITLE_EXCLUDE = '摘要|修订|法律意见|核查|独立财务顾问|财务顾问|管理办法|自查|更正|回复|问询|进展|出售|终止|届满|期满|展期|报告|说明|审核|议案'
QUIET_DAYS = 90
# Title lookback (calendar days) must cover quiet + entry shift + hold + slack
# so that every 90-day quiet gap of an active event is read, not assumed.
WINDOW_CAL_DAYS = 320
MIN_WINDOW_TITLES = 5


def score(context, codes):
    D = pd.Timestamp(context.inference_at)
    start_cal = (D - timedelta(days=560)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start_cal)])
    cal = np.array(sorted(days["trade_date"].astype(str).unique()))
    last = len(cal) - 1
    since = (D - timedelta(days=WINDOW_CAL_DAYS)).strftime("%Y%m%d")
    rows = titles.read(context, since, TITLE_PATTERN, exclude=TITLE_EXCLUDE)
    if len(rows) < MIN_WINDOW_TITLES:
        raise RuntimeError(f"the event title window holds only {len(rows)} matching titles; the text view may be broken")
    ev = rows.assign(code=rows["ts_code"].str.split(",")).explode("code")
    ev["ts_code"] = ev["code"].str.strip()
    ev = ev[ev["ts_code"].str.match(r"^\d{6}\.(SZ|SH)$", na=False)]
    ev = ev[["ts_code", "day"]].drop_duplicates().sort_values(["ts_code", "day"], kind="stable")
    quiet = titles.first_after_quiet(ev, QUIET_DAYS, since)
    if quiet.empty:
        values = pd.Series(np.nan, index=list(codes), dtype="float64")
        return values, {"scored": 0, "window_start": since, "hold_days": knobs.HOLD_DAYS, "late_shift": knobs.LATE_SHIFT}
    target = (pd.to_datetime(quiet["day"], format="%Y%m%d") + pd.Timedelta(days=2)).dt.strftime("%Y%m%d").to_numpy()
    j = np.searchsorted(cal, target)
    ok = j < len(cal)
    p_entry = np.full(len(quiet), -1, dtype="int64")
    p_entry[ok] = j[ok]
    p_entry = p_entry + knobs.LATE_SHIFT
    active = (p_entry >= 0) & (p_entry <= last) & ((last - p_entry) < knobs.HOLD_DAYS)
    if not active.any():
        values = pd.Series(np.nan, index=list(codes), dtype="float64")
        return values, {"scored": 0, "window_start": since, "hold_days": knobs.HOLD_DAYS, "late_shift": knobs.LATE_SHIFT}
    pos = pd.Series(p_entry[active], index=quiet["ts_code"].to_numpy()[active])
    pos = pos.groupby(level=0).max()
    values = pos.reindex(list(codes)).astype("float64")
    return values, {"scored": int(values.notna().sum()), "window_start": since,
                    "hold_days": knobs.HOLD_DAYS, "late_shift": knobs.LATE_SHIFT}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [float(picked.min()), float(picked.max())] if len(picked) else None}
