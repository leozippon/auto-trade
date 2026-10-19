"""Point-in-time reads of the fundamentals domain: a correctness aid, not a score.

The rules these functions follow are the "财务域" section of the mounted
operating memory (research-8y-data-surface); the arithmetic is the one the
seed was checked with. In short:

    read(context, dataset, columns)  one dataset as this decision sees it
    per_period(rows, keep)           one version per (ts_code, end_date)
    newest_period(rows)              each name's newest report period
    single_quarter(rows, column)     year-to-date flow -> that quarter alone
    trailing_year(rows, column)      year-to-date flow -> last four quarters
    calendar(context, lookback_days) visible trading days plus today
    event_days(stamps, days)         first decision day that sees each stamp

`read` always projects columns and filters on `dataset`: the domain is one wide
table of ten datasets, and a read without projection is killed at the 16 GiB
container limit on late views. Each vendor version is its own row at its own
`available_at`; `update_flag` does not mark revisions, so versions are told
apart by stamp only -- the newest is the current state, the earliest the
announcement (the event). Statement flows are year-to-date sums keyed by the
period's last day (MMDD 0331 / 0630 / 0930 / 1231); a quarter or a trailing
year is computed only when every period it needs is in `rows`, NaN otherwise.
Read at a review, not on every decision: the as-of directory gains one part
per trading day.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

KEYS = ["ts_code", "end_date", "available_at"]
TIMEZONE = "Asia/Shanghai"
DECISION_TIME = pd.Timedelta(hours=8, minutes=30)
# The domain is released into the as-of view by the PIT node at 03:35 on
# Tuesday..Saturday (Monday is 0): a Friday 18:00 stamp reaches Monday's
# decision, a Saturday or Sunday stamp Tuesday's.
RELEASE_TIME = pd.Timedelta(hours=3, minutes=35)
QUARTER = {"0331": 1, "0630": 2, "0930": 3, "1231": 4}
PREVIOUS_END = {"0630": "0331", "0930": "0630", "1231": "0930"}


def read(context, dataset, columns):
    """One dataset's rows in this decision's as-of view, with `stamp` (UTC).

    Projects the key columns and `columns`, filters on `dataset` while reading,
    and returns the rows sorted by name, period and stamp. A column the domain
    does not carry fails the read.
    """

    rows = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=list(dict.fromkeys([*KEYS, *columns])),
        filters=[("dataset", "==", dataset)],
    )
    rows = rows.assign(
        stamp=pd.to_datetime(rows["available_at"], utc=True),
        ts_code=rows["ts_code"].astype(str),
        end_date=rows["end_date"].astype(str),
    )
    return rows.sort_values(["ts_code", "end_date", "stamp"], kind="stable").reset_index(drop=True)


def per_period(rows, keep="last"):
    """One row per (ts_code, end_date): the newest visible version (`"last"`,
    the current state) or the first announcement (`"first"`, the event).
    Versions sharing a stamp carry the same figures; the pick among them is
    deterministic."""

    if keep not in ("first", "last"):
        raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")
    ordered = rows.sort_values(["ts_code", "end_date", "stamp"], kind="stable")
    return ordered.drop_duplicates(["ts_code", "end_date"], keep=keep).reset_index(drop=True)


def newest_period(rows):
    """Each name's row with the latest `end_date`. By period, never by stamp:
    a restatement of an old period is stamped later than the current report."""

    return rows.sort_values(["ts_code", "end_date"], kind="stable").drop_duplicates("ts_code", keep="last")


def _lookup(rows, column, codes, ends):
    """`column` of the period (code, end) in `rows`, NaN where it is absent."""

    table = rows.set_index(["ts_code", "end_date"])[column]
    if not table.index.is_unique:
        raise ValueError("pass one version per (ts_code, end_date), e.g. per_period(rows)")
    keys = pd.MultiIndex.from_arrays([np.asarray(codes, dtype=object), np.asarray(ends, dtype=object)])
    return table.reindex(keys).to_numpy(dtype="float64")


def single_quarter(rows, column):
    """The quarter alone of a year-to-date flow, aligned with `rows`.

    Q1 is its own year-to-date value; Q2..Q4 subtract the previous quarter of
    the same fiscal year, NaN when that period is not in `rows`.
    """

    ends = rows["end_date"].astype(str)
    day = ends.str[4:]
    previous = (ends.str[:4] + day.map(PREVIOUS_END)).fillna("")
    prior = _lookup(rows, column, rows["ts_code"], previous)
    value = rows[column].to_numpy(dtype="float64")
    out = np.where(day.eq("0331"), value, value - prior)
    return pd.Series(np.where(day.map(QUARTER).notna(), out, np.nan), index=rows.index)


def trailing_year(rows, column):
    """The last four quarters of a year-to-date flow, aligned with `rows`.

    A full year is its own value; otherwise this year-to-date plus last year's
    full year minus last year's same year-to-date, NaN unless both are in `rows`.
    """

    ends = rows["end_date"].astype(str)
    day = ends.str[4:]
    last_year = (ends.str[:4].astype(int) - 1).astype(str)
    annual = _lookup(rows, column, rows["ts_code"], last_year + "1231")
    same = _lookup(rows, column, rows["ts_code"], last_year + day)
    value = rows[column].to_numpy(dtype="float64")
    out = np.where(day.eq("1231"), value, value + annual - same)
    return pd.Series(np.where(day.map(QUARTER).notna(), out, np.nan), index=rows.index)


def calendar(context, lookback_days):
    """Visible trading days over `lookback_days` calendar days, plus the decision day.

    The newest `daily` row at an 08:30 decision is the previous trading day, so
    the decision day itself is appended.
    """

    start = (context.inference_at - timedelta(days=int(lookback_days))).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    today = pd.Timestamp(context.inference_at).tz_convert(TIMEZONE).strftime("%Y%m%d")
    return sorted({*days["trade_date"].astype(str), today})


def event_days(stamps, days):
    """For each stamp, the first day in `days` whose 08:30 decision sees it.

    A row reaches the as-of view at the first 03:35 node run, Tuesday to
    Saturday, at or after its stamp, and a decision sees what was released
    before it. `days` is a sorted list of YYYYMMDD trading days (for instance
    `calendar`); a missing stamp, or one released after the last of them, maps
    to None. Offline, on the research-end snapshot, it reproduces the replay's timing.
    """

    local = pd.DatetimeIndex(pd.to_datetime(pd.Series(stamps), utc=True)).tz_convert(TIMEZONE)
    midnight = local.normalize()
    run = midnight + RELEASE_TIME + pd.to_timedelta(np.where(local > midnight + RELEASE_TIME, 1, 0), unit="D")
    weekday = run.weekday.to_numpy()
    release = run + pd.to_timedelta(np.select([weekday == 6, weekday == 0], [2, 1], 0), unit="D")
    opens = pd.DatetimeIndex(pd.to_datetime(list(days), format="%Y%m%d")).tz_localize(TIMEZONE) + DECISION_TIME
    where = np.searchsorted(opens.asi8, release.asi8, side="left")
    return [
        None if pd.isna(stamp) or i >= len(days) else days[int(i)]
        for stamp, i in zip(local, where)
    ]
