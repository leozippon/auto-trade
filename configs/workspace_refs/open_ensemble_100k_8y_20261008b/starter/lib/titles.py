"""Point-in-time reads of the announcement titles (text domain, `anns_d`): a correctness aid, not a score.

    read(context, since, *patterns, exclude=None)
        visible titles whose local day is on or after `since`, matching every
        regular expression in `patterns` and not `exclude`: one row per name,
        day and title
    first_after_quiet(rows, quiet_days, start)
        each name's first day after more than `quiet_days` calendar days
        without a matching title, dated on or after `start`

`text_index` is one row per document -- about ten million by the end of the
research period -- with the stock code in `ts_codes` and the first 200
characters of the title; bodies are never read. A read projects four columns
and filters on `dataset` and `available_at` while reading, and belongs at a
review, not at every decision. The companions filed with one announcement
(summary, revised version, legal and adviser opinions, grantee lists) are
rows of their own, so an event is a name and a day, not a row.

Visibility. `available_at` is the vendor's receipt time when that lies within
-1..+3 days of the announcement date, otherwise 23:59:59 local on the
announcement date; nearly every research-period row is the latter. `day` here
is the LOCAL CALENDAR DATE of `available_at`. The replay publishes titles at
the nightly 23:15 text node, so a research title dated D first reaches the
08:30 decision of calendar day D + 2 (a Friday title the next Monday). `read`
returns only titles already in the view, so a book built on it never acts
early; date an event by the first trading day after its `day`, which keeps
the research-period stamps and the later receipt-time stamps on one rule.
"""

from datetime import timedelta

import pandas as pd

DATASET = "anns_d"
TIMEZONE = "Asia/Shanghai"
COLUMNS = ["dataset", "ts_codes", "title", "available_at"]


def read(context, since, *patterns, exclude=None):
    """(ts_code, day, title) of every visible title whose day (YYYYMMDD) is on or
    after `since`, that matches every pattern and, when given, not `exclude`;
    sorted by name, day and title, duplicates dropped."""

    # One calendar day of slack: a stamp written in UTC can carry the previous date.
    lower = (pd.Timestamp(since) - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = pd.read_parquet(
        context.asof_dir + "/text_index",
        columns=COLUMNS,
        filters=[("dataset", "==", DATASET), ("available_at", ">=", lower)],
    )
    title = rows["title"].fillna("").astype(str)
    keep = pd.Series(True, index=rows.index)
    for pattern in patterns:
        keep &= title.str.contains(pattern)
    if exclude:
        keep &= ~title.str.contains(exclude)
    rows, title = rows[keep], title[keep]
    stamp = pd.to_datetime(rows["available_at"], utc=True)
    if (stamp > pd.Timestamp(context.inference_at)).any():
        raise RuntimeError("the as-of text index holds a title stamped after this decision")
    out = pd.DataFrame({
        "ts_code": rows["ts_codes"].fillna("").astype(str).to_numpy(),
        "day": stamp.dt.tz_convert(TIMEZONE).dt.strftime("%Y%m%d").to_numpy(),
        "title": title.to_numpy(),
    })
    out = out[(out["ts_code"] != "") & (out["day"] >= since)]
    return out.drop_duplicates().sort_values(["ts_code", "day", "title"], kind="stable").reset_index(drop=True)


def first_after_quiet(rows, quiet_days, start):
    """(ts_code, day) of each name's days that follow more than `quiet_days`
    calendar days without a row, dated on or after `start`. `rows` must reach
    back at least `quiet_days` + 1 days before `start`, so that every quiet
    period is read rather than assumed."""

    days = rows[["ts_code", "day"]].drop_duplicates().sort_values(["ts_code", "day"], kind="stable")
    gap = pd.to_datetime(days["day"], format="%Y%m%d").groupby(days["ts_code"]).diff().dt.days
    first = gap.isna() | (gap > quiet_days)
    return days[first & (days["day"] >= start)].reset_index(drop=True)
