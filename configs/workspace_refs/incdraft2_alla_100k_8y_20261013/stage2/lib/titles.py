"""Incentive-plan drafts from the announcement titles of the text domain, point in time.

The event is the first restricted-stock or stock-option plan draft a company
announces after QUIET_DAYS calendar days without one. A title is a draft when
it names the plan (PLAN), says draft (DRAFT) and names none of the companion
documents filed around it (NOT): the summary, revised versions, legal and
adviser opinions, verification notes, the grantee list and the appraisal
rules. Employee-ownership plans are not in PLAN. The rule is round 20261012's
offline rule unchanged; `refs/README.md` lists the companions it is known to
admit. Each (name, day) carries `option`: whether any draft title of that day
names a stock option (OPTION). It splits the events, it never makes one.

`available_at` is the vendor's receipt time when that lies within -1..+3
days of the announcement date, otherwise 23:59:59 local on the announcement
date; every research-window row is the latter. An event's day here is the
LOCAL CALENDAR DATE of `available_at`, and `lib/book.py` counts from the first
trading day after it. The replay publishes titles at the nightly 23:15 text
node, so a research title dated D first reaches the 08:30 decision of
calendar day D + 2 (a Monday-to-Thursday title the second trading day after
D, a Friday title the next Monday); a receipt-time stamp before 23:15 reaches
the next morning. Either way a title is in the view no later than one trading
day after the day the book counts from, so the review that buys it always
sees it and no event falls between two reviews. The book assumes nothing
else: it only ever reads titles already in the view.

`text_index` is one row per document, with the stock code in `ts_codes` and
the first 200 characters of the title. It is large (thousands of rows a day):
every read projects four columns and filters on `dataset` and `available_at`.
Bodies under `text_library/` are never read.
"""

from datetime import timedelta

import pandas as pd

DATASET = "anns_d"
PLAN = "股权激励|限制性股票|股票期权"
DRAFT = "草案"
NOT = "摘要|修订|法律意见|核查|名单|考核|意见|独立财务顾问|自查|更正"
OPTION = "股票期权"
QUIET_DAYS = 90
TIMEZONE = "Asia/Shanghai"
COLUMNS = ["dataset", "ts_codes", "title", "available_at"]


def drafts(context, start):
    """(ts_code, day, option) of every visible draft whose day (YYYYMMDD) is on or after
    `start`, one row per name and day, sorted by name and day."""

    # One calendar day of slack: a stamp written in UTC can carry the previous date.
    since = (pd.Timestamp(start) - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = pd.read_parquet(
        context.asof_dir + "/text_index",
        columns=COLUMNS,
        filters=[("dataset", "==", DATASET), ("available_at", ">=", since)],
    )
    title = rows["title"].fillna("").astype(str)
    rows = rows[title.str.contains(PLAN) & title.str.contains(DRAFT) & ~title.str.contains(NOT)]
    stamp = pd.to_datetime(rows["available_at"], utc=True)
    if (stamp > pd.Timestamp(context.inference_at)).any():
        raise RuntimeError("the as-of text index holds a title stamped after this decision")
    out = pd.DataFrame({
        "ts_code": rows["ts_codes"].fillna("").astype(str).to_numpy(),
        "day": stamp.dt.tz_convert(TIMEZONE).dt.strftime("%Y%m%d").to_numpy(),
        "option": rows["title"].fillna("").astype(str).str.contains(OPTION).to_numpy(),
    })
    out = out[(out["ts_code"] != "") & (out["day"] >= start)]
    return out.groupby(["ts_code", "day"], as_index=False, sort=True)["option"].any()


def events(rows, start):
    """The first draft per name after QUIET_DAYS calendar days without one, dated on or
    after `start`. `rows` must reach at least QUIET_DAYS + 1 days before `start`, so that
    every event's quiet period is read, not assumed."""

    gap = pd.to_datetime(rows["day"], format="%Y%m%d").groupby(rows["ts_code"]).diff().dt.days
    first = gap.isna() | (gap > QUIET_DAYS)
    return rows[first & (rows["day"] >= start)].reset_index(drop=True)
