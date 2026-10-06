"""Announcement titles from the text domain, point in time: the ranker's documents and the draft rule.

`text_index` is one row per document, with one stock code in `ts_codes` and
the first 200 characters of the title. It is large (thousands of rows a day):
every read projects four columns and filters on `dataset` and `available_at`.
Bodies under `text_library/` are never read. An event's or document's day is
the LOCAL CALENDAR DATE of `available_at` (`lib/clock.py` says when it is
fresh); a title stamped after the decision stops the replay.

Documents (`documents`). Main-board and ChiNext codes only, the pool the
book buys; other codes (STAR, Beijing, bonds, pre-listing filings) drop out.
`normalize` removes what names the company rather than the announcement, so
the ranker reads the announcement and not the firm: a short-name prefix
ending in a colon (`开尔新材：`), a leading legal name ending in 有限公司 or
the like before 关于 / 第 / a year / a board (`山东步长制药股份有限公司关于`),
and every digit, which becomes 0 (years, amounts and percentages carry the
date and the size, not the kind of announcement). Exact duplicates on one
name and day -- the prefixed and unprefixed copies of one document -- are one
document.

The draft rule (`drafts`, `events`) is the incdraft lane's, character for
character: the first restricted-stock or stock-option plan draft a company
announces after QUIET_DAYS calendar days without one, companions excluded.
It is the control c_rule.
"""

from datetime import timedelta

import pandas as pd

DATASET = "anns_d"
COLUMNS = ["dataset", "ts_codes", "title", "available_at"]
TIMEZONE = "Asia/Shanghai"

STOCK = r"^(?:00|30|60)\d{4}\.(?:SZ|SH)$"
PREFIX = r"^[^：:]{1,20}[：:]"
ENTITY = r"^[^，。；、]{2,40}?(?:股份有限公司|有限责任公司|有限公司|集团公司)(?=关于|第|20\d\d|董事会|监事会|独立董事)"
DIGITS = r"[0-9０-９]"

PLAN = "股权激励|限制性股票|股票期权"
DRAFT = "草案"
NOT = "摘要|修订|法律意见|核查|名单|考核|意见|独立财务顾问|自查|更正"
QUIET_DAYS = 90


def _read(context, start):
    """Visible rows of the dataset whose stamp may fall on or after `start` (YYYYMMDD), with their local day."""

    # One calendar day of slack: a stamp written in UTC can carry the previous date.
    since = (pd.Timestamp(start) - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = pd.read_parquet(
        context.asof_dir + "/text_index",
        columns=COLUMNS,
        filters=[("dataset", "==", DATASET), ("available_at", ">=", since)],
    )
    stamp = pd.to_datetime(rows["available_at"], utc=True, format="ISO8601")
    if (stamp > pd.Timestamp(context.inference_at)).any():
        raise RuntimeError("the as-of text index holds a title stamped after this decision")
    local = stamp.dt.tz_convert(TIMEZONE)
    rows = rows.assign(
        ts_code=rows["ts_codes"].fillna("").astype(str),
        title=rows["title"].fillna("").astype(str),
        day=(local.dt.year * 10000 + local.dt.month * 100 + local.dt.day).astype(str),
    )
    return rows[rows["day"] >= start]


def normalize(title):
    """The announcement part of each title (a Series of str), digits as 0."""

    text = title.str.replace(PREFIX, "", regex=True).str.replace(ENTITY, "", regex=True)
    return text.str.replace(DIGITS, "0", regex=True).str.strip()


def documents(context, start):
    """(ts_code, day, text) of every visible main-board or ChiNext title dated on or after
    `start`, normalized, one row per distinct text on a name and day, sorted."""

    rows = _read(context, start)
    rows = rows[rows["ts_code"].str.match(STOCK)]
    out = pd.DataFrame({
        "ts_code": rows["ts_code"].to_numpy(),
        "day": rows["day"].to_numpy(),
        "text": normalize(rows["title"]).to_numpy(),
    })
    out = out[out["text"] != ""]
    return out.drop_duplicates().sort_values(["ts_code", "day", "text"], kind="stable").reset_index(drop=True)


def drafts(context, start):
    """(ts_code, day) of every visible draft whose day (YYYYMMDD) is on or after `start`,
    one row per name and day, sorted by name and day."""

    rows = _read(context, start)
    title = rows["title"]
    rows = rows[title.str.contains(PLAN) & title.str.contains(DRAFT) & ~title.str.contains(NOT)]
    out = pd.DataFrame({"ts_code": rows["ts_code"].to_numpy(), "day": rows["day"].to_numpy()})
    out = out[out["ts_code"] != ""]
    return out.drop_duplicates().sort_values(["ts_code", "day"], kind="stable").reset_index(drop=True)


def events(rows, start):
    """The first draft per name after QUIET_DAYS calendar days without one, dated on or
    after `start`. `rows` must reach at least QUIET_DAYS + 1 days before `start`, so that
    every event's quiet period is read, not assumed."""

    gap = pd.to_datetime(rows["day"], format="%Y%m%d").groupby(rows["ts_code"]).diff().dt.days
    first = gap.isna() | (gap > QUIET_DAYS)
    return rows[first & (rows["day"] >= start)].reset_index(drop=True)
