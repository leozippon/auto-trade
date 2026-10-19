"""Point-in-time announcement titles (text domain `anns_d`): what a decision sees, deduplicated and anonymised.

    read(context, calendar, since)  -> frame (ts_code, day, title): every
                                       visible title first seen by a decision
                                       on or after `since`, one row per name,
                                       decision day and anonymised title
    decision_days(stamps, days)        the first day in `days` whose 08:30
                                       decision sees each stamp: the pure
                                       rule, the text analogue of
                                       fund.event_days
    anonymise(titles, names)           the titles without their issuer's name
                                       or code
    table(rows, codes, days)           each name's title ids at each decision
                                       of `days`, as one integer array

Visibility. `available_at` is the vendor's receipt time when that lies within
-1..+3 days of the announcement date, otherwise 23:59:59 local on the
announcement date; nearly every research-period row is the latter, nearly
every forward-period and Paper row the former. The replay publishes titles at
the nightly 23:15 text node, and a decision at 08:30 on day T has seen the
node of the previous night: a row is visible when its stamp is at or before
(T-1) 23:15. So a date-only title dated D first reaches the decision of the
first trading day on or after D+2 (a Friday title the next Tuesday), and a
receipt-stamped title received the evening before D the decision of D itself.
`decision_days` is that rule; `read` applies it to the rows the view already
holds. One exception is the view's own: a research year's first decision reads
a view cut at the year's anchor instant (23:59:59 of the day before), not at
a node, so the anchor day's titles are already there a node early. The view
is the ground truth of what is visible, so `read` dates such a row by the
decision that holds it, and refuses only a row stamped after the decision.

Duplicates. Before 2025 the vendor sent most filings twice -- the plain title
and a copy prefixed with the issuer's short name, sometimes under a second
link -- and the view does not merge them. `anonymise` strips that prefix, so
the copies collapse onto one title and `read` keeps one row per (name,
decision day, title). Titles of one name and day come back in title order,
not arrival order, which the view does not record.

Anonymisation. The language model that reads these titles was pretrained on
text that covers the research years. Stripping the issuer's short name (the
prefix before the first colon, the `universe` name and any six-digit code)
and reducing full company names (...有限公司) to 公司 limits what the model can
recall about the issuer; it does not remove it, and the event itself may be
memorable. The README states the residual.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import knobs

DATASET = "anns_d"
TIMEZONE = "Asia/Shanghai"
COLUMNS = ["dataset", "ts_codes", "title", "available_at"]
DECISION_TIME = pd.Timedelta(hours=8, minutes=30)
# The nightly text node's cutoff: a row stamped later waits for the next night.
NODE_TIME = pd.Timedelta(hours=23, minutes=15)
# Calendar days read behind `since`: the longest exchange holiday plus the two
# days a date-only stamp waits.
READ_SLACK_DAYS = 15
# The patterns `anonymise` removes (pandas string methods; `re` is not an
# allowed import): a leading [tag], the issuer prefix before the first colon,
# a six-digit code, an ST marker on a name, and a full company name -- up to
# 16 characters, none of them a lead-in word that would pull the sentence
# before the name into the match, ending in (股份|集团)有限(责任)公司.
TAG = r"^\[[^\]]{1,10}\]"
PREFIX = r"^[^：:\s]{1,12}[：:]"
CODE = r"(?<!\d)\d{6}(?!\d)"
ST = r"^\*?ST\s*"
COMPANY = r"[^\s关于对与及暨向为的、，。；：:（）()\[\]]{2,16}?(?:股份|集团)?有限(?:责任)?公司"


def read(context, calendar, since):
    """(ts_code, day, title) of every visible title whose first visible
    decision day is on or after `since`, sorted by name, day and title.

    `calendar` is the sorted list of YYYYMMDD visible trading days ending
    with the decision day, and `since` one of its days after the first: a
    title older than the list would otherwise be dated by the list's first
    day instead of its own. A title the rule dates after the decision day is
    dated by the decision day (the view holds it, so this decision sees it).
    """

    days = [str(day) for day in calendar]
    today = context.inference_at.strftime("%Y%m%d")
    if days[-1] != today:
        raise ValueError(f"the calendar must end with the decision day {today}, it ends with {days[-1]}")
    if since not in days or days[0] >= since:
        raise ValueError(f"`since` {since} must be a day of the calendar after its first day {days[0]}")
    lower = (pd.Timestamp(since) - timedelta(days=READ_SLACK_DAYS)).strftime("%Y-%m-%d")
    rows = pd.read_parquet(
        context.asof_dir + "/text_index",
        columns=COLUMNS,
        filters=[("dataset", "==", DATASET), ("available_at", ">=", lower)],
    )
    stamp = pd.to_datetime(rows["available_at"], utc=True)
    if (stamp > pd.Timestamp(context.inference_at)).any():
        raise RuntimeError("the as-of text index holds a title stamped after this decision")
    day = np.array([today if seen is None else seen for seen in decision_days(stamp, days)], dtype=object)
    codes = rows["ts_codes"].fillna("").astype(str).to_numpy()
    keep = (day >= since) & (codes != "")
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "name"])
    names = universe.drop_duplicates("ts_code").set_index("ts_code")["name"]
    out = pd.DataFrame({"ts_code": codes[keep], "day": day[keep]})
    out["title"] = anonymise(
        rows["title"].fillna("").astype(str).to_numpy()[keep],
        names.reindex(out["ts_code"]).fillna("").to_numpy(),
    )
    out = out[out["title"] != ""]
    return out.drop_duplicates().sort_values(["ts_code", "day", "title"], kind="stable").reset_index(drop=True)


def decision_days(stamps, days):
    """For each stamp, the first day in `days` whose 08:30 decision sees it.

    A row joins the view at the first 23:15 node at or after its stamp, and a
    decision sees the nodes of the nights before it. `days` is a sorted list
    of YYYYMMDD trading days (for instance the visible ones plus today); a
    missing stamp, or one no day of the list sees, maps to None. Offline, on
    the research-end snapshot, it reproduces the replay's timing.
    """

    local = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(stamps)), utc=True)).tz_convert(TIMEZONE)
    midnight = local.normalize()
    # The calendar day whose node publishes the row: its own when stamped by
    # 23:15, else the next.
    node_day = midnight + pd.to_timedelta((local > midnight + NODE_TIME).astype(int), unit="D")
    first = node_day + pd.Timedelta(days=1) + DECISION_TIME
    opens = pd.DatetimeIndex(pd.to_datetime(list(days), format="%Y%m%d")).tz_localize(TIMEZONE) + DECISION_TIME
    where = np.searchsorted(opens.asi8, first.asi8, side="left")
    return [None if pd.isna(stamp) or i >= len(days) else days[int(i)] for stamp, i in zip(local, where)]


def anonymise(titles, names=None):
    """The titles without their issuer: a leading [tag], the prefix before the
    first colon (applied twice, the vendor sometimes doubled it), every
    six-digit code and the name in `names` (aligned with `titles`, "" where
    unknown, an ST marker ignored) are removed, full company names become
    公司, and whitespace is collapsed; "" where nothing else remains."""

    cleaned = pd.Series([str(title) for title in titles], dtype="object").str.strip()
    cleaned = cleaned.str.replace(TAG, "", regex=True)
    for _ in range(2):
        cleaned = cleaned.str.replace(PREFIX, "", regex=True).str.strip()
    cleaned = cleaned.str.replace(CODE, "", regex=True)
    if names is not None:
        issuers = pd.Series([str(name or "") for name in names], dtype="object").str.replace(ST, "", regex=True).str.strip()
        cleaned = pd.Series(
            [title.replace(name, "") if len(name) >= 2 else title for title, name in zip(cleaned, issuers)],
            dtype="object",
        )
    cleaned = cleaned.str.replace(COMPANY, "公司", regex=True)
    return cleaned.str.replace(r"\s+", " ", regex=True).str.strip().tolist()


def table(rows, codes, days):
    """(len(days), len(codes), MAX_TITLES) int64 array: for each decision day
    and name, the ids (`rows["title_id"]`) of the newest MAX_TITLES titles
    first seen within the last TEXT_DAYS days of `days` up to that one, -1
    where there are fewer. Rows of names or days outside the lists are
    ignored; within one day the order is the rows' own."""

    out = np.full((len(days), len(codes), knobs.MAX_TITLES), -1, dtype="int64")
    code = pd.Index(codes).get_indexer(rows["ts_code"].astype(str))
    seen = pd.Index([str(day) for day in days]).get_indexer(rows["day"].astype(str))
    title = rows["title_id"].to_numpy(dtype="int64")
    inside = (code >= 0) & (seen >= 0)
    frames = []
    for age in range(knobs.TEXT_DAYS):
        at = seen[inside] + age
        fits = at < len(days)
        frames.append(pd.DataFrame({
            "at": at[fits], "code": code[inside][fits], "age": age,
            "order": np.flatnonzero(inside)[fits], "title": title[inside][fits],
        }))
    stacked = pd.concat(frames, ignore_index=True).sort_values(["at", "code", "age", "order"], kind="stable")
    rank = stacked.groupby(["at", "code"], sort=False).cumcount().to_numpy()
    kept = stacked[rank < knobs.MAX_TITLES]
    out[kept["at"].to_numpy(), kept["code"].to_numpy(), rank[rank < knobs.MAX_TITLES]] = kept["title"].to_numpy()
    return out
