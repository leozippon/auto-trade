"""Title-defined company commitments from the text domain, point in time.

A category is a rule on the announcement title (RULES): every pattern in its
first element must be in the title, and the second element names the
companion documents filed around the act -- opinions, adjustments, progress
and completion notices, corrections -- none of which may be. An event of a set
of categories is the first title of any of them that a stock carries after
QUIET_DAYS calendar days without one; one category alone is the set of one.
`refs/README.md` §3 gives each rule's source and what it is known to admit.

`available_at` is the vendor's receipt time when that lies within -1..+3
days of the announcement date, otherwise 23:59:59 local on the announcement
date; all but 0.2 % of the research-window rows are the latter. An event's
day here is the LOCAL CALENDAR DATE of `available_at`, and `lib/book.py`
counts from the first trading day after it. The replay publishes titles at the nightly 23:15 text
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
# Category: (patterns the title must all contain, companions it must not name).
RULES = {
    # The board grants restricted stock or options under an approved plan;
    # reserved grants, unlocks, vesting and exercise are later acts.
    "grant": (
        ("授予", "限制性股票|股票期权", "公告"),
        "调整|核查|法律意见|名单|意见|预留|回购|注销|解除|归属|行权|上市|解锁|限售|解限",
    ),
    # A controlling shareholder, director or executive announces a plan to buy.
    "increase": (
        ("增持", "计划"),
        "减持|进展|完成|结果|实施|期限届满|终止|延期|变更|核查|法律意见",
    ),
    # The company proposes, plans or reports a programme to buy back its own
    # A shares; not the cancellation of plan shares or compensation shares.
    "buyback": (
        ("回购", "方案|预案|回购报告书|提议"),
        "限制性股票|股票期权|回购注销|补偿|调整|核查|法律意见|意见|进展|完成|结果|实施|期限届满|终止|变更"
        "|更正|补充|修订|出售|B股|前十名",
    ),
}
QUIET_DAYS = 90
TIMEZONE = "Asia/Shanghai"
COLUMNS = ["dataset", "ts_codes", "title", "available_at"]


def matched(title, categories):
    """Boolean mask over `title` (a str Series): the title is one of `categories`."""

    hit = pd.Series(False, index=title.index)
    for name in categories:
        must, never = RULES[name]
        # One pass over every title, then the rest only over the few it keeps.
        first = title[title.str.contains(must[0])]
        keep = ~first.str.contains(never)
        for pattern in must[1:]:
            keep &= first.str.contains(pattern)
        hit |= keep.reindex(title.index, fill_value=False)
    return hit


def titled(context, start, categories):
    """(ts_code, day) of every visible title of `categories` whose day (YYYYMMDD) is on or
    after `start`, one row per name and day, sorted by name and day."""

    # One calendar day of slack: a stamp written in UTC can carry the previous date.
    since = (pd.Timestamp(start) - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = pd.read_parquet(
        context.asof_dir + "/text_index",
        columns=COLUMNS,
        filters=[("dataset", "==", DATASET), ("available_at", ">=", since)],
    )
    rows = rows[matched(rows["title"].fillna("").astype(str), categories)]
    stamp = pd.to_datetime(rows["available_at"], utc=True)
    if (stamp > pd.Timestamp(context.inference_at)).any():
        raise RuntimeError("the as-of text index holds a title stamped after this decision")
    out = pd.DataFrame({
        "ts_code": rows["ts_codes"].fillna("").astype(str).to_numpy(),
        "day": stamp.dt.tz_convert(TIMEZONE).dt.strftime("%Y%m%d").to_numpy(),
    })
    out = out[(out["ts_code"] != "") & (out["day"] >= start)]
    return out.drop_duplicates().sort_values(["ts_code", "day"], kind="stable").reset_index(drop=True)


def events(rows, start):
    """The first title per name after QUIET_DAYS calendar days without one, dated on or
    after `start`. `rows` must reach at least QUIET_DAYS + 1 days before `start`, so that
    every event's quiet period is read, not assumed."""

    gap = pd.to_datetime(rows["day"], format="%Y%m%d").groupby(rows["ts_code"]).diff().dt.days
    first = gap.isna() | (gap > QUIET_DAYS)
    return rows[first & (rows["day"] >= start)].reset_index(drop=True)
