"""The three pre-registered exclusion screens, and the placebo that replaces them.

Each screen answers one question at the review date, on rows that were visible
then, and returns the codes it takes out of the screened pool. They are the only
thing this arm registers: the pool, the score and the book below them are the
sibling pack's, unchanged.

    audit    -- the name's latest visible audit opinion, for the most recent
                period it has one, is not an unqualified opinion
    unlock   -- the name has announced unlocks of restricted shares whose dates
                fall inside the next UNLOCK_HORIZON calendar days and whose
                float ratios add up to UNLOCK_MIN percent of total shares
    segment  -- the name sits in the most revenue-concentrated MAINBZ_SCREEN
                fraction of the pool by its largest product segment's share of
                the latest visible annual breakdown
    placebo  -- `c_placebo` only: one permanent draw per code, no data at all

Three rules hold for all of them and are the reason the arm can be read:

* Every screen EXCLUDES FLAGGED NAMES. It never admits only the names carrying a
  clean reading. A missing audit row is almost always a name whose opinion was
  never dated, not a clean opinion; a missing unlock row may be a gap in a
  source the data documentation still calls capped. Turning absence into a
  verdict would screen on coverage instead of on risk.
* Every screen reads its own panel with a bounded `columns=`/`filters=` read and
  judges visibility on the row-level `available_at` alone (`face.visible_rows`).
* Every screen is evaluated on the pool AFTER the pool floor and the accrual
  screen, so its removal count is a fraction of the names the book chooses from.
  `lib.trade` puts that count on every buy order.

Units and version rules are in `pit-field-map.md`; the traps that decide whether
these columns mean anything at all are repeated here, next to the code that
depends on them.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import face

# audit: the opinions that are not an unqualified opinion. An unqualified
# opinion with an emphasis-of-matter paragraph ("带强调事项段的无保留意见") is
# still unqualified and is NOT flagged at the starter value; adding it is a
# registered variant axis, not a judgement call at the console.
AUDIT_FLAGGED = ("保留意见", "无法表示意见", "否定意见")
AUDIT_PERIOD_DAYS = 900       # calendar days of audit periods read: two annual periods and a margin

# unlock: how far ahead a disclosed unlock date counts, and how much supply is
# enough. `float_ratio` is a PERCENT of total shares, so UNLOCK_MIN = 3 means
# three percent, not three tenths of a percent.
UNLOCK_HORIZON = 90           # calendar days ahead of the review
UNLOCK_MIN = 3.0              # summed percent of total shares

# segment: the product breakdown, and the fraction of the rankable names the
# screen removes from its concentrated end.
MAINBZ_CODE = "P"             # bz_code: P product, D region, I industry -- one dataset, three breakdowns
MAINBZ_AGGREGATE = ("产品", "合计特别调整")  # the vendor's own total row and its adjustment row, never segments
MAINBZ_SCREEN = 0.1
MAINBZ_PERIOD_DAYS = 900

# placebo: a permanent per-code draw whose rate matches the measured fraction
# the three screens remove together (`sources.md`). One draw per code, never per
# day: re-drawing would turn a choice-set control into a turnover control.
PLACEBO_RATE = 0.135
PLACEBO_SEED = 1618033988
CODE_SPACE = 1_000_000        # A-share numeric codes


def excluded(context, section, names):
    """Codes the named screens take out of `section`, and one count per screen.

    The union is what the book drops; the per-screen counts are reported on
    every buy order, because a screen that removes a large slice of the pool is
    not testing a tail -- it is a different construction.
    """
    drop = set()
    hits = {}
    for name in names:
        flagged = _SCREENS[name](context, section)
        hits[name] = len(flagged)
        drop |= flagged
    return drop, hits


def _audit(context, section):
    """Names whose latest visible opinion, for their most recent audited period,
    is not an unqualified one."""
    start = (context.inference_at - timedelta(days=AUDIT_PERIOD_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "audit_result"],
        filters=[("dataset", "=", "fina_audit"), ("end_date", ">=", start)],
    )
    frame = frame[frame["dataset"] == "fina_audit"]
    frame = face.visible_rows(frame, context).copy()
    if frame.empty:
        return set()
    frame["end_date"] = frame["end_date"].astype(str)
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    # the most recent period the name has an opinion for, then the last word on
    # that period: a corrected opinion replaces the one it corrects.
    frame = frame[frame["end_date"] == frame.groupby("ts_code")["end_date"].transform("max")]
    frame = frame.sort_values(["stamp", "ts_code"]).drop_duplicates("ts_code", keep="last")
    flagged = frame.loc[frame["audit_result"].isin(AUDIT_FLAGGED), "ts_code"]
    return set(flagged) & set(section["ts_code"])


def _unlock(context, section):
    """Names with announced unlocks inside the horizon adding up to UNLOCK_MIN percent.

    `float_date` is a DISCLOSED FUTURE date, so reading it forward is legitimate
    -- the schedule was published with the announcement. The supply figure is
    the hard part, and the panel carries the same lot more than once in two
    different ways, each of which alone reads a single holding as more than the
    whole company:

    * one scheduled unlock is re-announced every year until it happens, each
      time with the share count restated for the capital changes since and
      sometimes with the date moved by a day, so a lot sits in the panel three
      to five times under keys that differ. An announcement supersedes the
      announcements before it, so only each holder's LATEST one counts;
    * the panel is a union of two downloads, and one announcement then carries
      one holding twice under two neighbouring dates -- once with the share
      count restated and the ratio unchanged, once the other way round. Neither
      column identifies the lot on its own.

    So one holder's restricted position releasing inside the horizon is ONE row:
    its latest announcement, of that the largest share count, and of ties the
    smallest ratio -- the one measured against the largest, that is the most
    recent, total share count. The residual error is one-sided and conservative
    for a screen that excludes: a holder releasing two genuine tranches inside
    the same horizon is counted once, and a tranche an older announcement
    carried and the newest one does not mention is not counted at all.
    """
    decision = pd.Timestamp(context.inference_at)
    start = decision.strftime("%Y%m%d")
    end = (decision + timedelta(days=UNLOCK_HORIZON)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=["dataset", "available_at", "ts_code", "float_date", "float_share", "float_ratio",
                 "share_type", "holder_name"],
        filters=[("dataset", "=", "share_float_complete"),
                 ("float_date", ">=", start), ("float_date", "<=", end)],
    )
    frame = frame[frame["dataset"] == "share_float_complete"]
    frame = face.visible_rows(frame, context).copy()
    if frame.empty:
        return set()
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    holder = ["ts_code", "holder_name", "share_type"]
    frame = frame[frame["stamp"] == frame.groupby(holder)["stamp"].transform("max")]
    frame = frame.sort_values(["float_share", "float_ratio"], ascending=[False, True])
    frame = frame.drop_duplicates(holder, keep="first")
    supply = frame.groupby("ts_code")["float_ratio"].sum()
    return set(supply.index[supply >= UNLOCK_MIN]) & set(section["ts_code"])


def _segment(context, section):
    """The MAINBZ_SCREEN most revenue-concentrated names of the pool.

    Concentration is the largest product segment's share of the name's latest
    visible annual product breakdown. The trap is that the panel carries its own
    total in a row whose `bz_item` is the breakdown's name ("产品"), alongside an
    adjustment row: counting those as segments makes every denominator twice the
    truth and every concentration about 0.5, which is a screen on nothing.

    A name without a visible breakdown cannot be ranked and is never removed, so
    this screen takes at most MAINBZ_SCREEN of the pool and usually less.
    """
    start = (context.inference_at - timedelta(days=MAINBZ_PERIOD_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "bz_code", "bz_item", "bz_sales"],
        filters=[("dataset", "=", "fina_mainbz_vip"), ("end_date", ">=", start)],
    )
    frame = frame[(frame["dataset"] == "fina_mainbz_vip") & (frame["bz_code"] == MAINBZ_CODE)]
    frame = face.visible_rows(frame, context).copy()
    if frame.empty:
        return set()
    frame["end_date"] = frame["end_date"].astype(str)
    frame = frame[frame["end_date"].str.endswith("1231")]
    frame = frame[~frame["bz_item"].isin(MAINBZ_AGGREGATE) & (frame["bz_sales"] > 0)]
    if frame.empty:
        return set()
    frame = frame[frame["end_date"] == frame.groupby("ts_code")["end_date"].transform("max")]
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame.sort_values("stamp").drop_duplicates(["ts_code", "end_date", "bz_item"], keep="first")
    share = frame["bz_sales"] / frame.groupby("ts_code")["bz_sales"].transform("sum")
    concentration = share.groupby(frame["ts_code"]).max().reindex(section["ts_code"]).dropna()
    return set(concentration.nlargest(int(len(concentration) * MAINBZ_SCREEN)).index)


def _placebo(context, section):
    """A permanent per-code exclusion of the same expected size, from no data.

    Drawn once per A-share numeric code and never per decision: what separates
    "these names carry uncompensated risk" from "a smaller choice set is enough"
    is a control whose exclusions are as stable as the screens', not one that
    re-draws a fresh tenth of the pool every quarter. The seed must not move,
    between legs of a batch or between rounds.
    """
    table = np.random.default_rng(PLACEBO_SEED).random(CODE_SPACE)
    numeric = pd.to_numeric(section["ts_code"].str.slice(0, 6), errors="coerce").fillna(0).astype("int64")
    drawn = table[numeric.to_numpy() % CODE_SPACE]
    return set(section["ts_code"][drawn < PLACEBO_RATE])


_SCREENS = {"audit": _audit, "unlock": _unlock, "segment": _segment, "placebo": _placebo}
