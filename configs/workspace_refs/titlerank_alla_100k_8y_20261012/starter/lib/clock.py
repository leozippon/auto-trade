"""The book's clock: trading days, weekly reviews and when a title becomes fresh.

A review is the first decision of an ISO week: the newest visible trading day
(T-1) lies in an earlier week. Nothing trades on other days. A title dated D
(the local date of its `available_at`) is counted from its visible day, the
first trading day after D, and is fresh at the first review after that day:
at a review, the titles whose visible day falls on or after the previous
review and before today. In the research window a title dated D first reaches
the 08:30 decision of calendar day D + 2, never later than one trading day
after its visible day, so the review that takes it always sees it. These are
the incdraft lane's rules, shared here by `lib/events.py` (which scores the
fresh titles) and `lib/book.py` (which trades them).
"""

from datetime import timedelta

import numpy as np
import pandas as pd

TIMEZONE = "Asia/Shanghai"


def decision(context):
    return pd.Timestamp(context.inference_at).tz_convert(TIMEZONE)


def is_review(context):
    """True when this decision is the first of its ISO week."""

    today = decision(context)
    start = (today - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        return True
    latest = pd.Timestamp(str(days["trade_date"].max()))
    return tuple(latest.isocalendar())[:2] != tuple(today.isocalendar())[:2]


def calendar(context, start):
    """Visible trading days from `start` (YYYYMMDD), plus the decision day, as a sorted array."""

    rows = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    return np.array(sorted({*rows["trade_date"].astype(str), decision(context).strftime("%Y%m%d")}), dtype=object)


def reviews(days):
    """Positions in `days` of the first trading day of each ISO week (the first day itself excluded)."""

    weeks = [tuple(pd.Timestamp(day).isocalendar())[:2] for day in days]
    return np.array([i for i in range(1, len(days)) if weeks[i] != weeks[i - 1]], dtype=int)


def entries(days, review_positions, dates):
    """For each date (YYYYMMDD), the position in `days` of the review at which a title
    of that date is fresh, and its visible day's position; -1 when that review lies
    beyond `days`."""

    visible = np.searchsorted(days, np.asarray(dates, dtype=object), side="right")
    slot = np.searchsorted(review_positions, visible, side="right")
    entry = np.full(len(visible), -1, dtype=int)
    inside = slot < len(review_positions)
    entry[inside] = review_positions[slot[inside]]
    return entry, visible
