"""The clock lane: at which instants a review's swaps execute.

Fills exist only at 09:30 (the opening call auction's price) and 15:00 (the
closing auction's). A swap sells one name and buys another with the proceeds,
so the swapped money is idle for one segment: the clock chooses which.

    open_close   c_base. A review sells at 09:30 and buys at 15:00 of the same
                 day: the money is idle through the review day's session.
    close_open   A review sells at 15:00; the next trading day buys the empty
                 seats at 09:30 with that cash: the money is idle over the
                 night in between. The buys are ranked that morning, on data
                 through the review day.

Both clocks trade the same seats on the same review calendar: nothing here
changes how many swaps a review makes. A flat account (the first decision of
a replay) buys at once, at 15:00 under open_close and at 09:30 under
close_open. Under close_open a decision is a fill day when the newest visible
trading day was a review day; it buys into the seats that are empty at the
open and no others, so a buy the Broker rejects (a limit-up open, too little
cash) leaves its seat empty until the next review's fill, as c_base leaves a
seat its rejected 15:00 buy did not fill. A review whose previous trading day
was also a review (a week of one trading day) fills at 09:30 and sells at
15:00 in one call; the 15:00 proceeds do not fund that morning's buys.

Everything is stateless: whether yesterday was a review is read from the
as-of view's last two trading days, the same way `lib/book.py` decides
whether today is one.
"""

from datetime import timedelta

import pandas as pd

from lib import knobs

LOOKBACK_CALENDAR_DAYS = 20  # enough visible trading days to see the last two across any holiday
OPEN = (9, 30)
CLOSE = (15, 0)


def _at(decision, hour_minute):
    return decision.replace(hour=hour_minute[0], minute=hour_minute[1], second=0, microsecond=0)


def period(day):
    """The review period a trading day belongs to: its ISO week."""

    if knobs.REVIEW != "week":
        raise ValueError(f"knobs.REVIEW must be 'week', got {knobs.REVIEW!r}")
    return tuple(day.isocalendar())[:2]


def visible_days(context):
    """The visible trading days of the last LOOKBACK_CALENDAR_DAYS, oldest first."""

    start = (context.inference_at - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"],
                           filters=[("trade_date", ">=", start)])
    return [pd.Timestamp(str(day)) for day in sorted(days["trade_date"].astype(str).unique())]


def reviews(days, decision):
    """(is the decision day a review, was the newest visible trading day one), from the visible days.

    A day is a review when it opens a new period against the trading day
    before it; with no visible day the decision is a review.
    """

    if not days:
        return True, False
    today = period(days[-1]) != period(decision)
    yesterday = len(days) >= 2 and period(days[-2]) != period(days[-1])
    return today, yesterday


def instants(decision, flat, review_today, review_yesterday):
    """(sell_at, buy_at) of this decision under knobs.CLOCK; None where that side does not trade.

    open_close keeps the frozen book's rule exactly: a decision after 15:00
    trades nothing, one after 09:30 sells at 15:00 too.
    """

    sell_at, buy_at = _at(decision, OPEN), _at(decision, CLOSE)
    if knobs.CLOCK == "open_close":
        if decision > buy_at or not (flat or review_today):
            return None, None
        if decision > sell_at:
            sell_at = buy_at
        return (None if flat else sell_at), buy_at
    if knobs.CLOCK != "close_open":
        raise ValueError(f"knobs.CLOCK must be 'open_close' or 'close_open', got {knobs.CLOCK!r}")
    morning = _at(decision, OPEN) if decision <= _at(decision, OPEN) else None
    close = _at(decision, CLOSE) if decision <= _at(decision, CLOSE) else None
    if flat:
        return None, morning or close
    return (close if review_today else None), (morning if review_yesterday else None)
