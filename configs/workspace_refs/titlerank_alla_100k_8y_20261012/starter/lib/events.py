"""What `fit` does each decision day: refit in a new quarter, record the events of a review.

`main.py` declares a daily `fit`, so this runs before every decision, with the
decision's own context. It refits the leg's model at the first decision of
each calendar quarter, and of the replay (`lib/ranker.py`; c_rule fits
nothing). On a review day (`lib/clock.py`) it then records that review's
events, the names the book may buy today:

- nn and ridge legs: the units fresh at this review that score at or above
  the threshold of the model in force, with their scores;
- c_rule: the incentive-plan drafts fresh at this review (`lib/titles.py`),
  each scored minus its visible day, so the book takes them oldest first,
  ties by code, exactly as the incdraft book does.

Each review's events go to their own file under `context.state_dir`, and the
review's date, the units scored and the events kept are appended to the
index. A review is scored once, by the model in force that day, and never
rescored: `lib/book.py` reads these records, so a holding keeps the event it
was bought on through later refits, and a cold worker replays the same book.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import clock, knobs, ranker, titles

FIT_RECORD = "/fit_record.npy"
INDEX = "/events_index.npy"
EVENTS = "/events_{}.parquet"
# Calendar days before today from which the drafts of today's review are read.
RULE_DAYS = 30


def update(context):
    knobs.leg()
    today = clock.decision(context)
    quarter = today.year * 4 + (today.month - 1) // 3
    if knobs.MODEL != "rule" and quarter != _last_quarter(context):
        record = ranker.train(context)
        np.save(context.state_dir + FIT_RECORD, np.array([
            quarter, int(today.strftime("%Y%m%d")), record["train_units"], record["valid_units"],
            record["train_reviews"], record["setting"], record["threshold"],
        ], dtype=np.float64))
    if not clock.is_review(context):
        return
    events, scored = _drafts(context, today) if knobs.MODEL == "rule" else ranker.fresh(context)
    day = int(today.strftime("%Y%m%d"))
    events.to_parquet(context.state_dir + EVENTS.format(day), index=False)
    try:
        index = np.load(context.state_dir + INDEX)
    except FileNotFoundError:
        index = np.zeros((0, 3), dtype=np.int64)
    np.save(context.state_dir + INDEX, np.vstack([index, [[day, scored, len(events)]]]).astype(np.int64))


def _last_quarter(context):
    try:
        return int(np.load(context.state_dir + FIT_RECORD)[0])
    except FileNotFoundError:
        return None


def _drafts(context, today):
    first_day = (today - timedelta(days=RULE_DAYS)).strftime("%Y%m%d")
    quiet_day = (today - timedelta(days=RULE_DAYS + titles.QUIET_DAYS + 1)).strftime("%Y%m%d")
    days = clock.calendar(context, first_day)
    drafts = titles.events(titles.drafts(context, quiet_day), first_day)
    entry, visible = clock.entries(days, clock.reviews(days), drafts["day"].to_numpy(dtype=object))
    fresh = drafts[entry == len(days) - 1]
    out = pd.DataFrame({"ts_code": fresh["ts_code"].to_numpy(), "score": -visible[entry == len(days) - 1].astype(float)})
    return out, len(out)
