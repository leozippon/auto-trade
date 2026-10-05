"""Which kind of fit this is: a cold refit, a warm refit, or a monthly update.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so one small record there answers the question each fit
asks, and a replay stays reproducible from PIT data alone.

Quarterly refits are c_bag4's: the first fit of a replay is cold, and after
it every `COLD_EVERY`-th quarterly refit starts from a fresh initialisation
(a yearly reset) while the others warm-start every member from its
checkpoint. A fit in a new calendar quarter is a quarterly refit. With
`REFIT_PERIOD = "month"` (the cl lane) the host also calls `fit` at the other
two month starts of each quarter; a fit in the same quarter as the last
quarterly refit is a monthly update (`lib/model.py`). With a quarterly cadence
every fit after the first falls in a new quarter, so the schedule is exactly
c_bag4's, and every leg of either lane resets cold on the same refits as its
c_bag4: a control that refits cold while the candidate warm-starts is not a
control.

The record, one float array: [quarterly refits done, kind of the last fit
(1 cold, 0 warm, -1 update), its reading, the calendar quarter of the last
quarterly refit (year x 4 + quarter index), the newest labelled date the last
fit could see (YYYYMMDD)].
"""

import numpy as np

COLD_EVERY = 4               # quarterly refits between two cold starts: a yearly reset

META_FILE = "/refit_meta.npy"
KINDS = {"cold": 1.0, "warm": 0.0, "update": -1.0}


def _path(context):
    return context.state_dir + META_FILE


def _meta(context):
    """The record of the previous fit of this replay, or None before the first."""

    try:
        return np.load(_path(context))
    except FileNotFoundError:
        return None


def _quarter(context):
    day = context.inference_at
    return day.year * 4 + (day.month - 1) // 3


def kind(context):
    """"cold", "warm" or "update" for the fit about to run."""

    meta = _meta(context)
    if meta is None:
        return "cold"
    if int(meta[3]) == _quarter(context):
        return "update"
    return "cold" if int(meta[0]) % COLD_EVERY == 0 else "warm"


def newest(context):
    """The newest labelled date (YYYYMMDD) the previous fit of this replay could see."""

    return int(_meta(context)[4])


def record(context, kind, reading, newest):
    """Record the completed fit: an update keeps the quarterly count and quarter it follows."""

    meta = _meta(context)
    done = 0 if meta is None else int(meta[0])
    if kind == "update":
        quarter = int(meta[3])
    else:
        done, quarter = done + 1, _quarter(context)
    np.save(_path(context), np.asarray([done, KINDS[kind], float(reading), quarter, newest], dtype=np.float64))
