"""Entry module: the 100k sequence bag with four seeds per head (c_bag4), and two novel methods on it.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
lane the arm runs, which values may run and in which batch. With every switch
at its first value this package is `c_bag4`:

    b4s4     four seeds each of four structurally different heads -- flat MLP
             on the last bar and the 60-day mean, LSTM, dilated TCN, temporal
             self-attention -- per-member cross-sectional ranks averaged, on
             the CSI 1000 beta-residual label (adjusted prices), refitted
             quarterly on a trailing three-year window, held in a 12-seat
             equal-cash book: weekly review, at most 4 swaps, keep band
             2 x seats, one-lot affordability as a pool filter

The ssm lane changes which heads are in the bag (`knobs.SSM`, a selective
state-space head added or in place of the MLP); the cl lane adds a monthly
update of every member between the quarterly refits (`knobs.UPDATE`, with
the refit cadence below set to monthly). Inputs, label, training window,
quarterly refits and book are the baseline's in every leg.

The first fit of a replay is cold; later quarterly refits warm-start every
member from its own checkpoint and every fourth one resets cold; a monthly
fit inside a quarter is an update (`lib/state.py`). The panel is built once
per fit and shared by every member.
"""

from lib import book, knobs, model, panel, state

REFIT_PERIOD = "quarter"


def fit(context):
    knobs.leg()
    knobs.check_refit(REFIT_PERIOD)
    kind = state.kind(context)
    data = panel.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    if kind == "update":
        reading = model.update(context, data, state.newest(context))
    else:
        reading = model.fit(context, data, kind == "cold")
    state.record(context, kind, reading, model.newest_labelled(context, data))


def generate_orders(context):
    return book.run(context)
