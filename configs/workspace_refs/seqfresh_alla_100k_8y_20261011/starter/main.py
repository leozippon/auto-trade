"""Entry module: the 100k sequence bag on its frozen book (c_base), and how fresh its training is.

`lib/knobs.py` holds every switch but the refit cadence below; `refs/README.md`
is the authority for which values may run and in which batch. With every
switch at its first value and the cadence quarterly this package is `c_base`:

    b4s2     two seeds each of four structurally different heads -- flat MLP
             on the last bar and the 60-day mean, LSTM, dilated TCN, temporal
             self-attention -- per-member cross-sectional ranks averaged, on
             the CSI 1000 beta-residual label (adjusted prices), refitted
             quarterly on a trailing three-year window, held in a 12-seat
             equal-cash book: weekly review, at most 4 swaps, keep band
             2 x seats, one-lot affordability as a pool filter

The fresh lane changes only what the heads learn from and when: the window
length (`knobs.TRAIN_YEARS`), the weight of a training date by its age
(`knobs.RECENCY_HALFLIFE`), both in `lib/model.py`, and the refit cadence,
which is `REFIT_PERIOD` here because the host reads it from this file as a
literal: "quarter" (c_base) or "month". Heads, inputs, label and book are the
baseline's in every leg.

The first fit of a replay is cold, later ones warm-start every member from
its own checkpoint, and the cold reset comes back once a year on either
cadence (`lib/state.py`), so a monthly leg's cold refits fall on c_base's. The
panel is built once per refit and shared by every member.
"""

from lib import book, knobs, model, panel, state

REFIT_PERIOD = "quarter"


def fit(context):
    knobs.leg(REFIT_PERIOD)
    cold = state.is_cold(context, REFIT_PERIOD)
    data = panel.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    reading = model.fit(context, data, cold)
    state.bump(context, cold, reading)


def generate_orders(context):
    return book.run(context, REFIT_PERIOD)
