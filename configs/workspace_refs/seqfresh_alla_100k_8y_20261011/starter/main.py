"""Entry module: the 100k sequence bag with four seeds per head (c_bag4), and how fresh its training is.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
values may run and in which batch. With every switch at its first value this
package is `c_bag4`:

    b4s4     four seeds each of four structurally different heads -- flat MLP
             on the last bar and the 60-day mean, LSTM, dilated TCN, temporal
             self-attention -- per-member cross-sectional ranks averaged, on
             the CSI 1000 beta-residual label (adjusted prices), refitted
             quarterly on a trailing three-year window, held in a 12-seat
             equal-cash book: weekly review, at most 4 swaps, keep band
             2 x seats, one-lot affordability as a pool filter

The frozen bag is the same with two seeds per head (`b4s2`). The fresh lane
changes only what the heads learn from: the window length
(`knobs.TRAIN_YEARS`) and the weight of a training date by its age
(`knobs.RECENCY_HALFLIFE`), both in `lib/model.py`. Heads, seeds per head,
inputs, label, cadence and book are the baseline's in every leg.

Refitting is quarterly. The first fit of a replay is cold, later ones
warm-start every member from its own checkpoint, and every fourth one resets
cold (`lib/state.py`). The panel is built once per refit and shared by every
member.
"""

from lib import book, knobs, model, panel, state

REFIT_PERIOD = "quarter"


def fit(context):
    knobs.leg()
    cold = state.is_cold(context)
    data = panel.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    reading = model.fit(context, data, cold)
    state.bump(context, cold, reading)


def generate_orders(context):
    return book.run(context)
