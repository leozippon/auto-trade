"""Entry module: the 100k sequence bag on its frozen book (c_base), and one lane of book changes at a time.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
values may run and in which batch. With every switch at its first value this
package is `c_base`:

    b4s2     two seeds each of four structurally different heads -- flat MLP
             on the last bar and the 60-day mean, LSTM, dilated TCN, temporal
             self-attention -- per-member cross-sectional ranks averaged, on
             the CSI 1000 beta-residual label (adjusted prices), held in a
             12-seat equal-cash book: weekly review, at most 4 swaps, keep
             band 2 x seats, one-lot affordability as a pool filter

The bag lane changes the seeds per head (`lib/model.py`), the cost lane how
the book follows the score (`lib/book.py`: swaps, cadence, keep band, score
smoothing), the pool lane what the book may buy (`lib/pool.py`). The model,
its inputs and its label are the baseline's in every leg: the panel
(`lib/panel.py`), the label (`lib/label.py`) and the cold-reset counter
(`lib/state.py`) do not change.

Refitting is quarterly on a trailing three-year window. The first fit of a
replay is cold, later ones warm-start every member from its own checkpoint,
and every fourth one resets cold -- the recipe's cadence, the same for every
leg. The panel is built once per refit and shared by every member.
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
