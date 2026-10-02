"""Entry module: the rank-average bag of four [FEAT-14] heads on a 100k book, and its two controls.

`knobs.CANDIDATE` selects what this package is; `refs/README.md` is the
authority for which values may run and in which batch:

    b4s2        two seeds each of four structurally different heads -- flat MLP
                on the last bar and the 60-day mean, LSTM, dilated TCN,
                temporal self-attention -- per-member cross-sectional ranks
                averaged                                         the candidate
    c_single    the MLP alone, one seed: the bag's own first MLP member, so the
                bag against it is "what the other seven members add"
                                                                 control, never nominated
    c_lgbm      LightGBM on the same window tensor, flattened    control, never nominated

`lib/knobs.py` holds every switch (candidate, residualising against the tree,
the seed base, the book's seats, swap cap and review, and the label index the
session reconciles). `lib/model.py` holds the heads and the one training loop;
`lib/tree.py` the tree. Everything else is shared by every candidate: the
panel (`lib/panel.py`), the CSI 1000 beta-residual label on adjusted prices
(`lib/label.py`), the cold-reset counter (`lib/state.py`) and the equal-cash
book with its one-lot pool filter (`lib/book.py`). Changing the candidate
changes nothing else, which is what makes the controls controls.

Refitting is quarterly on a trailing three-year window. The first fit of a
replay is cold, later ones warm-start every member from its own checkpoint,
and every fourth one resets cold -- the recipe's cadence, the same for every
candidate. The panel is built once per refit and shared by every learner.
"""

from lib import book, knobs, model, panel, state, tree

REFIT_PERIOD = "quarter"


def fit(context):
    cold = state.is_cold(context)
    data = panel.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    reading = 0.0
    if knobs.CANDIDATE != "c_lgbm":
        reading = model.fit(context, data, cold, knobs.CANDIDATE)
    if knobs.CANDIDATE == "c_lgbm" or knobs.RESIDUALISE:
        control = tree.fit(context, data, cold)
        reading = control if knobs.CANDIDATE == "c_lgbm" else reading
    state.bump(context, cold, reading)


def generate_orders(context):
    return book.run(context)
