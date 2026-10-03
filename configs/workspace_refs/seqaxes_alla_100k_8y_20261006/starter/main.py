"""Entry module: the 100k sequence bag (c_base) and one axis of it at a time -- the label or the fundamentals.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
values may run and in which batch. With `LABEL = "base"` and `FUND = "off"`
this package is the 100k bag on its settled book, `c_base`:

    b4s2     two seeds each of four structurally different heads -- flat MLP
             on the last bar and the 60-day mean, LSTM, dilated TCN, temporal
             self-attention -- per-member cross-sectional ranks averaged, on
             the CSI 1000 beta-residual label (adjusted prices), optionally
             residualised against the tree (`RESIDUALISE`)

The label lane changes only the training target (`lib/label.py`): `bench`,
`size` or `hold`. The fundamentals lane adds point-in-time fundamentals
(`lib/fusion.py`): `late` blends a fundamentals tree's rank into the bag's,
`early` joins the features to every head's last layer, and `FUND_PERM` turns
either into its control `c_perm`. Everything else is shared by every leg: the
panel (`lib/panel.py`), the trees (`lib/tree.py`), the cold-reset counter
(`lib/state.py`) and the equal-cash book with its one-lot pool filter
(`lib/book.py`).

Refitting is quarterly on a trailing three-year window. The first fit of a
replay is cold, later ones warm-start every member from its own checkpoint,
and every fourth one resets cold -- the recipe's cadence, the same for every
leg. The panel is built once per refit and shared by every learner.
"""

from lib import book, fusion, knobs, model, panel, state, tree

REFIT_PERIOD = "quarter"


def fit(context):
    knobs.leg()
    cold = state.is_cold(context)
    data = panel.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    fx = None if knobs.FUND == "off" else fusion.features(context, data)
    reading = model.fit(context, data, cold, fx if knobs.FUND == "early" else None)
    if knobs.RESIDUALISE:
        tree.fit(context, data, cold)
    if knobs.FUND == "late":
        tree.fit_fund(context, data, fx)
    state.bump(context, cold, reading)


def generate_orders(context):
    return book.run(context)
