"""Entry: a non-learned northbound-holdings score on the CSI 300 book, with its controls.

`CANDIDATE` selects what this package is; `families.md` is the authority and
`lib/candidates.py` the one place each name is defined:

    "n1"       the candidate: the rank average of the northbound (HKSCC) share
               of float and its change against its own trailing average, from
               the UNION of the two top-10 holder tables (lib/holdings.py).
               Not learned: `fit` does nothing for it
    "c_shuf"   control: n1's values shuffled within the decision day
    "c_base"   control for the register only: the Alpha158 + LightGBM carrier
    "n2"       Family 2: the rank average of the carrier and the holdings
               change, entered only after n1 passed the offline screen

All four trade the same 50-seat monthly equal-cash book (lib/trade.py) on the
same universe (lib/data.py, lib/index.py). Only the carrier legs fit: a
quarterly refit of the booster with warm continuation (lib/primary.py,
lib/state.py) on the carrier's label (lib/labels.py).

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import candidates, data, knobs, labels, primary, state, trade

CANDIDATE = "n1"
REFIT_PERIOD = "quarter"


def fit(context):
    if not candidates.needs_fit(CANDIDATE):
        return
    cold = state.is_cold(context)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    target, realized = labels.rank_target(labels.residual(panel, knobs.HORIZON))
    samples = data.fit_samples(panel, target, realized)
    del panel
    reading = primary.fit(context, samples, cold)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
