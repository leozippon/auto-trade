"""Entry: the confirmed Alpha158 + LightGBM carrier, with its learner moved and nothing else.

`CANDIDATE` selects what this package is; `families.md` is the authority and
`primary.LEGS` the one place each name is defined:

    "c_base"   the carrier alone -- 158 Alpha operators, the benchmark-residual
               forward rank label, the constituent universe, one LightGBM L2
               booster, quarterly refits with warm continuation. The required
               control, always the first leg of a batch, never nominated
    "de1"      the same features, label, split and parameter point, trained as
               a DoubleEnsemble (lib/double_ensemble.py): sub-models reweighted
               by their loss curves and fed shuffle-selected features. Cold on
               every refit by construction
    "lr1"      the carrier with its L2 objective replaced by lambdarank, one
               query per decision bar, relevance = the same rank target cut
               into integer grades. Keeps the carrier's warm chain
    "de_lr"    both moves at once; registered as its own leg, run only after
               de1 and lr1 have each beaten c_base
    "c_cold"   the carrier refitted cold every quarter. A diagnostic that
               separates "DoubleEnsemble helps" from "dropping the warm chain
               helps"; never nominated

All share the panel (lib/data.py), the universe and the training dates
(lib/index.py), the label (lib/labels.py), the split (lib/common.py), the refit
counter (lib/state.py) and the 50-seat monthly book (lib/trade.py). Only
lib/primary.py and lib/double_ensemble.py read the candidate.

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import data, knobs, labels, primary, state, trade

CANDIDATE = "c_base"
REFIT_PERIOD = "quarter"


def fit(context):
    cold = state.is_cold(context) or not primary.warm(CANDIDATE)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    target, realized = labels.rank_target(labels.residual(panel, knobs.HORIZON))
    samples = data.fit_samples(panel, target, realized)
    del panel
    reading = primary.fit(context, samples, cold, CANDIDATE)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
