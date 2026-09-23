"""Entry: the confirmed Alpha158 + LightGBM carrier, with only its construction moved.

`CANDIDATE` selects what this package is; `families.md` is the authority and
`knobs.LEGS` the one place each name is defined:

    "c_base"   the carrier's monthly book exactly as the head packs trade it:
               first decision of each calendar month, at most 8 names changed
               per review, a holding kept while ranked inside the top 100.
               The required control, always the first leg of a batch, never
               nominated
    "f1"       the same score in a book that tracks it: no swap cap, a
               holding kept only while ranked inside the top 60, reviewed on
               the registered cadence (from the score's measured IC decay),
               and empty seats refilled between reviews. The H2 candidate
    "f2"       f1 plus a greedy pull of beta, size, volatility and industry
               back toward the pool (no optimiser). The H3 candidate, in the
               same first batch as f1 and c_base
    "f3"       f1 with the label horizon matched to f1's realised holding
               period; conditional
    "f4"       f1 with EPO weights at entry instead of equal cash;
               conditional, only after f1 has beaten c_base

Every leg shares the panel (lib/data.py), the universe (lib/index.py), the
label (lib/labels.py), the split (lib/common.py), the refit counter
(lib/state.py) and the booster (lib/primary.py); for all but f3 the fitted
booster is the same file. Only lib/book.py and lib/trade.py read the leg.

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import data, knobs, labels, primary, state, trade

CANDIDATE = "c_base"
REFIT_PERIOD = "quarter"


def fit(context):
    horizon = knobs.LEGS[CANDIDATE]["horizon"]
    cold = state.is_cold(context)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    target, realized = labels.rank_target(labels.residual(panel, horizon))
    samples = data.fit_samples(panel, target, realized)
    del panel
    reading = primary.fit(context, samples, cold, horizon)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
