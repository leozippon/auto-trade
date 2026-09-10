"""Entry module: intraday signed order-flow ranker with the moneyflow hard-gate switch.

`CANDIDATE` selects which pre-registered leg this package is. It maps to
(feature face, scorer) in `lib.flow.CANDIDATES`:

    "o1"      -> ofi face, equal-weight rank composite  (main candidate, no learner)
    "o2"      -> ofi face, constrained LightGBM         (does learning add anything?)
    "c_mf"    -> moneyflow face, the same LightGBM      (hard attribution control)
    "c_mf_ew" -> moneyflow face, the same EW composite  (hard control for O1)

The four differ in nothing else: same pool, same label, same 20-day cadence,
same topk=15 basket, same score-level neutralization. One line changes; the
comparison is the batch. A control leg is never nominated.
"""

from lib import model
from lib import trade

REFIT_PERIOD = "quarter"
CANDIDATE = "o1"


def fit(context):
    return model.fit(context, CANDIDATE)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
