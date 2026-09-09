"""Entry module: events-feature cross-sectional ranker with the attribution control switch.

`CONTROL` selects which pre-registered variant this package is:

    "events" -> main candidate, 53 columns (daily price/volume + five event families)
    "pv"     -> C-PV attribution control, the same learner on the 14 daily columns only
    "ew"     -> C-EW control, the same event features with no learner at all

The three must differ in nothing else: same pool, same label, same cadence,
same basket rules, same neutralization. Change one line, validate, compare.
"""

from lib import model
from lib import trade

REFIT_PERIOD = "quarter"
CONTROL = "events"


def fit(context):
    if CONTROL == "ew":
        return None
    return model.fit(context, CONTROL == "pv")


def generate_orders(context):
    return trade.run(context, CONTROL == "pv", CONTROL != "ew")
