"""Entry module: the graduated Alpha158 + LightGBM primary inside the CSI 300, with a meta-labelling second stage.

The one constant below selects what this package is; `families.md` is the
authority for all four:

    m1          the candidate. The primary ranks the constituents on a
                benchmark-residual 10-day label; a binary classifier trained on
                the primary's own purged out-of-fold top-decile picks predicts
                whether a pick beats that bar's median residual return, and a
                pick below the probability floor is passed over for the next
                primary rank that clears it. Equal cash (lib/meta.py,
                lib/trade.py)
    c_primary   the required control: the same primary, the same label, the
                same universe, the same seats and cadence, with the second
                stage removed. Never nominated
    m_size      the same gate used to SIZE rather than to filter; not equal
                cash, so it owes a same-score-equal-cash leg in its own batch
    m_stack     m1 with one extra meta feature, the rank of a horizon twin
                primary fitted on the 20-day residual label. Same family, a
                different horizon -- deliberately not a second alpha family,
                because every obvious one is on the reopen ban list

Everything else is shared: the constituent panel and its Alpha158 columns
(lib/data.py, lib/features.py), the residual label and the median outcome
(lib/labels.py), the embargoed split and the purged folds (lib/common.py), the
primary (lib/primary.py), the sections (lib/index.py) and the 12-seat weekly
book (lib/trade.py). Changing the constant changes nothing else, which is what
makes the control a control.

Refitting is quarterly. The first fit of a replay is cold because the state
directory starts empty; every later one continues the boosters it saved, and
every fourth one resets cold so a chain cannot drift forever (lib/state.py).

Knobs, one per registered axis: trade.SEATS / INDUSTRY_CAP / MAX_SWAPS /
P_FLOOR / PICK_POOL, meta.TOP_SHARE / BASE_FEATURES, common.OOF_FOLDS,
data.TRAIN_YEARS, primary.PARAMS, state.COLD_EVERY.
"""

from lib import meta, state, trade

CANDIDATE = "m1"
REFIT_PERIOD = "quarter"


def fit(context):
    cold = state.is_cold(context)
    reading = meta.fit(context, cold, CANDIDATE)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
