"""Entry: a relational graph ranker on the CSI 300 constituent cross-section.

`CANDIDATE` selects what this package is; `families.md` is the authority:

    "g1"         relational message passing over an industry co-membership
                 graph and a return-correlation kNN         main candidate
    "c_nograph"  the same encoder and head with the message passing off:
                 what the graph itself adds             mechanism control
    "c_lgbm"     LightGBM on the same features, label, universe and training
                 dates                          reported control, never nominated

All three share the panel (lib/panel.py), the constituent universe and
training dates, the label, and the equal-cash book with its keep band and
swap cap (lib/trade.py). `g1` and `c_nograph` train on CUDA and have no CPU
path; `c_lgbm` runs on the container's CPUs.

Every other default the session is expected to change lives in lib/knobs.py,
each one exactly once.
"""

from lib import control, model, trade

CANDIDATE = "g1"
REFIT_PERIOD = "quarter"


def fit(context):
    if CANDIDATE == "c_lgbm":
        control.fit(context)
    elif CANDIDATE in ("g1", "c_nograph"):
        model.fit(context, CANDIDATE == "g1")
    else:
        raise ValueError(f"unknown candidate: {CANDIDATE}")


def generate_orders(context):
    return trade.run(context, CANDIDATE)
