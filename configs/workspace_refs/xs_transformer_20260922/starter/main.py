"""Entry: market-gated cross-sectional attention on the CSI 300 constituents.

`CANDIDATE` selects what this package is; `families.md` is the authority:

    "x1"          a temporal encoder over each name's WINDOW_DAYS window,
                  market-gated, then one attention layer ACROSS the names of
                  that date                                  main candidate
    "c_temporal"  the same encoder and head with the cross-sectional
                  attention off: what attending across names adds
                                                         mechanism control
    "c_lgbm"      LightGBM on the same features, label, universe and training
                  dates                          reported control, never nominated

All three share the panel (lib/panel.py), the constituent universe and
training dates, the label, and the equal-cash book with its keep band and
swap cap (lib/trade.py). `x1` and `c_temporal` train on CUDA and have no CPU
path; `c_lgbm` runs on the container's CPUs.

Every other default the session is expected to change lives in lib/knobs.py,
each one exactly once.
"""

from lib import control, model, trade

CANDIDATE = "x1"
REFIT_PERIOD = "quarter"


def fit(context):
    if CANDIDATE == "c_lgbm":
        control.fit(context)
    elif CANDIDATE in ("x1", "c_temporal"):
        model.fit(context, CANDIDATE == "x1")
    else:
        raise ValueError(f"unknown candidate: {CANDIDATE}")


def generate_orders(context):
    return trade.run(context, CANDIDATE)
