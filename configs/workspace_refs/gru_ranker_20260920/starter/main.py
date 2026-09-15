"""Entry module: the GRU sequence ranker, its reported tree control and the ensemble variant.

`CANDIDATE` selects what this package is (families.md is the authority):

    "g1"      -> three fixed-seed GRUs over 60-day daily price-volume sequences,
                 trained on a CUDA device (lib/model.py)            (main candidate)
    "c_lgbm"  -> the frozen artifact's Alpha158 + LightGBM ranker (lib/tree.py)
                                                                     (reported control, never nominated)
    "v_ens"   -> the mean of the g1 and c_lgbm ranks                (registered variant)

All three share everything else: the panels (lib/panel.py), the trailing
three-year quarterly refit dates, the pool and the weekly top-15 book with its
keep band and swap cap (lib/trade.py). `g1` and `v_ens` have no CPU path.
"""

from lib import model, trade, tree

CANDIDATE = "g1"
REFIT_PERIOD = "quarter"


def fit(context):
    if CANDIDATE in ("g1", "v_ens"):
        model.fit(context)
    if CANDIDATE in ("c_lgbm", "v_ens"):
        tree.fit(context)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
