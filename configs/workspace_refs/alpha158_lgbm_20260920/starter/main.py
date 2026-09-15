"""Entry module: Alpha158 + growth / forecast-event + money-flow + listing-age columns into one
LightGBM cross-sectional ranker, refit every quarter on the trailing three years, held as a
weekly-reviewed top-15 book.

`fit` builds the trailing-window panel (lib/data.py), trains the graduate's 4-point grid and
saves the selected booster under context.state_dir (lib/score_lgbm.py). `generate_orders`
reviews the book on the first decision of each ISO week: it scores the newest cross-section
with the same feature builder and swaps at most two names (lib/trade.py).

Variant knobs, one per axis: data.FEATURE_GROUPS and data.TRAIN_YEARS, score_lgbm.LEAVES /
LRS / FIXED_PARAMS, trade.TOP_N / KEEP_BAND / MAX_SWAPS.
"""

from lib import score_lgbm, trade

REFIT_PERIOD = "quarter"


def fit(context):
    score_lgbm.fit(context)


def generate_orders(context):
    return trade.run(context)
