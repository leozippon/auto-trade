"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module and no docstring
restates a value from this file; they refer to these names only.

Three names live outside this file on purpose:

    CANDIDATE / REFIT_PERIOD    the loader reads them off `main.py`
    index.INDEX_CODE            the benchmark this book is built in and graded
                                against; it belongs beside the read that uses
                                it, and the run facts override it

Groups, in the order a session usually touches them:

    label       what the primary is asked to predict
    primary     the carrier: Alpha158 + LightGBM, the graduated lineage
    block       the new information family this arm is here to price
    state       warm continuation and the cold-reset cadence
    book        how a score becomes a basket this account can hold
"""

# --- label ---------------------------------------------------------------
# Forward trading days of the primary label. Registered in `hypothesis`
# before the first batch; changing it is a new registration, not a variant.
HORIZON = 10
# Trailing OLS beta of the label's benchmark leg: window, minimum window,
# shrinkage toward 1, and the clip applied after shrinking.
BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)

# --- primary -------------------------------------------------------------
# Trailing years of the rolling training window, and how its tail is split
# into a validation segment with an embargo the width of the label.
TRAIN_YEARS = 3
VALID_DAYS = 60
EMBARGO_DAYS = 10
MIN_SAMPLES = 200
# One registered parameter point, not a grid: a grid buys fit seconds, and
# every extra validated revision raises this arm's own deflated-Sharpe bar.
PARAMS = {
    "objective": "regression",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    # Pinned: the container has 8 CPUs and a three-way batch shares the host.
    "num_threads": 8,
    "seed": 7,
    "verbose": -1,
}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50

# --- block ---------------------------------------------------------------
# Rolling windows of the intraday-statistics block, in trading days. The short
# one is the weekly horizon the realized-jump evidence was measured at; the
# long one is about a month, the horizon of the China realized-skew evidence.
ISTATS_SHORT = 5
ISTATS_LONG = 20
# A stock-day whose statistics rest on fewer session-grid bars than this (a
# gap in the minute layer, a partial session), or that has no realized
# variance (a board sealed from the open), is masked to NaN before any rolling
# window (`pit-field-map.md`).
ISTATS_MIN_BARS = 200
# Calendar days of the events read. Must cover ISTATS_LONG trading days plus
# the warm-up the fit window needs; holidays make the ratio worse than 7/5.
ISTATS_LOOKBACK_DAYS = 120

# --- state ---------------------------------------------------------------
# Refits between two cold starts: with a quarterly refit, every fourth one
# starts from a fresh booster so a warm chain cannot run a whole replay.
COLD_EVERY = 4

# --- book ----------------------------------------------------------------
SEATS = 50
# "month" = the first decision of each calendar month, "week" = of each ISO
# week. Monthly is this round's registered default: review cadence, not seat
# count or keep-band width, is the turnover lever nobody has pulled.
REVIEW = "month"
# A holding ranked inside SEATS * KEEP_BAND is kept; at most MAX_SWAPS names
# change a review, forced exits counted first.
KEEP_BAND = 2.0
MAX_SWAPS = 8
CASH_BUFFER = 0.97
