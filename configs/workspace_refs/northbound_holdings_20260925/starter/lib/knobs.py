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

    holdings    the northbound score this arm is here to price (lib/holdings.py)
    book        how a score becomes a basket this account can hold
    label       what the carrier legs' booster is asked to predict
    primary     the carrier: Alpha158 + LightGBM, read by c_base and n2 only
    state       the carrier's warm continuation and cold-reset cadence
"""

# --- holdings ------------------------------------------------------------
# Earlier quarter-ends averaged in the denominator of `chg` (the quarterly
# analog of a trailing 12-month mean). Registered in `families.md`.
CHG_QUARTERS = 4
# Calendar days of the holder-table read, keyed on `ann_date`: must reach the
# oldest quarter-end the denominator can use plus its announcement lag.
HOLDINGS_LOOKBACK_DAYS = 640
# Weight of `lvl` in n1's rank average; `chg` carries the rest. A missing
# component takes the neutral rank 0.5.
LEVEL_WEIGHT = 0.5
# Weight of the carrier's rank in n2 (Family 2); the holdings change carries
# the rest.
N2_CARRIER_WEIGHT = 0.5

# --- book ----------------------------------------------------------------
SEATS = 50
# "month" = the first decision of each calendar month, "week" = of each ISO
# week. Monthly is the registered default; the holdings score itself moves
# only when a quarterly report lands.
REVIEW = "month"
# A holding ranked inside SEATS * KEEP_BAND is kept; at most MAX_SWAPS names
# change a review, forced exits counted first.
KEEP_BAND = 2.0
MAX_SWAPS = 8
CASH_BUFFER = 0.97

# --- label ---------------------------------------------------------------
# Forward trading days of the carrier's label (c_base, n2).
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
# The carrier's one registered parameter point, unchanged from its lineage.
PARAMS = {
    "objective": "regression",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    # Pinned: the container has 8 CPUs and a batch shares the host.
    "num_threads": 8,
    "seed": 7,
    "verbose": -1,
}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50

# --- state ---------------------------------------------------------------
# Refits between two cold starts: with a quarterly refit, every fourth one
# starts from a fresh booster so a warm chain cannot run a whole replay.
COLD_EVERY = 4
