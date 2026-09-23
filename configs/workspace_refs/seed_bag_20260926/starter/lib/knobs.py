"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module and no docstring
restates a value from this file; they refer to these names only.

Three names live outside this file on purpose:

    CANDIDATE / REFIT_PERIOD    the loader reads them off `main.py`
    index.INDEX_CODE            the benchmark this book is built in and graded
                                against; it belongs beside the read that uses
                                it, and the run facts override it

Groups, in file order:

    label       what the carrier is asked to predict
    legs        the training seeds each candidate averages -- the only thing
                this arm moves (`refs/families.md` is the authority for what each
                means; a new seed list is a new candidate)
    book        the monthly 50-seat book every leg shares, the head packs'
                `c_base` constants
    primary     the carrier: Alpha158 + LightGBM, the graduated lineage; its
                parameter point has no seed, the leg supplies it
    state       warm continuation and the cold-reset cadence
"""

# --- label ---------------------------------------------------------------
# Forward trading days of the carrier's label, and the width of the embargo
# before its validation segment.
HORIZON = 10
# Trailing OLS beta of the label's benchmark leg: window, minimum window,
# shrinkage toward 1, and the clip applied after shrinking.
BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)

# --- legs ----------------------------------------------------------------
# The LightGBM training seeds a leg fits, one booster per seed. The leg's score
# is the average over its boosters of each booster's cross-sectional
# percentile rank; with one seed that is the booster's own ranking. `b5x` and
# `b10x` are the disjoint-seed twins a passing bag must be replicated with
# before nomination, registered here so their seeds are fixed in advance.
SEEDS = {
    "c_s7": [7],
    "c_s8": [8],
    "b5": [7, 8, 9, 10, 11],
    "b10": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "b5x": [17, 18, 19, 20, 21],
    "b10x": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
}

# --- book ----------------------------------------------------------------
# Reviewed at the first decision of each calendar month (`lib/trade.py`).
SEATS = 50
CASH_BUFFER = 0.97
# Names that may change at one review, forced exits counted first.
MAX_SWAPS = 8
# A holding ranked inside SEATS * KEEP_BAND is kept.
KEEP_BAND = 2.0

# --- primary -------------------------------------------------------------
# Trailing years of the rolling training window, and how its tail is split
# into a validation segment (the embargo before it is the label horizon).
TRAIN_YEARS = 3
VALID_DAYS = 60
MIN_SAMPLES = 200
# One registered parameter point, not a grid: a grid buys fit seconds, and
# every extra validated revision raises this arm's own deflated-Sharpe bar.
# The seed is not here: `lib/primary.py` adds each seed of the leg.
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
    "verbose": -1,
}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50

# --- state ---------------------------------------------------------------
# Refits between two cold starts: with a quarterly refit, every fourth one
# starts from fresh boosters so a warm chain cannot run a whole replay.
COLD_EVERY = 4
