"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module and no docstring
restates a value from this file; they refer to these names only.

Three names live outside this file on purpose:

    CANDIDATE / REFIT_PERIOD    the loader reads them off `main.py`
    index.INDEX_CODE            the benchmark this book is built in and graded
                                against; it belongs beside the read that uses
                                it, and the run facts override it

Groups, in file order (`legs` is the one a session edits; the label comes
first only because `C_BASE` reads its horizon):

    label       what the carrier is asked to predict
    legs        the construction each candidate runs -- the only thing this
                arm moves (`families.md` is the authority for what each means;
                f1's review is the registered value the pack's census derived
                from the score's IC decay -- changing it is a new candidate)
    book        what every leg shares: seats, cash buffer, and the parameters
                of the two conditional mechanisms (exposure pull-back, EPO)
    primary     the carrier: Alpha158 + LightGBM, the graduated lineage
    state       warm continuation and the cold-reset cadence
"""

# --- label ---------------------------------------------------------------
# Forward trading days of the carrier's label, and the width of the embargo
# before its validation segment. A leg may carry its own (`f3`).
HORIZON = 10
# Trailing OLS beta of the label's benchmark leg: window, minimum window,
# shrinkage toward 1, and the clip applied after shrinking. The exposure
# pull-back reads the same beta.
BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)

# --- legs ----------------------------------------------------------------
# review      "month" = the first decision of each calendar month; a whole
#             number k = the first decision of each k-week bucket (weeks run
#             Monday..Sunday; buckets are counted from a fixed calendar origin,
#             so a cold worker and a warm one agree)
# max_swaps   names that may change at one review, forced exits counted
#             first; None = no cap
# keep_band   a holding ranked inside SEATS * keep_band is kept
# pull        greedy pull of beta, size, volatility and industry back toward
#             the pool (`book.pull_back`)
# weights     "equal" cash per seat, or "epo" at entry (`book.epo`)
# horizon     the label horizon this leg's booster is trained on
# refill      between reviews, buy into seats left empty by a rejected or
#             cash-short entry (limit-up, no price) on the following trading
#             days, best-ranked affordable names first; no sells between reviews
C_BASE = {"review": "month", "max_swaps": 8, "keep_band": 2.0, "pull": False,
          "weights": "equal", "horizon": HORIZON, "refill": False}
F1 = {**C_BASE, "review": 3, "max_swaps": None, "keep_band": 1.2, "refill": True}
LEGS = {
    "c_base": C_BASE,
    "f1": F1,
    "f2": {**F1, "pull": True},
    "f3": {**F1, "horizon": 20},
    "f4": {**F1, "weights": "epo"},
}

# --- book ----------------------------------------------------------------
SEATS = 50
CASH_BUFFER = 0.97
# Pull-back tolerances: the book's mean exposure within this many pool
# standard deviations of the pool mean (beta, log circulating cap, 60-day
# volatility), and no SW level-1 industry more than this far above its share
# of the pool.
PULL_STYLE_TOL = 0.10
PULL_INDUSTRY_TOL = 0.05
VOL_DAYS = 60
# EPO at entry: trailing days of the covariance, the shrink of correlations
# toward zero (1 = inverse variance), and the cap on one name's cash as a
# multiple of a seat.
EPO_DAYS = 120
EPO_SHRINK = 0.5
EPO_CAP = 2.0

# --- primary -------------------------------------------------------------
# Trailing years of the rolling training window, and how its tail is split
# into a validation segment (the embargo before it is the leg's horizon).
TRAIN_YEARS = 3
VALID_DAYS = 60
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

# --- state ---------------------------------------------------------------
# Refits between two cold starts: with a quarterly refit, every fourth one
# starts from a fresh booster so a warm chain cannot run a whole replay.
COLD_EVERY = 4
