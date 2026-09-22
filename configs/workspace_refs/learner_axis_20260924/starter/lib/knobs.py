"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module and no docstring
restates a value from this file; they refer to these names only.

Three names live outside this file on purpose:

    CANDIDATE / REFIT_PERIOD    the loader reads them off `main.py`
    index.INDEX_CODE            the benchmark this book is built in and graded
                                against; it belongs beside the read that uses
                                it, and the run facts override it

Which learner and which objective each candidate runs is a definition, not a
default, and lives in `primary.LEGS`.

Groups, in the order a session usually touches them:

    label       what the primary is asked to predict
    primary     the carrier: Alpha158 + LightGBM, the graduated lineage
    learner     the two moves this arm prices: DoubleEnsemble and lambdarank
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
# Every leg trains its boosters on this point; lambdarank legs override only
# the objective and its metric (`primary.params`).
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

# --- learner -------------------------------------------------------------
# DoubleEnsemble (`lib/double_ensemble.py`, `references/double-ensemble.md`).
# `models` sub-models on PARAMS: the first early-stops exactly like the
# carrier (COLD_ROUNDS / EARLY_STOP), the others train that many trees;
# sample reweighting over `bins_sr` bins of h = alpha1*h1 + alpha2*h2 with
# weight 1 / (decay^k * mean h of the bin + 0.1); feature selection over
# `bins_fs` bins of the shuffle g-value, keeping `sample_ratios` of each bin
# from the most to the least important. The values are Qlib's Alpha158
# configuration of the same algorithm; `models` is the cost lever (each extra
# sub-model is one more cold fit plus its share of the shuffle pass).
DOUBLE_ENSEMBLE = {
    "models": 3,
    "decay": 0.5,
    "alpha1": 1.0,
    "alpha2": 1.0,
    "bins_sr": 10,
    "bins_fs": 5,
    "sample_ratios": (0.8, 0.7, 0.6, 0.5, 0.4),
}
# lambdarank: one query per decision bar; relevance = the carrier's own rank
# target cut into RANK_GRADES equal-width integer grades (0 = worst) with a
# LINEAR gain (gain = grade); pairs are formed only above RANK_TRUNCATION
# (twice the registered SEATS); early stopping reads NDCG at SEATS on the
# validation bars. The top-weighting comes from NDCG's position discount and
# the truncation. LightGBM's default exponential gain was measured as a
# dispersion bet on this carrier, not a ranking (`sources.md`).
RANK_GRADES = 20
RANK_TRUNCATION = 100

# --- state ---------------------------------------------------------------
# Refits between two cold starts of a warm-chained leg: with a quarterly
# refit, every fourth one starts from a fresh booster so a warm chain cannot
# run a whole replay. DoubleEnsemble legs and `c_cold` ignore it: they are
# cold on every refit (`primary.LEGS`).
COLD_EVERY = 4

# --- book ----------------------------------------------------------------
SEATS = 50
# "month" = the first decision of each calendar month, "week" = of each ISO
# week. Monthly is the round's registered default.
REVIEW = "month"
# A holding ranked inside SEATS * KEEP_BAND is kept; at most MAX_SWAPS names
# change a review, forced exits counted first.
KEEP_BAND = 2.0
MAX_SWAPS = 8
CASH_BUFFER = 0.97
