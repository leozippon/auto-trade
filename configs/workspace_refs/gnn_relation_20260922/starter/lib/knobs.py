"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module and no docstring
restates a value from this file; they refer to these names only.

`CANDIDATE` and `REFIT_PERIOD` are the two exceptions: the loader reads them
off `main.py`, so they live there.

Groups, in the order a session usually touches them:

    label       what the model is asked to predict
    graph       how names are wired to each other
    network     capacity and optimisation, including the warm start
    book        how a score becomes a basket this account can hold
"""

# --- label ---------------------------------------------------------------
# Forward trading days of the label; register one value in `hypothesis`
# before the first batch and do not change it afterwards without re-probing.
HORIZON = 10
# Subtract the SW L1 industry mean from the benchmark-residual return before
# ranking. False = residualised against the benchmark only.
LABEL_INDUSTRY_DEMEAN = False
# Trailing years of the rolling training window, and how its tail is split.
TRAIN_YEARS = 2
VALID_DAYS = 60

# --- graph ---------------------------------------------------------------
# Trading days of the return-correlation window and neighbours kept per name
# on the dynamic graph. The static graph is SW L1 co-membership.
CORR_WINDOW = 60
KNN_K = 10
# "gat" = per-relation attention over the neighbourhood;
# "mean" = degree-normalised mean over it (the GCN reading).
GRAPH_MODE = "gat"
GNN_LAYERS = 2

# --- network -------------------------------------------------------------
HIDDEN = 64
DROPOUT = 0.1
SEEDS = (1000, 1001)
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 4
# The refit reloads the previous weights from `state_dir` and continues from
# them at this rate for this many epochs. WARM_START = False makes every
# refit a cold one, which is what every learned arm before this one did.
WARM_START = True
WARM_LEARNING_RATE = 3e-4
WARM_EPOCHS = 8

# --- book ----------------------------------------------------------------
SEATS = 50
# "week" = the first decision of each ISO week, "month" = of each month.
REVIEW = "week"
# A holding ranked inside SEATS * KEEP_BAND is kept; at most MAX_SWAPS names
# change a review, forced exits counted first.
KEEP_BAND = 2.0
MAX_SWAPS = 3
CASH_BUFFER = 0.97
