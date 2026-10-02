"""Every default this arm registers as an open axis or aligns to the run facts, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` lists the allowed values of each axis and
the batch each one may be spent in. Everything not here is the recipe and
fixed.
"""

# --- the arm -------------------------------------------------------------
# The label's benchmark leg and the only index code in the package. A strategy
# cannot read the run fact `benchmark_index`; round 0 checks that it is
# 000852.SH and stops otherwise.
INDEX = "000852.SH"

# --- the model (lib/model.py) --------------------------------------------
# b4s2 (the bag: four heads, two seeds each), c_single (its first MLP member,
# a control) or c_lgbm (the tree on the same inputs, a control).
CANDIDATE = "b4s2"
# True: the bag's score becomes its residual on the tree's score
# (cross-sectional OLS), re-ranked; fit then also trains the tree. Decided on
# the validation rows' benchmark.active_max_drawdown, never on equity drawdown.
RESIDUALISE = False
# Member k of head h trains with seed SEED_BASE + 100 x (h's position) + k and
# the tree with seed 7 + (SEED_BASE - 1000). 1000 is the registered value;
# 2000 only for the seed replicate before a nomination.
SEED_BASE = 1000

# --- the book (lib/book.py) ----------------------------------------------
# Equal-cash seats: 12, 20 or 30. The keep band (rank < 2 x seats) scales
# with it, and so does the one-lot affordability filter of the pool.
SEATS = 30
# Band exits per review: 2 or 4. Forced exits are always sold on top.
MAX_SWAPS = 4
# "week" (first decision of each ISO week) or "biweek" (first decision of
# each half calendar month: days 1-15 and 16-end).
REVIEW = "week"
