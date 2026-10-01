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

# --- the bag (lib/model.py) ----------------------------------------------
# A key of model.BAGS (a pre-registered bag, or the control c_single), or
# c_lgbm for the tree control.
CANDIDATE = "b4"
# True: every network candidate's score becomes its residual on the tree's
# score (cross-sectional OLS), re-ranked; fit then also trains the tree.
RESIDUALISE = False
# Member k of head h trains with seed SEED_BASE + 100 x (h's position) + k.
# 1000 is the registered value; 2000 only for the shifted-seed re-fit.
SEED_BASE = 1000

# --- the book (lib/book.py) ----------------------------------------------
# Equal-cash seats; the keep band (2 x seats) and the swap cap (4) are fixed.
SEATS = 30
