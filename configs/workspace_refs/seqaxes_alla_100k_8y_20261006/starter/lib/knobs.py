"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values each lane may run and in
which batch. Everything not here is the baseline recipe and fixed.
"""

# --- the arm -------------------------------------------------------------
# The label's benchmark leg and the only index code in the package. A strategy
# cannot read the run fact `benchmark_index`; round 0 checks that it is
# 000852.SH and stops otherwise.
INDEX = "000852.SH"

# --- the label lane (lib/label.py) ---------------------------------------
# "base": the baseline label (10-day beta residual, adjusted prices).
# "bench": the pure benchmark residual, not scaled by beta.
# "size": the baseline residual demeaned within size quintiles of each date.
# "hold": the baseline residual over HOLD_ALIGNED days instead of 10.
LABEL = "base"
# The label horizon aligned to the baseline book's holding period, in trading
# days: the registered rule (median round trip rounded to ten, clipped to
# [20, 40]) on the baseline book's own full-span row gives 10 -> 20; the
# seat-day-weighted median round trip of that row is 20 as well.
HOLD_ALIGNED = 20

# --- the fundamentals lane (lib/fusion.py) -------------------------------
# "off": no fundamentals. "late": the bag's score rank-blended with a
# fundamentals tree fitted inside fit. "early": fundamentals features joined
# to every head's last layer.
FUND = "off"
# True: the control c_perm -- the same model with each date's fundamentals
# rows permuted across the stocks of that date.
FUND_PERM = False

# --- the baseline model (lib/model.py, lib/tree.py) ----------------------
# Member k of head h trains with seed SEED_BASE + 100 x (h's position) + k and
# the tree with seed 7 + (SEED_BASE - 1000). 1000 and 2000 are the two
# registered seed bases (every leg runs on both); 3000 only for the third seed
# base of a finalist and c_base. c_base (and c_perm) always on the same base.
SEED_BASE = 1000

# --- the baseline book (lib/book.py) -------------------------------------
# The book the 100k bag arm froze (12 seats, 4 swaps, weekly, not
# residualised); fixed in this package, not an axis.
# True: the bag's score becomes its residual on the tree's score
# (cross-sectional OLS), re-ranked; fit then also trains the tree.
RESIDUALISE = False
# Equal-cash seats; the keep band (rank < 2 x seats) and the one-lot
# affordability filter of the pool scale with it.
SEATS = 12
# Band exits per review. Forced exits are always sold on top.
MAX_SWAPS = 4
# "week" (first decision of each ISO week) or "biweek" (first decision of
# each half calendar month: days 1-15 and 16-end).
REVIEW = "week"


def leg():
    """The leg name the orders carry, after checking the switches: one lane at a time."""

    if LABEL not in ("base", "bench", "size", "hold"):
        raise ValueError(f"LABEL must be base, bench, size or hold, got {LABEL!r}")
    if FUND not in ("off", "late", "early"):
        raise ValueError(f"FUND must be off, late or early, got {FUND!r}")
    if LABEL != "base" and FUND != "off":
        raise ValueError("one lane at a time: LABEL or FUND, not both")
    if FUND_PERM and FUND == "off":
        raise ValueError("FUND_PERM permutes fundamentals, so it needs FUND late or early")
    if LABEL != "base":
        return "lab_" + LABEL
    if FUND != "off":
        return ("c_perm_" if FUND_PERM else "fund_") + FUND
    return "c_base"
