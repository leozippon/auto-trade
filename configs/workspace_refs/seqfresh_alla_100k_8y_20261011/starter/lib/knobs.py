"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values may run and in which
batch. With every switch at its first value this package is `c_bag4`, the
100k sequence bag with four seeds per head on the frozen bag's book; the
heads, the seeds per head, inputs, label, refit cadence and book are that
baseline's and are not switches here.
"""

# --- the arm -------------------------------------------------------------
# The label's benchmark leg and the only index code the label reads. A
# strategy cannot read the run fact `benchmark_index`; round 0 checks that it
# is 000852.SH and stops otherwise.
INDEX = "000852.SH"

# --- every leg -----------------------------------------------------------
# Member j of head h trains with seed SEED_BASE + 100 x (h's position) + j.
# 1000, 2000, 3000 or 4000; a leg's c_bag4 always runs on the same base.
SEED_BASE = 1000

# --- the fresh lane: what the heads learn from (lib/model.py) -------------
# Years of labelled history behind each refit: 3 (c_bag4), 2 or 5.
TRAIN_YEARS = 3
# Half-life in years of the exponential loss weight on a training date's age
# inside the window: 0.0 (c_bag4, every date weighs the same) or 1.0.
RECENCY_HALFLIFE = 0.0

# --- the frozen bag's book, not an axis -------------------------------------
SEATS = 12

ALLOWED = {
    "SEED_BASE": (1000, 2000, 3000, 4000),
    "TRAIN_YEARS": (3, 2, 5),
    "RECENCY_HALFLIFE": (0.0, 1.0),
}


def leg():
    """The leg name the orders carry, after checking every switch against its registered values.

    Switches combine freely; which combinations run, and when, is the
    README's batch plan.
    """

    for name, allowed in ALLOWED.items():
        value = globals()[name]
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    parts = []
    if TRAIN_YEARS != 3:
        parts.append(f"win{TRAIN_YEARS}")
    if RECENCY_HALFLIFE:
        parts.append(f"rec{RECENCY_HALFLIFE:g}")
    return "+".join(parts) or "c_bag4"
