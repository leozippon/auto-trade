"""Every switch of this package but one, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values may run and in which
batch. With every switch at its first value, and `REFIT_PERIOD` in `main.py`
at "quarter", this package is `c_base`, the 100k sequence bag on its frozen
book; the heads, inputs, label and book are that baseline's and are not
switches here.

The refit cadence is the one switch that lives elsewhere: the host reads
`REFIT_PERIOD` from `main.py` as a literal, so it is set there, and `leg()`
and the cold-start counter (`lib/state.py`) receive it from `main.py`.
"""

# --- the arm -------------------------------------------------------------
# The label's benchmark leg and the only index code the label reads. A
# strategy cannot read the run fact `benchmark_index`; round 0 checks that it
# is 000852.SH and stops otherwise.
INDEX = "000852.SH"

# --- every leg -----------------------------------------------------------
# Member j of head h trains with seed SEED_BASE + 100 x (h's position) + j.
# 1000, 2000 or 3000; a leg's c_base always runs on the same base.
SEED_BASE = 1000

# --- the fresh lane: what the heads learn from (lib/model.py) -------------
# Years of labelled history behind each refit: 3 (c_base), 2 or 5.
TRAIN_YEARS = 3
# Half-life in years of the exponential loss weight on a training date's age
# inside the window: 0.0 (c_base, every date weighs the same) or 1.0.
RECENCY_HALFLIFE = 0.0

# --- the frozen book, not an axis ------------------------------------------
SEATS = 12

ALLOWED = {
    "SEED_BASE": (1000, 2000, 3000),
    "TRAIN_YEARS": (3, 2, 5),
    "RECENCY_HALFLIFE": (0.0, 1.0),
}
REFIT_PERIODS = ("quarter", "month")


def leg(refit_period):
    """The leg name the orders carry, after checking every switch against its registered values.

    `refit_period` is `main.REFIT_PERIOD`. Switches combine freely; which
    combinations run, and when, is the README's batch plan.
    """

    values = {name: globals()[name] for name in ALLOWED}
    for name, value in values.items():
        if value not in ALLOWED[name]:
            raise ValueError(f"{name} must be one of {ALLOWED[name]}, got {value!r}")
    if refit_period not in REFIT_PERIODS:
        raise ValueError(f"main.REFIT_PERIOD must be one of {REFIT_PERIODS}, got {refit_period!r}")
    parts = []
    if TRAIN_YEARS != 3:
        parts.append(f"win{TRAIN_YEARS}")
    if refit_period != "quarter":
        parts.append(refit_period)
    if RECENCY_HALFLIFE:
        parts.append(f"rec{RECENCY_HALFLIFE:g}")
    return "+".join(parts) or "c_base"
