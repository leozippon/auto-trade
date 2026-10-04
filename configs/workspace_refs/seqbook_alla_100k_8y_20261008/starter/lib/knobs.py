"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values each lane may run and in
which batch. With every lane switch at its first value this package is
`c_base`, the 100k sequence bag on its frozen book; the model, inputs and
label are that baseline's and are not switches here.
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

# --- the bag lane (lib/model.py) -----------------------------------------
# Seeds per head: 2 (c_base), 4 or 6.
SEEDS_PER_HEAD = 2

# --- the cost lane (lib/book.py) -----------------------------------------
# Band exits per review: 4 (c_base), 3 or 2. Forced exits are sold on top.
MAX_SWAPS = 4
# "week" (c_base; first decision of each ISO week) or "biweek" (first
# decision of each half calendar month: days 1-15 and 16-end).
REVIEW = "week"
# A holding is kept while its rank is below KEEP_BAND x SEATS: 2.0 (c_base)
# or 3.0.
KEEP_BAND = 2.0
# Score smoothing: 0 (c_base, off) or 2, the half-life in weeks of the
# exponential average of the bag's rank score over the past weekly rows.
SMOOTH_HALFLIFE = 0

# --- the pool lane (lib/pool.py) -----------------------------------------
# Share of the day's tradable names with the lowest 20-day median turnover
# value that the book may not buy: 0.0 (c_base) or 0.4.
LIQ_DROP = 0.0
# Share with the lowest circulating market value the book may not buy: 0.0
# (c_base) or 0.3 (the host size factor's small leg is the smallest 30 %).
MV_DROP = 0.0
# "all" (c_base), "csi1000" (constituents of 000852.SH) or "csi1500"
# (constituents of 000852.SH or 000905.SH), newest visible sections.
INDEX_POOL = "all"

# --- the frozen book, not an axis ------------------------------------------
SEATS = 12

ALLOWED = {
    "SEED_BASE": (1000, 2000, 3000),
    "SEEDS_PER_HEAD": (2, 4, 6),
    "MAX_SWAPS": (4, 3, 2),
    "REVIEW": ("week", "biweek"),
    "KEEP_BAND": (2.0, 3.0),
    "SMOOTH_HALFLIFE": (0, 2),
    "LIQ_DROP": (0.0, 0.4),
    "MV_DROP": (0.0, 0.3),
    "INDEX_POOL": ("all", "csi1000", "csi1500"),
}
LANES = {
    "bag": ("SEEDS_PER_HEAD",),
    "cost": ("MAX_SWAPS", "REVIEW", "KEEP_BAND", "SMOOTH_HALFLIFE"),
    "pool": ("LIQ_DROP", "MV_DROP", "INDEX_POOL"),
}


def leg():
    """The leg name the orders carry, after checking every switch: registered values, one lane at a time."""

    values = {name: globals()[name] for name in ALLOWED}
    for name, value in values.items():
        if value not in ALLOWED[name]:
            raise ValueError(f"{name} must be one of {ALLOWED[name]}, got {value!r}")
    moved = [lane for lane, names in LANES.items() if any(values[n] != ALLOWED[n][0] for n in names)]
    if len(moved) > 1:
        raise ValueError(f"one lane at a time, got switches of {moved}")
    parts = []
    if SEEDS_PER_HEAD != 2:
        parts.append(f"bag{SEEDS_PER_HEAD}")
    if MAX_SWAPS != 4:
        parts.append(f"swap{MAX_SWAPS}")
    if REVIEW != "week":
        parts.append(REVIEW)
    if KEEP_BAND != 2.0:
        parts.append(f"band{KEEP_BAND:g}")
    if SMOOTH_HALFLIFE:
        parts.append(f"smooth{SMOOTH_HALFLIFE}")
    if LIQ_DROP:
        parts.append(f"liq{round(LIQ_DROP * 100)}")
    if MV_DROP:
        parts.append(f"mv{round(MV_DROP * 100)}")
    if INDEX_POOL != "all":
        parts.append(INDEX_POOL)
    return "+".join(parts) or "c_base"
