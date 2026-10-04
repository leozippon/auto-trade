"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values each lane may run and in
which batch. With every lane switch at its first value this package is
`c_base`, the 100k sequence bag on its frozen book; the model, inputs, label
and book are that baseline's and are not switches here.
"""

# --- the arm -------------------------------------------------------------
# The label's benchmark leg, the beta lane's index and the only index code the
# package reads. A strategy cannot read the run fact `benchmark_index`; round
# 0 checks that it is 000852.SH and stops otherwise.
INDEX = "000852.SH"

# --- every leg -----------------------------------------------------------
# Member j of head h trains with seed SEED_BASE + 100 x (h's position) + j.
# 1000, 2000 or 3000; a leg's c_base always runs on the same base.
SEED_BASE = 1000

# --- the clock lane (lib/clock.py) ---------------------------------------
# "open_close" (c_base): a review sells at 09:30 and buys at 15:00 of the same
# day. "close_open": a review sells at 15:00 and the empty seats are bought at
# 09:30 of the next trading day.
CLOCK = "open_close"

# --- the beta lane (lib/beta.py) -----------------------------------------
# A name may be bought only while its ex-ante beta to INDEX is at least this:
# 0.0 (c_base, no floor), 1.0 or 1.1. Holdings are never sold for their beta.
BETA_FLOOR = 0.0
# Trailing trading days of the ex-ante beta: 120 (the label's window) or 60.
BETA_DAYS = 120
# True only for the placebo c_betashuf: the same floor on a date-seeded random
# score instead of the bag's, with no fit.
BETA_PLACEBO = False

# --- the frozen bag and book, not axes ---------------------------------------
SEATS = 12
SEEDS_PER_HEAD = 2
MAX_SWAPS = 4
REVIEW = "week"
KEEP_BAND = 2.0

ALLOWED = {
    "SEED_BASE": (1000, 2000, 3000),
    "CLOCK": ("open_close", "close_open"),
    "BETA_FLOOR": (0.0, 1.0, 1.1),
    "BETA_DAYS": (120, 60),
    "BETA_PLACEBO": (False, True),
}
LANES = {
    "clock": ("CLOCK",),
    "beta": ("BETA_FLOOR", "BETA_DAYS", "BETA_PLACEBO"),
}


def leg():
    """The leg name the orders carry, after checking every switch: registered values, one lane at a time."""

    values = {name: globals()[name] for name in ALLOWED}
    for name, value in values.items():
        if value not in ALLOWED[name] or type(value) is not type(ALLOWED[name][0]):
            raise ValueError(f"{name} must be one of {ALLOWED[name]}, got {value!r}")
    moved = [lane for lane, names in LANES.items() if any(values[n] != ALLOWED[n][0] for n in names)]
    if len(moved) > 1:
        raise ValueError(f"one lane at a time, got switches of {moved}")
    if not BETA_FLOOR and (BETA_DAYS != 120 or BETA_PLACEBO):
        raise ValueError("BETA_DAYS and BETA_PLACEBO apply to a beta floor; set BETA_FLOOR first")
    if BETA_PLACEBO and SEED_BASE != 1000:
        raise ValueError("the placebo fits nothing, so the seed base cannot reach it: run it on SEED_BASE 1000 only")
    parts = []
    if CLOCK != "open_close":
        parts.append(CLOCK)
    if BETA_FLOOR:
        parts.append(f"beta{BETA_FLOOR:g}")
    if BETA_DAYS != 120:
        parts.append(f"w{BETA_DAYS}")
    if BETA_PLACEBO:
        parts.append("shuf")
    return "+".join(parts) or "c_base"
