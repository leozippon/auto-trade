"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which lane the arm runs, which values
may run and in which batch. With every switch at its first value this package
is `c_bag4`, the 100k sequence bag with four seeds per head; its four heads,
the seeds per head, inputs, label, training window and book are that
baseline's and are not switches here. The refit cadence is `REFIT_PERIOD` in
`main.py`, because the host reads it there as a literal; `check_refit` ties it
to `UPDATE`.
"""

# --- the arm -------------------------------------------------------------
# The label's benchmark leg and the only index code the label reads. A
# strategy cannot read the run fact `benchmark_index`; round 0 checks that it
# is 000852.SH and stops otherwise.
INDEX = "000852.SH"

# --- every leg -----------------------------------------------------------
# Member j of the head in seed slot k trains with seed SEED_BASE + 100 x k + j
# (`lib/model.py`). 1000, 2000, 3000 or 4000; a leg's c_bag4 always runs on
# the same base.
SEED_BASE = 1000

# --- the ssm lane: a selective state-space head in the bag (lib/model.py) ---
# "off" (c_bag4), "add" (a fifth head) or "replace" (in place of the mlp head).
SSM = "off"

# --- the cl lane: monthly updates between the quarterly refits (lib/model.py) ---
# "off" (c_bag4), "warm" (the newest dates only) or "replay" (the newest dates
# and as many dates replayed from the trailing window). Any value but "off"
# needs REFIT_PERIOD = "month" in main.py.
UPDATE = "off"

# --- either lane ---------------------------------------------------------
# True runs the lane's placebo: the new seed slot holds four re-seeded LSTM
# members instead of the state-space head, or the monthly update trains on
# the newest dates with each date's labels permuted across names.
PLACEBO = False

# --- the frozen bag's book, not an axis -------------------------------------
SEATS = 12

ALLOWED = {
    "SEED_BASE": (1000, 2000, 3000, 4000),
    "SSM": ("off", "add", "replace"),
    "UPDATE": ("off", "warm", "replay"),
    "PLACEBO": (False, True),
}


def leg():
    """The leg name the orders carry, after checking every switch against its registered values."""

    for name, allowed in ALLOWED.items():
        value = globals()[name]
        if not any(type(value) is type(option) and value == option for option in allowed):
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    if SSM != "off" and UPDATE != "off":
        raise ValueError("one lane at a time: SSM and UPDATE cannot both be on")
    prefix = "plc_" if PLACEBO else None
    if SSM != "off":
        return (prefix or "ssm_") + {"add": "add", "replace": "rep"}[SSM]
    if UPDATE != "off":
        return (prefix or "cl_") + UPDATE
    if PLACEBO:
        raise ValueError("PLACEBO belongs to a lane: set SSM or UPDATE first")
    return "c_bag4"


def check_refit(refit_period):
    """Refuse a REFIT_PERIOD (main.py) that does not match UPDATE: monthly exactly when the cl lane is on."""

    wanted = "quarter" if UPDATE == "off" else "month"
    if refit_period != wanted:
        raise ValueError(f"UPDATE = {UPDATE!r} needs REFIT_PERIOD = {wanted!r} in main.py, got {refit_period!r}")
