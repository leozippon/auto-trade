"""Every switch of this package, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only. `refs/README.md` says which
values may run and in which batch. With every switch at its first value the
package is the candidate `G`.
"""

# The benchmark the arm is graded against. A strategy cannot read the run fact
# `benchmark_index`; round 0 checks that it is 000852.SH and stops otherwise.
# The canonical book uses it to report how much of itself sits in the index.
INDEX = "000852.SH"
# "alla": every tradable non-Beijing, non-STAR, non-ST A-share. Fixed.
POOL = "alla"
# 12 (three industries of four names) or 16 (four industries of four).
SEATS = 12
# Share of the book value spread over the seats; the rest is the buffer
# between the T-1 close the orders are sized at and the 15:00 fill price.
CAPITAL = 0.97
# At a review a holding ranked inside SEATS * KEEP_BAND is kept. With
# lib/score.py's six scored names per industry this keeps a holding while its
# industry ranks inside the top four (12 seats) or five (16 seats).
KEEP_BAND = 2.0
# Names per Shenwan L1 industry: four, so twelve seats hold three industries.
INDUSTRY_CAP = 4
# "month": the first decision of each calendar month. Fixed.
REVIEW = "month"
# False: the industries' own ranking. True: the control c_shuf, the same
# industry blocks put in an order drawn from the decision's calendar quarter.
SHUFFLE = False

# --- the cycle lane (lib/industry.py, lib/score.py) --------------------------
# Which industries rank first: "G" (reported growth accelerating), "GI" (G
# less half of capacity build), or "px" (the control c_px: the industries'
# trailing 120-day return, no fundamentals).
SIGNAL = "G"
# Which names of a chosen industry come first: "cap" (largest float cap) or
# "accel" (the name's own revenue acceleration).
WITHIN = "cap"

ALLOWED = {
    "SEATS": (12, 16),
    "SIGNAL": ("G", "GI", "px"),
    "WITHIN": ("cap", "accel"),
    "SHUFFLE": (False, True),
}


def leg():
    """The leg name the orders carry, after checking every switch against its registered values."""

    for name, values in ALLOWED.items():
        value = globals()[name]
        if value not in values or type(value) is not type(values[0]):
            raise ValueError(f"{name} must be one of {values}, got {value!r}")
    if POOL != "alla" or REVIEW != "month" or INDUSTRY_CAP != 4 or KEEP_BAND != 2.0:
        raise ValueError("POOL, REVIEW, INDUSTRY_CAP and KEEP_BAND are fixed by refs/README.md")
    if SHUFFLE and SIGNAL == "px":
        raise ValueError("c_shuf permutes a fundamentals ranking; c_px is a control already")
    parts = ["c_shuf" if SHUFFLE else "c_px" if SIGNAL == "px" else SIGNAL]
    if SEATS != 12:
        parts.append(f"s{SEATS}")
    if WITHIN != "cap":
        parts.append(WITHIN)
    return "+".join(parts)
