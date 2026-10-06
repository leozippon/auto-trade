"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values may run and in which
batch. With every switch at its first value this package is `grant`, the
incentive-grant book on 20 seats held 40 trading days.
"""

from lib import titles

# --- registered axes -------------------------------------------------------
# The title categories the book buys (keys of lib/titles.py RULES): one alone,
# or -- only as the registered later step of refs/README.md §7 -- a union of
# several, listed in the order RULES lists them.
CATEGORIES = ("grant",)
# Trading days from the review that bought a name to the first review that
# may sell it: 40 or 60.
HOLD = 40

# --- the leg ---------------------------------------------------------------
# "event": fresh events enter at the review after they become visible.
# "late": the control c_late, the same events entered LATE_DAYS trading days
# later (lib/book.py), on the same categories, seats and hold.
CANDIDATE = "event"

# --- fixed, not axes ---------------------------------------------------------
# Equal-cash seats at 100k, the same for every category.
SEATS = 20
HOLDS = (40, 60)


def leg():
    """The leg name the orders carry, after checking every switch against its registered values."""

    order = list(titles.RULES)
    if (
        not isinstance(CATEGORIES, tuple)
        or not CATEGORIES
        or any(name not in order for name in CATEGORIES)
        or [order.index(name) for name in CATEGORIES] != sorted({order.index(name) for name in CATEGORIES})
    ):
        raise ValueError(f"CATEGORIES must be a tuple of distinct names from {tuple(order)} in that order, got {CATEGORIES!r}")
    if HOLD not in HOLDS:
        raise ValueError(f"HOLD must be one of {HOLDS}, got {HOLD!r}")
    if CANDIDATE not in ("event", "late"):
        raise ValueError(f"CANDIDATE must be 'event' or 'late', got {CANDIDATE!r}")
    name = ("c_late_" if CANDIDATE == "late" else "") + "+".join(CATEGORIES)
    return name if HOLD == HOLDS[0] else f"{name}_h{HOLD}"
