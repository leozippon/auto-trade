"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which values may run and in which
batch. With every switch at its first value this package is `b16`, the 100k
baseline; the 500k arm's baseline `b40` is the same package with the account
line and the seat line set as round 0 of that arm says.
"""

# --- the arm ---------------------------------------------------------------
# The account the book is shaped for. A strategy cannot read the run fact
# `broker_replay.initial_cash`; round 0 reads it and sets this line to it,
# 100_000 or 500_000, and stops for any other value. The account fixes the
# execution and the liquidity floor below; it is not an axis.
ACCOUNT = 100_000

# --- registered axes -------------------------------------------------------
# Seats, equal cash. 100k: 16 (b16), 20, 24 or 30. 500k: 40 (b40) or 30.
SEATS = 30
# Trading days from the review that bought a name to the first review that
# may sell it: 40 or 60.
HOLD = 40

# --- the leg ---------------------------------------------------------------
# "event": fresh drafts enter at the review after they become visible.
# "late": the control c_late, the same events entered LATE_DAYS trading days
# later (lib/book.py), on the same seats and hold.
CANDIDATE = "event"

# --- fixed by the account, not axes ------------------------------------------
ALLOWED_SEATS = {100_000: (16, 20, 24, 30), 500_000: (40, 30)}
HOLDS = (40, 60)
# "open": sell at 09:30, buy at 15:00. "close": both legs in the closing
# auction (15:00), where a 500k ticket is a smaller share of the session.
EXECUTION = {100_000: "open", 500_000: "close"}
# Median daily amount (CNY) over the last 20 visible trading days that a name
# needs before it is bought: none at 100k, CNY 40m at 500k.
MIN_AMOUNT = {100_000: 0.0, 500_000: 4.0e7}


def leg():
    """The leg name the orders carry, after checking every switch against its registered values."""

    if ACCOUNT not in ALLOWED_SEATS:
        raise ValueError(f"ACCOUNT must be one of {tuple(ALLOWED_SEATS)}, got {ACCOUNT!r}")
    if SEATS not in ALLOWED_SEATS[ACCOUNT]:
        raise ValueError(f"SEATS must be one of {ALLOWED_SEATS[ACCOUNT]} at {ACCOUNT}, got {SEATS!r}")
    if HOLD not in HOLDS:
        raise ValueError(f"HOLD must be one of {HOLDS}, got {HOLD!r}")
    if CANDIDATE not in ("event", "late"):
        raise ValueError(f"CANDIDATE must be 'event' or 'late', got {CANDIDATE!r}")
    name = ("c_late" if CANDIDATE == "late" else "b") + str(SEATS)
    return name if HOLD == HOLDS[0] else f"{name}_h{HOLD}"
