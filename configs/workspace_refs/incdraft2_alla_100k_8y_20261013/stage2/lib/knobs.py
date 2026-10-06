"""Every switch of the stage-2 package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which legs may run and in which
batch. With every switch at its first value this package is `rs20`. The
unsplit legs b20, b30, c_late20 and c_late30 are not legs of this package:
they run from `refs/starter/` byte for byte.
"""

# --- the arm ---------------------------------------------------------------
# The account the book is shaped for; round 0 checks the run fact
# `broker_replay.initial_cash` is 100_000. It is not an axis.
ACCOUNT = 100_000

# --- registered switches -----------------------------------------------------
# Which side of the instrument split the book trades. An event's side is read
# from every draft title of its day (lib/titles.py): "opt" when one of them
# names a stock option, "rs" otherwise; "all" trades both.
SIDE = "rs"
# Seats, equal cash: 20 or 30 on the "rs" side, 12 on the "opt" side, 20 for
# the delay control.
SEATS = 20
# Trading days added to every event's visible day: 3 only for the delay
# control b20_d3.
DELAY = 0

# --- the leg ---------------------------------------------------------------
# "event": fresh drafts enter at the review after they become visible.
# "late": the control c_late, the same events entered LATE_DAYS trading days
# later (lib/book.py), on the same seats and hold.
CANDIDATE = "event"

# --- fixed, not axes ---------------------------------------------------------
HOLD = 40
LEGS = ("rs20", "op12", "c_late_rs20", "c_late_op12", "b20_d3", "rs30", "c_late_rs30")
EXECUTION = {100_000: "open"}
MIN_AMOUNT = {100_000: 0.0}


def leg():
    """The leg name the orders carry, after checking the switches name a registered leg."""

    if ACCOUNT != 100_000:
        raise ValueError(f"ACCOUNT must be 100_000, got {ACCOUNT!r}")
    side = {"all": "", "rs": "rs", "opt": "op"}.get(SIDE)
    if side is None:
        raise ValueError(f"SIDE must be 'all', 'rs' or 'opt', got {SIDE!r}")
    if CANDIDATE not in ("event", "late"):
        raise ValueError(f"CANDIDATE must be 'event' or 'late', got {CANDIDATE!r}")
    if CANDIDATE == "late":
        name = "c_late" + (f"_{side}" if side else "") + str(SEATS)
    else:
        name = (side or "b") + str(SEATS)
    if DELAY:
        name += f"_d{DELAY}"
    if name not in LEGS:
        raise ValueError(f"{name} is not a leg of this package: {LEGS}")
    return name
