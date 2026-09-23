"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only.

Groups:

    book     how the score becomes a basket this account can hold
    score    the same-calendar-month lags (registered in `refs/families.md`;
             changing one of these is a new registration)
"""

# --- book ----------------------------------------------------------------
# The benchmark the book is built in and graded against. A strategy cannot
# read the run fact `benchmark_index`, so round 0 reads it and changes this
# constant to match it -- the run fact overrides this literal, never the other
# way round.
INDEX = "000905.SH"
SEATS = 50
# Share of the book value spread over the seats; the rest is the buffer
# between the T-1 close the orders are sized at and the fill price.
CAPITAL = 0.97
# A holding ranked inside SEATS * KEEP_BAND is kept at a review; 1.0 holds
# exactly the month's top SEATS.
KEEP_BAND = 1.0
# False: the candidate s1. True: the control c_shuf, s1's own values permuted
# among the scored names of the decision day.
SHUFFLE = False

# --- score ---------------------------------------------------------------
# Annual lags k of the calendar month m read at a decision in m: months
# m - 12k for k = FIRST_LAG .. LAGS; a name needs at least MIN_LAGS of them.
FIRST_LAG = 1
LAGS = 4
MIN_LAGS = 3
