"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only.

Groups:

    book     how the score becomes a basket this account can hold
    score    the CSI 300 entry rule the score reproduces (registered in
             `refs/families.md`; changing one of these is a new registration)
"""

# --- book ----------------------------------------------------------------
# The benchmark the book is built in and graded against. A strategy cannot
# read the run fact `benchmark_index`, so round 0 reads it and changes this
# constant to match it -- the run fact overrides this literal, never the other
# way round.
INDEX = "000905.SH"
SEATS = 30
# Share of the book value spread over the seats; the rest is the buffer
# between the T-1 close the orders are sized at and the fill price.
CAPITAL = 0.97
# A holding ranked inside SEATS * KEEP_BAND is kept at a review.
KEEP_BAND = 2.0
# False: the candidate m1. True: the control c_shuf, m1's own values permuted
# among the scored names of the decision day.
SHUFFLE = False

# --- score ---------------------------------------------------------------
# The index whose entry rule is reproduced. Part of the score, not a benchmark.
TARGET = "000300.SH"
# Calendar days of the averaging window ending at T-1 ("the past year").
WINDOW_DAYS = 365
# Liquidity screen: the top share by mean daily amount is eligible; current
# TARGET members are eligible within the wider incumbent share.
LIQUIDITY_CUT = 0.5
INCUMBENT_CUT = 0.6
# Listing age for the sample space: STAR / ChiNext need a year, others a
# quarter.
LISTED_DAYS = 91
LISTED_DAYS_GROWTH = 365
