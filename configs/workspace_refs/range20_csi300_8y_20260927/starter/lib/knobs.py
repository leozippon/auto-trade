"""Every default this arm is expected to change, in one place.

Each assignment appears exactly once in the whole package and no other module
restates a value from this file. The three range20 packs of this round ship
the same starter; their `lib/knobs.py` files are the only difference.

Round 0 aligns two knobs to the run facts, never the reverse: INDEX to
`benchmark_index` (the host grades beta, tracking error, the neutralisation
and the zero-skill panel's membership matching against that index) and
CAPITAL to `broker_replay.initial_cash`. A strategy cannot read either fact at
decision time, which is why they are constants the session aligns once.
"""

# --- the arm -------------------------------------------------------------
INDEX = "000300.SH"
# The account this shape was registered for. A flat account whose cash is
# not within a factor of two of it stops the replay (lib/trade.py): a 50-seat
# book on a 10万 account, or a 12-seat book on a 100万 one, is another
# registration, not a quiet variant of this one.
CAPITAL = 1_000_000

# --- the book ------------------------------------------------------------
SEATS = 50
# A holding is kept while its rank in the scored pool is below
# SEATS * KEEP_BAND; there is no swap cap (the round-0 census found that a
# seats/5 cap would bind at most reviews and hold names the score dropped).
KEEP_BAND = 2.0
# At most floor(INDUSTRY_CAP * SEATS) names per as-of SW level-1 industry,
# kept names and buys together; below 0.30 because equal cash drifts.
INDUSTRY_CAP = 0.28
CASH_BUFFER = 0.97

# --- the score -----------------------------------------------------------
# Own visible bars averaged: 20 is r1 (and c_shuf); 60 is the neighbour r2.
RANGE_DAYS = 20
# True turns the package into the control c_shuf.
SHUFFLE = False
