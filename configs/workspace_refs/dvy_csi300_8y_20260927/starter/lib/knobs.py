"""Every default this arm is expected to change, in one place.

Each assignment appears exactly once in the whole package and no other module
restates a value from this file. The four value / turnover packs of this round
ship the same starter; their `lib/knobs.py` files differ only in INDEX and
SCORE.

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
# book on a 10万 account is another registration, not a quiet variant.
CAPITAL = 1_000_000

# --- the book ------------------------------------------------------------
SEATS = 50
# A holding is kept while its rank in the scored pool is below
# SEATS * KEEP_BAND; there is no swap cap.
KEEP_BAND = 2.0
# At most floor(INDUSTRY_CAP * SEATS) names per as-of SW level-1 industry,
# kept names and buys together; below 0.30 because equal cash drifts.
INDUSTRY_CAP = 0.28
CASH_BUFFER = 0.97

# --- the score (defined once, in lib/score.py) ----------------------------
# "ep", "dvy", "abturn" and "ch3" are the round's four registered scores.
# "range20" is the other eight-year arms' score, kept only as a labelled
# diagnostic: never a leg of this arm.
SCORE = "dvy"
# ep and ch3: the earnings multiple read at T-1. "pe" (the latest fiscal
# year's earnings) is the registered neighbour of ep and ch3.
EP_COLUMN = "pe_ttm"
# dvy: the yield columns in order of use; the first positive one counts.
# ("dv_ratio",) alone is the registered neighbour of dvy.
DV_COLUMNS = ("dv_ttm", "dv_ratio")
# abturn and ch3: market sessions of the short and the long mean turnover.
# TURN_LONG = 120 is the registered neighbour of abturn and ch3.
TURN_SHORT = 20
TURN_LONG = 250
# range20 (diagnostic only): own bars averaged.
RANGE_DAYS = 20
# True turns the package into the control c_shuf.
SHUFFLE = False
