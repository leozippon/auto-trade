"""Every default this arm registers as an open axis or aligns to the run facts, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` lists the allowed values of each axis.

Round 0 aligns INDEX to the run fact `benchmark_index` and CAPITAL to
`broker_replay.initial_cash`, never the reverse: the strategy cannot read
either fact at decision time.
"""

# --- the arm -------------------------------------------------------------
# The benchmark: the beta leg of the label the signs are fitted on. The census
# features read CSI 1000 for their own definitions whatever this says.
INDEX = "000852.SH"
# The account the shape was registered for; a flat account holding less than
# half or more than twice this stops the replay (lib/book.py).
CAPITAL = 100_000

# --- the book (lib/book.py) ------------------------------------------------
SEATS = 12
# "biweekly" (halves of the calendar month) or "monthly".
REVIEW = "biweekly"
# Fixed in this arm: a holding is kept while its rank is below SEATS * KEEP_BAND,
# at most round(SEATS * SWAP_SHARE) band exits a review, no industry cap.
KEEP_BAND = 2.0
SWAP_SHARE = 0.16
INDUSTRY_CAP = None
CASH_BUFFER = 0.97

# --- the composite (lib/learn.py) ------------------------------------------
# Fixed: label horizon in sessions, open of t+1 to open of t+1+HOLD.
HOLD = 20
# Trailing years of signal dates each quarterly sign fit reads.
SIGN_YEARS = 3
# "feature": every feature weighs 1/44. "family": every census family 1/19.
WEIGHTING = "feature"
