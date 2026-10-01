"""Every default this arm registers as an open axis or aligns to the run facts, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` lists the allowed values of each axis.
The two census_gbdt packs ship the same starter: this file is the only one in
which they differ.

Round 0 aligns INDEX to the run fact `benchmark_index` and CAPITAL to
`broker_replay.initial_cash`, never the reverse: the strategy cannot read
either fact at decision time.
"""

# --- the arm -------------------------------------------------------------
# The benchmark: the beta leg of the label. The census features read CSI 1000
# for their own definitions whatever this says.
INDEX = "000852.SH"
# The account the shape was registered for; a flat account holding less than
# half or more than twice this stops the replay (lib/book.py).
CAPITAL = 1_000_000
# "alla": every A-share except Beijing, STAR and ST / 退 names, with at least
# learn.MIN_BARS bars. "csi1000": the same, inside the CSI 1000 section in force.
POOL = "alla"

# --- the book ------------------------------------------------------------
SEATS = 50
# "weekly", "biweekly" (halves of the calendar month) or "monthly".
REVIEW = "biweekly"
# A holding is kept while its rank in the scored pool is below SEATS * KEEP_BAND.
KEEP_BAND = 2.0
# At most round(SEATS * SWAP_SHARE) band exits per review; forced exits always go.
SWAP_SHARE = 0.16
# None, or a fraction: at most floor(INDUSTRY_CAP * SEATS) names per SW level-1.
INDUSTRY_CAP = None
CASH_BUFFER = 0.97

# --- the model -----------------------------------------------------------
# Label horizon in sessions: open of t+1 to open of t+1+HOLD, minus beta x INDEX.
HOLD = 20
# Trailing years of signal dates each quarterly fit trains on.
TRAIN_YEARS = 3
# LightGBM's seed. A second value is the mandatory replicate before nomination.
SEED = 7
# Census families removed from the model's input; only for registered
# ablation controls (family keys: census_features.families()).
DROP_FAMILIES = ()
