"""Every switch of this arm, in one place.

Each assignment below appears exactly once in the package, so an `edit_file`
on any of them matches one line; no other module restates a value from here.
`refs/README.md` is the authority for the allowed values of each axis and the
batch in which a value other than the starter's may be spent.
"""

# The benchmark. A strategy cannot read the run fact `benchmark_index`, so
# round 0 checks it against this constant and stops if they differ. The book
# uses it for the pool's CSI 1000 membership and to report how much of itself
# sits inside the index.
INDEX = "000852.SH"

# The leg this package is: "f_all" (main candidate), "f_vol" or "f_beta"
# (secondary), "c_orig" or "c_shuf" (controls). lib/score.py defines them.
CANDIDATE = "f_all"

# --- open axes (refs/README.md) -------------------------------------------
# "alla": every A-share except Beijing, STAR and ST / 退 names.
# "csi1000": the same, inside the newest visible INDEX section.
POOL = "alla"
# Equal-cash seats: 30 or 50.
SEATS = 50
# At a review a holding ranked inside SEATS * KEEP_BAND is kept: 2.0 is the
# band, 1.0 switches it off (the book is re-cut to the top SEATS).
KEEP_BAND = 2.0
# "equal": every score of the leg weighs the same. "family": the scores are
# averaged inside each census family first, then the families equally.
WEIGHTING = "equal"

# --- fixed ----------------------------------------------------------------
# At most a fifth of the seats per Shenwan level-1 industry of the replay
# year's universe (names without a label share the bucket "未分类").
INDUSTRY_CAP = SEATS // 5
# Share of the book value spread over the seats; the rest is the buffer
# between the T-1 close the orders are sized at and the fill price.
CAPITAL = 0.97
REVIEW = "month"
# The control c_shuf trades f_all's values permuted among the scored names.
SHUFFLE = CANDIDATE == "c_shuf"
