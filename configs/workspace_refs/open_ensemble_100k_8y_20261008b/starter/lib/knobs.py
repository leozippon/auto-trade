"""Every book default this arm is expected to decide, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only. None of these values is a
finding: they are placeholders for a skeleton that runs.
"""

# The benchmark the arm is graded against. A strategy cannot read the run fact
# `benchmark_index`, so round 0 reads it and changes this constant to match it
# -- the run fact overrides this literal, never the other way round. The book
# uses it for the pool "csi1000" and to report how much of itself sits inside
# the index sections.
INDEX = "000852.SH"
# "alla": every tradable non-Beijing, non-STAR, non-ST A-share.
# "csi1000": only the members of the newest visible INDEX section.
POOL = "alla"
SEATS = 12
# Share of the book value spread over the seats; the rest is the buffer
# between the T-1 close the orders are sized at and the 15:00 fill price.
CAPITAL = 0.97
# At a review a holding ranked inside SEATS * KEEP_BAND is kept.
KEEP_BAND = 2.0
# At most this many names per Shenwan L1 industry of the replay year's
# universe (names without a label share the bucket "未分类").
INDUSTRY_CAP = 3
# "week": the first decision of each ISO week; "month": of each calendar month.
REVIEW = "month"
# False: the score's own ranking. True: the control c_shuf, the same values on
# stand-in names fixed across reviews (lib/controls.held_shuffle), which keeps
# the candidate's holding period and turnover.
SHUFFLE = False
