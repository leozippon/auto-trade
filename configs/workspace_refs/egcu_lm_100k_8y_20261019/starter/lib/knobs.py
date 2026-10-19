"""Every book and model default this arm is expected to decide, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only. None of these values is a
finding: they are placeholders for a baseline that runs.
"""

# --- the book (lib/trade.py) ---

# The benchmark the arm is graded against. A strategy cannot read the run fact
# `benchmark_index`, so round 0 reads it and changes this constant to match it
# -- the run fact overrides this literal, never the other way round. The book
# uses it for the pool "csi1000", the residual label's market leg, and to
# report how much of itself sits inside the index sections.
INDEX = "000852.SH"
# "alla": every tradable non-Beijing, non-STAR, non-ST A-share.
# "csi1000": only the members of the newest visible INDEX section.
POOL = "alla"
SEATS = 20
# Share of the book value spread over the seats; the rest is the buffer
# between the T-1 close the orders are sized at and the 15:00 fill price.
CAPITAL = 0.97
# At a review a holding ranked inside SEATS * KEEP_BAND is kept.
KEEP_BAND = 2.0
# At most this many names per Shenwan L1 industry of the replay year's
# universe (names without a label share the bucket "未分类").
INDUSTRY_CAP = 3
# "week": the first decision of each ISO week; "month": of each calendar month.
# Measured on one quarter with a frozen head: monthly reviews at this keep
# band sell about five books a year, weekly ones about twenty.
REVIEW = "month"
# False: the score's own ranking. True: the control c_shuf, the same values
# moved onto stand-in names fixed across reviews (lib/controls.held_shuffle).
SHUFFLE = False

# --- the model (lib/learn.py, lib/score.py): one seed line and the baseline ---

# The one integer every draw derives from; a seed replicate changes this line.
SEED = 1000
# How the fitted head changes over a replay.
#   "cold_quarter"  refitted from its initialisation on the trailing window at
#                   the first fit of every quarter, untouched in between: the
#                   platform's usual practice and the control every proposal
#                   must beat
#   "frozen"        fitted once, before the first decision, never again
# An online update rule is the Agent's to write: lib/update.py.
UPDATE = "cold_quarter"
# False: the numeric features alone, the title encoder never loaded -- the
# text-off twin of any candidate, which isolates the text channel.
TEXT = True
# The frozen title encoder, a directory under context.pretrained_dir:
#   "Qwen3.5-0.8B"  24 layers, three Gated DeltaNet (linear attention) layers
#                   per full-attention layer, hidden size 1024
#   "Qwen3-0.6B"    28 plain-attention layers, hidden size 1024; several times
#                   faster while the image lacks the fused linear-attention
#                   kernels (lib/encoder.py)
LM = "Qwen3.5-0.8B"
# Titles per forward pass. On the fallback linear-attention path the 0.8B
# model peaks near 5 GiB at 128; the 0.6B model near 1.3 GiB.
ENCODE_BATCH = 128
# A name's text at a decision: the newest MAX_TITLES of its titles that first
# became visible within the last TEXT_DAYS decision days, this one included.
TEXT_DAYS = 5
MAX_TITLES = 8
# The head (lib/model.py): the text embedding projected to TEXT_DIM beside a
# has-text flag and the ten numeric features, through one HIDDEN-wide layer.
TEXT_DIM = 32
HIDDEN = 64
# A fit: EPOCHS passes over the newest PRETRAIN_DAYS trading days at LR, in
# batches of DAYS_PER_BATCH days.
PRETRAIN_DAYS = 500
EPOCHS = 3
LR = 1e-3
DAYS_PER_BATCH = 4
# The label is the HORIZON-day forward residual return, rank-centred across
# the day's names (lib/features.py).
HORIZON = 20
