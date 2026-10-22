"""Every book, memory and model default this arm is expected to decide, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only. None of these values is a
finding: they are placeholders for a baseline that runs.
"""

# --- the book (lib/trade.py) ---

# The benchmark the arm is graded against. A strategy cannot read the run fact
# `benchmark_index`, so round 0 reads it and changes this constant to match it
# -- the run fact overrides this literal, never the other way round. The book
# uses it for the pool "csi1000", the label's market leg and the patterns'
# excess returns, and to report how much of itself sits inside the index.
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
REVIEW = "month"
# False: the score's own ranking. True: the control c_shuf, the same values
# moved onto stand-in names fixed across reviews (lib/controls.held_shuffle).
SHUFFLE = False

# --- the scorer (lib/learn.py): one seed line, the memory and the carrier ---

# The one integer every draw derives from; a seed replicate changes this line.
# The retrieval scorer draws nothing, so its fills do not depend on it; the
# carrier's initialisation, training sample and batch order do.
SEED = 1000
# What scores the names each day, from the same keys and the same memory:
#   "retrieval"     the outcomes of the K nearest matured cases in the memory,
#                   kernel-weighted by their distance (lib/memory.py)
#   "cold_quarter"  C1: a small network fitted from its initialisation on the
#                   matured cases at the first fit of every quarter, untouched
#                   in between (lib/model.py)
SCORER = "retrieval"
# How the memory grows over a replay:
#   "continuous"  every case enters the day its outcome is known
#   "frozen"      only the cases matured before the replay's first decision
MEMORY = "continuous"
# Which outcomes a retrieval averages:
#   "similar"  the retrieved cases' own outcomes
#   "placebo"  the same retrieval, each case's outcome swapped for another
#              case's of the same day by a fixed hash: random neighbours that
#              keep the candidate's day-to-day score dynamics and turnover
NEIGHBOURS = "similar"
# False: keys are the price-volume pattern alone, the title encoder never
# loaded -- the text-off twin of any candidate.
TEXT = True

# --- the keys (lib/features.py, lib/encoder.py) ---

# The pattern: the last WINDOW trading days of excess return and turnover,
# scaled by the name's own VOL_DAYS-day history.
WINDOW = 20
VOL_DAYS = 60
# The title encoder, a directory under context.pretrained_dir, and the leading
# dimensions of its (Matryoshka-trained) embedding that a key keeps.
ENCODER = "Qwen3-Embedding-0.6B"
TEXT_DIM = 64
# Titles per forward pass of the encoder.
ENCODE_BATCH = 256
# A name's text at a decision: every title first visible within the last
# TEXT_DAYS decision days, this one included, averaged.
TEXT_DAYS = 5
# The text block's share of a key: its squared length is TEXT_WEIGHT times
# the pattern block's expected squared length, so 1.0 weighs them alike.
TEXT_WEIGHT = 1.0

# --- the memory (lib/memory.py) ---

# Decision days of cases the first fit of a replay files before its first
# decision; the memory then grows by one day per trading day.
MEMORY_DAYS = 500
# Neighbours per query, and the kernel: weights exp(-d2 / (BANDWIDTH * the
# median d2 of the query's K neighbours)).
K = 64
BANDWIDTH = 0.5

# --- the carrier C1 (lib/model.py) ---

# A refit draws at most TRAIN_CASES matured cases and makes EPOCHS passes over
# them at LR in batches of BATCH cases, through one HIDDEN-wide layer.
TRAIN_CASES = 2_000_000
EPOCHS = 3
LR = 1e-3
BATCH = 8192
HIDDEN = 64

# --- the label (lib/features.py) ---

# The HORIZON-day forward residual return, rank-centred across the day's names.
HORIZON = 20
