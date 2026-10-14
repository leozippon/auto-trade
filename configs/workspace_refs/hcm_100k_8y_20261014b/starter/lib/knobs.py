"""Every book and model default this arm is expected to decide, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module restates a value
from this file; they refer to these names only. None of these values is a
finding: they are placeholders for a skeleton that runs.
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
# False: the score's own ranking. True: the control c_shuf, the same values
# moved onto stand-in names fixed across reviews (lib/controls.held_shuffle).
SHUFFLE = False

# --- the model (lib/hcm.py): one seed line and the three memory levels ---

# The one integer every draw derives from; a seed replicate changes this line.
SEED = 1000
# Parameter memory, what the daily fit does to the weights.
#   "graded"       the head, body and input groups each step at their own
#                  learning rate and period below (full HCM)
#   "uniform"      every group at the head's rate and period (A3)
#   "reversed"     the input group at the head's pair, the head at the input's (A4)
#   "frozen"       no update after the initial fit (C2, A1)
#   "cold_quarter" re-initialised and pretrained afresh at the first fit of
#                  every quarter, untouched in between (C1, the census practice)
UPDATE = "graded"
# Short-term memory, the recurrent state of each name.
#   "carried"      rolled one day forward at every daily fit and read as it
#                  stands by the decision; never reset while the replay runs
#   "window"       recomputed from zero over the last SEQ_DAYS at every call
#                  (C1, C2: the conventional sequence model)
STATE = "carried"
# External memory: the episode store, which feeds the head a retrieved signal
# and replays stored episodes into the body and head. False removes both (A2).
MEMORY = True
# Learning rate and update period (trading days) of each parameter group
# under "graded": fastest at the output, slowest at the input.
HEAD_LR = 1e-3
HEAD_PERIOD = 1
BODY_LR = 3e-4
BODY_PERIOD = 5
INPUT_LR = 1e-4
INPUT_PERIOD = 20
# The store holds at most STORE_SIZE episodes, kept by reservoir sampling over
# everything offered; a query reads its KNN nearest stored embeddings by
# cosine; a daily step replays REPLAY_BATCH stored episodes.
STORE_SIZE = 200_000
KNN = 32
REPLAY_BATCH = 2048
# The initial fit, and every cold refit: PRETRAIN_EPOCHS passes over the
# newest PRETRAIN_DAYS trading days at PRETRAIN_LR for every group.
PRETRAIN_DAYS = 500
PRETRAIN_EPOCHS = 3
PRETRAIN_LR = 1e-3
# The model: HIDDEN-wide state and embedding; the recurrent block reads the
# last SEQ_DAYS days of features; the label is the HORIZON-day forward
# residual return, rank-centred across the day's names.
HIDDEN = 80
SEQ_DAYS = 20
HORIZON = 20
