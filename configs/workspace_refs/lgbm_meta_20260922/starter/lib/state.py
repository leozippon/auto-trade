"""The refit counter that makes a warm start possible, and the cold-reset cadence.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers one question each refit asks: is this
the first fit of this replay, or is there a booster from the previous one to
continue? Nothing about the count reaches outside the replay -- a replay is
still reproducible from PIT data alone, it simply carries the trees it grew in
its own earlier quarters forward instead of discarding them.

`COLD_EVERY` bounds how far a warm chain may run before it is reset: with a
quarterly refit, every fourth one starts from a fresh booster. Without that, an
ensemble started in the first research year would keep growing all the way
through the last one with no path back, and no reading would tell the arm
whether the chain or the data was doing the work.

The counter lives here, not in a model file, because the candidate and its
control must reset on exactly the same refits; a control that refits cold while
the candidate continues is not a control. The out-of-fold pass inside a refit is
always cold regardless of this counter -- a score from a booster that already
read the fold is not an out-of-fold score.
"""

import numpy as np

COLD_EVERY = 4               # refits between two cold starts: quarterly refit -> a yearly reset

META_FILE = "/refit_meta.npy"


def _path(context):
    return context.state_dir + META_FILE


def count(context):
    """Refits this replay has already completed; 0 on the empty state directory of a fresh replay."""

    try:
        return int(np.load(_path(context))[0])
    except FileNotFoundError:
        return 0


def is_cold(context):
    return (count(context) % COLD_EVERY) == 0


def bump(context, cold, reading):
    """Record the completed refit: [count, was it cold, the primary's validation L2].

    The reading is kept only so a session can see what the last refit achieved
    without rerunning it; nothing in the decision path depends on it.
    """

    np.save(_path(context), np.asarray([count(context) + 1, 1.0 if cold else 0.0, float(reading)]))
