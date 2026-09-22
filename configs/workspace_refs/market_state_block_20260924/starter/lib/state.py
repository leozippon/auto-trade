"""The refit counter that makes a warm start possible, and the cold-reset cadence.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers one question each refit asks: is this
the first fit of this replay, or is there a booster from the previous one to
continue? Nothing about the count reaches outside the replay -- a replay is
still reproducible from PIT data alone, it simply carries the trees it grew in
its own earlier quarters forward instead of discarding them.

`knobs.COLD_EVERY` bounds how far a warm chain may run before it is reset.
Without that, a booster started in the first research year would keep growing
all the way through the last one with no path back, and no reading would tell
the arm whether the chain or the data was doing the work.

The counter lives here, not in a model file, because the candidate and its
control must reset on exactly the same refits; a control that refits cold while
the candidate continues is not a control.
"""

import numpy as np

from lib import knobs

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
    return (count(context) % knobs.COLD_EVERY) == 0


def bump(context, cold, reading):
    """Record the completed refit: [count, was it cold, the primary's validation L2]."""

    np.save(_path(context), np.asarray([count(context) + 1, 1.0 if cold else 0.0, float(reading)]))
