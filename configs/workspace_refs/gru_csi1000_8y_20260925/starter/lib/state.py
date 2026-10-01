"""The refit counter that makes a warm start possible, and the cold-reset cadence.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers one question each refit asks: is this
the first fit of this replay, or is there a checkpoint from the previous one to
continue from? Nothing about the count reaches outside the replay -- a replay
is still reproducible from PIT data alone, it simply carries what it learned in
its own earlier quarters forward instead of discarding it.

`COLD_EVERY` bounds how far a warm chain may run before it is reset: with a
quarterly refit, every fourth one starts from a fresh initialisation. Without
that, a checkpoint fitted in the first research year could still be steering the
model in the last one with no path back, and no reading would tell the arm
whether the chain or the data was doing the work.

The counter lives here, not in a model file, because the candidate and its
control must reset on exactly the same refits; a control that refits cold while
the candidate warm-starts is not a control.
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
    """Record the completed refit: [count, was it cold, the fitting path's own validation reading].

    The reading is the GRU's mean best validation IC or, when the control is the
    candidate, the control's validation L2 -- two different numbers, kept only so
    a session can read what the last refit achieved without rerunning it. Nothing
    in the decision path depends on it.
    """

    np.save(_path(context), np.asarray([count(context) + 1, 1.0 if cold else 0.0, float(reading)]))
