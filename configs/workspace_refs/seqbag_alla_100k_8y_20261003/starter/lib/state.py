"""The refit counter behind the warm start, and the cold-reset cadence.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers the one question each refit asks: is
this the first fit of the replay, or is there a checkpoint from the previous
one to continue from? A replay stays reproducible from PIT data alone.

`COLD_EVERY` bounds a warm chain: with a quarterly refit, every fourth refit
starts from a fresh initialisation (the recipe's cadence). The counter lives
here because every candidate -- bag, single head and tree -- must reset on the
same refits; a control that refits cold while the candidate warm-starts is not
a control.
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
    """Record the completed refit: [count, was it cold, the fitting path's validation reading]."""

    np.save(_path(context), np.asarray([count(context) + 1, 1.0 if cold else 0.0, float(reading)]))
