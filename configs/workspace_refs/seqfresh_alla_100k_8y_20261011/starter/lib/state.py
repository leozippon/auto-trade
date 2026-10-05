"""The refit counter behind the warm start, and the cold-reset cadence.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers the one question each refit asks: is
this the first fit of the replay, or is there a checkpoint from the previous
one to continue from? A replay stays reproducible from PIT data alone.

`COLD_EVERY` bounds a warm chain: the recipe starts every fourth quarterly
refit from a fresh initialisation, a yearly reset, and a monthly cadence
keeps that reset yearly (every twelfth refit). The cold refits of a monthly
leg therefore fall on the same decision days as c_base's -- the replay's first
fit and every twelve months after it -- and the leg differs from c_base only
in the warm refits between them. Every leg must reset on those days; a
control that refits cold while the candidate warm-starts is not a control.
"""

import numpy as np

COLD_EVERY = {"quarter": 4, "month": 12}   # refits between two cold starts, per main.REFIT_PERIOD

META_FILE = "/refit_meta.npy"


def _path(context):
    return context.state_dir + META_FILE


def count(context):
    """Refits this replay has already completed; 0 on the empty state directory of a fresh replay."""

    try:
        return int(np.load(_path(context))[0])
    except FileNotFoundError:
        return 0


def is_cold(context, refit_period):
    return (count(context) % COLD_EVERY[refit_period]) == 0


def bump(context, cold, reading):
    """Record the completed refit: [count, was it cold, the fitting path's validation reading]."""

    np.save(_path(context), np.asarray([count(context) + 1, 1.0 if cold else 0.0, float(reading)]))
