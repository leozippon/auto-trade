"""The refit counter that makes a warm start possible, the cold-reset cadence, and its reading.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers the one question each refit asks: is
this the first fit of this replay, or is there a booster from the previous one
to continue? Nothing about the count reaches outside the replay -- a replay is
still reproducible from PIT data alone, it simply carries the trees it grew in
its own earlier quarters forward instead of discarding them.

`knobs.COLD_EVERY` bounds how far a warm chain may run before it is reset.
Without that, a booster started in the first research year would keep growing
all the way through the last one with no path back, and no reading would tell
the arm whether the chain or the data was doing the work.

The counter lives here, not in a model file, because the candidates and their
control must count refits on exactly the same schedule: every leg of this arm
continues its booster on this cadence, so the fits of `c_base`, `f1`, `f2` and
`f4` are the same booster file for file, and a difference between them is the
construction's alone.

`report` exists because the state directory is discarded when a replay ends:
without it the arm cannot say afterwards how many refits happened and how many
of them were cold. It is read-only and is called from `generate_orders`, where
the directory is mounted read-only.
"""

import numpy as np

from lib import knobs

META_FILE = "/refit_meta.npy"


def _path(context):
    return context.state_dir + META_FILE


def _record(context):
    try:
        return np.load(_path(context))
    except FileNotFoundError:
        return None


def count(context):
    """Refits this replay has already completed; 0 on the empty state directory of a fresh replay."""

    record = _record(context)
    return 0 if record is None else int(record[0])


def is_cold(context):
    return (count(context) % knobs.COLD_EVERY) == 0


def bump(context, cold, reading):
    """Record the completed refit: [count, was it cold, cold refits so far, the validation reading]."""

    record = _record(context)
    cold_so_far = (0 if record is None else int(record[2])) + (1 if cold else 0)
    np.save(_path(context), np.asarray(
        [count(context) + 1, 1.0 if cold else 0.0, cold_so_far, float(reading)]))


def report(context):
    """What the last refit did, for the order metadata."""

    record = _record(context)
    if record is None:
        raise RuntimeError("no refit record under the state directory")
    return {"refits": int(record[0]), "cold_refits": int(record[2]),
            "last_refit_cold": bool(record[1]), "valid_metric": round(float(record[3]), 6)}
