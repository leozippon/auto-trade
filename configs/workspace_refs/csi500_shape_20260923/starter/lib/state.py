"""The refit counter that makes a warm start possible, the cold-reset cadence, and its reading.

`context.state_dir` is empty at the start of every replay and writable only
while `fit` runs, so this counter answers the one question each refit asks: is
this the first fit of this replay, or is there a booster from the previous one
to continue? Nothing about the count reaches outside the replay -- a replay is
still reproducible from PIT data alone, it simply carries the trees it grew in
its own earlier quarters forward instead of discarding them.

COLD_EVERY bounds how far a warm chain may run before it is reset: with a
quarterly refit, every fourth one starts from a fresh booster. Without that, an
ensemble started in the first research year would keep growing all the way
through the last one with no path back, and no reading would tell the arm
whether the chain or the data was doing the work.

`COLD_EVERY = 1` makes every refit cold. That is the arm's registered
persistence ablation, and it is a CONTROL, not a candidate: this repository has
measured cold beating warm on one controlled leg, no monotonic warm-start gain
on another, and warm refits that were not even faster, three for three with no
counter-evidence. A pack that ships continuation without pricing it would be
repeating a claim the repository has already failed to confirm.

`report` exists because the state directory is discarded when a replay ends:
without it the arm cannot say afterwards how many refits happened and how many
of them were cold, and `exploration-plan.md` requires exactly that reading. It
is read-only and is called from `generate_orders`, where the directory is
mounted read-only.
"""

import numpy as np

COLD_EVERY = 4               # refits between two cold starts; 1 = the persistence ablation

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
    return (count(context) % COLD_EVERY) == 0


def bump(context, cold, reading):
    """Record the completed refit: [count, was it cold, cold refits so far, the validation L2]."""

    record = _record(context)
    cold_so_far = (0 if record is None else int(record[2])) + (1 if cold else 0)
    np.save(_path(context), np.asarray(
        [count(context) + 1, 1.0 if cold else 0.0, cold_so_far, float(reading)]))


def report(context):
    """What the last refit did, for the order metadata: refits so far, cold ones, last L2."""

    record = _record(context)
    if record is None:
        raise RuntimeError("no refit record under the state directory")
    return {"refits": int(record[0]), "cold_refits": int(record[2]),
            "last_refit_cold": bool(record[1]), "valid_l2": round(float(record[3]), 6)}
