"""The embargoed train/validation split the carrier fits on.

A label built from an N-trading-day forward window overlaps its neighbours, so
two rows N bars apart are not independent. A split that ignores that leaks the
answer across the boundary and the leak looks exactly like skill. Validation is
the last `knobs.VALID_DAYS` decision bars of the window; training stops one
label width (the leg's horizon) before the validation start.

This is the graduated lineage's split, unchanged for `knobs.HORIZON`; a leg
that trains on another horizon (`f3`) widens the embargo with it.
"""

import numpy as np

from lib import knobs


def split_fit_dates(bars, embargo):
    """(train mask, valid mask, training rows, distinct training bars)."""

    present = np.unique(bars)
    if present.size <= knobs.VALID_DAYS:
        empty = np.zeros(bars.shape, dtype=bool)
        return empty, empty, 0, int(present.size)
    valid_bars = present[-knobs.VALID_DAYS:]
    train_max = int(valid_bars[0]) - embargo
    train = bars <= train_max
    valid = np.isin(bars, valid_bars)
    return train, valid, int(train.sum()), int(np.unique(bars[train]).size)
