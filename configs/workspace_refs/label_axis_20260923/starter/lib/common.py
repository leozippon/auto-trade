"""The embargoed train/validation split, with the embargo tied to the label's own horizon.

A label built from an N-trading-day forward window overlaps its neighbours: two
rows fewer than N bars apart share part of their outcome. A split that ignores
that leaks the answer across the boundary, and the leak looks exactly like
skill.

The graduated lineage hard-coded a 10-bar embargo because its label was always
10 bars long. This arm varies the horizon, so the embargo is an ARGUMENT, not a
constant: `main.fit` passes the horizon of the label it is fitting. A 40-day
label split with a 10-bar embargo would put 30 bars of shared outcome on both
sides of the boundary and read as a better model than it is -- and it would do
so only for the longer-horizon members of the family, i.e. it would bias the
very comparison this arm exists to make.

Validation is the last VALID_DAYS decision bars of the window; training stops
`embargo` bars before the validation segment starts.
"""

import numpy as np

VALID_DAYS = 60
MIN_SAMPLES = 200


def split_fit_dates(bars, embargo):
    """(train mask, valid mask, training rows, distinct training bars) with an `embargo`-bar gap."""

    if embargo < 1:
        raise ValueError(f"embargo must be at least one bar, got {embargo}")
    present = np.unique(bars)
    if present.size <= VALID_DAYS:
        empty = np.zeros(bars.shape, dtype=bool)
        return empty, empty, 0, int(present.size)
    valid_bars = present[-VALID_DAYS:]
    train_max = int(valid_bars[0]) - embargo
    train = bars <= train_max
    valid = np.isin(bars, valid_bars)
    return train, valid, int(train.sum()), int(np.unique(bars[train]).size)
