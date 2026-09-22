"""The embargoed train/validation split the primary fits on.

A label built from a HORIZON-trading-day forward window overlaps its
neighbours, so two rows HORIZON bars apart are not independent. A split that
ignores that leaks the answer across the boundary and the leak looks exactly
like skill. Validation is the last `knobs.VALID_DAYS` decision bars of the
window; training stops `knobs.EMBARGO_DAYS` bars before the validation start.

This is the graduated lineage's split, unchanged. The purged out-of-fold pass
that lineage's meta successor added is NOT here: this arm has no second stage
to feed, and an out-of-fold score it never reads would only cost fit seconds.
"""

import numpy as np

from lib import knobs


def split_fit_dates(bars):
    """(train mask, valid mask, training rows, distinct training bars)."""

    present = np.unique(bars)
    if present.size <= knobs.VALID_DAYS:
        empty = np.zeros(bars.shape, dtype=bool)
        return empty, empty, 0, int(present.size)
    valid_bars = present[-knobs.VALID_DAYS:]
    train_max = int(valid_bars[0]) - knobs.EMBARGO_DAYS
    train = bars <= train_max
    valid = np.isin(bars, valid_bars)
    return train, valid, int(train.sum()), int(np.unique(bars[train]).size)
