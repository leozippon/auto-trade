"""Shared fit-time constants and the embargoed train/valid date split.

Imported by the model scorers. Each candidate dir ships this file, so no scorer
may import a sibling scorer.
"""

import numpy as np

VALID_DAYS = 60
EMBARGO_DAYS = 10
MIN_SAMPLES = 200


def split_fit_dates(date_idx):
    """(train_mask, valid_mask, n_train, n_train_dates) with the h=10 embargo.

    Validation segment = the last VALID_DAYS decision bars of the visible window; the
    training segment is restricted to bars <= (validation first bar - EMBARGO_DAYS),
    i.e. at least EMBARGO_DAYS trading bars before the validation start.
    """
    present = np.unique(date_idx)
    if present.size <= VALID_DAYS:
        # Too few distinct decision bars for an embargoed split: nothing to train.
        tr = np.zeros(date_idx.shape, dtype=bool)
        va = np.zeros(date_idx.shape, dtype=bool)
        return tr, va, 0, int(present.size)
    va_dates = present[-VALID_DAYS:]
    train_max = int(va_dates[0]) - EMBARGO_DAYS
    tr = date_idx <= train_max
    va = np.isin(date_idx, va_dates)
    return tr, va, int(tr.sum()), int(np.unique(date_idx[tr]).size)
