"""The pre-registered leg table and the embargoed train/valid split of `fit`.

`LEGS` is the one place that says what each leg is: which score it ranks on and
whether its book is filled inside the exposure buckets. `lib.trade` reads both
entries; `lib.score_lgbm` reads the first, so a leg that carries no learned
score cannot quietly train one. The table lives here rather than in either
module so both can read it without importing each other.
"""

import numpy as np

# leg -> (score source, bucketed fill). "lgbm" is the only source that trains.
LEGS = {
    "s_pvb": ("lgbm", True),
    "c_uncapped": ("lgbm", False),
    "c_rand": ("random", True),
    "c_pool": ("pool", False),
}

VALID_DAYS = 60
EMBARGO_DAYS = 10
MIN_SAMPLES = 200


def trains(candidate):
    """True when the leg ranks on the learned score, i.e. when `fit` is real."""
    return LEGS[candidate][0] == "lgbm"


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
