"""The pre-registered leg table, the capital split, and the embargoed train/valid split of `fit`.

`LEGS` is the one place that says what each leg is: which sleeves it runs, what
share of capital each sleeve gets and how many seats it fills. `lib.trade` reads
the whole table; `lib.score_lgbm` reads only whether the learned sleeve is in it,
so a leg that runs no learned sleeve cannot quietly train one. The table lives
here rather than in either module so both can read it without importing each other.

The two sleeve names are the two mechanisms, not two scores of one book:

    "composite"  the fit-free monthly rank composite on a float-cap-floored pool
    "pv"         the learned weekly Alpha158 + LightGBM cross-sectional ranker

`W_A` is the blend's whole free parameter (variant axis a): sleeve `composite`
takes `W_A` of the capital and sleeve `pv` takes the rest. Nothing else in the
package depends on the split.
"""

import numpy as np

SLEEVE_A = "composite"
SLEEVE_B = "pv"

W_A = 0.30                   # sleeve A's share of capital in the blend; sleeve B takes 1 - W_A
SEATS = 15                   # seats per sleeve

# leg -> {sleeve: (capital weight, seats)}. The four legs differ in this line alone.
LEGS = {
    "s_blend": {SLEEVE_A: (W_A, SEATS), SLEEVE_B: (1.0 - W_A, SEATS)},
    "c_pv": {SLEEVE_B: (1.0, SEATS)},
    "c_comp": {SLEEVE_A: (1.0, SEATS)},
    "c_wide30": {SLEEVE_B: (1.0, 2 * SEATS)},
}

VALID_DAYS = 60
EMBARGO_DAYS = 10
MIN_SAMPLES = 200


def trains(candidate):
    """True when the leg runs the learned sleeve, i.e. when `fit` is real."""
    return SLEEVE_B in LEGS[candidate]


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
