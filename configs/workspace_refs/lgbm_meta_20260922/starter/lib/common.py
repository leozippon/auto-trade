"""The embargoed train/validation split and the purged folds the meta stage needs.

Both exist for the same reason: a label built from a 10-trading-day forward
window overlaps its neighbours, so two rows ten bars apart are not independent.
A split that ignores that leaks the answer across the boundary and the leak
looks exactly like skill.

`split_fit_dates` is the frozen lineage's split, unchanged: validation is the
last `VALID_DAYS` decision bars of the window and training stops `EMBARGO_DAYS`
bars before the validation start.

`purged_folds` is what the meta stage adds. The primary must be scored on bars
it did not train on, or the meta classifier learns the primary's in-sample
overfit instead of its out-of-sample behaviour; contiguous folds with an
`EMBARGO_DAYS` purge on BOTH sides give that. Contiguous, not random: a random
k-fold over bars puts a row's neighbours in the training set by construction and
turns the purge into decoration.
"""

import numpy as np

VALID_DAYS = 60
EMBARGO_DAYS = 10
MIN_SAMPLES = 200
OOF_FOLDS = 4


def split_fit_dates(bars):
    """(train mask, valid mask, training rows, distinct training bars) with the h=10 embargo."""

    present = np.unique(bars)
    if present.size <= VALID_DAYS:
        empty = np.zeros(bars.shape, dtype=bool)
        return empty, empty, 0, int(present.size)
    valid_bars = present[-VALID_DAYS:]
    train_max = int(valid_bars[0]) - EMBARGO_DAYS
    train = bars <= train_max
    valid = np.isin(bars, valid_bars)
    return train, valid, int(train.sum()), int(np.unique(bars[train]).size)


def purged_folds(bars, folds=OOF_FOLDS):
    """[(train mask, test mask)] over contiguous bar blocks, purged by EMBARGO_DAYS on both sides."""

    present = np.unique(bars)
    if present.size < folds * (VALID_DAYS // 2):
        raise RuntimeError(f"only {present.size} distinct bars for {folds} purged folds")
    blocks = np.array_split(present, folds)
    out = []
    for block in blocks:
        low, high = int(block[0]), int(block[-1])
        test = (bars >= low) & (bars <= high)
        train = (bars < low - EMBARGO_DAYS) | (bars > high + EMBARGO_DAYS)
        if train.sum() < MIN_SAMPLES or test.sum() < MIN_SAMPLES:
            raise RuntimeError("a purged fold is too small to train or to score")
        out.append((train, test))
    return out
