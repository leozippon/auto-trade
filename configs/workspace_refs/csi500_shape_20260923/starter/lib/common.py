"""The embargoed train/validation split, with the embargo tied to the label's horizon.

A label built from an N-trading-day forward window overlaps its neighbours: two
rows fewer than N bars apart share part of their outcome. A split that ignores
that leaks the answer across the boundary, and the leak looks exactly like
skill.

The embargo is passed in rather than fixed here, and `main.fit` passes
`labels.HOLD`, so the two can never drift apart. The label's horizon is not a
knob in this arm (`lib/labels.py`), but the coupling is written this way anyway:
a constant that silently stops matching its label is the cheapest way to turn a
correct package into a leaking one during a later edit.

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
