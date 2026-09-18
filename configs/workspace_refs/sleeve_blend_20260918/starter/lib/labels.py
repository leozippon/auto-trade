"""Label construction: y(bar s) = open_qfq[D+10] / open_qfq[D] - 1 with D = s+1,
i.e. O[s+11] / O[s+1] - 1. Only bars whose label is fully realized (index s+11 exists)
carry a value; the cross-sectional rank target is computed per bar over realized rows.
"""

import numpy as np


def build_targets(W, hold=10):
    """(S, T) forward-open return per bar; NaN where not fully realized."""
    O = W["O"]
    S, T = O.shape
    y = np.full((S, T), np.nan)
    if T > hold + 1:
        # y[:, s] = O[:, s+1+hold] / O[:, s+1] - 1, valid for s = 0..T-2-hold
        with np.errstate(divide="ignore", invalid="ignore"):
            y[:, : T - (hold + 1)] = O[:, hold + 1:] / O[:, 1 : T - hold] - 1.0
    return y


def target_matrix(y):
    """Per-bar cross-sectional target = pct_rank(y) - 0.5 over realized rows only.

    Returns (tgt (S, T), realized (S, T) bool). pct_rank uses average ranks / n_realized
    (ties averaged), so target lies in [-0.5, 0.5].
    """
    S, T = y.shape
    tgt = np.full((S, T), np.nan)
    for t in range(T):
        col = y[:, t]
        m = np.isfinite(col)
        n = int(m.sum())
        if n < 2:
            continue
        v = col[m]
        _uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        starts = cum - counts
        avg_rank = starts[inv] + (counts[inv] - 1) / 2.0
        tgt[m, t] = avg_rank / n - 0.5
    return tgt, np.isfinite(y)
