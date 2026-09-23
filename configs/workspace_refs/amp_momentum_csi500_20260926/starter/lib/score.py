"""The amplitude-split momentum score (开源证券 2020, "A factor").

Over the last `knobs.WINDOW` sessions, a name's valid sessions are ordered by
amplitude, (high - low) / pre_close; the score is the mean daily return over
the round(share * n) lowest-amplitude ones, n being its valid sessions there
(n >= `knobs.MIN_SESSIONS`, else NaN). For a name with every session present
this ranks exactly like the report's sum. Ties at the cutoff go to the earlier
session. With share = 1 it is the plain mean return of the window: the
`c_mom` control. A limit-locked day (high == low) has amplitude zero and is a
low-amplitude session like any other, as in the report.

The report orders sessions by high / low - 1; the two orders agree almost
everywhere (`refs/sources.md`), and the pack registers the pre_close form.
"""

import numpy as np

from lib import knobs


def amplitude_split(panel, share):
    """(names,) mean return over the lowest-amplitude `share` of each name's valid sessions."""

    window = knobs.WINDOW
    high, low, pre, ret = (panel[key][:, -window:] for key in ("H", "L", "PC", "R"))
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = (high - low) / pre
    valid = np.isfinite(amp) & np.isfinite(ret) & (pre > 0)
    n = valid.sum(axis=1)
    k = np.floor(share * n + 0.5)
    order = np.argsort(np.where(valid, amp, np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(order.shape[0])[:, None], order] = np.arange(window)[None, :]
    chosen = rank < k[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(valid, ret, 0.0) * chosen).sum(axis=1) / k
    return np.where(n >= knobs.MIN_SESSIONS, mean, np.nan)
