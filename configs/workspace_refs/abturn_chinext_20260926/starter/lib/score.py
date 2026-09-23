"""Abnormal turnover (Liu, Stambaugh and Yuan 2019, "one-month abnormal turnover").

The score is minus the ratio of a name's mean daily turnover rate over the last
`knobs.SHORT` market sessions to its mean over the last `knobs.LONG`: a name
trading unusually little relative to its own past year ranks high (the paper's
long leg is the low-abnormal-turnover decile). A session without a bar (a
full-day suspension) is skipped, never counted as zero turnover; a name needs
`knobs.MIN_SHORT` and `knobs.MIN_LONG` valid sessions, otherwise NaN, which
also keeps names listed for less than about a year out of the pool.
"""

import numpy as np

from lib import knobs


def abnormal_turnover(panel):
    """(names,) -(mean turnover over SHORT sessions / mean over LONG sessions)."""

    rate = np.where(panel["TR"] > 0, panel["TR"], np.nan)
    short, long = rate[:, -knobs.SHORT:], rate[:, -knobs.LONG:]
    n_short, n_long = np.isfinite(short).sum(axis=1), np.isfinite(long).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nansum(short, axis=1) / n_short / (np.nansum(long, axis=1) / n_long)
    return np.where((n_short >= knobs.MIN_SHORT) & (n_long >= knobs.MIN_LONG), -ratio, np.nan)
