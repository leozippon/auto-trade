"""The beta lane: which names the book may buy, by their ex-ante beta to the index.

The bag ranks calm names first, so the book's beta to CSI 1000 sits well below
one. This lane asks whether the bag's selection survives when the book may
buy only names whose ex-ante beta is at least `knobs.BETA_FLOOR`:

    ex-ante beta  the label's own estimator (`label.trailing_beta`): OLS of
                  the name's daily adjusted close returns on the index's over
                  the trailing `knobs.BETA_DAYS` trading days up to T-1,
                  shrunk 0.7 toward 1 and clipped to [0.3, 2.0]; NaN, and so
                  not buyable, without that many observations (at least 60)
    the floor     gates buys only, like one-lot affordability: a holding whose
                  beta falls under the floor is kept and leaves by rank

The placebo `c_betashuf` (`knobs.BETA_PLACEBO`) keeps the floor and the book
and replaces the bag's score with a uniform random score drawn from a
generator seeded by the decision date, so the same date always draws the same
ranks. It needs no model: `main.fit` does nothing for it. It measures what the
floor alone earns on the host's graded series, without any selection.
"""

import numpy as np

from lib import knobs, label, panel as P


def exante(context, data, calendar_days):
    """(names,) ex-ante beta of every name of the panel on its newest row; NaN without enough history."""

    benchmark = P.read_benchmark(context, calendar_days)
    return label.trailing_beta(data, benchmark, knobs.BETA_DAYS).to_numpy()[-1]


def buyable(context, data, calendar_days):
    """(names,) bool: may this name be bought today. All True without a floor; beta per name beside it."""

    names = len(data["codes"])
    if not knobs.BETA_FLOOR:
        return np.ones(names, dtype=bool), np.full(names, np.nan)
    beta = exante(context, data, calendar_days)
    with np.errstate(invalid="ignore"):
        return np.isfinite(beta) & (beta >= knobs.BETA_FLOOR), beta


def placebo_score(context, data, t):
    """(names,) uniform random scores on date index t's tradable names, seeded by the decision date."""

    rng = np.random.default_rng(int(context.inference_at.strftime("%Y%m%d")))
    draw = rng.random(len(data["codes"]))
    return np.where(P.tradable(data, t), draw, np.nan)
