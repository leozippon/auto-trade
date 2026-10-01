"""Entry: a LightGBM ranker over the census feature library, and its two control legs.

The CANDIDATE constant below selects the leg; everything else is shared and
the three differ in that one line only (`refs/README.md` is the authority):

    g1      LightGBM on the 44 census features, trained in `fit` on the trailing
            window against the benchmark-residual forward rank   main candidate
    c_shuf  g1's scores permuted among the pool on each decision day
                                                          control, never nominated
    c_lin   no model: the equal-weight composite of the ranked features with
            per-feature signs fitted in `fit`             control, never nominated

`lib/census_features.py` builds the panel and the features, `lib/learn.py` the
pool, label, fit and scores, `lib/book.py` the equal-cash book; every axis
value lives once in `lib/knobs.py`. `fit` rebuilds its state from scratch at
the first decision of a replay and of every quarter, under `context.state_dir`.
"""

from lib import book, learn

CANDIDATE = "g1"
REFIT_PERIOD = "quarter"


def fit(context):
    learn.fit(context, CANDIDATE)


def generate_orders(context):
    return book.run(context, CANDIDATE)
