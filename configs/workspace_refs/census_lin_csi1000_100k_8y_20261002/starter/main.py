"""Entry: the sign-fitted linear composite of the 44 census features, and its two control legs.

The CANDIDATE constant below selects the leg; the three differ in that one
line only (`refs/README.md` is the authority):

    l1      equal-weight composite of the per-date ranks, each sign refitted on
            the trailing window at every quarterly fit           main candidate
    c_shuf  l1's scores permuted among the pool on each decision day
                                                          control, never nominated
    c_fix   the composite with the signs of the replay's first fit kept for
            the whole replay                              control, never nominated

`lib/census_features.py` builds the panel and the features, `lib/learn.py` the
pool, label, sign fit and scores, `lib/book.py` the equal-cash book; every axis
value lives once in `lib/knobs.py`. `fit` writes the signs under
`context.state_dir` at the first decision of a replay and of every quarter.
"""

from lib import book, learn

CANDIDATE = "l1"
REFIT_PERIOD = "quarter"


def fit(context):
    learn.fit(context, CANDIDATE)


def generate_orders(context):
    return book.run(context, CANDIDATE)
