"""Entry: an abnormal-turnover score on the ChiNext book, with its control.

`CANDIDATE` selects what this package is; `refs/families.md` is the authority and
`lib/candidates.py` the one place each name is defined:

    "t1"      the candidate: minus the ratio of a name's 20-session mean
              turnover rate to its 250-session mean (lib/score.py). A rule,
              not learned
    "c_shuf"  control: t1's values shuffled within the decision day's pool
    "t1i"     the registered variant: t1 ranked within each SW L1 industry

All three trade the same 20-seat monthly equal-cash book (lib/trade.py) on the
same pool (lib/data.py, lib/index.py). Nothing is fitted, so there is no `fit`
(and no refit period); every review or refill recomputes the score from the
as-of view.

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import trade

CANDIDATE = "t1"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
