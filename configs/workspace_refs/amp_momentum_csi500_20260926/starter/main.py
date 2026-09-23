"""Entry: an amplitude-split momentum score on the CSI 500 book, with its controls.

`CANDIDATE` selects what this package is; `refs/families.md` is the authority and
`lib/candidates.py` the one place each name is defined:

    "a1"      the candidate: the mean daily return over the 70 % lowest-amplitude
              sessions of the last 160 (lib/score.py). A rule, not learned
    "c_shuf"  control: a1's values shuffled within the decision day's pool
    "c_mom"   control: the same mean over all 160 sessions (plain momentum)
    "a1i"     the registered variant: a1 ranked within each SW L1 industry

All four trade the same 50-seat monthly equal-cash book (lib/trade.py) on the
same pool (lib/data.py, lib/index.py). Nothing is fitted, so there is no `fit`
(and no refit period); every review or refill recomputes the score from the
as-of view.

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import trade

CANDIDATE = "a1"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
