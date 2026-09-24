"""Entry: range20, a rule score with no learner, on the benchmark's own constituents.

The score is minus the 20-bar mean of each name's daily range,
(high - low) / pre_close, over its own last `knobs.RANGE_DAYS` visible bars:
the calmest constituents rank first. `lib/score.py` defines it, `lib/data.py`
builds the pool it is ranked in, `lib/trade.py` turns the ranking into an
equal-cash book reviewed on the first decision of each calendar month.

What this package is follows from `lib/knobs.py` alone, one line per knob:

    INDEX       the benchmark the book is built in (aligned to the run fact)
    CAPITAL     the account the shape was registered for (aligned to the run fact)
    SEATS       equal-cash seats
    KEEP_BAND   a holding is kept while ranked inside SEATS x KEEP_BAND
    SHUFFLE     False: the candidate r1. True: the control c_shuf, the same
                scores permuted among the pool within each decision day
    RANGE_DAYS  20 for r1 and c_shuf; 60 for the registered neighbour r2

Nothing is fitted, so there is no `fit` and no `REFIT_PERIOD`: every decision
reads its own window from the as-of view and nothing is cached across calls.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
