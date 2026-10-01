"""Entry: one untrained rule score on the benchmark's own constituents.

`knobs.SCORE` picks the score (`lib/score.py` defines all of them in one
place): ep (earnings yield, 1 / pe_ttm), dvy (trailing cash dividend yield),
abturn (minus one-month abnormal turnover) or ch3 (the equal-weight rank
average of ep and abturn); range20 is only a labelled diagnostic. `lib/data.py`
builds the pool the score is ranked in, `lib/trade.py` turns the ranking into
an equal-cash book reviewed on the first decision of each calendar month.

What this package is follows from `lib/knobs.py` alone, one line per knob:

    INDEX       the benchmark the book is built in (aligned to the run fact)
    CAPITAL     the account the shape was registered for (aligned to the run fact)
    SEATS       equal-cash seats
    KEEP_BAND   a holding is kept while ranked inside SEATS x KEEP_BAND
    SCORE       the arm's registered score
    EP_COLUMN, DV_COLUMNS, TURN_SHORT, TURN_LONG
                the score's inputs; the registered neighbour changes one of them
    SHUFFLE     False: the candidate. True: the control c_shuf, the same
                scores permuted among the pool within each decision day

Nothing is fitted, so there is no `fit` and no `REFIT_PERIOD`: every decision
reads its own window from the as-of view and nothing is cached across calls.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
