"""Entry module: post-announcement earnings-surprise book with the two control switches.

`CANDIDATE` selects which pre-registered leg this package is (see
`lib.surprise.CANDIDATES`):

    "s1"        -> consensus surprise + quarterly SUE, freshness-weighted   (main candidate)
    "c_growth"  -> the same report's yoy net-profit growth only              (gate-1 control)
    "c_timing"  -> announcement freshness only                               (gate-2 control)

The three differ in nothing else: same pool, same half-month review, same
top-15 equal-cash book, same score-level neutralization. There is no `fit`:
every decision is a bounded read of the as-of view plus one least-squares.
"""

from lib import trade

CANDIDATE = "s1"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
