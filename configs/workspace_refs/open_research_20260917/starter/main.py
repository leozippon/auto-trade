"""Entry module: the v1 working baseline -- a size-neutral rank composite held as a monthly top-20 book.

Four legs from four information families, each a percentile rank (lib/composite.py):
cash-flow quality of the latest first-version statement, low 60-day residual
volatility on the CSI300, 20-day reversal, and value (earnings and book yield). Their
mean is residualised on size, the top 20 are held with a keep band and reviewed on
the first decision of each calendar month (lib/trade.py). The cross-section also
carries the decision day's SW L1 industry, and `trade.INDUSTRY_CAP` -- off by default
-- fills the book under a per-industry counter seeded with the retained holdings.
There is no `fit`.

It is a baseline to beat, not a recommended mechanism: replace any part or all of it.
Knobs: composite.LEGS, trade.TOP_N / KEEP_BAND / MAX_REPLACE / INDUSTRY_CAP.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
