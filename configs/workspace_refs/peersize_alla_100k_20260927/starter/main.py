"""Size versus the industry median, equal-cash, monthly.

m1 is minus circ_mv over the Shenwan level-1 median. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
