"""Amount following the prior day's return, equal-cash, monthly.

m1 correlates amount with the previous session's return. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
