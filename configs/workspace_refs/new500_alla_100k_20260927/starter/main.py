"""Newer CSI 500 members, equal-cash, monthly.

m1 is minus the count of month-end sections. c_shuf permutes those values.
No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
