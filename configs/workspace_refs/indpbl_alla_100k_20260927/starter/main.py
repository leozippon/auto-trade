"""Cheap Shenwan industries, equal-cash, monthly.

m1 is minus the industry pb. c_shuf permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
