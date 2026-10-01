"""Stable CSI 1000 weights, equal-cash, monthly.

m1 is minus the standard deviation of month-end weights. c_shuf permutes
those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
