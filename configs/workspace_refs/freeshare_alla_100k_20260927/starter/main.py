"""Free-float share of the float, equal-cash, monthly.

m1 is free_share / float_share. c_shuf permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
