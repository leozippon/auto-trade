"""Newer and larger CSI 500 members, equal-cash, monthly.

m1 ranks tenure first and size only inside one tenure. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
