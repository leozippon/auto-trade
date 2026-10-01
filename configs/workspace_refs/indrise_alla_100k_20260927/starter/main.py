"""Shenwan industry amount rise, equal-cash, monthly.

m1 is the 21-session rise in the industry index amount. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
