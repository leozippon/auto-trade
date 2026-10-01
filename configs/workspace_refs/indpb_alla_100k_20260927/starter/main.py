"""Shenwan industry pb change, equal-cash, monthly.

m1 is minus the 21-session change in the industry index pb. c_shuf permutes
those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
