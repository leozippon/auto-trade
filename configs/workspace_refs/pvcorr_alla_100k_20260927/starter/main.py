"""Return-amount correlation, equal-cash, monthly.

m1 is the 20-session correlation of pct_chg and amount. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
