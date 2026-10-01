"""Steadiness of traded amount, equal-cash, monthly.

m1 is minus the 20-session coefficient of variation of amount. c_shuf permutes
those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
