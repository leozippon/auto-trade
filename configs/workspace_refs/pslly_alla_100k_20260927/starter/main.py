"""Lagged cheap static ps on CSI 1000 main-board names. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
