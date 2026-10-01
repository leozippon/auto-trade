"""Lagged cheap ps_ttm outside CSI 1000, main board only. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
