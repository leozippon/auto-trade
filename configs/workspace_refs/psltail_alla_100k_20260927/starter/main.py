"""Lagged cheap ps_ttm without the cheapest tenth. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
