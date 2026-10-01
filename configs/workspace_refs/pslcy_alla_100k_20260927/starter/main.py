"""Lagged cheap ps_ttm inside the ChiNext index. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
