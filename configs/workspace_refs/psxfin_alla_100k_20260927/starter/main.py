"""Cheap ps_ttm inside CSI 1000, excluding finance. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
