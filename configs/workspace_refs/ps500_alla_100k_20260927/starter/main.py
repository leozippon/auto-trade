"""Cheaper ps_ttm inside CSI 500. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
