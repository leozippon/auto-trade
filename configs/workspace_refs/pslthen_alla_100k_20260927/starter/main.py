"""Lagged cheap ps_ttm on the CSI 1000 section from that same time. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
