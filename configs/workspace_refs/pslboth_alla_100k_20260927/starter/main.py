"""Lagged cheap ps_ttm on names in CSI 1000 both then and now. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
