"""Correlation with market turnover changes. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
