"""Older listings outside the major indexes. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
