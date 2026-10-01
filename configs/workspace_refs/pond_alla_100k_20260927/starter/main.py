"""Thin CSI 500 industries, larger names first. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
