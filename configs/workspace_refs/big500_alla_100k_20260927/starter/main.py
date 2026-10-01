"""Largest current CSI 500 members. No tenure term. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
