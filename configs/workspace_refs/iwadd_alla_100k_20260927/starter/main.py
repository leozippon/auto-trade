"""Industry weight-share change. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
