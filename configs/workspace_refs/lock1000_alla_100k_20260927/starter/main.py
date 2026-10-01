"""Lower circulating share of total value inside CSI 1000. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
