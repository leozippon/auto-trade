"""CSI 500 members larger than their industry. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
