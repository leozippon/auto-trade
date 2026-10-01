"""CSI 300 tenure score graded against CSI 300. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
