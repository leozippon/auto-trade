"""Recent CSI 500 deletions, graded against CSI 500. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
