"""Recent CSI 500 deletions, larger ones first. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
