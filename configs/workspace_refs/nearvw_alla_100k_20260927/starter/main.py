"""Close near the day's average trade. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
