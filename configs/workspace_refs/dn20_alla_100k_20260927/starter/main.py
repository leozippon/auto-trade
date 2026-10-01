"""Down-day amount share over twenty sessions. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
