"""Daily body share, equal-cash, monthly.

m1 is the 20-session mean of abs(close - open) / (high - low). c_shuf permutes
those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
