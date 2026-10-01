"""Room under the up limit, equal-cash, monthly.

m1 is (up_limit - close) / close. c_shuf permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
