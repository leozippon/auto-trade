"""Price-to-sales rank, equal-cash, monthly.

m1 is minus ps_ttm. c_shuf permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
