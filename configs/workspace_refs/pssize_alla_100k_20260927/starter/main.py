"""Size-residual price-to-sales, equal-cash, monthly.

m1 is minus the residual of ps_ttm on log circulating market value. c_shuf
permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
