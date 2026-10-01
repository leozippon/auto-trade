"""Amount co-movement with the industry, equal-cash, monthly.

m1 correlates the name's amount with its Shenwan level-1 amount. c_shuf
permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
