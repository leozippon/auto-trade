"""Free-share growth rank, equal-cash, monthly.

m1 is minus the 60-session growth in free_share. c_shuf permutes those values.
No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
