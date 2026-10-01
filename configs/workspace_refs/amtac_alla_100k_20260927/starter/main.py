"""Amount autocorrelation, equal-cash, monthly.

m1 is the lag-1 correlation of amount over 20 sessions. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
