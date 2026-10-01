"""Listing-age rank, equal-cash, monthly.

m1 is days since list_date. c_shuf permutes those values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
