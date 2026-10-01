"""Newer CSI 500 members, cheaper sales yield inside one tenure.

m1 ranks tenure first and ps_ttm only inside one tenure. c_shuf permutes those
values. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
