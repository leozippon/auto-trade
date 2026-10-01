"""CSI 1000 industry weight share. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
