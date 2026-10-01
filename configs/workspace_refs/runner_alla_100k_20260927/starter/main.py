"""CSI 500 industry weight share, excluding the leader. No fit.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
