"""esop60: Employee shareholding plan DRAFT announcements; entry at T0+1, held HOLD_DAYS=60 trading days.

Skeleton: the canonical equal-cash book (lib/trade.py) with soft seats, driven
by lib/knobs.py and the event score in lib/score.py. No fit, no state.
"""

from sleeves.esop60 import trade


def generate_orders(context):
    return trade.run(context)
