"""Entry: one book, six shapes, switched by an explicit constant.

`BOOK` is the shape. It defaults to `quality_overlay` and is never inferred
from the account or from whether a tracking mandate is in force -- those are
two independent run facts, and the strategy context carries neither.

    quality_overlay   CSI 300 benchmark weights with a quality tilt
    high52_overlay    CSI 300, 52-week near-high, SW L1-centred
    rmax_overlay      CSI 300, −max of the last 20 daily pct_chg
    pacc_index        CSI 300, 12 seats by index L1 weights, −percent accrual
    net_issuance      affordable whole-market pool, −252-day net issuance
    lottery_reverse   affordable whole-market pool, −within-L1 lottery rank

Nothing here is a thesis. The first day needs a real book; the session replaces
the score, the seats and even the shape after reading the arm's directive.

Knobs: trade.BOOK / SEATS / SEAT_CASH_MIN / KEEP_BAND / MAX_REPLACE,
composite.OVERLAY_K.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
