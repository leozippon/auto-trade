"""Entry: one book, four shapes, switched by an explicit constant.

`BOOK` is the shape. It defaults to `gp_overlay` and is never inferred
from the account or from whether a tracking mandate is in force -- those are
two independent run facts, and the strategy context carries neither.

    gp_overlay     CSI 300 benchmark weights with a gross-profit tilt
    cma_overlay    CSI 300, −asset growth
    noa_index      CSI 300, 12 seats by index L1 weights, −NOA / lagged TA
    cashdiv_pool   affordable whole-market pool, trailing cash dividend / close

Nothing here is a thesis. The first day needs a real book; the session replaces
the score, the seats and even the shape after reading the arm's directive.

Knobs: trade.BOOK / SEATS / SEAT_CASH_MIN / KEEP_BAND / MAX_REPLACE,
composite.OVERLAY_K.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
