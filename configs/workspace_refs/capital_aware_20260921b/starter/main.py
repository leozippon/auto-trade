"""Entry: one book, four shapes, switched by an explicit constant.

`BOOK` is the shape. It defaults to `index_overlay` and is never inferred from
the account or from whether a tracking mandate is in force -- those are two
independent run facts, and the strategy context carries neither.

    index_overlay     CSI 300 benchmark weights with an unbounded score overlay
    index_indneutral  industry weights matched to the index; value (EP/BP) inside
    pool_momentum     affordable whole-market pool, 60-120d residual momentum
    pool_eyield       affordable whole-market pool, earnings / cash-flow yield

Nothing here is a thesis. The first day needs a real book; the session replaces
the score, the seats and even the shape after reading the arm's directive.

Knobs: trade.BOOK / SEATS / SEAT_CASH_MIN / KEEP_BAND / MAX_REPLACE / MIN_ADV,
composite.MOM_WINDOW / OVERLAY_TILT.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
