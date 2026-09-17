"""Entry module: the bucketed low-residual-volatility book and its three controls.

`CANDIDATE` selects which pre-registered leg this package is (see
`lib.face.CANDIDATES` for the score and `lib.trade.SHAPES` for the
construction):

    "s_lb"        -> ivol60 ranks inside industry x float-cap buckets  (main candidate)
    "c_uncapped"  -> the same score and the same pool, no buckets      (control, not nominable)
    "c_rand"      -> the same buckets, a fixed random score            (control, not nominable)
    "c_pool"      -> a score-free book spread over the eligible pool   (control, not nominable)

The four differ in nothing else: same pool, same quarterly review with a keep
band, same top-15 equal-cash book, same score-level neutralization. There is no
`fit`: every decision is four bounded reads of the as-of view, one 60-day
rolling regression and one least squares.
"""

from lib import trade

CANDIDATE = "s_lb"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
