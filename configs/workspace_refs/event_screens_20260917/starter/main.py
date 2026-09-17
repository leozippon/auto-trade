"""Entry module: the borrowed low-volatility book behind three exclusion screens.

`CANDIDATE` selects which pre-registered leg this package is (see
`lib.screens` for the screens, `lib.trade.SCREENS` for which of them each leg
runs and `lib.face.CANDIDATES` for the score):

    "s_scr"       -> audit, unlock and segment screens on         (main candidate)
    "c_noscreen"  -> the identical book with no screen at all     (control, not nominable)
    "c_placebo"   -> a permanent random exclusion of the same size (control, not nominable)
    "c_pool"      -> a score-free book spread over the pool        (control, not nominable)

The four differ in nothing else: same pool, same quarterly review with a keep
band, same top-15 equal-cash book inside industry and float-cap buckets, same
score-level neutralization. `c_noscreen` is the sibling pack
`lowvol_bucketed_20260917`'s main candidate byte for byte. There is no `fit`:
every decision is four bounded reads of the as-of view, one 60-day rolling
regression and one least squares, plus three more bounded reads on the four
review days of the year.
"""

from lib import trade

CANDIDATE = "s_scr"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
