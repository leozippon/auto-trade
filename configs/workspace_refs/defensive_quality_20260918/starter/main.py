"""Entry module: defensive-quality book with the ablation and control switches.

`CANDIDATE` selects which pre-registered leg this package is (see
`lib.face.CANDIDATES`):

    "s1"        -> rank(cash-flow quality) + rank(low 60-day residual volatility)  (main candidate)
    "s_cfq"     -> cash-flow quality alone                                          (ablation, nominable)
    "s_lowvol"  -> low 60-day residual volatility alone                             (ablation, nominable)
    "c_vol20"   -> the closed 20-day low-vol face                                   (gate-1 control)
    "c_growth"  -> the same report's yoy net-profit growth                          (gate-2 control)
    "c_es"      -> the running earnings-surprise arm's composite                    (gate-2 control)

The six differ in nothing else: same pool, same monthly review with a keep
band, same top-15 equal-cash book, same score-level neutralization (a control
is not neutralized on its own face). There is no `fit`: every decision is a
few bounded reads of the as-of view plus one least squares.
"""

from lib import trade

CANDIDATE = "s1"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
