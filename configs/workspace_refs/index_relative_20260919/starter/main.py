"""Entry module: an equal-cash book built inside the CSI 300 and its two zero-skill controls.

`CANDIDATE` selects which pre-registered leg this package is; `lib.trade.LEGS`
says what each one holds and with how many seats:

    "s_idx"    the four-leg rank composite's top `trade.BASKET_N` constituents  (main candidate)
    "c_rand"   `trade.BASKET_N` constituents drawn by a permanent fixed-seed
               lottery -- the zero-skill baseline at the candidate's own size   (control)
    "c_index"  `trade.PROXY_N` constituents from the same draw -- the equal-weight
               index proxy at the largest size this account can carry           (control)

The three differ in nothing else: the same as-of constituent universe, the same
monthly review calendar, the same keep band, the same nearest-lot sizing, the
same 09:30-sell / 15:00-buy timing. Both controls must run in the same batch as
every candidate, and neither may be nominated.

`trade.BASKET_N` is the one constant this arm sets, inside the basket range its
directive gives; `trade.PROXY_N` is the largest seat count the account can
carry and belongs to the control alone. The composite score is a runnable
baseline, not this arm's thesis -- which families carry alpha among index
constituents is what the arm is there to find out. There is no `fit`.

Knobs: trade.BASKET_N / PROXY_N / KEEP_BAND / MAX_REPLACE, composite.LEGS.
"""

from lib import trade

CANDIDATE = "s_idx"


def generate_orders(context):
    return trade.run(context, CANDIDATE)
