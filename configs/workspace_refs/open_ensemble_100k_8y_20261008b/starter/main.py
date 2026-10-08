"""The combined book: the two fixed Paper sleeves on one account; no fit, no state.

As it stands this package is the control "merged without the new sleeve": the
ESOP draft book esop60 and the incentive-draft book b30, each on half of the
account at 16 seats, combined by lib/combine.py. The arm's own sleeve is added
as one more `Sleeve` with its own `run` and `holds` (lib/combine.py says what
each must do), placed before b30, and the shares are changed here; the seats
of all sleeves together stay at most 36. LEG picks what one copy of this
package runs (`lib.combine.leg`): the candidate is "merged"; the two required
controls are "without" and each given sleeve alone ("esop60", "b30"), copies
of the same package that differ in LEG only and are registered control: true
on the candidate's span. What may change and what may not is in
refs/README.md:

    sleeves/b30/, sleeves/esop60/   the frozen strategies, byte for byte but
                                    for their import lines; their logic is not
                                    an axis of this arm
    lib/combine.py                  sub-accounts, ownership and cash
    lib/controls.py                 the standard controls
    lib/data.py, lib/index.py, lib/fund.py, lib/titles.py, lib/trade.py,
    lib/knobs.py, lib/score.py      the common starter for an own sleeve
                                    (lib/score.py is a placeholder, never a
                                    sleeve)

The LAST sleeve takes every holding no earlier sleeve claims and must sell
what it holds no reason for; keep b30 last.
"""

from lib.combine import Sleeve, combine, holds_b30, holds_esop60, leg
from sleeves.b30 import main as b30
from sleeves.esop60 import main as esop60

# "merged" (the candidate), "without", "esop60" or "b30" (the controls).
LEG = "merged"

SLEEVES = [
    Sleeve("esop60", esop60.generate_orders, holds_esop60, 0.5),
    Sleeve("b30", b30.generate_orders, holds_b30, 0.5),
]


def generate_orders(context):
    return combine(context, leg(SLEEVES, LEG))
