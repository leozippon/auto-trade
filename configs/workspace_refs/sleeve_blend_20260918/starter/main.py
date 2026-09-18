"""Entry module: the two-sleeve blend and its three matched controls.

`CANDIDATE` selects which pre-registered leg this package is; `lib.common.LEGS` says
which sleeves each one runs, at what share of capital and with how many seats:

    "s_blend"   composite 0.30 x 15 seats + pv 0.70 x 15 seats   (main candidate)
    "c_pv"      pv alone, 1.00 x 15 seats                        (control, not nominable)
    "c_comp"    composite alone, 1.00 x 15 seats                 (control, not nominable)
    "c_wide30"  pv alone, 1.00 x 30 seats                        (control, not nominable)

The four differ in nothing else: the same two sleeve constructions, the same review
calendars (monthly for `composite`, the first decision of each ISO week for `pv`), the
same keep band, the same 09:30-sell / 15:00-buy timing. `c_pv` is the attribution
control -- the blend must beat the better leg on information ratio, or the blend is a
hedge and not an edge. `c_wide30` is the shape the ban table closed and is expected to
lose: it separates "two orthogonal top-15s" from "one score's top-30".

`c_comp` runs no learned sleeve, so it is the one leg whose main.py differs in more
than `CANDIDATE`: delete `fit` and the `REFIT_PERIOD` line (the loader refuses a
package that declares `REFIT_PERIOD` without a `fit` entrypoint, `None` included), and
import `trade` alone. No fit container is then started and no booster exists to be read
back by mistake; `fit` raises if that edit was forgotten.

Variant knobs, one per axis: common.W_A, common.SEATS, composite.CAP_FLOOR,
composite.LEGS, data.FEATURE_GROUPS / TRAIN_YEARS, trade.KEEP_BAND / MAX_SWAPS.
"""

from lib import score_lgbm, trade

CANDIDATE = "s_blend"

REFIT_PERIOD = "quarter"


def fit(context):
    score_lgbm.fit(context, CANDIDATE)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
