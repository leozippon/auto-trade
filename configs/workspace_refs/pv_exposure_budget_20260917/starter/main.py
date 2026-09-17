"""Entry module: the exposure-budgeted price-volume book and its three controls.

`CANDIDATE` selects which pre-registered leg this package is; `lib.common.LEGS` says
what each one is (which score it ranks on, whether the fill is bucketed):

    "s_pvb"       -> the borrowed LightGBM score, filled inside industry x cap buckets  (main candidate)
    "c_uncapped"  -> the same trained score, same pool and cadence, no buckets  (control, not nominable)
    "c_rand"      -> the same buckets, a fixed random score                     (control, not nominable)
    "c_pool"      -> a score-free book spread over the eligible pool            (control, not nominable)

The four differ in nothing else: same pool, same weekly review with a keep band, same
top-15 equal-cash book, same 09:30-sell / 15:00-buy timing. The score is a borrowed
fixed component -- `fit` builds the trailing-window panel (lib/data.py) and trains the
single-point grid into context.state_dir (lib/score_lgbm.py); this arm registers the
fill in lib/trade.py, not the score.

`c_rand` and `c_pool` rank on no learned score, so they are the two legs whose main.py
differs in more than `CANDIDATE`: delete `fit` and the `REFIT_PERIOD` line (the loader
refuses a package that declares `REFIT_PERIOD` without a `fit` entrypoint, `None`
included), and import `trade` alone. No fit container is then started and no booster
exists to be read back by mistake; `fit` raises if that edit was forgotten.

Variant knobs, one per axis: trade.INDUSTRY_CAP, trade.CAP_TERCILES, trade.MAX_SWAPS,
data.TRAIN_YEARS, score_lgbm.LEAVES / LRS.
"""

from lib import score_lgbm, trade

CANDIDATE = "s_pvb"

REFIT_PERIOD = "quarter"


def fit(context):
    score_lgbm.fit(context, CANDIDATE)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
