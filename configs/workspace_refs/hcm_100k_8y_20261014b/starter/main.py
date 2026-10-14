"""The HCM book: a never-reinitialised scorer, updated daily, over the shared equal-cash book.

`fit` (lib/hcm.py) is called before the first decision and then every trading
day: the first call pretrains on the lookback window, every later one rolls
the recurrent state a day forward, takes one small update on the day whose
label just matured, feeds the episode store, and writes everything under
`context.state_dir`. `generate_orders` scores from that state (lib/score.py)
and runs the canonical book (lib/trade.py) with `lib/knobs.py`. The controls
and ablations are knob settings, not other packages:

    UPDATE   graded (HCM) | uniform (A3) | reversed (A4) | frozen (C2, A1) | cold_quarter (C1)
    STATE    carried (HCM, A1..A4) | window (C1, C2)
    MEMORY   True (HCM, A3, A4) | False (C1, C2, A1, A2)

The other modules are point-in-time read helpers:

    lib/data.py      the decision day's pool and windowed daily panels
    lib/features.py  the feature panel and the residual forward label
    lib/index.py     the newest visible index section
    lib/fund.py      the fundamentals domain, which the skeleton does not read
    lib/device.py    the torch device, keeping the CPU path the contract requires

REFIT_PERIOD stays "day" for every setting: a daily call that has nothing to
do returns at once.
"""

from lib import hcm, trade

REFIT_PERIOD = "day"


def fit(context):
    hcm.fit(context)


def generate_orders(context):
    return trade.run(context)
