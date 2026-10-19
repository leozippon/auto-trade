"""The baseline book: a frozen language model reads each name's announcement titles, a small head scores it.

`fit` (lib/learn.py) runs before the first decision and then every trading
day: the first call pretrains the head on the trailing window and saves it
under `context.state_dir`; under `knobs.UPDATE = "cold_quarter"` the first
call of every quarter refits it from scratch, under `"frozen"` it never
changes; every other call goes to the hook `lib/update.daily`, which is empty
here and is where an online update rule belongs. `generate_orders` scores
from that state (lib/score.py) and runs the canonical book (lib/trade.py)
with `lib/knobs.py`.

The other modules are point-in-time read helpers and the model's parts:

    lib/data.py      the decision day's pool and windowed daily panels
    lib/features.py  the ten numeric features and the residual forward label
    lib/text.py      each name's visible titles, deduplicated and anonymised
    lib/encoder.py   the frozen title encoder and its in-process cache
    lib/model.py     the head, and the pooling of titles into a text input
    lib/index.py     the newest visible index section
    lib/fund.py      the fundamentals domain, which the baseline does not read
    lib/controls.py  the standard controls a nomination reports
    lib/device.py    the torch device, keeping the CPU path the contract requires

REFIT_PERIOD stays "day" for every setting: a daily call that has nothing to
do returns at once.
"""

from lib import learn, trade

REFIT_PERIOD = "day"


def fit(context):
    learn.fit(context)


def generate_orders(context):
    return trade.run(context)
