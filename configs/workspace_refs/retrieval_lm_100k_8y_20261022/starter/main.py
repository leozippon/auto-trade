"""The baseline book: retrieve each name's nearest past cases by price-volume pattern and titles, and buy the best predicted.

`fit` (lib/learn.py) runs before every decision (REFIT_PERIOD "day") and does
the day's work where the state directory is writable: the first call of a
replay fills the case memory from the PIT history, every call files the cases
whose outcomes have just matured, keys today's names and scores them --
by their nearest cases (lib/memory.py) or, for the carrier C1, by a network
refitted on the same cases each quarter (lib/model.py). `generate_orders`
reads the day's scores (lib/score.py) and runs the canonical book
(lib/trade.py) with `lib/knobs.py`.

The other modules are the read helpers and the keys' parts:

    lib/data.py      the decision day's pool and windowed daily panels
    lib/features.py  the price-volume patterns and the residual forward label
    lib/text.py      each name's visible titles, deduplicated and anonymised
    lib/encoder.py   the frozen title encoder and its in-process cache
    lib/index.py     the newest visible index section
    lib/controls.py  the standard controls a nomination reports
    lib/device.py    the torch device, keeping the CPU path the contract requires
"""

from lib import learn, trade

REFIT_PERIOD = "day"


def fit(context):
    learn.fit(context)


def generate_orders(context):
    return trade.run(context)
