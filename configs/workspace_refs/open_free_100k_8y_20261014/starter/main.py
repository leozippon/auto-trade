"""Skeleton book with the placeholder score p0; no fit, no state.

The package runs as it stands, but p0 (lib/score.py) orders names by a hash of
their code and carries no information by construction: it is a scaffold, not a
candidate. `lib/trade.py` is the canonical equal-cash book every pack's starter
shares, driven by `lib/knobs.py` and the score. The other modules are
point-in-time read helpers the baseline does not call:

    lib/data.py    the decision day's pool and windowed daily panels
    lib/index.py   the newest visible index section
    lib/fund.py    the fundamentals domain (statements, forecasts, express
                   reports, dividends, audit opinions, segments, calendars)
    lib/titles.py  the announcement titles of the text domain
    lib/device.py  the torch device, keeping the CPU path the contract requires

Keep what the mechanism needs and delete the rest before the first validation.
A fitted mechanism adds `fit(context)` and `REFIT_PERIOD` here, as
output/README.md describes.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
