"""Skeleton book with the placeholder score p0_placeholder; no fit, no state.

The package runs as it stands, but p0_placeholder (lib/score.py) orders names
by a hash of their code: a fixed basket that carries no information by
construction, a scaffold and never a candidate or a core. `lib/trade.py` is
the canonical equal-cash book every pack's starter shares, driven by
`lib/knobs.py` and the score. The other modules are helpers the baseline does
not call:

    lib/controls.py  the standard controls a nomination reports: late entry,
                     shuffled assignment, held shuffle, random skip, static
                     version, and the turnover the controls are checked on
    lib/data.py      the decision day's pool and windowed daily panels
    lib/index.py     the newest visible index section
    lib/fund.py      the fundamentals domain (statements, forecasts, express
                     reports, dividends, audit opinions, segments, calendars)
    lib/titles.py    the announcement titles of the text domain
    lib/events.py    the events domain, present only in packs whose arm
                     mounts it (refs/README.md says whether)

Keep what the mechanism needs and delete the rest before the first validation.
A fitted mechanism adds `fit(context)` and `REFIT_PERIOD` here, as
output/README.md describes; this arm has no GPU, so a model trains on the CPU.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
