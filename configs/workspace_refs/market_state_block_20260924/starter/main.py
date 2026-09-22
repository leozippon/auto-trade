"""Entry: the confirmed Alpha158 + LightGBM carrier, with a market-state block bolted on.

`CANDIDATE` selects what this package is; `families.md` is the authority:

    "c_base"   the carrier alone -- 158 Alpha operators, the benchmark-residual
               10-day rank label, the constituent universe, quarterly refits
               with warm continuation. The required control, always the first
               leg of a batch, never nominated
    "m1"       the same code with twelve date-level market-state columns
               appended to the same booster's columns: benchmark returns,
               volatility, trend and turnover, constituent dispersion and
               breadth. Every name on a bar carries the same values
    "m_int"    the same code with twelve state x stock interactions instead:
               three stock factors ranked inside the bar, each times four of
               the states. The same question asked in a form a split can use
               without isolating whole dates

Both candidates change the RANKING only. Seats, equal cash, the monthly review
and full investment are the carrier's; no leg varies gross exposure or cash by
state, which is what separates this arm from the closed timing family.

All three share the panel (lib/data.py), the universe and the training dates
(lib/index.py), the label (lib/labels.py), the split (lib/common.py), the
booster (lib/primary.py), the warm chain (lib/state.py) and the 50-seat
monthly book (lib/trade.py). The block itself is lib/market_state.py.

Refitting is quarterly. The first fit of a replay is cold because the state
directory starts empty; every later one continues the booster it saved, and
every `knobs.COLD_EVERY`-th one resets cold so a chain cannot drift forever.

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import data, knobs, labels, primary, state, trade

CANDIDATE = "c_base"
REFIT_PERIOD = "quarter"


def _values(context, panel):
    """The block's raw matrix, or a zero-width one when the candidate does not read it."""

    if CANDIDATE == "c_base":
        return data.empty_block(panel)
    return data.build_block(panel, CANDIDATE)


def fit(context):
    cold = state.is_cold(context)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    values = _values(context, panel)
    target, realized = labels.rank_target(labels.residual(panel, knobs.HORIZON))
    samples = data.fit_samples(panel, values, target, realized, CANDIDATE)
    del panel, values
    reading, _ = primary.fit(context, samples, cold, CANDIDATE)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE, _values)
