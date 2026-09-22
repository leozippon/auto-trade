"""Entry: the confirmed Alpha158 + LightGBM carrier, with a signed order-flow block bolted on.

`CANDIDATE` selects what this package is; `families.md` is the authority:

    "c_base"   the carrier alone -- 158 Alpha operators, the benchmark-residual
               10-day rank label, the constituent universe, quarterly refits
               with warm continuation. The required control, always the first
               leg of a batch, never nominated
    "f1"       the same code with the 14-column flow block appended to the same
               booster's columns. The candidate. Everything else -- split,
               rounds, seed, refit cadence, book -- is identical, which is what
               makes the difference an ablation of the block
    "f_only"   the block alone, no Alpha158 columns. A diagnostic: it answers
               "does this block carry anything at all", which is a different
               question from "does it add anything", and `families.md` forbids
               nominating it

All three share the panel (lib/data.py), the universe and the training dates
(lib/index.py), the label (lib/labels.py), the split (lib/common.py), the
booster (lib/primary.py), the warm chain (lib/state.py) and the 50-seat
monthly book (lib/trade.py). The block itself is lib/flow.py.

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
    return data.build_block(context, panel)


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
