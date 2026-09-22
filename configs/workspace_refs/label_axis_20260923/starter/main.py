"""Entry module: the confirmed carrier, unchanged, with the label on the variant axis.

The one constant below selects which registered label the carrier is fitted on;
`families.md` is the authority for all seven and `lib/labels.py` holds the
family itself:

    c_base   the control and the first leg of every batch: the round's default
             label, the 20260922 one unchanged -- a benchmark-residual 10-day
             open-to-open return, ranked across that bar's constituents
    l_h20    the same label over 20 trading days
    l_h40    the same label over 40 trading days
    l_exec   10 days measured between the two prices this book actually trades
             at: the close it buys on and the open it sells on
    l_cc     10 days measured close to close
    l_ind    10 days, residualised against the benchmark AND demeaned inside the
             name's own SW level-1 industry
    l_vol    10 days, the residual divided by the name's own trailing volatility
             before it is ranked

Nothing else differs between two legs. The panel (lib/data.py), the 158 Alpha
operators (lib/features.py), the constituent sections (lib/index.py), the
LightGBM carrier and its single parameter point (lib/primary.py), the embargoed
split (lib/common.py), the quarterly continuation and its cold reset
(lib/state.py) and the 12-seat monthly book (lib/trade.py) are shared by all
seven, which is what makes `c_base` a control rather than a seventh candidate.

The embargo between the training and validation segments is the label's own
horizon, so a 40-day label is split 40 bars apart and a 10-day label 10. That is
the one thing in the fit path that MUST move with the label: leaving it at 10
would let the longer horizons read as better models than they are, and it would
do so only for them -- biasing the exact comparison this arm exists to make.

Refitting is quarterly. The first fit of a replay is cold because the state
directory starts empty; every later one continues the booster it saved, and
every `state.COLD_EVERY`-th one resets cold so a chain cannot drift forever.
`state.COLD_EVERY = 1` is the registered persistence ablation and is a control.

Knobs, one per registered axis, each assigned exactly once:
labels.FAMILY, trade.SEATS / INDUSTRY_CAP / KEEP_BAND / MAX_SWAPS,
data.TRAIN_YEARS, primary.PARAMS, state.COLD_EVERY, common.VALID_DAYS.
"""

from lib import data, labels, primary, state, trade

CANDIDATE = "c_base"
REFIT_PERIOD = "quarter"


def fit(context):
    label = labels.spec(CANDIDATE)
    cold = state.is_cold(context)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    target, realized = labels.target(panel, label)
    samples = data.fit_samples(panel, target, realized)
    reading = primary.fit(context, samples, cold, label.horizon)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
