"""Entry module: the confirmed carrier, unchanged, on CSI 500 membership at CNY 1m.

Nothing about the model moves in this arm. The 158 Alpha operators
(lib/features.py), the LightGBM carrier and its single parameter point
(lib/primary.py), the benchmark-residual 10-day open-to-open rank label
(lib/labels.py), the quarterly continuation with its cold reset (lib/state.py)
and the embargoed split (lib/common.py) are the confirmed carrier exactly as the
previous round ran it. What moves is the POOL: `lib/index.py` reads the CSI 500
sections instead of the CSI 300 ones, and the host grades the book against the
CSI 500 because the arm is created with `benchmark_index = "000905.SH"`.

The one constant below selects which of the two registered legs this package is;
`families.md` is the authority:

    s1        the candidate: the carrier on CSI 500 membership, 50 equal-cash
              seats, monthly review
    c_cold    the same book with EVERY quarterly refit cold. It prices the
              continuation this round ships by default, which this repository
              has three times failed to confirm as an advantage and once
              measured as a 1.02 pp/yr disadvantage. Control, never nominated

There is no cross-index control and this pack does not pretend otherwise. An arm
is keyed to one `benchmark_index`, so a CSI 300 leg inside this batch would be
scored against the CSI 500's return and the CSI 500's panel membership -- the
mismatch every pack of the previous round bans by name. The CSI 300 reading this
arm is measured against is the previous round's host reading of this same
carrier (active IR 0.4642 and 0.18 on two arms), which is NOT same-batch and is
therefore evidence, not a control. `families.md` says how far that comparison may
be pushed, and `exploration-plan.md` round 0 registers the offline dispersion
diagnostic that is the only same-code cross-index comparison available at zero
replay cost.

Refitting is quarterly. The first fit of a replay is cold because the state
directory starts empty; every later one continues the booster it saved, and
every `state.COLD_EVERY`-th one resets cold.

Knobs, one per registered axis, each assigned exactly once:
index.INDEX_CODE, labels.HOLD, trade.SEATS / KEEP_BAND / MAX_SWAPS,
data.TRAIN_YEARS, primary.PARAMS, state.COLD_EVERY, common.VALID_DAYS.
"""

from lib import data, labels, primary, state, trade

CANDIDATE = "s1"
REFIT_PERIOD = "quarter"

REGISTERED = ("s1", "c_cold")


def fit(context):
    if CANDIDATE not in REGISTERED:
        raise RuntimeError(f"'{CANDIDATE}' is not a registered leg; families.md lists {REGISTERED}")
    cold = CANDIDATE == "c_cold" or state.is_cold(context)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    target, realized = labels.target(panel)
    samples = data.fit_samples(panel, target, realized)
    reading = primary.fit(context, samples, cold, labels.HOLD)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
