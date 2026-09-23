"""Entry: the confirmed Alpha158 + LightGBM carrier, scored by one seed or a bag of seeds.

`CANDIDATE` selects what this package is; `refs/families.md` is the authority and
`knobs.SEEDS` the one place each name is defined:

    "c_s7"   the carrier's monthly book exactly as the head packs trade it,
             one LightGBM booster with training seed 7. A required control,
             always the first leg of a batch, never nominated
    "c_s8"   the same book with training seed 8, nothing else changed. The
             second required control: the pair measures the seed spread in
             this arm's own batch
    "b5"     the same book scored by the rank average of five boosters that
             differ only in their training seed (7..11). The arm's candidate
    "b10"    the same with ten seeds (7..16); a second candidate only when the
             fit budget carries it
    "b5x"    `b5`'s disjoint-seed twin (17..21), and "b10x" `b10`'s (17..26):
             the replication a bag that clears the freeze statistics must pass
             before it is nominated; never run otherwise

Every leg shares the panel (lib/data.py), the universe (lib/index.py), the
label (lib/labels.py), the split (lib/common.py), the refit counter
(lib/state.py), the booster call (lib/primary.py) and the monthly book
(lib/book.py, lib/trade.py). A bag member is byte for byte the booster its
seed trains alone, so `c_s7` and `c_s8` are two members of `b5`, and a leg's
score differs from another's only by which boosters are averaged.

Every other default a session is expected to change lives in lib/knobs.py,
each one exactly once, except the benchmark this book is built in, which lives
beside the read that uses it in lib/index.py and is overridden by the run fact.
"""

from lib import data, knobs, labels, primary, state, trade

CANDIDATE = "c_s7"
REFIT_PERIOD = "quarter"


def fit(context):
    seeds = knobs.SEEDS[CANDIDATE]
    cold = state.is_cold(context)
    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    target, realized = labels.rank_target(labels.residual(panel, knobs.HORIZON))
    samples = data.fit_samples(panel, target, realized)
    del panel
    reading = primary.fit(context, samples, cold, seeds)
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
