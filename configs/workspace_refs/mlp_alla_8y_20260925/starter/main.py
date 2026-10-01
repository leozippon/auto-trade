"""Entry module: a persisted GRU sequence ranker inside the CSI 300, its tree control and the orthogonal variant.

The one constant below selects what this package is; `families.md` is the
authority for all three:

    g1       three fixed-seed GRUs over 60-day daily sequences of price, volume,
             turnover, limit-touch and margin-flow channels, trained on a CUDA
             device and WARM-STARTED at every refit from the checkpoint the
             previous one saved (lib/model.py)                 main candidate
    c_lgbm   the same panel, the same residual label and the same constituent
             universe, flattened into per-channel lags and window moments for
             LightGBM (lib/tree.py)      required control, never nominated
    g1r      the GRU score with the control's score regressed out of it
             (lib/trade.py)                               registered variant

Everything else is shared: the panel builder and its window normalisation
(lib/panel.py), the benchmark-residual 10-day label (lib/label.py), the
constituent sections (lib/index.py), the trailing three-year quarterly refit
dates and the 30-seat equal-cash weekly book (lib/trade.py). Changing the
constant changes nothing else, which is what makes the control a control.

Refitting is quarterly. The first fit of a replay is cold because the state
directory starts empty; every later one continues from the saved checkpoint,
and every fourth one resets cold so a chain cannot drift forever. The candidate
and its control reset on the same refits (lib/state.py).

Knobs, one per registered axis: trade.SEATS / KEEP_BAND / MAX_SWAPS,
model.TRAIN_YEARS / SEEDS / WARM_EPOCHS, state.COLD_EVERY, panel.SEQ_LEN,
label.INDUSTRY_DEMEAN.
"""

from lib import model, state, trade, tree

CANDIDATE = "g1"
REFIT_PERIOD = "quarter"


def fit(context):
    cold = state.is_cold(context)
    reading = 0.0
    if CANDIDATE in ("g1", "g1r"):
        reading = model.fit(context, cold)
    if CANDIDATE in ("c_lgbm", "g1r"):
        control = tree.fit(context, cold)
        reading = control if CANDIDATE == "c_lgbm" else reading
    state.bump(context, cold, reading)


def generate_orders(context):
    return trade.run(context, CANDIDATE)
