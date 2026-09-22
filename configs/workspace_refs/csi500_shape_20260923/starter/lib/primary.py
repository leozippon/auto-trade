"""The carrier: the graduated lineage's LightGBM ranker, unchanged, on whatever label it is given.

Mechanism, objective and parameters are the frozen artifact's, with the one
simplification its successor pack already made: a SINGLE registered parameter
point rather than a four-point grid. The grid bought seconds, not readings, in
the calibration arm, and every distinct revision this arm validates raises its
own deflated-Sharpe bar -- so a grid that does not change the answer makes the
freeze harder for nothing. `families.md` keeps the grid as a registered variant
axis and does not open it by default.

This file is the part of the arm that must NOT move. The whole claim under test
is "the same carrier, a different label," so `PARAMS`, the rounds, the early
stop and the persistence cadence are identical across every leg of a batch; the
only thing that differs between two legs is the target vector handed to `fit`
and the embargo that target's horizon requires.

Persistence. The booster is written under `context.state_dir`; a warm refit
continues it with WARM_ROUNDS more trees (`init_model`) rather than starting
over, and `lib/state.py` resets the chain cold every `state.COLD_EVERY`-th
refit so a chain cannot run a whole replay unbroken.

There is no fallback scorer. A fit with too few samples raises, and a review day
with no usable state raises, so a replay fails instead of quietly trading
something else.
"""

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from lib import common, data

PARAMS = {
    "objective": "regression",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    "num_leaves": 31,
    "learning_rate": 0.05,
    # Pinned: the container has 8 CPUs and a batch may replay three legs at once.
    "num_threads": 8,
    "seed": 7,
    "verbose": -1,
}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50
MODEL_FILE = "/primary.txt"


def model_path(context):
    return context.state_dir + MODEL_FILE


def load(context):
    try:
        return lgb.Booster(model_file=model_path(context))
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def pct_rank(prediction):
    """Cross-sectional percentile rank in [0, 1]; ties average, non-finite stays NaN."""

    prediction = np.asarray(prediction, dtype=np.float64)
    out = np.full(prediction.shape, np.nan)
    finite = np.isfinite(prediction)
    count = int(finite.sum())
    if count == 0:
        return out
    out[finite] = (rankdata(prediction[finite], method="average") - 0.5) / count
    return out


def fit(context, samples, cold, embargo):
    """Train the carrier on the window and replace its state file; returns its validation L2."""

    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"], embargo)
    if rows < common.MIN_SAMPLES or bars < common.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(
            f"fit has {rows} training rows on {bars} bars; need >= {common.MIN_SAMPLES} of each")
    previous = None if cold else load(context)
    train_set = lgb.Dataset(samples["X"][train_mask], samples["y"][train_mask], free_raw_data=True)
    valid_set = lgb.Dataset(samples["X"][valid_mask], samples["y"][valid_mask],
                            reference=train_set, free_raw_data=True)
    booster = lgb.train(
        PARAMS, train_set, num_boost_round=COLD_ROUNDS if previous is None else WARM_ROUNDS,
        valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    booster.save_model(model_path(context))
    return float(booster.best_score.get("valid", {}).get("l2", 0.0))


def score(context, panel):
    """(names,) RAW prediction of the carrier on the newest bar; NaN off the scorable set."""

    last = len(panel["dates"]) - 1
    keep = data.tradable(panel, last)
    out = np.full(len(panel["symbols"]), np.nan)
    if not keep.any():
        return out
    booster = load(context)
    if booster is None:
        raise RuntimeError("no fitted booster under the state directory")
    features = data.decision_features(panel)
    out[keep] = booster.predict(np.ascontiguousarray(features[keep], dtype=np.float32))
    return out
