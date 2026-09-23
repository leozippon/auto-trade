"""The carrier: the graduated lineage's LightGBM ranker on the index and a residual label.

Read by two legs only: `c_base` trades its score as is (a register-only
control), `n2` rank-averages it with the holdings change (`lib/candidates.py`).
Mechanism, objective and the single registered parameter point are the
lineage's, unchanged, so `c_base` here is the same book the register's other
`c_base` readings came from.

Persistence. The booster is written under `context.state_dir`; a warm refit
continues it with `knobs.WARM_ROUNDS` more trees (`init_model`) rather than
starting over, and `lib/state.py` resets the chain cold every
`knobs.COLD_EVERY`-th refit.

There is no fallback scorer. A fit with too few samples raises, and a review
day with no usable state raises, so a replay fails instead of trading something
else.
"""

import lightgbm as lgb
import numpy as np

from lib import common, data, knobs

MODEL_FILE = "/primary.txt"
NAMES_FILE = "/primary_features.npy"


def model_path(context):
    return context.state_dir + MODEL_FILE


def load(context):
    try:
        return lgb.Booster(model_file=model_path(context))
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def fit(context, samples, cold):
    """Train the carrier on the window and replace its state; returns the validation L2."""

    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"])
    if rows < knobs.MIN_SAMPLES or bars < knobs.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(
            f"fit has {rows} training rows on {bars} bars; need >= {knobs.MIN_SAMPLES} of each")
    previous = None if cold else load(context)
    rounds = knobs.COLD_ROUNDS if previous is None else knobs.WARM_ROUNDS
    train_set = lgb.Dataset(samples["X"][train_mask], samples["y"][train_mask],
                            feature_name=data.FEATURE_NAMES, free_raw_data=True)
    valid_set = lgb.Dataset(samples["X"][valid_mask], samples["y"][valid_mask],
                            reference=train_set, free_raw_data=True)
    booster = lgb.train(
        knobs.PARAMS, train_set, num_boost_round=rounds,
        valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
        callbacks=[lgb.early_stopping(knobs.EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    booster.save_model(model_path(context))
    np.save(context.state_dir + NAMES_FILE, np.array(data.FEATURE_NAMES))
    return float(booster.best_score.get("valid", {}).get("l2", 0.0))


def score(context, panel):
    """(names,) RAW prediction of the carrier on the newest bar; NaN off the universe."""

    last = len(panel["dates"]) - 1
    keep = data.tradable(panel, last)
    out = np.full(len(panel["symbols"]), np.nan)
    if not keep.any():
        return out
    booster = load(context)
    if booster is None:
        raise RuntimeError("no fitted booster under the state directory")
    stored = [str(name) for name in np.load(context.state_dir + NAMES_FILE)]
    if stored != data.FEATURE_NAMES:
        raise RuntimeError("the fitted booster's columns differ from data.FEATURE_NAMES")
    matrix = data.decision_features(panel)
    out[keep] = booster.predict(np.ascontiguousarray(matrix[keep], dtype=np.float32))
    return out
