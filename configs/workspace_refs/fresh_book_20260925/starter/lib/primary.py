"""The carrier: one LightGBM booster on the 158 Alpha columns, and how it scores.

Every leg trains the same booster -- the carrier's 158 Alpha columns, its
benchmark-residual rank target, its embargoed split, the parameter point
`knobs.PARAMS`, warm continuation between the cold resets of `lib/state.py`.
The only leg whose booster differs is `f3`, and only in the label horizon
(and therefore the embargo) it is trained on. Everything this arm moves lives
downstream of the score, in `lib/book.py` and `lib/trade.py`.

This is the head packs' `c_base` booster call, argument for argument.

There is no fallback scorer. A fit with too few samples raises, and a review
day with no usable state raises, so a replay fails instead of trading something
else.
"""

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from lib import common, data, knobs, state

MODEL_FILE = "/primary.txt"
NAMES_FILE = "/primary_features.npy"


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


def _load(context):
    try:
        return lgb.Booster(model_file=context.state_dir + MODEL_FILE)
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def fit(context, samples, cold, horizon):
    """Train (or continue) the booster on the window and replace the state; returns the validation L2."""

    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"], horizon)
    if rows < knobs.MIN_SAMPLES or bars < knobs.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(
            f"fit has {rows} training rows on {bars} bars; need >= {knobs.MIN_SAMPLES} of each")
    train_set = lgb.Dataset(samples["X"][train_mask], samples["y"][train_mask],
                            feature_name=data.FEATURE_NAMES, free_raw_data=True)
    valid_set = lgb.Dataset(samples["X"][valid_mask], samples["y"][valid_mask],
                            reference=train_set, free_raw_data=True)
    previous = None if cold else _load(context)
    booster = lgb.train(
        dict(knobs.PARAMS), train_set,
        num_boost_round=knobs.COLD_ROUNDS if previous is None else knobs.WARM_ROUNDS,
        valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
        callbacks=[lgb.early_stopping(knobs.EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    booster.save_model(context.state_dir + MODEL_FILE)
    np.save(context.state_dir + NAMES_FILE, np.array(data.FEATURE_NAMES))
    return float(next(iter(booster.best_score["valid"].values())))


def report(context):
    """The readings every buy order carries so a result note never has to guess."""

    return state.report(context)


def score(context, panel):
    """(names,) RAW score of the booster on the newest bar; NaN off the universe."""

    last = len(panel["dates"]) - 1
    keep = data.tradable(panel, last)
    out = np.full(len(panel["symbols"]), np.nan)
    if not keep.any():
        return out
    stored = [str(name) for name in np.load(context.state_dir + NAMES_FILE)]
    if stored != data.FEATURE_NAMES:
        raise RuntimeError("the fitted state's columns differ from data.FEATURE_NAMES")
    booster = _load(context)
    if booster is None:
        raise RuntimeError("no fitted booster under the state directory")
    matrix = np.ascontiguousarray(data.decision_features(panel)[keep], dtype=np.float32)
    out[keep] = booster.predict(matrix)
    return out
