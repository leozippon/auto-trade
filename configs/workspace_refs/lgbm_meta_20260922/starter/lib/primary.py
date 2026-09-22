"""The primary LightGBM ranker: the graduated lineage's scorer, on the index and on a residual label.

Mechanism, parameters and objective are the frozen artifact's, with one
deliberate simplification: a SINGLE registered parameter point rather than the
four-point grid. The grid bought seconds, not readings, in the calibration arm,
and every distinct revision this arm validates raises its own deflated-Sharpe
bar -- so a grid that does not change the answer makes the freeze harder for
nothing. `families.md` keeps the grid as a registered variant axis.

Persistence. The booster is written under `context.state_dir`; a warm refit
continues it with `WARM_ROUNDS` more trees (`init_model`) rather than starting
over, and `lib/state.py` resets the chain cold every `state.COLD_EVERY`-th
refit so a chain cannot run the whole replay. Continuation is what makes the
primary a model that accumulates: quarter four's booster still contains the
trees quarter one fitted, re-weighted by everything since.

There is no fallback scorer. A fit with too few samples raises, and a review day
with no usable state raises, so a replay fails instead of trading something else.
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
    "num_threads": 8,
    "seed": 7,
    "verbose": -1,
}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50


def model_path(context, tag):
    return context.state_dir + "/primary_" + tag + ".txt"


def load(context, tag):
    try:
        return lgb.Booster(model_file=model_path(context, tag))
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


def train(x_train, y_train, x_valid, y_valid, booster, rounds):
    train_set = lgb.Dataset(x_train, y_train, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set, free_raw_data=True)
    return lgb.train(
        PARAMS, train_set, num_boost_round=rounds,
        valid_sets=[valid_set], valid_names=["valid"], init_model=booster,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )


def fit(context, samples, cold, tag):
    """Train the primary on the window and replace its state file; returns its validation L2."""

    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"])
    if rows < common.MIN_SAMPLES or bars < common.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(f"fit has {rows} training rows on {bars} bars; need >= {common.MIN_SAMPLES} of each")
    previous = None if cold else load(context, tag)
    booster = train(samples["X"][train_mask], samples["y"][train_mask],
                    samples["X"][valid_mask], samples["y"][valid_mask],
                    previous, COLD_ROUNDS if previous is None else WARM_ROUNDS)
    booster.save_model(model_path(context, tag))
    return float(booster.best_score.get("valid", {}).get("l2", 0.0))


def out_of_fold(samples):
    """(rows,) out-of-fold prediction for every training row, from purged contiguous folds.

    Each fold trains a fresh booster on the rest of the window and scores the
    fold it never saw. Cold by construction: an out-of-fold score continued from
    a booster that already read the fold would not be out of fold.
    """

    out = np.full(samples["y"].shape, np.nan)
    for train_mask, test_mask in common.purged_folds(samples["bar"]):
        inner_train, inner_valid, rows, _ = common.split_fit_dates(samples["bar"][train_mask])
        if rows < common.MIN_SAMPLES or not inner_valid.any():
            raise RuntimeError("a purged fold has no embargoed validation segment")
        x_fold = samples["X"][train_mask]
        y_fold = samples["y"][train_mask]
        booster = train(x_fold[inner_train], y_fold[inner_train],
                        x_fold[inner_valid], y_fold[inner_valid], None, COLD_ROUNDS)
        out[test_mask] = booster.predict(samples["X"][test_mask])
        del booster, x_fold, y_fold
    if not np.isfinite(out).any():
        raise RuntimeError("the out-of-fold pass produced no prediction")
    return out


def score(context, panel, tag):
    """(names,) RAW prediction of the primary on the newest bar; NaN off the universe.

    Raw, not ranked: the meta stage's own scaling is bar-relative and has to see
    the spread of the day's scores, and `pct_rank` recovers the ordering the
    book needs whenever it needs it.
    """

    last = len(panel["dates"]) - 1
    keep = data.tradable(panel, last)
    out = np.full(len(panel["symbols"]), np.nan)
    if not keep.any():
        return out
    booster = load(context, tag)
    if booster is None:
        raise RuntimeError(f"no fitted primary booster '{tag}' under the state directory")
    features = data.decision_features(panel)
    out[keep] = booster.predict(np.ascontiguousarray(features[keep], dtype=np.float32))
    return out
