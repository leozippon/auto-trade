"""The carrier: the graduated lineage's LightGBM ranker, on the index and a residual label.

Mechanism, objective and parameters are the frozen artifact's, with one
deliberate simplification kept from its successor pack: a SINGLE registered
parameter point rather than a four-point grid. The grid bought seconds, not
readings, and every distinct revision this arm validates raises its own
deflated-Sharpe bar -- so a grid that does not change the answer makes the
freeze harder for nothing.

The candidate and the control are THE SAME CODE with a different column set
(`data.feature_names`). That is the whole design of an ablation arm: if the
control differed in its split, its rounds, its seed or its refit cadence, the
difference the ablation measures would not be the block.

Persistence. The booster is written under `context.state_dir`; a warm refit
continues it with `knobs.WARM_ROUNDS` more trees (`init_model`) rather than
starting over, and `lib/state.py` resets the chain cold every
`knobs.COLD_EVERY`-th refit. The control resets on exactly the same refits.

Gain share. `gain_share` is the fraction of the booster's total split gain that
lands on the block's columns. It is the second half of this arm's offline gate
and it is free: a booster that has already been fitted carries it. Read it as a
NECESSARY condition, never a sufficient one -- a tree will always spend some
gain on any column with variance, which is exactly why the gate pairs it with
an ablation on the active series and why the threshold is a floor rather than a
target.

There is no fallback scorer. A fit with too few samples raises, and a review
day with no usable state raises, so a replay fails instead of trading something
else.
"""

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from lib import common, data, knobs

MODEL_FILE = "/primary.txt"
NAMES_FILE = "/primary_features.npy"
GAIN_FILE = "/primary_gain.npy"


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


def gain_share(booster, names, block_names):
    """Share of the booster's total split gain that lands on the block's columns."""

    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
    total = float(gain.sum())
    if total <= 0.0:
        return 0.0
    used = [position for position, name in enumerate(names) if name in set(block_names)]
    return float(gain[used].sum() / total)


def fit(context, samples, cold, candidate):
    """Train the carrier on the window and replace its state; returns (validation L2, gain share)."""

    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"])
    if rows < knobs.MIN_SAMPLES or bars < knobs.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(
            f"fit has {rows} training rows on {bars} bars; need >= {knobs.MIN_SAMPLES} of each")
    names = data.feature_names(candidate)
    previous = None if cold else load(context)
    rounds = knobs.COLD_ROUNDS if previous is None else knobs.WARM_ROUNDS
    train_set = lgb.Dataset(samples["X"][train_mask], samples["y"][train_mask],
                            feature_name=names, free_raw_data=True)
    valid_set = lgb.Dataset(samples["X"][valid_mask], samples["y"][valid_mask],
                            reference=train_set, free_raw_data=True)
    booster = lgb.train(
        knobs.PARAMS, train_set, num_boost_round=rounds,
        valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
        callbacks=[lgb.early_stopping(knobs.EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    booster.save_model(model_path(context))
    np.save(context.state_dir + NAMES_FILE, np.array(names))
    share = gain_share(booster, names, data.BLOCK_NAMES)
    np.save(context.state_dir + GAIN_FILE, np.asarray([share], dtype=np.float64))
    return float(booster.best_score.get("valid", {}).get("l2", 0.0)), share


def report(context):
    """The readings every buy order carries so a result note never has to guess."""

    try:
        share = float(np.load(context.state_dir + GAIN_FILE)[0])
    except FileNotFoundError:
        share = float("nan")
    return {"block_gain_share": round(share, 5)}


def score(context, panel, values, candidate):
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
    if stored != data.feature_names(candidate):
        raise RuntimeError("the fitted booster's columns differ from feature_names(candidate)")
    matrix = data.decision_features(panel, values, candidate)
    out[keep] = booster.predict(np.ascontiguousarray(matrix[keep], dtype=np.float32))
    return out
