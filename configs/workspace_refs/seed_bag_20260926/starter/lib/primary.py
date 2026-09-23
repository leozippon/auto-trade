"""The carrier: LightGBM boosters on the 158 Alpha columns, one per training seed, and how they score.

Every booster is the head packs' `c_base` booster call, argument for argument:
the carrier's 158 Alpha columns, its benchmark-residual rank target, its
embargoed split, the parameter point `knobs.PARAMS`, warm continuation between
the cold resets of `lib/state.py`. The one argument a leg supplies is the
training seed, and a leg fits one booster per seed in `knobs.SEEDS`, each on
its own freshly built dataset, so a bag member is the file its seed would
train alone and `c_s7`'s booster is `b5`'s first member.

Every booster continues its own previous file on a warm refit; the refit
counter is shared, so every member is cold or warm on the same refit.

The score is the average over the leg's boosters of each booster's
cross-sectional percentile rank on the newest bar (`members`), so every
member carries the same weight whatever the scale of its raw predictions.
With one seed the average is that booster's own ranking, and the book is the
template's order for order.

There is no fallback scorer. A fit with too few samples raises, a warm refit
or a review day without a member's booster raises, so a replay fails instead
of trading something else.
"""

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from lib import common, data, knobs, state

NAMES_FILE = "/primary_features.npy"


def _model_file(seed):
    return f"/primary_s{int(seed)}.txt"


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


def _load(context, seed):
    try:
        return lgb.Booster(model_file=context.state_dir + _model_file(seed))
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError) as error:
        raise RuntimeError(f"no fitted booster for seed {seed} under the state directory") from error


def fit(context, samples, cold, seeds):
    """Train (or continue) one booster per seed and replace the state; returns the mean validation L2."""

    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"], knobs.HORIZON)
    if rows < knobs.MIN_SAMPLES or bars < knobs.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(
            f"fit has {rows} training rows on {bars} bars; need >= {knobs.MIN_SAMPLES} of each")
    x_train, y_train = samples["X"][train_mask], samples["y"][train_mask]
    x_valid, y_valid = samples["X"][valid_mask], samples["y"][valid_mask]
    readings = []
    for seed in seeds:
        train_set = lgb.Dataset(x_train, y_train, feature_name=data.FEATURE_NAMES, free_raw_data=True)
        valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set, free_raw_data=True)
        previous = None if cold else _load(context, seed)
        booster = lgb.train(
            {**knobs.PARAMS, "seed": int(seed)}, train_set,
            num_boost_round=knobs.COLD_ROUNDS if previous is None else knobs.WARM_ROUNDS,
            valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
            callbacks=[lgb.early_stopping(knobs.EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )
        booster.save_model(context.state_dir + _model_file(seed))
        readings.append(float(next(iter(booster.best_score["valid"].values()))))
        del booster, previous, train_set, valid_set
    np.save(context.state_dir + NAMES_FILE, np.array(data.FEATURE_NAMES))
    return float(np.mean(readings))


def report(context):
    """The readings every buy order carries so a result note never has to guess."""

    return state.report(context)


def members(context, panel, seeds):
    """(len(seeds), names) percentile rank of each seed's booster on the newest bar; NaN off the universe."""

    last = len(panel["dates"]) - 1
    keep = data.tradable(panel, last)
    out = np.full((len(seeds), len(panel["symbols"])), np.nan)
    if not keep.any():
        return out
    stored = [str(name) for name in np.load(context.state_dir + NAMES_FILE)]
    if stored != data.FEATURE_NAMES:
        raise RuntimeError("the fitted state's columns differ from data.FEATURE_NAMES")
    matrix = np.ascontiguousarray(data.decision_features(panel)[keep], dtype=np.float32)
    for row, seed in enumerate(seeds):
        out[row, keep] = pct_rank(_load(context, seed).predict(matrix))
    return out


def score(context, panel, seeds):
    """(names,) the leg's score: the mean of its members' percentile ranks; NaN off the universe."""

    return members(context, panel, seeds).mean(axis=0)
