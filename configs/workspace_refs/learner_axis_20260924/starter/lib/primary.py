"""The learner: which booster(s) a candidate trains on the carrier's rows, and how they score.

Every leg trains on the same rows -- the carrier's 158 Alpha columns, its
benchmark-residual rank target, its embargoed split -- at the same parameter
point `knobs.PARAMS`. What a leg may move is written once, in `LEGS`:

    learner     "single"  one booster, the carrier's
                "double"  a DoubleEnsemble of `knobs.DOUBLE_ENSEMBLE["models"]`
                          boosters (lib/double_ensemble.py)
    objective   "regression"  the carrier's L2 on the rank target
                "lambdarank"  one query per decision bar, relevance = the rank
                              target cut into `knobs.RANK_GRADES` grades
                              (`labels.grades`), NDCG truncated at
                              `knobs.RANK_TRUNCATION`, early stopping on NDCG at
                              `knobs.SEATS` -- the book's own depth
    warm        whether a refit continues the previous booster (`init_model`,
                +`knobs.WARM_ROUNDS` trees) between the cold resets of
                `lib/state.py`. A DoubleEnsemble cannot: every refit recomputes
                its sample weights and feature subsets from freshly trained
                sub-models, and a booster cannot continue on a different column
                set. So "double" legs are cold on every refit by construction,
                and `c_cold` exists to price exactly that difference.

`c_base` goes through the same `_booster` call the head packs make, argument
for argument, so its booster is the carrier's.

There is no fallback scorer. A fit with too few samples raises, and a review
day with no usable state raises, so a replay fails instead of trading something
else.
"""

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from lib import common, data, double_ensemble, knobs, labels, state

LEGS = {
    # candidate: (learner, objective, warm chain)
    "c_base": ("single", "regression", True),
    "c_cold": ("single", "regression", False),
    "lr1": ("single", "lambdarank", True),
    "de1": ("double", "regression", False),
    "de_lr": ("double", "lambdarank", False),
}

MODEL_FILE = "/primary.txt"
MEMBER_FILE = "/double_{}.txt"
SUBSET_FILE = "/double_subsets.npy"
NAMES_FILE = "/primary_features.npy"


def warm(candidate):
    return LEGS[candidate][2]


def params(objective):
    """The registered parameter point, with the ranking objective swapped in where asked."""

    if objective == "regression":
        return dict(knobs.PARAMS)
    return {**knobs.PARAMS, "objective": "lambdarank", "metric": "ndcg", "eval_at": [knobs.SEATS],
            "lambdarank_truncation_level": knobs.RANK_TRUNCATION,
            "label_gain": [float(grade) for grade in range(knobs.RANK_GRADES)]}


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


def _group(bars):
    """Query sizes of rows grouped by bar, in row order."""

    return np.diff(np.r_[double_ensemble.starts_of(bars), len(bars)])


def _booster(objective, train, valid, rounds, columns=None, weights=None, previous=None, stop=True):
    """Train one booster on (X, y, bar) row sets; `columns` restricts both sets alike.

    `stop=False` trains exactly `rounds` trees (DoubleEnsemble sub-models after the
    first); the validation set is then only recorded, never used to stop.
    """

    names = data.FEATURE_NAMES if columns is None else [data.FEATURE_NAMES[c] for c in columns]
    sets = []
    for rows in (train, valid):
        matrix = rows["X"] if columns is None else np.ascontiguousarray(rows["X"][:, columns])
        if objective == "regression":
            sets.append((matrix, rows["y"], None))
        else:
            sets.append((matrix, labels.grades(rows["y"]), _group(rows["bar"])))
    (x_train, y_train, q_train), (x_valid, y_valid, q_valid) = sets
    train_set = lgb.Dataset(x_train, y_train, weight=weights, group=q_train, feature_name=names,
                            free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, y_valid, group=q_valid, reference=train_set, free_raw_data=True)
    return lgb.train(
        params(objective), train_set, num_boost_round=rounds,
        valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
        callbacks=([lgb.early_stopping(knobs.EARLY_STOP, verbose=False)] if stop else [])
        + [lgb.log_evaluation(0)],
    )


def _reading(booster):
    """The booster's best validation metric: L2 for regression, NDCG at SEATS for lambdarank."""

    return float(next(iter(booster.best_score["valid"].values())))


def _load_single(context):
    try:
        return lgb.Booster(model_file=context.state_dir + MODEL_FILE)
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def _load_double(context):
    try:
        subsets = np.load(context.state_dir + SUBSET_FILE)
        return [(lgb.Booster(model_file=context.state_dir + MEMBER_FILE.format(k)), np.flatnonzero(mask))
                for k, mask in enumerate(subsets)]
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def fit(context, samples, cold, candidate):
    """Train this candidate's learner on the window and replace its state; returns its validation reading."""

    learner, objective, _ = LEGS[candidate]
    train_mask, valid_mask, rows, bars = common.split_fit_dates(samples["bar"])
    if rows < knobs.MIN_SAMPLES or bars < knobs.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(
            f"fit has {rows} training rows on {bars} bars; need >= {knobs.MIN_SAMPLES} of each")
    train = {key: samples[key][train_mask] for key in ("X", "y", "bar")}
    valid = {key: samples[key][valid_mask] for key in ("X", "y", "bar")}
    if learner == "single":
        previous = None if cold else _load_single(context)
        rounds = knobs.COLD_ROUNDS if previous is None else knobs.WARM_ROUNDS
        booster = _booster(objective, train, valid, rounds, previous=previous)
        booster.save_model(context.state_dir + MODEL_FILE)
        reading = _reading(booster)
    else:
        members = double_ensemble.fit(
            lambda columns, weights, trees: _booster(
                objective, train, valid, trees or knobs.COLD_ROUNDS, columns=columns,
                weights=weights, stop=trees is None),
            train["X"], train["y"], train["bar"], objective, knobs.DOUBLE_ENSEMBLE,
            knobs.PARAMS["seed"])
        masks = np.zeros((len(members), len(data.FEATURE_NAMES)), dtype=bool)
        for k, (booster, subset) in enumerate(members):
            booster.save_model(context.state_dir + MEMBER_FILE.format(k))
            masks[k, subset] = True
        np.save(context.state_dir + SUBSET_FILE, masks)
        reading = float(np.mean([_reading(booster) for booster, _ in members]))
    np.save(context.state_dir + NAMES_FILE, np.array(data.FEATURE_NAMES))
    return reading


def report(context, candidate):
    """The readings every buy order carries so a result note never has to guess."""

    learner, objective, _ = LEGS[candidate]
    out = {"learner": learner, "objective": objective, **state.report(context)}
    if learner == "double":
        out["subset_sizes"] = [int(size) for size in np.load(context.state_dir + SUBSET_FILE).sum(axis=1)]
    return out


def score(context, panel, candidate):
    """(names,) RAW score of this candidate on the newest bar; NaN off the universe."""

    last = len(panel["dates"]) - 1
    keep = data.tradable(panel, last)
    out = np.full(len(panel["symbols"]), np.nan)
    if not keep.any():
        return out
    stored = [str(name) for name in np.load(context.state_dir + NAMES_FILE)]
    if stored != data.FEATURE_NAMES:
        raise RuntimeError("the fitted state's columns differ from data.FEATURE_NAMES")
    matrix = np.ascontiguousarray(data.decision_features(panel)[keep], dtype=np.float32)
    learner, objective, _ = LEGS[candidate]
    if learner == "single":
        booster = _load_single(context)
        if booster is None:
            raise RuntimeError("no fitted booster under the state directory")
        out[keep] = booster.predict(matrix)
    else:
        members = _load_double(context)
        if members is None:
            raise RuntimeError("no fitted DoubleEnsemble under the state directory")
        out[keep] = double_ensemble.predict(members, matrix, np.zeros(matrix.shape[0]), objective)
    return out
