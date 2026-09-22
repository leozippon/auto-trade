"""The reported control: LightGBM on the same features, label, universe and dates.

This is the leg the learner has to beat to be worth a full-span batch. It is
the shape that actually graduated once in this repository -- Alpha158
operators combined by a gradient-boosted tree on a cross-sectional rank label
-- restricted here to the constituent panel and to the identical training and
validation dates the network gets from `lib/panel.py`.

Its parameters are NOT knobs. A control whose settings move with the
candidate stops being a fixed reference, and the incremental-information gate
in `families.md` is measured against it. One point, not a grid: the pack's
fit-cost note is that widening the grid buys fit seconds, not information.

The control refits cold every REFIT_PERIOD by design: the warm start is the
candidate's mechanism, and a control that also carried state would confound
the two. The booster is written to `state_dir` with `save_model` and read
back per review with `Booster(model_file=...)`, so a refit between two
reviews is picked up without a worker restart.

There is no fallback scorer: too few samples, or a review with no usable
state, raises. A replay that fails is better than one that quietly trades a
different strategy.
"""

import lightgbm as lgb
import numpy as np

from lib import panel

MODEL_FILE = "/control_lgbm.txt"
META_FILE = "/control_meta.npy"
PARAMS = {
    "objective": "regression",
    # 15, not the 31 the whole-market lineage used: that lineage fitted 2,400-3,500
    # names a date, this one 300. Measured on two decision vintages, 15 is in the top
    # two at both (validation IC 0.057 / 0.123 against 31's 0.044 / 0.121) and halves
    # the train-validation gap on the harder one. The neighbourhood 7/15/23/31/63 was
    # read as a shape, not a point.
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    # Pinned: the container has 8 CPUs and three replays may share the host.
    "num_threads": 8,
    "seed": 7,
    "verbose": -1,
}
ROUNDS = 600
EARLY_STOP = 50


def _rows(data, dates):
    features, labels, sizes = [], [], []
    for t in dates:
        rows = np.nonzero(data["member"][t] & np.isfinite(data["yrank"][t]))[0]
        features.append(data["feat"][t, rows])
        labels.append(data["yrank"][t, rows])
        sizes.append(len(rows))
    return np.vstack(features), np.concatenate(labels), sizes


def _date_ic(prediction, label, sizes):
    """Mean per-date Pearson IC -- the same number the networks early-stop on.

    The booster still stops on L2; this is measured after the fact so a result
    note can put the control's own validation IC beside the candidate's instead
    of comparing two differently selected models on trust.
    """

    scores, at = [], 0
    for count in sizes:
        left, right = prediction[at:at + count], label[at:at + count]
        at += count
        if count > 2 and left.std() > 0 and right.std() > 0:
            scores.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def fit(context):
    data, train, valid = panel.fit_window(context)
    train_x, train_y, _ = _rows(data, train)
    valid_x, valid_y, valid_sizes = _rows(data, valid)
    del data
    training = lgb.Dataset(train_x, train_y, free_raw_data=True)
    validation = lgb.Dataset(valid_x, valid_y, reference=training, free_raw_data=True)
    booster = lgb.train(
        PARAMS, training, num_boost_round=ROUNDS,
        valid_sets=[validation], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    best = booster.best_iteration or ROUNDS
    booster.save_model(context.state_dir + MODEL_FILE, num_iteration=best)
    np.save(context.state_dir + META_FILE,
            np.asarray([float(best),
                        _date_ic(booster.predict(valid_x, num_iteration=best),
                                 valid_y, valid_sizes)], dtype=np.float64))


def score(context, data):
    """Percentile rank in [0, 1] of every scorable name on the newest row."""

    last = len(data["dates"]) - 1
    rows = panel.scorable(data, last)
    if len(rows) == 0:
        raise RuntimeError("no visible constituent with a bar on the newest row")
    booster = lgb.Booster(model_file=context.state_dir + MODEL_FILE)
    prediction = booster.predict(np.ascontiguousarray(data["feat"][last, rows], dtype=np.float32))
    out = np.full(len(data["codes"]), np.nan)
    order = np.argsort(np.argsort(prediction, kind="stable"), kind="stable")
    out[rows] = order / max(1, len(rows) - 1)
    return out


def report(context):
    meta = np.load(context.state_dir + META_FILE)
    return {"control_rounds": int(meta[0]), "valid_ic": round(float(meta[1]), 5)}
