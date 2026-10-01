"""The registered LightGBM control `c_lgbm`: the same panel, label and universe, flattened.

The round's rule is that a learned arm must carry a tree control on the SAME
features, the SAME label and the SAME universe, so that "the sequence model
found something" can be separated from "a price-volume learner found something".
This file is that control and nothing else.

Same features. A tree cannot read a 60-step sequence, so each name's window --
the very array `panel.window` hands the GRU -- is reduced to a flat row: the
channel values at `LAGS` steps counted back from the window's last day, plus the
window mean and standard deviation of each channel. `SEQ_FEATURES * (len(LAGS)
+ 2)` columns, built from the identical normalised tensor. Nothing is added that
the GRU cannot see, and nothing the GRU sees is dropped except the shape of the
path between the sampled lags -- which is exactly the thing the sequence model
is being asked to prove it uses.

Same label, same dates, same names: `model.fit_dates` and `model.scorable`.

Persistence. The booster is written under `context.state_dir` and a warm refit
continues it with `WARM_ROUNDS` more trees (`init_model`), resetting cold on
exactly the refits `lib/state.py` resets the GRU on. A control that refits cold
while the candidate warm-starts would not be a control.
"""

import lightgbm as lgb
import numpy as np

from lib import model, panel as P

LAGS = (0, 4, 9, 19, 39, 59)   # steps back from the last day of the window
PARAMS = {"objective": "regression", "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "min_data_in_leaf": 200, "lambda_l2": 10, "num_leaves": 31,
          "learning_rate": 0.05, "num_threads": 8, "seed": 7, "verbose": -1}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50
FEATURE_COUNT = P.SEQ_FEATURES * (len(LAGS) + 2)


def model_path(context):
    return context.state_dir + "/lgbm_control.txt"


def flatten(window):
    """(names, SEQ_LEN, SEQ_FEATURES) -> (names, FEATURE_COUNT) float32."""

    last = window.shape[1] - 1
    taken = [window[:, last - lag, :] for lag in LAGS]
    taken.append(window.mean(axis=1))
    taken.append(window.std(axis=1))
    return np.concatenate(taken, axis=1).astype(np.float32)


def rows(data, dates):
    """(X, y) over the requested date indexes, built from the same windows the GRU trains on."""

    features, targets = [], []
    for t in dates:
        idx = model.scorable(data, t)
        idx = idx[np.isfinite(data["yrank"][t, idx])]
        if len(idx) < model.MIN_NAMES:
            continue
        features.append(flatten(P.window(data, t, idx)))
        targets.append(data["yrank"][t, idx].astype(np.float32))
    if not features:
        raise RuntimeError("the control has no scorable date in this segment")
    return np.vstack(features), np.concatenate(targets)


def _previous(context):
    try:
        return lgb.Booster(model_file=model_path(context))
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def fit(context, cold):
    """Train the control on the trailing window; cold from scratch, warm by continuation."""

    data = P.build(context, model.FIT_CALENDAR_DAYS, labels=True)
    train, valid = model.fit_dates(context, data)
    x_train, y_train = rows(data, train)
    x_valid, y_valid = rows(data, valid)
    booster = None if cold else _previous(context)
    train_set = lgb.Dataset(x_train, y_train, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set, free_raw_data=True)
    trained = lgb.train(
        PARAMS, train_set,
        num_boost_round=COLD_ROUNDS if booster is None else WARM_ROUNDS,
        valid_sets=[valid_set], valid_names=["valid"], init_model=booster,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    trained.save_model(model_path(context))
    return float(trained.best_score.get("valid", {}).get("l2", 0.0))


def score(context, data):
    """Cross-sectional rank in [0, 1] of the control's prediction on the newest row."""

    last = len(data["dates"]) - 1
    idx = model.scorable(data, last)
    out = np.full(len(data["codes"]), np.nan)
    if len(idx) < model.MIN_NAMES:
        return out
    booster = _previous(context)
    if booster is None:
        raise RuntimeError("no fitted control booster under the state directory")
    prediction = booster.predict(flatten(P.window(data, last, idx)))
    order = np.argsort(np.argsort(prediction, kind="stable"), kind="stable")
    out[idx] = order / max(1, len(idx) - 1)
    return out
