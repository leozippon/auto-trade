"""The two LightGBM learners: the tree `knobs.RESIDUALISE` regresses out, and late fusion's fundamentals tree.

The residualising tree is the 100k bag's tree control unchanged. A tree cannot
read a 60-step sequence, so each name's window -- the very array
`panel.window` hands the heads -- is reduced to a flat row: the channel values
at `LAGS` steps back from the window's last day plus the window mean and
standard deviation of each channel, 12 x 8 = 96 columns of the identical
normalised tensor, on the heads' dates and names (`model.fit_dates`,
`model.scorable`). Its booster is written under `context.state_dir`; a warm
refit continues it with `WARM_ROUNDS` more trees and resets cold on exactly
the refits `lib/state.py` resets the heads on.

The fundamentals tree (`knobs.FUND == "late"`) is the same learner on the
`fusion.FEATURES` ranks of the same names, label and fit dates, keeping only
the dates on which at least `fusion.MIN_COVERAGE` of the names have a report
(the domain's history starts inside the first refits' windows). It is fitted
from scratch at every refit: it is not part of the recipe and carries no warm
chain.
"""

import lightgbm as lgb
import numpy as np

from lib import fusion, knobs, model, panel as P

LAGS = (0, 4, 9, 19, 39, 59)   # steps back from the last day of the window
# Seed 7 at SEED_BASE 1000 (the recipe's tree); another seed base moves the
# trees with the heads, so a leg on that base changes every seed at once.
PARAMS = {"objective": "regression", "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "min_data_in_leaf": 200, "lambda_l2": 10, "num_leaves": 31,
          "learning_rate": 0.05, "num_threads": 8, "seed": 7 + knobs.SEED_BASE - 1000, "verbose": -1}
# The fundamentals tree also pins LightGBM's histogram layout and its
# deterministic mode (seed and threads are fixed above): left to itself the
# library picks the layout by a timing test, so two fits of the same rows can
# differ, and c_perm is compared with its candidate run by run.
FUND_PARAMS = {**PARAMS, "deterministic": True, "force_row_wise": True}
COLD_ROUNDS = 600
WARM_ROUNDS = 150
EARLY_STOP = 50


def model_path(context):
    return context.state_dir + "/lgbm_control.txt"


def fund_path(context):
    return context.state_dir + "/lgbm_fund.txt"


def flatten(window):
    """(names, SEQ_LEN, SEQ_FEATURES) -> (names, SEQ_FEATURES * (len(LAGS) + 2)) float32."""

    last = window.shape[1] - 1
    taken = [window[:, last - lag, :] for lag in LAGS]
    taken.append(window.mean(axis=1))
    taken.append(window.std(axis=1))
    return np.concatenate(taken, axis=1).astype(np.float32)


def rows(data, dates):
    """(X, y) over the requested date indexes, built from the same windows the heads train on."""

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


def fund_rows(data, fx, dates):
    """(X, y) of the fundamentals tree over the dates with enough reports."""

    features, targets = [], []
    for t in dates:
        idx = model.scorable(data, t)
        idx = idx[np.isfinite(data["yrank"][t, idx])]
        if len(idx) < model.MIN_NAMES:
            continue
        x = fusion.take(fx, data, t, idx)
        if np.isfinite(x).any(axis=1).mean() < fusion.MIN_COVERAGE:
            continue
        features.append(x)
        targets.append(data["yrank"][t, idx].astype(np.float32))
    if not features:
        raise RuntimeError("no date of this segment has fundamentals for enough names")
    return np.vstack(features), np.concatenate(targets)


def _previous(path):
    try:
        return lgb.Booster(model_file=path)
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def _train(train, valid, rounds, init, params):
    train_set = lgb.Dataset(*train, free_raw_data=True)
    valid_set = lgb.Dataset(*valid, reference=train_set, free_raw_data=True)
    return lgb.train(
        params, train_set, num_boost_round=rounds,
        valid_sets=[valid_set], valid_names=["valid"], init_model=init,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )


def fit(context, data, cold):
    """Train the residualising tree on the trailing window; cold from scratch, warm by continuation."""

    train, valid = model.fit_dates(context, data)
    booster = None
    if not cold:
        booster = _previous(model_path(context))
        if booster is None:
            raise RuntimeError("warm refit without a fitted control booster")
    trained = _train(rows(data, train), rows(data, valid), COLD_ROUNDS if booster is None else WARM_ROUNDS,
                     booster, PARAMS)
    trained.save_model(model_path(context))


def fit_fund(context, data, fx):
    """Train the fundamentals tree from scratch on the trailing window."""

    train, valid = model.fit_dates(context, data)
    trained = _train(fund_rows(data, fx, train), fund_rows(data, fx, valid), COLD_ROUNDS, None, FUND_PARAMS)
    trained.save_model(fund_path(context))


def _ranked(prediction, idx, size):
    out = np.full(size, np.nan)
    order = np.argsort(np.argsort(prediction, kind="stable"), kind="stable")
    out[idx] = order / max(1, len(idx) - 1)
    return out


def score(context, data):
    """Cross-sectional rank in [0, 1] of the residualising tree's prediction on the newest row."""

    last = len(data["dates"]) - 1
    idx = model.scorable(data, last)
    if len(idx) < model.MIN_NAMES:
        return np.full(len(data["codes"]), np.nan)
    booster = _previous(model_path(context))
    if booster is None:
        raise RuntimeError("no fitted control booster under the state directory")
    return _ranked(booster.predict(flatten(P.window(data, last, idx))), idx, len(data["codes"]))


def score_fund(context, data, fx):
    """Cross-sectional rank in [0, 1] of the fundamentals tree's prediction on the newest row."""

    last = len(data["dates"]) - 1
    idx = model.scorable(data, last)
    if len(idx) < model.MIN_NAMES:
        return np.full(len(data["codes"]), np.nan)
    booster = _previous(fund_path(context))
    if booster is None:
        raise RuntimeError("no fitted fundamentals booster under the state directory")
    return _ranked(booster.predict(fusion.take(fx, data, last, idx)), idx, len(data["codes"]))
