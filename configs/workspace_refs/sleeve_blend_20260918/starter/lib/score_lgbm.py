"""LightGBM cross-sectional ranker: the frozen artifact's grid, trained in fit, scored on a review day.

Mechanism (the frozen artifact's scorer): pre-registered 4-point grid num_leaves {31, 63} x
learning_rate {0.05, 0.1}; objective=regression on the per-bar label rank - 0.5,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200,
lambda_l2=10, num_threads=8; validation = the last VALID_DAYS training bars after an
EMBARGO_DAYS gap (lib/common.py); 600 rounds with early stopping 50; among the points
within 5% of the minimum validation L2 the lexicographically smallest
(num_leaves, learning_rate) wins. Each refit trains one new booster and replaces the
previous one.

State files under context.state_dir, rewritten by every fit:
  model.txt           the selected booster at its best iteration (save_model)
  model_meta.npy      float64[3] = [leaf_idx, lr_idx, best_iter]
  feature_names.npy   the column names the booster was trained on

There is no fallback scorer: a fit with too few samples raises, and a review day with
no usable state raises, so a replay fails instead of trading a different strategy.
"""

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from lib import common, data

LEAVES = (31, 63)
LRS = (0.05, 0.1)
NUM_ROUNDS = 600
EARLY_STOP = 50
FIXED_PARAMS = {
    "objective": "regression",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    "verbose": -1,
    # Pin the thread count: lgbm's default (auto) can oversubscribe the container quota.
    "num_threads": 8,
}


def _pct_rank(pred):
    """Cross-sectional percentile rank in [0, 1]; only finite values participate, ties
    get the average rank, mapped to (r - 0.5) / n_finite; non-finite input stays NaN."""
    pred = np.asarray(pred, dtype=np.float64)
    out = np.full(pred.shape, np.nan, dtype=np.float64)
    m = np.isfinite(pred)
    n = int(m.sum())
    if n == 0:
        return out
    out[m] = (rankdata(pred[m], method="average") - 0.5) / n
    return out


def _train_grid(dtr, dva, Xva, yva, names, seed=None, num_boost_round=NUM_ROUNDS,
                early_stop=EARLY_STOP):
    """4-point grid + embargoed early stopping + 5% plateau / lexicographic selection.

    dtr/dva are the prebuilt lgb.Dataset pair (constructed once by fit with
    free_raw_data=True): lightgbm copies the dense data into its C++ container
    lazily on the first train and, with free_raw_data, drops that copy after
    binning, so the same pair serves all 4 grid points (rebuild-per-point
    churn gone; verified bit-identical l2/best_iter vs a fresh-dataset control).
    After the first train the python-side raw refs on dtr are cleared so the
    big train buffer is released while the remaining points run on bins.
    Xva/yva stay for the validation l2 metric. Returns (best, l2s): best is
    {"li","ri","l2","bi","booster"} and non-best boosters are released.
    Raises on failure."""
    results = []
    first = True
    for li, nl in enumerate(LEAVES):
        for ri, lr in enumerate(LRS):
            params = dict(FIXED_PARAMS, num_leaves=nl, learning_rate=lr)
            if seed is not None:
                params["seed"] = int(seed)
            booster = lgb.train(
                params, dtr, num_boost_round=num_boost_round,
                valid_sets=[dva], valid_names=["val"],
                callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
            )
            bi = booster.best_iteration or num_boost_round
            if first:
                # First train constructed the C++ bin container; the raw data
                # is binned now. Drop the python-side refs (fit already dropped
                # its own) so the train buffer is freed for the remaining points.
                dtr.data = None
                dtr.label = None
                first = False
            pred = booster.predict(Xva, num_iteration=bi)
            l2 = float(np.mean((pred - yva) ** 2))
            results.append({"li": li, "ri": ri, "l2": l2, "bi": int(bi), "booster": booster})
    l2s = np.array([r["l2"] for r in results])
    cands = [i for i in range(len(results)) if l2s[i] <= 1.05 * l2s.min()]
    sel = min(cands, key=lambda i: (results[i]["li"], results[i]["ri"]))
    best = results[sel]
    for r in results:
        if r is not best:
            r["booster"] = None
    return best, [float(x) for x in l2s]


def fit(context, candidate):
    """Train one booster on the trailing window and replace the state files.

    A leg that runs no learned sleeve has nothing to train: rather than burn a fit
    container on a booster no review will read, it must declare no `fit` at all, and
    this raises if that edit was forgotten.

    Row layout (the premise for the zero-copy split below): build_fit_samples
    appends per-bar blocks in ascending bar order, so the sample rows are
    (bar, symbol) major and date_idx is globally non-decreasing. Hence
    split_fit_dates' train segment (date_idx <= train_max) is exactly the row
    PREFIX [0, k_tr) and its valid segment (the last VALID_DAYS bars) exactly
    the row SUFFIX [k_va0, N); both are taken as views instead of fancy-index
    copies. The layout is asserted before slicing and a violation raises
    RuntimeError (no silent fallback to fancy indexing).

    Dataset lifecycle (F6a): the (train, valid) lgb.Dataset pair is built ONCE
    here for the whole 4-point grid (was: one pair per grid point, 4x dense
    copy/free churn). lightgbm copies the dense data into its C++ container
    lazily on the first train; free_raw_data releases that copy after binning
    and _train_grid drops dtr's python-side raw refs right after the first
    train, so the ~2.3 GiB train buffer is gone while the remaining points
    train on the binned data. The valid pair's reference relationship and all
    training parameters are exactly the former per-point construction's."""
    if not common.trains(candidate):
        raise RuntimeError(
            f"{candidate} runs no learned sleeve and trains nothing: "
            "delete fit() and the REFIT_PERIOD line from main.py"
        )
    samples = data.build_fit_samples(context)
    X, y, date_idx = samples["X"], samples["y"], samples["date_idx"]
    tr, va, n_train, n_train_dates = common.split_fit_dates(date_idx)
    if n_train < common.MIN_SAMPLES or n_train_dates < common.MIN_SAMPLES or not va.any():
        raise RuntimeError(
            f"fit has {n_train} training rows on {n_train_dates} dates and "
            f"{int(va.sum())} validation rows; need >= {common.MIN_SAMPLES} of each"
        )
    N = X.shape[0]
    # (bar, symbol) major layout => the split segments are contiguous blocks.
    if not bool(np.all(date_idx[1:] >= date_idx[:-1])):
        raise RuntimeError("fit row layout violated: date_idx is not non-decreasing")
    k_tr = n_train
    if not bool(tr[:k_tr].all()) or bool(tr[k_tr:].any()):
        raise RuntimeError("fit row layout violated: train segment is not the row prefix")
    k_va0 = N - int(va.sum())
    if not bool(va[k_va0:].all()) or bool(va[:k_va0].any()):
        raise RuntimeError("fit row layout violated: valid segment is not the row suffix")
    if k_tr > k_va0:
        raise RuntimeError("fit row layout violated: train prefix overlaps valid suffix")
    # Zero-copy views (replaces the X[tr] / X[va] fancy-index copies).
    Xva, yva = X[k_va0:], y[k_va0:]
    names = [str(n) for n in data.FEATURE_NAMES]
    dtr = lgb.Dataset(X[:k_tr], y[:k_tr], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=True)
    del X, y, samples
    best, _l2s = _train_grid(dtr, dva, Xva, yva, names)
    del dtr, dva, Xva, yva
    best["booster"].save_model(context.state_dir + "/model.txt", num_iteration=best["bi"])
    np.save(context.state_dir + "/model_meta.npy",
            np.asarray((best["li"], best["ri"], best["bi"]), dtype=np.float64))
    np.save(context.state_dir + "/feature_names.npy", np.array(names))


def score(context, X_z):
    """Percentile ranks of the booster's predictions for one cross-section (S, F).

    Loaded per call, so a refit that replaced the state files between two reviews is
    picked up without a worker restart. Returns (scores (S,) float64, order metadata)."""
    names = [str(x) for x in np.load(context.state_dir + "/feature_names.npy")]
    if names != [str(n) for n in data.FEATURE_NAMES]:
        raise RuntimeError("the fitted booster's feature names differ from data.FEATURE_NAMES")
    meta = np.load(context.state_dir + "/model_meta.npy")
    booster = lgb.Booster(model_file=context.state_dir + "/model.txt")
    pred = booster.predict(np.ascontiguousarray(X_z, dtype=np.float32))
    extra = {"num_leaves": int(LEAVES[int(meta[0])]),
             "learning_rate": float(LRS[int(meta[1])]),
             "best_iter": int(meta[2])}
    return _pct_rank(pred), extra
