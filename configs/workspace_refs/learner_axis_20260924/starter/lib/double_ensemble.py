"""DoubleEnsemble over LightGBM: loss-curve sample reweighting and shuffle-based feature selection.

Source. A port of Qlib's `DEnsembleModel` (`qlib/contrib/model/double_ensemble.py`,
Microsoft, MIT licence, vendored in this repository under `external_references/qlib/`)
of the algorithm in Zhang, Li, Li, Wang, Liu, Chen, Yang and Liu, "DoubleEnsemble: A New
Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data
Analysis", ICDM 2020 (arXiv:2010.01265), Algorithms 1-3. `references/double-ensemble.md`
is the one-page reading of both. Hyperparameters are `knobs.DOUBLE_ENSEMBLE`.

The algorithm, as ported:

1. Sub-model 1 sees every feature with equal weights -- the carrier's own cold fit.
2. After sub-model k < K, on the TRAINING rows:
   - its loss curve: every row's loss after each of its trees, each tree's column
     replaced by its rank across rows; l_start / l_end = the row's mean rank over the
     first / last 10 % of the trees;
   - the loss of the current ensemble (the mean of sub-models 1..k);
   - SR: h = alpha1 * rank(-ensemble loss) + alpha2 * rank(l_end / l_start); rows cut
     into `bins_sr` equal-width bins of h; a row's weight is
     1 / (decay^k * mean h of its bin + 0.1). Easy rows (low loss) and noisy rows (loss
     that did not fall while the others' did) are down-weighted, hard rows up-weighted;
   - FS: for every feature, shuffle its column across the training rows, re-predict the
     ensemble, and take g = mean(dL) / std(dL) of the per-row loss increase; features
     cut into `bins_fs` equal-width bins of g, and ceil(ratio * bin size) drawn at
     random from each non-empty bin, the most important bin with the largest ratio.
3. Sub-model k + 1 trains on those weights and that feature subset.
4. The prediction is the plain mean of the K sub-models.

Where this port departs from Qlib's code. None of it changes the algorithm:

- the loss curve is evaluated only at the trees SR reads (first and last 10 %), one
  column at a time, instead of materialising an N x T frame;
- Qlib's reported configuration trains every sub-model a fixed number of rounds.
  Here sub-model 1 early-stops exactly like the carrier and sub-models 2..K train the
  number of trees it kept, without early stopping: they optimise a REWEIGHTED loss,
  and stopping them on the unweighted validation loss ends them before the
  reweighting acts (measured: 3-4 trees against the first model's 49, `sources.md`).
  The loss curve covers the trees a sub-model keeps;
- the shuffle pass re-predicts only the sub-models that contain the shuffled feature;
  the others' predictions cannot change, so reusing them is exact;
- randomness comes from one seeded generator, so a replay is reproducible;
- sub-model 1 gets no weight vector (Qlib passes ones), which makes it the very booster
  `c_cold` fits;
- Qlib's code raises for any loss but MSE. A ranking objective has no per-row loss, so
  for `lambdarank` the per-row loss is the squared difference between the per-bar
  z-score of the score and the per-bar z-score of the rank target, and the ensemble
  averages per-bar z-scores instead of raw scores (lambdarank score scales are
  arbitrary per sub-model). This is the pack's registered adaptation for `de_lr`.

Rows are grouped by bar in ascending order, which is how `data.fit_samples` stacks
them and what a LightGBM query array needs anyway.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CURVE_SHARE = 0.1   # the paper's "first and last 10 %" of a loss curve
WEIGHT_FLOOR = 0.1  # the "+ 0.1" of the weight formula
G_EPSILON = 1e-7    # Qlib's guard on the std of a g-value


def starts_of(bars):
    """Row offsets at which each bar's contiguous run of rows begins."""

    bars = np.asarray(bars)
    if bars.size == 0:
        raise ValueError("no rows")
    return np.flatnonzero(np.r_[True, bars[1:] != bars[:-1]])


def standardize(values, starts):
    """Per-bar z-score of a vector whose rows are grouped by bar; a flat bar maps to 0."""

    values = np.asarray(values, dtype=np.float64)
    counts = np.diff(np.r_[starts, values.size])
    centered = values - np.repeat(np.add.reduceat(values, starts) / counts, counts)
    scale = np.sqrt(np.add.reduceat(centered * centered, starts) / counts)
    return centered / np.repeat(np.where(scale > 0.0, scale, 1.0), counts)


def combine(predictions, starts, objective):
    """The ensemble score: the plain mean, of per-bar z-scores for a ranking objective."""

    if objective == "regression":
        return np.mean(predictions, axis=0)
    return np.mean([standardize(prediction, starts) for prediction in predictions], axis=0)


def loss_function(target, starts, objective):
    """Per-row loss of a score against the continuous rank target."""

    target = np.asarray(target, dtype=np.float64)
    if objective == "regression":
        return lambda prediction: (target - prediction) ** 2
    if objective == "lambdarank":
        standard = standardize(target, starts)
        return lambda prediction: (standardize(prediction, starts) - standard) ** 2
    raise ValueError(f"no per-row loss for objective {objective!r}")


def kept_trees(booster):
    """Trees the booster predicts with: its best iteration when it early-stopped, else all."""

    return booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()


def curve_ends(booster, features, loss_of):
    """(l_start, l_end): each row's mean per-tree loss rank over the first and last 10 % of trees."""

    trees = kept_trees(booster)
    part = max(int(trees * CURVE_SHARE), 1)
    rows = features.shape[0]

    def walk(first, last, prediction):
        total = np.zeros(rows)
        for tree in range(first, last):
            prediction = prediction + booster.predict(features, start_iteration=tree, num_iteration=1)
            total += rankdata(loss_of(prediction)) / rows
        return total / (last - first)

    head = walk(0, part, np.zeros(rows))
    tail_from = trees - part
    # num_iteration <= 0 means "every tree" to LightGBM, so an empty prefix is spelled out.
    prefix = booster.predict(features, num_iteration=tail_from) if tail_from > 0 else np.zeros(rows)
    return head, walk(tail_from, trees, prefix)


def sample_weights(ensemble_loss, l_start, l_end, trained, config):
    """SR (Algorithm 2): one weight per training row after `trained` sub-models."""

    h1 = pd.Series(-ensemble_loss).rank(pct=True).to_numpy()
    h2 = pd.Series(l_end / l_start).rank(pct=True).to_numpy()
    h = config["alpha1"] * h1 + config["alpha2"] * h2
    bins = np.asarray(pd.cut(h, config["bins_sr"], labels=False), dtype=np.int64)
    sizes = np.bincount(bins, minlength=config["bins_sr"])
    means = np.bincount(bins, weights=h, minlength=config["bins_sr"]) / np.maximum(sizes, 1)
    return 1.0 / (config["decay"] ** trained * means[bins] + WEIGHT_FLOOR)


def select_features(models, subsets, features, predictions, base_loss, loss_of, starts,
                    objective, config, rng):
    """FS (Algorithm 3): column indices for the next sub-model, drawn from ALL features."""

    rows, width = features.shape
    views = [np.ascontiguousarray(features[:, subset]) for subset in subsets]
    g = np.zeros(width)
    for column in range(width):
        order = rng.permutation(rows)
        shuffled = list(predictions)
        for which, (booster, subset) in enumerate(zip(models, subsets)):
            position = np.flatnonzero(subset == column)
            if position.size == 0:
                continue
            view = views[which]
            original = view[:, position[0]].copy()
            view[:, position[0]] = original[order]
            shuffled[which] = booster.predict(view)
            view[:, position[0]] = original
        increase = loss_of(combine(shuffled, starts, objective)) - base_loss
        g[column] = increase.mean() / (increase.std() + G_EPSILON)
    g = np.nan_to_num(g, nan=0.0)
    bins = np.asarray(pd.cut(g, config["bins_fs"], labels=False), dtype=np.int64)
    chosen = []
    for rank, which in enumerate(sorted(np.unique(bins), reverse=True)):
        members = np.flatnonzero(bins == which)
        take = int(np.ceil(config["sample_ratios"][rank] * members.size))
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())
    return np.array(sorted(set(chosen)), dtype=np.int64)


def fit(train_submodel, features, target, bars, objective, config, seed):
    """Algorithm 1. Returns [(booster, column indices)] for the K sub-models.

    `train_submodel(columns, weights, trees)` trains one booster on the training
    rows restricted to `columns` with per-row `weights` (None = equal) -- early
    stopping when `trees` is None, exactly `trees` trees otherwise -- and returns it;
    `features`, `target` and `bars` are those same training rows.
    """

    if len(config["sample_ratios"]) != config["bins_fs"]:
        raise ValueError("DOUBLE_ENSEMBLE: one sample ratio per feature bin")
    starts = starts_of(bars)
    loss_of = loss_function(target, starts, objective)
    rng = np.random.default_rng(seed)
    columns = np.arange(features.shape[1], dtype=np.int64)
    weights, trees = None, None
    models, subsets, predictions = [], [], []
    for trained in range(1, config["models"] + 1):
        booster = train_submodel(columns, weights, trees)
        trees = trees or kept_trees(booster)
        models.append(booster)
        subsets.append(columns)
        if trained == config["models"]:
            break
        view = np.ascontiguousarray(features[:, columns])
        predictions.append(booster.predict(view))
        base_loss = loss_of(combine(predictions, starts, objective))
        l_start, l_end = curve_ends(booster, view, loss_of)
        del view
        weights = sample_weights(base_loss, l_start, l_end, trained, config)
        columns = select_features(models, subsets, features, predictions, base_loss, loss_of,
                                  starts, objective, config, rng)
    return list(zip(models, subsets))


def predict(members, features, bars, objective):
    """The ensemble score of rows grouped by bar (one bar at a decision)."""

    predictions = [booster.predict(np.ascontiguousarray(features[:, subset]))
                   for booster, subset in members]
    return combine(predictions, starts_of(bars), objective)
