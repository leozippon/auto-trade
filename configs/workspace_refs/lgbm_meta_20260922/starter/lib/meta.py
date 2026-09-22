"""The second stage: a classifier over the primary's own out-of-fold picks.

The graduated lineage uses its primary score directly -- rank, take the top N,
equal cash. Everything the score knows about how RELIABLE a given pick is, it
has no way to say. Meta-labelling asks that second question with a second model:

    primary       ranks the constituents on the residual N-day label
    out-of-fold   purged contiguous folds give every training bar a primary
                  score from a booster that never saw that bar
    picks         the top `TOP_SHARE` of each out-of-fold bar -- the names the
                  book would actually have bought
    meta          a binary LightGBM over `feature_names(...)`: did this pick
                  beat that bar's cross-sectional median residual return
                  (lib/labels.py)
    use           filter, or size, per `main.CANDIDATE` (lib/trade.py)

Why these features and no others. They are the state a pick sits in, not a
second alpha: where it sits in the day's score distribution, what kind of name
it is (volatility, beta, turnover, size, distance to its own 60-day high), where
it ranks inside its own industry, and what the market was doing. A meta stage
fed new return predictors would be a second alpha wearing a gate's name, and
`families.md` forbids that.

Every feature is scale-free within its bar -- percentile ranks, z-scores, and
one robust gap `(score - median) / IQR` -- because the classifier is TRAINED on
out-of-fold boosters' raw scores and DEPLOYED on a different booster's. That
handles the scale; it does not handle the other, registered asymmetry: the
deployed primary saw its whole window, so its score distribution is better
separated than any out-of-fold one. There is no way around it inside one `fit`,
so it is the first thing the incremental-information probe has to survive.

Persistence. The meta booster is continued exactly as the primary is, and reset
cold on the same refits (`lib/state.py`).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import common, data, labels, primary

TOP_SHARE = 0.10             # the primary's top decile of a bar is what the meta stage judges
MIN_PICKS = 5
MIN_BAR_NAMES = 50
BASE_FEATURES = ("score_rank", "score_gap", "vol20", "beta", "turn20", "size_rank",
                 "high60", "ind_rank", "mkt_ret20", "mkt_vol20", "disp_ret20")
STACK_FEATURE = "twin_rank"
PARAMS = {
    "objective": "binary",
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 10,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "num_threads": 8,
    "seed": 11,
    "verbose": -1,
}
COLD_ROUNDS = 300
WARM_ROUNDS = 80
EARLY_STOP = 50
TWIN_HOLD = 20               # the stacking variant's horizon twin: same family, different horizon


def feature_names(candidate):
    return list(BASE_FEATURES) + ([STACK_FEATURE] if candidate == "m_stack" else [])


def model_path(context):
    return context.state_dir + "/meta_model.txt"


def names_path(context):
    return context.state_dir + "/meta_names.npy"


def load(context):
    try:
        return lgb.Booster(model_file=model_path(context))
    except (FileNotFoundError, OSError, lgb.basic.LightGBMError):
        return None


def _roll(matrix, window, how):
    rolling = pd.DataFrame(matrix.T).rolling(window, min_periods=max(5, window // 2))
    return getattr(rolling, how)().to_numpy().T


def state_columns(panel):
    """(names, dates) and (dates,) state matrices the meta features index into."""

    closes = panel["C"]
    total = closes.shape[1]
    returns = pd.DataFrame(closes.T).pct_change(fill_method=None).to_numpy().T
    ret20 = np.full_like(closes, np.nan)
    market = np.full(total, np.nan)
    if total > 20:
        with np.errstate(divide="ignore", invalid="ignore"):
            ret20[:, 20:] = closes[:, 20:] / closes[:, :total - 20] - 1.0
            market[20:] = panel["bench_close"][20:] / panel["bench_close"][:total - 20] - 1.0
    bench_return = pd.Series(panel["bench_close"]).pct_change(fill_method=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        high60 = closes / _roll(closes, 60, "max")
    dispersion = pd.DataFrame(np.where(panel["member"], ret20, np.nan)).std(axis=0, skipna=True).to_numpy()
    return {
        "vol20": _roll(returns, 20, "std"),
        "turn20": _roll(panel["turn_f"], 20, "mean"),
        "size": np.log(np.clip(panel["circ_mv"], 1.0, None)),
        "high60": high60,
        "beta": labels.trailing_beta(panel),
        "mkt_ret20": market,
        "mkt_vol20": bench_return.rolling(20, min_periods=10).std().to_numpy() * np.sqrt(244.0),
        "disp_ret20": dispersion,
    }


def _z(values):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros(len(values))
    centre = values[finite].mean()
    spread = values[finite].std()
    return np.nan_to_num((values - centre) / (spread if spread > 0 else 1.0), nan=0.0,
                         posinf=0.0, neginf=0.0)


def _rank(values):
    return np.nan_to_num(pd.Series(values).rank(pct=True).to_numpy(), nan=0.5)


def _gap(values):
    """Robust, booster-independent distance above the bar's own middle."""

    finite = np.isfinite(values)
    if finite.sum() < 4:
        return np.zeros(len(values))
    low, middle, high = np.percentile(values[finite], [25, 50, 75])
    spread = high - low
    out = (values - middle) / (spread if spread > 0 else 1.0)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)


def bar_features(panel, columns, bar, names, score, twin=None):
    """(len(names), len(feature_names)) meta features for ONE bar's whole scorable cross-section.

    Always called with the full cross-section, never with a pre-selected subset:
    every column is relative to the bar, so selecting first would measure the
    picks against each other instead of against the day.
    """

    industry = pd.Series(panel["industry"][names])
    block = [
        _rank(score),
        _gap(score),
        _z(columns["vol20"][names, bar]),
        _z(columns["beta"][names, bar]),
        _z(columns["turn20"][names, bar]),
        _rank(columns["size"][names, bar]),
        np.nan_to_num(columns["high60"][names, bar], nan=1.0),
        np.nan_to_num(pd.Series(score).groupby(industry).rank(pct=True).to_numpy(), nan=0.5),
        np.full(len(names), float(np.nan_to_num(columns["mkt_ret20"][bar]))),
        np.full(len(names), float(np.nan_to_num(columns["mkt_vol20"][bar]))),
        np.full(len(names), float(np.nan_to_num(columns["disp_ret20"][bar]))),
    ]
    if twin is not None:
        block.append(_rank(twin))
    return np.stack(block, axis=1).astype(np.float32)


def training_rows(panel, columns, samples, oof, outcome, twin_oof=None):
    """Meta rows over the primary's out-of-fold top-decile picks, one block per bar."""

    order = np.argsort(samples["bar"], kind="stable")
    bar_axis = samples["bar"][order]
    starts = np.searchsorted(bar_axis, np.unique(bar_axis), side="left").tolist() + [len(order)]
    rows, targets, bars = [], [], []
    for start, stop in zip(starts[:-1], starts[1:]):
        chunk = order[start:stop]
        bar = int(samples["bar"][chunk[0]])
        names = samples["name"][chunk]
        score = oof[chunk]
        target = outcome[names, bar]
        usable = np.isfinite(score) & np.isfinite(target)
        if usable.sum() < MIN_BAR_NAMES:
            continue
        names, score, target = names[usable], score[usable], target[usable]
        twin = twin_oof[chunk][usable] if twin_oof is not None else None
        block = bar_features(panel, columns, bar, names, score, twin)
        take = max(MIN_PICKS, int(round(TOP_SHARE * len(score))))
        picks = np.argsort(-score, kind="stable")[:take]
        rows.append(block[picks])
        targets.append(target[picks].astype(np.float32))
        bars.append(np.full(len(picks), bar, dtype=np.int64))
    if not rows:
        raise RuntimeError("the out-of-fold pass produced no meta training rows")
    return np.vstack(rows), np.concatenate(targets), np.concatenate(bars)


def fit(context, cold, candidate):
    """Train the primary (and, for the stacking variant, its horizon twin) and the meta stage."""

    panel = data.wide(context, data.FIT_CALENDAR_DAYS)
    residual = labels.residual(panel, data.HOLD)
    target, realized = labels.rank_target(residual)
    samples = data.fit_samples(panel, target, realized)
    reading = primary.fit(context, samples, cold, "h10")
    twin_samples = None
    if candidate == "m_stack":
        twin_target, _ = labels.rank_target(labels.residual(panel, TWIN_HOLD))
        twin_y = twin_target[samples["name"], samples["bar"]].astype(np.float32)
        twin_samples = dict(samples)
        twin_samples["y"] = np.where(np.isfinite(twin_y), twin_y, 0.0)
        primary.fit(context, twin_samples, cold, "h20")
    if candidate == "c_primary":
        return reading
    columns = state_columns(panel)
    outcome = labels.beats_median(residual)
    oof = primary.out_of_fold(samples)
    twin_oof = primary.out_of_fold(twin_samples) if twin_samples is not None else None
    x_meta, y_meta, bars = training_rows(panel, columns, samples, oof, outcome, twin_oof)
    train_mask, valid_mask, rows, _ = common.split_fit_dates(bars)
    if rows < common.MIN_SAMPLES or not valid_mask.any():
        raise RuntimeError(f"the meta stage has {rows} training rows; need >= {common.MIN_SAMPLES}")
    names = feature_names(candidate)
    previous = None if cold else load(context)
    train_set = lgb.Dataset(x_meta[train_mask], y_meta[train_mask], feature_name=names, free_raw_data=True)
    valid_set = lgb.Dataset(x_meta[valid_mask], y_meta[valid_mask], reference=train_set, free_raw_data=True)
    booster = lgb.train(
        PARAMS, train_set, num_boost_round=COLD_ROUNDS if previous is None else WARM_ROUNDS,
        valid_sets=[valid_set], valid_names=["valid"], init_model=previous,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    booster.save_model(model_path(context))
    np.save(names_path(context), np.array(names))
    return reading


def decision(context, panel, candidate):
    """(raw primary score, meta probability) over every name; NaN off the scorable cross-section."""

    score = primary.score(context, panel, "h10")
    probability = np.full(len(score), np.nan)
    if candidate == "c_primary":
        return score, probability
    booster = load(context)
    if booster is None:
        raise RuntimeError("no fitted meta booster under the state directory")
    stored = [str(name) for name in np.load(names_path(context))]
    if stored != feature_names(candidate):
        raise RuntimeError("the fitted meta booster's features differ from feature_names(candidate)")
    names = np.nonzero(np.isfinite(score))[0]
    if len(names) < MIN_PICKS:
        return score, probability
    bar = len(panel["dates"]) - 1
    twin = None
    if candidate == "m_stack":
        twin = primary.score(context, panel, "h20")[names]
    rows = bar_features(panel, state_columns(panel), bar, names, score[names], twin)
    probability[names] = booster.predict(rows)
    return score, probability
