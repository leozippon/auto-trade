"""Units, one refit's training set, and the scores of the units fresh at a review.

Unit. One row per (name, review): the distinct normalised texts of the name's
titles that became fresh at that review (`lib/clock.py`, `lib/titles.py`),
sorted, at most MAX_TITLES of them (a report season files dozens; 95 % of
units hold 18 or fewer). The book acts on reviews, so this is the row it
ranks: every title fresh for a name at one review shares one entry and one
label, and a row per title or per day would repeat that label and weigh a
name by how many documents it files. Each text names one stock; a document
filed for several names is one row per name in the vendor's index.

Label (`lib/label.py`): the unit's size-matched return over the book's
holding, ranked across the units of its review (0 mean, -0.5..0.5). A refit
uses the units of the trailing TRAIN_YEARS whose label is a visible bar; the
text history starts in 2016-01, so the first research year's refits see less.
The last VALID_REVIEWS labelled reviews are the validation segment, and the
training segment ends HOLD trading days before it, so no training label
overlaps a validation entry. Each model chooses its one search setting on the
validation segment (`lib/models.py`); the book's threshold is the (1 - TOP)
quantile of the chosen model's validation scores.

Controls. `knobs.LABELS = "shuffled"` permutes every label across the units
of its review (seeded by `knobs.SEED` and the review), training and
validation alike, before anything is fitted. `knobs.TEXT = "blind"` replaces a
unit's texts by one descriptor naming its float-cap decile at T-1, its board
and its number of distinct documents (9 or more as 9): the same unit, label,
model and book without the words.
"""

import numpy as np
import pandas as pd

from lib import clock, knobs, label, models, titles

TRAIN_YEARS = 3
VALID_REVIEWS = 13
MIN_TRAIN_REVIEWS = 26
MAX_TITLES = 16
TOP = 0.01
# Calendar days of titles read before the first review a set needs, more than
# the longest gap between two reviews.
LOOKBACK_DAYS = 21
DECILES = 10
BOARDS = {"60": "沪", "00": "深", "30": "创"}


def units(docs, days, review_positions):
    """(entry, ts_code, texts, documents): one row per name and review, entry a position in `days`."""

    entry, _ = clock.entries(days, review_positions, docs["day"].to_numpy(dtype=object))
    docs = docs.assign(entry=entry)[entry >= 0].drop_duplicates(["entry", "ts_code", "text"])
    docs = docs.sort_values(["entry", "ts_code", "text"], kind="stable")
    grouped = docs.groupby(["entry", "ts_code"], sort=True)
    out = grouped["text"].size().rename("documents").to_frame()
    out["texts"] = docs.groupby(["entry", "ts_code"], sort=True).head(MAX_TITLES).groupby(
        ["entry", "ts_code"], sort=True)["text"].agg(list)
    return out.reset_index()


def blind(frame, ranks):
    """One descriptor per unit: float-cap decile, board, documents. `ranks` is each unit's cap rank."""

    decile = np.where(np.isfinite(ranks), np.minimum((np.nan_to_num(ranks) * DECILES).astype(int), DECILES - 1), -1)
    board = frame["ts_code"].str[:2].map(BOARDS).to_numpy()
    documents = np.minimum(frame["documents"].to_numpy(), 9)
    return [[f"规{d}板{b}件{n}"] for d, b, n in zip(decile, board, documents)]


def train(context):
    """Fit the leg's model on the trailing window and save it; a small record of the fit."""

    today = clock.decision(context)
    first = (today - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y%m%d")
    start = (pd.Timestamp(first) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    days = clock.calendar(context, start)
    data = label.panel(context, days)
    frame = units(titles.documents(context, start), days, clock.reviews(days))
    last_bar = len(days) - 2
    frame = frame[(frame["entry"] + knobs.HOLD <= last_bar) & (days[frame["entry"].to_numpy()] >= first)]

    code = pd.Index(data["codes"]).get_indexer(frame["ts_code"])
    raw = np.full(len(frame), np.nan)
    ranks = np.full(len(frame), np.nan)
    for t in np.unique(frame["entry"]):
        rows = np.nonzero((frame["entry"].to_numpy() == t) & (code >= 0))[0]
        raw[rows] = label.abnormal(data, t)[code[rows]]
        ranks[rows] = label.size_rank(data["mv"][t - 1])[code[rows]]
    keep = np.isfinite(raw)
    frame, raw, ranks = frame[keep].reset_index(drop=True), raw[keep], ranks[keep]
    y = pd.Series(raw).groupby(frame["entry"].to_numpy()).rank().to_numpy()
    sizes = frame.groupby("entry")["entry"].transform("size").to_numpy()
    y = (y - 0.5) / sizes - 0.5
    if knobs.LABELS == "shuffled":
        for t in np.unique(frame["entry"]):
            rows = np.nonzero(frame["entry"].to_numpy() == t)[0]
            y[rows] = y[rows][np.random.default_rng([knobs.SEED, int(days[t])]).permutation(len(rows))]
    inputs = blind(frame, ranks) if knobs.TEXT == "blind" else list(frame["texts"])

    reviews = np.unique(frame["entry"])
    valid = reviews[-VALID_REVIEWS:]
    trained = reviews[reviews + knobs.HOLD <= valid[0]]
    if len(trained) < MIN_TRAIN_REVIEWS:
        raise RuntimeError(f"fit has {len(trained)} training reviews in its window, fewer than {MIN_TRAIN_REVIEWS}")
    segments = {}
    for name, chosen in (("train", trained), ("valid", valid)):
        rows = np.nonzero(np.isin(frame["entry"].to_numpy(), chosen))[0]
        segments[name] = {"inputs": [inputs[i] for i in rows], "y": y[rows], "groups": frame["entry"].to_numpy()[rows]}
    params, scores = models.fit(segments["train"], segments["valid"], int(today.strftime("%Y%m%d")))
    threshold = float(np.quantile(scores, 1.0 - TOP))
    models.save(context, params, threshold)
    return {
        "train_units": len(segments["train"]["y"]),
        "valid_units": len(segments["valid"]["y"]),
        "train_reviews": len(trained),
        "setting": float(params.get("alpha", params.get("epochs", 0))),
        "threshold": threshold,
    }


def fresh(context):
    """(ts_code, score) of the units fresh at today's review that score at or above the
    threshold of the last refit, and how many units were scored."""

    today = clock.decision(context)
    start = (today - pd.Timedelta(days=2 * LOOKBACK_DAYS)).strftime("%Y%m%d")
    days = clock.calendar(context, start)
    frame = units(titles.documents(context, start), days, clock.reviews(days))
    frame = frame[frame["entry"] == len(days) - 1].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame({"ts_code": pd.Series(dtype=object), "score": pd.Series(dtype=float)}), 0
    if knobs.TEXT == "blind":
        bars = pd.read_parquet(context.asof_dir + "/daily", columns=["ts_code", "trade_date", "circ_mv"],
                               filters=[("trade_date", "==", days[-2])])
        bars = bars[bars["ts_code"].astype(str).str.match(titles.STOCK)]
        rank = pd.Series(label.size_rank(bars["circ_mv"].to_numpy(dtype=float)), index=bars["ts_code"].astype(str))
        inputs = blind(frame, rank.reindex(frame["ts_code"]).to_numpy())
    else:
        inputs = list(frame["texts"])
    scores, threshold = models.predict(context, inputs)
    out = pd.DataFrame({"ts_code": frame["ts_code"].to_numpy(), "score": scores})
    return out[out["score"] >= threshold].reset_index(drop=True), len(frame)
