"""The flipped census composite and its legs, chosen by `knobs.CANDIDATE`.

Each decision day, every census score of the leg (lib/census.py) is turned
into a cross-sectional percentile rank over the day's pool, after multiplying
it by the leg's sign: -1 flips it, so the names the census arm would have
bought last rank first. The leg's value is the mean of those ranks
(`knobs.WEIGHTING`: over all its scores, or over census families first); a
name needs at least MIN_SHARE of the leg's scores to have one. Higher is
bought first.

    leg      scores                           sign
    f_all    every family of census.FAMILIES  -1   main candidate
    f_vol    the volume / turnover family     -1   secondary
    f_beta   the beta family                  -1   secondary
    c_orig   every family                     +1   control: the census direction
    c_shuf   every family                     -1   control: trade.py permutes the values

What `trade.run` relies on: score(context, codes) -> (values, meta), with
values a float Series over `codes` (NaN = not scored) and meta a small
JSON-safe dict; shuffle(values, decision_at); describe(values, book); NAME.
"""

import numpy as np
import pandas as pd

from lib import census, knobs

NAME = knobs.CANDIDATE
EVERY = tuple(census.FAMILIES)
LEGS = {
    "f_all": (EVERY, -1.0),
    "f_vol": (("volume",), -1.0),
    "f_beta": (("beta",), -1.0),
    "c_orig": (EVERY, 1.0),
    "c_shuf": (EVERY, -1.0),
}
# A name needs at least this share of the leg's scores.
MIN_SHARE = 0.5
# A score covering fewer of the pool's names than this is a broken read.
MIN_SCORED = 100


def score(context, codes):
    families, sign = LEGS[knobs.CANDIDATE]
    names = [name for family in families for name in census.FAMILIES[family]]
    codes = list(codes)
    ranks = {}
    for name, values in census.compute(context, names).items():
        values = values.reindex(codes)
        scored = int(values.notna().sum())
        if scored < MIN_SCORED:
            raise RuntimeError(f"census score {name} covers {scored} of {len(codes)} pool names")
        ranks[name] = (sign * values).rank(pct=True)
    table = pd.DataFrame(ranks, index=codes)
    if knobs.WEIGHTING == "equal":
        composite = table.mean(axis=1)
    elif knobs.WEIGHTING == "family":
        composite = pd.DataFrame(
            {family: table[list(census.FAMILIES[family])].mean(axis=1) for family in families}
        ).mean(axis=1)
    else:
        raise ValueError(f"knobs.WEIGHTING must be 'equal' or 'family', got {knobs.WEIGHTING!r}")
    composite = composite.where(table.notna().sum(axis=1) >= MIN_SHARE * len(names))
    return composite, {"scores": len(names), "composite_scored": int(composite.notna().sum())}


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    picked = values.reindex(book).dropna()
    return {"book_score_span": [round(float(picked.min()), 4), round(float(picked.max()), 4)] if len(picked) else None}
