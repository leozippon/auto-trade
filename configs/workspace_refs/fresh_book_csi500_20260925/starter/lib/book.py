"""How a ranked pool becomes a book -- the only thing this arm moves.

Pure functions over codes, ranks and arrays: nothing here reads a file or
touches `context`, so the round-0 census can walk exactly these rules over a
panel of offline scores (book path only, no Broker) and `lib/trade.py` turns
their answer into orders. What each leg sets is `knobs.LEGS`:

    exits       forced exits (left the section, lost the score or the bar)
                first, then holdings ranked outside SEATS * keep_band, worst
                first, at most `max_swaps` in all (None = no cap)
    entrants    the best-ranked names this money can buy that are not held,
                until the seats are full
    pull_back   (`pull`) greedy substitution toward the pool's average beta,
                size, volatility and industry mix -- no optimiser, no
                covariance in the selection
    epo         (`weights = "epo"`) entry cash of the names bought at this
                review, from a shrunk covariance of the whole target book

`c_base` runs exits and entrants with the head packs' constants and is the
head packs' book, order for order. `transfer` is a reading, not a rule: the
correlation between the book's equal-cash active weights over the pool and the
rank-normal score -- 0.68 for a freshly bought top 50 of about 283 names.
"""

import numpy as np
from scipy.stats import norm

from lib import knobs, labels

STYLES = ("beta", "logcap", "vol")


def exits(held, rank, pool_size, leg):
    """Holdings to sell at this review, forced exits first."""

    forced = [code for code in held if code not in rank]
    outside = [code for code in held
               if code in rank and rank[code] >= knobs.SEATS * leg["keep_band"]]
    outside.sort(key=lambda code: (-rank[code], code))
    available = max(0, pool_size - len(held))
    room = available if leg["max_swaps"] is None else min(leg["max_swaps"] - len(forced), available)
    return forced + outside[:max(0, room)]


def entrants(keep, held, affordable, leg):
    """Best-ranked affordable names not already held, until SEATS are filled.

    `affordable` is in rank order. A name sold at this review is still held
    when the review is decided, so it is never bought back the same day.
    """

    return [code for code in affordable if code not in held][:max(0, knobs.SEATS - len(keep))]


def exposures(panel, rows):
    """(len(rows), 3) beta, log circulating cap and daily volatility on the newest bar.

    Beta is the label's own shrunk trailing beta; volatility is the standard
    deviation of the last `knobs.VOL_DAYS` adjusted-close returns.
    """

    last = len(panel["dates"]) - 1
    beta = labels.trailing_beta(panel)[rows, last]
    closes = panel["C"][rows, -(knobs.VOL_DAYS + 1):]
    with np.errstate(invalid="ignore", divide="ignore"):
        vol = np.nanstd(closes[:, 1:] / closes[:, :-1] - 1.0, axis=1)
        logcap = np.log(panel["circ_mv"][rows, last])
    return np.column_stack([beta, logcap, vol])


def standardise(values):
    """Columns standardised over the pool; a missing value sits at the pool mean."""

    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0)
    z = (values - mean) / np.where(sd > 0, sd, 1.0)
    return np.where(np.isfinite(z), z, 0.0)


def _worst(book, z, industry, share):
    """The largest breach in tolerance units: (size, kind, key, sign), or None inside every tolerance."""

    mean = np.mean([z[code] for code in book], axis=0)
    worst = None
    for j, name in enumerate(STYLES):
        size = abs(mean[j]) / knobs.PULL_STYLE_TOL
        if size > 1 and (worst is None or size > worst[0]):
            worst = (size, "style", j, 1.0 if mean[j] > 0 else -1.0)
    counts = {}
    for code in book:
        counts[industry[code]] = counts.get(industry[code], 0) + 1
    for name, count in counts.items():
        size = (count / len(book) - share.get(name, 0.0)) / knobs.PULL_INDUSTRY_TOL
        if size > 1 and (worst is None or size > worst[0]):
            worst = (size, "industry", name, 1.0)
    return worst


def pull_back(book, reserve, rank, z, industry, share):
    """Swap names until the book sits inside the pull tolerances, or no swap helps.

    Each step takes the worst breach, drops the LOWEST-ranked book name that
    feeds it, and brings in the HIGHEST-ranked reserve name (walking down the
    ranked list) that works against it, so the score gives up as little as
    possible. `z` maps a code to its three standardised exposures, `industry`
    to its SW level-1 name, `share` an industry to its share of the pool.
    """

    book, reserve = list(book), list(reserve)
    for _ in range(knobs.SEATS):
        breach = _worst(book, z, industry, share)
        if breach is None:
            break
        _, kind, key, sign = breach
        if kind == "style":
            feeds = [code for code in book if sign * z[code][key] > 0]
            fights = [code for code in reserve if sign * z[code][key] < 0]
        else:
            feeds = [code for code in book if industry[code] == key]
            fights = [code for code in reserve if industry[code] != key]
        if not feeds or not fights:
            break
        out = max(feeds, key=lambda code: rank[code])
        book[book.index(out)] = fights[0]
        reserve.remove(fights[0])
    return book


def deviation(book, z, industry, share):
    """The book's mean standardised exposures and its largest industry overweight."""

    mean = np.mean([z[code] for code in book], axis=0)
    counts = {}
    for code in book:
        counts[industry[code]] = counts.get(industry[code], 0) + 1
    over = max(count / len(book) - share.get(name, 0.0) for name, count in counts.items())
    return {**{name: round(float(mean[j]), 3) for j, name in enumerate(STYLES)},
            "industry_over": round(float(over), 3)}


def epo(returns, signal):
    """Long-only EPO weights, summing to 1, of the names whose daily returns are the columns.

    Pedersen, Babu and Levine's enhanced portfolio optimisation in its simple
    form: w ~ inverse(shrunk covariance) x signal, the correlations shrunk
    toward zero by `knobs.EPO_SHRINK`; negative weights are dropped and one
    name is capped at `knobs.EPO_CAP` times an equal share. No benchmark
    anchor and no tracking-error target.
    """

    returns = np.where(np.isfinite(returns), returns, 0.0)
    covariance = np.cov(returns, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1.0 - knobs.EPO_SHRINK) * covariance + knobs.EPO_SHRINK * diagonal
    raw = np.clip(np.linalg.solve(shrunk + 1e-12 * np.eye(len(signal)), signal), 0.0, None)
    if raw.sum() <= 0:
        return np.full(len(signal), 1.0 / len(signal))
    weights = raw / raw.sum()
    cap = knobs.EPO_CAP / len(signal)
    for _ in range(len(signal)):
        over = weights > cap
        if not over.any():
            break
        spare = (weights[over] - cap).sum()
        weights[over] = cap
        free = ~over & (weights > 0)
        if not free.any():
            break
        weights[free] += spare * weights[free] / weights[free].sum()
    return weights


def transfer(book, codes, pct):
    """Correlation of the equal-cash book's active weights with the rank-normal score over the pool."""

    held = np.isin(np.asarray(codes), list(book)).astype(np.float64)
    if held.all() or not held.any():
        return float("nan")
    return float(np.corrcoef(held, norm.ppf(np.asarray(pct, dtype=np.float64)))[0, 1])
