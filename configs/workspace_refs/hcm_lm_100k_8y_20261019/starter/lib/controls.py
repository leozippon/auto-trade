"""The standard controls a nomination reports (open-rules §3), as constructions a book calls.

Each one changes exactly one thing about the candidate and leaves the rest of
the book -- seats, sizing, hold, review, exits -- to the candidate's own code,
so the control and the candidate differ by that one thing only:

    late_entry(events, calendar, k)        event books: the same names, entered
                                           k trading days after the event
    shuffled_assignment(events, eligible)  event books, and screens that pick
                                           the whole list afresh at every
                                           review: the same days and counts,
                                           names drawn from the eligible pool
    held_shuffle(values, eligible)         screens that keep names (a keep band,
                                           a slow review): the same scores on
                                           stand-in names fixed across reviews,
                                           so each is held as long as the name
                                           it stands for
    random_skip(candidates, n_skip, key)   exclusion layers: as many names
                                           skipped, at random instead of by rule
    static_version(adaptive, value)        adaptive rules: the same call, one
                                           fixed parameter declared beforehand
    most_common(choices)                   that parameter: the adaptive rule's
                                           most frequent choice on the probe years
    turnover_per_year(trades, cash, days)  the check every control passes: its
                                           turnover within 25 % of the
                                           candidate's, or it differs in more
                                           than the one thing

`events` is a frame with at least `ts_code` and `day` (YYYYMMDD strings), one
row per name and event day; any other column rides along unchanged. A late or
reassigned event is still an event of the book: read the event window far
enough back to cover k plus the hold plus any quiet period, as for the
candidate. Applied to the events, `shuffled_assignment` keeps the hold: the
drawn name is held for the event's window. Applied to a screen's picks at
every review it redraws the whole book each time, so a screen that would have
kept its names churns several times over (round 20261015: 14.4 against 3.6 a
year); such a screen takes `held_shuffle`.

Every random draw is a fixed hash of (code, day or key, salt), never a seeded
generator over a list: a name's place in the draw does not move when the pool
around it gains or loses a name, so the drawn set changes only by the names
that entered or left, and a cold worker, a warm one and a later review that
recomputes an old event draw alike. The draws are a function of the inputs
alone and reproduce across processes.
"""

from collections import Counter

import numpy as np
import pandas as pd


def late_entry(events, calendar, k):
    """`events` with each `day` moved to the k-th trading day after the first
    trading day on or after it, on the sorted YYYYMMDD `calendar` (the visible
    trading days, optionally plus today). Events whose late day lies past the
    calendar's end are dropped: they are not due yet and come back at the
    decision that reaches them."""

    if int(k) < 1:
        raise ValueError(f"a late-entry control needs k >= 1 trading day, got {k!r}")
    days = np.asarray(calendar, dtype=object)
    if len(days) == 0 or (days[1:] < days[:-1]).any():
        raise ValueError("calendar must be a non-empty sorted sequence of YYYYMMDD strings")
    position = np.searchsorted(days, events["day"].astype(str).to_numpy(dtype=object), side="left") + int(k)
    due = position < len(days)
    late = events[due].copy()
    late["day"] = days[position[due]]
    return late.reset_index(drop=True)


def shuffled_assignment(events, eligible, salt=0):
    """`events` with each row's `ts_code` replaced by a name drawn from
    `eligible` (the codes the book could buy, e.g. the decision's tradable
    pool): on every event day as many distinct names as that day has rows,
    without replacement, by a fixed hash of (code, day, salt). Days and counts
    are the candidate's; only which names get the signal is random."""

    pool = sorted({str(code) for code in eligible})
    out = events.copy().reset_index(drop=True)
    days = out["day"].astype(str)
    for day, rows in out.groupby(days, sort=True).groups.items():
        if len(rows) > len(pool):
            raise ValueError(f"{len(rows)} events on {day} but only {len(pool)} eligible names")
        drawn = [pool[i] for i in _order(pool, f"{day}|{salt}")[: len(rows)]]
        out.loc[rows, "ts_code"] = drawn
    return out


def held_shuffle(values, eligible, salt=0):
    """`values` (scores indexed by code, NaN not scored, every scored code in
    `eligible`) moved onto the names of `eligible` by a relabeling that does
    not depend on the day: the eligible names sorted by a fixed hash of (code,
    salt), each takes the score of the next name in that order, the last the
    first's; the names that take no score are dropped. A book with a keep band
    run on the result holds each stand-in exactly as long as the candidate
    holds the name it stands for, so the control keeps the candidate's holding
    period and turnover and changes only which names carry the scores. The
    stand-in is fixed per name rather than per holding: a name the candidate
    holds twice is stood in for by the same name twice. A name entering or
    leaving `eligible` re-partners only its neighbour in the order."""

    pool = sorted({str(code) for code in eligible})
    scored = values.dropna()
    scored.index = scored.index.astype(str)
    outside = sorted(set(scored.index) - set(pool))
    if outside:
        raise ValueError(f"{len(outside)} scored names are not eligible, first {outside[0]}")
    order = _order(pool, f"held|{salt}")
    scores = scored.reindex(pool).to_numpy(dtype=float)
    moved = np.empty(len(pool), dtype=float)
    moved[order] = scores[np.roll(order, -1)]
    return pd.Series(moved, index=pool).dropna()


def turnover_per_year(trades, initial_cash, trading_days):
    """Filled notional, buys and sells alike (|price x quantity|), over
    `initial_cash`, per 244 trading days: the host's span `turnover` on a
    yearly scale, which a batch row gives as row["turnover"] * 244 /
    row["replayed_trade_days"]. `trades` is a frame or records with `price` and
    `quantity`, and `status` where some did not fill (only "filled" counts).
    A control whose figure differs from its candidate's by more than a quarter
    of the candidate's differs in holding and cost as well as in names, and the
    comparison is void (open-rules §3)."""

    if float(initial_cash) <= 0 or int(trading_days) <= 0:
        raise ValueError(f"need positive cash and days, got {initial_cash!r} and {trading_days!r}")
    frame = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(list(trades))
    if frame.empty:
        return 0.0
    if "status" in frame.columns:
        frame = frame[frame["status"] == "filled"]
    notional = (frame["price"].astype(float) * frame["quantity"].astype(float)).abs().sum()
    return float(notional) / float(initial_cash) * 244.0 / int(trading_days)


def random_skip(candidates, n_skip, key, salt=0):
    """`candidates` (codes in the book's order of preference) with `n_skip` of
    them removed by a fixed hash of (code, key, salt) instead of by the
    exclusion rule, the rest kept in order. `n_skip` is how many the rule
    skipped from this same list; `key` is the decision day, so the draw is
    redrawn at every decision and fixed within one."""

    codes = [str(code) for code in candidates]
    n_skip = int(n_skip)
    if not 0 <= n_skip <= len(codes):
        raise ValueError(f"cannot skip {n_skip} of {len(codes)} candidates")
    skipped = {codes[i] for i in _order(codes, f"{key}|{salt}")[:n_skip]}
    return [code for code in codes if code not in skipped]


def static_version(adaptive, value):
    """The static control of an adaptive rule: a callable that takes whatever
    `adaptive` takes and always returns `value`. Declare `value` before the run
    (most_common over the probe years, or the grid's middle), never from the
    years the nomination is tested on."""

    def static(*args, **kwargs):
        return value

    static.__name__ = f"static_{getattr(adaptive, '__name__', 'rule')}"
    return static


def most_common(choices):
    """The most frequent of `choices` (hashable values in decision order), the
    earliest one on a tie."""

    counts = Counter(choices)
    if not counts:
        raise ValueError("no choices to take the most common of")
    top = max(counts.values())
    return next(choice for choice in choices if counts[choice] == top)


def _order(codes, key):
    """Positions of `codes` sorted by a fixed 64-bit hash of (code, key)."""

    labels = np.array([f"{code}|{key}" for code in codes], dtype=object)
    return np.argsort(pd.util.hash_array(labels), kind="stable")
