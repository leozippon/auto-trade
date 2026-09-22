"""The market-state block: where the market stands, appended to every stock-day row.

Alpha158 describes each name against its own history. It never says what the
whole market was doing on that bar, so the booster has to learn one ordering
rule for a rebound, a crash and a quiet grind alike. This block hands it the
market's state as columns, so a split can condition the cross-sectional
ordering on the regime. The book does not move: seats, cash and cadence are
the carrier's, and only which names rank on top can change. That is what keeps
this arm out of the closed portfolio-timing family (`families.md`).

Two forms, one per candidate, each a 12-column block so the two legs dilute the
carrier's columns identically:

    STATE_NAMES          "m1": twelve date-level columns, one value per bar
                         shared by every name on it
        benchmark        mkt_r1, mkt_r5, mkt_r_s, mkt_r_l   trailing returns
                         mkt_vol_s                          realized vol, annualized
                         mkt_vol_ratio                      log(short vol / long vol)
                         mkt_ma_s, mkt_ma_l                 close over its moving average, minus 1
                         mkt_dd_l                           close over its rolling high, minus 1
                         turn_z                             5-bar mean of log traded amount,
                                                            z-scored on the long window
        constituents     xs_disp_s                          inter-quartile range of the
                                                            constituents' short-window returns
                         breadth_l                          share of constituents above their own
                                                            long moving average, minus one half
    INTERACTION_NAMES    "m_int": three stock factors -- short and long return,
                         short realized vol -- each ranked inside the bar's
                         constituents and centred on 0, times four of the
                         states above (short return, long MA gap, vol ratio,
                         breadth). Every name gets its own value on every bar

Why the state columns earn gain only through interaction. The label is a
per-bar rank centred on zero, so every bar's labels average to the same value.
A split on a date-level column separates whole bars and leaves both children
with that same mean: the split itself gains nothing. It gains only below a
split on a stock column, where the conditional mean does move with the state.
The interaction form hands the tree that product directly, and because it
varies across the names of a bar, a split on it cannot isolate a handful of
dates the way a split on a date-level column can.

Scaling. Neither form is z-scored across the bar (`data.py`): a date-level
column is constant on a bar, so a cross-sectional z-score would zero it, and
an interaction's z-score collapses to sign(state) x z(stock). The states are
returns, log ratios, shares and a z-score -- already comparable across bars --
and the stock factors are centred ranks in [-0.5, 0.5].

Density. Every window is at most `knobs.STATE_LONG` bars and must fit inside
the carrier's warm-up (`data.WARMUP_BARS`): the benchmark series in this
snapshot starts 2020-01-02, the panel is clipped to it, and the first training
bar of the first research year sits exactly one warm-up past that start. A
state column missing on the early part of a training window would be a hidden
time stamp, so `build` raises when any state value is non-finite on a bar the
fit or the review can use, instead of letting LightGBM route the gap.

Nothing here reads data: the benchmark close and amount and the constituent
panel are the ones `data.wide` already built, so the block adds no read, no
dataset and no visibility rule.
"""

import numpy as np
import pandas as pd

from lib import knobs

TRADING_DAYS = 244
MIN_NAMES = 20                     # constituents a cross-sectional state needs on a bar

STATE_NAMES = ("mkt_r1", "mkt_r5", "mkt_r_s", "mkt_r_l",
               "mkt_vol_s", "mkt_vol_ratio", "mkt_ma_s", "mkt_ma_l", "mkt_dd_l",
               "turn_z", "xs_disp_s", "breadth_l")
STOCK_FACTORS = ("ret_s", "ret_l", "vol_s")
INTERACTION_STATES = ("mkt_r_s", "mkt_ma_l", "mkt_vol_ratio", "breadth_l")
INTERACTION_NAMES = tuple(f"x_{stock}_{state}" for stock in STOCK_FACTORS
                          for state in INTERACTION_STATES)
BLOCK_NAMES = STATE_NAMES + INTERACTION_NAMES


def _lagged(series, lag):
    """Return over `lag` bars along the last axis; NaN where the start is missing."""

    out = np.full(series.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[..., lag:] = series[..., lag:] / series[..., :-lag] - 1.0
    return out


def _roll(matrix, window, how, min_periods):
    """Rolling statistic along the date axis of a (names, dates) matrix."""

    rolling = pd.DataFrame(matrix.T).rolling(window, min_periods=min_periods)
    return getattr(rolling, how)().to_numpy().T


def _across(matrix, member, how):
    """(dates,) statistic across each bar's constituents with a finite value."""

    masked = pd.DataFrame(np.where(member, matrix, np.nan))
    counted = masked.notna().sum(axis=0).to_numpy()
    value = how(masked).to_numpy(dtype=np.float64)
    return np.where(counted >= MIN_NAMES, value, np.nan)


def states(panel):
    """(dates, len(STATE_NAMES)) date-level market state on every panel bar."""

    short, long = knobs.STATE_SHORT, knobs.STATE_LONG
    close = pd.Series(panel["bench_close"], dtype=np.float64)
    daily = close.pct_change(fill_method=None)
    vol_s = daily.rolling(short, min_periods=short).std() * np.sqrt(TRADING_DAYS)
    vol_l = daily.rolling(long, min_periods=long).std() * np.sqrt(TRADING_DAYS)
    amount = np.log(pd.Series(panel["bench_amount"], dtype=np.float64).where(lambda a: a > 0))
    turn_z = ((amount.rolling(5, min_periods=5).mean()
               - amount.rolling(long, min_periods=long).mean())
              / amount.rolling(long, min_periods=long).std())

    member, prices = panel["member"], panel["C"]
    stock_ret = _lagged(prices, short)
    average = _roll(prices, long, "mean", long)
    above = np.where(np.isfinite(average) & np.isfinite(prices),
                     (prices > average).astype(np.float64), np.nan)
    breadth = _across(above, member, lambda frame: frame.mean(axis=0)) - 0.5
    dispersion = _across(stock_ret, member,
                         lambda frame: frame.quantile(0.75, axis=0) - frame.quantile(0.25, axis=0))

    series = close.to_numpy()
    columns = {
        "mkt_r1": _lagged(series, 1),
        "mkt_r5": _lagged(series, 5),
        "mkt_r_s": _lagged(series, short),
        "mkt_r_l": _lagged(series, long),
        "mkt_vol_s": vol_s.to_numpy(),
        "mkt_vol_ratio": np.log(vol_s / vol_l).to_numpy(),
        "mkt_ma_s": (close / close.rolling(short, min_periods=short).mean() - 1.0).to_numpy(),
        "mkt_ma_l": (close / close.rolling(long, min_periods=long).mean() - 1.0).to_numpy(),
        "mkt_dd_l": (close / close.rolling(long, min_periods=long).max() - 1.0).to_numpy(),
        "turn_z": turn_z.to_numpy(),
        "xs_disp_s": dispersion,
        "breadth_l": breadth,
    }
    return np.stack([np.asarray(columns[name], dtype=np.float64) for name in STATE_NAMES], axis=1)


def _stock_factors(panel):
    """{factor: (names, dates)} centred rank inside each bar's constituents; NaN off them."""

    short, long = knobs.STATE_SHORT, knobs.STATE_LONG
    prices, member = panel["C"], panel["member"]
    raw = {
        "ret_s": _lagged(prices, short),
        "ret_l": _lagged(prices, long),
        "vol_s": _roll(_lagged(prices, 1), short, "std", max(2, short // 2)),
    }
    out = {}
    for name, matrix in raw.items():
        masked = pd.DataFrame(np.where(member, matrix, np.nan))
        count = np.maximum(masked.notna().sum(axis=0).to_numpy(), 1)
        out[name] = (masked.rank(axis=0).to_numpy() - 0.5) / count - 0.5
    return out


def _require_dense(state, panel, first_bar):
    """Raise when a state is missing on a bar the fit or the review can use."""

    usable = np.zeros(len(panel["dates"]), dtype=bool)
    usable[first_bar:] = panel["member"][:, first_bar:].any(axis=0)
    bad = usable & ~np.isfinite(state).all(axis=1)
    if bad.any():
        where = int(np.argmax(bad))
        missing = [name for name, ok in zip(STATE_NAMES, np.isfinite(state[where])) if not ok]
        raise RuntimeError(
            f"market state {missing} is missing on {panel['dates'][where]} ({int(bad.sum())} usable "
            "bars in all); a state window longer than the carrier's warm-up reaches before the "
            "benchmark series starts")


def build(panel, candidate, first_bar):
    """(names, dates, 12) raw block of `candidate`, aligned to the panel axes."""

    state = states(panel)
    _require_dense(state, panel, first_bar)
    names = len(panel["symbols"])
    if candidate == "m1":
        return np.broadcast_to(state[None, :, :], (names, *state.shape)).copy()
    if candidate != "m_int":
        raise ValueError(f"no market-state block for candidate {candidate}")
    factors = _stock_factors(panel)
    column = {name: position for position, name in enumerate(STATE_NAMES)}
    return np.stack([factors[stock] * state[None, :, column[name]]
                     for stock in STOCK_FACTORS for name in INTERACTION_STATES], axis=2)


def coverage(values, panel):
    """Per-bar share of the bar's constituents carrying a finite value in EVERY block column.

    Date-level states are 1 or 0 per bar by construction; the interaction form
    falls below 1 where a constituent lacks its own factor window (a recent
    listing, a long suspension) and enters the booster as the neutral 0.
    """

    member = panel["member"]
    if values.shape[2] == 0:
        return np.full(len(panel["dates"]), np.nan)
    finite = np.isfinite(values).all(axis=2) & member
    total = member.sum(axis=0)
    return np.where(total > 0, finite.sum(axis=0) / np.maximum(total, 1), np.nan)
