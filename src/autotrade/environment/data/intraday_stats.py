"""Derived daily realized-moment and intraday-profile summary over the 1-minute bar lake.

One row per ``(ts_code, trade_date)``, computed from that trade date's own
minute bars and nothing else, built and stamped exactly like
:mod:`intraday_flow` (same source, same build, same close contract) so a
strategy can use intraday statistics without mounting the minute domain.

Bars. Only the standard session grid is used: the bar labelled 09:30, then
09:31-11:30 and 13:01-15:00 (labels are bar END times, so a complete day is 241
bars). The 09:30 bar carries the opening call-auction match (for Shenzhen names
also the first seconds of continuous trading) and the 15:00 bar carries the
closing call-auction match; both are ordinary bars here. Any bar off that grid
is ignored. ``.BJ`` names are excluded as in :mod:`intraday_flow`: their minute
close at 15:00 is not the exchange's official close and part of their history
carries 15:01-15:30 after-hours bars.

Price path. Bars before the stock-day's first traded bar (``vol > 0``) only
repeat the previous close, so the path starts at that bar's ``open`` -- the
opening auction price, or the first trade when the auction did not match --
and then takes every later bar's ``close``:

    p_0 = open of the first traded bar;  p_k = close of the k-th bar from it
    r_k = log(p_k / p_{k-1}),  k = 1..N    (N = 241 on a complete day)

The lunch break is not a gap in this path (11:30 close -> 13:01 close is one
return), and a missing bar simply makes the next return span it, so
``sum_k r_k = log(close / open)`` always holds. Moments (Amaya, Christoffersen,
Jacobs and Vasquez 2015; Bollerslev, Li and Zhao 2020):

    rv            = sum r_k^2                       realized variance
    rv_down_share = sum_{r_k<0} r_k^2 / rv          downside share of rv
    rsk           = sqrt(N) * sum r_k^3 / rv^1.5    realized skewness
    rku           = N * sum r_k^4 / rv^2            realized kurtosis
    sjv           = sum_{r_k>0} r_k^2 - sum_{r_k<0} r_k^2   signed jump variation

``rv_down_share``, ``rsk`` and ``rku`` are undefined (null) when ``rv == 0``, a
day whose price never moved (a limit board sealed from the open).

Intraday profile, with ``P(t)`` the close of the last bar at or before label
``t`` on the path (``p_0`` if there is none) and simple returns:

    ret_open30  = P(10:00) / p_0 - 1           opening price -> 10:00
    ret_mid     = P(14:30) / P(10:00) - 1      10:00 -> 14:30
    ret_close30 = P(15:00) / P(14:30) - 1      14:30 -> close
    vwap_dev    = P(15:00) / VWAP - 1,  VWAP = sum amount / sum vol
    vol_open30_share  = vol of bars 09:30-10:00 / day vol
    vol_close30_share = vol of bars 14:31-15:00 / day vol

``n_bars`` counts the grid bars present (241 on a complete day) as a quality
diagnostic; it is never applied as a filter here. A stock-day with no traded
volume at all has no price path and no VWAP and is not written, as in
:mod:`intraday_flow`. Rolling windows are not precomputed, for the same reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.environment.data.contracts import INTRADAY_STATS_CONTRACT
from autotrade.environment.data.intraday_flow import MINUTE_COLUMNS, stamp_available_at

INTRADAY_STATS_DATASET = "intraday_stats"
# The flow columns plus the bar open, which prices the start of the path.
INTRADAY_STATS_MINUTE_COLUMNS = (*MINUTE_COLUMNS, "open")

INTRADAY_STATS_COLUMNS = (
    "ts_code",
    "trade_date",
    "rv",
    "rv_down_share",
    "rsk",
    "rku",
    "sjv",
    "ret_open30",
    "ret_mid",
    "ret_close30",
    "vwap_dev",
    "vol_open30_share",
    "vol_close30_share",
    "n_bars",
    "available_at",
    "available_at_rule",
)
_FLOAT_COLUMNS = INTRADAY_STATS_COLUMNS[2:13]
_FLAT_DAY_UNDEFINED = ("rv_down_share", "rsk", "rku")


def aggregate_intraday_stats(minutes: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw minute bars to one PIT-stamped statistics row per stock-day.

    ``minutes`` must carry :data:`INTRADAY_STATS_MINUTE_COLUMNS` and may span
    several trade dates; every quantity is grouped inside one
    ``(ts_code, trade_date)``, so the result never depends on call shape.
    """
    missing = [c for c in INTRADAY_STATS_MINUTE_COLUMNS if c not in minutes.columns]
    if missing:
        raise ValueError(f"minute frame is missing columns {missing}")
    frame = minutes[~minutes["ts_code"].str.endswith(".BJ")]
    frame = frame.sort_values(["ts_code", "trade_date", "trade_time"], kind="stable")
    # Read the clock once per distinct timestamp rather than once per row.
    stamp, stamps = pd.factorize(frame["trade_time"])
    clock = pd.Index(stamps).str.slice(11, 16)
    on_grid = (((clock >= "09:30") & (clock <= "11:30")) | ((clock >= "13:01") & (clock <= "15:00")))[stamp]
    if not on_grid.any():
        return _empty()
    frame, stamp = frame[on_grid], stamp[on_grid]
    upto_1000 = (clock <= "10:00")[stamp]
    upto_1430 = (clock <= "14:30")[stamp]
    code = frame["ts_code"].to_numpy()
    date = frame["trade_date"].to_numpy()
    opens = frame["open"].to_numpy(dtype="float64")
    close = frame["close"].to_numpy(dtype="float64")
    volume = frame["vol"].to_numpy(dtype="float64")
    amount = frame["amount"].to_numpy(dtype="float64")
    if not (np.isfinite(opens).all() and np.isfinite(close).all() and (opens > 0).all() and (close > 0).all()):
        raise ValueError("minute frame carries a non-positive or missing session price")

    first = np.concatenate(([True], (code[1:] != code[:-1]) | (date[1:] != date[:-1])))
    group = np.cumsum(first) - 1
    n_groups = int(group[-1]) + 1

    def per_group(values: np.ndarray, index: np.ndarray = group) -> np.ndarray:
        return np.bincount(index, weights=values, minlength=n_groups)

    total_vol = per_group(volume)
    total_amt = per_group(amount)
    # A stock-day with no traded volume (whole-day halt) has no path and no
    # VWAP; it has no `daily` bar either. Drop it rather than invent values.
    keep = total_vol > 0
    if not keep.any():
        return _empty()

    # The path starts at each stock-day's first traded bar; the bars before it
    # carry no price of the day. Traded-bar counts reset per stock-day.
    traded = volume > 0
    running = np.cumsum(traded)
    starts = np.flatnonzero(first)
    before = running[starts] - traded[starts]
    path = np.flatnonzero(running - before[group] > 0)
    path_group = group[path]
    path_first = np.concatenate(([True], path_group[1:] != path_group[:-1]))
    path_close = close[path]
    previous = np.concatenate(([np.nan], path_close[:-1]))
    returns = np.log(path_close / np.where(path_first, opens[path], previous))

    squared = returns * returns
    n_returns = np.bincount(path_group, minlength=n_groups)
    rv = per_group(squared, path_group)
    rv_down = per_group(np.where(returns < 0, squared, 0.0), path_group)
    rv_up = per_group(np.where(returns > 0, squared, 0.0), path_group)
    cubed = per_group(squared * returns, path_group)
    fourth = per_group(squared * squared, path_group)

    p_open = np.full(n_groups, np.nan)
    p_open[path_group[path_first]] = opens[path][path_first]
    p_1000 = _last_per_group(path_close, path_group, upto_1000[path], p_open)
    p_1430 = _last_per_group(path_close, path_group, upto_1430[path], p_open)
    p_close = _last_per_group(path_close, path_group, np.ones(len(path), dtype=bool), p_open)

    moved = rv > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = {
            "rv": rv,
            "rv_down_share": np.where(moved, rv_down / rv, np.nan),
            "rsk": np.where(moved, np.sqrt(n_returns) * cubed / rv**1.5, np.nan),
            "rku": np.where(moved, n_returns * fourth / rv**2, np.nan),
            "sjv": rv_up - rv_down,
            "ret_open30": p_1000 / p_open - 1.0,
            "ret_mid": p_1430 / p_1000 - 1.0,
            "ret_close30": p_close / p_1430 - 1.0,
            "vwap_dev": p_close * total_vol / total_amt - 1.0,
            "vol_open30_share": per_group(np.where(upto_1000, volume, 0.0)) / total_vol,
            "vol_close30_share": per_group(np.where(upto_1430, 0.0, volume)) / total_vol,
        }
    out = pd.DataFrame(
        {
            "ts_code": code[first],
            "trade_date": date[first],
            **stats,
            "n_bars": np.bincount(group, minlength=n_groups).astype("int32"),
        }
    )[keep].reset_index(drop=True)
    # Only the three rv-normalised moments may be undefined, and only on a day
    # whose price never moved; anything else non-finite (say turnover 0 on a
    # bar with volume) is a source defect and must not be published.
    defined = [column for column in _FLOAT_COLUMNS if column not in _FLAT_DAY_UNDEFINED]
    if not np.isfinite(out[defined].to_numpy()).all():
        raise ValueError("intraday_stats produced a non-finite value outside the flat-day moments")
    return stamp_available_at(out, INTRADAY_STATS_CONTRACT)[list(INTRADAY_STATS_COLUMNS)]


def _last_per_group(
    values: np.ndarray, groups: np.ndarray, mask: np.ndarray, default: np.ndarray
) -> np.ndarray:
    """Last masked value of each (sorted, contiguous) group, else ``default``."""
    out = default.copy()
    values, groups = values[mask], groups[mask]
    if len(groups):
        last = np.concatenate((groups[1:] != groups[:-1], [True]))
        out[groups[last]] = values[last]
    return out


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": pd.Series(dtype="object"),
            "trade_date": pd.Series(dtype="object"),
            **{column: pd.Series(dtype="float64") for column in _FLOAT_COLUMNS},
            "n_bars": pd.Series(dtype="int32"),
            "available_at": pd.Series(dtype="object"),
            "available_at_rule": pd.Series(dtype="object"),
        }
    )
