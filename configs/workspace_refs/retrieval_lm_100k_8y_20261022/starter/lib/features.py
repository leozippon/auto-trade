"""Per-name price-volume patterns and the residual forward label, point in time.

    panel(context, days) -> Panel   the newest `days` trading days up to T-1

Everything comes from one `data.panel` read of `daily` plus the benchmark's
visible rows of `index_daily`, on adjusted prices (open and close times
`adj_factor`).

A pattern is the shape of a name's last WINDOW bars, made comparable across
names and years. With e the day's log return in excess of the benchmark's and
s the name's standard deviation of e over the VOL_DAYS bars ending the window:

    path[k]  the cumulative sum of e / s over the window's first k + 1 days,
             divided by sqrt(k + 1): under a random walk each element is
             about N(0, 1), whatever the name's volatility or the market did
    flow[k]  log(turnover / the name's mean turnover over the VOL_DAYS bars
             ending that day) on the window's k-th day, over FLOW_SCALE

so the pattern is 2 * WINDOW numbers of about unit scale. A return is defined
only between two consecutive bars (the step a resumption makes after a
suspension is left out), a missing step or turnover reads 0, and nothing is
standardised across names, so a name's pattern depends only on its own bars in
the window and the VOL_DAYS before it, never on how far back the panel reads
or which other names it holds. A pattern is `valid` when the name has a bar on
the window's last day, at least MIN_BARS bars in the window, and a finite
volatility and mean turnover.

The label follows the residual-label operating memory: the HORIZON-day forward
return from the next open, minus beta times the benchmark's, beta the trailing
OLS slope of daily returns on the benchmark's over BETA_DAYS (at least
MIN_BETA_DAYS days on which both are known), rank-centred across the day's
names to [-0.5, 0.5]. The label of bar t is known on the trading day that
closes its forward window, `due[t]`; only from that day may the case enter the
memory or be trained on. Bars whose window has not closed carry NaN labels and
no due day.

The trailing statistics are cumulative sums over the day axis, so a panel of
several hundred days over every listed name takes seconds, not minutes.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, knobs

COLUMNS = ["open", "close", "adj_factor", "turnover_rate"]
BETA_DAYS = 120
MIN_BETA_DAYS = 60
MIN_VOL_DAYS = 40
MIN_BARS = 15
FLOW_SCALE = 0.5
CLIP = 5.0
# Bars a panel needs behind its first pattern, and behind the first label.
PATTERN_LEAD = knobs.VOL_DAYS + knobs.WINDOW
LABEL_LEAD = BETA_DAYS
PATTERN = 2 * knobs.WINDOW


@dataclass
class Panel:
    dates: list  # YYYYMMDD, ascending
    codes: list  # ts_code, the columns of every array
    excess: np.ndarray  # (days, names) excess log return, NaN where undefined
    scale: np.ndarray  # (days, names) trailing standard deviation of `excess`
    flow: np.ndarray  # (days, names) log turnover over its trailing mean, NaN where unknown
    valid: np.ndarray  # (days, names) bool: a pattern is defined
    label: np.ndarray  # (days, names) float32, NaN where not computable
    due: list  # per day: the YYYYMMDD its labels are known on, or None


def panel(context, days):
    frames = data.panel(context, COLUMNS, days)
    dates = [str(day) for day in frames["close"].index]
    codes = [str(code) for code in frames["close"].columns]
    raw = {name: frame.to_numpy(dtype="float64") for name, frame in frames.items()}
    close = np.where(raw["close"] > 0, raw["close"] * raw["adj_factor"], np.nan)
    opened = np.where(raw["open"] > 0, raw["open"] * raw["adj_factor"], np.nan)
    index = _index_bars(context, days).reindex(dates).to_numpy(dtype="float64")
    step = np.full(close.shape, np.nan)
    step[1:] = np.log(close[1:] / close[:-1])
    index_step = np.full(len(dates), np.nan)
    index_step[1:] = np.log(index[1:, 1] / index[:-1, 1])
    excess = step - index_step[:, None]
    scale = _rolling_std(excess, knobs.VOL_DAYS, MIN_VOL_DAYS)
    turnover = np.where(np.isfinite(close), raw["turnover_rate"], np.nan)
    mean_turnover = _rolling_mean(turnover, knobs.VOL_DAYS, MIN_VOL_DAYS)
    flow = np.log((turnover + 1e-4) / (mean_turnover + 1e-4))
    bars, _ = _window_sums(np.where(np.isfinite(close), 1.0, np.nan), knobs.WINDOW)
    valid = (
        np.isfinite(close) & (bars >= MIN_BARS) & np.isfinite(scale) & (scale > 0) & np.isfinite(mean_turnover)
    )
    valid[: PATTERN_LEAD - 1] = False
    label = _residual_label(close, opened, index)
    horizon = knobs.HORIZON + 1
    due = [dates[i + horizon] if i + horizon < len(dates) else None for i in range(len(dates))]
    return Panel(dates, codes, excess, scale, flow, valid, label, due)


def patterns(p, day):
    """(names, PATTERN) float32 patterns of every name at bar `day` (an index
    into p.dates), zeros where the pattern is not valid."""

    start = day - knobs.WINDOW + 1
    if start < 1:
        raise ValueError(f"bar {day} has fewer than {knobs.WINDOW} bars behind it in this panel")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = p.excess[start:day + 1] / p.scale[day]
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -CLIP, CLIP)
    path = np.cumsum(z, axis=0) / np.sqrt(np.arange(1, knobs.WINDOW + 1))[:, None]
    flow = np.clip(np.nan_to_num(p.flow[start:day + 1], nan=0.0, posinf=0.0, neginf=0.0) / FLOW_SCALE, -CLIP, CLIP)
    out = np.concatenate([path.T, flow.T], axis=1).astype("float32")
    out[~p.valid[day]] = 0.0
    return out


def _index_bars(context, days):
    """The benchmark's visible open and close, indexed by YYYYMMDD."""

    start = (context.inference_at - timedelta(days=int(1.6 * days) + 14)).strftime("%Y%m%d")
    rows = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=["dataset", "ts_code", "trade_date", "open", "close", "available_at"],
        filters=[("dataset", "==", "index_daily"), ("ts_code", "==", knobs.INDEX), ("trade_date", ">=", start)],
    )
    stamp = pd.to_datetime(rows["available_at"], utc=True)
    rows = rows[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    rows = rows.assign(trade_date=rows["trade_date"].astype(str)).drop_duplicates("trade_date", keep="last")
    return rows.set_index("trade_date").sort_index()[["open", "close"]].astype("float64")


def _residual_label(close, opened, index):
    """(days x names) rank-centred residual forward return, NaN where unknown."""

    stock_ret = close[1:] / close[:-1] - 1.0
    index_ret = index[1:, 1] / index[:-1, 1] - 1.0
    beta = np.full(close.shape, np.nan)
    beta[1:] = _rolling_beta(stock_ret, index_ret, BETA_DAYS, MIN_BETA_DAYS)
    # Entered at the next open, left HORIZON days later: days 1..D-h over days h..D-1.
    horizon = knobs.HORIZON + 1
    span = len(close) - horizon
    forward = np.full(close.shape, np.nan)
    forward[:span] = opened[horizon:] / opened[1:span + 1] - 1.0
    index_forward = np.full(len(index), np.nan)
    index_forward[:span] = index[horizon:, 0] / index[1:span + 1, 0] - 1.0
    residual = forward - beta * index_forward[:, None]
    return (pd.DataFrame(residual).rank(axis=1, pct=True) - 0.5).to_numpy(dtype="float32")


def _window_sums(values, window):
    """(sum, count) of the finite values over each day's trailing `window` days, inclusive."""

    finite = np.isfinite(values)
    total = np.cumsum(np.where(finite, values, 0.0), axis=0)
    count = np.cumsum(finite, axis=0, dtype="float64")
    total[window:] -= total[:-window].copy()
    count[window:] -= count[:-window].copy()
    return total, count


def _rolling_mean(values, window, min_periods):
    total, count = _window_sums(values, window)
    return total / np.where(count >= min_periods, count, np.nan)


def _rolling_std(values, window, min_periods):
    total, count = _window_sums(values, window)
    squares, _ = _window_sums(values * values, window)
    n = np.where(count >= max(min_periods, 2), count, np.nan)
    return np.sqrt(np.clip((squares - total * total / n) / (n - 1.0), 0.0, None))


def _rolling_beta(stock_ret, index_ret, window, min_periods):
    """Each column's trailing OLS slope on `index_ret`, over the days both are finite."""

    index_ret = np.broadcast_to(index_ret[:, None], stock_ret.shape)
    paired = np.isfinite(stock_ret) & np.isfinite(index_ret)
    x = np.where(paired, stock_ret, np.nan)
    y = np.where(paired, index_ret, np.nan)
    sx, n = _window_sums(x, window)
    sy, _ = _window_sums(y, window)
    sxy, _ = _window_sums(x * y, window)
    syy, _ = _window_sums(y * y, window)
    n = np.where(n >= min_periods, n, np.nan)
    covariance = sxy - sx * sy / n
    variance = syy - sy * sy / n
    return covariance / np.where(variance > 0, variance, np.nan)
