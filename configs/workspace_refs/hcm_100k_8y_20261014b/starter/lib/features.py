"""Per-name daily features and the residual forward label, point in time.

    panel(context, days) -> Panel   the newest `days` trading days up to T-1
    latest_day(context)            the newest visible trading day (YYYYMMDD)

Everything comes from one `data.panel` read of `daily` plus the benchmark's
visible rows of `index_daily`, on adjusted prices (open and close times
`adj_factor`). The ten features per name and day -- log returns over 1, 5, 20
and 60 days, 20-day realised volatility, log 20-day turnover, the 5-day to
60-day turnover ratio, log 20-day amount, log float market cap and log
book-to-price -- are each standardised across the names of the day, clipped at
CLIP standard deviations, and 0 where unknown. A name is `valid` on a day when
it has a bar, its 20-day return and volatility, and its market cap.

The label follows the residual-label operating memory: the HORIZON-day forward
return from the next open, minus beta times the benchmark's, beta the trailing
OLS slope of daily returns on the benchmark's over BETA_DAYS (at least
MIN_BETA_DAYS days on which both are known), rank-centred across the day's
names to [-0.5, 0.5]. The label of day t is known on the trading day that
closes its forward window, `due[t]`; only from that day may the episode be
trained on or enter the store. Days whose window has not closed carry NaN
labels and no due day.

The trailing statistics are cumulative sums over the day axis (a 162 x 3,500
panel in well under a second); a pandas rolling over thousands of columns took
several seconds per daily fit.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from lib import data, knobs

COLUMNS = ["open", "close", "adj_factor", "amount", "turnover_rate", "circ_mv", "pb"]
FEATURES = 10
# The longest trailing window a feature needs; `valid` is False before it.
HISTORY_DAYS = 60
BETA_DAYS = 120
MIN_BETA_DAYS = 60
CLIP = 3.0


@dataclass
class Panel:
    dates: list  # YYYYMMDD, ascending
    codes: list  # ts_code, the columns of every array
    x: np.ndarray  # (days, names, FEATURES) float32
    valid: np.ndarray  # (days, names) bool
    label: np.ndarray  # (days, names) float32, NaN where not computable
    due: list  # per day: the YYYYMMDD its labels are known on, or None


def panel(context, days):
    frames = data.panel(context, COLUMNS, days)
    dates = [str(day) for day in frames["close"].index]
    codes = [str(code) for code in frames["close"].columns]
    raw = {name: frame.to_numpy(dtype="float64") for name, frame in frames.items()}
    close = np.where(raw["close"] > 0, raw["close"] * raw["adj_factor"], np.nan)
    opened = np.where(raw["open"] > 0, raw["open"] * raw["adj_factor"], np.nan)
    logc = pd.DataFrame(np.log(close)).ffill().to_numpy()
    r1 = _lag_diff(logc, 1)
    vol20 = _rolling_std(r1, 20, 15)
    turn = raw["turnover_rate"]
    circ = np.where(raw["circ_mv"] > 0, raw["circ_mv"], np.nan)
    columns = [
        r1,
        _lag_diff(logc, 5),
        _lag_diff(logc, 20),
        _lag_diff(logc, HISTORY_DAYS),
        vol20,
        np.log(_rolling_mean(turn, 20, 15) + 1e-4),
        np.log((_rolling_mean(turn, 5, 3) + 1e-4) / (_rolling_mean(turn, HISTORY_DAYS, 40) + 1e-4)),
        np.log(_rolling_mean(raw["amount"], 20, 15) + 1.0),
        np.log(circ),
        -np.log(np.where(raw["pb"] > 0, raw["pb"], np.nan)),
    ]
    x = np.stack([_standardised(column) for column in columns], axis=-1).astype("float32")
    valid = np.isfinite(close) & np.isfinite(columns[2]) & np.isfinite(vol20) & np.isfinite(circ)
    index = _index_bars(context, days).reindex(dates).to_numpy(dtype="float64")
    label = _residual_label(close, opened, index)
    horizon = knobs.HORIZON + 1
    due = [dates[i + horizon] if i + horizon < len(dates) else None for i in range(len(dates))]
    return Panel(dates, codes, x, valid, label, due)


def latest_day(context):
    start = (context.inference_at - timedelta(days=12)).strftime("%Y%m%d")
    days = pd.read_parquet(context.asof_dir + "/daily", columns=["trade_date"], filters=[("trade_date", ">=", start)])
    if days.empty:
        raise RuntimeError("the as-of view holds no recent bar")
    return str(days["trade_date"].astype(str).max())


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


def _lag_diff(values, lag):
    out = np.full(values.shape, np.nan)
    out[lag:] = values[lag:] - values[:-lag]
    return out


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


def _standardised(values):
    """Each day's values centred and scaled across its finite names, clipped, 0 where unknown."""

    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    n = finite.sum(axis=1, keepdims=True)
    mean = filled.sum(axis=1, keepdims=True) / np.maximum(n, 1)
    variance = (np.where(finite, filled - mean, 0.0) ** 2).sum(axis=1, keepdims=True) / np.maximum(n - 1, 1)
    spread = np.sqrt(variance)
    z = (values - mean) / np.where(spread > 0, spread, np.nan)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -CLIP, CLIP)
