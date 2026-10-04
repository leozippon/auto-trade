"""The one data pipeline: a dense daily sequence panel over the all-A pool.

`build(context, calendar_days, labels)` reads the `daily` window once and
returns (dates x names) arrays. `fit` builds one trailing-window panel with the
label and hands it to every learner it trains; a review or fill day builds a
short panel (`lib/book.py`) and scores its newest row. Both go through
`window()`, so every head is trained on and scores identically built
sequences.

Pool. The name axis is every A-share with a bar in the window except the
Beijing exchange (`.BJ`) and STAR (`688`/`689`); there is no price, liquidity or
listing-age filter. `tradable()` further drops names without a bar on that row,
names with fewer than `SEQ_LEN` bars, and names whose `universe` name contains
ST or 退 -- the name as the replay year's `universe` version spells it (the
as-of `universe` is replaced each 07-01 and fixed within the year), applied to
every date of the window.
The heads train and score on this whole tradable pool; the one-lot
affordability filter of a 100k account is the book's (`lib/book.py`), and so
is the beta lane's floor (`lib/beta.py`).

Visibility. `daily` has no `available_at` column: the as-of view holds exactly
the rows visible at the decision, so at an 08:30 decision its newest row is the
previous trading day (T-1). The benchmark rows in `macro` do carry
`available_at` and are filtered on it.

Units (data contract): prices CNY/share unadjusted, `adj_factor` cumulative,
`vol` shares, `amount` CNY, `turnover_rate` / `turnover_rate_f` decimals,
`volume_ratio` a ratio, `up_limit` / `down_limit` CNY/share, `index_daily`
prices index points.

Channels per name-day (12), raw in `seq`, transformed in `window()`:

    0-4  O, H, L, C, VWAP   adjusted, forward-filled over days without a bar
    5    V                  split-consistent share volume, 0 without a bar
    6    TURN               turnover rate, percent, 0 without a bar
    7    HAS_BAR            1.0 on a day with a bar, else 0.0
    8    TURN_F             free-float turnover rate, percent
    9    VRATIO             vendor volume ratio
    10   LIM_UP             1.0 when the day's high touched the limit-up price
    11   LIM_DN             1.0 when the day's low touched the limit-down price

The recipe's two margin-financing channels are DROPPED, not zero-filled: the
eight-year data selection mounts no events domain, so they were constant 0 in
every arm that ran this recipe, and reading an absent domain behind a
try/except is a silent fallback this package does not carry.

"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import knobs, label

SEQ_LEN = 60                 # trading days in one input sequence
SEQ_FEATURES = 12
CLIP = 3.0

DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                 "adj_factor", "turnover_rate", "turnover_rate_f", "volume_ratio",
                 "up_limit", "down_limit"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "open", "close"]
UNIVERSE_COLUMNS = ["ts_code", "name"]
LIMIT_EPS = 1e-6


def _start(context, calendar_days):
    return (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")


def _read_daily(context, calendar_days):
    """Every A-share bar in the window, except the Beijing exchange and STAR."""

    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", _start(context, calendar_days))],
    )
    missing = [name for name in DAILY_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    code = frame["ts_code"].astype(str)
    frame = frame.loc[~code.str.endswith(".BJ") & ~code.str.startswith(("688", "689"))]
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    return frame.assign(trade_date=frame["trade_date"].astype(str), ts_code=frame["ts_code"].astype(str))


def read_benchmark(context, calendar_days):
    """The label index's opens and closes, one row per trading day, PIT by available_at.

    `index_daily` rows carry an empty `index_code`; the series is keyed on
    `ts_code`.
    """

    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=INDEX_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", knobs.INDEX),
                 ("trade_date", ">=", _start(context, calendar_days))],
    )
    missing = [name for name in INDEX_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_daily is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    if frame.empty:
        raise RuntimeError(f"no visible {knobs.INDEX} daily rows in the macro domain")
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    return frame.drop_duplicates("trade_date", keep="last").set_index("trade_date")


def build(context, calendar_days, labels):
    """Dense arrays over the window and every non-STAR, non-Beijing A-share in it."""

    frame = _read_daily(context, calendar_days)
    if frame.empty:
        raise RuntimeError("the daily window of the as-of view holds no rows")
    dates = np.array(sorted(frame["trade_date"].unique()))
    codes = np.array(sorted(frame["ts_code"].unique()))
    rows, names = len(dates), len(codes)
    ti = np.searchsorted(dates, frame["trade_date"].to_numpy())
    si = np.searchsorted(codes, frame["ts_code"].to_numpy())

    def dense(values, fill=np.nan):
        out = np.full((rows, names), fill, dtype=np.float64)
        out[ti, si] = values
        return out

    factor = frame["adj_factor"].to_numpy(dtype=np.float64)
    volume = frame["vol"].to_numpy(dtype=np.float64)
    amount = frame["amount"].to_numpy(dtype=np.float64)
    vwap = np.where(volume > 0, amount / np.where(volume > 0, volume, 1.0), np.nan)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    up_limit = frame["up_limit"].to_numpy(dtype=np.float64)
    down_limit = frame["down_limit"].to_numpy(dtype=np.float64)

    opens = dense(frame["open"].to_numpy(dtype=np.float64) * factor)
    closes = dense(frame["close"].to_numpy(dtype=np.float64) * factor)
    seq = np.zeros((rows, names, SEQ_FEATURES), dtype=np.float32)
    prices = (opens, dense(high * factor), dense(low * factor), closes, dense(vwap * factor))
    for k, matrix in enumerate(prices):
        seq[:, :, k] = pd.DataFrame(matrix).ffill().to_numpy(dtype=np.float32)
    seq[:, :, 5] = np.nan_to_num(dense(volume / factor), nan=0.0)
    seq[:, :, 6] = np.nan_to_num(dense(frame["turnover_rate"].to_numpy(dtype=np.float64) * 100.0), nan=0.0)
    has_bar = dense(np.ones(len(frame)), fill=0.0) > 0
    seq[:, :, 7] = has_bar
    seq[:, :, 8] = np.nan_to_num(dense(frame["turnover_rate_f"].to_numpy(dtype=np.float64) * 100.0), nan=0.0)
    seq[:, :, 9] = np.nan_to_num(dense(frame["volume_ratio"].to_numpy(dtype=np.float64)), nan=0.0)
    seq[:, :, 10] = np.nan_to_num(dense((high >= up_limit - LIMIT_EPS).astype(np.float64)), nan=0.0)
    seq[:, :, 11] = np.nan_to_num(dense((low <= down_limit + LIMIT_EPS).astype(np.float64)), nan=0.0)

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    name = universe.drop_duplicates("ts_code").set_index("ts_code")["name"].reindex(pd.Index(codes))
    data = {
        "dates": dates,
        "codes": codes,
        "seq": seq,
        "has_bar": has_bar,
        "nbars": np.cumsum(has_bar, axis=0),
        "close": dense(frame["close"].to_numpy(dtype=np.float64)),
        "close_adj": closes,
        "open_adj": opens,
        "not_st": ~name.fillna("").astype(str).str.contains("ST|退").to_numpy(),
    }
    if labels:
        data["yrank"] = label.target(data, read_benchmark(context, calendar_days))
    return data


def tradable(data, t):
    """Names this account may hold at date index t: a bar, a full sequence, not ST."""

    return data["has_bar"][t] & (data["nbars"][t] >= SEQ_LEN) & data["not_st"]


def window(data, t, idx):
    """(len(idx), SEQ_LEN, SEQ_FEATURES) float32 normalised inputs for date index t.

    Prices become log ratios to the window's last close, volume and the two
    turnover columns and the volume ratio log1p transforms, and every
    (lag, channel) is z-scored across the names of that date and clipped. One
    function for every head and every review.
    """

    raw = data["seq"][t - SEQ_LEN + 1: t + 1, idx, :].transpose(1, 0, 2).astype(np.float64)
    last_close = raw[:, -1:, 3:4]
    out = np.empty_like(raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, :, 0:5] = np.log(np.clip(raw[:, :, 0:5] / np.clip(last_close, 1e-4, None), 1e-4, None))
        volume = raw[:, :, 5]
        out[:, :, 5] = np.log1p(volume / (volume.mean(axis=1, keepdims=True) + 1e-9))
        out[:, :, 6] = np.log1p(np.clip(raw[:, :, 6], 0.0, None))
        out[:, :, 7] = raw[:, :, 7]
        out[:, :, 8] = np.log1p(np.clip(raw[:, :, 8], 0.0, None))
        out[:, :, 9] = np.log1p(np.clip(raw[:, :, 9], 0.0, None))
        out[:, :, 10:12] = raw[:, :, 10:12]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    return np.clip((out - mean) / (std + 1e-6), -CLIP, CLIP).astype(np.float32)
