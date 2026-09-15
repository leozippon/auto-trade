"""Dense daily price-volume panels from the rolling as-of view: one builder for training and decisions.

`build(context, calendar_days)` reads the last `calendar_days` of the `daily`
domain (bounded, projected) and returns (dates x names) arrays. `fit` builds a
trailing-three-year panel with labels; a review day builds a 130-day panel and
scores its last row. Because both call this one function, the sequence a model
is trained on and the sequence it scores are built identically.

Visibility: `daily` has no `available_at` column. The as-of view holds exactly
the rows visible at the decision, so at an 08:30 decision the newest row is the
previous trading day (T-1) and no timestamp filter is needed or possible.

Units (snapshot contract): prices CNY/share, `vol` shares, `amount` CNY,
`turnover_rate` a decimal; `adj_factor` is the cumulative adjustment factor, so
price * adj_factor is continuous across splits and dividends and share volume /
adj_factor is the split-consistent volume.

Sequence inputs per name-day (SEQ_FEATURES):
    O, H, L, C, VWAP   adjusted prices, forward-filled over days without a bar
    V                  split-consistent share volume, 0 on days without a bar
    TURN               turnover rate in percent, 0 on days without a bar
    HAS_BAR            1.0 on a trading day with a bar, else 0.0
Label (training only): the 10-day open-to-open return O[t+11] / O[t+1] - 1 of a
signal on day t (a decision on t+1 buys at or near that open), ranked per date
into (rank - 1) / n - 0.5 over the names that have it.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

SEQ_LEN = 60                 # trading days in one input sequence
HOLD = 10                    # label horizon in trading days
SEQ_FEATURES = 8
DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                 "adj_factor", "turnover_rate"]

PRICE_CAP = 30.0             # T-1 close, CNY: one 100-share lot stays under half a position
PRICE_FLOOR = 1.0
ADV_FLOOR = 3.0e7            # 20-day mean amount, CNY
ADV_DAYS = 20
MIN_LISTED_DAYS = 120


def read_daily(context, calendar_days):
    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna() & ~frame["ts_code"].str.endswith(".BJ")]
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def read_universe(context):
    return pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "name", "list_date"])


def build(context, calendar_days, labels):
    """Arrays over the visible trading days of the window and every non-BJ name with a bar in it."""

    frame = read_daily(context, calendar_days)
    dates = np.array(sorted(frame["trade_date"].unique()))
    codes = np.array(sorted(frame["ts_code"].unique()))
    T, S = len(dates), len(codes)
    ti = np.searchsorted(dates, frame["trade_date"].to_numpy())
    si = np.searchsorted(codes, frame["ts_code"].to_numpy())

    def dense(values, fill=np.nan):
        out = np.full((T, S), fill, dtype=np.float64)
        out[ti, si] = values
        return out

    factor = frame["adj_factor"].to_numpy(dtype=np.float64)
    vol = frame["vol"].to_numpy(dtype=np.float64)
    amount = frame["amount"].to_numpy(dtype=np.float64)
    vwap = np.where(vol > 0, amount / np.where(vol > 0, vol, 1.0), np.nan)
    opens = dense(frame["open"].to_numpy(dtype=np.float64) * factor)
    seq = np.zeros((T, S, SEQ_FEATURES), dtype=np.float32)
    prices = (opens, dense(frame["high"].to_numpy(dtype=np.float64) * factor),
              dense(frame["low"].to_numpy(dtype=np.float64) * factor),
              dense(frame["close"].to_numpy(dtype=np.float64) * factor), dense(vwap * factor))
    for k, matrix in enumerate(prices):
        seq[:, :, k] = pd.DataFrame(matrix).ffill().to_numpy(dtype=np.float32)
    seq[:, :, 5] = np.nan_to_num(dense(vol / factor), nan=0.0)
    seq[:, :, 6] = np.nan_to_num(dense(frame["turnover_rate"].to_numpy(dtype=np.float64) * 100.0), nan=0.0)
    has_bar = dense(np.ones(len(frame)), fill=0.0) > 0
    seq[:, :, 7] = has_bar
    panel = {
        "dates": dates,
        "codes": codes,
        "seq": seq,
        "has_bar": has_bar,
        "nbars": np.cumsum(has_bar, axis=0),
        "close": dense(frame["close"].to_numpy(dtype=np.float64)),
        "amount": dense(amount),
    }
    if labels:
        y = np.full((T, S), np.nan)
        y[: T - HOLD - 1] = opens[HOLD + 1:] / opens[1: T - HOLD] - 1.0
        ranks = pd.DataFrame(y).rank(axis=1, pct=True).to_numpy()
        count = np.isfinite(y).sum(axis=1, keepdims=True)
        panel["yrank"] = np.where(np.isfinite(y), ranks - 1.0 / np.maximum(count, 1) - 0.5, np.nan).astype(np.float32)
    return panel


def pool_mask(context, panel):
    """The account pool on the newest row (T-1): the names a review may hold or buy."""

    last = len(panel["dates"]) - 1
    codes = pd.Series(panel["codes"])
    universe = read_universe(context).drop_duplicates("ts_code").set_index("ts_code")
    names = universe["name"].reindex(codes).fillna("")
    listed = pd.to_datetime(universe["list_date"].reindex(codes), format="%Y%m%d", errors="coerce")
    decision_day = pd.Timestamp(context.inference_at.date())
    listed_days = (decision_day - listed).dt.days.to_numpy()
    amount = panel["amount"][-ADV_DAYS:]
    present = np.isfinite(amount).sum(axis=0)
    adv = np.where(present >= ADV_DAYS // 2, np.nansum(amount, axis=0) / np.maximum(present, 1), 0.0)
    close = panel["close"][last]
    return (
        panel["has_bar"][last]
        & (panel["nbars"][last] >= SEQ_LEN)
        & ~codes.str.startswith(("688", "689")).to_numpy()
        & ~names.str.contains("ST|退").to_numpy()
        & (np.nan_to_num(listed_days, nan=-1.0) >= MIN_LISTED_DAYS)
        & (close > PRICE_FLOOR) & (close <= PRICE_CAP)
        & (adv >= ADV_FLOOR)
    )
