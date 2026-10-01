"""The one panel builder: a dense daily sequence panel over the CSI 300 constituent union.

`build(context, calendar_days, labels)` reads the window once and returns
(dates x names) arrays. `fit` builds a trailing-window panel with the residual
label; a review day builds a short panel and scores its last row. Because both
call this one function and both normalise a window through `window()`, the
sequence a model is trained on and the sequence it scores are built identically
-- and so are the flat columns the LightGBM control sees (`lib/tree.py` only
reshapes `window()`'s output).

The name axis is the UNION of every constituent of every visible section in the
window, not the whole market: the arm's universe is the index, so the panel is
read, normalised and ranked inside it. `member[t]` says which of those names
were in the section in force on date t.

Visibility. `daily` has no `available_at` column -- the as-of view holds exactly
the rows visible at the decision, so at an 08:30 decision its newest row is the
previous trading day (T-1) and no timestamp filter is possible. The constituent
sections and the margin rows DO carry `available_at` and are placed by it:
a row is mapped to the first 08:30 decision strictly after its stamp, and
therefore onto the panel row that decision reads as T-1. No rule string is
hard-coded; the arithmetic is the rule.

Units (snapshot contract): prices CNY/share unadjusted, `adj_factor` cumulative,
`vol` shares, `amount` CNY, `turnover_rate` / `turnover_rate_f` decimals,
`volume_ratio` a ratio, `up_limit` / `down_limit` CNY/share, `circ_mv` CNY,
`margin_detail.rzye` / `rzmre` / `rzche` CNY, `index_daily` prices index points.

Sequence channels per name-day (SEQ_FEATURES = 14), raw in `seq`, transformed
in `window()`:

    0-4  O, H, L, C, VWAP   adjusted, forward-filled over days without a bar
    5    V                  split-consistent share volume, 0 without a bar
    6    TURN               turnover rate, percent, 0 without a bar
    7    HAS_BAR            1.0 on a day with a bar, else 0.0
    8    TURN_F             free-float turnover rate, percent
    9    VRATIO             volume ratio against its own 5-day mean (vendor column)
    10   LIM_UP             1.0 when the day's high touched the limit-up price
    11   LIM_DN             1.0 when the day's low touched the limit-down price
    12   MRG_BAL            margin financing balance / circulating market value
    13   MRG_FLOW           (financing buys - financing repayments) / circulating market value

Channels 8-13 are what this arm adds to the eight the earlier sequence arm had.
They are inputs, never a direction: `families.md` forbids turning any of them
into a standalone score.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import index, label

SEQ_LEN = 60                 # trading days in one input sequence
SEQ_FEATURES = 14
HOLD = 10                    # label horizon in trading days, registered up front
CLIP = 3.0

DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                 "adj_factor", "turnover_rate", "turnover_rate_f", "volume_ratio",
                 "up_limit", "down_limit", "circ_mv"]
MARGIN_COLUMNS = ["dataset", "available_at", "ts_code", "trade_date", "rzye", "rzmre", "rzche"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "open", "close"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]

SECTION_EXTRA_DAYS = 45      # a section is stamped at its own month end; look back past one gap
LIMIT_EPS = 1e-6
MV_FLOOR = 1.0e6


def _read_daily(context, calendar_days, codes):
    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start), ("ts_code", "in", list(codes))],
    )
    missing = [name for name in DAILY_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    return frame.assign(trade_date=frame["trade_date"].astype(str))


def _read_margin(context, calendar_days, codes):
    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=MARGIN_COLUMNS,
        filters=[("dataset", "=", "margin_detail"), ("trade_date", ">=", start),
                 ("ts_code", "in", list(codes))],
    )
    missing = [name for name in MARGIN_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"margin_detail is missing columns {missing}")
    return frame


def _read_benchmark(context, calendar_days):
    """Adjusted-free CSI 300 opens and closes, one row per trading day, PIT by available_at."""

    start = (context.inference_at - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=INDEX_COLUMNS,
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", index.INDEX_CODE),
                 ("trade_date", ">=", start)],
    )
    missing = [name for name in INDEX_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"index_daily is missing columns {missing}")
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    if frame.empty:
        raise RuntimeError(f"no visible {index.INDEX_CODE} daily rows in the macro domain")
    frame = frame.assign(trade_date=frame["trade_date"].astype(str))
    return frame.drop_duplicates("trade_date", keep="last").set_index("trade_date")


def _decision_stamps(context, dates):
    """UTC ns of the 08:30 decision of each panel date, plus this call's own decision.

    A row visible at stamp A is first usable by the first decision strictly
    after A; that decision reads the previous trading day as its newest row, so
    the row belongs on the panel row before it.
    """

    decision = pd.Timestamp(context.inference_at)
    clock = decision.tz_convert("Asia/Shanghai")
    at = pd.to_datetime(pd.Index(np.asarray(dates).astype("U8")), format="%Y%m%d")
    at = at.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=clock.hour, minutes=clock.minute)
    stamps = at.tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[ns]")
    tail = np.array([decision.tz_convert("UTC").tz_localize(None).to_datetime64()], dtype="datetime64[ns]")
    return np.concatenate([stamps, tail])


def _place_by_stamp(frame, dates, codes, columns, context):
    """(len(dates), len(codes), len(columns)) placed on the panel row its stamp allows, forward-filled."""

    out = np.full((len(dates), len(codes), len(columns)), np.nan)
    if frame.empty:
        return out
    stamps = _decision_stamps(context, dates)
    seen = pd.to_datetime(frame["available_at"], utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    row = np.searchsorted(stamps, seen, side="right") - 1
    position = pd.Series(np.arange(len(codes)), index=pd.Index(codes))
    column = position.reindex(frame["ts_code"].to_numpy()).to_numpy()
    keep = (row >= 0) & (row < len(dates)) & np.isfinite(column)
    row = row[keep].astype(np.int64)
    column = column[keep].astype(np.int64)
    for k, name in enumerate(columns):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)[keep]
        out[row, column, k] = values
    for k in range(len(columns)):
        out[:, :, k] = pd.DataFrame(out[:, :, k]).ffill().to_numpy()
    return out


def build(context, calendar_days, labels):
    """Dense arrays over the window's trading days and the constituent union of its sections."""

    section_frame = index.sections(context, calendar_days + SECTION_EXTRA_DAYS)
    union = sorted(section_frame["ts_code"].unique())
    frame = _read_daily(context, calendar_days, union)
    if frame.empty:
        raise RuntimeError("the daily window of the as-of view holds no constituent rows")
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

    market_value = pd.DataFrame(dense(frame["circ_mv"].to_numpy(dtype=np.float64))).ffill().to_numpy()
    market_value = np.where(market_value > MV_FLOOR, market_value, np.nan)
    margin = _place_by_stamp(_read_margin(context, calendar_days, union), dates, codes,
                             ["rzye", "rzmre", "rzche"], context)
    seq[:, :, 12] = np.nan_to_num(margin[:, :, 0] / market_value, nan=0.0, posinf=0.0, neginf=0.0)
    seq[:, :, 13] = np.nan_to_num((margin[:, :, 1] - margin[:, :, 2]) / market_value,
                                  nan=0.0, posinf=0.0, neginf=0.0)

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(pd.Index(codes))
    data = {
        "dates": dates,
        "codes": codes,
        "seq": seq,
        "has_bar": has_bar,
        "nbars": np.cumsum(has_bar, axis=0),
        "close": dense(frame["close"].to_numpy(dtype=np.float64)),
        "open_adj": opens,
        "amount": dense(amount),
        "member": index.membership(section_frame, dates, codes),
        "industry": info["l1_name"].fillna("未分类").astype(str).to_numpy(),
        "name": info["name"].fillna("").astype(str).to_numpy(),
        "sections": section_frame,
    }
    if labels:
        data["yrank"] = label.residual_rank(data, _read_benchmark(context, calendar_days), HOLD)
    return data


def tradable(data, t):
    """Names this account may hold at date index t: in the section, with a bar and a full sequence."""

    codes = pd.Series(data["codes"])
    names = pd.Series(data["name"])
    return (
        data["member"][t]
        & data["has_bar"][t]
        & (data["nbars"][t] >= SEQ_LEN)
        & ~codes.str.startswith(("688", "689")).to_numpy()
        & ~codes.str.endswith(".BJ").to_numpy()
        & ~names.str.contains("ST|退").to_numpy()
    )


def window(data, t, idx):
    """(len(idx), SEQ_LEN, SEQ_FEATURES) float32 normalised inputs for date index t.

    Prices become log ratios to the window's last close, volume and the two
    turnover columns log1p ratios, and every (lag, channel) is z-scored across
    the names of that date and clipped. One function for the GRU, its control
    and every review, so a flat column and a sequence step never disagree.
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
        out[:, :, 10:14] = raw[:, :, 10:14]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    return np.clip((out - mean) / (std + 1e-6), -CLIP, CLIP).astype(np.float32)
