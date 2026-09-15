"""PIT daily reads, (S, T) wide matrices and the one feature builder of fit and decision.

The daily domain has no available_at column: at a 08:30 decision on day T only rows with
trade_date < T are visible. All reads are bounded (calendar-day window) and rooted at
context.asof_dir + "/daily" (directory of parquet parts). Beijing Stock Exchange names
(.BJ) are dropped at the read, for training and decisions alike.

Feature columns (FEATURE_NAMES): the 158 Alpha columns, then the blocks named in
FEATURE_GROUPS in that order -- vipf2 (6 growth / forecast-event columns), mfflow (8
money-flow columns), newage (listing age). Both paths z-score the raw columns per
cross-section with features.robust_zscore.

Training window: the fit reads FIT_CALENDAR_DAYS of daily rows (TRAIN_YEARS plus a
warm-up) and trains only on bars after the first WARMUP_BARS, so every training row
has its full 60-bar Alpha158 window.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import features, labels, mfflow, newage, vipf2

# Final 11 columns (task A decision 1). VWAP = amount / vol is derived.
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol",
        "amount", "adj_factor", "is_suspended", "up_limit"]

DECISION_LOOKBACK_DAYS = 130   # >= 63 trading days: covers the d=60 window + margin
HOLD_DAYS = 10                 # label holding period (open[D+10] / open[D] - 1)
TRAIN_YEARS = 3
WARMUP_BARS = 60               # leading bars of the fit window that only warm the 60-bar operators
FIT_CALENDAR_DAYS = int(365.25 * TRAIN_YEARS) + 110   # the training years plus >= WARMUP_BARS trading days

# name -> (columns, decision-path frame, fit-path panels); FEATURE_GROUPS picks and orders them.
BLOCKS = {
    "vipf2": (vipf2.FEATURES, vipf2.vipf2_frame, vipf2.vipf2_fit_panels),
    "mfflow": (mfflow.FEATURES, mfflow.mfflow_frame, mfflow.mfflow_fit_panels),
    "newage": (newage.FEATURES, newage.newage_frame, newage.newage_fit_panels),
}
FEATURE_GROUPS = ("vipf2", "mfflow", "newage")
FEATURE_NAMES = list(features.ALPHA158) + [name for group in FEATURE_GROUPS for name in BLOCKS[group][0]]

_WIDE_FIELDS = ("open", "high", "low", "close", "vol", "amount")
_WIDE_KEYS = {"open": "O", "high": "H", "low": "L", "close": "C", "vol": "V", "amount": "A"}
_RAW_FIELDS = ("open", "close", "vol", "amount", "up_limit")
_RAW_KEYS = {"open": "raw_open", "close": "raw_close", "vol": "raw_vol",
             "amount": "raw_amount", "up_limit": "raw_up_limit"}


def decision_date(context):
    """YYYYMMDD string of the decision day (inference date)."""
    return context.inference_at.strftime("%Y%m%d")


def visible_frame(context, cal_days):
    """Visible daily rows of the last cal_days calendar days: trade_date < T, no .BJ."""
    t = decision_date(context)
    start = (context.inference_at - timedelta(days=cal_days)).strftime("%Y%m%d")
    df = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=COLS,
        filters=[("trade_date", ">=", start)],
    )
    df = df[(df["trade_date"] < t) & ~df["ts_code"].str.endswith(".BJ")]
    return df.reset_index(drop=True)


def to_wide(df):
    """Build qfq wide matrices from a visible frame.

    qfq anchor (T-1 frozen): factor per symbol = adj_factor / adj_factor of the symbol's
    last visible row; prices and volume/amount are adjusted with the same factor
    (the factor cancels in VWAP = amount / vol, so VWAP uses raw values).
    Returns dict with dates (list[str]), symbols (np array), O/H/L/C/V/A (qfq, S x T),
    VWAP, raw_open/raw_close/raw_vol/raw_amount/raw_up_limit (S x T), suspended (bool S x T),
    n_bars (visible bars per symbol).
    """
    if df is None or df.empty:
        empty = {k: np.zeros((0, 0)) for k in
                 ("O", "H", "L", "C", "V", "A", "VWAP", "raw_open", "raw_close",
                  "raw_vol", "raw_amount", "raw_up_limit")}
        empty["suspended"] = np.zeros((0, 0), dtype=bool)
        empty["dates"] = []
        empty["symbols"] = np.array([], dtype=object)
        empty["n_bars"] = np.zeros((0,), dtype=np.int64)
        return empty

    last_idx = df.groupby("ts_code", sort=False)["trade_date"].idxmax()
    adj_last = df.loc[last_idx].set_index("ts_code")["adj_factor"]
    factor = (df["adj_factor"].to_numpy() / adj_last.loc[df["ts_code"]].to_numpy())
    adj = pd.DataFrame(index=df.index)
    adj["trade_date"] = df["trade_date"].to_numpy()
    adj["ts_code"] = df["ts_code"].to_numpy()
    for c in _WIDE_FIELDS:
        adj[c] = df[c].to_numpy() * factor

    pivot = adj.pivot(index="trade_date", columns="ts_code", values=list(_WIDE_FIELDS))
    dates = sorted(pivot.index.unique())
    syms = sorted(pivot.columns.get_level_values(1).unique())
    raw_df = df[["trade_date", "ts_code", "open", "close", "vol", "amount",
                 "up_limit", "is_suspended"]]
    praw = raw_df.pivot(index="trade_date", columns="ts_code", values=list(_RAW_FIELDS))
    psus = raw_df.pivot(index="trade_date", columns="ts_code", values="is_suspended")

    W = {}
    for field in _WIDE_FIELDS:
        m = pivot[field].reindex(index=dates, columns=syms)
        W[_WIDE_KEYS[field]] = m.to_numpy(dtype=np.float64).T
    for field in _RAW_FIELDS:
        m = praw[field].reindex(index=dates, columns=syms)
        W[_RAW_KEYS[field]] = m.to_numpy(dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        W["VWAP"] = np.where(W["raw_vol"] > 0.0, W["raw_amount"] / np.where(W["raw_vol"] > 0.0, W["raw_vol"], 1.0), np.nan)
    msus = psus.reindex(index=dates, columns=syms).astype("boolean")
    W["suspended"] = msus.fillna(False).to_numpy(dtype=bool).T
    cnt = df.groupby("ts_code", sort=False).size()
    W["n_bars"] = cnt.reindex(syms).fillna(0).astype(np.int64).to_numpy()
    W["dates"] = list(dates)
    W["symbols"] = np.array(syms, dtype=object)
    return W


def decision_wide(context):
    """Bounded decision-day window (DECISION_LOOKBACK_DAYS) as wide matrices."""
    return to_wide(visible_frame(context, DECISION_LOOKBACK_DAYS))


def decision_features(context, W):
    """(S, len(FEATURE_NAMES)) float32 z-scored features of the newest visible bar."""
    raw = features.compute_158(W)
    X_raw = np.stack([raw[name][:, -1] for name in features.ALPHA158], axis=1).astype(np.float32)
    for group in FEATURE_GROUPS:
        block = BLOCKS[group][1](context, W["symbols"]).to_numpy()
        X_raw = np.hstack((X_raw, block.astype(np.float32)))
    return features.robust_zscore(X_raw)


def build_fit_samples(context, block_days=120, overlap_days=60, hold=HOLD_DAYS):
    """Chunked PIT feature + label construction over the trailing training window.

    Blocks of block_days trading days stride by block_days - overlap_days; each block is
    computed on the block plus overlap_days of leading context bars so every kept row has
    its full window. Per decision bar: features z-scored cross-sectionally (axis=1),
    label rows kept only where the label is fully realized and the bar is past the
    WARMUP_BARS warm-up.

    Returns dict: X (N, len(FEATURE_NAMES)) float32, y (N,) float32, date_idx (N,) int64
    (bar position), dates (list of all visible bar dates), symbols (list).
    """
    df = visible_frame(context, FIT_CALENDAR_DAYS)
    W = to_wide(df)
    del df
    # Fit only needs the feature inputs + dates/symbols; drop the raw_* wide matrices
    # before the block loop.
    for k in ("raw_open", "raw_close", "raw_vol", "raw_amount", "raw_up_limit"):
        del W[k]
    T = len(W["dates"])
    S = W["symbols"].shape[0]
    F = len(FEATURE_NAMES)
    if T <= WARMUP_BARS or S == 0:
        raise RuntimeError(f"fit window holds {T} bars and {S} names; need more than {WARMUP_BARS} bars")

    # Extra blocks: PIT panels over the full bar grid (each bar date evaluated at its
    # 08:30 decision time; panel expansion = ffill across trade_date).
    panels = [(BLOCKS[group][0], BLOCKS[group][2](context, W["dates"], W["symbols"])) for group in FEATURE_GROUPS]

    y = labels.build_targets(W, hold=hold)
    tgt, realized = labels.target_matrix(y)
    realized[:, :WARMUP_BARS] = False

    X_rows, y_rows, d_idx = [], [], []
    step = block_days - overlap_days
    n_alpha = len(features.ALPHA158)
    for start in range(0, T, step):
        seg_end = min(T, start + block_days)
        ctx_start = max(0, start - overlap_days)
        # The block COMPUTES up to 180 bars (60 leading context + 120 target) but only
        # KEEPS the first `step` bars (the stride) so every (bar, symbol) sample is stored
        # exactly once; the final block keeps the tail and ends the loop (the frozen artifact
        # kept looping and stored the last partial block twice).
        is_last = start + block_days >= T
        keep_end = seg_end if is_last else min(start + overlap_days, seg_end)
        seg = {k: W[k][:, ctx_start:seg_end].copy()
               for k in ("O", "H", "L", "C", "V", "VWAP")}
        raw = features.compute_158(seg)
        keep0 = start - ctx_start
        m = keep_end - start
        Xs = np.empty((S, m, F), dtype=np.float32)
        for j, name in enumerate(features.ALPHA158):
            Xs[:, :, j] = raw[name][:, keep0:keep0 + m]
            del raw[name]
        col = n_alpha
        for names, panel in panels:
            for name in names:
                Xs[:, :, col] = panel[name][start:keep_end].T
                col += 1
        Xz = features.robust_zscore(Xs.transpose(1, 0, 2))  # (m, S, F) float32
        rows_, cols_ = np.nonzero(realized[:, start:keep_end].T)  # (kept-bar, symbol)
        X_rows.append(Xz[rows_, cols_, :])
        y_rows.append(tgt[cols_, start + rows_].astype(np.float32))  # tgt is (S, T)
        d_idx.append(np.asarray(start + rows_, dtype=np.int64))
        del Xs, Xz, raw, seg
        if is_last:
            break

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    date_idx = np.concatenate(d_idx)
    return {"X": X, "y": y, "date_idx": date_idx,
            "dates": W["dates"], "symbols": W["symbols"].tolist()}
