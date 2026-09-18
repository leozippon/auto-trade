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
TRAIN_YEARS = 2
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

    Blocks stride by step = block_days - overlap_days (60) trading days: block k
    COMPUTES 120 bars (60 leading context + 60 kept, block 0 only its 60) so
    every kept bar has its full d=60 Alpha158 window, and KEEPS its 60 bars
    (the final block, start + step >= T, keeps the tail up to T); a block with
    no realized samples is skipped entirely (the warm-up head and the
    unrealized label tail). The 158 Alpha operators run in two back-to-back
    segments (features.compute_158_part1/_part2, 75 + 87 operators) so at most
    87 (S, T) f64 matrices are alive inside a block. Per decision bar: features z-scored
    cross-sectionally (axis=1), label rows kept only where the label is fully
    realized and the bar is past the WARMUP_BARS warm-up.

    The group blocks (vipf2 / mfflow / newage) are read and written only for
    the groups named in FEATURE_GROUPS: an empty FEATURE_GROUPS makes X hold
    exactly the 158 Alpha columns and no vipf2/mfflow/newage frame or panel is
    read.

    The sample rows are preallocated from the realized matrix (one np.empty per
    array, written in place - no per-block list + final vstack), and the wide
    matrices and the PIT structures are released as soon as the block loop
    ends (dates / symbols / n_bars are kept).

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

    # Extra blocks (PIT, each bar date evaluated at its 08:30 decision time), read
    # only for the groups in FEATURE_GROUPS:
    # vipf2 / mfflow keep the precomputed per-symbol maps and are evaluated per
    # block inside the loop (no full (T, S) panels alive; bars are independent,
    # so per-block evaluation is exactly the restriction of the full-panel one);
    # newage stays a full panel (cheap). The reads reach DECISION_LOOKBACK_DAYS
    # before the first bar (asof guard = context.inference_at).
    dates = W["dates"]
    symbols = W["symbols"]
    t_all = v_maps = m_maps = n_panel = None
    if FEATURE_GROUPS:
        t_all = np.array([vipf2._t_of_bar_date(d) for d in dates], dtype=np.int64)
        span_days = (pd.Timestamp(context.inference_at)
                     - pd.Timestamp(int(t_all[0]), unit="s", tz="UTC")).days + 1
        if "vipf2" in FEATURE_GROUPS:
            v_frame = vipf2.read_fundamentals(context,
                                              lookback=span_days + vipf2.DECISION_LOOKBACK_DAYS)
            v_maps = vipf2._build_maps(v_frame)
            del v_frame
        if "mfflow" in FEATURE_GROUPS:
            m_frame = mfflow.read_mf(context, lookback=span_days + mfflow.DECISION_LOOKBACK_DAYS)
            m_dframe = mfflow.read_daily(context, lookback=span_days + mfflow.DECISION_LOOKBACK_DAYS)
            m_maps = mfflow._build_maps(m_frame, m_dframe)
            del m_frame, m_dframe
        if "newage" in FEATURE_GROUPS:
            n_panel = newage.newage_fit_panels(context, dates, symbols)

    y = labels.build_targets(W, hold=hold)
    tgt, realized = labels.target_matrix(y)
    realized[:, :WARMUP_BARS] = False
    del y
    # The block loop only reads O/H/L/C/V/VWAP; release the rest of the wide
    # matrices now.
    del W["A"], W["suspended"]

    step = block_days - overlap_days
    n_alpha = len(features.ALPHA158)
    # kept bars per block (the final block keeps the tail to T) and the row
    # count each stores, from the realized matrix alone (independent of features)
    blocks = []
    for start in range(0, T, step):
        is_last = start + step >= T
        keep_end = T if is_last else start + step
        ctx_start = max(0, start - overlap_days)
        blocks.append((start, keep_end, ctx_start,
                       int(realized[:, start:keep_end].sum())))
    N = sum(b[3] for b in blocks)
    X = np.empty((N, F), dtype=np.float32)
    y_rows = np.empty((N,), dtype=np.float32)
    d_idx = np.empty((N,), dtype=np.int64)
    cursor = 0
    for start, keep_end, ctx_start, n_rows in blocks:
        if n_rows == 0:
            continue  # warm-up head / unrealized label tail: no samples to store
        m = keep_end - start
        seg = {k: W[k][:, ctx_start:keep_end].copy()
               for k in ("O", "H", "L", "C", "V", "VWAP")}
        keep0 = start - ctx_start
        Xs = np.empty((S, m, F), dtype=np.float32)
        # Two ~79-operator segments back-to-back (memory): at most 87 of the 158
        # (S, T) f64 operator matrices are alive at once instead of all 158; the
        # segments share no operator, so the stored columns are unchanged.
        raw = features.compute_158_part1(seg)
        for j, name in enumerate(features.ALPHA158_PART1_NAMES):
            Xs[:, :, j] = raw[name][:, keep0:keep0 + m]
            del raw[name]
        del raw
        raw = features.compute_158_part2(seg)
        for j, name in enumerate(features.ALPHA158_PART2_NAMES):
            Xs[:, :, len(features.ALPHA158_PART1_NAMES) + j] = raw[name][:, keep0:keep0 + m]
            del raw[name]
        del raw
        # Group block columns 158.., written only for the groups in FEATURE_GROUPS
        # (col always stays within Xs' axis-2 size by construction).
        col = n_alpha
        vP = mP = None
        if "vipf2" in FEATURE_GROUPS or "mfflow" in FEATURE_GROUPS:
            t_block = t_all[start:keep_end]
        if "vipf2" in FEATURE_GROUPS:
            vP = vipf2._evaluate_maps(v_maps, symbols, t_block)
            for name in vipf2.FEATURES:
                Xs[:, :, col] = vP[name].T
                col += 1
        if "mfflow" in FEATURE_GROUPS:
            mP = mfflow._evaluate_maps(m_maps, symbols, t_block)
            for name in mfflow.FEATURES:
                Xs[:, :, col] = mP[name].T
                col += 1
        if "newage" in FEATURE_GROUPS:
            for name in newage.FEATURES:
                Xs[:, :, col] = n_panel[name][start:keep_end].T
                col += 1
        del seg, vP, mP
        Xz = features.robust_zscore(Xs.transpose(1, 0, 2))  # (m, S, F) float32
        rows_, cols_ = np.nonzero(realized[:, start:keep_end].T)  # (kept-bar, symbol)
        X[cursor:cursor + n_rows] = Xz[rows_, cols_, :]
        y_rows[cursor:cursor + n_rows] = tgt[cols_, start + rows_].astype(np.float32)  # tgt is (S, T)
        d_idx[cursor:cursor + n_rows] = np.asarray(start + rows_, dtype=np.int64)
        cursor += n_rows
        del Xs, Xz
    # Release the PIT structures and the wide matrices now that the samples are stored.
    del v_maps, m_maps, n_panel, t_all, tgt, realized
    for k in ("O", "H", "L", "C", "V", "VWAP"):
        del W[k]
    return {"X": X, "y": y_rows, "date_idx": d_idx,
            "dates": W["dates"], "symbols": W["symbols"].tolist()}
