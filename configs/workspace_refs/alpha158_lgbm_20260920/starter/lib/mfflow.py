"""mfflow: 8-column moneyflow PIT feature block (Fold 5 R2, candidate mfp =
mf_persist@topk16; foundation topk16 + moneyflow re-entry axis, prereg_round2).

Dataset (events domain; single dataset, per-day latest visibility):
  moneyflow  per-trading-day main-force money flow (万元): net_mf_amount,
             buy_elg_amount / sell_elg_amount (extra-large), buy_lg_amount /
             sell_lg_amount (large); available_at = trade_date 19:00+08:00
             (rule official_19_from:trade_date, verified 100% on the audited
             window). ~1 row per stock per trading day, near-full coverage;
             0 same-(ts_code, trade_date) duplicate groups in the 2021-01..
             2022-12 window (dedup rule below is kept for contract safety and
             is verified on synthetic rows).

Features (raw values; appended AFTER the 158 Alpha + 6 vipF2 columns, the same
172-column RobustZScore; T = decision day 08:30+08:00; amounts 万元 x 1e4 -> CNY):
  mf_amt_5     sum of net_mf_amount x 1e4 over the 5 calendar days before T,
               divided by circ_mv (no visible row in window -> NaN)
  mf_amt_20    same over the 20 calendar days (no visible row -> NaN)
  mf_pos20     count of days with net_mf_amount > 0 inside the 20 calendar
               days, divided by 20 (persistence; no visible row -> NaN)
  mf_trend520  mf_amt_5 / (mf_amt_20 / 4) (5d flow vs 20d mean speed;
               |mf_amt_20 / 4| <= 1e-8 -> NaN)
  mf_vola5_20  population std (ddof=0) of the 5d per-day series / population
               std of the 20d series, element = that day's
               net_mf_amount x 1e4 / that day's own daily-bar circ_mv
               (same-day daily row; missing/<=0/NaN circ_mv -> element NaN);
               20d visible days < 2 -> NaN; 20d std <= 0 (incl. < 2 finite
               elements) -> NaN; 5d window empty (or no finite element) -> NaN
  mf_elg_net5  sum of (buy_elg_amount - sell_elg_amount) x 1e4 over the 5
               calendar days / circ_mv (no visible row -> NaN)
  mf_lg_net20  sum of (buy_lg_amount - sell_lg_amount) x 1e4 over the 20
               calendar days / circ_mv (no visible row -> NaN)
  mf_amtturn20 sum of net_mf_amount x 1e4 over the 20 calendar days / sum of
               daily.amount over the same 20 calendar days (flow / turnover;
               no visible moneyflow row -> NaN; denominator <= 0 -> NaN)

Calendar-day windows: "the K calendar days before T" = visible rows with
trade_date >= T - K days, i.e. {T-K, ..., T-1}; visible rows can never carry
trade_date == T at a 08:30 decision (availability is T 19:00 > T 08:30).

PIT hard rules (parent vipf2 conventions, ported):
  - A row is visible at decision time t iff available_at <= t. The moneyflow
    same-day-19:00 rule makes trade_date <= T-1 automatic at 08:30; the
    visibility filter is available_at ONLY - no bare `trade_date < T`
    shortcut is ever used (a trade_date == T row is excluded solely because
    its available_at (T 19:00) > t).
  - Same (ts_code, trade_date) multiple rows: the in-effect row at t is the
    max available_at version <= t; identical available_at resolves to the
    last file occurrence under a stable sort (vipf2._end_versions convention). The dedup happens BEFORE the window masks.
  - Window-empty cells -> NaN (not 0): with near-full moneyflow coverage a 0
    convention would create an atomic-zero / median-atom mass that flattens
    the column under the parent RobustZScore (the fc_dir mechanism). NaN keeps the cross-sectional statistics over the
    row-bearing symbols and fills the no-row symbols with 0 AFTER zscoring.
  - Daily join (circ_mv / amount): per bar date d the LAST VISIBLE daily row
    is the row with the max trade_date < d (the same root bar as the parent
    universe(); daily PIT = trade_date < T, already guaranteed by the asof
    view + read filter). Scalar features use that bar's circ_mv (missing /
    NaN / <= 0 -> NaN by the denominator rule). mf_vola5_20 uses each day's
    OWN daily-bar circ_mv; mf_amtturn20 sums the daily-bar amount over the
    20 calendar days (suspended days simply absent from the sum).
  - Ratio features with a NaN or <= 0 denominator -> NaN.
  - A NaN in a moneyflow amount propagates to the affected cell (the audited
    window has 0 nulls in the 5 amount columns; the rule is recorded).

Only data sources: context.asof_dir + "/events" and + "/daily" (directories
of parquet parts, same layout as the parent domains). Reads are bounded: the
decision path uses a DECISION_LOOKBACK_DAYS calendar-day window
(pit-read-budget, 120d covers the 20d feature window + margin; moneyflow is
near-fully covered so no long history is needed), the fit path the same window
before its first training bar. The decision path and the fit panels share _evaluate (single
source of truth; the naive cross-check reference is the only other
implementation of these semantics).
"""

from datetime import timedelta

import numpy as np
import pandas as pd

DS_MF = "moneyflow"
DATASETS = (DS_MF,)

# Parquet projection for read_mf (spec-locked): the dataset pin column +
# the columns used by the 8 features.
READ_COLS = ["dataset", "ts_code", "trade_date", "net_mf_amount",
             "buy_elg_amount", "sell_elg_amount", "buy_lg_amount",
             "sell_lg_amount", "available_at"]

# The 8 feature columns, in the exact order appended after the 158 Alpha +
# 6 vipF2 columns. data.FEATURE_NAMES is built from it, for fit and decision alike.
FEATURES = ("mf_amt_5", "mf_amt_20", "mf_pos20", "mf_trend520",
            "mf_vola5_20", "mf_elg_net5", "mf_lg_net20", "mf_amtturn20")

# Decision-path available_at window (calendar days): 20d feature window +
# margin (moneyflow is near-fully covered, no long history needed).
DECISION_LOOKBACK_DAYS = 120

# Daily join projection (per-bar last-visible row + 20d amount window).
DAILY_COLS = ["ts_code", "trade_date", "circ_mv", "amount"]

_DAY_NS = 86400 * 10**9
_DAY_S = 86400
_TREND_EPS = 1e-8
_W5 = 5
_W20 = 20


def _prep_frame(df, asof_ts, lookback):
    """PIT filter + normalization of the raw moneyflow frame.

    Keeps rows with available_at in [asof_ts - lookback, asof_ts]. File order is
    preserved (deterministic stable ties, the vipf2 convention)."""
    end = pd.Timestamp(asof_ts)
    av = pd.to_datetime(df["available_at"])
    m = (av <= end) & (av >= end - pd.Timedelta(days=int(lookback)))
    if not bool(m.all()):
        df = df.loc[m].reset_index(drop=True)
    out = df.copy()
    out["ts_code"] = out["ts_code"].astype(str)
    out["trade_date"] = out["trade_date"].astype(str).str.zfill(8)
    out["avail_s"] = (pd.to_datetime(out["available_at"]).astype("int64")
                      // 10**9).to_numpy(dtype=np.int64)
    out["trade_ord"] = (pd.to_datetime(out["trade_date"], format="%Y%m%d")
                        .astype("int64") // _DAY_NS).to_numpy(dtype=np.int64)
    for c in ("net_mf_amount", "buy_elg_amount", "sell_elg_amount",
              "buy_lg_amount", "sell_lg_amount"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_mf(context, lookback):
    """Visible moneyflow rows from the asof view (directory of parquet parts).

    The dataset pin goes into both the pushdown filter and the column projection
    (multi-dataset files share ts_code); the trade_date pushdown bounds the read
    (a row stamped trade_date 19:00 is inside the available_at window exactly
    when its trade_date is on or after the window's first day)."""
    start = (context.inference_at - timedelta(days=int(lookback))).strftime("%Y%m%d")
    df = pd.read_parquet(context.asof_dir + "/events", columns=READ_COLS,
                         filters=[("dataset", "in", DATASETS), ("trade_date", ">=", start)])
    return _prep_frame(df, context.inference_at, lookback)


def read_daily(context, lookback):
    """Visible daily rows (ts_code, trade_date, circ_mv, amount).

    Daily PIT = trade_date < T (same root-bar convention as data.visible_frame
    / parent universe()); lookback bounds the read from below."""
    t = context.inference_at.strftime("%Y%m%d")
    start = (context.inference_at - timedelta(days=int(lookback))).strftime("%Y%m%d")
    df = pd.read_parquet(
        context.asof_dir + "/daily", columns=DAILY_COLS,
        filters=[("trade_date", ">=", start)],
    )
    df = df[df["trade_date"] < t]
    if df.empty:
        return df
    out = df.reset_index(drop=True).copy()
    out["ts_code"] = out["ts_code"].astype(str)
    out["circ_mv"] = pd.to_numeric(out["circ_mv"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["date_ord"] = (pd.to_datetime(out["trade_date"], format="%Y%m%d")
                       .astype("int64") // _DAY_NS).to_numpy(dtype=np.int64)
    return out


def _t_of_bar_date(d):
    """08:30+08:00 epoch seconds for a YYYYMMDD bar date (the decision time)."""
    return int(pd.Timestamp("%s-%s-%s 08:30:00+08:00" % (d[:4], d[4:6], d[6:8]))
               .value // 10**9)


def _pstd(s1, s2, cnt):
    """Population std (ddof=0) per bar from sum/sumsq/counts; 0 where cnt==0,
    tiny negative variance (FP round-off) clipped to 0."""
    c = np.where(cnt > 0, cnt, 1)
    mu = s1 / c
    var = s2 / c - mu * mu
    var = np.where(var > 0.0, var, 0.0)
    return np.sqrt(var)


def _evaluate(frame, dframe, symbols, t_arr):
    """Evaluate the 8 feature columns at query times t_arr (int64 epoch seconds).

    frame: normalized moneyflow frame (read_mf output, may be empty); dframe:
    normalized daily frame (read_daily output, may be empty); symbols: ordered
    symbol list; t_arr: (m,) int64 epoch seconds. Returns {feature: (m, S)
    float64} aligned to symbols - the single shared constructor of the
    decision path (m = 1) and the fit path (m = len(dates)). Window-empty /
    no-row cells are NaN; RobustZScore turns them into 0 AFTER the
    cross-sectional median/MAD is computed."""
    t_arr = np.asarray(t_arr, dtype=np.int64)
    m = len(t_arr)
    S = len(symbols)
    P = {n: np.full((m, S), np.nan, dtype=np.float64) for n in FEATURES}
    if S == 0 or m == 0:
        return P
    if frame.empty:
        return P
    # Date ordinal (days since epoch) of each query time: t is +08:00, so the
    # calendar date is (t + 8h) floored to the day in UTC.
    T_ord = (t_arr + 8 * 3600) // _DAY_S

    day_map = {}
    if dframe is not None and not dframe.empty:
        for ts, g in dframe.groupby("ts_code", sort=False):
            g = g.sort_values("date_ord", kind="stable")
            day_map[ts] = (g["date_ord"].to_numpy(dtype=np.int64),
                           g["circ_mv"].to_numpy(dtype=np.float64),
                           g["amount"].to_numpy(dtype=np.float64))
    mf_map = {ts: g for ts, g in frame[frame["dataset"] == DS_MF].groupby("ts_code", sort=False)}

    for si, ts in enumerate(symbols):
        d = day_map.get(ts)
        if d is not None:
            do, dmv, damt = d
            ci = np.searchsorted(do, T_ord, "left") - 1  # last row with trade_date < T
            cmv = np.where(ci >= 0, dmv[np.clip(ci, 0, None)], np.nan)
        else:
            do = dmv = damt = None
            cmv = np.full(m, np.nan)

        g = mf_map.get(ts)
        if g is None:
            continue  # all 8 features stay NaN
        n = len(g)
        od = g["trade_ord"].to_numpy(dtype=np.int64)
        av = g["avail_s"].to_numpy(dtype=np.int64)
        net = g["net_mf_amount"].to_numpy(dtype=np.float64)
        elg = (g["buy_elg_amount"].to_numpy(dtype=np.float64)
               - g["sell_elg_amount"].to_numpy(dtype=np.float64))
        lg = (g["buy_lg_amount"].to_numpy(dtype=np.float64)
              - g["sell_lg_amount"].to_numpy(dtype=np.float64))
        # canonical row order (trade_date, available_at, file order)
        o = np.lexsort((np.arange(n), av, od))
        od, av, net, elg, lg = od[o], av[o], net[o], elg[o], lg[o]
        # same-(ts, trade_date) dedup: sorted by (od, av), so the last
        # occurrence in each od group is the max-available_at version
        # (stable file-order tie -> last wins; vipf2._prep_frame convention)
        _, fi = np.unique(od, return_index=True)
        keep = np.concatenate((fi[1:] - 1, np.array([n - 1], dtype=np.int64)))
        od, av, net, elg, lg = od[keep], av[keep], net[keep], elg[keep], lg[keep]
        n2 = len(od)
        vis = av[:, None] <= t_arr[None, :]
        w5 = vis & (od[:, None] >= (T_ord - _W5)[None, :])
        w20 = vis & (od[:, None] >= (T_ord - _W20)[None, :])
        c5 = w5.sum(axis=0)
        c20 = w20.sum(axis=0)
        s5net = (net[:, None] * w5).sum(axis=0)
        s20net = (net[:, None] * w20).sum(axis=0)
        s5elg = (elg[:, None] * w5).sum(axis=0)
        s20lg = (lg[:, None] * w20).sum(axis=0)
        pos20 = (w20 & (net[:, None] > 0.0)).sum(axis=0)
        cmv_ok = np.isfinite(cmv) & (cmv > 0.0)
        cmv_safe = np.where(cmv_ok, cmv, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            a5 = s5net * 1e4 / cmv_safe
            a20 = s20net * 1e4 / cmv_safe
            e5 = s5elg * 1e4 / cmv_safe
            l20 = s20lg * 1e4 / cmv_safe
            a20q = a20 / 4.0
        P["mf_amt_5"][:, si] = np.where((c5 > 0) & cmv_ok, a5, np.nan)
        P["mf_amt_20"][:, si] = np.where((c20 > 0) & cmv_ok, a20, np.nan)
        P["mf_pos20"][:, si] = np.where(c20 > 0, pos20 / float(_W20), np.nan)
        P["mf_trend520"][:, si] = np.where(
            (c5 > 0) & (c20 > 0) & cmv_ok & np.isfinite(a5) & np.isfinite(a20q)
            & (np.abs(a20q) > _TREND_EPS),
            a5 / np.where(np.abs(a20q) > _TREND_EPS, a20q, 1.0), np.nan)
        P["mf_elg_net5"][:, si] = np.where((c5 > 0) & cmv_ok, e5, np.nan)
        P["mf_lg_net20"][:, si] = np.where((c20 > 0) & cmv_ok, l20, np.nan)
        # ---- mf_vola5_20: per-day element = net x 1e4 / same-day circ_mv ----
        if do is not None and len(do) > 0:
            mi = np.searchsorted(do, od, "left")
            mi_c = np.clip(mi, 0, len(do) - 1)
            match = (mi < len(do)) & (do[mi_c] == od)
            cmvd = np.where(match, dmv[mi_c], np.nan)
        else:
            cmvd = np.full(n2, np.nan)
        okd = np.isfinite(cmvd) & (cmvd > 0.0)
        e = np.where(okd, net * 1e4 / np.where(okd, cmvd, 1.0), np.nan)
        f5 = okd[:, None] & w5
        f20 = okd[:, None] & w20
        n5f = f5.sum(axis=0)
        n20f = f20.sum(axis=0)
        e1, e2 = e[:, None], (e[:, None]) ** 2
        std5 = _pstd((np.where(f5, e1, 0.0)).sum(axis=0),
                     (np.where(f5, e2, 0.0)).sum(axis=0), n5f)
        std20 = _pstd((np.where(f20, e1, 0.0)).sum(axis=0),
                      (np.where(f20, e2, 0.0)).sum(axis=0), n20f)
        okv = ((c20 >= 2) & (n20f >= 1) & (std20 > 0.0) & (c5 > 0)
               & (n5f >= 1) & np.isfinite(std20) & np.isfinite(std5))
        P["mf_vola5_20"][:, si] = np.where(okv, std5 / np.where(okv, std20, 1.0), np.nan)
        # ---- mf_amtturn20: sum(net x 1e4) / sum(daily.amount) over 20d ----
        if do is not None and len(do) > 0:
            dw = (do[:, None] >= (T_ord - _W20)[None, :]) & (do[:, None] < T_ord[None, :])
            amt20 = (damt[:, None] * dw).sum(axis=0)
        else:
            amt20 = np.zeros(m, dtype=np.float64)
        den_ok = np.isfinite(amt20) & (amt20 > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            at20 = s20net * 1e4 / np.where(den_ok, amt20, 1.0)
        P["mf_amtturn20"][:, si] = np.where((c20 > 0) & den_ok, at20, np.nan)
    return P


def mfflow_frame(context, symbols):
    """Decision-path entry: rebalance day t x symbols -> 8 raw feature columns.

    t = context.inference_at (08:30+08:00). Returns a float64 DataFrame
    indexed by symbols, column order = FEATURES (the order appended after
    the 158 Alpha + 6 vipF2 columns), shape/semantics aligned with
    vipf2_frame."""
    t = int(pd.Timestamp(context.inference_at).value // 10**9)
    frame = read_mf(context, lookback=DECISION_LOOKBACK_DAYS)
    dframe = read_daily(context, lookback=DECISION_LOOKBACK_DAYS)
    P = _evaluate(frame, dframe, list(symbols), np.array([t], dtype=np.int64))
    return pd.DataFrame({n: P[n][0] for n in FEATURES},
                        index=np.asarray(symbols, dtype=object))


def mfflow_fit_panels(context, dates, symbols):
    """Fit-path entry: 8 panels over the fit window's bar grid (panel expansion).

    Each bar date is evaluated at its 08:30+08:00 decision time, the same
    semantics as the decision path (_evaluate is the shared constructor). The
    reads reach DECISION_LOOKBACK_DAYS before the first bar (asof guard =
    context.inference_at). Returns {feature: (T, S) float64} aligned to
    dates/symbols, same shape/key order as vipf2_fit_panels."""
    t_arr = np.array([_t_of_bar_date(d) for d in dates], dtype=np.int64)
    span_days = (pd.Timestamp(context.inference_at) - pd.Timestamp(int(t_arr[0]), unit="s", tz="UTC")).days + 1
    frame = read_mf(context, lookback=span_days + DECISION_LOOKBACK_DAYS)
    dframe = read_daily(context, lookback=span_days + DECISION_LOOKBACK_DAYS)
    return _evaluate(frame, dframe, list(symbols), t_arr)
