"""vipF2: 6-column growth / forecast-event PIT feature block (Fold 4 R2, candidate C).

Ported item-by-item from the validated screen-panel implementation
(notes/screen_signals/prep.py; panels panel_sig_{or_yoy,netprofit_yoy,
dt_netprofit_yoy,op_yoy,fc_dir,fc_exp_rev}.parquet), so values are identical by
construction; see the d3 cross-validation report.

Features (raw values; appended AFTER the 158 Alpha columns, same RobustZScore):
  or_yoy / netprofit_yoy / dt_netprofit_yoy / op_yoy
      fina_indicator_vip: in-effect version (max available_at <= t) at time t of
      the latest visible end (end with the max first availability <= t), source
      percent -> decimal (/100); no visible row -> NaN.
  fc_dir
      latest visible forecast_vip row (any end_date, max available_at <= t):
      0 when the actual income_vip report of that end_date is already visible
      (first income availability <= t); else type in {预增,扭亏,略增} -> +1,
      type in {预减,首亏,略减} -> -1, any other/unknown type -> 0; no forecast -> 0.
  fc_exp_rev
      latest visible express_vip row, paired with the forecast of the SAME
      (ts_code, end_date): fc_mid = ((net_profit_min + net_profit_max) / 2) /
      last_parent_net - 1 (10k_CNY self-consistent; 0 unless all three finite and
      |last_parent_net| > 0); exp_gr = (n_income - yoy_net_profit) /
      |yoy_net_profit| (CNY self-consistent; 0 unless both finite and
      yoy_net_profit != 0); +1 if exp_gr > fc_mid, -1 if exp_gr < fc_mid, 0 if ==.
      No express row, or no forecast row for that end -> 0.
      Units stay self-consistent per feature (万元/万元, 元/元); never mixed.

PIT hard rules (identical to prep.py):
  - a row is visible at decision time t (decision day 08:30+08:00) iff
    available_at <= t; available_at is announcement-day 18:00, i.e. exactly
    ann_date <= T-1 at a 08:30 decision;
  - same (dataset, ts_code, end_date) multi-version: the in-effect version at t
    is the max available_at version <= t; fully duplicated rows (identical
    available_at) resolve to the last file occurrence under a stable sort;
  - missing fundamentals never fabricate a value: fi columns stay NaN (RobustZScore
    maps them to 0, the same neutral as the Alpha columns), the event columns are
    0 by construction (no forecast / no pair / actual already disclosed).

income_vip rows carry no numeric values here: they only mark the first visibility
time of each (ts_code, end_date) for the fc_dir "actual already disclosed" rule.
Only data source: context.asof_dir + "/fundamentals" (directory of parquet parts,
same layout as the daily domain). Both reads are bounded by available_at: the
decision path keeps a DECISION_LOOKBACK_DAYS calendar-day window, the fit path the
same window before its first training bar; the parquet read itself is cut on
ann_date ANN_DATE_MARGIN_DAYS earlier still, which only drops announcements that
reached the lake more than that long after they were made.
"""

import numpy as np
import pandas as pd

DS_FI = "fina_indicator_vip"
DS_FC = "forecast_vip"
DS_EX = "express_vip"
DS_IN = "income_vip"
DATASETS = (DS_FI, DS_FC, DS_EX, DS_IN)

# Parquet projection for read_fundamentals (spec-locked). ann_date / p_change_* /
# revenue are read for contract completeness but unused by the 6 features.
READ_COLS = ["dataset", "ts_code", "end_date", "ann_date", "available_at",
             "or_yoy", "netprofit_yoy", "dt_netprofit_yoy", "op_yoy", "type",
             "p_change_min", "p_change_max", "net_profit_min", "net_profit_max",
             "last_parent_net", "n_income", "yoy_net_profit", "revenue"]

# The 6 feature columns, in the exact order appended after the 158 Alpha columns.
# data.FEATURE_NAMES is built from it, for fit and decision alike.
FEATURES = ("or_yoy", "netprofit_yoy", "dt_netprofit_yoy", "op_yoy",
            "fc_dir", "fc_exp_rev")

FI_COLS = ["or_yoy", "netprofit_yoy", "dt_netprofit_yoy", "op_yoy"]  # percent -> /100
FC_NUM = ["net_profit_min", "net_profit_max", "last_parent_net"]     # 10k_CNY
EX_NUM = ["n_income", "yoy_net_profit"]                              # CNY

FC_UP = {"预增": 1.0, "扭亏": 1.0, "略增": 1.0}
FC_DOWN = {"预减": -1.0, "首亏": -1.0, "略减": -1.0}

DECISION_LOOKBACK_DAYS = 1100  # decision-path available_at window (calendar days)
ANN_DATE_MARGIN_DAYS = 400     # parquet pushdown on ann_date, this much wider than the window


def _prep_frame(df, asof_ts, lookback):
    """PIT filter + normalization of a raw fundamentals frame (any datasets).

    Keeps rows with available_at in [asof_ts - lookback, asof_ts]. File order is
    preserved (deterministic stable ties)."""
    end = pd.Timestamp(asof_ts)
    av = pd.to_datetime(df["available_at"])
    m = (av <= end) & (av >= end - pd.Timedelta(days=int(lookback)))
    if not bool(m.all()):
        df = df.loc[m].reset_index(drop=True)
    out = df.copy()
    out["ts_code"] = out["ts_code"].astype(str)
    out["end_date"] = out["end_date"].astype(str).str.zfill(8)
    out["avail_s"] = (pd.to_datetime(out["available_at"]).astype("int64")
                      // 10**9).to_numpy(dtype=np.int64)
    for c in FI_COLS + FC_NUM + EX_NUM + ["revenue"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_fundamentals(context, lookback):
    """Visible fundamentals rows from the asof view (directory of parquet parts).

    The dataset pin goes into both the pushdown filter and the column projection
    (multi-dataset files share ts_code); the ann_date pushdown bounds the read."""
    start = (context.inference_at - pd.Timedelta(days=int(lookback) + ANN_DATE_MARGIN_DAYS)).strftime("%Y%m%d")
    df = pd.read_parquet(context.asof_dir + "/fundamentals", columns=READ_COLS,
                         filters=[("dataset", "in", DATASETS), ("ann_date", ">=", start)])
    return _prep_frame(df, context.inference_at, lookback)


def _end_versions(g, cols):
    """Per-end version arrays (port of prep.py EndDS.__init__): stable sort by
    available_at inside each end. Returns (ends ordered by (first availability,
    end_date), ver, fvt, first_vis)."""
    ver = {}
    for e, ge in g.groupby("end_date", sort=True):
        ge = ge.sort_values("avail_s", kind="stable")
        ver[e] = (ge["avail_s"].to_numpy(dtype=np.int64),
                  ge[cols].to_numpy(dtype=np.float64))
    ends = sorted(ver.keys(), key=lambda e: (ver[e][0][0], e))
    fvt = np.array([ver[e][0][0] for e in ends], dtype=np.int64)
    first_vis = {e: ver[e][0][0] for e in ends}
    return ends, ver, fvt, first_vis


def _end_table(ends, ver, fvt, t_arr, ncol):
    """C[i, j] = in-effect value of end i at query time j (NaN if not yet
    visible); L[j] = index of the latest visible end at query j (-1 if none).
    Port of prep.py EndDS.table + latest_col index selection."""
    m = len(t_arr)
    C = np.full((len(ends), m, ncol), np.nan)
    for i, e in enumerate(ends):
        a, v = ver[e]
        mask = t_arr >= a[0]
        if not mask.any():
            continue
        rows_i = np.nonzero(mask)[0]
        idx = np.searchsorted(a, t_arr[rows_i], "right") - 1
        C[i, rows_i, :] = v[idx]
    L = np.searchsorted(fvt, t_arr, "right") - 1
    return C, L


def _flat_rows(g, cols):
    """Flat per-ts rows (port of prep.py FlatDS.__init__): stable sort by
    (available_at, end_date). Returns (avail_s, end_date, vals, type)."""
    g = g.sort_values(["avail_s", "end_date"], kind="stable")
    return (g["avail_s"].to_numpy(dtype=np.int64),
            g["end_date"].to_numpy(),
            g[cols].to_numpy(dtype=np.float64),
            g["type"].to_numpy())


def _evaluate(frame, symbols, t_arr):
    """Evaluate the 6 feature columns at query times t_arr (int64 epoch seconds).

    frame: normalized fundamentals frame (read_fundamentals output); symbols:
    ordered symbol list; t_arr: (m,) int64 epoch seconds. Returns
    {feature: (m, len(symbols)) float64} aligned to symbols. Port of the
    per-symbol block of prep.py main(), evaluated directly at the query times
    (equivalent to prep's breakpoint build + searchsorted day mapping)."""
    t_arr = np.asarray(t_arr, dtype=np.int64)
    m = len(t_arr)
    S = len(symbols)
    P = {n: (np.zeros((m, S), dtype=np.float64) if n in ("fc_dir", "fc_exp_rev")
             else np.full((m, S), np.nan, dtype=np.float64)) for n in FEATURES}
    if frame.empty or S == 0 or m == 0:
        return P
    fi_map = {ts: g for ts, g in frame[frame["dataset"] == DS_FI].groupby("ts_code", sort=False)}
    fc_map = {ts: g for ts, g in frame[frame["dataset"] == DS_FC].groupby("ts_code", sort=False)}
    ex_map = {ts: g for ts, g in frame[frame["dataset"] == DS_EX].groupby("ts_code", sort=False)}
    inc_map = {ts: g for ts, g in frame[frame["dataset"] == DS_IN].groupby("ts_code", sort=False)}
    for si, ts in enumerate(symbols):
        fi_g = fi_map.get(ts)
        fc_g = fc_map.get(ts)
        ex_g = ex_map.get(ts)
        if fi_g is None and fc_g is None and ex_g is None:
            continue  # defaults: fi NaN, fc_dir 0, fc_exp_rev 0
        # ---- fi 4 columns: latest visible end, in-effect version, percent / 100 ----
        if fi_g is not None:
            ends, ver, fvt, _fv = _end_versions(fi_g, FI_COLS)
            C, L = _end_table(ends, ver, fvt, t_arr, len(FI_COLS))
            ok = L >= 0
            jj = np.nonzero(ok)[0]
            for c, n in enumerate(FI_COLS):
                v = np.full(m, np.nan)
                v[jj] = C[L[jj], jj, c] / 100.0
                P[n][:, si] = v
        # ---- fc_dir: latest visible forecast row (any end) ----
        fc_flat = _flat_rows(fc_g, FC_NUM) if fc_g is not None else None
        if fc_flat is not None:
            avail, end, _vals, typ = fc_flat
            k = np.searchsorted(avail, t_arr, "right") - 1
            inc_g = inc_map.get(ts)
            inc_fv = ({e: int(g2["avail_s"].min())
                       for e, g2 in inc_g.groupby("end_date", sort=False)}
                      if inc_g is not None else {})
            out = np.zeros(m)
            for r in np.nonzero(k >= 0)[0]:
                fv = inc_fv.get(end[k[r]])
                if fv is not None and fv <= t_arr[r]:
                    continue  # actual income already disclosed -> 0
                tt = typ[k[r]]
                if tt in FC_UP:
                    out[r] = 1.0
                elif tt in FC_DOWN:
                    out[r] = -1.0
            P["fc_dir"][:, si] = out
        # ---- fc_exp_rev: latest visible express row paired with same (ts, end) ----
        if fc_flat is not None and ex_g is not None:
            ex_avail, ex_end, ex_vals, _et = _flat_rows(ex_g, EX_NUM)
            kx = np.searchsorted(ex_avail, t_arr, "right") - 1
            f_ends, f_ver, f_fvt, _ffv = _end_versions(fc_g, FC_NUM)
            Cf, _Lf = _end_table(f_ends, f_ver, f_fvt, t_arr, len(FC_NUM))
            pos_fce = {e: i for i, e in enumerate(f_ends)}
            ex_ni = ex_vals[:, 0]
            ex_y = ex_vals[:, 1]
            out = np.zeros(m)
            for r in np.nonzero(kx >= 0)[0]:
                i = pos_fce.get(ex_end[kx[r]])
                if i is None:
                    continue  # no forecast row for this end -> 0
                a, b, c0 = Cf[i, r, 0], Cf[i, r, 1], Cf[i, r, 2]
                if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c0)
                        and abs(c0) > 0):
                    fc_mid = 0.0
                else:
                    fc_mid = (a + b) / 2.0 / c0 - 1.0
                e_ni, e_yoy = ex_ni[kx[r]], ex_y[kx[r]]
                if not (np.isfinite(e_ni) and np.isfinite(e_yoy)) or e_yoy == 0:
                    exp_gr = 0.0
                else:
                    exp_gr = (e_ni - e_yoy) / abs(e_yoy)
                if exp_gr > fc_mid:
                    out[r] = 1.0
                elif exp_gr < fc_mid:
                    out[r] = -1.0
            P["fc_exp_rev"][:, si] = out
    return P


def _t_of_bar_date(d):
    """08:30+08:00 epoch seconds for a YYYYMMDD bar date (the decision time)."""
    return int(pd.Timestamp("%s-%s-%s 08:30:00+08:00" % (d[:4], d[4:6], d[6:8]))
               .value // 10**9)


def vipf2_frame(context, symbols):
    """Decision-path entry: rebalance day t x symbols -> 6 raw feature columns.

    t = context.inference_at (08:30+08:00). Returns a float64 DataFrame indexed
    by symbols, column order = FEATURES (the order appended after the 158)."""
    t = int(pd.Timestamp(context.inference_at).value // 10**9)
    frame = read_fundamentals(context, lookback=DECISION_LOOKBACK_DAYS)
    P = _evaluate(frame, list(symbols), np.array([t], dtype=np.int64))
    return pd.DataFrame({n: P[n][0] for n in FEATURES},
                        index=np.asarray(symbols, dtype=object))


def vipf2_fit_panels(context, dates, symbols):
    """Fit-path entry: 6 panels over the fit window's bar grid (panel expansion).

    Fundamental rows become visible at their available_at and are forward-carried
    (ffill) across the trade_date grid: bar date d is evaluated at the decision
    time d 08:30+08:00, the same semantics as the decision path. The read reaches
    DECISION_LOOKBACK_DAYS before the first bar (asof guard = context.inference_at).
    Returns {feature: (T, S) float64} aligned to dates/symbols."""
    t_arr = np.array([_t_of_bar_date(d) for d in dates], dtype=np.int64)
    span_days = (pd.Timestamp(context.inference_at) - pd.Timestamp(int(t_arr[0]), unit="s", tz="UTC")).days + 1
    frame = read_fundamentals(context, lookback=span_days + DECISION_LOOKBACK_DAYS)
    return _evaluate(frame, list(symbols), t_arr)
