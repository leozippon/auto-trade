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
      latest visible forecast row (any end_date, max available_at <= t):
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

Evaluation (perf, semantics unchanged): per-symbol structures are precomputed
once in numpy (per-dataset stable (symbol, ...) sorts; per-end version tables in
(first availability, end_date) order; flat (available_at, end_date) rows) and
then evaluated vectorized over all query times at once, with no pandas inside
the symbol loop. The per-query selection is exactly the old searchsorted-based
one (latest visible end -> in-effect version; latest visible flat row -> type
map / express-forecast pair), so the outputs are identical to the per-symbol
reference implementation.
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


def _reorder_ends(be, ba, bv):
    """Per-symbol end reorder (port of the prep.py EndDS end ordering).

    be: (sz,) int64 end per row, rows already in stable (end, available_at)
    order; ba: (sz,) int64 availability; bv: (sz, k) values. Returns (fvt (E,)
    int64, starts (E+1,) int64, fa (sz,) int64, fv (sz, k), ue (E,) int64) with
    the end groups in (first availability, end_date) order - exactly the order
    `sorted(ver.keys(), key=lambda e: (ver[e][0][0], e))` produced - and each
    group's rows in stable available_at order."""
    ue, sp = np.unique(be, return_index=True)
    fvt = ba[sp]
    eo = np.lexsort((ue, fvt))  # (first availability, end_date)
    seg = np.concatenate((sp, np.array([be.size], dtype=np.int64)))
    gsz = (seg[1:] - seg[:-1])[eo]
    starts = np.concatenate(([0], np.cumsum(gsz)))
    idx = np.concatenate([np.arange(seg[eo[k]], seg[eo[k] + 1]) for k in range(len(eo))])
    return fvt[eo], starts, ba[idx], bv[idx], ue[eo]


def _type_map(typ):
    """(n,) float64: +1 / -1 / 0 per forecast type, unknown type -> 0."""
    tm = np.zeros(typ.shape, dtype=np.float64)
    for s, val in FC_UP.items():
        tm[typ == s] = val
    for s, val in FC_DOWN.items():
        tm[typ == s] = val
    return tm


def _build_maps(frame):
    """Per-dataset per-symbol numpy structures, keyed by ts_code string.

    All sorts are stable (file order preserved on ties); end_date is zero-padded
    YYYYMMDD, so integer and string order coincide. Returns (fi_map, fc_map,
    ex_map, inc_map) with:
      fi_map[ts]  (fvt (E,), starts (E+1,), av (sz,), v (sz, 4)): per-end version
                   rows in (first availability, end) end order
      fc_map[ts]  (avail (n,), end (n,), tmap (n,), fvt (E,), vstarts (E+1,),
                   avv (sz2,), vv (sz2, 3), ue (E,)): flat rows in stable
                   (available_at, end) order, the same rows as per-end version
                   rows in (first availability, end) order, and the end list of
                   the latter (for express-forecast pairing)
      ex_map[ts]  (avail (n,), end (n,), n_income (n,), yoy (n,)): flat rows in
                   stable (available_at, end) order
      inc_map[ts] (end (Ei,) ascending, first_av (Ei,)): first income visibility
                   per end
    """
    uniq, codes = np.unique(frame["ts_code"].to_numpy(), return_inverse=True)
    end_i = frame["end_date"].to_numpy().astype(np.int64)
    av0 = frame["avail_s"].to_numpy()
    ds = frame["dataset"].to_numpy()

    fi_map, fc_map, ex_map, inc_map = {}, {}, {}, {}

    rows = np.nonzero(ds == DS_FI)[0]
    if rows.size:
        c, e, a = codes[rows], end_i[rows], av0[rows]
        v = frame[FI_COLS].to_numpy(dtype=np.float64)[rows]
        o = np.lexsort((a, e, c))  # (symbol, end, available_at), stable
        c, e, a, v = c[o], e[o], a[o], v[o]
        uv, i0, sz = np.unique(c, return_index=True, return_counts=True)
        for i in range(len(uv)):
            b0 = i0[i]
            fvt, starts, fa, fv, _ue = _reorder_ends(e[b0:b0 + sz[i]],
                                                     a[b0:b0 + sz[i]],
                                                     v[b0:b0 + sz[i]])
            fi_map[uniq[uv[i]]] = (fvt, starts, fa, fv)

    rows = np.nonzero(ds == DS_FC)[0]
    if rows.size:
        c, e, a = codes[rows], end_i[rows], av0[rows]
        vv = frame[FC_NUM].to_numpy(dtype=np.float64)[rows]
        typ = frame["type"].to_numpy()[rows]
        o1 = np.lexsort((e, a, c))  # (symbol, available_at, end), stable
        o2 = np.lexsort((a, e, c))  # (symbol, end, available_at), stable
        c1, a1, e1 = c[o1], a[o1], e[o1]
        tm1 = _type_map(typ[o1])
        c2, a2, e2, v2 = c[o2], a[o2], e[o2], vv[o2]
        uv1, i1, sz1 = np.unique(c1, return_index=True, return_counts=True)
        _uv2, i2, _sz2 = np.unique(c2, return_index=True, return_counts=True)
        for i in range(len(uv1)):
            b0a = i1[i]  # block start in (symbol, available_at, end) order
            b0v = i2[i]  # block start in (symbol, end, available_at) order
            fvt, vstarts, fav, fvv, uev = _reorder_ends(
                e2[b0v:b0v + sz1[i]], a2[b0v:b0v + sz1[i]],
                v2[b0v:b0v + sz1[i]])
            fc_map[uniq[uv1[i]]] = (a1[b0a:b0a + sz1[i]], e1[b0a:b0a + sz1[i]],
                                    tm1[b0a:b0a + sz1[i]], fvt, vstarts, fav, fvv, uev)

    rows = np.nonzero(ds == DS_EX)[0]
    if rows.size:
        c, e, a = codes[rows], end_i[rows], av0[rows]
        vv = frame[EX_NUM].to_numpy(dtype=np.float64)[rows]
        o = np.lexsort((e, a, c))  # (symbol, available_at, end), stable
        c, a, e, vv = c[o], a[o], e[o], vv[o]
        uv, i0, sz = np.unique(c, return_index=True, return_counts=True)
        for i in range(len(uv)):
            b0 = i0[i]
            ex_map[uniq[uv[i]]] = (a[b0:b0 + sz[i]], e[b0:b0 + sz[i]],
                                   vv[b0:b0 + sz[i], 0],
                                   vv[b0:b0 + sz[i], 1])

    rows = np.nonzero(ds == DS_IN)[0]
    if rows.size:
        c, e, a = codes[rows], end_i[rows], av0[rows]
        o = np.lexsort((a, e, c))  # (symbol, end, available_at), stable
        c, e, a = c[o], e[o], a[o]
        uv, i0, sz = np.unique(c, return_index=True, return_counts=True)
        for i in range(len(uv)):
            b0 = i0[i]
            b1 = b0 + sz[i]
            ue, sp = np.unique(e[b0:b1], return_index=True)
            inc_map[uniq[uv[i]]] = (ue, a[b0:b1][sp])
    return fi_map, fc_map, ex_map, inc_map


def _evaluate_maps(maps, symbols, t_arr):
    """Evaluate the 6 feature columns from precomputed per-symbol structures.

    maps: the (fi_map, fc_map, ex_map, inc_map) tuple from _build_maps (four
    empty dicts for an empty frame); symbols: ordered symbol list; t_arr: (m,)
    int64 epoch seconds. Returns {feature: (m, len(symbols)) float64} aligned
    to symbols. Per query time the same searchsorted rules apply (latest
    visible end -> in-effect version; latest visible flat row -> type map /
    express-forecast pair), fully vectorized over t_arr. Each bar of a fit
    block is one entry of t_arr, so evaluating per block on the same maps is
    exactly the restriction of the full-panel evaluation (bars are independent).
    """
    t_arr = np.asarray(t_arr, dtype=np.int64)
    m = len(t_arr)
    S = len(symbols)
    P = {n: (np.zeros((m, S), dtype=np.float64) if n in ("fc_dir", "fc_exp_rev")
             else np.full((m, S), np.nan, dtype=np.float64)) for n in FEATURES}
    if S == 0 or m == 0:
        return P
    fi_map, fc_map, ex_map, inc_map = maps
    inf = np.float64(np.inf)
    for si, ts in enumerate(symbols):
        fi_s = fi_map.get(ts)
        fc_s = fc_map.get(ts)
        ex_s = ex_map.get(ts)
        if fi_s is None and fc_s is None and ex_s is None:
            continue  # defaults: fi NaN, fc_dir 0, fc_exp_rev 0
        # ---- fi 4 columns: latest visible end, in-effect version, percent / 100 ----
        if fi_s is not None:
            fvt, starts, fa, fv = fi_s
            E = fvt.size
            C = np.full((E, m, 4), np.nan)
            for i in range(E):
                a = fa[starts[i]:starts[i + 1]]
                v = fv[starts[i]:starts[i + 1]]
                idx = np.searchsorted(a, t_arr, "right") - 1
                ok = idx >= 0
                if ok.any():
                    C[i, ok] = v[idx[ok]]
            L = np.searchsorted(fvt, t_arr, "right") - 1
            jj = np.nonzero(L >= 0)[0]
            if jj.size:
                for c, n in enumerate(FI_COLS):
                    col = np.full(m, np.nan)
                    col[jj] = C[L[jj], jj, c] / 100.0
                    P[n][:, si] = col
        # ---- fc_dir: latest visible forecast row (any end) ----
        if fc_s is not None:
            avail, endf, tm, _fv, _vs, _fa, _vv, _ue = fc_s
            k = np.searchsorted(avail, t_arr, "right") - 1
            okk = k >= 0
            if okk.any():
                inc_s = inc_map.get(ts)
                if inc_s is not None:
                    ie, ifv = inc_s
                    pos = np.searchsorted(ie, endf, "left")
                    pc = np.clip(pos, 0, ie.size - 1)
                    match = (pos < ie.size) & (ie[pc] == endf)
                    inc_first = np.where(match, ifv[pc], inf)
                else:
                    inc_first = np.full(avail.size, inf)
                kv = k[okk]
                out = np.zeros(m)
                out[okk] = np.where(inc_first[kv] <= t_arr[okk], 0.0, tm[kv])
                P["fc_dir"][:, si] = out
        # ---- fc_exp_rev: latest visible express row paired with same (ts, end) ----
        if fc_s is not None and ex_s is not None:
            _avail, _endf, _tm, fvt, vstarts, fav, fvv, ue_f = fc_s
            exa, exe, eni, eyoy = ex_s
            kx = np.searchsorted(exa, t_arr, "right") - 1
            okx = kx >= 0
            if okx.any():
                Ef = fvt.size
                Cf = np.full((Ef, m, 3), np.nan)
                for i in range(Ef):
                    a = fav[vstarts[i]:vstarts[i + 1]]
                    v = fvv[vstarts[i]:vstarts[i + 1]]
                    idx = np.searchsorted(a, t_arr, "right") - 1
                    ok = idx >= 0
                    if ok.any():
                        Cf[i, ok] = v[idx[ok]]
                # ue_f is in (first-availability, end) order, not sorted:
                # map ex-row ends to ue_f row indices via a dict (old: pos_fce)
                posf = {e: i for i, e in enumerate(ue_f.tolist())}
                ue_x, ixx = np.unique(exe, return_inverse=True)
                irow = np.array([posf.get(e, -1) for e in ue_x],
                                dtype=np.int64)[ixx]
                r = np.nonzero(okx)[0]
                iq = irow[kx[okx]]
                good = iq >= 0
                A = np.full((r.size, 3), np.nan)
                A[good] = Cf[iq[good], r[good], :]
                a, b, c0 = A[:, 0], A[:, 1], A[:, 2]
                okm = (np.isfinite(a) & np.isfinite(b) & np.isfinite(c0)
                       & (np.abs(c0) > 0))
                fc_mid = np.where(okm, (a + b) / 2.0 / c0 - 1.0, 0.0)
                niq, yq = eni[kx[okx]], eyoy[kx[okx]]
                oke = np.isfinite(niq) & np.isfinite(yq) & (yq != 0)
                exp_gr = np.where(oke, (niq - yq) / np.abs(yq), 0.0)
                sgn = np.where(exp_gr > fc_mid, 1.0,
                               np.where(exp_gr < fc_mid, -1.0, 0.0))
                sgn[~good] = 0.0
                out = np.zeros(m)
                out[r] = sgn
                P["fc_exp_rev"][:, si] = out
    return P


def _evaluate(frame, symbols, t_arr):
    """Evaluate the 6 feature columns at query times t_arr (int64 epoch seconds).

    frame: normalized fundamentals frame (read_fundamentals output); symbols:
    ordered symbol list; t_arr: (m,) int64 epoch seconds. Returns
    {feature: (m, len(symbols)) float64} aligned to symbols. Same semantics as
    the per-symbol reference of prep.py main(): per-symbol structures are
    precomputed once in numpy, then every query time is selected with the same
    searchsorted rules, fully vectorized over t_arr."""
    if frame is None or frame.empty:
        return _evaluate_maps(({}, {}, {}, {}), symbols, t_arr)
    return _evaluate_maps(_build_maps(frame), symbols, t_arr)


def _t_of_bar_date(d):
    """08:30+08:00 epoch seconds for a YYYYMMDD bar date (the decision time)."""
    return int(pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]} 08:30:00+08:00")
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
