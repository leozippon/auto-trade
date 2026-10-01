"""newage: 1-column listing-age PIT feature block (Fold 8 R2, variant V2,
age-only trim per parent ruling after the 5% importance gate).

Gate result (notes/f8_gate/RESULTS.md): of the pre-registered 3 newage
columns, `age` PASSED on both splits (gain share S1 0.08513 / S2 0.05535)
while unlock_up60 (0.01047 / 0.00616) and unlock_d60 (0.00568 / 0.00272)
FAILED; the pre-registration column-reduction clause trims the variant to
`age` only (173 model columns). Mechanism narrowed to listing age; the
share_float_complete event domain is NOT read at all anymore (fit/decision
cost ~= parent), so the decision path no longer touches /events.

Reads (only context.asof_dir + "/universe"):
  universe  static listing dimension table (ts_code, list_date, YYYYMMDD
            string). A name absent from the view -> age NaN by the
            pre-registration (both asof-view forms are handled: the table
            may or may not be cut by list_date <= T; absent -> NaN either
            way).

Feature (raw value, appended after the 172 parent columns):
  age          log1p(T - list_date) in calendar days; T = the evaluation day
               (decision day on the decision path; the bar date on the fit
               path, each bar evaluated at its 08:30 decision time like the
               parent vipF2/mfflow fit panels). T < list_date -> NaN
               (defensive guard; cannot occur for a bar-bearing symbol).
  age passes through the SAME main-path RobustZScore as the 172 parent
  columns (no rank standardization - that applied only to the two removed
  unlock columns).

The decision path and the fit panels share _evaluate (single source of
truth).
"""

import numpy as np
import pandas as pd

# The 1 feature column, in the exact order appended after the 158 Alpha +
# 6 vipF2 + 8 mfflow columns. data.FEATURE_NAMES is built from it, for fit and
# decision alike.
FEATURES = ("age",)

UNIV_COLS = ["ts_code", "list_date"]
_DAY_NS = 86400 * 10**9
_DAY_S = 86400


def read_universe(context):
    """Static listing dimension table -> {ts_code: list_date ordinal days}.

    Only legal data source: context.asof_dir + "/universe" (directory of
    parquet parts). The table is static history (a scored name always has a
    bar, no future leak); the asof view may cut it by list_date <= T, both
    forms handled (absent name -> simply missing from the dict -> NaN)."""
    df = pd.read_parquet(context.asof_dir + "/universe", columns=UNIV_COLS)
    df = df.dropna(subset=["list_date"])
    ord_ = (pd.to_datetime(df["list_date"].astype(str).str.zfill(8),
                           format="%Y%m%d")
            .astype("int64") // _DAY_NS).to_numpy(dtype=np.int64)
    return {str(ts): int(o) for ts, o in
            zip(df["ts_code"].astype(str), ord_)}


def _t_of_bar_date(d):
    """08:30+08:00 epoch seconds for a YYYYMMDD bar date (the decision time)."""
    return int(pd.Timestamp("%s-%s-%s 08:30:00+08:00" % (d[:4], d[4:6], d[6:8]))
               .value // 10**9)


def _evaluate(univ, symbols, t_arr):
    """Evaluate the 1 feature column at query times t_arr (int64 epoch
    seconds). univ: read_universe dict; symbols: ordered symbol list;
    t_arr: (m,) int64 epoch seconds. Returns {feature: (m, S) float64}
    aligned to symbols - the single shared constructor of the decision path
    (m = 1) and the fit path (m = len(dates))."""
    t_arr = np.asarray(t_arr, dtype=np.int64)
    m = len(t_arr)
    S = len(symbols)
    P = {n: np.full((m, S), np.nan, dtype=np.float64) for n in FEATURES}
    if S == 0 or m == 0:
        return P
    # Date ordinal (days since epoch) of each query time: t is +08:00, so
    # the calendar date is (t + 8h) floored to the day in UTC.
    T_ord = (t_arr + 8 * 3600) // _DAY_S
    for si, ts in enumerate(symbols):
        lo = univ.get(str(ts))
        if lo is not None:
            age = T_ord - lo
            P["age"][:, si] = np.where(
                age >= 0, np.log1p(np.maximum(age, 0)), np.nan)
    return P


def newage_frame(context, symbols):
    """Decision-path entry: rebalance day t x symbols -> 1 raw feature
    column. t = context.inference_at (08:30+08:00). Returns a float64
    DataFrame indexed by symbols, column order = FEATURES (the order
    appended after the 158 Alpha + 6 vipF2 + 8 mfflow columns)."""
    t = int(pd.Timestamp(context.inference_at).value // 10**9)
    univ = read_universe(context)
    P = _evaluate(univ, list(symbols), np.array([t], dtype=np.int64))
    return pd.DataFrame({n: P[n][0] for n in FEATURES},
                        index=np.asarray(symbols, dtype=object))


def newage_fit_panels(context, dates, symbols):
    """Fit-path entry: 1 panel over the full visible bar grid (panel
    expansion). Each bar date is evaluated at its 08:30+08:00 decision time,
    the same semantics as the decision path (_evaluate is the shared
    constructor). Returns {feature: (T, S) float64} aligned to
    dates/symbols, same shape/key order as vipf2_fit_panels."""
    univ = read_universe(context)
    t_arr = np.array([_t_of_bar_date(d) for d in dates], dtype=np.int64)
    return _evaluate(univ, list(symbols), t_arr)
