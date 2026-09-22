"""The one panel builder: Alpha158 inputs over the constituent union, for fit and for decisions.

`wide(context, calendar_days)` reads the window once and returns (names x dates)
matrices; `fit_samples` and `decision_features` turn slices of them into the
158 Alpha columns, cross-sectionally robust-z-scored over the constituents of
each bar. `fit` and a review call the same builder on the same window
definition, so the matrix a booster is trained on and the matrix it scores are
built identically.

The name axis is the UNION of every constituent of every visible section in the
window, not the whole market, and `member[s, t]` then restricts every bar to the
section that was in force on it. Reading only the union is also what keeps the
fit small -- a few hundred names instead of the 3,500 of an all-market panel --
which is why seven registered labels can be screened for what the graduated
lineage paid to fit one.

Nothing in this module depends on WHICH label is being built: it returns prices,
volumes and the benchmark's own series, and `lib/labels.py` turns them into a
target. That separation is the point of this arm -- if the panel moved with the
label, a label comparison would not be a label comparison.

Visibility. `daily` has no `available_at` column -- the as-of view holds exactly
the rows visible at the decision, so its newest row at an 08:30 decision is the
previous trading day (T-1). The constituent sections carry `available_at` and
are placed by it (`lib/index.py`); `member[s, t]` says whether name s was in the
section in force on bar t.

Adjustment. Prices and volume are forward-adjusted to each symbol's newest
visible bar (`adj_factor / adj_factor of that bar`), the convention the frozen
artifact used; every Alpha158 operator is a ratio inside its own window, so the
anchor cancels and only split consistency matters. VWAP = amount / vol is built
from raw values, where the factor cancels by construction.

Units (snapshot contract): prices CNY/share unadjusted, `vol` shares, `amount`
CNY, `index_daily` prices index points. Only the columns the operators, the
label and the order sizing actually consume are read: a column a package does
not use is a column its field map has to explain for nothing.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import features, index

TRAIN_YEARS = 3
WARMUP_BARS = 60               # leading bars of the fit window that only warm the operators
DECISION_LOOKBACK_DAYS = 200   # >= 63 trading days of operator window plus holidays
FIT_CALENDAR_DAYS = int(365.25 * TRAIN_YEARS) + 120
SECTION_EXTRA_DAYS = 45
BLOCK = 180
CONTEXT = 60

DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                 "adj_factor"]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "open", "close"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
FEATURE_NAMES = list(features.ALPHA158)


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
    return frame.assign(trade_date=frame["trade_date"].astype(str)).reset_index(drop=True)


def _read_benchmark(context, calendar_days):
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
    return frame.assign(trade_date=frame["trade_date"].astype(str)).drop_duplicates(
        "trade_date", keep="last").set_index("trade_date")


def wide(context, calendar_days):
    """(names x dates) matrices over the constituent union of the window's visible sections."""

    sections = index.sections(context, calendar_days + SECTION_EXTRA_DAYS)
    frame = _read_daily(context, calendar_days, sorted(sections["ts_code"].unique()))
    if frame.empty:
        raise RuntimeError("the daily window of the as-of view holds no constituent rows")
    dates = sorted(frame["trade_date"].unique())
    codes = sorted(frame["ts_code"].unique())
    si = np.searchsorted(np.array(codes), frame["ts_code"].to_numpy())
    ti = np.searchsorted(np.array(dates), frame["trade_date"].to_numpy())

    def dense(values, fill=np.nan, dtype=np.float64):
        out = np.full((len(codes), len(dates)), fill, dtype=dtype)
        out[si, ti] = values
        return out

    last = frame.groupby("ts_code", sort=False)["trade_date"].idxmax()
    anchor = frame.loc[last].set_index("ts_code")["adj_factor"]
    factor = frame["adj_factor"].to_numpy(dtype=np.float64) / anchor.loc[frame["ts_code"]].to_numpy()
    volume = frame["vol"].to_numpy(dtype=np.float64)
    amount = frame["amount"].to_numpy(dtype=np.float64)
    panel = {
        "dates": dates,
        "symbols": np.array(codes, dtype=object),
        "O": dense(frame["open"].to_numpy(dtype=np.float64) * factor),
        "H": dense(frame["high"].to_numpy(dtype=np.float64) * factor),
        "L": dense(frame["low"].to_numpy(dtype=np.float64) * factor),
        "C": dense(frame["close"].to_numpy(dtype=np.float64) * factor),
        "V": dense(np.where(factor > 0, volume / factor, np.nan)),
        "VWAP": dense(np.where(volume > 0, amount / np.where(volume > 0, volume, 1.0), np.nan)),
        "raw_close": dense(frame["close"].to_numpy(dtype=np.float64)),
        "has_bar": dense(np.ones(len(frame)), fill=0.0) > 0,
        "member": index.membership(sections, np.array(dates), np.array(codes)).T,
        "sections": sections,
    }
    panel["n_bars"] = np.cumsum(panel["has_bar"], axis=1)
    benchmark = _read_benchmark(context, calendar_days)
    panel["bench_open"] = pd.to_numeric(benchmark["open"], errors="coerce").reindex(
        pd.Index(dates)).ffill().to_numpy()
    panel["bench_close"] = pd.to_numeric(benchmark["close"], errors="coerce").reindex(
        pd.Index(dates)).ffill().to_numpy()
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(pd.Index(codes))
    panel["industry"] = info["l1_name"].fillna("未分类").astype(str).to_numpy()
    panel["name"] = info["name"].fillna("").astype(str).to_numpy()
    return panel


def tradable(panel, t):
    """(names,) bool: in the section in force on bar t, with a bar and a full operator window."""

    codes = pd.Series([str(code) for code in panel["symbols"]])
    names = pd.Series(panel["name"])
    return (
        panel["member"][:, t]
        & panel["has_bar"][:, t]
        & (panel["n_bars"][:, t] >= WARMUP_BARS)
        & ~codes.str.startswith(("688", "689")).to_numpy()
        & ~codes.str.endswith(".BJ").to_numpy()
        & ~names.str.contains("ST|退").to_numpy()
    )


def _zscored_block(panel, lo, hi, keep):
    """(len(keep), names, 158) float32 z-scored features for the bars in `keep` (absolute indexes)."""

    segment = {key: panel[key][:, lo:hi].copy() for key in ("O", "H", "L", "C", "V", "VWAP")}
    for key in ("O", "H", "L", "C", "V", "VWAP"):
        segment[key][~panel["has_bar"][:, lo:hi]] = np.nan
    raw = features.compute_158(segment)
    columns = np.stack([raw[name][:, [t - lo for t in keep]] for name in features.ALPHA158], axis=2)
    block = np.transpose(columns, (1, 0, 2)).astype(np.float32)
    for row, t in enumerate(keep):
        block[row][~tradable(panel, t)] = np.nan
    return features.robust_zscore(block)


def decision_features(panel):
    """(names, 158) float32 z-scored features of the newest visible bar."""

    last = len(panel["dates"]) - 1
    lo = max(0, last - CONTEXT - 5)
    return _zscored_block(panel, lo, last + 1, [last])[0]


def fit_samples(panel, target, realized):
    """Stacked training rows over every bar past the warm-up whose label is realised.

    Blocks of BLOCK bars with CONTEXT leading bars of warm-up; each block keeps
    only its own stride, so a (bar, name) pair is stored exactly once.
    """

    total = len(panel["dates"])
    rows, labels, bars, names = [], [], [], []
    stride = BLOCK - CONTEXT
    for start in range(0, total, stride):
        end = min(total, start + BLOCK)
        lo = max(0, start - CONTEXT)
        is_last = start + BLOCK >= total
        keep_end = end if is_last else min(start + stride, end)
        wanted = [t for t in range(start, keep_end) if t >= WARMUP_BARS and realized[:, t].any()]
        if wanted:
            block = _zscored_block(panel, lo, end, wanted)
            for row, t in enumerate(wanted):
                idx = np.nonzero(realized[:, t] & tradable(panel, t))[0]
                if idx.size == 0:
                    continue
                rows.append(block[row][idx])
                labels.append(target[idx, t].astype(np.float32))
                bars.append(np.full(idx.size, t, dtype=np.int64))
                names.append(idx.astype(np.int64))
            del block
        if is_last:
            break
    if not rows:
        raise RuntimeError("the fit window produced no labelled rows")
    return {"X": np.vstack(rows), "y": np.concatenate(labels),
            "bar": np.concatenate(bars), "name": np.concatenate(names)}
