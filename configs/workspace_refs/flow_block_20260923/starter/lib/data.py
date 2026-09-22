"""The one panel builder: the carrier's Alpha158 inputs, the new block, and their union.

`wide(context, calendar_days)` reads the window once and returns (names x dates)
matrices; `fit_samples(...)` and `decision_features(...)` turn slices of them
into the feature matrix a booster is trained on and the one it scores. `fit`
and a review call the same functions on the same window definition, so the two
matrices are built identically -- the single most common way a two-source
feature set goes wrong is a training path and a decision path that assemble
their columns in different orders.

The name axis is the UNION of every constituent of every visible section in the
window, not the whole market: about 420 names and 820 bars for a three-year
window instead of 3,500 names, which is what keeps the fit inside the
container and lets the events read push `ts_code` down.

Visibility. `daily` has no `available_at` column -- the as-of view holds exactly
the rows visible at the decision, so its newest row at an 08:30 decision is the
previous trading day (T-1). The constituent sections and every events dataset
carry `available_at` and are filtered by it (`lib/index.py`, `lib/events.py`).

Adjustment. Prices and volume are forward-adjusted to each symbol's newest
visible bar (`adj_factor / adj_factor of that bar`), the convention the frozen
artifact used; every Alpha158 operator is a ratio inside its own window, so the
anchor cancels and only split consistency matters. VWAP = amount / vol is built
from raw values, where the factor cancels by construction. `raw_amount` is the
UNADJUSTED turnover in CNY: it is the denominator the block's money ratios
divide by, and an adjusted one would not match the vendor's own amounts.

Candidates. `c_base` is the carrier alone -- the 158 operators and nothing
else. `f1` is the carrier plus the block. `f_only` is the block alone, which is
a diagnostic and never a nomination: it answers "does this block carry anything
at all", which is a different question from "does it add anything".
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import features, flow as block, index, knobs

WARMUP_BARS = 60               # leading bars of the fit window that only warm the operators
DECISION_LOOKBACK_DAYS = 200   # >= 63 trading days of operator window plus holidays
FIT_CALENDAR_DAYS = int(365.25 * knobs.TRAIN_YEARS) + 120
SECTION_EXTRA_DAYS = 45
BLOCK = 180
CONTEXT = 60

DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                 "adj_factor"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
CARRIER_NAMES = list(features.ALPHA158)
BLOCK_NAMES = list(block.BLOCK_NAMES)


def block_coverage(panel, values):
    """Per-bar share of the bar's constituents whose block columns are all finite."""

    return block.coverage(values, panel)


def empty_block(panel):
    """A zero-width block, for the control leg that does not read one."""

    return np.zeros((len(panel["symbols"]), len(panel["dates"]), 0))


def build_block(context, panel):
    return block.build(context, panel)


def feature_names(candidate):
    """The columns this candidate's booster is fitted on, in this exact order."""

    if candidate == "c_base":
        return list(CARRIER_NAMES)
    if candidate == "f_only":
        return list(BLOCK_NAMES)
    if candidate == "f1":
        return CARRIER_NAMES + BLOCK_NAMES
    raise ValueError(f"unknown candidate: {candidate}")


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


def wide(context, calendar_days):
    """(names x dates) matrices over the constituent union of the window's visible sections."""

    sections = index.sections(context, calendar_days + SECTION_EXTRA_DAYS)
    frame = _read_daily(context, calendar_days, sorted(sections["ts_code"].unique()))
    if frame.empty:
        raise RuntimeError("the daily window of the as-of view holds no constituent rows")
    benchmark = index.benchmark(context, calendar_days).set_index("trade_date")
    # The benchmark is a leg of the label, so a date without it is not a usable
    # training date. Clip the panel to the benchmark rather than widening the
    # read: widening would hand the first section's names to earlier dates.
    first = str(benchmark.index.min())
    frame = frame[frame["trade_date"] >= first]
    if frame.empty:
        raise RuntimeError("no constituent bar falls inside the visible benchmark series")
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
        "raw_amount": dense(amount),
        "has_bar": dense(np.ones(len(frame)), fill=0.0) > 0,
        "member": index.membership(sections, np.array(dates), np.array(codes)).T,
        "sections": sections,
    }
    panel["n_bars"] = np.cumsum(panel["has_bar"], axis=1)
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


def _carrier_block(panel, lo, hi, keep):
    """(len(keep), names, 158) float32 z-scored Alpha158 columns for the bars in `keep`."""

    segment = {key: panel[key][:, lo:hi].copy() for key in ("O", "H", "L", "C", "V", "VWAP")}
    for key in ("O", "H", "L", "C", "V", "VWAP"):
        segment[key][~panel["has_bar"][:, lo:hi]] = np.nan
    raw = features.compute_158(segment)
    columns = np.stack([raw[name][:, [t - lo for t in keep]] for name in features.ALPHA158], axis=2)
    stacked = np.transpose(columns, (1, 0, 2)).astype(np.float32)
    for row, t in enumerate(keep):
        stacked[row][~tradable(panel, t)] = np.nan
    return features.robust_zscore(stacked)


def _new_block(panel, values, keep):
    """(len(keep), names, K) float32 z-scored block columns for the bars in `keep`.

    The same robust cross-sectional z-score the carrier's columns get, over the
    same constituent set of the same bar, so the booster sees one homogeneous
    matrix and no column arrives on a different scale.
    """

    stacked = np.transpose(values[:, keep, :], (1, 0, 2)).astype(np.float32)
    for row, t in enumerate(keep):
        stacked[row][~tradable(panel, t)] = np.nan
    return features.robust_zscore(stacked)


def _matrix(panel, values, lo, hi, keep, candidate):
    if candidate == "c_base":
        return _carrier_block(panel, lo, hi, keep)
    if candidate == "f_only":
        return _new_block(panel, values, keep)
    return np.concatenate([_carrier_block(panel, lo, hi, keep),
                           _new_block(panel, values, keep)], axis=2)


def decision_features(panel, values, candidate):
    """(names, len(feature_names)) float32 features of the newest visible bar."""

    last = len(panel["dates"]) - 1
    lo = max(0, last - CONTEXT - 5)
    return _matrix(panel, values, lo, last + 1, [last], candidate)[0]


def fit_samples(panel, values, target, realized, candidate):
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
            matrix = _matrix(panel, values, lo, end, wanted, candidate)
            for row, t in enumerate(wanted):
                idx = np.nonzero(realized[:, t] & tradable(panel, t))[0]
                if idx.size == 0:
                    continue
                rows.append(matrix[row][idx])
                labels.append(target[idx, t].astype(np.float32))
                bars.append(np.full(idx.size, t, dtype=np.int64))
                names.append(idx.astype(np.int64))
            del matrix
        if is_last:
            break
    if not rows:
        raise RuntimeError("the fit window produced no labelled rows")
    return {"X": np.vstack(rows), "y": np.concatenate(labels),
            "bar": np.concatenate(bars), "name": np.concatenate(names)}
