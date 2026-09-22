"""The PIT constituent panel: one builder for training and for a decision.

`build` reads a bounded window of the `daily` domain restricted to the names
that were CSI 300 constituents at some point inside it, plus the benchmark's
own series, and returns dense (dates x names) arrays. `fit_window` builds the
training panel with labels and the train/validation split; `review_window`
builds a short panel whose last row is the cross-section a review scores.
Because both go through this one function, the cross-section a model is
trained on and the one it scores are built identically -- the pack's one hard
requirement.

Visibility. `daily` and `universe` carry no `available_at` column: the as-of
view holds exactly the rows visible at the decision, so at an 08:30 decision
the newest `daily` row is the previous trading day (T-1) and no timestamp
filter is possible. The constituent sections and the benchmark series do carry
one and are filtered on it in `lib/index.py`.

Units (snapshot contract): prices CNY/share unadjusted, `adj_factor` is the
cumulative adjustment factor, `vol` shares, `amount` CNY, `turnover_rate` a
decimal. Adjusted price = price * adj_factor and split-consistent volume =
vol / adj_factor, so every ratio inside a window is adjustment-free.

Universe. A date's cross-section is the constituents of the newest section
dated at or before it. Training uses every such name; the tradability filter
that a *book* needs (ST, STAR/BSE lots) is applied in `lib/trade.py` on the
decision row only, because `universe` is a single decision-day vintage and
pushing today's ST labels back over a training window would be a look-ahead
that quietly drops future losers.

Features. The 158 Alpha158 operators of `lib/alpha158.py` on the adjusted
series, cross-sectionally robust-z-scored over that date's constituents,
missing -> 0, clipped at +-3. Non-constituents are masked out before the
statistics are taken, so the z-score is the cross-section the book trades.

Label. The forward HORIZON-day open-to-open return of a signal on day t (a
decision on t+1 buys at or near that open), residualised against the
benchmark's own forward return with a trailing beta, optionally demeaned
within the SW L1 industry, then ranked per date into (rank - 1) / n - 0.5
over the constituents that have it. Only signal dates whose label is realised
by T-1 exist, so no label can cross the decision.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from lib import alpha158, index, knobs

DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close",
                 "vol", "amount", "adj_factor", "turnover_rate"]
UNIVERSE_COLUMNS = ["ts_code", "name", "l1_name"]
UNCLASSIFIED = "未分类"

# Structural, not knobs: the label's benchmark leg and the feature warm-up.
BETA_WINDOW = 120
BETA_MIN_OBS = 60
BETA_SHRINK = 0.7
BETA_FLOOR = 0.3
BETA_CAP = 2.0
FEATURE_BLOCK = 120
FEATURE_CONTEXT = max(alpha158.WINDOWS)
MIN_TRAIN_DATES = 120
# Calendar days read on a review: the longest operator window, the correlation
# and market windows, holidays and a margin.
REVIEW_CALENDAR_DAYS = 260
# Extra calendar days ahead of the training window so its first date is fed.
FIT_WARMUP_DAYS = 260


def warmup():
    """Date indexes below this have no usable feature row."""

    return int(max(FEATURE_CONTEXT, knobs.CORR_WINDOW, BETA_MIN_OBS)) + 1


def fit_calendar_days():
    return int(365.25 * knobs.TRAIN_YEARS) + FIT_WARMUP_DAYS


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
    if frame.empty:
        raise RuntimeError("no visible daily rows for the constituent union")
    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def _membership(dates, sec, codes):
    """(T, S) bool: constituent on that date, per the newest section at or before it."""

    section_dates = np.array(sorted(sec["trade_date"].unique()))
    table = np.zeros((len(section_dates), len(codes)), dtype=bool)
    order = {code: position for position, code in enumerate(codes)}
    for row, section in enumerate(section_dates):
        names = sec.loc[sec["trade_date"] == section, "ts_code"]
        table[row, [order[code] for code in names if code in order]] = True
    position = np.searchsorted(section_dates, dates, side="right") - 1
    known = position >= 0
    if not known.any():
        raise RuntimeError("no constituent section dated at or before the panel's first day")
    # A date before the first visible section has NO known membership. Carrying the
    # first section backwards would hand those dates a future name list, so they are
    # simply not constituents and drop out of training, the z-score and the graph.
    out = table[np.maximum(position, 0)]
    out[~known] = False
    return out


def _features(wide, usable):
    """(T, S, 158) float32: the operators, z-scored over each date's constituents."""

    symbols, days = wide["C"].shape
    out = np.zeros((days, symbols, len(alpha158.ALPHA158)), dtype=np.float32)
    for start in range(0, days, FEATURE_BLOCK):
        end = min(days, start + FEATURE_BLOCK)
        low = max(0, start - FEATURE_CONTEXT)
        raw = alpha158.compute_158({name: series[:, low:end] for name, series in wide.items()})
        block = np.stack([raw[name] for name in alpha158.ALPHA158], axis=2)[:, start - low:, :]
        block = np.transpose(block, (1, 0, 2)).astype(np.float64)
        block[~usable[start:end]] = np.nan
        out[start:end] = alpha158.robust_zscore(block)
        del raw, block
    return out


def _beta(returns, benchmark_returns):
    """(T, S) trailing OLS slope on the benchmark, shrunk to 1 and clipped."""

    stock = pd.DataFrame(returns)
    bench = pd.Series(benchmark_returns, index=stock.index)
    present = stock.notna().astype("float64")
    paired = present.mul(bench, axis=0)
    roll = dict(window=BETA_WINDOW, min_periods=BETA_MIN_OBS)
    count = present.rolling(**roll).sum()
    sum_x = stock.rolling(**roll).sum()
    sum_y = paired.rolling(**roll).sum()
    sum_xy = stock.mul(bench, axis=0).rolling(**roll).sum()
    sum_yy = paired.mul(bench, axis=0).rolling(**roll).sum()
    numerator = (count * sum_xy - sum_x * sum_y).to_numpy()
    denominator = (count * sum_yy - sum_y * sum_y).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(np.abs(denominator) > 0.0, numerator / denominator, np.nan)
    beta = np.where(np.isfinite(beta), beta, 1.0)
    return np.clip(BETA_SHRINK * beta + (1.0 - BETA_SHRINK), BETA_FLOOR, BETA_CAP)


def _forward(series, horizon):
    """Open-to-open forward return of a signal on t, NaN where it is not realised."""

    days = series.shape[0]
    out = np.full(series.shape, np.nan)
    span = days - 1 - horizon
    if span > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:span] = series[1 + horizon:] / series[1: days - horizon] - 1.0
    return out


def _labels(open_adjusted, bench_open, beta, member, industry):
    forward = _forward(open_adjusted, knobs.HORIZON)
    bench_forward = _forward(bench_open.reshape(-1, 1), knobs.HORIZON)
    residual = forward - beta * bench_forward
    residual = np.where(member & np.isfinite(residual), residual, np.nan)
    if knobs.LABEL_INDUSTRY_DEMEAN:
        frame = pd.DataFrame(residual)
        group = frame.T.groupby(industry).transform("mean").T.to_numpy()
        residual = residual - np.where(np.isfinite(group), group, 0.0)
    frame = pd.DataFrame(residual)
    ranks = frame.rank(axis=1, pct=True).to_numpy()
    count = np.isfinite(residual).sum(axis=1, keepdims=True)
    return np.where(np.isfinite(residual),
                    ranks - 1.0 / np.maximum(count, 1) - 0.5, np.nan).astype(np.float32)


def build(context, calendar_days, labels):
    """Dense arrays over the window's trading days and its constituent union."""

    sec = index.sections(context, calendar_days)
    bench = index.benchmark(context, calendar_days).set_index("trade_date")
    frame = _read_daily(context, calendar_days, sorted({str(code) for code in sec["ts_code"]}))
    # The panel can only start where the benchmark starts: the label's benchmark leg
    # needs it on every date. The daily domain reaches five years back while the macro
    # domain's own window may start later, so the panel is trimmed to the benchmark
    # rather than widened -- a date with no benchmark is not a usable training date.
    frame = frame[frame["trade_date"] >= str(bench.index.min())]
    if frame.empty:
        raise RuntimeError("no daily rows at or after the benchmark series' first visible day")
    dates = np.array(sorted(frame["trade_date"].unique()))
    codes = np.array(sorted(frame["ts_code"].unique()))
    days, symbols = len(dates), len(codes)
    date_index = np.searchsorted(dates, frame["trade_date"].to_numpy())
    code_index = np.searchsorted(codes, frame["ts_code"].to_numpy())

    def dense(values, fill=np.nan):
        out = np.full((symbols, days), fill, dtype=np.float64)
        out[code_index, date_index] = values
        return out

    factor = frame["adj_factor"].to_numpy(dtype=np.float64)
    volume = frame["vol"].to_numpy(dtype=np.float64)
    amount = frame["amount"].to_numpy(dtype=np.float64)
    vwap = np.where(volume > 0, amount / np.where(volume > 0, volume, 1.0), np.nan)
    wide = {
        "O": dense(frame["open"].to_numpy(dtype=np.float64) * factor),
        "H": dense(frame["high"].to_numpy(dtype=np.float64) * factor),
        "L": dense(frame["low"].to_numpy(dtype=np.float64) * factor),
        "C": dense(frame["close"].to_numpy(dtype=np.float64) * factor),
        "VWAP": dense(vwap * factor),
        "V": dense(volume / factor),
    }
    has_bar = np.isfinite(wide["C"]).T
    member = _membership(dates, sec, codes) & has_bar

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    missing = [name for name in UNIVERSE_COLUMNS if name not in universe.columns]
    if missing:
        raise RuntimeError(f"universe is missing columns {missing}")
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(codes)
    labels_l1 = info["l1_name"].fillna(UNCLASSIFIED).astype(str).to_numpy()
    if (labels_l1 == UNCLASSIFIED).all():
        raise RuntimeError("no visible SW L1 industry on the constituent union")
    industry = pd.factorize(labels_l1)[0].astype(np.int64)

    bench_close = pd.to_numeric(bench["close"], errors="coerce").reindex(dates).ffill()
    bench_open = pd.to_numeric(bench["open"], errors="coerce").reindex(dates).ffill()
    if not np.isfinite(bench_close.to_numpy()).all():
        raise RuntimeError("the benchmark series has gaps over the panel's trading days")

    adjusted_close = wide["C"].T
    returns = np.full((days, symbols), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = adjusted_close[1:] / adjusted_close[:-1] - 1.0
    bench_returns = np.full(days, np.nan)
    bench_returns[1:] = bench_close.to_numpy()[1:] / bench_close.to_numpy()[:-1] - 1.0
    bench_returns[0] = 0.0

    panel = {
        "dates": dates,
        "codes": codes,
        "names": info["name"].fillna("").astype(str).to_numpy(),
        "industry": industry,
        "industry_name": labels_l1,
        "has_bar": has_bar,
        "member": member,
        "close": dense(frame["close"].to_numpy(dtype=np.float64)).T,
        "returns": returns,
        "bench_close": bench_close.to_numpy(),
        "feat": _features(wide, member),
    }
    if labels:
        beta = _beta(returns, bench_returns)
        panel["yrank"] = _labels(wide["O"].T, bench_open.to_numpy(), beta, member, industry)
        panel["beta"] = beta
    return panel


def fit_dates(context, data):
    """(training, validation) date indexes of one refit; every candidate shares them."""

    dates = data["dates"]
    first = (pd.Timestamp(context.inference_at.date())
             - pd.DateOffset(years=knobs.TRAIN_YEARS)).strftime("%Y%m%d")
    embargo = knobs.HORIZON + 2
    usable = [t for t in range(warmup(), len(dates) - knobs.HORIZON)
              if dates[t] >= first and np.isfinite(data["yrank"][t]).sum() > 1]
    if len(usable) < knobs.VALID_DAYS + embargo + MIN_TRAIN_DATES:
        raise RuntimeError(f"fit has only {len(usable)} labelled dates in its window")
    valid = usable[-knobs.VALID_DAYS:]
    train = [t for t in usable if t <= valid[0] - embargo]
    if len(train) < MIN_TRAIN_DATES:
        raise RuntimeError(f"fit has only {len(train)} training dates after the embargo")
    return train, valid


def fit_window(context):
    """(panel, training dates, validation dates) -- the one training entry point."""

    data = build(context, fit_calendar_days(), labels=True)
    train, valid = fit_dates(context, data)
    return data, train, valid


def review_window(context):
    """The short panel whose last row is the cross-section this review scores."""

    return build(context, REVIEW_CALENDAR_DAYS, labels=False)


def scorable(data, t):
    """Name indexes that are constituents with a bar on date index t."""

    return np.nonzero(data["member"][t])[0]
