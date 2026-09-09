"""Family 1 building blocks: PIT reads, the consensus revision, and neutralization.

Every read is rooted at ``context.asof_dir`` and bounded by an explicit column
list and date window.  The reference pack's field map is the authority on units
and stamp rules; this module only implements what it declares.

Column names follow the landed ``report_rc`` events slice (commit 65edfae).  It
is opt-in: confirm it is in this run's ``events.datasets`` before any backtest,
and abstain rather than fall back to the title-only text domain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Declared invariants, not fitted parameters.  See families.md "共同底座".
RECENT_TRADING_DAYS = 20
PRIOR_TRADING_DAYS = 60
MIN_BROKERS = 2
SCALE_FLOOR = 0.05          # CNY/share; keeps near-zero base forecasts from exploding
WINSOR = 0.01
DAILY_LOOKBACK_DAYS = 130   # calendar days: enough for a 60-trading-day control window
REPORT_LOOKBACK_DAYS = 150  # calendar days: enough for the 60-trading-day prior window

DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "close",
    "amount",
    "turnover_rate",
    "circ_mv",
    "adj_factor",
    "is_suspended",
]
REPORT_COLUMNS = [
    "dataset",
    "available_at",
    "create_time",
    "ts_code",
    "report_date",
    "org_name",
    "author_name",
    "quarter",
    "eps",              # CNY_per_share; np/tp are 10k_CNY and are not read here
]
UNIVERSE_COLUMNS = ["ts_code", "name", "list_date"]
BUSINESS_KEY = ["ts_code", "report_date", "org_name", "author_name", "quarter"]


def _as_naive(series):
    """Drop the timezone after normalising, so date arithmetic stays comparable."""
    stamps = pd.to_datetime(series, errors="coerce", utc=True)
    return stamps.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def load_daily(context):
    """Bounded PIT daily tail: only the columns and calendar window we use."""
    start = (context.inference_at - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[frame["close"] > 0].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce")
    return frame.dropna(subset=["trade_date"]).sort_values(["ts_code", "trade_date"])


def load_universe(context):
    frame = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    frame["list_date"] = pd.to_datetime(frame["list_date"], format="%Y%m%d", errors="coerce")
    return frame


def load_reports(context):
    """Visible ``report_rc`` rows, deduplicated to one row per broker/day/fiscal year.

    ``available_at`` is the only visibility gate: neither ``report_date`` nor
    ``create_time`` may be used for it (rows clear the nightly text node at
    23:15, so they are visible from T-1).  ``quarter`` is the fiscal YEAR label
    (``2024Q4`` = FY2024); rows whose label does not parse are dropped rather
    than guessed.
    """
    frame = pd.read_parquet(context.asof_dir + "/events", columns=REPORT_COLUMNS)
    frame = frame[frame["dataset"] == "report_rc"].copy()
    frame["available_at"] = _as_naive(frame["available_at"])
    asof = pd.Timestamp(context.inference_at).tz_localize(None)
    floor = asof - pd.Timedelta(days=REPORT_LOOKBACK_DAYS)
    frame = frame[(frame["available_at"] <= asof) & (frame["available_at"] > floor)]

    label = frame["quarter"].astype("string")
    fiscal_year = pd.to_numeric(label.str.slice(0, 4), errors="coerce")
    frame = frame[label.str.fullmatch(r"\d{4}Q\d").fillna(False) & fiscal_year.notna()].copy()
    frame["fiscal_year"] = fiscal_year.loc[frame.index].astype(int)

    report_day = pd.to_datetime(frame["report_date"], format="%Y%m%d", errors="coerce")
    frame = frame[report_day.notna()].copy()
    frame["report_year"] = report_day.loc[frame.index].dt.year
    frame = frame[frame["fiscal_year"] == frame["report_year"]]     # current fiscal year only
    frame = frame.dropna(subset=["eps"])

    # The declared business key is NOT unique: the union layer drops only
    # byte-identical rows, and vendor re-pushes carry a different create_time and
    # sometimes a different eps (~0.9% of rows).  Keep the latest create_time per
    # key; available_at breaks the ties that conservative 22:00:00 stamps create.
    frame = frame.sort_values(["create_time", "available_at"])
    return frame.drop_duplicates(BUSINESS_KEY, keep="last")


def consensus_revision(reports, trading_days):
    """Family 1: mean current-FY EPS over the last 20 trading days minus the prior 40.

    Within each window a broker contributes its latest forecast only, so a house
    that publishes weekly does not outvote one that publishes once.
    """
    if len(trading_days) < PRIOR_TRADING_DAYS or reports.empty:
        return pd.Series(dtype="float64")
    recent_start = trading_days[-RECENT_TRADING_DAYS]
    prior_start = trading_days[-PRIOR_TRADING_DAYS]
    stamp = reports["available_at"].dt.normalize()

    recent = reports[stamp >= recent_start]
    prior = reports[(stamp >= prior_start) & (stamp < recent_start)]
    if recent.empty or prior.empty:
        return pd.Series(dtype="float64")

    def _consensus(window):
        latest = window.sort_values("available_at").drop_duplicates(
            ["ts_code", "org_name"], keep="last"
        )
        return latest.groupby("ts_code")["eps"].agg(["mean", "size"])

    now = _consensus(recent)
    before = _consensus(prior)
    joined = now.join(before, lsuffix="_now", rsuffix="_before", how="inner")
    joined = joined[
        (joined["size_now"] >= MIN_BROKERS) & (joined["size_before"] >= MIN_BROKERS)
    ]
    if joined.empty:
        return pd.Series(dtype="float64")
    scale = joined["mean_before"].abs().clip(lower=SCALE_FLOOR)
    revision = (joined["mean_now"] - joined["mean_before"]) / scale
    return revision.replace([np.inf, -np.inf], np.nan).dropna()


def controls(daily, trading_days):
    """Size, momentum, turnover and volatility at the last visible trading day."""
    if len(trading_days) < PRIOR_TRADING_DAYS:
        return pd.DataFrame()
    latest = trading_days[-1]
    frame = daily.copy()
    frame["adj_close"] = frame["close"] * frame["adj_factor"]
    grouped = frame.groupby("ts_code")

    out = pd.DataFrame(index=frame.index)
    out["ts_code"] = frame["ts_code"]
    out["trade_date"] = frame["trade_date"]
    out["close"] = frame["close"]
    out["circ_mv"] = frame["circ_mv"]
    out["is_suspended"] = frame["is_suspended"].astype(bool)
    out["mom20"] = grouped["adj_close"].pct_change(RECENT_TRADING_DAYS)
    out["mom60"] = grouped["adj_close"].pct_change(PRIOR_TRADING_DAYS)
    out["turn20"] = grouped["turnover_rate"].transform(
        lambda s: s.rolling(RECENT_TRADING_DAYS).mean()
    )
    out["adv20"] = grouped["amount"].transform(lambda s: s.rolling(RECENT_TRADING_DAYS).mean())
    daily_return = grouped["adj_close"].pct_change()
    out["vol20"] = daily_return.groupby(frame["ts_code"]).transform(
        lambda s: s.rolling(RECENT_TRADING_DAYS).std()
    )
    cross = out[out["trade_date"] == latest].copy()
    cross["log_size"] = np.log(cross["circ_mv"].clip(lower=1.0))
    return cross.set_index("ts_code")


def _rank(values):
    ranked = pd.Series(values).rank(pct=True)
    return (2.0 * (ranked - 0.5)).to_numpy()


def neutralize(score, control_frame, control_columns):
    """Residual of the winsorized signal against ranked controls, then re-ranked."""
    aligned = control_frame.loc[control_frame.index.intersection(score.index)]
    values = score.loc[aligned.index].astype(float)
    usable = aligned[control_columns].notna().all(axis=1) & np.isfinite(values)
    aligned, values = aligned[usable], values[usable]
    if len(values) < 30:
        return pd.Series(dtype="float64")

    low, high = values.quantile(WINSOR), values.quantile(1.0 - WINSOR)
    target = _rank(values.clip(lower=low, upper=high).to_numpy())
    design = np.column_stack(
        [np.ones(len(aligned))] + [_rank(aligned[c].to_numpy()) for c in control_columns]
    )
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return pd.Series(_rank(residual), index=aligned.index).sort_values(ascending=False)


def eligible(control_frame, universe, min_adv, price_cap, min_listed_days, latest):
    """Declared tradability invariants; none of them is searched on Validation."""
    frame = control_frame.join(universe.set_index("ts_code"), how="inner")
    code = frame.index.to_series()
    keep = (
        (~frame["is_suspended"])
        & (frame["adv20"] >= min_adv)
        & (frame["close"] <= price_cap)
        & (frame["close"] > 0)
        & frame["circ_mv"].notna()
        & (frame["list_date"] <= latest - pd.Timedelta(days=min_listed_days))
        & (~frame["name"].astype("string").str.contains("ST", na=False))
        & (~code.str.startswith(("688", "689")))
        & (~code.str.endswith(".BJ"))
    )
    return frame[keep]
