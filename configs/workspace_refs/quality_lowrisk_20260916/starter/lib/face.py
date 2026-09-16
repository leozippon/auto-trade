"""PIT-safe construction of the cash-flow quality face and its registered controls.

One builder serves every decision, so a cold and a warm worker score the same
cross-section. Everything is read from the rolling as-of view with column
projection and date filters; nothing is fitted, nothing is cached across calls,
and the heaviest steps are one 60-day rolling regression on a dates x names
matrix and one least squares.

The ranking face is cash-flow earnings quality:

    accr   = (n_income_attr_p - n_cashflow_act) / total_assets
    ocf_ta = n_cashflow_act / total_assets * 4 / quarter
    cfq2   = mean of the percentile ranks of -accr and ocf_ta

from the FIRST announced version of the latest statement (income_vip,
cashflow_vip, balancesheet_vip joined on (ts_code, end_date), report_type "1",
CNY, YTD flows annualised by the quarter number); a statement older than
STATE_MAX_AGE trading days carries no signal.

The risk columns the construction uses are built here too:

    ivol60  = std of the residual of the stock's daily return on the CSI300
              return over the last VOL_WINDOW visible trading days
              (>= MIN_VOL_DAYS); with no macro index the leg is the total
              60-day volatility and every buy order says so (`vol_basis`)
    circ_mv = float market cap, the column the pool floor sorts on

Statements are stamped f_ann_date_or_ann_date 18:00 -> a decision at 08:30 sees
announcements up to T-1; the index row is stamped 17:30 -> T-1. Units come from
the snapshot contract: daily prices CNY/share, `amount` and `circ_mv` CNY,
`turnover_rate` and `pct_chg` decimals; fundamentals CNY; `index_daily.pct_chg`
a percent number (divided by 100 here).
"""

from datetime import timedelta

import numpy as np
import pandas as pd

EPS = 1e-12

# candidate -> which score it ranks on. `main.CANDIDATE` is the only line that
# differs between the pre-registered legs of one batch; `lib.trade.SHAPES` says
# how each of them turns that score into a book.
CANDIDATES = {
    "s_ql": "cfq",       # main candidate: quality ranks, risk-shaped construction
    "c_q15": "cfq",      # control: the same ranks, unshaped
    "c_ivol": "lowvol",  # control: the construction with the quality rank removed
    "c_pool": "pool",    # control: a score-free book spread over the eligible pool
}

BOOK_SIZE = 15                # names held; `lib.trade` holds exactly this many
VOL_WINDOW = 60
MIN_VOL_DAYS = 40
STATE_MAX_AGE = 130           # trading days a statement stays live (about two quarters)
DAILY_LOOKBACK_DAYS = 220     # calendar days of daily rows read per decision: longer than STATE_MAX_AGE trading days, so a statement announced before the window is dead
STATEMENT_LOOKBACK_DAYS = 700 # calendar days of statements read per decision (the year-ago quarter of a 130-day-old statement)
PERIOD_RECENCY_DAYS = 400     # a statement whose period ended earlier than this before its announcement is a restatement of an old period, not an event
INDEX_CODE = "000300.SH"
ASSET_FLOOR = 1.0             # CNY, total_assets floor
GROWTH_FLOOR = 1.0e6          # CNY, |year-ago profit| floor for yoy_np

NEUTRAL_COLUMNS = ["log_circ_mv", "mom_20", "mom_60", "r_5", "turn_20", "vol_20", "max_20", "ep", "yoy_np"]

PRICE_CAP = 30.0              # T-1 close, CNY: one 100-share lot <= half a position
ADV_FLOOR = 3.0e7             # CNY
MIN_LISTED_DAYS = 120


def kind_of(candidate):
    return CANDIDATES[candidate]


# --------------------------------------------------------------------------- reads


def _visible(frame, context):
    stamp = pd.to_datetime(frame["available_at"], utc=True)
    return frame[stamp <= pd.Timestamp(context.inference_at).tz_convert("UTC")]


def read_daily(context):
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close", "pct_chg", "amount", "adj_factor",
                 "turnover_rate", "circ_mv", "pe_ttm", "is_suspended"],
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    anchor = frame.groupby("ts_code", sort=False)["adj_factor"].transform("last")
    frame["q_close"] = frame["close"] * frame["adj_factor"] / anchor
    return frame


def read_universe(context):
    return pd.read_parquet(
        context.asof_dir + "/universe",
        columns=["ts_code", "name", "list_date", "l1_code"],
    )


def read_index_returns(context):
    """CSI300 daily returns as decimals keyed by trade_date; empty when the round carries no macro index."""
    start = (context.inference_at - timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/macro",
        columns=["dataset", "available_at", "ts_code", "trade_date", "pct_chg"],
        filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX_CODE), ("trade_date", ">=", start)],
    )
    frame = frame[(frame["dataset"] == "index_daily") & (frame["ts_code"] == INDEX_CODE)]
    frame = _visible(frame, context)
    frame = frame.dropna(subset=["pct_chg"]).copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    return pd.Series(frame["pct_chg"].astype("float64").to_numpy() / 100.0, index=frame["trade_date"].to_numpy())


def read_statements(context):
    """First announced version of every income / cash-flow / balance-sheet statement visible at the decision."""
    start = (context.inference_at - timedelta(days=STATEMENT_LOOKBACK_DAYS)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/fundamentals",
        columns=["dataset", "available_at", "ts_code", "end_date", "ann_date", "report_type",
                 "n_income_attr_p", "n_cashflow_act", "total_assets"],
        filters=[("dataset", "in", ["income_vip", "cashflow_vip", "balancesheet_vip"]), ("ann_date", ">=", start)],
    )
    frame = _visible(frame, context).copy()
    frame = frame[frame["report_type"] == "1"]
    frame["stamp"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["end_date"] = frame["end_date"].astype(str)
    # A restatement of an old period whose first version predates the read window
    # looks like a first version; only a period that ended recently can be an event.
    ann_day = frame["stamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    period_end = pd.to_datetime(frame["end_date"], format="%Y%m%d", errors="coerce")
    frame = frame[period_end >= ann_day - pd.Timedelta(days=PERIOD_RECENCY_DAYS)]
    parts = {}
    for dataset, column in (("income_vip", "n_income_attr_p"), ("cashflow_vip", "n_cashflow_act"),
                            ("balancesheet_vip", "total_assets")):
        part = frame[(frame["dataset"] == dataset) & frame[column].notna()]
        part = part.sort_values("stamp").drop_duplicates(["ts_code", "end_date"], keep="first")
        parts[dataset] = part[["ts_code", "end_date", "stamp", column]].rename(columns={"stamp": f"stamp_{dataset}"})
    joined = parts["income_vip"].merge(parts["cashflow_vip"], on=["ts_code", "end_date"], how="inner")
    joined = joined.merge(parts["balancesheet_vip"], on=["ts_code", "end_date"], how="inner")
    # visible only when all three parts are: the event is the last of the three stamps
    joined["stamp"] = joined[["stamp_income_vip", "stamp_cashflow_vip", "stamp_balancesheet_vip"]].max(axis=1)
    return joined[["ts_code", "end_date", "stamp", "n_income_attr_p", "n_cashflow_act", "total_assets"]]


# ------------------------------------------------------------------ statement values


def statement_values(statements):
    """One row per statement: the two quality legs and the yoy growth control column."""
    frame = statements.copy()
    frame["qn"] = frame["end_date"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    frame = frame.dropna(subset=["qn"]).sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    assets = frame["total_assets"].clip(lower=ASSET_FLOOR)
    frame["accr"] = (frame["n_income_attr_p"] - frame["n_cashflow_act"]) / assets
    frame["ocf_ta"] = frame["n_cashflow_act"] / assets * (4.0 / frame["qn"])
    # `yoy_np` is a neutralization column only -- the quality face must not be growth.
    # Same caliber as the earnings-surprise pack: cumulative profit against the
    # statement four rows earlier for the same name.
    cum_lag4 = frame.groupby("ts_code", sort=False)["n_income_attr_p"].shift(4)
    frame["yoy_np"] = (frame["n_income_attr_p"] - cum_lag4) / cum_lag4.abs().clip(lower=GROWTH_FLOOR)
    return frame[["ts_code", "end_date", "stamp", "accr", "ocf_ta", "yoy_np"]]


def latest_with_age(events, trading_days, latest_day):
    """Each name's most recent event (by stamp), with its age in visible trading days.

    An event stamped 18:00 on day d is first usable at the next trading day's
    decision; its signal date is the first trading day on or after d, and
    age = index(latest_day) - index(signal date). The latest event is chosen by
    stamp, never by age: every event announced before the window's first day
    would tie at the same age, and a tie-break by row order picks an arbitrary
    -- in practice the oldest -- statement. Such an event is older than the
    window, which DAILY_LOOKBACK_DAYS keeps longer than STATE_MAX_AGE trading
    days, so its age is reported as the window length and it is dead.
    """
    if events.empty:
        return events.assign(age=np.nan).iloc[0:0]
    latest = events.sort_values("stamp").drop_duplicates("ts_code", keep="last").copy()
    days = np.array(trading_days)
    ann_day = latest["stamp"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy()
    idx = np.searchsorted(days, ann_day, side="left")
    latest_idx = int(np.searchsorted(days, latest_day, side="left"))
    latest["age"] = np.where(ann_day < days[0], len(days), latest_idx - idx)
    return latest[latest["age"] >= 0]


# ------------------------------------------------------------------- daily context


def daily_context(daily):
    """Neutralization inputs, risk columns and screening columns at every visible date.

    The NEUTRAL_COLUMNS are never candidate features: `vol_20` and `max_20` are
    what make the 60-day residual column a different leg from the closed 20-day
    lottery face, and `ep`/`yoy_np` are what make the quality face neither value
    nor growth. Dropping any of them changes the mechanism, not just the noise.
    """
    grouped = daily.groupby("ts_code", sort=False)
    ret = grouped["q_close"].pct_change()
    close = daily["q_close"]
    frame = daily[["ts_code", "trade_date", "close", "circ_mv", "is_suspended"]].copy()
    frame["ret"] = ret
    frame["log_circ_mv"] = np.log(daily["circ_mv"].clip(lower=1.0))
    frame["mom_20"] = close / grouped["q_close"].shift(20) - 1.0
    frame["mom_60"] = close / grouped["q_close"].shift(60) - 1.0
    frame["r_5"] = close / grouped["q_close"].shift(5) - 1.0
    frame["turn_20"] = grouped["turnover_rate"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    frame["vol_20"] = ret.groupby(daily["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).std())
    frame["max_20"] = ret.groupby(daily["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).max())
    frame["adv_20"] = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    pe = daily["pe_ttm"].where(daily["pe_ttm"] > 0)
    frame["ep"] = (1.0 / pe).fillna(0.0)
    return frame


def residual_volatility(context_frame, trading_days, market):
    """60-day residual (or total) volatility per name at the latest date.

    Returns (Series keyed by ts_code, basis) with basis "residual" when the
    CSI300 series covers the window and "total" otherwise -- the declared
    fallback, never a silent one.
    """
    window_days = trading_days[-VOL_WINDOW:]
    rows = context_frame[context_frame["trade_date"].isin(window_days)]
    wide = rows.pivot(index="trade_date", columns="ts_code", values="ret").reindex(window_days)
    returns = wide.to_numpy(dtype="float64")
    present = np.isfinite(returns)
    count = present.sum(axis=0)
    filled = np.where(present, returns, 0.0)
    mean_r = filled.sum(axis=0) / np.maximum(count, 1)
    dev_r = np.where(present, returns - mean_r, 0.0)
    var_r = (dev_r ** 2).sum(axis=0) / np.maximum(count - 1, 1)
    basis = "total"
    resid_var = var_r
    m = market.reindex(window_days).to_numpy(dtype="float64") if market is not None and len(market) else None
    if m is not None and np.isfinite(m).sum() >= MIN_VOL_DAYS:
        m_ok = np.isfinite(m)
        both = present & m_ok[:, None]
        n_both = both.sum(axis=0)
        m_filled = np.where(m_ok, m, 0.0)
        mean_m = (np.where(both, m_filled[:, None], 0.0)).sum(axis=0) / np.maximum(n_both, 1)
        dev_m = np.where(both, m_filled[:, None] - mean_m, 0.0)
        dev_rb = np.where(both, returns - mean_r, 0.0)
        cov = (dev_rb * dev_m).sum(axis=0) / np.maximum(n_both - 1, 1)
        var_m = (dev_m ** 2).sum(axis=0) / np.maximum(n_both - 1, 1)
        beta = np.where(var_m > EPS, cov / np.where(var_m > EPS, var_m, 1.0), 0.0)
        resid_var = np.clip(var_r - beta * cov, 0.0, None)
        basis = "residual"
    vol = np.sqrt(resid_var)
    vol = np.where(count >= MIN_VOL_DAYS, vol, np.nan)
    return pd.Series(vol, index=wide.columns), basis


def eligible(section, universe, latest_day):
    """Declared pool, shared by every leg: no BJ/STAR/ST, listed >= 120 days,
    ADV floor, price cap, not halted, enough volatility days."""
    codes = section["ts_code"]
    keep = (~codes.str.endswith(".BJ")) & (~codes.str.startswith(("688", "689")))
    keep &= (section["close"] <= PRICE_CAP) & (section["adv_20"] >= ADV_FLOOR)
    keep &= ~section["is_suspended"].fillna(False).astype(bool)
    keep &= section["ivol60"].notna()
    info = universe.set_index("ts_code").reindex(codes)
    name = info["name"].astype(str)
    keep &= ~(name.str.contains("ST") | name.str.contains("退")).to_numpy()
    listed = pd.to_datetime(info["list_date"], format="%Y%m%d", errors="coerce")
    keep &= (listed <= pd.Timestamp(latest_day) - pd.Timedelta(days=MIN_LISTED_DAYS)).to_numpy()
    return section[keep.to_numpy()]


# --------------------------------------------------------------------- scoring


def pct_rank(values):
    return values.astype("float64").replace([np.inf, -np.inf], np.nan).rank(pct=True)


def _pool_spread(size):
    """A score-free ranking that spreads BOOK_SIZE names evenly over the pool.

    The eligible pool cannot be held on this account -- a thousand names at one
    lot each is an order of magnitude more cash than the book has -- so the
    equal-weight-pool control is a stratified sample of it at book size: the
    names sitting at BOOK_SIZE evenly spaced positions of the pool ordered by
    `ts_code`, which spreads the sample across boards and code ranges and
    carries no information about either leg. Descending order puts those
    BOOK_SIZE names first and their neighbours behind them, so the keep band
    and the swap cap work exactly as for a scored leg.
    """
    positions = np.arange(size)
    anchors = np.round(np.arange(BOOK_SIZE) * size / BOOK_SIZE).astype(int)
    return -np.abs(positions[:, None] - anchors[None, :]).min(axis=1).astype("float64")


def build_cross_section(context, candidate):
    """Scored, neutralization-ready cross-section at the latest visible date.

    Returns (frame, latest_day, vol_basis): one row per eligible name carrying
    `raw` (the candidate's score before neutralization), the NEUTRAL_COLUMNS,
    `ivol60` and `circ_mv` for the construction, `close` for lot sizing and
    `l1_code`. Names without a live statement (age <= STATE_MAX_AGE) or without
    the volatility window are excluded, for every leg alike: the pool is the
    same for the candidate and its controls or their readings do not compare.
    """
    daily = read_daily(context)
    if daily.empty:
        return None, None, "none"
    trading_days = sorted(daily["trade_date"].unique())
    latest_day = trading_days[-1]
    universe = read_universe(context)
    context_frame = daily_context(daily)
    vol, vol_basis = residual_volatility(context_frame, trading_days, read_index_returns(context))
    section = context_frame[context_frame["trade_date"] == latest_day].copy()
    section["ivol60"] = section["ts_code"].map(vol)
    statements = read_statements(context)
    latest = latest_with_age(statement_values(statements), trading_days, latest_day)
    latest = latest[latest["age"] <= STATE_MAX_AGE]
    section = section.merge(latest[["ts_code", "age", "accr", "ocf_ta", "yoy_np"]], on="ts_code", how="inner")
    section["yoy_np"] = section["yoy_np"].fillna(0.0)
    section = eligible(section, universe, latest_day)
    section = section[section["accr"].notna() & section["ocf_ta"].notna()].copy()
    if section.empty:
        return None, latest_day, vol_basis
    section = section.sort_values("ts_code").reset_index(drop=True)
    kind = kind_of(candidate)
    if kind == "cfq":
        raw = (1.0 - pct_rank(section["accr"]) + pct_rank(section["ocf_ta"])) / 2.0
    elif kind == "lowvol":
        raw = 1.0 - pct_rank(section["ivol60"])
    else:  # pool: no face at all
        raw = pd.Series(_pool_spread(len(section)), index=section.index)
    section["raw"] = raw
    section = section[section["raw"].notna()].copy()
    industry = universe.set_index("ts_code")["l1_code"].reindex(section["ts_code"]).fillna("NA").to_numpy()
    section["l1_code"] = industry
    return section.reset_index(drop=True), latest_day, vol_basis


def neutralize(section, candidate):
    """Residualize the raw score on the NEUTRAL_COLUMNS ranks and SW-L1 dummies.

    The pool control has no face to residualize -- neutralizing an even spread
    would turn a score-free book into a book of whatever the residual happens to
    rank -- so it keeps its spread and its construction stays unshaped.
    """
    if kind_of(candidate) == "pool" or len(section) < 30:
        return section["raw"].to_numpy(dtype="float64")
    design = [np.ones((len(section), 1))]
    for column in NEUTRAL_COLUMNS:
        ranked = section[column].astype("float64").rank(pct=True).fillna(0.5).to_numpy()
        design.append(ranked.reshape(-1, 1))
    dummies = pd.get_dummies(section["l1_code"].astype(str), drop_first=True, dtype="float64")
    if dummies.shape[1] > 0:
        design.append(dummies.to_numpy())
    matrix = np.hstack(design)
    target = np.nan_to_num(section["raw"].to_numpy(dtype="float64"), nan=0.0)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    return target - matrix @ solution
