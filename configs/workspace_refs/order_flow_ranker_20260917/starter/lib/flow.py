"""PIT-safe construction of the two feature faces this arm compares.

One builder serves both the training panel and the decision cross-section, so
the two can never drift apart. `build_face` reads one daily events table per
face and forms the rolling windows; a decision that needs the last window and a
`fit` that needs two years run the same code.

The ofi face comes from the derived events dataset `intraday_flow`: one row per
(ts_code, trade_date), reduced from that trade date's own 1-minute bars by the
data layer, so this arm never mounts the minute domain. Its construction is the
minute-close tick rule

    r_m      = close_m - close_{m-1}, only between contiguous minutes of the
               same stock-day; the 09:30 auction bar has no predecessor and
               11:30 -> 13:01 is the lunch break, so a full 241-bar session
               yields at most 239 signed minutes
    ofi_1d   = sum_m sign(r_m) * vol_m / sum_m vol_m

and `ofi_amt_1d` is the same quantity weighted by turnover. A no-trade minute
repeats the previous close with vol = 0 and contributes zero to both sums, so
paused minutes are neutral by construction.

The two research thresholds stay here, in strategy code, because the data layer
deliberately ships them as diagnostics rather than filters: a stock-day is kept
only when `nret >= MIN_CONTIGUOUS_RETURNS` and it is not a sealed limit board
(`zero_share < MAX_ZERO_BAR_SHARE`, published as `sealed_limit`). Both drop
rates are reported per fold. The rolling windows and the up-day/down-day split
are formed here too, for the same reason.

The moneyflow face is the vendor's daily tick classification. Its eight
`buy_*`/`sell_*_amount` legs are a size-tier split of one turnover, so their
buy and sell totals are equal for 71% of names and their algebraic net is
identically zero: a control built from those eight alone would be artificially
weak. `net_mf_amount` is the vendor's own separate net and is what the study
measured, so it is in the block. See `families.md`.

Units come from the snapshot contract: `intraday_flow.ofi_1d`/`ofi_amt_1d` are
dimensionless ratios in [-1, 1] and `zero_share` a decimal share; `daily`
prices are CNY/share, `circ_mv`/`amount` CNY, `pct_chg` and `turnover_rate`
decimals; events `moneyflow` amounts are 万元, which cancel in every ratio here.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

EPS = 1e-12

# candidate -> (feature face, scorer). `main.CANDIDATE` is the only line that
# differs between the pre-registered legs of one batch.
CANDIDATES = {
    "o1": ("ofi", "ew"),
    "o2": ("ofi", "lgb"),
    "c_mf": ("mf", "lgb"),
    "c_mf_ew": ("mf", "ew"),
}

OFI_BLOCK = ["ofi_21", "ofi_5", "ofi_amt_21", "ofi_up_21", "ofi_dn_21"]
MF_BLOCK = ["mfimb_21", "mfimb_5", "imb_lg_21", "imb_elg_21", "imb_sm_21", "sh_elg_21"]

# Declared directions for the estimator-free composite. Every ofi column is the
# same signed-imbalance quantity under a different weighting or day subset, so
# the probe's positive sign carries to all of them. On the moneyflow side only
# the vendor net was probed; the four tier columns have no declared sign and are
# therefore excluded from the composite, and learned by the LightGBM leg only.
EW_SIGNS = {
    "ofi": {name: 1.0 for name in OFI_BLOCK},
    "mf": {"mfimb_21": 1.0, "mfimb_5": 1.0},
}

NEUTRAL_COLUMNS = ["log_circ_mv", "mom_20", "mom_60", "turn_20", "vol_20", "max_20"]

WINDOW = 21                     # trading days: the probed ofi_21 / mfimb_21 window
MIN_WINDOW_OBS = 15             # rolling min_periods, as probed
FAST_WINDOW = 5
MIN_CONDITIONAL_OBS = 5         # up-day / down-day legs
MIN_CONTIGUOUS_RETURNS = 200    # signed minutes a kept stock-day must have (of 239)
MAX_ZERO_BAR_SHARE = 0.5        # sealed-limit filter; equals the published `sealed_limit`
FACE_LOOKBACK_BUFFER_DAYS = 90  # calendar slack so the first requested date has its window
PRICE_CAP = 30.0                # T-1 close: one 100-share lot <= half a position
ADV_FLOOR = 3.0e7               # CNY
MIN_LISTED_DAYS = 120

MF_TIERS = ("sm", "md", "lg", "elg")
MF_LEGS = tuple(f"{side}_{tier}_amount" for tier in MF_TIERS for side in ("buy", "sell"))


def face_of(candidate):
    return CANDIDATES[candidate][0]


def scorer_of(candidate):
    return CANDIDATES[candidate][1]


def feature_names(candidate):
    """Declared column order; the model persists it and the decision path checks it."""
    return list(OFI_BLOCK if face_of(candidate) == "ofi" else MF_BLOCK)


# --------------------------------------------------------------------------- reads


def read_daily(context, lookback_days):
    """Visible daily rows plus a frozen-anchor forward-adjusted price block."""
    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "high", "low", "close", "pre_close", "vol",
                 "amount", "pct_chg", "adj_factor", "turnover_rate", "circ_mv",
                 "up_limit", "is_suspended"],
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    anchor = frame.groupby("ts_code", sort=False)["adj_factor"].transform("last")
    frame["q_close"] = frame["close"] * frame["adj_factor"] / anchor
    return frame


def read_universe(context):
    return pd.read_parquet(
        context.asof_dir + "/universe",
        columns=["ts_code", "name", "list_date", "l1_code"],
    )


def read_moneyflow(context, lookback_days):
    """The vendor moneyflow face, restricted to rows visible at the decision."""
    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=["dataset", "available_at", "ts_code", "trade_date",
                 "net_mf_amount", *MF_LEGS],
        filters=[("dataset", "=", "moneyflow"), ("trade_date", ">=", start)],
    )
    frame = frame[frame["dataset"] == "moneyflow"].copy()
    visible = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[visible <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.drop(columns=["dataset", "available_at"])


def read_intraday_flow(context, lookback_days):
    """The derived daily order-flow rows visible at the decision, already filtered.

    `nret` and `sealed_limit` travel with every row as diagnostics; the data
    layer does not apply them, so the two research thresholds are applied here
    and their drop rates are what `exploration-plan.md` asks the fold to report.
    """
    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=["dataset", "available_at", "ts_code", "trade_date",
                 "ofi_1d", "ofi_amt_1d", "nret", "zero_share", "sealed_limit"],
        filters=[("dataset", "=", "intraday_flow"), ("trade_date", ">=", start)],
    )
    frame = frame[frame["dataset"] == "intraday_flow"].copy()
    visible = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame[visible <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
    frame["trade_date"] = frame["trade_date"].astype(str)
    keep = ((frame["nret"] >= MIN_CONTIGUOUS_RETURNS)
            & (frame["zero_share"] < MAX_ZERO_BAR_SHARE)
            & frame["ofi_1d"].notna())
    frame = frame[keep].rename(columns={"ofi_1d": "ofi", "ofi_amt_1d": "ofi_amt"})
    return frame[["ts_code", "trade_date", "ofi", "ofi_amt", "nret", "zero_share"]].sort_values(
        ["ts_code", "trade_date"]).reset_index(drop=True)


# ------------------------------------------------------------------ ofi face


def _rolling(frame, column, window, min_obs):
    rolled = frame.groupby("ts_code", sort=False)[column].rolling(window, min_periods=min_obs)
    return rolled.mean().reset_index(level=0, drop=True)


def ofi_face(stock_days, day_sign):
    """The five declared ofi columns, indexed by (ts_code, trade_date)."""
    if stock_days.empty:
        return pd.DataFrame(columns=OFI_BLOCK)
    frame = stock_days.merge(day_sign, on=["ts_code", "trade_date"], how="left")
    frame["ofi_on_up"] = frame["ofi"].where(frame["day_up"] == 1.0)
    frame["ofi_on_dn"] = frame["ofi"].where(frame["day_up"] == 0.0)
    frame["ofi_21"] = _rolling(frame, "ofi", WINDOW, MIN_WINDOW_OBS)
    frame["ofi_5"] = _rolling(frame, "ofi", FAST_WINDOW, FAST_WINDOW - 1)
    frame["ofi_amt_21"] = _rolling(frame, "ofi_amt", WINDOW, MIN_WINDOW_OBS)
    frame["ofi_up_21"] = _rolling(frame, "ofi_on_up", WINDOW, MIN_CONDITIONAL_OBS)
    frame["ofi_dn_21"] = _rolling(frame, "ofi_on_dn", WINDOW, MIN_CONDITIONAL_OBS)
    return frame.set_index(["ts_code", "trade_date"])[OFI_BLOCK]


# --------------------------------------------------------------------------- mf face


def mf_face(moneyflow):
    """The six declared moneyflow columns, indexed by (ts_code, trade_date).

    Every column is a ratio against the same gross (buy + sell over the four
    size tiers, which equals twice the day's turnover), so the vendor's 万元
    unit cancels and no column is a size proxy.
    """
    if moneyflow.empty:
        return pd.DataFrame(columns=MF_BLOCK)
    frame = moneyflow.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    gross = sum(frame[leg] for leg in MF_LEGS).clip(lower=EPS)
    frame["mfimb"] = frame["net_mf_amount"] / gross
    for tier in MF_TIERS:
        frame[f"imb_{tier}"] = (frame[f"buy_{tier}_amount"] - frame[f"sell_{tier}_amount"]) / gross
    frame["sh_elg"] = (frame["buy_elg_amount"] + frame["sell_elg_amount"]) / gross
    frame["mfimb_21"] = _rolling(frame, "mfimb", WINDOW, MIN_WINDOW_OBS)
    frame["mfimb_5"] = _rolling(frame, "mfimb", FAST_WINDOW, FAST_WINDOW - 1)
    for column in ("imb_lg", "imb_elg", "imb_sm", "sh_elg"):
        frame[column + "_21"] = _rolling(frame, column, WINDOW, MIN_WINDOW_OBS)
    return frame.set_index(["ts_code", "trade_date"])[MF_BLOCK]


# ------------------------------------------------------------------- daily context


def daily_context(daily):
    """Size / momentum / turnover / volatility / lottery controls and ADV20.

    These are neutralization and screening inputs, never candidate features:
    putting them in a face would make the hard gate a comparison of two daily
    price-volume models.
    """
    frame = daily[["ts_code", "trade_date"]].copy()
    grouped = daily.groupby("ts_code", sort=False)
    ret = grouped["q_close"].pct_change()
    close = daily["q_close"]
    frame["log_circ_mv"] = np.log(daily["circ_mv"].clip(lower=1.0))
    frame["mom_20"] = close / grouped["q_close"].shift(20) - 1.0
    frame["mom_60"] = close / grouped["q_close"].shift(60) - 1.0
    frame["turn_20"] = grouped["turnover_rate"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    frame["vol_20"] = ret.groupby(daily["ts_code"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).std())
    frame["max_20"] = ret.groupby(daily["ts_code"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).max())
    frame["adv20"] = grouped["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    frame["day_up"] = (daily["pct_chg"] > 0).astype("float64")
    return frame


def build_face(context, candidate, daily, dates):
    """The candidate's feature face over `dates`, indexed by (ts_code, trade_date).

    Both faces are daily events tables now, so the build is a bounded read plus
    a per-stock rolling mean -- no minute domain and no streaming aggregation.
    """
    context_frame = daily_context(daily)
    span = (pd.Timestamp(max(dates)) - pd.Timestamp(min(dates))).days + FACE_LOOKBACK_BUFFER_DAYS
    if face_of(candidate) == "ofi":
        stock_days = read_intraday_flow(context, span)
        block = ofi_face(stock_days, context_frame[["ts_code", "trade_date", "day_up"]])
    else:
        block = mf_face(read_moneyflow(context, span))
    return block, context_frame


def cross_section(block, context_frame, names, feature_date, pool):
    """One standardized feature date restricted to the declared pool.

    `context_frame` is the flat frame `build_face` returns; the neutralization
    columns are attached here so the panel and the decision share one shape.
    """
    index = pd.MultiIndex.from_product([pool, [feature_date]], names=["ts_code", "trade_date"])
    section = block.reindex(index).droplevel("trade_date").dropna(how="all")
    if section.empty:
        return section
    controls = context_frame[context_frame["trade_date"] == feature_date]
    controls = controls.set_index("ts_code").reindex(section.index)
    section = standardize(section, names)
    for column in NEUTRAL_COLUMNS:
        section[column] = controls[column].to_numpy()
    return section


# ----------------------------------------------------------------- shared helpers


def standardize(frame, columns):
    """Cross-sectional median/MAD z-score clipped to +-3."""
    out = frame.copy()
    for column in columns:
        values = out[column].astype("float64").replace([np.inf, -np.inf], np.nan)
        median = values.median()
        mad = (values - median).abs().median() * 1.4826
        if not np.isfinite(mad) or mad <= EPS:
            out[column] = 0.0
            continue
        out[column] = ((values - median) / mad).clip(-3.0, 3.0)
    return out


def equal_weight_score(frame, candidate):
    """The estimator-free leg: signed cross-sectional ranks, equally weighted."""
    signs = EW_SIGNS[face_of(candidate)]
    total = np.zeros(len(frame), dtype="float64")
    used = 0
    for column, sign in sorted(signs.items()):
        if column not in frame.columns:
            continue
        values = frame[column].astype("float64")
        if values.notna().sum() < 20:
            continue
        ranked = values.rank(pct=True)
        total = total + sign * np.nan_to_num(ranked.to_numpy(), nan=0.5)
        used += 1
    if used == 0:
        return total
    return total / used


def neutralize(scores, frame, industry):
    """Residualize against size / momentum / turnover / volatility / max and SW-L1."""
    design = [np.ones((len(scores), 1))]
    for column in NEUTRAL_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].astype("float64").to_numpy()
        design.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1))
    dummies = pd.get_dummies(industry.astype(str), drop_first=True, dtype="float64")
    if dummies.shape[1] > 0:
        design.append(dummies.to_numpy())
    matrix = np.hstack(design)
    target = np.nan_to_num(np.asarray(scores, dtype="float64"), nan=0.0)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    return target - matrix @ solution


def eligible(context, daily, universe, context_frame, feature_date):
    """Declared pool: no BJ/STAR/ST, listed >= 120 days, ADV20 floor, price cap, not halted."""
    day = daily[daily["trade_date"] == feature_date]
    if day.empty:
        return pd.Index([])
    adv = context_frame[context_frame["trade_date"] == feature_date].set_index("ts_code")["adv20"]
    day = day.assign(adv20=adv.reindex(day["ts_code"]).to_numpy())
    day = day[(day["is_suspended"] != True) & (day["close"] <= PRICE_CAP)  # noqa: E712
              & (day["adv20"] >= ADV_FLOOR)]
    codes = day["ts_code"]
    codes = codes[~codes.str.endswith(".BJ") & ~codes.str.startswith(("688", "689"))]
    info = universe.set_index("ts_code").reindex(codes)
    name = info["name"].astype(str)
    keep = ~(name.str.contains("ST") | name.str.contains("退"))
    listed = pd.to_datetime(info["list_date"], format="%Y%m%d", errors="coerce")
    keep = keep & (listed <= pd.Timestamp(feature_date) - pd.Timedelta(days=MIN_LISTED_DAYS))
    return pd.Index(sorted(codes[keep.to_numpy()]))
