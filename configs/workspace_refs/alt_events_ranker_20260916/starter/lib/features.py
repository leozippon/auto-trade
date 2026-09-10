"""PIT-safe feature construction from the as-of view.

One builder serves both the training panel and the decision cross-section, so
the two can never drift apart (the `panel-semantics-parity` lesson). Every
event table is read once, then restricted to ``available_at <= as_of(T)`` for
whichever feature date ``T`` is being built; the declared visibility semantics
are **per dataset, latest visible**, never a cross-dataset union.

as_of(T) = Timestamp(T) + 1 day at 08:30, capped at ``context.inference_at``.
It is uniform for the panel and the decision, and never later than the real
decision instant, so a Friday feature date deliberately drops rows announced
over the weekend rather than letting the two sides disagree.

Units follow the snapshot contract: daily prices CNY/share, `amount`/`circ_mv`
CNY, `turnover_rate`/`pct_chg` decimals; block_trade `vol` 万股 / `amount` 万元;
margin_detail balances CNY and `rqyl` shares; winner_rate / float_ratio /
change_ratio / hold_float_ratio are percent numbers.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

EPS = 1e-12

PRICE_BLOCK = [
    "mom_20", "mom_60_skip5", "rev_5", "vol_20", "vol_60", "max_20", "turn_20",
    "turn_ratio", "amihud_20", "log_circ_mv", "close_pos_60", "ep", "bp", "dist_up_limit",
]
ANALYST_BLOCK = [
    "eps_rev_fy0", "eps_rev_fy1", "np_rev_fy0", "rev_breadth_60", "disp_60",
    "cover_60", "cover_chg", "rating_score_60", "has_analyst",
]
MARGIN_BLOCK = [
    "rzye_chg_20", "rzye_chg_5", "rzye_to_circ", "rzmre_share_5", "rq_ratio",
    "rzye_z_60", "has_margin",
]
BLOCK_BLOCK = [
    "disc_pre_20", "disc_close_20", "press_60", "inst_buy_share_20",
    "inst_sell_share_20", "n_events_60", "has_block_20",
]
HOLDER_BLOCK = [
    "hn_chg_1", "hn_chg_2", "log_hn", "mv_per_holder", "insider_net_60",
    "insider_net_180", "top10_float_chg", "has_holder",
]
CHIP_BLOCK = [
    "winner_rate", "winner_chg_20", "cost_spread", "px_to_cost50", "px_to_wavg",
    "unlock_next90", "unlock_past30", "has_unlock_90",
]
EVENT_BLOCKS = {
    "A": ANALYST_BLOCK, "B": MARGIN_BLOCK, "C": BLOCK_BLOCK,
    "D": HOLDER_BLOCK, "E": CHIP_BLOCK,
}

# Equal-weight control signs (families.md); columns absent here stay out of C-EW.
EW_SIGNS = {
    "eps_rev_fy0": 1.0, "eps_rev_fy1": 1.0, "np_rev_fy0": 1.0, "rev_breadth_60": 1.0,
    "disp_60": -1.0, "cover_chg": 1.0, "rating_score_60": 1.0,
    "rzye_chg_20": -1.0, "rzye_chg_5": -1.0, "rq_ratio": -1.0,
    "disc_pre_20": 1.0, "inst_buy_share_20": 1.0, "press_60": -1.0,
    "hn_chg_1": -1.0, "insider_net_60": 1.0, "top10_float_chg": 1.0,
    "winner_rate": -1.0, "unlock_next90": -1.0,
}

RATING_MAP = {
    "买入": 5.0, "强烈推荐": 5.0, "推荐": 4.5, "优于大市": 4.0, "跑赢行业": 4.0,
    "增持": 4.0, "谨慎推荐": 3.5, "审慎推荐": 3.5, "中性": 3.0, "持有": 3.0,
    "同步大市": 3.0, "观望": 2.5, "减持": 2.0, "回避": 1.5, "卖出": 1.0,
}

NEUTRAL_COLUMNS = ["log_circ_mv", "mom_20", "turn_20", "vol_60"]


def feature_names(daily_only):
    """Declared column order; the model persists it and the decision path checks it."""
    if daily_only:
        return list(PRICE_BLOCK)
    names = list(PRICE_BLOCK)
    for key in ("A", "B", "C", "D", "E"):
        names.extend(EVENT_BLOCKS[key])
    return names


# --------------------------------------------------------------------------- reads


def read_daily(context, lookback_days):
    """Visible daily rows plus a frozen-anchor forward-adjusted price block."""
    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol",
            "amount", "pct_chg", "adj_factor", "turnover_rate", "circ_mv", "pe_ttm", "pb",
            "up_limit", "is_suspended",
        ],
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[(frame["close"] > 0) & frame["adj_factor"].notna()]
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    anchor = frame.groupby("ts_code", sort=False)["adj_factor"].transform("last")
    for column in ("open", "high", "low", "close"):
        frame["q_" + column] = frame[column] * frame["adj_factor"] / anchor
    return frame


def read_universe(context):
    return pd.read_parquet(
        context.asof_dir + "/universe",
        columns=["ts_code", "name", "list_date", "l1_code"],
    )


def read_events(context, dataset, columns):
    """One events dataset, already filtered to rows visible at the decision."""
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=["dataset", "available_at"] + columns,
        filters=[("dataset", "=", dataset)],
    )
    frame = frame[frame["dataset"] == dataset].copy()
    frame["visible_at"] = pd.to_datetime(frame["available_at"], utc=True)
    limit = pd.Timestamp(context.inference_at).tz_convert("UTC")
    return frame[frame["visible_at"] <= limit].drop(columns=["dataset", "available_at"])


def empty_tables():
    """Shape-compatible empties for the daily-only control, which reads no events."""
    columns = ["ts_code", "visible_at", "_vis_ns"]
    return {key: pd.DataFrame(columns=columns) for key in
            ("rc", "md", "bt", "cy", "hn", "ht", "tf", "sf")}


def read_all_events(context):
    """Every table this pack uses, one read each. Empty frames stay usable."""
    tables = {}
    tables["rc"] = read_events(context, "report_rc", [
        "ts_code", "report_date", "create_time", "org_name", "author_name",
        "quarter", "eps", "np", "rating"])
    tables["md"] = read_events(context, "margin_detail", [
        "ts_code", "trade_date", "rzye", "rqye", "rzmre"])
    tables["bt"] = read_events(context, "block_trade", [
        "ts_code", "trade_date", "price", "vol", "amount", "buyer", "seller"])
    tables["cy"] = read_events(context, "cyq_perf", [
        "ts_code", "trade_date", "winner_rate", "cost_15pct", "cost_50pct",
        "cost_85pct", "weight_avg"])
    tables["hn"] = read_events(context, "stk_holdernumber", [
        "ts_code", "ann_date", "end_date", "holder_num"])
    tables["ht"] = read_events(context, "stk_holdertrade", [
        "ts_code", "ann_date", "holder_name", "in_de", "change_vol"])
    tables["tf"] = read_events(context, "top10_floatholders", [
        "ts_code", "ann_date", "end_date", "holder_name", "hold_float_ratio"])
    tables["sf"] = read_events(context, "share_float_complete", [
        "ts_code", "ann_date", "float_date", "float_ratio", "holder_name", "share_type"])
    return tables


# --------------------------------------------------------------------------- helpers


def as_of(context, feature_date):
    """Uniform as-of instant for one feature date, never past the real decision."""
    stamp = pd.Timestamp(feature_date) + pd.Timedelta(days=1, hours=8, minutes=30)
    stamp = stamp.tz_localize("Asia/Shanghai").tz_convert("UTC")
    return min(stamp, pd.Timestamp(context.inference_at).tz_convert("UTC"))


def sort_by_visible(frame):
    """Sort once on visibility and cache the nanosecond key the window slicer needs.

    Every per-date read is then a slice, not a full-table scan: with ~100 feature
    dates and multi-million-row event tables the scan is what makes `fit` miss
    its budget.
    """
    if frame.empty:
        return frame.assign(_vis_ns=pd.Series(dtype="int64"))
    out = frame.sort_values("visible_at").reset_index(drop=True)
    return out.assign(_vis_ns=out["visible_at"].astype("int64"))


def _window(frame, end_ts, lo_days, hi_days):
    """Rows visible in (end - hi_days, end - lo_days]; requires `sort_by_visible`."""
    if frame.empty:
        return frame
    keys = frame["_vis_ns"].to_numpy()
    low = (end_ts - pd.Timedelta(days=hi_days)).value
    high = (end_ts - pd.Timedelta(days=lo_days)).value
    return frame.iloc[int(np.searchsorted(keys, low, side="right")):
                      int(np.searchsorted(keys, high, side="right"))]


def _upto(frame, end_ts):
    """Rows visible at or before `end_ts`; requires `sort_by_visible`."""
    if frame.empty:
        return frame
    keys = frame["_vis_ns"].to_numpy()
    return frame.iloc[:int(np.searchsorted(keys, end_ts.value, side="right"))]


def daily_panel(frame, date_column):
    """Index a dense per-trading-day event table by (ts_code, date position).

    `margin_detail` and `cyq_perf` carry one row per stock per trading day, so
    "the k-th visible row back" is just a date position: resolve it once here
    instead of re-sorting the whole table at every feature date.
    """
    if frame.empty:
        return {"frame": frame, "vis": np.zeros(0, dtype="int64"), "dates": []}
    dates = sorted(frame[date_column].unique())
    position = {value: index for index, value in enumerate(dates)}
    indexed = frame.assign(date_idx=frame[date_column].map(position))
    visible = indexed.groupby("date_idx")["visible_at"].max().sort_index()
    indexed = indexed.set_index(["date_idx", "ts_code"]).sort_index()
    return {"frame": indexed, "vis": visible.astype("int64").to_numpy(), "dates": dates}


def _panel_cut(panel, end_ts):
    """Highest date position whose rows are visible at `end_ts`, or None."""
    if not len(panel["vis"]):
        return None
    cut = int(np.searchsorted(panel["vis"], end_ts.value, side="right")) - 1
    return cut if cut >= 0 else None


def _panel_at(panel, cut, lag, columns):
    """Rows of the panel `lag` trading days before the cut, indexed by ts_code."""
    position = cut - lag
    if position < 0:
        return pd.DataFrame(columns=columns)
    try:
        return panel["frame"].xs(position, level="date_idx")[columns]
    except KeyError:
        return pd.DataFrame(columns=columns)


def _panel_tail(panel, cut, window, column):
    """`window` trading days ending at the cut, as a groupby on ts_code."""
    low = max(0, cut - window + 1)
    section = panel["frame"].loc[(slice(low, cut), slice(None)), column]
    return section.groupby(level="ts_code")


def _rank_from_end(frame, key, order):
    """0 = most recent row of each `key` group after sorting on `order`."""
    out = frame.sort_values(list(key) + list(order))
    out = out.assign(_rk=out.groupby(list(key), sort=False).cumcount(ascending=False))
    return out


def _roll(frame, column, window, how):
    rolled = frame.groupby("ts_code", sort=False)[column].rolling(
        window, min_periods=max(3, window // 2))
    return getattr(rolled, how)().reset_index(level=0, drop=True)


def _safe_ratio(numerator, denominator, floor):
    denominator = denominator.abs().clip(lower=floor)
    return numerator / denominator


# --------------------------------------------------------------------------- block P


def price_block(daily):
    """14 daily price/volume columns, one row per (ts_code, trade_date)."""
    out = daily[["ts_code", "trade_date"]].copy()
    grouped = daily.groupby("ts_code", sort=False)
    ret = grouped["q_close"].pct_change()
    daily = daily.assign(_ret=ret, _abs_ret_amt=ret.abs() / daily["amount"].clip(lower=1.0))
    out["mom_20"] = grouped["q_close"].pct_change(20)
    out["mom_60_skip5"] = grouped["q_close"].shift(5) / grouped["q_close"].shift(60) - 1.0
    out["rev_5"] = -grouped["q_close"].pct_change(5)
    out["vol_20"] = _roll(daily, "_ret", 20, "std")
    out["vol_60"] = _roll(daily, "_ret", 60, "std")
    out["max_20"] = _roll(daily, "pct_chg", 20, "max")
    out["turn_20"] = _roll(daily, "turnover_rate", 20, "mean")
    out["turn_ratio"] = _roll(daily, "turnover_rate", 5, "mean") / (
        _roll(daily, "turnover_rate", 60, "mean") + EPS)
    out["amihud_20"] = _roll(daily, "_abs_ret_amt", 20, "mean") * 1e9
    out["log_circ_mv"] = np.log(daily["circ_mv"].clip(lower=1.0))
    high60 = _roll(daily, "q_high", 60, "max")
    low60 = _roll(daily, "q_low", 60, "min")
    out["close_pos_60"] = (daily["q_close"] - low60) / (high60 - low60 + EPS)
    out["ep"] = np.where(daily["pe_ttm"] > 0, 1.0 / daily["pe_ttm"], np.nan)
    out["bp"] = np.where(daily["pb"] > 0, 1.0 / daily["pb"], np.nan)
    out["dist_up_limit"] = (daily["up_limit"] - daily["close"]) / daily["close"]
    return out


# --------------------------------------------------------------------------- block A


def _rc_prepare(rc):
    if rc.empty:
        return rc
    out = rc.copy()
    # `quarter` is a fiscal-year label ("2024Q4"), but the raw table also
    # carries bare "Q" and NULL (about 0.3% of rows); a hard cast raises there,
    # so coerce and let the NaN fall out of the fy_off filter downstream.
    out["fy_off"] = (
        pd.to_numeric(out["quarter"].astype(str).str[:4], errors="coerce")
        - pd.to_numeric(out["report_date"].astype(str).str[:4], errors="coerce")
    )
    out["rating_score"] = out["rating"].map(RATING_MAP)
    return out


def _rc_latest(rc, end_ts, lo_days, hi_days, fy_off):
    """Business-key dedup (max create_time, deterministic tie-break) then per-broker latest."""
    window = _window(rc, end_ts, lo_days, hi_days)
    if window.empty:
        return window
    window = window[window["fy_off"] == fy_off]
    if window.empty:
        return window
    window = window.sort_values(["create_time", "report_date", "org_name", "author_name"])
    window = window.drop_duplicates(
        ["ts_code", "report_date", "org_name", "author_name", "quarter"], keep="last")
    return window.drop_duplicates(["ts_code", "org_name"], keep="last")


def analyst_features(rc, end_ts, index):
    out = pd.DataFrame(index=index, columns=ANALYST_BLOCK, dtype="float64")
    if rc.empty:
        out["has_analyst"] = 0.0
        return out
    near0 = _rc_latest(rc, end_ts, 0, 20, 0.0)
    far0 = _rc_latest(rc, end_ts, 20, 60, 0.0)
    near1 = _rc_latest(rc, end_ts, 0, 20, 1.0)
    far1 = _rc_latest(rc, end_ts, 20, 60, 1.0)
    wide0 = _rc_latest(rc, end_ts, 0, 60, 0.0)
    prior0 = _rc_latest(rc, end_ts, 60, 120, 0.0)

    def revision(near, far, column, floor):
        if near.empty or far.empty:
            return pd.Series(np.nan, index=index)
        mn = near.groupby("ts_code")[column].mean()
        mf = far.groupby("ts_code")[column].mean()
        cn = near.groupby("ts_code")["org_name"].nunique()
        cf = far.groupby("ts_code")["org_name"].nunique()
        value = _safe_ratio(mn - mf, mf, floor)
        value = value.where((cn >= 2) & (cf >= 2))
        return value.reindex(index)

    out["eps_rev_fy0"] = revision(near0, far0, "eps", 0.05)
    out["eps_rev_fy1"] = revision(near1, far1, "eps", 0.05)
    out["np_rev_fy0"] = revision(near0, far0, "np", 1e3)
    if not wide0.empty:
        grouped = wide0.groupby("ts_code")
        std = grouped["eps"].std()
        mean = grouped["eps"].mean()
        count = grouped["org_name"].nunique()
        out["disp_60"] = _safe_ratio(std, mean, 0.05).where(count >= 3).reindex(index)
        out["cover_60"] = np.log1p(count).reindex(index)
        out["rating_score_60"] = grouped["rating_score"].mean().reindex(index)
        out["has_analyst"] = count.reindex(index).notna().astype("float64")
    else:
        out["has_analyst"] = 0.0
    if not prior0.empty and not wide0.empty:
        prior = prior0.groupby("ts_code")["org_name"].nunique()
        now = wide0.groupby("ts_code")["org_name"].nunique()
        out["cover_chg"] = (np.log1p(now) - np.log1p(prior.reindex(now.index).fillna(0.0))).reindex(index)
    out["rev_breadth_60"] = _revision_breadth(rc, end_ts).reindex(index)
    out["has_analyst"] = out["has_analyst"].fillna(0.0)
    return out


def _revision_breadth(rc, end_ts):
    """(#up - #down)/#pairs over each broker's last two visible FY0 forecasts."""
    window = _window(rc, end_ts, 0, 60)
    window = window[window["fy_off"] == 0.0] if not window.empty else window
    if window.empty:
        return pd.Series(dtype="float64")
    window = window.sort_values(["create_time", "report_date", "org_name", "author_name"])
    window = window.drop_duplicates(
        ["ts_code", "report_date", "org_name", "author_name", "quarter"], keep="last")
    pair = window.groupby(["ts_code", "org_name"], sort=False).tail(2)
    agg = pair.groupby(["ts_code", "org_name"], sort=False)["eps"].agg(["first", "last", "size"])
    agg = agg[agg["size"] >= 2]
    if agg.empty:
        return pd.Series(dtype="float64")
    sign = pd.Series(np.sign(agg["last"] - agg["first"]).to_numpy(), index=agg.index)
    breadth = sign.groupby(level=0).mean()
    pairs = sign.groupby(level=0).size()
    return breadth.where(pairs >= 3)


# --------------------------------------------------------------------------- block B


def margin_features(md, daily_last, amount_lookup, end_ts, index):
    out = pd.DataFrame(index=index, columns=MARGIN_BLOCK, dtype="float64")
    out["has_margin"] = 0.0
    cut = _panel_cut(md, end_ts)
    if cut is None:
        return out
    latest = _panel_at(md, cut, 0, ["rzye", "rqye", "rzmre", "trade_date"])
    if latest.empty:
        return out
    for name, lag in (("rzye_chg_20", 20), ("rzye_chg_5", 5)):
        older = _panel_at(md, cut, lag, ["rzye"])["rzye"]
        out[name] = (latest["rzye"] / older.reindex(latest.index).clip(lower=1.0) - 1.0).reindex(index)
    out["rzye_to_circ"] = (latest["rzye"] / daily_last["circ_mv"].reindex(latest.index)).reindex(index)
    out["rq_ratio"] = (latest["rqye"] / latest["rzye"].clip(lower=1.0)).reindex(index)
    shares = []
    for lag in range(5):
        row = _panel_at(md, cut, lag, ["rzye", "rqye", "rzmre", "trade_date"])
        if row.empty:
            continue
        keys = pd.MultiIndex.from_arrays([row.index, row["trade_date"]])
        amount = pd.Series(amount_lookup.reindex(keys).to_numpy(), index=row.index)
        shares.append((row["rzmre"] / amount).replace([np.inf, -np.inf], np.nan))
    if shares:
        out["rzmre_share_5"] = pd.concat(shares, axis=1).mean(axis=1).reindex(index)
    tail = _panel_tail(md, cut, 60, "rzye")
    out["rzye_z_60"] = ((latest["rzye"] - tail.mean()) / (tail.std() + EPS)).reindex(index)
    out["has_margin"] = latest["rzye"].notna().reindex(index).fillna(False).astype("float64")
    return out


# --------------------------------------------------------------------------- block C


def block_events(bt, daily):
    """Level one: rows -> (ts_code, trade_date) events, priced off the same day's daily row."""
    if bt.empty:
        return bt
    frame = bt.copy()
    frame["amount_cny"] = frame["amount"] * 1e4
    prices = daily[["ts_code", "trade_date", "close", "pre_close"]]
    frame = frame.merge(prices, on=["ts_code", "trade_date"], how="inner")
    if frame.empty:
        return frame
    # Bridge trades (identical buyer and seller seat, 9.4% of rows) are self-transfers:
    # they leave both sides of the institutional share, numerator and denominator.
    bridge = frame["buyer"].astype(str) == frame["seller"].astype(str)
    frame["disc_pre"] = frame["price"] / frame["pre_close"] - 1.0
    frame["disc_close"] = frame["price"] / frame["close"] - 1.0
    frame["flow_amount"] = np.where(bridge, 0.0, frame["amount_cny"])
    frame["inst_buy"] = np.where(
        (frame["buyer"].astype(str) == "机构专用") & (~bridge), frame["amount_cny"], 0.0)
    frame["inst_sell"] = np.where(
        (frame["seller"].astype(str) == "机构专用") & (~bridge), frame["amount_cny"], 0.0)
    frame["w_pre"] = frame["disc_pre"] * frame["amount_cny"]
    frame["w_close"] = frame["disc_close"] * frame["amount_cny"]
    event = frame.groupby(["ts_code", "trade_date"], as_index=False).agg(
        amount_cny=("amount_cny", "sum"), flow_amount=("flow_amount", "sum"),
        w_pre=("w_pre", "sum"), w_close=("w_close", "sum"),
        inst_buy=("inst_buy", "sum"), inst_sell=("inst_sell", "sum"),
        visible_at=("visible_at", "max"))
    return event


def block_features(events, adv_sum60, end_ts, index):
    out = pd.DataFrame(index=index, columns=BLOCK_BLOCK, dtype="float64")
    out["has_block_20"] = 0.0
    if events.empty:
        return out
    near = _window(events, end_ts, 0, 30)
    wide = _window(events, end_ts, 0, 90)
    if not near.empty:
        grouped = near.groupby("ts_code")
        total = grouped["amount_cny"].sum()
        out["disc_pre_20"] = (grouped["w_pre"].sum() / total.clip(lower=1.0)).reindex(index)
        out["disc_close_20"] = (grouped["w_close"].sum() / total.clip(lower=1.0)).reindex(index)
        flow = grouped["flow_amount"].sum()
        out["inst_buy_share_20"] = (
            grouped["inst_buy"].sum() / flow.clip(lower=1.0)).where(flow > 0).reindex(index)
        out["inst_sell_share_20"] = (
            grouped["inst_sell"].sum() / flow.clip(lower=1.0)).where(flow > 0).reindex(index)
        out["has_block_20"] = total.reindex(index).notna().astype("float64")
    if not wide.empty:
        grouped = wide.groupby("ts_code")
        out["press_60"] = (grouped["amount_cny"].sum() / adv_sum60.reindex(index).clip(lower=1.0)).reindex(index)
        out["n_events_60"] = np.log1p(grouped.size()).reindex(index)
    return out


# --------------------------------------------------------------------------- block D

REPORT_ENDS = ("0331", "0630", "0930", "1231")


def _report_periods(frame, end_ts):
    """Visible rows, exact-duplicate dropped, latest ann_date per (ts_code, end_date),
    restricted to the four report-period ends."""
    visible = _upto(frame, end_ts)
    if visible.empty:
        return visible
    visible = visible.drop_duplicates()
    visible = visible[visible["end_date"].astype(str).str[4:].isin(REPORT_ENDS)]
    if visible.empty:
        return visible
    keep = visible.groupby(["ts_code", "end_date"])["ann_date"].transform("max")
    return visible[visible["ann_date"] == keep]


def holder_features(hn, ht, tf, daily_last, end_ts, index):
    out = pd.DataFrame(index=index, columns=HOLDER_BLOCK, dtype="float64")
    out["has_holder"] = 0.0
    periods = _report_periods(hn, end_ts)
    if not periods.empty:
        ranked = _rank_from_end(periods.drop_duplicates(["ts_code", "end_date"]),
                                ("ts_code",), ("end_date",))
        level = {k: ranked[ranked["_rk"] == k].set_index("ts_code")["holder_num"] for k in (0, 1, 2)}
        out["hn_chg_1"] = (level[0] / level[1].reindex(level[0].index) - 1.0).reindex(index)
        out["hn_chg_2"] = (level[0] / level[2].reindex(level[0].index) - 1.0).reindex(index)
        out["log_hn"] = np.log(level[0].clip(lower=1.0)).reindex(index)
        out["mv_per_holder"] = np.log(
            (daily_last["circ_mv"].reindex(level[0].index) / level[0].clip(lower=1.0)).clip(lower=1.0)
        ).reindex(index)
        out["has_holder"] = level[1].reindex(index).notna().astype("float64")
    if not ht.empty:
        float_shares = (daily_last["circ_mv"] / daily_last["close"]).clip(lower=1.0)
        signed = ht.assign(signed=np.where(ht["in_de"].astype(str) == "IN", 1.0, -1.0)
                           * ht["change_vol"].fillna(0.0))
        signed = signed.drop_duplicates(["ts_code", "ann_date", "holder_name", "in_de", "change_vol"])
        for name, days in (("insider_net_60", 60), ("insider_net_180", 180)):
            window = _window(signed, end_ts, 0, days)
            if window.empty:
                continue
            net = window.groupby("ts_code")["signed"].sum()
            out[name] = (net / float_shares.reindex(net.index)).reindex(index)
    periods_tf = _report_periods(tf, end_ts)
    if not periods_tf.empty:
        total = periods_tf.drop_duplicates(
            ["ts_code", "end_date", "holder_name", "hold_float_ratio"]
        ).groupby(["ts_code", "end_date"], as_index=False)["hold_float_ratio"].sum()
        ranked = _rank_from_end(total, ("ts_code",), ("end_date",))
        cur = ranked[ranked["_rk"] == 0].set_index("ts_code")["hold_float_ratio"]
        prev = ranked[ranked["_rk"] == 1].set_index("ts_code")["hold_float_ratio"]
        out["top10_float_chg"] = (cur - prev.reindex(cur.index)).reindex(index)
    return out


# --------------------------------------------------------------------------- block E


def unlock_events(sf):
    """Collapse the per-holder unlock rows to one row per announcement version.

    The vendor union re-announces the same (ts_code, float_date) under 2-5
    ann_dates, so summing across versions double-counts (measured maximum 371%
    of total capital). Summing the per-holder rows *inside* one version is
    correct; picking which version is in effect is a per-decision question, so
    it stays in `chip_features` rather than being resolved once here.
    """
    if sf.empty:
        return sf.assign(_vis_ns=pd.Series(dtype="int64"), ratio=pd.Series(dtype="float64"))
    frame = sf.drop_duplicates()
    versions = frame.groupby(["ts_code", "float_date", "ann_date"], as_index=False).agg(
        ratio=("float_ratio", "sum"), visible_at=("visible_at", "max"))
    return sort_by_visible(versions)


def chip_features(cy, unlocks, daily_last, feature_date, end_ts, index):
    out = pd.DataFrame(index=index, columns=CHIP_BLOCK, dtype="float64")
    out["has_unlock_90"] = 0.0
    cut = _panel_cut(cy, end_ts)
    if cut is not None:
        columns = ["winner_rate", "cost_15pct", "cost_50pct", "cost_85pct", "weight_avg"]
        latest = _panel_at(cy, cut, 0, columns)
        older = _panel_at(cy, cut, 20, columns).reindex(columns=columns).astype("float64")
    if cut is not None and not latest.empty:
        rate = latest["winner_rate"].clip(lower=0.0, upper=100.0)
        out["winner_rate"] = rate.reindex(index)
        out["winner_chg_20"] = (
            rate - older["winner_rate"].clip(lower=0.0, upper=100.0).reindex(rate.index)
        ).reindex(index)
        out["cost_spread"] = (
            (latest["cost_85pct"] - latest["cost_15pct"]) / latest["weight_avg"].abs().clip(lower=EPS)
        ).reindex(index)
        # Unadjusted same-day close over the same-day cost level: both raw, one date.
        close = daily_last["close"].reindex(latest.index)
        out["px_to_cost50"] = (
            close / latest["cost_50pct"].abs().clip(lower=EPS) - 1.0).reindex(index)
        out["px_to_wavg"] = (
            close / latest["weight_avg"].abs().clip(lower=EPS) - 1.0).reindex(index)
    out["unlock_next90"] = 0.0
    out["unlock_past30"] = 0.0
    visible_versions = _upto(unlocks, end_ts)
    if not visible_versions.empty:
        # In-effect version = the latest announcement of that unlock visible now.
        keep = visible_versions.groupby(["ts_code", "float_date"])["ann_date"].transform("max")
        current = visible_versions[visible_versions["ann_date"] == keep]
        day = pd.Timestamp(feature_date)
        today = day.strftime("%Y%m%d")
        ahead = current[(current["float_date"] > today)
                        & (current["float_date"] <= (day + pd.Timedelta(days=90)).strftime("%Y%m%d"))]
        behind = current[(current["float_date"] >= (day - pd.Timedelta(days=30)).strftime("%Y%m%d"))
                         & (current["float_date"] < today)]
        for name, subset in (("unlock_next90", ahead), ("unlock_past30", behind)):
            total = subset.groupby("ts_code")["ratio"].sum().clip(upper=100.0)
            out[name] = total.reindex(index).fillna(0.0)
        out["has_unlock_90"] = (out["unlock_next90"] > 0).astype("float64")
    return out


# --------------------------------------------------------------------------- assembly


def cross_section(context, feature_date, daily, price, tables, block_ev, unlocks,
                  adv_sum60, amount_lookup, daily_only):
    """All declared columns for one feature date, indexed by ts_code."""
    day = price[price["trade_date"] == feature_date].set_index("ts_code")
    if day.empty:
        return day
    index = day.index
    frame = day[PRICE_BLOCK].copy()
    if daily_only:
        return frame
    end_ts = as_of(context, feature_date)
    daily_last = daily[daily["trade_date"] == feature_date].set_index("ts_code")
    parts = [
        analyst_features(tables["rc"], end_ts, index),
        margin_features(tables["md"], daily_last, amount_lookup, end_ts, index),
        block_features(block_ev, adv_sum60, end_ts, index),
        holder_features(tables["hn"], tables["ht"], tables["tf"], daily_last, end_ts, index),
        chip_features(tables["cy"], unlocks, daily_last, feature_date, end_ts, index),
    ]
    return pd.concat([frame] + parts, axis=1)


def prepare(context, daily, tables):
    """Per-run precomputation shared by every feature date.

    Everything that does not depend on the feature date is done once: the daily
    block, the block-trade event level, the unlock version table, and the two
    dense daily event panels. `tables` is mutated in place so the caller's dict
    holds the prepared frames.
    """
    price = price_block(daily)
    block_ev = sort_by_visible(block_events(tables["bt"], daily))
    unlocks = unlock_events(tables["sf"])
    tables["rc"] = sort_by_visible(_rc_prepare(tables["rc"]))
    tables["ht"] = sort_by_visible(tables["ht"])
    tables["hn"] = sort_by_visible(tables["hn"])
    tables["tf"] = sort_by_visible(tables["tf"])
    tables["md"] = daily_panel(tables["md"], "trade_date")
    tables["cy"] = daily_panel(tables["cy"], "trade_date")
    adv = daily[["ts_code", "trade_date"]].assign(
        amount_sum60=_roll(daily, "amount", 60, "sum"),
        adv20=_roll(daily, "amount", 20, "mean"))
    amount_lookup = daily.set_index(["ts_code", "trade_date"])["amount"]
    amount_lookup = amount_lookup[~amount_lookup.index.duplicated()]
    return price, block_ev, unlocks, adv, amount_lookup


def adv_sum_at(adv, feature_date):
    day = adv[adv["trade_date"] == feature_date]
    return day.set_index("ts_code")["amount_sum60"]


def standardize(frame, columns):
    """Cross-sectional median/MAD z-score clipped to +-3; indicator columns pass through."""
    out = frame.copy()
    for column in columns:
        if column.startswith("has_"):
            out[column] = out[column].fillna(0.0)
            continue
        values = out[column].astype("float64").replace([np.inf, -np.inf], np.nan)
        median = values.median()
        mad = (values - median).abs().median() * 1.4826
        if not np.isfinite(mad) or mad <= EPS:
            out[column] = 0.0
            continue
        out[column] = ((values - median) / mad).clip(-3.0, 3.0)
    return out


def eligible(context, daily, universe, feature_date, price_cap, adv_floor, adv):
    """Declared pool: no BJ/STAR/ST, listed >= 120 days, ADV20 floor, price cap, not suspended.

    `adv` is the frame `prepare` returns: the rolling windows are computed once
    for the whole history, never per feature date.
    """
    day = daily[daily["trade_date"] == feature_date]
    if day.empty:
        return pd.Index([])
    day = day.assign(adv20=adv["adv20"].reindex(day.index))
    day = day[(day["is_suspended"] != True) & (day["close"] <= price_cap)  # noqa: E712
              & (day["adv20"] >= adv_floor)]
    codes = day["ts_code"]
    codes = codes[~codes.str.endswith(".BJ") & ~codes.str.startswith(("688", "689"))]
    info = universe.set_index("ts_code").reindex(codes)
    name = info["name"].astype(str)
    keep = ~(name.str.contains("ST") | name.str.contains("退"))
    listed = pd.to_datetime(info["list_date"], format="%Y%m%d", errors="coerce")
    keep = keep & (listed <= pd.Timestamp(feature_date) - pd.Timedelta(days=120))
    return pd.Index(codes[keep.to_numpy()])


def neutralize(scores, frame):
    """Residualize a score against size/momentum/turnover/volatility and SW-L1 dummies."""
    columns = [c for c in NEUTRAL_COLUMNS if c in frame.columns]
    design = [np.ones((len(scores), 1))]
    for column in columns:
        values = frame[column].astype("float64").to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        design.append(values.reshape(-1, 1))
    industry = frame.get("l1_code")
    if industry is not None:
        dummies = pd.get_dummies(industry.astype(str), drop_first=True, dtype="float64")
        if dummies.shape[1] > 0:
            design.append(dummies.to_numpy())
    matrix = np.hstack(design)
    target = np.nan_to_num(np.asarray(scores, dtype="float64"), nan=0.0)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    return target - matrix @ solution


def equal_weight_score(frame):
    """C-EW: signed cross-sectional ranks of the declared event columns, equally weighted."""
    total = np.zeros(len(frame), dtype="float64")
    used = 0
    for column, sign in EW_SIGNS.items():
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
