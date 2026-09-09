"""Block-trade discount: PIT reads, two-level aggregation, neutralised ranking.

Every function here takes the decision context or frames derived from it; no
path is built outside ``context.asof_dir``. Units follow the data contract:
``block_trade.vol`` is 10k shares and ``block_trade.amount`` is 10k CNY, while
``daily.amount`` and ``daily.circ_mv`` are already normalised to CNY.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

# 10k CNY -> CNY. The single most error-prone conversion in this arm.
BLOCK_AMOUNT_TO_CNY = 1e4
INSTITUTIONAL_SEAT = "机构专用"

DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "close",
    "pre_close",
    "amount",
    "turnover_rate",
    "circ_mv",
    "is_suspended",
]
BLOCK_COLUMNS = [
    "dataset",
    "available_at",
    "ts_code",
    "trade_date",
    "price",
    "vol",
    "amount",
    "buyer",
    "seller",
]


def read_daily(context, lookback_days):
    """Visible daily rows, bounded by a calendar lookback."""

    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[frame["close"] > 0]
    return frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def read_universe(context):
    return pd.read_parquet(
        context.asof_dir + "/universe",
        columns=["ts_code", "name", "list_date", "market", "l1_code"],
    )


def read_blocks(context, start_date):
    """Visible ``block_trade`` rows. Visibility is decided by ``available_at``."""

    frame = pd.read_parquet(context.asof_dir + "/events", columns=BLOCK_COLUMNS)
    frame = frame[frame["dataset"] == "block_trade"]
    visible = pd.to_datetime(frame["available_at"]) <= pd.Timestamp(context.inference_at)
    frame = frame[visible & (frame["trade_date"] >= start_date)]
    return frame.dropna(subset=["ts_code", "trade_date", "price", "amount"])


def window_discount(blocks, daily, window_dates, drop_self_dealing=True):
    """Two-level amount-weighted discount over ``window_dates``.

    Level one aggregates prints into (ts_code, trade_date) events, level two
    aggregates events into the window. The discount denominator is the same
    day's unadjusted close, which only ``daily`` carries: the ``close`` column
    of the events union is empty on block-trade rows.
    """

    window = blocks[blocks["trade_date"].isin(window_dates)]
    if drop_self_dealing:
        window = window[window["buyer"].fillna("") != window["seller"].fillna("")]
    closes = daily[daily["trade_date"].isin(window_dates)][["ts_code", "trade_date", "close"]]
    window = window.merge(closes, on=["ts_code", "trade_date"], how="inner")
    if window.empty:
        return pd.DataFrame(columns=["ts_code", "discount", "amount_cny", "events", "seller_inst_share"])
    window = window.assign(
        amount_cny=window["amount"].astype(float) * BLOCK_AMOUNT_TO_CNY,
        discount=window["price"].astype(float) / window["close"].astype(float) - 1.0,
    )
    window = window[np.isfinite(window["discount"]) & (window["amount_cny"] > 0)]
    if window.empty:
        return pd.DataFrame(columns=["ts_code", "discount", "amount_cny", "events", "seller_inst_share"])
    window = window.assign(
        weighted=window["discount"] * window["amount_cny"],
        seller_inst=np.where(
            window["seller"].fillna("") == INSTITUTIONAL_SEAT, window["amount_cny"], 0.0
        ),
    )
    # Level one: prints -> events.
    events = window.groupby(["ts_code", "trade_date"], as_index=False)[
        ["weighted", "amount_cny", "seller_inst"]
    ].sum()
    # Level two: events -> window.
    totals = events.groupby("ts_code", as_index=False).agg(
        weighted=("weighted", "sum"),
        amount_cny=("amount_cny", "sum"),
        seller_inst=("seller_inst", "sum"),
        events=("trade_date", "size"),
    )
    totals["discount"] = totals["weighted"] / totals["amount_cny"]
    totals["seller_inst_share"] = totals["seller_inst"] / totals["amount_cny"]
    return totals[["ts_code", "discount", "amount_cny", "events", "seller_inst_share"]]


def cross_section(daily, universe, last_date, rules):
    """Tradability screen plus the neutralisation controls, as of ``last_date``."""

    tail = daily[daily["trade_date"] <= last_date]
    recent = tail.groupby("ts_code").tail(20)
    stats = recent.groupby("ts_code", as_index=False).agg(
        adv20=("amount", "mean"), turn20=("turnover_rate", "mean")
    )
    latest = tail[tail["trade_date"] == last_date][
        ["ts_code", "close", "circ_mv", "is_suspended"]
    ]
    frame = latest.merge(stats, on="ts_code", how="left").merge(universe, on="ts_code", how="left")
    listed_before = (
        pd.Timestamp(last_date) - timedelta(days=rules["min_listed_calendar_days"])
    ).strftime("%Y%m%d")
    keep = (
        (~frame["is_suspended"].fillna(True).astype(bool))
        & (frame["adv20"] >= rules["min_adv_cny"])
        & (frame["close"] <= rules["max_price"])
        & (frame["close"] > 1.0)
        & (frame["circ_mv"] > 0)
        & (frame["turn20"] > 0)
        & frame["list_date"].fillna("").astype(str).le(listed_before)
        & (~frame["name"].fillna("").str.contains("ST"))
        & (~frame["market"].fillna("").isin(rules["excluded_markets"]))
    )
    return frame[keep].reset_index(drop=True)


def _unit_rank(values):
    ranked = pd.Series(values).rank(pct=True).to_numpy()
    return 2.0 * (ranked - 0.5)


def neutral_score(frame, column):
    """Winsorise, rank, residualise on size and turnover, then demean by industry."""

    raw = frame[column].astype(float).to_numpy()
    lo, hi = np.nanquantile(raw, 0.01), np.nanquantile(raw, 0.99)
    signal = _unit_rank(np.clip(raw, lo, hi))
    size = _unit_rank(np.log(frame["circ_mv"].astype(float).to_numpy()))
    turn = _unit_rank(np.log(frame["turn20"].astype(float).to_numpy()))
    design = np.column_stack([np.ones(len(frame)), size, turn])
    beta, *_ = np.linalg.lstsq(design, signal, rcond=None)
    resid = signal - design @ beta
    industry = frame["l1_code"].fillna("__none__").to_numpy()
    out = pd.Series(resid, index=frame.index)
    return out - out.groupby(pd.Series(industry, index=frame.index)).transform("mean")


def select(frame, score_column, top_n, mode):
    """``signal``: the top ``top_n`` scores. ``control``: the ``top_n`` names
    closest to the cross-sectional median of the same eligible set."""

    ordered = frame.sort_values([score_column, "ts_code"], ascending=[False, True])
    if mode == "control":
        median = ordered[score_column].median()
        ordered = ordered.assign(distance=(ordered[score_column] - median).abs())
        ordered = ordered.sort_values(["distance", "ts_code"], ascending=[True, True])
    return ordered.head(top_n)
