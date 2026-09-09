"""PIT-safe construction of the seasoned margin-roster inclusion cohort.

Every rule here is a declared constant from ``families.md``, not a fitted
parameter. The module never reads a date column to decide visibility: the
caller filters ``available_at <= context.inference_at`` once, and the visible
``margin_secs`` trade dates are then used as this arm's trading calendar.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

# --- declared constants (families.md) ---------------------------------------
HOLD_DAYS = 40                 # trading days a name stays in the book
BASKET_CAP = 12                # equal-weight names held at once
ROUNDTRIP_DAYS = 60            # F2: a name removed this recently is not a new event
MIN_LIST_AGE_DAYS = 120        # F3: calendar days since listing
CONFIRM_DAYS = 3               # F4: margin_detail rows checked before the event
MIN_ADV20_CNY = 3.0e7          # tradability: 20-day mean amount
MAX_CLOSE_CNY = 40.0           # tradability: one 100-share lot must fit the budget
ADV_WINDOW = 20
CHURN_START = "20240401"       # declared exclusion: SSE roster slice oscillation
CHURN_END = "20240531"
EXCLUDED_PREFIXES = ("688", "689")
EXCLUDED_SUFFIX = ".BJ"

EVENT_COLUMNS = ["dataset", "trade_date", "ts_code", "available_at"]
DAILY_COLUMNS = ["ts_code", "trade_date", "close", "amount", "is_suspended"]
UNIVERSE_COLUMNS = ["ts_code", "name", "list_date"]


def visible_events(context, lookback_days):
    """``events`` rows of both margin datasets, visible at the decision time."""
    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/events",
        columns=EVENT_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    frame = frame[frame["dataset"].isin(["margin_secs", "margin_detail"])]
    visible = pd.to_datetime(frame["available_at"]) <= pd.Timestamp(context.inference_at)
    frame = frame[visible & frame["trade_date"].notna() & frame["ts_code"].notna()]
    return frame[["dataset", "trade_date", "ts_code"]]


def visible_daily(context, lookback_days):
    """Daily rows visible at the decision time, bounded to a calendar lookback."""
    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    return frame.sort_values(["ts_code", "trade_date"])


def visible_universe(context):
    return pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)


def roster_calendar(events):
    """Sorted visible ``margin_secs`` trade dates: this arm's trading calendar."""
    roster = events[events["dataset"] == "margin_secs"]
    return sorted(roster["trade_date"].unique())


def _roster_sets(events, dates):
    roster = events[events["dataset"] == "margin_secs"]
    roster = roster[roster["trade_date"].isin(dates)]
    return {date: frozenset(codes) for date, codes in roster.groupby("trade_date")["ts_code"]}


def _detail_keys(events, dates):
    detail = events[events["dataset"] == "margin_detail"]
    detail = detail[detail["trade_date"].isin(dates)]
    return set(zip(detail["trade_date"], detail["ts_code"]))


def raw_diffs(rosters, dates):
    """``(date, ts_code, kind)`` roster set differences between adjacent dates."""
    rows = []
    for index in range(1, len(dates)):
        current, previous = dates[index], dates[index - 1]
        now, before = rosters.get(current), rosters.get(previous)
        if now is None or before is None:
            continue
        for code in now - before:
            rows.append((current, code, "add"))
        for code in before - now:
            rows.append((current, code, "drop"))
    return pd.DataFrame(rows, columns=["event_date", "ts_code", "kind"])


def inclusion_cohort(events, universe, daily, dates):
    """Seasoned inclusion events on the ``HOLD_DAYS`` most recent roster dates.

    Applies, in order: the declared churn-window exclusion, F1 (universe and
    board), F2 (60-trading-day round trip), F3 (listing age), F4 (no
    ``margin_detail`` row on the three roster dates before the event) and the
    tradability invariants measured at the event date.
    """
    if len(dates) < HOLD_DAYS + CONFIRM_DAYS + 2:
        return pd.DataFrame(columns=["event_date", "ts_code", "adv20"])
    rosters = _roster_sets(events, dates)
    diffs = raw_diffs(rosters, dates)
    if diffs.empty:
        return pd.DataFrame(columns=["event_date", "ts_code", "adv20"])
    position = {date: index for index, date in enumerate(dates)}
    detail = _detail_keys(events, dates)

    drops = diffs[diffs["kind"] == "drop"]
    drop_index = {}
    for date, code in zip(drops["event_date"], drops["ts_code"]):
        drop_index.setdefault(code, []).append(position[date])

    window = set(dates[-HOLD_DAYS:])
    adds = diffs[(diffs["kind"] == "add") & diffs["event_date"].isin(window)]
    adds = adds[~((adds["event_date"] >= CHURN_START) & (adds["event_date"] <= CHURN_END))]
    if adds.empty:
        return pd.DataFrame(columns=["event_date", "ts_code", "adv20"])

    listed = universe.dropna(subset=["list_date"]).drop_duplicates("ts_code")
    names = dict(zip(listed["ts_code"], listed["name"].astype(str)))
    list_dates = dict(zip(listed["ts_code"], listed["list_date"].astype(str)))

    kept = []
    for date, code in zip(adds["event_date"], adds["ts_code"]):
        if code not in list_dates or code.endswith(EXCLUDED_SUFFIX):
            continue                                              # F1
        if code.startswith(EXCLUDED_PREFIXES) or "ST" in names.get(code, ""):
            continue
        index = position[date]
        if any(0 < index - earlier <= ROUNDTRIP_DAYS for earlier in drop_index.get(code, ())):
            continue                                              # F2
        age = pd.Timestamp(date) - pd.Timestamp(list_dates[code])
        if age.days < MIN_LIST_AGE_DAYS:
            continue                                              # F3
        confirm = dates[max(index - CONFIRM_DAYS, 0):index]
        if any((day, code) in detail for day in confirm):
            continue                                              # F4
        kept.append((date, code))
    if not kept:
        return pd.DataFrame(columns=["event_date", "ts_code", "adv20"])

    cohort = pd.DataFrame(kept, columns=["event_date", "ts_code"])
    return _apply_tradability(cohort, daily)


def _apply_tradability(cohort, daily):
    """Keep events whose event-date daily row passes the declared invariants."""
    frame = daily[daily["ts_code"].isin(set(cohort["ts_code"]))].copy()
    if frame.empty:
        return pd.DataFrame(columns=["event_date", "ts_code", "adv20"])
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    frame["adv20"] = amount.groupby(frame["ts_code"]).transform(
        lambda series: series.rolling(ADV_WINDOW).mean()
    )
    frame = frame[["ts_code", "trade_date", "close", "adv20", "is_suspended"]]
    merged = cohort.merge(
        frame, left_on=["ts_code", "event_date"], right_on=["ts_code", "trade_date"], how="inner"
    )
    close = pd.to_numeric(merged["close"], errors="coerce")
    suspended = merged["is_suspended"].fillna(False).astype(bool)
    keep = (
        (merged["adv20"] >= MIN_ADV20_CNY)
        & (close > 0)
        & (close <= MAX_CLOSE_CNY)
        & ~suspended
    )
    return merged.loc[keep, ["event_date", "ts_code", "adv20"]].reset_index(drop=True)


def target_book(cohort):
    """Equal-weight target names: the most liquid ``BASKET_CAP`` live events."""
    if cohort.empty:
        return []
    ranked = cohort.sort_values(["adv20", "ts_code"], ascending=[False, True])
    ranked = ranked.drop_duplicates("ts_code")
    return list(ranked["ts_code"].head(BASKET_CAP))


def last_close(daily, codes):
    """Latest visible close per symbol, for lot sizing only."""
    frame = daily[daily["ts_code"].isin(set(codes))]
    if frame.empty:
        return {}
    latest = frame.groupby("ts_code").tail(1)
    prices = pd.to_numeric(latest["close"], errors="coerce")
    return {
        code: float(price)
        for code, price in zip(latest["ts_code"], prices)
        if pd.notna(price) and price > 0
    }
