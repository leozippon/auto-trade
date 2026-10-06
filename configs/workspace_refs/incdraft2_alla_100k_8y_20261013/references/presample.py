"""The pre-sample event study of refs/README.md section 6: the draft drift before the research period.

usage: python refs/references/presample.py [--split none|instrument] [--snapshot /mnt/snapshot]

Events are the book's own (stage2/lib/titles.py: the title rule, one event per
name after 90 quiet days, the instrument side), main board and ChiNext only,
dated 2016-09-28..2017-04-28: the view's titles start 2016-06-30, so that is the
first day with a full quiet period read, and a 40-day window from the last
event still closes before the research period. Entry is the close of the second
trading day after the event's date, skipped when that close is limit-up or the
name has no bar. The abnormal return is the name's 40-trading-day return
(adjusted close) minus the mean over the other main-board and ChiNext names
with a bar that day in the same quintile of the previous day's float market
value: the expectation of a same-day comparable random name. Only prices dated
2016-07-01..2017-06-30 are read. t is clustered by entry month.
"""

import argparse

import numpy as np
import pandas as pd

PLAN = "股权激励|限制性股票|股票期权"
DRAFT = "草案"
NOT = "摘要|修订|法律意见|核查|名单|考核|意见|独立财务顾问|自查|更正"
OPTION = "股票期权"
QUIET_DAYS, HORIZON = 90, 40
FIRST_EVENT, LAST_EVENT, PRICES_FROM, PRICES_TO = "20160928", "20170428", "20160701", "20170630"


def _permitted(codes):
    return ~codes.str.startswith(("688", "689")) & ~codes.str.endswith(".BJ")


def events(snapshot):
    rows = pd.read_parquet(f"{snapshot}/text_index.parquet", columns=["dataset", "ts_codes", "title", "available_at"],
                           filters=[("dataset", "==", "anns_d"), ("available_at", "<", "2017-07-01")])
    title = rows["title"].fillna("").astype(str)
    rows = rows[title.str.contains(PLAN) & title.str.contains(DRAFT) & ~title.str.contains(NOT)]
    out = pd.DataFrame({
        "ts_code": rows["ts_codes"].fillna("").astype(str).to_numpy(),
        "day": pd.to_datetime(rows["available_at"], utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").to_numpy(),
        "option": rows["title"].fillna("").astype(str).str.contains(OPTION).to_numpy(),
    })
    out = out[out["ts_code"] != ""].groupby(["ts_code", "day"], as_index=False, sort=True)["option"].any()
    gap = pd.to_datetime(out["day"], format="%Y%m%d").groupby(out["ts_code"]).diff().dt.days
    out = out[(gap.isna() | (gap > QUIET_DAYS)) & out["day"].between(FIRST_EVENT, LAST_EVENT)]
    return out[_permitted(out["ts_code"])].reset_index(drop=True)


def abnormal(snapshot, ev):
    bars = pd.read_parquet(f"{snapshot}/daily.parquet",
                           columns=["ts_code", "trade_date", "close", "adj_factor", "circ_mv", "up_limit", "is_suspended"],
                           filters=[("trade_date", ">=", PRICES_FROM), ("trade_date", "<=", PRICES_TO)])
    bars = bars.assign(ts_code=bars["ts_code"].astype(str), trade_date=bars["trade_date"].astype(str))
    bars = bars[_permitted(bars["ts_code"]) & ~bars["is_suspended"].eq(True) & (bars["close"] > 0)]
    wide = lambda column: bars.pivot(index="trade_date", columns="ts_code", values=column).sort_index()
    close = wide("close")
    dates, codes = close.index.to_numpy(), {c: i for i, c in enumerate(close.columns)}
    has = close.notna().to_numpy()
    adj = (close * wide("adj_factor").reindex_like(close)).ffill(limit=120).to_numpy()
    locked = (close >= wide("up_limit").reindex_like(close) * 0.9995).to_numpy()
    rank = wide("circ_mv").reindex_like(close).shift(1).rank(axis=1, pct=True).to_numpy()
    quintile = np.where(np.isnan(rank), -1, np.minimum(rank * 5, 4)).astype(int)
    rows = []
    for e in ev.itertuples():
        t = int(np.searchsorted(dates, e.day, side="right")) + 1
        c = codes.get(e.ts_code)
        if c is None or t + HORIZON >= len(dates) or not has[t, c] or locked[t, c] or quintile[t, c] < 0:
            continue
        peers = np.where((quintile[t] == quintile[t, c]) & has[t])[0]
        forward = adj[t + HORIZON] / adj[t] - 1
        rows.append({"ts_code": e.ts_code, "month": dates[t][:6], "option": e.option,
                     "ar40": forward[c] - np.nanmean(forward[peers[peers != c]])})
    return pd.DataFrame(rows)


def report(label, frame, events_found):
    monthly = frame.groupby("month")["ar40"].mean()
    t = monthly.mean() / monthly.std(ddof=1) * np.sqrt(len(monthly)) if len(monthly) > 1 else float("nan")
    print(f"{label}: events {events_found}, usable {len(frame)}, mean ar40 {100 * frame['ar40'].mean():+.2f} %, "
          f"month-clustered t {t:.2f} ({len(monthly)} months)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("none", "instrument"), default="none")
    parser.add_argument("--snapshot", default="/mnt/snapshot")
    args = parser.parse_args()
    ev = events(args.snapshot)
    frame = abnormal(args.snapshot, ev)
    if args.split == "none":
        report("all", frame, len(ev))
        return
    for side, flag in (("rs", False), ("opt", True)):
        report(side, frame[frame["option"] == flag], int((ev["option"] == flag).sum()))


if __name__ == "__main__":
    main()
