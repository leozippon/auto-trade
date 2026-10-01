"""The census scores this arm flips, point in time, in the census direction.

Each function below is one arm of the 2026-09-27 census, named in its
docstring as `<stem>_alla_100k_20260927`: the same columns, calendar window,
filters and minimum counts as that run's validated main candidate, so a
higher value here is what that arm bought first. Nothing in this module knows
about the flip: `lib/score.py` ranks these values and reverses them.

One difference from the census, on purpose. Seven census arms scored only the
current CSI 1000 members (`roe1k` and `mgnlag` also left out 300 / 301 codes).
Here every score covers every name that has the bars it needs; which names are
eligible is the pool's business (`knobs.POOL`), not the score's.

Reads, once per call: one projected `daily` window of WINDOW_DAYS calendar
days, the `index_daily` rows of MARKET, the Shenwan level-1 rows of `sw_daily`
and the universe's level-1 codes. Each score then cuts its own census window
out of these. Percent units follow the data contract: `daily.pct_chg` is a
decimal, `index_daily.pct_chg` a percent number (divided by 100 here), and
`index_daily.amount` is only ever used as a ratio of itself.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

# The market leg inside the census definitions (beta, delayed beta,
# coskewness, relative down days, amount response). Part of each score's
# definition, so it is not the benchmark knob.
MARKET = "000852.SH"
# The longest census window below (`freegrow`: 60 sessions in 120 days).
WINDOW_DAYS = 120
DAILY_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "amount", "up_limit",
    "circ_mv", "total_mv", "free_share", "float_share", "pb", "pe", "ps_ttm", "pe_ttm",
]
INDEX_COLUMNS = ["dataset", "ts_code", "trade_date", "pct_chg", "amount"]
SW_COLUMNS = ["dataset", "ts_code", "trade_date", "available_at", "close", "amount"]
SW_LEVEL1 = r"801\d\d0\.SI"

# The census families (the review's grouping), each a tuple of score names.
FAMILIES = {
    "volume": ("amtlag", "amtac", "pvcorr", "volag", "dn20", "dn1000", "reldn"),
    "beta": ("delay60", "beta60", "csk1k"),
    "peer": ("decouple", "peerlag"),
    "candle": ("room", "body"),
    "float": ("lock1000", "freeshare", "freegrow"),
    "valuation": ("pb", "roe1k", "mgnlag"),
    "industry": ("indflow",),
}


def compute(context, names):
    """{score name: float Series indexed by ts_code, census direction} for `names`."""

    data = _Inputs(context)
    return {name: SCORES[name](data) for name in names}


class _Inputs:
    """The decision day's windowed reads, shared by every score of one call."""

    def __init__(self, context):
        self.at = pd.Timestamp(context.inference_at)
        start = self.start(WINDOW_DAYS)
        daily = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS,
                                filters=[("trade_date", ">=", start)])
        missing = [name for name in DAILY_COLUMNS if name not in daily.columns]
        if missing or daily.empty:
            raise RuntimeError(f"daily window from {start}: missing columns {missing} or no rows")
        daily = daily.assign(trade_date=daily["trade_date"].astype(str), ts_code=daily["ts_code"].astype(str))
        for name in DAILY_COLUMNS[2:]:
            daily[name] = pd.to_numeric(daily[name], errors="coerce")
        self.daily = daily.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        index = pd.read_parquet(context.asof_dir + "/macro", columns=INDEX_COLUMNS,
                                filters=[("dataset", "=", "index_daily"), ("ts_code", "=", MARKET),
                                         ("trade_date", ">=", start)])
        index = index.assign(trade_date=index["trade_date"].astype(str),
                             pct_chg=pd.to_numeric(index["pct_chg"], errors="coerce"),
                             amount=pd.to_numeric(index["amount"], errors="coerce"))
        self.index = index.drop_duplicates("trade_date").set_index("trade_date").sort_index()
        if self.index["pct_chg"].dropna().empty:
            raise RuntimeError(f"no visible {MARKET} index_daily rows from {start}")
        sw = pd.read_parquet(context.asof_dir + "/macro", columns=SW_COLUMNS,
                             filters=[("dataset", "=", "sw_daily"), ("trade_date", ">=", self.start(50))])
        stamp = pd.to_datetime(sw["available_at"], utc=True)
        sw = sw[stamp <= self.at.tz_convert("UTC")]
        sw = sw.assign(ts_code=sw["ts_code"].map(_norm), trade_date=sw["trade_date"].astype(str))
        self.sw = sw[sw["ts_code"].str.fullmatch(SW_LEVEL1)]
        universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code", "l1_code"])
        self.industry = universe.drop_duplicates("ts_code").set_index("ts_code")["l1_code"].map(_norm)

    def start(self, days):
        return (self.at - timedelta(days=days)).strftime("%Y%m%d")

    def bars(self, days):
        """The daily rows of the last `days` calendar days, ordered by code then date."""
        return self.daily[self.daily["trade_date"] >= self.start(days)]

    def market(self, column, days):
        """The MARKET column over the last `days` calendar days, indexed by trade_date."""
        return self.index.loc[self.index.index >= self.start(days), column]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    if text.endswith(".SI"):
        return text
    if text.isdigit():
        return text + ".SI"
    return text


def _last(bars):
    """Each name's newest row."""
    return bars.drop_duplicates("ts_code", keep="last").set_index("ts_code")


def _sessions(bars, count):
    """The rows on the newest `count` trading dates of the window (any name)."""
    dates = sorted(bars["trade_date"].unique())
    return bars[bars["trade_date"].isin(set(dates[-count:]))], len(dates)


def _moments(frame, x, y, min_rows):
    """Per name over rows with both x and y: (rows, Sxx, Sxy, Syy) about the name's own means."""
    use = frame.loc[frame[x].notna() & frame[y].notna(), ["ts_code", x, y]]
    groups = use.groupby("ts_code")
    dx = use[x] - groups[x].transform("mean")
    dy = use[y] - groups[y].transform("mean")
    parts = pd.DataFrame({"ts_code": use["ts_code"], "xx": dx * dx, "xy": dx * dy, "yy": dy * dy})
    sums = parts.groupby("ts_code")[["xx", "xy", "yy"]].sum()
    rows = groups.size()
    return sums[rows >= min_rows]


def _slope(frame, x, y, min_rows):
    """OLS slope of y on x per name, the census loops' (x - x̄)·(y - ȳ) / (x - x̄)²."""
    sums = _moments(frame, x, y, min_rows)
    sums = sums[sums["xx"] > 0]
    return sums["xy"] / sums["xx"]


def _corr(frame, x, y, min_rows):
    """Pearson correlation per name; a name with a constant side has none."""
    sums = _moments(frame, x, y, min_rows)
    sums = sums[(sums["xx"] > 0) & (sums["yy"] > 0)]
    return sums["xy"] / np.sqrt(sums["xx"] * sums["yy"])


# --- volume / turnover: co-movement -------------------------------------------------------------


def amtlag(data):
    """`amtlag_alla_100k_20260927`: corr(amount_t, pct_chg_{t-1}) over the newest 20 sessions
    of a 40-day window, consecutive rows of the name, at least 15 pairs. High = amount
    follows yesterday's rise."""
    window, dates = _sessions(data.bars(40), 20)
    if dates < 16:
        raise RuntimeError("amtlag: fewer than 16 sessions in 40 days")
    window = window.assign(prev=window.groupby("ts_code")["pct_chg"].shift(1))
    return _corr(window, "amount", "prev", 15)


def amtac(data):
    """`amtac_alla_100k_20260927`: corr(amount_t, amount_{t-1}) over the newest 20 sessions
    of a 40-day window, at least 15 pairs. High = sticky amount."""
    window, dates = _sessions(data.bars(40), 20)
    if dates < 16:
        raise RuntimeError("amtac: fewer than 16 sessions in 40 days")
    window = window.assign(prev=window.groupby("ts_code")["amount"].shift(1))
    return _corr(window, "amount", "prev", 15)


def pvcorr(data):
    """`pvcorr_alla_100k_20260927`: corr(pct_chg, amount) over the newest 20 sessions of a
    40-day window, at least 15 sessions. High = amount swells with the rise."""
    window, dates = _sessions(data.bars(40), 20)
    if dates < 15:
        raise RuntimeError("pvcorr: fewer than 15 sessions in 40 days")
    return _corr(window, "pct_chg", "amount", 15)


def volag(data):
    """`volag_alla_100k_20260927` (census: CSI 1000 members only): slope of the name's daily
    amount change on the previous day's MARKET amount change over 100 days, both changes
    below 10 in absolute value, at least 30 days. High = amount answers the market late."""
    bars = data.bars(100)
    market = data.market("amount", 100)
    market = market.where(market > 0)
    lagged = (market / market.shift(1) - 1.0).shift(1)
    change = bars["amount"] / bars.groupby("ts_code")["amount"].shift(1) - 1.0
    frame = pd.DataFrame({"ts_code": bars["ts_code"], "chg": change, "mkt": bars["trade_date"].map(lagged)})
    frame = frame[frame["chg"].abs().lt(10) & frame["mkt"].abs().lt(10)]
    return _slope(frame, "mkt", "chg", 30)


# --- volume / turnover: down-day amount share -----------------------------------------------------


def _share(rows, down, min_rows):
    """Amount on `down` rows over all amount, per name with at least `min_rows` rows; a name
    without a single down row has no score (as in the census)."""
    counts = rows.groupby("ts_code").size()
    total = rows.groupby("ts_code")["amount"].sum()
    lost = rows.loc[down].groupby("ts_code")["amount"].sum()
    share = (lost / total).reindex(counts.index)
    return share[counts >= min_rows].dropna()


def dn20(data):
    """`dn20_alla_100k_20260927`: share of amount traded on down days (pct_chg < 0) over each
    name's newest 20 usable sessions (priced, amount > 0) of a 35-day window, at least 10."""
    bars = data.bars(35)
    use = bars[bars["pct_chg"].notna() & bars["amount"].notna() & (bars["amount"] > 0)]
    last = use.groupby("ts_code", sort=False).tail(20)
    return _share(last, last["pct_chg"] < 0, 10)


def dn1000(data):
    """`dn1000_alla_100k_20260927` (census: CSI 1000 members only): share of amount on down
    days over every usable session of a 40-day window, at least 10."""
    bars = data.bars(40)
    use = bars[bars["pct_chg"].notna() & bars["amount"].notna() & (bars["amount"] > 0)]
    return _share(use, use["pct_chg"] < 0, 10)


def reldn(data):
    """`reldn_alla_100k_20260927`: share of amount on days the name trailed MARKET over a
    40-day window, at least 10 usable days."""
    bars = data.bars(40)
    market = data.market("pct_chg", 40) / 100.0
    bars = bars.assign(mkt=bars["trade_date"].map(market))
    use = bars[bars["pct_chg"].notna() & bars["mkt"].notna() & bars["amount"].notna() & (bars["amount"] > 0)]
    return _share(use, use["pct_chg"] < use["mkt"], 10)


# --- beta / index sensitivity ---------------------------------------------------------------------


def delay60(data):
    """`delay60_alla_100k_20260927`: slope of the name's daily return on the previous day's
    MARKET return over 100 days (about 60 sessions), at least 30 days. High = slow response."""
    bars = data.bars(100)
    lagged = (data.market("pct_chg", 100) / 100.0).shift(1)
    return _slope(bars.assign(mkt=bars["trade_date"].map(lagged)), "mkt", "pct_chg", 30)


def beta60(data):
    """`beta60_alla_100k_20260927`: minus the same-day beta to MARKET over 100 days, at least
    30 days. High = low beta. (`beta1k_alla_100k_20260927` is this score on CSI 1000 members.)"""
    bars = data.bars(100)
    market = data.market("pct_chg", 100) / 100.0
    return -_slope(bars.assign(mkt=bars["trade_date"].map(market)), "mkt", "pct_chg", 30)


def csk1k(data):
    """`csk1k_alla_100k_20260927` (census: CSI 1000 members only): minus the slope of the
    name's daily return on the squared demeaned MARKET return over 100 days, at least 30
    days; the mean is the window's. High = does worse on large index moves."""
    bars = data.bars(100)
    market = data.market("pct_chg", 100) / 100.0
    squared = (market - market.mean()) ** 2
    return -_slope(bars.assign(x=bars["trade_date"].map(squared)), "x", "pct_chg", 30)


# --- peer-relative ----------------------------------------------------------------------------------


def _sw_rows(data, column):
    """The level-1 rows with a usable `column` (as `value`), one per code and date, by date."""
    rows = data.sw.assign(value=pd.to_numeric(data.sw[column], errors="coerce")).dropna(subset=["value"])
    rows = rows.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")
    return rows


def decouple(data):
    """`decouple_alla_100k_20260927`: minus the correlation of the name's daily return with
    its Shenwan level-1 index over the newest 21 Shenwan sessions, at least 15 pairs. Returns
    are on the unadjusted close and a day without a bar repeats the last close (a zero
    return), as in the census code. High = does not track its industry."""
    rows = _sw_rows(data, "close")
    wide = rows.pivot(index="trade_date", columns="ts_code", values="value").sort_index()
    if len(wide) < 22:
        raise RuntimeError("decouple: fewer than 22 Shenwan sessions")
    wide = wide.iloc[-22:]
    industry_returns = wide.ffill().pct_change(fill_method=None).iloc[1:]
    bars = data.bars(50)
    bars = bars[bars["close"] > 0]
    prices = bars.pivot(index="trade_date", columns="ts_code", values="close").reindex(wide.index)
    stock = prices.ffill().pct_change(fill_method=None).iloc[1:]
    industry = data.industry.reindex(stock.columns)
    codes = [code for code in stock.columns if industry.get(code, "") in industry_returns.columns]
    stock = stock[codes].to_numpy()
    peer = industry_returns[[industry[code] for code in codes]].to_numpy()
    both = ~np.isnan(stock) & ~np.isnan(peer)
    rows_used = both.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(both, stock, 0.0)
        y = np.where(both, peer, 0.0)
        dx = np.where(both, x - x.sum(axis=0) / rows_used, 0.0)
        dy = np.where(both, y - y.sum(axis=0) / rows_used, 0.0)
        sxx = (dx * dx).sum(axis=0)
        syy = (dy * dy).sum(axis=0)
        value = -(dx * dy).sum(axis=0) / np.sqrt(sxx * syy)
    keep = (rows_used >= 15) & (sxx > 0) & (syy > 0)
    return pd.Series(value[keep], index=np.asarray(codes, dtype=object)[keep], dtype="float64")


def peerlag(data):
    """`peerlag_alla_100k_20260927`: minus (the name's 21-session close-to-close return minus
    its Shenwan level-1 return over its own newest 22 rows). Unadjusted close, as in the
    census code. High = lagged its industry."""
    rows = _sw_rows(data, "close")
    industry_return = {}
    for code, group in rows.groupby("ts_code"):
        closes = group["value"].to_numpy()
        if len(closes) >= 22 and closes[-22] > 0:
            industry_return[code] = float(closes[-1] / closes[-22] - 1.0)
    if not industry_return:
        raise RuntimeError("peerlag: no Shenwan industry has a 21-session return")
    bars = data.bars(50)
    bars = bars[bars["close"] > 0]
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < 22:
        raise RuntimeError("peerlag: fewer than 22 sessions in 50 days")
    last = bars[bars["trade_date"] == dates[-1]].drop_duplicates("ts_code").set_index("ts_code")["close"]
    first = bars[bars["trade_date"] == dates[-22]].drop_duplicates("ts_code").set_index("ts_code")["close"]
    both = pd.concat([last.rename("close"), first.rename("close0")], axis=1, join="inner")
    own = both["close"] / both["close0"] - 1.0
    peer = data.industry.reindex(own.index).map(industry_return)
    return (-(own - peer))[peer.notna()]


# --- price range / candle shape -------------------------------------------------------------------


def room(data):
    """`room_alla_100k_20260927`: (up_limit - close) / close on the newest bar of a 15-day
    window. High = far below the limit-up price."""
    last = _last(data.bars(15))
    value = (last["up_limit"] - last["close"]) / last["close"].where(last["close"] > 0)
    return value.where(last["up_limit"] > 0)


def body(data):
    """`body_alla_100k_20260927`: mean of |close - open| / (high - low) over the newest 20
    sessions of a 40-day window, at least 15 measurable days. High = full-bodied candles."""
    window, dates = _sessions(data.bars(40), 20)
    if dates < 15:
        raise RuntimeError("body: fewer than 15 sessions in 40 days")
    span = window["high"] - window["low"]
    shape = (window["close"] - window["open"]).abs() / span.where(span > 0)
    groups = shape.groupby(window["ts_code"])
    return groups.mean().where(groups.count() >= 15)


# --- free float / share structure -----------------------------------------------------------------


def lock1000(data):
    """`lock1000_alla_100k_20260927` (census: CSI 1000 members only): minus circ_mv / total_mv
    on the newest bar of an 80-day window, both positive. High = small tradable share."""
    last = _last(data.bars(80))
    return (-(last["circ_mv"] / last["total_mv"])).where((last["circ_mv"] > 0) & (last["total_mv"] > 0))


def freeshare(data):
    """`freeshare_alla_100k_20260927`: free_share / float_share on the newest bar of a 15-day
    window, both positive. High = little of the float is locked by large holders."""
    last = _last(data.bars(15))
    ratio = last["free_share"] / last["float_share"].where(last["float_share"] > 0)
    return ratio.where(last["free_share"] > 0)


def freegrow(data):
    """`freegrow_alla_100k_20260927`: minus the growth of free_share from the 60th-newest
    session to the newest one of a 120-day window, both positive. High = free float grew
    least."""
    bars = data.bars(WINDOW_DAYS)
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < 60:
        raise RuntimeError("freegrow: fewer than 60 sessions in 120 days")
    early = bars[bars["trade_date"] == dates[-60]].drop_duplicates("ts_code").set_index("ts_code")["free_share"]
    late = bars[bars["trade_date"] == dates[-1]].drop_duplicates("ts_code").set_index("ts_code")["free_share"]
    growth = late / early.where(early > 0) - 1.0
    return -growth.where(late > 0)


# --- valuation ratios -----------------------------------------------------------------------------


def pb(data):
    """`pb_alla_100k_20260927`: minus pb on the newest bar of a 15-day window, pb positive.
    High = cheap on book."""
    last = _last(data.bars(15))
    return -last["pb"].where(last["pb"] > 0)


def roe1k(data):
    """`roe1k_alla_100k_20260927` (census: CSI 1000 members outside 300 / 301 only): pb / pe
    on the newest bar of a 40-day window, both positive. High = high return on equity."""
    last = _last(data.bars(40))
    return (last["pb"] / last["pe"]).where((last["pb"] > 0) & (last["pe"] > 0))


def mgnlag(data):
    """`mgnlag_alla_100k_20260927` (census: CSI 1000 members outside 300 / 301 only):
    ps_ttm / pe_ttm on each name's newest bar dated at or before the 22nd-newest session
    of an 80-day window, both positive. High = high net margin a month ago."""
    bars = data.bars(80)
    dates = sorted(bars["trade_date"].unique())
    if len(dates) <= 21:
        raise RuntimeError("mgnlag: 21 or fewer sessions in 80 days")
    last = _last(bars[bars["trade_date"] <= dates[-22]])
    return (last["ps_ttm"] / last["pe_ttm"]).where((last["ps_ttm"] > 0) & (last["pe_ttm"] > 0))


# --- industry aggregates --------------------------------------------------------------------------


def indflow(data):
    """`indflow_alla_100k_20260927`: the name's Shenwan level-1 amount on its newest row over
    its amount on the 21st-newest row, minus one; every name of an industry shares it.
    (`indrise_alla_100k_20260927` is the same score.) High = industry amount rising."""
    rows = _sw_rows(data, "amount")
    growth = {}
    for code, group in rows.groupby("ts_code"):
        amounts = group["value"].to_numpy()
        if len(amounts) >= 21 and amounts[-21] > 0:
            growth[code] = float(amounts[-1] / amounts[-21] - 1.0)
    if not growth:
        raise RuntimeError("indflow: no Shenwan industry has 21 amount rows")
    return data.industry.map(growth).dropna()


SCORES = {
    "amtlag": amtlag, "amtac": amtac, "pvcorr": pvcorr, "volag": volag,
    "dn20": dn20, "dn1000": dn1000, "reldn": reldn,
    "delay60": delay60, "beta60": beta60, "csk1k": csk1k,
    "decouple": decouple, "peerlag": peerlag,
    "room": room, "body": body,
    "lock1000": lock1000, "freeshare": freeshare, "freegrow": freegrow,
    "pb": pb, "roe1k": roe1k, "mgnlag": mgnlag,
    "indflow": indflow,
}
