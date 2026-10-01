"""The census feature library: 44 per-date cross-sectional features of every A-share.

Round 20260927 ran 148 one-score arms on the eight-year seed, each ranking the
A-share cross-section by one rule; they fall into 19 families and 31
sub-families. This module recomputes those rules as the inputs of ONE learned
ranker. Inclusion is by family coverage, never by how a family read: one
feature per sub-family, a second or third only where the sub-family's arms
define structurally different quantities (beta / delayed beta / coskewness),
none for a re-timing of a feature already here (ps lags 10 and 42, the
60-session down-day share) or for an exact function of two features already
here (static ps = ps_gap and sp_ttm). Every feature function's docstring names
the census arms whose score it reproduces: "exact" = same quantity and window,
"approx." = the window convention named there.

The SAME file ships in census_gbdt_alla_8y_20261001 and
census_gbdt_csi1000_100k_8y_20261001; edit both or neither.

Point in time. Only `context.asof_dir` is read. `daily` has no `available_at`:
the view holds exactly what is visible, so at an 08:30 decision its newest row
is T-1. Every `macro` row is kept only if its own `available_at` is not later
than the decision. Panel row t is a trade date and every feature on row t uses
rows <= t, so the newest row a review scores and the rows `fit` trains on come
out of the same code. Keys: `index_weight` rows carry the index in
`index_code` and the stock in `con_code` (their `ts_code` is empty); `index_daily`
and `sw_daily` rows carry the series in `ts_code` (their `index_code` is
empty). Filtering the wrong column returns nothing or everything, silently.

Units (normalised daily): prices CNY/share, `vol` and share counts in shares,
`amount` / `circ_mv` / `total_mv` CNY, `pct_chg` and `dv_*` decimals. Index and
Shenwan returns are computed from closes; the percent-number `pct_chg` of the
macro series is never read.

History. Windows count market sessions (panel rows), not calendar days. A
value that cannot be computed is NaN, never a fill: index features need the
index's sections (CSI 300 / 500 from 2014-07-31, CSI 1000 from 2014-10-31 in
this seed), tenure needs TENURE_SECTIONS visible sections and exit
EXIT_SECTIONS + 1, Shenwan features need `sw_daily` (from 2014-07-01). The
model reads NaN as "unknown".
"""

from datetime import timedelta

import numpy as np
import pandas as pd

CSI300, CSI500, CSI1000 = "000300.SH", "000905.SH", "000852.SH"
TIERS = ((CSI300, 3), (CSI500, 2), (CSI1000, 1))
# Calendar days a review reads: 14 visible month-end sections plus the slack below,
# and 244 sessions for the entry rule, both with holidays.
DECISION_DAYS = 480
SECTION_SLACK_DAYS = 40
TENURE_SECTIONS = 14
EXIT_SECTIONS = 6

DAILY_COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol",
                 "amount", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_share",
                 "float_share", "free_share", "total_mv", "circ_mv", "up_limit", "down_limit", "adj_factor"]
UNIVERSE_COLUMNS = ["ts_code", "name", "list_date", "l1_code", "l1_name"]
WEIGHT_COLUMNS = ["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"]
SERIES_COLUMNS = ["dataset", "available_at", "ts_code", "trade_date", "open", "close", "amount", "pb"]
SW_LEVEL1 = r"801\d\d0\.SI"

FEATURES = []   # (name, family, function) in registration order


def _feature(family):
    def register(function):
        FEATURES.append((function.__name__, family, function))
        return function
    return register


# ----------------------------------------------------------------- reading


def read_panel(context, calendar_days):
    """Dense (sessions x names) arrays of every non-Beijing A-share in the window, plus the macro rows.

    STAR names are read (the CSI 300 entry rule ranks them) and left to the
    pool rule to exclude. Raises when a column the features need is missing.
    """

    decision = pd.Timestamp(context.inference_at)
    start = (decision - timedelta(days=calendar_days)).strftime("%Y%m%d")
    frame = pd.read_parquet(context.asof_dir + "/daily", columns=DAILY_COLUMNS,
                            filters=[("trade_date", ">=", start)])
    missing = [name for name in DAILY_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    frame = frame.assign(ts_code=frame["ts_code"].astype(str), trade_date=frame["trade_date"].astype(str))
    frame = frame[~frame["ts_code"].str.endswith(".BJ")]
    if frame.empty:
        raise RuntimeError("the daily window of the as-of view holds no row")
    dates = np.array(sorted(frame["trade_date"].unique()))
    codes = np.array(sorted(frame["ts_code"].unique()))
    ti = np.searchsorted(dates, frame["trade_date"].to_numpy())
    si = np.searchsorted(codes, frame["ts_code"].to_numpy())
    daily = {}
    for column in DAILY_COLUMNS[2:]:
        dense = np.full((len(dates), len(codes)), np.nan)
        dense[ti, si] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        daily[column] = dense
    has_bar = np.zeros((len(dates), len(codes)), dtype=bool)
    has_bar[ti, si] = True
    del frame

    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNIVERSE_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code").reindex(pd.Index(codes))
    listed = pd.to_datetime(info["list_date"].astype(str), format="%Y%m%d", errors="coerce")

    cutoff = decision.tz_convert("UTC")
    section_start = (decision - timedelta(days=calendar_days + SECTION_SLACK_DAYS)).strftime("%Y%m%d")
    weights = pd.read_parquet(
        context.asof_dir + "/macro", columns=WEIGHT_COLUMNS,
        filters=[("dataset", "=", "index_weight"), ("index_code", "in", [code for code, _ in TIERS]),
                 ("trade_date", ">=", section_start)])
    weights = weights[(pd.to_datetime(weights["available_at"], utc=True) <= cutoff)
                      & weights["con_code"].notna() & (pd.to_numeric(weights["weight"], errors="coerce") > 0)]
    weights = weights.assign(trade_date=weights["trade_date"].astype(str), con_code=weights["con_code"].astype(str),
                             weight=pd.to_numeric(weights["weight"], errors="coerce"))
    series = pd.read_parquet(
        context.asof_dir + "/macro", columns=SERIES_COLUMNS,
        filters=[("dataset", "in", ["index_daily", "sw_daily"]), ("trade_date", ">=", start)])
    series = series[pd.to_datetime(series["available_at"], utc=True) <= cutoff]
    series = series.assign(ts_code=series["ts_code"].astype(str), trade_date=series["trade_date"].astype(str))
    index_rows = series[series["dataset"] == "index_daily"]
    sw_rows = series[(series["dataset"] == "sw_daily") & series["ts_code"].str.fullmatch(SW_LEVEL1)]
    return {
        "decision": decision,
        "dates": dates,
        "codes": codes,
        "d": daily,
        "has_bar": has_bar,
        "name": info["name"].fillna("").astype(str).to_numpy(),
        "list_date": listed.to_numpy(dtype="datetime64[ns]"),
        "l1_code": info["l1_code"].fillna("").astype(str).to_numpy(),
        "l1_name": info["l1_name"].fillna("").astype(str).to_numpy(),
        "weights": weights[["index_code", "trade_date", "con_code", "weight"]].reset_index(drop=True),
        "index": {code: _wide(index_rows[index_rows["ts_code"] == code], dates)
                  for code in sorted(index_rows["ts_code"].unique())},
        "sw": _wide(sw_rows, dates, by_code=True),
        "_cache": {},
    }


def _wide(rows, dates, by_code=False):
    """{column: array on the panel's dates}; by code: {column: (sessions x codes)} plus the code list."""

    rows = rows.drop_duplicates(["ts_code", "trade_date"], keep="last")
    out = {}
    if not by_code:
        frame = rows.set_index("trade_date").reindex(pd.Index(dates))
        for column in ("open", "close"):
            out[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        return out
    out["codes"] = np.array(sorted(rows["ts_code"].unique()))
    for column in ("close", "amount", "pb"):
        wide = rows.pivot(index="trade_date", columns="ts_code", values=column)
        wide = wide.reindex(index=pd.Index(dates), columns=pd.Index(out["codes"]))
        out[column] = wide.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return out


def compute(panel):
    """(features, sessions, names) float32: every registered feature on every panel row."""

    out = np.full((len(FEATURES), len(panel["dates"]), len(panel["codes"])), np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (_name, _family, function) in enumerate(FEATURES):
            out[k] = function(panel)
    return out


def names():
    return tuple(name for name, _family, _function in FEATURES)


def families():
    return tuple(family for _name, family, _function in FEATURES)


# ----------------------------------------------------------------- helpers


def _cached(panel, key, build):
    cache = panel["_cache"]
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _positive(a):
    return np.where(a > 0, a, np.nan)


def _lag(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def _ffill(a, limit):
    return pd.DataFrame(a).ffill(limit=limit).to_numpy()


def _roll(a, window, minimum, how):
    return getattr(pd.DataFrame(a).rolling(window, min_periods=minimum), how)().to_numpy()


def _pair(x, y):
    both = np.isfinite(x) & np.isfinite(y)
    return np.where(both, x, np.nan), np.where(both, y, np.nan)


def _roll_corr(x, y, window, minimum):
    x, y = _pair(x, y)
    return pd.DataFrame(x).rolling(window, min_periods=minimum).corr(pd.DataFrame(y)).to_numpy()


def rolling_slope(y, x, window, minimum):
    """Rolling OLS slope of y (sessions x names) on x (sessions x names), pairs only."""

    x, y = _pair(x, y)
    cov = pd.DataFrame(y).rolling(window, min_periods=minimum).cov(pd.DataFrame(x)).to_numpy()
    var = _roll(x, window, minimum, "var")
    return cov / np.where(var > 0, var, np.nan)


def _wide_by_name(panel, values):
    """Broadcast a (sessions,) series to (sessions x names)."""

    return np.repeat(values[:, None], len(panel["codes"]), axis=1)


def daily_returns(panel):
    """Daily decimal returns, the listing day of each name masked (its pct_chg is against the issue price)."""

    def build():
        ret = panel["d"]["pct_chg"].copy()
        listed = panel["list_date"].astype("datetime64[D]")
        day = pd.to_datetime(pd.Index(panel["dates"]), format="%Y%m%d").to_numpy(dtype="datetime64[D]")
        ret[day[:, None] == listed[None, :]] = np.nan
        return ret
    return _cached(panel, "ret", build)


def index_returns(panel, code):
    close = panel["index"].get(code, {}).get("close")
    if close is None:
        return np.full(len(panel["dates"]), np.nan)
    previous = np.concatenate([[np.nan], close[:-1]])
    return close / previous - 1.0


def _adj_close(panel):
    return _cached(panel, "adj_close", lambda: panel["d"]["close"] * panel["d"]["adj_factor"])


def _industry_columns(panel):
    """(names,) column of each name's Shenwan level-1 series in panel['sw'], -1 when unmapped."""

    def build():
        position = {code: k for k, code in enumerate(panel["sw"]["codes"])}
        return np.array([position.get(code, -1) for code in panel["l1_code"]], dtype=np.int64)
    return _cached(panel, "sw_col", build)


def _by_industry(panel, wide):
    """(sessions x Shenwan codes) -> (sessions x names), NaN for names without an industry series."""

    column = _industry_columns(panel)
    out = wide[:, np.clip(column, 0, None)] if wide.shape[1] else np.full((wide.shape[0], len(column)), np.nan)
    out = np.array(out, dtype=np.float64)
    out[:, column < 0] = np.nan
    return out


def _groups(labels):
    """{label: name columns} over non-empty labels."""

    out = {}
    for k, label in enumerate(labels):
        if label:
            out.setdefault(label, []).append(k)
    return {label: np.array(columns) for label, columns in out.items()}


# ------------------------------------------------------- index sections


def _sections(panel):
    """Per tier: section dates, (sections x names) weights, the section in force on each row.

    The section in force on row t is the newest one dated <= t: it is stamped
    17:30 on its own date, so the 08:30 decision that reads row t as T-1 sees it.
    """

    def build():
        out = {}
        frame = panel["weights"]
        position = pd.Series(np.arange(len(panel["codes"])), index=pd.Index(panel["codes"]))
        for code, tier in TIERS:
            rows = frame[frame["index_code"] == code]
            dates = np.array(sorted(rows["trade_date"].unique()))
            weight = np.zeros((len(dates), len(panel["codes"])))
            column = position.reindex(rows["con_code"].to_numpy()).to_numpy()
            keep = np.isfinite(column)
            weight[np.searchsorted(dates, rows["trade_date"].to_numpy()[keep]), column[keep].astype(np.int64)] = \
                rows["weight"].to_numpy(dtype=np.float64)[keep]
            slot = np.searchsorted(dates, panel["dates"], side="right") - 1
            out[code] = {"tier": tier, "dates": dates, "weight": weight, "slot": slot}
        return out
    return _cached(panel, "sections", build)


def _on_rows(section_values, slot, valid_from):
    """Expand (sections x names) values to rows; NaN where fewer than valid_from + 1 sections are visible."""

    rows = np.full((len(slot), section_values.shape[1]), np.nan)
    ok = slot >= valid_from
    rows[ok] = section_values[slot[ok]]
    return rows


def _tier(panel):
    """3 CSI 300, 2 CSI 500, 1 CSI 1000, 0 none of them; NaN while any of the three has no visible section."""

    def build():
        sections = _sections(panel)
        tier = np.zeros((len(panel["dates"]), len(panel["codes"])))
        known = np.ones(len(panel["dates"]), dtype=bool)
        for code, value in reversed(TIERS):
            entry = sections[code]
            member = _on_rows(entry["weight"] > 0, entry["slot"], 0)
            tier = np.where(member == 1, value, tier)
            known &= entry["slot"] >= 0
        tier[~known] = np.nan
        return tier
    return _cached(panel, "tier", build)


def _per_tier(panel, section_function, valid_from):
    """First non-NaN over the three indices of a per-section, per-member quantity."""

    out = np.full((len(panel["dates"]), len(panel["codes"])), np.nan)
    for code, _tier_value in TIERS:
        entry = _sections(panel)[code]
        if not len(entry["dates"]):
            continue
        values = section_function(entry)
        rows = _on_rows(values, entry["slot"], valid_from)
        out = np.where(np.isnan(out), rows, out)
    return out


# ----------------------------------------------------- price-to-sales


@_feature("ps")
def sp_ttm(panel):
    """1 / ps_ttm on the row (positive ps_ttm only). Reproduces ps, ps300, ps500, ps1000, psmain, psxfin,
    psgem, pscy, psout (score -ps_ttm on their pools; exact rank)."""

    return 1.0 / _positive(panel["d"]["ps_ttm"])


@_feature("ps")
def sp_lag21(panel):
    """1 / ps_ttm 21 sessions before the row (a name's last bar up to 10 sessions back). Reproduces pslag,
    pslagm, pslage, pslthen, pslbig, pslsml, pslboth, psltail, psl500, psl300, pslcy, psloutm (exact rank
    for names with a bar on the lag day); psl10 / psl42 / psstay are re-timings of it."""

    return _lag(_ffill(1.0 / _positive(panel["d"]["ps_ttm"]), 10), 21)


@_feature("ps")
def ps_gap(panel):
    """log(ps) - log(ps_ttm), both positive. Reproduces psacc, psgap (exact); with sp_ttm it also carries
    the static multiple of psly, psly1k, psly500, psmly, pslly."""

    return np.log(_positive(panel["d"]["ps"])) - np.log(_positive(panel["d"]["ps_ttm"]))


@_feature("ps")
def sp_ind_rel(panel):
    """Shenwan L1 median ps_ttm minus own ps_ttm, over names with a positive ps_ttm on the row. Reproduces
    psind (exact up to the Beijing names the panel does not read), ps1ind (approx.: median over all names,
    not CSI 1000 members); pssize / ps1sz (size residual) are not reproduced separately."""

    ps = _positive(panel["d"]["ps_ttm"])
    out = np.full_like(ps, np.nan)
    for columns in _groups(panel["l1_name"]).values():
        block = ps[:, columns]
        median = np.nanmedian(np.where(np.isfinite(block).any(axis=1, keepdims=True), block, 0.0), axis=1)
        out[:, columns] = median[:, None] - block
    return out


# ------------------------------------------------- other valuation ratios


@_feature("valuation")
def ep_ttm(panel):
    """1 / pe_ttm; 0 for a loss (pe_ttm null with daily_basic present); NaN without daily_basic.
    Reproduces pe, pe1000, pelagm (exact rank among positive pe_ttm) and ep_csi500_8y."""

    pe = panel["d"]["pe_ttm"]
    basic = panel["d"]["total_mv"] > 0
    return np.where(pe > 0, 1.0 / np.where(pe > 0, pe, 1.0), np.where(basic, 0.0, np.nan))


@_feature("valuation")
def bp(panel):
    """1 / pb, positive pb only. Reproduces pb, pb1000, pblagm (score -pb; exact rank)."""

    return 1.0 / _positive(panel["d"]["pb"])


@_feature("valuation")
def pe_gap(panel):
    """log(pe) - log(pe_ttm), both positive. Reproduces earnac (exact)."""

    return np.log(_positive(panel["d"]["pe"])) - np.log(_positive(panel["d"]["pe_ttm"]))


@_feature("valuation")
def roe_proxy(panel):
    """pb / pe (static), both positive: the book's earnings yield. Reproduces roe1k, roeout (exact);
    mgnlag (ps_ttm / pe_ttm, 21 sessions back) is not reproduced separately."""

    return _positive(panel["d"]["pb"]) / _positive(panel["d"]["pe"])


@_feature("valuation")
def dv(panel):
    """Trailing cash dividend yield: dv_ttm, else dv_ratio, else 0 (no dividend); NaN without daily_basic.
    Reproduces dvy_csi300_8y and the dvy_* sweep (exact)."""

    first = np.where(panel["d"]["dv_ttm"] > 0, panel["d"]["dv_ttm"], np.nan)
    first = np.where(np.isnan(first) & (panel["d"]["dv_ratio"] > 0), panel["d"]["dv_ratio"], first)
    return np.where(np.isnan(first) & (panel["d"]["total_mv"] > 0), 0.0, first)


# ------------------------------------------------- pb / ps, sales, size


@_feature("pbps")
def pb_over_ps(panel):
    """pb / ps (static), both positive: sales over book. Reproduces aturn, at1k, atcy (exact); atlag is a
    re-timing; iat1k (industry median) is not reproduced separately."""

    return _positive(panel["d"]["pb"]) / _positive(panel["d"]["ps"])


@_feature("sales")
def log_sales(panel):
    """log(circ_mv / ps_ttm): the trailing sales the float carries. Reproduces sale1000, sale500, salem
    (exact rank); ssch1000 (industry share) is not reproduced separately."""

    return np.log(_positive(panel["d"]["circ_mv"]) / _positive(panel["d"]["ps_ttm"]))


@_feature("size_in")
def log_circ_mv(panel):
    """log circ_mv. Reproduces big500, sz1k (exact rank on their pools); bigind / peersize (industry-relative)
    approx."""

    return np.log(_positive(panel["d"]["circ_mv"]))


@_feature("size_out")
def size_out3(panel):
    """log circ_mv of names in none of CSI 300 / 500 / 1000 (tier 0), NaN otherwise. Reproduces next1k
    (exact rank), nextmb, nextlo (its subsets)."""

    return np.where(_tier(panel) == 0, np.log(_positive(panel["d"]["circ_mv"])), np.nan)


# --------------------------------------------------- index membership


@_feature("tenure")
def member_tier(panel):
    """3 CSI 300, 2 CSI 500, 1 CSI 1000, 0 none, by the section in force. The pool split of every census
    arm that scored one index's members (ps300, ps500, ps1000, new500, ...)."""

    return _tier(panel)


@_feature("tenure")
def tenure_own(panel):
    """Of the last TENURE_SECTIONS sections of its own index, how many hold the name; 0 for a name in none
    of the three. Reproduces newmem, new500, new300 (-count), stay500 (count), midnew, midflat, mid1000,
    mid300 (|count - 4|): exact at a month-start decision, where the census's 400 calendar days hold 14
    sections; newbig, psnew, newbmk, midbmk, mid3bmk are tie-break or benchmark variants; mid50, midcy,
    newcy (SSE 50 / ChiNext) are not covered."""

    def count(entry):
        member = (entry["weight"] > 0).astype(np.float64)
        total = np.cumsum(member, axis=0)
        before = np.zeros_like(total)
        before[TENURE_SECTIONS:] = total[:-TENURE_SECTIONS]
        return np.where(member > 0, total - before, np.nan)
    out = _per_tier(panel, count, TENURE_SECTIONS - 1)
    tier = _tier(panel)
    return np.where(tier == 0, 0.0, np.where(tier > 0, out, np.nan))


@_feature("exit")
def exit_ago(panel):
    """Sections since the name last left CSI 300, 500 or 1000 (the smallest over the three), 1..6; 7 when
    it left none in the last EXIT_SECTIONS; NaN while any index has fewer than 7 visible sections. Reproduces
    exit300, exit500, exit1000, exitbmk, exitwide (approx.: no size tie-break); exit50 / exitcy (SSE 50,
    ChiNext) are not covered."""

    out = np.full((len(panel["dates"]), len(panel["codes"])), float(EXIT_SECTIONS + 1))
    known = np.ones(len(panel["dates"]), dtype=bool)
    for code, _tier_value in TIERS:
        entry = _sections(panel)[code]
        member = entry["weight"] > 0
        last_in = np.where(member, np.arange(len(member))[:, None], -1)
        last_in = np.maximum.accumulate(last_in, axis=0) if len(member) else last_in
        ago = np.arange(len(member))[:, None] - last_in
        value = np.where(~member & (last_in >= 0) & (ago <= EXIT_SECTIONS), ago, np.nan)
        rows = _on_rows(value, entry["slot"], EXIT_SECTIONS)
        out = np.fmin(out, rows)
        known &= entry["slot"] >= EXIT_SECTIONS
    out[~known] = np.nan
    return out


@_feature("weight")
def weight_ind_share(panel):
    """Index weight over the summed weight of the same index's members in the same Shenwan L1 (at least 4
    of them). Reproduces champ, champ300, champ1000, champbmk, ch3bmk (exact on their members); runner,
    nlead, pond approx.; wtheavy (raw weight) is not reproduced separately."""

    groups = _groups(panel["l1_name"])

    def share(entry):
        weight = entry["weight"]
        out = np.full_like(weight, np.nan)
        for columns in groups.values():
            block = weight[:, columns]
            count = (block > 0).sum(axis=1, keepdims=True)
            total = block.sum(axis=1, keepdims=True)
            out[:, columns] = np.where((block > 0) & (count >= 4), block / np.where(total > 0, total, 1.0), np.nan)
        return out
    return _per_tier(panel, share, 0)


@_feature("weight")
def weight_delta(panel):
    """Weight in the newest section minus the previous one (absent = 0) for the index's current members.
    Reproduces wdelta (exact on CSI 1000 members); chdelta (share change), wtstab, iwadd approx."""

    def delta(entry):
        weight = entry["weight"]
        previous = np.vstack([np.full((1, weight.shape[1]), np.nan), weight[:-1]])
        return np.where(weight > 0, weight - previous, np.nan)
    return _per_tier(panel, delta, 1)


@_feature("entry")
def entry_rank(panel):
    """CSI 300 entry rank: among non-ST names listed > 91 days (365 for 300/301/688/689), the top half by
    mean daily amount (CSI 300 members: top 60 %) ranked by mean total_mv, both over the last 244 sessions
    with a positive amount. 1 = largest. Reproduces entry (-rank) and boundary (-|rank - 270|), approx.:
    244 sessions, not 365 calendar days."""

    d = panel["d"]
    traded = d["amount"] > 0
    amount = _roll(np.where(traded, d["amount"], np.nan), 244, 20, "mean")
    value = _roll(np.where(traded, d["total_mv"], np.nan), 244, 20, "mean")
    day = pd.to_datetime(pd.Index(panel["dates"]), format="%Y%m%d").to_numpy(dtype="datetime64[D]")
    age = (day[:, None] - panel["list_date"].astype("datetime64[D]")[None, :]).astype(np.float64)
    codes = pd.Series(panel["codes"])
    need = np.where(codes.str.startswith(("300", "301", "688", "689")).to_numpy(), 365.0, 91.0)
    st = pd.Series(panel["name"]).str.contains("ST|退").to_numpy()
    space = np.isfinite(amount) & np.isfinite(value) & (age > need[None, :]) & ~st[None, :]
    liquidity = pd.DataFrame(np.where(space, amount, np.nan)).rank(axis=1, ascending=False, method="first").to_numpy()
    liquidity = liquidity / space.sum(axis=1, keepdims=True).clip(1)
    in300 = _tier(panel) == 3
    eligible = space & ((liquidity <= 0.5) | (in300 & (liquidity <= 0.6)))
    rank = pd.DataFrame(np.where(eligible, value, np.nan)).rank(axis=1, ascending=False, method="first").to_numpy()
    known = _sections(panel)[CSI300]["slot"] >= 0
    rank[~known] = np.nan
    return rank


# ----------------------------------------------- industry aggregates (Shenwan L1 indexes)


@_feature("industry")
def ind_ret21(panel):
    """The name's Shenwan L1 index close-to-close return over the last 21 closes. Reproduces indrev
    (exact), indcalm (-|ret|)."""

    close = panel["sw"]["close"]
    return _by_industry(panel, close / _lag(close, 20) - 1.0)


@_feature("industry")
def ind_amt_chg21(panel):
    """The Shenwan L1 index's amount over its amount 20 sessions earlier, minus one. Reproduces indrise
    (exact), indflow approx."""

    amount = _positive(panel["sw"]["amount"])
    return _by_industry(panel, amount / _lag(amount, 20) - 1.0)


@_feature("industry")
def ind_bp(panel):
    """1 / pb of the Shenwan L1 index. Reproduces indpbl (score -pb; exact rank); indpb (pb change), swpe
    (pe), iroe (pb / pe) approx.; indto, indmv, inddisp, broad are not reproduced separately."""

    return _by_industry(panel, 1.0 / _positive(panel["sw"]["pb"]))


# ------------------------------------------------------- peer-relative


@_feature("peer")
def ret21_vs_ind(panel):
    """The name's adjusted return over 20 sessions minus its Shenwan L1 index's. Reproduces peerlag,
    peerlead (opposite signs) approx."""

    close = _adj_close(panel)
    return close / _lag(close, 20) - 1.0 - ind_ret21(panel)


@_feature("peer")
def amt_share_ind(panel):
    """The row's amount over the summed amount of every name with the same Shenwan L1 code. Reproduces
    leader (exact up to Beijing names); peerliq approx."""

    amount = np.where(panel["d"]["amount"] > 0, panel["d"]["amount"], np.nan)
    out = np.full_like(amount, np.nan)
    for columns in _groups(panel["l1_code"]).values():
        block = amount[:, columns]
        total = np.nansum(block, axis=1, keepdims=True)
        out[:, columns] = block / np.where(total > 0, total, np.nan)
    return out


@_feature("peer")
def ind_corr21(panel):
    """Correlation of the name's daily return with its Shenwan L1 index's over 21 sessions (15 pairs).
    Reproduces decouple (score -corr) approx.: decimal returns, not unadjusted close changes."""

    close = panel["sw"]["close"]
    industry = _by_industry(panel, close / _lag(close, 1) - 1.0)
    return _roll_corr(daily_returns(panel), industry, 21, 15)


# ----------------------------------------------------------- listing age


@_feature("age")
def log_age(panel):
    """log(1 + days since list_date) on the row's date. Reproduces listed, age1k, ageout (exact rank)."""

    day = pd.to_datetime(pd.Index(panel["dates"]), format="%Y%m%d").to_numpy(dtype="datetime64[D]")
    age = (day[:, None] - panel["list_date"].astype("datetime64[D]")[None, :]).astype(np.float64)
    return np.log1p(np.where(age > 0, age, np.nan))


# ---------------------------------------------------- free float / share structure


@_feature("float")
def float_ratio(panel):
    """circ_mv / total_mv. Reproduces freefloat (exact), lock1000 (opposite sign)."""

    return panel["d"]["circ_mv"] / _positive(panel["d"]["total_mv"])


@_feature("float")
def free_over_float(panel):
    """free_share / float_share. Reproduces freeshare (exact)."""

    return panel["d"]["free_share"] / _positive(panel["d"]["float_share"])


@_feature("float")
def free_chg60(panel):
    """free_share on the row over free_share 59 sessions earlier, minus one. Reproduces freegrow (score
    -growth; exact for names with a bar on both days)."""

    free = _positive(panel["d"]["free_share"])
    return free / _lag(free, 59) - 1.0


# ------------------------------------------------------ volume / turnover


@_feature("volume")
def amt_cv20(panel):
    """Coefficient of variation of amount over 20 sessions (positive amounts, at least 15). Reproduces amtcv
    (score -cv; exact)."""

    amount = np.where(panel["d"]["amount"] > 0, panel["d"]["amount"], np.nan)
    return _roll(amount, 20, 15, "std") / _roll(amount, 20, 15, "mean")


@_feature("volume")
def vwap_dev(panel):
    """Mean |close - amount / vol| / close over 27 sessions (at least 10). Reproduces nearvw (score -gap)
    approx.: 27 sessions stand for the census's rows of the last 40 calendar days."""

    d = panel["d"]
    ok = (d["close"] > 0) & (d["vol"] > 0) & (d["amount"] > 0)
    gap = np.abs(d["close"] - d["amount"] / np.where(ok, d["vol"], np.nan)) / np.where(ok, d["close"], np.nan)
    return _roll(gap, 27, 10, "mean")


@_feature("volume")
def pv_corr20(panel):
    """Correlation of the daily return with amount over 20 sessions (at least 15). Reproduces pvcorr (exact)."""

    return _roll_corr(panel["d"]["pct_chg"], panel["d"]["amount"], 20, 15)


@_feature("volume")
def amt_ac20(panel):
    """Correlation of amount with the previous session's amount, pairs inside the last 20 sessions (at least
    15). Reproduces amtac (exact for names without a gap in the window)."""

    amount = panel["d"]["amount"]
    return _roll_corr(amount, _lag(amount, 1), 19, 15)


@_feature("volume")
def amt_lagret20(panel):
    """Correlation of amount with the previous session's return, pairs inside the last 20 sessions (at least
    15). Reproduces amtlag (exact for names without a gap in the window)."""

    return _roll_corr(panel["d"]["amount"], _lag(panel["d"]["pct_chg"], 1), 19, 15)


@_feature("volume")
def dn_share(panel):
    """Share of the amount traded on down days over 27 sessions (at least 10 bars). Reproduces dn20, dn1000
    approx.: 27 sessions stand for the census's rows of the last 40 calendar days; dn60 is a re-timing;
    reldn (down against CSI 1000) approx."""

    d = panel["d"]
    ok = np.isfinite(d["pct_chg"]) & (d["amount"] > 0)
    amount = np.where(ok, d["amount"], np.nan)
    down = np.where(ok, np.where(d["pct_chg"] < 0, d["amount"], 0.0), np.nan)
    return _roll(down, 27, 10, "sum") / _roll(amount, 27, 10, "sum")


# ------------------------------------------------- beta / index sensitivity (CSI 1000)
# 67 sessions stand for the census's rows of the last 100 calendar days.


@_feature("beta")
def beta(panel):
    """Slope of the daily return on CSI 1000's over 67 sessions (at least 30). Reproduces beta60, beta1k
    (score -beta) approx."""

    market = _wide_by_name(panel, index_returns(panel, CSI1000))
    return rolling_slope(daily_returns(panel), market, 67, 30)


@_feature("beta")
def delay_beta(panel):
    """Slope of the daily return on CSI 1000's return of the previous session, 67 sessions (at least 30).
    Reproduces delay60, delay1k approx."""

    market = _wide_by_name(panel, _lag(index_returns(panel, CSI1000)[:, None], 1)[:, 0])
    return rolling_slope(daily_returns(panel), market, 67, 30)


@_feature("beta")
def coskew(panel):
    """Slope of the daily return on the squared demeaned CSI 1000 return (demeaned by its own window mean),
    67 sessions (at least 30). Reproduces csk1k (score -slope) approx."""

    window, minimum = 67, 30
    m = index_returns(panel, CSI1000)
    centre = pd.Series(m).rolling(window, min_periods=1).mean().to_numpy()[:, None]
    y, mm = _pair(daily_returns(panel), _wide_by_name(panel, m))

    def total(a):
        return _roll(a, window, 1, "sum")
    n = _roll(y, window, 1, "count")
    s1, s2, s3, s4 = total(mm), total(mm ** 2), total(mm ** 3), total(mm ** 4)
    sy, sym, sym2 = total(y), total(y * mm), total(y * mm ** 2)
    sx = s2 - 2 * centre * s1 + n * centre ** 2
    sxx = s4 - 4 * centre * s3 + 6 * centre ** 2 * s2 - 4 * centre ** 3 * s1 + n * centre ** 4
    sxy = sym2 - 2 * centre * sym + centre ** 2 * sy
    denominator = n * sxx - sx ** 2
    slope = (n * sxy - sx * sy) / np.where(denominator > 0, denominator, np.nan)
    return np.where(n >= minimum, slope, np.nan)


# ----------------------------------------------------- price range / candle shape


@_feature("range")
def range20(panel):
    """Mean (high - low) / pre_close over 20 sessions (at least 15). Reproduces range20_csi300_8y,
    range20_csi500_8y (score -range) approx.: 20 sessions, not the name's own last 20 bars."""

    d = panel["d"]
    return _roll((d["high"] - d["low"]) / _positive(d["pre_close"]), 20, 15, "mean")


@_feature("range")
def body20(panel):
    """Mean |close - open| / (high - low) over 20 sessions (days with high > low, at least 15). Reproduces
    body (exact)."""

    d = panel["d"]
    span = d["high"] - d["low"]
    return _roll(np.abs(d["close"] - d["open"]) / np.where(span > 0, span, np.nan), 20, 15, "mean")


@_feature("range")
def limdn60(panel):
    """Share of the bars of the last 60 sessions whose low touched the down limit (at least 40 bars).
    Reproduces limdn (score -share; exact)."""

    d = panel["d"]
    bar = panel["has_bar"].astype(np.float64)
    hit = np.where(np.isfinite(d["low"]) & np.isfinite(d["down_limit"]) & (d["low"] <= d["down_limit"] + 0.01),
                   1.0, 0.0) * bar
    bars = _roll(bar, 60, 1, "sum")
    return np.where(bars >= 40, _roll(hit, 60, 1, "sum") / np.where(bars > 0, bars, np.nan), np.nan)


@_feature("range")
def room_up(panel):
    """(up_limit - close) / close on the row. Reproduces room (exact)."""

    d = panel["d"]
    return np.where(d["up_limit"] > 0, (d["up_limit"] - d["close"]) / _positive(d["close"]), np.nan)


# ------------------------------------------------- corporate actions / dividends


@_feature("corp")
def cash_share(panel):
    """Cash share of the name's last distribution in the trailing 244 sessions, read from the bars: an
    ex-date is an adj_factor step; the stock part is the total_share step settled two sessions later; the
    cash per share is the cum close minus pre_close x (1 + stock ratio). Placed on the row two sessions
    after the ex-date. A step across a suspension counts only if the previous bar is at most 20 sessions
    back. Reproduces cashmix approx. (stock part marked at the cum close, not the decision close); rights
    issues (negative cash) are skipped. paylag needs the pay date, which no mounted table carries, and is
    not covered."""

    d = panel["d"]
    factor = _ffill(d["adj_factor"], 20)
    shares = _ffill(d["total_share"], 20)
    close = _ffill(d["close"], 20)
    event = (factor / _lag(factor, 1) > 1.0 + 1e-6) & panel["has_bar"]
    settled = np.vstack([shares[2:], np.full((2, shares.shape[1]), np.nan)])
    stock = np.clip(settled / _lag(shares, 1) - 1.0, 0.0, None)
    stock = np.where(stock < 1e-4, 0.0, stock)
    cum = _lag(close, 1)
    cash = cum - d["pre_close"] * (1.0 + stock)
    cash = np.where(cash > -0.005 * cum, np.clip(cash, 0.0, None), np.nan)
    whole = cash + cum * stock
    value = np.where(event & (whole > 0), cash / np.where(whole > 0, whole, np.nan), np.nan)
    placed = _lag(value, 2)
    return _ffill(placed, 243)


# ------------------------------------------------------------- calendar


@_feature("calendar")
def tom_ret(panel):
    """Mean daily return on month-turn sessions (first three and last three of a month) over the last 12
    completed months (at least 30 such returns). A month is complete once a later session or the decision
    falls in a later month. Reproduces tom1k approx.: completed months only, not 400 calendar days."""

    ret = daily_returns(panel)
    month = np.array([int(day[:6]) for day in panel["dates"]])
    edges = np.flatnonzero(np.diff(month)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(month)]])
    turn = np.zeros(len(month), dtype=bool)
    for begin, end in zip(starts, ends):
        turn[begin:begin + 3] = True
        turn[max(begin, end - 3):end] = True
    decision_month = int(panel["decision"].tz_convert("Asia/Shanghai").strftime("%Y%m"))
    complete = np.ones(len(starts), dtype=bool)
    complete[-1] = month[-1] != decision_month
    ok = np.isfinite(ret) & turn[:, None]
    sums = np.add.reduceat(np.where(ok, ret, 0.0), starts, axis=0)
    counts = np.add.reduceat(ok.astype(np.float64), starts, axis=0)
    zero = np.zeros((1, ret.shape[1]))
    cum_sum = np.vstack([zero, np.cumsum(sums, axis=0)])
    cum_count = np.vstack([zero, np.cumsum(counts, axis=0)])
    out = np.full(ret.shape, np.nan)
    position = np.repeat(np.arange(len(starts)), ends - starts)
    last_of_month = np.zeros(len(month), dtype=bool)
    last_of_month[ends - 1] = True
    through = np.where(last_of_month & complete[position], position, position - 1)
    for t in range(len(month)):
        e = through[t]
        if e < 12:      # the window's first month may be partial: twelve full months must precede it
            continue
        count = cum_count[e + 1] - cum_count[e - 11]
        total = cum_sum[e + 1] - cum_sum[e - 11]
        out[t] = np.where(count >= 30, total / np.where(count > 0, count, np.nan), np.nan)
    return out
