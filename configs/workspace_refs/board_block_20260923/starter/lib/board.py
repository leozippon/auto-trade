"""The limit-board and dragon-tiger block: an information family no arm has ever read.

Four datasets sit in every experiment's snapshot by default -- `limit_list_d`,
`kpl_list`, `top_list`, `top_inst` -- and across every frozen strategy revision
and every reference pack in this repository, zero files reference a single one
of their columns. They are whole-market since 2020-01, unit-verified for the
tradable-relevant columns, and structurally different from anything Alpha158
reads: a limit-up is not a large return, it is a return that stopped at a
regulated wall with a queue behind it, and a dragon-tiger listing is the
exchange naming the five seats that did the trading.

What the block encodes, in five groups and seventeen columns:

    limits        lim_up_l, lim_up_x, lim_dn_x, lim_brk_x, lim_streak
                  how often this name hit the wall, in which direction, how
                  often the seal broke, and how long the current run is
    quality       lim_open_x, lim_seal_l, lim_first_l
                  how many times the board opened, how big the sealing order
                  was against the day's turnover, and how early in the session
                  the board first closed -- the three things that separate a
                  strong limit from a weak one
    dragon-tiger  dt_count_x, dt_net_x, inst_net_x, inst_shr_x
                  how often the exchange listed it, the net listed flow, the
                  net taken by INSTITUTIONAL seats (`exalter` = 机构专用), and
                  the institutional share of the listed buy side
    board lists   kpl_up_l, kpl_bid_l, kpl_dn_l, kpl_theme_l
                  vendor limit / auction / down-limit list membership, and the
                  concept-heat membership count: how many themes this name was
                  listed under over the window
    presence      board_any_x
                  did ANY of the four tables carry a row for this name in the
                  long window

Structural zeros, not missing values. These tables are EVENT-CONDITIONAL: a
row exists only when something happened. "No limit-up in sixty days" is not a
missing observation, it is the most common true observation, and encoding it
as NaN would throw away the majority of the signal and hand the booster a
missingness pattern to learn instead. Every column therefore emits an explicit
0.0 where no event occurred, and `board_any_x` is carried so the booster can
separate "quiet" from "weak" without inferring it. `coverage()` reports the
ACTIVE share -- the fraction of a bar's constituents with a non-zero block --
because for this family that, not the non-null share, is the number a result
note has to defend.

Visibility, per dataset, from the snapshot's own rules (`pit-field-map.md`):
`limit_list_d` stamps 16:00 of its own trade date, `top_list` and `top_inst`
20:00 of theirs, so all three are readable at the next morning's 08:30
decision. `kpl_list` stamps 08:30 of the NEXT day, and the as-of view also
cuts each event dataset at its refresh job: `kpl_list` lands with the 08:50
pre-open backfill, so a 08:30 decision sees only what the previous evening's
23:35 refresh held. Its newest visible list is therefore T-2 at a
Tuesday-to-Friday decision and the prior Friday's at a Monday decision -- one
day more lag than the stamp suggests, the same as in live trading.

Duplicate business keys. `top_list` and `top_inst` can carry several rows for
one (name, day): the same day is listed under several reasons, and one reason
may be a multi-day cumulative listing. The repository's own data documentation
says these keys must be aggregated before use rather than assumed unique. This
module reduces each (name, day) -- and, for seats, each (name, day, seat,
side) -- by MEAN, which treats several listings as restatements of one event
instead of as several events. It is a registered choice, not a neutral one,
and `families.md` carries it as a variant axis.

Units, once, here: `limit_list_d.amount` and `fd_amount` are CNY (the same
dataset's `amount` is 1000x `daily.amount`'s raw-lake scale, but the snapshot
normalises `daily.amount` to CNY too, so the RATIO of the two `limit_list_d`
columns is taken within the dataset and never across it); `top_list.l_buy`,
`l_sell`, `net_amount` and `top_inst.buy`, `sell`, `net_buy` are CNY;
`open_times` and `limit_times` are counts; `limit` is categorical (U up-limit,
D down-limit, Z board broken); `first_time` is an HHMMSS clock reading, not a
number to average.
"""

import numpy as np
import pandas as pd

from lib import events, knobs

LIMIT_COLUMNS = ["limit", "first_time", "open_times", "limit_times", "fd_amount", "amount"]
KPL_COLUMNS = ["tag", "theme"]
TOP_LIST_COLUMNS = ["l_buy", "net_amount", "amount"]
TOP_INST_COLUMNS = ["exalter", "side", "buy", "net_buy"]
INSTITUTIONAL_SEAT = "机构专用"
UP_TAGS = ("涨停", "自然涨停")
SESSION_OPEN, SESSION_CLOSE = 9.5 * 3600.0, 15.0 * 3600.0

BLOCK_NAMES = ("lim_up_l", "lim_up_x", "lim_dn_x", "lim_brk_x", "lim_streak",
               "lim_open_x", "lim_seal_l", "lim_first_l",
               "dt_count_x", "dt_net_x", "inst_net_x", "inst_shr_x",
               "kpl_up_l", "kpl_bid_l", "kpl_dn_l", "kpl_theme_l",
               "board_any_x")


def _roll_sum(matrix, window):
    """Rolling sum of a structural-zero matrix; the leading bars are still zeros."""

    return pd.DataFrame(matrix.T).rolling(window, min_periods=1).sum().to_numpy().T


def _last_within(matrix, window):
    """Forward-fill the newest event value for at most `window` bars, 0 where none."""

    filled = pd.DataFrame(matrix.T).ffill(limit=window).to_numpy().T
    return np.where(np.isfinite(filled), filled, 0.0)


def _ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / np.where(np.abs(denominator) > 0.0, denominator, np.nan)
    return np.where(np.isfinite(out), out, 0.0)


def _clock_fraction(values):
    """HHMMSS -> how early in the session, 1.0 at the open and 0.0 at the close."""

    digits = pd.to_numeric(pd.Series(values.ravel()), errors="coerce").to_numpy()
    hour, rest = np.divmod(digits, 10000.0)
    minute, second = np.divmod(rest, 100.0)
    seconds = hour * 3600.0 + minute * 60.0 + second
    part = 1.0 - (seconds - SESSION_OPEN) / (SESSION_CLOSE - SESSION_OPEN)
    return np.clip(part, 0.0, 1.0).reshape(values.shape)


def _event_flag(frame, dates, codes):
    """(names, dates) 1.0 where this dataset carries any row, 0.0 elsewhere."""

    flag = events.dense(frame.assign(_one=1.0), "_one", dates, codes, aggregate="max")
    return np.where(np.isfinite(flag), 1.0, 0.0)


def build(context, panel):
    """(names, dates, len(BLOCK_NAMES)) raw block values aligned to the panel axes."""

    dates, codes = panel["dates"], [str(code) for code in panel["symbols"]]
    window = knobs.BOARD_LOOKBACK_DAYS + int(len(dates) * 1.6)
    limits = events.read(context, "limit_list_d", LIMIT_COLUMNS, window, codes)
    kpl = events.read(context, "kpl_list", KPL_COLUMNS, window, codes)
    top = events.read(context, "top_list", TOP_LIST_COLUMNS, window, codes)
    inst = events.read(context, "top_inst", TOP_INST_COLUMNS, window, codes)

    long, extra = knobs.BOARD_LONG, knobs.BOARD_EXTRA
    zeros = np.zeros((len(codes), len(dates)))

    def limit_flag(mark):
        rows = limits[limits["limit"].astype(str) == mark]
        return _event_flag(rows, dates, codes) if not rows.empty else zeros

    up_flag, down_flag, break_flag = limit_flag("U"), limit_flag("D"), limit_flag("Z")
    up_rows = limits[limits["limit"].astype(str) == "U"]
    streak = events.dense(limits, "limit_times", dates, codes, aggregate="max")
    opens = events.dense(limits, "open_times", dates, codes, aggregate="max")
    seal = _ratio(events.dense(limits, "fd_amount", dates, codes, aggregate="mean"),
                  events.dense(limits, "amount", dates, codes, aggregate="mean"))
    first = _clock_fraction(events.dense(up_rows, "first_time", dates, codes, aggregate="max")
                            if not up_rows.empty else np.full((len(codes), len(dates)), np.nan))
    first = np.where(up_flag > 0, first, np.nan)

    # Duplicate business keys: several listings of one (name, day) are treated
    # as restatements of one event, reduced by mean, before anything is summed.
    top_key = top.groupby(["ts_code", "trade_date"], as_index=False)[["l_buy", "net_amount"]].mean()
    inst_seats = inst[inst["exalter"].astype(str).str.contains(INSTITUTIONAL_SEAT, na=False)]
    if inst_seats.empty:
        inst_key = pd.DataFrame(columns=["ts_code", "trade_date", "buy", "net_buy"])
    else:
        inst_key = (inst_seats.groupby(["ts_code", "trade_date", "exalter", "side"], as_index=False)
                    [["buy", "net_buy"]].mean()
                    .groupby(["ts_code", "trade_date"], as_index=False)[["buy", "net_buy"]].sum())

    amount = np.where(np.isfinite(panel["raw_amount"]), panel["raw_amount"], 0.0)
    dt_flag = _event_flag(top, dates, codes)
    dt_net = np.nan_to_num(events.dense(top_key, "net_amount", dates, codes), nan=0.0)
    dt_buy = np.nan_to_num(events.dense(top_key, "l_buy", dates, codes), nan=0.0)
    inst_net = np.nan_to_num(events.dense(inst_key, "net_buy", dates, codes), nan=0.0)
    inst_buy = np.nan_to_num(events.dense(inst_key, "buy", dates, codes), nan=0.0)

    def kpl_flag(tags):
        rows = kpl[kpl["tag"].astype(str).isin(tags)]
        return _event_flag(rows, dates, codes) if not rows.empty else zeros

    theme_n = events.dense(
        kpl.assign(_n=kpl["theme"].astype(str).str.count("、").add(1).where(kpl["theme"].notna(), 0.0)),
        "_n", dates, codes, aggregate="sum")
    theme_n = np.nan_to_num(theme_n, nan=0.0)

    any_flag = np.maximum.reduce([up_flag, down_flag, break_flag, dt_flag,
                                  _event_flag(inst, dates, codes), _event_flag(kpl, dates, codes)])
    block = {
        "lim_up_l": _roll_sum(up_flag, long) / long,
        "lim_up_x": _roll_sum(up_flag, extra) / extra,
        "lim_dn_x": _roll_sum(down_flag, extra) / extra,
        "lim_brk_x": _roll_sum(break_flag, extra) / extra,
        "lim_streak": _last_within(streak, long),
        "lim_open_x": _ratio(_roll_sum(np.nan_to_num(opens, nan=0.0), extra),
                             _roll_sum(np.maximum.reduce([up_flag, down_flag, break_flag]), extra)),
        "lim_seal_l": _last_within(np.where(up_flag > 0, seal, np.nan), long),
        "lim_first_l": _last_within(first, long),
        "dt_count_x": _roll_sum(dt_flag, extra) / extra,
        "dt_net_x": _ratio(_roll_sum(dt_net, extra), _roll_sum(amount, extra)),
        "inst_net_x": _ratio(_roll_sum(inst_net, extra), _roll_sum(amount, extra)),
        "inst_shr_x": _ratio(_roll_sum(inst_buy, extra), _roll_sum(dt_buy, extra)),
        "kpl_up_l": _roll_sum(kpl_flag(UP_TAGS), long) / long,
        "kpl_bid_l": _roll_sum(kpl_flag(("竞价",)), long) / long,
        "kpl_dn_l": _roll_sum(kpl_flag(("跌停",)), long) / long,
        "kpl_theme_l": _roll_sum(theme_n, long),
        "board_any_x": np.minimum(_roll_sum(any_flag, extra), 1.0),
    }
    return np.stack([block[name] for name in BLOCK_NAMES], axis=2)


def coverage(values, panel):
    """Per-bar share of the bar's constituents whose block is ACTIVE, not merely non-null.

    Every column of this block is finite by construction, so a non-null share
    would read 1.0 on every bar and say nothing. What a result note has to
    defend is how many names the block actually distinguishes, which is the
    share carrying `board_any_x` -- at least one limit, listing or board-list
    row in the long window. On a large-cap constituent universe that share is
    the single most important number about this arm, and it belongs in every
    reading, not only in round 0.
    """

    member = panel["member"]
    if values.shape[2] == 0:
        return np.full(len(panel["dates"]), np.nan)
    active = (values[:, :, BLOCK_NAMES.index("board_any_x")] > 0) & member
    total = member.sum(axis=0)
    return np.where(total > 0, active.sum(axis=0) / np.maximum(total, 1), np.nan)
