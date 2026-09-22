"""The intraday price-path block: what a day's shape says that its OHLC cannot.

Alpha158 reads each stock-day through its open, high, low, close, volume and
turnover-weighted average price. Two stock-days with the same six numbers can
have travelled completely different paths: one drifted up all session in small
steps, the other sat flat, collapsed in three minutes and recovered. The SIGN
of the day's variation, the tails of its minute-return distribution, WHEN in
the session the price moved and WHEN the volume traded are invisible to every
one of the 158 operators.

This block reads them from `intraday_stats`: one row per stock-day, reduced
from that day's own 1-minute bars by the data layer (the path starts at the
day's first traded open, so the overnight gap is not an intraday jump) and
stamped with the same close contract as `intraday_flow`, so day t's row is
first visible at the 08:30 decision of t+1 and lands on panel bar t exactly
like day t's `daily` bar. No minute mount is needed to read it.

Four groups, fifteen columns, every one scale-free before it leaves this
module, so the cross-sectional z-score over the bar's constituents that
`data.py` applies to the carrier's columns applies to these unchanged:

    asymmetry   rsj_1, rsj_s, rsj_l, sj_rel                the sign of the variation
    moments     rsk_s, rsk_l, rku_l                        the tails
    profile     ro30_s, rmid_l, rc30_s, rc30_l, vwd_s      when the price moved
    volume      vo30_l, vc30_l, vc30_z                     when the volume traded

RSJ is the relative signed jump variation of Bollerslev, Li & Zhao (2020),
`sjv / rv` = (RV+ - RV-) / RV in [-1, 1]. It is an exact function of the
dataset's `rv_down_share` (RSJ = 1 - 2 * rv_down_share), so the downside share
enters through RSJ and is deliberately not a second, collinear column. Every
window statistic is the MEAN of the daily measure over the window, the paper's
own weekly construction; `sj_rel` is the newest signed jump in units of the
name's trailing mean variance, which separates a large signed jump from a quiet
day whose tiny variance happened to fall on one side.

Two dataset columns are read and deliberately not used as levels: the one-day
`vwap_dev` is already Alpha158's VWAP0 (the daily VWAP is amount / volume
either way), so only its short mean enters; and `rv` enters only as the
normaliser of `sj_rel`, because Alpha158's STD columns already read the size
of the variation -- this block is about its shape.

Masking, not dropping. A stock-day whose statistics rest on fewer than
`knobs.ISTATS_MIN_BARS` session-grid bars (a gap in the minute layer), or that
has no realized variance (a board sealed from the open, where the dataset
leaves the moments null), is set to NaN in every column before any rolling
window: its moments do not describe a price path. NaN is the honest value --
the windows skip it, and `data.py` fills what survives to the neutral 0 only at
the very end, in the same cross-section as every Alpha158 column.

Units, once, here (`pit-field-map.md` is the authority): returns are
fractions, `rv` and `sjv` are fractions squared, `rsk` and `rku` are
dimensionless, the volume shares are fractions in [0, 1], `n_bars` is a count.
"""

import numpy as np
import pandas as pd

from lib import events, knobs

STATS_COLUMNS = ["rv", "sjv", "rsk", "rku", "ret_open30", "ret_mid", "ret_close30",
                 "vwap_dev", "vol_open30_share", "vol_close30_share", "n_bars"]

BLOCK_NAMES = ("rsj_1", "rsj_s", "rsj_l", "sj_rel",
               "rsk_s", "rsk_l", "rku_l",
               "ro30_s", "rmid_l", "rc30_s", "rc30_l", "vwd_s",
               "vo30_l", "vc30_l", "vc30_z")


def _roll(matrix, window, how):
    """(names, dates) trailing `how` over `window` bars; needs half the window present."""

    rolling = pd.DataFrame(matrix.T).rolling(window, min_periods=max(2, window // 2))
    return getattr(rolling, how)().to_numpy().T


def _ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / np.where(np.abs(denominator) > 0.0, denominator, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def build(context, panel):
    """(names, dates, len(BLOCK_NAMES)) raw block values aligned to the panel axes."""

    dates, codes = panel["dates"], [str(code) for code in panel["symbols"]]
    window = knobs.ISTATS_LOOKBACK_DAYS + int(len(dates) * 1.6)
    stats = events.read(context, "intraday_stats", STATS_COLUMNS, window, codes)
    day = {name: events.dense(stats, name, dates, codes) for name in STATS_COLUMNS}
    usable = (day["n_bars"] >= knobs.ISTATS_MIN_BARS) & (day["rv"] > 0.0)
    x = {name: np.where(usable, values, np.nan) for name, values in day.items()}

    short, long = knobs.ISTATS_SHORT, knobs.ISTATS_LONG
    rsj = _ratio(x["sjv"], x["rv"])
    close_share = x["vol_close30_share"]
    close_share_l = _roll(close_share, long, "mean")
    block = {
        "rsj_1": rsj,
        "rsj_s": _roll(rsj, short, "mean"),
        "rsj_l": _roll(rsj, long, "mean"),
        "sj_rel": _ratio(x["sjv"], _roll(x["rv"], long, "mean")),
        "rsk_s": _roll(x["rsk"], short, "mean"),
        "rsk_l": _roll(x["rsk"], long, "mean"),
        "rku_l": _roll(x["rku"], long, "mean"),
        "ro30_s": _roll(x["ret_open30"], short, "mean"),
        "rmid_l": _roll(x["ret_mid"], long, "mean"),
        "rc30_s": _roll(x["ret_close30"], short, "mean"),
        "rc30_l": _roll(x["ret_close30"], long, "mean"),
        "vwd_s": _roll(x["vwap_dev"], short, "mean"),
        "vo30_l": _roll(x["vol_open30_share"], long, "mean"),
        "vc30_l": close_share_l,
        "vc30_z": _ratio(close_share - close_share_l, _roll(close_share, long, "std")),
    }
    return np.stack([block[name] for name in BLOCK_NAMES], axis=2)


def coverage(values, panel):
    """Per-bar share of the bar's constituents carrying a finite value in EVERY column.

    Reported by the round-0 census and by every result note. A block that is
    mostly missing is a finding about the data, not a property of the model,
    and it has to be read before any ablation reading is believed.
    """

    member = panel["member"]
    if values.shape[2] == 0:
        return np.full(len(panel["dates"]), np.nan)
    finite = np.isfinite(values).all(axis=2) & member
    total = member.sum(axis=0)
    return np.where(total > 0, finite.sum(axis=0) / np.maximum(total, 1), np.nan)
