"""The signed order-flow block: what the price/volume operators cannot say.

Alpha158 reads four prices, a volume and a turnover-derived average price. It
sees HOW MUCH traded and WHERE the price ended, never WHICH SIDE was taking.
Two stock-days with the same close, the same range and the same volume look
identical to every one of the 158 operators and can be opposite in intent: one
lifted through the offer all afternoon, the other was distributed into a bid.

This block is that missing sign, from two independent sources that measure it
in different ways and therefore disagree in a usable way:

    intraday_flow   one row per stock-day, derived locally from the minute
                    bars by a tick rule on consecutive minute closes, weighted
                    by that minute's volume (`ofi_1d`) or amount
                    (`ofi_amt_1d`) and divided by the session total. It is a
                    MICROSTRUCTURE measure: it counts minutes, not orders, and
                    it needs no minute mount to read
    moneyflow       the vendor's own per-stock-day classification of turnover
                    into small / medium / large / extra-large buy and sell
                    amounts. It is an ORDER-SIZE measure: it counts money by
                    the size of the print, and it knows nothing about when in
                    the session that print happened

Neither is a return predictor on its own in this repository's record -- minute
OFI as a standalone score is closed (no long leg at portfolio scale) and the
vendor `moneyflow` series was absorbed into a price/volume reference score
years ago. What has never been asked is the question this arm asks: does the
SIGN, its persistence, and the DISAGREEMENT between two ways of measuring it
add anything to the carrier that already reads the price and the size?

Four groups, fourteen columns, every one scale-free before it leaves this
module so a cross-sectional z-score over the bar's constituents is meaningful:

    level          ofi_1, ofi_s, ofi_l, ofi_z, ofia_s      where the sign sits
    quality        ofi_std, zero_l                          how trustworthy it is
    money          mf_1, mf_s, mf_l, lg_share_l, elg_net_l  the vendor's sign
    disagreement   disagree_l, agree_s                      where they differ

Masking, not dropping. A stock-day whose session produced fewer than
`knobs.FLOW_MIN_NRET` signed minutes, or whose board sealed, is set to NaN
before any rolling window: its imbalance is a queue artefact. NaN is the
honest value -- the rolling windows skip it, and `data.py` fills what survives
to the feature matrix's neutral 0 only at the very end, in the same
cross-section as every Alpha158 column.

Units, once, here (`pit-field-map.md` is the authority): `ofi_1d` and
`ofi_amt_1d` are dimensionless ratios in [-1, 1]; `zero_share` is a
dimensionless ratio; every `moneyflow` amount is in UNITS OF 10,000 CNY while
`daily.amount` is in CNY, so the vendor amounts are multiplied by 1e4 before
any ratio against turnover is taken. Mixing those two scales is a silent
factor of ten thousand that no assertion in the replay would catch.
"""

import numpy as np
import pandas as pd

from lib import events, knobs

FLOW_COLUMNS = ["ofi_1d", "ofi_amt_1d", "nret", "zero_share", "sealed_limit"]
MONEYFLOW_COLUMNS = ["net_mf_amount", "buy_lg_amount", "sell_lg_amount",
                     "buy_elg_amount", "sell_elg_amount", "buy_md_amount", "sell_md_amount",
                     "buy_sm_amount", "sell_sm_amount"]
VENDOR_AMOUNT_SCALE = 1e4          # moneyflow amounts are 10k CNY; daily.amount is CNY

BLOCK_NAMES = ("ofi_1", "ofi_s", "ofi_l", "ofi_z", "ofia_s",
               "ofi_std", "zero_l",
               "mf_1", "mf_s", "mf_l", "lg_share_l", "elg_net_l",
               "disagree_l", "agree_s")


def _roll(matrix, window, how, min_periods):
    rolling = pd.DataFrame(matrix.T).rolling(window, min_periods=min_periods)
    return getattr(rolling, how)().to_numpy().T


def _bar_rank(matrix, member):
    """(names, dates) percentile rank inside each bar's constituent set; NaN elsewhere."""

    masked = np.where(member, matrix, np.nan)
    return pd.DataFrame(masked).rank(axis=0, pct=True).to_numpy()


def _ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / np.where(np.abs(denominator) > 0.0, denominator, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def build(context, panel):
    """(names, dates, len(BLOCK_NAMES)) raw block values aligned to the panel axes."""

    dates, codes = panel["dates"], [str(code) for code in panel["symbols"]]
    window = knobs.FLOW_LOOKBACK_DAYS + int(len(dates) * 1.6)
    flow = events.read(context, "intraday_flow", FLOW_COLUMNS, window, codes)
    money = events.read(context, "moneyflow", MONEYFLOW_COLUMNS, window, codes)

    ofi = events.dense(flow, "ofi_1d", dates, codes)
    ofia = events.dense(flow, "ofi_amt_1d", dates, codes)
    nret = events.dense(flow, "nret", dates, codes)
    zero = events.dense(flow, "zero_share", dates, codes)
    sealed = events.dense(flow, "sealed_limit", dates, codes)
    usable = (nret >= knobs.FLOW_MIN_NRET)
    if knobs.FLOW_DROP_SEALED:
        usable &= ~(sealed > 0)
    ofi = np.where(usable, ofi, np.nan)
    ofia = np.where(usable, ofia, np.nan)

    amount = panel["raw_amount"]
    net = events.dense(money, "net_mf_amount", dates, codes) * VENDOR_AMOUNT_SCALE
    big = sum(events.dense(money, name, dates, codes) for name in
              ("buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount")) * VENDOR_AMOUNT_SCALE
    gross = big + sum(events.dense(money, name, dates, codes) for name in
                      ("buy_md_amount", "sell_md_amount", "buy_sm_amount",
                       "sell_sm_amount")) * VENDOR_AMOUNT_SCALE
    elg_net = (events.dense(money, "buy_elg_amount", dates, codes)
               - events.dense(money, "sell_elg_amount", dates, codes)) * VENDOR_AMOUNT_SCALE

    short, long = knobs.FLOW_SHORT, knobs.FLOW_LONG
    half = max(2, long // 2)
    ofi_s = _roll(ofi, short, "mean", max(2, short // 2))
    ofi_l = _roll(ofi, long, "mean", half)
    ofi_std = _roll(ofi, long, "std", half)
    mf_1 = _ratio(net, amount)
    mf_s = _ratio(_roll(net, short, "sum", max(2, short // 2)),
                  _roll(amount, short, "sum", max(2, short // 2)))
    mf_l = _ratio(_roll(net, long, "sum", half), _roll(amount, long, "sum", half))
    agree = np.where(np.isfinite(ofi) & np.isfinite(net),
                     (np.sign(ofi) == np.sign(net)).astype(np.float64), np.nan)

    member = panel["member"]
    block = {
        "ofi_1": ofi,
        "ofi_s": ofi_s,
        "ofi_l": ofi_l,
        "ofi_z": _ratio(ofi - ofi_l, ofi_std),
        "ofia_s": _roll(ofia, short, "mean", max(2, short // 2)),
        "ofi_std": ofi_std,
        "zero_l": _roll(zero, long, "mean", half),
        "mf_1": mf_1,
        "mf_s": mf_s,
        "mf_l": mf_l,
        "lg_share_l": _ratio(_roll(big, long, "sum", half), _roll(gross, long, "sum", half)),
        "elg_net_l": _ratio(_roll(elg_net, long, "sum", half), _roll(amount, long, "sum", half)),
        "disagree_l": _bar_rank(ofi_l, member) - _bar_rank(mf_l, member),
        "agree_s": _roll(agree, long, "mean", half),
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
