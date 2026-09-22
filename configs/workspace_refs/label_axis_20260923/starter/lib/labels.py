"""The label family this arm exists to test, and the one function that builds any member of it.

Every learned arm in this repository has ranked the same thing: a 5- or
10-trading-day forward OPEN-to-OPEN return, latterly residualised against the
benchmark. No arm's closure reason has ever named the label, and no arm ever
reached a label-axis change, so the label is measured-but-untested as a lever.
This module makes it a lever: one registered family, four components, and one
member selected by `main.CANDIDATE`.

    horizon         N forward trading days, N in HORIZONS
    timing          which price pair the N days are measured between
    residual        what is subtracted before the cross-section is ranked
    scaling         rank, or rank of the return divided by its own trailing
                    volatility

Timing is the component with a factual answer rather than a taste, and the
incumbent gets it wrong by one session. Bar t is the newest bar visible at the
08:30 decision of the NEXT trading day D, and this book executes its sells at
D's 09:30 open and its buys at D's 15:00 close (`lib/trade.py`). So a name
entered on review day D is bought at C[D] and sold, on the review day N trading
days later, at O[D+N]:

    "co"  exit O[t+1+N] over entry C[t+1]      what this book actually earns
    "oo"  exit O[t+1+N] over entry O[t+1]      the incumbent: entry one session early
    "cc"  exit C[t+1+N] over entry C[t+1]      exit one session late

`oo` therefore hands the model the decision day's own open-to-close move as if
the book had captured it, when the book was still flat for all of it; `cc`
hands it the exit day's open-to-close move, which the book has already left.
Neither is fatal -- the error is one session out of N -- but it is an error, it
is the same sign for every name only on average, and it has never been
measured. `c_base` keeps `oo` because the round's control must be the incumbent
label unchanged; `l_exec` is the registered variant that fixes it.

The benchmark leg of a residual uses the SAME two timestamps as the stock leg,
so a timing change never silently compares a stock's close-to-open against a
benchmark's open-to-open.

Residualisation. `beta` subtracts the name's trailing OLS beta times the
benchmark's move over those same two timestamps -- trailing, so nothing after
bar t enters it, and shrunk toward 1 because an unshrunk beta on large caps
carries enough estimation noise to re-import the exposure it is meant to
remove. `ind` subtracts the cross-sectional mean of that bar's own SW level-1
industry, over that bar's constituents only. `both` does beta first, then
industry. `none` is the raw return, which is what the graduated lineage ranked.

Scaling. `rank` is the incumbent: the cross-sectional rank over that bar's
constituents, mapped to (rank - 1) / n - 0.5. `volrank` divides the residual by
the name's own trailing VOL_DAYS realised volatility before ranking, so the
target is a risk-adjusted move rather than a move; a name whose volatility is
missing or zero drops out of that bar instead of being given a default.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

BETA_DAYS = 120
BETA_MIN_DAYS = 60
BETA_SHRINK = 0.7
BETA_CLIP = (0.3, 2.0)
VOL_DAYS = 20
VOL_MIN_DAYS = 10

HORIZONS = (10, 20, 40)
TIMINGS = ("oo", "cc", "co")
RESIDUALS = ("none", "beta", "ind", "both")
SCALINGS = ("rank", "volrank")


@dataclass(frozen=True)
class Label:
    """One member of the family. Every field is registered before the first batch."""

    horizon: int
    timing: str
    residual: str
    scaling: str

    def __post_init__(self):
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon {self.horizon} is outside the registered family {HORIZONS}")
        if self.timing not in TIMINGS:
            raise ValueError(f"timing {self.timing} is outside the registered family {TIMINGS}")
        if self.residual not in RESIDUALS:
            raise ValueError(f"residual {self.residual} is outside the registered family {RESIDUALS}")
        if self.scaling not in SCALINGS:
            raise ValueError(f"scaling {self.scaling} is outside the registered family {SCALINGS}")


# The whole registered family, one line each. `c_base` is the round's default
# label and the arm's control; every other member moves EXACTLY ONE component
# away from it, which is what makes an offline screen readable as an ablation.
FAMILY = {
    "c_base": Label(10, "oo", "beta", "rank"),     # the control: the 20260922 default, unchanged
    "l_h20":  Label(20, "oo", "beta", "rank"),     # horizon: matches the monthly review cadence
    "l_h40":  Label(40, "oo", "beta", "rank"),     # horizon: two review intervals
    "l_exec": Label(10, "co", "beta", "rank"),     # timing: what this book actually earns
    "l_cc":   Label(10, "cc", "beta", "rank"),     # timing: close to close
    "l_ind":  Label(10, "oo", "both", "rank"),     # residual: beta, then industry-demeaned
    "l_vol":  Label(10, "oo", "beta", "volrank"),  # scaling: risk-adjusted move
}


def spec(candidate):
    """The registered label of `candidate`; unknown names fail instead of falling back."""

    try:
        return FAMILY[candidate]
    except KeyError:
        raise RuntimeError(
            f"'{candidate}' is not a registered label; families.md lists {sorted(FAMILY)}") from None


def _lead(values, offset):
    """values[..., t + offset] written at column t; NaN where the column runs off the panel."""

    out = np.full_like(values, np.nan)
    total = values.shape[-1]
    if total > offset:
        out[..., : total - offset] = values[..., offset:]
    return out


def _legs(panel, label):
    """(entry, exit) price matrices and the benchmark's own pair, aligned to the signal bar.

    Column t of every returned array belongs to the signal formed on bar t,
    which is acted on at the decision of the next trading day.
    """

    stock_entry = panel["C"] if label.timing in ("cc", "co") else panel["O"]
    stock_exit = panel["C"] if label.timing == "cc" else panel["O"]
    bench_entry = panel["bench_close"] if label.timing in ("cc", "co") else panel["bench_open"]
    bench_exit = panel["bench_close"] if label.timing == "cc" else panel["bench_open"]
    offset = 1 + label.horizon
    return (_lead(stock_entry, 1), _lead(stock_exit, offset),
            _lead(bench_entry, 1), _lead(bench_exit, offset))


def trailing_beta(panel):
    """(names, dates) trailing beta against the benchmark, shrunk toward 1 and clipped."""

    closes = pd.DataFrame(panel["C"].T, index=pd.Index(panel["dates"]))
    bench = pd.Series(panel["bench_close"], index=pd.Index(panel["dates"]))
    if bench.notna().sum() < BETA_MIN_DAYS + 1:
        raise RuntimeError("the benchmark series does not cover the panel window")
    stock_return = closes.pct_change(fill_method=None)
    bench_return = bench.pct_change(fill_method=None)
    covariance = stock_return.rolling(BETA_DAYS, min_periods=BETA_MIN_DAYS).cov(bench_return)
    variance = bench_return.rolling(BETA_DAYS, min_periods=BETA_MIN_DAYS).var()
    raw = covariance.div(variance.replace(0.0, np.nan), axis=0)
    beta = BETA_SHRINK * raw + (1.0 - BETA_SHRINK)
    return beta.clip(*BETA_CLIP).fillna(1.0).to_numpy().T


def trailing_vol(panel):
    """(names, dates) trailing realised volatility of daily adjusted-close returns."""

    closes = pd.DataFrame(panel["C"].T)
    returns = closes.pct_change(fill_method=None)
    vol = returns.rolling(VOL_DAYS, min_periods=VOL_MIN_DAYS).std().to_numpy().T
    return np.where(np.isfinite(vol) & (vol > 0.0), vol, np.nan)


def _industry_demean(values, panel):
    """Subtract each bar's own SW level-1 industry mean, over that bar's constituents only."""

    frame = pd.DataFrame(np.where(panel["member"], values, np.nan))
    groups = pd.Series(panel["industry"], index=frame.index)
    return (frame - frame.groupby(groups).transform("mean")).to_numpy()


def residual(panel, label):
    """(names, dates) the label's residual forward return; NaN off that bar's constituents."""

    entry, exit_, bench_entry, bench_exit = _legs(panel, label)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = exit_ / entry - 1.0
        move = bench_exit / bench_entry - 1.0
    if label.residual in ("beta", "both"):
        out = out - trailing_beta(panel) * move[None, :]
    if label.residual in ("ind", "both"):
        out = _industry_demean(out, panel)
    if label.scaling == "volrank":
        out = out / trailing_vol(panel)
    return np.where(panel["member"] & np.isfinite(out), out, np.nan)


def rank_target(values):
    """(names, dates) cross-sectional rank of a residual in [-0.5, 0.5], and where it exists."""

    frame = pd.DataFrame(values)
    ranks = frame.rank(axis=0, pct=True).to_numpy()
    count = np.isfinite(values).sum(axis=0, keepdims=True)
    target = ranks - 1.0 / np.maximum(count, 1) - 0.5
    realized = np.isfinite(values)
    return np.where(realized, target, np.nan), realized


def target(panel, label):
    """(target, realised) for one registered label: the single entry point `main.fit` calls."""

    return rank_target(residual(panel, label))
