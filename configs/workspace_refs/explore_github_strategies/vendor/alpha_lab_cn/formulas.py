# SPDX-License-Identifier: MIT
# Source and copyright notice: SOURCE.md

"""Reference-only portable adaptations of selected Alpha Lab CN formulas.

Do not import this file from the formal strategy. Rewrite the selected formula
inside output/main.py after checking the current data and unit contracts.
"""

# Reference snippets run in the sandbox/quant environment; the repository LSP
# does not resolve that environment's scientific packages.
# pyright: reportMissingImports=false

import numpy as np
import pandas as pd


def information_weighted_reversal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """Negative rolling return weighted by each day's share of window volume."""
    close, volume = close.align(volume, join="inner", axis=None)
    daily_return = close.pct_change(fill_method=None)
    volume_sum = volume.rolling(lookback, min_periods=max(1, lookback - 1)).sum()
    volume_weight = (volume / volume_sum.replace(0, np.nan)).clip(upper=0.5)
    weighted_return = daily_return * volume_weight.fillna(1.0 / lookback)
    return -weighted_return.rolling(lookback, min_periods=max(1, lookback - 1)).sum()


def positive_ep(pe_ttm: pd.Series) -> pd.Series:
    """Earnings-to-price from positive trailing PE only."""
    pe = pd.to_numeric(pe_ttm, errors="coerce")
    return (1.0 / pe.replace(0, np.nan)).where(pe > 0)


def low_turnover(turnover_rate: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Negative trailing mean turnover; larger output means calmer turnover."""
    return -turnover_rate.rolling(lookback, min_periods=lookback // 2).mean()
