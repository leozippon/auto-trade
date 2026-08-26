# SPDX-License-Identifier: MIT
# Source and copyright notice: SOURCE.md

"""WorldQuant Alpha101 operator controls, not primary strategy candidates.

The previous GitHub-reference experiment already used #6/#12/#101. These
functions are useful for checking wide-panel operators and signal direction.
Do not import this file from the formal strategy.
"""

# Reference snippets run in the sandbox/quant environment; the repository LSP
# does not resolve that environment's scientific packages.
# pyright: reportMissingImports=false

import numpy as np
import pandas as pd


def cross_sectional_rank(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(axis=1, pct=True)


def alpha006(open_price: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """-correlation(open, volume, 10)."""
    return -open_price.rolling(10).corr(volume)


def alpha012(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """sign(delta(volume, 1)) * -delta(close, 1)."""
    return np.sign(volume.diff()) * -close.diff()


def alpha033(open_price: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """rank(-(1 - open / close))."""
    raw = -(1.0 - open_price / close.replace(0, np.nan))
    return cross_sectional_rank(raw)


def alpha101(
    open_price: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """(close - open) / (high - low + 0.001)."""
    return (close - open_price) / (high - low + 0.001)
