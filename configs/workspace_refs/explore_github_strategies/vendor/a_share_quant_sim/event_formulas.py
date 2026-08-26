# SPDX-License-Identifier: MIT
# Source and copyright notice: SOURCE.md

"""Reference-only event arithmetic adapted from A-share Quant Sim.

The source's SQLite and wall-clock calls are deliberately removed. These
helpers accept explicit PIT-filtered tables and an explicit inference time.
Do not import this file from output/main.py.
"""

# Reference snippets run in the sandbox/quant environment; the repository LSP
# does not resolve that environment's scientific packages.
# pyright: reportMissingImports=false

import numpy as np
import pandas as pd


def earnings_decay(events: pd.DataFrame, inference_at) -> pd.Series:
    """Latest visible positive event decays for 20d; negative for 10d."""
    if events.empty:
        return pd.Series(dtype=float)
    now = pd.Timestamp(inference_at)
    frame = events.copy()
    available = pd.to_datetime(frame["available_at"], errors="coerce")
    if available.dt.tz is None and now.tzinfo is not None:
        available = available.dt.tz_localize(now.tzinfo)
    frame = frame[available.notna() & (available <= now)].assign(_available=available)
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame.sort_values("_available").groupby("ts_code", as_index=False).tail(1)
    age = (now.normalize() - frame["_available"].dt.normalize()).dt.days
    positive = frame["is_positive"].fillna(False).astype(bool)
    negative = frame["is_negative"].fillna(False).astype(bool)
    score = pd.Series(0.0, index=frame.index)
    score.loc[positive & age.between(0, 20)] = (
        1.0 - 0.0475 * age.loc[positive & age.between(0, 20)]
    ).clip(lower=0.05)
    score.loc[negative & age.between(0, 10)] = (
        -1.0 + 0.09 * age.loc[negative & age.between(0, 10)]
    ).clip(upper=-0.1)
    return pd.Series(score.to_numpy(), index=frame["ts_code"], name="earnings_decay")


def holder_accumulation(holder_events: pd.DataFrame, inference_at) -> pd.Series:
    """Time-decayed signed holder change ratio over the prior 90 days."""
    if holder_events.empty:
        return pd.Series(dtype=float)
    now = pd.Timestamp(inference_at)
    frame = holder_events.copy()
    available = pd.to_datetime(frame["available_at"], errors="coerce")
    if available.dt.tz is None and now.tzinfo is not None:
        available = available.dt.tz_localize(now.tzinfo)
    age = (now.normalize() - available.dt.normalize()).dt.days
    frame = frame[available.notna() & (available <= now) & age.between(0, 90)].copy()
    age = age.loc[frame.index]
    if frame.empty:
        return pd.Series(dtype=float)
    direction = frame["in_de"].astype(str).str.upper().map({"IN": 1.0, "DE": -1.0})
    signed_ratio = (
        pd.to_numeric(frame["change_ratio"], errors="coerce").abs() * direction
    )
    weight = np.select([age <= 30, age <= 60], [1.0, 0.5], default=0.25)
    return (signed_ratio * weight).groupby(frame["ts_code"]).sum(min_count=1)


def announced_unlock_pressure(
    unlock_events: pd.DataFrame, inference_at, days: int = 30
) -> pd.Series:
    """Maximum already-visible unlock ratio scheduled in the next N days."""
    if unlock_events.empty:
        return pd.Series(dtype=float)
    now = pd.Timestamp(inference_at)
    frame = unlock_events.copy()
    available = pd.to_datetime(frame["available_at"], errors="coerce")
    if available.dt.tz is None and now.tzinfo is not None:
        available = available.dt.tz_localize(now.tzinfo)
    unlock_date = pd.to_datetime(frame["float_date"], errors="coerce")
    horizon = (
        now.tz_localize(None).normalize() + pd.Timedelta(days=days)
        if now.tzinfo
        else now.normalize() + pd.Timedelta(days=days)
    )
    start = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    frame = frame[
        available.notna()
        & (available <= now)
        & unlock_date.notna()
        & unlock_date.between(start, horizon)
    ].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    ratio = pd.to_numeric(frame["float_ratio"], errors="coerce")
    return ratio.groupby(frame["ts_code"]).max()
