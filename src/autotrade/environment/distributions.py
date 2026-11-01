"""The ex-date distribution passed from market data to account settlement."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from autotrade.environment.dividend_tax import dividend_tax_rule


@dataclass(frozen=True)
class Distribution:
    cash_per_share: float = 0.0
    # None means an older table cannot separate 送股 from 转增.
    bonus_per_share: float | None = 0.0
    record_date: str | None = None


def distribution_record_date(dates: Iterable[str]) -> str:
    """Validate a taxable group before summing it on one record-date share base.

    Even dates in the same tax regime cannot silently be collapsed: they may
    describe different entitled holdings. Untaxable conversions need no date.
    """

    values = set(dates)
    if len(values) != 1:
        raise ValueError(f"conflicting dividend record_date values: {sorted(values)}")
    value = next(iter(values))
    dividend_tax_rule(value)
    return value
