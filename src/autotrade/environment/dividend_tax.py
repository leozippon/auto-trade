"""Individual A-share dividend tax under 财税〔2012〕85号 /〔2015〕101号.

The record date selects the rule permanently; the acquisition-to-sale holding
period selects its final rate. Settlement stays with the Broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

BONUS_SHARE_PAR_CNY = 1.0
DIVIDEND_TAX_CUTOVER = "20150908"


@dataclass(frozen=True)
class DividendTaxRule:
    key: str
    withholding_rate: float
    tiers: tuple[tuple[int, float], ...]
    rate_beyond: float


_HOLDING_TIERS = ((1, 0.20), (12, 0.10))
OLD_RULE = DividendTaxRule("2012_85", 0.05, _HOLDING_TIERS, 0.05)
NEW_RULE = DividendTaxRule("2015_101", 0.0, _HOLDING_TIERS, 0.0)


def dividend_tax_rule(record_date: str) -> DividendTaxRule:
    """Refuse unknown/unsupported dates rather than infer from the ex/sale day."""

    _day(record_date)
    if record_date < "20130101":
        raise ValueError(f"dividend record_date {record_date} predates the supported 2013 tax rules")
    return OLD_RULE if record_date < DIVIDEND_TAX_CUTOVER else NEW_RULE


def dividend_tax_rate(acquired: str, sold: str, rule: DividendTaxRule) -> float:
    """Final rate, holding through the day before sale in natural months/years."""

    start, sale = _day(acquired), _day(sold)
    if sale <= start:
        raise ValueError(f"shares acquired on {acquired} cannot be sold on {sold}")
    for months, rate in rule.tiers:
        if sale <= _months_later(start, months):
            return rate
    return rule.rate_beyond


def dividend_tax_due(acquired: str, sold: str, *, old_income: float, new_income: float) -> float:
    """Sale-time supplement only: old income has already paid its 5% in cash."""

    return sum(
        income * (dividend_tax_rate(acquired, sold, rule) - rule.withholding_rate)
        for rule, income in ((OLD_RULE, old_income), (NEW_RULE, new_income))
        if income
    )


def _day(text: str) -> date:
    if not isinstance(text, str) or len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid dividend tax date: {text!r}; expected YYYYMMDD record_date/trade date")
    return date.fromisoformat(text)


def _months_later(day: date, months: int) -> date:
    """Same day months later, or first of the next month if that day is absent."""

    year, month = divmod(day.year * 12 + day.month - 1 + months, 12)
    try:
        return day.replace(year=year, month=month + 1)
    except ValueError:
        return date(year + (month + 1) // 12, (month + 1) % 12 + 1, 1)
