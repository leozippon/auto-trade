"""Small, deterministic cost and fill helpers for daily stock execution.

Pure stdlib (no ``autotrade``/``pandas`` import), but host-only: this module is NOT
shipped into the Agent sandbox image. Its single consumer is the authoritative host
:class:`~autotrade.environment.broker.DailyBroker`, which projects every order's
money/share outcome from the functions here (commission, stamp duty, slippage, lot
sizing). Only this deterministic math lives here; bar-level gates (suspension, price
limits, T+1 sellable) and position bookkeeping stay with the broker, which holds the
market data and position state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LOT_SIZE = 100
STAR_MIN_LOT_SIZE = 200
STAMP_DUTY_CUTOVER = "20230828"  # sell-side stamp duty halved to 0.05% from this date
# The board a name is listed on, told by its code: the one vocabulary of the
# research universe's board screen (``SnapshotConfig.screen_boards``) and of the
# boards an account may buy (``BrokerProfile.permitted_boards``). A Beijing
# name is told by its ``.BJ`` suffix instead of a prefix.
BOARD_PREFIXES: dict[str, tuple[str, ...]] = {
    "main": ("600", "601", "603", "605", "000", "001", "002", "003"),
    "gem": ("300", "301", "302"),
    "star": ("688", "689"),
    "bj": (),
}
BOARDS: tuple[str, ...] = tuple(BOARD_PREFIXES)


def board_of(symbol: str) -> str | None:
    """The board ``symbol`` is listed on, or None for a code on none of them."""

    code = str(symbol).upper()
    if code.endswith(".BJ"):
        return "bj"
    return next((board for board, prefixes in BOARD_PREFIXES.items() if code.startswith(prefixes)), None)


def is_star_market(symbol: str) -> bool:
    code = str(symbol).upper()
    return code.endswith(".SH") and code[:3] in {"688", "689"}


def is_bse_market(symbol: str) -> bool:
    return str(symbol).upper().endswith(".BJ")


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    min_commission_cny: float = 5.0
    stamp_duty_sell_bps_before_cutover: float = 10.0
    stamp_duty_sell_bps_from_cutover: float = 5.0
    transfer_fee_bps: float = 0.1  # 过户费 0.01‰ = 0.1 bps, both buy and sell side.
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "commission_bps",
            "min_commission_cny",
            "stamp_duty_sell_bps_before_cutover",
            "stamp_duty_sell_bps_from_cutover",
            "transfer_fee_bps",
            "slippage_bps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")

    def fill_price(self, price: float, *, action: str) -> float:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("market price must be a positive finite number")
        direction = 1.0 if action == "buy" else -1.0
        return float(price) * (1.0 + direction * self.slippage_bps / 10_000.0)

    def commission(self, notional: float) -> float:
        return max(notional * self.commission_bps / 10_000.0, self.min_commission_cny)

    def transfer_fee(self, notional: float) -> float:
        return notional * self.transfer_fee_bps / 10_000.0

    def trade_fee(self, notional: float) -> float:
        return self.commission(notional) + self.transfer_fee(notional)

    def stamp_duty_on_sale(self, notional: float, trade_date: str) -> float:
        bps = (
            self.stamp_duty_sell_bps_from_cutover
            if str(trade_date) >= STAMP_DUTY_CUTOVER
            else self.stamp_duty_sell_bps_before_cutover
        )
        return notional * bps / 10_000.0

    def fees(self, notional: float, *, action: str, trade_date: str) -> tuple[float, float]:
        """Fee and stamp duty for one fill.

        The first element is the whole trade fee (佣金 plus 过户费), matching how
        the fill's cash movement is settled; ``stamp_duty`` is sell-side only and
        follows the 2023-08-28 cutover.
        """
        if not math.isfinite(notional) or notional <= 0:
            raise ValueError("notional must be a positive finite number")
        stamp_duty = self.stamp_duty_on_sale(notional, trade_date) if action == "sell" else 0.0
        return self.trade_fee(notional), stamp_duty


def validate_buy_lot(quantity: int, symbol: str = "") -> None:
    """Board-aware buy declaration ladder.

    STAR (688/689.SH) declares at least 200 shares then 1-share increments; the
    BSE declares at least 100 shares then 1-share increments; every other board
    declares whole 100-share lots.
    """
    if is_star_market(symbol):
        if quantity < STAR_MIN_LOT_SIZE:
            raise ValueError(f"buy quantity must be at least {STAR_MIN_LOT_SIZE} shares")
        return
    if is_bse_market(symbol):
        if quantity < LOT_SIZE:
            raise ValueError(f"buy quantity must be at least {LOT_SIZE} shares")
        return
    if quantity % LOT_SIZE:
        raise ValueError(f"buy quantity must be a multiple of {LOT_SIZE}")


def reduce_amount_reject(shares: int, sellable: int, symbol: str) -> str | None:
    """Sell-side lot rule for a positive ``shares`` request against ``sellable``.

    Exchange rules let a holder declare whole lots, or one declaration that carries
    the ENTIRE sub-lot odd tail (零股必须一次性申报卖出) — corporate actions (送转)
    legitimately create odd positions, so reduces cannot reuse the strict buy
    ladder. STAR/BSE positions below their minimum declaration are likewise
    exitable only in full."""
    if is_star_market(symbol):
        return None if shares >= STAR_MIN_LOT_SIZE or shares == sellable else "amount_below_lot_size"
    if is_bse_market(symbol):
        return None if shares >= LOT_SIZE or shares == sellable else "amount_below_lot_size"
    if shares % LOT_SIZE == 0:
        return None
    odd = sellable % LOT_SIZE
    if odd and shares % LOT_SIZE == odd and shares <= sellable:
        return None
    return "amount_not_lot_aligned" if shares >= LOT_SIZE else "amount_below_lot_size"
