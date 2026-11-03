"""Small, deterministic board and lot rules for daily stock execution.

Pure stdlib (no ``autotrade``/``pandas`` import), but host-only: this module is NOT
shipped into the Agent sandbox image. Its consumers are the authoritative host
:class:`~autotrade.environment.broker.DailyBroker`, which sizes every order's lots
from the functions here and prices it through its
:class:`~autotrade.environment.broker.BrokerProfile` (commission, stamp duty from
``STAMP_DUTY_CUTOVER``, slippage), and the screens that read a name's board. Bar-level
gates (suspension, price limits, T+1 sellable) and position bookkeeping stay with
the broker, which holds the market data and position state.
"""

from __future__ import annotations

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
