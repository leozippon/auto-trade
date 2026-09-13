"""The human-readable order sheet of one Paper session.

Rendered from the book's own state and order journal, so the sheet, the
console and the account can never disagree. Reference prices are the previous
close; the book fills at the session's open, so notionals are estimates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path

from .book import Book
from .engine import PAPER_STATE_NAME, REFERENCE_KEY
from .storage import read_json, read_jsonl

WEEKDAYS = "一二三四五六日"
LATEST_NAME = "latest_orders.md"
DISCLAIMER = (
    "参考价为前一交易日收盘价；账簿按当日开盘价撮合，涨跌停、停牌或资金不足时订单会被拒，"
    "金额与现金均为估算，未计佣金、印花税与滑点。"
)


def orders_file_name(trade_date: str) -> str:
    return f"{trade_date}_orders.md"


def render_orders(book: Book, trade_date: str) -> str:
    state = read_json(book.root / PAPER_STATE_NAME)
    decision = next(
        (row for row in state.get("decisions") or () if row.get("trade_date") == trade_date), None
    )
    if decision is None:
        raise ValueError(f"the book has no decision for {trade_date}")
    rows, skipped = read_jsonl(book.root / f"orders_{trade_date}.jsonl")
    held = {str(symbol): int(quantity) for symbol, quantity in dict(decision["positions"]).items()}
    # Everything below is the decision's own record, so an old sheet renders
    # exactly as it did on its morning.
    quotes = {
        **{str(row["symbol"]): dict(row.get(REFERENCE_KEY) or {}) for row in rows},
        **dict(decision.get("quotes") or {}),
    }
    cash = float(decision["cash"])
    equity = float(decision["equity"])

    def name(symbol: str) -> str:
        return str(quotes.get(symbol, {}).get("name") or "")

    def price(symbol: str) -> float | None:
        value = quotes.get(symbol, {}).get("close")
        return float(value) if isinstance(value, (int, float)) else None

    lines = [f"# Paper 订单 · {_day(trade_date)}", ""]
    if book.note:
        lines += [f"> {book.note}", ""]
    lines += [
        (
            f"- 账簿：{book.experiment_id} / {book.artifact_id}（{book.candidate_source}），"
            f"{_day(str(state.get('start_date') or trade_date))} 起，初始资金 {_money(book.profile.initial_cash)}"
        ),
        (
            f"- 决策：{_day(trade_date)} {book.schedule.inference_time}，数据截至 {_day(str(decision['data_through']))} 收盘"
            f"（发布 {str(decision['generation_id'])[:12]}）"
            + _fit_note(decision, str(state.get("last_fit_date") or ""))
        ),
        f"- 重放此前决策 {decision['replayed_calls']} 次，其中 {decision['replayed_matching_journal']} 次与账簿记录的订单一致",
        (
            f"- 账户（{_day(str(state.get('settled_through') or decision['data_through']))} 收盘估值）："
            f"总资产 {_money(equity)}，现金 {_money(cash)}，持仓 {len(held)} 只"
        ),
        "",
    ]
    if rows:
        lines += [
            f"## 订单（{len(rows)} 笔）",
            "",
            "| 时间 | 代码 | 名称 | 方向 | 股数 | 参考价 | 约计金额 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        flows = {"buy": 0.0, "sell": 0.0}
        for row in rows:
            symbol, quantity, action = str(row["symbol"]), int(row["quantity"]), str(row["action"])
            quote = price(symbol)
            notional = quote * quantity if quote is not None else None
            if notional is not None:
                flows[action] += notional
            lines.append(
                f"| {datetime.fromisoformat(str(row['execute_at'])).strftime('%H:%M')} | {symbol} | {name(symbol)} "
                f"| {'买入' if action == 'buy' else '卖出'} | {quantity:,} | {_price(quote)} | {_money(notional)} |"
            )
        lines += ["", f"卖出合计约 {_money(flows['sell'])}，买入合计约 {_money(flows['buy'])}。", ""]
        if skipped:
            lines += [f"订单日志有 {skipped} 行无法解析，已跳过；请检查 orders_{trade_date}.jsonl。", ""]
    else:
        lines += ["## 订单", "", "今日无订单，持仓不变。", ""]
        flows = {"buy": 0.0, "sell": 0.0}
    target = _positions_after(held, rows)
    lines += [
        f"## 成交后目标持仓（{len(target)} 只）",
        "",
        "| 代码 | 名称 | 股数 | 参考价 | 参考市值 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    value = 0.0
    for symbol, quantity in target.items():
        quote = price(symbol)
        value += quote * quantity if quote is not None else 0.0
        lines.append(
            f"| {symbol} | {name(symbol)} | {quantity:,} | {_price(quote)} "
            f"| {_money(quote * quantity if quote is not None else None)} |"
        )
    remaining = cash + flows["sell"] - flows["buy"]
    lines += ["", f"成交后现金约 {_money(remaining)}，参考市值合计 {_money(value)}。", ""]
    for warning in state.get("warnings") or ():
        lines += [f"注意：{warning}", ""]
    lines += [DISCLAIMER, ""]
    return "\n".join(lines)


def render_failure(book: Book, trade_date: str, error: BaseException) -> str:
    message = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    lines = [f"# Paper 订单 · {_day(trade_date)} · 未生成", ""]
    if book.note:
        lines += [f"> {book.note}", ""]
    lines += [
        f"本次运行失败，{_day(trade_date)} 没有订单。原因：{type(error).__name__}: {message}",
        "",
        (
            "修复原因后重跑同一命令（已完成的结算不会重复）："
            f"`python scripts/paper/run_paper.py run --trade-date {trade_date}`。不要手工修改账簿文件。"
        ),
        "",
    ]
    return "\n".join(lines)


def write_orders(orders_dir: str | Path, trade_date: str, text: str) -> Path:
    """Write the dated sheet and refresh ``latest_orders.md`` to the same text."""

    directory = Path(orders_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / orders_file_name(trade_date)
    for path in (target, directory / LATEST_NAME):
        staging = path.with_name(f".{path.name}.tmp")
        staging.write_text(text, encoding="utf-8")
        staging.replace(path)
    return target


def _positions_after(positions: Mapping[str, int], orders: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Holdings once every order fills in full."""

    result = dict(positions)
    for order in orders:
        symbol, quantity = str(order["symbol"]), int(order["quantity"])
        result[symbol] = result.get(symbol, 0) + (quantity if order["action"] == "buy" else -quantity)
    return {symbol: quantity for symbol, quantity in sorted(result.items()) if quantity > 0}


def _fit_note(decision: Mapping[str, object], last_fit_date: str) -> str:
    if decision.get("fitted"):
        return "，本次重新拟合"
    return f"，沿用 {_day(last_fit_date)} 的拟合" if last_fit_date else ""


def _day(value: str) -> str:
    day = date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    return f"{day.isoformat()}（周{WEEKDAYS[day.weekday()]}）"


def _money(value: float | None) -> str:
    return "—" if value is None else f"¥{value:,.2f}"


def _price(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


__all__ = ["LATEST_NAME", "orders_file_name", "render_failure", "render_orders", "write_orders"]
