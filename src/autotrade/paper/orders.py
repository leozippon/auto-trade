"""The human-readable order sheet of one Paper session.

Rendered from the book's own state and order journal, so the sheet, the
console and the account can never disagree. Reference prices are the previous
close; each order fills at the price of the time it names (the session's open
or close), so notionals are estimates.

Every order names the window the operator places it in, read off its own
``execute_at`` exactly as the fill model reads it (``resolve_execution_price``):
09:30 fills at the open, which the 09:15-09:25 opening call auction sets; 15:00
fills at the close, which the 14:57-15:00 closing call auction sets; any other
minute is continuous trading at that minute. Nothing here assumes which side
uses which window: a strategy may sell at the close and buy at the next open.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path

from autotrade.environment.replay.engine import MARKET_CLOSE, MARKET_OPEN
from autotrade.environment.strategy import CN_TZ

from .book import Book
from .engine import PAPER_STATE_NAME, REFERENCE_KEY, PaperAwaitingFills
from .storage import read_json, read_jsonl

WEEKDAYS = "一二三四五六日"
LATEST_NAME = "latest_orders.md"
DISCLAIMER = (
    "参考价为前一交易日收盘价；模拟账户按每笔订单自己的时点撮合：开盘竞价的按当日开盘价，"
    "收盘竞价的按当日收盘价，均另计滑点；涨跌停、停牌或资金不足时订单会被拒。"
    "金额与现金均为按参考价的估算，未计佣金、印花税与滑点。"
)
OPEN_AUCTION = "open_auction"
CLOSE_AUCTION = "close_auction"
CONTINUOUS = "continuous"
# (name, declaration span, what the fill price is, how the window works). The
# one wording of every place the operator reads a window: the Markdown sheet,
# the console's sheet and the copied text, which take it from the payload.
_WINDOWS = {
    OPEN_AUCTION: (
        "开盘集合竞价", "09:15–09:25", "按开盘价成交",
        "09:15–09:20 可以申报和撤单，09:20–09:25 只能申报、不能撤单；09:25 按单一价格撮合出开盘价。",
    ),
    CLOSE_AUCTION: (
        "收盘集合竞价", "14:57–15:00", "按收盘价成交",
        "14:57–15:00 申报，申报后不能撤单；15:00 按单一价格撮合出收盘价。",
    ),
}
LIMIT_GUIDANCE = (
    "限价怎么填：集合竞价按单一价格撮合，买入限价不低于、卖出限价不高于撮合价的申报都按撮合价成交，"
    "成交价是撮合价而不是你填的限价。模拟账户按开盘价或收盘价成交（另计滑点），只在撮合价触及涨停（买入）"
    "或跌停（卖出）时拒单，所以要与模拟一致，买入限价可以填当日涨停价、卖出限价填当日跌停价。"
    "买入按申报价冻结资金：资金不够时，把买入限价降到当时价格上方几个百分点，但撮合价高于限价时不会成交，"
    "请如实记录为未成交。申报价不能超出当日涨跌停价，交易所与券商可能另有价格范围限制，以券商提示为准；"
    "收盘竞价下单前先看一眼 14:57 的最新价，参考价是前一交易日的收盘价。"
)


def orders_file_name(trade_date: str) -> str:
    return f"{trade_date}_orders.md"


def order_window(execute_at: datetime) -> str:
    """One order's declaration window, from the time it asks to fill at.

    The same comparison the fill model makes (``resolve_execution_price``):
    exactly 09:30 is the day's open, exactly 15:00 its close; any other minute
    is continuous trading at that minute.
    """

    local = execute_at.astimezone(CN_TZ).time()
    if local == MARKET_OPEN:
        return OPEN_AUCTION
    if local == MARKET_CLOSE:
        return CLOSE_AUCTION
    return CONTINUOUS


def window_text(execute_at: datetime, sheet_day: str) -> tuple[str, str, str]:
    """(short label, heading, how the window works) of one order's window, in
    plain words. An order dated for another session than the sheet's says which."""

    local = execute_at.astimezone(CN_TZ)
    clock = local.strftime("%H:%M")
    window = order_window(local)
    name, span, fill, how = _WINDOWS.get(window) or (
        "连续竞价", clock, "按该分钟的价格成交",
        f"在 {clock} 前后的连续竞价里下单；模拟账户按这一分钟的收盘价撮合。",
    )
    half = "上午" if local.hour < 12 else "下午"
    session = local.strftime("%Y%m%d")
    if session != sheet_day:
        half = f"{_day(session)}{half}"
    return f"{half} {name} {span}", f"{half} · {name} {span} 申报 · {fill}", how


def _order_groups(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Orders that share one window, in placement order: the batches an
    operator places at one sitting, morning and afternoon apart."""

    groups: list[dict[str, object]] = []
    for index, row in enumerate(orders):
        key = (row["session"], row["window"], row["time"])
        if not groups or groups[-1]["key"] != key:
            groups.append({
                "key": key, "session": row["session"], "window": row["window"], "time": row["time"],
                "title": row["window_title"], "how": row["window_how"], "orders": [],
                "flows": {"buy": 0.0, "sell": 0.0},
            })
        group = groups[-1]
        group["orders"].append(index)
        if row["notional"] is not None:
            group["flows"][row["action"]] += row["notional"]
    for group in groups:
        del group["key"]
    return groups


def order_sheet(
    root: str | Path, trade_date: str, *, state: Mapping[str, object] | None = None
) -> dict[str, object]:
    """One decision's order sheet as data: the decision record, its orders at
    their reference quotes, and the holdings and cash once every order fills.

    The one computation behind the Markdown sheet and the console's signal
    panels. Everything comes from the decision's own record and its order
    journal, so an old sheet reads exactly as it did on its morning. ``state``
    is the book's parsed state when the caller already holds it.
    """

    root = Path(root)
    state = read_json(root / PAPER_STATE_NAME) if state is None else state
    decision = next(
        (row for row in state.get("decisions") or () if row.get("trade_date") == trade_date), None
    )
    if decision is None:
        raise ValueError(f"the book has no decision for {trade_date}")
    rows, skipped = read_jsonl(root / f"orders_{trade_date}.jsonl")
    held = {str(symbol): int(quantity) for symbol, quantity in dict(decision["positions"]).items()}
    quotes = {
        **{str(row["symbol"]): dict(row.get(REFERENCE_KEY) or {}) for row in rows},
        **dict(decision.get("quotes") or {}),
    }

    def name(symbol: str) -> str:
        return str(quotes.get(symbol, {}).get("name") or "")

    def price(symbol: str) -> float | None:
        value = quotes.get(symbol, {}).get("close")
        return float(value) if isinstance(value, (int, float)) else None

    orders = []
    flows = {"buy": 0.0, "sell": 0.0}
    # Placement order: the strategy's own order within one execution time.
    for row in sorted(rows, key=lambda item: datetime.fromisoformat(str(item["execute_at"]))):
        symbol, quantity, action = str(row["symbol"]), int(row["quantity"]), str(row["action"])
        quote = price(symbol)
        notional = quote * quantity if quote is not None else None
        if notional is not None:
            flows[action] += notional
        execute_at = datetime.fromisoformat(str(row["execute_at"])).astimezone(CN_TZ)
        label, title, how = window_text(execute_at, trade_date)
        orders.append({
            "execute_at": str(row["execute_at"]), "session": execute_at.strftime("%Y%m%d"),
            "time": execute_at.strftime("%H:%M"), "window": order_window(execute_at),
            "window_label": label, "window_title": title, "window_how": how,
            "symbol": symbol, "name": name(symbol), "action": action,
            "quantity": quantity, "reference_price": quote, "notional": notional,
        })
    target = []
    for symbol, quantity in _positions_after(held, rows).items():
        quote = price(symbol)
        target.append({
            "symbol": symbol, "name": name(symbol), "quantity": quantity, "reference_price": quote,
            "value": quote * quantity if quote is not None else None,
        })
    cash = float(decision["cash"])
    target_value = sum(row["value"] or 0.0 for row in target)
    cash_after = cash + flows["sell"] - flows["buy"]
    # Weights of the estimated post-trade account: holdings at reference prices plus cash.
    total = target_value + cash_after
    for row in target:
        row["weight"] = row["value"] / total if row["value"] is not None and total > 0 else None
    return {
        "decision": decision,
        "cash": cash,
        "equity": float(decision["equity"]),
        "held_count": len(held),
        "orders": orders,
        "groups": _order_groups(orders),
        "skipped_lines": skipped,
        "flows": flows,
        "target": target,
        "target_value": target_value,
        "cash_after": cash_after,
        "cash_weight": cash_after / total if total > 0 else None,
    }


def render_orders(book: Book, trade_date: str) -> str:
    state = read_json(book.root / PAPER_STATE_NAME)
    sheet = order_sheet(book.root, trade_date, state=state)
    decision = sheet["decision"]
    cash, equity, rows = sheet["cash"], sheet["equity"], sheet["orders"]

    lines = [f"# Paper 订单 · {_day(trade_date)}", ""]
    if book.note:
        lines += [f"> {book.note}", ""]
    lines += [
        (
            f"- 模拟账户：{book.experiment_id} / {book.artifact_id}（{book.candidate_source}），"
            f"{_day(str(state.get('start_date') or trade_date))} 起，初始资金 {_money(book.profile.initial_cash)}"
        ),
        (
            f"- 决策：{_day(trade_date)} {book.schedule.inference_time}，数据截至 {_day(str(decision['data_through']))} 收盘"
            f"（发布 {str(decision['generation_id'])[:12]}）"
            + _fit_note(decision, str(state.get("last_fit_date") or ""))
        ),
        f"- 重放此前决策 {decision['replayed_calls']} 次，其中 {decision['replayed_matching_journal']} 次与模拟账户记录的订单一致",
        (
            f"- 账户（{_day(str(state.get('settled_through') or decision['data_through']))} 收盘估值）："
            f"总资产 {_money(equity)}，现金 {_money(cash)}，持仓 {sheet['held_count']} 只"
        ),
        *_real_fills_lines(state),
        "",
    ]
    if rows:
        groups = sheet["groups"]
        lines += [f"## 订单（{len(rows)} 笔{_batches(groups, rows)}）", ""]
        for group in groups:
            members = [rows[index] for index in group["orders"]]
            lines += [
                f"### {group['title']}（{len(members)} 笔）",
                "",
                group["how"],
                "",
                "| 方向 | 代码 | 名称 | 股数 | 参考价 | 约计金额 | 下单时段 |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
            for row in members:
                lines.append(
                    f"| {'买入' if row['action'] == 'buy' else '卖出'} | {row['symbol']} | {row['name']} "
                    f"| {row['quantity']:,} | {_price(row['reference_price'])} | {_money(row['notional'])} "
                    f"| {row['window_label']} |"
                )
            lines += ["", _flows_line(group["flows"]), ""]
        if any(row["window"] != CONTINUOUS for row in rows):
            lines += [LIMIT_GUIDANCE, ""]
        if sheet["skipped_lines"]:
            lines += [f"订单日志有 {sheet['skipped_lines']} 行无法解析，已跳过；请检查 orders_{trade_date}.jsonl。", ""]
    else:
        lines += ["## 订单", "", "今日无订单，持仓不变。", ""]
    lines += [
        f"## 成交后目标持仓（{len(sheet['target'])} 只）",
        "",
        "| 代码 | 名称 | 股数 | 参考价 | 参考市值 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in sheet["target"]:
        lines.append(
            f"| {row['symbol']} | {row['name']} | {row['quantity']:,} | {_price(row['reference_price'])} "
            f"| {_money(row['value'])} |"
        )
    lines += [
        "",
        f"成交后现金约 {_money(sheet['cash_after'])}，参考市值合计 {_money(sheet['target_value'])}。",
        "",
    ]
    for warning in state.get("warnings") or ():
        lines += [f"注意：{warning}", ""]
    lines += [DISCLAIMER, ""]
    return "\n".join(lines)


def render_killed(book: Book, trade_date: str, transition: Mapping[str, object]) -> str:
    """The sheet of a killed book: no decision, and the holdings its last
    sheet leaves once every order fills, for the owner to exit by hand."""

    state = read_json(book.root / PAPER_STATE_NAME)
    decisions = state.get("decisions") or ()
    target = order_sheet(book.root, str(decisions[-1]["trade_date"]), state=state)["target"] if decisions else []
    lines = [f"# Paper 订单 · {_day(trade_date)} · 已终止（killed — exit these holdings by hand）", ""]
    if book.note:
        lines += [f"> {book.note}", ""]
    lines += [
        (
            f"本模拟账户（{book.experiment_id} / {book.artifact_id}，{book.candidate_source}）在 "
            f"{_day(str(transition['date']))}（Paper 第 {transition['days']} 个结算日）被判定终止："
            f"{transition['reason']}。它不再决策，今后不出订单；请手工卖出下列持仓。"
        ),
        "",
    ]
    if target:
        lines += [
            f"## 待手工退出的持仓（{len(target)} 只，按上一张订单全部成交计）",
            "",
            "| 代码 | 名称 | 股数 |",
            "| --- | --- | ---: |",
            *(f"| {row['symbol']} | {row['name']} | {row['quantity']:,} |" for row in target),
            "",
        ]
    else:
        lines += ["账户没有持仓。", ""]
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
    ]
    if isinstance(error, PaperAwaitingFills):
        lines += [
            (
                "这个模拟账户按实际成交跟踪：先在控制台模拟账户页的「成交记录」里记下上面那一场每笔订单的结果"
                "（全部按模拟成交、全部未成交，或逐笔填实际成交价与股数），再重跑下面的命令或等下一次定时运行。"
            ),
            "",
        ]
    lines += [
        (
            "修复原因后重跑同一命令（已完成的结算不会重复）："
            f"`python scripts/paper/run_paper.py run --trade-date {trade_date}`。不要手工修改模拟账户文件。"
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


def _batches(groups: list[Mapping[str, object]], rows: list[Mapping[str, object]]) -> str:
    """The heading's batch count: a sheet placed at more than one sitting says
    so before its first table, so a morning batch never hides an afternoon one."""

    if len(groups) < 2:
        return ""
    parts = [
        f"{rows[group['orders'][0]]['window_label'].rsplit(' ', 1)[0]} {len(group['orders'])} 笔"
        for group in groups
    ]
    return f"，分 {len(groups)} 批下单：" + "，".join(parts)


def _flows_line(flows: Mapping[str, float]) -> str:
    parts = [f"{label}约 {_money(flows[side])}" for side, label in (("sell", "卖出"), ("buy", "买入")) if flows[side]]
    return ("这一批" + "，".join(parts) + "。") if parts else "这一批没有参考价，金额未估算。"


def _real_fills_lines(state: Mapping[str, object]) -> list[str]:
    """The sheet's line on how the book settles, for a book that follows the
    owner's recorded fills; a simulated book prints nothing new."""

    mode = state.get("real_fills")
    if not isinstance(mode, Mapping):
        return []
    after = str(mode.get("after") or "")
    scope = f"{_day(after)} 之后的每一场" if after else "每一场"
    return [
        (
            f"- 成交：按实际成交跟踪，{scope}都按控制台记录的成交结算。收盘后请记录本单每笔的实际结果；"
            "有订单而未记录的一场不会结算，下一张订单单也不会生成。"
        )
    ]


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


__all__ = [
    "CLOSE_AUCTION",
    "CONTINUOUS",
    "LATEST_NAME",
    "LIMIT_GUIDANCE",
    "OPEN_AUCTION",
    "order_sheet",
    "order_window",
    "orders_file_name",
    "render_failure",
    "render_killed",
    "render_orders",
    "window_text",
    "write_orders",
]
