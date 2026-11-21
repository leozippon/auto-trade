"""A two-name Paper book whose sessions and fills the operator records.

Two names (``A`` and ``B``) over seven sessions; ``_book`` runs ``PLAN_STRATEGY``
over them at 05:45 of each morning, as the cron does, with fill recording
enabled unless a test asks for the simulated book."""

from __future__ import annotations

import json
from datetime import datetime as real_datetime
from pathlib import Path

import pandas as pd

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.strategy import CN_TZ
from autotrade.paper import engine as engine_module
from autotrade.paper.engine import DailyPaperEngine
from autotrade.paper.fills import append_records

SESSIONS = ("20260102", "20260105", "20260106", "20260107", "20260108", "20260109", "20260112")
A, B = "000001.SZ", "600000.SH"
PROFILE = BrokerProfile(initial_cash=100_000.0)

PLAN_STRATEGY = """PLAN = {plan!r}


def generate_orders(context):
    day = context.inference_at.strftime("%Y%m%d")
    stamp = context.inference_at.strftime("%Y-%m-%d")
    return [
        {{"symbol": symbol, "action": action, "quantity": quantity, "execute_at": stamp + "T" + clock + ":00+08:00"}}
        for symbol, action, quantity, clock in PLAN.get(day, [])
    ]
"""


def _bars(days) -> pd.DataFrame:
    rows = []
    for index, day in enumerate(days):
        for offset, symbol in enumerate((A, B)):
            base = 10.0 + 5 * offset + 0.1 * index
            rows.append({
                "trade_date": day, "symbol": symbol, "open": base, "close": base + 0.05,
                "pre_close": base - 0.05, "up_limit": round(base * 1.1, 2), "down_limit": round(base * 0.9, 2),
                "high": base + 0.1, "low": base - 0.1, "vol": 1.0e6, "amount": (base + 0.02) * 1.0e6,
            })
    return pd.DataFrame(rows)


class _Data:
    def __init__(self, release_end: str) -> None:
        self.sessions = SESSIONS
        self.release_end = release_end
        self.generation_id = f"gen_{release_end}"
        self.vintages: dict = {}
        self.market = DailyMarketData(_bars([day for day in SESSIONS if day <= release_end]))

    def context_data(self, inference_at):
        return StrategyDataView("", "", inference_at.isoformat())

    def execution_price(self, symbol, when):
        return None

    def references(self, symbols, before):
        return {symbol: {"name": {A: "平安银行", B: "浦发银行"}[symbol], "close": 10.0} for symbol in symbols}

    def close(self) -> None:
        pass


class _Book:
    def __init__(self, root: Path, source: str, monkeypatch) -> None:
        self.root = root
        strategy = root.parent / f"{root.name}_strategy" / "main.py"
        strategy.parent.mkdir(parents=True, exist_ok=True)
        strategy.write_text(source, encoding="utf-8")
        self.builds = 0
        self.monkeypatch = monkeypatch
        self.engine = DailyPaperEngine(
            strategy_path=strategy, strategy_revision="revision_1", state_root=root,
            data_factory=self._data, profile=PROFILE,
            executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
                path, state_dir=state_dir, models_dir=models_dir
            ),
        )

    def _data(self, _start: str, trade_date: str) -> _Data:
        self.builds += 1
        return _Data(max(day for day in SESSIONS if day < trade_date))

    def run(self, day: str) -> dict[str, object]:
        """Decide ``day`` at 05:45 that morning, as the cron does."""

        morning = real_datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 5, 45, tzinfo=CN_TZ)

        class Clock(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return morning.astimezone(tz) if tz else morning

        self.monkeypatch.setattr(engine_module, "datetime", Clock)
        return self.engine.run_day(day)

    def record(self, *records: dict[str, object]) -> None:
        append_records(self.root, records)

    def journal(self, name: str) -> list[dict[str, object]]:
        path = self.root / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def state(self) -> dict[str, object]:
        return json.loads((self.root / ".paper_state.json").read_text(encoding="utf-8"))


def _book(tmp_path: Path, monkeypatch, *, plan=None, source=None, real=True, name="book") -> _Book:
    book = _Book(tmp_path / name, source or PLAN_STRATEGY.format(plan=plan or {}), monkeypatch)
    if real:
        book.record({"kind": "enable"})
    return book


def _session(day: str, outcome: str = "simulated") -> dict[str, object]:
    return {"kind": "session", "trade_date": day, "outcome": outcome}


def _fill(day: str, clock: str, symbol: str, action: str, outcome: str = "filled", **fields) -> dict[str, object]:
    return {"kind": "fill", "trade_date": day, "time": clock, "symbol": symbol, "action": action,
            "outcome": outcome, **fields}


def _write_slot(root: Path, benchmark: str, index_pct: dict[str, float]) -> None:
    """The replay slot a Paper run leaves (``paper/pit.py``) over the
    reconcile fixture's bars, with its decision view beside it: the index's
    daily ``pct_chg`` (percent) and a constituent section in the slot, the
    section dated before the book's first session in the decision view."""

    views = root / "pit" / "gen_1" / "pit_views"
    decision, slot = views / "decision" / "20251231T235959", views / "replay" / "paper" / f"{SESSIONS[0]}_{SESSIONS[-1]}_20251231T235959"

    def macro(rows: list[dict[str, object]]) -> pd.DataFrame:
        columns = ["dataset", "ts_code", "trade_date", "pct_chg", "index_code", "con_code"]
        return pd.DataFrame([{column: row.get(column) for column in columns} for row in rows]).astype({"pct_chg": float})

    for path in (decision, slot):
        path.mkdir(parents=True)
    weight = {"dataset": "index_weight", "index_code": benchmark}
    macro([{**weight, "trade_date": "20251231", "con_code": code} for code in (A, B)]).to_parquet(decision / "macro.parquet")
    macro([
        *({"dataset": "index_daily", "ts_code": benchmark, "trade_date": day, "pct_chg": value} for day, value in index_pct.items()),
        {**weight, "trade_date": "20260106", "con_code": A},
    ]).to_parquet(slot / "macro.parquet")
    _bars(SESSIONS).to_parquet(slot / "daily.parquet")
    pd.DataFrame({"symbol": pd.Series(dtype=str), "ex_date": pd.Series(dtype=str), "cash_per_share": pd.Series(dtype=float)}).to_parquet(
        slot / "corporate_actions.parquet"
    )
    (slot / "manifest.json").write_text(json.dumps({"domains": {"corporate_actions": {}}}), encoding="utf-8")
