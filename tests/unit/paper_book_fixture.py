"""Paper accounts the real engine writes, laid out as the Paper state root holds them.

The market is one name over six sessions (``SESSIONS``), committed run by run
(``_Data``); ``_Book`` drives the engine over it directly. Each book trades a
strategy copied into its own directory with a trusted executor, so a book's
journals, state and snapshot are exactly what a live run writes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.environment.replay.engine import StrategyDataView
from autotrade.environment.replay.market import DailyMarketData
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper import DailyPaperEngine
from autotrade.paper.book import Book, create_book
from tests.unit.webui_research_experiment import build_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ("20260105", "20260106", "20260107", "20260108", "20260109", "20260112")
SYMBOL = "000001.SZ"
PROFILE = BrokerProfile(
    initial_cash=100_000.0, min_commission_cny=0, slippage_bps=0, transfer_fee_bps=0, permitted_boards=("main", "gem")
)

# A module-level decision counter, the way a registered strategy may pace its
# rebalance: it buys on the first call of a worker and sells on the third.
COUNTER_STRATEGY = """_CALLS = 0


def generate_orders(context):
    global _CALLS
    call = _CALLS
    _CALLS += 1
    day = context.inference_at.strftime("%Y-%m-%d")
    if call % 3 == 0 and not context.account.positions:
        return [{"symbol": "000001.SZ", "action": "buy", "quantity": 100, "execute_at": day + "T09:30:00+08:00", "call": call}]
    if call % 3 == 2 and context.account.positions:
        return [{"symbol": "000001.SZ", "action": "sell", "quantity": 100, "execute_at": day + "T09:30:00+08:00", "call": call}]
    return []
"""


def _bars(days) -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_date": day, "symbol": SYMBOL, "open": 10.0 + index, "close": 10.5 + index,
         "pre_close": 10.5 + index - 1, "up_limit": 30.0, "down_limit": 1.0}
        for index, day in enumerate(days)
    ])


class _Data:
    """The committed data of one run: bars through ``release_end``."""

    def __init__(self, sessions, release_end: str, *, execution_price=None) -> None:
        self.sessions = tuple(sessions)
        self.release_end = release_end
        self.generation_id = f"gen_{release_end}"
        self.vintages: dict = {}
        self.market = DailyMarketData(_bars([day for day in self.sessions if day <= release_end]))
        self.views: list[str] = []
        self.closed = False
        self._execution_price = execution_price

    def context_data(self, inference_at):
        self.views.append(inference_at.isoformat())
        return StrategyDataView("", "", str(len(self.views)))

    def execution_price(self, symbol, when):
        return self._execution_price(symbol, when) if self._execution_price else None

    def references(self, symbols, before):
        return {symbol: {"name": "平安银行", "close": 10.5} for symbol in symbols}

    def close(self) -> None:
        self.closed = True


class _Book:
    """A book whose committed data ends wherever the test says it does."""

    def __init__(self, tmp_path: Path, source: str, *, sessions=SESSIONS, schedule=None) -> None:
        self.strategy = tmp_path / "strategy" / "main.py"
        self.strategy.parent.mkdir(parents=True, exist_ok=True)
        self.strategy.write_text(source, encoding="utf-8")
        self.root = tmp_path / "paper"
        self.sessions = sessions
        self.release_end = sessions[0]
        self.built: list[_Data] = []
        self.execution_price = None
        self.engine = DailyPaperEngine(
            strategy_path=self.strategy,
            strategy_revision="revision_1",
            state_root=self.root,
            data_factory=self._data,
            schedule=schedule,
            profile=PROFILE,
            executor_factory=lambda path, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
                path, state_dir=state_dir, models_dir=models_dir
            ),
        )

    def _data(self, start: str, trade_date: str) -> _Data:
        data = _Data(self.sessions, self.release_end, execution_price=self.execution_price)
        self.built.append(data)
        return data

    def run(self, trade_date: str) -> dict[str, object]:
        """Run the pre-open decision for ``trade_date`` once the prior session landed."""

        self.release_end = max(day for day in self.sessions if day < trade_date)
        return self.engine.run_day(trade_date)

    def journal(self, name: str) -> list[dict[str, object]]:
        path = self.root / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def state(self) -> dict[str, object]:
        return json.loads((self.root / ".paper_state.json").read_text(encoding="utf-8"))


def _sheet_book(book: _Book, note: str = "") -> object:
    """The frozen identity ``render_orders`` reads beside the book's own files."""

    return type("SheetBook", (), {
        "root": book.root.resolve(), "note": note,
        "experiment_id": "exp", "artifact_id": "art",
        "profile": PROFILE, "schedule": StrategySchedule(),
    })()


BOOK_SESSIONS = ("20260102", *SESSIONS)
FAILING_STRATEGY = """def generate_orders(context):
    raise RuntimeError("strategy broke")
"""


def paper_root(repo_root: Path) -> Path:
    return repo_root / "data/trading/paper"


def write_book_record(root: Path, **overrides: object) -> None:
    """The ``book.json`` fields the console and the migration read."""
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment_id": "exp", "artifact_id": "art", "benchmark_index": "000300.SH",
        "note": "参考簿（观察中）", "strategy_path": "strategy/main.py",
        "profile": {"initial_cash": PROFILE.initial_cash}, "schedule": {"inference_time": "08:30"},
        "raw_dir": "/private/lake", **overrides,
    }
    (root / "book.json").write_text(json.dumps(record), encoding="utf-8")


def run_days(root: Path, *days: str, strategy: str = COUNTER_STRATEGY) -> None:
    """Decide ``days`` in order for the book at ``root``, each once its prior session landed."""
    path = root / "strategy" / "main.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(strategy, encoding="utf-8")
    release = {"end": BOOK_SESSIONS[0]}
    engine = DailyPaperEngine(
        strategy_path=path,
        strategy_revision="revision_1",
        state_root=root,
        data_factory=lambda _start, _target: _Data(BOOK_SESSIONS, release["end"]),
        profile=PROFILE,
        executor_factory=lambda source, _sandbox, _view, state_dir, models_dir: TrustedStrategyExecutor.from_path(
            source, state_dir=state_dir, models_dir=models_dir
        ),
    )
    for day in days:
        release["end"] = max(session for session in BOOK_SESSIONS if session < day)
        engine.run_day(day)


def engine_book(repo_root: Path, *days: str, book: str = "exp", strategy: str = COUNTER_STRATEGY) -> Path:
    """The book ``book`` under the repo's Paper state root, run through ``days``;
    a later call continues it."""
    root = paper_root(repo_root) / book
    write_book_record(root, experiment_id=book)
    run_days(root, *days, strategy=strategy)
    return root


def _registered_book(tmp_path: Path, book_id: str = "exp") -> Book:
    experiment = build_experiment(tmp_path / "experiments", book_id, "registered_with_record")
    return create_book(
        paper_root(tmp_path) / book_id,
        experiment_dir=experiment,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
    )


def _run_paper():
    spec = importlib.util.spec_from_file_location("run_paper", REPO_ROOT / "scripts/paper/run_paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
