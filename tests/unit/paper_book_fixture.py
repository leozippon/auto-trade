"""Paper books the real engine writes, laid out as the Paper state root holds them.

Each book trades a strategy copied into its own directory, over the synthetic
bars of ``test_paper_trading`` with a trusted executor, so a book's journals,
state and snapshot are exactly what a live run writes.
"""

from __future__ import annotations

import json
from pathlib import Path

from autotrade.environment.executor import TrustedStrategyExecutor
from autotrade.paper import DailyPaperEngine
from tests.unit.test_paper_trading import COUNTER_STRATEGY, PROFILE, SESSIONS, _Data

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
        "experiment_id": "exp", "artifact_id": "art", "candidate_source": "graduated",
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
