"""The Paper verdict: its status machine, its reading of a book's journals and
newest replay slot, and a killed book's run."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.replay.null_control import (
    MATCHED_MEMBERSHIP,
    run_null_control,
)
from autotrade.paper import verdict as paper_verdict
from autotrade.paper.book import (
    VERDICT_LOG_NAME,
    Book,
    create_book,
)
from autotrade.paper.storage import append_jsonl_once, read_jsonl
from autotrade.paper.verdict import evidence_reading, paper_reading
from tests.unit.paper_book_fixture import (
    BOOK_SESSIONS,
    paper_root,
    run_days,
)
from tests.unit.test_paper_reconcile import PROFILE as RECONCILE_PROFILE
from tests.unit.test_paper_reconcile import SESSIONS, A, B, _fill, _session
from tests.unit.test_paper_reconcile import _bars as _two_name_bars
from tests.unit.test_paper_reconcile import _book as _reconcile_book
from tests.unit.test_paper_trading import SYMBOL, _bars
from tests.unit.webui_research_arm import build_arm

REPO_ROOT = Path(__file__).resolve().parents[2]


def _graduated_book(tmp_path: Path, book_id: str = "exp") -> Book:
    arm = build_arm(tmp_path / "experiments", book_id, "graduated")
    return create_book(
        paper_root(tmp_path) / book_id,
        experiment_dir=arm,
        artifact_id="strategy_research_abc",
        repo_root=tmp_path,
        track="graduated",
    )


def _kill(root: Path, day: str = "20260105") -> None:
    append_jsonl_once(
        root / VERDICT_LOG_NAME,
        {"event_id": "terminal", "status": "killed", "reason": "kill_upper_bound_below_zero", "date": day, "days": 126},
    )


def _days(count: int) -> list[str]:
    start = np.datetime64("2027-01-04")
    return [str(day).replace("-", "") for day in np.busday_offset(start, np.arange(count), roll="forward")]


def _read(book: Book, returns, *, previous=None) -> dict[str, object]:
    """The reading of an account that earned ``returns`` and never traded:
    its zero-skill panel is the idle account, so its active return is its own."""

    dates = _days(len(returns))
    equity = book.profile.initial_cash * np.cumprod(1.0 + np.asarray(returns))
    rows = [{"trade_date": day, "equity": float(value)} for day, value in zip(dates, equity, strict=True)]
    return evidence_reading(book, rows, [], _bars(dates), {day: 0.0 for day in dates}, previous=previous)


def _noisy(mean: float, count: int, seed: int = 7) -> np.ndarray:
    return mean + np.random.default_rng(seed).normal(0.0, 0.004, count)


def _statistical(book: Book) -> Book:
    """The book with its drawdown limits out of reach, so only the checkpoints act."""
    return replace(book, verdict_rules={**book.verdict_rules, "max_drawdown": 0.99, "active_max_drawdown": 0.99})


def test_statistical_checks_run_only_at_checkpoints_and_each_once(tmp_path: Path):
    book = _statistical(_graduated_book(tmp_path))
    losing = _noisy(-0.002, 260)

    before = _read(book, losing[:125])
    assert before["status"] == "observing" and before["transition"] is None
    assert before["kill_upper_bound"] < 0  # reported, but not acted on before the checkpoint
    assert (before["checked_through"], before["next_checkpoint"]) == (0, 126)

    at = _read(book, losing[:126])
    assert at["status"] == "killed" and at["checked_through"] == 126
    assert at["transition"]["reason"] == "kill_upper_bound_below_zero"
    assert at["transition"]["days"] == 126 and at["transition"]["date"] == _days(126)[-1]

    # A later reading does not check the 126th day again: nothing acts until 252.
    between = _read(book, losing[:200], previous={"checked_through": 126})
    assert between["status"] == "observing" and between["next_checkpoint"] == 252

    confirmed = _read(book, _noisy(0.003, 126))
    assert confirmed["status"] == "confirmed" and confirmed["confirm_lower_bound"] > 0
    assert confirmed["transition"]["reason"] == "confirm_lower_bound_above_zero"

    # One window and one seed read the same numbers every time.
    again = _read(book, _noisy(0.003, 126))
    assert {**again, "computed_at": None, "transition": None} == {**confirmed, "computed_at": None, "transition": None}
    assert again["transition"]["confirm_lower_bound"] == confirmed["transition"]["confirm_lower_bound"]


def test_a_drawdown_kills_on_an_ordinary_day(tmp_path: Path):
    book = _graduated_book(tmp_path)  # max_drawdown 0.45, active_max_drawdown 0.30
    calm = [0.001] * 10
    account = _read(book, [*calm, -0.50, *calm])
    assert account["status"] == "killed" and account["transition"]["reason"] == "account_drawdown_exceeded"
    assert account["transition"]["days"] == 11

    # 35 % is inside the account limit but past the active one.
    active = _read(book, [*calm, -0.35, *calm])
    assert active["status"] == "killed" and active["transition"]["reason"] == "active_drawdown_exceeded"
    assert active["transition"]["date"] == _days(11)[-1]


def test_a_terminal_status_is_never_left(tmp_path: Path):
    book = _graduated_book(tmp_path)
    _kill(book.root)
    reading = _read(book, _noisy(0.003, 130))
    assert reading["status"] == "killed" and reading["transition"]["date"] == "20260105"
    assert len(read_jsonl(book.root / VERDICT_LOG_NAME)[0]) == 1

    confirmed = _graduated_book(tmp_path, "conf")
    append_jsonl_once(confirmed.root / VERDICT_LOG_NAME, {"event_id": "terminal", "status": "confirmed", "date": "20270101", "days": 126})
    assert _read(confirmed, [0.001, -0.6, 0.001])["status"] == "confirmed"


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
    _two_name_bars(SESSIONS).to_parquet(slot / "daily.parquet")
    pd.DataFrame({"symbol": pd.Series(dtype=str), "ex_date": pd.Series(dtype=str), "cash_per_share": pd.Series(dtype=float)}).to_parquet(
        slot / "corporate_actions.parquet"
    )
    (slot / "manifest.json").write_text(json.dumps({"domains": {"corporate_actions": {}}}), encoding="utf-8")


def test_a_paper_reading_reads_the_fills_a_real_fill_book_holds_over_its_newest_slot(tmp_path: Path, monkeypatch):
    """End to end through the real engine and a real-fill correction: only the
    book's settled days are read, the panel is handed the corrected fill (not
    the one the session's journal still holds), and the slot's bars, index and
    membership -- with the decision view's earlier section -- reach it."""

    real = _reconcile_book(tmp_path, monkeypatch, plan={"20260105": [(B, "buy", 300, "15:00")]})
    real.run("20260105")
    real.record(_session("20260105"))
    real.run("20260106")
    real.record(_fill("20260105", "15:00", B, "buy", quantity=200, price=15.2, commission=5.0))
    real.run("20260107")
    real.run("20260108")
    assert real.journal("corrections.jsonl") and [
        row["quantity"] for row in real.journal("executions_20260105.jsonl") if row["status"] == "filled"
    ] == [300]
    book = replace(_graduated_book(tmp_path), root=real.root, profile=RECONCILE_PROFILE)
    with pytest.raises(FileNotFoundError, match="no replay slot"):
        paper_reading(book)

    index_pct = dict(zip(SESSIONS, (0.3, 1.0, -0.5, 0.2, 0.4, 0.1, -0.2), strict=True))
    _write_slot(real.root, book.benchmark_index, index_pct)
    handed: dict[str, object] = {}

    def spy(result, frame, benchmark, *args, **kwargs):
        handed.update(result=result, frame=frame, membership=kwargs["membership"])
        return run_null_control(result, frame, benchmark, *args, **kwargs)

    monkeypatch.setattr(paper_verdict, "run_null_control", spy)
    reading = paper_reading(book)

    settled = ["20260105", "20260106", "20260107"]
    assert (reading["first_day"], reading["last_day"], reading["days"]) == ("20260105", "20260107", 3)
    assert [row["equity"] for row in handed["result"].equity_curve] == [row["equity"] for row in real.journal("equity_daily.jsonl")]
    assert [
        (row["symbol"], row["action"], row["quantity"], row["price"], row["matched_at"])
        for row in handed["result"].executions if row["status"] == "filled"
    ] == [(B, "buy", 200, 15.2, "2026-01-05T15:00:00+08:00")]
    assert sorted(handed["frame"]["trade_date"].unique()) == settled
    assert handed["membership"] == {"20251231": frozenset({A, B}), "20260106": frozenset({A})}
    assert reading["panel"]["matched"] == MATCHED_MEMBERSHIP and reading["panel"]["round_trips"] == 1
    assert reading["index_return"] == pytest.approx(np.prod([1 + index_pct[day] / 100 for day in settled]) - 1)
    assert reading["status"] == "observing" and reading["confirm_lower_bound"] is None  # under one bootstrap block


def _run_paper():
    spec = importlib.util.spec_from_file_location("run_paper", REPO_ROOT / "scripts/paper/run_paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_killed_book_writes_its_exit_sheet_decides_nothing_and_is_not_a_failure(tmp_path: Path, monkeypatch):
    run_paper = _run_paper()
    book = _graduated_book(tmp_path)
    run_days(book.root, BOOK_SESSIONS[1], BOOK_SESSIONS[2])
    _kill(book.root, BOOK_SESSIONS[2])
    state_before = (book.root / ".paper_state.json").read_bytes()
    monkeypatch.setattr(run_paper, "load_sse_trading_days", lambda _raw: list(BOOK_SESSIONS))

    failures = run_paper.run_books(
        paper_root(tmp_path), ["exp"],
        lambda book_id, root: run_paper.run_book(book_id, root, BOOK_SESSIONS[3], tmp_path / "orders"),
    )
    assert failures == {}
    sheet = (tmp_path / "orders/exp" / f"{BOOK_SESSIONS[3]}_orders.md").read_text(encoding="utf-8")
    assert "killed — exit these holdings by hand" in sheet and f"| {SYMBOL} |" in sheet
    assert (book.root / ".paper_state.json").read_bytes() == state_before  # nothing decided or settled
