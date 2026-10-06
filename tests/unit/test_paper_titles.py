"""A Paper book that reads announcement titles decides only on landed titles."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from autotrade.data_sources.tushare.cron_update import DEFAULT_CONFIG, job_state_path
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.strategy import CN_TZ
from autotrade.paper.engine import PaperTitlesNotLanded
from autotrade.paper.orders import render_failure
from autotrade.paper.titles import require_landed_titles
from tests.unit.test_paper_trading import COUNTER_STRATEGY, _Book, _sheet_book

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB = "cn_nightly_anns_full"
TITLE_BOOK = SnapshotConfig(text_datasets=("anns_d",))


def _repo(tmp_path: Path) -> Path:
    """A repository root with the real schedule and no job state yet."""

    root = tmp_path / "repo"
    (root / DEFAULT_CONFIG).parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / DEFAULT_CONFIG, root / DEFAULT_CONFIG)
    return root


def _record(repo: Path, **state: object) -> None:
    """The title job's state, as the scheduler records it."""

    path = job_state_path(JOB, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"log_path": "logs/tushare/cron/anns.log", **state}), encoding="utf-8")


def _ok(repo: Path, end_date: str, finished: str) -> None:
    _record(repo, status="ok", returncode=0, start_date="20200101", end_date=end_date, updated_at=finished)


def _refused(repo: Path, day: str) -> str:
    at = datetime.strptime(day, "%Y%m%d").replace(hour=8, minute=30, tzinfo=CN_TZ)
    with pytest.raises(PaperTitlesNotLanded) as caught:
        require_landed_titles(TITLE_BOOK, at, repo)
    return f"{caught.value}\n{caught.value.action}"


def _decides(repo: Path, day: str, config: SnapshotConfig = TITLE_BOOK) -> None:
    at = datetime.strptime(day, "%Y%m%d").replace(hour=8, minute=30, tzinfo=CN_TZ)
    require_landed_titles(config, at, repo)


def test_a_failed_title_run_refuses_the_decision_after_settling(tmp_path: Path, monkeypatch):
    secret = "token-value-that-must-not-leak"
    monkeypatch.setenv("TUSHARE_TOKEN_TMP", secret)
    repo = _repo(tmp_path)
    book = _Book(tmp_path, COUNTER_STRATEGY, sessions=("20260921", "20260922", "20260923", "20260924", "20260928"))
    book.engine.decision_check = lambda at: require_landed_titles(TITLE_BOOK, at, repo)
    _ok(repo, "20260922", "2026-09-22T15:16:00+00:00")  # 23:16 Beijing
    book.run("20260923")

    # The next evening's run loses its token: nothing written, exit 77.
    _record(
        repo, status="error", returncode=77, start_date="20260823", end_date="20260923",
        updated_at="2026-09-23T15:15:03+00:00", last_ok_at="2026-09-22T15:16:00+00:00", last_ok_end_date="20260922",
        error="OfficialAccessError: anns_d could not be fetched with TUSHARE_TOKEN_TMP; nothing was written",
    )
    with pytest.raises(PaperTitlesNotLanded) as caught:
        book.run("20260924")
    state = book.state()
    assert [row["trade_date"] for row in state["decisions"]] == ["20260923"]
    assert state["settled_through"] == "20260923"  # settling reads no titles
    assert book.journal("orders_20260924.jsonl") == []
    assert "PaperTitlesNotLanded" in state["last_error"]

    sheet = render_failure(_sheet_book(book), "20260924", caught.value)
    assert f"{JOB} 2026-09-23 23:15 那一轮" in sheet
    assert "失败，返回码 77：OfficialAccessError" in sheet
    assert "最近一次成功：2026-09-22 23:16 结束、覆盖到 20260922 的那一次" in sheet
    assert "把有效的 TUSHARE_TOKEN_TMP 写进仓库根目录的 .env" in sheet
    assert f"--job {JOB} --end-date 20260923 --force-run" in sheet
    assert "run_paper.py run --trade-date 20260924" in sheet
    values = [value for value in os.environ.values() if len(value) >= 8]
    assert secret in values and not [value for value in values if value in sheet]

    # The missed evening rerun the next morning: the same session decides.
    _ok(repo, "20260923", "2026-09-24T00:10:00+00:00")
    book.run("20260924")
    assert [row["trade_date"] for row in book.state()["decisions"]] == ["20260923", "20260924"]


def test_a_book_without_titles_ignores_the_title_job(tmp_path: Path):
    repo = _repo(tmp_path)
    _record(repo, status="error", returncode=77, end_date="20260927", updated_at="2026-09-27T15:15:03+00:00")
    for config in (SnapshotConfig(text_datasets=("report_rc",)), SnapshotConfig(replay_include_text=False)):
        _decides(repo, "20260928", config)
    assert "公告标题没有落地" in _refused(repo, "20260928")


def test_the_due_run_is_the_previous_evening_of_the_real_schedule(tmp_path: Path):
    # The schedule runs the job every calendar evening, so weekends and
    # exchange holidays have their own runs and a decision needs the last one.
    [line] = [
        line for line in (REPO_ROOT / "ops/cron/tushare_update.cron").read_text(encoding="utf-8").splitlines()
        if line.rstrip().endswith(f"--job {JOB}")
    ]
    assert line.split()[:5] == ["15", "23", "*", "*", "*"]
    repo = _repo(tmp_path)
    assert "没有任何运行记录" in _refused(repo, "20260928")

    # Monday after a weekend and the Friday Mid-Autumn holiday: Sunday's run.
    _ok(repo, "20260926", "2026-09-26T15:16:00+00:00")
    assert "2026-09-27 23:15 那一轮" in _refused(repo, "20260928")
    _ok(repo, "20260927", "2026-09-27T15:16:00+00:00")
    _decides(repo, "20260928")

    # First session after the National Day week: the last holiday evening's run.
    _ok(repo, "20261006", "2026-10-06T15:16:00+00:00")
    assert "成功运行要覆盖到 20261007" in _refused(repo, "20261008")
    # A run of that window finished before 23:15 fetched only the morning's
    # titles, and the scheduler then skips the 23:15 run as already done.
    _ok(repo, "20261007", "2026-10-07T01:00:00+00:00")
    assert "--end-date 20261007 --force-run" in _refused(repo, "20261008")
    _ok(repo, "20261007", "2026-10-07T15:20:00+00:00")
    _decides(repo, "20261008")
