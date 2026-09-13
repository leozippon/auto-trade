"""HITL console backend tests: registry read-models, lifecycle guards, API routes.

No worker subprocesses, Docker, or LLM calls: worker spawn is patched out and
experiment state is synthesized on disk exactly as the orchestrator writes it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import LOCAL_QWEN_MODEL, MODEL_CHOICES
from autotrade.environment.runtime import (
    TRACE_PAYLOAD_HEAD_CHARS,
    AgentTraceWriter,
    write_json_atomic,
)
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY, AcceptanceRules
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    ControlState,
    StatusReporter,
    proc_start_ticks,
    read_control,
    read_status,
    status_pid_alive,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.prior import ExperimentPriorStore
from autotrade.pipelines.skills import ExperimentSkillsStore
from autotrade.webui.manager import (
    MAX_RUNNING_EXPERIMENTS,
    ExperimentManager,
    ManagerError,
)
from autotrade.webui.public_identity import PublicIdentity
from autotrade.webui.server import create_app, is_loopback_host

#: Every control action blocked once Test/Held-out numbers are on screen —
#: `manager._SEALED_BLOCKED_ACTIONS`, verbatim and in sorted order.
_SEALED_AFTER_REVEAL = (
    "cancel_skip_to_heldout",
    "inject_message",
    "rerun_fold",
    "restart",
    "resume",
    "rollback_fold",
    "set_directive",
    "skip_to_heldout",
)


# The evaluation backends name every result directory f"{mode}_{uuid4().hex}"
# (pipelines/pit_backend.py, pipelines/local_backend.py). Fixtures use that real
# shape so the console's prefix handling is exercised exactly as it ships.
VALID_RESULT_DIR = "valid_5b1d0a9c8e7f46329d1c4b7a2e6f8d03"
TEST_RESULT_DIR = "frozen_test_2c9f7a1e4d6b48305fa8e3c7b105d69e"
# One calendar-quarter row of the breakdown every result now carries.
_SUB_WINDOW = {
    "kind": "quarter",
    "label": "2022Q1",
    "start": "20220104",
    "end": "20220331",
    "trade_days": 58,
    "partial": False,
    "return": 0.04,
    "benchmark_return": 0.01,
    "excess_return": 0.03,
    "sharpe": 0.9,
    "max_drawdown": 0.03,
    "turnover": 1.1,
    "trade_count": 5,
}


def _write_ledger(experiment_dir: Path, records: list[dict[str, object]]) -> None:
    ledger = experiment_dir / "ledgers" / "experiment_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(
            json.dumps({"schema_version": 1, **record}) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def test_local_webui_health_schema_and_brand(tmp_path: Path):
    daily = tmp_path / "data/raw/daily"
    daily.mkdir(parents=True)
    for trade_date in (
        "20240102",
        "20240103",
        "20240201",
        "20240202",
        "20240301",
        "20240304",
        "20240429",
        "20240430",
    ):
        (daily / f"trade_date={trade_date}.parquet").touch()
    client = TestClient(create_app(tmp_path))
    # The loopback console reports operational health only: the closed
    # code-version/keepalive fields exist to serve a remote deployment stack
    # this version does not have.
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert "experiments_root" not in health
    assert str(tmp_path) not in json.dumps(health)
    assert health["max_running_experiments"] == MAX_RUNNING_EXPERIMENTS
    assert health["running"] == []
    assert health["unreadable_experiments"] == []
    assert health["raw_generation"] == {"state": "absent"}
    schema = client.get("/api/parameter-schema").json()
    fields = {
        field["key"]: field for group in schema["groups"] for field in group["fields"]
    }
    assert "fields" not in schema
    assert schema["schema_version"] == 3
    assert [group["name"] for group in schema["groups"]] == [
        "基本与排程",
        "数据窗口",
        "数据域",
        "股票筛选",
        "预算与验收",
        "Broker 账户",
        "运行控制",
        "模型与上下文",
    ]
    assert (
        not {
            "strategy_path",
            "data_backend",
            "daily_path",
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "execution_mode",
            "developer_mode",
        }
        & fields.keys()
    )
    for key, value in DEFAULT_RESEARCH_GEOMETRY.to_record().items():
        assert fields[key]["default"] == value
    assert fields["strategy_period"]["choices"] == ["day", "month", "quarter", "year"]
    assert fields["inference_time"]["default"] == "08:30"
    assert fields["daily_window_months"]["optional"] is True
    assert fields["include_intraday"]["default"] is WEB_CREATE_DEFAULTS["include_intraday"]
    assert fields["fundamental_datasets"]["type"] == "multi"
    assert fields["macro_datasets"]["choices"]
    assert fields["events_datasets"]["choices"]
    assert fields["text_datasets"]["choices"]
    assert fields["screen_boards"]["choices"] == ["main", "gem", "star", "bj"]
    model_fields = [
        field["key"]
        for group in schema["groups"]
        if group["name"] == "模型与上下文"
        for field in group["fields"]
    ]
    assert model_fields == [
        "model",
        "subagent_model",
        "nl_model",
        "compact_model",
        "reasoning_effort",
        "no_thinking",
        "disable_context_compact",
        "compact_token_threshold",
        "compact_keep_recent_messages",
        "compact_max_tokens",
        "compact_max_calls",
    ]
    assert fields["model"]["choices"] == list(MODEL_CHOICES)
    assert fields["model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["subagent_model"]["choices"] == fields["model"]["choices"]
    assert fields["subagent_model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["nl_model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["compact_model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["reasoning_effort"]["default"] == "xhigh"
    # Empty = derived from the model window by the worker.
    assert fields["compact_token_threshold"]["default"] is None
    assert fields["compact_token_threshold"]["optional"] is True
    assert fields["compact_keep_recent_messages"]["default"] == 10
    assert fields["compact_max_tokens"]["default"] == 10_000
    assert fields["compact_max_calls"]["default"] == 10
    assert (
        not {
            "llm_model",
            "llm_max_response_tokens",
            "llm_temperature",
            "nl_max_results",
            "nl_max_calls_per_decision",
            "nl_max_total_calls",
            "nl_deadline_seconds",
        }
        & fields.keys()
    )
    for key in (
        "research_sessions",
        "window_months",
        "max_steps_per_fold",
        "max_backtests_per_fold",
        "max_llm_calls",
    ):
        assert not fields[key].get("advanced", False)
    page = client.get("/")
    assert page.status_code == 200
    assert "ADM-Cube" in page.text
    assert "no-store" in page.headers["cache-control"]
    # Brand mark is a CSS background, so a failed fetch paints nothing rather
    # than the browser placeholder glyph in the top-left corner.
    assert '<span class="logo" aria-hidden="true"></span>' in page.text
    assert "<img" not in page.text.split("</header>")[0]
    assert 'url("/static/logo.png")' in client.get("/static/style.css").text
    assert 'rel="icon" href="/static/logo.png"' in page.text
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/png"
    # The nav strip carries a frontend-only 实盘交易 entry on its own #/qmt
    # route. What must stay absent is the live-trading console itself: no
    # backend, no communication, no execution path behind it.
    assert '<a href="#/qmt" data-nav="qmt">实盘交易</a>' in page.text
    assert "#/trading/live" not in page.text and 'data-nav="live"' not in page.text
    logo = client.get("/static/logo.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"


def test_health_incompatible_hitl_state_does_not_echo_paths(tmp_path: Path):
    hitl = tmp_path / "experiments" / "exp_incompat" / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "status.json").write_text(
        json.dumps({"schema_version": 99, "state": "created"}),
        encoding="utf-8",
    )
    response = TestClient(create_app(tmp_path)).get("/api/health")
    assert response.status_code == 200
    health = response.json()
    assert health["status"] == "degraded"
    assert health["unreadable_experiments"] == [
        {
            "experiment_id": "exp_incompat",
            "error": "ValueError: HITL control plane is unreadable",
        }
    ]
    dumped = json.dumps(health)
    assert str(tmp_path) not in dumped
    assert str(hitl) not in dumped


def test_local_webui_disables_openapi_docs(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code == 404


def test_site_footer_shows_icp_and_public_security_filings(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    page = client.get("/").text
    assert "津ICP备2024017854号-2" in page
    assert 'href="https://beian.miit.gov.cn/"' in page
    assert "津公网安备12010402002613号" in page
    assert (
        'href="https://beian.mps.gov.cn/#/query/webSearch?code=12010402002613"'
        in page
    )
    assert 'src="/static/gongan.png"' in page
    assert "hugo-next" not in page
    gongan = client.get("/static/gongan.png")
    assert gongan.status_code == 200
    assert gongan.headers["content-type"] == "image/png"
    assert gongan.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_exposes_distinct_session_and_subagent_model_choices() -> None:
    from scripts.experiments._cli import add_model_arguments

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.model == defaults.subagent_model == LOCAL_QWEN_MODEL
    assert not hasattr(defaults, "meta_model")
    mixed = parser.parse_args(
        ["--model", "deepseek-v4-flash", "--subagent-model", LOCAL_QWEN_MODEL]
    )
    assert mixed.model == "deepseek-v4-flash"
    assert mixed.subagent_model == LOCAL_QWEN_MODEL


def test_agent_trace_is_redacted_bounded_and_private(tmp_path: Path):
    path = tmp_path / "artifacts/traces/run_001.jsonl"
    writer = AgentTraceWriter(
        path,
        ids={"experiment_id": "demo", "run_id": "run_001"},
        max_bytes=900,
        max_event_bytes=450,
    )
    writer.emit(
        "tool_call",
        {
            "authorization": "Bearer should-not-appear",
            "arguments": {"api_key": "sk-abcdefghijk"},
            "content": "x" * 2_000,
        },
    )
    writer.emit("tool_call", {"content": "y" * 2_000})
    text = path.read_text(encoding="utf-8")
    assert "should-not-appear" not in text and "abcdefghijk" not in text
    assert path.stat().st_size <= 900
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_oversized_tool_call_event_keeps_a_readable_bounded_stub(tmp_path: Path):
    """A tool_call carries no ``content``: without a payload head the stub kept
    only identifiers, and the largest events — exactly the ones worth reading
    back — said nothing about what the tool was asked to do."""
    path = tmp_path / "artifacts/traces/run_002.jsonl"
    writer = AgentTraceWriter(
        path,
        ids={"experiment_id": "demo", "run_id": "run_002"},
        max_bytes=1_000_000,
        max_event_bytes=64 * 1024,
    )
    writer.emit(
        "tool_call",
        {
            "call_index": 7,
            "tool": "daily_backtest",
            "arguments": {"note": "rebalance", "api_key": "sk-abcdefghijk"},
            "result": {"ok": True, "per_stock": ["row" * 40 for _ in range(4_000)]},
        },
    )
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["truncated"] is True
    assert record["tool"] == "daily_backtest" and record["call_index"] == 7
    assert record["original_bytes"] > 64 * 1024
    head = record["payload_head"]
    assert len(head) <= TRACE_PAYLOAD_HEAD_CHARS
    assert '"note": "rebalance"' in head and '"per_stock"' in head
    assert "abcdefghijk" not in head
    # The reply preview stays empty: this event never had a ``content`` field.
    assert record["content_preview"] == ""


def test_experiment_endpoint_rejects_console_managed_unknown_and_missing_parameters(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    closed = client.post("/api/experiments", json={"data_backend": "daily"})
    assert closed.status_code == 400
    assert (
        closed.json()["detail"]
        == "console-managed parameters are not accepted: data_backend"
    )
    old_model = client.post("/api/experiments", json={"llm_model": "deepseek-chat"})
    assert old_model.status_code == 400
    assert (
        old_model.json()["detail"]
        == "console-managed parameters are not accepted: llm_model"
    )
    endpoint = client.post(
        "/api/experiments",
        json={"llm_base_url": "https://untrusted.example.test/v1"},
    )
    assert endpoint.status_code == 400
    assert endpoint.json()["detail"] == (
        "console-managed parameters are not accepted: llm_base_url"
    )
    unknown = client.post("/api/experiments", json={"period": "month"})
    assert unknown.status_code == 400
    assert unknown.json()["detail"] == "unknown experiment parameters: period"
    missing = client.post("/api/experiments", json={})
    assert missing.status_code == 400
    assert "experiment_id" in missing.json()["detail"]


def test_experiment_endpoint_creates_only_persistent_sandbox_research(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/experiments",
        json={
            "params": {
                "experiment_id": "persistent_demo",
                **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                "strategy_period": "quarter",
                "inference_time": "23:59",
                "daily_window_months": 18,
                "include_macro": False,
                "events_datasets": ["margin", "moneyflow"],
                "screen_boards": ["main", "gem"],
                "model": "deepseek-v4-flash",
                "subagent_model": LOCAL_QWEN_MODEL,
                "nl_model": "deepseek-v4-pro",
                "compact_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
                "no_thinking": True,
                "disable_context_compact": False,
                "compact_token_threshold": 90_000,
                "compact_keep_recent_messages": 10,
                "compact_max_tokens": 1_200,
                "compact_max_calls": 4,
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["experiment_id"] == "persistent_demo"
    assert response.json()["spawned"] is False
    params = json.loads(
        (tmp_path / "experiments/persistent_demo/hitl/params.json").read_text(
            encoding="utf-8"
        )
    )
    assert params["strategy_path"] == "configs/agent_output_template/main.py"
    assert params["data_backend"] == "pit"
    assert params["execution_mode"] == "sandbox"
    assert params["developer_mode"] == "llm"
    assert params["strategy_period"] == "quarter"
    assert params["inference_time"] == "23:59"
    assert params["daily_window_months"] == 18
    assert params["include_macro"] is False
    assert params["events_datasets"] == ["margin", "moneyflow"]
    assert params["screen_boards"] == ["main", "gem"]
    assert params["model"] == "deepseek-v4-flash"
    assert params["subagent_model"] == LOCAL_QWEN_MODEL
    assert params["nl_model"] == "deepseek-v4-pro"
    assert params["compact_model"] == "deepseek-v4-pro"
    assert params["reasoning_effort"] == "high"
    assert params["no_thinking"] is True
    assert params["compact_token_threshold"] == 90_000


def _fold_ref(experiment_dir: Path, raw_fold_id: str) -> str:
    return PublicIdentity(experiment_dir).fold_ref(raw_fold_id)


def _run_ref(experiment_dir: Path, raw_run_id: str) -> str:
    return PublicIdentity(experiment_dir).run_ref(raw_run_id)


def _session_ref(experiment_dir: Path, raw_session_key: str) -> str:
    return PublicIdentity(experiment_dir).public_session_key(raw_session_key)


def _live_pid_fields() -> dict[str, object]:
    pid = os.getpid()
    return {"pid": pid, "pid_start_ticks": proc_start_ticks(pid)}


# A console experiment is resolvable by definition — a worker ran it — and the
# prompt preview resolves one the very same way. Fixtures that exercise the
# preview therefore carry these parameters and the repository inputs below;
# the rest keep a bare repository root on purpose.
RESEARCH_PARAMS: dict[str, object] = {
    "strategy_path": "configs/agent_output_template/main.py",
    "data_backend": "pit",
    "raw_dir": "data/raw",
    "fundamental_events_root": "data/pit/fundamental_events",
    "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
    # Keeps the pinned release to the core datasets the fixture provides.
    "include_fundamentals": False,
    "include_macro": False,
    "include_events": False,
    "include_text": False,
    "include_intraday": False,
}


def _research_inputs(repo_root: Path) -> None:
    """The repository inputs the worker's parameter resolution reads.

    Only the trade calendar is parsed; the release's dataset partitions merely
    have to exist, and the daily partition names are the pipeline's trading
    calendar for the fold schedule.
    """
    template = repo_root / "configs" / "agent_output_template"
    template.mkdir(parents=True, exist_ok=True)
    (template / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    (repo_root / "data" / "pit" / "fundamental_events").mkdir(parents=True)
    raw = repo_root / "data" / "raw"
    days = [
        f"{year}{month:02d}{day:02d}"
        for year in range(2019, 2028)
        for month in range(1, 13)
        for day in (5, 20)
    ]
    calendar = raw / "trade_cal" / "exchange=SSE"
    calendar.mkdir(parents=True)
    pd.DataFrame({"cal_date": days, "is_open": ["1"] * len(days)}).to_parquet(
        calendar / "year=2019.parquet"
    )
    for dataset in ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d"):
        directory = raw / dataset
        directory.mkdir(parents=True)
        for day in days if dataset == "daily" else days[:1]:
            (directory / f"trade_date={day}.parquet").touch()


def _persistent_experiment(tmp_path: Path) -> Path:
    _research_inputs(tmp_path)
    directory = tmp_path / "experiments/demo"
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    write_control(hitl / "control.json", ControlState(mode="manual"))
    (hitl / "status.json").write_text(
        json.dumps({"schema_version": 1, "state": "created"}), encoding="utf-8"
    )
    (hitl / "params.json").write_text(
        json.dumps(
            {
                "experiment_id": "demo",
                "strategy_period": "day",
                "inference_time": "08:30",
                **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                **RESEARCH_PARAMS,
            }
        ),
        encoding="utf-8",
    )
    (hitl / "schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sessions": [
                    {
                        "session_key": "epoch_001/fold_2026Q1",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2026Q1",
                        "fold_index": 0,
                    },
                    {
                        "key": "heldout",
                        "kind": "heldout",
                        "epoch_id": "epoch_001",
                        "periods": [
                            {"label": "2026Q2", "start": "20260401", "end": "20260630"}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    strategy = directory / "artifacts/strategy/frozen/strategy_001/output"
    strategy.mkdir(parents=True)
    (strategy / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    trace_writer = AgentTraceWriter(
        directory / "artifacts/traces/run_001.jsonl",
        ids={"experiment_id": "demo", "run_id": "run_001"},
    )
    trace_writer.emit(
        "session_start",
        {
            "mode": "fold",
            "system_prompt": "遵守 PIT 合同；不要暴露 fold_2026Q1",
            "instruction": "改进 fold_2026Q1 当前策略",
        },
    )
    trace_writer.emit(
        "llm_call",
        {"status": "ok", "content": "检查验证表现", "usage": {"total_tokens": 12}},
    )
    result = directory / "artifacts/results/valid_001/result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "equity_curve": [
                    {
                        "trade_date": "20260102",
                        "initial_equity": 1_000_000,
                        "equity": 1_010_000,
                    },
                    {
                        "trade_date": "20260105",
                        "initial_equity": 1_000_000,
                        "equity": 1_020_000,
                    },
                ],
                "executions": [
                    {
                        "symbol": "000001.SZ",
                        "action": "buy",
                        "quantity": 100,
                        "status": "filled",
                        "price": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    skills_source = tmp_path / "skills-source"
    first_skill = skills_source / "schema-notes"
    second_skill = skills_source / "workflow-notes"
    first_skill.mkdir(parents=True)
    second_skill.mkdir(parents=True)
    (first_skill / "SKILL.md").write_text(
        "# Schema Notes\n\nRead schema first.\n", encoding="utf-8"
    )
    (second_skill / "SKILL.md").write_text(
        "# Workflow Notes\n\nKeep checks bounded.\n", encoding="utf-8"
    )
    references = first_skill / "references"
    references.mkdir()
    used_bytes = sum(
        path.stat().st_size for path in skills_source.rglob("*") if path.is_file()
    )
    (references / "detail.txt").write_text(
        "x" * (512 - used_bytes), encoding="utf-8"
    )
    skills = ExperimentSkillsStore(directory).publish(
        skills_source, generation_id="epoch_001_fold_2026Q1_run_001"
    )
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        {
            "record_type": "fold",
            "experiment_id": "demo",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2026Q1",
            "run_id": "run_001",
            "session_key": "epoch_001/fold_2026Q1",
            "fold_status": "frozen",
            "skills_ref": skills.skills_ref,
            "skills_generation_id": skills.generation_id,
            "skills_count": skills.stats.count,
            "skills_files": skills.stats.files,
            "skills_bytes": skills.stats.bytes,
            "skills_published": True,
            "run_wall_seconds": 12.5,
            "selected_step_id": "step_001",
            "steps": [
                {
                    "step_id": "step_001",
                    "revision_id": "revision_001",
                    "complete_validation": True,
                    "validation_result_ref": str(result),
                }
            ],
            "frozen_strategy_artifact_id": "strategy_001",
            "frozen_strategy_artifact_path": str(strategy),
            "validation_result": {
                "total_return": 0.02,
                "max_drawdown": -0.01,
                "filled_orders": 2,
            },
            # The Test window this Fold was scheduled on. A Fold record always
            # carries it (``None`` on a schedule without a Test stage), and it
            # is what says the Fold owed a frozen-Test series at all.
            "test_period": "20260201..20260228",
            "test_result": {"total_return": 0.01, "max_drawdown": -0.02},
        }
    )
    style = result.parent / "style_analysis.json"
    style.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "valid",
                "benchmark": {"ts_code": "000300.SH", "label": "沪深300"},
                "benchmark_regression": {
                    "available": True,
                    "reason": None,
                    "n_days": 10,
                    "benchmark_return": 0.01,
                    "beta": 0.8,
                    "alpha_annualized": 0.05,
                    "r2": 0.4,
                },
                "style": {
                    "available": True,
                    "reason": None,
                    "days": 2,
                    "tilts": {"size": -0.1, "pb": 0.2, "turnover": 0.0},
                    "industries": [],
                    "avg_names": 1.0,
                    "avg_long_gross": 1000.0,
                    "avg_short_gross": 0.0,
                },
                "strategy_daily": [["20260102", 0.01], ["20260105", 0.00990099]],
                "benchmark_daily": [["20260102", 0.005], ["20260105", -0.002]],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_revealed_equity_includes_test_and_heldout_csi300(tmp_path: Path):
    directory = _persistent_experiment(tmp_path)
    fold_ref = _fold_ref(directory, "fold_2026Q1")
    test_dir = directory / "artifacts/results/frozen_test_001"
    test_dir.mkdir()
    test_dir.joinpath("result.json").write_text(
        json.dumps(
            {
                "equity_curve": [
                    {
                        "trade_date": "20260202",
                        "initial_equity": 1_000_000,
                        "equity": 1_030_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    test_dir.joinpath("style_analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "frozen_test",
                "benchmark_daily": [["20260202", 0.004]],
            }
        ),
        encoding="utf-8",
    )
    heldout_dir = directory / "artifacts/results/heldout_001"
    heldout_dir.mkdir()
    heldout_dir.joinpath("result.json").write_text(
        json.dumps(
            {
                "equity_curve": [
                    {
                        "trade_date": "20260504",
                        "initial_equity": 1_000_000,
                        "equity": 990_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    heldout_dir.joinpath("style_analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "heldout",
                "benchmark_daily": [["20260504", -0.001]],
            }
        ),
        encoding="utf-8",
    )
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    fold = json.loads(ledger.path.read_text(encoding="utf-8").splitlines()[0])
    fold["test_result_ref"] = str(test_dir / "result.json")
    ledger.path.write_text(json.dumps(fold) + "\n", encoding="utf-8")
    ledger.append(
        {
            "record_type": "heldout",
            "experiment_id": "demo",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_2026Q2",
            "run_id": "run_heldout",
            "result_ref": str(heldout_dir / "result.json"),
            "result": {"total_return": -0.01},
        }
    )
    write_control(directory / "hitl/control.json", ControlState(mode="manual", test_revealed=True))
    client = TestClient(create_app(tmp_path))
    curve = client.get("/api/experiments/demo/equity").json()
    assert set(curve["benchmark"]["dates"]) == {"20260102", "20260105", "20260202", "20260504"}
    fold_curve = client.get(
        f"/api/experiments/demo/folds/epoch_001/{fold_ref}/equity"
    ).json()
    assert set(fold_curve["benchmark"]["dates"]) == {"20260102", "20260105", "20260202"}
    assert {series["key"] for series in curve["series"]} >= {"valid", "test", "heldout"}


def _result_artifact(directory: Path, name: str, days: list[tuple[str, float]]) -> str:
    """One replay result whose equity curve starts from 100."""
    path = directory / "artifacts/results" / name / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "equity_curve": [
                    {"trade_date": day, "initial_equity": 100.0, "equity": equity}
                    for day, equity in days
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _walk_forward_experiment(tmp_path: Path, folds: list[dict[str, object]]) -> Path:
    """A console experiment whose ledger is exactly ``folds``.

    The schedule lists the same Folds, because public Fold identity is minted
    from it: without it no ``fold_ref_`` resolves back to a record and the
    per-Fold routes cannot be exercised at all.
    """
    directory = tmp_path / "experiments/walk"
    AgentRefStore(directory)
    (directory / "hitl").mkdir(parents=True, exist_ok=True)
    write_control(directory / "hitl/control.json", ControlState(mode="manual"))
    (directory / "hitl/status.json").write_text(
        json.dumps({"schema_version": 1, "state": "created"}), encoding="utf-8"
    )
    write_json_atomic(
        directory / "hitl/schedule.json",
        {
            "schema_version": 1,
            "epochs": 1,
            "sessions": [
                {
                    "key": f"epoch_001/{fold['fold_id']}",
                    "kind": "fold",
                    "epoch_id": "epoch_001",
                    "fold_id": fold["fold_id"],
                }
                for fold in folds
            ],
        },
    )
    _write_ledger(
        directory,
        [
            {
                "record_type": "fold",
                "experiment_id": "walk",
                "epoch_id": "epoch_001",
                **fold,
            }
            for fold in folds
        ],
    )
    return directory


def test_the_forward_line_chains_the_parent_controls_on_their_scored_spans(
    tmp_path: Path,
):
    """The out-of-sample counterpart of the Validation chain, day by day.

    The Validation chain's new quarter is the strategy the Fold left in force,
    which for a Fold that froze is a candidate selected on that very quarter.
    The forward line chains the other leg -- the inherited parent replayed on
    ground nobody chose it on -- over exactly the spans
    ``ledger.transition_result`` grades and the graduation term counts. A
    trailing window must contribute its new quarter alone: taking the whole
    parent replay would compound the quarters its predecessors already scored.
    """
    import math

    from autotrade.pipelines.ledger import latest_fold_records, transition_result
    from autotrade.webui import equity, registry

    directory = tmp_path / "experiments/walk"
    first = _result_artifact(
        directory,
        "valid_first",
        [("20220701", 110.0), ("20221003", 121.0), ("20230103", 133.1), ("20230331", 146.41)],
    )
    second = _result_artifact(
        directory, "valid_second", [("20230403", 150.0), ("20230630", 300.0)]
    )
    fourth = _result_artifact(
        directory, "valid_fourth", [("20231009", 90.0), ("20231229", 45.0)]
    )
    # The parent of Fold 2 replayed over Fold 2's whole trailing window: only
    # 2023Q2 is ground it had never seen.
    control_second = _result_artifact(
        directory,
        "control_second",
        [("20221003", 200.0), ("20230103", 400.0), ("20230403", 440.0), ("20230630", 484.0)],
    )
    control_third = _result_artifact(
        directory,
        "control_third",
        [("20230103", 50.0), ("20230403", 25.0), ("20230703", 20.0), ("20230929", 10.0)],
    )
    _walk_forward_experiment(
        tmp_path,
        [
            {
                "fold_id": "fold_2023Q1",
                "validation_period": "20220701..20230331",
                "fold_status": "frozen",
                "selected_step_id": "step_1",
                "steps": [{"step_id": "step_1", "validation_result_ref": first}],
            },
            {
                "fold_id": "fold_2023Q2",
                "validation_period": "20221001..20230630",
                "fold_status": "frozen",
                "selected_step_id": "step_2",
                "steps": [{"step_id": "step_2", "validation_result_ref": second}],
                "parent_control": {
                    "status": "ok",
                    "validation_result_ref": control_second,
                    "validation_result": {"total_return": 3.84},
                    "step_result": {"start": "20230403", "end": "20230630", "total_return": 0.21},
                },
            },
            {
                # Kept its parent, so this Fold's new quarter is the same
                # replay on both lines -- the honest case.
                "fold_id": "fold_2023Q3",
                "validation_period": "20230101..20230930",
                "fold_status": "no_update",
                "parent_control": {
                    "status": "ok",
                    "validation_result_ref": control_third,
                    "validation_result": {"total_return": -0.90},
                    "step_result": {"start": "20230703", "end": "20230930", "total_return": -0.60},
                },
            },
            {
                # The control never completed: a transition that proved
                # nothing, and a quarter the forward line cannot draw.
                "fold_id": "fold_2023Q4",
                "validation_period": "20230401..20231231",
                "fold_status": "frozen",
                "selected_step_id": "step_4",
                "steps": [{"step_id": "step_4", "validation_result_ref": fourth}],
                "parent_control": {
                    "status": "failed",
                    "error": "TimeoutError: parent control exceeded the deadline",
                },
            },
        ],
    )

    payload = equity.experiment_equity_payload(tmp_path / "experiments", "walk")
    forward = next(row for row in payload["series"] if row["key"] == "forward")
    assert forward["label"] == "父本对照前向（样本外过渡）"
    # Only the scored spans, and each exactly once: the shared 2022Q4/2023Q1
    # quarters of the second control never reach the line.
    assert forward["dates"] == ["20230403", "20230630", "20230703", "20230929"]
    assert forward["cum"] == [0.1, 0.21, -0.032, -0.516]
    # ... and the line compounds to the product of the very returns the ledger
    # grades those transitions on, so the chart and the transition table are
    # one record read two ways.
    records = registry.read_ledger_records(directory)
    scored = [
        transition_result(record.get("parent_control"))
        for record in registry.walk_forward_folds(
            list(latest_fold_records(records).values())
        )[1:]
    ]
    drawn = math.prod(
        1.0 + row["total_return"] for row in scored if row is not None
    )
    assert forward["final"] == pytest.approx(drawn - 1.0)

    # The Validation chain is a different record over the same calendar: its
    # 2023Q2 is the candidate frozen on it, worth +100%, not the parent's +21%.
    valid = next(row for row in payload["series"] if row["key"] == "valid")
    assert valid["dates"][-1] == "20231229"
    assert valid["final"] != forward["final"]
    # The quarter the failed control owed is named, so the line's early stop
    # is stated rather than left to be read off the axis.
    assert payload["missing"] == {"forward": ["2023Q4"]}
    # Full-cycle statistics are served for it like every other chained series.
    assert payload["stats"]["forward"]["n_days"] == 4


def test_an_experiment_without_folds_carries_no_walk_forward_term(tmp_path: Path):
    """Nothing recorded yet is not a walk-forward record of zero positives.

    A freshly started arm must be able to say 「尚无过渡」 on its card rather
    than 「0/0 正」 under a floor it has not been measured against, so the list
    payload offers no Epoch row to read at all.
    """
    from autotrade.webui import registry

    _walk_forward_experiment(tmp_path, [])
    listed = registry.list_experiments(tmp_path / "experiments")
    assert [row["experiment_id"] for row in listed] == ["walk"]
    assert listed[0]["metrics_by_epoch"] == []
    assert listed[0]["metrics"]["epoch_id"] is None


def test_a_fold_whose_in_force_replay_is_unreadable_still_names_its_source(
    tmp_path: Path,
):
    """An unreadable artifact is not a Fold that left no strategy.

    The two are one blank pane apart, and only the source tells them apart:
    the label stands on what the ledger says the Fold left in force, so the
    console can say the replay is missing instead of that nothing traded.
    """
    from autotrade.webui import equity

    directory = tmp_path / "experiments/walk"
    _walk_forward_experiment(
        tmp_path,
        [
            {
                "fold_id": "fold_2022Q3",
                "validation_period": "20220701..20220930",
                "fold_status": "no_update",
                "parent_control": {
                    "status": "ok",
                    "validation_result_ref": str(
                        directory / "artifacts/results/valid_gone/result.json"
                    ),
                },
            }
        ],
    )
    payload = equity.fold_equity_payload(
        tmp_path / "experiments",
        "walk",
        "epoch_001",
        PublicIdentity(directory).fold_ref("fold_2022Q3"),
    )
    assert payload["strategy_in_force"] == "parent_control"
    assert payload["series"] == []


def test_missing_quarters_are_read_off_the_chained_days_not_off_failed_folds(
    tmp_path: Path,
):
    """Only quarters a line truly has no day in are reported missing.

    Trailing windows overlap, so when a Fold has no replay a later Fold's
    window still fills its quarters on the Validation line: naming that Fold
    told the operator 2023Q2 was absent from a curve visibly drawn over it.
    The forward line takes only each Fold's new quarter, so the same failed
    control leaves 2023Q2 truly uncovered there, and that is reported.
    """
    from autotrade.webui import equity

    directory = tmp_path / "experiments/walk"
    first = _result_artifact(
        directory,
        "valid_first",
        [("20220701", 110.0), ("20221003", 121.0), ("20230103", 133.1)],
    )
    third = _result_artifact(
        directory,
        "control_third",
        [("20221003", 100.0), ("20230103", 101.0), ("20230403", 102.0), ("20230703", 103.0)],
    )
    _walk_forward_experiment(
        tmp_path,
        [
            {
                "fold_id": "fold_2023Q1",
                "validation_period": "20220701..20230331",
                "fold_status": "frozen",
                "selected_step_id": "step_1",
                "steps": [{"step_id": "step_1", "validation_result_ref": first}],
            },
            {
                "fold_id": "fold_2023Q2",
                "validation_period": "20221001..20230630",
                "fold_status": "no_update",
                "parent_control": {"status": "failed", "error": "TimeoutError: deadline"},
            },
            {
                "fold_id": "fold_2023Q3",
                "validation_period": "20230101..20230930",
                "fold_status": "no_update",
                "parent_control": {
                    "status": "ok",
                    "validation_result_ref": third,
                    "step_result": {"start": "20230701", "end": "20230930"},
                },
            },
        ],
    )

    payload = equity.experiment_equity_payload(tmp_path / "experiments", "walk")
    valid = next(row for row in payload["series"] if row["key"] == "valid")
    assert "20230403" in valid["dates"]
    assert payload["missing"] == {"forward": ["2023Q2"]}


def test_static_console_keeps_macro_style_surfaces_without_closed_capabilities(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    page = client.get("/").text
    script = client.get("/static/app.js").text
    assert "ADM-Cube" in page and "/static/logo.png" in page
    for label in ("系统提示词预览", "Paper 模拟交易"):
        assert label in script
    for label in (
        "验证期日度累计收益 vs 沪深300（含回撤）",
        "总耗时",
        "Step 产物树",
        "Fold 策略分析",
    ):
        assert label in script
    # The console has no research-report surface.
    assert "研究报告" not in script
    assert "实时 Agent Trace" in script and "/trace/download" in script
    assert "实盘交易" in page and "实盘交易" in script
    assert "后端未连接" in script
    assert "仅前端" not in script
    qmt_source = script.split("function renderQmtPage()", 1)[1]
    assert not any(
        text in qmt_source
        for text in (
            "连接不可用",
            "交易不可用",
            "暂无可显示",
            "未连接 QMT 后端",
            "委托（不可用）",
            "查询与下单",
        )
    )
    assert 'el("div", { class: "empty" }, "后端未连接")' in qmt_source
    assert "status.awaiting_question" not in script
    assert "status.awaiting_step" not in script
    assert "gpus" in client.get("/api/gpus").json()


def test_home_progress_uses_csp_compatible_native_control(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    script = client.get("/static/app.js").text
    stylesheet = client.get("/static/style.css").text
    card_source = script.split("function experimentCard(item)", 1)[1].split(
        "function pickBestExperiment(list)", 1
    )[0]

    # The native control is still the el("progress", ...) call; app.js is now
    # prettier-formatted, so match across the multi-line argument list.
    assert re.search(r'el\(\s*"progress",', card_source)
    assert "value: progressValue" in card_source
    assert "max: numericTotal" in card_source
    assert '"aria-label"' in card_source and '"aria-valuetext"' in card_source
    assert "style:" not in card_source
    assert ".progress > div" not in stylesheet
    for selector in (
        "progress.progress::-webkit-progress-bar",
        "progress.progress::-webkit-progress-value",
        "progress.progress::-moz-progress-bar",
        "progress.progress.done::-webkit-progress-value",
        "progress.progress.done::-moz-progress-bar",
    ):
        assert selector in stylesheet


def test_gpu_allocation_bar_uses_csp_compatible_native_control(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    script = client.get("/static/app.js").text
    stylesheet = client.get("/static/style.css").text
    source = script.split("function gpuAllocationRow(", 1)[1].split(
        "async function sendControlAction(", 1
    )[0]
    assert re.search(r'el\(\s*"progress",', source)
    assert "value: gpu.memory_free_mib" in source
    assert "max: gpu.memory_total_mib" in source
    assert "style:" not in source.split("const renderGpus", 1)[1]
    assert ".gpu-bar > span" not in stylesheet
    assert 'select.value === "" ? experimentDefault' in source


def test_style_api_rejects_result_reference_outside_experiment(tmp_path: Path):
    directory = _persistent_experiment(tmp_path)
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    record = ledger.read()[0]
    outside = tmp_path / "outside/result.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    (outside.parent / "style_analysis.json").write_text(
        json.dumps({"schema_version": 1, "mode": "valid"}), encoding="utf-8"
    )
    record["steps"][0]["validation_result_ref"] = str(outside)
    ledger.rewrite([record])

    response = TestClient(create_app(tmp_path)).get(
        "/api/experiments/demo/style",
        params={"run_id": _run_ref(directory, "run_001"), "prefix": "valid"},
    )
    assert response.status_code == 404


def test_qmt_backend_is_absent_and_loopback_validation_is_single_source(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/trading/live/health").status_code == 404
    assert client.get("/api/trading/sim/health").status_code == 404
    assert is_loopback_host("127.0.0.2")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("example.test")


def test_worker_liveness_rejects_unstamped_or_recycled_process_ids():
    pid = os.getpid()
    ticks = proc_start_ticks(pid)
    assert ticks is not None
    assert status_pid_alive({"pid": pid, "pid_start_ticks": ticks})
    assert not status_pid_alive({"pid": pid})
    assert not status_pid_alive({"pid": pid, "pid_start_ticks": ticks + 1})


def test_public_identity_keeps_cjk_slash_lists_and_redacts_host_paths(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "experiments/demo"
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "schedule.json").write_text(
        json.dumps({"schema_version": 1, "sessions": []}), encoding="utf-8"
    )
    identity = PublicIdentity(directory)
    omitted = "[host path omitted]"
    chinese_list = (
        "原型 + 机械测试通过（30买/无churn/卖出路径/冷热同单/严格JSON）。"
    )
    buy_sell = "买入路径与卖出路径均保留，冷热同单。"
    assert identity.public_text(chinese_list) == chinese_list
    assert identity.public_text(buy_sell) == buy_sell
    assert identity.public_status({"error": chinese_list})["error"] == chinese_list

    assert identity.public_text("打开/Data2/lzp/secret") == f"打开{omitted}"
    assert identity.public_text("中文/mnt/private/secret") == f"中文{omitted}"
    assert identity.public_text("/var/lib/private/result.json") == omitted
    assert identity.public_text("file:///tmp/private/result.json") == omitted
    assert identity.public_text("file://server/share/private.txt") == omitted
    assert identity.public_text(r"C:\Users\private\result.json") == omitted
    assert identity.public_text("/mnt/private/secret") == omitted
    assert identity.public_text("/无churn/secret") == omitted

    kept = identity.public_text(
        "see https://example.test/docs/path and/or 3/4 relative/path; "
        "GET /api/experiments/x; /mnt/agent/workspace/main.py; "
        "/mnt/artifacts/a; /mnt/snapshot/s; /mnt/snapshots/t; "
        "python /mnt/tools/screen.py --signal notes/idea.py"
    )
    for token in (
        "https://example.test/docs/path",
        "and/or",
        "3/4",
        "relative/path",
        "/api/experiments/x",
        "/mnt/agent/workspace/main.py",
        "/mnt/artifacts/a",
        "/mnt/snapshot/s",
        "/mnt/snapshots/t",
        "python /mnt/tools/screen.py --signal notes/idea.py",
    ):
        assert token in kept
    assert omitted not in kept

    # Two segments is the whole threshold: the shortest host path still goes.
    assert identity.public_text("cat /srv/dump.log 后再看") == f"cat {omitted} 后再看"
    # A single-segment token is prose or arithmetic, never a host path.
    formulas = identity.public_text(
        'k_mid=(C-O)/C；z=(x-mean)/std；pd.read_parquet(asof_dir + "/daily")'
    )
    assert omitted not in formulas
    # An allow-listed sandbox root followed by CJK must not be absorbed.
    assert identity.public_text("读 /mnt/snapshot。pandas") == "读 /mnt/snapshot。pandas"
    # Container FHS frames stay readable: they are image-fixed, not host identity.
    traceback_frame = "/usr/local/lib/python3.11/x.py:12: RuntimeWarning"
    assert identity.public_text(traceback_frame) == traceback_frame
    assert identity.public_text("2>/dev/null 会丢证据") == "2>/dev/null 会丢证据"


class WebuiBackendTest(unittest.TestCase):
    """HITL console backend: registry read-models, lifecycle guards, API routes.

    No worker subprocesses, Docker, or LLM calls: worker spawn is patched out
    and experiment state is synthesized on disk exactly as the orchestrator
    writes it.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.experiments_root.mkdir(parents=True)
        self._build_hitl_experiment("exp_hitl")
        self.app = create_app(self.repo_root, self.experiments_root)
        self.client = TestClient(self.app)

    def _identity(self, experiment_id: str = "exp_hitl") -> PublicIdentity:
        return PublicIdentity(self.experiments_root / experiment_id)

    def _fold_ref(self, raw_fold_id: str, experiment_id: str = "exp_hitl") -> str:
        return self._identity(experiment_id).fold_ref(raw_fold_id)

    def _run_ref(self, raw_run_id: str, experiment_id: str = "exp_hitl") -> str:
        return self._identity(experiment_id).run_ref(raw_run_id)

    def _session_ref(
        self, raw_session_key: str, experiment_id: str = "exp_hitl"
    ) -> str:
        return self._identity(experiment_id).public_session_key(raw_session_key)

    # ---- fixtures ------------------------------------------------------------
    def _build_hitl_experiment(self, experiment_id: str) -> Path:
        experiment_dir = self.experiments_root / experiment_id
        AgentRefStore(experiment_dir)
        hitl = experiment_dir / "hitl"
        hitl.mkdir(parents=True)
        write_json_atomic(
            hitl / "params.json",
            {
                "experiment_id": experiment_id,
                **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                **RESEARCH_PARAMS,
                "_created_at": "2026-07-06T00:00:00+00:00",
            },
        )
        write_control(hitl / "control.json", ControlState(mode="manual"))
        write_json_atomic(
            hitl / "status.json",
            {
                "schema_version": 1,
                "pid": 999_999_999,
                "state": "running_session",
                "session_key": "epoch_001/fold_2022Q2",
            },
        )
        write_json_atomic(
            hitl / "schedule.json",
            {
                "schema_version": 1,
                "epochs": 1,
                "sessions": [
                    {
                        "key": "epoch_001/meta_learning",
                        "kind": "meta_learning",
                        "epoch_id": "epoch_001",
                    },
                    {
                        "key": "epoch_001/fold_2022Q1",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022Q1",
                    },
                    {
                        "key": "epoch_001/fold_2022Q2",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022Q2",
                    },
                    {
                        "key": "heldout",
                        "kind": "heldout",
                        "epoch_id": "epoch_001",
                        "periods": [],
                    },
                ],
            },
        )
        strategy_dir = (
            experiment_dir
            / "artifacts"
            / "strategy"
            / "frozen"
            / "strategy_epoch_001_fold_2022Q1"
            / "output"
        )
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "main.py").write_text(
            "def generate_orders(context):\n    return []\n", encoding="utf-8"
        )
        prior_text = "fixture PRIOR\n"
        valid_result = (
            experiment_dir
            / "artifacts"
            / "run_001"
            / "results"
            / VALID_RESULT_DIR
            / "result.json"
        )
        valid_result.parent.mkdir(parents=True)
        valid_result.write_text(
            json.dumps(
                {
                    "equity_curve": [
                        {
                            "trade_date": "20211001",
                            "initial_equity": 100.0,
                            "equity": 110.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        # The frozen Test replay behind the ledger's test_result: the console's
        # cumulative returns are the final value of the chained daily series,
        # so the summary and the artifact have to tell the same story (+0.20).
        test_result = valid_result.parent.parent / TEST_RESULT_DIR / "result.json"
        test_result.parent.mkdir(parents=True)
        test_result.write_text(
            json.dumps(
                {
                    "equity_curve": [
                        {
                            "trade_date": "20220104",
                            "initial_equity": 100.0,
                            "equity": 120.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        _write_ledger(
            experiment_dir,
            [
                {
                    "record_type": "meta_learning",
                    "experiment_id": experiment_id,
                    "epoch_id": "epoch_001",
                    "meta_learning_id": "epoch_001",
                    "trigger_after_folds": 0,
                    "fold_id": "epoch_001_meta_learning",
                    "run_id": "run_meta",
                    "session_key": "epoch_001/meta_learning",
                    "prior": prior_text,
                },
                {
                    "record_type": "fold",
                    "experiment_id": experiment_id,
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_2022Q1",
                    "run_id": "run_001",
                    "session_key": "epoch_001/fold_2022Q1",
                    "fold_status": "frozen",
                    "validation_period": "20211001..20211231",
                    "test_period": "20220101..20220331",
                    "test_decision_time": "2021-12-31T23:59:59+08:00",
                    "frozen_strategy_artifact_id": "strategy_epoch_001_fold_2022Q1",
                    "frozen_strategy_artifact_path": str(strategy_dir),
                    "validation_result": {
                        "total_return": 0.10,
                        "sharpe": 1.0,
                        "max_drawdown": 0.05,
                        "long_return": 0.08,
                        "sub_windows": [_SUB_WINDOW],
                    },
                    "test_result": {
                        "total_return": 0.20,
                        "sharpe": 1.5,
                        "max_drawdown": 0.04,
                        "long_return": 0.15,
                        "sub_windows": [_SUB_WINDOW],
                    },
                    "selected_step_id": "step_001",
                    "steps": [
                        {
                            "step_id": "step_000",
                            "revision_id": "revision_000",
                            "validation_result_ref": str(valid_result),
                        },
                        {
                            "step_id": "step_001",
                            "revision_id": "revision_001",
                            "validation_result_ref": str(valid_result),
                        },
                    ],
                },
                {
                    "record_type": "heldout",
                    "experiment_id": experiment_id,
                    "epoch_id": "epoch_001",
                    "fold_id": "heldout_2023Q1",
                    "run_id": "run_heldout",
                    "session_key": "heldout",
                    "period": "2023Q1",
                    "result": {
                        "total_return": -0.03,
                        "sharpe": -0.2,
                        "max_drawdown": 0.08,
                    },
                },
            ],
        )
        orders_dir = experiment_dir / "artifacts" / "run_001" / "results" / VALID_RESULT_DIR
        pd.DataFrame(
            [
                {
                    "order_id": "o1",
                    "account": "stock",
                    "ts_code": "000001.SZ",
                    "action": "buy",
                    "requested_amount": 500,
                    "filled_quantity": 500,
                    "price": 10.0,
                    "status": "filled",
                    "reject_reason": "",
                    "decision_time": "09:32",
                    "trade_date": "20220104",
                },
                {
                    "order_id": "o2",
                    "account": "stock",
                    "ts_code": "000001.SZ",
                    "action": "sell",
                    "requested_amount": 500,
                    "filled_quantity": 500,
                    "price": 11.0,
                    "status": "filled",
                    "reject_reason": "",
                    "decision_time": "10:00",
                    "trade_date": "20220105",
                },
                {
                    "order_id": "o3",
                    "account": "stock",
                    "ts_code": "600000.SH",
                    "action": "buy",
                    "requested_amount": 200,
                    "filled_quantity": 0,
                    "price": None,
                    "status": "rejected",
                    "reject_reason": "limit_up_blocked_buy",
                    "decision_time": "09:33",
                    "trade_date": "20220104",
                },
            ]
        ).to_parquet(orders_dir / "orders.parquet", index=False)
        trace_dir = experiment_dir / "artifacts" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        events = [
            {
                "event_type": "llm_call",
                "seq": 0,
                "run_id": "run_001",
                "fold_id": "fold_2022Q1",
                "content": (
                    f"run_001 inspected fold_2022Q1 under {experiment_dir}"
                ),
                "usage": {
                    "total_tokens": 1000,
                    "prompt_tokens": 800,
                    "completion_tokens": 200,
                },
            },
            {
                "event_type": "llm_call",
                "seq": 1,
                "content": "/mnt/agent/output uses run_001",
                "usage": {
                    "total_tokens": 2000,
                    "prompt_tokens": 1500,
                    "completion_tokens": 500,
                },
            },
            {"event_type": "tool_call", "seq": 2, "tool": "shell"},
        ]
        (trace_dir / "run_001.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        analysis_dir = hitl / "analysis"
        analysis_dir.mkdir()
        fold_ref = PublicIdentity(experiment_dir).fold_ref("fold_2022Q1")
        (analysis_dir / f"epoch_001__{fold_ref}.md").write_text(
            "## 策略逻辑概述\nfold_2022Q1 / run_001\n",
            encoding="utf-8",
        )
        return experiment_dir

    def _build_walk_forward_experiment(self, experiment_id: str) -> Path:
        """Four yearly Folds of one Epoch, exactly as the pipeline records them.

        The first Fold of the first Epoch inherits nothing and has no control;
        the three after it open with the previous Fold's frozen strategy
        replayed on their own Validation window (one beats its benchmark, one
        does not, one crashed in the strategy's own code), so the Epoch has
        three walk-forward transitions of which one is positive — short of the
        two-thirds term (b) requires.
        """
        experiment_dir = self.experiments_root / experiment_id
        AgentRefStore(experiment_dir)
        hitl = experiment_dir / "hitl"
        hitl.mkdir(parents=True)
        write_json_atomic(
            hitl / "params.json",
            {
                "experiment_id": experiment_id,
                **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                "_created_at": "2026-07-06T00:00:00+00:00",
            },
        )
        write_control(hitl / "control.json", ControlState(mode="manual"))
        write_json_atomic(
            hitl / "schedule.json",
            {
                "schema_version": 1,
                "epochs": 1,
                "sessions": [
                    *(
                        session
                        for year in ("2022", "2023", "2024", "2025")
                        for session in (
                            {
                                "key": f"epoch_001/meta_learning#{year}",
                                "kind": "meta_learning",
                                "epoch_id": "epoch_001",
                            },
                            {
                                "key": f"epoch_001/fold_{year}",
                                "kind": "fold",
                                "epoch_id": "epoch_001",
                                "fold_id": f"fold_{year}",
                            },
                        )
                    ),
                    {
                        "key": "heldout",
                        "kind": "heldout",
                        "epoch_id": "epoch_001",
                        "periods": [],
                    },
                ],
            },
        )
        controls = {
            "2022": None,
            # Beat the benchmark: a positive transition.
            "2023": {
                "status": "ok",
                "parent_strategy_artifact_id": "strategy_epoch_001_fold_2022",
                "step_id": "step_control",
                "validation_result": {
                    "total_return": 0.08,
                    "sharpe": 0.60,
                    "max_drawdown": 0.07,
                    "benchmark": {
                        "benchmark_return": 0.03,
                        "neutralized_excess_return": 0.05,
                    },
                },
            },
            # Lost to the benchmark. A trailing window, so the console must
            # read the new period alone and its own null control, not the
            # window's flattering numbers.
            "2024": {
                "status": "ok",
                "parent_strategy_artifact_id": "strategy_epoch_001_fold_2023",
                "step_id": "step_control",
                "validation_result": {
                    "total_return": 0.30,
                    "sharpe": 1.50,
                    "max_drawdown": 0.09,
                    "benchmark": {"benchmark_return": 0.04},
                },
                "step_result": {
                    "label": "2024Q4",
                    "start": "20241008",
                    "end": "20241231",
                    "total_return": 0.01,
                    "sharpe": 0.10,
                    "max_drawdown": 0.09,
                    "benchmark": {
                        "benchmark_return": 0.04,
                        "neutralized_excess_return": -0.03,
                    },
                },
                "null_control": {
                    "k": 500,
                    "excess_percentile": 0.99,
                    "step": {"excess_percentile": 0.42},
                },
            },
            # The parent's own code crashed: a completed, non-positive transition.
            "2025": {
                "status": "failed",
                "failure": "strategy_error",
                "parent_strategy_artifact_id": "strategy_epoch_001_fold_2024",
                # Host-generated text, host path and all: the console must
                # publish it the way the Agent reads it, not verbatim.
                "error": (
                    "BacktestError: generate_orders failed: KeyError 'close' "
                    "(/srv/experiments/exp_wf/steps/step_control.log)"
                ),
            },
        }
        # How wide each Fold's search was and how much of the frozen
        # candidate's Sharpe that width alone explains; the last Fold ran a
        # single candidate, which leaves the probability unavailable.
        selections = {
            year: {
                "candidates_evaluated": candidates,
                # 2024 ran three candidates but one carried no finite Sharpe,
                # so the formula's N is smaller than the search width.
                "trials": candidates - 1 if year == "2024" else candidates,
                "deflated_sharpe_probability": probability,
                "sharpe_star": 0.42 if probability is not None else None,
                "observed_sharpe": 1.0,
                "unavailable_reason": (
                    None if probability is not None else "fewer_than_two_trials"
                ),
            }
            for year, candidates, probability in (
                ("2022", 6, 0.81),
                ("2023", 4, 0.55),
                ("2024", 3, 0.12),
                ("2025", 1, None),
            )
        }
        records: list[dict[str, object]] = [
            {
                "record_type": "fold",
                "experiment_id": experiment_id,
                "epoch_id": "epoch_001",
                "fold_id": f"fold_{year}",
                "run_id": f"run_{year}",
                "session_key": f"epoch_001/fold_{year}",
                "fold_status": "frozen",
                "validation_period": f"{year}0104..{year}1230",
                "test_period": None,
                "test_result": None,
                "parent_control": controls[year],
                "selection_statistics": selections[year],
                "validation_result": {
                    "total_return": 0.10,
                    "sharpe": 1.0,
                    "max_drawdown": 0.05,
                    "benchmark": {"benchmark_return": 0.04, "excess_return": 0.06},
                },
                "steps": [],
            }
            for year in ("2022", "2023", "2024", "2025")
        ]
        heldout_result = {
            "total_return": 0.05,
            "sharpe": 0.80,
            "max_drawdown": 0.10,
            "benchmark": {"benchmark_return": 0.02, "neutralized_excess_return": 0.01},
        }
        records.append(
            {
                "record_type": "heldout",
                "experiment_id": experiment_id,
                "epoch_id": "epoch_001",
                "fold_id": "heldout_2026",
                "run_id": "run_heldout",
                "session_key": "heldout",
                "period": "2026",
                "result": heldout_result,
                # The verdict the pipeline stamps: terms (a) and (c) pass, term
                # (b) does not, so the failing reason is the walk-forward one.
                "verdict": AcceptanceRules().heldout_verdict(
                    heldout_result,
                    {
                        "source": "parent_control",
                        "epoch_id": "epoch_001",
                        "transitions": 3,
                        "positive_excess": 1,
                    },
                    # Selection evidence of the Fold that froze the shipped
                    # artifact, shaped exactly as ledger.frozen_selection
                    # returns it — including the raw Fold token the console
                    # must never echo back out.
                    {
                        "fold_id": "fold_2023",
                        "candidates_evaluated": 4,
                        "deflated_sharpe_probability": 0.55,
                        "deflated_sharpe_trials": 4,
                        "validation_excess_percentile": 0.61,
                    },
                    # The shipped artifact's own share of those transitions,
                    # shaped exactly as ledger.final_artifact_transitions
                    # returns it to run_heldout: the last Fold kept the parent,
                    # so the artifact Held-out replays was confirmed forward
                    # twice before it got there.
                    {
                        "artifact_id": "strategy_epoch_001_fold_2023",
                        "epoch_id": "epoch_001",
                        "transitions": 2,
                        "positive_excess": 2,
                        "failed": 0,
                        "unmeasured": 0,
                    },
                ),
            }
        )
        _write_ledger(experiment_dir, records)
        return experiment_dir

    def _reveal(self, experiment_id: str = "exp_hitl") -> None:
        response = self.client.post(
            f"/api/experiments/{experiment_id}/control",
            json={"action": "reveal_test_results"},
        )
        self.assertEqual(response.status_code, 200, response.text)

    # ---- schema & listing ----------------------------------------------------
    def test_frontend_assets_use_clean_urls_and_revalidate(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn('src="/static/app.js"', index.text)
        self.assertNotIn("app.js?v=", index.text)

    def test_parameter_schema_defaults_track_worker_defaults(self) -> None:
        schema = self.client.get("/api/parameter-schema").json()
        fields = {
            field["key"]: field
            for group in schema["groups"]
            for field in group["fields"]
        }
        self.assertEqual(
            fields["research_sessions"]["default"], WEB_CREATE_DEFAULTS["research_sessions"]
        )
        self.assertEqual(fields["model"]["default"], WEB_CREATE_DEFAULTS["model"])
        for hidden in (
            "experiments_root",
            "work_root",
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "template_dir",
            "local_dev",
        ):
            self.assertNotIn(hidden, fields, hidden)
        for model_field in ("model", "subagent_model", "nl_model", "compact_model"):
            self.assertNotIn("deepseek-chat", fields[model_field]["choices"])
            self.assertNotIn("deepseek-reasoner", fields[model_field]["choices"])
        visible_copy = "\n".join(
            str(field.get(key, ""))
            for field in fields.values()
            for key in ("label", "help")
        )
        self.assertNotIn("DeepSeek", visible_copy)
        self.assertNotIn("provider", visible_copy)
        self.assertEqual(fields["no_thinking"]["label"], "禁用推理模式")
        # The geometry dates are plain YYYYMMDD strings the worker validates.
        self.assertEqual(fields["research_start"]["type"], "string")
        for retired in ("meta_learning_directive", "meta_learning_fold_interval", "meta_model"):
            self.assertNotIn(retired, fields)
        self.assertEqual(fields["fold_exploration_directive"]["type"], "text")
        self.assertEqual(fields["fold_exploration_directive"]["default"], "")
        self.assertTrue(fields["fold_exploration_directive"]["wide"])
        self.assertTrue(
            {
                "auction_enabled",
                "auction_preopen_time",
                "auction_decision_time",
                "auction_close_time",
            }.isdisjoint(fields)
        )

    def test_public_params_never_echo_hidden_keys(self) -> None:
        # The console API refuses HIDDEN_KEYS at creation, but params.json is
        # also a worker-side ops channel — the read model must not echo them.
        from autotrade.webui.registry import _public_params

        # Operator-only keys: source roots, manager-owned roots and credential
        # env *names*. The API refuses them at creation, but params.json is
        # also a worker-side ops channel where they legitimately exist.
        operator_only = (
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "experiments_root",
            "work_root",
            "llm_api_key_env",
            "llm_env_file",
            "llm_base_url",
        )
        # Only the held-out calendar is sealed; the development window is
        # public research scope the session labels already carry.
        development = {
            "development_first_period": "2024Q1",
            "development_last_period": "2024Q4",
        }
        sealed_periods = {
            "heldout_first_period": "2025Q1",
            "heldout_last_period": "2025Q4",
        }
        params = {
            "model": "deepseek-v4-pro",
            **development,
            **sealed_periods,
            **{key: f"secret-{key}" for key in operator_only},
        }
        public = _public_params(params, test_revealed=False)
        self.assertEqual(public, {"model": "deepseek-v4-pro", **development})
        self.assertEqual(
            _public_params(params, test_revealed=True),
            {"model": "deepseek-v4-pro", **development, **sealed_periods},
        )

    def test_historical_endpoint_is_absent_from_list_and_detail_api(self) -> None:
        params_path = self.experiments_root / "exp_hitl/hitl/params.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        private_endpoint = "https://historical-private.example.test/v1"
        params["llm_base_url"] = private_endpoint
        write_json_atomic(params_path, params)

        listed = self.client.get("/api/experiments")
        detail = self.client.get("/api/experiments/exp_hitl")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn(private_endpoint, listed.text)
        self.assertNotIn(private_endpoint, detail.text)
        self.assertNotIn("llm_base_url", detail.json()["params"])

    # ---- reveal / seal --------------------------------------------------------
    def test_reveal_seals_learning_actions(self) -> None:
        self._reveal()
        # `resume` restarts learning on a sealed experiment, so it is blocked
        # together with the rest of the learning-affecting control set.
        for action in (
            "rerun_fold",
            "rollback_fold",
            "set_directive",
            "resume",
        ):
            refused = self.client.post(
                "/api/experiments/exp_hitl/control",
                json={
                    "action": action,
                    "session_key": self._session_ref("epoch_001/fold_2022Q2"),
                    "directive": "x",
                },
            )
            self.assertEqual(refused.status_code, 400, action)
            self.assertIn("封存", refused.json()["detail"])
        # Lifecycle controls stay available on a sealed experiment.
        ok = self.client.post(
            "/api/experiments/exp_hitl/control", json={"action": "stop"}
        )
        self.assertEqual(ok.status_code, 200)

    def test_the_sealed_test_calendar_stays_sealed_until_the_reveal(self) -> None:
        """The fold record names the window the Test evaluation will use.

        ``_public_params`` and ``public_session`` already refuse to publish
        those dates; a fold record that carried them anyway would hand out the
        same calendar through a different route.
        """
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}"
        record = self.client.get(url).json()["record"]
        self.assertNotIn("test_period", record)
        self.assertNotIn("test_decision_time", record)
        # The Validation window is public and stays visible.
        self.assertEqual(record["validation_period"], "20211001..20211231")
        self._reveal()
        revealed = self.client.get(url).json()["record"]
        self.assertEqual(revealed["test_period"], "20220101..20220331")
        self.assertEqual(
            revealed["test_decision_time"], "2021-12-31T23:59:59+08:00"
        )

    def test_sub_windows_ride_with_the_result_they_belong_to(self) -> None:
        """The per-quarter breakdown follows its result through the same gate:
        Validation is public, the Test copy only exists after the reveal."""
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}"
        detail = self.client.get(url).json()
        rows = detail["record"]["validation_result"]["sub_windows"]
        self.assertEqual(rows[0]["label"], "2022Q1")
        self.assertEqual(detail["test_audit"], {"hidden": True})
        self._reveal()
        revealed = self.client.get(url).json()
        self.assertEqual(
            revealed["test_audit"]["test_result"]["sub_windows"][0]["label"],
            "2022Q1",
        )

    def test_fold_detail_separates_test_audit_from_record(self) -> None:
        fold_ref = self._fold_ref("fold_2022Q1")
        detail = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}"
        ).json()
        self.assertNotIn("test_result", detail["record"])
        # Hidden until revealed; revealing seals the experiment.
        self.assertEqual(detail["test_audit"], {"hidden": True})
        self._reveal()
        detail = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}"
        ).json()
        self.assertEqual(detail["test_audit"]["test_result"]["total_return"], 0.20)
        # Downloads are ZIP-only: no per-file listing or file endpoint.
        self.assertNotIn("strategy_files", detail)
        self.assertTrue(detail["analysis"]["available"])

    def test_style_route_gated_until_reveal(self) -> None:
        run_ref = self._run_ref("run_001")
        missing_run_ref = self._run_ref("run_missing")
        results = (
            self.experiments_root / "exp_hitl" / "artifacts" / "run_001" / "results"
        )
        for prefix in (VALID_RESULT_DIR, TEST_RESULT_DIR):
            directory = results / prefix
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.json").write_text("{}", encoding="utf-8")
            style_mode = prefix.rsplit("_", maxsplit=1)[0]
            (directory / "style_analysis.json").write_text(
                json.dumps({"schema_version": 1, "mode": style_mode}),
                encoding="utf-8",
            )
        record_path = (
            self.experiments_root / "exp_hitl" / "ledgers" / "experiment_ledger.jsonl"
        )
        records = [
            json.loads(line)
            for line in record_path.read_text(encoding="utf-8").splitlines()
        ]
        for record in records:
            if record["record_type"] == "fold":
                record["test_result_ref"] = str(results / TEST_RESULT_DIR / "result.json")
        record_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        url = "/api/experiments/exp_hitl/style"
        valid = self.client.get(url, params={"run_id": run_ref, "prefix": "valid"})
        self.assertEqual(valid.status_code, 200, valid.text)
        self.assertEqual(valid.json()["mode"], "valid")
        hidden = self.client.get(url, params={"run_id": run_ref, "prefix": "test"})
        self.assertEqual(hidden.status_code, 404)
        # Indistinguishable from a run without a rollup: existence must not leak.
        absent = self.client.get(
            url, params={"run_id": missing_run_ref, "prefix": "valid"}
        )
        self.assertEqual(hidden.json()["detail"], absent.json()["detail"])
        self._reveal()
        revealed = self.client.get(url, params={"run_id": run_ref, "prefix": "test"})
        self.assertEqual(revealed.status_code, 200, revealed.text)
        self.assertEqual(revealed.json()["mode"], "frozen_test")

    def test_fold_orders_gated_until_reveal(self) -> None:
        results = (
            self.experiments_root / "exp_hitl" / "artifacts" / "run_001" / "results"
        )
        test_dir = results / TEST_RESULT_DIR
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "result.json").write_text(
            json.dumps(
                {
                    "executions": [
                        {
                            "symbol": "000001.SZ",
                            "action": "buy",
                            "quantity": 100,
                            "execute_at": "2022-04-01T09:30:00+08:00",
                            "status": "filled",
                            "price": 9.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders"
        hidden = self.client.get(url, params={"result": TEST_RESULT_DIR})
        self.assertEqual(hidden.status_code, 404)
        listing = self.client.get(url).json()
        self.assertEqual(listing["result"], VALID_RESULT_DIR)
        # The visible enumeration must not leak the sealed result's existence.
        self.assertEqual(listing["available"], [VALID_RESULT_DIR])
        csv_hidden = self.client.get(url + ".csv", params={"result": TEST_RESULT_DIR})
        self.assertEqual(csv_hidden.status_code, 404)
        self._reveal()
        revealed = self.client.get(url, params={"result": TEST_RESULT_DIR})
        self.assertEqual(revealed.status_code, 200, revealed.text)
        self.assertEqual(revealed.json()["result"], TEST_RESULT_DIR)
        # A revealed test result is served on request but never enters
        # `available`: the default selection reads that list, so a test result
        # can never silently become the pane's default.
        listing = self.client.get(url).json()
        self.assertEqual(listing["available"], [VALID_RESULT_DIR])
        self.assertEqual(listing["result"], VALID_RESULT_DIR)
        csv_ok = self.client.get(url + ".csv", params={"result": TEST_RESULT_DIR})
        self.assertEqual(csv_ok.status_code, 200)
        self.assertEqual(len(csv_ok.text.strip().splitlines()), 2)  # header + 1 order

    def test_fold_detail_exposes_real_test_result_id(self) -> None:
        """The console links the test-period order export by the id the
        read-model serves: result directories are named frozen_test_<uuid>, so
        any guessed name would only ever 404."""

        results = (
            self.experiments_root / "exp_hitl" / "artifacts" / "run_001" / "results"
        )
        test_dir = results / TEST_RESULT_DIR
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "result.json").write_text(
            json.dumps(
                {
                    "executions": [
                        {
                            "symbol": "000001.SZ",
                            "action": "buy",
                            "quantity": 100,
                            "execute_at": "2022-04-01T09:30:00+08:00",
                            "status": "filled",
                            "price": 9.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        ledger = (
            self.experiments_root / "exp_hitl" / "ledgers" / "experiment_ledger.jsonl"
        )
        records = [
            json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
        ]
        for record in records:
            if record["record_type"] == "fold":
                record["test_result_ref"] = str(test_dir / "result.json")
        ledger.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}"
        self.assertEqual(self.client.get(url).json()["test_audit"], {"hidden": True})
        self._reveal()
        audit = self.client.get(url).json()["test_audit"]
        self.assertEqual(audit["result"], TEST_RESULT_DIR)
        export = self.client.get(
            f"{url}/orders.csv", params={"result": audit["result"]}
        )
        self.assertEqual(export.status_code, 200, export.text)

    def test_fold_orders_rows_and_csv_export(self) -> None:
        results = (
            self.experiments_root
            / "exp_hitl"
            / "artifacts"
            / "run_001"
            / "results"
            / VALID_RESULT_DIR
        )
        (results / "result.json").write_text(
            json.dumps(
                {
                    "executions": [
                        {
                            "symbol": "000001.SZ",
                            "action": "buy",
                            "quantity": 500,
                            "execute_at": "2022-01-04T09:32:00+08:00",
                            "status": "filled",
                            "price": 10.0,
                        },
                        {
                            "symbol": "000001.SZ",
                            "action": "sell",
                            "quantity": 500,
                            "execute_at": "2022-01-05T10:00:00+08:00",
                            "status": "filled",
                            "price": 11.0,
                        },
                        {
                            "symbol": "600000.SH",
                            "action": "buy",
                            "quantity": 200,
                            "execute_at": "2022-01-04T09:33:00+08:00",
                            "status": "rejected",
                            "reason": "limit_up_blocked_buy",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        fold_ref = self._fold_ref("fold_2022Q1")
        data = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders"
        ).json()
        self.assertEqual(data["result"], VALID_RESULT_DIR)
        self.assertEqual(data["row_count"], 3)
        self.assertEqual(
            [row["action"] for row in data["rows"]], ["buy", "sell", "buy"]
        )
        self.assertEqual(data["rows"][2]["reason"], "limit_up_blocked_buy")
        csv_response = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders.csv",
            params={"result": VALID_RESULT_DIR},
        )
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("attachment", csv_response.headers.get("content-disposition", ""))
        self.assertEqual(
            len(csv_response.text.strip().splitlines()), 4
        )  # header + 3 orders
        missing = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders.csv",
            params={"result": "nope"},
        )
        self.assertEqual(missing.status_code, 404)

    def test_fold_orders_payload_carries_exactly_the_console_key_names(self) -> None:
        """The five-key projection the order pane reads.

        Pinned as an exact key set: a rename in either direction, or an extra
        field nothing consumes, breaks the pane silently otherwise.
        """
        import shutil

        expected_keys = ["available", "result", "row_count", "rows", "stats"]
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders"
        payload = self.client.get(url).json()
        self.assertEqual(sorted(payload), expected_keys)
        self.assertEqual(payload["available"], [VALID_RESULT_DIR])
        # Three orders are far under the 500-row cap.
        self.assertEqual(payload["row_count"], len(payload["rows"]))
        # The empty selection returns the same shape, so the pane never has to
        # branch on which keys are present.
        shutil.rmtree(self.experiments_root / "exp_hitl/artifacts/run_001/results")
        empty = self.client.get(url).json()
        self.assertEqual(sorted(empty), expected_keys)
        self.assertEqual(empty["result"], None)
        self.assertEqual(empty["available"], [])
        self.assertEqual(empty["rows"], [])
        self.assertEqual(empty["row_count"], 0)

    def test_fold_orders_caps_the_table_at_five_hundred_rows_but_not_the_export(
        self,
    ) -> None:
        """`row_count` and the stats describe the whole stream, `rows` is capped.

        The pane renders 「共 M 条」 from `row_count`, so a cap that also
        shrank the count would report a lie; and the CSV is the escape hatch
        from the cap, so it must stay uncapped.
        """
        results = self.experiments_root / f"exp_hitl/artifacts/run_001/results/{VALID_RESULT_DIR}"
        (results / "result.json").write_text(
            json.dumps(
                {
                    "executions": [
                        {
                            "symbol": "000001.SZ",
                            "action": "buy",
                            "quantity": 100,
                            "execute_at": "2022-01-04T09:32:00+08:00",
                            "status": "filled",
                            "price": 10.0,
                        }
                        for _ in range(501)
                    ]
                }
            ),
            encoding="utf-8",
        )
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders"
        payload = self.client.get(url).json()
        self.assertEqual(len(payload["rows"]), 500)
        self.assertEqual(payload["row_count"], 501)
        self.assertEqual(payload["stats"]["orders"], 501)
        self.assertEqual(payload["stats"]["filled"], 501)
        csv_response = self.client.get(url + ".csv", params={"result": VALID_RESULT_DIR})
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(
            len(csv_response.text.strip().splitlines()), 502
        )  # header + 501

    # ---- lifecycle guards ------------------------------------------------------
    def test_corrupt_identity_store_fails_closed_without_host_paths(self) -> None:
        directory = self._build_hitl_experiment("exp_bad_refs")
        store_path = directory / ".host/agent-refs.json"
        store_path.write_text("{broken", encoding="utf-8")

        responses = [
            self.client.get("/api/experiments"),
            self.client.get("/api/experiments/exp_bad_refs"),
            self.client.get("/api/experiments/exp_bad_refs/status"),
            self.client.get("/api/experiments/exp_bad_refs/steps"),
            self.client.post(
                "/api/experiments/exp_bad_refs/control", json={"action": "pause"}
            ),
        ]
        self.assertEqual(responses[0].status_code, 200)
        bad_summary = next(
            row
            for row in responses[0].json()["experiments"]
            if row["experiment_id"] == "exp_bad_refs"
        )
        self.assertEqual(bad_summary["state"], "unreadable")
        self.assertEqual(responses[1].json()["state"], "unreadable")
        self.assertEqual(responses[2].status_code, 409)
        self.assertEqual(responses[3].status_code, 409)
        self.assertEqual(responses[4].status_code, 400)
        for response in responses:
            public_text = response.text
            self.assertNotIn(str(self.repo_root), public_text)
            self.assertNotIn(str(directory), public_text)
        self.assertEqual(store_path.read_text(encoding="utf-8"), "{broken")

    def test_dead_live_session_degrades_to_interrupted(self) -> None:
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": 999_999_999,
                "state": "running_session",
                "session_key": "epoch_001/fold_2022Q2",
            },
        )
        status = self.client.get("/api/experiments/exp_hitl/status").json()
        self.assertEqual(status["state"], "interrupted")
        self.assertFalse(status["worker_alive"])
        self.assertEqual(status["status"]["state"], "running_session")
        self.assertEqual(
            status["status"]["session_key"],
            self._session_ref("epoch_001/fold_2022Q2"),
        )

    def test_strategy_zip_contains_output_tree(self) -> None:
        fold_ref = self._fold_ref("fold_2022Q1")
        response = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/strategy.zip"
        )
        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        self.assertEqual(archive.namelist(), ["output/main.py"])

    def test_delete_requires_confirm_and_no_live_worker(self) -> None:
        missing_confirm = self.client.delete("/api/experiments/exp_hitl")
        self.assertEqual(missing_confirm.status_code, 400)
        # Simulate a live worker on the HITL experiment (our own pid is alive;
        # liveness requires the recorded kernel start ticks to match).
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "pid_start_ticks": proc_start_ticks(os.getpid()),
                "state": "running_session",
            },
        )
        alive = self.client.delete(
            "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
        )
        self.assertEqual(alive.status_code, 409)
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        gone = self.client.delete(
            "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
        )
        self.assertEqual(gone.status_code, 200)
        self.assertFalse((self.experiments_root / "exp_hitl").exists())

    def test_failed_experiment_with_readonly_hardlinked_artifacts_can_be_deleted(
        self,
    ) -> None:
        directory = self.experiments_root / "exp_readonly"
        hitl = directory / "hitl"
        readonly = directory / "artifacts/results/valid_repro/asof"
        hitl.mkdir(parents=True)
        readonly.mkdir(parents=True)
        write_json_atomic(
            hitl / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        shared = self.repo_root / "shared-cache.parquet"
        shared.write_bytes(b"shared")
        shared.chmod(0o444)
        os.link(shared, readonly / "part.parquet")
        readonly.chmod(0o555)

        response = self.client.delete(
            "/api/experiments/exp_readonly", params={"confirm": "exp_readonly"}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertEqual(shared.read_bytes(), b"shared")
        self.assertEqual(shared.stat().st_mode & 0o777, 0o444)

    def test_delete_filesystem_failure_returns_detail_without_false_success(
        self,
    ) -> None:
        directory = self.experiments_root / "exp_delete_error"
        hitl = directory / "hitl"
        hitl.mkdir(parents=True)
        write_json_atomic(
            hitl / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        with patch(
            "autotrade.webui.manager._remove_readonly_tree",
            side_effect=PermissionError("readonly artifact"),
        ):
            response = self.client.delete(
                "/api/experiments/exp_delete_error",
                params={"confirm": "exp_delete_error"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("was not fully deleted", response.json()["detail"])
        self.assertIn("PermissionError", response.json()["detail"])
        self.assertTrue(directory.exists())

    def test_delete_rejects_derived_sandbox_symlink_without_touching_target(
        self,
    ) -> None:
        directory = self._build_hitl_experiment("exp_sandbox_link")
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        external = self.repo_root / "external-sandbox"
        external.mkdir()
        marker = external / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        expected = self.repo_root / ".runtime/sandboxes/exp_sandbox_link"
        expected.parent.mkdir(parents=True)
        expected.symlink_to(external, target_is_directory=True)

        with patch("autotrade.webui.manager._remove_sandbox_tree") as remove:
            response = self.client.delete(
                "/api/experiments/exp_sandbox_link",
                params={"confirm": "exp_sandbox_link"},
            )

        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("symbolic link", response.json()["detail"])
        remove.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertTrue(directory.exists())

    def test_delete_rejects_symlinked_sandbox_root_without_touching_external_tree(
        self,
    ) -> None:
        experiment_id = "exp_sandbox_root_link"
        directory = self._build_hitl_experiment(experiment_id)
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        runtime_root = self.repo_root / ".runtime"
        runtime_root.mkdir()
        with tempfile.TemporaryDirectory() as external_tmp:
            external = Path(external_tmp)
            outside = external / experiment_id
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (runtime_root / "sandboxes").symlink_to(external, target_is_directory=True)

            with patch("autotrade.webui.manager._remove_sandbox_tree") as remove:
                response = self.client.delete(
                    f"/api/experiments/{experiment_id}",
                    params={"confirm": experiment_id},
                )

            self.assertEqual(response.status_code, 500, response.text)
            self.assertIn("symbolic-link sandbox root", response.json()["detail"])
            remove.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertTrue(directory.exists())

    def test_delete_rejects_symlinked_runtime_root_without_touching_external_tree(
        self,
    ) -> None:
        experiment_id = "exp_runtime_root_link"
        directory = self._build_hitl_experiment(experiment_id)
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        with tempfile.TemporaryDirectory() as external_tmp:
            external = Path(external_tmp)
            outside = external / "sandboxes" / experiment_id
            outside.mkdir(parents=True)
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (self.repo_root / ".runtime").symlink_to(external, target_is_directory=True)

            with patch("autotrade.webui.manager._remove_sandbox_tree") as remove:
                response = self.client.delete(
                    f"/api/experiments/{experiment_id}",
                    params={"confirm": experiment_id},
                )

            self.assertEqual(response.status_code, 500, response.text)
            self.assertIn("symbolic-link runtime root", response.json()["detail"])
            remove.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertTrue(directory.exists())

    def test_delete_with_corrupt_params_uses_only_derived_sandbox_path(self) -> None:
        directory = self._build_hitl_experiment("exp_corrupt_params")
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        explicit = self.repo_root / "explicit-work-root"
        explicit.mkdir()
        marker = explicit / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        (directory / "hitl/params.json").write_text(
            f'{{"work_root": {json.dumps(str(explicit))}, broken',
            encoding="utf-8",
        )
        sandbox = self.repo_root / ".runtime/sandboxes/exp_corrupt_params"
        sandbox.mkdir(parents=True)
        (sandbox / "runtime.txt").write_text("remove", encoding="utf-8")

        response = self.client.delete(
            "/api/experiments/exp_corrupt_params",
            params={"confirm": "exp_corrupt_params"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertFalse(sandbox.exists())
        self.assertEqual(response.json()["removed_work_root"], str(sandbox))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_delete_with_unreadable_params_uses_only_derived_sandbox_path(self) -> None:
        directory = self._build_hitl_experiment("exp_unreadable_params")
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        params_path = directory / "hitl/params.json"
        explicit = self.repo_root / "unreadable-explicit-work-root"
        explicit.mkdir()
        marker = explicit / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        write_json_atomic(params_path, {"work_root": str(explicit)})
        sandbox = self.repo_root / ".runtime/sandboxes/exp_unreadable_params"
        sandbox.mkdir(parents=True)
        (sandbox / "runtime.txt").write_text("remove", encoding="utf-8")
        original_read_text = Path.read_text

        def deny_params(path: Path, *args, **kwargs):
            if path == params_path:
                raise PermissionError("params unreadable")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", new=deny_params):
            response = self.client.delete(
                "/api/experiments/exp_unreadable_params",
                params={"confirm": "exp_unreadable_params"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertFalse(sandbox.exists())
        self.assertEqual(response.json()["removed_work_root"], str(sandbox))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_delete_removes_console_sandbox_tree_under_shared_work_root(self) -> None:
        # Console-created experiments record the shared sandbox root in
        # params.json and the worker runs under <work_root>/<experiment_id>.
        # Deletion must reclaim exactly that subtree and nothing around it.
        experiment_id = "exp_console_layout"
        directory = self._build_hitl_experiment(experiment_id)
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        params_path = directory / "hitl/params.json"
        shared_root = self.repo_root / ".runtime/sandboxes"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        params["work_root"] = str(shared_root)
        write_json_atomic(params_path, params)
        sandbox = shared_root / experiment_id
        (sandbox / "run_0001").mkdir(parents=True)
        (sandbox / "run_0001/runtime.txt").write_text("remove", encoding="utf-8")
        sibling = shared_root / "exp_other_layout"
        sibling.mkdir()
        keep = sibling / "keep.txt"
        keep.write_text("keep", encoding="utf-8")

        response = self.client.delete(
            f"/api/experiments/{experiment_id}",
            params={"confirm": experiment_id},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["removed_work_root"], str(sandbox))
        self.assertFalse(sandbox.exists())
        self.assertFalse(directory.exists())
        self.assertTrue(shared_root.is_dir())
        self.assertEqual(keep.read_text(encoding="utf-8"), "keep")

    def test_delete_rejects_symlink_alias_to_another_experiment(self) -> None:
        target = self.experiments_root / "exp_target"
        target.mkdir()
        (target / "marker").write_text("keep", encoding="utf-8")
        (self.experiments_root / "exp_alias").symlink_to(
            target, target_is_directory=True
        )
        manager = ExperimentManager(self.repo_root, self.experiments_root)

        with self.assertRaisesRegex(ManagerError, "invalid experiment ID"):
            manager.delete_experiment("exp_alias")
        self.assertEqual((target / "marker").read_text(encoding="utf-8"), "keep")

    def test_delete_refused_while_analysis_pending(self) -> None:
        # AnalysisService worker threads keep writing into hitl/analysis/ after
        # their HTTP request returns; deleting the experiment tree under them
        # would race those writes. The server wires the service's pending view
        # into the manager, which must refuse with 409 until the work drains.
        from autotrade.webui.analysis import AnalysisService

        manager = ExperimentManager(
            self.repo_root,
            self.experiments_root,
            analysis_pending=lambda experiment_id: experiment_id == "exp_hitl",
        )
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        with self.assertRaisesRegex(ManagerError, "analysis in progress"):
            manager.delete_experiment("exp_hitl")
        self.assertTrue((self.experiments_root / "exp_hitl").exists())
        # End-to-end through create_app: the wiring exists and maps to 409.
        with patch.object(
            AnalysisService, "pending_for_experiment", return_value=True
        ) as pending:
            client = TestClient(create_app(self.repo_root, self.experiments_root))
            refused = client.delete(
                "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
            )
            self.assertEqual(refused.status_code, 409)
            self.assertIn("analysis in progress", refused.json()["detail"])
            self.assertTrue((self.experiments_root / "exp_hitl").exists())
            pending.return_value = False  # analysis drained -> delete proceeds
            done = client.delete(
                "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
            )
            self.assertEqual(done.status_code, 200)
            self.assertFalse((self.experiments_root / "exp_hitl").exists())

    def test_analysis_endpoint_serves_existing_markdown(self) -> None:
        payload = self.client.get(
            f"/api/experiments/exp_hitl/analysis/epoch_001/{self._fold_ref('fold_2022Q1')}"
        ).json()
        self.assertTrue(payload["available"])
        self.assertIn("策略逻辑概述", payload["content"])
        missing = self.client.get(
            f"/api/experiments/exp_hitl/analysis/epoch_001/{self._fold_ref('fold_2022Q2')}"
        ).json()
        self.assertFalse(missing["available"])

    def test_dataset_coverage_reads_partition_bounds(self) -> None:
        from autotrade.webui.registry import dataset_coverage

        raw = self.repo_root / "data" / "raw"
        (raw / "daily").mkdir(parents=True)
        for day in ("20200102", "20240105"):
            (raw / "daily" / f"trade_date={day}.parquet").write_bytes(b"")
        self.assertEqual(dataset_coverage(raw, "daily"), ("20200102", "20240105"))
        self.assertIsNone(dataset_coverage(raw, "stk_mins_1min_by_date"))

    def test_health_raw_generation_states_drive_status(self) -> None:
        from autotrade.environment.data.contracts import RAW_GENERATION_FILENAME

        # No stamp at all (dev/test roots without a lake): still healthy.
        payload = self.client.get("/api/health").json()
        self.assertEqual(payload["raw_generation"], {"state": "absent"})
        self.assertEqual(payload["status"], "ok")
        stamp = self.repo_root / "data" / "raw" / RAW_GENERATION_FILENAME
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "state": "committed",
                    "generation_id": "gen42",
                    "completed_at": "2026-07-28T04:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        payload = self.client.get("/api/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["raw_generation"]["state"], "committed")
        self.assertEqual(payload["raw_generation"]["generation_id"], "gen42")
        self.assertEqual(
            payload["raw_generation"]["completed_at"], "2026-07-28T04:00:00+00:00"
        )
        # A dirty lake (aborted mutating cron) must degrade health — the
        # 6-day production outage stayed green behind the hardcoded literal.
        stamp.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "state": "dirty",
                    "generation_id": "gen43",
                    "updated_at": "2026-07-23T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        payload = self.client.get("/api/health").json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["raw_generation"]["state"], "dirty")
        self.assertEqual(
            payload["raw_generation"]["updated_at"], "2026-07-23T00:00:00+00:00"
        )
        # A malformed stamp is reported, never a 500.
        stamp.write_text("{oops", encoding="utf-8")
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["raw_generation"]["state"], "unreadable")
        self.assertEqual(response.json()["status"], "degraded")

    def test_assert_no_live_writer_guards_migrations(self) -> None:
        from autotrade.pipelines.hitl_state import assert_no_live_writer

        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "pid_start_ticks": proc_start_ticks(os.getpid()),
                "state": "running_session",
            },
        )
        with self.assertRaises(RuntimeError):
            assert_no_live_writer(self.experiments_root / "exp_hitl")
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        assert_no_live_writer(self.experiments_root / "exp_hitl")

    def test_broken_experiment_is_isolated_from_creation_and_detail(self) -> None:
        broken = self.experiments_root / "exp_broken" / "hitl"
        broken.mkdir(parents=True)
        (broken / "status.json").write_text("{not json", encoding="utf-8")
        listing = {
            entry["experiment_id"]: entry
            for entry in self.client.get("/api/experiments").json()["experiments"]
        }
        # A broken experiment renders as a structured error, never a 500, and
        # stays inspectable and deletable from the console.
        self.assertEqual(listing["exp_broken"]["state"], "unreadable")
        self.assertIn("exp_hitl", listing)
        detail = self.client.get("/api/experiments/exp_broken")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["state"], "unreadable")
        self.assertEqual(detail.json()["sessions"], [])
        health = self.client.get("/api/health").json()
        self.assertIn(
            "exp_broken",
            [item["experiment_id"] for item in health["unreadable_experiments"]],
        )
        self.assertEqual(health["status"], "degraded")
        # A broken neighbour must never block creating a new experiment.
        created = self.client.post(
            "/api/experiments",
            json={
                "params": {
                    "experiment_id": "exp_new",
                    **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                }
            },
        )
        self.assertEqual(created.status_code, 200, created.text)

    def test_running_cap_allows_last_slot_and_blocks_overflow(self) -> None:
        manager = ExperimentManager(self.repo_root, self.experiments_root)
        running = [f"running_{index}" for index in range(MAX_RUNNING_EXPERIMENTS - 1)]
        with (
            patch.object(manager, "running_experiments", return_value=running),
            patch.object(manager, "_preflight"),
            patch.object(manager, "start_worker", return_value={"spawned": False}),
        ):
            created = manager.create_experiment(
                {
                    "experiment_id": "exp_last_slot",
                    **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                }
            )
        self.assertEqual(created["experiment_id"], "exp_last_slot")

        running.append("exp_last_slot")
        with (
            patch.object(manager, "running_experiments", return_value=running),
            self.assertRaisesRegex(
                ManagerError,
                rf"parallel experiment cap reached \({MAX_RUNNING_EXPERIMENTS}\)",
            ),
        ):
            manager.create_experiment(
                {
                    "experiment_id": "exp_overflow",
                    **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                }
            )
        self.assertFalse((self.experiments_root / "exp_overflow").exists())

    def test_running_cap_also_guards_worker_restart(self) -> None:
        manager = ExperimentManager(self.repo_root, self.experiments_root)
        running = [f"running_{index}" for index in range(MAX_RUNNING_EXPERIMENTS)]
        with (
            patch.object(manager, "running_experiments", return_value=running),
            self.assertRaisesRegex(
                ManagerError,
                rf"parallel experiment cap reached \({MAX_RUNNING_EXPERIMENTS}\)",
            ),
        ):
            manager.start_worker("exp_hitl")

    # ---- traces ----------------------------------------------------------------
    def test_trace_pagination_and_partial_tail(self) -> None:
        run_ref = self._run_ref("run_001")
        first = self.client.get(
            "/api/experiments/exp_hitl/trace/blocks", params={"run_id": run_ref}
        ).json()
        self.assertTrue(first["blocks"])
        self.assertTrue(first["eof"])
        again = self.client.get(
            "/api/experiments/exp_hitl/trace/blocks",
            params={"run_id": run_ref, "offset": first["next_offset"]},
        ).json()
        self.assertEqual(again["blocks"], [])
        self.assertEqual(again["next_offset"], first["next_offset"])

    def test_trace_run_id_traversal_is_rejected(self) -> None:
        for run_id in ("../run_001", "run_001/../..", "/etc/passwd"):
            response = self.client.get(
                "/api/experiments/exp_hitl/trace/blocks", params={"run_id": run_id}
            )
            # Never served, and never distinguishable from an absent trace.
            self.assertIn(response.status_code, (400, 404), run_id)
            self.assertNotIn("etc", response.text)
        # A traversing run id on the style route is refused outright.
        self.assertEqual(
            self.client.get(
                "/api/experiments/exp_hitl/style",
                params={"run_id": "../run_001", "prefix": "valid"},
            ).status_code,
            400,
        )

    def test_trace_tail_returns_recent_events_and_stream_offset(self) -> None:
        tail = self.client.get(
            "/api/experiments/exp_hitl/trace/blocks",
            params={"run_id": self._run_ref("run_001"), "tail_events": 2},
        ).json()
        self.assertEqual(
            [block["kind"] for block in tail["blocks"]],
            ["agent_output", "tool_group"],
        )
        self.assertTrue(tail["next_offset"] > 0)

    def test_trace_stats_counts_tokens_and_tool_calls(self) -> None:
        stats = self.client.get(
            "/api/experiments/exp_hitl/trace/stats",
            params={"run_id": self._run_ref("run_001")},
        ).json()
        self.assertEqual(stats["counts"]["llm_call"], 2)
        self.assertEqual(stats["tool_counts"], {"shell": 1})
        self.assertEqual(stats["llm_total_tokens"], 3000)
        self.assertEqual(stats["llm_prompt_tokens"], 2300)
        self.assertEqual(stats["llm_completion_tokens"], 700)
        self.assertEqual(stats["subagent_tasks"], 0)

    def test_trace_download_serves_public_jsonl(self) -> None:
        trace_ref = self._identity().trace_ref("run_001")
        response = self.client.get(
            "/api/experiments/exp_hitl/trace/download",
            params={"run_id": trace_ref},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.text.strip().splitlines()), 3)
        self.assertIn(trace_ref, response.headers["content-disposition"])
        self.assertNotIn("run_001", response.headers["content-disposition"])

class HitlControlActionTest(unittest.TestCase):
    """Positive paths for the seven learning-control actions.

    Each was covered only by its refusal path, so an action that accepted the
    request and then did nothing would have passed. Every test asserts the
    control-state change the worker actually reads back.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.experiments_root.mkdir(parents=True)
        self.directory = self._build("exp_ctl")
        self.client = TestClient(create_app(self.repo_root, self.experiments_root))

    def _build(self, experiment_id: str) -> Path:
        directory = self.experiments_root / experiment_id
        ref_store = AgentRefStore(directory)
        fold_ref = ref_store.get_or_create("fold", "fold_2022Q1")
        run_ref = ref_store.get_or_create("run", "run_001")
        self.fold_q1_node = (
            f"epoch_001__{fold_ref}__{run_ref}__valid_000"
        )
        hitl = directory / "hitl"
        hitl.mkdir(parents=True)
        write_json_atomic(hitl / "params.json", {"experiment_id": experiment_id})
        write_control(hitl / "control.json", ControlState(mode="manual"))
        write_json_atomic(
            hitl / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        write_json_atomic(
            hitl / "schedule.json",
            {
                "schema_version": 1,
                "epochs": 1,
                "sessions": [
                    {
                        "key": "epoch_001/fold_2022Q1",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022Q1",
                    },
                    {
                        "key": "epoch_001/fold_2022Q2",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022Q2",
                    },
                    {
                        "key": "heldout",
                        "kind": "heldout",
                        "epoch_id": "epoch_001",
                        "periods": [{"label": "2023Q1"}, {"label": "2023Q2"}],
                    },
                ],
            },
        )
        frozen = (
            directory
            / "artifacts/strategy/frozen/strategy_epoch_001_fold_2022Q1/output"
        )
        frozen.mkdir(parents=True)
        (frozen / "main.py").write_text(
            "def generate_orders(context):\n    return []\n", encoding="utf-8"
        )
        _write_ledger(
            directory,
            [
                {
                    "record_type": "fold",
                    "experiment_id": experiment_id,
                    "epoch_id": "epoch_001",
                    "fold_id": "fold_2022Q1",
                    "run_id": "run_001",
                    "session_key": "epoch_001/fold_2022Q1",
                    "fold_status": "frozen",
                    "frozen_strategy_artifact_id": "strategy_epoch_001_fold_2022Q1",
                    "frozen_strategy_artifact_path": str(frozen),
                    "validation_result": {"total_return": 0.1},
                    "selected_step_id": self.fold_q1_node,
                    "steps": [
                        {"step_id": self.fold_q1_node, "revision_id": "revision_001"}
                    ],
                },
            ],
        )
        return directory

    def _control(self) -> ControlState:
        return read_control(self.directory / "hitl/control.json")

    def _post(self, **payload):
        session_key = payload.get("session_key")
        if isinstance(session_key, str) and session_key:
            from autotrade.webui.public_identity import PublicIdentity

            payload["session_key"] = PublicIdentity(self.directory).public_session_key(
                session_key
            )
        return self.client.post("/api/experiments/exp_ctl/control", json=payload)

    def _step_tree(
        self, *, fold_id: str = "fold_2022Q1", run_id: str = "run_001"
    ) -> str:
        from autotrade.environment.artifacts import new_revision_id
        from autotrade.environment.step_tree import StepTree

        # The step tree stores opaque refs, and the console resolves them only
        # through the experiment's host mapping.
        ref_store = AgentRefStore(self.directory)
        fold_ref = ref_store.get_or_create("fold", fold_id)
        run_ref = ref_store.get_or_create("run", run_id)
        output = (
            self.directory
            / "artifacts/strategy/frozen/strategy_epoch_001_fold_2022Q1/output"
        )
        tree = StepTree(self.directory / "steps")
        return tree.record_step(
            output,
            epoch_id="epoch_001",
            fold_id=fold_ref,
            run_id=run_ref,
            result_name="valid_000",
            revision_id=new_revision_id("revision"),
            metrics={"total_return": 0.1},
        )

    def test_set_directive_stores_and_clears_a_per_session_directive(self) -> None:
        response = self._post(
            action="set_directive",
            session_key="epoch_001/fold_2022Q2",
            directive="控制回撤",
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._control().directives["epoch_001/fold_2022Q2"], "控制回撤"
        )
        # An empty directive clears it rather than storing a blank.
        self._post(
            action="set_directive", session_key="epoch_001/fold_2022Q2", directive=""
        )
        self.assertNotIn("epoch_001/fold_2022Q2", self._control().directives)
        # The operator defines the PIT policy, so a directive that names a date
        # is stored as written: the rule against it is written guidance (see
        # docs/agent-design.md), not a server-side refusal. The calendar gate
        # stays on Agent-authored memory (PRIOR, skills).
        dated = self._post(
            action="set_directive",
            session_key="epoch_001/fold_2022Q2",
            directive="在 2022Q1 减仓",
        )
        self.assertEqual(dated.status_code, 200, dated.text)
        self.assertEqual(
            self._control().directives["epoch_001/fold_2022Q2"], "在 2022Q1 减仓"
        )

    def test_skip_to_heldout_and_its_cancellation_round_trip(self) -> None:
        self.assertEqual(self._post(action="skip_to_heldout").status_code, 200)
        control = self._control()
        self.assertTrue(control.skip_to_heldout)
        self.assertIsNone(control.request)  # a pending pause must not block it
        self.assertEqual(self._post(action="cancel_skip_to_heldout").status_code, 200)
        self.assertFalse(self._control().skip_to_heldout)

    def test_skip_to_heldout_is_refused_before_any_fold_completes(self) -> None:
        bare = self._build("exp_bare")
        _write_ledger(bare, [])
        refused = self.client.post(
            "/api/experiments/exp_bare/control", json={"action": "skip_to_heldout"}
        )
        self.assertEqual(refused.status_code, 400)
        self.assertFalse(read_control(bare / "hitl/control.json").skip_to_heldout)

    def test_rerun_fold_issues_a_token_for_the_latest_completed_fold(self) -> None:
        response = self._post(action="rerun_fold", session_key="epoch_001/fold_2022Q1")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(self._control().rerun_sessions["epoch_001/fold_2022Q1"])

    def test_rerun_fold_refuses_a_fold_that_is_not_the_latest_completed(self) -> None:
        refused = self._post(action="rerun_fold", session_key="epoch_001/fold_2022Q2")
        self.assertEqual(refused.status_code, 400)
        self.assertEqual(self._control().rerun_sessions, {})

    def test_rollback_fold_withdraws_later_records_and_archives_the_ledger(
        self,
    ) -> None:
        ledger_path = self.directory / "ledgers/experiment_ledger.jsonl"
        ledger = ExperimentLedger(ledger_path)
        ledger.append(
            {
                "record_type": "heldout",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "heldout_2023Q1",
                "run_id": "run_heldout",
                "session_key": "heldout",
                "period": "2023Q1",
                "result": {"total_return": 0.01},
            }
        )
        self.assertEqual(len(ledger.read()), 2)
        node_id = self._step_tree()

        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(response.status_code, 200, response.text)

        # The later held-out record is withdrawn; the rolled-back fold stays.
        remaining = ExperimentLedger(ledger_path).read()
        self.assertEqual([record["record_type"] for record in remaining], ["fold"])
        # The withdrawn ledger is archived, never discarded.
        archives = sorted(
            (self.directory / "ledgers").glob("experiment_ledger.rollback_*.jsonl")
        )
        self.assertEqual(len(archives), 1)
        self.assertIn("heldout", archives[0].read_text(encoding="utf-8"))
        # A rollback clears the derived control state.
        control = self._control()
        self.assertIsNone(control.request)
        self.assertFalse(control.skip_to_heldout)
        self.assertEqual(control.rerun_sessions, {})
        self.assertTrue(node_id)

    def test_rollback_fold_clears_prior_current_when_no_generation_remains(self) -> None:

        store = ExperimentPriorStore(self.directory)
        store.publish("later workflow", generation_id="gen_2")
        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        ledger.append(
            {
                "record_type": "heldout",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "heldout_2023Q1",
                "run_id": "run_heldout",
                "session_key": "heldout",
                "period": "2023Q1",
                "result": {"total_return": 0.01},
            }
        )
        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(store.current_generation_id(), "")
        self.assertEqual(store.current_text(), "")
        self.assertFalse(store.current_pointer_path.exists())
        self.assertEqual(
            (store.root / "generations" / "gen_2" / "PRIOR.md")
            .read_text(encoding="utf-8")
            .strip(),
            "later workflow",
        )

    def test_rollback_fold_is_refused_while_a_worker_is_alive(self) -> None:
        write_json_atomic(
            self.directory / "hitl/status.json",
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "pid_start_ticks": proc_start_ticks(os.getpid()),
                "state": "running_session",
            },
        )
        refused = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(refused.status_code, 400)
        self.assertIn("先停止运行中的 worker", refused.json()["detail"])
        self.assertEqual(
            len(
                ExperimentLedger(
                    self.directory / "ledgers/experiment_ledger.jsonl"
                ).read()
            ),
            1,
        )

    def test_rollback_unlocks_a_flagged_first_fold(self) -> None:
        from autotrade.pipelines.ledger import assert_no_frozen_artifact_mutation

        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        records = ledger.read()
        records[0]["state_changed_during_test"] = True
        ledger.rewrite(records)
        frozen = Path(str(records[0]["frozen_strategy_artifact_path"]))
        self.assertTrue(frozen.is_dir())
        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(response.status_code, 200, response.text)
        remaining = ExperimentLedger(
            self.directory / "ledgers/experiment_ledger.jsonl"
        ).read()
        self.assertEqual(remaining, [])
        assert_no_frozen_artifact_mutation(remaining)
        self.assertFalse(frozen.is_dir())
        self.assertTrue(
            list((self.directory / "artifacts/strategy/_archive").glob("rollback_*"))
        )

    def test_rollback_drops_same_fold_flagged_rerun(self) -> None:
        from autotrade.pipelines.ledger import assert_no_frozen_artifact_mutation

        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        success = ledger.read()[0]
        flagged_id = "strategy_epoch_001_fold_2022Q1_flagged"
        flagged_dir = (
            self.directory / "artifacts/strategy/frozen" / flagged_id / "output"
        )
        flagged_dir.mkdir(parents=True)
        (flagged_dir / "main.py").write_text(
            "def generate_orders(context):\n    return []\n", encoding="utf-8"
        )
        ledger.append(
            {
                "record_type": "fold",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2022Q1",
                "run_id": "run_flagged",
                "session_key": "epoch_001/fold_2022Q1",
                "fold_status": "frozen",
                "frozen_strategy_artifact_id": flagged_id,
                "frozen_strategy_artifact_path": str(flagged_dir),
                "state_changed_during_test": True,
            }
        )
        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(response.status_code, 200, response.text)
        remaining = ledger.read()
        self.assertEqual([record["run_id"] for record in remaining], ["run_001"])
        assert_no_frozen_artifact_mutation(remaining)
        self.assertTrue(Path(str(success["frozen_strategy_artifact_path"])).is_dir())
        self.assertFalse(flagged_dir.is_dir())

    def test_rollback_heldout_unlocks_a_flagged_heldout(self) -> None:
        from autotrade.pipelines.ledger import assert_no_frozen_artifact_mutation

        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        success = ledger.read()[0]
        ledger.append(
            {
                "record_type": "heldout",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "heldout_2023Q1",
                "run_id": "run_heldout_flagged",
                "session_key": "heldout",
                "period": "2023Q1",
                "strategy_artifact_id": success["frozen_strategy_artifact_id"],
                "state_changed_during_test": True,
            }
        )
        response = self._post(action="rollback_fold", session_key="heldout")
        self.assertEqual(response.status_code, 200, response.text)
        remaining = ledger.read()
        self.assertEqual([record["record_type"] for record in remaining], ["fold"])
        assert_no_frozen_artifact_mutation(remaining)
        self.assertTrue(Path(str(success["frozen_strategy_artifact_path"])).is_dir())

    def test_rollback_does_not_archive_shared_no_update_freeze(self) -> None:
        from autotrade.pipelines.ledger import assert_no_frozen_artifact_mutation

        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        success = ledger.read()[0]
        shared_id = str(success["frozen_strategy_artifact_id"])
        shared_path = Path(str(success["frozen_strategy_artifact_path"]))
        ledger.append(
            {
                "record_type": "fold",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2022Q2",
                "run_id": "run_no_update",
                "session_key": "epoch_001/fold_2022Q2",
                "fold_status": "no_update",
                "frozen_strategy_artifact_id": shared_id,
                "frozen_strategy_artifact_path": str(shared_path),
                "state_changed_during_test": True,
            }
        )
        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q2"
        )
        self.assertEqual(response.status_code, 200, response.text)
        remaining = ledger.read()
        self.assertEqual([record["fold_id"] for record in remaining], ["fold_2022Q1"])
        assert_no_frozen_artifact_mutation(remaining)
        self.assertTrue(shared_path.is_dir())

    def _install_inherited_seed(
        self, artifact_id: str = "strategy_inherited_src"
    ) -> Path:
        """The read-only copy the console makes at creation for ``inherit_from``."""
        from autotrade.environment.artifacts import chmod_tree

        seed = self.directory / "artifacts/strategy/_inherited" / artifact_id
        seed.mkdir(parents=True)
        (seed / "main.py").write_text(
            "def generate_orders(context):\n    return []\n", encoding="utf-8"
        )
        chmod_tree(seed, file_mode=0o444, dir_mode=0o555)
        write_json_atomic(
            self.directory / "hitl/params.json",
            {
                "experiment_id": "exp_ctl",
                "inherit_from": "src",
                "_inherited_artifact": {
                    "artifact_id": artifact_id,
                    "path": str(seed),
                    "model_path": None,
                    "revision_id": "revision_src",
                    "source_fold_id": "fold_2021Q4",
                },
            },
        )
        return seed

    def _archived_names(self) -> set[str]:
        return {
            path.name
            for archive in (self.directory / "artifacts/strategy/_archive").glob(
                "rollback_*"
            )
            for path in archive.iterdir()
        }

    def test_rollback_keeps_the_inherited_seed_a_withdrawn_fold_kept(self) -> None:
        """Folds that never beat their inherited parent record the seed's own
        ``_inherited/`` path, so withdrawing one hands that path to the
        archiver. Matching kept records by ``frozen/<id>`` alone never matches
        the seed back, and moving it away leaves the resumed worker with no
        parent at all."""
        seed = self._install_inherited_seed()
        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        kept = ledger.read()[0] | {
            "fold_status": "no_update",
            "frozen_strategy_artifact_id": "strategy_inherited_src",
            "frozen_strategy_artifact_path": str(seed),
        }
        ledger.rewrite([kept])
        ledger.append(
            {
                "record_type": "fold",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2022Q2",
                "run_id": "run_002",
                "session_key": "epoch_001/fold_2022Q2",
                "fold_status": "no_update",
                "parent_strategy_artifact_id": "strategy_inherited_src",
                "frozen_strategy_artifact_id": "strategy_inherited_src",
                "frozen_strategy_artifact_path": str(seed),
            }
        )

        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [record["fold_id"] for record in ledger.read()], ["fold_2022Q1"]
        )
        self.assertTrue((seed / "main.py").is_file())
        self.assertEqual((seed / "main.py").stat().st_mode & 0o222, 0)
        self.assertNotIn(seed.name, self._archived_names())

    def test_rollback_keeps_the_inherited_seed_no_record_names_any_more(self) -> None:
        """The seed is creation state, not rollback output. Unlocking an
        integrity-flagged Fold that kept it withdraws the only record that
        named it; archiving it there would leave the experiment with neither a
        ledger artifact nor the seed its resume falls back to."""
        seed = self._install_inherited_seed()
        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        ledger.rewrite(
            [
                ledger.read()[0]
                | {
                    "fold_status": "no_update",
                    "frozen_strategy_artifact_id": "strategy_inherited_src",
                    "frozen_strategy_artifact_path": str(seed),
                    "state_changed_during_test": True,
                }
            ]
        )

        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(ledger.read(), [])
        self.assertTrue((seed / "main.py").is_file())
        self.assertEqual((seed / "main.py").stat().st_mode & 0o222, 0)
        self.assertNotIn(seed.name, self._archived_names())

    def test_set_gpu_count_round_trips_and_refuses_everything_else(self) -> None:
        """The console's per-session GPU allocation, and its four refusals.

        `params_schema` renders 1..4 as browser attributes; the handler is what
        actually holds the range, so it is asserted here rather than assumed.
        """
        key = "epoch_001/fold_2022Q2"
        public_key = PublicIdentity(self.directory).public_session_key(key)
        allocated = self._post(action="set_gpu_count", session_key=key, directive="2")
        self.assertEqual(allocated.status_code, 200, allocated.text)
        self.assertEqual(
            allocated.json()["control"]["gpu_counts"], {public_key: 2}
        )
        self.assertEqual(self._control().gpu_counts, {key: 2})
        cleared = self._post(action="set_gpu_count", session_key=key, directive="")
        self.assertEqual(cleared.json()["control"]["gpu_counts"], {})
        self.assertEqual(self._control().gpu_counts, {})
        zero = self._post(action="set_gpu_count", session_key=key, directive="0")
        self.assertEqual(zero.status_code, 200, zero.text)
        self.assertEqual(zero.json()["control"]["gpu_counts"], {public_key: 0})
        # 0 is a CPU-only allocation, not an absent one: it must survive the
        # write/read round trip through control.json, or the session silently
        # falls back to the experiment default the researcher just overrode.
        self.assertEqual(self._control().gpu_counts, {key: 0})
        cleared = self._post(action="set_gpu_count", session_key=key, directive="")
        self.assertEqual(cleared.json()["control"]["gpu_counts"], {})
        for directive, fragment in (("5", "0..4"), ("abc", "整数")):
            with self.subTest(directive=directive):
                refused = self._post(
                    action="set_gpu_count", session_key=key, directive=directive
                )
                self.assertEqual(refused.status_code, 400, refused.text)
                self.assertIn(fragment, refused.json()["detail"])
        missing = self._post(action="set_gpu_count", directive="2")
        self.assertEqual(missing.status_code, 400)
        self.assertIn("set_gpu_count requires session_key", missing.json()["detail"])
        unplanned = self.client.post(
            "/api/experiments/exp_ctl/control",
            json={
                "action": "set_gpu_count",
                "session_key": "epoch_009/fold_ref_00000000-0000-4000-8000-000000000001",
                "directive": "2",
            },
        )
        self.assertEqual(unplanned.status_code, 400)
        self.assertIn("unknown public session", unplanned.json()["detail"])
        self.assertEqual(
            self._control().gpu_counts, {}, "a refused request must persist nothing"
        )

    def test_every_action_the_server_accepts_is_reachable_through_the_route(
        self,
    ) -> None:
        from autotrade.webui.manager import _ACTIONS

        refused = self._post(action="not_a_real_action")
        self.assertEqual(refused.status_code, 400)
        self.assertIn("unknown control action", refused.json()["detail"])
        # The seal set is a subset of the action set: a typo in either would
        # silently leave an action permanently unsealed.
        from autotrade.webui.manager import _SEALED_BLOCKED_ACTIONS

        self.assertTrue(_SEALED_BLOCKED_ACTIONS <= _ACTIONS)
        self.assertEqual(
            sorted(_ACTIONS - _SEALED_BLOCKED_ACTIONS),
            [
                "pause",
                "reveal_test_results",
                "set_gpu_count",
                "stop",
                "terminate",
            ],
        )

    def test_the_seal_set_is_pinned_action_by_action(self) -> None:
        """The anti-leakage seal, spelled out rather than counted.

        `_SEALED_BLOCKED_ACTIONS` is what stops a researcher who has seen the Test /
        Held-out numbers from steering any further learning. A missing member
        is invisible to every other test — `resume` was once absent from this
        set — so pin the membership itself, not its size.
        """
        from autotrade.webui.manager import _ACTIONS, _SEALED_BLOCKED_ACTIONS

        self.assertEqual(sorted(_SEALED_BLOCKED_ACTIONS), list(_SEALED_AFTER_REVEAL))
        self.assertEqual(
            sorted(_ACTIONS),
            [
                "cancel_skip_to_heldout",
                "inject_message",
                "pause",
                "rerun_fold",
                "restart",
                "resume",
                "reveal_test_results",
                "rollback_fold",
                "set_directive",
                "set_gpu_count",
                "skip_to_heldout",
                "stop",
                "terminate",
            ],
        )

    def test_every_sealed_action_is_refused_after_the_reveal(self) -> None:
        """Behaviour, not membership: each sealed action really is blocked.

        Driven off the literal list, not off `_SEALED_BLOCKED_ACTIONS`, so shrinking
        the seal cannot make this test pass by iterating fewer actions.
        """
        self.assertEqual(self._post(action="reveal_test_results").status_code, 200)
        for action in _SEALED_AFTER_REVEAL:
            with self.subTest(action=action):
                refused = self._post(action=action, session_key="epoch_001/fold_2022Q1")
                self.assertEqual(refused.status_code, 400, refused.text)
                self.assertIn("测试结果已揭示", refused.json()["detail"])

    def test_the_unsealed_actions_still_work_after_the_reveal(self) -> None:
        """The other half of the seal: lifecycle control must survive it.

        Sealing pause/stop/terminate would strand a running worker on a
        revealed experiment with no way to stop it.
        """
        from autotrade.webui.manager import _ACTIONS, _SEALED_BLOCKED_ACTIONS

        self.assertEqual(self._post(action="reveal_test_results").status_code, 200)
        self.assertEqual(
            sorted(_ACTIONS - _SEALED_BLOCKED_ACTIONS),
            [
                "pause",
                "reveal_test_results",
                "set_gpu_count",
                "stop",
                "terminate",
            ],
        )
        self.assertEqual(self._post(action="pause").status_code, 200)
        self.assertEqual(self._control().request, "pause")
        allocated = self._post(
            action="set_gpu_count", session_key="epoch_001/fold_2022Q2", directive="2"
        )
        self.assertEqual(allocated.status_code, 200, allocated.text)
        self.assertEqual(self._control().gpu_counts, {"epoch_001/fold_2022Q2": 2})
        self.assertEqual(self._post(action="stop").status_code, 200)
        self.assertEqual(self._control().request, "stop")
        # `terminate` reaches its handler and refuses on its own terms (the
        # fixture's recorded pid is dead), not on the seal.
        dead = self._post(action="terminate")
        self.assertEqual(dead.status_code, 400)
        self.assertEqual(dead.json()["detail"], "no live worker to terminate")


def test_public_worker_log_is_repo_relative_and_never_a_host_path(tmp_path: Path):
    """The worker log location crosses the public boundary; a host path must not.

    ``worker_log`` is written into hitl/status.json and returned by the create
    and restart endpoints, both of which the console projects.
    """
    from autotrade.webui.registry import WORKER_LOG_DIR, worker_log_ref

    directory = tmp_path / "experiments" / "demo"
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "schedule.json").write_text(
        json.dumps({"schema_version": 1, "sessions": []}), encoding="utf-8"
    )

    relative = worker_log_ref("demo")
    assert relative == f"{WORKER_LOG_DIR}/demo.log"
    assert not Path(relative).is_absolute()

    identity = PublicIdentity(directory)
    public = identity.public_status(
        {"schema_version": 1, "state": "launching", "worker_log": relative}
    )
    # A relative location survives the projection unchanged...
    assert public["worker_log"] == relative
    # ...while the absolute form it replaced would have been redacted, which is
    # exactly why it must never be stored.
    assert identity.public_status(
        {"worker_log": f"/Data2/lzp/ADMCubeQuant/{relative}"}
    )["worker_log"] == "[host path omitted]"
    assert "/" != json.dumps(public)[0]
    assert not any(
        isinstance(value, str) and Path(value).is_absolute()
        for value in public.values()
    )

    # The launch write is transient: the worker replaces status.json with its
    # own record, so a running experiment must still carry the reference.
    from autotrade.webui.registry import experiment_state, summarize_experiment

    (hitl / "status.json").write_text(
        json.dumps(
            {"schema_version": 1, "state": "running_session", "run_id": "run_001"}
        ),
        encoding="utf-8",
    )
    assert "worker_log" not in experiment_state(directory)
    log_path = tmp_path / relative
    log_path.parent.mkdir(parents=True)
    log_path.write_text("boom\n", encoding="utf-8")
    running = experiment_state(directory)
    # A recorded running state whose pid is gone reads as interrupted — the
    # case where the log is the only explanation left.
    assert running["state"] == "interrupted"
    assert running["worker_log"] == relative
    assert "worker_log" not in running["status"]
    listed = summarize_experiment(directory)
    assert listed["worker_log"] == relative


def test_a_worker_that_dies_during_boot_reads_as_interrupted(tmp_path: Path) -> None:
    """The console's active vocabulary must contain the worker's boot state.

    ``StatusReporter`` writes that state before the first session opens. A
    boot state the console does not count as active never degrades when the
    pid dies, so the experiment keeps its boot badge forever: no 已中断, and
    no resume button on a worker that is gone. The state is read back from the
    writer rather than spelled a second time here, so the two cannot drift.
    """
    from autotrade.webui.manager import _TERMINAL_RESUMABLE_STATES
    from autotrade.webui.registry import ACTIVE_STATES, experiment_state

    directory = tmp_path / "experiments/demo"
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    reporter = StatusReporter(hitl / "status.json")
    reporter.start()
    reporter.stop()
    boot = read_status(hitl / "status.json")
    assert boot["state"] in ACTIVE_STATES

    write_json_atomic(
        hitl / "status.json", {**boot, "pid": 999_999_999, "pid_start_ticks": 1}
    )
    state = experiment_state(directory)
    assert state["state"] == "interrupted"
    # …and the console offers the researcher a way out of it.
    assert state["state"] in _TERMINAL_RESUMABLE_STATES


def test_guarded_fold_view_handles_a_fold_without_a_test_stage() -> None:
    """A development fold may have no Test stage at all.

    Nothing then needs sealing and nothing is missing: the view is the record,
    before and after the reveal, and the console renders no test block rather
    than an empty "pending" one.
    """
    from autotrade.webui.registry import guarded_fold_view

    record = {
        "record_type": "fold",
        "fold_id": "fold_dev",
        "validation_period": "20220101..20251231",
        "validation_result": {"total_return": 0.1, "sub_windows": [_SUB_WINDOW]},
    }
    assert guarded_fold_view(record, test_revealed=False) == record
    assert guarded_fold_view(record, test_revealed=True) == record


def test_guarded_fold_view_never_publishes_result_payloads() -> None:
    """Test evidence and its on-disk pointers leave through the labelled audit
    block only, revealed or not; only the calendar is reveal-gated."""
    from autotrade.webui.registry import guarded_fold_view

    record = {
        "record_type": "fold",
        "test_period": "20220101..20220331",
        "test_decision_time": "2021-12-31T23:59:59+08:00",
        "test_result": {"total_return": 0.2},
        "test_result_ref": "/host/path/result.json",
        "snapshot_ids": ["snap"],
    }
    sealed = guarded_fold_view(record, test_revealed=False)
    assert sealed == {"record_type": "fold"}
    revealed = guarded_fold_view(record, test_revealed=True)
    assert set(revealed) == {"record_type", "test_period", "test_decision_time"}
