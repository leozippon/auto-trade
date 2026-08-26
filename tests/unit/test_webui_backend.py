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
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import AgentTraceWriter, write_json_atomic
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    ControlState,
    StatusReporter,
    consume_step_approval,
    consume_user_reply,
    proc_start_ticks,
    read_control,
    read_status,
    status_pid_alive,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
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
    "approve",
    "approve_step",
    "cancel_skip_to_heldout",
    "inject_message",
    "reply_question",
    "rerun_fold",
    "restart",
    "resume",
    "rollback_fold",
    "set_directive",
    "set_parent_override",
    "set_prompt_override",
    "set_step_gate",
    "skip_to_heldout",
)


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
    assert health["max_running_experiments"] == 5
    assert health["running"] == []
    assert health["unreadable_experiments"] == []
    assert health["raw_generation"] == {"state": "absent"}
    schema = client.get("/api/parameter-schema").json()
    fields = {
        field["key"]: field for group in schema["groups"] for field in group["fields"]
    }
    assert "fields" not in schema
    assert schema["schema_version"] == 2
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
    assert fields["fold_period"]["choices"] == ["week", "month", "quarter", "year"]
    assert fields["strategy_period"]["choices"] == ["day", "month", "quarter", "year"]
    assert fields["inference_time"]["default"] == "08:30"
    assert fields["daily_window_months"]["optional"] is True
    assert fields["include_intraday"]["default"] is True
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
        "meta_model",
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
    assert fields["model"]["choices"] == [
        "qwen3.8-27b-local",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ]
    assert fields["meta_model"]["choices"] == fields["model"]["choices"]
    assert fields["model"]["default"] == "qwen3.8-27b-local"
    assert fields["meta_model"]["default"] == "qwen3.8-27b-local"
    assert fields["nl_model"]["default"] == "qwen3.8-27b-local"
    assert fields["compact_model"]["default"] == "qwen3.8-27b-local"
    assert fields["reasoning_effort"]["default"] == "max"
    assert fields["compact_token_threshold"]["default"] == 200_000
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
        "epochs",
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
    assert '<img class="logo" src="/static/logo.png"' in page.text
    # The nav strip carries a frontend-only 实盘交易 entry on its own #/qmt
    # route. What must stay absent is the live-trading console itself: no
    # backend, no communication, no execution path behind it.
    assert '<a href="#/qmt" data-nav="qmt">实盘交易</a>' in page.text
    assert "#/trading/live" not in page.text and 'data-nav="live"' not in page.text
    logo = client.get("/static/logo.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"


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


def test_cli_exposes_distinct_fold_and_meta_model_choices() -> None:
    from scripts.experiments._cli import add_model_arguments

    parser = argparse.ArgumentParser()
    add_model_arguments(parser, verbose_help=True)
    defaults = parser.parse_args([])
    assert defaults.model == defaults.meta_model == "qwen3.8-27b-local"
    mixed = parser.parse_args(
        [
            "--model",
            "deepseek-v4-flash",
            "--meta-model",
            "qwen3.8-27b-local",
        ]
    )
    assert mixed.model == "deepseek-v4-flash"
    assert mixed.meta_model == "qwen3.8-27b-local"


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
                "first_test_period": "2024Q1",
                "last_test_period": "2024Q1",
                "heldout_first_period": "2024Q2",
                "heldout_last_period": "2024Q2",
                "strategy_period": "quarter",
                "inference_time": "23:59",
                "daily_window_months": 18,
                "include_macro": False,
                "events_datasets": ["margin", "moneyflow"],
                "screen_boards": ["main", "gem"],
                "model": "deepseek-v4-flash",
                "meta_model": "deepseek-v4-pro",
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
    assert params["meta_model"] == "deepseek-v4-pro"
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


def _persistent_experiment(tmp_path: Path) -> Path:
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


def test_active_experiment_api_hides_historical_steps_analysis_and_reports(
    tmp_path: Path,
):
    directory = _persistent_experiment(tmp_path)
    fold_ref = _fold_ref(directory, "fold_2026Q1")
    run_ref = _run_ref(directory, "run_001")
    session_ref = _session_ref(directory, "epoch_001/fold_2026Q1")
    client = TestClient(create_app(tmp_path))

    home = client.get("/api/experiments").json()["experiments"][0]
    assert home["experiment_id"] == "demo"
    assert (home["completed_sessions"], home["total_sessions"]) == (1, 2)
    assert home["skills"] == {
        "count": 2,
        "files": 3,
        "bytes": 512,
    }

    detail = client.get("/api/experiments/demo").json()
    assert detail["sessions"][0]["record"]["fold_status"] == "frozen"
    assert "test_result" not in detail["sessions"][0]["record"]
    assert detail["sessions"][-1]["kind"] == "heldout"

    steps = client.get("/api/experiments/demo/steps")
    assert steps.status_code == 200
    assert steps.json()["nodes"] == []
    assert (
        client.get("/api/experiments/demo/steps/node_001/source.zip").status_code == 404
    )

    curve = client.get("/api/experiments/demo/equity").json()
    assert curve["series"][0]["key"] == "valid"
    assert curve["series"][0]["dates"] == ["20260102", "20260105"]
    fold_curve = client.get(
        f"/api/experiments/demo/folds/epoch_001/{fold_ref}/equity"
    ).json()
    assert fold_curve["series"][0]["key"] == "valid"
    assert fold_curve["series"][0]["dates"] == ["20260102", "20260105"]
    assert fold_curve["benchmark"]["key"] == "benchmark"
    assert fold_curve["benchmark"]["dates"] == ["20260102", "20260105"]
    assert curve["benchmark"]["label"] == "沪深300"
    orders = client.get(
        f"/api/experiments/demo/folds/epoch_001/{fold_ref}/orders"
    ).json()
    assert orders["rows"][0]["symbol"] == "000001.SZ"
    assert (
        client.get(
            f"/api/experiments/demo/folds/epoch_001/{fold_ref}/orders.csv",
            params={"result": orders["result"]},
        )
        .headers["content-type"]
        .startswith("text/csv")
    )
    style = client.get(
        "/api/experiments/demo/style", params={"run_id": run_ref, "prefix": "valid"}
    ).json()
    assert style["schema_version"] == 1
    assert style["benchmark_regression"]["beta"] == 0.8
    assert (
        client.get(
            "/api/experiments/demo/style",
            params={"run_id": "../run_001", "prefix": "valid"},
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/experiments/demo/style",
            params={"run_id": run_ref, "prefix": "test"},
        ).status_code
        == 404
    )

    preview = client.post(
        "/api/experiments/demo/prompt-preview",
        json={"session_key": session_ref, "directive": "控制回撤"},
    ).json()
    assert preview["kind"] == "fold"
    assert "控制回撤" in preview["prompt"]

    analysis = client.get(f"/api/experiments/demo/analysis/epoch_001/{fold_ref}")
    assert analysis.status_code == 200
    assert analysis.json()["available"] is False

    trace = client.get(f"/api/experiments/demo/trace?run_id={run_ref}").json()
    assert [event["event_type"] for event in trace["events"]] == [
        "session_start",
        "llm_call",
    ]
    assert (
        client.get(f"/api/experiments/demo/trace/stats?run_id={run_ref}").json()[
            "llm_total_tokens"
        ]
        == 12
    )
    assert (
        client.get(f"/api/experiments/demo/trace/download?run_id={run_ref}").status_code
        == 200
    )
    assert (
        client.get(f"/api/experiments/demo/trace/stream?run_id={run_ref}").status_code
        == 200
    )
    prompt = client.get(
        f"/api/experiments/demo/folds/epoch_001/{fold_ref}/initial-prompt"
    ).json()
    assert [message["role"] for message in prompt["messages"]] == ["system", "user"]
    rendered_prompt = json.dumps(prompt, ensure_ascii=False)
    assert "遵守 PIT 合同" in rendered_prompt
    assert fold_ref in rendered_prompt
    assert "fold_2026Q1" not in rendered_prompt

    assert client.post("/api/experiments/demo/reports").status_code == 404
    assert client.get("/api/experiments/demo/reports").status_code == 404
    assert client.get("/api/experiments/demo/reports/download").status_code == 404


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


def test_experiment_progress_comes_from_schedule_and_durable_ledger(tmp_path: Path):
    directory = _persistent_experiment(tmp_path)
    client = TestClient(create_app(tmp_path))
    ledger_path = directory / "ledgers/experiment_ledger.jsonl"
    fold_record = ledger_path.read_text(encoding="utf-8")
    status_path = directory / "hitl/status.json"

    def home_progress() -> tuple[int, int | None]:
        home = client.get("/api/experiments").json()["experiments"][0]
        return home["completed_sessions"], home["total_sessions"]

    # A just-started experiment has no durable success, even if a stale or
    # racing heartbeat claims that every planned session is complete.
    ledger_path.write_text("", encoding="utf-8")
    write_json_atomic(
        status_path,
        {
            "schema_version": 1,
            "state": "running_session",
            "pid": os.getpid(),
            "pid_start_ticks": proc_start_ticks(os.getpid()),
            "completed_sessions": 2,
            "total_sessions": 2,
        },
    )
    assert home_progress() == (0, 2)

    # Conversely, a startup heartbeat's legitimate zero cannot erase a Fold
    # that is already durable, and a bogus denominator cannot override plan.
    ledger_path.write_text(fold_record, encoding="utf-8")
    write_json_atomic(
        status_path,
        {
            "schema_version": 1,
            "state": "running_session",
            "pid": os.getpid(),
            "pid_start_ticks": proc_start_ticks(os.getpid()),
            "completed_sessions": 0,
            "total_sessions": 999,
        },
    )
    assert home_progress() == (1, 2)

    # Held-out is one planned session and becomes durable only after all of
    # its periods are recorded; terminal progress is therefore exactly N/N.
    schedule_path = directory / "hitl/schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["sessions"][-1]["periods"].append(
        {"label": "2026Q3", "start": "20260701", "end": "20260930"}
    )
    write_json_atomic(schedule_path, schedule)
    ledger = ExperimentLedger(ledger_path)
    for label in ("2026Q2", "2026Q3"):
        ledger.append(
            {
                "record_type": "heldout",
                "experiment_id": "demo",
                "epoch_id": "epoch_001",
                "fold_id": f"heldout_{label}",
                "run_id": f"run_heldout_{label}",
                "period": label,
                "result": {"total_return": 0.01},
            }
        )
        expected = (1, 2) if label == "2026Q2" else (2, 2)
        assert home_progress() == expected

    write_json_atomic(
        status_path,
        {
            "schema_version": 1,
            "state": "completed",
            "completed_sessions": 0,
            "total_sessions": 1,
        },
    )
    assert home_progress() == (2, 2)


def test_current_question_and_step_controls_use_exact_one_shot_keys(tmp_path: Path):
    directory = _persistent_experiment(tmp_path)
    client = TestClient(create_app(tmp_path))
    session_key = "epoch_001/fold_2026Q1"
    question_key = f"{session_key}#q1"
    public_session_key = _session_ref(directory, session_key)
    public_question_key = f"{public_session_key}#q1"
    status_path = directory / "hitl/status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "waiting_user_reply",
                "session_key": session_key,
                "question_key": question_key,
                "question": "继续当前假设吗？",
            }
        ),
        encoding="utf-8",
    )

    wrong_question = client.post(
        "/api/experiments/demo/control",
        json={
            "action": "reply_question",
            "session_key": public_session_key,
            "directive": "",
        },
    )
    assert wrong_question.status_code == 400
    assert "current question key" in wrong_question.json()["detail"]

    replied = client.post(
        "/api/experiments/demo/control",
        json={
            "action": "reply_question",
            "session_key": public_question_key,
            "directive": "",
        },
    )
    assert replied.status_code == 200
    assert replied.json()["control"]["user_replies"] == {public_question_key: ""}
    assert consume_user_reply(directory / "hitl/control.json", question_key) == (
        True,
        "",
    )
    assert read_control(directory / "hitl/control.json").user_replies == {}

    status_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "waiting_step_user",
                "session_key": session_key,
                "step_index": 2,
            }
        ),
        encoding="utf-8",
    )
    wrong_step = client.post(
        "/api/experiments/demo/control",
        json={
            "action": "approve_step",
            "session_key": public_session_key,
            "step_index": 1,
            "directive": "继续控制回撤",
        },
    )
    assert wrong_step.status_code == 400
    assert "current waiting Step" in wrong_step.json()["detail"]
    malformed_step = client.post(
        "/api/experiments/demo/control",
        json={
            "action": "approve_step",
            "session_key": public_session_key,
            "step_index": 2,
            "directive": {"text": "继续"},
        },
    )
    assert malformed_step.status_code == 400
    assert malformed_step.json()["detail"] == "approve_step directive must be a string"

    approved = client.post(
        "/api/experiments/demo/control",
        json={
            "action": "approve_step",
            "session_key": public_session_key,
            "step_index": 2,
            "directive": "继续控制回撤",
        },
    )
    assert approved.status_code == 200
    control = approved.json()["control"]
    assert control["step_go"] == {public_session_key: 2}
    assert control["step_directives"] == {
        f"{public_session_key}#2": "继续控制回撤"
    }
    assert consume_step_approval(directory / "hitl/control.json", session_key, 2) == (
        True,
        "继续控制回撤",
    )
    consumed = read_control(directory / "hitl/control.json")
    assert consumed.step_go == {}
    assert consumed.step_directives == {}


def test_status_reporter_keeps_only_the_current_wait_payload(tmp_path: Path):
    path = tmp_path / "status.json"
    reporter = StatusReporter(path)
    reporter.set(
        state="waiting_user_reply",
        question_key="epoch_001/fold_2026Q1#q1",
        question="继续吗？",
        question_summary="当前假设",
    )
    reporter.set(
        state="waiting_step_user",
        step_index=1,
        step_summary={"complete": True},
    )
    status = read_status(path)
    assert status["step_index"] == 1
    assert not {"question_key", "question", "question_summary"} & status.keys()

    reporter.set(state="running_session")
    status = read_status(path)
    assert (
        not {"step_index", "step_summary", "question_key", "question"} & status.keys()
    )


def test_partial_heldout_does_not_reveal_out_of_sample_results(tmp_path: Path):
    directory = _persistent_experiment(tmp_path)
    schedule_path = directory / "hitl/schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["sessions"][-1]["periods"] = [
        {"label": "2026Q2", "start": "20260401", "end": "20260630"},
        {"label": "2026Q3", "start": "20260701", "end": "20260930"},
    ]
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl").append(
        {
            "record_type": "heldout",
            "experiment_id": "demo",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_2026Q2",
            "run_id": "run_heldout",
            "period": "2026Q2",
            "result": {"total_return": 0.03},
        }
    )
    status = directory / "hitl/status.json"
    status.write_text(
        json.dumps({"schema_version": 1, "state": "running_heldout"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))
    detail = client.get("/api/experiments/demo").json()
    assert detail["test_revealed"] is False
    assert "test_result" not in detail["sessions"][0]["record"]

    status.write_text(
        json.dumps({"schema_version": 1, "state": "completed"}),
        encoding="utf-8",
    )
    detail = client.get("/api/experiments/demo").json()
    assert detail["test_revealed"] is False
    assert "test_result" not in detail["sessions"][0]["record"]
    assert [record["record_type"] for record in detail["ledger"]] == ["fold", "heldout"]
    heldout = detail["ledger"][-1]
    assert heldout["hidden"] is True
    # Only identity survives redaction: no result, no reference to one.
    assert not {"result", "result_ref", "test_result"} & heldout.keys()


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
    assert "session_key: questionKey" in script
    assert "step_index: stepIndex" in script
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
                "first_test_period": "2022Q1",
                "last_test_period": "2022Q2",
                "heldout_first_period": "2023Q1",
                "heldout_last_period": "2023Q1",
                "analysis_model": "deepseek-v4-flash",
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
            / "valid_000"
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
                    "frozen_strategy_artifact_id": "strategy_epoch_001_fold_2022Q1",
                    "frozen_strategy_artifact_path": str(strategy_dir),
                    "validation_result": {
                        "total_return": 0.10,
                        "sharpe": 1.0,
                        "max_drawdown": 0.05,
                        "long_return": 0.08,
                    },
                    "test_result": {
                        "total_return": 0.20,
                        "sharpe": 1.5,
                        "max_drawdown": 0.04,
                        "long_return": 0.15,
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
        orders_dir = experiment_dir / "artifacts" / "run_001" / "results" / "valid_000"
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
        self.assertEqual(fields["epochs"]["default"], WEB_CREATE_DEFAULTS["epochs"])
        self.assertEqual(fields["model"]["default"], WEB_CREATE_DEFAULTS["model"])
        self.assertEqual(
            fields["meta_model"]["default"], WEB_CREATE_DEFAULTS["meta_model"]
        )
        self.assertEqual(
            fields["initial_control_mode"]["default"],
            WEB_CREATE_DEFAULTS["initial_control_mode"],
        )
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
        for model_field in (
            "model",
            "meta_model",
            "nl_model",
            "compact_model",
            "analysis_model",
        ):
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
        # No trade calendar under the tmp repo root: period pickers degrade to text.
        self.assertEqual(schema["period_options"], {})
        self.assertEqual(fields["first_test_period"]["type"], "string")
        # Filled per-epoch on the detail page instead of at creation.
        self.assertNotIn("meta_learning_directive", fields)
        self.assertEqual(fields["meta_learning_fold_interval"]["default"], 1)
        self.assertEqual(fields["meta_learning_fold_interval"]["min"], 0)
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

    def test_period_options_and_defaults_from_calendar(self) -> None:
        from autotrade.webui.params_schema import (
            build_period_options,
            parameter_schema,
            suggest_period_defaults,
        )

        trading_days = [
            day.strftime("%Y%m%d")
            for day in pd.date_range("2023-01-02", "2024-07-05", freq="B")
        ]
        options = build_period_options(trading_days)
        self.assertEqual(options["year"], ["2023"])
        self.assertEqual(options["quarter"][0], "2023Q1")
        self.assertEqual(
            options["quarter"][-1], "2024Q2"
        )  # ends 20240630 <= last trading day
        self.assertIn("202401", options["month"])
        self.assertNotIn("202407", options["month"])  # incomplete month excluded
        self.assertTrue(all(len(label) == 8 for label in options["week"]))
        defaults = suggest_period_defaults(options)
        quarter = defaults["quarter"]
        # Calendar ends 2024Q2, so the pinned 2025Q4/2026Q1 preset cannot apply.
        self.assertEqual(quarter["heldout_first_period"], "2024Q2")
        self.assertEqual(quarter["last_test_period"], "2024Q1")
        self.assertLess(quarter["first_test_period"], quarter["last_test_period"])
        # first_test never takes the very first option (its validation period
        # must also exist in the calendar).
        self.assertNotEqual(quarter["first_test_period"], options["quarter"][0])
        schema = parameter_schema(trading_days=trading_days)
        fields = {
            field["key"]: field
            for group in schema["groups"]
            for field in group["fields"]
        }
        self.assertEqual(fields["first_test_period"]["type"], "period")
        self.assertEqual(fields["heldout_first_period"]["default"], "2024Q2")

    def test_quarter_defaults_prefer_the_console_research_window(self) -> None:
        from autotrade.webui.params_schema import suggest_period_defaults

        labels = [f"{year}Q{quarter}" for year in range(2022, 2027) for quarter in range(1, 5)]
        defaults = suggest_period_defaults({"quarter": labels})["quarter"]
        self.assertEqual(defaults["first_test_period"], "2024Q2")
        self.assertEqual(defaults["last_test_period"], "2025Q4")
        self.assertEqual(defaults["heldout_first_period"], "2026Q1")
        self.assertEqual(defaults["heldout_last_period"], "2026Q2")

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
        sealed_periods = {
            "first_test_period": "2024Q1",
            "last_test_period": "2024Q4",
            "heldout_first_period": "2025Q1",
            "heldout_last_period": "2025Q4",
        }
        params = {
            "model": "deepseek-v4-pro",
            **sealed_periods,
            **{key: f"secret-{key}" for key in operator_only},
        }
        public = _public_params(params, test_revealed=False)
        self.assertEqual(public, {"model": "deepseek-v4-pro"})
        self.assertEqual(
            _public_params(params, test_revealed=True),
            {"model": "deepseek-v4-pro", **sealed_periods},
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

    def test_list_experiments_marks_kind_state_and_metrics(self) -> None:
        payload = self.client.get("/api/experiments").json()
        by_id = {entry["experiment_id"]: entry for entry in payload["experiments"]}
        hitl = by_id["exp_hitl"]
        self.assertEqual(hitl["kind"], "hitl")
        self.assertEqual(hitl["state"], "interrupted")  # recorded pid is not alive
        self.assertFalse(hitl["test_revealed"])
        self.assertEqual(
            [row["fold_ref"] for row in hitl["fold_returns"]],
            [self._fold_ref("fold_2022Q1")],
        )
        self.assertIsNone(hitl["environment_stage"])
        self.assertIsNone(hitl["metrics"]["cum_heldout_return"])
        self.assertIsNone(hitl["metrics"]["cum_test_return"])

    def test_list_exposes_the_live_environment_stage(self) -> None:
        status_path = self.experiments_root / "exp_hitl" / "hitl" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["environment_stage"] = "pit_snapshot"
        status["environment_stage_started_at"] = "2026-08-17T23:46:10+00:00"
        status["session_started_at"] = "2026-08-17T23:46:10+00:00"
        status["environment_progress"] = {"day_index": 3, "total_days": 10}
        write_json_atomic(status_path, status)
        hitl = {
            entry["experiment_id"]: entry
            for entry in self.client.get("/api/experiments").json()["experiments"]
        }["exp_hitl"]
        self.assertEqual(hitl["environment_stage"], "pit_snapshot")
        self.assertEqual(
            hitl["environment_stage_started_at"], "2026-08-17T23:46:10+00:00"
        )
        self.assertEqual(hitl["session_started_at"], "2026-08-17T23:46:10+00:00")
        self.assertEqual(
            hitl["environment_progress"], {"day_index": 3, "total_days": 10}
        )

    # ---- reveal / seal --------------------------------------------------------
    def test_heldout_completion_auto_reveals_and_seals(self) -> None:
        from autotrade.webui.registry import test_results_revealed

        experiment_dir = self.experiments_root / "exp_hitl"
        hitl = experiment_dir / "hitl"
        schedule = json.loads((hitl / "schedule.json").read_text(encoding="utf-8"))
        schedule["sessions"][-1]["periods"] = [{"label": "2023Q1"}, {"label": "2023Q2"}]
        write_json_atomic(hitl / "schedule.json", schedule)

        # Partial held-out (fixture records only 2023Q1): stays hidden so the
        # worker can still be resumed to finish the remaining periods.
        self.assertFalse(test_results_revealed(experiment_dir))
        detail = self.client.get("/api/experiments/exp_hitl").json()
        self.assertFalse(detail["test_revealed"])

        # Recording the last planned period auto-reveals without any click.
        ledger_path = experiment_dir / "ledgers" / "experiment_ledger.jsonl"
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "record_type": "heldout",
                        "experiment_id": "exp_hitl",
                        "epoch_id": "epoch_001",
                        "fold_id": "heldout_2023Q2",
                        "run_id": "run_heldout_2",
                        "period": "2023Q2",
                        "result": {
                            "total_return": 0.01,
                            "sharpe": 0.1,
                            "max_drawdown": 0.02,
                        },
                    }
                )
                + "\n"
            )
        self.assertTrue(test_results_revealed(experiment_dir))
        detail = self.client.get("/api/experiments/exp_hitl").json()
        self.assertTrue(detail["test_revealed"])
        listing = self.client.get("/api/experiments").json()
        entry = {item["experiment_id"]: item for item in listing["experiments"]}[
            "exp_hitl"
        ]
        self.assertTrue(entry["test_revealed"])
        self.assertAlmostEqual(entry["metrics"]["cum_test_return"], 0.20)

        # Auto-reveal applies the same seal as a manual reveal: the guard must
        # read test_results_revealed(), not only the control flag.
        response = self.client.post(
            "/api/experiments/exp_hitl/control",
            json={"action": "approve", "session_key": "heldout"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("封存", response.json()["detail"])

    def test_reveal_seals_learning_actions(self) -> None:
        self._reveal()
        # `resume` restarts learning on a sealed experiment, so it is blocked
        # together with the rest of the learning-affecting control set.
        for action in (
            "approve",
            "rerun_fold",
            "rollback_fold",
            "approve_step",
            "reply_question",
            "set_step_gate",
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
        for prefix in ("valid_000", "test_000"):
            directory = results / prefix
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.json").write_text("{}", encoding="utf-8")
            style_mode = "frozen_test" if prefix.startswith("test") else prefix.split("_")[0]
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
                record["test_result_ref"] = str(results / "test_000" / "result.json")
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
        test_dir = results / "test_000"
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
        hidden = self.client.get(url, params={"result": "test_000"})
        self.assertEqual(hidden.status_code, 404)
        listing = self.client.get(url).json()
        self.assertEqual(listing["result"], "valid_000")
        # The visible enumeration must not leak the sealed result's existence.
        self.assertEqual(listing["available"], ["valid_000"])
        csv_hidden = self.client.get(url + ".csv", params={"result": "test_000"})
        self.assertEqual(csv_hidden.status_code, 404)
        self._reveal()
        revealed = self.client.get(url, params={"result": "test_000"})
        self.assertEqual(revealed.status_code, 200, revealed.text)
        self.assertEqual(revealed.json()["result"], "test_000")
        # Revealed test results surface in their own list, never in `available`:
        # the default selection reads `available`, so a test result can never
        # silently become the pane's default.
        listing = self.client.get(url).json()
        self.assertIn("test_000", listing["test_results"])
        self.assertEqual(listing["available"], ["valid_000"])
        self.assertEqual(listing["result"], "valid_000")
        csv_ok = self.client.get(url + ".csv", params={"result": "test_000"})
        self.assertEqual(csv_ok.status_code, 200)
        self.assertEqual(len(csv_ok.text.strip().splitlines()), 2)  # header + 1 order

    def test_fold_orders_rows_and_csv_export(self) -> None:
        results = (
            self.experiments_root
            / "exp_hitl"
            / "artifacts"
            / "run_001"
            / "results"
            / "valid_000"
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
        self.assertEqual(data["result"], "valid_000")
        self.assertEqual(data["row_count"], 3)
        self.assertEqual(
            [row["action"] for row in data["rows"]], ["buy", "sell", "buy"]
        )
        self.assertEqual(data["rows"][2]["reason"], "limit_up_blocked_buy")
        csv_response = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders.csv",
            params={"result": "valid_000"},
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
        """The seven-key projection the order pane and the audit block read.

        Pinned as an exact key set: a rename in either direction, or an extra
        field nothing consumes, breaks the pane silently otherwise.
        """
        import shutil

        expected_keys = [
            "available",
            "result",
            "row_count",
            "rows",
            "stats",
            "test_results",
            "truncated",
        ]
        fold_ref = self._fold_ref("fold_2022Q1")
        url = f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders"
        payload = self.client.get(url).json()
        self.assertEqual(sorted(payload), expected_keys)
        self.assertEqual(payload["available"], ["valid_000"])
        # Pre-reveal the sealed list is empty, and three orders are far under
        # the 500-row cap.
        self.assertEqual(payload["test_results"], [])
        self.assertIs(payload["truncated"], False)
        self.assertEqual(payload["row_count"], len(payload["rows"]))
        # The empty selection returns the same shape, so the pane never has to
        # branch on which keys are present.
        shutil.rmtree(self.experiments_root / "exp_hitl/artifacts/run_001/results")
        empty = self.client.get(url).json()
        self.assertEqual(sorted(empty), expected_keys)
        self.assertEqual(empty["result"], None)
        self.assertEqual(empty["available"], [])
        self.assertEqual(empty["test_results"], [])
        self.assertEqual(empty["rows"], [])
        self.assertEqual(empty["row_count"], 0)
        self.assertIs(empty["truncated"], False)

    def test_fold_orders_caps_the_table_at_five_hundred_rows_but_not_the_export(
        self,
    ) -> None:
        """`row_count` and the stats describe the whole stream, `rows` is capped.

        The pane renders 「共 M 条」 from `row_count`, so a cap that also
        shrank the count would report a lie; and the CSV is the escape hatch
        from the cap, so it must stay uncapped.
        """
        results = self.experiments_root / "exp_hitl/artifacts/run_001/results/valid_000"
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
        self.assertIs(payload["truncated"], True)
        self.assertEqual(payload["stats"]["orders"], 501)
        self.assertEqual(payload["stats"]["filled"], 501)
        csv_response = self.client.get(url + ".csv", params={"result": "valid_000"})
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(
            len(csv_response.text.strip().splitlines()), 502
        )  # header + 501

    def test_summary_carries_per_period_heldout_returns(self) -> None:
        self._reveal()
        payload = self.client.get("/api/experiments").json()
        hitl = next(
            e for e in payload["experiments"] if e["experiment_id"] == "exp_hitl"
        )
        self.assertEqual(
            hitl["heldout_returns"],
            [{"fold_ref": self._fold_ref("heldout_2023Q1"), "return": -0.03}],
        )
        self.assertAlmostEqual(hitl["metrics"]["cum_heldout_return"], -0.03)
        row = hitl["fold_returns"][0]
        self.assertAlmostEqual(row["valid_long"], 0.08)

    # ---- lifecycle guards ------------------------------------------------------
    def test_experiment_detail_merges_schedule_and_records(self) -> None:
        detail = self.client.get("/api/experiments/exp_hitl").json()
        sessions = {session["key"]: session for session in detail["sessions"]}
        q1_key = self._session_ref("epoch_001/fold_2022Q1")
        q2_key = self._session_ref("epoch_001/fold_2022Q2")
        self.assertIn("record", sessions[q1_key])
        self.assertNotIn("record", sessions[q2_key])
        self.assertTrue(sessions[q1_key]["analysis_available"])
        self.assertEqual(detail["control"]["mode"], "manual")
        sealed_period_fields = {
            "first_test_period",
            "last_test_period",
            "heldout_first_period",
            "heldout_last_period",
        }
        self.assertTrue(sealed_period_fields.isdisjoint(detail["params"]))
        self._reveal()
        revealed = self.client.get("/api/experiments/exp_hitl").json()
        self.assertEqual(
            {key: revealed["params"][key] for key in sealed_period_fields},
            {
                "first_test_period": "2022Q1",
                "last_test_period": "2022Q2",
                "heldout_first_period": "2023Q1",
                "heldout_last_period": "2023Q1",
            },
        )
        self.assertEqual(self.client.get("/api/experiments/nope").status_code, 404)

    def test_modern_public_api_exposes_only_opaque_identities(self) -> None:
        identity = self._identity()
        fold_ref = identity.fold_ref("fold_2022Q1")
        run_ref = identity.run_ref("run_001")
        trace_ref = identity.trace_ref("run_001")
        session_ref = identity.public_session_key("epoch_001/fold_2022Q2")
        forbidden = (
            "fold_2022Q1",
            "fold_2022Q2",
            "run_001",
            "run_meta",
            "run_heldout",
            str(self.repo_root),
            str(self.experiments_root),
        )

        json_responses = [
            self.client.get("/api/experiments"),
            self.client.get("/api/experiments/exp_hitl"),
            self.client.get("/api/experiments/exp_hitl/status"),
            self.client.get(
                f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}"
            ),
            self.client.get("/api/experiments/exp_hitl/steps"),
            self.client.get(
                "/api/experiments/exp_hitl/trace", params={"run_id": run_ref}
            ),
            self.client.get(
                "/api/experiments/exp_hitl/trace/stats",
                params={"run_id": run_ref},
            ),
            self.client.get(
                "/api/experiments/exp_hitl/trace/blocks",
                params={"run_id": trace_ref},
            ),
            self.client.get(
                f"/api/experiments/exp_hitl/analysis/epoch_001/{fold_ref}"
            ),
            self.client.post(
                "/api/experiments/exp_hitl/prompt-preview",
                json={"session_key": session_ref, "directive": "控制回撤"},
            ),
        ]
        for response in json_responses:
            self.assertEqual(response.status_code, 200, response.text)
            public_text = json.dumps(response.json(), ensure_ascii=False)
            for marker in forbidden:
                self.assertNotIn(marker, public_text)

        trace_download = self.client.get(
            "/api/experiments/exp_hitl/trace/download",
            params={"run_id": trace_ref},
        )
        self.assertEqual(trace_download.status_code, 200, trace_download.text)
        strategy_download = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/strategy.zip"
        )
        self.assertEqual(strategy_download.status_code, 200)
        orders_download = self.client.get(
            f"/api/experiments/exp_hitl/folds/epoch_001/{fold_ref}/orders.csv",
            params={"result": "valid_000"},
        )
        self.assertEqual(orders_download.status_code, 200)
        for response in (trace_download, strategy_download, orders_download):
            disposition = response.headers.get("content-disposition", "")
            for marker in forbidden:
                self.assertNotIn(marker, disposition)
        for marker in forbidden:
            self.assertNotIn(marker, trace_download.text)

        self.assertEqual(
            self.client.get(
                "/api/experiments/exp_hitl/folds/epoch_001/fold_2022Q1"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                "/api/experiments/exp_hitl/trace", params={"run_id": "run_001"}
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                "/api/experiments/exp_hitl/control",
                json={
                    "action": "set_gpu_count",
                    "session_key": "epoch_001/fold_2022Q2",
                    "directive": "1",
                },
            ).status_code,
            400,
        )

        embedded_host_paths = (
            "/var/lib/private/result.json",
            "/secret",
            "path:/secret",
            "path:/var/lib/private/result.json",
            "file:///tmp/private/result.json",
            "file://server/share/private.txt",
            r"C:\Users\private\result.json",
        )
        error = "failed opening " + " and ".join(embedded_host_paths)
        public_status = identity.public_status(
            {
                "state": "failed",
                "error": error,
                "final_strategy_artifact": "strategy_secret_raw",
            }
        )
        public_status_text = json.dumps(public_status)
        for embedded_host_path in embedded_host_paths:
            self.assertNotIn(embedded_host_path, public_status_text)
        self.assertNotIn("strategy_secret_raw", public_status_text)
        self.assertTrue(
            str(public_status["final_strategy_ref"]).startswith("strategy_ref_")
        )
        public_event = identity.public_record(
            {
                "event_type": "tool_call",
                "run_id": "run_001",
                "error": error,
                "safe_paths": (
                    "GET /api/experiments/x; read /mnt/agent/workspace/main.py; "
                    "see https://example.test/docs/path"
                ),
            },
            heldout_revealed=False,
        )
        public_event_text = json.dumps(public_event)
        for embedded_host_path in embedded_host_paths:
            self.assertNotIn(embedded_host_path, public_event_text)
        self.assertIn("/api/experiments/x", public_event_text)
        self.assertIn("/mnt/agent/workspace/main.py", public_event_text)
        self.assertIn("https://example.test/docs/path", public_event_text)

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

    def test_dead_question_wait_degrades_to_interrupted(self) -> None:
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": 999_999_999,
                "state": "waiting_user_reply",
                "session_key": "epoch_001/fold_2022Q2",
            },
        )
        status = self.client.get("/api/experiments/exp_hitl/status").json()
        self.assertEqual(status["state"], "interrupted")
        self.assertFalse(status["worker_alive"])
        self.assertEqual(status["status"]["state"], "waiting_user_reply")
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
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                }
            },
        )
        self.assertEqual(created.status_code, 200, created.text)

    def test_running_cap_allows_fifth_and_blocks_sixth(self) -> None:
        manager = ExperimentManager(self.repo_root, self.experiments_root)
        running = [f"running_{index}" for index in range(MAX_RUNNING_EXPERIMENTS - 1)]
        with (
            patch.object(manager, "running_experiments", return_value=running),
            patch.object(manager, "_preflight"),
            patch.object(manager, "start_worker", return_value={"spawned": False}),
        ):
            created = manager.create_experiment(
                {
                    "experiment_id": "exp_fifth",
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                }
            )
        self.assertEqual(created["experiment_id"], "exp_fifth")

        running.append("exp_fifth")
        with (
            patch.object(manager, "running_experiments", return_value=running),
            self.assertRaisesRegex(
                ManagerError, r"parallel experiment cap reached \(5\)"
            ),
        ):
            manager.create_experiment(
                {
                    "experiment_id": "exp_sixth",
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                }
            )
        self.assertFalse((self.experiments_root / "exp_sixth").exists())

    def test_running_cap_also_guards_worker_restart(self) -> None:
        manager = ExperimentManager(self.repo_root, self.experiments_root)
        running = [f"running_{index}" for index in range(MAX_RUNNING_EXPERIMENTS)]
        with (
            patch.object(manager, "running_experiments", return_value=running),
            self.assertRaisesRegex(
                ManagerError, r"parallel experiment cap reached \(5\)"
            ),
        ):
            manager.start_worker("exp_hitl")

    # ---- traces ----------------------------------------------------------------
    def test_trace_pagination_and_partial_tail(self) -> None:
        run_ref = self._run_ref("run_001")
        first = self.client.get(
            "/api/experiments/exp_hitl/trace", params={"run_id": run_ref}
        ).json()
        self.assertEqual(len(first["events"]), 3)
        self.assertTrue(first["eof"])
        again = self.client.get(
            "/api/experiments/exp_hitl/trace",
            params={"run_id": run_ref, "offset": first["next_offset"]},
        ).json()
        self.assertEqual(again["events"], [])
        self.assertEqual(again["next_offset"], first["next_offset"])

    def test_trace_run_id_traversal_is_rejected(self) -> None:
        for run_id in ("../run_001", "run_001/../..", "/etc/passwd"):
            response = self.client.get(
                "/api/experiments/exp_hitl/trace", params={"run_id": run_id}
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
            "/api/experiments/exp_hitl/trace",
            params={"run_id": self._run_ref("run_001"), "tail_events": 2},
        ).json()
        self.assertEqual([event["seq"] for event in tail["events"]], [1, 2])
        self.assertTrue(tail["next_offset"] > 0)

    def test_trace_stats_counts_tokens_and_tool_calls(self) -> None:
        stats = self.client.get(
            "/api/experiments/exp_hitl/trace/stats",
            params={"run_id": self._run_ref("run_001")},
        ).json()
        self.assertEqual(stats["counts"]["llm_call"], 2)
        self.assertEqual(stats["total_events"], 3)
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

    def test_prompt_preview_embeds_directive_and_guards_heldout(self) -> None:
        preview = self.client.post(
            "/api/experiments/exp_hitl/prompt-preview",
            json={
                "session_key": self._session_ref("epoch_001/fold_2022Q2"),
                "directive": "控制回撤",
            },
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        payload = preview.json()
        self.assertEqual(payload["kind"], "fold")
        self.assertIn("控制回撤", payload["prompt"])
        # The preview is a prompt, not an evidence channel: the held-out
        # schedule never reaches it.
        self.assertNotIn("2023Q1", payload["prompt"])
        self.assertNotIn("test_period", payload["prompt"])
        self.assertNotIn("fold_2022Q2", payload["prompt"])
        self.assertNotIn("2022Q2", payload["prompt"])
        refused = self.client.post(
            "/api/experiments/exp_hitl/prompt-preview", json={"session_key": "heldout"}
        )
        self.assertEqual(refused.status_code, 400)


class InheritFromTest(unittest.TestCase):
    """Creating an experiment with ``inherit_from`` must actually inherit.

    The console copies the source experiment's latest frozen fold output into
    the new experiment as a read-only snapshot, and the worker starts its first
    Fold from that snapshot instead of the blank template. Both halves are
    asserted here: accepting the parameter without seeding the run would be a
    decorative feature.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.experiments_root.mkdir(parents=True)
        self.manager = ExperimentManager(self.repo_root, self.experiments_root)

    def _source_experiment(
        self, experiment_id: str = "exp_source", *, with_fold: bool = True
    ) -> Path:
        directory = self.experiments_root / experiment_id
        (directory / "hitl").mkdir(parents=True)
        write_json_atomic(
            directory / "hitl/params.json", {"experiment_id": experiment_id}
        )
        write_json_atomic(
            directory / "hitl/status.json", {"schema_version": 1, "state": "completed"}
        )
        if not with_fold:
            _write_ledger(directory, [])
            return directory
        frozen = (
            directory
            / "artifacts/strategy/frozen/strategy_epoch_001_fold_2022Q1/output"
        )
        frozen.mkdir(parents=True)
        (frozen / "main.py").write_text(
            "def generate_orders(context):\n    return []  # inherited\n",
            encoding="utf-8",
        )
        models = (
            directory
            / "artifacts/strategy/frozen/strategy_epoch_001_fold_2022Q1/models"
        )
        models.mkdir(parents=True)
        (models / "params.json").write_text('{"alpha": 1}\n', encoding="utf-8")
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
                    "test_period": "20220101..20220331",
                    "frozen_strategy_artifact_id": "strategy_epoch_001_fold_2022Q1",
                    "frozen_strategy_artifact_path": str(frozen),
                    "frozen_model_artifact_path": str(models),
                }
            ],
        )
        return directory

    def test_inherit_import_copies_and_locks_the_source_artifact(self) -> None:
        self._source_experiment()
        target = self.experiments_root / "exp_child"
        (target / "hitl").mkdir(parents=True)
        payload = self.manager._import_inherited_artifact(target, "exp_source")

        self.assertEqual(payload["source_experiment_id"], "exp_source")
        self.assertEqual(payload["source_fold_id"], "fold_2022Q1")
        self.assertEqual(
            payload["source_artifact_id"], "strategy_epoch_001_fold_2022Q1"
        )
        copied = Path(str(payload["path"]))
        self.assertTrue((copied / "main.py").exists())
        self.assertIn("inherited", (copied / "main.py").read_text(encoding="utf-8"))
        self.assertTrue(copied.is_relative_to(target))
        models = Path(str(payload["model_path"]))
        self.assertTrue((models / "params.json").exists())
        # The snapshot is immutable evidence: read-only stands in for a digest.
        self.assertEqual((copied / "main.py").stat().st_mode & 0o222, 0)
        self.assertEqual((models / "params.json").stat().st_mode & 0o222, 0)

    def test_inherit_from_a_source_without_a_recorded_fold_is_refused(self) -> None:
        self._source_experiment("exp_bare", with_fold=False)
        target = self.experiments_root / "exp_child"
        (target / "hitl").mkdir(parents=True)
        with self.assertRaises(ManagerError):
            self.manager._import_inherited_artifact(target, "exp_bare")
        self.assertFalse((target / "artifacts/strategy/_inherited").exists())

    def test_created_experiment_records_the_inherited_seed(self) -> None:
        self._source_experiment()
        with patch.object(
            ExperimentManager, "start_worker", return_value={"spawned": False}
        ):
            self.manager.create_experiment(
                {
                    "experiment_id": "exp_child",
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                    "inherit_from": "exp_source",
                }
            )
        params = json.loads(
            (self.experiments_root / "exp_child/hitl/params.json").read_text(
                encoding="utf-8"
            )
        )
        inherited = params["_inherited_artifact"]
        self.assertEqual(inherited["source_experiment_id"], "exp_source")
        self.assertTrue(Path(inherited["path"]).is_dir())

    def test_worker_starts_the_first_fold_from_the_inherited_artifact(self) -> None:
        from autotrade.pipelines.worker import _load_inherited_parent

        self._source_experiment()
        with patch.object(
            ExperimentManager, "start_worker", return_value={"spawned": False}
        ):
            self.manager.create_experiment(
                {
                    "experiment_id": "exp_child",
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                    "inherit_from": "exp_source",
                }
            )
        child = self.experiments_root / "exp_child"

        parent = _load_inherited_parent(child)
        self.assertIsNotNone(parent)
        self.assertEqual(parent.source_fold_id, "fold_2022Q1")
        self.assertIn(
            "inherited", (parent.path / "main.py").read_text(encoding="utf-8")
        )
        self.assertIsNotNone(parent.model_path)

        # An experiment created without inherit_from starts blank.
        with patch.object(
            ExperimentManager, "start_worker", return_value={"spawned": False}
        ):
            self.manager.create_experiment(
                {
                    "experiment_id": "exp_blank",
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                }
            )
        self.assertIsNone(_load_inherited_parent(self.experiments_root / "exp_blank"))

    def test_tampered_inherited_snapshot_stops_the_run(self) -> None:
        from autotrade.environment.artifacts import ArtifactError
        from autotrade.pipelines.worker import _load_inherited_parent

        self._source_experiment()
        with patch.object(
            ExperimentManager, "start_worker", return_value={"spawned": False}
        ):
            self.manager.create_experiment(
                {
                    "experiment_id": "exp_child",
                    "first_test_period": "2024Q1",
                    "last_test_period": "2024Q1",
                    "heldout_first_period": "2024Q2",
                    "heldout_last_period": "2024Q2",
                    "inherit_from": "exp_source",
                }
            )
        child = self.experiments_root / "exp_child"
        seed = Path(
            json.loads((child / "hitl/params.json").read_text(encoding="utf-8"))[
                "_inherited_artifact"
            ]["path"]
        )
        # A seed that became writable again is unverified strategy code.
        (seed / "main.py").chmod(0o644)
        with self.assertRaisesRegex(ArtifactError, "writable"):
            _load_inherited_parent(child)
        # A seed that vanished is refused just as loudly.
        moved = seed.with_name("moved")
        seed.rename(moved)
        with self.assertRaisesRegex(
            RuntimeError, "inherited artifact directory is missing"
        ):
            _load_inherited_parent(child)
        moved.rename(seed)
        (seed / "main.py").chmod(0o444)
        self.assertIsNotNone(_load_inherited_parent(child))

    def test_inherit_choices_list_only_sources_with_a_recorded_fold(self) -> None:
        self._source_experiment("exp_source")
        self._source_experiment("exp_bare", with_fold=False)
        client = TestClient(create_app(self.repo_root, self.experiments_root))
        schema = client.get("/api/parameter-schema").json()
        fields = {
            field["key"]: field
            for group in schema["groups"]
            for field in group["fields"]
        }
        choices = fields["inherit_from"]["choices"]
        self.assertEqual(choices[0], "")  # blank = start from the template
        self.assertIn("exp_source", choices)
        self.assertNotIn("exp_bare", choices)


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
            complete_validation=True,
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
        dated = self._post(
            action="set_directive",
            session_key="epoch_001/fold_2022Q2",
            directive="在 2022Q1 减仓",
        )
        self.assertEqual(dated.status_code, 400, dated.text)
        self.assertIn("日历日期", dated.json()["detail"])
        self.assertNotIn("epoch_001/fold_2022Q2", self._control().directives)

    def test_set_prompt_override_stores_and_clears_a_replacement_prompt(self) -> None:
        self.assertEqual(
            self._post(
                action="set_prompt_override",
                session_key="epoch_001/fold_2022Q2",
                directive="完整替换提示词",
            ).status_code,
            200,
        )
        self.assertEqual(
            self._control().prompt_overrides["epoch_001/fold_2022Q2"], "完整替换提示词"
        )
        self._post(
            action="set_prompt_override",
            session_key="epoch_001/fold_2022Q2",
            directive="",
        )
        self.assertNotIn("epoch_001/fold_2022Q2", self._control().prompt_overrides)

    def test_set_step_gate_toggles_per_session_step_holding(self) -> None:
        self.assertEqual(
            self._post(
                action="set_step_gate",
                session_key="epoch_001/fold_2022Q2",
                directive="on",
            ).status_code,
            200,
        )
        self.assertIs(self._control().step_gate["epoch_001/fold_2022Q2"], True)
        self._post(
            action="set_step_gate", session_key="epoch_001/fold_2022Q2", directive="off"
        )
        self.assertIs(self._control().step_gate["epoch_001/fold_2022Q2"], False)
        self._post(
            action="set_step_gate", session_key="epoch_001/fold_2022Q2", directive=""
        )
        self.assertNotIn("epoch_001/fold_2022Q2", self._control().step_gate)

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

    def test_set_parent_override_records_and_clears_a_validated_node(self) -> None:
        node_id = self._step_tree()
        response = self._post(
            action="set_parent_override",
            session_key="epoch_001/fold_2022Q2",
            directive=node_id,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self._control().parent_overrides["epoch_001/fold_2022Q2"], node_id
        )
        self._post(
            action="set_parent_override",
            session_key="epoch_001/fold_2022Q2",
            directive="",
        )
        self.assertNotIn("epoch_001/fold_2022Q2", self._control().parent_overrides)

    def test_set_parent_override_refuses_a_later_fold_node(self) -> None:
        later = self._step_tree(fold_id="fold_2022Q2", run_id="run_002")
        refused = self._post(
            action="set_parent_override",
            session_key="epoch_001/fold_2022Q1",
            directive=later,
        )
        self.assertEqual(refused.status_code, 400)
        self.assertIn("未来验证信息泄漏", refused.json()["detail"])
        self.assertEqual(self._control().parent_overrides, {})

    def test_rerun_fold_issues_a_token_and_resets_the_session_gates(self) -> None:
        control = self._control()
        control.approved_sessions = ("epoch_001/fold_2022Q1",)
        control.step_go["epoch_001/fold_2022Q1"] = 3
        write_control(self.directory / "hitl/control.json", control)

        response = self._post(action="rerun_fold", session_key="epoch_001/fold_2022Q1")
        self.assertEqual(response.status_code, 200, response.text)
        updated = self._control()
        self.assertTrue(updated.rerun_sessions["epoch_001/fold_2022Q1"])
        # The re-run must be re-approved and its step gating starts afresh.
        self.assertNotIn("epoch_001/fold_2022Q1", updated.approved_sessions)
        self.assertNotIn("epoch_001/fold_2022Q1", updated.step_go)

    def test_rerun_fold_in_auto_mode_returns_the_fold_to_a_real_gate(self) -> None:
        """A re-run must actually wait for the researcher, in every mode.

        `rerun_fold`'s own comment promises "the re-run must be re-approved
        (prompt edits land first)". Dropping the session from
        `approved_sessions` delivers that in manual/step mode only:
        `InteractiveExperimentRunner._gate` returns immediately when
        `control.mode == "auto"`, so an auto-mode re-run starts before the
        researcher can edit anything -- which is the entire point of the
        action. `terminate` already routes through
        `_require_session_reapproval`, which drops the approval AND falls back
        to manual; `rerun_fold` inlines only the first half.
        """
        control = self._control()
        control.mode = "auto"
        control.approved_sessions = ("epoch_001/fold_2022Q1",)
        write_control(self.directory / "hitl/control.json", control)

        response = self._post(action="rerun_fold", session_key="epoch_001/fold_2022Q1")
        self.assertEqual(response.status_code, 200, response.text)
        updated = self._control()
        self.assertNotIn("epoch_001/fold_2022Q1", updated.approved_sessions)
        self.assertEqual(updated.mode, "manual")

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
        self.assertEqual(control.parent_overrides, {})
        self.assertTrue(node_id)

    def test_rollback_fold_restores_prior_current_immediately(self) -> None:
        from autotrade.pipelines.prior import ExperimentPriorStore

        write_json_atomic(
            self.directory / "hitl/schedule.json",
            {
                "schema_version": 1,
                "epochs": 1,
                "sessions": [
                    {
                        "key": "epoch_001/meta_learning",
                        "kind": "meta",
                        "epoch_id": "epoch_001",
                    },
                    {
                        "key": "epoch_001/fold_2022Q1",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022Q1",
                    },
                    {
                        "key": "epoch_001/meta_learning_after_fold_001",
                        "kind": "meta",
                        "epoch_id": "epoch_001",
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
        store = ExperimentPriorStore(self.directory)
        store.publish("first workflow", generation_id="gen_1")
        store.publish("second workflow", generation_id="gen_2")
        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        ledger.append(
            {
                "record_type": "meta_learning",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "epoch_001_meta_learning",
                "run_id": "run_meta_0",
                "session_key": "epoch_001/meta_learning",
                "prior": "first workflow",
                "prior_generation_id": "gen_1",
            }
        )
        ledger.append(
            {
                "record_type": "meta_learning",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "epoch_001_after_fold_001",
                "run_id": "run_meta_1",
                "session_key": "epoch_001/meta_learning_after_fold_001",
                "prior": "second workflow",
                "prior_generation_id": "gen_2",
            }
        )
        self.assertEqual(store.current_generation_id(), "gen_2")
        response = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(response.status_code, 200, response.text)
        remaining = ExperimentLedger(
            self.directory / "ledgers/experiment_ledger.jsonl"
        ).read()
        self.assertEqual(
            [record["record_type"] for record in remaining],
            ["fold", "meta_learning"],
        )
        self.assertEqual(remaining[-1]["prior_generation_id"], "gen_1")
        self.assertEqual(store.current_generation_id(), "gen_1")
        self.assertEqual(store.current_text().strip(), "first workflow")
        self.assertEqual(
            (store.root / "generations" / "gen_2" / "PRIOR.md")
            .read_text(encoding="utf-8")
            .strip(),
            "second workflow",
        )

    def test_rollback_fold_clears_prior_current_when_no_generation_remains(self) -> None:
        from autotrade.pipelines.prior import ExperimentPriorStore

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

    def test_rollback_fold_fails_if_remaining_prior_generation_is_missing(self) -> None:
        from autotrade.pipelines.prior import ExperimentPriorStore

        write_json_atomic(
            self.directory / "hitl/schedule.json",
            {
                "schema_version": 1,
                "epochs": 1,
                "sessions": [
                    {
                        "key": "epoch_001/meta_learning",
                        "kind": "meta",
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
                        "periods": [{"label": "2023Q1"}, {"label": "2023Q2"}],
                    },
                ],
            },
        )
        store = ExperimentPriorStore(self.directory)
        store.publish("later workflow", generation_id="gen_2")
        ledger = ExperimentLedger(self.directory / "ledgers/experiment_ledger.jsonl")
        ledger.append(
            {
                "record_type": "meta_learning",
                "experiment_id": "exp_ctl",
                "epoch_id": "epoch_001",
                "fold_id": "epoch_001_meta_learning",
                "run_id": "run_meta_ghost",
                "session_key": "epoch_001/meta_learning",
                "prior": "gone",
                "prior_generation_id": "ghost",
            }
        )
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
        refused = self._post(
            action="rollback_fold", session_key="epoch_001/fold_2022Q1"
        )
        self.assertEqual(refused.status_code, 400, refused.text)
        self.assertIn("ghost", refused.json()["detail"])
        self.assertEqual(store.current_generation_id(), "gen_2")

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
                "set_mode",
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
                "approve",
                "approve_step",
                "cancel_skip_to_heldout",
                "inject_message",
                "pause",
                "reply_question",
                "rerun_fold",
                "restart",
                "resume",
                "reveal_test_results",
                "rollback_fold",
                "set_directive",
                "set_gpu_count",
                "set_mode",
                "set_parent_override",
                "set_prompt_override",
                "set_step_gate",
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
                "set_mode",
                "stop",
                "terminate",
            ],
        )
        self.assertEqual(self._post(action="pause").status_code, 200)
        self.assertEqual(self._control().request, "pause")
        self.assertEqual(self._post(action="set_mode", mode="step").status_code, 200)
        self.assertEqual(self._control().mode, "step")
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
