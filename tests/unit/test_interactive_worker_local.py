from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import (
    LEGACY_LOCAL_QWEN_MODEL,
    LOCAL_QWEN_MODEL,
    DeepSeekProxy,
    OpenAICompatibleProxy,
    ProviderResponse,
    ScriptedLLM,
    ToolCall,
)
from autotrade.environment.tools import CommandResult
from autotrade.pipelines.agent_views import compact_fold_history
from autotrade.pipelines.hitl_state import (
    ControlState,
    DevelopmentSession,
    read_control,
    read_status,
    write_control,
)
from autotrade.pipelines.interactive import InteractiveExperimentRunner
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.local_backend import SessionBudgetLLM, SessionCallBudget
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from autotrade.webui.manager import ExperimentManager
from autotrade.webui.server import create_app

_FOLD_DELEGATION_ROLES = ("auditor", "developer")


def _experiment(
    tmp_path: Path, *, developer_mode: str = "baseline"
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    experiment = repo / "experiments" / "smoke"
    hitl = experiment / "hitl"
    hitl.mkdir(parents=True)
    strategy = repo / "strategies" / "main.py"
    strategy.parent.mkdir(parents=True)
    strategy.write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    daily = repo / "data" / "daily.parquet"
    daily.parent.mkdir(parents=True)
    days = pd.bdate_range("2025-09-01", "2026-09-30")
    pd.DataFrame(
        {
            "trade_date": [stamp.strftime("%Y%m%d") for stamp in days],
            "symbol": ["000001.SZ"] * len(days),
            "open": [10.0] * len(days),
            "close": [10.0] * len(days),
        }
    ).to_parquet(daily, index=False)
    (hitl / "params.json").write_text(
        json.dumps(
            {
                "experiment_id": "smoke",
                "strategy_path": "strategies/main.py",
                "daily_path": "data/daily.parquet",
                "execution_mode": "trusted",
                "developer_mode": developer_mode,
                "data_backend": "daily",
                "initial_control_mode": "auto",
                "strategy_period": "day",
                "inference_time": "08:30",
                "initial_cash": 100_000,
                "epochs": 1,
                "first_test_period": "2026Q1",
                "last_test_period": "2026Q1",
                "heldout_first_period": "2026Q2",
                "heldout_last_period": "2026Q2",
            }
        ),
        encoding="utf-8",
    )
    (hitl / "control.json").write_text(
        json.dumps({"schema_version": 1, "mode": "auto"}), encoding="utf-8"
    )
    return repo, experiment


def test_analysis_enabled_defaults_off_and_can_be_enabled(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    options = load_worker_options(experiment, repo_root=repo)
    assert options.analysis_enabled is False

    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["analysis_enabled"] = True
    path.write_text(json.dumps(params), encoding="utf-8")
    options = load_worker_options(experiment, repo_root=repo)
    assert options.analysis_enabled is True


def test_local_worker_runs_real_baseline_valid_test_and_heldout(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(options)
    assert result["state"] == "completed"
    assert result["developer_mode"] == "deterministic_baseline_no_agent_improvement"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    # The ledger is append-only: completion adds no summary row and rewrites nothing.
    assert [record["record_type"] for record in records] == ["fold", "heldout"]
    assert result["final_strategy_artifact"].startswith("strategy_")
    heldout = records[-1]
    assert heldout["result"]["total_return"] == 0.0
    assert heldout["strategy_artifact_id"] == result["final_strategy_artifact"]
    assert (
        records[0]["frozen_strategy_artifact_id"] == result["final_strategy_artifact"]
    )
    schedule = json.loads(
        (experiment / "hitl" / "schedule.json").read_text(encoding="utf-8")
    )
    assert [row["kind"] for row in schedule["sessions"]] == ["fold", "heldout"]
    assert schedule["sessions"][-1]["periods"] == [
        {"label": "2026Q2", "start": "20260401", "end": "20260630"}
    ]


def test_worker_rejects_llm_mode_without_provider_credentials(tmp_path: Path):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    with pytest.raises(ValueError, match="requires the gateway API key"):
        load_worker_options(experiment, repo_root=repo)


def test_worker_maps_model_context_params_to_role_gateways_and_compactor(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "model": "deepseek-v4-flash",
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
    )
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    settings = load_worker_options(experiment, repo_root=repo).llm
    assert settings is not None
    assert settings.meta_model == settings.model
    main = settings.build_gateway("main").config
    nl = settings.build_gateway("nl").config
    compact = settings.build_gateway("compact").config
    assert (main.model, nl.model, compact.model) == (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-pro",
    )
    assert main.thinking_enabled is False and nl.thinking_enabled is False
    assert main.reasoning_effort == "high" and nl.reasoning_effort == "high"
    assert compact.thinking_enabled is False and compact.reasoning_effort is None
    assert compact.max_tokens == 1_200
    assert settings.compact_enabled is True
    assert settings.compaction.token_threshold == 90_000
    assert settings.compaction.keep_recent_messages == 10
    assert settings.compaction.max_response_tokens == 1_200
    assert settings.compaction.max_calls == 4


def test_worker_canonicalizes_all_legacy_model_roles_without_rewriting_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "model": LEGACY_LOCAL_QWEN_MODEL,
            "meta_model": LEGACY_LOCAL_QWEN_MODEL,
            "nl_model": LEGACY_LOCAL_QWEN_MODEL,
            "compact_model": LEGACY_LOCAL_QWEN_MODEL,
            "analysis_model": LEGACY_LOCAL_QWEN_MODEL,
            "analysis_enabled": True,
            "compact_token_threshold": 20_000,
        }
    )
    path.write_text(json.dumps(params), encoding="utf-8")
    persisted = path.read_bytes()
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    options = load_worker_options(experiment, repo_root=repo)

    assert options.llm is not None
    assert (
        options.llm.model,
        options.llm.meta_model,
        options.llm.nl_model,
        options.llm.compact_model,
        options.analysis_model,
    ) == (LOCAL_QWEN_MODEL,) * 5
    assert path.read_bytes() == persisted


def test_worker_ignores_historical_endpoint_param_and_uses_trusted_env(
    tmp_path: Path,
    monkeypatch,
):
    from autotrade.pipelines.worker import _ALLOWED_PARAMS, NON_PERSISTABLE_PARAMS

    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["llm_base_url"] = "https://untrusted-snapshot.example.test/v1"
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    monkeypatch.setenv("VLLM_BASE_URL", "https://trusted-runtime.example.test/v1")

    settings = load_worker_options(experiment, repo_root=repo).llm
    assert settings is not None
    assert "llm_base_url" in NON_PERSISTABLE_PARAMS
    assert "llm_base_url" not in _ALLOWED_PARAMS
    assert (
        settings.build_gateway("main").config.base_url
        == "https://trusted-runtime.example.test/v1"
    )


def test_worker_resolves_mixed_local_and_deepseek_roles_with_real_timeout(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "model": LOCAL_QWEN_MODEL,
            "meta_model": "deepseek-v4-pro",
            "nl_model": "deepseek-v4-flash",
            "compact_model": "deepseek-v4-flash",
            "analysis_model": LOCAL_QWEN_MODEL,
            "compact_token_threshold": 20_000,
            "per_call_timeout_seconds": 120,
        }
    )
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")

    options = load_worker_options(experiment, repo_root=repo)
    assert options.llm is not None
    main = options.llm.build_gateway("main")
    meta = options.llm.build_gateway("meta")
    nl = options.llm.build_gateway("nl")
    analysis = options.llm.build_gateway(
        "analysis", model=options.analysis_model, max_tokens=options.analysis_max_tokens
    )
    assert isinstance(main, OpenAICompatibleProxy)
    assert main.provider == "vllm"
    assert isinstance(meta, DeepSeekProxy)
    assert meta.provider == "deepseek"
    assert meta.model == "deepseek-v4-pro"
    assert isinstance(nl, DeepSeekProxy)
    assert nl.provider == "deepseek"
    assert analysis.provider == "vllm"
    assert main.config.max_tokens == 32_768
    assert analysis.config.max_tokens == 6_000
    assert main.config.timeout_seconds == 120
    assert main.config.reasoning_effort == "xhigh"


def test_worker_applies_local_output_cap_to_each_role_budget(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "model": LOCAL_QWEN_MODEL,
            "meta_model": LOCAL_QWEN_MODEL,
            "nl_model": LOCAL_QWEN_MODEL,
            "compact_model": LOCAL_QWEN_MODEL,
            "analysis_model": LOCAL_QWEN_MODEL,
            "compact_token_threshold": 20_000,
            "compact_max_tokens": 20_000,
            "analysis_max_tokens": 6_000,
        }
    )
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    options = load_worker_options(experiment, repo_root=repo)
    assert options.llm is not None
    assert options.llm.compaction.max_response_tokens == 20_000
    assert options.llm.max_tokens_for("main") == 32_768
    assert options.llm.max_tokens_for("meta") == 32_768
    assert options.llm.max_tokens_for("nl", requested=1_200) == 1_200
    assert options.llm.max_tokens_for("nl", requested=20_000) == 20_000
    assert (
        options.llm.max_tokens_for(
            "analysis", model=options.analysis_model, requested=6_000
        )
        == 6_000
    )
    for role in ("main", "meta", "nl"):
        assert options.llm.build_gateway(role).config.max_tokens == 32_768
    assert options.llm.build_gateway("compact").config.max_tokens == 20_000


def test_worker_rejects_local_context_threshold_before_launch(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["model"] = LOCAL_QWEN_MODEL
    params["compact_token_threshold"] = 300_000
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    with pytest.raises(ValueError, match="compact_token_threshold must be <= 227328"):
        load_worker_options(experiment, repo_root=repo)


def test_model_roles_share_one_session_call_budget():
    first = ScriptedLLM([ProviderResponse(content="main")])
    second = ScriptedLLM([ProviderResponse(content="nl")])
    shared = SessionCallBudget(
        max_calls=1, deadline=__import__("time").monotonic() + 10
    )
    main = SessionBudgetLLM(first, budget=shared)
    nl = SessionBudgetLLM(second, budget=shared)
    main.complete([])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        nl.complete([])
    assert shared.calls == 1


def test_interactive_hooks_consume_current_controls_without_retaining_content(
    tmp_path: Path,
):
    session_key = "epoch_001/fold_2026Q1"
    control_path = tmp_path / "control.json"
    status_path = tmp_path / "status.json"
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(),
        execute_session=lambda _session, _context: None,
        ledger=ledger,
        control_path=control_path,
        status_path=status_path,
        poll_seconds=0.01,
    )

    write_control(
        control_path,
        ControlState(
            mode="step",
            step_go={session_key: 2},
            step_directives={f"{session_key}#2": "继续控制回撤"},
        ),
    )
    assert runner.step_gate_hook(session_key)(2, {"complete": True}) == "继续控制回撤"
    consumed = read_control(control_path)
    assert consumed.step_go == {}
    assert consumed.step_directives == {}
    assert read_status(status_path)["state"] == "running_session"

    write_control(
        control_path,
        ControlState(mode="manual", user_replies={f"{session_key}#q1": ""}),
    )
    assert runner.user_question_hook(session_key)("继续吗？") == ""
    assert read_control(control_path).user_replies == {}
    assert read_status(status_path)["state"] == "running_session"


def test_interactive_runner_publishes_current_session_timing(tmp_path: Path):
    session_key = "epoch_001/fold_2026Q1"
    control_path = tmp_path / "control.json"
    status_path = tmp_path / "status.json"
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    write_control(control_path, ControlState(mode="auto"))
    captured: dict[str, object] = {}

    def execute(session, context):
        progress = context["progress_hook"]
        assert callable(progress)
        progress("pit_snapshot", {"run_id": "run_001"})
        status = read_status(status_path)
        captured.update(status)
        timing = context["session_timing"]
        assert callable(timing)
        captured["timing"] = timing()
        return {
            "record_type": "fold",
            "experiment_id": "demo",
            "epoch_id": session.epoch_id,
            "fold_id": "fold_2026Q1",
            "run_id": "run_001",
        }

    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(DevelopmentSession(session_key, "fold", "epoch_001", None),),
        execute_session=execute,
        ledger=ledger,
        control_path=control_path,
        status_path=status_path,
        ref_store=AgentRefStore(tmp_path / "experiment"),
        poll_seconds=0.01,
    )
    assert runner.run()["status"] == "complete"
    assert captured["state"] == "running_session"
    assert captured["session_key"] == session_key
    assert captured["session_started_at"]
    assert captured["researcher_wait_seconds"] == 0.0
    assert captured["run_id"] == "run_001"
    assert captured["environment_stage"] == "pit_snapshot"
    assert captured["environment_stage_started_at"]
    timing = captured["timing"]
    assert isinstance(timing, dict)
    assert timing["run_wall_seconds"] >= 0.0
    assert timing["researcher_wait_seconds"] == 0.0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model", "unknown-model", "unsupported DeepSeek model"),
        ("meta_model", "unknown-model", "unsupported DeepSeek model"),
        ("reasoning_effort", "ultra", "reasoning_effort"),
        ("no_thinking", 1, "must be a boolean"),
        ("compact_token_threshold", 0, "must be a positive integer"),
        ("compact_max_calls", -1, "must be a non-negative integer"),
    ],
)
def test_worker_rejects_invalid_model_context_params(
    tmp_path: Path,
    monkeypatch,
    key: str,
    value: object,
    message: str,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params[key] = value
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    with pytest.raises(ValueError, match=message):
        load_worker_options(experiment, repo_root=repo)


class _NoShellRunner:
    def run(self, argv, *, cwd, timeout_seconds, max_output_chars, input_text=None):
        del argv, cwd, timeout_seconds, max_output_chars, input_text
        return CommandResult(126, stderr="shell is disabled in this test")


def _explore_then(
    *tool_calls: ToolCall,
    roles: tuple[str, ...] = ("auditor",),
    summary: str = "委托完成",
    implement: dict[str, object] | None = None,
) -> tuple[ProviderResponse, ...]:
    explores = tuple(
        ToolCall(f"ex_{role}", "explore", {"role": role, "task": f"review {role}"})
        for role in roles
    )
    responses: list[ProviderResponse] = [
        ProviderResponse(tool_calls=(*explores, *tool_calls))
    ]
    for role in roles:
        if implement is not None and role == "developer":
            responses.append(
                ProviderResponse(
                    tool_calls=(
                        ToolCall("w", "write_file", dict(implement)),
                    )
                )
            )
        responses.append(ProviderResponse(content=summary))
    return tuple(responses)


def test_llm_worker_runs_real_meta_fold_validation_and_heldout(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    assert options.llm is not None
    assert isinstance(options.llm.build_gateway(), OpenAICompatibleProxy)
    source = "def generate_orders(context):\n    return []\n"
    llm = ScriptedLLM(
        [
            *_explore_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer small daily changes"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_explore_then(
                ToolCall(
                    "ask",
                    "ask_user",
                    {"question": "Continue with the bounded validation?"},
                ),
                ToolCall("check", "modification_check", {}),
                ToolCall("valid", "daily_backtest", {}),
                ToolCall("finish_fold", "finish_fold", {}),
                roles=_FOLD_DELEGATION_ROLES,
                implement={"path": "output/main.py", "content": source},
            ),
        ]
    )
    result = run_local_interactive_worker(
        options,
        llm=llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )
    assert result["state"] == "completed"
    assert result["developer_mode"] == "llm_fold_meta_agent"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == [
        "meta_learning",
        "fold",
        "heldout",
    ]
    meta, fold, heldout = records
    assert meta["prior"] == "prefer small daily changes"
    assert fold["steps"][0]["revision_id"].startswith("revision_")
    # The manifest the Agent and later Meta sessions read is the COLLECTED
    # copy under experiments/<id>/artifacts/<run_id>/, not the sandbox's
    # host-only scratch, which is cleaned up at session end.
    manifest_ref = Path(fold["run_manifest_ref"])
    assert manifest_ref.name == "run_manifest.json"
    assert manifest_ref.is_file()
    assert manifest_ref.parent.parent == experiment / "artifacts"
    assert (manifest_ref.parent / "host_run_manifest.json").is_file()
    fold_host_manifest = json.loads(
        (manifest_ref.parent / "host_run_manifest.json").read_text(encoding="utf-8")
    )
    meta_host_manifest = json.loads(
        (
            experiment / "artifacts" / str(meta["run_id"]) / "host_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert fold_host_manifest["llm"] == {
        "provider": "scripted",
        "model": "scripted",
    }
    assert meta_host_manifest["llm"] == {
        "provider": "scripted",
        "model": "scripted",
    }
    # The property the path assertion was only ever a proxy for.
    summaries = compact_fold_history(
        fold, ref_store=AgentRefStore(experiment)
    )["backtest_summaries"]
    assert summaries, (
        "Meta history has no backtest summaries: the manifest did not survive"
    )
    assert summaries[0]["mode"] == "valid"
    assert summaries[0]["status"] == "ok"
    # The cross-fold step tree is published where the console and the worker read it.
    tree = json.loads((experiment / "steps/tree.json").read_text(encoding="utf-8"))
    assert [node["node_id"] for node in tree["nodes"]] == [fold["selected_step_id"]]
    assert heldout["result"]["total_return"] == 0.0
    assert heldout["strategy_artifact_id"] == result["final_strategy_artifact"]
    validation_ref = Path(fold["steps"][0]["validation_result_ref"])
    style = json.loads(
        (validation_ref.parent / "style_analysis.json").read_text(encoding="utf-8")
    )
    assert style["schema_version"] == 1 and style["mode"] == "valid"
    assert style["benchmark_regression"]["available"] is False
    heldout_style = json.loads(
        (Path(heldout["result_ref"]).parent / "style_analysis.json").read_text(encoding="utf-8")
    )
    assert heldout_style["schema_version"] == 1 and heldout_style["mode"] == "heldout"
    api_style = TestClient(create_app(repo)).get(
        "/api/experiments/smoke/style",
        params={
            "run_id": AgentRefStore(experiment).get_or_create(
                "run", str(fold["run_id"])
            ),
            "prefix": "valid",
        },
    )
    assert api_style.status_code == 200
    assert api_style.json() == style
    traces = sorted((experiment / "artifacts/traces").glob("*.jsonl"))
    assert len(traces) == 2
    assert all("session_start" in path.read_text(encoding="utf-8") for path in traces)
    assert not any("test_result" in path.read_text(encoding="utf-8") for path in traces)
    fold_trace = next(
        path.read_text(encoding="utf-8")
        for path in traces
        if '"session_kind": "fold"' in path.read_text(encoding="utf-8")
    )
    assert '"stage": "frozen_test"' in fold_trace
    assert '"stage": "publishing"' in fold_trace
    # Completion prunes nothing: the run evidence a later audit reads stays on disk.
    assert (options.work_root / options.experiment_id).is_dir()
    assert not any((experiment / "artifacts/strategy/revisions").iterdir())
    frozen = list((experiment / "artifacts/strategy/frozen").iterdir())
    assert len(frozen) == 1 and frozen[0].name == fold["frozen_strategy_artifact_id"]
    assert len(llm.calls) == 6
    meta_tool_names = {item["function"]["name"] for item in llm.calls[0]["tools"]}
    assert {"write_file", "finish_meta", "explore"}.issubset(meta_tool_names)
    # The Meta session may regularize the working copy, so it holds the typed
    # writers and modification_check — but it stays offline and never backtests.
    assert {"write_file", "edit_file", "modification_check", "todo"}.issubset(
        meta_tool_names
    )
    assert {"shell", "daily_backtest", "step_rollback"}.isdisjoint(meta_tool_names)
    fold_tool_names = {item["function"]["name"] for item in llm.calls[2]["tools"]}
    # Fold parent holds typed writers; shell is debug-only and must not edit strategy.
    assert {
        "ask_user",
        "daily_backtest",
        "edit_file",
        "explore",
        "finish_fold",
        "shell",
        "step_rollback",
        "todo",
        "write_file",
    }.issubset(fold_tool_names)
    assert all(
        "test_period" not in (message.content or "")
        for call in llm.calls
        for message in call["messages"]
        if message.role in {"system", "user"}
    )
    resumed = run_local_interactive_worker(
        options,
        llm=llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )
    assert resumed["heldout_runs"] == 0
    assert len(ExperimentLedger(options.rolling.ledger_path).read()) == 3
    assert len(llm.calls) == 6


def test_second_llm_fold_prompt_excludes_prior_test_diagnostic(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    params_path = experiment / "hitl/params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.update(
        {
            "first_test_period": "2026Q1",
            "last_test_period": "2026Q2",
            "heldout_first_period": "2026Q3",
            "heldout_last_period": "2026Q3",
        }
    )
    params_path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    def _fold_script(source: str) -> tuple[ProviderResponse, ...]:
        return _explore_then(
            ToolCall("check", "modification_check", {}),
            ToolCall("valid", "daily_backtest", {}),
            ToolCall("finish_fold", "finish_fold", {}),
            roles=_FOLD_DELEGATION_ROLES,
            implement={"path": "output/main.py", "content": source},
        )

    llm = ScriptedLLM(
        [
            *_explore_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer simple signals"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_fold_script("def generate_orders(context):\n    return []\n"),
            *_fold_script(
                "def generate_orders(context):\n    _ = context.inference_at\n    return []\n"
            ),
        ]
    )

    result = run_local_interactive_worker(
        load_worker_options(experiment, repo_root=repo),
        llm=llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    assert result["state"] == "completed"
    assert len(llm.calls) == 10
    second_fold_context = "\n".join(
        message.content or ""
        for message in llm.calls[6]["messages"]
        if message.role in {"system", "user"}
    )
    assert '"development_history"' in second_fold_context
    assert '"validation_result"' in second_fold_context
    assert "test_diagnostic" not in second_fold_context
    assert "test_result" not in second_fold_context


def test_local_worker_resume_skips_durable_sessions_and_heldout(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    options = load_worker_options(experiment, repo_root=repo)
    run_local_interactive_worker(options)
    before = ExperimentLedger(options.rolling.ledger_path).read()
    resumed = run_local_interactive_worker(options)
    after = ExperimentLedger(options.rolling.ledger_path).read()
    assert resumed["state"] == "completed"
    assert resumed["heldout_runs"] == 0
    assert after == before


def test_webui_worker_discards_process_output_without_log_files(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    experiment = repo / "experiments/demo"
    (experiment / "hitl").mkdir(parents=True)
    (experiment / "hitl/status.json").write_text(
        '{"schema_version":1,"state":"created"}', encoding="utf-8"
    )
    worker = repo / "scripts/experiments/run_interactive_experiment.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    class Process:
        pid = __import__("os").getpid()

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr("autotrade.webui.manager.subprocess.Popen", fake_popen)
    result = ExperimentManager(repo).start_worker("demo")

    devnull = __import__("subprocess").DEVNULL
    assert captured["stdout"] == devnull and captured["stderr"] == devnull
    assert "log_path" not in result
    assert not (repo / "logs").exists()


def test_worker_params_reject_unknown_and_partial_periods(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl" / "params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["typo_budget"] = 3
    path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown experiment parameters"):
        load_worker_options(experiment, repo_root=repo)
    params.pop("typo_budget")
    for key in ("last_test_period", "heldout_first_period", "heldout_last_period"):
        params.pop(key)
    path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="all four"):
        load_worker_options(experiment, repo_root=repo)


def test_worker_maps_data_domain_controls_to_snapshot_config(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl" / "params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "window_months": 24,
            "daily_window_months": 18,
            "fundamentals_window_months": 30,
            "events_window_months": 12,
            "macro_window_months": 36,
            "text_window_months": 6,
            "intraday_trade_days": 10,
            "include_fundamentals": True,
            "include_macro": False,
            "include_events": True,
            "include_text": True,
            "include_intraday": False,
            "fundamental_datasets": ["forecast_vip"],
            "events_datasets": ["margin", "moneyflow"],
            "text_datasets": ["anns_d"],
            "screen_exclude_st": True,
            "screen_exclude_new_listed_days": 90,
            "screen_min_circ_mv_yi": 20.0,
            "screen_max_price": 100.0,
            "screen_boards": ["main", "gem"],
        }
    )
    path.write_text(json.dumps(params), encoding="utf-8")

    config = load_worker_options(experiment, repo_root=repo).snapshot_config

    assert config.window_months == 24
    assert config.daily_window_months == 18
    assert config.fundamentals_window_months == 30
    assert config.events_window_months == 12
    assert config.macro_window_months == 36
    assert config.text_window_months == 6
    assert config.intraday_trade_days == 10
    assert config.fundamental_datasets == ("forecast_vip",)
    assert config.macro_datasets == ()
    assert config.events_datasets == ("margin", "moneyflow")
    assert config.text_datasets == ("anns_d",)
    assert config.include_intraday is False
    assert config.replay_include_minutes is False
    assert config.replay_include_macro is False
    assert config.screen_exclude_st is True
    assert config.screen_exclude_new_listed_days == 90
    assert config.screen_min_circ_mv_yi == 20.0
    assert config.screen_max_price == 100.0
    assert config.screen_boards == ("main", "gem")


def test_worker_rejects_unknown_data_domain_selection(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl" / "params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["events_datasets"] = ["not_a_dataset"]
    path.write_text(json.dumps(params), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown events_datasets"):
        load_worker_options(experiment, repo_root=repo)


def test_webui_persistent_create_uses_available_worker_entrypoint(
    tmp_path: Path, monkeypatch
):
    repository = Path(__file__).resolve().parents[2]
    manager = ExperimentManager(repository, tmp_path / "experiments")
    assert manager.worker_script.is_file()
    monkeypatch.setattr(
        manager,
        "start_worker",
        lambda experiment_id: {"spawned": True, "worker": experiment_id},
    )
    created = manager.create_experiment(
        {
            "experiment_id": "worker_smoke",
            "first_test_period": "2026Q1",
            "last_test_period": "2026Q1",
            "heldout_first_period": "2026Q2",
            "heldout_last_period": "2026Q2",
            "initial_control_mode": "manual",
        }
    )
    assert created["spawned"] is True
    assert created["worker"] == "worker_smoke"
    params = json.loads(
        (tmp_path / "experiments/worker_smoke/hitl/params.json").read_text(
            encoding="utf-8"
        )
    )
    assert params["strategy_path"] == "configs/agent_output_template/main.py"
    assert params["data_backend"] == "pit"
    assert params["execution_mode"] == "sandbox"
    assert params["developer_mode"] == "llm"


def test_console_gpu_allocation_reaches_the_run_manifests_sandbox_spec(
    tmp_path: Path,
    monkeypatch,
):
    """The whole `set_gpu_count` chain, end to end, on the real worker.

    control.json -> InteractiveExperimentRunner session context ->
    RollingExperimentPipeline.run_fold -> FoldSessionRequest ->
    LocalDeveloperBackend's derived SandboxSpec -> the run manifest the
    sandbox is started from. Asserting the manifest is what distinguishes a
    knob that works from one the console merely accepts.
    """
    from autotrade.agent import runner as agent_runner

    RealAgentSessionRunner = agent_runner.AgentSessionRunner

    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    params_path = experiment / "hitl/params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["finalize_before_deadline_seconds"] = 600
    params_path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    assembled_configs = []

    class RecordingAgentSessionRunner(RealAgentSessionRunner):
        def __init__(self, *args, **kwargs):
            assembled_configs.append(kwargs.get("config"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(agent_runner, "AgentSessionRunner", RecordingAgentSessionRunner)
    write_control(
        experiment / "hitl/control.json",
        ControlState(mode="auto", gpu_counts={"epoch_001/fold_2026Q1": 3}),
    )
    options = load_worker_options(experiment, repo_root=repo)
    source = "def generate_orders(context):\n    return []\n"
    llm = ScriptedLLM(
        [
            *_explore_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer small daily changes"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_explore_then(
                ToolCall("check", "modification_check", {}),
                ToolCall("valid", "daily_backtest", {}),
                ToolCall("finish_fold", "finish_fold", {}),
                roles=_FOLD_DELEGATION_ROLES,
                implement={"path": "output/main.py", "content": source},
            ),
        ]
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )
    assert result["state"] == "completed"
    fold = ExperimentLedger(options.rolling.ledger_path).read("fold")[0]
    manifest = json.loads(
        (Path(fold["run_manifest_ref"]).parent / "host_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    spec = manifest["sandbox_spec"]
    assert spec["gpu_count"] == 3, (
        "the console allocation never reached the sandbox spec"
    )
    # The default selection policy the count is interpreted against: without
    # gpu="auto" the count would be inert and the L20 filter unused.
    assert spec["gpu"] == "auto"
    assert spec["gpu_name_filter"] == "L20"
    fold_configs = [config for config in assembled_configs if config.mode == "fold"]
    assert len(fold_configs) == 1
    assert fold_configs[0].finalize_before_deadline_seconds == 600
    assert not hasattr(fold_configs[0], "required_explore_roles")
    meta_configs = [config for config in assembled_configs if config.mode == "meta"]
    assert not hasattr(meta_configs[0], "required_explore_roles")
    # One-shot, like every other per-session control.
    assert read_control(experiment / "hitl/control.json").gpu_counts == {}


#: (params.json key, offending value, worker error message). Every entry is a
#: value the console create form once refused in the browser through a
#: `params_schema` min/max attribute. Those open-only attributes were removed;
#: the guard that actually protects the run lives in the worker, and this is
#: where it is proved to still be there.
_REMOVED_BROWSER_BOUNDS = (
    ("epochs", 0, "epochs must be a positive integer"),
    (
        "meta_memory_max_epochs",
        -1,
        "meta_memory_max_epochs must be a non-negative integer",
    ),
    ("window_months", 0, "window_months must be a positive integer"),
    ("daily_window_months", 0, "daily_window_months must be a positive integer"),
    (
        "fundamentals_window_months",
        0,
        "fundamentals_window_months must be a positive integer",
    ),
    ("macro_window_months", 0, "macro_window_months must be a positive integer"),
    ("events_window_months", 0, "events_window_months must be a positive integer"),
    ("text_window_months", 0, "text_window_months must be a positive integer"),
    ("intraday_trade_days", 0, "intraday_trade_days must be a positive integer"),
    (
        "screen_exclude_new_listed_days",
        -1,
        "screen_exclude_new_listed_days must be a non-negative integer",
    ),
    (
        "screen_min_circ_mv_yi",
        -1.0,
        "screen_min_circ_mv_yi must be a non-negative finite number",
    ),
    (
        "screen_max_circ_mv_yi",
        -1.0,
        "screen_max_circ_mv_yi must be a non-negative finite number",
    ),
    ("screen_min_price", -1.0, "screen_min_price must be a non-negative finite number"),
    ("screen_max_price", -1.0, "screen_max_price must be a non-negative finite number"),
    ("max_fold_minutes", 0, "max_fold_minutes must be a positive integer"),
    ("max_drawdown", 1.5, "max_drawdown must be between 0.0 and 1.0"),
    ("max_drawdown", -0.5, "max_drawdown must be a non-negative finite number"),
    ("max_steps_per_fold", 0, "max_steps_per_fold must be a positive integer"),
    ("max_backtests_per_fold", 0, "max_backtests_per_fold must be a positive integer"),
    ("max_llm_calls", 0, "max_llm_calls must be a positive integer"),
    ("initial_cash", 0.0, "initial_cash must be a positive finite number"),
    ("analysis_max_tokens", 0, "analysis_max_tokens must be a positive integer"),
    (
        "compact_token_threshold",
        0,
        "compact_token_threshold must be a positive integer",
    ),
    (
        "compact_keep_recent_messages",
        0,
        "compact_keep_recent_messages must be a positive integer",
    ),
    ("compact_max_tokens", 0, "compact_max_tokens must be a positive integer"),
    ("compact_max_calls", -1, "compact_max_calls must be a non-negative integer"),
)


def test_the_worker_rejects_every_value_the_create_form_no_longer_bounds(
    tmp_path: Path,
    monkeypatch,
    subtests,
):
    """The browser is an affordance; the worker is the guard.

    `params_schema` no longer carries min/max attributes for these keys, so a
    value typed past the form (or posted straight to the API) reaches
    `params.json`. It must then fail fast and explicitly at worker startup
    rather than configuring a nonsense run.
    """
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    path = experiment / "hitl/params.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    # Without this the probes below would prove nothing: a fixture that cannot
    # load makes every rejection look like a guard.
    load_worker_options(experiment, repo_root=repo)
    for key, value, message in _REMOVED_BROWSER_BOUNDS:
        with subtests.test(key=key, value=value):
            path.write_text(json.dumps({**baseline, key: value}), encoding="utf-8")
            with pytest.raises(ValueError, match=re.escape(message)):
                load_worker_options(experiment, repo_root=repo)
    path.write_text(json.dumps(baseline), encoding="utf-8")


def _console_params(repo: Path, experiments_root: Path, **overrides) -> dict:
    """The exact params dict `create_experiment` hands to the pre-flight."""
    from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS, WEB_INTERNAL_PARAMS

    params = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in WEB_CREATE_DEFAULTS.items()
    }
    params.update(WEB_INTERNAL_PARAMS)
    params.update(
        {
            "experiment_id": "narrowing",
            "experiments_root": str(experiments_root),
            "work_root": str(repo / ".runtime/sandboxes"),
            "_creation_surface": "webui",
        }
    )
    params.update(overrides)
    return params


def test_preflight_narrows_exactly_three_things_and_nothing_else(
    tmp_path: Path, monkeypatch
):
    """`preflight=True` validates the request; `preflight=False` the deployment.

    The pre-flight exists so the console can reject a bad create before it
    writes anything, which means it must skip the three checks that describe
    the host rather than the request: input-path existence, the API key, and
    the release-pin/calendar steps. Each is pinned at the stage where it starts
    to matter, so a future edit cannot quietly widen the exemption into (say)
    skipping a parameter check.
    """
    from autotrade.pipelines.worker import resolve_worker_options

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    repo = tmp_path / "repo"
    experiments_root = repo / "experiments"
    experiments_root.mkdir(parents=True)
    directory = experiments_root / "narrowing"

    def resolve(preflight: bool, **overrides):
        return resolve_worker_options(
            _console_params(repo, experiments_root, **overrides),
            experiment_dir=directory,
            repo_root=repo,
            preflight=preflight,
        )

    # (1) input paths: nothing on disk yet.
    assert resolve(True) is not None
    with pytest.raises(FileNotFoundError):
        resolve(False)

    template = repo / "configs/agent_output_template"
    template.mkdir(parents=True)
    (template / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    (repo / "data/raw").mkdir(parents=True)
    (repo / "data/pit/fundamental_events").mkdir(parents=True)

    # (2) the API key: deployment state, reported by /api/health.
    assert resolve(True) is not None
    with pytest.raises(ValueError, match="requires the gateway API key"):
        resolve(False)

    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    # (3) the research-release pin and the calendar-dependent fold schedule.
    assert resolve(True) is not None
    with pytest.raises(RuntimeError, match="lacks configured raw datasets"):
        resolve(False)

    # What it does NOT narrow: repository containment is still enforced, so the
    # relaxed path check cannot be used to reach outside the repo. (Checked
    # without the console surface marker, which rejects these keys earlier.)
    for key, escape, label in (
        ("strategy_path", "../evil.py", "baseline strategy"),
        ("strategy_path", "/etc/passwd", "baseline strategy"),
        ("raw_dir", "../outside", "raw_dir"),
        ("raw_dir", "/etc", "raw_dir"),
    ):
        params = _console_params(repo, experiments_root, **{key: escape})
        params.pop("_creation_surface")
        with pytest.raises(
            ValueError, match=f"{label} must stay inside the repository"
        ):
            resolve_worker_options(
                params, experiment_dir=directory, repo_root=repo, preflight=True
            )

    # And every parameter check still runs in pre-flight mode.
    with pytest.raises(ValueError, match="gpu_count must be between 0 and 4"):
        resolve(True, gpu_count=9)
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        resolve(True, epochs=-1)
