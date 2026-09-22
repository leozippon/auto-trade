"""The worker's parameter contract, its model roles and its runner controls.

The research arm itself runs end to end in ``test_research_arm_worker.py``;
the scripted Agent helpers at the bottom of this module are shared with it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

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
from autotrade.environment.nl import NLConfig
from autotrade.environment.tools import CommandResult
from autotrade.pipelines.hitl_state import (
    ControlState,
    PlannedSession,
    read_control,
    read_status,
    write_control,
)
from autotrade.pipelines.interactive import InteractiveExperimentRunner
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.local_backend import SessionBudgetLLM, SessionCallBudget
from autotrade.pipelines.worker import (
    NL_REASONING_EFFORT,
    load_worker_options,
)
from autotrade.webui.manager import ExperimentManager
from tests.unit.gpu_probe import stubbed_gpu_probe

_FOLD_DELEGATION_ROLES = ("Explore", "general-purpose")


def _experiment(
    tmp_path: Path, *, developer_mode: str = "baseline", max_replay_years: int = 1
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
    # Reaches into the default Held-out, which is all the loader checks.
    days = pd.bdate_range("2026-06-01", "2026-09-30")
    pd.DataFrame(
        {
            "trade_date": [stamp.strftime("%Y%m%d") for stamp in days],
            "symbol": ["000001.SZ"] * len(days),
            "open": [10.0] * len(days),
            "close": [10.0] * len(days),
            "pre_close": [10.0] * len(days),
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
                "max_replay_years": max_replay_years,
            }
        ),
        encoding="utf-8",
    )
    (hitl / "control.json").write_text(
        json.dumps({"schema_version": 1, "mode": "auto"}), encoding="utf-8"
    )
    return repo, experiment



def test_worker_rejects_llm_mode_without_provider_credentials(tmp_path: Path):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    with pytest.raises(ValueError, match="requires an API key: set VLLM_API_KEY"):
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
    assert settings.subagent_model == settings.model
    main = settings.build_gateway("main").config
    nl = settings.build_gateway("nl").config
    compact = settings.build_gateway("compact").config
    assert (main.model, nl.model, compact.model) == (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-pro",
    )
    assert main.thinking_enabled is False and nl.thinking_enabled is False
    # Thinking is off, so no role sends a reasoning effort.
    assert main.reasoning_effort is None and nl.reasoning_effort is None
    assert compact.thinking_enabled is False and compact.reasoning_effort is None
    assert compact.max_tokens == 1_200
    assert settings.compact_enabled is True
    # The configured 90,000 is clamped to the DeepSeek bound
    # 128,000 − 32,768 output ceiling − 8,192 margin.
    assert settings.compaction.token_threshold == 87_040
    assert settings.compaction.keep_recent_messages == 10
    assert settings.compaction.max_response_tokens == 1_200
    assert settings.compaction.max_calls == 4



def test_nl_cost_controls_reach_the_worker_without_being_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NL is the only real model inference inside a backtest's wall clock.

    Both of its cost controls have to hold with nothing configured: a real
    per-backtest call ceiling, and a reasoning tier of its own instead of the
    strategy-design dialogues' effort.
    """
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    options = load_worker_options(experiment, repo_root=repo)
    # Unset: the evaluation backend derives it from each replay's own length.
    assert options.nl_config.max_total_calls is None
    assert options.nl_config.max_calls_per_decision == NLConfig().max_calls_per_decision

    settings = options.llm
    assert settings is not None
    assert settings.reasoning_effort == "xhigh"
    assert settings.thinking_enabled is True
    # The default is the native `xhigh` tier and passes through the local Qwen
    # profile unmapped; NL keeps its own native tier.
    assert settings.build_gateway("main").config.reasoning_effort == "xhigh"
    assert settings.build_gateway("nl").config.reasoning_effort == NL_REASONING_EFFORT
    assert NL_REASONING_EFFORT == "medium"



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
            "subagent_model": LEGACY_LOCAL_QWEN_MODEL,
            "nl_model": LEGACY_LOCAL_QWEN_MODEL,
            "compact_model": LEGACY_LOCAL_QWEN_MODEL,
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
        options.llm.subagent_model,
        options.llm.nl_model,
        options.llm.compact_model,
    ) == (LOCAL_QWEN_MODEL,) * 4
    assert path.read_bytes() == persisted



def test_worker_ignores_historical_endpoint_and_credential_params(
    tmp_path: Path,
    monkeypatch,
):
    """Neither endpoint nor credential can be redirected by a params file.

    Both keys are retired operator overrides that historical snapshots may
    still carry. A ``llm_api_key_env`` naming any other variable in the
    process environment would otherwise send that secret to the provider as a
    bearer token, which is exactly what the trusted profile exists to prevent.
    """

    from autotrade.pipelines.worker import _ALLOWED_PARAMS, NON_PERSISTABLE_PARAMS

    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["llm_base_url"] = "https://untrusted-snapshot.example.test/v1"
    params["llm_api_key_env"] = "UNRELATED_SECRET"
    params["subagent_model"] = "deepseek-v4-pro"
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("UNRELATED_SECRET", "not-a-provider-credential")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    monkeypatch.setenv("VLLM_BASE_URL", "https://trusted-runtime.example.test/v1")

    settings = load_worker_options(experiment, repo_root=repo).llm
    assert settings is not None
    for key in ("llm_base_url", "llm_api_key_env"):
        assert key in NON_PERSISTABLE_PARAMS
        assert key not in _ALLOWED_PARAMS
    main = settings.build_gateway("main").config
    assert main.base_url == "https://trusted-runtime.example.test/v1"
    assert main.api_key == "local-test-key"
    assert settings.build_gateway("subagent").config.api_key == "deepseek-test-key"



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
            "subagent_model": "deepseek-v4-pro",
            "nl_model": "deepseek-v4-flash",
            "compact_model": "deepseek-v4-flash",
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
    subagent = options.llm.build_gateway("subagent")
    nl = options.llm.build_gateway("nl")
    assert isinstance(main, OpenAICompatibleProxy)
    assert main.provider == "vllm"
    assert isinstance(subagent, DeepSeekProxy)
    assert subagent.provider == "deepseek"
    assert subagent.model == "deepseek-v4-pro"
    assert isinstance(nl, DeepSeekProxy)
    assert nl.provider == "deepseek"
    assert main.config.max_tokens == 32_768
    assert main.config.timeout_seconds == 120
    assert main.config.reasoning_effort == "xhigh"
    with pytest.raises(ValueError, match="unknown model role: meta"):
        options.llm.model_for("meta")



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
            "nl_model": LOCAL_QWEN_MODEL,
            "compact_model": LOCAL_QWEN_MODEL,
            "compact_token_threshold": 20_000,
            "compact_max_tokens": 20_000,
        }
    )
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    options = load_worker_options(experiment, repo_root=repo)
    assert options.llm is not None
    assert options.llm.compaction.max_response_tokens == 20_000
    assert options.llm.max_tokens_for("main") == 32_768
    assert options.llm.max_tokens_for("nl", requested=1_200) == 1_200
    assert options.llm.max_tokens_for("nl", requested=20_000) == 20_000
    for role in ("main", "subagent", "nl"):
        assert options.llm.build_gateway(role).config.max_tokens == 32_768
    assert options.llm.build_gateway("compact").config.max_tokens == 20_000



def test_worker_derives_the_compaction_threshold_from_the_model_context(
    tmp_path: Path,
    monkeypatch,
):
    """The default threshold is window − output ceiling − margin, and a
    configured value is clamped to that same bound."""
    from autotrade.environment.llm import AGENT_MAX_OUTPUT_TOKENS
    from autotrade.pipelines.worker import COMPACTION_SAFETY_MARGIN_TOKENS

    assert COMPACTION_SAFETY_MARGIN_TOKENS == 8_192
    derived = 262_144 - AGENT_MAX_OUTPUT_TOKENS - COMPACTION_SAFETY_MARGIN_TOKENS
    assert derived == 221_184
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["model"] = LOCAL_QWEN_MODEL
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    for absent in (None, ""):
        params["compact_token_threshold"] = absent
        path.write_text(json.dumps(params), encoding="utf-8")
        options = load_worker_options(experiment, repo_root=repo)
        assert options.llm.compaction.token_threshold == derived
    params.pop("compact_token_threshold")
    path.write_text(json.dumps(params), encoding="utf-8")
    assert load_worker_options(experiment, repo_root=repo).llm.compaction.token_threshold == derived
    params["compact_token_threshold"] = 300_000
    path.write_text(json.dumps(params), encoding="utf-8")
    assert load_worker_options(experiment, repo_root=repo).llm.compaction.token_threshold == derived
    params["compact_token_threshold"] = 90_000
    path.write_text(json.dumps(params), encoding="utf-8")
    assert load_worker_options(experiment, repo_root=repo).llm.compaction.token_threshold == 90_000
    # A larger output ceiling lowers the derived threshold by the same amount.
    params["compact_token_threshold"] = None
    params["llm_max_response_tokens"] = AGENT_MAX_OUTPUT_TOKENS + 1_000
    path.write_text(json.dumps(params), encoding="utf-8")
    assert load_worker_options(experiment, repo_root=repo).llm.compaction.token_threshold == derived - 1_000



def test_worker_default_threshold_fits_deepseek_and_compaction_can_be_disabled(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["model"] = "deepseek-v4-flash"
    params.pop("compact_token_threshold", None)
    path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    # 128,000 - 32,768 output ceiling - 8,192 margin: shipped defaults launch.
    assert options.llm.compaction.token_threshold == 87_040
    params["disable_context_compact"] = True
    path.write_text(json.dumps(params), encoding="utf-8")
    disabled = load_worker_options(experiment, repo_root=repo)
    assert disabled.llm.compact_enabled is False



def test_worker_gives_subagents_their_own_gateway_and_compaction_budget(
    tmp_path: Path,
    monkeypatch,
):
    """``subagent_model`` is a second gateway with the parent's quota, and
    every conversation role compacts at its own model's window − output
    budget − margin, bounded by what the compaction model can read."""
    from autotrade.pipelines.worker import resolve_worker_options

    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "model": "deepseek-v4-flash",
            "subagent_model": LOCAL_QWEN_MODEL,
            "nl_model": LOCAL_QWEN_MODEL,
            "compact_model": LOCAL_QWEN_MODEL,
            "compact_max_tokens": 1_600,
        }
    )
    params.pop("compact_token_threshold", None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    def settings_for(**overrides: object):
        params.update(overrides)
        path.write_text(json.dumps(params), encoding="utf-8")
        return load_worker_options(experiment, repo_root=repo).llm

    settings = settings_for()
    assert settings.subagent_model == LOCAL_QWEN_MODEL
    child = settings.build_gateway("subagent")
    assert isinstance(child, OpenAICompatibleProxy)
    assert (child.provider, child.model) == ("vllm", LOCAL_QWEN_MODEL)
    assert child.config.thinking_enabled and child.config.reasoning_effort == "xhigh"
    assert child.config.max_tokens == 32_768
    assert settings.build_gateway("main").provider == "deepseek"
    # Parents on the 128,000 window, children on the 262,144 one.
    assert settings.compaction.token_threshold == 87_040
    assert settings.compaction_for("main") == settings.compaction
    assert settings.compaction_for("subagent").token_threshold == 221_184
    with pytest.raises(ValueError, match="unknown conversation role"):
        settings.compaction_for("nl")
    # A DeepSeek compaction model reads the whole child conversation:
    # 128,000 − 1,600 − 8,192 bounds the children too.
    settings = settings_for(compact_model="deepseek-v4-flash")
    assert settings.compaction.token_threshold == 87_040
    assert settings.compaction_for("subagent").token_threshold == 118_208
    # A configured threshold is clamped per role.
    settings = settings_for(compact_model=LOCAL_QWEN_MODEL, compact_token_threshold=100_000)
    assert settings.compaction.token_threshold == 87_040
    assert settings.compaction_for("subagent").token_threshold == 100_000
    # Every gateway is validated at create preflight without credentials.
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.delenv("VLLM_API_KEY")
    with pytest.raises(ValueError, match="API key"):
        load_worker_options(experiment, repo_root=repo)
    assert (
        resolve_worker_options(
            params, experiment_dir=experiment, repo_root=repo, preflight=True
        ).llm.subagent_model
        == LOCAL_QWEN_MODEL
    )



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



def test_interactive_runner_publishes_current_session_timing(tmp_path: Path):
    session_key = "research"
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
        # The real executor records the session through the pipeline, which is
        # also what stamps the session key; the runner never appends for it.
        ledger.append(
            {
                "record_type": "research_session",
                "experiment_id": "demo",
                "epoch_id": "research",
                "fold_id": session.session_key,
                "run_id": "run_001",
                "session_key": session.session_key,
            }
        )

    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(PlannedSession(session_key, "research"),),
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
    assert captured["run_id"] == "run_001"
    assert captured["environment_stage"] == "pit_snapshot"
    assert captured["environment_stage_started_at"]
    timing = captured["timing"]
    assert isinstance(timing, dict)
    assert timing["run_wall_seconds"] >= 0.0



def test_session_boundary_restart_keeps_the_finished_session_and_stops_the_next(
    tmp_path: Path,
):
    """A deferred restart costs no work: the session in flight is recorded,
    the forward replay is not started, and the entrypoint is told to re-exec."""

    control_path = tmp_path / "control.json"
    status_path = tmp_path / "status.json"
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl")
    write_control(control_path, ControlState(mode="auto"))
    ran: list[str] = []

    def execute(session, context):
        del context
        ran.append(session.session_key)
        # The console records the request while this session is running.
        pending = read_control(control_path)
        pending.restart_pending = True
        write_control(control_path, pending)
        ledger.append(
            {
                "record_type": "research_session",
                "experiment_id": "demo",
                "epoch_id": "research",
                "fold_id": session.session_key,
                "run_id": f"run_{len(ran):03d}",
                "session_key": session.session_key,
            }
        )

    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(
            PlannedSession("research", "research"),
            PlannedSession("forward", "forward"),
        ),
        execute_session=execute,
        ledger=ledger,
        control_path=control_path,
        status_path=status_path,
        ref_store=AgentRefStore(tmp_path / "experiment"),
        poll_seconds=0.01,
    )

    result = runner.run()

    assert result == {"status": "restart", "sessions_run": 1}
    assert ran == ["research"]
    assert [row["session_key"] for row in ledger.read()] == ["research"]
    # One-shot, and the console sees one worker coming up rather than a stop.
    assert read_control(control_path).restart_pending is False
    assert read_status(status_path)["state"] == "launching"



def test_session_boundary_restart_is_taken_before_the_next_session_starts(
    tmp_path: Path,
):
    """Requested before a session starts, the swap happens at the gate: the
    next session must not run hours of the old code first."""

    control_path = tmp_path / "control.json"
    status_path = tmp_path / "status.json"
    write_control(control_path, ControlState(mode="manual", restart_pending=True))

    def execute(session, context):  # pragma: no cover - must not run
        del session, context
        raise AssertionError("the gate started a session instead of restarting")

    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(PlannedSession("research", "research"),),
        execute_session=execute,
        ledger=ExperimentLedger(tmp_path / "ledger.jsonl"),
        control_path=control_path,
        status_path=status_path,
        poll_seconds=0.01,
    )

    result = runner.run()

    assert result["status"] == "restart" and result["sessions_run"] == 0
    assert read_control(control_path).restart_pending is False



@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model", "unknown-model", "unsupported DeepSeek model"),
        ("subagent_model", "unknown-model", "unsupported DeepSeek model"),
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



# The working copy as a one-candidate batch_validate round: the scripted
# sessions write output/ through a child and validate it as it stands.
VALIDATE_WORKING_COPY = ToolCall(
    "valid",
    "batch_validate",
    {
        "offline_trials": 0,
        "candidates": [
            {
                "name": "working_copy",
                "hypothesis": "the implemented working copy beats its baseline",
                "path": "output",
                "control": False,
            }
        ],
    },
)



# A scripted finish that names the working-copy row it has just read: node ids
# carry the session's opaque run ref, which a script cannot know in advance.
LAST_WORKING_COPY_NODE = "<working copy node>"



class _NominatingLLM(ScriptedLLM):
    def complete(self, messages, **kwargs):
        response = super().complete(messages, **kwargs)
        if not any(
            call.arguments.get("node_id") == LAST_WORKING_COPY_NODE
            for call in response.tool_calls
        ):
            return response
        rows = re.findall(
            r'"name":\s*"working_copy".*?"node_id":\s*"([^"]+)"',
            "\n".join(message.content or "" for message in messages),
            flags=re.DOTALL,
        )
        return ProviderResponse(
            tool_calls=tuple(
                ToolCall(call.id, call.name, {**call.arguments, "node_id": rows[-1]})
                if call.arguments.get("node_id") == LAST_WORKING_COPY_NODE
                else call
                for call in response.tool_calls
            )
        )



def _agent_then(
    *tool_calls: ToolCall,
    roles: tuple[str, ...] = ("Explore",),
    summary: str = "委托完成",
    implement: dict[str, object] | None = None,
) -> tuple[ProviderResponse, ...]:
    launches = tuple(
        ToolCall(f"ex_{role}", "agent", {"agent": role, "task": f"review {role}"})
        for role in roles
    )
    responses: list[ProviderResponse] = [
        ProviderResponse(tool_calls=(*launches, *tool_calls))
    ]
    for role in roles:
        if implement is not None and role == "general-purpose":
            responses.append(
                ProviderResponse(
                    tool_calls=(
                        ToolCall("w", "write_file", dict(implement)),
                    )
                )
            )
        responses.append(
            ProviderResponse(
                content=summary,
                usage={"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57},
            )
        )
    return tuple(responses)



def test_webui_worker_output_is_recoverable_from_a_per_experiment_log(
    tmp_path: Path, monkeypatch
):
    """A worker traceback must survive the run that produced it.

    stdout/stderr used to go to /dev/null, so an unhandled exception in a live
    session was unrecoverable — the run could only be described as "it stopped".
    """
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

    log_path = repo / "logs" / "workers" / "demo.log"
    # Repo-relative only: this value crosses the public API/status boundary.
    assert result["worker_log"] == "logs/workers/demo.log"
    assert not Path(result["worker_log"]).is_absolute()
    # stdin stays closed; stderr is folded into the same stream so an
    # interleaved traceback keeps its ordering.
    assert captured["stdin"] == __import__("subprocess").DEVNULL
    assert captured["stderr"] == __import__("subprocess").STDOUT
    assert captured["stdout"].name == str(log_path)
    # The parent releases its handle; the detached child keeps writing.
    assert captured["stdout"].closed

    status = json.loads(
        (experiment / "hitl/status.json").read_text(encoding="utf-8")
    )
    assert status["worker_log"] == "logs/workers/demo.log"
    assert str(repo) not in json.dumps(status)
    # Appended, never truncated: a restart must not erase the crash before it.
    assert "===== worker start" in log_path.read_text(encoding="utf-8")
    (experiment / "hitl/status.json").write_text(
        '{"schema_version":1,"state":"created"}', encoding="utf-8"
    )
    ExperimentManager(repo).start_worker("demo")
    assert log_path.read_text(encoding="utf-8").count("===== worker start") == 2



def test_worker_params_reject_unknown_keys_fold_era_keys_and_a_bad_geometry(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl" / "params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    options = load_worker_options(experiment, repo_root=repo)
    assert options.rolling.geometry.research_end == "20250630"
    assert options.rolling.max_research_minutes == 2400
    for key, value in (
        ("typo_budget", 3),
        ("development_first_period", "2022"),
        ("epochs", 1),
        ("inherit_from", "other"),
        ("meta_model", LOCAL_QWEN_MODEL),
    ):
        path.write_text(json.dumps({**params, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown experiment parameters"):
            load_worker_options(experiment, repo_root=repo)
    for key, value, message in (
        ("research_start", "20210101", "July 1"),
        ("forward_end", "20261231", "twelve months after research"),
        ("research_end", 20250630, "YYYYMMDD string"),
        ("max_research_minutes", 0, "max_research_minutes must be a positive integer"),
    ):
        path.write_text(json.dumps({**params, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
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
    with stubbed_gpu_probe():
        created = manager.create_experiment(
            {"experiment_id": "worker_smoke", "initial_control_mode": "manual"}
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



def test_a_cpu_only_allocation_survives_the_control_round_trip(tmp_path: Path) -> None:
    """0 GPUs is a request, not an unset value.

    The console accepts 0..4 and every layer below honours 0 explicitly
    (``_optional_gpu_count``, the developer's ``gpu=None, gpu_count=0``), so a
    reader that drops it silently runs the CPU-only session on the experiment
    default.
    """

    control = tmp_path / "control.json"
    write_control(
        control,
        ControlState(
            mode="auto",
            gpu_counts={"research": 0},
        ),
    )
    state = read_control(control)
    assert state.gpu_counts == {"research": 0}



#: (params.json key, offending value, worker error message). Every entry is a
#: value the console create form once refused in the browser through a
#: `params_schema` min/max attribute. Those open-only attributes were removed;
#: the guard that actually protects the run lives in the worker, and this is
#: where it is proved to still be there.
_REMOVED_BROWSER_BOUNDS = (
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
    ("max_research_minutes", 0, "max_research_minutes must be a positive integer"),
    ("max_drawdown", 1.5, "max_drawdown must be between zero and one"),
    ("max_drawdown", -0.5, "max_drawdown must be between zero and one"),
    ("active_max_drawdown", 1.5, "active_max_drawdown must be between zero and one"),
    ("tracking_error_cap", -0.08, "tracking_error_cap must be positive"),
    # This arm's capital derives no tracking mandate, so a band has no cap.
    ("beta_min", 0.9, "are set together or not at all"),
    ("max_replay_years", 0, "max_replay_years must be a positive integer"),
    ("max_llm_calls", 0, "max_llm_calls must be a positive integer"),
    ("initial_cash", 0.0, "initial_cash must be a positive finite number"),
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
    with pytest.raises(ValueError, match="requires an API key: set VLLM_API_KEY"):
        resolve(False)

    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    # (3) the research-release pin and the release's reach into Held-out.
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
    with pytest.raises(ValueError, match="max_research_minutes must be a positive integer"):
        resolve(True, max_research_minutes=-1)



def test_compaction_gateway_makes_one_attempt(tmp_path: Path, monkeypatch):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    settings = load_worker_options(experiment, repo_root=repo).llm
    assert settings.build_gateway("compact", max_retries=0).config.max_retries == 0
    assert settings.build_gateway("main").config.max_retries == settings.max_retries



def test_worker_reasoning_effort_offers_wire_levels_and_maps_legacy_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The console enum is exactly what the local Qwen template distinguishes;
    older params.json files that still say high/max keep launching as xhigh."""
    from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
    from autotrade.pipelines.worker import DEFAULT_REASONING_EFFORT, REASONING_EFFORTS
    from autotrade.webui.params_schema import _FIELDS

    assert REASONING_EFFORTS == ("low", "medium", "xhigh")
    assert WEB_CREATE_DEFAULTS["reasoning_effort"] == DEFAULT_REASONING_EFFORT == "xhigh"
    field = next(item for item in _FIELDS if item["key"] == "reasoning_effort")
    assert set(field["choices"]) == set(REASONING_EFFORTS)

    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    path = experiment / "hitl/params.json"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    for configured, effective in (("max", "xhigh"), ("high", "xhigh"), ("medium", "medium")):
        params = json.loads(path.read_text(encoding="utf-8"))
        params["reasoning_effort"] = configured
        path.write_text(json.dumps(params), encoding="utf-8")
        settings = load_worker_options(experiment, repo_root=repo).llm
        assert settings is not None
        assert settings.reasoning_effort == effective
        assert settings.build_gateway("main").config.reasoning_effort == effective

