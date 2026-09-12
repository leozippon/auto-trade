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
from autotrade.environment.nl import NLConfig
from autotrade.environment.runtime import chmod_tree
from autotrade.environment.tools import CommandResult
from autotrade.pipelines import worker
from autotrade.pipelines.agent_views import compact_fold_history
from autotrade.pipelines.config import FoldSessionResult
from autotrade.pipelines.hitl_state import (
    ControlState,
    DevelopmentSession,
    read_control,
    read_status,
    write_control,
)
from autotrade.pipelines.inherited_memory import (
    import_inherited_memory,
    inherited_prior_header,
    load_inherited_memory,
    prior_provenance,
)
from autotrade.pipelines.interactive import InteractiveExperimentRunner
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.local_backend import SessionBudgetLLM, SessionCallBudget
from autotrade.pipelines.prior import ExperimentPriorStore
from autotrade.pipelines.skills import ExperimentSkillsStore
from autotrade.pipelines.worker import (
    NL_REASONING_EFFORT,
    _heldout_epoch_id,
    load_worker_options,
    run_local_interactive_worker,
)
from autotrade.webui.manager import ExperimentManager
from autotrade.webui.server import create_app

_FOLD_DELEGATION_ROLES = ("auditor", "developer")


def _experiment(
    tmp_path: Path, *, developer_mode: str = "baseline", max_backtests: int = 1
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
                "epochs": 1,
                # One Validation is the whole Fold budget by default: these
                # sessions script a single daily_backtest, and finish_fold
                # waives its batch-round floor only when no round fits. A test
                # that exercises the floor or the early-stop gate raises it.
                "max_backtests_per_fold": max_backtests,
                # Two-Fold windows: these sessions script their own nomination
                # in every Fold, so the confirmation tail is switched off here
                # and covered by its own tests (test_fold_calendar,
                # test_finish_fold, and the regular-Fold worker test below).
                "confirmation_folds": 0,
                "fold_period": "quarter",
                # Rolling design: validation 2025Q4, frozen Test 2026Q1. The
                # default single-window design is exercised separately.
                "development_first_period": "2025Q4",
                "development_last_period": "2026Q1",
                "test_stage": True,
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


def test_local_worker_regular_folds_go_straight_to_held_out(tmp_path: Path):
    """The default research design end to end: one regular Fold per period,
    no frozen Test, the host's parent control before every Fold that has a
    parent, an automatic Held-out replay, and the graduation verdict on the
    ledger and the terminal status."""
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(
        {
            "development_first_period": "2025Q4",
            "development_last_period": "2026Q1",
            "test_stage": False,
            # One confirmation Fold: the second Fold may only keep the parent,
            # which is what this stub developer nominates anyway, and term (c)
            # then reads that one transition in the verdict below.
            "confirmation_folds": 1,
        }
    )
    path.write_text(json.dumps(params), encoding="utf-8")
    seed = _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    assert options.rolling.test_stage is False
    result = run_local_interactive_worker(options)
    assert result["state"] == "completed"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == ["fold", "fold", "heldout"]
    first, second, heldout = records
    assert (first["fold_id"], second["fold_id"]) == ("fold_2025Q4", "fold_2026Q1")
    assert first["validation_period"] == "20251001..20251231"
    assert second["validation_period"] == "20260101..20260331"
    for fold in (first, second):
        assert fold["test_period"] is None
        assert fold["test_result"] is None
        assert fold["snapshot_ids"]["test_decision_input"] is None
    assert not list((experiment / "artifacts/results").glob("frozen_test_*"))
    # Selection statistics ride on every Fold record. The deterministic
    # baseline runs exactly one candidate, so the trial count is honest and
    # the deflated Sharpe reports itself unavailable rather than 0.
    for fold in (first, second):
        statistics = fold["selection_statistics"]
        assert statistics["candidates_evaluated"] == 1
        assert statistics["deflated_sharpe_probability"] is None
        assert statistics["unavailable_reason"] == "fewer_than_two_trials"
    # Parent carry-forward: this experiment starts from an inherited seed (a
    # parentless first Fold could only freeze a baseline anchor, which is a
    # control the run refuses to deliver), so the host replayed a parent
    # control before both Folds and the lineage head is a real artifact.
    assert first["parent_control"]["parent_strategy_artifact_id"] == seed
    # The local daily fixture ships no benchmark series, so the excess deltas
    # stay unavailable instead of invented.
    assert second["vs_parent"] == {
        "excess_return_delta": None,
        "neutralized_excess_return_delta": None,
        "max_drawdown_delta": 0.0,
        "beats_parent": None,
    }
    assert second["parent_strategy_artifact_id"] == first["frozen_strategy_artifact_id"]
    # The deterministic local developer edits nothing, so every Fold's
    # nomination is the inherited parent itself: the lineage head is retained
    # rather than reissued under a second id, and the row says why.
    assert (first["fold_status"], second["fold_status"]) == ("no_update", "no_update")
    assert first["nominated_identical_to_parent"] is True
    assert second["nominated_identical_to_parent"] is True
    assert first["frozen_strategy_artifact_id"] == seed
    assert second["frozen_strategy_artifact_id"] == first["frozen_strategy_artifact_id"]
    control = second["parent_control"]
    assert control["status"] == "ok"
    assert control["parent_strategy_artifact_id"] == first["frozen_strategy_artifact_id"]
    assert control["validation_result"]["total_return"] == 0.0
    assert Path(control["validation_result_ref"]).is_file()
    plan = json.loads((experiment / "hitl/schedule.json").read_text(encoding="utf-8"))
    assert [row["kind"] for row in plan["sessions"]] == ["fold", "fold", "heldout"]
    # The deterministic baseline holds cash (zero Sharpe) and the local daily
    # fixture carries no benchmark series (so no neutralized excess either), so
    # the verdict names all three, and the one walk-forward transition (cash vs
    # no benchmark) proves nothing: with neither a recorded neutralized excess
    # nor a style sidecar to derive one from, its grade is unmeasurable and the
    # verdict says so in its own reason rather than reading the raw excess
    # instead. That transition replayed the very artifact Held-out ships (the
    # second Fold kept it), so term (c) has evidence to read and finds it
    # unproven.
    assert heldout["verdict"]["status"] == "discarded"
    assert heldout["verdict"]["reasons"] == [
        "missing_benchmark_return",
        "missing_neutralized_excess_return",
        "sharpe_not_positive",
        "walkforward_excess_inconsistent(0/1<1)",
        "missing_transition_neutralized_excess(1/1)",
        "final_artifact_forward_excess_inconsistent(0/1<1)",
    ]
    assert result["verdict"]["status"] == "discarded"
    assert result["verdict"]["periods"][0]["period"] == "2026Q2"
    status = read_status(experiment / "hitl/status.json")
    assert status["verdict"] == result["verdict"]


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
    _seed_parent(experiment)
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
        {
            "label": "2026Q2",
            "start": "20260401",
            "end": "20260630",
            "requested_end": "20260630",
            "truncation_reason": None,
        }
    ]


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
    assert settings.meta_model == settings.model
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
            "meta_model": LEGACY_LOCAL_QWEN_MODEL,
            "subagent_model": LEGACY_LOCAL_QWEN_MODEL,
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
        options.llm.subagent_model,
        options.llm.nl_model,
        options.llm.compact_model,
        options.analysis_model,
    ) == (LOCAL_QWEN_MODEL,) * 6
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
    params["meta_model"] = "deepseek-v4-pro"
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
    assert settings.build_gateway("meta").config.api_key == "deepseek-test-key"


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
    # The analysis model is not one of this dataclass's roles: both call sites
    # pass it, and any default here could only be another role's model.
    with pytest.raises(ValueError, match="unknown model role: analysis"):
        options.llm.model_for("analysis")


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
            "meta_model": "deepseek-v4-flash",
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
    assert settings.build_gateway("meta").provider == "deepseek"
    # Parents on the 128,000 window, children on the 262,144 one.
    assert settings.compaction.token_threshold == 87_040
    assert settings.compaction_for("main") == settings.compaction
    assert settings.compaction_for("meta").token_threshold == 87_040
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
    # Auto mode has nobody to answer: no hook, so no ask_user tool is registered.
    write_control(control_path, ControlState(mode="auto"))
    assert runner.user_question_hook(session_key) is None


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
        # The real executor records the session through the pipeline, which is
        # also what stamps the session key; the runner never appends for it.
        ledger.append(
            {
                "record_type": "fold",
                "experiment_id": "demo",
                "epoch_id": session.epoch_id,
                "fold_id": "fold_2026Q1",
                "run_id": "run_001",
                "session_key": session.session_key,
            }
        )

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


def test_session_boundary_restart_keeps_the_finished_session_and_stops_the_next(
    tmp_path: Path,
):
    """A deferred restart costs no work: the session in flight is recorded,
    the next one is not started, and the entrypoint is told to re-exec."""

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
                "record_type": "fold",
                "experiment_id": "demo",
                "epoch_id": session.epoch_id,
                "fold_id": session.session_key.rsplit("/", 1)[-1],
                "run_id": f"run_{len(ran):03d}",
                "session_key": session.session_key,
            }
        )

    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(
            DevelopmentSession("epoch_001/fold_2026Q1", "fold", "epoch_001", None),
            DevelopmentSession("epoch_001/fold_2026Q2", "fold", "epoch_001", None),
        ),
        execute_session=execute,
        ledger=ledger,
        control_path=control_path,
        status_path=status_path,
        ref_store=AgentRefStore(tmp_path / "experiment"),
        poll_seconds=0.01,
    )

    result = runner.run()

    assert result == {"status": "restart", "sessions_run": 1, "reran_sessions": []}
    assert ran == ["epoch_001/fold_2026Q1"]
    assert [row["session_key"] for row in ledger.read()] == ["epoch_001/fold_2026Q1"]
    # One-shot, and the console sees one worker coming up rather than a stop.
    assert read_control(control_path).restart_pending is False
    assert read_status(status_path)["state"] == "launching"


def test_session_boundary_restart_is_taken_before_the_next_session_starts(
    tmp_path: Path,
):
    """Requested while the worker waits at an approval gate, the swap happens
    there: the next session must not run hours of the old code first."""

    control_path = tmp_path / "control.json"
    status_path = tmp_path / "status.json"
    write_control(control_path, ControlState(mode="manual", restart_pending=True))

    def execute(session, context):  # pragma: no cover - must not run
        del session, context
        raise AssertionError("the gate started a session instead of restarting")

    runner = InteractiveExperimentRunner(
        experiment_id="demo",
        sessions=(
            DevelopmentSession("epoch_001/fold_2026Q1", "fold", "epoch_001", None),
        ),
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
        ("meta_model", "unknown-model", "unsupported DeepSeek model"),
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


def _agent_then(
    *tool_calls: ToolCall,
    roles: tuple[str, ...] = ("auditor",),
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
        if implement is not None and role == "developer":
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


def test_llm_worker_runs_real_meta_fold_validation_and_heldout(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    assert options.llm is not None
    assert isinstance(options.llm.build_gateway(), OpenAICompatibleProxy)
    # Different executable logic from the inherited seed: a Fold with a
    # parent may only nominate a different hypothesis (or an explicit
    # keep-parent after one existed).
    source = "def generate_orders(context):\n    if context is None:\n        return []\n    return []\n"
    llm = ScriptedLLM(
        [
            *_agent_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer small daily changes"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_agent_then(
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
        "subagent": {"provider": "scripted", "model": "scripted"},
    }
    assert meta_host_manifest["llm"] == {
        "provider": "scripted",
        "model": "scripted",
        "subagent": {"provider": "scripted", "model": "scripted"},
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
    # The cross-fold step tree is published where the console and the worker
    # read it: the host's parent control of the inherited seed, then the
    # candidate this Fold nominated.
    tree = json.loads((experiment / "steps/tree.json").read_text(encoding="utf-8"))
    assert [node["node_id"] for node in tree["nodes"]][-1] == fold["selected_step_id"]
    assert [node["result_name"] for node in tree["nodes"]] == [
        "parent_control",
        "valid_001",
    ]
    assert heldout["result"]["total_return"] == 0.0
    assert heldout["strategy_artifact_id"] == result["final_strategy_artifact"]
    # The nominated candidate's own result, not the host's parent control that
    # opened the session.
    selected = next(
        step for step in fold["steps"] if step["step_id"] == fold["selected_step_id"]
    )
    validation_ref = Path(selected["validation_result_ref"])
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
    # The Meta trace writer must keep sub-agent identity, progress and usage
    # on disk: the console cards, trace stats and Meta process summaries are
    # built from these files, not from in-memory events.
    from autotrade.webui.traces import project_trace_blocks, trace_stats

    meta_trace_path = next(
        path
        for path in traces
        if '"session_kind": "fold"' not in path.read_text(encoding="utf-8")
    )
    meta_events = [
        json.loads(line)
        for line in meta_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_type: dict[str, list[dict[str, object]]] = {}
    for event in meta_events:
        by_type.setdefault(str(event["event_type"]), []).append(event)
    started = by_type["subagent_task"]
    assert started and all(
        event["task_id"].startswith("agent_") and event["role"] == "auditor"
        and event["status"] == "started" and event["mode"] == "meta"
        and "thinking" in event
        for event in started
    )
    assert all(
        event["task_id"] == started[0]["task_id"] or event["task_id"].startswith("agent_")
        for event in by_type["subagent_llm"]
    )
    assert all(
        event["round"] >= 1 and event["usage"]["total_tokens"] == 57
        and "content" not in event
        for event in by_type["subagent_llm"]
    )
    ended = by_type["subagent"]
    assert ended and all(
        event["task_id"] == start["task_id"] for event, start in zip(ended, started)
    )
    assert all(
        event["status"] == "completed" and event["usage_totals"]["total_tokens"] == 57
        and event["llm_calls"] == 1 and "summary" not in event
        for event in ended
    )
    stats = trace_stats(meta_trace_path)
    assert stats["subagent_tasks"] == len(started)
    assert stats["subagent_running"] == 0
    assert stats["subagent_total_tokens"] == 57 * len(started)
    cards = [block for block in project_trace_blocks(meta_events) if block.get("kind") == "subagent"]
    assert len(cards) == len(started)
    assert cards[0]["status"] == "completed" and cards[0]["role"] == "auditor"
    assert cards[0]["usage"]["total_tokens"] == 57
    # Completion prunes nothing: the run evidence a later audit reads stays on disk.
    assert (options.work_root / options.experiment_id).is_dir()
    assert not any((experiment / "artifacts/strategy/revisions").iterdir())
    frozen = list((experiment / "artifacts/strategy/frozen").iterdir())
    assert len(frozen) == 1 and frozen[0].name == fold["frozen_strategy_artifact_id"]
    assert len(llm.calls) == 6
    meta_tool_names = {item["function"]["name"] for item in llm.calls[0]["tools"]}
    assert {"write_file", "finish_meta", "agent"}.issubset(meta_tool_names)
    # The Meta session may regularize the working copy, so it holds the typed
    # writers and modification_check — but it stays offline and never backtests.
    assert {"write_file", "edit_file", "modification_check"}.issubset(
        meta_tool_names
    )
    assert {"shell", "daily_backtest", "step_rollback"}.isdisjoint(meta_tool_names)
    fold_tool_names = {item["function"]["name"] for item in llm.calls[2]["tools"]}
    # Fold parent holds typed writers; shell is debug-only and must not edit strategy.
    assert {
        "daily_backtest",
        "edit_file",
        "agent",
        "finish_fold",
        "shell",
        "step_rollback",
        "write_file",
    }.issubset(fold_tool_names)
    # Auto mode registers no ask_user: nobody would answer it.
    assert "ask_user" not in fold_tool_names
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


_EARLY_STOP_REASON = (
    "两轮 batch_validate 已证伪动量与反转两类假设；仍未检验的只有需要分钟线的"
    "日内假设，本 Fold 拿不到该数据，剩余回测预算留给下一个 Fold 更有价值。"
)


def _candidate_source(weight: str) -> str:
    return (
        "def generate_orders(context):\n"
        f"    weight = {weight}\n"
        "    return []\n"
    )


def _batch_round(names: tuple[str, str]) -> tuple[ToolCall, ...]:
    return (
        *(
            ToolCall(
                f"write_{name}",
                "write_file",
                {
                    "path": f"candidates/{name}/main.py",
                    "content": _candidate_source(f"0.{index}"),
                },
            )
            for index, name in enumerate(names, start=1)
        ),
        ToolCall(
            f"batch_{names[0]}",
            "batch_validate",
            {
                "candidates": [
                    {
                        "name": name,
                        "hypothesis": f"{name} beats the parent on this window",
                        "path": f"candidates/{name}",
                    }
                    for name in names
                ]
            },
        ),
    )


INHERITED_PRIOR = (
    "Closed: small-cap reversal (null percentile near 0.5). "
    "Next round: post-event drift against a matched control."
)
INHERITED_STRATEGY = (
    "SOURCE_ARM = 'src'\n\n\ndef generate_orders(context):\n    return []\n"
)
INHERITED_ARTIFACT_ID = "strategy_epoch_001_fold_2025Q3"


def _seed_parent(experiment: Path) -> str:
    """Start the experiment from a read-only inherited artifact.

    A parentless first Fold can only freeze a baseline anchor, and an anchor
    is a control the experiment refuses to deliver (docs/pipeline-design.md §2.2):
    a run whose developer never improves on it therefore has nothing to send
    to Held-out. These tests are about the orchestration around the delivery,
    not about earning one, so they inherit a real artifact the way the console
    seeds ``inherit_from`` and the lineage starts from a deliverable.
    """

    root = experiment / "inherited" / "output"
    root.mkdir(parents=True)
    (root / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    chmod_tree(root, file_mode=0o444, dir_mode=0o555)
    artifact_id = "strategy_inherited_seed"
    _update_params(
        experiment,
        {
            "_inherited_artifact": {
                "artifact_id": artifact_id,
                "path": str(root),
                "revision_id": "revision_inherited_seed",
                "source_fold_id": "fold_seed",
            }
        },
    )
    return artifact_id


def _update_params(experiment: Path, values: dict[str, object]) -> None:
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(values)
    path.write_text(json.dumps(params), encoding="utf-8")


def _memory_source(repo: Path, source_id: str = "src") -> Path:
    """A source experiment whose ledger published a PRIOR and a skills tree."""
    source = repo / "experiments" / source_id
    ExperimentPriorStore(source).publish(INHERITED_PRIOR, generation_id="gen_1")
    tree = repo / "skills_src" / source_id / "skills"
    (tree / "closed-families").mkdir(parents=True)
    (tree / "closed-families" / "SKILL.md").write_text(
        "# Closed Families\n\nSmall-cap reversal is closed; do not re-test it.\n",
        encoding="utf-8",
    )
    skills = ExperimentSkillsStore(source).publish(tree, generation_id="gen_1")
    ExperimentLedger(source / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "meta_learning",
            "experiment_id": source_id,
            "epoch_id": "epoch_001",
            "fold_id": "meta_001",
            "run_id": "run_m",
            "prior": INHERITED_PRIOR,
            "prior_generation_id": "gen_1",
            "skills_ref": skills.skills_ref,
            "skills_generation_id": skills.generation_id,
            **skills.stats.ledger_fields(),
            "skills_published": True,
        }
    )
    return source


def _artifact_source(repo: Path, source_id: str = "src") -> Path:
    """A source experiment whose latest Fold froze a strategy and its models."""
    source = repo / "experiments" / source_id
    frozen = source / "artifacts/strategy/frozen" / INHERITED_ARTIFACT_ID
    (frozen / "output").mkdir(parents=True)
    (frozen / "output" / "main.py").write_text(INHERITED_STRATEGY, encoding="utf-8")
    (frozen / "models").mkdir()
    (frozen / "models" / "params.json").write_text('{"alpha": 1}\n', encoding="utf-8")
    ExperimentLedger(source / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "fold",
            "experiment_id": source_id,
            "epoch_id": "epoch_001",
            "fold_id": "fold_2025Q3",
            "run_id": "run_f",
            "session_key": "epoch_001/fold_2025Q3",
            "fold_status": "frozen",
            "test_period": "20250701..20250930",
            "frozen_strategy_artifact_id": INHERITED_ARTIFACT_ID,
            "frozen_strategy_artifact_path": str(frozen / "output"),
            "frozen_model_artifact_path": str(frozen / "models"),
        }
    )
    return source


def _inherit_artifact(repo: Path, experiment: Path, source_id: str) -> dict:
    """Seed the child through the console's own copier, the way create does."""
    manager = ExperimentManager(repo, repo / "experiments")
    payload = manager._import_inherited_artifact(experiment, source_id)
    _update_params(
        experiment, {"inherit_from": source_id, "_inherited_artifact": payload}
    )
    return payload


def test_llm_worker_starts_from_inherited_memory(tmp_path: Path, monkeypatch):
    """``inherit_memory_from``: the first Meta reads the inherited PRIOR as the
    previous generation and may keep it, the first Fold's system prompt
    carries that PRIOR and its workspace mounts the inherited skills -- the
    same view either session has after a Meta publication -- and the Fold,
    having no frozen parent, anchors the lineage on its nomination.

    That anchor is also where this arm stops: an anchor is the lineage's
    control, so a development window that ends with one still in force has
    nothing to deliver and the run fails explicitly instead of spending a
    Held-out on the placebo. Every record written before that stays."""
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    source = _memory_source(repo)
    payload = import_inherited_memory(experiment, source, source_id="src")
    _update_params(
        experiment, {"inherit_memory_from": "src", "_inherited_memory": payload}
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    llm = ScriptedLLM(
        [
            # The Meta keeps the inherited PRIOR as it stands.
            *_agent_then(ToolCall("finish_meta", "finish_meta", {})),
            *_agent_then(
                ToolCall("check", "modification_check", {}),
                ToolCall("valid", "daily_backtest", {}),
                ToolCall("finish_fold", "finish_fold", {}),
                roles=_FOLD_DELEGATION_ROLES,
                implement={
                    "path": "output/main.py",
                    "content": "def generate_orders(context):\n    return []\n",
                },
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="baseline anchor in force"):
        run_local_interactive_worker(
            options,
            llm=llm,
            command_runner_factory=lambda _workspace: _NoShellRunner(),
        )
    status = read_status(experiment / "hitl/status.json")
    assert status["state"] == "failed"
    assert "an anchor is the lineage's control" in status["error"]
    records = ExperimentLedger(options.rolling.ledger_path).read()
    # Development ran in full; only the delivery is refused.
    assert [record["record_type"] for record in records] == ["meta_learning", "fold"]
    meta, fold = records
    assert meta["prior"] == INHERITED_PRIOR
    assert meta["prior_generation_id"] == "inherited_src"
    assert meta["prior_published"] is False
    assert fold["skills_ref"] == payload["skills_ref"] and fold["skills_count"] == 1
    assert fold["fold_status"] == "frozen" and fold["baseline_anchor"] is True
    fold_prompts = [
        message.content or ""
        for call in llm.calls
        for message in call["messages"]
        if message.role == "system" and "# 提交合同" in (message.content or "")
    ]
    assert fold_prompts and all(INHERITED_PRIOR in prompt for prompt in fold_prompts)
    manifest = json.loads(Path(fold["run_manifest_ref"]).read_text(encoding="utf-8"))
    assert manifest["skills"]["count"] == 1
    fold_workspace = Path(fold["run_manifest_ref"]).parent / "workspace"
    assert (fold_workspace / "skills" / "closed-families" / "SKILL.md").is_file()

    # Provenance: both sessions are told the PRIOR and the skills came from
    # another experiment's ledger, so the fold and artifact ids that PRIOR
    # cites are not read as this experiment's own missing records.
    memory = load_inherited_memory(experiment)
    refs = AgentRefStore(experiment)
    origin = prior_provenance(memory, INHERITED_PRIOR, ref_store=refs)
    assert origin is not None and origin["source_experiment"] == "src"
    meta_workspace = experiment / "artifacts" / str(meta["run_id"]) / "workspace"
    meta_context = json.loads(
        (meta_workspace / "inputs" / "meta_context.json").read_text(encoding="utf-8")
    )
    assert meta_context["review_window"]["fold_count"] == 0
    assert (
        meta_context["review_window"]["previous_meta_ref"]
        == origin["source_generation_id"]
    )
    assert meta_context["review_window"]["prior_provenance"] == origin
    meta_facts = [
        message.content or ""
        for call in llm.calls
        for message in call["messages"]
        if message.role == "system" and "prior_provenance" in (message.content or "")
    ]
    assert meta_facts and all("skills_provenance" in prompt for prompt in meta_facts)

    # The Fold reads the same body through the mount and the prompt, both with
    # the mount-time header; the read-only inherited generation keeps the exact
    # bytes the source published.
    header = inherited_prior_header(origin, has_parent=False)
    mounted = (fold_workspace / "inputs" / "PRIOR.md").read_text(encoding="utf-8")
    assert mounted == f"{header}\n\n{INHERITED_PRIOR}\n"
    assert all(header in prompt for prompt in fold_prompts)
    assert Path(str(payload["prior_ref"])).read_text(
        encoding="utf-8"
    ) == INHERITED_PRIOR + "\n"
    fold_facts = json.loads(
        (fold_workspace / "inputs" / "fold_context.json").read_text(encoding="utf-8")
    )
    assert fold_facts["prior_provenance"] == origin
    assert fold_facts["skills_provenance"]["source_experiment"] == "src"


def _inherited_artifact_llm() -> ScriptedLLM:
    """One Meta that only writes a PRIOR, then one Fold that edits the parent."""
    return ScriptedLLM(
        [
            *_agent_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer small daily changes"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_agent_then(
                ToolCall("check", "modification_check", {}),
                ToolCall("valid", "daily_backtest", {}),
                ToolCall("finish_fold", "finish_fold", {}),
                roles=_FOLD_DELEGATION_ROLES,
                implement={
                    "path": "output/main.py",
                    # A real logic change: finish_fold refuses a candidate that
                    # differs from the parent only in comments.
                    "content": INHERITED_STRATEGY.replace("'src'", "'child'"),
                },
            ),
        ]
    )


def test_llm_worker_starts_from_the_inherited_artifact(tmp_path: Path, monkeypatch):
    """``inherit_from``: the seed the console copied into ``_inherited/`` is the
    parent of BOTH sessions that precede this experiment's first freeze.

    The Epoch-start Meta opens its working copy on it and the first Fold's
    host parent control replays it on that Fold's Validation window -- neither
    may look for it under this experiment's own ``frozen/``, where a seed
    copied from another experiment never lives. The seed itself stays the
    read-only snapshot it was created as."""
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    _artifact_source(repo)
    payload = _inherit_artifact(repo, experiment, "src")
    seed = Path(str(payload["path"]))
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(
        options,
        llm=_inherited_artifact_llm(),
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )
    assert result["state"] == "completed"
    meta, fold, _heldout = ExperimentLedger(options.rolling.ledger_path).read()

    # The Meta session took the inherited seed as its working copy instead of
    # the blank template, and its host manifest names the inherited artifact.
    meta_run = experiment / "artifacts" / str(meta["run_id"])
    assert (meta_run / "workspace/output/main.py").read_text(
        encoding="utf-8"
    ) == INHERITED_STRATEGY
    meta_manifest = json.loads(
        (meta_run / "host_run_manifest.json").read_text(encoding="utf-8")
    )
    assert meta_manifest["parent_strategy_artifact_id"] == payload["artifact_id"]
    assert meta_manifest["is_initial_artifact"] is False
    assert meta_manifest["template_ref"] is None
    # The Meta only published a PRIOR, so the seed is still the Fold's parent.
    assert meta["status"] == "prior_only_kept_parent"

    # The Fold's host parent control replayed the seed on this Fold's window,
    # so the first Fold already has walk-forward evidence for its parent and
    # does not have to anchor the lineage on its own nomination.
    control = fold["parent_control"]
    assert control["status"] == "ok"
    assert control["parent_strategy_artifact_id"] == payload["artifact_id"]
    assert Path(control["validation_result_ref"]).is_file()
    assert fold["parent_strategy_artifact_id"] == payload["artifact_id"]
    assert "baseline_anchor" not in fold
    assert fold["fold_status"] == "frozen"
    assert fold["frozen_strategy_artifact_id"] != payload["artifact_id"]

    # A snapshot the source experiment can no longer influence: still whole,
    # still read-only, still outside the store the run prunes.
    assert (seed / "main.py").read_text(encoding="utf-8") == INHERITED_STRATEGY
    assert (seed / "main.py").stat().st_mode & 0o222 == 0
    assert seed.parent.name == "_inherited"
    assert not (experiment / "artifacts/strategy/frozen" / seed.name).exists()


def test_a_kept_inherited_seed_survives_a_restart_and_the_resume_continues(
    tmp_path: Path, monkeypatch
):
    """An arm that inherits a seed and never beats it must still be resumable.

    Every Fold here abstains, so the seed stays the parent and each fold record
    names its ``_inherited/`` tree. The console restarts the worker at a session
    boundary; the resume rebuilds the parent from that record. Deriving
    ``frozen/<id>`` from the recorded id instead looked for the seed in the
    store of this experiment's own freezes and killed the restarted worker
    before it could run a single remaining session.
    """
    repo, experiment = _experiment(tmp_path)
    _update_params(experiment, {"test_stage": False})
    _artifact_source(repo)
    payload = _inherit_artifact(repo, experiment, "src")
    seed = Path(str(payload["path"]))
    control_path = experiment / "hitl/control.json"
    ran: list[str] = []

    class Abstaining:
        """Abstains on every Fold; the console asks for a restart during the
        first one, exactly as the arm that failed was restarted."""

        def __init__(self, **_options: object) -> None:
            pass

        def __call__(self, request):
            ran.append(request.fold.fold_id)
            if len(ran) == 1:
                pending = read_control(control_path)
                pending.restart_pending = True
                write_control(control_path, pending)
            return FoldSessionResult(
                conversation_id=f"conv_{request.fold.fold_id}",
                steps=(),
                no_edge_reason="nothing beat the inherited parent",
            )

    monkeypatch.setattr(worker, "DeterministicBaselineDeveloper", Abstaining)
    options = load_worker_options(experiment, repo_root=repo)

    interrupted = run_local_interactive_worker(options)

    assert interrupted["status"] == "restart"
    assert ran == ["fold_2025Q4"]
    kept = ExperimentLedger(options.rolling.ledger_path).read()[0]
    assert kept["fold_status"] == "no_update"
    assert kept["frozen_strategy_artifact_id"] == payload["artifact_id"]
    assert Path(str(kept["frozen_strategy_artifact_path"])) == seed

    resumed = run_local_interactive_worker(options)

    assert resumed["state"] == "completed"
    # The finished Fold is not re-run, the remaining one is, and the Held-out
    # scores the seed the ledger still names as the graduate.
    assert ran == ["fold_2025Q4", "fold_2026Q1"]
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == ["fold", "fold", "heldout"]
    assert [record["fold_status"] for record in records[:2]] == ["no_update"] * 2
    assert records[2]["strategy_artifact_id"] == payload["artifact_id"]
    # The seed is still the console's read-only snapshot, and resolving it
    # never copied it into this experiment's own store.
    assert (seed / "main.py").stat().st_mode & 0o222 == 0
    assert not (experiment / "artifacts/strategy/frozen" / seed.name).exists()


def test_llm_worker_inherits_an_artifact_and_a_memory_together(
    tmp_path: Path, monkeypatch
):
    """``inherit_from`` and ``inherit_memory_from`` are independent seeds and a
    console create may set both: the run then starts from the source's frozen
    artifact AND from its PRIOR and skills."""
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    source = _memory_source(repo)
    _artifact_source(repo)
    memory = import_inherited_memory(experiment, source, source_id="src")
    _update_params(
        experiment, {"inherit_memory_from": "src", "_inherited_memory": memory}
    )
    payload = _inherit_artifact(repo, experiment, "src")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(
        options,
        llm=_inherited_artifact_llm(),
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )
    assert result["state"] == "completed"
    meta, fold, _heldout = ExperimentLedger(options.rolling.ledger_path).read()
    # Memory: the inherited PRIOR was the Meta's previous generation, and the
    # Fold mounted the inherited skills.
    assert meta["prior"] == "prefer small daily changes"
    assert meta["prior_published"] is True
    assert fold["skills_ref"] == memory["skills_ref"]
    # Provenance follows the body: this experiment's first Meta replaced the
    # inherited PRIOR, so the Fold mounts a PRIOR of its own -- no inherited
    # header, no ``prior_provenance``.
    fold_workspace = Path(fold["run_manifest_ref"]).parent / "workspace"
    assert (fold_workspace / "inputs" / "PRIOR.md").read_text(
        encoding="utf-8"
    ) == "prefer small daily changes\n"
    fold_facts = json.loads(
        (fold_workspace / "inputs" / "fold_context.json").read_text(encoding="utf-8")
    )
    assert "prior_provenance" not in fold_facts
    # Artifact: the same run's parent chain starts at the inherited seed.
    meta_run = experiment / "artifacts" / str(meta["run_id"])
    assert (meta_run / "workspace/output/main.py").read_text(
        encoding="utf-8"
    ) == INHERITED_STRATEGY
    assert fold["parent_control"]["parent_strategy_artifact_id"] == payload[
        "artifact_id"
    ]


def test_a_voluntary_early_finish_must_justify_itself_and_reaches_the_ledger(
    tmp_path: Path,
    monkeypatch,
):
    """``finish_fold``'s early-stop gate, driven by the session's real counters.

    Two ``batch_validate`` rounds clear the round floor while the Fold still
    holds more than a third of its backtest budget: the bare finish is refused
    with the unused budget named, the justified one is accepted, and the
    Agent's own reason is what the fold ledger record and the Meta review carry.
    """

    repo, experiment = _experiment(tmp_path, developer_mode="llm", max_backtests=9)
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    llm = ScriptedLLM(
        [
            *_agent_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer small daily changes"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_agent_then(
                *_batch_round(("momentum_a", "momentum_b")),
                roles=_FOLD_DELEGATION_ROLES,
                implement={
                    "path": "output/main.py",
                    "content": _candidate_source("0.5"),
                },
            ),
            ProviderResponse(tool_calls=_batch_round(("reversal_a", "reversal_b"))),
            ProviderResponse(
                tool_calls=(
                    ToolCall("check", "modification_check", {}),
                    ToolCall("valid", "daily_backtest", {}),
                )
            ),
            # 5 of 9 backtests spent and another round still fits: refused.
            ProviderResponse(tool_calls=(ToolCall("early", "finish_fold", {}),)),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "finish",
                        "finish_fold",
                        {"early_stop_reason": _EARLY_STOP_REASON},
                    ),
                )
            ),
        ]
    )
    run_local_interactive_worker(
        options,
        llm=llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    records = ExperimentLedger(options.rolling.ledger_path).read()
    fold = next(record for record in records if record["record_type"] == "fold")
    assert fold["fold_status"] == "frozen"
    assert fold["early_stop_reason"] == _EARLY_STOP_REASON
    # The Meta review reads the Agent's own account of the early finish.
    summary = compact_fold_history(fold, ref_store=AgentRefStore(experiment))
    assert summary["early_stop_reason"] == _EARLY_STOP_REASON

    fold_trace = next(
        path
        for path in sorted((experiment / "artifacts/traces").glob("*.jsonl"))
        if '"session_kind": "fold"' in path.read_text(encoding="utf-8")
    )
    events = [
        json.loads(line)
        for line in fold_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refused = [
        event
        for event in events
        if event.get("event_type") == "tool_call"
        and event.get("tool") == "finish_fold"
        and isinstance(event.get("result"), dict)
        and event["result"].get("ok") is False
    ]
    assert len(refused) == 1
    # The refusal states what this session would leave unused, from the real
    # counters the backtest tool tracks.
    assert "4/9 backtests" in str(refused[0]["result"].get("error"))


def test_second_llm_fold_prompt_excludes_prior_test_diagnostic(
    tmp_path: Path,
    monkeypatch,
):
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    params_path = experiment / "hitl/params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.update(
        {
            "fold_period": "quarter",
            # Two rolling Folds: 2025Q4 -> 2026Q1 and 2026Q1 -> 2026Q2.
            "development_first_period": "2025Q4",
            "development_last_period": "2026Q2",
            "test_stage": True,
            "heldout_first_period": "2026Q3",
            "heldout_last_period": "2026Q3",
            # One Epoch-start Meta only: this asserts what the SECOND Fold
            # prompt carries, not the periodic meta cadence.
            "meta_learning_fold_interval": 0,
        }
    )
    params_path.write_text(json.dumps(params), encoding="utf-8")
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    def _fold_script(source: str) -> tuple[ProviderResponse, ...]:
        return _agent_then(
            ToolCall("check", "modification_check", {}),
            ToolCall("valid", "daily_backtest", {}),
            ToolCall("finish_fold", "finish_fold", {}),
            roles=_FOLD_DELEGATION_ROLES,
            implement={"path": "output/main.py", "content": source},
        )

    llm = ScriptedLLM(
        [
            *_agent_then(
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
    # The second Fold reads the first Fold's verdict, never its per-candidate
    # trial log: the system prompt is never compacted, so its size must not
    # grow with how many candidates an earlier Fold ran.
    assert '"selection_statistics"' in second_fold_context
    assert '"backtest_summaries"' not in second_fold_context
    assert "test_diagnostic" not in second_fold_context
    assert "test_result" not in second_fold_context


def test_the_meta_after_an_anchor_fold_opens_the_anchor_as_its_parent(
    tmp_path: Path,
    monkeypatch,
):
    """The Meta that follows the lineage's FIRST freeze gets that freeze as its
    parent, and opens its working copy on the artifact's own tree.

    A baseline anchor is a control rather than a deliverable, but it is the
    lineage head all the same: the session that follows it must be handed the
    same kind of artifact record an inherited seed or a later freeze produces,
    or it cannot resolve the parent at all."""
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    _update_params(
        experiment,
        {
            # Two rolling Folds with the default meta cadence between them:
            # 2025Q4 -> 2026Q1 and 2026Q1 -> 2026Q2.
            "development_first_period": "2025Q4",
            "development_last_period": "2026Q2",
            "test_stage": True,
            "heldout_first_period": "2026Q3",
            "heldout_last_period": "2026Q3",
        },
    )
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")

    def _meta_script(prior: str) -> tuple[ProviderResponse, ...]:
        return _agent_then(
            ToolCall("prior", "write_file", {"path": "PRIOR.md", "content": prior}),
            ToolCall("finish_meta", "finish_meta", {}),
        )

    def _fold_script(source: str) -> tuple[ProviderResponse, ...]:
        return _agent_then(
            ToolCall("check", "modification_check", {}),
            ToolCall("valid", "daily_backtest", {}),
            ToolCall("finish_fold", "finish_fold", {}),
            roles=_FOLD_DELEGATION_ROLES,
            implement={"path": "output/main.py", "content": source},
        )

    anchor_source = "def generate_orders(context):\n    return []\n"
    llm = ScriptedLLM(
        [
            *_meta_script("prefer simple signals"),
            # No parent exists, so a passing nomination anchors the lineage.
            *_fold_script(anchor_source),
            *_meta_script("keep the anchor under review"),
            *_fold_script(
                "def generate_orders(context):\n    _ = context.inference_at\n    return []\n"
            ),
        ]
    )
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(
        options,
        llm=llm,
        command_runner_factory=lambda _workspace: _NoShellRunner(),
    )

    assert result["state"] == "completed"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == [
        "meta_learning",
        "fold",
        "meta_learning",
        "fold",
        "heldout",
    ]
    _, anchor_fold, meta, second_fold, _heldout = records
    assert anchor_fold["fold_status"] == "frozen"
    assert anchor_fold["baseline_anchor"] is True

    # The Meta session resolved the anchor from the artifact record it was
    # handed: its manifest names it and its working copy carries its bytes.
    manifest = json.loads(
        (
            experiment / "artifacts" / str(meta["run_id"]) / "host_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["parent_strategy_artifact_id"] == (
        anchor_fold["frozen_strategy_artifact_id"]
    )
    assert (
        experiment / "artifacts" / str(meta["run_id"]) / "workspace/output/main.py"
    ).read_text(encoding="utf-8") == anchor_source

    # An anchor is a valid parent for what follows; it is only kept out of
    # graduation and delivery. The next Fold controls against it and replaces
    # it, so the run reaches Held-out with a candidate rather than the anchor.
    assert second_fold["parent_strategy_artifact_id"] == (
        anchor_fold["frozen_strategy_artifact_id"]
    )
    assert second_fold["fold_status"] == "frozen"
    assert "baseline_anchor" not in second_fold


def test_early_finish_grades_the_epoch_that_actually_ran(tmp_path: Path):
    """Skip-to-Held-out ends development inside Epoch 1 of a three-Epoch
    schedule. Graduation term (b) must be scored on the Epoch that produced the
    Folds: scoring the configured last Epoch finds no fold record, reports zero
    transitions, and silently waives the walk-forward requirement."""
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["epochs"] = 3
    path.write_text(json.dumps(params), encoding="utf-8")
    (experiment / "hitl/control.json").write_text(
        json.dumps({"schema_version": 1, "mode": "auto", "skip_to_heldout": True}),
        encoding="utf-8",
    )
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(options)
    assert result["state"] == "completed"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    folds = [row for row in records if row["record_type"] == "fold"]
    assert [row["epoch_id"] for row in folds] == ["epoch_001"]
    heldout = [row for row in records if row["record_type"] == "heldout"]
    assert [row["epoch_id"] for row in heldout] == ["epoch_001"]
    # The one frozen Test that ran is a transition, and holding cash does not
    # beat the benchmark: the term fails instead of reporting itself absent.
    walk_forward = heldout[0]["verdict"]["walk_forward"]
    assert walk_forward["status"] == "inconsistent"
    assert walk_forward["transitions"] == 1
    assert "walkforward_excess_inconsistent(0/1<1)" in heldout[0]["verdict"]["reasons"]
    # No fold record at all leaves only the configured schedule to name.
    assert _heldout_epoch_id(ExperimentLedger(tmp_path / "empty.jsonl"), 3) == "epoch_003"


def _graduate_every_heldout(monkeypatch) -> None:
    """The deterministic baseline holds cash and never graduates; the
    deployment path needs a graduated verdict, which is the ledger's word."""
    from autotrade.pipelines.config import AcceptanceRules

    monkeypatch.setattr(
        AcceptanceRules,
        "heldout_verdict",
        lambda self, summary, *args, window=None, **kwargs: {
            "status": "graduated",
            "reasons": [],
            "window": window,
        },
    )


def _deployment_experiment(tmp_path: Path, **params: object) -> tuple[Path, Path]:
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl/params.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(
        {
            "development_first_period": "2025Q4",
            "development_last_period": "2026Q1",
            "test_stage": False,
            "deployment_adjustment_start": "20260401",
            **params,
        }
    )
    path.write_text(json.dumps(current), encoding="utf-8")
    return repo, experiment


def test_local_worker_runs_the_deployment_adjustment_after_graduation(
    tmp_path: Path, monkeypatch
):
    """A graduated experiment runs the post-seal deployment adjustment on the
    window from the configured start to the release end, records one row,
    names the Paper candidate in the terminal status, and a resume republishes
    without re-running anything."""
    _graduate_every_heldout(monkeypatch)
    repo, experiment = _deployment_experiment(tmp_path)
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(options)
    assert result["state"] == "completed"
    assert result["verdict"]["status"] == "graduated"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == [
        "fold",
        "fold",
        "heldout",
        "deployment_adjustment",
    ]
    graduated = records[1]["frozen_strategy_artifact_id"]
    adjustment = records[-1]
    assert adjustment["session_key"] == "deployment_adjustment"
    assert adjustment["fold_id"] == "deployment_20260401..20260930"
    assert adjustment["validation_period"] == "20260401..20260930"
    assert adjustment["valid_decision_time"].startswith("2026-03-31T23:59:59")
    assert adjustment["parent_strategy_artifact_id"] == graduated
    # The host replayed the graduate on the window; the deterministic
    # developer nominated the graduate's own content, so nothing was adjusted.
    assert adjustment["parent_control"]["status"] == "ok"
    assert adjustment["status"] == "no_update"
    assert adjustment["nominated_identical_to_parent"] is True
    assert adjustment["adjusted_strategy_artifact_id"] is None
    assert result["paper_candidate"]["source"] == "graduated"
    assert result["paper_candidate"]["artifact_id"] == graduated
    assert result["final_strategy_artifact"] == graduated
    status = read_status(experiment / "hitl/status.json")
    assert status["paper_candidate"] == result["paper_candidate"]
    plan = json.loads((experiment / "hitl/schedule.json").read_text(encoding="utf-8"))
    assert [row["kind"] for row in plan["sessions"]] == [
        "fold",
        "fold",
        "heldout",
        "deployment_adjustment",
    ]
    assert plan["sessions"][-1]["period"] == {"start": "20260401", "end": "20260930"}
    assert result["total_sessions"] == 4
    # Nothing outstanding: a resume republishes the same terminal status.
    resumed = run_local_interactive_worker(options)
    assert resumed["state"] == "completed"
    assert resumed["heldout_runs"] == 0
    assert resumed["paper_candidate"] == result["paper_candidate"]
    assert ExperimentLedger(options.rolling.ledger_path).read() == records


def test_a_discarded_experiment_runs_no_deployment_adjustment(tmp_path: Path):
    repo, experiment = _deployment_experiment(tmp_path)
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    result = run_local_interactive_worker(options)
    assert result["state"] == "completed"
    assert result["verdict"]["status"] == "discarded"
    assert result["paper_candidate"] is None
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == ["fold", "fold", "heldout"]
    resumed = run_local_interactive_worker(options)
    assert resumed["state"] == "completed"
    assert ExperimentLedger(options.rolling.ledger_path).read() == records


def test_a_crashed_deployment_adjustment_is_recorded_and_resumed(
    tmp_path: Path, monkeypatch
):
    """A crash leaves attempt_failed and the session stays due; the next
    worker start goes straight to it without re-running development or the
    Held-out."""
    from autotrade.pipelines.local_backend import DeterministicBaselineDeveloper

    _graduate_every_heldout(monkeypatch)
    repo, experiment = _deployment_experiment(tmp_path, session_max_attempts=1)
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    original = DeterministicBaselineDeveloper.__call__

    def crash_on_deployment(self, request):
        if request.session_kind == "deployment_adjustment":
            raise RuntimeError("sandbox died")
        return original(self, request)

    monkeypatch.setattr(DeterministicBaselineDeveloper, "__call__", crash_on_deployment)
    with pytest.raises(RuntimeError, match="sandbox died"):
        run_local_interactive_worker(options)
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == [
        "fold",
        "fold",
        "heldout",
        "attempt_failed",
    ]
    assert records[-1]["session_key"] == "deployment_adjustment"
    assert read_status(experiment / "hitl/status.json")["state"] == "failed"

    monkeypatch.setattr(DeterministicBaselineDeveloper, "__call__", original)
    resumed = run_local_interactive_worker(options)
    assert resumed["state"] == "completed"
    after = ExperimentLedger(options.rolling.ledger_path).read()
    assert after[:4] == records
    assert [record["record_type"] for record in after] == [
        "fold",
        "fold",
        "heldout",
        "attempt_failed",
        "deployment_adjustment",
    ]
    assert resumed["paper_candidate"]["source"] == "graduated"


def test_a_deployment_adjustment_requested_after_completion_runs_on_resume(
    tmp_path: Path, monkeypatch
):
    """An experiment that graduated before the knob existed: setting the
    start on its params and resuming runs just the adjustment."""
    _graduate_every_heldout(monkeypatch)
    repo, experiment = _deployment_experiment(tmp_path, deployment_adjustment_start="")
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    completed = run_local_interactive_worker(options)
    assert completed["state"] == "completed"
    assert completed["paper_candidate"]["source"] == "graduated"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    assert [record["record_type"] for record in records] == ["fold", "fold", "heldout"]

    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["deployment_adjustment_start"] = "20260401"
    path.write_text(json.dumps(params), encoding="utf-8")
    options = load_worker_options(experiment, repo_root=repo)
    resumed = run_local_interactive_worker(options)
    assert resumed["state"] == "completed"
    after = ExperimentLedger(options.rolling.ledger_path).read()
    assert after[:3] == records
    assert after[-1]["record_type"] == "deployment_adjustment"
    plan = json.loads((experiment / "hitl/schedule.json").read_text(encoding="utf-8"))
    assert plan["sessions"][-1]["kind"] == "deployment_adjustment"


def test_local_worker_resume_skips_durable_sessions_and_heldout(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    run_local_interactive_worker(options)
    before = ExperimentLedger(options.rolling.ledger_path).read()
    resumed = run_local_interactive_worker(options)
    after = ExperimentLedger(options.rolling.ledger_path).read()
    assert resumed["state"] == "completed"
    assert resumed["heldout_runs"] == 0
    assert after == before


def test_development_that_freezes_nothing_fails_and_stays_terminal(
    tmp_path: Path, monkeypatch
):
    """The documented zero-freeze end of development, and its resume.

    Every Fold abstains, so nothing is ever frozen: the run must fail (there is
    no artifact to evaluate), the message must name the reason already in the
    ledger, and a restart must republish that same terminal failure instead of
    walking the finished plan only to raise again.
    """
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["test_stage"] = False
    path.write_text(json.dumps(params), encoding="utf-8")
    ran: list[str] = []
    assembled: list[str] = []

    class Abstaining:
        """A developer that finishes every Fold with an explicit no-edge."""

        def __init__(self, **_options: object) -> None:
            assembled.append("developer")

        def __call__(self, request):
            ran.append(request.fold.fold_id)
            return FoldSessionResult(
                conversation_id=f"conv_{request.fold.fold_id}",
                steps=(),
                no_edge_reason="nothing beat holding cash",
            )

    monkeypatch.setattr(worker, "DeterministicBaselineDeveloper", Abstaining)
    options = load_worker_options(experiment, repo_root=repo)
    expected = (
        "Development completed without a frozen baseline artifact: "
        "2/2 folds ended agent_no_edge (baseline_missing)"
    )
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        run_local_interactive_worker(options)
    assert ran == ["fold_2025Q4", "fold_2026Q1"]
    ledger = ExperimentLedger(options.rolling.ledger_path)
    before = ledger.read()
    assert [record["fold_status"] for record in before] == ["baseline_missing"] * 2
    status = read_status(experiment / "hitl/status.json")
    assert status["state"] == "failed"
    assert status["error"] == f"RuntimeError: {expected}"

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        run_local_interactive_worker(options)
    # Republished, not re-run: the second invocation never even assembled the
    # pipeline, so no session executed and no record was appended.
    assert assembled == ["developer"]
    assert ran == ["fold_2025Q4", "fold_2026Q1"]
    assert ledger.read() == before
    republished = read_status(experiment / "hitl/status.json")
    assert (republished["state"], republished["error"]) == ("failed", status["error"])


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


def test_worker_params_reject_unknown_and_partial_periods(tmp_path: Path):
    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl" / "params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params["typo_budget"] = 3
    path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown experiment parameters"):
        load_worker_options(experiment, repo_root=repo)
    params.pop("typo_budget")
    for key in ("development_last_period", "heldout_first_period", "heldout_last_period"):
        params.pop(key)
    path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="all four"):
        load_worker_options(experiment, repo_root=repo)


def test_worker_parses_the_deployment_adjustment_knobs(tmp_path: Path):
    from autotrade.pipelines.worker import _deployment_pit_views_seed

    repo, experiment = _experiment(tmp_path)
    path = experiment / "hitl/params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    options = load_worker_options(experiment, repo_root=repo)
    assert options.rolling.deployment_adjustment_start == ""
    assert options.rolling.deployment_max_backtests == 6
    assert options.deployment_pit_views_seed is None
    params.update({"deployment_adjustment_start": "2026-04-01", "deployment_max_backtests": 3})
    path.write_text(json.dumps(params), encoding="utf-8")
    options = load_worker_options(experiment, repo_root=repo)
    assert options.rolling.deployment_adjustment_start == "20260401"
    assert options.rolling.deployment_max_backtests == 3
    params["deployment_max_backtests"] = 0
    path.write_text(json.dumps(params), encoding="utf-8")
    with pytest.raises(ValueError, match="deployment_max_backtests"):
        load_worker_options(experiment, repo_root=repo)
    # A named deployment seed is a decision: it must exist and carry this
    # experiment's snapshot configuration under the current cache format.
    with pytest.raises(ValueError, match="existing directory"):
        _deployment_pit_views_seed("data/pit_views_seed/absent", repo, options.snapshot_config)
    (repo / "data/pit_views_seed/bare").mkdir(parents=True)
    with pytest.raises(ValueError, match="provider.json"):
        _deployment_pit_views_seed("data/pit_views_seed/bare", repo, options.snapshot_config)


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
            "fold_period": "quarter",
            "development_first_period": "2026Q1",
            "development_last_period": "2026Q1",
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
    _seed_parent(experiment)
    options = load_worker_options(experiment, repo_root=repo)
    # Different executable logic from the inherited seed: a Fold with a
    # parent may only nominate a different hypothesis (or an explicit
    # keep-parent after one existed).
    source = "def generate_orders(context):\n    if context is None:\n        return []\n    return []\n"
    llm = ScriptedLLM(
        [
            *_agent_then(
                ToolCall(
                    "prior",
                    "write_file",
                    {"path": "PRIOR.md", "content": "prefer small daily changes"},
                ),
                ToolCall("finish_meta", "finish_meta", {}),
            ),
            *_agent_then(
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
    assert not hasattr(fold_configs[0], "required_subagent_roles")
    meta_configs = [config for config in assembled_configs if config.mode == "meta"]
    assert not hasattr(meta_configs[0], "required_subagent_roles")
    # One-shot, like every other per-session control.
    assert read_control(experiment / "hitl/control.json").gpu_counts == {}


def test_a_cpu_only_allocation_survives_the_control_round_trip(tmp_path: Path) -> None:
    """0 GPUs is a request, not an unset value.

    The console accepts 0..4 and every layer below honours 0 explicitly
    (``_optional_gpu_count``, the developer's ``gpu=None, gpu_count=0``), so a
    reader that drops it silently runs the CPU-only fold on the experiment
    default. Step indexes are the opposite case: they start at 1, so a 0 there
    addresses no step and stays dropped.
    """

    control = tmp_path / "control.json"
    write_control(
        control,
        ControlState(
            mode="auto",
            gpu_counts={"epoch_001/fold_a": 0, "epoch_001/fold_b": 2},
            step_go={"epoch_001/fold_a": 0, "epoch_001/fold_b": 2},
        ),
    )
    state = read_control(control)
    assert state.gpu_counts == {"epoch_001/fold_a": 0, "epoch_001/fold_b": 2}
    assert state.step_go == {"epoch_001/fold_b": 2}


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
    with pytest.raises(ValueError, match="requires an API key: set VLLM_API_KEY"):
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


def test_meta_trace_payload_keeps_prompts_and_tool_status_without_bodies() -> None:
    from autotrade.pipelines.local_backend import _safe_meta_trace_payload

    start = _safe_meta_trace_payload(
        "session_start",
        {"mode": "meta", "system_prompt": "SYS", "instruction": "GO", "extra": 1},
    )
    assert start == {"mode": "meta", "system_prompt": "SYS", "instruction": "GO"}
    failed = _safe_meta_trace_payload(
        "tool_call",
        {
            "call_index": 3,
            "tool_call_id": "t1",
            "tool": "read_file",
            "arguments": {"path": "inputs/x"},
            "result": {
                "ok": False,
                "error": "search path does not exist",
                "value": {"error_type": "not_found", "content": "body"},
            },
        },
    )
    assert failed == {
        "call_index": 3,
        "tool_call_id": "t1",
        "tool": "read_file",
        "argument_keys": ["path"],
        "result": {
            "ok": False,
            "error": "search path does not exist",
            "error_type": "not_found",
        },
    }
    succeeded = _safe_meta_trace_payload(
        "tool_call",
        {"tool": "read_file", "result": {"ok": True, "value": {"content": "test evidence"}}},
    )
    assert succeeded["result"] == {"ok": True}
    call = _safe_meta_trace_payload(
        "llm_call",
        {"call_index": 1, "status": "ok", "content": "model text", "usage": {"total_tokens": 5}},
    )
    assert "content" not in call and call["usage"] == {"total_tokens": 5}


def test_meta_trace_payload_keeps_shape_only_usefulness_signals() -> None:
    from autotrade.pipelines.local_backend import _safe_meta_trace_payload

    ended = _safe_meta_trace_payload(
        "subagent",
        {"task_id": "agent_1", "role": "auditor", "status": "completed", "summary": "十个字的总结文本啊", "truncated": True},
    )
    assert ended["summary_chars"] == 9 and "summary" not in ended and ended["truncated"] is True
    call = _safe_meta_trace_payload("llm_call", {"call_index": 2, "content": "abc", "status": "ok"})
    assert call["content_chars"] == 3 and "content" not in call
    child_call = _safe_meta_trace_payload(
        "subagent_llm", {"task_id": "agent_1", "role": "auditor", "round": 1, "content": "xy"}
    )
    assert child_call["content_chars"] == 2 and child_call["role"] == "auditor"
    tool = _safe_meta_trace_payload(
        "tool_call",
        {"tool": "agent", "arguments": {"agent": "auditor", "description": "d", "thinking": "medium"}, "result": {"ok": True, "value": {"task_id": "agent_2"}}},
    )
    assert tool["argument_keys"] == ["agent", "description", "thinking"] and "arguments" not in tool
    child_tool = _safe_meta_trace_payload(
        "subagent_tool", {"task_id": "agent_1", "role": "Explore", "round": 2, "tool": "read_file", "result": {"ok": True, "value": {"content": "secret"}}}
    )
    assert child_tool["role"] == "Explore" and child_tool["result"] == {"ok": True}
    reminder = _safe_meta_trace_payload(
        "delegation_reminder",
        {"own_work_calls": 8, "running_children": [{"task_id": "agent_1", "role": "auditor", "description": "d"}], "queued_children": []},
    )
    assert reminder["running_children"][0]["description"] == "d" and reminder["queued_children"] == []
    wrap = _safe_meta_trace_payload(
        "subagent_wrap_up", {"task_id": "agent_1", "role": "auditor", "round": 22, "rounds_limit": 24, "parent_call_id": "c1"}
    )
    assert wrap["rounds_limit"] == 24 and wrap["role"] == "auditor"


def test_meta_trace_payload_keeps_child_argument_keys_and_parent_truncation() -> None:
    from autotrade.pipelines.local_backend import _safe_meta_trace_payload

    child_tool = _safe_meta_trace_payload(
        "subagent_tool",
        {"task_id": "agent_1", "role": "Explore", "round": 2, "tool": "read_file",
         "arguments": {"root": "workspace", "path": "inputs/x"}, "result": {"ok": True, "value": {"content": "s"}}},
    )
    assert child_tool["argument_keys"] == ["path", "root"] and "arguments" not in child_tool
    cut = _safe_meta_trace_payload(
        "output_truncated", {"call_index": 7, "completion_tokens": 12000, "max_tokens": 12000, "content": "x"}
    )
    assert cut == {"call_index": 7, "completion_tokens": 12000, "max_tokens": 12000}


def test_meta_trace_payload_keeps_sub_agent_brief_thinking_and_failure_events() -> None:
    from autotrade.pipelines.local_backend import _safe_meta_trace_payload

    started = _safe_meta_trace_payload(
        "subagent_task",
        {"task_id": "agent_1", "role": "auditor", "parent_call_id": "call_p", "status": "started",
         "mode": "meta", "model": "local-model", "thinking": "medium", "thinking_applied": True,
         "rounds_limit": 12, "inherit_context": False, "description": "trace audit",
         "task": "读 inputs/agent_traces/x.jsonl，回答委托质量问题。"},
    )
    assert started["thinking_applied"] is True and started["description"] == "trace audit"
    assert started["rounds_limit"] == 12
    assert started["task"] == "读 inputs/agent_traces/x.jsonl，回答委托质量问题。"
    child_cut = _safe_meta_trace_payload(
        "subagent_output_truncated",
        {"task_id": "agent_1", "role": "auditor", "round": 3, "completion_tokens": 12000,
         "max_tokens": 12000, "continuation": 1, "parent_call_id": "call_p", "content": "reasoning"},
    )
    assert child_cut == {"task_id": "agent_1", "role": "auditor", "round": 3,
                         "completion_tokens": 12000, "max_tokens": 12000, "continuation": 1,
                         "parent_call_id": "call_p"}
    child_error = _safe_meta_trace_payload(
        "subagent_llm_error",
        {"task_id": "agent_1", "role": "auditor", "round": 2, "provider": "vllm",
         "model": "local-model", "llm_error": "HTTP 503 from model service",
         "parent_call_id": "call_p"},
    )
    assert child_error == {"task_id": "agent_1", "role": "auditor", "round": 2, "provider": "vllm",
                           "model": "local-model", "llm_error": "HTTP 503 from model service",
                           "parent_call_id": "call_p"}
    ended = _safe_meta_trace_payload(
        "subagent",
        {"task_id": "agent_1", "status": "error", "rounds": 4, "tool_calls": 2, "llm_calls": 5,
         "provider": "vllm", "model": "local-model", "usage_totals": {"total_tokens": 900},
         "summary": "报告正文", "mode": "meta", "role": "auditor", "thinking": "medium",
         "thinking_applied": True, "rounds_limit": 12, "inherit_context": False, "truncated": True,
         "truncated_rounds": 3, "llm_errors": 1, "error": "output budget exhausted on reasoning"},
    )
    assert ended["thinking_applied"] is True and ended["rounds_limit"] == 12
    assert ended["truncated_rounds"] == 3 and ended["llm_errors"] == 1
    assert "summary" not in ended and ended["summary_chars"] == 4
    # Report delivery is counters and a spill reference, never the child's text.
    delivered = _safe_meta_trace_payload(
        "subagent_attempt",
        {"attempt": 2, "role": "auditor", "ok": True, "status": "completed",
         "task_id": "agent_1", "summary_chars": 9000, "summary_delivered_chars": 6000,
         "summary_truncated": True, "result_ref": "logs/tool_results/subagent_report_ab12/report.txt",
         "summary": "报告正文", "report": {"summary": "报告正文"}},
    )
    assert delivered == {"attempt": 2, "role": "auditor", "ok": True, "status": "completed",
                         "task_id": "agent_1", "summary_chars": 9000,
                         "summary_delivered_chars": 6000, "summary_truncated": True,
                         "result_ref": "logs/tool_results/subagent_report_ab12/report.txt"}
    # A parent instruction is counted and attributed, never quoted.
    steer = _safe_meta_trace_payload(
        "subagent_steer",
        {"task_id": "agent_1", "role": "auditor", "round": 2, "chars": 40,
         "delivery": "delivered", "parent_call_id": "call_p", "text": "改范围"},
    )
    assert steer == {"task_id": "agent_1", "role": "auditor", "round": 2, "chars": 40,
                     "delivery": "delivered", "parent_call_id": "call_p"}


# ---------------------------------------------------------------------------
# vs_parent: every candidate a Fold session validates is compared to that
# Fold's own parent control, on the tool observation, in the run manifest, and
# therefore in the development history Meta reads back.
# ---------------------------------------------------------------------------


def _benchmarked(evaluator, *, benchmark_return: float):
    """Wrap a replay so its summaries carry the benchmark block a real one has.

    The neutralized excess is half the raw excess, so a reader that confuses
    the two gets a different number rather than the same one.
    """

    class _Wrapped:
        def evaluate(self, request, max_days=None):
            result = evaluator.evaluate(request, max_days)
            excess = result.summary["total_return"] - benchmark_return
            result.summary["benchmark"] = {
                "label": "CSI 300",
                "benchmark_return": benchmark_return,
                "excess_return": excess,
                "neutralized_excess_return": excess / 2,
            }
            return result

    return _Wrapped()


def test_batch_rows_compare_every_candidate_to_this_folds_own_control(tmp_path: Path):
    """A batch row's ``vs_parent`` is the candidate minus the control the Fold
    itself ran; without a control there is no baseline and no block at all."""
    from dataclasses import replace

    from autotrade.pipelines.config import EvaluationResult
    from tests.unit.test_batch_validate import _Session, _strategy

    session = _Session(tmp_path / "no_control")
    # No inherited parent on this Fold: nothing to compare against.
    assert session.backtest.parent_control_summary is None
    session.candidate("a", _strategy("1"))
    session.candidate("b", _strategy("22222"))
    rows = session.call("a", "b").value["candidates"]
    assert all(row["vs_parent"] is None and row["vs_parent_note"] for row in rows)

    session = _Session(tmp_path / "with_control")
    session.backtest.evaluator = _benchmarked(session.evaluator, benchmark_return=0.01)
    sources = {"a": _strategy("1"), "b": _strategy("22222")}
    # The harness prices a replay by its source length, so the two candidates
    # straddle a control placed at their midpoint.
    excesses = {name: 0.01 * len(source) - 0.01 for name, source in sources.items()}
    midpoint = (excesses["a"] + excesses["b"]) / 2
    control = EvaluationResult(
        {
            "total_return": midpoint + 0.01,
            "sharpe": 1.0,
            "max_drawdown": 0.08,
            "benchmark": {
                "label": "CSI 300",
                "benchmark_return": 0.01,
                "excess_return": midpoint,
                "neutralized_excess_return": midpoint / 2,
            },
        },
        "result/parent_control",
    )
    session.backtest.request = replace(
        session.backtest.request, parent_control=control
    )
    assert session.backtest.parent_control_summary is control.summary
    for name, source in sources.items():
        session.candidate(name, source)
    rows = {row["name"]: row for row in session.call("a", "b").value["candidates"]}

    loser = rows["a"]["vs_parent"]
    assert loser["excess_return_delta"] == pytest.approx(excesses["a"] - midpoint)
    assert loser["neutralized_excess_return_delta"] == pytest.approx(
        (excesses["a"] - midpoint) / 2
    )
    # The harness fixes every candidate's drawdown at 0.05 against a control at
    # 0.08: the winner also drew down less.
    assert loser["max_drawdown_delta"] == pytest.approx(-0.03)
    assert loser["beats_parent"] is False

    winner = rows["b"]["vs_parent"]
    assert winner["excess_return_delta"] == pytest.approx(excesses["b"] - midpoint)
    assert winner["beats_parent"] is True


def test_meta_history_carries_each_candidates_parent_comparison(tmp_path: Path):
    """The per-candidate comparison survives into the development history: a
    key the projection does not name is dropped, so it has to be listed."""

    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "backtest_summaries": [
                    {
                        "result_name": "valid_001",
                        "mode": "valid",
                        "status": "ok",
                        "complete_validation": True,
                        "total_return": 0.12,
                        "vs_parent": {
                            "excess_return_delta": 0.06,
                            "neutralized_excess_return_delta": 0.04,
                            "max_drawdown_delta": -0.02,
                            "beats_parent": True,
                        },
                        "host_only_note": "/host/path/should/never/cross",
                    },
                    {
                        "result_name": "parent_control",
                        "mode": "valid",
                        "status": "ok",
                        "complete_validation": True,
                        "parent_control": True,
                        "total_return": 0.06,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    history = compact_fold_history(
        {
            "record_type": "fold",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2024",
            "run_manifest_ref": str(manifest),
        },
        ref_store=AgentRefStore(tmp_path / "experiment"),
    )
    candidate, control = history["backtest_summaries"]
    assert candidate["vs_parent"]["beats_parent"] is True
    assert candidate["vs_parent"]["excess_return_delta"] == 0.06
    assert "host_only_note" not in candidate
    # The control is the baseline, not a candidate: it is never compared to
    # itself, so it carries no block.
    assert "vs_parent" not in control
