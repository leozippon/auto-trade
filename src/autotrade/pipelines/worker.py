"""Strict local assembly for persistent interactive daily experiments."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from autotrade.agent.compact import ContextCompactionConfig
from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
    # Single source of the frozen-artifact immutability rule: enforce the
    # read-only tree directly whenever a frozen artifact is consumed.
    _assert_readonly_tree,
)
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.research_release import pin_research_release
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import (
    AGENT_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
    LOCAL_QWEN_MODEL,
    LLMProxy,
    build_model_gateway,
    canonicalize_model_name,
    effective_max_output_tokens,
    model_profile,
)
from autotrade.environment.nl import NLConfig
from autotrade.environment.runtime import utc_now_iso, write_json_atomic
from autotrade.environment.sandbox import DEFAULT_IMAGE, SandboxConfig, SandboxSpec
from autotrade.environment.sandbox_images import prepare_experiment_sandbox_image
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.tools.base import CommandRunner

from .config import AcceptanceRules, FrozenArtifact, RollingExperimentConfig
from .experiment import RollingExperimentPipeline
from .folds import build_fold_schedule, heldout_periods, load_sse_trading_days
from .hitl_state import (
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    StatusReporter,
    build_session_plan,
    iter_development_sessions,
    read_control,
    read_json,
    read_status,
)
from .interactive import InteractiveExperimentRunner
from .ledger import (
    ExperimentLedger,
    FrozenArtifactMutated,
    assert_no_frozen_artifact_mutation,
)
from .local_backend import (
    DeterministicBaselineDeveloper,
    LLMFoldDeveloper,
    LLMMetaLearner,
    LocalDailyEvaluationBackend,
    LocalDailySnapshotProvider,
)
from .pit_backend import (
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    required_release_raw_datasets,
)
from .pit_views_seed import DEFAULT_PIT_VIEWS_SEED
from .prior import latest_prior_text, restore_current_from_records
from .skills import latest_skills_snapshot

_ALLOWED_PARAMS = {
    "experiment_id",
    "strategy_path",
    "baseline_strategy_path",
    "daily_path",
    "data_backend",
    "raw_dir",
    "fundamental_events_root",
    "fundamental_events_status",
    "pit_cache_root",
    "execution_mode",
    "strategy_period",
    "inference_time",
    "initial_cash",
    "initial_control_mode",
    "analysis_enabled",
    "analysis_model",
    "analysis_max_tokens",
    "daily_window_months",
    "fundamentals_window_months",
    "events_window_months",
    "macro_window_months",
    "text_window_months",
    "intraday_trade_days",
    "include_fundamentals",
    "include_macro",
    "include_events",
    "include_text",
    "include_intraday",
    "fundamental_datasets",
    "macro_datasets",
    "events_datasets",
    "text_datasets",
    "screen_exclude_st",
    "screen_exclude_new_listed_days",
    "screen_min_circ_mv_yi",
    "screen_max_circ_mv_yi",
    "screen_min_price",
    "screen_max_price",
    "screen_boards",
    "nl_max_results",
    "nl_max_calls_per_decision",
    "nl_max_total_calls",
    "nl_deadline_seconds",
    "max_intraday_row_group_rows",
    "developer_mode",
    "fold_period",
    "first_test_period",
    "last_test_period",
    "heldout_first_period",
    "heldout_last_period",
    "epochs",
    "window_months",
    "min_region_trade_days",
    "max_steps_per_fold",
    "max_backtests_per_fold",
    "max_llm_calls",
    "session_max_attempts",
    "max_fold_minutes",
    "min_return",
    "min_sharpe",
    "max_drawdown",
    "meta_learning_fold_interval",
    "meta_memory_max_epochs",
    "inherit_from",
    "meta_learning_directive",
    "fold_exploration_directive",
    "workspace_reference",
    "disable_step_tree",
    "record_failed_attempts",
    "convergence_start_epoch",
    "nl_failure_policy",
    "finalize_before_deadline_seconds",
    "per_call_timeout_seconds",
    "commission_bps",
    "slippage_bps",
    "max_total_holdings",
    "max_single_name_weight",
    "gpu_count",
    "disable_meta_sandbox_rebuild",
    "meta_sandbox_rebuild_timeout_seconds",
    "meta_sandbox_image_keep",
    "experiments_root",
    "work_root",
    "llm_api_key_env",
    "llm_env_file",
    "llm_model",
    "llm_timeout_seconds",
    "llm_max_retries",
    "llm_retry_backoff_seconds",
    "llm_temperature",
    "llm_max_response_tokens",
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
    "agent_sandbox_image",
    "agent_sandbox_cpus",
    "agent_sandbox_memory",
    "agent_sandbox_pids",
    "agent_sandbox_tmpfs",
}

# Single source for the NL budget defaults advertised to experiment parameters.
NL_DEFAULTS = NLConfig()

# NL Sub Agent reasoning tier. Independent of the experiment's
# ``reasoning_effort``, which governs the strategy-design dialogues; `medium` is
# a native Qwen tier and passes through the shared gateway unmapped.
NL_REASONING_EFFORT = "medium"

# The experiment-level ``reasoning_effort`` offers exactly the levels that are
# distinct on the wire for the local Qwen profile (its chat template knows
# low/medium/xhigh; ``high`` was always sent as ``xhigh``). The legacy aliases
# stay readable so existing params.json files keep launching.
REASONING_EFFORTS = ("low", "medium", "xhigh")
DEFAULT_REASONING_EFFORT = "xhigh"
LEGACY_REASONING_EFFORTS = {"high": "xhigh", "max": "xhigh"}

# Historical snapshots may contain this former operator override.  It is
# deliberately ignored rather than interpreted or exposed: provider endpoints
# now come only from the trusted model profile's fixed environment key.
NON_PERSISTABLE_PARAMS = frozenset({"llm_base_url"})


@dataclass(frozen=True)
class LLMWorkerSettings:
    api_key_env: str
    env_file: Path
    model: str
    meta_model: str
    nl_model: str
    compact_model: str
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    temperature: float
    max_response_tokens: int
    thinking_enabled: bool
    reasoning_effort: str
    compact_enabled: bool
    compaction: ContextCompactionConfig

    def max_tokens_for(
        self,
        role: str,
        *,
        model: str | None = None,
        requested: int | None = None,
    ) -> int:
        if role not in {"main", "meta", "nl", "compact", "analysis"}:
            raise ValueError(f"unknown model role: {role}")
        selected_model = (
            model
            or {
                "main": self.model,
                "meta": self.meta_model,
                "nl": self.nl_model,
                "compact": self.compact_model,
                "analysis": self.model,
            }[role]
        )
        configured = (
            requested
            if requested is not None
            else self.compaction.max_response_tokens
            if role == "compact"
            else self.max_response_tokens
        )
        return effective_max_output_tokens(selected_model, configured)

    def build_gateway(
        self,
        role: str = "main",
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        require_credentials: bool = True,
        max_retries: int | None = None,
    ) -> LLMProxy:
        if role not in {"main", "meta", "nl", "compact", "analysis"}:
            raise ValueError(f"unknown model role: {role}")
        selected_model = (
            model
            or {
                "main": self.model,
                "meta": self.meta_model,
                "nl": self.nl_model,
                "compact": self.compact_model,
                "analysis": self.model,
            }[role]
        )
        effective_max_tokens = self.max_tokens_for(
            role, model=selected_model, requested=max_tokens
        )
        return build_model_gateway(
            selected_model,
            env_file=self.env_file,
            deepseek_api_key_env=self.api_key_env,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries if max_retries is None else max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            max_tokens=effective_max_tokens,
            temperature=self.temperature,
            thinking_enabled=self.thinking_enabled if role != "compact" else False,
            reasoning_effort=self.reasoning_effort_for(role),
            require_credentials=require_credentials,
        )

    def reasoning_effort_for(self, role: str) -> str | None:
        """The effort a role sends, or None when its thinking is off."""

        if role == "compact" or not self.thinking_enabled:
            return None
        if role == "analysis":
            return "high"
        if role == "nl":
            # NL extracts evidence from already-retrieved PIT text and answers
            # short or enum-bounded questions; it is also the only LLM
            # inference inside a backtest's wall clock, so it runs at a
            # deliberately lower effort than the strategy-design dialogues.
            return NL_REASONING_EFFORT
        return self.reasoning_effort


@dataclass(frozen=True)
class InteractiveWorkerOptions:
    experiment_id: str
    experiment_dir: Path
    repo_root: Path
    baseline_strategy: Path
    daily_path: Path | None
    data_backend: str
    execution_mode: str
    developer_mode: str
    initial_control_mode: str
    rolling: RollingExperimentConfig
    work_root: Path
    raw_dir: Path | None = None
    fundamental_events_root: Path | None = None
    fundamental_events_status: Path | None = None
    pit_cache_root: Path | None = None
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
    nl_config: NLConfig = field(default_factory=NLConfig)
    max_intraday_row_group_rows: int = 2_000_000
    analysis_enabled: bool = False
    analysis_model: str = LOCAL_QWEN_MODEL
    analysis_max_tokens: int = 6_000
    llm: LLMWorkerSettings | None = None
    agent_sandbox: SandboxSpec | None = None


def load_worker_options(
    experiment_dir: str | Path,
    *,
    repo_root: str | Path,
) -> InteractiveWorkerOptions:
    directory = Path(experiment_dir).resolve(strict=True)
    params_path = directory / "hitl" / "params.json"
    params = read_json(params_path)
    if not params:
        raise ValueError(f"missing experiment params: {params_path}")
    return resolve_worker_options(params, experiment_dir=directory, repo_root=repo_root)


def resolve_worker_options(
    params: Mapping[str, object],
    *,
    experiment_dir: str | Path,
    repo_root: str | Path,
    preflight: bool = False,
) -> InteractiveWorkerOptions:
    """Validate one experiment's parameters and assemble its worker options.

    The single validation source for both the worker (which reads the durable
    ``hitl/params.json``) and the console's create-time pre-flight, so a
    parameter the console accepts cannot fail seconds later inside the worker.

    ``preflight`` validates parameters against an experiment directory that
    does not exist yet, so it checks the request rather than the deployment:
    input paths are checked for repository containment but not for existence,
    and the two steps that consume the durable data inputs are skipped — the
    immutable research-release pin (which materialises state inside the
    experiment directory) and the trading-calendar-dependent fold schedule.
    Every parameter check runs either way.
    """
    params = {
        key: value for key, value in params.items() if key not in NON_PERSISTABLE_PARAMS
    }
    directory = Path(experiment_dir).resolve()
    repository = Path(repo_root).resolve(strict=True)
    # The deployment's template file and data roots are identical for every
    # create on a host, so their absence is not a property of the request.
    repo_file = _repo_path if preflight else _repo_file
    repo_dir = _repo_path if preflight else _repo_dir
    unknown = sorted(
        key for key in params if not key.startswith("_") and key not in _ALLOWED_PARAMS
    )
    if unknown:
        raise ValueError(f"unknown experiment parameters: {unknown}")
    if params.get("_creation_surface") == "webui":
        altered = [
            key
            for key, value in WEB_INTERNAL_PARAMS.items()
            if params.get(key) != value
        ]
        if altered:
            raise ValueError(
                "WebUI experiment has altered console-managed parameters: "
                + ", ".join(sorted(altered))
            )
    experiment_id = _required_text(params, "experiment_id")
    if directory.name != experiment_id:
        raise ValueError("experiment_id does not match the experiment directory")
    if "experiments_root" in params:
        configured_root = _path_value(
            params["experiments_root"], repository, "experiments_root"
        )
        if configured_root != directory.parent:
            raise ValueError(
                "experiments_root does not match the experiment directory parent"
            )
    work_root = _path_value(
        params.get("work_root", ".runtime/sandboxes"),
        repository,
        "work_root",
    )
    if repository != work_root and repository not in work_root.parents:
        raise ValueError("work_root must stay inside the repository")
    baseline_value = params.get("baseline_strategy_path", params.get("strategy_path"))
    baseline = repo_file(repository, baseline_value, "baseline strategy")
    requested_backend = params.get("data_backend")
    if requested_backend in (None, ""):
        production_raw = repository / "data" / "raw"
        data_backend = (
            "pit"
            if str(params.get("developer_mode") or "baseline") == "llm"
            and production_raw.is_dir()
            else "daily"
        )
    else:
        data_backend = str(requested_backend)
    if data_backend not in {"daily", "pit"}:
        raise ValueError("data_backend must be daily or pit")
    daily = (
        repo_file(repository, params.get("daily_path"), "daily Parquet")
        if data_backend == "daily"
        else None
    )
    execution_mode = str(params.get("execution_mode") or "sandbox")
    if execution_mode not in {"sandbox", "trusted"}:
        raise ValueError("execution_mode must be sandbox or trusted")
    developer_mode = str(params.get("developer_mode") or "baseline")
    if developer_mode not in {"baseline", "llm"}:
        raise ValueError("developer_mode must be baseline or llm")
    llm_settings, sandbox_spec = (
        _llm_settings(params, repository, preflight=preflight)
        if developer_mode == "llm"
        else (None, None)
    )
    initial_control_mode = str(params.get("initial_control_mode") or "manual")
    if initial_control_mode not in {"auto", "manual", "step"}:
        raise ValueError("initial_control_mode must be auto, manual, or step")
    snapshot_config = _snapshot_config(params)
    raw_dir = (
        repo_dir(repository, params.get("raw_dir", "data/raw"), "raw_dir")
        if data_backend == "pit"
        else None
    )
    events_root = (
        repo_dir(
            repository,
            params.get("fundamental_events_root", "data/pit/fundamental_events"),
            "fundamental_events_root",
        )
        if data_backend == "pit"
        else None
    )
    events_status = (
        _repo_path(
            repository,
            params.get(
                "fundamental_events_status",
                "results/data_quality/fundamental_events_status.json",
            ),
            "fundamental_events_status",
        )
        if data_backend == "pit"
        else None
    )
    pit_cache_root = (
        _path_value(
            params.get("pit_cache_root", str(directory / "pit_views")),
            repository,
            "pit_cache_root",
        )
        if data_backend == "pit"
        else None
    )
    if pit_cache_root is not None and not (
        pit_cache_root == directory or pit_cache_root.is_relative_to(directory)
    ):
        raise ValueError("pit_cache_root must stay inside the experiment directory")
    trading_days: list[str] = []
    if preflight:
        pass  # the release pin writes into the experiment dir; see the docstring
    elif data_backend == "daily":
        assert daily is not None
        frame = pd.read_parquet(daily, columns=["trade_date"])
        trading_days = sorted(set(frame["trade_date"].map(_date_key).tolist()))
    else:
        assert (
            raw_dir is not None
            and events_root is not None
            and events_status is not None
        )
        release = pin_research_release(
            experiment_dir=directory,
            raw_dir=raw_dir,
            fundamental_events_root=events_root,
            fundamental_events_status=events_status,
            required_raw_datasets=required_release_raw_datasets(snapshot_config),
        )
        trading_days = load_sse_trading_days(release.raw_dir)
    if not trading_days and not preflight:
        raise ValueError("daily Parquet has no trading days")
    fold_period = str(params.get("fold_period") or "quarter")
    supplied_periods = [
        params.get("first_test_period"),
        params.get("last_test_period"),
        params.get("heldout_first_period"),
        params.get("heldout_last_period"),
    ]
    if not all(value not in (None, "") for value in supplied_periods):
        raise ValueError("all four Development/Held-out period fields are required")
    first_test, last_test, first_heldout, last_heldout = (
        str(value) for value in supplied_periods
    )
    schedule = StrategySchedule(
        str(params.get("strategy_period") or "day"),  # type: ignore[arg-type]
        str(params.get("inference_time") or "08:30"),
    )
    initial_cash = _positive_float(
        params.get("initial_cash", 1_000_000), "initial_cash"
    )
    rolling = RollingExperimentConfig(
        experiment_id=experiment_id,
        experiments_root=directory.parent,
        first_test_period=first_test,
        last_test_period=last_test,
        heldout_first_period=first_heldout,
        heldout_last_period=last_heldout,
        fold_period=fold_period,
        epochs=_positive_int(params.get("epochs", 1), "epochs"),
        window_months=_positive_int(params.get("window_months", 21), "window_months"),
        min_region_trade_days=_positive_int(
            params.get("min_region_trade_days", 2), "min_region_trade_days"
        ),
        max_steps_per_fold=_positive_int(
            params.get("max_steps_per_fold", 10), "max_steps_per_fold"
        ),
        max_backtests_per_fold=_positive_int(
            params.get("max_backtests_per_fold", 15), "max_backtests_per_fold"
        ),
        max_llm_calls=_positive_int(params.get("max_llm_calls", 400), "max_llm_calls"),
        session_max_attempts=_positive_int(
            params.get("session_max_attempts", 3), "session_max_attempts"
        ),
        max_fold_minutes=_positive_int(
            params.get("max_fold_minutes", 240), "max_fold_minutes"
        ),
        meta_learning_fold_interval=_nonnegative_int(
            params.get("meta_learning_fold_interval", 0), "meta_learning_fold_interval"
        ),
        meta_memory_max_epochs=_nonnegative_int(
            params.get("meta_memory_max_epochs", 3), "meta_memory_max_epochs"
        ),
        meta_learning_directive=_calendar_free_text(
            params.get("meta_learning_directive"), "meta_learning_directive"
        ),
        fold_exploration_directive=_calendar_free_text(
            params.get("fold_exploration_directive"), "fold_exploration_directive"
        ),
        workspace_reference=_optional_workspace_reference(
            params.get("workspace_reference"), repository
        ),
        step_tree_enabled=not _strict_bool(
            params.get("disable_step_tree", False), "disable_step_tree"
        ),
        record_failed_attempts=_strict_bool(
            params.get("record_failed_attempts", True), "record_failed_attempts"
        ),
        convergence_start_epoch=_positive_int(
            params.get("convergence_start_epoch", 3), "convergence_start_epoch"
        ),
        nl_failure_policy=_nl_failure_policy(
            params.get("nl_failure_policy", "return_error_with_audit")
        ),
        finalize_before_deadline_seconds=_nonnegative_int(
            params.get("finalize_before_deadline_seconds", 300),
            "finalize_before_deadline_seconds",
        ),
        per_call_timeout_seconds=_positive_int(
            params.get("per_call_timeout_seconds", 3600), "per_call_timeout_seconds"
        ),
        meta_sandbox_rebuild_enabled=not _strict_bool(
            params.get("disable_meta_sandbox_rebuild", False),
            "disable_meta_sandbox_rebuild",
        ),
        meta_sandbox_rebuild_timeout_seconds=_nonnegative_int(
            params.get("meta_sandbox_rebuild_timeout_seconds", 1800),
            "meta_sandbox_rebuild_timeout_seconds",
        ),
        meta_sandbox_image_keep=_nonnegative_int(
            params.get("meta_sandbox_image_keep", 3), "meta_sandbox_image_keep"
        ),
        acceptance=AcceptanceRules(
            min_return=_finite_float(params.get("min_return", 0.0), "min_return"),
            min_sharpe=_finite_float(params.get("min_sharpe", 0.0), "min_sharpe"),
            max_drawdown=_bounded_float(
                params.get("max_drawdown", 0.25), "max_drawdown", 0.0, 1.0
            ),
        ),
        schedule=schedule,
        broker_profile=BrokerProfile(
            initial_cash=initial_cash,
            commission_bps=_nonnegative_float(
                params.get("commission_bps", 1.0), "commission_bps"
            ),
            slippage_bps=_nonnegative_float(
                params.get("slippage_bps", 5.0), "slippage_bps"
            ),
            max_total_holdings=_optional_positive_int(
                params.get("max_total_holdings"), "max_total_holdings"
            ),
            max_single_name_weight=_optional_positive_float(
                params.get("max_single_name_weight"), "max_single_name_weight"
            ),
        ),
    )
    # Validate the derived/supplied schedule before the worker advertises it.
    if not preflight:
        build_fold_schedule(
            rolling.first_test_period,
            rolling.last_test_period,
            trading_days,
            window_months=rolling.window_months,
            period=rolling.fold_period,
            min_region_trade_days=rolling.min_region_trade_days,
        )
    analysis_enabled = _strict_bool(
        params.get("analysis_enabled", WEB_CREATE_DEFAULTS["analysis_enabled"]),
        "analysis_enabled",
    )
    analysis_model = canonicalize_model_name(
        str(params.get("analysis_model") or LOCAL_QWEN_MODEL)
    )
    analysis_max_tokens = _positive_int(
        params.get("analysis_max_tokens", 6_000), "analysis_max_tokens"
    )
    if analysis_enabled and llm_settings is not None:
        llm_settings.build_gateway(
            "analysis",
            model=analysis_model,
            max_tokens=analysis_max_tokens,
            require_credentials=not preflight,
        )
    return InteractiveWorkerOptions(
        experiment_id=experiment_id,
        experiment_dir=directory,
        repo_root=repository,
        baseline_strategy=baseline,
        daily_path=daily,
        data_backend=data_backend,
        execution_mode=execution_mode,
        developer_mode=developer_mode,
        initial_control_mode=initial_control_mode,
        rolling=rolling,
        work_root=work_root,
        raw_dir=raw_dir,
        fundamental_events_root=events_root,
        fundamental_events_status=events_status,
        pit_cache_root=pit_cache_root,
        snapshot_config=snapshot_config,
        # NLConfig owns the NL budget defaults; an absent parameter keeps the
        # shipped default rather than a second copy of it living here.
        nl_config=NLConfig(
            max_results=_positive_int(
                params.get("nl_max_results", NL_DEFAULTS.max_results),
                "nl_max_results",
            ),
            max_calls_per_decision=_positive_int(
                params.get(
                    "nl_max_calls_per_decision", NL_DEFAULTS.max_calls_per_decision
                ),
                "nl_max_calls_per_decision",
            ),
            max_total_calls=_optional_positive_int(
                params.get("nl_max_total_calls", NL_DEFAULTS.max_total_calls),
                "nl_max_total_calls",
            ),
            deadline_seconds=_positive_float(
                params.get("nl_deadline_seconds", NL_DEFAULTS.deadline_seconds),
                "nl_deadline_seconds",
            ),
        ),
        analysis_enabled=analysis_enabled,
        analysis_model=analysis_model,
        analysis_max_tokens=analysis_max_tokens,
        max_intraday_row_group_rows=_positive_int(
            params.get("max_intraday_row_group_rows", 2_000_000),
            "max_intraday_row_group_rows",
        ),
        llm=llm_settings,
        agent_sandbox=sandbox_spec,
    )


def _strategy_sandbox_from_spec(spec: SandboxSpec | None) -> SandboxConfig:
    if spec is None:
        return SandboxConfig(image=DEFAULT_IMAGE)
    return SandboxConfig(
        image=spec.image,
        docker_executable=spec.docker_executable,
    )


def _activate_experiment_sandbox(
    spec: SandboxSpec,
    *,
    developer: LLMFoldDeveloper,
    evaluator: PITDailyEvaluationBackend | LocalDailyEvaluationBackend,
) -> None:
    """Publish one active image to both Agent and formal evaluation paths."""

    developer.set_sandbox_spec(spec)
    evaluator.sandbox = replace(
        evaluator.sandbox,
        image=spec.image,
        docker_executable=spec.docker_executable,
    )


def run_local_interactive_worker(
    options: InteractiveWorkerOptions,
    *,
    llm: LLMProxy | None = None,
    command_runner_factory: Callable[[Path], CommandRunner] | None = None,
    poll_seconds: float = 2.0,
) -> dict[str, object]:
    ref_store = AgentRefStore(options.experiment_dir)
    hitl = options.experiment_dir / "hitl"
    ledger = ExperimentLedger(options.rolling.ledger_path)
    store = FilesystemArtifactStore(options.experiment_dir / "artifacts" / "strategy")
    try:
        assert_no_frozen_artifact_mutation(ledger.read())
    except FrozenArtifactMutated as exc:
        write_json_atomic(
            hitl / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    completed = read_status(hitl / "status.json")
    if str(completed.get("state")) == "completed" and not _has_outstanding_work(
        hitl, ledger
    ):
        # A finished experiment is terminal: every session and the held-out
        # evaluation are already durable in the ledger, so a resume must
        # republish the completion status instead of re-running anything.
        payload = _terminal_status(completed)
        write_json_atomic(hitl / "status.json", payload)
        return payload
    if (
        command_runner_factory is None
        and (options.execution_mode == "sandbox" or options.developer_mode == "llm")
    ):
        options = replace(
            options,
            agent_sandbox=prepare_experiment_sandbox_image(
                options.agent_sandbox or SandboxSpec(gpu=None),
                experiment_id=options.experiment_id,
                experiment_dir=options.experiment_dir,
            ),
        )
    fold_gateway = llm or (
        options.llm.build_gateway("main")
        if options.developer_mode == "llm" and options.llm
        else None
    )
    meta_gateway = llm or (
        options.llm.build_gateway("meta")
        if options.developer_mode == "llm" and options.llm
        else None
    )
    nl_gateway = llm or (
        options.llm.build_gateway("nl")
        if options.developer_mode == "llm" and options.llm
        else None
    )
    # A failed compaction falls through to the emergency fit path by design;
    # provider retries would only add their full latency to that failure.
    compact_gateway = llm or (
        options.llm.build_gateway("compact", max_retries=0)
        if options.developer_mode == "llm"
        and options.llm
        and options.llm.compact_enabled
        else None
    )
    strategy_sandbox = _strategy_sandbox_from_spec(options.agent_sandbox)
    if options.data_backend == "pit":
        if (
            options.raw_dir is None
            or options.fundamental_events_root is None
            or options.fundamental_events_status is None
        ):
            raise ValueError("data_backend=pit is missing validated raw/PIT paths")
        snapshots = ResearchPITSnapshotProvider(
            experiment_dir=options.experiment_dir,
            raw_dir=options.raw_dir,
            fundamental_events_root=options.fundamental_events_root,
            fundamental_events_status=options.fundamental_events_status,
            config=options.snapshot_config,
            cache_root=options.pit_cache_root,
            pit_views_seed=options.repo_root / DEFAULT_PIT_VIEWS_SEED,
        )
        evaluator = PITDailyEvaluationBackend(
            options.experiment_dir / "artifacts" / "results",
            execution_mode=options.execution_mode,
            nl_llm=nl_gateway,
            nl_config=options.nl_config,
            nl_failure_policy=options.rolling.nl_failure_policy,
            max_intraday_row_group_rows=options.max_intraday_row_group_rows,
            sandbox=strategy_sandbox,
        )
        trading_days = snapshots.trading_days
    else:
        if options.daily_path is None:
            raise ValueError("data_backend=daily requires daily_path")
        snapshots = LocalDailySnapshotProvider(options.daily_path)
        evaluator = LocalDailyEvaluationBackend(
            options.daily_path,
            options.experiment_dir / "artifacts" / "results",
            execution_mode=options.execution_mode,
            sandbox=strategy_sandbox,
        )
        trading_days = evaluator.trading_days
    if options.developer_mode == "llm":
        if options.llm is None or options.agent_sandbox is None:
            raise ValueError(
                "developer_mode=llm is missing validated LLM or sandbox settings"
            )
        if fold_gateway is None or meta_gateway is None:
            raise ValueError(
                "developer_mode=llm requires initialized Fold and Meta LLM gateways"
            )
        runtime_root = options.work_root / options.experiment_id
        developer = LLMFoldDeveloper(
            llm=fold_gateway,
            subagent_llm=fold_gateway,
            compact_llm=compact_gateway,
            context_compaction=options.llm.compaction,
            baseline_strategy=options.baseline_strategy,
            artifact_store=store,
            evaluator=evaluator,
            schedule=options.rolling.schedule,
            broker_profile=options.rolling.broker_profile,
            ledger=ledger,
            experiment_dir=options.experiment_dir,
            runtime_root=runtime_root,
            sandbox_spec=options.agent_sandbox,
            command_runner_factory=command_runner_factory,
            # One ceiling for the parent conversation and its children.
            max_response_tokens=options.llm.max_tokens_for("main"),
            step_tree_enabled=options.rolling.step_tree_enabled,
            fold_exploration_directive=options.rolling.fold_exploration_directive,
            workspace_reference=options.rolling.workspace_reference,
            repo_root=options.repo_root,
        )
        meta_learner = LLMMetaLearner(
            llm=meta_gateway,
            subagent_llm=meta_gateway,
            compact_llm=compact_gateway,
            context_compaction=options.llm.compaction,
            baseline_strategy=options.baseline_strategy,
            artifact_store=store,
            experiment_dir=options.experiment_dir,
            runtime_root=runtime_root,
            max_llm_calls=options.rolling.max_llm_calls,
            deadline_seconds=options.rolling.max_fold_minutes * 60,
            max_response_tokens=options.llm.max_tokens_for("meta"),
            meta_learning_directive=options.rolling.meta_learning_directive,
            fold_exploration_directive=options.rolling.fold_exploration_directive,
            workspace_reference=options.rolling.workspace_reference,
            repo_root=options.repo_root,
            regularization_constraints=options.rolling.regularization_constraints,
            sandbox_spec=options.agent_sandbox,
            # A sandboxless smoke run must not shell out to docker build.
            use_docker=command_runner_factory is None,
            rebuild_enabled=options.rolling.meta_sandbox_rebuild_enabled,
            rebuild_timeout_seconds=options.rolling.meta_sandbox_rebuild_timeout_seconds,
            image_keep=options.rolling.meta_sandbox_image_keep,
            # A derived image becomes the single image for both later Fold
            # Agent sessions and every formal evaluation mode.
            sandbox_spec_sink=lambda spec: _activate_experiment_sandbox(
                spec,
                developer=developer,
                evaluator=evaluator,
            ),
        )
        meta_enabled = True
        developer_label = "llm_fold_meta_agent"
    else:
        developer = DeterministicBaselineDeveloper(
            baseline_strategy=options.baseline_strategy,
            artifact_store=store,
            evaluator=evaluator,
            schedule=options.rolling.schedule,
            broker_profile=options.rolling.broker_profile,
            ref_store=ref_store,
        )
        meta_learner = None
        meta_enabled = False
        developer_label = "deterministic_baseline_no_agent_improvement"
    pipeline = RollingExperimentPipeline(
        options.rolling,
        snapshots=snapshots,
        artifacts=store,
        evaluator=evaluator,
        developer=developer,
        meta_learner=meta_learner,
        ledger=ledger,
    )
    folds = build_fold_schedule(
        options.rolling.first_test_period,
        options.rolling.last_test_period,
        trading_days,
        window_months=options.rolling.window_months,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
    )
    heldout = heldout_periods(
        options.rolling.heldout_first_period,
        options.rolling.heldout_last_period,
        trading_days,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
    )
    sessions = iter_development_sessions(
        options.rolling.epochs,
        folds,
        meta_enabled=meta_enabled,
        meta_learning_fold_interval=options.rolling.meta_learning_fold_interval,
    )
    plan = build_session_plan(
        options.rolling.epochs,
        folds,
        heldout,
        meta_enabled=meta_enabled,
        meta_learning_fold_interval=options.rolling.meta_learning_fold_interval,
    )
    write_json_atomic(hitl / "schedule.json", plan)
    # Inherited seed (from another experiment's frozen output) replaces the
    # blank template as the first fold's parent; a resumed experiment takes its
    # parent from its own ledger instead.
    _restore_prior_store(options.experiment_dir, ledger)
    # Skills have no mutable CURRENT pointer: validating the final remaining
    # successful Fold/Meta row is the complete resume/rollback restore step.
    latest_skills_snapshot(ledger.read(), experiment_dir=options.experiment_dir)
    state = {
        "parent": _latest_artifact(ledger, store)
        or _load_inherited_parent(options.experiment_dir),
        "prior": _latest_prior(ledger, options.experiment_dir),
    }

    def execute(session, context):
        if session.fold is None:
            raise RuntimeError(f"unsupported local session kind: {session.kind}")
        if session.kind == "meta":
            # A Meta session may regularize the artifact; the next Fold then
            # starts from the regularized parent, not the pre-Meta one.
            state["prior"], state["parent"] = pipeline.run_meta_session(
                session.epoch_id,
                session.fold_index,
                session.fold,
                parent=state["parent"],
                previous_prior=str(state["prior"]),
                session_context=context,
            )
            return
        if session.kind != "fold":
            raise RuntimeError(f"unsupported local session kind: {session.kind}")
        override_node = str(context.get("parent_override") or "")
        session_parent = (
            _parent_from_step_node(
                options.experiment_dir, override_node, session.session_key
            )
            if override_node
            else state["parent"]
        )
        outcome = pipeline.run_fold(
            session.epoch_id,
            session.fold,
            parent=session_parent,
            prior=str(state["prior"]),
            session_context=context,
        )
        state["parent"] = outcome.frozen
        return  # run_fold already appended the canonical ledger record

    interactive = InteractiveExperimentRunner(
        experiment_id=options.experiment_id,
        sessions=sessions,
        execute_session=execute,
        ledger=ledger,
        control_path=hitl / "control.json",
        status_path=hitl / "status.json",
        ref_store=ref_store,
        poll_seconds=poll_seconds,
        post_fold_hook=_build_post_fold_hook(options, hitl / "analysis"),
        session_max_attempts=options.rolling.session_max_attempts,
    )
    result = interactive.run()
    if result["status"] != "complete":
        return result
    final = state["parent"] or _latest_artifact(ledger, store)
    if final is None:
        raise RuntimeError("Development completed without a frozen baseline artifact")
    final_status = StatusReporter(hitl / "status.json")
    final_status.start()
    completed_development = len(
        {
            str(row.get("session_key") or row.get("run_id"))
            for row in ledger.read()
            if row.get("record_type") in {"fold", "meta_learning"}
        }
    )
    final_status.set(
        state="running_heldout",
        developer_mode=developer_label,
        session_key="heldout",
        session_started_at=utc_now_iso(),
        completed_sessions=completed_development,
        total_sessions=len(sessions) + 1,
        environment_stage="heldout",
    )
    try:
        heldout_runs = pipeline.run_heldout(
            f"epoch_{options.rolling.epochs:03d}",
            final,
            trading_days,
            replay=bool(result.get("reran_sessions")),
        )
    finally:
        final_status.stop()
    payload = _terminal_status(
        {
            "completed_at": utc_now_iso(),
            "developer_mode": developer_label,
            "completed_sessions": completed_development + 1,
            "total_sessions": len(sessions) + 1,
            "final_strategy_artifact": final.artifact_id,
        },
        developer_mode=developer_label,
        heldout_runs=heldout_runs,
    )
    write_json_atomic(hitl / "status.json", payload)
    return payload


def _latest_artifact(
    ledger: ExperimentLedger,
    store: FilesystemArtifactStore,
) -> FrozenArtifact | None:
    records = ledger.read()
    assert_no_frozen_artifact_mutation(records)
    current_id = ""
    current_path = ""
    current_record: dict[str, object] | None = None
    requires_validation = False
    for record in records:
        record_type = record.get("record_type")
        if record_type == "fold":
            artifact_id = str(record.get("frozen_strategy_artifact_id") or "")
            path = str(record.get("frozen_strategy_artifact_path") or "")
            if not artifact_id:
                continue
            current_id = artifact_id
            current_path = path
            current_record = record
            requires_validation = False
        elif (
            record_type == "meta_learning"
            and record.get("status") == "meta_regularized"
        ):
            artifact_id = str(record.get("frozen_strategy_artifact_id") or "")
            if not artifact_id:
                continue
            current_id = artifact_id
            current_path = str(record.get("frozen_strategy_artifact_path") or "")
            current_record = record
            requires_validation = True
    if not current_id or current_record is None:
        return None
    try:
        frozen = store.frozen(
            current_id,
            expected_path=current_path or None,
            experiment_id=str(current_record.get("experiment_id") or ""),
        )
    except Exception as exc:
        raise RuntimeError(
            f"ledger artifact failed validation: {current_id}: {exc}"
        ) from exc
    return FrozenArtifact(
        current_id,
        Path(frozen.path),
        Path(frozen.model_path) if frozen.model_path is not None else None,
        str(frozen.source_run_id),
        str(frozen.source_fold_id),
        str(frozen.source_step_id),
        str(frozen.revision_id),
        requires_validation=requires_validation,
    )


def _terminal_status(
    source: Mapping[str, object],
    *,
    developer_mode: str | None = None,
    heldout_runs: int = 0,
) -> dict[str, object]:
    """Durable completion status. The experiment's own evidence stays in the
    append-only ledger; status.json only records that the run reached its end,
    so a resume republishes it without re-running or re-recording anything
    (``heldout_runs`` counts what THIS invocation executed)."""
    return {
        "schema_version": 1,
        "state": "completed",
        "pid": os.getpid(),
        "completed_at": source.get("completed_at") or utc_now_iso(),
        "developer_mode": developer_mode or source.get("developer_mode"),
        "completed_sessions": source.get("completed_sessions"),
        "total_sessions": source.get("total_sessions"),
        "heldout_runs": heldout_runs,
        "final_strategy_artifact": source.get("final_strategy_artifact"),
    }


def _has_outstanding_work(hitl: Path, ledger: ExperimentLedger) -> bool:
    """Whether a completed experiment still has console-requested work.

    A rollback drops every held-out record (the frontier moved back) and a
    rerun request leaves a token no fold record has absorbed yet. In both cases
    the worker must resume instead of republishing the terminal status, or the
    console operation would look accepted and silently do nothing."""
    records = ledger.read()
    if not any(record.get("record_type") == "heldout" for record in records):
        return True
    pending = read_control(hitl / "control.json").rerun_sessions
    if not pending:
        return False
    absorbed: dict[str, str] = {}
    for record in records:
        if record.get("record_type") == "fold":
            absorbed[str(record.get("session_key") or "")] = str(
                record.get("rerun_id") or ""
            )
    return any(absorbed.get(key) != token for key, token in pending.items())


def _load_inherited_parent(experiment_dir: Path) -> FrozenArtifact | None:
    """The read-only artifact snapshot the console copied in at creation.

    The snapshot is validated here rather than trusted: the console locked it
    read-only, so a tree that is missing or has become writable again is a
    tampered seed and must stop the run instead of silently starting from
    unverified strategy code."""
    payload = read_json(Path(experiment_dir) / "hitl/params.json").get(
        "_inherited_artifact"
    )
    if not isinstance(payload, dict):
        return None
    path = Path(str(payload.get("path") or ""))
    if not path.is_dir():
        raise RuntimeError(f"inherited artifact directory is missing: {path}")
    _assert_readonly_tree(path)
    model_path = payload.get("model_path")
    models = Path(str(model_path)) if model_path else None
    if models is not None:
        if not models.is_dir():
            raise RuntimeError(
                f"inherited model artifact directory is missing: {models}"
            )
        _assert_readonly_tree(models)
    return FrozenArtifact(
        artifact_id=str(payload.get("artifact_id") or ""),
        path=path,
        model_path=models,
        source_run_id="",
        source_fold_id=str(payload.get("source_fold_id") or ""),
        source_step_id="",
        revision_id=str(payload.get("revision_id") or ""),
    )


def _parent_from_step_node(
    experiment_dir: Path, node_id: str, session_key: str
) -> FrozenArtifact:
    """Build the session parent from a validated step-tree node snapshot.

    The worker re-validates chronology itself: control.json is a plain file,
    so the console-side check alone would not stop a hand-edited override
    from leaking a later fold's validated strategy backwards."""
    from autotrade.environment.step_tree import (
        NODE_MODELS_DIR,
        NODE_OUTPUT_DIR,
        StepTree,
    )

    from .hitl_state import assert_node_not_from_later_fold

    steps_root = Path(experiment_dir) / "steps"
    tree = StepTree(steps_root)
    node = tree.get_node(node_id)  # ValueError on unknown ids -- fail fast
    if node.get("status") == "failed" or not node.get("complete_validation"):
        raise RuntimeError(
            f"parent override {node_id} is not a validated node with a snapshot"
        )
    schedule = read_json(Path(experiment_dir) / "hitl/schedule.json")
    raw_sessions = schedule.get("sessions")
    sessions: list[object] = raw_sessions if isinstance(raw_sessions, list) else []
    fold_keys = [
        str(item.get("session_key") or item.get("key") or "")
        for item in sessions
        if isinstance(item, dict) and item.get("kind") == "fold"
    ]
    assert_node_not_from_later_fold(
        node,
        session_key,
        fold_keys,
        ref_store=AgentRefStore(experiment_dir),
    )
    output_dir = steps_root / node_id / NODE_OUTPUT_DIR
    if not output_dir.is_dir():
        raise RuntimeError(
            f"parent override {node_id} has no strategy snapshot on disk"
        )
    models_dir = steps_root / node_id / NODE_MODELS_DIR
    return FrozenArtifact(
        artifact_id=f"stepnode_{node_id}",
        path=output_dir,
        model_path=models_dir if models_dir.is_dir() else None,
        source_run_id=str(node.get("run_id") or ""),
        source_fold_id=str(node.get("fold_id") or ""),
        source_step_id=node_id,
        revision_id=str(node.get("revision_id") or ""),
    )


def _build_post_fold_hook(
    options: InteractiveWorkerOptions, out_dir: Path
) -> Callable[[dict[str, object]], None] | None:
    """Fold-completion strategy analysis, when enabled and a provider exists."""

    if not options.analysis_enabled or options.llm is None:
        return None
    from .fold_analysis import analyze_fold

    effective_analysis_max_tokens = options.llm.max_tokens_for(
        "analysis",
        model=options.analysis_model,
        requested=options.analysis_max_tokens,
    )
    proxy = options.llm.build_gateway(
        "analysis",
        model=options.analysis_model,
        max_tokens=effective_analysis_max_tokens,
    )
    ref_store = AgentRefStore(options.experiment_dir)

    def post_fold_hook(record: dict[str, object]) -> None:
        strategy_dir = record.get("frozen_strategy_artifact_path")
        if not strategy_dir:
            raise ValueError("fold record has no frozen strategy artifact to analyse")
        model_dir = record.get("frozen_model_artifact_path")
        analyze_fold(
            proxy,
            ledger_record=record,
            ref_store=ref_store,
            strategy_dir=Path(str(strategy_dir)),
            model_dir=Path(str(model_dir)) if model_dir else None,
            out_dir=out_dir,
            max_tokens=effective_analysis_max_tokens,
            output_identity=(
                str(record.get("epoch_id") or "epoch_unknown"),
                ref_store.get_or_create("fold", str(record.get("fold_id") or "fold_unknown")),
            ),
        )

    return post_fold_hook


def _latest_prior(ledger: ExperimentLedger, experiment_dir: Path) -> str:
    return latest_prior_text(
        ledger.read("meta_learning"), experiment_dir=experiment_dir
    )


def _restore_prior_store(experiment_dir: Path, ledger: ExperimentLedger) -> None:
    """Align CURRENT with the last remaining Meta generation after resume/rollback."""
    restore_current_from_records(experiment_dir, ledger.read("meta_learning"))


def _repo_file(repo_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is required")
    raw = Path(value)
    path = (
        (repo_root / raw).resolve(strict=True)
        if not raw.is_absolute()
        else raw.resolve(strict=True)
    )
    if repo_root != path and repo_root not in path.parents:
        raise ValueError(f"{label} must stay inside the repository")
    if not path.is_file():
        raise ValueError(f"{label} must be a file")
    return path


def _repo_dir(repo_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is required")
    raw = Path(value)
    path = (
        (repo_root / raw).resolve(strict=True)
        if not raw.is_absolute()
        else raw.resolve(strict=True)
    )
    if repo_root != path and repo_root not in path.parents:
        raise ValueError(f"{label} must stay inside the repository")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    return path


def _repo_path(repo_root: Path, value: object, label: str) -> Path:
    """Resolve an input path without requiring the mutable live copy to exist.

    A dirty raw lake is allowed to fall back to a pinned immutable research
    release.  Its quality status lives inside that release, so requiring the
    corresponding live status file before pinning would reject the valid
    fallback path prematurely.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is required")
    raw = Path(value)
    path = (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if repo_root != path and repo_root not in path.parents:
        raise ValueError(f"{label} must stay inside the repository")
    return path


def _path_value(value: object, repo_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _snapshot_config(params: dict[str, object]) -> SnapshotConfig:
    """Map persisted WebUI/CLI parameters to the exact PIT snapshot contract."""

    base = SnapshotConfig()

    def enabled(name: str) -> bool:
        value = params.get(name, True)
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
        return value

    def selection(name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
        value = params.get(name, ())
        if value in (None, ""):
            selected: tuple[str, ...] = ()
        elif isinstance(value, (list, tuple)):
            selected = tuple(str(item).strip() for item in value)
        else:
            raise ValueError(f"{name} must be an array of dataset names")
        if any(not item for item in selected):
            raise ValueError(f"{name} must contain non-empty dataset names")
        if len(selected) != len(set(selected)):
            raise ValueError(f"{name} must not contain duplicates")
        unknown = sorted(set(selected) - set(allowed))
        if unknown:
            raise ValueError(f"unknown {name}: {unknown}")
        return selected

    def datasets(
        name: str, is_enabled: bool, defaults: tuple[str, ...]
    ) -> tuple[str, ...]:
        selected = selection(name, defaults)
        return (selected or defaults) if is_enabled else ()

    include_fundamentals = enabled("include_fundamentals")
    include_macro = enabled("include_macro")
    include_events = enabled("include_events")
    include_text = enabled("include_text")
    include_intraday = enabled("include_intraday")
    screen_exclude_st = params.get("screen_exclude_st", False)
    if type(screen_exclude_st) is not bool:
        raise ValueError("screen_exclude_st must be boolean")
    screen_boards = selection("screen_boards", ("main", "gem", "star", "bj"))
    return SnapshotConfig(
        window_months=_positive_int(params.get("window_months", 21), "window_months"),
        daily_window_months=_optional_positive_int(
            params.get("daily_window_months"), "daily_window_months"
        ),
        fundamentals_window_months=_optional_positive_int(
            params.get("fundamentals_window_months"), "fundamentals_window_months"
        ),
        events_window_months=_optional_positive_int(
            params.get("events_window_months"), "events_window_months"
        ),
        macro_window_months=_optional_positive_int(
            params.get("macro_window_months"), "macro_window_months"
        ),
        text_window_months=_optional_positive_int(
            params.get("text_window_months"), "text_window_months"
        ),
        intraday_trade_days=_positive_int(
            params.get("intraday_trade_days", base.intraday_trade_days),
            "intraday_trade_days",
        ),
        fundamental_datasets=datasets(
            "fundamental_datasets",
            include_fundamentals,
            base.fundamental_datasets,
        ),
        macro_datasets=datasets("macro_datasets", include_macro, base.macro_datasets),
        events_datasets=datasets(
            "events_datasets", include_events, base.events_datasets
        ),
        text_datasets=datasets("text_datasets", include_text, base.text_datasets),
        include_intraday=include_intraday,
        replay_include_fundamentals=include_fundamentals,
        replay_include_macro=include_macro,
        replay_include_events=include_events,
        replay_include_text=include_text,
        replay_include_minutes=include_intraday,
        screen_exclude_st=screen_exclude_st,
        screen_exclude_new_listed_days=_nonnegative_int(
            params.get("screen_exclude_new_listed_days", 0),
            "screen_exclude_new_listed_days",
        ),
        screen_min_circ_mv_yi=_optional_nonnegative_float(
            params.get("screen_min_circ_mv_yi"), "screen_min_circ_mv_yi"
        ),
        screen_max_circ_mv_yi=_optional_nonnegative_float(
            params.get("screen_max_circ_mv_yi"), "screen_max_circ_mv_yi"
        ),
        screen_min_price=_optional_nonnegative_float(
            params.get("screen_min_price"), "screen_min_price"
        ),
        screen_max_price=_optional_nonnegative_float(
            params.get("screen_max_price"), "screen_max_price"
        ),
        screen_boards=screen_boards,
    )


def _llm_settings(
    params: dict[str, object],
    repository: Path,
    *,
    preflight: bool = False,
) -> tuple[LLMWorkerSettings, SandboxSpec]:
    api_key_env = str(params.get("llm_api_key_env") or "DEEPSEEK_API_KEY")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError("llm_api_key_env must be an environment variable name")
    env_file = _path_value(
        params.get("llm_env_file", ".env"), repository, "llm_env_file"
    )
    if repository != env_file and repository not in env_file.parents:
        raise ValueError("llm_env_file must stay inside the repository")
    fold_model = canonicalize_model_name(
        str(params.get("model") or params.get("llm_model") or LOCAL_QWEN_MODEL)
    )
    meta_model = canonicalize_model_name(str(params.get("meta_model") or fold_model))
    nl_model = canonicalize_model_name(
        str(params.get("nl_model") or LOCAL_QWEN_MODEL)
    )
    compact_model = canonicalize_model_name(
        str(params.get("compact_model") or LOCAL_QWEN_MODEL)
    )
    compact_max_tokens = _positive_int(
        params.get("compact_max_tokens", 1_600),
        "compact_max_tokens",
    )
    # None = derived from the model windows; ``_resolve_compaction_threshold``
    # sets the effective value on the settings either way.
    configured_threshold = _optional_positive_int(
        params.get("compact_token_threshold"), "compact_token_threshold"
    )
    compaction = ContextCompactionConfig(
        token_threshold=configured_threshold or ContextCompactionConfig.token_threshold,
        keep_recent_messages=_positive_int(
            params.get("compact_keep_recent_messages", 12),
            "compact_keep_recent_messages",
        ),
        max_response_tokens=effective_max_output_tokens(
            compact_model, compact_max_tokens
        ),
        max_calls=_nonnegative_int(
            params.get("compact_max_calls", 8),
            "compact_max_calls",
        ),
    )
    settings = LLMWorkerSettings(
        api_key_env=api_key_env,
        env_file=env_file,
        model=fold_model,
        meta_model=meta_model,
        nl_model=nl_model,
        compact_model=compact_model,
        timeout_seconds=_positive_float(
            params.get(
                "llm_timeout_seconds",
                params.get("per_call_timeout_seconds", 3600),
            ),
            "llm_timeout_seconds",
        ),
        max_retries=_nonnegative_int(
            params.get("llm_max_retries", DEFAULT_LLM_MAX_RETRIES), "llm_max_retries"
        ),
        retry_backoff_seconds=_nonnegative_float(
            params.get("llm_retry_backoff_seconds", DEFAULT_LLM_RETRY_BACKOFF_SECONDS),
            "llm_retry_backoff_seconds",
        ),
        temperature=_bounded_float(
            params.get("llm_temperature", 0.0), "llm_temperature", 0.0, 2.0
        ),
        max_response_tokens=_positive_int(
            params.get("llm_max_response_tokens", AGENT_MAX_OUTPUT_TOKENS),
            "llm_max_response_tokens",
        ),
        thinking_enabled=not _strict_bool(
            params.get("no_thinking", False), "no_thinking"
        ),
        reasoning_effort=_reasoning_effort(
            params.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
        ),
        compact_enabled=not _strict_bool(
            params.get("disable_context_compact", False),
            "disable_context_compact",
        ),
        compaction=compaction,
    )
    # Whether the host holds a credential is deployment state, not a property
    # of a WebUI create request.  Every model/role combination is still
    # validated at preflight with a non-secret placeholder.
    for role in ("main", "meta", "nl", "compact"):
        settings.build_gateway(role, require_credentials=not preflight)
    settings = _resolve_compaction_threshold(settings, configured_threshold)
    gpu_count = _gpu_count(params.get("gpu_count", SandboxSpec().gpu_count))
    sandbox = SandboxSpec(
        image=str(params.get("agent_sandbox_image") or DEFAULT_IMAGE),
        cpus=_positive_float(
            params.get("agent_sandbox_cpus", SandboxSpec().cpus), "agent_sandbox_cpus"
        ),
        memory=_memory_limit(
            params.get("agent_sandbox_memory", SandboxSpec().memory), "agent_sandbox_memory"
        ),
        pids_limit=_positive_int(
            params.get("agent_sandbox_pids", 512), "agent_sandbox_pids"
        ),
        tmpfs_size=_memory_limit(
            params.get("agent_sandbox_tmpfs", "1g"), "agent_sandbox_tmpfs"
        ),
        gpu=None if gpu_count == 0 else "auto",
        gpu_count=gpu_count,
    )
    return settings, sandbox


# Tokens the compaction threshold leaves below "window − output budget". The
# threshold is checked before the sub-agent completion observations (≤6,000
# chars each), inbox messages and budget notices of that turn are appended,
# so the margin lets one such addition ride along without a forced
# compaction; the gateway keeps its own 2,048-token tokenizer slack on top.
COMPACTION_SAFETY_MARGIN_TOKENS = 8_192


def _resolve_compaction_threshold(
    settings: LLMWorkerSettings, configured: int | None
) -> LLMWorkerSettings:
    """The effective compaction threshold: derived from the windows, clamped.

    Every model role with a known context window bounds the threshold at
    ``window − that role's output budget − COMPACTION_SAFETY_MARGIN_TOKENS``,
    so prompt plus output never exceeds the window; the smallest bound is the
    default, and a configured value is clamped to it. The result reaches the
    run facts through the compaction budget. Only a window with no room for
    the output budget at all is a launch error.
    """

    roles = [("main", settings.model), ("meta", settings.meta_model)]
    if settings.compact_enabled:
        roles.append(("compact", settings.compact_model))
    bounds: list[int] = []
    for role, model in roles:
        window = model_profile(model).context_window_tokens
        if window is None:
            continue
        maximum = window - settings.max_tokens_for(role) - COMPACTION_SAFETY_MARGIN_TOKENS
        if maximum <= 0:
            raise ValueError(
                f"{role} model output budget leaves no context capacity"
            )
        bounds.append(maximum)
    if not bounds and configured is None:
        raise ValueError(
            "compact_token_threshold is required when no model role declares "
            "a context window"
        )
    effective = min([*bounds, *([configured] if configured is not None else [])])
    return replace(
        settings,
        compaction=replace(settings.compaction, token_threshold=effective),
    )


def _optional_workspace_reference(value: object, repo_root: Path) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ValueError("workspace_reference must be a string")
    text = value.strip()
    if not text:
        return ""
    # Config seed, not a data lake: existence is required even at create preflight.
    _repo_dir(repo_root, text, "workspace_reference")
    return text


def _required_text(params: dict[str, object], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _gpu_count(value: object) -> int:
    """Per-session GPU allocation shares this range with `set_gpu_count`."""
    count = _nonnegative_int(value, "gpu_count")
    if count > 4:
        raise ValueError("gpu_count must be between 0 and 4")
    return count


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")  # noqa: TRY004
    return value


def _reasoning_effort(value: object) -> str:
    effort = str(value)
    # Older params.json files may still carry the shared-scale aliases; on
    # the wire they were never distinct from xhigh, so they stay valid and
    # resolve to it rather than failing a running experiment.
    if effort in LEGACY_REASONING_EFFORTS:
        return LEGACY_REASONING_EFFORTS[effort]
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of "
            + ", ".join(REASONING_EFFORTS)
            + " (legacy high/max resolve to xhigh)"
        )
    return effort


def _optional_positive_int(value: object, name: str) -> int | None:
    if value in (None, ""):
        return None
    return _positive_int(value, name)


def _optional_nonnegative_float(value: object, name: str) -> float | None:
    if value in (None, ""):
        return None
    return _nonnegative_float(value, name)


def _optional_positive_float(value: object, name: str) -> float | None:
    if value in (None, ""):
        return None
    return _positive_float(value, name)


def _calendar_free_text(value: object, name: str) -> str:
    text = str(value or "")
    if not text.strip():
        return ""
    from autotrade.environment.tools.prior_policy import calendar_policy_violation

    reason = calendar_policy_violation(text)
    if reason:
        raise ValueError(f"{name} contains a non-transferable calendar date: {reason}")
    return text


def _nl_failure_policy(value: object) -> str:
    policy = str(value or "return_error_with_audit")
    if policy not in NL_FAILURE_POLICIES:
        raise ValueError(
            f"nl_failure_policy must be one of {sorted(NL_FAILURE_POLICIES)}"
        )
    return policy


NL_FAILURE_POLICIES = frozenset({"return_error_with_audit", "fail"})


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-negative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _bounded_float(value: object, name: str, lower: float, upper: float) -> float:
    number = _nonnegative_float(value, name)
    if not lower <= number <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return number


def _memory_limit(value: object, name: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*(?:[kKmMgG])?", text):
        raise ValueError(f"{name} must be a positive Docker memory limit")
    return text


def _date_key(value: object) -> str:
    return pd.Timestamp(str(value)).strftime("%Y%m%d")


__all__ = [
    "InteractiveWorkerOptions",
    "LLMWorkerSettings",
    "load_worker_options",
    "run_local_interactive_worker",
]
