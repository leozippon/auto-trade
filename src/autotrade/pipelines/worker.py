"""Strict local assembly for persistent interactive daily experiments."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple, NoReturn

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
from autotrade.environment.data.snapshot import (
    DEFAULT_DATASETS,
    SELECTABLE_DATASETS,
    SnapshotConfig,
)
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
from autotrade.environment.sandbox import (
    DEFAULT_IMAGE,
    SandboxConfig,
    SandboxLimits,
    SandboxSpec,
)
from autotrade.environment.sandbox_images import prepare_experiment_sandbox_image
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.tools.base import CommandRunner

from .config import (
    DEFAULT_PIT_VIEWS_SEED,
    AcceptanceRules,
    FrozenArtifact,
    RollingExperimentConfig,
    rolling_default,
)
from .experiment import RollingExperimentPipeline
from .folds import (
    FoldSpec,
    build_fold_schedule,
    deployment_fold,
    heldout_periods,
    load_sse_trading_days,
    yyyymmdd,
)
from .hitl_state import (
    CONTROL_MODES,
    DEPLOYMENT_SESSION_KEY,
    SCHEDULE_NAME,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    DevelopmentSession,
    StatusReporter,
    build_session_plan,
    epoch_ids,
    iter_development_sessions,
    read_control,
    read_json,
    read_status,
)
from .inherited_memory import load_inherited_memory
from .interactive import InteractiveExperimentRunner
from .ledger import (
    ExperimentLedger,
    FrozenArtifactMutated,
    RunMarkers,
    assert_no_frozen_artifact_mutation,
    baseline_anchor_artifacts,
    deployment_adjustment_due,
    experiment_verdict,
    is_durable_success_record,
    latest_fold_records,
    latest_heldout_records,
    paper_candidate,
    rerun_absorbed,
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
from .pit_views_seed import assert_seed_snapshot_config
from .prior import latest_prior_text, restore_current_from_records
from .skills import latest_skills_snapshot, resolve_operating_memory

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
    "pit_views_seed",
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
    "development_first_period",
    "development_last_period",
    "test_stage",
    "heldout_first_period",
    "heldout_last_period",
    "epochs",
    "window_months",
    "validation_periods",
    "min_region_trade_days",
    "max_steps_per_fold",
    "max_backtests_per_fold",
    "max_null_controls_per_fold",
    "max_llm_calls",
    "session_max_attempts",
    "max_fold_minutes",
    "min_return",
    "min_sharpe",
    "max_drawdown",
    "cost_stress_multiplier",
    "heldout_min_trades",
    "confirmation_folds",
    "deployment_adjustment_start",
    "deployment_max_backtests",
    "deployment_pit_views_seed",
    "meta_learning_fold_interval",
    "meta_memory_max_epochs",
    "inherit_from",
    "inherit_memory_from",
    "meta_learning_directive",
    "fold_exploration_directive",
    "workspace_reference",
    "operating_memory",
    "disable_step_tree",
    "record_failed_attempts",
    "convergence_start_epoch",
    "nl_failure_policy",
    "finalize_before_deadline_seconds",
    "per_call_timeout_seconds",
    "strategy_fit_timeout_seconds",
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
    "llm_env_file",
    "llm_model",
    "llm_timeout_seconds",
    "llm_max_retries",
    "llm_retry_backoff_seconds",
    "llm_temperature",
    "llm_max_response_tokens",
    "model",
    "meta_model",
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

# Historical snapshots may contain these former operator overrides. They are
# deliberately ignored rather than interpreted or exposed: provider endpoints
# and credentials now come only from the trusted model profile's fixed
# environment keys, so no experiment parameter can redirect either.
NON_PERSISTABLE_PARAMS = frozenset({"llm_api_key_env", "llm_base_url"})


@dataclass(frozen=True)
class LLMWorkerSettings:
    env_file: Path
    model: str
    meta_model: str
    # The ``agent`` sub-agents of both Fold and Meta sessions; they keep the
    # parent's call quota, budget wrapper and time budget on this gateway.
    subagent_model: str
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
    # The Fold parent conversation's compaction budget; ``compaction_for``
    # derives the other conversation roles' from the same knobs.
    compaction: ContextCompactionConfig
    # The console's ``compact_token_threshold``; None = derived per role.
    compact_token_threshold: int | None = None

    def model_for(self, role: str) -> str:
        """The model of a role this dataclass owns.

        ``analysis`` is deliberately absent: its model lives on
        ``InteractiveWorkerOptions``, and both call sites pass it explicitly.
        A default here could only be another role's model.
        """

        models = {
            "main": self.model,
            "meta": self.meta_model,
            "subagent": self.subagent_model,
            "nl": self.nl_model,
            "compact": self.compact_model,
        }
        if role not in models:
            raise ValueError(f"unknown model role: {role}")
        return models[role]

    def max_tokens_for(
        self,
        role: str,
        *,
        model: str | None = None,
        requested: int | None = None,
    ) -> int:
        selected_model = model or self.model_for(role)
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
        selected_model = model or self.model_for(role)
        effective_max_tokens = self.max_tokens_for(
            role, model=selected_model, requested=max_tokens
        )
        return build_model_gateway(
            selected_model,
            env_file=self.env_file,
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

    def compaction_for(self, role: str) -> ContextCompactionConfig:
        """One conversation role's compaction budget (main, meta or subagent).

        Its threshold is ``window − that role's output budget −
        COMPACTION_SAFETY_MARGIN_TOKENS`` for the role's own model, so prompt
        plus output never exceeds the window, and the same bound for the
        compaction model, which must read the whole conversation; a configured
        ``compact_token_threshold`` is clamped to them. The value reaches the
        run facts (parents) and the compaction events (children). Only a
        window with no room for the output budget at all is a launch error.
        """

        if role not in {"main", "meta", "subagent"}:
            raise ValueError(f"unknown conversation role: {role}")
        bounds = [self.compact_token_threshold] if self.compact_token_threshold else []
        for bound_role in (role, "compact") if self.compact_enabled else (role,):
            window = model_profile(self.model_for(bound_role)).context_window_tokens
            if window is None:
                continue
            bound = (
                window
                - self.max_tokens_for(bound_role)
                - COMPACTION_SAFETY_MARGIN_TOKENS
            )
            if bound <= 0:
                raise ValueError(
                    f"{bound_role} model output budget leaves no context capacity"
                )
            bounds.append(bound)
        if not bounds:
            raise ValueError(
                "compact_token_threshold is required when no model role declares "
                "a context window"
            )
        return replace(self.compaction, token_threshold=min(bounds))


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
    pit_views_seed: Path | None = None
    # An explicitly chosen seed must apply or the run fails; the default one is
    # an optimisation, so a missing or non-matching default cold-builds.
    pit_views_seed_required: bool = False
    # The tree the deployment adjustment takes its two views from when the
    # experiment's own views cannot be extended (docs/pipeline-design.md §3.4);
    # None falls back to an explicitly named pit_views_seed, else a cold build.
    deployment_pit_views_seed: Path | None = None
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

    def knob(name: str) -> object:
        """One rolling-config knob, defaulted from the dataclass.

        ``RollingExperimentConfig`` is the single source of truth for the
        pipeline defaults. A second literal here would silently reconfigure
        every experiment whose ``params.json`` predates the knob.
        """
        return params.get(name, rolling_default(name))

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
    if initial_control_mode not in CONTROL_MODES:
        raise ValueError(f"initial_control_mode must be one of {CONTROL_MODES}")
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
    pit_views_seed, pit_views_seed_required = (
        _pit_views_seed(params.get("pit_views_seed"), repository, snapshot_config)
        if data_backend == "pit"
        else (None, False)
    )
    deployment_seed = (
        _deployment_pit_views_seed(
            params.get("deployment_pit_views_seed"), repository, snapshot_config
        )
        if data_backend == "pit"
        else None
    )
    trading_days: list[str] = []
    if preflight:
        pass  # the release pin writes into the experiment dir; see the docstring
    elif data_backend == "daily":
        assert daily is not None
        frame = pd.read_parquet(daily, columns=["trade_date"])
        trading_days = sorted(set(frame["trade_date"].map(yyyymmdd).tolist()))
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
    fold_period = str(params.get("fold_period") or rolling_default("fold_period"))
    supplied_periods = [
        params.get("development_first_period"),
        params.get("development_last_period"),
        params.get("heldout_first_period"),
        params.get("heldout_last_period"),
    ]
    if not all(value not in (None, "") for value in supplied_periods):
        raise ValueError("all four Development/Held-out period fields are required")
    first_development, last_development, first_heldout, last_heldout = (
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
        development_first_period=first_development,
        development_last_period=last_development,
        heldout_first_period=first_heldout,
        heldout_last_period=last_heldout,
        fold_period=fold_period,
        test_stage=_strict_bool(knob("test_stage"), "test_stage"),
        epochs=_positive_int(knob("epochs"), "epochs"),
        window_months=_positive_int(knob("window_months"), "window_months"),
        validation_periods=_positive_int(
            knob("validation_periods"), "validation_periods"
        ),
        min_region_trade_days=_positive_int(
            knob("min_region_trade_days"), "min_region_trade_days"
        ),
        max_steps_per_fold=_positive_int(
            knob("max_steps_per_fold"), "max_steps_per_fold"
        ),
        max_backtests_per_fold=_positive_int(
            knob("max_backtests_per_fold"), "max_backtests_per_fold"
        ),
        max_null_controls_per_fold=_nonnegative_int(
            knob("max_null_controls_per_fold"), "max_null_controls_per_fold"
        ),
        max_llm_calls=_positive_int(knob("max_llm_calls"), "max_llm_calls"),
        session_max_attempts=_positive_int(
            knob("session_max_attempts"), "session_max_attempts"
        ),
        max_fold_minutes=_positive_int(knob("max_fold_minutes"), "max_fold_minutes"),
        meta_learning_fold_interval=_nonnegative_int(
            knob("meta_learning_fold_interval"), "meta_learning_fold_interval"
        ),
        meta_memory_max_epochs=_nonnegative_int(
            knob("meta_memory_max_epochs"), "meta_memory_max_epochs"
        ),
        meta_learning_directive=str(params.get("meta_learning_directive") or ""),
        fold_exploration_directive=str(
            params.get("fold_exploration_directive") or ""
        ),
        workspace_reference=_optional_workspace_reference(
            params.get("workspace_reference"), repository
        ),
        operating_memory=resolve_operating_memory(params.get("operating_memory")),
        step_tree_enabled=not _strict_bool(
            params.get("disable_step_tree", False), "disable_step_tree"
        ),
        record_failed_attempts=_strict_bool(
            knob("record_failed_attempts"), "record_failed_attempts"
        ),
        convergence_start_epoch=_positive_int(
            knob("convergence_start_epoch"), "convergence_start_epoch"
        ),
        deployment_adjustment_start=_deployment_start(
            knob("deployment_adjustment_start")
        ),
        deployment_max_backtests=_positive_int(
            knob("deployment_max_backtests"), "deployment_max_backtests"
        ),
        nl_failure_policy=_nl_failure_policy(knob("nl_failure_policy")),
        finalize_before_deadline_seconds=_nonnegative_int(
            knob("finalize_before_deadline_seconds"),
            "finalize_before_deadline_seconds",
        ),
        per_call_timeout_seconds=_positive_int(
            knob("per_call_timeout_seconds"), "per_call_timeout_seconds"
        ),
        strategy_fit_timeout_seconds=_positive_int(
            knob("strategy_fit_timeout_seconds"), "strategy_fit_timeout_seconds"
        ),
        meta_sandbox_rebuild_enabled=not _strict_bool(
            params.get("disable_meta_sandbox_rebuild", False),
            "disable_meta_sandbox_rebuild",
        ),
        meta_sandbox_rebuild_timeout_seconds=_nonnegative_int(
            knob("meta_sandbox_rebuild_timeout_seconds"),
            "meta_sandbox_rebuild_timeout_seconds",
        ),
        meta_sandbox_image_keep=_nonnegative_int(
            knob("meta_sandbox_image_keep"), "meta_sandbox_image_keep"
        ),
        acceptance=AcceptanceRules(
            min_return=_finite_float(params.get("min_return", 0.0), "min_return"),
            min_sharpe=_finite_float(params.get("min_sharpe", 0.0), "min_sharpe"),
            max_drawdown=_bounded_float(
                params.get("max_drawdown", 0.25), "max_drawdown", 0.0, 1.0
            ),
            cost_stress_multiplier=_finite_float(
                params.get("cost_stress_multiplier", 1.0), "cost_stress_multiplier"
            ),
            heldout_min_trades=_nonnegative_int(
                params.get("heldout_min_trades", 0), "heldout_min_trades"
            ),
            confirmation_folds=_nonnegative_int(
                params.get("confirmation_folds", AcceptanceRules().confirmation_folds),
                "confirmation_folds",
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
            rolling.development_first_period,
            rolling.development_last_period,
            trading_days,
            window_months=rolling.window_months,
            period=rolling.fold_period,
            min_region_trade_days=rolling.min_region_trade_days,
            test_stage=rolling.test_stage,
            validation_periods=rolling.validation_periods,
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
        pit_views_seed=pit_views_seed,
        pit_views_seed_required=pit_views_seed_required,
        deployment_pit_views_seed=deployment_seed,
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


def _strategy_sandbox_from_spec(
    spec: SandboxSpec | None, *, fit_timeout_seconds: float
) -> SandboxConfig:
    """The strategy container's boundary, derived from the Agent session's spec.

    The experiment's GPU request travels with it: ``fit(context)`` is where a
    model is trained, and it runs in the strategy container of every formal
    replay (Validation, parent control, frozen Test, Held-out), not in the
    session. The request is the experiment-level one — a per-session HITL
    ``sandbox_gpu_count`` override moves only that session's own container.
    The strategy container always uses the free-memory selector, so a spec that
    pins explicit device indexes is honoured as a device count, not as those
    exact devices.
    """

    if spec is None:
        return SandboxConfig(
            image=DEFAULT_IMAGE,
            limits=SandboxLimits(fit_timeout_seconds=float(fit_timeout_seconds)),
        )
    return SandboxConfig(
        image=spec.image,
        limits=SandboxLimits(
            fit_timeout_seconds=float(fit_timeout_seconds),
            gpu_count=spec.gpu_count if spec.gpu is not None else 0,
            gpu_name_filter=spec.gpu_name_filter,
        ),
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


class ExperimentPipelineBuild(NamedTuple):
    """One assembled experiment, as either driver of it needs to see it."""

    pipeline: RollingExperimentPipeline
    trading_days: list[str]
    meta_enabled: bool
    developer_label: str


def build_experiment_pipeline(
    options: InteractiveWorkerOptions,
    *,
    ledger: ExperimentLedger,
    store: FilesystemArtifactStore,
    ref_store: AgentRefStore,
    llm: LLMProxy | None = None,
    command_runner_factory: Callable[[Path], CommandRunner] | None = None,
) -> ExperimentPipelineBuild:
    """Assemble one experiment's providers, backends, Agents and pipeline.

    The single assembly for both drivers: the console's session loop
    (``run_local_interactive_worker``) and the single-session audit entrypoint
    (``scripts/experiments/run_audit_session.py``). A second hand-written
    assembly is a session configured differently from the one the console runs
    while reported as the same, so everything that shapes a session lives here
    -- the gateway roles and their retry policy, the snapshot provider and
    evaluator selection, the strategy sandbox wall clocks, and both Agent
    adapters.

    ``command_runner_factory`` replaces the Fold sandbox with a trusted
    in-process runner (the non-Docker test path) and, being sandboxless, also
    keeps the Meta session from shelling out to ``docker build``. The one step
    that is not shared is the worker's per-experiment derived image: it is
    applied to ``options`` before this call, because the audit entrypoint takes
    an explicit ``--sandbox-image`` instead.
    """

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
    subagent_gateway = llm or (
        options.llm.build_gateway("subagent")
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
    strategy_sandbox = _strategy_sandbox_from_spec(
        options.agent_sandbox,
        fit_timeout_seconds=options.rolling.strategy_fit_timeout_seconds,
    )
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
            pit_views_seed=options.pit_views_seed,
            pit_views_seed_required=options.pit_views_seed_required,
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
            subagent_llm=subagent_gateway,
            compact_llm=compact_gateway,
            context_compaction=options.llm.compaction,
            subagent_compaction=options.llm.compaction_for("subagent"),
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
            operating_memory=options.rolling.operating_memory,
            repo_root=options.repo_root,
        )
        meta_learner = LLMMetaLearner(
            llm=meta_gateway,
            subagent_llm=subagent_gateway,
            compact_llm=compact_gateway,
            context_compaction=options.llm.compaction_for("meta"),
            subagent_compaction=options.llm.compaction_for("subagent"),
            baseline_strategy=options.baseline_strategy,
            artifact_store=store,
            experiment_dir=options.experiment_dir,
            runtime_root=runtime_root,
            max_llm_calls=options.rolling.max_llm_calls,
            deadline_seconds=options.rolling.max_fold_minutes * 60,
            decision_timeout_seconds=strategy_sandbox.limits.timeout_seconds,
            fit_timeout_seconds=strategy_sandbox.limits.fit_timeout_seconds,
            strategy_gpu_count=strategy_sandbox.limits.gpu_count,
            max_response_tokens=options.llm.max_tokens_for("meta"),
            meta_learning_directive=options.rolling.meta_learning_directive,
            fold_exploration_directive=options.rolling.fold_exploration_directive,
            workspace_reference=options.rolling.workspace_reference,
            operating_memory=options.rolling.operating_memory,
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
    # Memory seeded at creation from another experiment: its skills are the
    # head until the first session row, and its provenance is what a Meta
    # reading the inherited PRIOR needs, for either driver of this assembly.
    inherited_memory = load_inherited_memory(options.experiment_dir)
    pipeline = RollingExperimentPipeline(
        options.rolling,
        snapshots=snapshots,
        artifacts=store,
        evaluator=evaluator,
        developer=developer,
        meta_learner=meta_learner,
        ledger=ledger,
        inherited_memory=inherited_memory,
    )
    return ExperimentPipelineBuild(
        pipeline=pipeline,
        trading_days=trading_days,
        meta_enabled=meta_enabled,
        developer_label=developer_label,
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
    # A run killed outright (SIGKILL, OOM kill, host reset) cannot append its
    # own attempt_failed record. Worker start is the only moment at which no run
    # of this experiment is in flight, so the markers those runs left behind
    # become their ledger evidence here, before any new session begins.
    RunMarkers(options.experiment_dir).recover(ledger)
    completed = read_status(hitl / "status.json")
    if str(completed.get("state")) == "completed" and not _has_outstanding_work(
        hitl, ledger, options.rolling
    ):
        # A finished experiment is terminal: every session and the held-out
        # evaluation are already durable in the ledger, so a resume must
        # republish the completion status instead of re-running anything.
        records = ledger.read()
        payload = _terminal_status(
            completed,
            verdict=experiment_verdict(records),
            paper_candidate=paper_candidate(records),
        )
        write_json_atomic(hitl / "status.json", payload)
        return payload
    if (
        str(completed.get("state")) == "completed"
        and ledger.read("heldout")
        and not _pending_rerun(hitl, ledger)
    ):
        # Only the post-seal deployment adjustment is outstanding (a crashed
        # attempt, or a request made after completion): nothing before the
        # Held-out may run again, so the worker goes straight to it.
        return _run_deployment_adjustment(
            options,
            ledger=ledger,
            store=store,
            ref_store=ref_store,
            llm=llm,
            command_runner_factory=command_runner_factory,
            poll_seconds=poll_seconds,
            heldout_runs=0,
        )
    # Development that already walked its whole plan without freezing anything
    # is terminal: a resume could only re-walk the finished plan and end at the
    # same failure, so republish it here instead -- before any snapshot,
    # sandbox or gateway preparation, and without re-running a single session.
    if _development_is_exhausted(hitl, ledger) and not _pending_rerun(hitl, ledger):
        # Nothing to deliver means no frozen artifact at all, or only the
        # baseline anchor the first Fold was forced to freeze -- a control, not
        # a candidate (docs/pipeline-design.md §2.2).
        in_force = _latest_artifact(
            ledger, store, options.experiment_dir
        ) or _load_inherited_parent(options.experiment_dir)
        if in_force is None or _is_baseline_anchor(ledger, in_force):
            _fail_without_frozen_artifact(hitl, ledger, artifact=in_force)
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
    pipeline, trading_days, meta_enabled, developer_label = build_experiment_pipeline(
        options,
        ledger=ledger,
        store=store,
        ref_store=ref_store,
        llm=llm,
        command_runner_factory=command_runner_factory,
    )
    folds = build_fold_schedule(
        options.rolling.development_first_period,
        options.rolling.development_last_period,
        trading_days,
        window_months=options.rolling.window_months,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
        test_stage=options.rolling.test_stage,
        validation_periods=options.rolling.validation_periods,
    )
    sessions, plan = _write_session_plan(
        options, hitl, folds, trading_days, meta_enabled=meta_enabled
    )
    # Inherited seeds (another experiment's frozen output, another's PRIOR and
    # skills) stand in for the blank template and the empty memory only until
    # this experiment's own ledger rows exist; a resumed experiment takes its
    # parent, PRIOR and skills from its own ledger instead.
    memory = load_inherited_memory(options.experiment_dir)
    _restore_prior_store(
        options.experiment_dir,
        ledger,
        fallback_generation_id=memory.prior_generation_id if memory else "",
    )
    # Skills have no mutable CURRENT pointer: validating the final remaining
    # successful Fold/Meta row is the complete resume/rollback restore step.
    latest_skills_snapshot(
        ledger.read(),
        experiment_dir=options.experiment_dir,
        inherited=memory.skills if memory else None,
    )
    state = {
        "parent": _latest_artifact(ledger, store, options.experiment_dir)
        or _load_inherited_parent(options.experiment_dir),
        "prior": latest_prior_text(ledger.read("meta_learning"))
        or (memory.prior_text if memory else ""),
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
        outcome = pipeline.run_fold(
            session.epoch_id,
            session.fold,
            parent=state["parent"],
            prior=str(state["prior"]),
            confirmation=session.confirmation,
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
    final = state["parent"] or _latest_artifact(ledger, store, options.experiment_dir)
    # A baseline anchor is the lineage's control, never the deliverable: an
    # experiment whose last word is the placebo it was forced to freeze takes
    # the same explicit exit as one that froze nothing at all, rather than
    # spending a Held-out on it (docs/pipeline-design.md §2.2).
    if final is None or _is_baseline_anchor(ledger, final):
        _fail_without_frozen_artifact(hitl, ledger, artifact=final)
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
            _heldout_epoch_id(ledger, options.rolling.epochs),
            final,
            trading_days,
            replay=bool(result.get("reran_sessions")),
        )
    finally:
        final_status.stop()
    records = ledger.read()
    payload = _terminal_status(
        {
            "completed_at": utc_now_iso(),
            "developer_mode": developer_label,
            "completed_sessions": completed_development + 1,
            "total_sessions": len(plan["sessions"]),
            "final_strategy_artifact": final.artifact_id,
        },
        developer_mode=developer_label,
        heldout_runs=heldout_runs,
        verdict=experiment_verdict(records),
        paper_candidate=paper_candidate(records),
    )
    write_json_atomic(hitl / "status.json", payload)
    if deployment_adjustment_due(
        records, start=options.rolling.deployment_adjustment_start
    ):
        return _run_deployment_adjustment(
            options,
            ledger=ledger,
            store=store,
            ref_store=ref_store,
            llm=llm,
            command_runner_factory=command_runner_factory,
            poll_seconds=poll_seconds,
            heldout_runs=heldout_runs,
        )
    return payload


def _write_session_plan(
    options: InteractiveWorkerOptions,
    hitl: Path,
    folds: list[FoldSpec],
    trading_days: list[str],
    *,
    meta_enabled: bool,
) -> tuple[tuple[DevelopmentSession, ...], dict[str, object]]:
    """The plan of record (``schedule.json``): development sessions, the
    Held-out, and the deployment adjustment when its start is configured."""
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
        confirmation_folds=options.rolling.acceptance.confirmation_folds,
    )
    plan = build_session_plan(
        options.rolling.epochs,
        folds,
        heldout,
        meta_enabled=meta_enabled,
        meta_learning_fold_interval=options.rolling.meta_learning_fold_interval,
        confirmation_folds=options.rolling.acceptance.confirmation_folds,
        deployment=_deployment_fold(options, trading_days),
    )
    write_json_atomic(hitl / SCHEDULE_NAME, plan)
    return sessions, plan


def _deployment_fold(
    options: InteractiveWorkerOptions, trading_days: list[str]
) -> FoldSpec | None:
    start = options.rolling.deployment_adjustment_start
    if not start:
        return None
    return deployment_fold(
        start,
        trading_days,
        window_months=options.rolling.window_months,
        min_region_trade_days=options.rolling.min_region_trade_days,
    )


def _run_deployment_adjustment(
    options: InteractiveWorkerOptions,
    *,
    ledger: ExperimentLedger,
    store: FilesystemArtifactStore,
    ref_store: AgentRefStore,
    llm: LLMProxy | None,
    command_runner_factory: Callable[[Path], CommandRunner] | None,
    poll_seconds: float,
    heldout_runs: int,
) -> dict[str, object]:
    """The post-Held-out deployment adjustment (docs/pipeline-design.md §3.4).

    Assembled on its own PIT cache root, ``pit_views/deployment``: the
    graduate's own views may have been built under an older cache format,
    which this code can neither extend nor read, so the session's two views
    are hardlinked there from the named seed (or cold-built when none is
    named) and the rest of the experiment's views stay as they are. The
    session runs through the interactive runner past the reveal, and the
    terminal status names the Paper candidate.
    """
    hitl = options.experiment_dir / "hitl"
    graduated = _latest_artifact(ledger, store, options.experiment_dir)
    if graduated is None:
        raise RuntimeError("the deployment adjustment needs the graduated artifact")
    scored = {
        str(row.get("strategy_artifact_id") or "")
        for row in latest_heldout_records(ledger.read())
    }
    if scored != {graduated.artifact_id}:
        raise RuntimeError(
            f"the Held-out scored {sorted(scored)}, not the ledger's latest "
            f"artifact {graduated.artifact_id}; refusing to adjust it"
        )
    if command_runner_factory is None and (
        options.execution_mode == "sandbox" or options.developer_mode == "llm"
    ):
        options = replace(
            options,
            agent_sandbox=prepare_experiment_sandbox_image(
                options.agent_sandbox or SandboxSpec(gpu=None),
                experiment_id=options.experiment_id,
                experiment_dir=options.experiment_dir,
            ),
        )
    deployment_options = replace(
        options,
        pit_cache_root=(
            options.experiment_dir / "pit_views" / "deployment"
            if options.data_backend == "pit"
            else None
        ),
        pit_views_seed=None,
        pit_views_seed_required=False,
    )
    pipeline, trading_days, meta_enabled, developer_label = build_experiment_pipeline(
        deployment_options,
        ledger=ledger,
        store=store,
        ref_store=ref_store,
        llm=llm,
        command_runner_factory=command_runner_factory,
    )
    folds = build_fold_schedule(
        options.rolling.development_first_period,
        options.rolling.development_last_period,
        trading_days,
        window_months=options.rolling.window_months,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
        test_stage=options.rolling.test_stage,
        validation_periods=options.rolling.validation_periods,
    )
    _sessions, plan = _write_session_plan(
        options, hitl, folds, trading_days, meta_enabled=meta_enabled
    )
    fold = _deployment_fold(options, trading_days)
    assert fold is not None  # the caller checked deployment_adjustment_due
    seed = options.deployment_pit_views_seed or (
        options.pit_views_seed if options.pit_views_seed_required else None
    )
    if seed is not None and options.data_backend == "pit":
        pipeline.snapshots.link_seed_slots(
            seed,
            phase="valid",
            start=fold.validation_start,
            end=fold.validation_end,
            decision_time=fold.valid_decision_time,
        )
    memory = load_inherited_memory(options.experiment_dir)
    prior = latest_prior_text(ledger.read("meta_learning")) or (
        memory.prior_text if memory else ""
    )
    session = DevelopmentSession(
        DEPLOYMENT_SESSION_KEY,
        "deployment_adjustment",
        _heldout_epoch_id(ledger, options.rolling.epochs),
        fold,
    )

    def execute(session: DevelopmentSession, context: dict[str, object]) -> None:
        assert session.fold is not None
        pipeline.run_deployment_adjustment(
            session.epoch_id,
            session.fold,
            graduated=graduated,
            prior=prior,
            session_context=context,
        )

    interactive = InteractiveExperimentRunner(
        experiment_id=options.experiment_id,
        sessions=(session,),
        execute_session=execute,
        ledger=ledger,
        control_path=hitl / "control.json",
        status_path=hitl / "status.json",
        ref_store=ref_store,
        poll_seconds=poll_seconds,
        session_max_attempts=options.rolling.session_max_attempts,
        after_reveal=True,
    )
    result = interactive.run()
    if result["status"] != "complete":
        return result
    records = ledger.read()
    payload = _terminal_status(
        {
            "completed_at": utc_now_iso(),
            "developer_mode": developer_label,
            "completed_sessions": len(plan["sessions"]),
            "total_sessions": len(plan["sessions"]),
            "final_strategy_artifact": graduated.artifact_id,
        },
        developer_mode=developer_label,
        heldout_runs=heldout_runs,
        verdict=experiment_verdict(records),
        paper_candidate=paper_candidate(records),
    )
    write_json_atomic(hitl / "status.json", payload)
    return payload


def _heldout_epoch_id(ledger: ExperimentLedger, configured_epochs: int) -> str:
    """Epoch that graduation term (b) is scored on: the last one that actually
    produced a Fold. Development can stop before the configured last Epoch (the
    console's skip-to-Held-out), and an Epoch with no fold record has no
    walk-forward transition, which would waive the requirement instead of
    failing it. Without any fold record the configured last Epoch is the only
    answer available, and it is equally empty."""
    epochs = {epoch for epoch, _ in latest_fold_records(ledger.read("fold"))}
    return max(epochs) if epochs else epoch_ids(configured_epochs)[-1]


def _artifact_from_record(
    artifact_id: str,
    recorded_path: str,
    record: Mapping[str, object],
    *,
    store: FilesystemArtifactStore,
    experiment_dir: Path,
) -> FrozenArtifact:
    """Rebuild the artifact a ledger record names, from the path it recorded.

    ``frozen/`` holds only what this experiment froze itself, so a Fold that
    kept its parent records the console-installed seed's own ``_inherited/``
    tree (docs/pipeline-design.md §3.1) and deriving ``frozen/<id>`` from the
    id would look for that seed where it never lives. The recorded path decides
    which tree is loaded; each tree keeps the validator that owns it -- the
    store's manifest and immutability check for this experiment's own freezes,
    the seed's read-only snapshot check for the inherited one.
    """
    if recorded_path and not Path(recorded_path).resolve().is_relative_to(
        store.frozen_root.resolve()
    ):
        inherited = _load_inherited_parent(experiment_dir)
        if (
            inherited is None
            or inherited.artifact_id != artifact_id
            or inherited.path.resolve() != Path(recorded_path).resolve()
        ):
            raise RuntimeError(
                "the recorded path is neither this experiment's own frozen "
                f"artifact nor its inherited seed: {recorded_path}"
            )
        return inherited
    frozen = store.frozen(
        artifact_id,
        expected_path=recorded_path or None,
        experiment_id=str(record.get("experiment_id") or ""),
    )
    return FrozenArtifact(
        artifact_id,
        Path(frozen.path),
        Path(frozen.model_path) if frozen.model_path is not None else None,
        str(frozen.source_run_id),
        str(frozen.source_fold_id),
        str(frozen.source_step_id),
        str(frozen.revision_id),
    )


def _latest_artifact(
    ledger: ExperimentLedger,
    store: FilesystemArtifactStore,
    experiment_dir: Path,
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
        artifact = _artifact_from_record(
            current_id,
            current_path,
            current_record,
            store=store,
            experiment_dir=experiment_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"ledger artifact failed validation: {current_id}: {exc}"
        ) from exc
    return replace(artifact, requires_validation=requires_validation)


def _terminal_status(
    source: Mapping[str, object],
    *,
    developer_mode: str | None = None,
    heldout_runs: int = 0,
    verdict: Mapping[str, object] | None = None,
    paper_candidate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Durable completion status. The experiment's own evidence stays in the
    append-only ledger; status.json only records that the run reached its end,
    so a resume republishes it without re-running or re-recording anything
    (``heldout_runs`` counts what THIS invocation executed). ``verdict`` is the
    ledger-derived graduation verdict and ``paper_candidate`` the artifact
    Paper pins (``ledger.paper_candidate``), both re-read on every publish."""
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
        "verdict": dict(verdict) if verdict is not None else None,
        "paper_candidate": dict(paper_candidate) if paper_candidate is not None else None,
    }


def _has_outstanding_work(
    hitl: Path, ledger: ExperimentLedger, rolling: RollingExperimentConfig
) -> bool:
    """Whether a completed experiment still has console-requested work.

    A rollback drops every held-out record (the frontier moved back), a
    rerun request leaves a token no fold record has absorbed yet, and a
    graduated experiment whose deployment adjustment is configured but not
    recorded still owes that session. In each case the worker must resume
    instead of republishing the terminal status, or the console operation
    would look accepted and silently do nothing."""
    records = ledger.read()
    if not any(record.get("record_type") == "heldout" for record in records):
        return True
    if deployment_adjustment_due(records, start=rolling.deployment_adjustment_start):
        return True
    return _pending_rerun(hitl, ledger)


def _pending_rerun(hitl: Path, ledger: ExperimentLedger) -> bool:
    """Whether a console re-run request is still waiting for its fold record."""
    pending = read_control(hitl / "control.json").rerun_sessions
    if not pending:
        return False
    records = ledger.read("fold")
    return any(
        not rerun_absorbed(records, key, token) for key, token in pending.items()
    )


def _development_is_exhausted(hitl: Path, ledger: ExperimentLedger) -> bool:
    """Whether every planned development session already has a durable record.

    The plan of record is the worker's own ``schedule.json``, the same plan the
    console resolves its session operations against, so the question is settled
    without rebuilding the calendar or touching any data."""
    sessions = read_json(hitl / SCHEDULE_NAME).get("sessions")
    if not isinstance(sessions, list):
        return False
    planned = {
        str(item.get("session_key") or "")
        for item in sessions
        if isinstance(item, dict) and item.get("kind") in {"fold", "meta"}
    }
    if not planned or "" in planned:
        return False
    completed = {
        str(record.get("session_key") or "")
        for record in ledger.read()
        if is_durable_success_record(record, record_types=("fold", "meta_learning"))
    }
    return planned <= completed


def _is_baseline_anchor(ledger: ExperimentLedger, artifact: FrozenArtifact) -> bool:
    """Whether the artifact in force is the lineage's baseline anchor."""

    return artifact.artifact_id in baseline_anchor_artifacts(ledger.read("fold"))


def _fail_without_frozen_artifact(
    hitl: Path, ledger: ExperimentLedger, *, artifact: FrozenArtifact | None = None
) -> NoReturn:
    """The documented zero-deliverable end of development (§4.3): development
    ran out of folds without ever freezing an artifact worth evaluating, so
    there is nothing to evaluate and the run fails. ``artifact`` is the
    baseline anchor left in force, when that is why there is nothing to
    deliver. The status is published here as well as raised so the terminal
    state is the same whichever caller drove the worker."""
    error = (
        _baseline_anchor_end_error(artifact)
        if artifact is not None
        else _development_end_error(ledger)
    )
    write_json_atomic(
        hitl / "status.json",
        {
            "schema_version": 1,
            "state": "failed",
            "pid": os.getpid(),
            "failed_at": utc_now_iso(),
            "error": f"RuntimeError: {error}",
        },
    )
    raise RuntimeError(error)


def _baseline_anchor_end_error(artifact: FrozenArtifact) -> str:
    """Name the anchor, so the failure is not read as a lost artifact.

    The experiment did freeze something; what it never did is replace the weak
    baseline the anchor rule forced on its first Fold with a candidate anyone
    judged worth shipping."""

    return (
        "Development ended with a baseline anchor in force "
        f"({artifact.artifact_id}): an anchor is the lineage's control, not a "
        "deliverable, and no candidate ever replaced it, so there is nothing "
        "to evaluate on Held-out"
    )


def _development_end_error(ledger: ExperimentLedger) -> str:
    """Name the research cause the ledger already holds. The bare sentence left
    the reader to reconstruct by hand whether every Fold abstained, every
    nomination was hard-rejected, or no session ever nominated anything."""
    folds = latest_fold_records(ledger.read("fold"))
    outcomes = Counter(
        (
            str(record.get("finish_mode") or "unknown"),
            str(record.get("fold_status") or "unknown"),
        )
        for record in folds.values()
    )
    detail = (
        ", ".join(
            f"{count}/{len(folds)} folds ended {mode} ({status})"
            for (mode, status), count in outcomes.most_common()
        )
        if outcomes
        else "no fold recorded a result"
    )
    return f"Development completed without a frozen baseline artifact: {detail}"


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


def parent_from_step_node(
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
        strategy_path = Path(str(strategy_dir))
        # A frozen artifact is ``frozen/<id>/output`` plus a sibling
        # ``frozen/<id>/models`` (docs/pipeline-design.md §2.3); the ledger
        # records only the output path, so derive the models one from it.
        # analyze_fold lists the directory only when it exists.
        analyze_fold(
            proxy,
            ledger_record=record,
            ref_store=ref_store,
            strategy_dir=strategy_path,
            model_dir=strategy_path.parent / "models",
            out_dir=out_dir,
            max_tokens=effective_analysis_max_tokens,
            output_identity=(
                str(record.get("epoch_id") or "epoch_unknown"),
                ref_store.get_or_create("fold", str(record.get("fold_id") or "fold_unknown")),
            ),
        )

    return post_fold_hook


def _restore_prior_store(
    experiment_dir: Path, ledger: ExperimentLedger, *, fallback_generation_id: str = ""
) -> None:
    """Align CURRENT with the last remaining Meta generation after resume/rollback,
    or with the inherited generation before the first Meta."""
    restore_current_from_records(
        experiment_dir,
        ledger.read("meta_learning"),
        fallback_generation_id=fallback_generation_id,
    )


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

    def datasets(name: str, is_enabled: bool, domain: str) -> tuple[str, ...]:
        # Selecting nothing means this domain's default scope; an explicit
        # selection may name ANY dataset the domain can load, which is how an
        # experiment opts into a series the default scope leaves out.
        selected = selection(name, SELECTABLE_DATASETS[domain])
        return (selected or DEFAULT_DATASETS[domain]) if is_enabled else ()

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
        window_months=_positive_int(
            params.get("window_months", rolling_default("window_months")), "window_months"
        ),
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
            "fundamental_datasets", include_fundamentals, "fundamentals"
        ),
        macro_datasets=datasets("macro_datasets", include_macro, "macro"),
        events_datasets=datasets("events_datasets", include_events, "events"),
        text_datasets=datasets("text_datasets", include_text, "text"),
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
    env_file = _path_value(
        params.get("llm_env_file", ".env"), repository, "llm_env_file"
    )
    if repository != env_file and repository not in env_file.parents:
        raise ValueError("llm_env_file must stay inside the repository")
    fold_model = canonicalize_model_name(
        str(params.get("model") or params.get("llm_model") or LOCAL_QWEN_MODEL)
    )
    meta_model = canonicalize_model_name(str(params.get("meta_model") or fold_model))
    subagent_model = canonicalize_model_name(
        str(params.get("subagent_model") or fold_model)
    )
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
    # None = derived from the model windows; ``compaction_for`` sets the
    # effective per-role value either way.
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
        env_file=env_file,
        model=fold_model,
        meta_model=meta_model,
        subagent_model=subagent_model,
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
        compact_token_threshold=configured_threshold,
    )
    # Whether the host holds a credential is deployment state, not a property
    # of a WebUI create request.  Every model/role combination is still
    # validated at preflight with a non-secret placeholder.
    for role in ("main", "meta", "subagent", "nl", "compact"):
        settings.build_gateway(role, require_credentials=not preflight)
    # Every conversation role must leave room for its output budget; the Fold
    # parent's derived budget becomes the settings' own.
    for role in ("meta", "subagent"):
        settings.compaction_for(role)
    settings = replace(settings, compaction=settings.compaction_for("main"))
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


def _pit_views_seed(
    value: object, repo_root: Path, snapshot_config: SnapshotConfig
) -> tuple[Path, bool]:
    """The PIT view seed this experiment hardlinks from, and whether it must apply.

    The default seed is an optimisation over cold-building: it carries the
    default dataset selection, and an experiment that asks for anything else
    simply finds no matching contract and builds its own views. A seed named
    explicitly is a decision — the only way an arm whose selection differs from
    the default gets prebuilt views at all — so it is checked here, at create
    time: the tree must exist and must already carry exactly this snapshot
    configuration. Silently cold-building instead would cost hours and look
    like a slow experiment rather than a wrong parameter.
    """

    default = repo_root / DEFAULT_PIT_VIEWS_SEED
    if value in (None, ""):
        return default, False
    if not isinstance(value, str):
        raise ValueError("pit_views_seed must be a string")  # noqa: TRY004
    seed = _repo_path(repo_root, value.strip(), "pit_views_seed")
    if seed == default:
        return default, False
    if not seed.is_dir() or seed.is_symlink():
        raise ValueError(f"pit_views_seed must be an existing directory: {seed}")
    assert_seed_snapshot_config(seed, snapshot_config)
    return seed, True


def _deployment_pit_views_seed(
    value: object, repo_root: Path, snapshot_config: SnapshotConfig
) -> Path | None:
    """The seed the deployment adjustment links its two views from, if named.

    Checked like an explicit ``pit_views_seed``: the tree must exist and carry
    this experiment's snapshot configuration under the current cache format,
    because the two slots are read by this code, whatever format the
    experiment's own views were built under.
    """

    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("deployment_pit_views_seed must be a string")  # noqa: TRY004
    seed = _repo_path(repo_root, value.strip(), "deployment_pit_views_seed")
    if not seed.is_dir() or seed.is_symlink():
        raise ValueError(
            f"deployment_pit_views_seed must be an existing directory: {seed}"
        )
    assert_seed_snapshot_config(seed, snapshot_config)
    return seed


def _deployment_start(value: object) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ValueError("deployment_adjustment_start must be a YYYYMMDD string")  # noqa: TRY004
    return yyyymmdd(value.strip())


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


__all__ = [
    "ExperimentPipelineBuild",
    "InteractiveWorkerOptions",
    "LLMWorkerSettings",
    "build_experiment_pipeline",
    "load_worker_options",
    "parent_from_step_node",
    "run_local_interactive_worker",
]
