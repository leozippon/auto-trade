"""Strict local assembly for persistent interactive daily experiments."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from autotrade.agent.compact import ContextCompactionConfig
from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
)
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.contracts import benchmark_index_label
from autotrade.environment.data.research_release import (
    pin_research_release,
    published_research_release,
    require_benchmark_index,
)
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
    experiment_container_labels,
)
from autotrade.environment.sandbox_images import prepare_experiment_sandbox_image
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.tools.base import CommandRunner

from .calendar import (
    GEOMETRY_PARAMETERS,
    ResearchGeometry,
    load_sse_trading_days,
    yyyymmdd,
)
from .config import (
    DEFAULT_PIT_VIEWS_SEED,
    AcceptanceRules,
    RollingExperimentConfig,
    acceptance_for,
    rolling_default,
)
from .experiment import RollingExperimentPipeline
from .hitl_state import (
    CONTROL_MODES,
    SCHEDULE_NAME,
    WEB_INTERNAL_PARAMS,
    PlannedSession,
    build_session_plan,
    planned_sessions,
    read_json,
    read_status,
)
from .interactive import InteractiveExperimentRunner
from .ledger import (
    FOLD_ERA_RECORD_TYPES,
    ExperimentLedger,
    FrozenArtifactMutated,
    RunMarkers,
    assert_no_frozen_artifact_mutation,
    experiment_verdict,
    frozen_record,
    paper_candidate,
    research_records,
)
from .local_backend import (
    DeterministicBaselineDeveloper,
    LLMResearchDeveloper,
    LocalDailyEvaluationBackend,
    LocalDailySnapshotProvider,
)
from .pit_backend import (
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    required_release_raw_datasets,
)
from .pit_views_seed import assert_seed_snapshot_config
from .skills import latest_skills_snapshot, resolve_operating_memory

# Knobs no longer read by anything. They stay accepted, and only accepted, so
# every experiment created before their removal keeps launching and keeps
# listing: rejecting a key the console itself wrote would make those arms
# unreadable. The acceptance targets set warnings no run ever recorded.
RETIRED_PARAMS = ("min_return", "min_sharpe")

_ALLOWED_PARAMS = {
    "experiment_id",
    *GEOMETRY_PARAMETERS,
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
    "benchmark_index",
    "window_months",
    "max_replay_years",
    "max_null_controls",
    "max_llm_calls",
    "session_max_attempts",
    "max_research_minutes",
    "max_drawdown",
    "active_max_drawdown",
    "tracking_error_cap",
    "beta_min",
    "beta_max",
    "cost_stress_multiplier",
    "min_active_ir",
    "min_dsr_probability",
    "min_positive_year_share",
    "min_full_span_validations",
    "forward_confidence",
    "recency_months",
    "min_mean_gross",
    "min_round_trips_per_month",
    "heldout_tolerance_z",
    "research_directive",
    "workspace_reference",
    "operating_memory",
    "record_failed_attempts",
    "nl_failure_policy",
    "finalize_before_deadline_seconds",
    "per_call_timeout_seconds",
    "strategy_fit_timeout_seconds",
    "commission_bps",
    "slippage_bps",
    "max_total_holdings",
    "max_single_name_weight",
    "gpu_count",
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
    *RETIRED_PARAMS,
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
    # The ``agent`` sub-agents of research sessions; they keep the parent's
    # call quota, budget wrapper and time budget on this gateway.
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
    # The session parent conversation's compaction budget; ``compaction_for``
    # derives the other conversation roles' from the same knobs.
    compaction: ContextCompactionConfig
    # The console's ``compact_token_threshold``; None = derived per role.
    compact_token_threshold: int | None = None

    def model_for(self, role: str) -> str:
        """The model of one gateway role."""

        models = {
            "main": self.model,
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
        if role == "nl":
            # NL extracts evidence from already-retrieved PIT text and answers
            # short or enum-bounded questions; it is also the only LLM
            # inference inside a backtest's wall clock, so it runs at a
            # deliberately lower effort than the strategy-design dialogues.
            return NL_REASONING_EFFORT
        return self.reasoning_effort

    def compaction_for(self, role: str) -> ContextCompactionConfig:
        """One conversation role's compaction budget (main or subagent).

        Its threshold is ``window − that role's output budget −
        COMPACTION_SAFETY_MARGIN_TOKENS`` for the role's own model, so prompt
        plus output never exceeds the window, and the same bound for the
        compaction model, which must read the whole conversation; a configured
        ``compact_token_threshold`` is clamped to them. The value reaches the
        run facts (parents) and the compaction events (children). Only a
        window with no room for the output budget at all is a launch error.
        """

        if role not in {"main", "subagent"}:
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
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
    nl_config: NLConfig = field(default_factory=NLConfig)
    max_intraday_row_group_rows: int = 2_000_000
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
    and the immutable research-release pin (which materialises state inside the
    experiment directory) is skipped. An experiment naming a PIT view seed pins
    the release that seed was built from, which is already published, so the
    pre-flight still reads that release and checks it reaches into Held-out;
    without a named seed the release is not known until the pin, and that
    check waits for it. Every parameter check runs either way.
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
    # The arm's benchmark: refused here if it is not an index the lake carries,
    # and checked against the pinned release's partitions below.
    benchmark_index = str(knob("benchmark_index"))
    benchmark_index_label(benchmark_index)
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
    pit_views_seed, seed_release = (
        _pit_views_seed(params.get("pit_views_seed"), repository, snapshot_config)
        if data_backend == "pit"
        else (None, None)
    )
    default_geometry = rolling_default("geometry")
    geometry = ResearchGeometry(
        **{
            name: params.get(name, getattr(default_geometry, name))
            for name in GEOMETRY_PARAMETERS
        }
    )
    # None: a pre-flight without a named seed reads no data (see the docstring).
    trading_days: list[str] | None = None
    if data_backend == "daily":
        if not preflight:
            assert daily is not None
            frame = pd.read_parquet(daily, columns=["trade_date"])
            trading_days = sorted(set(frame["trade_date"].map(yyyymmdd).tolist()))
    else:
        assert (
            raw_dir is not None
            and events_root is not None
            and events_status is not None
        )
        required_raw_datasets = required_release_raw_datasets(snapshot_config)
        release_raw_dir: Path | None = None
        if seed_release is not None:
            assert pit_views_seed is not None
            release_raw_dir, trading_days = _seed_release_trading_days(
                pit_views_seed,
                seed_release,
                experiment_dir=directory,
                raw_dir=raw_dir,
                fundamental_events_root=events_root,
                fundamental_events_status=events_status,
                required_raw_datasets=required_raw_datasets,
                preflight=preflight,
            )
        elif not preflight:
            release = pin_research_release(
                experiment_dir=directory,
                raw_dir=raw_dir,
                fundamental_events_root=events_root,
                fundamental_events_status=events_status,
                required_raw_datasets=required_raw_datasets,
            )
            release_raw_dir = release.raw_dir
            trading_days = load_sse_trading_days(release.raw_dir)
        # The benchmark the arm will be graded against has to exist in the
        # release it pins, from before the research period on, and a create
        # request is the last place that can say so: by replay end the missing
        # series is only an unmeasured verdict.
        if release_raw_dir is not None:
            require_benchmark_index(
                release_raw_dir,
                benchmark_index,
                datasets=required_raw_datasets,
                research_start=geometry.research_start,
            )
    if trading_days is not None and not trading_days:
        raise ValueError("daily Parquet has no trading days")
    if trading_days is not None:
        # The release must reach into Held-out; the forward replay clips the
        # Held-out slot to its last trading day.
        geometry.heldout(trading_days)
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
        geometry=geometry,
        benchmark_index=benchmark_index,
        window_months=_positive_int(knob("window_months"), "window_months"),
        max_replay_years=_positive_int(knob("max_replay_years"), "max_replay_years"),
        max_null_controls=_nonnegative_int(
            knob("max_null_controls"), "max_null_controls"
        ),
        max_llm_calls=_positive_int(knob("max_llm_calls"), "max_llm_calls"),
        session_max_attempts=_positive_int(
            knob("session_max_attempts"), "session_max_attempts"
        ),
        max_research_minutes=_positive_int(
            knob("max_research_minutes"), "max_research_minutes"
        ),
        research_directive=str(
            params.get("research_directive") or ""
        ),
        workspace_reference=_optional_workspace_reference(
            params.get("workspace_reference"), repository
        ),
        operating_memory=resolve_operating_memory(params.get("operating_memory")),
        record_failed_attempts=_strict_bool(
            knob("record_failed_attempts"), "record_failed_attempts"
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
        # A tracking mandate exactly where the request named a cap; every
        # limit absent or null takes its default (``config.acceptance_for``).
        acceptance=acceptance_for(
            {
                name: None if params.get(name) is None else _finite_float(params[name], name)
                for name in AcceptanceRules().to_record()
            }
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
        pit_views_seed_required=seed_release is not None,
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
        max_intraday_row_group_rows=_positive_int(
            params.get("max_intraday_row_group_rows", 2_000_000),
            "max_intraday_row_group_rows",
        ),
        llm=llm_settings,
        agent_sandbox=sandbox_spec,
    )


def _strategy_sandbox_from_spec(
    spec: SandboxSpec | None,
    *,
    fit_timeout_seconds: float,
    experiment_id: str | None = None,
) -> SandboxConfig:
    """The strategy container's boundary, derived from the Agent session's spec.

    ``experiment_id`` labels every strategy container the experiment starts
    like its session container, so the console reclaims them together.

    The experiment's GPU request travels with it: ``fit(context)`` is where a
    model is trained, and it runs in the strategy container of every formal
    replay (validation and forward alike), not in the session. The request is the experiment-level one — a per-session HITL
    ``sandbox_gpu_count`` override moves only that session's own container.
    The strategy container always uses the free-memory selector, so a spec that
    pins explicit device indexes is honoured as a device count, not as those
    exact devices.
    """

    labels = experiment_container_labels(experiment_id) if experiment_id else {}
    if spec is None:
        return SandboxConfig(
            image=DEFAULT_IMAGE,
            limits=SandboxLimits(fit_timeout_seconds=float(fit_timeout_seconds)),
            labels=labels,
        )
    return SandboxConfig(
        image=spec.image,
        limits=SandboxLimits(
            fit_timeout_seconds=float(fit_timeout_seconds),
            gpu_count=spec.gpu_count if spec.gpu is not None else 0,
            gpu_name_filter=spec.gpu_name_filter,
        ),
        docker_executable=spec.docker_executable,
        labels=labels,
    )


class ExperimentPipelineBuild(NamedTuple):
    """One assembled experiment, as either driver of it needs to see it."""

    pipeline: RollingExperimentPipeline
    trading_days: list[str]
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
    """Assemble one experiment's providers, backends, Agent and pipeline.

    The single assembly for both drivers: the console's session loop
    (``run_local_interactive_worker``) and the single-session audit entrypoint
    (``scripts/experiments/run_audit_session.py``). A second hand-written
    assembly is a session configured differently from the one the console runs
    while reported as the same, so everything that shapes a session lives here
    -- the gateway roles and their retry policy, the snapshot provider and
    evaluator selection, the strategy sandbox wall clocks, and the Agent
    adapter.

    ``command_runner_factory`` replaces the session sandbox with a trusted
    in-process runner (the non-Docker test path). The one step that is not
    shared is the worker's per-experiment image: it is applied to ``options``
    before this call, because the audit entrypoint takes an explicit
    ``--sandbox-image`` instead.
    """

    agent = options.developer_mode == "llm" and options.llm is not None
    main_gateway = llm or (options.llm.build_gateway("main") if agent else None)
    subagent_gateway = llm or (options.llm.build_gateway("subagent") if agent else None)
    nl_gateway = llm or (options.llm.build_gateway("nl") if agent else None)
    # A failed compaction falls through to the emergency fit path by design;
    # provider retries would only add their full latency to that failure.
    compact_gateway = llm or (
        options.llm.build_gateway("compact", max_retries=0)
        if agent and options.llm.compact_enabled
        else None
    )
    strategy_sandbox = _strategy_sandbox_from_spec(
        options.agent_sandbox,
        fit_timeout_seconds=options.rolling.strategy_fit_timeout_seconds,
        experiment_id=options.experiment_id,
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
            benchmark_index=options.rolling.benchmark_index,
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
            benchmark_index=options.rolling.benchmark_index,
            sandbox=strategy_sandbox,
        )
        trading_days = evaluator.trading_days
    if options.developer_mode == "llm":
        if options.llm is None or options.agent_sandbox is None:
            raise ValueError(
                "developer_mode=llm is missing validated LLM or sandbox settings"
            )
        if main_gateway is None:
            raise ValueError("developer_mode=llm requires an initialized LLM gateway")
        developer = LLMResearchDeveloper(
            llm=main_gateway,
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
            runtime_root=options.work_root / options.experiment_id,
            sandbox_spec=options.agent_sandbox,
            command_runner_factory=command_runner_factory,
            # One ceiling for the parent conversation and its children.
            max_response_tokens=options.llm.max_tokens_for("main"),
            research_directive=options.rolling.research_directive,
            workspace_reference=options.rolling.workspace_reference,
            operating_memory=options.rolling.operating_memory,
            repo_root=options.repo_root,
        )
        developer_label = "llm_research_agent"
    else:
        developer = DeterministicBaselineDeveloper(
            baseline_strategy=options.baseline_strategy,
            artifact_store=store,
            evaluator=evaluator,
            schedule=options.rolling.schedule,
            broker_profile=options.rolling.broker_profile,
            ref_store=ref_store,
        )
        developer_label = "deterministic_baseline_no_agent_improvement"
    pipeline = RollingExperimentPipeline(
        options.rolling,
        snapshots=snapshots,
        artifacts=store,
        evaluator=evaluator,
        developer=developer,
        trading_days=trading_days,
        ledger=ledger,
    )
    return ExperimentPipelineBuild(
        pipeline=pipeline,
        trading_days=trading_days,
        developer_label=developer_label,
    )


def run_local_interactive_worker(
    options: InteractiveWorkerOptions,
    *,
    llm: LLMProxy | None = None,
    command_runner_factory: Callable[[Path], CommandRunner] | None = None,
    poll_seconds: float = 2.0,
) -> dict[str, object]:
    """Run the arm from wherever its ledger ends: the research session, then
    the forward replay once an artifact froze, then the terminal status."""

    ref_store = AgentRefStore(options.experiment_dir)
    hitl = options.experiment_dir / "hitl"
    ledger = ExperimentLedger(options.rolling.ledger_path)
    store = FilesystemArtifactStore(options.experiment_dir / "artifacts" / "strategy")
    records = ledger.read()
    fold_era = sorted(
        {str(record.get("record_type")) for record in records}
        & set(FOLD_ERA_RECORD_TYPES)
    )
    if fold_era:
        raise ValueError(
            f"{options.experiment_id} holds a Fold-era ledger ({', '.join(fold_era)} "
            "records); this pipeline does not resume it"
        )
    try:
        assert_no_frozen_artifact_mutation(records)
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
    if experiment_verdict(ledger.read()) is not None:
        # A finished arm is terminal: its verdict is durable, so a resume
        # republishes the completion status instead of re-running anything,
        # before any snapshot, sandbox or gateway preparation.
        payload = _terminal_status(ledger, read_status(hitl / "status.json"))
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
    pipeline, trading_days, developer_label = build_experiment_pipeline(
        options,
        ledger=ledger,
        store=store,
        ref_store=ref_store,
        llm=llm,
        command_runner_factory=command_runner_factory,
    )
    sessions = _write_session_plan(options, hitl, trading_days)
    # Skills have no mutable pointer: validating the recorded generation is
    # their whole restore step.
    latest_skills_snapshot(ledger.read(), experiment_dir=options.experiment_dir)

    def execute(session: PlannedSession, context: dict[str, object]) -> None:
        if session.kind == "research":
            pipeline.run_research_session(session_context=context)
        else:
            pipeline.run_forward(session_context=context)

    interactive = InteractiveExperimentRunner(
        experiment_id=options.experiment_id,
        sessions=sessions,
        execute_session=execute,
        ledger=ledger,
        control_path=hitl / "control.json",
        status_path=hitl / "status.json",
        ref_store=ref_store,
        poll_seconds=poll_seconds,
        session_max_attempts=options.rolling.session_max_attempts,
    )
    result = interactive.run()
    if result["status"] != "complete":
        return result
    payload = _terminal_status(
        ledger, {"completed_at": utc_now_iso()}, developer_mode=developer_label
    )
    write_json_atomic(hitl / "status.json", payload)
    return payload


def _write_session_plan(
    options: InteractiveWorkerOptions, hitl: Path, trading_days: list[str]
) -> tuple[PlannedSession, ...]:
    """Write the plan of record (``schedule.json``) and return its sessions.

    The forward entry states the replay span, the Held-out end clipped to the
    release; ``geometry.heldout`` refuses a release that does not reach it.
    """

    geometry = options.rolling.geometry
    heldout = geometry.heldout(trading_days)
    plan = build_session_plan(
        forward={
            "start": geometry.forward_start,
            "forward_end": geometry.forward_end,
            "heldout_start": heldout.start,
            "replay_end": heldout.end,
            "requested_end": heldout.requested_end,
            "truncation_reason": heldout.truncation_reason,
        },
    )
    write_json_atomic(hitl / SCHEDULE_NAME, plan)
    return planned_sessions()


def _terminal_status(
    ledger: ExperimentLedger,
    source: Mapping[str, object],
    *,
    developer_mode: str | None = None,
) -> dict[str, object]:
    """Durable completion status. The arm's evidence stays in the append-only
    ledger; status.json only records that the run reached its end, so a resume
    republishes it without re-running or re-recording anything. ``verdict`` and
    ``paper_candidate`` are re-read from the ledger on every publish."""

    records = ledger.read()
    frozen = frozen_record(records)
    return {
        "schema_version": 1,
        "state": "completed",
        "pid": os.getpid(),
        "completed_at": source.get("completed_at") or utc_now_iso(),
        "developer_mode": developer_mode or source.get("developer_mode"),
        "completed_sessions": len(research_records(records))
        + sum(1 for record in records if record.get("record_type") == "forward"),
        "final_strategy_artifact": (
            str(frozen["frozen"]["artifact_id"]) if frozen is not None else None  # type: ignore[index]
        ),
        "verdict": experiment_verdict(records),
        "paper_candidate": paper_candidate(records),
    }


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
    for role in ("main", "subagent", "nl", "compact"):
        settings.build_gateway(role, require_credentials=not preflight)
    # Every conversation role must leave room for its output budget; the
    # session parent's derived budget becomes the settings' own.
    settings.compaction_for("subagent")
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
) -> tuple[Path, tuple[str, str] | None]:
    """The PIT view seed this experiment hardlinks from, and the release it binds.

    The default seed is an optimisation over cold-building: it carries the
    default dataset selection, and an experiment that asks for anything else
    simply finds no matching contract and builds its own views. A seed named
    explicitly is a decision — the only way an arm whose selection differs from
    the default gets prebuilt views at all — so it is checked here, at create
    time: the tree must exist and must already carry exactly this snapshot
    configuration. Silently cold-building instead would cost hours and look
    like a slow experiment rather than a wrong parameter. A named seed also
    binds the experiment's research release: the ``(generation_id,
    release_raw_dir)`` it was built from is returned, and None means the seed
    is the optional default, which binds nothing and need not apply.
    """

    default = repo_root / DEFAULT_PIT_VIEWS_SEED
    if value in (None, ""):
        return default, None
    if not isinstance(value, str):
        raise ValueError("pit_views_seed must be a string")  # noqa: TRY004
    seed = _repo_path(repo_root, value.strip(), "pit_views_seed")
    if seed == default:
        return default, None
    if not seed.is_dir() or seed.is_symlink():
        raise ValueError(f"pit_views_seed must be an existing directory: {seed}")
    return seed, assert_seed_snapshot_config(seed, snapshot_config)


def _seed_release_trading_days(
    seed: Path,
    seed_release: tuple[str, str],
    *,
    experiment_dir: Path,
    raw_dir: Path,
    fundamental_events_root: Path,
    fundamental_events_status: Path,
    required_raw_datasets: tuple[str, ...],
    preflight: bool,
) -> tuple[Path, list[str]]:
    """Bind a named seed's research release; return its raw dir and trading days.

    The seed's views were built from one release, so that release is this
    experiment's: the worker pins it rather than the newest generation, and a
    pre-flight, which must not write, reads the same published release. Either
    way a seed whose release this deployment cannot provide is a wrong
    parameter, refused before any view is linked.
    """

    generation_id, seed_raw_dir = seed_release
    try:
        if preflight:
            release = published_research_release(
                generation_id=generation_id,
                raw_dir=raw_dir,
                fundamental_events_root=fundamental_events_root,
                fundamental_events_status=fundamental_events_status,
                required_raw_datasets=required_raw_datasets,
            )
        else:
            release = pin_research_release(
                experiment_dir=experiment_dir,
                generation_id=generation_id,
                raw_dir=raw_dir,
                fundamental_events_root=fundamental_events_root,
                fundamental_events_status=fundamental_events_status,
                required_raw_datasets=required_raw_datasets,
            )
        trading_days = load_sse_trading_days(release.raw_dir)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"pit_views_seed {seed} was built from research release {generation_id!r}, "
            f"which this experiment cannot pin: {exc}"
        ) from exc
    if str(release.raw_dir) != seed_raw_dir:
        raise ValueError(
            f"pit_views_seed {seed} records release raw dir {seed_raw_dir}, but research "
            f"release {generation_id} is at {release.raw_dir} in this repository"
        )
    return release.raw_dir, trading_days


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
    "run_local_interactive_worker",
]
