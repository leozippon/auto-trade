"""Strict local assembly for persistent interactive daily experiments."""

from __future__ import annotations

import math
import os
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from autotrade.agent.compact import ContextCompactionConfig
from autotrade.common.records import read_json_object, write_json_atomic
from autotrade.environment.artifacts import (
    FilesystemArtifactStore,
)
from autotrade.environment.broker import BrokerProfile, stamped_permitted_boards
from autotrade.environment.broker_core import BOARDS
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
    LLMProxy,
    build_model_gateway,
    effective_max_output_tokens,
    model_profile,
)
from autotrade.environment.pretrained import pretrained_pin
from autotrade.environment.runtime import utc_now_iso
from autotrade.environment.sandbox import (
    STRATEGY_TMPFS_SIZE,
    GpuMode,
    SandboxConfig,
    SandboxSpec,
    experiment_container_labels,
    remove_sandbox_tree,
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
    ExperimentConfig,
    acceptance_for,
)
from .experiment import ExperimentPipeline
from .experiment_params import (
    ExperimentParams,
    read_param,
    read_param_or_default,
    session_models,
)
from .hitl_state import (
    SCHEDULE_NAME,
    WEB_INTERNAL_PARAMS,
    ExperimentState,
    PlannedSession,
    build_session_plan,
    planned_sessions,
    read_gpu_claim,
    read_status,
)
from .interactive import InteractiveExperimentRunner
from .ledger import (
    ExperimentLedger,
    RegisteredArtifactMutated,
    RunMarkers,
    assert_no_registered_artifact_mutation,
    experiment_verdict,
    paper_candidate,
    registered_record,
    research_records,
)
from .local_backend import (
    DeterministicBaselineDeveloper,
    LocalDailyEvaluationBackend,
    LocalDailySnapshotProvider,
)
from .pit_backend import (
    PITDailyEvaluationBackend,
    ResearchPITSnapshotProvider,
    required_release_raw_datasets,
)
from .pit_views_seed import (
    assert_seed_carries_bonus_split,
    assert_seed_snapshot_config,
    assert_seed_timing_contract,
)
from .research_session import LLMResearchDeveloper
from .skills import latest_skills_snapshot, resolve_skill_library

# The experiment-level ``reasoning_effort`` offers exactly the levels that are
# distinct on the wire for the local Qwen profile (its chat template knows
# low/medium/xhigh; ``high`` was always sent as ``xhigh``).
REASONING_EFFORTS = ("low", "medium", "xhigh")

@dataclass(frozen=True)
class LLMWorkerSettings:
    env_file: Path
    model: str
    # The ``agent`` sub-agents of research sessions; they keep the parent's
    # call quota, budget wrapper and time budget on this gateway.
    subagent_model: str
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
    # The experiment's ``compact_token_threshold``; None = each role's bound
    # (``compaction_for``).
    compact_token_threshold: int | None = None
    # The experiment's ``llm_key``: which relay credential every role starts on.
    llm_key: str = "primary"

    def model_for(self, role: str) -> str:
        """The model of one gateway role."""

        models = {
            "main": self.model,
            "subagent": self.subagent_model,
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
            llm_key=self.llm_key,
        )

    def reasoning_effort_for(self, role: str) -> str | None:
        """The effort a role sends, or None when its thinking is off."""

        if role == "compact" or not self.thinking_enabled:
            return None
        return self.reasoning_effort

    def compaction_for(self, role: str) -> ContextCompactionConfig:
        """One conversation role's compaction budget (main or subagent).

        Its bound is ``window − that role's output budget −
        COMPACTION_SAFETY_MARGIN_TOKENS`` for the role's own model, so prompt
        plus output never exceeds the window, and the same bound for the
        compaction model, which must read the whole conversation. The
        threshold is that bound unless the experiment's ``compact_token_threshold``
        names a lower one; a higher one is refused, as is a window with no
        room for the output budget at all. The value reaches the run facts
        (parents) and the compaction events (children).
        """

        if role not in {"main", "subagent"}:
            raise ValueError(f"unknown conversation role: {role}")
        bounds: list[int] = []
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
        bound = min(bounds, default=None)
        threshold = self.compact_token_threshold or bound
        if threshold is None:
            raise ValueError(
                "compact_token_threshold is required when no model role declares "
                "a context window"
            )
        if bound is not None and threshold > bound:
            raise ValueError(
                f"compact_token_threshold {threshold:,} exceeds the {role} conversation's "
                f"bound {bound:,}: the context window of its model and of the compaction "
                f"model, less each one's output budget and "
                f"{COMPACTION_SAFETY_MARGIN_TOKENS:,} tokens"
            )
        return replace(self.compaction, token_threshold=threshold)


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
    experiment: ExperimentConfig
    work_root: Path
    raw_dir: Path | None = None
    fundamental_events_root: Path | None = None
    fundamental_events_status: Path | None = None
    pit_cache_root: Path | None = None
    pit_views_seed: Path | None = None
    # A named seed must apply or the run fails; with none every view is built
    # cold.
    pit_views_seed_required: bool = False
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
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
    params = read_json_object(params_path, missing_ok=True)
    if not params:
        raise ValueError(f"missing experiment params: {params_path}")
    options = resolve_worker_options(params, experiment_dir=directory, repo_root=repo_root)
    # The devices the console claimed for this run (webui.manager.start_worker):
    # the session container and every replay's strategy containers attach
    # exactly these. Without a claim the request names no device, and the
    # first container that asks for one refuses to start.
    claim = read_gpu_claim(directory / "hitl")
    spec = options.agent_sandbox
    if claim is None or spec is None or spec.gpu is None:
        return options
    if len(claim) != spec.gpu_count:
        raise ValueError(
            f"GPU claim {list(claim)} does not match gpu_count {spec.gpu_count}"
        )
    return replace(options, agent_sandbox=replace(spec, gpu=claim))


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
    experiment directory) is skipped. An experiment naming a prebuilt view cache pins
    the release that seed was built from, which is already published, so the
    pre-flight still reads that release and checks it reaches into Held-out;
    without a named seed the release is not known until the pin, and that
    check waits for it. Every parameter check runs either way.
    """

    directory = Path(experiment_dir).resolve()
    repository = Path(repo_root).resolve(strict=True)
    # The deployment's template file is identical for every create on a host,
    # so its absence is not a property of the request.
    repo_file = _repo_path if preflight else _repo_file
    # Every key a parameter, every value of its declared type
    # (``experiment_params.ExperimentParams``). The checks below read the file through
    # ``read_param``: a key it leaves out takes the parameter's one declared
    # default (for a pipeline knob ``ExperimentConfig``'s), never a
    # literal of the worker's own that could silently reconfigure every
    # experiment whose params.json predates the key; a value it states, null
    # included, is checked as stated.
    ExperimentParams.from_record(params)
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
    work_root = _path_value(read_param(params, "work_root"), repository, "work_root")
    if repository != work_root and repository not in work_root.parents:
        raise ValueError("work_root must stay inside the repository")
    baseline_value = params.get("baseline_strategy_path", params.get("strategy_path"))
    baseline = repo_file(repository, baseline_value, "baseline strategy")
    requested_backend = params.get("data_backend")
    if requested_backend in (None, ""):
        production_raw = repository / "data" / "raw"
        data_backend = (
            "pit"
            if str(read_param_or_default(params, "developer_mode")) == "llm"
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
    execution_mode = str(read_param_or_default(params, "execution_mode"))
    if execution_mode not in {"sandbox", "trusted"}:
        raise ValueError("execution_mode must be sandbox or trusted")
    developer_mode = str(read_param_or_default(params, "developer_mode"))
    if developer_mode not in {"baseline", "llm"}:
        raise ValueError("developer_mode must be baseline or llm")
    llm_settings, sandbox_spec = (
        _llm_settings(params, repository, preflight=preflight)
        if developer_mode == "llm"
        else (None, None)
    )
    snapshot_config = _snapshot_config(params)
    # The experiment's benchmark: refused here if it is not an index the lake carries,
    # and checked against the pinned release's partitions below.
    benchmark_index = str(read_param(params, "benchmark_index"))
    benchmark_index_label(benchmark_index)
    raw_dir = (
        _repo_path(repository, read_param(params, "raw_dir"), "raw_dir")
        if data_backend == "pit"
        else None
    )
    events_root = (
        _repo_path(repository, read_param(params, "fundamental_events_root"), "fundamental_events_root")
        if data_backend == "pit"
        else None
    )
    events_status = (
        _repo_path(repository, read_param(params, "fundamental_events_status"), "fundamental_events_status")
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
    pit_views_seed, seed_generation = (
        _pit_views_seed(read_param(params, "pit_views_seed"), repository, snapshot_config)
        if data_backend == "pit"
        else (None, None)
    )
    # Every experiment's account pays the dividend tax, which reads each replay
    # slot's bonus shares.
    if pit_views_seed is not None and pit_views_seed.is_dir():
        assert_seed_carries_bonus_split(pit_views_seed)
    # A named seed must have been built under this code's refresh nodes and
    # stamps. Checked when an experiment is created, never at a worker start: a
    # running experiment keeps the views it was created on.
    if preflight and seed_generation is not None:
        assert_seed_timing_contract(pit_views_seed, snapshot_config)
    geometry = ResearchGeometry(**{name: read_param(params, name) for name in GEOMETRY_PARAMETERS})
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
        if seed_generation is not None:
            assert pit_views_seed is not None
            release_raw_dir, trading_days = _seed_release_trading_days(
                pit_views_seed,
                seed_generation,
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
        # The benchmark the experiment will be graded against has to exist in the
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
        str(read_param_or_default(params, "strategy_period")),  # type: ignore[arg-type]
        str(read_param_or_default(params, "inference_time")),
    )
    initial_cash = _positive_float(read_param(params, "initial_cash"), "initial_cash")
    config = ExperimentConfig(
        experiment_id=experiment_id,
        experiments_root=directory.parent,
        geometry=geometry,
        benchmark_index=benchmark_index,
        window_months=_positive_int(read_param(params, "window_months"), "window_months"),
        max_null_controls=_nonnegative_int(
            read_param(params, "max_null_controls"), "max_null_controls"
        ),
        # Range-checked by the config itself (1 up to the candidate cap).
        batch_validate_max_concurrency=_positive_int(
            read_param(params, "batch_validate_max_concurrency"), "batch_validate_max_concurrency"
        ),
        max_llm_calls=_positive_int(read_param(params, "max_llm_calls"), "max_llm_calls"),
        max_api_spend_usd=_positive_float(read_param(params, "max_api_spend_usd"), "max_api_spend_usd"),
        session_max_attempts=_positive_int(
            read_param(params, "session_max_attempts"), "session_max_attempts"
        ),
        max_research_minutes=_positive_int(
            read_param(params, "max_research_minutes"), "max_research_minutes"
        ),
        research_directive=str(read_param_or_default(params, "research_directive")),
        workspace_reference=_optional_workspace_reference(
            read_param(params, "workspace_reference"), repository
        ),
        skill_library=resolve_skill_library(read_param(params, "skill_library")),
        record_failed_attempts=_strict_bool(
            read_param(params, "record_failed_attempts"), "record_failed_attempts"
        ),
        per_call_timeout_seconds=_positive_int(
            read_param(params, "per_call_timeout_seconds"), "per_call_timeout_seconds"
        ),
        strategy_fit_timeout_seconds=_positive_int(
            read_param(params, "strategy_fit_timeout_seconds"), "strategy_fit_timeout_seconds"
        ),
        strategy_inference_timeout_seconds=_positive_int(
            read_param(params, "strategy_inference_timeout_seconds"), "strategy_inference_timeout_seconds"
        ),
        strategy_cpus=_positive_float(read_param(params, "strategy_cpus"), "strategy_cpus"),
        strategy_memory=_memory_limit(read_param(params, "strategy_memory"), "strategy_memory"),
        # A tracking mandate exactly where the request named a cap; every
        # limit absent or null takes its default, and the rules check each
        # value's type and range (``config.acceptance_for``).
        acceptance=acceptance_for(params),
        schedule=schedule,
        broker_profile=BrokerProfile(
            initial_cash=initial_cash,
            commission_bps=_nonnegative_float(read_param(params, "commission_bps"), "commission_bps"),
            slippage_bps=_nonnegative_float(read_param(params, "slippage_bps"), "slippage_bps"),
            dividend_tax=True,
            # Stamped at creation; an experiment without the key buys on every board.
            permitted_boards=stamped_permitted_boards(params),
            max_total_holdings=_optional_positive_int(
                read_param(params, "max_total_holdings"), "max_total_holdings"
            ),
            max_single_name_weight=_optional_positive_float(
                read_param(params, "max_single_name_weight"), "max_single_name_weight"
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
        experiment=config,
        work_root=work_root,
        raw_dir=raw_dir,
        fundamental_events_root=events_root,
        fundamental_events_status=events_status,
        pit_cache_root=pit_cache_root,
        pit_views_seed=pit_views_seed,
        pit_views_seed_required=seed_generation is not None,
        snapshot_config=snapshot_config,
        max_intraday_row_group_rows=_positive_int(
            read_param(params, "max_intraday_row_group_rows"), "max_intraday_row_group_rows"
        ),
        llm=llm_settings,
        agent_sandbox=sandbox_spec,
    )


def strategy_boundary(config: ExperimentConfig) -> dict[str, Any]:
    """The experiment's strategy-container boundary as ``strategy_sandbox_from_spec``
    takes it: the one mapping every builder of an experiment's strategy sandbox (the
    pipeline, a Paper account, the console's prompt preview) passes on."""

    return {
        "fit_timeout_seconds": config.strategy_fit_timeout_seconds,
        "inference_timeout_seconds": config.strategy_inference_timeout_seconds,
        "cpus": config.strategy_cpus,
        "memory": config.strategy_memory,
    }


def strategy_sandbox_from_spec(
    spec: SandboxSpec | None,
    *,
    fit_timeout_seconds: float,
    inference_timeout_seconds: float | None = None,
    cpus: float | None = None,
    memory: str | None = None,
    experiment_id: str | None = None,
) -> SandboxConfig:
    """The strategy container's boundary, derived from the Agent session's spec.

    The wall clocks, CPU quota and memory ceiling are the experiment's own
    (``ExperimentConfig.strategy_*``); one left as None keeps the
    executor's default (``SandboxConfig``, ``SandboxSpec``), which is also the
    experiment default. The process guard, ``/dev/shm`` and the user are the
    container defaults and ``/tmp`` is ``STRATEGY_TMPFS_SIZE``, whatever the
    session container was given.

    ``experiment_id`` labels every strategy container the experiment starts
    like its session container, so the console reclaims them together.

    The experiment's GPU request travels with it: ``fit(context)`` is where a
    model is trained, and it runs in the strategy container of every formal
    replay (validation and forward alike), not in the session. The request is the experiment-level one — a per-session HITL
    ``sandbox_gpu_count`` override moves only that session's own container.
    A spec carrying the devices taken for this run (a tuple of indexes: the
    console's claim, see ``load_worker_options``, or another caller's
    ``hitl_state.select_gpus``) pins every strategy container to them; a spec
    still on ``GpuMode.AUTO`` leaves the container none, and it refuses to
    start.

    The experiment's pretrained-store pin travels the same way, so every strategy
    container mounts the store the session container does.
    """

    session = spec if spec is not None else SandboxSpec(gpu=None)
    resources: dict[str, Any] = {}
    if cpus is not None:
        resources["cpus"] = float(cpus)
    if memory is not None:
        resources["memory"] = memory
    clocks: dict[str, Any] = {"fit_timeout_seconds": float(fit_timeout_seconds)}
    if inference_timeout_seconds is not None:
        clocks["timeout_seconds"] = float(inference_timeout_seconds)
    return SandboxConfig(
        spec=SandboxSpec(
            image=session.image,
            docker_executable=session.docker_executable,
            pretrained_manifest_sha256=session.pretrained_manifest_sha256,
            gpu=session.gpu,
            gpu_count=session.gpu_count,
            gpu_name_filter=session.gpu_name_filter,
            tmpfs_size=STRATEGY_TMPFS_SIZE,
            **resources,
        ),
        labels=experiment_container_labels(experiment_id) if experiment_id else {},
        **clocks,
    )


class ExperimentPipelineBuild(NamedTuple):
    """One assembled experiment, as either driver of it needs to see it."""

    pipeline: ExperimentPipeline
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
    # A failed compaction falls through to the emergency fit path by design;
    # provider retries would only add their full latency to that failure.
    compact_gateway = llm or (
        options.llm.build_gateway("compact", max_retries=0)
        if agent and options.llm.compact_enabled
        else None
    )
    strategy_sandbox = strategy_sandbox_from_spec(
        options.agent_sandbox,
        **strategy_boundary(options.experiment),
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
            max_intraday_row_group_rows=options.max_intraday_row_group_rows,
            sandbox=strategy_sandbox,
            benchmark_index=options.experiment.benchmark_index,
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
            benchmark_index=options.experiment.benchmark_index,
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
            schedule=options.experiment.schedule,
            broker_profile=options.experiment.broker_profile,
            ledger=ledger,
            experiment_dir=options.experiment_dir,
            runtime_root=options.work_root / options.experiment_id,
            sandbox_spec=options.agent_sandbox,
            command_runner_factory=command_runner_factory,
            # One ceiling for the parent conversation and its children.
            max_response_tokens=options.llm.max_tokens_for("main"),
            research_directive=options.experiment.research_directive,
            workspace_reference=options.experiment.workspace_reference,
            skill_library=options.experiment.skill_library,
            repo_root=options.repo_root,
        )
        developer_label = "llm_research_agent"
    else:
        developer = DeterministicBaselineDeveloper(
            baseline_strategy=options.baseline_strategy,
            artifact_store=store,
            evaluator=evaluator,
            schedule=options.experiment.schedule,
            broker_profile=options.experiment.broker_profile,
            ref_store=ref_store,
        )
        developer_label = "deterministic_baseline_no_agent_improvement"
    pipeline = ExperimentPipeline(
        options.experiment,
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
    """Run the experiment from wherever its ledger ends: the research session, then
    the forward replay once an artifact registered, then the terminal status."""

    ref_store = AgentRefStore(options.experiment_dir)
    hitl = options.experiment_dir / "hitl"
    ledger = ExperimentLedger(options.experiment.ledger_path)
    store = FilesystemArtifactStore(options.experiment_dir / "artifacts" / "strategy")
    records = ledger.read()
    try:
        assert_no_registered_artifact_mutation(records)
    except RegisteredArtifactMutated as exc:
        write_json_atomic(
            hitl / "status.json",
            {
                "schema_version": 1,
                "state": ExperimentState.FAILED,
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
        # A finished experiment is terminal: its verdict is durable, so a resume
        # republishes the completion status instead of re-running anything,
        # before any snapshot, sandbox or gateway preparation.
        payload = _terminal_status(ledger, read_status(hitl / "status.json"))
        write_json_atomic(hitl / "status.json", payload)
        _release_finished_experiment(options, payload)
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
        session_max_attempts=options.experiment.session_max_attempts,
    )
    result = interactive.run()
    if result["status"] != "complete":
        return result
    payload = _terminal_status(
        ledger, {"completed_at": utc_now_iso()}, developer_mode=developer_label
    )
    # No book opens here: the operator opens a registered strategy's Paper account
    # from the console, on the experiment page's 读出 view or the best-experiment
    # panel (``paper.books.open_registered_book``).
    write_json_atomic(hitl / "status.json", payload)
    _release_finished_experiment(options, payload)
    return payload


def _write_session_plan(
    options: InteractiveWorkerOptions, hitl: Path, trading_days: list[str]
) -> tuple[PlannedSession, ...]:
    """Write the plan of record (``schedule.json``) and return its sessions.

    The forward entry states the replay span, the Held-out end clipped to the
    release; ``geometry.heldout`` refuses a release that does not reach it.
    """

    geometry = options.experiment.geometry
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


def _release_finished_experiment(
    options: InteractiveWorkerOptions, payload: dict[str, object]
) -> None:
    """Drop the derived trees of an experiment that will not resume, once its
    terminal status is written: the session workspace and the PIT view cache.

    Best-effort: a cache that stays is named in the status, and the experiment's
    outcome stands. A stopped or failed experiment never gets here and keeps both.
    """

    release_finished_sandbox(options)
    try:
        release_pit_views(options.experiment_dir)
    except Exception as exc:  # noqa: BLE001 - the verdict stands; the failure is recorded
        payload["pit_views_release_error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(options.experiment_dir / "hitl" / "status.json", payload)


def release_finished_sandbox(options: InteractiveWorkerOptions) -> None:
    """Remove the session workspace once this experiment will not resume.

    The ledger and registered strategy stay in the experiment directory. An
    interrupted attempt keeps the tree: the next start continues in it.
    """

    root = options.work_root.resolve()
    path = (options.work_root / options.experiment_id).resolve()
    if path == root or root not in path.parents or not path.is_dir():
        return
    remove_sandbox_tree(path)


def pit_views_tree(experiment_dir: Path) -> Path | None:
    """The experiment's PIT view cache, ``<experiment>/pit_views``, or None without one.

    The fixed name, not the ``pit_cache_root`` parameter, which may name the
    experiment directory itself. Raises ValueError when ``pit_views`` is not a
    plain directory: deleting through a link could reach a tree the experiment does
    not own.
    """

    views = experiment_dir / "pit_views"
    if views.is_symlink() or (views.exists() and not views.is_dir()):
        raise ValueError(f"pit_views is not a plain directory: {views}")
    return views if views.is_dir() else None


def release_pit_views(experiment_dir: Path) -> None:
    """Delete a finished experiment's PIT view cache; the worker's terminal step and
    ``scripts/experiments/prune_pit_views.py`` share it.

    Its only reader is this experiment's snapshot provider, which a worker with a
    verdict never builds, and a missing tree is rebuilt on demand. Directory
    modes are unlocked and file modes left alone: the views are hard links into
    the seed and into every other experiment seeded from it. Raises OSError when the
    tree remains.
    """

    views = pit_views_tree(experiment_dir)
    if views is not None and not remove_sandbox_tree(views):
        raise OSError(f"pit_views still exists after removal: {views}")


def _terminal_status(
    ledger: ExperimentLedger,
    source: Mapping[str, object],
    *,
    developer_mode: str | None = None,
) -> dict[str, object]:
    """Durable completion status. The experiment's evidence stays in the append-only
    ledger; status.json only records that the run reached its end, so a resume
    republishes it without re-running or re-recording anything. ``verdict`` and
    ``paper_candidate`` are re-read from the ledger on every publish."""

    records = ledger.read()
    registered = registered_record(records)
    return {
        "schema_version": 1,
        "state": ExperimentState.COMPLETED,
        "pid": os.getpid(),
        "completed_at": source.get("completed_at") or utc_now_iso(),
        "developer_mode": developer_mode or source.get("developer_mode"),
        "completed_sessions": len(research_records(records))
        + sum(1 for record in records if record.get("record_type") == "forward"),
        "final_strategy_artifact": (
            str(registered["registered"]["artifact_id"]) if registered is not None else None  # type: ignore[index]
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

    An experiment reads its data from a pinned immutable research release: a
    lake being written falls back to one, the quality status lives inside it, and
    a deployment that carries only releases and seeds (the A100 instance) has
    no live lake at all. Requiring the live roots here would refuse all of
    them; a read that does need the live lake fails where it reads.
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

    def enabled(name: str) -> bool:
        value = read_param(params, name)
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
        return value

    def selection(name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
        value = read_param(params, name)
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
    screen_exclude_st = read_param(params, "screen_exclude_st")
    if type(screen_exclude_st) is not bool:
        raise ValueError("screen_exclude_st must be boolean")
    screen_boards = selection("screen_boards", BOARDS)
    return SnapshotConfig(
        window_months=_positive_int(read_param(params, "window_months"), "window_months"),
        daily_window_months=_optional_positive_int(
            read_param(params, "daily_window_months"), "daily_window_months"
        ),
        fundamentals_window_months=_optional_positive_int(
            read_param(params, "fundamentals_window_months"), "fundamentals_window_months"
        ),
        events_window_months=_optional_positive_int(
            read_param(params, "events_window_months"), "events_window_months"
        ),
        macro_window_months=_optional_positive_int(
            read_param(params, "macro_window_months"), "macro_window_months"
        ),
        text_window_months=_optional_positive_int(
            read_param(params, "text_window_months"), "text_window_months"
        ),
        intraday_trade_days=_positive_int(
            read_param(params, "intraday_trade_days"), "intraday_trade_days"
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
            read_param(params, "screen_exclude_new_listed_days"),
            "screen_exclude_new_listed_days",
        ),
        screen_min_circ_mv_yi=_optional_nonnegative_float(
            read_param(params, "screen_min_circ_mv_yi"), "screen_min_circ_mv_yi"
        ),
        screen_max_circ_mv_yi=_optional_nonnegative_float(
            read_param(params, "screen_max_circ_mv_yi"), "screen_max_circ_mv_yi"
        ),
        screen_min_price=_optional_nonnegative_float(
            read_param(params, "screen_min_price"), "screen_min_price"
        ),
        screen_max_price=_optional_nonnegative_float(
            read_param(params, "screen_max_price"), "screen_max_price"
        ),
        screen_boards=screen_boards,
    )


def _llm_settings(
    params: dict[str, object],
    repository: Path,
    *,
    preflight: bool = False,
) -> tuple[LLMWorkerSettings, SandboxSpec]:
    env_file = _path_value(read_param(params, "llm_env_file"), repository, "llm_env_file")
    if repository != env_file and repository not in env_file.parents:
        raise ValueError("llm_env_file must stay inside the repository")
    fold_model, subagent_model, compact_model = session_models(params)
    compact_max_tokens = _positive_int(
        read_param(params, "compact_max_tokens"), "compact_max_tokens"
    )
    # None = derived from the model windows; ``compaction_for`` sets the
    # effective per-role value either way.
    configured_threshold = _optional_positive_int(
        read_param(params, "compact_token_threshold"), "compact_token_threshold"
    )
    compaction = ContextCompactionConfig(
        token_threshold=configured_threshold or ContextCompactionConfig.token_threshold,
        keep_recent_messages=_positive_int(
            read_param(params, "compact_keep_recent_messages"),
            "compact_keep_recent_messages",
        ),
        max_response_tokens=effective_max_output_tokens(
            compact_model, compact_max_tokens
        ),
        max_calls=_nonnegative_int(
            read_param(params, "compact_max_calls"), "compact_max_calls"
        ),
    )
    settings = LLMWorkerSettings(
        env_file=env_file,
        model=fold_model,
        subagent_model=subagent_model,
        compact_model=compact_model,
        timeout_seconds=_positive_float(
            params.get("llm_timeout_seconds", read_param(params, "per_call_timeout_seconds")),
            "llm_timeout_seconds",
        ),
        max_retries=_nonnegative_int(read_param(params, "llm_max_retries"), "llm_max_retries"),
        retry_backoff_seconds=_nonnegative_float(
            read_param(params, "llm_retry_backoff_seconds"), "llm_retry_backoff_seconds"
        ),
        temperature=_bounded_float(
            read_param(params, "llm_temperature"), "llm_temperature", 0.0, 2.0
        ),
        max_response_tokens=_positive_int(
            read_param(params, "llm_max_response_tokens"), "llm_max_response_tokens"
        ),
        thinking_enabled=not _strict_bool(read_param(params, "no_thinking"), "no_thinking"),
        reasoning_effort=_reasoning_effort(read_param(params, "reasoning_effort")),
        compact_enabled=not _strict_bool(
            read_param(params, "disable_context_compact"), "disable_context_compact"
        ),
        compaction=compaction,
        compact_token_threshold=configured_threshold,
        # Checked by every role's gateway just below, pre-flight included.
        llm_key=read_param(params, "llm_key"),  # type: ignore[arg-type]
    )
    # Whether the host holds a credential is deployment state, not a property
    # of a WebUI create request.  Every model/role combination is still
    # validated at preflight with a non-secret placeholder.
    for role in ("main", "subagent", "compact"):
        settings.build_gateway(role, require_credentials=not preflight)
    # The Agent's typed grep tool runs ripgrep on this host (``tools.search``),
    # not in the sandbox image. Like a credential it is deployment state, so it
    # is refused here, before a session starts, instead of by the session.
    if not preflight and shutil.which("rg") is None:
        raise ValueError(
            "the Agent's grep tool runs ripgrep on this host, and 'rg' is not on PATH: "
            "install ripgrep"
        )
    # Every conversation role's threshold is held to its bound (create
    # pre-flight included); the session parent's becomes the settings' own.
    settings.compaction_for("subagent")
    settings = replace(settings, compaction=settings.compaction_for("main"))
    gpu_count = _gpu_count(read_param(params, "gpu_count"))
    # One boundary with the strategy container: the experiment's strategy CPUs and
    # memory, unless the internal agent_sandbox_* knobs name others; an experiment that
    # names neither keeps the shared default.
    cpus_key = "agent_sandbox_cpus" if "agent_sandbox_cpus" in params else "strategy_cpus"
    memory_key = "agent_sandbox_memory" if "agent_sandbox_memory" in params else "strategy_memory"
    sandbox = SandboxSpec(
        image=str(read_param_or_default(params, "agent_sandbox_image")),
        cpus=_positive_float(read_param(params, cpus_key), cpus_key),
        memory=_memory_limit(read_param(params, memory_key), memory_key),
        pids=_positive_int(read_param(params, "agent_sandbox_pids"), "agent_sandbox_pids"),
        tmpfs_size=_memory_limit(read_param(params, "agent_sandbox_tmpfs"), "agent_sandbox_tmpfs"),
        gpu=None if gpu_count == 0 else GpuMode.AUTO,
        gpu_count=gpu_count,
        # The store the experiment pinned at creation; an experiment created before the store
        # has none and its containers mount nothing.
        pretrained_manifest_sha256=pretrained_pin(read_param(params, "pretrained_manifest_sha256")),
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
) -> tuple[Path, str | None]:
    """The prebuilt view cache this experiment hardlinks from, and the release it binds.

    A named seed, the console's default included, is checked here, at create
    time: the tree must exist and must already carry exactly this snapshot
    configuration. Silently cold-building instead would cost hours and look
    like a slow experiment rather than a wrong parameter. A seed also binds the
    experiment's research release: the generation id it was built from is
    returned (the path the seed records beside it is where that release was
    on the host that built the seed, not a condition). No seed (``""``)
    returns ``(None, None)``: every view is built cold, on the newest release.
    """

    if value in (None, ""):
        return None, None
    if not isinstance(value, str):
        raise ValueError("pit_views_seed must be a string")
    seed = _repo_path(repo_root, value.strip(), "pit_views_seed")
    if not seed.is_dir() or seed.is_symlink():
        raise ValueError(f"pit_views_seed must be an existing directory: {seed}")
    generation_id, _release_raw_dir = assert_seed_snapshot_config(seed, snapshot_config)
    return seed, generation_id


def _seed_release_trading_days(
    seed: Path,
    generation_id: str,
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
    return release.raw_dir, trading_days


def _optional_workspace_reference(value: object, repo_root: Path) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ValueError("workspace_reference must be a string")
    text = value.strip()
    if not text:
        return ""
    # Config seed, not a data lake: existence is required even at create
    # preflight, and its absence is a refused parameter like any other (the
    # console and the round launcher report ValueError, not an OSError).
    if not (repo_root / text).is_dir():
        raise ValueError(f"workspace_reference {text} is not a directory in the repository")
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
        raise ValueError(f"{name} must be a boolean")
    return value


def _reasoning_effort(value: object) -> str:
    effort = str(value)
    if effort not in REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be one of " + ", ".join(REASONING_EFFORTS))
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
    "pit_views_tree",
    "release_pit_views",
    "run_local_interactive_worker",
]
