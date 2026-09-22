"""Runnable baseline and LLM backends for the daily JSON rolling pipeline.

The Agent-facing tools a research session runs on (``smoke_backtest``,
``batch_validate``, ``run_null_control``) and the Validation engine behind them
live in :mod:`autotrade.pipelines.session_tools`; this module composes them.
"""

from __future__ import annotations

import copy
import json
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from autotrade.agent.compact import ContextCompactionConfig
from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.environment.artifacts import (
    WORKSPACE_DIR_MODE,
    WORKSPACE_FILE_MODE,
    FilesystemArtifactStore,
    copy_artifact,
    copy_model_artifacts,
    overlay_artifact,
    readonly_baseline,
    restore_working_artifacts_writable,
)
from autotrade.environment.data.summary import write_agent_data_summary
from autotrade.environment.executor import PersistentCommandRunner
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm.model_profiles import AGENT_MAX_OUTPUT_TOKENS
from autotrade.environment.replay.stats import (
    PhaseTimer,
    attach_cost_sensitivity,
    attach_sub_window_benchmark,
    finalize_summary_timing,
)
from autotrade.environment.replay.style import (
    benchmark_summary_block,
    replay_style_analysis,
    write_style_rollup,
)
from autotrade.environment.runtime import (
    AgentTraceWriter,
    RunManifest,
    agent_trace_path,
    agent_transcript_dir,
    chmod_tree,
    utc_now_iso,
    write_json_atomic,
)
from autotrade.environment.sandbox import (
    SandboxLimits,
    DockerSandbox,
    LocalSandbox,
    SandboxConfig,
    SandboxSpec,
    experiment_container_labels,
    link_copytree,
)
from autotrade.environment.step_tree import StepTree
from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
)
from autotrade.environment.tools.base import (
    CommandRunner,
    Tool,
    ToolRegistry,
)
from autotrade.environment.tools.compact import CompactTool
from autotrade.environment.tools.files import EditFileTool, WriteFileTool
from autotrade.environment.tools.finish_session import FinishSessionTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.report_issue import (
    ReportIssueTool,
    issue_reports_path,
)
from autotrade.environment.tools.search import (
    GlobTool,
    GrepTool,
    ReadFileTool,
    SearchRoots,
)
from autotrade.environment.tools.shell import SandboxShellTool
from autotrade.environment.tools.skill_feedback import (
    SkillFeedbackTool,
    skill_feedback_path,
)
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.environment.tools.workspace import SafeWorkspace

from .calendar import FULL_SPAN, replay_window, yyyymmdd
from .config import (
    ArtifactRevision,
    BudgetUsed,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    FrozenArtifact,
    ResearchSessionRequest,
    ResearchSessionResult,
    SnapshotBundle,
    StepResult,
    StrategyExperimentConfig,
)
from .experiment import DailyStrategyPipeline
from .ledger import RESEARCH_STAGE, ExperimentLedger
from .session_tools import (
    BatchValidateTool,
    NullControlTool,
    SessionValidations,
    SmokeBacktestTool,
    another_batch_round_fits,
    batch_candidate_resources,
    batch_candidate_stats,
    session_budget_status,
)
from .skills import (
    OPERATING_MEMORY_DIRNAME,
    SKILLS_INDEX_PATH,
    DeleteSkillTool,
    MemorySource,
    WriteSkillTool,
    _assert_skills_absent_from_formal,
    ensure_operating_memory_snapshot,
    install_operating_memory,
    install_workspace_skills,
    mounted_skill_refs,
    write_skills_index,
)

if TYPE_CHECKING:
    from autotrade.environment.llm import ChatMessage, LLMProxy, ProviderResponse


class LocalDailySnapshotProvider:
    """Bind every phase to one immutable path selected at worker startup."""

    def __init__(self, daily_path: str | Path) -> None:
        self.daily_path = Path(daily_path).resolve(strict=True)
        if not self.daily_path.is_file():
            raise ValueError("daily_path must be a local Parquet file")

    def prepare(
        self,
        *,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle:
        del decision_time
        if phase not in {"valid", "heldout"}:
            raise ValueError(f"unsupported local snapshot phase: {phase}")
        return SnapshotBundle(
            snapshot_id=f"local_daily_{phase}_{start}_{end}",
            decision_ref=str(self.daily_path),
            replay_ref=str(self.daily_path),
            generation_id="local_daily",
        )

    def prepare_decision(self, *, decision_time: datetime) -> SnapshotBundle:
        return SnapshotBundle(
            snapshot_id=f"local_daily_decision_{decision_time:%Y%m%d}",
            decision_ref=str(self.daily_path),
            replay_ref="",
            generation_id="local_daily",
        )


class LocalDailyEvaluationBackend:
    """Evaluate an immutable strategy revision through DailyStrategyPipeline."""

    def __init__(
        self,
        daily_path: str | Path,
        results_root: str | Path,
        *,
        execution_mode: str,
        benchmark_index: str,
        sandbox: SandboxConfig | None = None,
        executor_factory=None,
    ) -> None:
        if execution_mode not in {"sandbox", "trusted"}:
            raise ValueError("execution_mode must be sandbox or trusted")
        # The arm's benchmark, which the style sidecar records; a plain daily
        # file carries no index series, so nothing is regressed against it.
        self.benchmark_index = benchmark_index
        self.daily_path = Path(daily_path).resolve(strict=True)
        self.results_root = Path(results_root).resolve()
        self.execution_mode = execution_mode
        self.sandbox = sandbox or SandboxConfig()
        self.executor_factory = executor_factory
        self._daily = pd.read_parquet(self.daily_path)
        if "trade_date" not in self._daily.columns:
            raise ValueError("daily Parquet must contain trade_date")
        self._daily = self._daily.copy()
        self._daily["trade_date"] = self._daily["trade_date"].map(yyyymmdd)

    @property
    def trading_days(self) -> list[str]:
        return sorted(set(self._daily["trade_date"].tolist()))

    def frame_between(self, start: str, end: str) -> pd.DataFrame:
        return self._daily[
            (self._daily["trade_date"] >= yyyymmdd(start))
            & (self._daily["trade_date"] <= yyyymmdd(end))
        ].copy()

    def evaluate(
        self,
        request: EvaluationRequest,
        *,
        max_days: int | None = None,
        start_day: str | None = None,
    ) -> EvaluationResult:
        """``start_day``/``max_days`` truncate the replay (``calendar.replay_window``).

        Same contract as the PIT backend: an unofficial smoke run gets the real
        replay path over a short window, never a short window dressed up as a
        full Validation.
        """
        if max_days is not None and max_days <= 0:
            raise ValueError("max_days must be a positive integer")
        if request.mode not in {"valid", "heldout"}:
            raise ValueError(f"unsupported local evaluation mode: {request.mode}")
        strategy_path = Path(request.revision.output_path) / "main.py"
        if not strategy_path.is_file():
            raise FileNotFoundError(
                f"strategy revision has no main.py: {strategy_path}"
            )
        validate_strategy_package(strategy_path)
        started_at = utc_now_iso()
        timer = PhaseTimer()
        with timer.phase("replay_frames"):
            frame = self.frame_between(request.start, request.end)
            replay_start = str(request.start)
            replay_end = str(request.end)
            if max_days is not None or start_day is not None:
                kept = replay_window(
                    set(frame["trade_date"]), start=start_day, max_days=max_days
                )
                frame = frame[frame["trade_date"].isin(set(kept))].copy()
                # A truncated replay must not claim it covered the whole span.
                if kept:
                    replay_start, replay_end = kept[0], kept[-1]
        if frame.empty:
            raise ValueError(
                f"daily replay is empty for {request.start}..{request.end}"
            )
        config = StrategyExperimentConfig(
            strategy_path=strategy_path,
            schedule=request.schedule,
            broker_profile=request.broker_profile,
            execution_mode=self.execution_mode,  # type: ignore[arg-type]
            sandbox=self.sandbox,
        )
        replay = DailyStrategyPipeline(
            config,
            executor_factory=self.executor_factory,
        ).run(frame)
        record = replay.to_record(start=replay_start, end=replay_end)
        with timer.phase("style_analysis"):
            style = replay_style_analysis(
                replay,
                frame,
                replay_dir=None,
                universes=(),
                mode=request.mode,
                benchmark_index=self.benchmark_index,
            )
        summary = record.get("stats")
        if not isinstance(summary, dict):
            raise TypeError("daily replay omitted stats")
        benchmark = benchmark_summary_block(style)
        if benchmark is not None:
            summary["benchmark"] = benchmark
        attach_sub_window_benchmark(summary, style)
        attach_cost_sensitivity(summary, request.broker_profile.slippage_bps)
        finalize_summary_timing(
            summary, started_at=started_at, setup_phases=timer.to_record()
        )
        result_id = f"{request.mode}_{uuid.uuid4().hex}"
        target = self.results_root / result_id / "result.json"
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_text(
            json.dumps(
                record, ensure_ascii=False, allow_nan=False, default=str, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        write_style_rollup(target.parent, style)
        return EvaluationResult(dict(summary), str(target))


class DeterministicBaselineDeveloper:
    """Replay the template unchanged on the research span and nominate it.

    Nothing is modified and nothing is judged: the arm's one session validates
    the template twice on the full span (the freeze gate counts full-span
    validations, the nominee included, and needs two) and nominates the
    second replay for the gate.
    """

    def __init__(
        self,
        *,
        baseline_strategy: str | Path,
        artifact_store: FilesystemArtifactStore,
        evaluator: EvaluationBackend,
        schedule,
        broker_profile,
        ref_store: AgentRefStore,
    ) -> None:
        self.baseline_strategy = Path(baseline_strategy).resolve(strict=True)
        self.ref_store = ref_store
        self.artifact_store = artifact_store
        self.evaluator = evaluator
        self.schedule = schedule
        self.broker_profile = broker_profile
        if not self.baseline_strategy.is_file():
            raise ValueError("baseline strategy must be a file")
        validate_strategy_package(self.baseline_strategy)
        self.baseline_root = self.artifact_store.root / "baseline_source"
        self.baseline_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.baseline_strategy, self.baseline_root / "main.py")

    def __call__(self, request: ResearchSessionRequest) -> ResearchSessionResult:
        source = self.baseline_root
        _assert_skills_absent_from_formal(source, None)
        # Step ids reach the Agent-facing projections, so they carry the same
        # opaque refs every agent-visible surface uses.
        prefix = (
            f"baseline_{self.ref_store.get_or_create('session', request.session_key)}__"
            f"{self.ref_store.get_or_create('run', request.run_id)}"
        )
        steps: list[StepResult] = []
        for index in (1, 2):
            revision = self.artifact_store.create_revision(source, models_path=None)
            typed_revision = ArtifactRevision(
                str(revision.revision_id),
                Path(revision.output_path),
                Path(revision.models_path) if revision.models_path is not None else None,
            )
            validation = self.evaluator.evaluate(
                request.validation.request(
                    typed_revision, schedule=self.schedule, broker_profile=self.broker_profile
                )
            )
            steps.append(
                StepResult(
                    f"{prefix}__valid_{index:03d}",
                    typed_revision.revision_id,
                    validation,
                    span=request.validation.label,
                )
            )
        return ResearchSessionResult(
            f"deterministic_baseline_{request.run_id}",
            tuple(steps),
            "freeze",
            node_id=steps[-1].step_id,
            finish_reason="deterministic_baseline_replay_no_agent_improvement",
        )


_SESSION_CALL_BUDGET_REFERENCE_MAX = 400
_SESSION_SUBAGENT_CALL_CAP_AT_REFERENCE = 200
_SESSION_PARENT_MAIN_RESERVE_AT_REFERENCE = 50
SESSION_LLM_CALL_ROLES = ("main", "subagent", "compact")


def session_role_quotas(max_calls: int) -> tuple[int, int]:
    """Sub-agent cumulative cap and parent-main reserve for one shared call budget."""
    if max_calls <= 0:
        raise ValueError("max_calls must be positive")
    subagent_cap = (
        max_calls
        * _SESSION_SUBAGENT_CALL_CAP_AT_REFERENCE
        // _SESSION_CALL_BUDGET_REFERENCE_MAX
    )
    parent_reserve = (
        max_calls
        * _SESSION_PARENT_MAIN_RESERVE_AT_REFERENCE
        // _SESSION_CALL_BUDGET_REFERENCE_MAX
    )
    if max_calls >= 2:
        subagent_cap = max(subagent_cap, 1)
    return subagent_cap, parent_reserve


class SessionCallBudget:
    """One counter and deadline shared across all model roles in a session.

    The total ``max_calls`` cap is hard. Sub-agents have a cumulative ceiling and
    compact/subagent cannot consume the parent-main reserve. This is not two
    independent budgets.
    """

    def __init__(
        self,
        *,
        max_calls: int,
        deadline: float | None = None,
        time_budget: InferenceTimeBudget | None = None,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if (deadline is None) == (time_budget is None):
            raise ValueError("provide exactly one of deadline or time_budget")
        self.max_calls = max_calls
        self.subagent_cap, self.parent_reserve = session_role_quotas(max_calls)
        self.time_budget = time_budget or InferenceTimeBudget(deadline=deadline)
        self.calls = 0
        self._subagent_calls = 0
        self._main_calls = 0
        self._compact_calls = 0
        self._lock = threading.Lock()

    @property
    def deadline(self) -> float:
        return self.time_budget.deadline

    @property
    def subagent_calls(self) -> int:
        return self._subagent_calls

    @property
    def main_calls(self) -> int:
        return self._main_calls

    @property
    def compact_calls(self) -> int:
        return self._compact_calls

    def seed(self, used: BudgetUsed) -> None:
        """Continue the counters from what earlier attempts of the session spent."""

        with self._lock:
            self.calls = int(used.llm_calls)
            self._main_calls = int(used.main_calls)
            self._subagent_calls = int(used.subagent_calls)
            self._compact_calls = int(used.compact_calls)

    def claim(self, role: str = "main") -> None:
        if role not in SESSION_LLM_CALL_ROLES:
            raise ValueError(f"unknown LLM call role: {role}")
        with self._lock:
            self.time_budget.check()
            if self.calls >= self.max_calls:
                raise RuntimeError("Agent session LLM call budget exhausted")
            if role == "subagent" and self._subagent_calls >= self.subagent_cap:
                raise RuntimeError("Agent session LLM call budget exhausted")
            if role != "main":
                parent_needed = max(0, self.parent_reserve - self._main_calls)
                if self.max_calls - self.calls - 1 < parent_needed:
                    raise RuntimeError("Agent session LLM call budget exhausted")
            self.calls += 1
            if role == "subagent":
                self._subagent_calls += 1
            elif role == "main":
                self._main_calls += 1
            else:
                self._compact_calls += 1

    def check_deadline(self) -> None:
        self.time_budget.check()


class SessionBudgetLLM(SessionTimeBudgetAware):
    """Apply a shared session budget to one model-role gateway."""

    def __init__(
        self,
        delegate: LLMProxy,
        *,
        max_calls: int | None = None,
        deadline: float | None = None,
        budget: SessionCallBudget | None = None,
        role: str = "main",
    ) -> None:
        if role not in SESSION_LLM_CALL_ROLES:
            raise ValueError(f"unknown LLM call role: {role}")
        if budget is None:
            if max_calls is None or deadline is None:
                raise ValueError(
                    "max_calls and deadline are required without a shared budget"
                )
            budget = SessionCallBudget(max_calls=max_calls, deadline=deadline)
        self.delegate = delegate
        self.budget = budget
        self.role = role
        self.provider = str(getattr(delegate, "provider", ""))
        self.model = str(getattr(delegate, "model", ""))
        window = getattr(delegate, "context_window_tokens", None)
        self.context_window_tokens = (
            window if isinstance(window, int) and not isinstance(window, bool) else None
        )

    @property
    def calls(self) -> int:
        return self.budget.calls

    @property
    def time_budget(self) -> InferenceTimeBudget:
        return self.budget.time_budget

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.time_budget

    def with_thinking(self, *, enabled: bool, reasoning_effort: str | None) -> LLMProxy:
        """Clone this wrapper over a gateway carrying another thinking setting.

        The clone keeps the same shared budget and role, so a per-call thinking
        level never forks the session's accounting. A delegate that cannot take
        a level (test doubles) leaves the wrapper unchanged, which the caller
        recognizes by identity.
        """

        clone_gateway = getattr(self.delegate, "with_thinking", None)
        if clone_gateway is None:
            return self
        clone = copy.copy(self)
        clone.delegate = clone_gateway(
            enabled=enabled, reasoning_effort=reasoning_effort
        )
        return clone

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = "auto",
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.budget.claim(self.role)
        response = self.delegate.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        self.budget.check_deadline()
        return response


AGENT_DATA_CONTRACT_FILES = ("data_summary.json", "unit_reference.json")


def install_agent_data_contract(
    paths,
    *,
    kind: str,
    session_ref: str,
    views: Mapping[str, tuple[Path, str]],
) -> None:
    """Publish the data contract a session's run manifest advertises: the
    summary and unit reference of its own mounted views, the session id opaqued
    so no calendar label leaks through them."""

    for name in AGENT_DATA_CONTRACT_FILES:
        # A resumed attempt rewrites the files its predecessor locked.
        target = paths.artifacts / name
        if target.is_file():
            target.chmod(0o644)
    write_agent_data_summary(
        paths.data_summary, kind=kind, session_ref=session_ref, views=views
    )
    for name in AGENT_DATA_CONTRACT_FILES:
        target = paths.artifacts / name
        if target.is_file():
            # Materialise as indented, key-sorted JSON: a single-line file
            # pages as one line and spills on every read_file.
            target.chmod(0o644)
            payload = json.loads(target.read_text(encoding="utf-8"))
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target.chmod(0o444)


def build_subagent_tools(
    search_roots: SearchRoots,
    workspace: SafeWorkspace,
    command_runner: CommandRunner,
    modification: ModificationCheckTool,
    smoke: Tool,
) -> list[Tool]:
    """Tools handed to the session's ``SubAgentEngine``: the parent's research
    and write surface plus the unofficial smoke run; no formal backtest."""
    return [
        ReadFileTool(search_roots),
        GrepTool(search_roots),
        GlobTool(search_roots),
        WriteSkillTool(workspace),
        DeleteSkillTool(workspace),
        WriteFileTool(workspace),
        EditFileTool(workspace),
        SandboxShellTool(workspace, command_runner, result_store=search_roots),
        modification,
        smoke,
    ]


class LLMResearchDeveloper:
    """Adapter from the native Agent loop to ``ResearchDeveloper``."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        subagent_llm: LLMProxy | None = None,
        compact_llm: LLMProxy | None = None,
        context_compaction: ContextCompactionConfig | None = None,
        # The children's compaction budget, derived from their own model's
        # window; defaults to the parent's.
        subagent_compaction: ContextCompactionConfig | None = None,
        baseline_strategy: str | Path,
        artifact_store: FilesystemArtifactStore,
        evaluator: EvaluationBackend,
        schedule,
        broker_profile,
        ledger: ExperimentLedger,
        experiment_dir: str | Path,
        runtime_root: str | Path,
        sandbox_spec: SandboxSpec | None = None,
        command_runner_factory: Callable[[Path], CommandRunner] | None = None,
        # One completion ceiling for the parent conversation and its children.
        max_response_tokens: int = AGENT_MAX_OUTPUT_TOKENS,
        research_directive: str = "",
        workspace_reference: str = "",
        operating_memory: str = "none",
        repo_root: str | Path | None = None,
    ) -> None:
        self.llm = llm
        self.subagent_llm = subagent_llm or llm
        self.compact_llm = compact_llm
        self.context_compaction = context_compaction or ContextCompactionConfig()
        self.subagent_compaction = subagent_compaction or self.context_compaction
        self.baseline_strategy = Path(baseline_strategy).resolve(strict=True)
        self.artifact_store = artifact_store
        self.experiment_dir = Path(experiment_dir).resolve()
        self.ref_store = AgentRefStore(self.experiment_dir)
        self.evaluator = evaluator
        self.schedule = schedule
        self.broker_profile = broker_profile
        self.ledger = ledger
        self.runtime_root = Path(runtime_root).resolve()
        self.sandbox_spec = sandbox_spec or SandboxSpec()
        self.command_runner_factory = command_runner_factory
        self.max_response_tokens = max_response_tokens
        self.research_directive = research_directive
        self.workspace_reference = workspace_reference
        self.operating_memory = str(operating_memory)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        validate_strategy_package(self.baseline_strategy)

    @property
    def decision_timeout_seconds(self) -> float:
        """The formal executor's per-decision inference wall clock."""

        sandbox = getattr(self.evaluator, "sandbox", None)
        limits = getattr(sandbox, "limits", None) or SandboxLimits()
        return float(limits.timeout_seconds)

    @property
    def fit_timeout_seconds(self) -> float:
        """The formal executor's wall clock for one strategy ``fit(context)``."""

        sandbox = getattr(self.evaluator, "sandbox", None)
        limits = getattr(sandbox, "limits", None) or SandboxLimits()
        return float(limits.fit_timeout_seconds)

    @property
    def strategy_gpu_count(self) -> int:
        """GPUs attached to the formal executor's strategy container.

        Read from the evaluator's own limits, so it is the number ``fit`` and
        ``generate_orders`` will actually see rather than a second derivation
        of the experiment request. It can differ from this session's container
        (``runtime_env.json``'s ``sandbox_spec``), which a per-session HITL
        override may move on its own.
        """

        sandbox = getattr(self.evaluator, "sandbox", None)
        limits = getattr(sandbox, "limits", None) or SandboxLimits()
        return int(limits.gpu_count)

    def __call__(self, request: ResearchSessionRequest) -> ResearchSessionResult:
        from autotrade.agent.compact import ContextCompactor, compaction_summary_message
        from autotrade.agent.prompts import (
            SESSION_DEFAULT_INSTRUCTION,
            build_resume_instruction,
            build_system_prompt,
        )
        from autotrade.agent.runner import (
            AgentSessionBudgetExhausted,
            AgentSessionConfig,
            AgentSessionRunner,
        )
        from autotrade.agent.subagent import (
            SubAgentConfig,
            SubAgentEngine,
        )
        from autotrade.pipelines.agent_inbox import bind_session_inbox

        # One runtime root per session, reused by every attempt: the
        # workspace, the working copy, the candidates and the step tree are
        # simply still there when an interrupted attempt is resumed.
        resume = request.resume
        root = self.runtime_root / request.session_key
        if root.exists() and resume is None:
            raise FileExistsError(f"session runtime already exists: {root.name}")
        if resume is not None and not (root / "agent" / "workspace" / "output" / "main.py").is_file():
            raise RuntimeError(
                "cannot resume the research session: the workspace of the interrupted "
                f"attempt is missing under the session runtime root {root.name}"
            )
        # The replay-years the counter continues from, not the trace's figure
        # alone: the facts and the resume note show what the session has left.
        used = replace(request.budget_used, replay_years=request.replay_years_spent)
        session_ref = self.ref_store.get_or_create("session", request.session_key)
        run_ref = self.ref_store.get_or_create("run", request.run_id)
        # The Agent-readable transcript of every attempt, appended beside the
        # host trace and offered to the session as the ``trace`` read root.
        transcripts = agent_transcript_dir(self.artifact_store.root.parent)
        transcripts.mkdir(parents=True, exist_ok=True)
        trace = AgentTraceWriter(
            agent_trace_path(self.artifact_store.root.parent, request.run_id),
            ids={
                "experiment_id": request.experiment_id,
                "epoch_id": RESEARCH_STAGE,
                "session_ref": session_ref,
                "run_id": run_ref,
                "session_kind": RESEARCH_STAGE,
            },
            transcript_dir=transcripts,
        )
        _environment_phase(request.progress_hook, "sandbox_layout", request.run_id)
        local = LocalSandbox(root)
        paths = local.prepare_layout()
        # Per-session HITL override; the "auto" selector still picks that many
        # GPUs by free memory at container start.
        if request.sandbox_gpu_count is None:
            sandbox_spec = self.sandbox_spec
        elif int(request.sandbox_gpu_count) == 0:
            sandbox_spec = replace(self.sandbox_spec, gpu=None, gpu_count=0)
        else:
            sandbox_spec = replace(
                self.sandbox_spec, gpu_count=int(request.sandbox_gpu_count)
            )
        # RunManifest publishes two views of the same data: the host audit copy
        # under runtime/, and the allowlisted Agent-visible copy mounted at
        # /mnt/artifacts/run_manifest.json. It is also where every backtest
        # summary accumulates. Research dates only: the forward period is
        # never known to a session.
        manifest = RunManifest.create(
            paths.run_manifest,
            {
                "experiment_id": request.experiment_id,
                "epoch_id": RESEARCH_STAGE,
                # Raw on the host manifest (issue reports link on it);
                # RunManifest's Agent-visible view and build_experiment_facts
                # both project it through the experiment reference store.
                "fold_id": request.session_key,
                "run_id": request.run_id,
                "session_key": request.session_key,
                "kind": RESEARCH_STAGE,
                "llm": {
                    "provider": str(getattr(self.llm, "provider", "")),
                    "model": str(getattr(self.llm, "model", "")),
                    "subagent": {
                        "provider": str(getattr(self.subagent_llm, "provider", "")),
                        "model": str(getattr(self.subagent_llm, "model", "")),
                    },
                },
                "conversation_id": request.run_id,
                "runtime_env_ref": "/mnt/artifacts/runtime_env.json",
                "data_summary_ref": "/mnt/artifacts/data_summary.json",
                "research": research_geometry_record(
                    request.research_years,
                    input_window_start=request.input_window_start,
                    decision_time=request.decision_time.isoformat(),
                ),
                # The index this arm is graded against; the Agent-visible facts
                # publish it so a session designs against the same benchmark
                # the host measures it on.
                "benchmark_index": request.benchmark_index,
                "snapshot_config": dict(request.snapshot_config),
                "snapshots": {
                    "decision_input": {"snapshot_id": request.snapshot.snapshot_id}
                },
                "start": start_record(),
                "arm": arm_record(request.steps_before),
                "modification_constraints": request.modification_constraints.to_record(),
                "acceptance_rules": dict(request.acceptance_rules),
                "schedule": self.schedule.to_record(),
                "broker_profile": self.broker_profile.to_record(),
                "nl_failure_policy": request.nl_failure_policy,
                "record_failed_attempts": request.record_failed_attempts,
                "attempt": resume.attempt if resume is not None else 1,
                "finalize_before_deadline_seconds": request.finalize_before_deadline_seconds,
                "sandbox_spec": sandbox_spec.to_record(),
                "exploration_directive": self.research_directive.strip(),
                "budgets": {
                    "max_replay_years": request.max_replay_years,
                    "max_null_controls": request.max_null_controls,
                    "max_llm_calls": request.max_llm_calls,
                    # deadline_seconds is the whole session wall clock;
                    # the grace is the trailing wrap-up slice of it, so
                    # the manifest carries both and the run facts can
                    # publish the split instead of one opaque total.
                    "deadline_seconds": request.deadline_seconds,
                    "deadline_grace_seconds": request.deadline_grace_seconds,
                    "strategy_inference_timeout_seconds": self.decision_timeout_seconds,
                    "strategy_fit_timeout_seconds": self.fit_timeout_seconds,
                    # What ``fit`` may spend those wall clocks on: the formal
                    # strategy container's own GPU allocation.
                    "strategy_gpu_count": self.strategy_gpu_count,
                    # The totals above are the arm's; what earlier attempts
                    # already spent of them.
                    "used_before_this_attempt": used.to_record(),
                },
            },
            ref_store=self.ref_store,
        )
        workspace_root = paths.workspace
        output_dir = workspace_root / "output"
        models_dir = workspace_root / "models"
        inputs_dir = workspace_root / "inputs"
        source = self.baseline_strategy.parent
        source_models = None
        if self.baseline_strategy.name != "main.py":
            raise ValueError(
                "baseline strategy file must be named main.py for research sessions"
            )
        if resume is None:
            copy_artifact(source, output_dir)
            copy_model_artifacts(source_models, models_dir)
            # A reference pack that ships a runnable starter is what the arm
            # starts from; the template supplies the contract files it keeps.
            seed_output_from_starter(
                output_dir, self.workspace_reference, repo_root=self.repo_root
            )
        restore_working_artifacts_writable(output_dir, models_dir)
        # Pin the read-only contract files to what this session actually
        # received, and record it for the audit: an initial artifact seeds them
        # from the live repository template, and editing that template must not
        # retroactively fail a session already running against the old bytes.
        seeded_readonly = readonly_baseline(output_dir)
        manifest.update(readonly_baseline=seeded_readonly)
        if inputs_dir.exists():
            chmod_tree(inputs_dir, file_mode=0o644, dir_mode=0o755)
        inputs_dir.mkdir(exist_ok=True)
        # Mount before the index is written: the curated entries and this
        # experiment's own skills reach the Agent through the same index.
        # Fixed at creation: a session mounts this experiment's snapshot and
        # never re-resolves the library. An experiment created before
        # snapshotting existed gets its snapshot here, once.
        memory_snapshot = ensure_operating_memory_snapshot(
            self.experiment_dir,
            mode=self.operating_memory,
            repo_root=self.repo_root,
            experiments_root=self.experiment_dir.parent,
        )
        # The mounted memory and refs are re-copied on a resume (their content
        # is fixed); the skills tree is the Agent's own work and stays.
        _remove_mounted_tree(workspace_root / OPERATING_MEMORY_DIRNAME)
        mounted_memory = install_operating_memory(workspace_root, self.experiment_dir)
        manifest.update(
            operating_memory=_operating_memory_record(memory_snapshot, mounted_memory)
        )
        if resume is None:
            skills_stats = install_workspace_skills(
                request.skills_source_ref or None,
                workspace_root,
                index_path=inputs_dir / "skills_index.json",
            )
        else:
            skills_stats = write_skills_index(
                workspace_root / "skills", inputs_dir / "skills_index.json"
            )
        manifest.update(
            skills={
                "index_path": SKILLS_INDEX_PATH,
                "count": skills_stats.count,
                "files": skills_stats.files,
                "bytes": skills_stats.bytes,
            }
        )
        _remove_mounted_tree(workspace_root / _WORKSPACE_REFS_DIR)
        install_workspace_reference(
            workspace_root,
            self.workspace_reference,
            repo_root=self.repo_root,
        )
        _environment_phase(request.progress_hook, "pit_view", request.run_id)
        self._install_snapshot_view(
            local,
            request,
            start=request.input_window_start,
            end=request.validation.end,
        )
        safe = SafeWorkspace(workspace_root)
        # Read-only exploration reaches the PIT view, the start node's
        # artifacts, the backtest results, the step lineage and the session's
        # own transcript, not just the writable workspace.
        search_roots = SearchRoots(safe, paths=paths, trace_root=transcripts)
        tree = self._install_step_tree(paths)
        sandbox: DockerSandbox | None = None
        emit_event = trace.emit
        try:
            if self.command_runner_factory is not None:
                command_runner = self.command_runner_factory(workspace_root)
            else:
                _environment_phase(
                    request.progress_hook, "sandbox_start", request.run_id
                )
                sandbox = DockerSandbox(
                    local,
                    sandbox_spec,
                    labels=experiment_container_labels(
                        request.experiment_id, run_id=request.run_id
                    ),
                )
                sandbox.start()
                command_runner = PersistentCommandRunner(sandbox)
            facts = self._session_facts(
                request,
                manifest=manifest,
                paths=paths,
                models_dir=models_dir,
            )
            write_json_atomic(inputs_dir / SESSION_CONTEXT_NAME, facts)
            chmod_tree(inputs_dir, file_mode=0o444, dir_mode=0o555)

            modification = ModificationCheckTool(
                output_dir,
                parent_dir=source,
                models_dir=models_dir,
                parent_models_dir=source_models,
                constraints=request.modification_constraints,
                readonly_baseline=seeded_readonly,
                # The host seeded this tree, so the host restores the contract
                # file it owns there: a session that deleted or overwrote it
                # cannot write it back, and without this every replay stayed
                # blocked on bytes only the host may produce.
                readonly_seed=source,
            )
            # The arm's budgets minus what earlier attempts spent: the clock
            # holds only the remainder, the counters start from the spend.
            attempt_seconds = max(request.deadline_seconds - used.inference_seconds, 0.001)
            time_budget = InferenceTimeBudget(duration_seconds=attempt_seconds)
            shared_budget = SessionCallBudget(
                max_calls=request.max_llm_calls,
                time_budget=time_budget,
            )
            shared_budget.seed(used)
            backtest = SessionValidations(
                request=request,
                output_dir=output_dir,
                models_dir=models_dir,
                artifact_store=self.artifact_store,
                evaluator=self.evaluator,
                tree=tree,
                schedule=self.schedule,
                broker_profile=self.broker_profile,
                time_budget=time_budget,
                ref_store=self.ref_store,
                ledger=self.ledger,
                manifest=manifest,
                experiment_dir=self.experiment_dir,
            )
            smoke = SmokeBacktestTool(
                request=request,
                output_dir=output_dir,
                models_dir=models_dir,
                modification_check=modification,
                evaluator=self.evaluator,
                schedule=self.schedule,
                broker_profile=self.broker_profile,
                # Host-only runtime scratch: the rehearsal copy is outside every
                # Agent mount, so nothing in the session can reach the bytes it
                # is replaying.
                scratch_root=paths.runtime / "smoke",
                time_budget=time_budget,
            )
            tools: list[Tool] = [
                ReadFileTool(search_roots),
                GrepTool(search_roots),
                GlobTool(search_roots),
                WriteFileTool(safe),
                EditFileTool(safe),
                SandboxShellTool(safe, command_runner, result_store=search_roots),
                WriteSkillTool(safe),
                DeleteSkillTool(safe),
                # Parent-only: sub-agents report findings to their parent, the
                # parent files the report.
                ReportIssueTool(issue_reports_path(self.experiment_dir), manifest),
                # Parent-only, and negative only: one mounted memory entry this
                # session's own measurements contradict.
                SkillFeedbackTool(
                    skill_feedback_path(self.experiment_dir),
                    manifest,
                    mounted_skill_refs(mounted_memory),
                ),
                # Parent-only: the Agent's own context compaction.
                CompactTool(),
                modification,
                smoke,
                BatchValidateTool(
                    backtest=backtest,
                    workspace=safe,
                    # Same static gate as the live working copy, pointed at the
                    # candidate directory: one constraint set for every formal
                    # artifact this session produces.
                    modification_check_factory=lambda directory: ModificationCheckTool(
                        directory,
                        parent_dir=source,
                        models_dir=models_dir,
                        parent_models_dir=source_models,
                        constraints=request.modification_constraints,
                        readonly_baseline=seeded_readonly,
                        # Restored only in the tree the host seeded. A
                        # candidate directory is the Agent's own layout of the
                        # artifact it asks to freeze: the file is supplied
                        # there when absent, and one carrying different bytes
                        # is refused rather than silently corrected.
                        readonly_seed=source if directory == output_dir else None,
                    ),
                    trace_emit=trace.emit,
                ),
            ]
            null_control_tool = (
                NullControlTool(backtest, max_calls=request.max_null_controls)
                if request.max_null_controls > 0
                else None
            )
            if null_control_tool is not None:
                null_control_tool.used = int(used.null_controls)
                tools.append(null_control_tool)
            tools.append(
                StepRollbackTool(tree, output_dir, models_dir, session_ref=session_ref)
            )
            # Matches the opaque session ref the step tree stores, so the
            # current-session check compares like with like. One gate for the
            # session: ``finish_session`` refuses a failing nomination with it,
            # and the Runner labels every hard-finalization candidate with it.
            tools.append(
                FinishSessionTool(
                    tree,
                    session_ref=session_ref,
                    freeze_gate=backtest.freeze_gate,
                    another_round_fits=lambda: another_batch_round_fits(backtest),
                    budget_status=lambda: session_budget_status(backtest),
                )
            )
            def budget_used_now() -> dict[str, object]:
                """The session's cumulative spend, as every budgeted trace event records it."""

                return BudgetUsed(
                    inference_seconds=used.inference_seconds
                    + max(attempt_seconds - time_budget.remaining(), 0.0),
                    llm_calls=shared_budget.calls,
                    main_calls=shared_budget.main_calls,
                    subagent_calls=shared_budget.subagent_calls,
                    compact_calls=shared_budget.compact_calls,
                    replay_years=backtest.replay_years_used,
                    null_controls=(
                        null_control_tool.used
                        if null_control_tool is not None
                        else int(used.null_controls)
                    ),
                ).to_record()

            emit_event = _agent_event_sink(
                trace, request.progress_hook, request.run_id, budget_used=budget_used_now
            )
            budgeted = SessionBudgetLLM(self.llm, budget=shared_budget, role="main")
            subagent_budgeted = SessionBudgetLLM(
                self.subagent_llm,
                budget=shared_budget,
                role="subagent",
            )
            compact_budgeted = (
                SessionBudgetLLM(
                    self.compact_llm, budget=shared_budget, role="compact"
                )
                if self.compact_llm is not None
                else None
            )
            subagent_tools = ToolRegistry(
                build_subagent_tools(
                    search_roots, safe, command_runner, modification, smoke
                )
            )
            subagent = SubAgentEngine(
                llm=subagent_budgeted,
                tools=subagent_tools,
                config=SubAgentConfig(max_tokens=self.max_response_tokens),
                time_budget=time_budget,
                # The parent's compaction gateway and archive, at the
                # threshold the children's own model window allows.
                compactor=(
                    ContextCompactor(
                        compact_budgeted,
                        self.subagent_compaction,
                        result_store=search_roots,
                        trace_ref=run_ref,
                    )
                    if compact_budgeted is not None
                    else None
                ),
            )
            runner = AgentSessionRunner(
                llm=budgeted,
                tools=ToolRegistry(tools),
                system_prompt=build_system_prompt(
                    self.schedule,
                    experiment_facts=facts,
                    exploration_directive=self.research_directive,
                    session_directive=request.directive,
                ),
                config=AgentSessionConfig(
                    finalize_before_deadline_seconds=(
                        request.finalize_before_deadline_seconds
                    ),
                    deadline_grace_seconds=request.deadline_grace_seconds,
                    max_llm_calls=request.max_llm_calls,
                    deadline_seconds=request.deadline_seconds,
                    max_response_tokens=self.max_response_tokens,
                ),
                compactor=(
                    ContextCompactor(
                        compact_budgeted,
                        self.context_compaction,
                        result_store=search_roots,
                        trace_ref=run_ref,
                    )
                    if compact_budgeted is not None
                    else None
                ),
                subagent=subagent,
                time_budget=time_budget,
                event_sink=emit_event,
                inbox=bind_session_inbox(
                    self.experiment_dir,
                    session_key=request.session_key,
                    run_id=request.run_id,
                ),
                freeze_gate=backtest.freeze_gate,
                trace_ref=run_ref,
            )
            # The Validations earlier attempts recorded are this session's:
            # nominable at the finish and listed in a hard finalization.
            recorded_validations = [
                {
                    "node_id": step.step_id,
                    "revision_id": self.ref_store.get_or_create("strategy", step.revision_id),
                    "stats": batch_candidate_stats(step.validation.summary),
                    **batch_candidate_resources(step.validation.summary),
                }
                for step in request.steps_before
            ]
            if resume is None:
                preamble: list[ChatMessage] = []
                instruction = SESSION_DEFAULT_INSTRUCTION
            else:
                preamble = (
                    [
                        compaction_summary_message(
                            resume.compaction_summary, kind="resume", trace_ref=run_ref
                        )
                    ]
                    if resume.compaction_summary
                    else []
                )
                instruction = build_resume_instruction(
                    attempt=resume.attempt,
                    interrupted_at=resume.interrupted_at,
                    error=resume.error,
                    has_summary=bool(resume.compaction_summary),
                    transcripts=resume.transcripts,
                    used=used.to_record(),
                    totals={
                        "inference_seconds": request.deadline_seconds,
                        "llm_calls": request.max_llm_calls,
                        "replay_years": request.max_replay_years,
                        "null_controls": request.max_null_controls,
                    },
                )
            try:
                result = runner.run(
                    instruction,
                    preamble=preamble,
                    complete_validations=recorded_validations,
                    budget_total_seconds=request.deadline_seconds,
                )
                conversation_id = result.conversation_id
                outcome, node_id, reason = _session_outcome(result.finish_value)
                finish_reason = "llm_agent_finish_session"
            except AgentSessionBudgetExhausted as exc:
                # The session closed on an exhausted budget (its wrap-up grace
                # or its model calls); the Validations it completed are still
                # the arm's trials.
                conversation_id = exc.conversation_id
                outcome, node_id, reason = "deadline", None, ""
                finish_reason = exc.finish_reason
            chmod_tree(inputs_dir, file_mode=0o644, dir_mode=0o755)
            final_skills = write_skills_index(
                workspace_root / "skills", inputs_dir / "skills_index.json"
            )
            chmod_tree(inputs_dir, file_mode=0o444, dir_mode=0o555)
            manifest.update(
                skills={
                    "index_path": SKILLS_INDEX_PATH,
                    "count": final_skills.count,
                    "files": final_skills.files,
                    "bytes": final_skills.bytes,
                }
            )
            steps = tuple(backtest.steps)
            if outcome == "freeze" and node_id not in {step.step_id for step in steps}:
                raise RuntimeError(
                    "finish_session nominated a node absent from this session's Validations"
                )
            manifest.update(
                conversation_id=conversation_id,
                selected_step_id=node_id,
                finish_outcome=outcome,
            )
            backtest.publish_tree()
            collected = local.collect_artifacts(
                self.artifact_store.root.parent / request.run_id
            )
            return ResearchSessionResult(
                conversation_id,
                steps,
                outcome,
                node_id=node_id,
                reason=reason,
                finish_reason=finish_reason,
                # The nulls the session already drew, for the freeze to reuse.
                null_controls=(
                    dict(null_control_tool.blocks)
                    if null_control_tool is not None
                    else {}
                ),
                # The collected copy, not the live sandbox tree: it outlives
                # the sandbox cleanup.
                run_manifest_ref=str(collected / "run_manifest.json"),
                skills_source_ref=str(collected / "workspace" / "skills"),
                budget_used=BudgetUsed.from_record(budget_used_now()),
                attempt=resume.attempt if resume is not None else 1,
            )
        except Exception as exc:
            emit_event(
                "session_error",
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        finally:
            if sandbox is not None:
                sandbox.stop()

    def _install_step_tree(self, paths) -> StepTree:
        """Hand the experiment-level step tree to the session.

        A resumed attempt finds its predecessor's tree in the shared session
        root and keeps it; a fresh session seeds from the experiment copy.
        """
        experiment_tree = self.experiment_dir / "steps"
        if not (paths.steps / "tree.json").is_file() and experiment_tree.exists():
            link_copytree(experiment_tree, paths.steps)
        return StepTree(paths.steps)

    def _install_snapshot_view(
        self,
        local: LocalSandbox,
        request: ResearchSessionRequest,
        *,
        start: str,
        end: str,
    ) -> None:
        source = Path(request.snapshot.decision_ref).resolve(strict=True)
        target = local.paths.current_snapshot
        # The view is fixed for the session: a resumed attempt re-mounts the
        # same decision view over the read-only one the interrupted attempt left.
        _remove_mounted_tree(target)
        if source.is_dir():
            local.bind_snapshot_view(source)
        elif source.is_file():
            frame_between = getattr(self.evaluator, "frame_between", None)
            if not callable(frame_between):
                raise TypeError(
                    "file-backed snapshot requires an evaluator with frame_between"
                )
            visible = frame_between(start, end)
            if not isinstance(visible, pd.DataFrame) or visible.empty:
                raise ValueError(f"Agent daily view is empty for {start}..{end}")
            target.mkdir(parents=True, exist_ok=True)
            visible.to_parquet(target / "daily.parquet", index=False)
            write_json_atomic(
                target / "manifest.json",
                {
                    "snapshot_id": request.snapshot.snapshot_id,
                    "kind": "local_daily",
                    "period_start": yyyymmdd(start),
                    "period_end": yyyymmdd(end),
                },
            )
            chmod_tree(target, file_mode=0o444, dir_mode=0o555)
        else:  # pragma: no cover - resolve(strict=True) already rejects this
            raise ValueError(
                f"snapshot decision_ref is neither a file nor directory: {source}"
            )
        install_agent_data_contract(
            local.paths,
            kind=RESEARCH_STAGE,
            session_ref=self.ref_store.get_or_create("session", request.session_key),
            views={"snapshot": (target, "/mnt/snapshot")},
        )

    def _session_facts(
        self,
        request: ResearchSessionRequest,
        *,
        manifest: RunManifest,
        paths,
        models_dir: Path,
    ) -> dict[str, object]:
        """The Agent-visible operational-facts block for this session.

        ``build_experiment_facts`` is the single visibility contract: it reads
        the run manifest, the runtime env and the data summary and projects the
        raw session id through the experiment reference store.
        """
        return {
            **build_experiment_facts(
                manifest=dict(manifest.data),
                ref_store=self.ref_store,
                runtime_env=_read_json_if_exists(paths.runtime_env),
                data_summary=_read_json_if_exists(paths.data_summary),
                max_llm_calls=request.max_llm_calls,
                context_compaction={
                    "enabled": self.compact_llm is not None,
                    "token_threshold": self.context_compaction.token_threshold,
                    "max_calls": self.context_compaction.max_calls,
                },
                model_artifacts_empty=(
                    not any(models_dir.iterdir()) if models_dir.exists() else True
                ),
            ),
            **session_fact_blocks(paths.workspace),
        }


def session_fact_blocks(workspace: str | Path) -> dict[str, object]:
    """The facts a session gets beside the manifest projection: the workspace
    index and the forbidden list."""

    return {
        "workspace": session_workspace_map(workspace),
        "forbidden": SESSION_FORBIDDEN,
    }


def research_geometry_record(
    years: Sequence[object], *, input_window_start: str, decision_time: str
) -> dict[str, object]:
    """The research period as a session may know it: research dates only.

    ``years`` are the research years in order, anything with ``label``,
    ``start`` and ``end``. The periods after research end exist and are
    sealed; neither their dates nor their slots are named anywhere a session
    can read.
    """

    first, last = years[0], years[-1]
    return {
        "decision_time": decision_time,
        "input_window": f"{input_window_start}..{last.end}",  # type: ignore[attr-defined]
        "research_period": f"{first.start}..{last.end}",  # type: ignore[attr-defined]
        "years": [
            {"label": year.label, "start": year.start, "end": year.end}  # type: ignore[attr-defined]
            for year in years
        ],
        "spans": (
            f"{FULL_SPAN} (every year), one year such as Y1, or contiguous years such "
            f"as Y1..Y{len(years)}"
        ),
    }


def start_record() -> dict[str, object]:
    """Which template seeded the session's working copy and its contract files.

    A reference pack that ships ``starter/`` is then overlaid onto ``output/``
    (:func:`seed_output_from_starter`), so this names the template the contract
    files and ``output/README.md`` come from, not necessarily the strategy code
    the session starts reading.
    """

    return {"kind": "template", "template_ref": "agent_output_template"}


def arm_record(steps: Sequence[StepResult]) -> dict[str, object]:
    """The arm's selection state when the attempt starts.

    Trials are the distinct revisions the session's earlier attempts
    validated, the pool the freeze gate deflates over; a session only runs
    while nothing is frozen.
    """

    return {
        "frozen": False,
        "freezes_per_arm": 1,
        "trials_to_date": len({step.revision_id for step in steps}),
        "full_span_validations_to_date": sum(1 for step in steps if step.span == FULL_SPAN),
    }


# The read-only facts file of a session, under ``workspace/inputs``.
SESSION_CONTEXT_NAME = "session_context.json"

# The ``forbidden`` fact of a research session.
SESSION_FORBIDDEN = [
    "data_after_research_end",
    "forward_period",
    "heldout",
    "external_network",
    "host_control",
]


def _session_outcome(finish: Mapping[str, object]) -> tuple[str, str | None, str]:
    """``finish_session``'s finish as the research session's outcome."""

    outcome = str(finish.get("outcome") or "")
    reason = str(finish.get("reason") or "")
    node_id = str(finish.get("node_id") or "") or None
    if outcome == "freeze" and node_id is None:
        raise RuntimeError("research session Agent froze without naming a node")
    if outcome not in ("freeze", "no_edge"):
        raise RuntimeError(f"research session Agent finished with {outcome!r}")
    return outcome, node_id if outcome == "freeze" else None, reason


_WORKSPACE_REFS_DIR = "refs"
_WORKSPACE_STARTER_DIR = "starter"
_REFERENCE_SKIP_NAMES = frozenset({".git", "__pycache__", "node_modules", ".venv"})
_REFERENCE_PDF_MAX_BYTES = 256 * 1024


def session_workspace_map(workspace: str | Path) -> dict[str, str]:
    """Agent-visible workspace index. ``refs`` is omitted when the directory is absent."""
    mapping = {
        "strategy": "output/main.py",
        "models": "models/",
        "session_context": f"inputs/{SESSION_CONTEXT_NAME}",
        "data_summary": "/mnt/artifacts/data_summary.json",
        "snapshot_in_sandbox": "/mnt/snapshot",
    }
    if (Path(workspace) / _WORKSPACE_REFS_DIR).is_dir():
        mapping["refs"] = "refs/"
    return mapping


def _operating_memory_record(
    snapshot: Mapping[str, object], sources: Sequence[MemorySource]
) -> dict[str, object]:
    """What the run manifest records about mounted cross-experiment memory.

    The snapshot id, not a live resolution: what this run mounted is decided by
    the experiment's frozen snapshot, and ``created_from`` says whether that
    snapshot was taken at creation or by the first session of an experiment
    that predates snapshotting.
    """

    return {
        "mode": str(snapshot.get("mode") or ""),
        "snapshot_id": str(snapshot.get("snapshot_id") or ""),
        "snapshot_created_at": str(snapshot.get("created_at") or ""),
        "snapshot_created_from": str(snapshot.get("created_from") or ""),
        "sources": [source.to_record() for source in sources],
    }


def _resolve_workspace_reference(
    workspace_reference: str | Path | None,
    repo_root: str | Path | None = None,
) -> Path | None:
    """The reference pack directory, or ``None`` when the parameter is unset.

    One resolution for both things a pack supplies: the read-only ``refs/``
    tree and the starter that seeds ``output/``. An empty value is a no-op; a
    set path must exist, be a directory and stay inside the repository,
    otherwise this fails immediately.
    """
    raw = str(workspace_reference or "").strip()
    if not raw:
        return None
    seed = Path(raw)
    if not seed.is_absolute():
        if repo_root is None:
            raise FileNotFoundError(f"workspace_reference does not exist: {raw}")
        seed = Path(repo_root) / seed
    if not seed.exists():
        raise FileNotFoundError(f"workspace_reference does not exist: {raw}")
    try:
        seed = seed.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"workspace_reference does not exist: {raw}") from exc
    if not seed.is_dir():
        raise NotADirectoryError(f"workspace_reference must be a directory: {seed}")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        if seed != root and root not in seed.parents:
            raise ValueError("workspace_reference must stay inside the repository")
    return seed


def install_workspace_reference(
    workspace: str | Path,
    workspace_reference: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Copy the optional reference pack into ``workspace/refs/`` before sandbox start.

    The pack lands writable like everything else under the Agent's workspace
    mount. It was installed 0o444/0o555 once, to keep the reference copy from
    drifting; what that actually bought was a papercut in every arm, because
    ``cp`` reproduces the source's mode and the copies it seeds -- a candidate
    directory, a working file under ``output/`` -- came out unwritable for the
    typed writers, which run as the host user and cannot chmod a file the
    sandbox user owns. Drift costs nothing instead: every session start removes
    this tree and re-copies it from the checked-in pack.

    The copy writes only ``refs/``, never ``models/`` or ``inputs/``;
    ``output/`` is seeded separately by :func:`seed_output_from_starter`.
    """
    seed = _resolve_workspace_reference(workspace_reference, repo_root)
    if seed is None:
        return
    dest = Path(workspace) / _WORKSPACE_REFS_DIR
    if dest.exists():
        raise FileExistsError(f"workspace refs directory already exists: {dest}")
    dest.mkdir()
    _copy_workspace_reference_tree(seed, dest, seed_root=seed)
    chmod_tree(dest, file_mode=WORKSPACE_FILE_MODE, dir_mode=WORKSPACE_DIR_MODE)


def seed_output_from_starter(
    output_dir: str | Path,
    workspace_reference: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Seed a session's ``output/`` from the reference pack's starter package.

    A pack that ships ``starter/main.py`` ships a runnable strategy, so the
    working copy starts as that package rather than as the bare template: every
    arm otherwise spent its first substantive turn hand-porting the same files,
    and the copies it made from the read-only ``refs/`` tree came out
    unwritable. The template's read-only contract files are not overwritten —
    ``output/README.md`` remains the strategy contract, whatever the pack
    carries. Returns the relative paths seeded, empty when there is no pack or
    the pack ships no starter.
    """
    seed = _resolve_workspace_reference(workspace_reference, repo_root)
    if seed is None:
        return ()
    starter = seed / _WORKSPACE_STARTER_DIR
    if not (starter / "main.py").is_file():
        return ()
    return overlay_artifact(starter, output_dir)


def _remove_mounted_tree(path: Path) -> None:
    """Drop a read-only mount an earlier attempt left, so it can be re-copied."""

    if path.exists():
        chmod_tree(path, file_mode=0o644, dir_mode=0o755)
        shutil.rmtree(path)


def _skip_workspace_reference_name(name: str) -> bool:
    return name.startswith(".") or name in _REFERENCE_SKIP_NAMES


def _copy_workspace_reference_tree(
    source: Path, dest: Path, *, seed_root: Path
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if _skip_workspace_reference_name(child.name):
            continue
        if child.is_symlink():
            continue
        if child.is_dir():
            _copy_workspace_reference_tree(
                child, dest / child.name, seed_root=seed_root
            )
            continue
        if not child.is_file():
            continue
        if (
            child.suffix.lower() == ".pdf"
            and child.stat().st_size > _REFERENCE_PDF_MAX_BYTES
        ):
            continue
        try:
            resolved = child.resolve(strict=True)
        except OSError:
            continue
        if resolved != seed_root and seed_root not in resolved.parents:
            continue
        shutil.copy2(child, dest / child.name, follow_symlinks=False)


def _environment_phase(
    progress_hook,
    stage: str,
    run_id: str,
) -> None:
    if progress_hook is not None:
        progress_hook(stage, {"run_id": run_id})


# The events that carry the session's cumulative spend, so a resumed attempt
# reads it off the last one an interrupted attempt wrote.
BUDGET_EVENTS = frozenset({"llm_call", "tool_call", "session_end", "session_error"})


def _agent_event_sink(
    trace: AgentTraceWriter,
    progress_hook,
    run_id: str,
    *,
    budget_used: Callable[[], dict[str, object]] | None = None,
):
    def emit(event_type: str, payload: dict[str, object]) -> None:
        if budget_used is not None and event_type in BUDGET_EVENTS:
            payload = {**payload, "budget_used": budget_used()}
        trace.emit(event_type, payload)
        if progress_hook is None:
            return
        if event_type == "llm_call_started":
            stage = "llm_call"
        elif event_type == "tool_call_started":
            stage = (
                "backtest"
                if payload.get("tool") in ("batch_validate", "smoke_backtest")
                else "tool_call"
            )
        elif event_type == "subagent_wait_started":
            # The parent is idle waiting on a child. Without its own stage the
            # console kept showing the preceding llm_call, so a yielding parent
            # looked like a single model call running for tens of minutes.
            stage = "subagent_wait"
        elif event_type == "session_end":
            stage = "agent_complete"
        else:
            return
        public = {
            "run_id": run_id,
            **{
                key: payload[key]
                for key in ("call_index", "tool", "status", "llm_calls", "steps_used")
                if key in payload
            },
        }
        progress_hook(stage, public)

    return emit


def _read_json_if_exists(path: Path) -> dict[str, object]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = [
    "DeterministicBaselineDeveloper",
    "FilesystemArtifactStore",
    "session_workspace_map",
    "LLMResearchDeveloper",
    "LocalDailyEvaluationBackend",
    "LocalDailySnapshotProvider",
    "SessionBudgetLLM",
    "SessionCallBudget",
    "session_role_quotas",
]
