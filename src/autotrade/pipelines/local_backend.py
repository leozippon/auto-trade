"""Runnable baseline and LLM backends for the daily JSON rolling pipeline."""

from __future__ import annotations

import copy
import json
import math
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import pandas as pd

from autotrade.agent.compact import ContextCompactionConfig, safe_error_summary
from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.environment.artifacts import (
    READONLY_FILES,
    ArtifactSnapshotUnstable,
    FilesystemArtifactStore,
    copy_artifact,
    copy_artifact_snapshot,
    copy_model_artifacts,
    readonly_baseline,
    restore_working_artifacts_writable,
)
from autotrade.environment.data.summary import HOST_PATH_RE, write_agent_data_summary
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
    AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS,
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
    SessionInterrupt,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from autotrade.environment.tools.compact import CompactTool
from autotrade.environment.tools.files import EditFileTool, WriteFileTool
from autotrade.environment.tools.finish_session import (
    FinishSessionTool,
    SessionBudgetStatus,
)
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

from .agent_views import NULL_CONTROL_KEYS, allowed_keys
from .calendar import FULL_SPAN, yyyymmdd
from .config import (
    AcceptanceRules,
    ArtifactRevision,
    BudgetUsed,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    FrozenArtifact,
    ReplaySpan,
    ResearchSessionRequest,
    ResearchSessionResult,
    SnapshotBundle,
    StepResult,
    StrategyExperimentConfig,
    research_span,
)
from .experiment import (
    DailyStrategyPipeline,
    freeze_gate_for,
    null_control_seed,
    research_step_record,
)
from .ledger import RESEARCH_STAGE, ExperimentLedger
from .session_resume import record_step_sidecar
from .skills import (
    OPERATING_MEMORY_DIRNAME,
    SKILLS_INDEX_PATH,
    DeleteSkillTool,
    MemorySource,
    WriteSkillTool,
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
        sandbox: SandboxConfig | None = None,
        executor_factory=None,
    ) -> None:
        if execution_mode not in {"sandbox", "trusted"}:
            raise ValueError("execution_mode must be sandbox or trusted")
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
        self, request: EvaluationRequest, *, max_days: int | None = None
    ) -> EvaluationResult:
        """``max_days`` truncates the replay to its first N trading days.

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
            replay_end = str(request.end)
            if max_days is not None:
                kept = sorted(set(frame["trade_date"]))[:max_days]
                frame = frame[frame["trade_date"].isin(set(kept))].copy()
                # A truncated replay must not claim it covered the last quarter.
                replay_end = str(kept[-1]) if kept else replay_end
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
        record = replay.to_record(start=str(request.start), end=replay_end)
        with timer.phase("style_analysis"):
            style = replay_style_analysis(
                replay,
                frame,
                replay_dir=None,
                snapshot_dir=None,
                mode=request.mode,
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


def _public_error_text(exc: Exception) -> str:
    """The exact failure text with host paths redacted."""
    return HOST_PATH_RE.sub("[host_path]", safe_error_summary(exc))


def _public_validation_error(exc: Exception) -> str:
    """Agent-visible Validation failure: type, actionable reason, no host leaks."""
    return f"daily Validation failed: {_public_error_text(exc)}"


SMOKE_BACKTEST_DEFAULT_DAYS = 3
SMOKE_BACKTEST_MAX_DAYS = 5


class SmokeBacktestTool(SessionTimeBudgetAware):
    """Run the CURRENT working copy through the real replay for a few days.

    Hand-rolled shell smoke tests are what let seven of nine official backtests
    die on day one: a script that sets ``ctx.asof_dir = "/mnt/snapshot"`` and
    fakes an account object exercises the flat frozen snapshot and a dict-like
    account, while the replay hands the strategy a rolling directory-per-domain
    as-of view and a real ``AccountSnapshot``. This tool removes the reason to
    hand-roll one: same executor, same per-decision wall clock
    (``SandboxLimits.timeout_seconds``), same as-of layout, same account
    object, same modification gate — only the window is short.

    It is deliberately NOT an evaluation: no revision is committed, no step-tree
    node is written, nothing here can be selected at freeze time, and it does
    not consume the session's replay-year budget.

    Like every other replay tool it pauses the session's thinking clock. A
    5-day rehearsal is dominated by the same full ``fit`` a Validation runs,
    so charging it to the one budget the session cannot refill priced the
    cheap rehearsal the prompt requires before every batch in the scarce
    currency while the expensive verdict stayed free: 9.5 h across 48 audited
    sessions, three hour-long smokes dying at the fit cap costing one session
    ~3 h of its 10.17 h. The rehearsal still costs real host wall clock; it no
    longer costs the Agent its time to think. A sub-agent shares this tool
    object and this budget, so a smoke it starts pauses the parent's clock
    too -- the same union-of-pauses semantics an in-flight sub-agent already
    gets while the parent's own Validation is paused.
    """

    spec = ToolSpec(
        "smoke_backtest",
        "UNOFFICIAL smoke run of the CURRENT output/ over the first few trading "
        "days of the research period, on the real replay path: real rolling "
        "as-of view (each context.asof_dir/<domain>/ is a DIRECTORY of parquet "
        "parts, read it with pd.read_parquet(directory)), real AccountSnapshot "
        "object, same sandbox executor and per-decision timeout as "
        "a Validation replay. Returns per-day strategy and as-of seconds, order "
        "counts, the as-of domain directory names, and the exact exception text "
        "on failure. It commits no revision, creates no Step, cannot be frozen, "
        "and consumes no replay-years. Use it before batch_validate "
        "instead of hand-writing a shell smoke test against /mnt/snapshot.",
        {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": SMOKE_BACKTEST_MAX_DAYS,
                    "description": (
                        "Trading days from the start of the research period "
                        f"(default {SMOKE_BACKTEST_DEFAULT_DAYS})."
                    ),
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def __init__(
        self,
        *,
        request: ResearchSessionRequest,
        output_dir: Path,
        models_dir: Path,
        modification_check: ModificationCheckTool,
        evaluator: EvaluationBackend,
        schedule,
        broker_profile,
        scratch_root: Path,
        time_budget: InferenceTimeBudget,
    ) -> None:
        self.request = request
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.modification_check = modification_check
        self.evaluator = evaluator
        self.schedule = schedule
        self.broker_profile = broker_profile
        self.scratch_root = Path(scratch_root)
        self.time_budget = time_budget
        self.runs = 0

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        # Argument validation is the Agent's own mistake and stays on its
        # clock; the replay itself does not.
        days = self._days(arguments)
        with self.time_budget.pause():
            return self._invoke_exempt(days)

    def _invoke_exempt(self, days: int) -> ToolResult:
        self.runs += 1
        check = self.modification_check.invoke({})
        if not check.ok:
            # Same static gate as a Validation candidate, so a green smoke run
            # means the gate will not be what fails the official one.
            raise ToolError(f"smoke_backtest blocked by modification_check: {check.error}")
        # The rehearsal replays an immutable snapshot outside the Agent's mounts,
        # not the live tree: it can neither be frozen nor leave anything behind
        # in output/ (a trusted-mode run imports main.py and would drop
        # __pycache__ into the working copy, which the next modification_check
        # would reject), and the Agent's own writes during the run cannot reach
        # the bytes being replayed.
        scratch = self._scratch_dir()
        evaluation = None
        try:
            models_source = self.models_dir if self.models_dir.is_dir() else None
            fingerprint = copy_artifact_snapshot(
                self.output_dir,
                models_source,
                dest_output=scratch / "output",
                dest_models=scratch / "models",
            )
            if fingerprint != check.value["fingerprint"]:
                raise ArtifactSnapshotUnstable(
                    "output/ changed between modification_check and the smoke snapshot"
                )
            revision = ArtifactRevision(
                "smoke",
                scratch / "output",
                scratch / "models" if models_source is not None else None,
            )
            evaluation = self.evaluator.evaluate(
                self.request.validation.request(
                    revision, schedule=self.schedule, broker_profile=self.broker_profile
                ),
                max_days=days,
            )
        except SessionInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - the exception text IS the result
            return ToolResult(
                True,
                value={
                    "status": "failed",
                    "days_requested": days,
                    "official": False,
                    "counts_against_replay_budget": False,
                    "error": _public_error_text(exc),
                    "hint": _SMOKE_LAYOUT_HINT,
                },
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        result_dir = Path(evaluation.result_ref).parent
        try:
            return ToolResult(True, value=self._report(evaluation, result_dir, days))
        finally:
            # A smoke run leaves no result behind for the ledger or the Agent to
            # mistake for a Validation.
            shutil.rmtree(result_dir, ignore_errors=True)

    def _scratch_dir(self) -> Path:
        target = self.scratch_root / uuid.uuid4().hex
        target.mkdir(parents=True, exist_ok=False)
        return target

    def _days(self, arguments: Mapping[str, object]) -> int:
        raw = arguments.get("days", SMOKE_BACKTEST_DEFAULT_DAYS)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ToolError("smoke_backtest days must be an integer")
        if not 1 <= raw <= SMOKE_BACKTEST_MAX_DAYS:
            raise ToolError(
                f"smoke_backtest days must be between 1 and {SMOKE_BACKTEST_MAX_DAYS}"
            )
        return raw

    def _report(
        self, evaluation: EvaluationResult, result_dir: Path, days: int
    ) -> dict[str, object]:
        summary = dict(evaluation.summary)
        phases = summary.get("phase_seconds")
        phases = dict(phases) if isinstance(phases, dict) else {}
        replayed = summary.get("replayed_trade_days")
        replayed = int(replayed) if isinstance(replayed, int) else 0
        per_day = (
            {
                name: round(phases[name] / replayed, 3)
                for name in ("strategy", "data_view")
                if name in phases
            }
            if replayed
            else {}
        )
        return {
            "status": "ok",
            "official": False,
            "counts_against_replay_budget": False,
            "days_requested": days,
            "replayed_trade_days": replayed,
            "decision_calls": summary.get("decision_calls"),
            "order_count": summary.get("order_count"),
            "order_lifecycle": summary.get("order_lifecycle"),
            "reject_counts": summary.get("reject_counts"),
            "seconds_per_day": per_day,
            "phase_seconds": phases,
            "nl_calls": summary.get("nl_calls"),
            "asof_domains": _smoke_asof_domains(result_dir),
            "hint": _SMOKE_LAYOUT_HINT,
        }


_SMOKE_LAYOUT_HINT = (
    "context.asof_dir/<domain>/ is a directory of parquet parts: read it with "
    "pd.read_parquet(context.asof_dir + '/daily'), never "
    "context.asof_dir + '/daily.parquet'. context.account is an AccountSnapshot "
    "object (context.account.cash, context.account.positions), not a dict. Never "
    "fall back to context.snapshot_dir when an as-of read fails: the frozen "
    "snapshot stops at the decision time and using it is a PIT violation."
)


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


def _smoke_asof_domains(result_dir: Path) -> list[str]:
    """Domain directory names the replay actually exposed under asof_dir."""
    try:
        record = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    pit = record.get("pit")
    domains = pit.get("asof_domains") if isinstance(pit, dict) else None
    return [str(item) for item in domains] if isinstance(domains, list) else []


# One Validation's full replay record is attached to its step-tree node, and
# the node tree is mounted as the ``steps`` search root. This pair is the only
# Agent-readable reference to the full result, so the attachment site and the
# returned reference must name the same file.
VALIDATION_RESULT_ATTACHMENT = "validation/result.json"
STEP_TREE_SEARCH_ROOT = "steps"
# Summary blocks whose size scales with the replay: one row per closed position
# and one per week of the window. An inline copy is therefore not a fixed-cost
# observation — an audited session shipped 427 KB of ``per_stock`` into the
# conversation and forced a 174 s compaction plus twelve recovery calls. They
# stay in the referenced result.json; every other metric is O(1) and rides
# inline.
REPLAY_SCALED_SUMMARY_BLOCKS = ("per_stock", "weekly_returns")


def inline_backtest_stats(summary: Mapping[str, object]) -> dict[str, object]:
    """The backtest metrics an Agent observation may carry inline."""
    return {
        key: value
        for key, value in summary.items()
        if key not in REPLAY_SCALED_SUMMARY_BLOCKS
    }


def manifest_backtest_stats(summary: Mapping[str, object]) -> dict[str, object]:
    """The metrics one completed Validation contributes to the run manifest.

    Scalars are cheap enough to keep wholesale for host audit; structured
    values only earn their place when the Agent-visible projection actually
    carries them. Every completed Validation of a session projects through
    this one function, so no row is missing a block its siblings carry.
    """
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, (dict, list))
        or key in AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS
    }


# What a candidate row's provisional selection block is, and is not.
SELECTION_STATISTICS_NOTE = (
    "provisional: the freeze gate as it would read this node now, over every "
    "revision the arm has validated so far; the freeze recomputes it"
)


class SessionValidations:
    """The research session's Validation engine: one budget, one Step list, one tree.

    Not an Agent tool: ``batch_validate`` commits and replays every candidate
    through it, so no Validation can overspend or bypass the ledger behind
    another's back. The budget is counted in replay-years: a candidate costs
    one per research year its span covers.
    """

    def __init__(
        self,
        *,
        request: ResearchSessionRequest,
        output_dir: Path,
        models_dir: Path,
        artifact_store: FilesystemArtifactStore,
        evaluator: EvaluationBackend,
        tree: StepTree,
        schedule,
        broker_profile,
        time_budget: InferenceTimeBudget,
        ref_store: AgentRefStore,
        ledger: ExperimentLedger,
        manifest: RunManifest | None = None,
        experiment_dir: Path | None = None,
    ) -> None:
        self.request = request
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.artifact_store = artifact_store
        self.evaluator = evaluator
        self.tree = tree
        self.schedule = schedule
        self.broker_profile = broker_profile
        self.time_budget = time_budget
        self.ref_store = ref_store
        self.ledger = ledger
        self.manifest = manifest
        # Where each recorded Validation is made durable at once: the
        # experiment's step tree and its host-only sidecar, so an attempt that
        # dies keeps the nodes it validated for the attempt that resumes it.
        self.experiment_dir = experiment_dir
        # Continued from the earlier attempts' spend and Validations.
        self.replay_years_used = int(request.budget_used.replay_years)
        self.steps: list[StepResult] = list(request.steps_before)
        # Candidates reserved so far, which names each result.
        self.candidates_started = 0

    @property
    def replay_years_remaining(self) -> int:
        return max(self.request.max_replay_years - self.replay_years_used, 0)

    def span(self, label: object) -> ReplaySpan:
        """The research span ``label`` names, or a schema error naming the valid ones."""

        try:
            return research_span(self.request.research_years, str(label))
        except ValueError as exc:
            raise ToolError(
                str(exc), error_type="schema_error", blocked_target="span"
            ) from exc

    def freeze_gate(self, node_id: str) -> dict[str, object]:
        """The freeze gate as the Pipeline would read one Step of this session now.

        The Pipeline's own gate (``experiment.freeze_gate_for``) over the arm's
        recorded Steps and this session's completed ones, with the hard
        nomination rules of the run, so the Agent reads the verdict a freeze of
        this node would get today. A node that is not a Step of this session
        does not pass.
        """

        rows = [research_step_record(item) for item in self.steps]
        nominee = next((row for row in rows if row["step_id"] == node_id), None)
        if nominee is None:
            return {"passed": False, "reasons": ["freeze_needs_a_step_of_this_session"]}
        hard = (
            AcceptanceRules.from_record(self.request.acceptance_rules).evaluate(
                dict(nominee["summary"])  # type: ignore[arg-type]
            )[0]
            if self.request.acceptance_rules
            else []
        )
        return freeze_gate_for(self.ledger.read(), rows, nominee, hard_reasons=hard)

    def selection_statistics(self, step: StepResult) -> dict[str, object]:
        """The provisional freeze-gate reading one candidate row carries."""

        gate = self.freeze_gate(step.step_id)
        dsr = gate.get("deflated_sharpe")
        return {
            "freeze_gate_passed": gate["passed"],
            "freeze_gate_reasons": gate["reasons"],
            "deflated_sharpe_probability": (
                dsr.get("deflated_sharpe_probability") if isinstance(dsr, Mapping) else None
            ),
            "trials": dsr.get("trials") if isinstance(dsr, Mapping) else None,
            "full_span_validations": gate.get("full_span_validations"),
            "note": SELECTION_STATISTICS_NOTE,
        }

    def append_manifest_summary(self, summary: dict[str, object]) -> None:
        """Every backtest attempt, successful or not, lands in the run manifest.

        It is the only durable per-run record of what the session actually ran.
        """
        if self.manifest is not None:
            self.manifest.append_backtest_summary(summary)

    def commit_revision(
        self, source_output: Path, fingerprint: str, *, label: str
    ) -> ArtifactRevision:
        """Freeze one approved working tree into an immutable revision.

        The revision is verified to be the bytes ``modification_check`` just
        approved, so an approval can never be transferred to a tree that was
        still being written; a mismatch discards the revision and raises.

        It also records the artifact it descends from, so the revisions the arm
        keeps form the same lineage the Step tree draws.
        """
        revision = self.artifact_store.create_revision(
            source_output,
            models_path=self.models_dir,
            parent_revision_id=self._parent_revision_id(),
        )
        if revision.fingerprint != fingerprint:
            self.artifact_store.discard_revision(str(revision.revision_id))
            raise ArtifactSnapshotUnstable(
                f"{label} changed between modification_check and the "
                "Validation snapshot"
            )
        typed = ArtifactRevision(
            str(revision.revision_id),
            Path(revision.output_path),
            Path(revision.models_path)
            if revision.models_path is not None
            else None,
        )
        _assert_skills_absent_from_formal(typed.output_path, typed.models_path)
        return typed

    def _parent_revision_id(self) -> str | None:
        """The revision the working artifact descends from.

        A tree position the Agent set with ``step_rollback`` is the deliberate
        branch point and wins; otherwise the working copy descends from the last
        Validation the arm recorded, across attempts. ``batch_validate`` commits
        a whole round before it records any of it, so a round's revisions all
        resolve to the same parent and stay siblings. ``None`` before the arm's
        first Validation, and at a position the arm's own Validations do not
        account for (an inherited seed node), which starts a new lineage root
        rather than inventing a parent.
        """

        node_id = self.tree.current_node_id
        if node_id is None:
            return self.steps[-1].revision_id if self.steps else None
        for step in reversed(self.steps):
            if step.step_id == node_id:
                return step.revision_id
        return None

    def record_validation(
        self,
        revision: ArtifactRevision,
        evaluation: EvaluationResult,
        *,
        result_name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Append one completed Validation to the step tree under the current
        position; ``batch_validate`` repositions the tree before each record so
        its candidates are siblings of one parent."""
        node_id = self.tree.record_step(
            revision.output_path,
            epoch_id=RESEARCH_STAGE,
            # The session id is opaqued like every other Agent-visible id.
            session_ref=self.ref_store.get_or_create("session", self.request.session_key),
            run_id=self.ref_store.get_or_create("run", self.request.run_id),
            result_name=result_name,
            revision_id=self.ref_store.get_or_create(
                "strategy", revision.revision_id
            ),
            # tree.json is Agent-readable and accumulates one node per
            # Validation for the whole experiment, so it carries the
            # same fixed-size projection the observation does.
            metrics=inline_backtest_stats(evaluation.summary),
            models_root=revision.models_path,
            attachments={VALIDATION_RESULT_ATTACHMENT: evaluation.result_ref},
            metadata=metadata,
        )
        if self.experiment_dir is not None:
            span = str((metadata or {}).get("span") or "")
            if not span:
                raise ValueError("a recorded Validation needs its span in metadata")
            record_step_sidecar(
                self.experiment_dir,
                StepResult(node_id, revision.revision_id, evaluation, span=span),
            )
            self.publish_tree()
        return node_id

    def publish_tree(self) -> None:
        """Publish the session's tree to the experiment, node snapshots included."""

        if self.experiment_dir is not None and self.tree.tree_path.is_file():
            link_copytree(self.tree.root, self.experiment_dir / "steps")

    def validation_request(
        self, revision: ArtifactRevision, span: ReplaySpan
    ) -> EvaluationRequest:
        """The Validation replay of ``span`` for a revision the session accepted."""
        return span.request(
            revision, schedule=self.schedule, broker_profile=self.broker_profile
        )

    def reserve(self, count: int, span: ReplaySpan) -> list[str]:
        """Claim the replay-years of ``count`` candidates on ``span`` and name their results.

        A batch claims every replay-year before it commits anything, so a batch
        that does not fit is refused whole instead of half-run.
        """
        cost = count * span.slots
        remaining = self.replay_years_remaining
        if cost > remaining:
            raise ToolError(
                "the replay-year budget is spent"
                if remaining <= 0
                else f"the replay-year budget has {remaining} left and this batch "
                f"needs {cost} ({count} candidate(s) x {span.slots} year(s) of "
                f"span {span.label})",
                error_type="budget_exhausted",
            )
        names = [
            f"valid_{self.candidates_started + offset + 1:03d}" for offset in range(count)
        ]
        self.replay_years_used += cost
        self.candidates_started += count
        return names

    def release(self, count: int, span: ReplaySpan) -> None:
        """Refund candidates whose snapshot never held (no replay ran)."""
        self.replay_years_used = max(0, self.replay_years_used - count * span.slots)
        self.candidates_started = max(0, self.candidates_started - count)

    def charge_replay_year(self) -> bool:
        """Spend one replay-year on something that produced no Step.

        The only caller is ``batch_validate``'s repeated-rejection breaker, and
        it is the whole bounding mechanism there: a refused batch is otherwise
        free, so the budget is the only clock a retry loop can run down. False
        means the budget is already spent and there is nothing left to charge.
        """

        if self.replay_years_remaining <= 0:
            return False
        self.replay_years_used += 1
        return True

    def check_deadline(self) -> None:
        try:
            self.time_budget.check()
        except TimeoutError as exc:
            raise TimeoutError("research session deadline exceeded") from exc


# ``batch_validate``: one formal step that fans out a pre-registered candidate
# set over one span. Audited sessions reached at most two formal Validations
# each when every candidate was its own serial step, each branching off the
# last, which made the evidence per decision too thin. A batch fixes the parent
# and the span for every candidate, so their numbers are comparable, and
# pre-registers each hypothesis before any result exists. One candidate is a
# round too: the hypothesis is the same binding pre-registration at any width.
BATCH_VALIDATE_MIN_CANDIDATES = 1
# The session's replay-year budget is the real limit and is checked per call;
# this cap only bounds what one observation may carry, and six screening
# candidates already make a wide round.
BATCH_VALIDATE_MAX_CANDIDATES = 6
# Concurrent replays per batch. Each one holds its own result/as-of directory
# and its own strategy container — two when the candidate declares fit, whose
# read-write state bind needs a second worker — and the Timeview stash
# serializes part publication across evaluations, so the bound is host capacity
# (up to two containers per replay at SandboxLimits.cpus), not correctness —
# which only holds because both strategy deadlines scale with the batch's own
# width (``_batch_replay_timeouts``); a fixed clock made the outcome depend on
# how many siblings happened to share the host.
BATCH_VALIDATE_MAX_CONCURRENCY = 3
BATCH_NAME_MAX_CHARS = 40
BATCH_HYPOTHESIS_MAX_CHARS = 500
BATCH_PATH_MAX_CHARS = 200
# Workspace roots a candidate may not sit under: they are the working copy's
# own trees or not strategy trees at all. ``output`` itself is a valid path --
# the working copy validated as it stands; every revision is a snapshot
# verified against the bytes its check approved, so later edits cannot reach it.
_BATCH_RESERVED_ROOTS = frozenset({"output", "models", "inputs", "skills", "refs"})
_BATCH_WORKING_COPY = "output"
# Repeated identical rejections. A refused batch is free by design — nothing is
# committed and no slot is spent — which is also why nothing bounded the retry
# loop: one audited session spent 3.89 h of 10.17 h on 128 consecutive
# rejections carrying the same error, 393 parent LLM calls apart. A rejection is
# counted per session by its signature (error type plus the target it names,
# never the message text, which carries digests and so changes with the file);
# the third identical one says so and names the recovery for that signature, and
# from the seventh on each identical attempt consumes one replay-year, so the
# loop is bounded by the session's replay budget instead of by nothing.
BATCH_REJECTION_ESCALATE_AT = 3
BATCH_REJECTION_CHARGE_AFTER = 6
# The per-candidate projection an observation carries: a batch multiplies the
# fixed-size summary by N, so a row keeps what a screening decision is actually
# made on and points at the node's full record for everything else.
BATCH_CANDIDATE_SUMMARY_KEYS = (
    "total_return",
    "annualized_return",
    "long_return",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "turnover",
    "trade_count",
    "order_count",
    "decision_calls",
    "replayed_trade_days",
    "exposure",
    "benchmark",
    # Whether the excess survives worse execution, and how few trades and names
    # produced the gains: a screening decision made without them buys a result
    # that only exists at the modelled slippage or rests on one lucky name.
    "cost_sensitivity",
    "pnl_concentration",
    "sub_windows",
)
# A multi-year span has one sub-window row per July-June year, and a batch
# multiplies that by the number of candidates. A row keeps the columns a
# screening comparison is made on; the node's result.json keeps the full table.
# ``neutralized_excess_return`` rides with the raw one because the freeze gate
# and the verdict read the neutralized figure: a comparison made on the raw
# column alone reads a different number than they will.
BATCH_SUB_WINDOW_KEYS = (
    "label",
    "return",
    "excess_return",
    "neutralized_excess_return",
    "sharpe",
)


def batch_candidate_stats(summary: Mapping[str, object]) -> dict[str, object]:
    """The metrics one batch row carries inline."""
    stats = {
        key: value
        for key, value in summary.items()
        if key in BATCH_CANDIDATE_SUMMARY_KEYS
    }
    rows = stats.get("sub_windows")
    if isinstance(rows, list):
        stats["sub_windows"] = [
            {key: row.get(key) for key in BATCH_SUB_WINDOW_KEYS}
            for row in rows
            if isinstance(row, Mapping)
        ]
    return stats


@contextmanager
def _batch_replay_timeouts(evaluator: object, workers: int) -> Iterator[None]:
    """Widen both strategy deadlines to the replay width this batch creates.

    ``SandboxLimits.fit_timeout_seconds`` and ``timeout_seconds`` are
    runaway guards measured on host wall clock. Fanning ``workers`` replays
    out over the same host makes the same ``fit(context)`` and the same
    ``generate_orders(context)`` take longer without the strategy doing
    anything different, so a fixed cap makes the verdict depend on how many
    siblings a candidate happened to be batched with — the failure mode that
    cost one arm four Validations to fits solo reruns finished in
    1,550-2,027 s. Scaling both caps by ``workers`` keeps the guards (a
    runaway fit or decision still dies) while removing that dependence.

    The inference cap was deliberately left fixed when the fit cap was
    scaled, on the argument that a Validation makes hundreds of decision
    calls and widening the per-call cap would stretch a slow batch instead of
    failing it. Three arms then filed the counter-example on one day: a
    candidate measured at ~2.0 s/day over 243 serial decision days died at
    ``strategy inference exceeded 180s`` on one rebalance day inside a 2-way
    batch. Both caps bound ONE call, so both are equally distorted by the
    fan-out, and the whole-replay runaway is bounded elsewhere (the replay-year
    budget and the session's own deadline).

    The batch owns the evaluator for the pool's lifetime — a session runs
    one tool at a time — so mutating and restoring the shared config here is
    safe. An evaluator without sandbox limits (trusted mode, test doubles) has
    no clock to scale and is left alone.
    """

    config = getattr(evaluator, "sandbox", None)
    limits = getattr(config, "limits", None)
    if limits is None:
        yield
        return
    evaluator.sandbox = replace(  # type: ignore[attr-defined]
        config,
        limits=replace(
            limits,
            fit_timeout_seconds=limits.fit_timeout_seconds * workers,
            timeout_seconds=limits.timeout_seconds * workers,
        ),
    )
    try:
        yield
    finally:
        evaluator.sandbox = config  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _BatchCandidate:
    name: str
    hypothesis: str
    path: str
    directory: Path


class BatchValidateTool(SessionTimeBudgetAware):
    """The one Validation tool: pre-registered candidates as sibling Steps.

    Every candidate gets its own ``modification_check``, its own immutable
    revision, one replay over the batch's span and one Step node, and costs one
    replay-year per research year of that span. The candidates of a call share
    one parent node and one span, so a round's numbers are comparable and each
    hypothesis is registered before any result exists, at every width from one
    to six.

    Selection is never automatic: the Agent reads the table and nominates a
    winner with ``finish_session``.
    """

    spec = ToolSpec(
        "batch_validate",
        "Replay "
        f"{BATCH_VALIDATE_MIN_CANDIDATES}-{BATCH_VALIDATE_MAX_CANDIDATES} "
        "PRE-REGISTERED candidates over one span of the research period in one "
        "call; this is the only way to create a selectable node. span is full (the "
        "whole research period, the default), one research year such as Y2, or "
        "contiguous years such as Y2..Y4, as the research_geometry fact lists them; "
        "each span is replayed as one continuous book from the decision view at "
        "its first year. Each candidate is {name, hypothesis, path}: path is a "
        "workspace directory laid out like output/ (main.py plus its sibling "
        "modules; the read-only template files such as README.md are supplied for "
        "you; models/ is shared with the working copy), or output itself to "
        "validate the working copy as it stands, and hypothesis is the falsifiable "
        "statement you register BEFORE any result exists. The batch costs one "
        "replay-year per candidate per year of the span, reserved before anything "
        "runs; a candidate whose replay completes becomes its own immutable "
        "revision and Step node under the CURRENT node as shared parent, recorded "
        "with its span. A candidate whose replay fails keeps its cost, has no "
        "result_ref and, while record_failed_attempts is on, is recorded as a "
        "dead-end node, so later sessions see what was already tried. The returned "
        "replay_years_used/replay_years_remaining are the authoritative counters. "
        "The whole batch is refused before anything runs if the span is not one of "
        "the research years, if it does not fit the budget, if two candidates are "
        "byte-identical, or if one fails modification_check. A refusal is free the "
        "first times, but the same refusal repeated is not: the third identical one "
        "states the recovery for it, and from the "
        f"{BATCH_REJECTION_CHARGE_AFTER + 1}th on each identical attempt consumes "
        "one replay-year, so fix what the error names instead of calling again "
        "unchanged. The call waits briefly for background sub-agents that can "
        "write and is refused while one is still running; read-only audits keep "
        "running while the candidates replay concurrently. Returns one row per "
        "candidate: node id, headline metrics, the per-year return/excess/"
        "neutralized excess/Sharpe of sub_windows, the provisional "
        "selection_statistics (the freeze gate as it would read this node now, "
        "which the freeze recomputes), and wall seconds; a failed candidate's row "
        "carries its exact failure text instead — one failure never hides the "
        "others. Each completed row's result_ref reads back that candidate's full "
        "replay record. Selection stays yours: finish_session nominates a row as it "
        "is, and step_rollback(node_id) restores one as the working copy to build "
        "on.",
        {
            "type": "object",
            "properties": {
                "span": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 20,
                    "description": (
                        "full (default), one research year Yk, or contiguous years "
                        "Yi..Yj. Iterate on years; a freeze needs a full-span "
                        "validation."
                    ),
                },
                "candidates": {
                    "type": "array",
                    "minItems": BATCH_VALIDATE_MIN_CANDIDATES,
                    "maxItems": BATCH_VALIDATE_MAX_CANDIDATES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": BATCH_NAME_MAX_CHARS,
                                "description": "Short label, unique in the batch.",
                            },
                            "hypothesis": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": BATCH_HYPOTHESIS_MAX_CHARS,
                                "description": (
                                    "Falsifiable statement registered before "
                                    "the result exists; at most "
                                    f"{BATCH_HYPOTHESIS_MAX_CHARS} characters "
                                    "(keep the detail in your own notes)."
                                ),
                            },
                            "path": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": BATCH_PATH_MAX_CHARS,
                                "description": (
                                    "Workspace-relative directory laid out "
                                    "like output/, e.g. candidates/value."
                                ),
                            },
                        },
                        "required": ["name", "hypothesis", "path"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
        mutating=True,
        example={
            "span": "Y3..Y4",
            "candidates": [
                {
                    "name": "value_quality",
                    "hypothesis": "T-1 估值+质量 4 因子等权打分的中性化超额在两个年份都为正",
                    "path": "candidates/value_quality",
                },
                {
                    "name": "reversal",
                    "hypothesis": "21 日反转单因子的中性化超额在两个年份都为正",
                    "path": "candidates/reversal",
                },
            ],
        },
    )

    def __init__(
        self,
        *,
        backtest: SessionValidations,
        workspace: SafeWorkspace,
        modification_check_factory: Callable[[Path], ModificationCheckTool],
        trace_emit: Callable[[str, dict[str, object]], object] | None = None,
    ) -> None:
        self.backtest = backtest
        self.workspace = workspace
        self.modification_check_factory = modification_check_factory
        self._trace_emit = trace_emit
        # Per-session rejection counter, keyed by signature. Not persisted:
        # the loop it bounds is one session's retry loop.
        self._rejections: dict[tuple[str, str], int] = {}

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.backtest.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        self.backtest.check_deadline()
        with self.backtest.time_budget.pause():
            return self._invoke_exempt(arguments)

    def _invoke_exempt(self, arguments: Mapping[str, object]) -> ToolResult:
        # Everything that can refuse the batch runs before a single
        # replay-year is spent, so a rejected batch costs nothing and the Agent
        # can fix the offending input and call again. Because it costs nothing,
        # the same rejection can also repeat forever: ``_rejected`` counts it.
        try:
            span = self.backtest.span(arguments.get("span", FULL_SPAN))
            candidates = self._parse(arguments)
            self._supply_readonly_files(candidates)
            checks = self._precheck(candidates)
        except ToolError as exc:
            escalated = self._rejected(exc)
            if escalated is exc:
                raise
            raise escalated from exc
        result_names = self.backtest.reserve(len(candidates), span)
        batch_id = uuid.uuid4().hex[:12]
        parent_node_id = self.backtest.tree.current_node_id
        revisions = self._commit(candidates, checks, span)
        outcomes = self._replay(revisions, span)
        # A batch does not re-check the deadline here: every replay is already
        # paid for, and dropping N completed Validations because the clock ran
        # out during them would destroy real evidence. The session deadline is
        # enforced at the next dispatch and LLM call.
        rows: list[dict[str, object]] = []
        recorded: list[tuple[dict[str, object], StepResult]] = []
        try:
            for candidate, revision, result_name, outcome in zip(
                candidates, revisions, result_names, outcomes, strict=True
            ):
                evaluation, error, seconds = outcome
                # Every candidate — recorded or dead end — hangs off the node
                # the batch started at, never off the sibling before it.
                self.backtest.tree.set_position(parent_node_id)
                row: dict[str, object] = {
                    "name": candidate.name,
                    "hypothesis": candidate.hypothesis,
                    "path": candidate.path,
                    "result_name": result_name,
                    "wall_seconds": round(seconds, 1),
                }
                if evaluation is None:
                    row.update(
                        self._record_failure(
                            candidate, result_name, error, batch_id=batch_id, span=span
                        )
                    )
                else:
                    row.update(
                        self._record_success(
                            candidate,
                            revision,
                            evaluation,
                            result_name=result_name,
                            batch_id=batch_id,
                            span=span,
                        )
                    )
                    recorded.append((row, self.backtest.steps[-1]))
                rows.append(row)
        finally:
            # The batch never touched the working copy, so the tree position
            # comes back to where it branched from even if recording failed:
            # the Agent moves it deliberately with step_rollback once it picks
            # a winner.
            self.backtest.tree.set_position(parent_node_id)
        # Every row of the round deflates against the same trial pool: the
        # whole batch is complete by the time the table is returned.
        for row, step in recorded:
            row["selection_statistics"] = self.backtest.selection_statistics(step)
        if not recorded:
            raise ToolError(
                f"batch_validate: all {len(rows)} candidates failed their "
                "Validation; each still consumed its replay-years",
                error_type="validation_failed",
                details={"batch_id": batch_id, "candidates": rows},
            )
        return ToolResult(
            True,
            value={
                "batch_id": batch_id,
                "run_id": self.backtest.ref_store.get_or_create(
                    "run", self.backtest.request.run_id
                ),
                "parent_node_id": parent_node_id,
                "span": {
                    "label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "years": span.slots,
                },
                "candidates": rows,
                "complete_validations": len(recorded),
                "failed": len(rows) - len(recorded),
                "replay_years_used": self.backtest.replay_years_used,
                "replay_years_remaining": self.backtest.replay_years_remaining,
                "result_root": STEP_TREE_SEARCH_ROOT,
                "select_hint": batch_select_hint(
                    rows, replay_years_remaining=self.backtest.replay_years_remaining
                ),
            },
        )

    # ---- input ----

    def _parse(self, arguments: Mapping[str, object]) -> list[_BatchCandidate]:
        raw = arguments.get("candidates")
        if not isinstance(raw, list):
            raise ToolError(
                "batch_validate candidates must be an array",
                error_type="schema_error",
            )
        if not (
            BATCH_VALIDATE_MIN_CANDIDATES
            <= len(raw)
            <= BATCH_VALIDATE_MAX_CANDIDATES
        ):
            raise ToolError(
                f"batch_validate takes {BATCH_VALIDATE_MIN_CANDIDATES} to "
                f"{BATCH_VALIDATE_MAX_CANDIDATES} candidates, got {len(raw)}",
                error_type="schema_error",
            )
        names: set[str] = set()
        directories: set[str] = set()
        parsed: list[_BatchCandidate] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ToolError(
                    f"candidate {index} must be an object with name, "
                    "hypothesis and path",
                    error_type="schema_error",
                    blocked_target=str(index),
                )
            unknown = sorted(set(item) - {"name", "hypothesis", "path"})
            if unknown:
                raise ToolError(
                    f"candidate {index} has unknown field(s): {unknown}",
                    error_type="schema_error",
                    blocked_target=str(index),
                )
            name = _batch_text(item, "name", index, BATCH_NAME_MAX_CHARS)
            hypothesis = _batch_text(
                item, "hypothesis", index, BATCH_HYPOTHESIS_MAX_CHARS
            )
            path = _batch_text(item, "path", index, BATCH_PATH_MAX_CHARS)
            if name in names:
                raise ToolError(
                    f"duplicate candidate name: {name}",
                    error_type="schema_error",
                    blocked_target=name,
                )
            names.add(name)
            directory = self.workspace.resolve(path, must_exist=True, directory=True)
            if directory == self.workspace.root or (
                PurePosixPath(path).parts[0] in _BATCH_RESERVED_ROOTS
                and directory != self.workspace.root / _BATCH_WORKING_COPY
            ):
                raise ToolError(
                    f"candidate {name} points at a reserved workspace root "
                    f"({path}); pass output itself or copy the tree to its own "
                    "directory, e.g. candidates/<name>/",
                    error_type="path_error",
                    blocked_target=path,
                )
            if str(directory) in directories:
                raise ToolError(
                    f"duplicate candidate path: {path}",
                    error_type="schema_error",
                    blocked_target=path,
                )
            directories.add(str(directory))
            parsed.append(_BatchCandidate(name, hypothesis, path, directory))
        return parsed

    def _supply_readonly_files(self, candidates: Sequence[_BatchCandidate]) -> None:
        """Give each candidate the read-only template files it did not write.

        ``README.md`` is part of every formal artifact but carries no strategy
        content and the Agent may not edit it, so a candidate laid out from
        its strategy modules alone would be refused for "modifying" a file it
        never touched. The working copy's own read-only files are copied in
        where absent; a candidate that carries a different one is still
        refused by ``modification_check``. Only the read-only template names
        are touched — never a sibling module of the package.
        """

        for candidate in candidates:
            for name in READONLY_FILES:
                source = self.backtest.output_dir / name
                target = candidate.directory / name
                if not source.is_file() or target.exists():
                    continue
                try:
                    shutil.copyfile(source, target)
                except PermissionError as exc:
                    # A candidate directory copied out of a read-only artifact
                    # tree keeps mode 0444/0555, and the template cannot land
                    # in it. Say so with the remedy instead of failing as an
                    # unhandled host error.
                    raise ToolError(
                        f"candidate {candidate.name} ({candidate.path}) is not "
                        f"writable, so the read-only template {name} cannot be "
                        f"supplied: {_public_error_text(exc)}",
                        error_type="permission_denied",
                        blocked_target=candidate.path,
                    ) from exc

    def _precheck(self, candidates: Sequence[_BatchCandidate]) -> list[dict[str, object]]:
        """Static gate for every candidate, plus the batch-only rule that no
        two candidates may be the same bytes."""

        checks: list[dict[str, object]] = []
        fingerprints: dict[str, str] = {}
        for candidate in candidates:
            try:
                check = self.modification_check_factory(candidate.directory).invoke({})
            except ToolError as exc:
                # A read-only baseline violation is its own signature: the
                # repeated-rejection breaker names a different recovery for it
                # than for the size and validity failures.
                readonly = bool(exc.details.get("readonly_violations"))
                raise ToolError(
                    f"candidate {candidate.name} ({candidate.path}) failed "
                    f"modification_check: {exc}",
                    error_type=(
                        "readonly_baseline" if readonly else "modification_check_failed"
                    ),
                    blocked_target=candidate.path,
                    details=dict(exc.details) or None,
                ) from exc
            value = dict(check.value)
            fingerprint = str(value.get("fingerprint") or "")
            if fingerprint in fingerprints:
                raise ToolError(
                    f"candidates {fingerprints[fingerprint]} and "
                    f"{candidate.name} are byte-identical; every candidate "
                    "must carry a distinct hypothesis",
                    error_type="duplicate_candidate",
                    blocked_target=candidate.path,
                )
            fingerprints[fingerprint] = candidate.name
            checks.append(value)
        return checks

    # ---- repeated rejections ----

    def _rejected(self, exc: ToolError) -> ToolError:
        """Count one pre-reservation rejection and escalate a repeating signature.

        A first-time rejection is returned untouched: nothing about the free,
        fix-and-retry path changes. Only repetition is treated as evidence that
        retrying is not the fix.
        """

        signature = (exc.error_type, str(exc.blocked_target or ""))
        count = self._rejections.get(signature, 0) + 1
        self._rejections[signature] = count
        if count < BATCH_REJECTION_ESCALATE_AT:
            return exc
        charged = (
            self.backtest.charge_replay_year()
            if count > BATCH_REJECTION_CHARGE_AFTER
            else False
        )
        remaining = self.backtest.replay_years_remaining
        if count > BATCH_REJECTION_CHARGE_AFTER:
            cost = (
                f"This attempt consumed one replay-year ({remaining} left); "
                "so does every further identical one."
                if charged
                else "The replay-year budget is already spent; nothing is left "
                "to charge and no further batch can run."
            )
        else:
            cost = (
                f"From the {BATCH_REJECTION_CHARGE_AFTER + 1}th identical "
                "attempt on, each one consumes a replay-year."
            )
        recovery = _rejection_recovery(exc.error_type)
        message = (
            f"{exc}\n[repeated rejection] batch_validate has now refused this "
            f"exact rejection {count} times; calling it again unchanged returns "
            f"the same refusal. {recovery} {cost}"
        )
        if self._trace_emit is not None:
            self._trace_emit(
                "batch_rejection_escalated",
                {
                    "tool": "batch_validate",
                    "error_type": exc.error_type,
                    "blocked_target": signature[1],
                    "repeat_count": count,
                    "charged_replay_year": charged,
                    "replay_years_used": self.backtest.replay_years_used,
                },
            )
        details = dict(exc.details)
        details.update(
            {
                "repeat_count": count,
                "charged_replay_year": charged,
                "replay_years_remaining": remaining,
            }
        )
        return ToolError(
            message,
            error_type=exc.error_type,
            reason=exc.reason,
            retry_hint=recovery,
            blocked_target=exc.blocked_target,
            details=details,
        )

    # ---- execution ----

    def _commit(
        self,
        candidates: Sequence[_BatchCandidate],
        checks: Sequence[Mapping[str, object]],
        span: ReplaySpan,
    ) -> list[ArtifactRevision]:
        revisions: list[ArtifactRevision] = []
        try:
            for candidate, check in zip(candidates, checks, strict=True):
                revisions.append(
                    self.backtest.commit_revision(
                        candidate.directory,
                        str(check["fingerprint"]),
                        label=candidate.path,
                    )
                )
        except SessionInterrupt:
            raise
        except Exception as exc:
            # No replay ran, so the batch is infrastructure, not a Validation:
            # every revision and every reserved replay-year goes back.
            for revision in revisions:
                self.backtest.artifact_store.discard_revision(revision.revision_id)
            self.backtest.release(len(candidates), span)
            public_error = _public_error_text(exc)
            # Recorded as what it was: every attempt reaches the run manifest.
            self.backtest.append_manifest_summary(
                {
                    "mode": "valid",
                    "status": "infrastructure_error",
                    "complete_validation": False,
                    "span": span.label,
                    "error": public_error,
                }
            )
            raise ToolError(
                "batch_validate could not start: " + public_error,
                error_type="infrastructure_error",
                retry_hint=(
                    "A candidate directory was still being written when its "
                    "snapshot was taken and nothing ran; no budget was "
                    "consumed. Let any in-container job finish, then call "
                    "batch_validate again."
                ),
            ) from exc
        return revisions

    def _replay(
        self, revisions: Sequence[ArtifactRevision], span: ReplaySpan
    ) -> list[tuple[EvaluationResult | None, Exception | None, float]]:
        """Replay every committed revision over ``span``, bounded-concurrently.

        Each evaluation owns its result directory, its as-of view and its
        strategy container; the shared Timeview stash serializes part
        publication with its own file locks. Bookkeeping (revisions above,
        step-tree nodes below) stays on this thread and in input order, so the
        recorded lineage does not depend on which replay finished first.
        """

        outcomes: list[tuple[EvaluationResult | None, Exception | None, float] | None]
        outcomes = [None] * len(revisions)

        def run_one(index: int):
            started = time.perf_counter()
            try:
                evaluation = self.backtest.evaluator.evaluate(
                    self.backtest.validation_request(revisions[index], span)
                )
            except SessionInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - the text IS this row's result
                return None, exc, time.perf_counter() - started
            return evaluation, None, time.perf_counter() - started

        workers = min(len(revisions), BATCH_VALIDATE_MAX_CONCURRENCY)
        if workers <= 1:
            return [run_one(index) for index in range(len(revisions))]
        interrupt: SessionInterrupt | None = None
        with (
            _batch_replay_timeouts(self.backtest.evaluator, workers),
            ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="batch-validate"
            ) as pool,
        ):
            futures = {
                pool.submit(run_one, index): index for index in range(len(revisions))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcomes[index] = future.result()
                except SessionInterrupt as exc:
                    interrupt = exc
        if interrupt is not None:
            raise interrupt
        return [
            outcome if outcome is not None else (None, RuntimeError("replay did not run"), 0.0)
            for outcome in outcomes
        ]

    # ---- recording ----

    def _record_success(
        self,
        candidate: _BatchCandidate,
        revision: ArtifactRevision,
        evaluation: EvaluationResult,
        *,
        result_name: str,
        batch_id: str,
        span: ReplaySpan,
    ) -> dict[str, object]:
        node_id = self.backtest.record_validation(
            revision,
            evaluation,
            result_name=result_name,
            metadata={
                "batch_id": batch_id,
                "candidate": candidate.name,
                "hypothesis": candidate.hypothesis,
                "source_path": candidate.path,
                "span": span.label,
            },
        )
        self.backtest.steps.append(
            StepResult(node_id, revision.revision_id, evaluation, span=span.label)
        )
        self.backtest.append_manifest_summary(
            {
                "result_name": result_name,
                "mode": "valid",
                "status": "ok",
                "complete_validation": True,
                "batch_id": batch_id,
                "candidate": candidate.name,
                "hypothesis": candidate.hypothesis,
                "span": span.label,
                **manifest_backtest_stats(evaluation.summary),
            }
        )
        public_result_ref = f"{node_id}/{VALIDATION_RESULT_ATTACHMENT}"
        if not (self.backtest.tree.root / public_result_ref).is_file():
            raise ToolError(
                "Validation result attachment is missing for the recorded step: "
                f"{public_result_ref}"
            )
        return {
            "status": "ok",
            "node_id": node_id,
            "revision_id": self.backtest.ref_store.get_or_create(
                "strategy", revision.revision_id
            ),
            "stats": batch_candidate_stats(evaluation.summary),
            "result_ref": public_result_ref,
        }

    def _record_failure(
        self,
        candidate: _BatchCandidate,
        result_name: str,
        error: Exception | None,
        *,
        batch_id: str,
        span: ReplaySpan,
    ) -> dict[str, object]:
        public_error = _public_validation_error(
            error if error is not None else RuntimeError("unknown replay failure")
        )
        request = self.backtest.request
        metadata = {
            "batch_id": batch_id,
            "candidate": candidate.name,
            "hypothesis": candidate.hypothesis,
            "source_path": candidate.path,
            "span": span.label,
        }
        if request.record_failed_attempts:
            # record_failed_attempt leaves the tree position alone by design,
            # so a dead end never becomes anybody's parent.
            self.backtest.tree.record_failed_attempt(
                epoch_id=RESEARCH_STAGE,
                session_ref=self.backtest.ref_store.get_or_create(
                    "session", request.session_key
                ),
                run_id=self.backtest.ref_store.get_or_create("run", request.run_id),
                result_name=result_name,
                error=public_error,
                metadata=metadata,
            )
            self.backtest.publish_tree()
        self.backtest.append_manifest_summary(
            {
                "result_name": result_name,
                "mode": "valid",
                "status": "failed",
                "complete_validation": False,
                **{key: metadata[key] for key in ("batch_id", "candidate", "hypothesis", "span")},
                "error": public_error,
            }
        )
        return {"status": "failed", "error": public_error}


def batch_select_hint(
    rows: Sequence[Mapping[str, object]], *, replay_years_remaining: int
) -> str:
    """Name the row leading on the design's tie-breaker; select nothing.

    The neutralized excess is the figure the guidance ranks candidates on, so
    the hint says which row leads on it and on nothing else. A winning round
    is the starting point of the next pre-registered round, not the end of the
    session; once no batch fits the replay-year budget, the hint says the
    session is left with finishing.
    """

    ranked = [
        (_batch_row_neutralized_excess(row), row)
        for row in rows
        if row.get("status") == "ok"
    ]
    leading = max(ranked, key=lambda item: item[0], default=(float("-inf"), None))
    lead = (
        f"leading on neutralized excess: {leading[1].get('name')} "
        f"(node_id={leading[1].get('node_id')}); "
        if leading[1] is not None and math.isfinite(leading[0])
        else "no row carries a neutralized excess figure; "
    )
    if replay_years_remaining < 1:
        return (
            f"{lead}read every row yourself (whole span AND sub_windows) — nothing "
            "is selected for you. The replay-year budget is spent, so no further "
            "batch can run: write any skills and finish_session."
        )
    return (
        f"{lead}read every row yourself (whole span AND sub_windows) — nothing "
        "is selected for you. A winning round is the start of the next "
        "pre-registered round: step_rollback(node_id=<chosen>) restores it as "
        "the working copy; a freeze needs a full-span validation that passes the "
        "freeze gate, and finish_session is warranted only once the pre-registered "
        "hypotheses are resolved or the remaining budget no longer fits another "
        "round."
    )


def _batch_row_neutralized_excess(row: Mapping[str, object]) -> float:
    stats = row.get("stats")
    benchmark = stats.get("benchmark") if isinstance(stats, Mapping) else None
    value = (
        benchmark.get("neutralized_excess_return")
        if isinstance(benchmark, Mapping)
        else None
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("-inf")
    return float(value) if math.isfinite(value) else float("-inf")


def another_batch_round_fits(backtest: SessionValidations) -> bool:
    """Whether the session could still run one more ``batch_validate`` round.

    ``finish_session`` asks an early freeze for a reason only while this is
    true. It stops being true inside the deadline window — the finalize reserve
    before the main deadline, and the wrap-up grace behind it, where the Runner
    itself asks the session to finish — and once the replay-year budget cannot
    hold the smallest batch, one candidate on one year.
    """

    request = backtest.request
    main_remaining = backtest.time_budget.remaining() - request.deadline_grace_seconds
    if main_remaining <= request.finalize_before_deadline_seconds:
        return False
    return backtest.replay_years_remaining >= 1


def session_budget_status(backtest: SessionValidations) -> SessionBudgetStatus:
    """What this session still has when ``finish_session`` is called.

    The inference time is the same main-window figure ``another_batch_round_fits``
    reasons about: the grace reserve behind the deadline is wrap-up time, not
    budget the session chose to leave unused.
    """

    request = backtest.request
    return SessionBudgetStatus(
        replay_years_remaining=backtest.replay_years_remaining,
        replay_years_total=request.max_replay_years,
        inference_seconds_remaining=(
            backtest.time_budget.remaining() - request.deadline_grace_seconds
        ),
    )


NULL_CONTROL_NOTE = (
    "descriptive only: excess_percentile near 0.5 means the names carried no "
    "information the timing and sizing did not; nothing gates on it"
)


class NullControlTool(SessionTimeBudgetAware):
    """Rank one complete Validation against random-name replays of its trades.

    The same K=500 null control the Pipeline runs for the frozen node at
    freeze (``experiment._null_control``), drawn with the same seed through
    the same backend over the node's own span, so the figure the Agent reads
    before selecting is the figure the ledger records: the block is cached per
    node and handed to the Pipeline, which reuses it for the frozen node
    instead of drawing again. Each call is minutes of host replay, hence the
    per-session cap.
    """

    spec = ToolSpec(
        # The tool is a verb (like ``batch_validate``): ``null_control`` alone
        # is the result/ledger block it produces, and one name for both made
        # every mention of the block read as a tool reference.
        "run_null_control",
        "Random-portfolio null control of one complete Validation node.",
        {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "minLength": 1, "maxLength": 500},
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
        # Sequential, and locked after finish, like a formal backtest: it
        # pauses the session clock and spends a capped budget.
        mutating=True,
        example={"node_id": "<complete Validation node_id>"},
    )

    def __init__(self, backtest: SessionValidations, *, max_calls: int) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0:
            raise ValueError("run_null_control max_calls must be a positive integer")
        self.backtest = backtest
        self.max_calls = max_calls
        self.used = 0
        # Successful blocks by node id, exactly as the ledger records them.
        self.blocks: dict[str, dict[str, object]] = {}
        self.spec = ToolSpec(
            self.spec.name,
            "Rank one complete Validation node of this session against K=500 host "
            "replays of its own trade skeleton with random same-size names over the "
            "node's own span: the same null control the Pipeline runs for the frozen "
            "node at freeze. Returns the null_control block the ledger will carry "
            "(observed_excess, excess_percentile — near 0.5 means the names added "
            "nothing the timing and sizing did not — the null's mean and p05/p95, "
            "rejects_mean, dropped_trips_mean). Costs minutes of host replay per call "
            "(about 1.5 min per research year; the session clock pauses like a formal "
            f"backtest), consumes no replay-years and is capped at {max_calls} per "
            "session (max_null_controls in the budgets fact; every result reports "
            "null_controls_remaining); a node's block is cached, and the frozen node's "
            "block is reused at freeze instead of being drawn again. Use it on "
            "full-span finalists before finish_session, not on every candidate. "
            "Refused for a node that is not a complete Validation of this session, "
            "once the cap is spent, and while a background sub-agent that can write "
            "is still running; it is also unavailable once the session enters hard "
            "finalization, so rank the finalists before that.",
            self.spec.input_schema,
            mutating=True,
            example=self.spec.example,
        )

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.backtest.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        node_id = str(arguments.get("node_id") or "")
        self.backtest.check_deadline()
        step = next((item for item in self.backtest.steps if item.step_id == node_id), None)
        if step is None:
            raise ToolError(
                "run_null_control requires the node_id of a complete Validation of "
                f"this session; {node_id or '<empty>'} is not one",
                details={"candidates": [item.step_id for item in self.backtest.steps]},
            )
        if node_id in self.blocks:
            return ToolResult(True, value=self._report(node_id, self.blocks[node_id], cached=True))
        if self.used >= self.max_calls:
            raise ToolError(
                f"run_null_control budget exhausted: {self.max_calls} per session "
                "(max_null_controls). The frozen node's null control still runs at "
                "freeze.",
                error_type="null_control_budget_exhausted",
            )
        runner = getattr(self.backtest.evaluator, "null_control", None)
        if not callable(runner):
            raise ToolError("run_null_control is not available on this evaluation backend")
        span = research_span(self.backtest.request.research_years, step.span)
        # The attempt is charged before it runs: the compute is spent either way.
        self.used += 1
        with self.backtest.time_budget.pause():
            try:
                block = runner(
                    step.validation.result_ref,
                    start=span.start,
                    end=span.end,
                    profile=self.backtest.broker_profile,
                    schedule=self.backtest.schedule,
                    seed=null_control_seed(self.backtest.request.session_key, "frozen"),
                )
            except SessionInterrupt:
                raise
            except Exception as exc:
                raise ToolError(
                    "run_null_control failed: " + _public_error_text(exc),
                    error_type="null_control_failed",
                    details={
                        "null_controls_used": self.used,
                        "null_controls_remaining": self.max_calls - self.used,
                    },
                ) from exc
        self.blocks[node_id] = dict(block)
        return ToolResult(True, value=self._report(node_id, block, cached=False))

    def _report(
        self, node_id: str, block: Mapping[str, object], *, cached: bool
    ) -> dict[str, object]:
        return {
            "node_id": node_id,
            "null_control": allowed_keys(block, NULL_CONTROL_KEYS),
            "cached": cached,
            "null_controls_used": self.used,
            "null_controls_remaining": self.max_calls - self.used,
            "note": NULL_CONTROL_NOTE,
        }


def _rejection_recovery(error_type: str) -> str:
    """What to actually do about a ``batch_validate`` rejection that repeats.

    The first rejection already names the offending candidate; what it does not
    name is the remedy, which is why the audited loop kept re-sending the same
    batch. One sentence per signature class, concrete enough to act on.
    """

    if error_type == "readonly_baseline":
        return (
            "Recovery: call modification_check on that candidate directory and "
            "read delta.readonly_violations — the named file no longer holds "
            "the bytes this session was seeded with. Delete your copy of it "
            "from the candidate directory; batch_validate supplies the "
            "read-only template itself."
        )
    if error_type == "permission_denied":
        return (
            "Recovery: files copied out of a read-only tree keep mode 0444 and "
            "belong to the sandbox user, so run chmod -R a+w on that candidate "
            "directory through shell before calling again; u+w leaves the host-"
            "side writer locked out."
        )
    return (
        "Recovery: read the error text above and change the input it names; "
        "the call shape is not the problem."
    )


def _batch_text(
    item: Mapping[str, object], field_name: str, index: int, limit: int
) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(
            f"candidate {index} needs a non-empty {field_name} string",
            error_type="schema_error",
            blocked_target=field_name,
        )
    text = value.strip()
    if len(text) > limit:
        raise ToolError(
            f"candidate {index} {field_name} exceeds {limit} characters",
            error_type="schema_error",
            blocked_target=field_name,
        )
    return text


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
        used = request.budget_used
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
    """Where the session's working copy was seeded from: the template."""

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


def install_workspace_reference(
    workspace: str | Path,
    workspace_reference: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Copy optional operator notes into ``workspace/refs/`` before sandbox start.

    An empty ``workspace_reference`` is a no-op. A set path must exist and be a
    directory, otherwise this fails immediately. The copy writes only ``refs/``,
    never ``output/``, ``models/``, or ``inputs/``. Each research session has a
    fresh workspace, so later sessions see the notes only because this hook runs
    again.
    """
    raw = str(workspace_reference or "").strip()
    if not raw:
        return
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
    dest = Path(workspace) / _WORKSPACE_REFS_DIR
    if dest.exists():
        raise FileExistsError(f"workspace refs directory already exists: {dest}")
    dest.mkdir()
    _copy_workspace_reference_tree(seed, dest, seed_root=seed)
    chmod_tree(dest, file_mode=0o444, dir_mode=0o555)


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


def _assert_skills_absent_from_formal(
    output_dir: str | Path, models_dir: str | Path | None = None
) -> None:
    """Keep the shared knowledge tree out of every formal strategy revision."""

    roots = (("output", Path(output_dir)),)
    if models_dir is not None:
        roots += (("models", Path(models_dir)),)
    for label, root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (path.is_dir() and path.name == "skills") or (
                path.is_file() and path.name == "SKILL.md"
            ):
                relative = path.relative_to(root).as_posix()
                raise ValueError(
                    f"shared skills cannot enter formal {label}: {label}/{relative}"
                )


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
