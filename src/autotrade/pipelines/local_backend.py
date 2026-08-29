"""Runnable baseline and LLM backends for the daily JSON rolling pipeline."""

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

from autotrade.agent.compact import ContextCompactionConfig, safe_error_summary
from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.environment.artifacts import (
    ArtifactSnapshotUnstable,
    FilesystemArtifactStore,
    ModificationConstraints,
    copy_artifact,
    copy_artifact_snapshot,
    copy_model_artifacts,
    restore_working_artifacts_writable,
)
from autotrade.environment.data.summary import HOST_PATH_RE, write_agent_data_summary
from autotrade.environment.executor import PersistentCommandRunner
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm.model_profiles import AGENT_MAX_OUTPUT_TOKENS
from autotrade.environment.replay.stats import PhaseTimer, finalize_summary_timing
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
    link_copytree,
)
from autotrade.environment.sandbox_images import (
    SANDBOX_ENVIRONMENT_REQUEST_NAME,
    maybe_rebuild_sandbox_image,
    write_sandbox_environment_example,
)
from autotrade.environment.step_tree import StepTree
from autotrade.environment.strategy_loader import validate_strategy_source
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
from autotrade.environment.tools.files import EditFileTool, WriteFileTool
from autotrade.environment.tools.finish_fold import FinishFoldTool
from autotrade.environment.tools.hitl import AskUserTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.search import (
    GlobTool,
    GrepTool,
    ReadFileTool,
    SearchRoots,
)
from autotrade.environment.tools.shell import SandboxShellTool
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.environment.tools.workspace import SafeWorkspace

from .agent_views import compact_fold_history
from .config import (
    ArtifactRevision,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    FoldSessionRequest,
    FoldSessionResult,
    MetaSessionResult,
    SnapshotBundle,
    StepResult,
    StrategyExperimentConfig,
)
from .experiment import DailyStrategyPipeline
from .ledger import ExperimentLedger, latest_fold_records
from .skills import (
    SKILLS_INDEX_PATH,
    DeleteSkillTool,
    WriteSkillTool,
    install_workspace_skills,
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
        fold,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle:
        del fold
        if phase not in {"meta", "valid", "frozen_test", "heldout"}:
            raise ValueError(f"unsupported local snapshot phase: {phase}")
        return SnapshotBundle(
            snapshot_id=f"local_daily_{phase}_{start}_{end}",
            decision_ref=str(self.daily_path),
            replay_ref=str(self.daily_path),
            data_summary_ref="",
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
        self._daily["trade_date"] = self._daily["trade_date"].map(_date_key)

    @property
    def trading_days(self) -> list[str]:
        return sorted(set(self._daily["trade_date"].tolist()))

    def frame_between(self, start: str, end: str) -> pd.DataFrame:
        return self._daily[
            (self._daily["trade_date"] >= _date_key(start))
            & (self._daily["trade_date"] <= _date_key(end))
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
        if request.mode not in {"valid", "frozen_test", "heldout"}:
            raise ValueError(f"unsupported local evaluation mode: {request.mode}")
        strategy_path = Path(request.revision.output_path) / "main.py"
        if not strategy_path.is_file():
            raise FileNotFoundError(
                f"strategy revision has no main.py: {strategy_path}"
            )
        validate_strategy_source(
            strategy_path.read_text(encoding="utf-8"), filename="main.py"
        )
        started_at = utc_now_iso()
        timer = PhaseTimer()
        with timer.phase("replay_frames"):
            frame = self.frame_between(request.start, request.end)
            if max_days is not None:
                kept = sorted(set(frame["trade_date"]))[:max_days]
                frame = frame[frame["trade_date"].isin(set(kept))].copy()
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
        record = replay.to_record()
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
    """Create one Step by replaying the supplied baseline without modifying it."""

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
        validate_strategy_source(
            self.baseline_strategy.read_text(encoding="utf-8"),
            filename=self.baseline_strategy.name,
        )
        self.baseline_root = self.artifact_store.root / "baseline_source"
        self.baseline_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.baseline_strategy, self.baseline_root / "main.py")

    def __call__(self, request: FoldSessionRequest) -> FoldSessionResult:
        source = (
            request.parent.path if request.parent is not None else self.baseline_root
        )
        _assert_skills_absent_from_formal(
            source,
            request.parent.model_path if request.parent is not None else None,
        )
        revision = self.artifact_store.create_revision(
            source,
            models_path=request.parent.model_path
            if request.parent is not None
            else None,
        )
        typed_revision = ArtifactRevision(
            str(revision.revision_id),
            Path(revision.output_path),
            Path(revision.models_path) if revision.models_path is not None else None,
        )
        validation = self.evaluator.evaluate(
            EvaluationRequest(
                revision=typed_revision,
                snapshot=request.snapshot,
                mode="valid",
                start=request.fold.validation_start,
                end=request.fold.validation_end,
                schedule=self.schedule,
                broker_profile=self.broker_profile,
            )
        )
        # Step ids reach the Agent through the ledger's steps[] projection, so
        # they carry the same opaque fold ref every agent-visible surface uses.
        step_id = (
            f"baseline_{self.ref_store.get_or_create('fold', request.fold.fold_id)}__"
            f"{self.ref_store.get_or_create('run', request.run_id)}"
        )
        return FoldSessionResult(
            conversation_id=f"deterministic_baseline_{request.run_id}",
            steps=(
                StepResult(
                    step_id, typed_revision.revision_id, validation, selected=True
                ),
            ),
            selected_step_id=step_id,
            finish_reason="deterministic_baseline_replay_no_agent_improvement",
        )


SESSION_CALL_BUDGET_REFERENCE_MAX = 400
SESSION_SUBAGENT_CALL_CAP_AT_REFERENCE = 200
SESSION_PARENT_MAIN_RESERVE_AT_REFERENCE = 50
SESSION_LLM_CALL_ROLES = ("main", "subagent", "compact")


def session_role_quotas(max_calls: int) -> tuple[int, int]:
    """Sub-agent cumulative cap and parent-main reserve for one shared call budget."""
    if max_calls <= 0:
        raise ValueError("max_calls must be positive")
    subagent_cap = (
        max_calls
        * SESSION_SUBAGENT_CALL_CAP_AT_REFERENCE
        // SESSION_CALL_BUDGET_REFERENCE_MAX
    )
    parent_reserve = (
        max_calls
        * SESSION_PARENT_MAIN_RESERVE_AT_REFERENCE
        // SESSION_CALL_BUDGET_REFERENCE_MAX
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


class SandboxLost(SessionInterrupt):
    """The session sandbox was destroyed while a formal call held it.

    Every later tool call would fail against a container that no longer exists,
    so the session aborts here and the pipeline retries the Fold with a fresh
    one instead of letting the Agent spend its whole budget probing a dead
    sandbox.
    """


def _public_error_text(exc: Exception, *, hidden: Sequence[str] = ()) -> str:
    """The exact failure text, host paths and hidden calendar redacted."""
    text = HOST_PATH_RE.sub("[host_path]", safe_error_summary(exc))
    for value in sorted({item for item in hidden if item}, key=len, reverse=True):
        text = text.replace(value, "[redacted]")
    return text


def _public_validation_error(
    exc: Exception, *, hidden: Sequence[str] = ()
) -> str:
    """Agent-visible Validation failure: type, actionable reason, no host leaks."""
    return f"daily Validation failed: {_public_error_text(exc, hidden=hidden)}"


SMOKE_BACKTEST_DEFAULT_DAYS = 3
SMOKE_BACKTEST_MAX_DAYS = 5


class SmokeBacktestTool:
    """Run the CURRENT working copy through the real replay for a few days.

    Hand-rolled shell smoke tests are what let seven of nine official backtests
    die on day one: a script that sets ``ctx.asof_dir = "/mnt/snapshot"`` and
    fakes an account object exercises the flat frozen snapshot and a dict-like
    account, while the replay hands the strategy a rolling directory-per-domain
    as-of view and a real ``AccountSnapshot``. This tool removes the reason to
    hand-roll one: same executor, same 30 s per-decision cap, same as-of layout,
    same account object, same modification gate — only the window is short.

    It is deliberately NOT an evaluation: no revision is committed, no step-tree
    node is written, nothing here can be selected at freeze time, and it does
    not consume the Fold's backtest budget.
    """

    spec = ToolSpec(
        "smoke_backtest",
        "UNOFFICIAL smoke run of the CURRENT output/ over the first few trading "
        "days of the validation window, on the real replay path: real rolling "
        "as-of view (each context.asof_dir/<domain>/ is a DIRECTORY of parquet "
        "parts, read it with pd.read_parquet(directory)), real AccountSnapshot "
        "object, same sandbox executor and per-decision timeout as "
        "daily_backtest. Returns per-day strategy and as-of seconds, order "
        "counts, the as-of domain directory names, and the exact exception text "
        "on failure. It commits no revision, creates no Step, cannot be frozen, "
        "and does not consume the backtest budget. Use it before daily_backtest "
        "instead of hand-writing a shell smoke test against /mnt/snapshot.",
        {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": SMOKE_BACKTEST_MAX_DAYS,
                    "description": (
                        "Trading days from the start of the validation window "
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
        request: FoldSessionRequest,
        output_dir: Path,
        models_dir: Path,
        modification_check: ModificationCheckTool,
        evaluator: EvaluationBackend,
        schedule,
        broker_profile,
        scratch_root: Path,
    ) -> None:
        self.request = request
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.modification_check = modification_check
        self.evaluator = evaluator
        self.schedule = schedule
        self.broker_profile = broker_profile
        self.scratch_root = Path(scratch_root)
        self.runs = 0

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        days = self._days(arguments)
        self.runs += 1
        check = self.modification_check.invoke({})
        if not check.ok:
            # Same static gate as daily_backtest, so a green smoke run means the
            # gate will not be what fails the official one.
            raise ToolError(f"smoke_backtest blocked by modification_check: {check.error}")
        fold = self.request.fold
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
                EvaluationRequest(
                    revision=revision,
                    snapshot=self.request.snapshot,
                    mode="valid",
                    start=fold.validation_start,
                    end=fold.validation_end,
                    schedule=self.schedule,
                    broker_profile=self.broker_profile,
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
                    "counts_against_backtest_budget": False,
                    "error": _public_error_text(
                        exc, hidden=(fold.fold_id, fold.test_start, fold.test_end)
                    ),
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
            "counts_against_backtest_budget": False,
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
    fold_id: str | None = None,
    views: Mapping[str, tuple[Path, str]] | None = None,
    bundle_ref: str = "",
) -> bool:
    """Publish the data contract a session's run manifest advertises.

    A manifest that declares ``data_summary_ref`` must be backed by a file the
    session can actually read. Meta declared it and never installed it, so a
    Meta session read a missing path and published the false premise
    "data_summary.json is not mounted" into the immutable PRIOR that every later
    Fold reads. One installer for both session kinds, and the caller records the
    ref only when this returns True.

    A Fold builds the summary from its own mounted snapshot view (its fold id is
    opaqued so the calendar period cannot leak). Meta has no mounted snapshot,
    so it installs the prebuilt bundle summary its SnapshotBundle already names
    — the same numbers, for the Meta window's decision snapshot.
    """
    if views:
        write_agent_data_summary(
            paths.data_summary, kind=kind, fold_id=fold_id, views=views
        )
    else:
        source = Path(bundle_ref) if bundle_ref else None
        if source is None or not source.is_file():
            return False
        for name in AGENT_DATA_CONTRACT_FILES:
            candidate = source.parent / name
            if candidate.is_file():
                shutil.copy2(candidate, paths.artifacts / name)
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
    return paths.data_summary.is_file()


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
# observation — an audited Fold shipped 427 KB of ``per_stock`` into the
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


class FoldBacktestTool(SessionTimeBudgetAware):
    """Commit and evaluate the current work copy as one immutable Step."""

    spec = ToolSpec(
        "daily_backtest",
        "Commit the current output as an immutable revision and run the Fold Validation replay.",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def __init__(
        self,
        *,
        request: FoldSessionRequest,
        output_dir: Path,
        models_dir: Path,
        modification_check: ModificationCheckTool,
        artifact_store: FilesystemArtifactStore,
        evaluator: EvaluationBackend,
        tree: StepTree,
        schedule,
        broker_profile,
        time_budget: InferenceTimeBudget,
        ref_store: AgentRefStore,
        manifest: RunManifest | None = None,
        decision_timeout_seconds: float = SandboxLimits().timeout_seconds,
    ) -> None:
        # The executor's per-decision wall clock is the cap the Agent must
        # design for, so it rides in the tool description under its number.
        self.decision_timeout_seconds = float(decision_timeout_seconds)
        self.spec = ToolSpec(
            self.spec.name,
            "Commit the current output as an immutable revision and run the Fold "
            "Validation replay. One trading day's generate_orders inference over "
            f"{self.decision_timeout_seconds:g}s fails the whole backtest "
            "(strategy_inference_timeout_seconds); smoke-test timing on a few days first.",
            self.spec.input_schema,
            mutating=True,
        )
        self.request = request
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.modification_check = modification_check
        self.artifact_store = artifact_store
        self.evaluator = evaluator
        self.tree = tree
        self.schedule = schedule
        self.broker_profile = broker_profile
        self.time_budget = time_budget
        self.ref_store = ref_store
        self.manifest = manifest
        self.backtests = 0
        self.steps: list[StepResult] = []

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.time_budget

    def _append_manifest_summary(self, summary: dict[str, object]) -> None:
        """Every backtest attempt, successful or not, lands in the run manifest.

        It is the only durable per-run record of what the session actually ran:
        the ledger keeps one fold record, and ``compact_fold_history`` reads
        these summaries back out for the next Meta session.
        """
        if self.manifest is not None:
            self.manifest.append_backtest_summary(summary)

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        self._check_deadline()
        with self.time_budget.pause():
            return self._invoke_exempt()

    def _invoke_exempt(self) -> ToolResult:
        if self.backtests >= self.request.max_backtests:
            raise ToolError("Fold Validation backtest budget exhausted")
        if len(self.steps) >= self.request.max_steps:
            raise ToolError("Fold Step budget exhausted")
        # The attempt begins here and costs one Validation slot; only a snapshot
        # that never held is refunded below.
        self.backtests += 1
        result_name = f"valid_{self.backtests:03d}"
        revision_id = ""
        node_id = None
        evaluation = None
        check: ToolResult | None = None
        try:
            check = self.modification_check.invoke({})
            # The replay reads an immutable snapshot of the artifact, never the
            # Agent's live tree, so the session keeps running while it does. The
            # snapshot is verified to be the bytes the check just approved — a
            # write that lands after this line belongs to the next attempt and
            # cannot reach this one's result.
            revision = self.artifact_store.create_revision(
                self.output_dir,
                models_path=self.models_dir,
            )
            if revision.fingerprint != check.value["fingerprint"]:
                self.artifact_store.discard_revision(str(revision.revision_id))
                raise ArtifactSnapshotUnstable(
                    "output/ changed between modification_check and the "
                    "Validation snapshot"
                )
            revision_id = str(revision.revision_id)
            typed = ArtifactRevision(
                revision_id,
                Path(revision.output_path),
                Path(revision.models_path)
                if revision.models_path is not None
                else None,
            )
            _assert_skills_absent_from_formal(typed.output_path, typed.models_path)
            evaluation = self.evaluator.evaluate(
                EvaluationRequest(
                    revision=typed,
                    snapshot=self.request.snapshot,
                    mode="valid",
                    start=self.request.fold.validation_start,
                    end=self.request.fold.validation_end,
                    schedule=self.schedule,
                    broker_profile=self.broker_profile,
                )
            )
            self._check_deadline()
            revision_ref = self.ref_store.get_or_create("strategy", revision_id)
            node_id = self.tree.record_step(
                typed.output_path,
                epoch_id=self.request.epoch_id,
                # Opaque the fold id so the step-tree node names the Agent
                # reads (steps/tree.txt|tree.json) never leak the held-out
                # calendar period.
                fold_id=self.ref_store.get_or_create(
                    "fold", self.request.fold.fold_id
                ),
                run_id=self.ref_store.get_or_create("run", self.request.run_id),
                result_name=result_name,
                revision_id=revision_ref,
                # tree.json is Agent-readable and accumulates one node per
                # Validation for the whole experiment, so it carries the
                # same fixed-size projection the observation does.
                metrics=inline_backtest_stats(evaluation.summary),
                models_root=typed.models_path,
                attachments={
                    VALIDATION_RESULT_ATTACHMENT: evaluation.result_ref
                },
            )
        except SessionInterrupt:
            # The session is over: an environment failure must never be
            # recorded, or reported to the Agent, as a strategy failure.
            raise
        except ArtifactSnapshotUnstable as exc:
            # No snapshot was accepted and no replay ran, so this is
            # infrastructure, not a Validation: it costs neither a backtest slot
            # nor a Step.
            self.backtests -= 1
            fold = self.request.fold
            public_error = _public_error_text(
                exc, hidden=(fold.fold_id, fold.test_start, fold.test_end)
            )
            self._append_manifest_summary(
                {
                    "mode": "valid",
                    "status": "infrastructure_error",
                    "complete_validation": False,
                    "error": public_error,
                }
            )
            raise ToolError(
                f"daily_backtest could not start: {public_error}",
                error_type="infrastructure_error",
                retry_hint=(
                    "The strategy artifact was still being written when the "
                    "Validation snapshot was taken and no backtest ran; no "
                    "budget was consumed. Let any in-container job that writes "
                    "output/ or models/ finish, then call daily_backtest again."
                ),
            ) from exc
        except Exception as exc:
            fold = self.request.fold
            public_error = _public_validation_error(
                exc,
                hidden=(
                    fold.fold_id,
                    fold.test_start,
                    fold.test_end,
                ),
            )
            if self.request.record_failed_attempts:
                self.tree.record_failed_attempt(
                    epoch_id=self.request.epoch_id,
                    fold_id=self.ref_store.get_or_create(
                        "fold", self.request.fold.fold_id
                    ),
                    run_id=self.ref_store.get_or_create("run", self.request.run_id),
                    result_name=result_name,
                    error=public_error,
                    metrics=(
                        {
                            "revision_id": self.ref_store.get_or_create(
                                "strategy", revision_id
                            )
                        }
                        if revision_id
                        else None
                    ),
                )
            self._append_manifest_summary(
                {
                    "result_name": result_name,
                    "mode": "valid",
                    "status": "failed",
                    "complete_validation": False,
                    "error": public_error,
                }
            )
            if isinstance(exc, TimeoutError):
                raise TimeoutError(public_error) from exc
            raise ToolError(public_error) from exc
        assert node_id is not None and evaluation is not None and check is not None
        step = StepResult(node_id, revision_id, evaluation)
        self.steps.append(step)
        self._append_manifest_summary(
            {
                "result_name": result_name,
                "mode": "valid",
                "status": "ok",
                "complete_validation": True,
                # Scalars are cheap enough to keep wholesale for host audit;
                # structured values only earn their place in the manifest when
                # the Agent-visible projection actually carries them.
                **{
                    key: value
                    for key, value in evaluation.summary.items()
                    if not isinstance(value, (dict, list))
                    or key in AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS
                },
            }
        )
        # A returned EvaluationResult is by construction a full-window replay;
        # a partial one raises above and never reaches here, so a successful
        # result with node_id and revision_id is the complete Validation.
        summary = {
            "run_id": self.ref_store.get_or_create("run", self.request.run_id),
            "node_id": node_id,
            "revision_id": self.ref_store.get_or_create("strategy", revision_id),
            "stats": inline_backtest_stats(evaluation.summary),
        }
        directive = ""
        if self.request.step_gate_hook is not None:
            directive = self.request.step_gate_hook(
                len(self.steps), dict(summary)
            )
        # The full record — per-position rows, executions, equity curve — is the
        # attachment ``record_step`` just wrote under the node tree, which the
        # Agent reads through the ``steps`` root. A reference the Agent cannot
        # resolve is worse than none: it costs a round of blind searching.
        public_result_ref = f"{node_id}/{VALIDATION_RESULT_ATTACHMENT}"
        if not (self.tree.root / public_result_ref).is_file():
            raise ToolError(
                "Validation result attachment is missing for the recorded step: "
                f"{public_result_ref}"
            )
        return ToolResult(
            True,
            value={
                **summary,
                "result_root": STEP_TREE_SEARCH_ROOT,
                "result_ref": public_result_ref,
                "result_hint": (
                    "full replay record (per_stock, executions, equity_curve); "
                    f"read it with: read_file root='{STEP_TREE_SEARCH_ROOT}' "
                    f"path='{public_result_ref}'"
                ),
                "modification_check": dict(check.value),
                "step_directive": str(directive),
                "backtests_used": self.backtests,
                "backtests_remaining": self.request.max_backtests - self.backtests,
            },
        )

    def _check_deadline(self) -> None:
        try:
            self.time_budget.check()
        except TimeoutError as exc:
            raise TimeoutError("Fold deadline exceeded") from exc


def build_fold_subagent_tools(
    search_roots: SearchRoots,
    workspace: SafeWorkspace,
    command_runner: CommandRunner,
    modification: ModificationCheckTool,
    smoke: Tool,
) -> list[Tool]:
    """Tools handed to Fold ``SubAgentEngine``: the parent's research and
    write surface plus the unofficial smoke run; no formal backtest."""
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


def build_meta_subagent_tools(search_roots: SearchRoots) -> list[Tool]:
    """Tools handed to Meta ``SubAgentEngine``: read-only audit."""
    return [
        ReadFileTool(search_roots),
        GrepTool(search_roots),
        GlobTool(search_roots),
    ]


class LLMFoldDeveloper:
    """Adapter from the native Agent loop to ``FoldDeveloper``."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        subagent_llm: LLMProxy | None = None,
        compact_llm: LLMProxy | None = None,
        context_compaction: ContextCompactionConfig | None = None,
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
        step_tree_enabled: bool = True,
        fold_exploration_directive: str = "",
        workspace_reference: str = "",
        repo_root: str | Path | None = None,
    ) -> None:
        self.llm = llm
        self.subagent_llm = subagent_llm or llm
        self.compact_llm = compact_llm
        self.context_compaction = context_compaction or ContextCompactionConfig()
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
        self.step_tree_enabled = step_tree_enabled
        self.fold_exploration_directive = fold_exploration_directive
        self.workspace_reference = workspace_reference
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        validate_strategy_source(
            self.baseline_strategy.read_text(encoding="utf-8"),
            filename=self.baseline_strategy.name,
        )

    @property
    def decision_timeout_seconds(self) -> float:
        """The formal executor's per-decision inference wall clock."""

        sandbox = getattr(self.evaluator, "sandbox", None)
        limits = getattr(sandbox, "limits", None) or SandboxLimits()
        return float(limits.timeout_seconds)

    def set_sandbox_spec(self, spec: SandboxSpec) -> None:
        """Adopt the derived image a Meta session just built, for later Folds."""
        self.sandbox_spec = spec

    def __call__(self, request: FoldSessionRequest) -> FoldSessionResult:
        from autotrade.agent.compact import ContextCompactor
        from autotrade.agent.subagent import (
            SubAgentConfig,
            SubAgentEngine,
        )
        from autotrade.agent.prompts import build_system_prompt
        from autotrade.agent.runner import AgentSessionConfig, AgentSessionRunner
        from autotrade.pipelines.agent_inbox import bind_session_inbox

        root = self.runtime_root / request.run_id
        if root.exists():
            raise FileExistsError(f"Fold runtime already exists: {request.run_id}")
        fold_ref = self.ref_store.get_or_create("fold", request.fold.fold_id)
        run_ref = self.ref_store.get_or_create("run", request.run_id)
        trace = AgentTraceWriter(
            agent_trace_path(self.artifact_store.root.parent, request.run_id),
            ids={
                "experiment_id": request.experiment_id,
                "epoch_id": request.epoch_id,
                "fold_id": fold_ref,
                "run_id": run_ref,
                "session_kind": "fold",
            },
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
        # summary accumulates, which is what the next Meta session reads back
        # through compact_fold_history's run_manifest_ref.
        manifest = RunManifest.create(
            paths.run_manifest,
            {
                "experiment_id": request.experiment_id,
                "epoch_id": request.epoch_id,
                # Raw on the host manifest; RunManifest's Agent-visible view and
                # build_experiment_facts both project it through the experiment
                # reference store; projecting here would lose the raw host audit
                # identity and break correlation with the other host artifacts.
                "fold_id": request.fold.fold_id,
                "run_id": request.run_id,
                "session_key": request.session_key,
                "kind": "fold",
                "llm": {
                    "provider": str(getattr(self.llm, "provider", "")),
                    "model": str(getattr(self.llm, "model", "")),
                },
                "conversation_id": request.run_id,
                "runtime_env_ref": "/mnt/artifacts/runtime_env.json",
                "data_summary_ref": "/mnt/artifacts/data_summary.json",
                "fold": {
                    "fold_id": request.fold.fold_id,
                    "input_window": f"{request.fold.input_window_start}..{request.fold.input_window_end}",
                    "validation_period": f"{request.fold.validation_start}..{request.fold.validation_end}",
                    "valid_decision_time": request.fold.valid_decision_time.isoformat(),
                },
                "fold_period": request.fold_period,
                "snapshot_config": dict(request.snapshot_config),
                "snapshots": {
                    "valid_decision_input": {
                        "snapshot_id": request.snapshot.snapshot_id
                    }
                },
                "valid_decision_time": request.fold.valid_decision_time.isoformat(),
                "is_initial_artifact": request.parent is None,
                "parent_strategy_artifact_id": (
                    request.parent.artifact_id if request.parent is not None else None
                ),
                "template_ref": None
                if request.parent is not None
                else "agent_output_template",
                "modification_constraints": request.modification_constraints.to_record(),
                "acceptance_rules": dict(request.acceptance_rules),
                "schedule": self.schedule.to_record(),
                "broker_profile": self.broker_profile.to_record(),
                "nl_failure_policy": request.nl_failure_policy,
                "step_tree_enabled": self.step_tree_enabled,
                "record_failed_attempts": request.record_failed_attempts,
                "epoch_index": request.epoch_index,
                "phase": request.phase,
                "max_steps": request.max_steps,
                "max_backtests_per_fold": request.max_backtests,
                "deadline_seconds": request.deadline_seconds,
                "finalize_before_deadline_seconds": request.finalize_before_deadline_seconds,
                "sandbox_spec": sandbox_spec.to_record(),
                "prior_prompt": request.prior,
                "fold_exploration_directive": self.fold_exploration_directive.strip(),
                "budgets": {
                    "max_steps": request.max_steps,
                    "max_backtests": request.max_backtests,
                    "max_llm_calls": request.max_llm_calls,
                    "deadline_seconds": request.deadline_seconds,
                    "strategy_inference_timeout_seconds": self.decision_timeout_seconds,
                },
            },
            ref_store=self.ref_store,
        )
        workspace_root = paths.workspace
        output_dir = workspace_root / "output"
        models_dir = workspace_root / "models"
        inputs_dir = workspace_root / "inputs"
        source = (
            request.parent.path
            if request.parent is not None
            else self.baseline_strategy.parent
        )
        source_models = (
            request.parent.model_path if request.parent is not None else None
        )
        if request.parent is None and self.baseline_strategy.name != "main.py":
            raise ValueError(
                "baseline strategy file must be named main.py for Fold development"
            )
        copy_artifact(source, output_dir)
        copy_model_artifacts(source_models, models_dir)
        restore_working_artifacts_writable(output_dir, models_dir)
        inputs_dir.mkdir()
        skills_stats = install_workspace_skills(
            request.skills_source_ref or None,
            workspace_root,
            index_path=inputs_dir / "skills_index.json",
        )
        manifest.update(
            skills={
                "index_path": SKILLS_INDEX_PATH,
                "count": skills_stats.count,
                "files": skills_stats.files,
                "bytes": skills_stats.bytes,
            }
        )
        install_workspace_reference(
            workspace_root,
            self.workspace_reference,
            repo_root=self.repo_root,
        )
        # Fold sessions never see frozen Test metrics: only the meta session is
        # allowed that adaptive feedback (docs/pipeline-design.md §3.2).
        history = [
            compact_fold_history(record, ref_store=self.ref_store)
            for record in latest_fold_records(self.ledger.read()).values()
        ]
        _environment_phase(request.progress_hook, "pit_view", request.run_id)
        self._install_snapshot_view(
            local,
            request,
            start=request.fold.input_window_start,
            end=request.fold.validation_end,
        )
        safe = SafeWorkspace(workspace_root)
        # Read-only exploration reaches the PIT views, the inherited parent
        # artifacts, the backtest results and the step lineage, not just the
        # writable workspace.
        search_roots = SearchRoots(safe, paths=paths)
        tree = self._install_step_tree(paths, request.parent)
        sandbox: DockerSandbox | None = None
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
                    labels={
                        "adm.experiment": request.experiment_id,
                        "adm.run": request.run_id,
                    },
                )
                sandbox.start()
                command_runner = PersistentCommandRunner(sandbox)
            # Built once, after the runtime env and the data summary exist, so
            # the prompt and the workspace copy state the same facts.
            if request.prior.strip():
                (inputs_dir / "PRIOR.md").write_text(
                    request.prior.strip() + "\n", encoding="utf-8"
                )
            facts = self._fold_facts(
                request,
                history,
                manifest=manifest,
                paths=paths,
                models_dir=models_dir,
            )
            write_json_atomic(inputs_dir / "fold_context.json", facts)
            chmod_tree(inputs_dir, file_mode=0o444, dir_mode=0o555)

            modification = ModificationCheckTool(
                output_dir,
                parent_dir=source,
                models_dir=models_dir,
                parent_models_dir=source_models,
                constraints=request.modification_constraints,
            )
            time_budget = InferenceTimeBudget(duration_seconds=request.deadline_seconds)
            shared_budget = SessionCallBudget(
                max_calls=request.max_llm_calls,
                time_budget=time_budget,
            )
            backtest = FoldBacktestTool(
                request=request,
                output_dir=output_dir,
                models_dir=models_dir,
                modification_check=modification,
                artifact_store=self.artifact_store,
                evaluator=self.evaluator,
                tree=tree,
                schedule=self.schedule,
                broker_profile=self.broker_profile,
                time_budget=time_budget,
                ref_store=self.ref_store,
                manifest=manifest,
                decision_timeout_seconds=self.decision_timeout_seconds,
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
                modification,
                smoke,
                backtest,
            ]
            if self.step_tree_enabled:
                tools.append(StepRollbackTool(tree, output_dir, models_dir))
            if request.user_question_hook is not None:
                tools.append(
                    AskUserTool(request.user_question_hook, time_budget=time_budget)
                )
            # Matches the opaque fold ref the step tree stores, so the
            # current-session check compares like with like.
            tools.append(
                FinishFoldTool(
                    tree,
                    fold_id=fold_ref,
                    run_id=run_ref,
                    parent_main_py=(
                        (source / "main.py") if request.parent is not None else None
                    ),
                    current_output=output_dir,
                    current_models=models_dir,
                )
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
                build_fold_subagent_tools(
                    search_roots, safe, command_runner, modification, smoke
                )
            )
            subagent = SubAgentEngine(
                llm=subagent_budgeted,
                tools=subagent_tools,
                config=SubAgentConfig(max_tokens=self.max_response_tokens),
                time_budget=time_budget,
            )
            runner = AgentSessionRunner(
                llm=budgeted,
                tools=ToolRegistry(tools),
                system_prompt=build_system_prompt(
                    self.schedule,
                    mode="fold",
                    experiment_facts=facts,
                    phase=request.phase,
                    step_tree_enabled=self.step_tree_enabled,
                    prior_prompt=request.prior,
                    fold_exploration_directive=self.fold_exploration_directive,
                    fold_directive=request.directive,
                ),
                config=AgentSessionConfig(
                    mode="fold",
                    finalize_before_deadline_seconds=(
                        request.finalize_before_deadline_seconds
                    ),
                    deadline_grace_seconds=request.deadline_grace_seconds,
                    max_llm_calls=request.max_llm_calls,
                    max_steps=request.max_steps,
                    deadline_seconds=request.deadline_seconds,
                    max_response_tokens=self.max_response_tokens,
                ),
                compactor=(
                    ContextCompactor(compact_budgeted, self.context_compaction)
                    if compact_budgeted is not None
                    else None
                ),
                subagent=subagent,
                time_budget=time_budget,
                event_sink=_agent_event_sink(
                    trace, request.progress_hook, request.run_id
                ),
                inbox=bind_session_inbox(
                    self.experiment_dir,
                    session_key=request.session_key,
                    run_id=request.run_id,
                ),
            )
            result = runner.run(self._fold_instruction(request))
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
            selected_node = str(result.finish_value.get("node_id") or "")
            selected_revision_ref = str(
                result.finish_value.get("revision_id") or ""
            )
            if not selected_node or not selected_revision_ref:
                raise RuntimeError("Fold Agent did not select a validated revision")
            selected_revision = self.ref_store.resolve(
                "strategy", selected_revision_ref
            )
            steps = tuple(
                StepResult(
                    step.step_id,
                    step.revision_id,
                    step.validation,
                    selected=step.step_id == selected_node,
                )
                for step in backtest.steps
            )
            if selected_revision not in {step.revision_id for step in steps}:
                raise RuntimeError(
                    "finish_fold selected a revision absent from this Fold result"
                )
            manifest.update(
                conversation_id=result.conversation_id, selected_step_id=selected_node
            )
            if self.step_tree_enabled and paths.steps.exists():
                link_copytree(paths.steps, self.experiment_dir / "steps")
            collected = local.collect_artifacts(
                self.artifact_store.root.parent / request.run_id
            )
            return FoldSessionResult(
                result.conversation_id,
                steps,
                selected_node,
                "llm_agent_finish_fold",
                # The collected copy, not the live sandbox tree: the fold ledger
                # record carries it so a later Meta session can still read this
                # run's backtest summaries after the sandbox is cleaned up.
                run_manifest_ref=str(collected / "run_manifest.json"),
                skills_source_ref=str(collected / "workspace" / "skills"),
            )
        except Exception as exc:
            trace.emit(
                "session_error",
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        finally:
            if sandbox is not None:
                sandbox.stop()

    def _install_step_tree(self, paths, parent) -> StepTree:
        """Hand the experiment-level step tree to the fold and mark the start node.

        With the step tree disabled the fold still records its own run nodes —
        ``finish_fold`` selects one of them — but the lineage is not inherited
        from earlier folds and is not published back, so the ablation removes
        the cross-fold memory the knob is about.
        """
        experiment_tree = self.experiment_dir / "steps"
        if self.step_tree_enabled and experiment_tree.exists():
            link_copytree(experiment_tree, paths.steps)
        tree = StepTree(paths.steps)
        if self.step_tree_enabled:
            tree.set_position(
                tree.position_for_step(parent.source_step_id) if parent else None
            )
        return tree

    def _install_snapshot_view(
        self,
        local: LocalSandbox,
        request: FoldSessionRequest,
        *,
        start: str,
        end: str,
    ) -> None:
        source = Path(request.snapshot.decision_ref).resolve(strict=True)
        target = local.paths.current_snapshot
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
                    "period_start": _date_key(start),
                    "period_end": _date_key(end),
                },
            )
            chmod_tree(target, file_mode=0o444, dir_mode=0o555)
        else:  # pragma: no cover - resolve(strict=True) already rejects this
            raise ValueError(
                f"snapshot decision_ref is neither a file nor directory: {source}"
            )
        install_agent_data_contract(
            local.paths,
            kind="fold",
            # Agent-visible: opaque the fold id so the calendar period (e.g.
            # 2022Q1) cannot leak through data_summary.json. Host correlation
            # uses run_id. Same projection as the ledger and step-tree views.
            fold_id=self.ref_store.get_or_create("fold", request.fold.fold_id),
            views={"snapshot": (target, "/mnt/snapshot")},
        )

    def _fold_facts(
        self,
        request: FoldSessionRequest,
        history: list[dict[str, object]],
        *,
        manifest: RunManifest,
        paths,
        models_dir: Path,
    ) -> dict[str, object]:
        """The Agent-visible operational-facts block for this Fold session.

        ``build_experiment_facts`` is the single visibility contract: it reads
        the run manifest, the runtime env and the data summary and projects the
        raw fold id through the experiment reference store. The Fold-side development
        history rides alongside it — the only cross-fold evidence a Fold
        session gets besides the step tree.
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
            "development_history": history,
            "workspace": fold_workspace_map(paths.workspace),
            "forbidden": [
                "current_test",
                "future_data",
                "heldout",
                "external_network",
                "host_control",
            ],
        }

    @staticmethod
    def _fold_instruction(request: FoldSessionRequest) -> str:
        # The researcher's per-Fold directive is a system-prompt section
        # (build_fold_directive_section), not an instruction suffix: it must
        # carry the framing and precedence rules every session sees, and must
        # not be re-stated in the user turn.
        from autotrade.agent.prompts import FOLD_DEFAULT_INSTRUCTION

        return request.prompt_override.strip() or FOLD_DEFAULT_INSTRUCTION


class LLMMetaLearner:
    """Offline Meta adapter that delegates the session to ``MetaLearningAgent``."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        subagent_llm: LLMProxy | None = None,
        compact_llm: LLMProxy | None = None,
        context_compaction: ContextCompactionConfig | None = None,
        baseline_strategy: str | Path,
        artifact_store: FilesystemArtifactStore,
        experiment_dir: str | Path,
        runtime_root: str | Path,
        max_llm_calls: int,
        deadline_seconds: float,
        # One completion ceiling for the parent conversation and its children.
        max_response_tokens: int = AGENT_MAX_OUTPUT_TOKENS,
        meta_learning_directive: str = "",
        fold_exploration_directive: str = "",
        workspace_reference: str = "",
        repo_root: str | Path | None = None,
        regularization_constraints: ModificationConstraints | None = None,
        sandbox_spec: SandboxSpec | None = None,
        use_docker: bool = True,
        rebuild_enabled: bool = True,
        rebuild_timeout_seconds: int = 1800,
        image_keep: int = 3,
        sandbox_spec_sink: Callable[[SandboxSpec], None] | None = None,
    ) -> None:
        self.llm = llm
        self.subagent_llm = subagent_llm or llm
        self.compact_llm = compact_llm
        self.context_compaction = context_compaction or ContextCompactionConfig()
        self.baseline_strategy = Path(baseline_strategy).resolve(strict=True)
        self.artifact_store = artifact_store
        self.experiment_dir = Path(experiment_dir).resolve()
        self.ref_store = AgentRefStore(self.experiment_dir)
        self.runtime_root = Path(runtime_root).resolve()
        self.max_llm_calls = max_llm_calls
        self.deadline_seconds = deadline_seconds
        # Derived-image rebuild: a Meta session may declare stable dependencies
        # that later ordinary Folds inherit. The new tag reaches those folds
        # through ``sandbox_spec_sink``.
        self.sandbox_spec = sandbox_spec or SandboxSpec()
        self.use_docker = use_docker
        self.rebuild_enabled = rebuild_enabled
        self.rebuild_timeout_seconds = rebuild_timeout_seconds
        self.image_keep = image_keep
        self.sandbox_spec_sink = sandbox_spec_sink
        self.max_response_tokens = max_response_tokens
        self.meta_learning_directive = meta_learning_directive
        self.fold_exploration_directive = fold_exploration_directive
        self.workspace_reference = workspace_reference
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        # The limits a Meta regularization must satisfy before the Pipeline will
        # freeze it; published in the run manifest and enforced by the check.
        self.regularization_constraints = (
            regularization_constraints or ModificationConstraints()
        )

    def __call__(self, facts: dict[str, object]) -> MetaSessionResult:
        from autotrade.agent.compact import ContextCompactor
        from autotrade.agent.subagent import (
            SubAgentConfig,
            SubAgentEngine,
        )
        from autotrade.agent.prompts import (
            build_meta_learning_prompt,
            build_system_prompt,
        )
        from autotrade.agent.runner import (
            AgentSessionConfig,
            AgentSessionRunner,
            MetaLearningAgent,
        )
        from autotrade.environment.tools.finish_meta import FinishMetaTool
        from autotrade.environment.tools.prior_policy import visible_window_dates
        from autotrade.pipelines.agent_inbox import bind_session_inbox

        run_id = str(facts.get("run_id") or f"meta_{uuid.uuid4().hex}")
        root = self.runtime_root / run_id
        if root.exists():
            raise FileExistsError(f"Meta runtime already exists: {run_id}")
        experiment_id = str(facts.get("experiment_id") or "")
        epoch_id = str(facts.get("epoch_id") or "")
        session_id = str(facts.get("meta_learning_id") or "")
        progress_hook = facts.get("progress_hook")
        if progress_hook is not None and not callable(progress_hook):
            raise TypeError("progress_hook must be callable")
        meta_ref = self.ref_store.get_or_create("meta", session_id)
        run_ref = self.ref_store.get_or_create("run", run_id)
        trace = AgentTraceWriter(
            agent_trace_path(self.artifact_store.root.parent, run_id),
            ids={
                "experiment_id": experiment_id,
                "epoch_id": epoch_id,
                "fold_id": meta_ref,
                "run_id": run_ref,
                "session_kind": "meta_learning",
            },
        )
        _environment_phase(progress_hook, "sandbox_layout", run_id)
        local = LocalSandbox(root)
        paths = local.prepare_layout()
        # Meta reasons about the data contract too, so it gets the same
        # data_summary.json / unit_reference.json a Fold gets — from the Meta
        # window's own bundle. The manifest records the ref only if this lands.
        data_contract_installed = install_agent_data_contract(
            paths,
            kind="meta_learning",
            bundle_ref=str(facts.get("data_summary_ref") or ""),
        )
        # Only a Meta session may declare new sandbox dependencies, so only a
        # Meta workspace carries the request format example.
        write_sandbox_environment_example(paths.workspace)
        install_workspace_reference(
            paths.workspace,
            self.workspace_reference,
            repo_root=self.repo_root,
        )
        safe = SafeWorkspace(paths.workspace)
        search_roots = SearchRoots(safe, paths=paths)
        inputs = paths.workspace / "inputs"
        inputs.mkdir()
        skills_stats = install_workspace_skills(
            str(facts.get("skills_source_ref") or "") or None,
            paths.workspace,
            index_path=inputs / "skills_index.json",
        )
        parent_id = str(facts.get("parent_artifact_id") or "")
        if parent_id:
            if Path(parent_id).name != parent_id or parent_id.startswith("."):
                raise ValueError("parent_artifact_id must be one local path component")
            parent_root = (self.artifact_store.frozen_root / parent_id).resolve(
                strict=True
            )
            if not parent_root.is_relative_to(self.artifact_store.frozen_root):
                raise ValueError("parent artifact escaped the configured store")
            parent = (parent_root / "output").resolve(strict=True)
            parent_models = parent_root / "models"
            parent_models = parent_models if parent_models.is_dir() else None
        else:
            parent = self.baseline_strategy.parent
            parent_models = None
        # The parent is the Meta session's WORKING copy, not a read-only input:
        # a Meta session may regularize the strategy artifact within
        # regularization_constraints, and the Pipeline freezes the result.
        output_dir = paths.workspace / "output"
        models_dir = paths.workspace / "models"
        copy_artifact(parent, output_dir)
        copy_model_artifacts(parent_models, models_dir)
        restore_working_artifacts_writable(output_dir, models_dir)
        previous_prior = str(facts.get("previous_prior") or "").strip()
        # PRIOR.md is the sole writable Meta direction/memory channel. Seed the
        # current published body before the Agent starts; the first session gets
        # an empty file and must make it non-empty before finish_meta succeeds.
        (paths.workspace / "PRIOR.md").write_text(
            previous_prior + ("\n" if previous_prior else ""), encoding="utf-8"
        )
        public = {
            key: value
            for key, value in facts.items()
            if key
            not in {
                "user_question_hook",
                "progress_hook",
                "meta_learning_memory",
                "previous_prior",
                "session_key",
                "agent_trace_sidecars",
                "skills_source_ref",
                "host_visible_fold",
            }
        }
        public["run_id"] = run_ref
        public["meta_learning_id"] = meta_ref
        if parent_id:
            public["parent_artifact_id"] = self.ref_store.get_or_create(
                "strategy", parent_id
            )
        from autotrade.pipelines.meta_inputs import (
            AgentTraceFullSidecar,
            write_meta_agent_trace_sidecars,
        )

        raw_sidecars = facts.get("agent_trace_sidecars") or ()
        if not isinstance(raw_sidecars, (list, tuple)):
            raise TypeError("agent_trace_sidecars must be a sequence")
        sidecars: list[AgentTraceFullSidecar] = []
        for item in raw_sidecars:
            if not isinstance(item, AgentTraceFullSidecar):
                raise TypeError(
                    "agent_trace_sidecars must be AgentTraceFullSidecar values"
                )
            sidecars.append(item)
        # Materialize and fsync the referenced sidecars before publishing the
        # atomic context index, so a crash cannot leave metadata pointing at a
        # missing or partially-written file.
        write_meta_agent_trace_sidecars(paths.workspace, sidecars)
        write_json_atomic(inputs / "meta_context.json", public)
        # Raw prior Meta traces, bounded by meta_memory_max_epochs: a JSONL file
        # rather than a prompt field, because it is line-oriented and can be
        # long. Empty memory still writes the file so a first Epoch reads an
        # empty one instead of guessing.
        memory_path = inputs / "meta_learning_memory.jsonl"
        memory_path.write_text(
            str(facts.get("meta_learning_memory") or ""), encoding="utf-8"
        )
        chmod_tree(inputs, file_mode=0o444, dir_mode=0o555)
        host_visible_fold = (
            facts.get("host_visible_fold")
            if isinstance(facts.get("host_visible_fold"), dict)
            else {}
        )
        manifest = RunManifest.create(
            paths.run_manifest,
            {
                "experiment_id": experiment_id,
                "epoch_id": epoch_id,
                "meta_learning_id": session_id,
                "trigger_after_folds": facts.get("trigger_after_folds"),
                # Raw on the host manifest; both Agent-visible projections use
                # the experiment reference store themselves.
                "fold_id": session_id,
                "run_id": run_id,
                "session_key": str(facts.get("session_key") or ""),
                "kind": "meta_learning",
                "llm": {
                    "provider": str(getattr(self.llm, "provider", "")),
                    "model": str(getattr(self.llm, "model", "")),
                },
                "runtime_env_ref": "/mnt/artifacts/runtime_env.json",
                # Never advertise a path the session cannot read: a Meta run
                # that found this missing published the absence as a data fact
                # into PRIOR.
                **(
                    {"data_summary_ref": "/mnt/artifacts/data_summary.json"}
                    if data_contract_installed
                    else {}
                ),
                "meta_learning_visible_fold": dict(host_visible_fold),
                "valid_decision_time": host_visible_fold.get("valid_decision_time"),
                "snapshots": {
                    "valid_decision_input": {"snapshot_id": facts.get("snapshot_id")},
                },
                "parent_strategy_artifact_id": parent_id or None,
                "template_ref": None if parent_id else "agent_output_template",
                "is_initial_artifact": not parent_id,
                # Agent-facing manifest: sandbox mount paths, never host paths.
                "development_inputs": {
                    "meta_context": "/mnt/agent/workspace/inputs/meta_context.json",
                    "meta_learning_memory": "/mnt/agent/workspace/inputs/meta_learning_memory.jsonl",
                    "agent_traces": "/mnt/agent/workspace/inputs/agent_traces",
                    "agent_trace_full": {
                        "directory": "/mnt/agent/workspace/inputs/agent_traces",
                        "available": sum(
                            1 for item in sidecars if item.available
                        ),
                        "fold_count": len(sidecars),
                        "refs": [
                            {
                                "path": item.relative_path,
                                "available": item.available,
                            }
                            for item in sidecars
                        ],
                    },
                    "strategy_working_copy": "/mnt/agent/workspace/output",
                    "model_working_copy": "/mnt/agent/workspace/models",
                    "previous_prior": bool(previous_prior),
                },
                "prior_output": "/mnt/agent/workspace/PRIOR.md",
                "skills": {
                    "index_path": SKILLS_INDEX_PATH,
                    "count": skills_stats.count,
                    "files": skills_stats.files,
                    "bytes": skills_stats.bytes,
                },
                "modification_constraints": replace(
                    self.regularization_constraints, is_initial_artifact=not parent_id
                ).to_record(),
                "meta_learning_directive": self.meta_learning_directive.strip(),
                "fold_exploration_directive": self.fold_exploration_directive.strip(),
                "review_window": (
                    dict(public["review_window"])
                    if isinstance(public.get("review_window"), dict)
                    else {
                        "previous_meta_ref": None,
                        "fold_run_refs": [],
                        "fold_count": 0,
                    }
                ),
                "budgets": {
                    "max_llm_calls": self.max_llm_calls,
                    "deadline_seconds": self.deadline_seconds,
                },
            },
            ref_store=self.ref_store,
        )
        time_budget = InferenceTimeBudget(duration_seconds=self.deadline_seconds)
        shared_budget = SessionCallBudget(
            max_calls=self.max_llm_calls,
            time_budget=time_budget,
        )
        budgeted = SessionBudgetLLM(self.llm, budget=shared_budget, role="main")
        compact_budgeted = (
            SessionBudgetLLM(self.compact_llm, budget=shared_budget, role="compact")
            if self.compact_llm is not None
            else None
        )
        subagent_budgeted = SessionBudgetLLM(
            self.subagent_llm, budget=shared_budget, role="subagent"
        )
        subagent = SubAgentEngine(
            llm=subagent_budgeted,
            tools=ToolRegistry(build_meta_subagent_tools(search_roots)),
            config=SubAgentConfig(max_tokens=self.max_response_tokens),
            time_budget=time_budget,
            mode="meta",
        )
        modification = ModificationCheckTool(
            output_dir,
            parent_dir=parent,
            models_dir=models_dir,
            parent_models_dir=parent_models,
            constraints=replace(
                self.regularization_constraints, is_initial_artifact=not parent_id
            ),
        )
        tools: list[Tool] = [
            ReadFileTool(search_roots),
            GrepTool(search_roots),
            GlobTool(search_roots),
            # The Meta session's writable surface: PRIOR.md, the strategy/model
            # copy it may regularize, and the optional sandbox dependency request.
            WriteFileTool(safe),
            EditFileTool(safe),
            WriteSkillTool(safe),
            DeleteSkillTool(safe),
            modification,
        ]
        hook = facts.get("user_question_hook")
        if hook is not None:
            if not callable(hook):
                raise TypeError("user_question_hook must be callable")
            tools.append(AskUserTool(hook, time_budget=time_budget))
        tools.append(
            FinishMetaTool(
                safe,
                window_dates=visible_window_dates(manifest.data),
            )
        )
        instruction = str(
            public.get("prompt_override") or ""
        ).strip() or build_meta_learning_prompt(
            public.get("development_history")
            if isinstance(public.get("development_history"), dict)
            else {},
            experiment_directive=self.meta_learning_directive,
            fold_exploration_directive=self.fold_exploration_directive,
        )
        directive = str(public.get("directive") or "").strip()
        if directive:
            instruction += f"\n\nSupervising user directive:\n{directive}"
        runner = AgentSessionRunner(
            llm=budgeted,
            tools=ToolRegistry(tools),
            system_prompt=build_system_prompt(
                mode="meta",
                experiment_facts=build_experiment_facts(
                    manifest=manifest.data, ref_store=self.ref_store
                ),
            ),
            config=AgentSessionConfig(
                mode="meta",
                max_llm_calls=self.max_llm_calls,
                deadline_seconds=self.deadline_seconds,
                max_response_tokens=self.max_response_tokens,
            ),
            compactor=(
                ContextCompactor(compact_budgeted, self.context_compaction)
                if compact_budgeted is not None
                else None
            ),
            subagent=subagent,
            time_budget=time_budget,
            event_sink=_agent_event_sink(
                trace,
                progress_hook,
                run_id,
                include_content=False,
            ),
            inbox=bind_session_inbox(
                self.experiment_dir,
                session_key=str(facts.get("session_key") or ""),
                run_id=run_id,
            ),
        )
        try:
            result = MetaLearningAgent(runner, paths.workspace).learn(instruction)
            chmod_tree(inputs, file_mode=0o644, dir_mode=0o755)
            final_skills = write_skills_index(
                paths.workspace / "skills", inputs / "skills_index.json"
            )
            chmod_tree(inputs, file_mode=0o444, dir_mode=0o555)
            manifest.update(
                conversation_id=result.get("conversation_id"),
                skills={
                    "index_path": SKILLS_INDEX_PATH,
                    "count": final_skills.count,
                    "files": final_skills.files,
                    "bytes": final_skills.bytes,
                },
            )
            # Runs after the session returns, so it is Pipeline finalization
            # rather than an Agent action: whatever the Meta left in output/ and
            # models/ must satisfy the regularization constraints before it can
            # become the next Fold's parent.
            _environment_phase(progress_hook, "meta_finalize", run_id)
            check, allowed = _finalize_modification_check(modification)
            manifest.update(last_modification_check=check)
            revision_id = ""
            if parent_id and allowed and _check_has_changes(check):
                _assert_skills_absent_from_formal(output_dir, models_dir)
                validate_strategy_source(
                    (output_dir / "main.py").read_text(encoding="utf-8"),
                    filename="main.py",
                )
                revision = self.artifact_store.create_revision(
                    output_dir, models_path=models_dir
                )
                revision_id = str(revision.revision_id)
            _environment_phase(progress_hook, "environment_update", run_id)
            rebuild_error: RuntimeError | None = None
            try:
                _update, active_spec = maybe_rebuild_sandbox_image(
                    paths.workspace / SANDBOX_ENVIRONMENT_REQUEST_NAME,
                    base_spec=self.sandbox_spec,
                    experiment_id=experiment_id,
                    epoch_id=session_id or epoch_id,
                    experiment_dir=self.experiment_dir,
                    manifest=manifest,
                    use_docker=self.use_docker,
                    rebuild_enabled=self.rebuild_enabled,
                    timeout_seconds=self.rebuild_timeout_seconds,
                    image_keep=self.image_keep,
                )
            except RuntimeError as exc:
                rebuild_error = exc
            else:
                if active_spec is not self.sandbox_spec:
                    self.sandbox_spec = active_spec
                    if self.sandbox_spec_sink is not None:
                        self.sandbox_spec_sink(active_spec)
            # Collect first, then fail: PRIOR and the rebuild record must
            # survive a rebuild failure.
            collected = local.collect_artifacts(
                self.artifact_store.root.parent / run_id
            )
            if rebuild_error is not None:
                raise rebuild_error
            return MetaSessionResult(
                prior=str(result["prior"]),
                conversation_id=str(result.get("conversation_id") or ""),
                revision_id=revision_id,
                modification_check=check,
                allowed=allowed,
                skills_source_ref=str(collected / "workspace" / "skills"),
            )
        except Exception as exc:
            trace.emit(
                "session_error",
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise


_WORKSPACE_REFS_DIR = "refs"
_REFERENCE_SKIP_NAMES = frozenset({".git", "__pycache__", "node_modules", ".venv"})
_REFERENCE_PDF_MAX_BYTES = 256 * 1024


def fold_workspace_map(workspace: str | Path) -> dict[str, str]:
    """Agent-visible workspace index. ``refs`` is omitted when the directory is absent."""
    mapping = {
        "strategy": "output/main.py",
        "models": "models/",
        "fold_context": "inputs/fold_context.json",
        "data_summary": "/mnt/artifacts/data_summary.json",
        "snapshot_in_sandbox": "/mnt/snapshot",
    }
    if (Path(workspace) / _WORKSPACE_REFS_DIR).is_dir():
        mapping["refs"] = "refs/"
    if (Path(workspace) / "inputs" / "PRIOR.md").is_file():
        mapping["prior"] = "inputs/PRIOR.md"
    return mapping


def install_workspace_reference(
    workspace: str | Path,
    workspace_reference: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Copy optional operator notes into ``workspace/refs/`` before sandbox start.

    An empty ``workspace_reference`` is a no-op. A set path must exist and be a
    directory, otherwise this fails immediately. The copy writes only ``refs/``,
    never ``output/``, ``models/``, or ``inputs/``. Each Fold/Meta session has a
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


def _agent_event_sink(
    trace: AgentTraceWriter,
    progress_hook,
    run_id: str,
    *,
    include_content: bool = True,
):
    def emit(event_type: str, payload: dict[str, object]) -> None:
        trace.emit(
            event_type,
            payload
            if include_content
            else _safe_meta_trace_payload(event_type, payload),
        )
        if progress_hook is None:
            return
        if event_type == "llm_call_started":
            stage = "llm_call"
        elif event_type == "tool_call_started":
            stage = (
                "backtest"
                if payload.get("tool") in ("daily_backtest", "smoke_backtest")
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


def _safe_meta_trace_payload(
    event_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Keep Meta operations observable without exposing compact Test evidence.

    Sub-agent events keep their identity, progress and usage fields (the
    console cards, ``trace_stats`` and the Meta process summary need them) plus
    the parent-written brief (``description``, ``task``) that delegation
    quality is judged by, already clipped where it is emitted; the child's own
    text (``content``, ``summary``, tool ``result`` bodies) is dropped like the
    parent ``llm_call`` content. The session prompts are built before any Test
    data is read, so ``session_start`` keeps them, and every tool call keeps
    its ``ok``/``error_type``/``error`` status so a failing Meta session stays
    auditable.
    """

    subagent_identity = {
        "task_id",
        "role",
        "parent_call_id",
        "status",
        "mode",
        "model",
        "thinking",
        "thinking_applied",
        "rounds_limit",
        "inherit_context",
        "description",
        "resumed_from",
    }
    allowed = {
        "session_start": {"mode", "system_prompt", "instruction"},
        "subagent_task": subagent_identity | {"task"},
        "subagent_llm": {
            "task_id",
            "role",
            "round",
            "provider",
            "model",
            "usage",
            "tool_names",
            "parent_call_id",
        },
        "subagent_llm_error": {
            "task_id",
            "role",
            "round",
            "provider",
            "model",
            "llm_error",
            "parent_call_id",
        },
        "subagent_tool": {"task_id", "role", "round", "tool", "parent_call_id", "result"},
        "subagent_wrap_up": {"task_id", "role", "round", "rounds_limit", "parent_call_id"},
        # A parent instruction is counted, never quoted.
        "subagent_steer": {"task_id", "role", "round", "chars", "delivery", "parent_call_id"},
        "subagent_context_compaction": {
            "task_id",
            "role",
            "round",
            "parent_call_id",
            "compaction",
        },
        "subagent_context_edit": {
            "task_id",
            "role",
            "round",
            "parent_call_id",
            "context_edit",
        },
        "subagent_output_truncated": {
            "task_id",
            "role",
            "round",
            "completion_tokens",
            "max_tokens",
            "continuation",
            "parent_call_id",
        },
        "subagent": subagent_identity
        | {
            "rounds",
            "tool_calls",
            "llm_calls",
            "provider",
            "usage_totals",
            "error",
            "truncated",
            "truncated_rounds",
            "llm_errors",
        },
        "subagent_attempt": {
            "attempt",
            "role",
            "ok",
            "status",
            "task_id",
            "error",
            # Report delivery is a count-and-reference record, not the child's
            # own text: how much it wrote, how much the parent got, and where
            # the rest was spilled.
            "summary_chars",
            "summary_delivered_chars",
            "summary_lines",
            "resume_line",
            "summary_truncated",
            "result_ref",
        },
        "delegation_reminder": {"own_work_calls", "running_children", "queued_children"},
        "output_truncated": {"call_index", "completion_tokens", "max_tokens"},
        "llm_call_started": {"call_index", "status"},
        "llm_call": {"call_index", "status", "model", "usage", "tool_names", "error"},
        "tool_call_started": {"tool", "tool_call_id", "status"},
        "tool_call": {"call_index", "tool_call_id", "tool", "result"},
        "session_end": {"status", "llm_calls", "steps_used"},
        "user_message": {
            "message_id",
            "interrupt",
            "applied_at",
            "safe_point",
            "content",
        },
        "tool_skipped": {
            "tool_call_id",
            "tool",
            "reason",
            "message_id",
            "safe_point",
        },
    }.get(event_type, {"status", "error"})
    kept = {key: value for key, value in payload.items() if key in allowed}
    result = kept.get("result")
    if event_type in {"subagent_tool", "tool_call"} and isinstance(result, dict):
        # Status only: a tool result body may carry compact Test evidence.
        status = {key: result[key] for key in ("ok", "error") if key in result}
        value = result.get("value")
        if isinstance(value, dict) and "error_type" in value:
            status["error_type"] = value["error_type"]
        kept["result"] = status
    compaction = kept.get("compaction")
    if event_type == "subagent_context_compaction" and isinstance(compaction, dict):
        # Shape only, like the parent's own compaction record in a Meta trace:
        # the summary text is the child's compacted history.
        kept["compaction"] = {
            key: compaction[key]
            for key in (
                "status",
                "error",
                "skip_reason",
                "estimated_tokens",
                "messages_before",
                "messages_after",
                "summary_chars",
            )
            if key in compaction
        }
    # Bounded shape-only signals so a Meta child's usefulness stays auditable
    # without any model text: how long its summary/content was and which
    # argument keys the parent used.
    if event_type == "subagent" and isinstance(payload.get("summary"), str):
        kept["summary_chars"] = len(payload["summary"])
    if event_type in {"llm_call", "subagent_llm"} and isinstance(payload.get("content"), str):
        kept["content_chars"] = len(payload["content"])
    arguments = payload.get("arguments")
    if event_type in {"tool_call", "subagent_tool"} and isinstance(arguments, Mapping):
        kept["argument_keys"] = sorted(str(key) for key in arguments)
    return kept


def _finalize_modification_check(
    tool: ModificationCheckTool,
) -> tuple[dict[str, object], bool]:
    """Run the post-session check, turning a refusal into an audited verdict.

    A Meta session that leaves the artifact outside its constraints must not
    fail the whole session — PRIOR is still valid — but its edits must be
    rejected rather than frozen, so the refusal is recorded and reported.
    """
    try:
        result = tool.invoke({})
    except ToolError as exc:
        return {"allowed_to_backtest": False, "reasons": [str(exc)]}, False
    return {"allowed_to_backtest": True, **dict(result.value)}, True


def _check_has_changes(check: Mapping[str, object]) -> bool:
    delta = check.get("delta")
    model_delta = check.get("model_delta")
    if not isinstance(delta, Mapping):
        delta = {}
    if not isinstance(model_delta, Mapping):
        model_delta = {}
    return any(
        [
            int(delta.get("changed_file_count") or 0) > 0,
            int(delta.get("diff_lines") or 0) > 0,
            int(delta.get("code_diff_lines") or 0) > 0,
            int(model_delta.get("changed_file_count") or 0) > 0,
        ]
    )


def _read_json_if_exists(path: Path) -> dict[str, object]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _date_key(value: object) -> str:
    return pd.Timestamp(str(value)).strftime("%Y%m%d")


__all__ = [
    "DeterministicBaselineDeveloper",
    "FilesystemArtifactStore",
    "fold_workspace_map",
    "LLMFoldDeveloper",
    "LLMMetaLearner",
    "LocalDailyEvaluationBackend",
    "LocalDailySnapshotProvider",
    "SESSION_CALL_BUDGET_REFERENCE_MAX",
    "SESSION_SUBAGENT_CALL_CAP_AT_REFERENCE",
    "SESSION_PARENT_MAIN_RESERVE_AT_REFERENCE",
    "SessionBudgetLLM",
    "SessionCallBudget",
    "session_role_quotas",
]
