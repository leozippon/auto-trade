"""Configuration for one scheduled strategy experiment."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import MISSING, KW_ONLY, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from autotrade.environment.artifacts import ModificationConstraints
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.sandbox import SandboxConfig, SandboxLimits
from autotrade.environment.strategy import StrategySchedule

from .skills import DEFAULT_OPERATING_MEMORY
from .folds import FoldSpec, assert_no_overlap

ExecutionMode = Literal["sandbox", "trusted"]

# Increment only when the cached snapshot/replay on-disk contract changes.
# Source revisions are intentionally not cache inputs: harmless code changes
# should not invalidate every expensive data view.
# v6: events/macro unions carry the full configured-dataset schema (typed
# zero-row contributions for datasets without visible rows in the window).
# v7: the snapshot manifest carries dataset_columns for the unit reference.
# v8: incomplete/unusable vendor columns are dropped from the union domains
# (SNAPSHOT_EXCLUDED_COLUMNS in environment.data.snapshot), so a v7 view still
# carries fields the Agent must no longer see.
# v9: the daily join drops its duplicate close_basic/pre_close_limit columns.
SNAPSHOT_CACHE_FORMAT_VERSION = 9

# Trailing wrap-up grace added to the Fold session budget. Not a console/worker
# HITL knob; FoldSessionRequest carries the seconds to AgentSessionConfig.
DEFAULT_DEADLINE_GRACE_MINUTES = 10

# Research calendar cadence: the unit the development and held-out labels are
# written in. The default development window is whole years, long enough to
# contain more than one market state.
DEFAULT_FOLD_PERIOD = "year"


@dataclass(frozen=True)
class StrategyExperimentConfig:
    strategy_path: Path
    schedule: StrategySchedule = field(default_factory=StrategySchedule)
    broker_profile: BrokerProfile = field(default_factory=BrokerProfile)
    execution_mode: ExecutionMode = "sandbox"
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    # The strategy's frozen ``models/`` tree, mounted read-only when it has
    # one. The per-replay ``fit`` state directory is not a configuration knob:
    # the replay creates it empty and discards it with the run.
    models_dir: Path | None = None

    def __post_init__(self) -> None:
        path = Path(self.strategy_path).resolve()
        if not path.is_file():
            raise ValueError(f"strategy file does not exist: {path}")
        if self.execution_mode not in ("sandbox", "trusted"):
            raise ValueError("execution_mode must be sandbox or trusted")
        object.__setattr__(self, "strategy_path", path)
        if self.models_dir is not None:
            models = Path(self.models_dir).resolve()
            if not models.is_dir():
                raise ValueError(f"models directory does not exist: {models}")
            object.__setattr__(self, "models_dir", models)


# The one-line summary of AcceptanceRules.evaluate handed to the Agent as the
# ``acceptance_semantics`` fact. It lives beside the rules it describes so the
# projection cannot drift from the behavior.
ACCEPTANCE_SEMANTICS_SUMMARY = (
    "max_drawdown+finite_metrics=hard; return/sharpe=warn-only targets"
)


@dataclass(frozen=True)
class AcceptanceRules:
    """Validation acceptance checks (docs/pipeline-design.md §2.2): drawdown and
    finiteness are HARD rejects; min_return/min_sharpe are warn-only targets — a
    shortfall records a warning and never resets the fold."""

    min_return: float = 0.0
    min_sharpe: float = 0.0
    max_drawdown: float = 0.25

    def __post_init__(self) -> None:
        for name in ("min_return", "min_sharpe", "max_drawdown"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be between zero and one")

    def to_record(self) -> dict[str, object]:
        return {
            "min_return": self.min_return,
            "min_sharpe": self.min_sharpe,
            "max_drawdown": self.max_drawdown,
        }

    def heldout_verdict(
        self,
        summary: Mapping[str, object] | None,
        walk_forward: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Graduation verdict of one Held-out replay (docs/pipeline-design.md §3.3).

        ``graduated`` iff (a) the frozen strategy beat its benchmark on
        Held-out (excess return > 0), earned a positive annualized Sharpe, and
        stayed within the experiment's ``max_drawdown``, and (b) the final
        Epoch's walk-forward transitions (``ledger.walk_forward_transitions``)
        show a positive excess return in at least two thirds of them (rounded
        up). Otherwise ``discarded`` with every failing reason. A missing or
        non-finite input is itself a failing reason: a replay that cannot
        prove the conditions did not pass. Term (b) is ``not_applicable`` when
        the schedule has no transitions, and the verdict then rests on (a).
        """
        reasons: list[str] = []
        values: dict[str, float | None] = {}
        source = summary if isinstance(summary, Mapping) else {}
        if not source or source.get("status") == "failed":
            reasons.append("heldout_failed")
        benchmark = source.get("benchmark")
        benchmark_return = (
            benchmark.get("benchmark_return") if isinstance(benchmark, Mapping) else None
        )
        for name, value in (
            ("total_return", source.get("total_return")),
            ("benchmark_return", benchmark_return),
            ("sharpe", source.get("sharpe")),
            ("max_drawdown", source.get("max_drawdown")),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                values[name] = None
                reasons.append(f"missing_{name}")
            else:
                values[name] = float(value)
        total_return, bench, sharpe, drawdown = (
            values["total_return"],
            values["benchmark_return"],
            values["sharpe"],
            values["max_drawdown"],
        )
        excess = (
            total_return - bench if total_return is not None and bench is not None else None
        )
        if excess is not None and excess <= 0:
            reasons.append("excess_return_not_positive")
        if sharpe is not None and sharpe <= 0:
            reasons.append("sharpe_not_positive")
        if drawdown is not None and abs(drawdown) > self.max_drawdown:
            reasons.append("max_drawdown_exceeded")
        consistency = self.walk_forward_consistency(walk_forward)
        if consistency["status"] == "inconsistent":
            reasons.append(
                "walkforward_excess_inconsistent("
                f"{consistency['positive_excess']}/{consistency['transitions']}"
                f"<{consistency['required']})"
            )
        return {
            "status": "discarded" if reasons else "graduated",
            "reasons": reasons,
            "excess_return": excess,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "max_drawdown_limit": self.max_drawdown,
            "walk_forward": consistency,
        }

    @staticmethod
    def walk_forward_consistency(
        walk_forward: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Term (b) of graduation: positive excess in >= ceil(2/3) of transitions."""
        transitions = int((walk_forward or {}).get("transitions") or 0)
        if transitions <= 0:
            return {"status": "not_applicable", "transitions": 0}
        positive = int((walk_forward or {}).get("positive_excess") or 0)
        required = math.ceil(2 * transitions / 3)
        return {
            "status": "consistent" if positive >= required else "inconsistent",
            "source": str((walk_forward or {}).get("source") or ""),
            "transitions": transitions,
            "positive_excess": positive,
            "required": required,
        }

    def evaluate(self, summary: dict[str, object]) -> tuple[list[str], list[str]]:
        """(hard_reasons, warnings). Integrity failures are hard rejects:
        non-finite metrics (every IEEE comparison against NaN is False, so a NaN
        metric would otherwise pass all thresholds). The max_drawdown cap stays a
        hard risk limit. Return/Sharpe shortfalls are WARNINGS only — the fold
        still freezes its validated update; a weak step recorded with a warning
        beats silently resetting the fold chain. Only a summary from a completed
        full-window evaluation reaches here; an aborted replay never produces
        one."""
        hard: list[str] = []
        warnings: list[str] = []
        values: dict[str, float] = {}
        for key in ("total_return", "max_drawdown"):
            value = summary.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                hard.append(f"non_finite_{key}")
            else:
                values[key] = float(value)
        sharpe = summary.get("sharpe")
        if sharpe is not None:
            if (
                isinstance(sharpe, bool)
                or not isinstance(sharpe, (int, float))
                or not math.isfinite(float(sharpe))
            ):
                hard.append("non_finite_sharpe")
            else:
                values["sharpe"] = float(sharpe)
        if abs(values.get("max_drawdown", 0.0)) > self.max_drawdown:
            hard.append("max_drawdown_exceeded")
        if values.get("total_return", float("-inf")) < self.min_return:
            warnings.append("return_below_target")
        if "sharpe" in values and values["sharpe"] < self.min_sharpe:
            warnings.append("sharpe_below_target")
        return hard, warnings


@dataclass(frozen=True)
class RollingExperimentConfig:
    experiment_id: str
    experiments_root: Path
    # Development window as inclusive cadence labels (``2022``..``2025``) or one
    # explicit ``YYYYMMDD..YYYYMMDD`` range written in both fields.
    development_first_period: str
    development_last_period: str
    heldout_first_period: str
    heldout_last_period: str
    fold_period: str = DEFAULT_FOLD_PERIOD
    # False: one regular Fold per cadence period of the window, no frozen
    # Test; the last frozen strategy goes straight to Held-out, which is the
    # verdict. True: rolling Folds inside the window (first period validation
    # only, each later period a test with the preceding period as its
    # validation).
    test_stage: bool = False
    # Passes over the whole development window; the Fold chain and the Meta
    # cadence continue across the Epoch boundary.
    epochs: int = 3
    # The macro data floor is 2020-01, so 24 months before the default 2022-01
    # development start is the most history available.
    window_months: int = 24
    min_region_trade_days: int = 2
    # Per-Fold budgets sized for a one-year Validation region with
    # ``batch_validate`` available; the host's parent control before the
    # session is never charged against them.
    max_steps_per_fold: int = 30
    max_backtests_per_fold: int = 30
    max_llm_calls: int = 1600
    session_max_attempts: int = 3
    max_fold_minutes: int = 720
    # Trailing wrap-up grace added to the Fold session budget and forwarded on
    # FoldSessionRequest.deadline_grace_seconds. Implementation default only.
    deadline_grace_minutes: int = DEFAULT_DEADLINE_GRACE_MINUTES
    finalize_before_deadline_seconds: int = 300
    per_call_timeout_seconds: int = 3600
    # Wall clock for one ``fit(context)`` invocation of the formal strategy;
    # the executor default is the single source. A slower fit fails the backtest.
    strategy_fit_timeout_seconds: int = int(SandboxLimits().fit_timeout_seconds)
    # Individual NL Sub Agent failures return audited error results by default
    # so Agent code can decide whether to ignore, retry, or fail closed.
    nl_failure_policy: str = "return_error_with_audit"
    # Epoch index (1-based) from which folds enter the convergence phase
    # (fewer modifications while holding returns, down to zero changes).
    convergence_start_epoch: int = 3
    # Preserve the Epoch-start Meta session. A positive value additionally
    # triggers Meta after every N completed Folds, before the next Fold; 0
    # disables the within-Epoch triggers. The interval counts Folds: the
    # default 1 puts a Meta session between every two consecutive Folds, and
    # the Epoch-start session covers the boundary between Epochs.
    meta_learning_fold_interval: int = 1
    # Raw prior meta-learning traces handed to the next meta session are bounded
    # to the most recent N epochs (0 disables raw memory). Unbounded concatenation
    # grows O(epochs^2); older sessions persist via PRIOR and compact fold history.
    meta_memory_max_epochs: int = 3
    # Optional experiment-level research direction injected only into the
    # active meta-learning prompt.
    meta_learning_directive: str = ""
    # Optional experiment-level exploration direction injected into every
    # automatically assembled ordinary Fold prompt. Per-Fold HITL directives
    # remain a separate, additive control surface.
    fold_exploration_directive: str = ""
    # Optional repo-relative directory of Agent-readable notes copied into each
    # Fold/Meta session's workspace/refs/. Empty keeps the historical no-copy
    # behavior; a set path must exist and be a directory.
    workspace_reference: str = ""
    # Which cross-experiment memory tiers mount read-only into every Fold and
    # Meta workspace: the curated repository library alone, plus the skills of
    # every graduated experiment, or nothing.
    operating_memory: str = DEFAULT_OPERATING_MEMORY
    # If meta-learning writes workspace/sandbox_environment.json, Pipeline can
    # build a derived Docker image and use it for later ordinary Fold runs.
    meta_sandbox_rebuild_enabled: bool = True
    meta_sandbox_rebuild_timeout_seconds: int = 1800
    # Keep at most this many derived sandbox images for this experiment; older ones
    # are best-effort pruned after a successful rebuild (0 disables GC).
    meta_sandbox_image_keep: int = 3
    # Step artifact tree (lineage across folds); toggleable for ablations.
    step_tree_enabled: bool = True
    # Also record failed validation attempts as lightweight dead-end nodes
    # (no output snapshot) so later folds can see what was already tried.
    record_failed_attempts: bool = True
    schedule: StrategySchedule = field(default_factory=StrategySchedule)
    broker_profile: BrokerProfile = field(default_factory=BrokerProfile)
    acceptance: AcceptanceRules = field(default_factory=AcceptanceRules)
    step_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )
    regularization_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.experiment_id):
            raise ValueError(
                "experiment_id must contain only letters, digits, underscore, or dash"
            )
        for name in (
            "epochs",
            "window_months",
            "min_region_trade_days",
            "max_steps_per_fold",
            "max_backtests_per_fold",
            "max_llm_calls",
            "session_max_attempts",
            "max_fold_minutes",
            "per_call_timeout_seconds",
            "strategy_fit_timeout_seconds",
            "convergence_start_epoch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "meta_learning_fold_interval",
            "meta_memory_max_epochs",
            "deadline_grace_minutes",
            "finalize_before_deadline_seconds",
            "meta_sandbox_rebuild_timeout_seconds",
            "meta_sandbox_image_keep",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.test_stage, bool):
            raise ValueError("test_stage must be boolean")
        assert_no_overlap(
            self.development_last_period, self.heldout_first_period, period=self.fold_period
        )
        object.__setattr__(self, "experiments_root", Path(self.experiments_root))

    @property
    def experiment_dir(self) -> Path:
        return self.experiments_root / self.experiment_id

    @property
    def ledger_path(self) -> Path:
        return self.experiment_dir / "ledgers" / "experiment_ledger.jsonl"


_ROLLING_FIELDS = {field_obj.name: field_obj for field_obj in fields(RollingExperimentConfig)}


def rolling_default(name: str) -> object:
    """Default of one ``RollingExperimentConfig`` field.

    The dataclass is the single source of truth for the pipeline defaults. A
    caller that fills in an absent value (the ``params.json`` loader, a CLI)
    reads it from here instead of restating a literal that can drift.
    """
    field_obj = _ROLLING_FIELDS.get(name)
    if field_obj is None or field_obj.default is MISSING:
        raise KeyError(f"{name} is not a defaulted RollingExperimentConfig field")
    return field_obj.default


def fold_session_deadline_seconds(
    max_fold_minutes: float,
    deadline_grace_minutes: float = DEFAULT_DEADLINE_GRACE_MINUTES,
) -> float:
    """Total Fold session budget: main deadline plus trailing wrap-up grace."""
    return float(max_fold_minutes) * 60.0 + float(deadline_grace_minutes) * 60.0


@dataclass(frozen=True)
class SnapshotBundle:
    snapshot_id: str
    decision_ref: str
    replay_ref: str
    data_summary_ref: str = ""
    generation_id: str = ""


class SnapshotProvider(Protocol):
    def prepare(
        self,
        *,
        fold: FoldSpec | None,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle: ...


@dataclass(frozen=True)
class ArtifactRevision:
    revision_id: str
    output_path: Path
    models_path: Path | None = None


@dataclass(frozen=True)
class FrozenArtifact:
    artifact_id: str
    path: Path
    model_path: Path | None
    source_run_id: str
    source_fold_id: str
    source_step_id: str
    revision_id: str = ""
    # A meta-regularized artifact enters the next Fold without ever having been
    # backtested. The Fold may only fall back to it after validating identical
    # content in that Fold (experiment.run_fold), never silently.
    requires_validation: bool = False


class ArtifactStore(Protocol):
    def revision(self, revision_id: str) -> ArtifactRevision: ...

    def freeze_revision(
        self,
        revision_id: str,
        *,
        artifact_id: str,
        experiment_id: str,
        epoch_id: str,
        fold_id: str,
        run_id: str,
        step_id: str,
    ) -> FrozenArtifact: ...


@dataclass(frozen=True)
class EvaluationRequest:
    revision: ArtifactRevision
    snapshot: SnapshotBundle
    mode: str
    start: str
    end: str
    schedule: StrategySchedule
    broker_profile: BrokerProfile


@dataclass(frozen=True)
class EvaluationResult:
    """One completed full-window evaluation.

    A backend either returns this or raises: a partial or aborted replay never
    produces an EvaluationResult, so there is no "incomplete" variant to carry.
    """

    summary: dict[str, object]
    result_ref: str


class EvaluationBackend(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


@dataclass(frozen=True)
class StepResult:
    step_id: str
    revision_id: str
    validation: EvaluationResult
    selected: bool = False
    # The host's parent control recorded as an in-session Step node: the
    # inherited parent replayed unchanged on this Fold's Validation window
    # before the Agent started. It is never charged against the Step budget.
    parent_control: bool = False


@dataclass(frozen=True)
class FoldSessionRequest:
    experiment_id: str
    epoch_id: str
    fold: FoldSpec
    run_id: str
    parent: FrozenArtifact | None
    snapshot: SnapshotBundle
    max_steps: int
    max_backtests: int
    max_llm_calls: int
    deadline_seconds: float
    # Trailing wrap-up grace reserved from deadline_seconds. Live Fold sessions
    # always set this from RollingExperimentConfig.deadline_grace_minutes.
    deadline_grace_seconds: float = DEFAULT_DEADLINE_GRACE_MINUTES * 60.0
    directive: str = ""
    prior: str = ""
    prompt_override: str = ""
    # Per-session HITL override of the experiment's default sandbox GPU count;
    # None keeps the experiment default. The "auto" selector still picks which
    # devices by free memory at container start.
    sandbox_gpu_count: int | None = None
    # Experiment contract the session must publish in its run manifest and
    # enforce while it runs. Closed source carries the same values on the
    # manifest the pipeline writes; here the pipeline hands them to the
    # sandbox owner, which is the component that writes the manifest.
    fold_period: str = DEFAULT_FOLD_PERIOD
    # Whether a frozen Test follows this Fold (rolling development) or the
    # Held-out replay is the next and final evaluation (regular Folds).
    test_stage: bool = False
    # The parent's completed Validation on this Fold's window, replayed by the
    # host before the session; None without a parent or when it failed. The
    # developer records it as the session's first Step node.
    parent_control: EvaluationResult | None = None
    epoch_index: int = 1
    phase: str = "exploration"
    acceptance_rules: Mapping[str, object] = field(default_factory=dict)
    modification_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )
    snapshot_config: Mapping[str, object] = field(default_factory=dict)
    record_failed_attempts: bool = True
    nl_failure_policy: str = "return_error_with_audit"
    finalize_before_deadline_seconds: int = 300
    step_gate_hook: Callable[[int, dict[str, object]], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    user_question_hook: Callable[[str, str], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    progress_hook: Callable[[str, dict[str, object] | None], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    session_key: str = ""
    # Trusted host source for the current experiment-level skills snapshot.
    # The sandbox adapter copies it to workspace/skills but never exposes this
    # host path through Agent-visible facts or manifests.
    skills_source_ref: str = ""


@dataclass(frozen=True)
class FoldSessionResult:
    conversation_id: str
    steps: tuple[StepResult, ...]
    selected_step_id: str | None = None
    finish_reason: str = "fold_finished"
    # Keyword-only from here: these are independent optional records, so a new
    # one must never be able to land in an older field's positional slot.
    _: KW_ONLY
    # The Agent's own account of a voluntary early finish, as ``finish_fold``
    # recorded it. Empty when the session did not finish early.
    early_stop_reason: str = ""
    # Host path of this run's manifest. The fold ledger record carries it so a
    # later Meta session can read the run's backtest summaries back out.
    run_manifest_ref: str = ""
    # Trusted host path to this run's collected workspace/skills audit copy.
    skills_source_ref: str = ""


@dataclass(frozen=True)
class MetaSessionResult:
    """What one Meta session produced.

    ``revision_id`` is set only when the session regularized the strategy
    artifact and the modification check allowed it; the Pipeline owns the
    freeze, exactly as it does for a Fold's selected Step.
    """

    prior: str
    conversation_id: str = ""
    revision_id: str = ""
    modification_check: Mapping[str, object] = field(default_factory=dict)
    allowed: bool = True
    # Trusted host path to this run's collected workspace/skills audit copy.
    skills_source_ref: str = ""


FoldDeveloper = Callable[[FoldSessionRequest], FoldSessionResult]
MetaLearner = Callable[[dict[str, object]], "MetaSessionResult"]


@dataclass(frozen=True)
class FoldOutcome:
    fold_id: str
    run_id: str
    fold_status: str
    # None when the fold recorded no freezable outcome (first fold without an
    # acceptable baseline): the run continues and only fails at the end if no
    # fold ever froze an artifact.
    frozen: FrozenArtifact | None
    validation_summary: dict[str, object] | None
    test_summary: dict[str, object] | None


__all__ = [
    "ACCEPTANCE_SEMANTICS_SUMMARY",
    "DEFAULT_DEADLINE_GRACE_MINUTES",
    "DEFAULT_FOLD_PERIOD",
    "AcceptanceRules",
    "ArtifactRevision",
    "ArtifactStore",
    "EvaluationBackend",
    "EvaluationRequest",
    "EvaluationResult",
    "ExecutionMode",
    "FoldDeveloper",
    "FoldOutcome",
    "FoldSessionRequest",
    "FoldSessionResult",
    "FrozenArtifact",
    "MetaLearner",
    "MetaSessionResult",
    "RollingExperimentConfig",
    "SnapshotBundle",
    "SnapshotProvider",
    "StepResult",
    "StrategyExperimentConfig",
    "fold_session_deadline_seconds",
    "rolling_default",
]
