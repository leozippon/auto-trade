"""Configuration for one scheduled strategy experiment."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from autotrade.environment.artifacts import ModificationConstraints
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import StrategySchedule

from .folds import FoldSpec, assert_no_overlap

ExecutionMode = Literal["sandbox", "trusted"]

# Increment only when the cached snapshot/replay on-disk contract changes.
# Source revisions are intentionally not cache inputs: harmless code changes
# should not invalidate every expensive data view.
# v6: events/macro unions carry the full configured-dataset schema (typed
# zero-row contributions for datasets without visible rows in the window).
# v7: the snapshot manifest carries dataset_columns for the unit reference.
SNAPSHOT_CACHE_FORMAT_VERSION = 7


@dataclass(frozen=True)
class StrategyExperimentConfig:
    strategy_path: Path
    schedule: StrategySchedule = field(default_factory=StrategySchedule)
    broker_profile: BrokerProfile = field(default_factory=BrokerProfile)
    execution_mode: ExecutionMode = "sandbox"
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    def __post_init__(self) -> None:
        path = Path(self.strategy_path).resolve()
        if not path.is_file():
            raise ValueError(f"strategy file does not exist: {path}")
        if self.execution_mode not in ("sandbox", "trusted"):
            raise ValueError("execution_mode must be sandbox or trusted")
        object.__setattr__(self, "strategy_path", path)


ExperimentConfig = StrategyExperimentConfig


@dataclass(frozen=True)
class AcceptanceRules:
    """Validation acceptance checks (docs/pipeline-design.md §2.2): drawdown,
    finiteness and completeness are HARD rejects; min_return/min_sharpe are
    warn-only targets — a shortfall records a warning and never resets the fold."""

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

    def evaluate(
        self, summary: dict[str, object], *, complete: bool
    ) -> tuple[list[str], list[str]]:
        """(hard_reasons, warnings). Integrity failures are hard rejects:
        non-finite metrics (every IEEE comparison against NaN is False, so a NaN
        metric would otherwise pass all thresholds) and incomplete validation.
        The max_drawdown cap stays a hard risk limit. Return/Sharpe shortfalls
        are WARNINGS only — the fold still freezes its validated update; a weak
        step recorded with a warning beats silently resetting the fold chain."""
        hard: list[str] = []
        warnings: list[str] = []
        if not complete:
            hard.append("incomplete_validation")
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
    first_test_period: str
    last_test_period: str
    heldout_first_period: str
    heldout_last_period: str
    fold_period: str = "quarter"
    epochs: int = 3
    window_months: int = 21
    min_region_trade_days: int = 2
    max_steps_per_fold: int = 10
    max_backtests_per_fold: int = 15
    max_llm_calls: int = 400
    session_max_attempts: int = 3
    max_fold_minutes: int = 240
    # Trailing wrap-up grace handed to the Agent session on top of the main
    # Fold deadline (experiment.py adds it to the session budget; the runner
    # reserves the trailing window). Keep aligned with the runner's
    # DEFAULT_DEADLINE_GRACE_SECONDS (10 minutes) — the session-side reservation
    # is the runner default because the fold backend does not forward this
    # field per request.
    deadline_grace_minutes: int = 10
    finalize_before_deadline_seconds: int = 300
    per_call_timeout_seconds: int = 3600
    # Individual NL Sub Agent failures return audited error results by default
    # so Agent code can decide whether to ignore, retry, or fail closed.
    nl_failure_policy: str = "return_error_with_audit"
    # Epoch index (1-based) from which folds enter the convergence phase
    # (fewer modifications while holding returns, down to zero changes).
    convergence_start_epoch: int = 3
    # Preserve the Epoch-start Meta session. A positive value additionally
    # triggers Meta after every N completed Folds, before the next Fold; 0
    # disables the within-Epoch triggers. Default 2 on a quarterly calendar.
    meta_learning_fold_interval: int = 2
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
        assert_no_overlap(
            self.last_test_period, self.heldout_first_period, period=self.fold_period
        )
        object.__setattr__(self, "experiments_root", Path(self.experiments_root))

    @property
    def experiment_dir(self) -> Path:
        return self.experiments_root / self.experiment_id

    @property
    def ledger_path(self) -> Path:
        return self.experiment_dir / "ledgers" / "experiment_ledger.jsonl"


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
    summary: dict[str, object]
    result_ref: str
    complete: bool = True


class EvaluationBackend(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


@dataclass(frozen=True)
class StepResult:
    step_id: str
    revision_id: str
    validation: EvaluationResult
    selected: bool = False


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
    fold_period: str = "quarter"
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
    "AcceptanceRules",
    "ArtifactRevision",
    "ArtifactStore",
    "EvaluationBackend",
    "EvaluationRequest",
    "EvaluationResult",
    "ExecutionMode",
    "ExperimentConfig",
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
]
