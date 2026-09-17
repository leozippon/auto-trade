"""Configuration for one research arm."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, MISSING, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from autotrade.environment.artifacts import ModificationConstraints
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.sandbox import SandboxConfig, SandboxLimits
from autotrade.environment.strategy import StrategySchedule

from . import verdict
from .calendar import FULL_SPAN, ResearchGeometry
from .ledger import RESEARCH_SESSION_KEY
from .skills import DEFAULT_OPERATING_MEMORY

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
# v10: same-day macro tables carry the close contract stamp
# (MACRO_DATASET_CONTRACTS) instead of the raw date-EOD placeholder, so a v9
# macro view shows T rows a trading day later than a v10 one.
# v11: daily.is_suspended counts only suspend_d halts (suspend_type "S"); v10
# also flagged resumption rows ("R"), i.e. normally traded sessions.
SNAPSHOT_CACHE_FORMAT_VERSION = 11

# Trailing wrap-up grace added to the research session budget. Not a console or
# worker knob; ResearchSessionRequest carries the seconds to the Agent runner.
DEFAULT_DEADLINE_GRACE_MINUTES = 10
# A research session's time, replay-year, null-control and model-call budgets
# are arm-level: one session per arm spends them across every attempt.
DEFAULT_MAX_RESEARCH_MINUTES = 2400
DEFAULT_MAX_REPLAY_YEARS = 96
DEFAULT_MAX_NULL_CONTROLS = 12
DEFAULT_MAX_LLM_CALLS = 6400

# Research on four July-June years, a twelve-month forward test after them,
# and a Held-out quarter the replay clips to the release end
# (docs/pipeline-design.md). The PIT view seed is planned over it.
DEFAULT_RESEARCH_GEOMETRY = ResearchGeometry(
    research_start="20210701",
    research_end="20250630",
    forward_end="20260630",
    heldout_end="20260930",
)

# The exploration PIT view seed an experiment hardlinks completed views from
# unless its ``pit_views_seed`` parameter names another tree, and the scratch
# directory the offline prebuild pins its research release in. Repo-relative
# and gitignored; the contract that decides reuse lives in
# ``autotrade.pipelines.pit_views_seed``. Defined here because both the console
# creation defaults and that module need the path.
DEFAULT_PIT_VIEWS_SEED = Path("data/pit_views_seed/explore")
DEFAULT_PIT_VIEWS_SEED_WORKSPACE = Path("data/pit_views_seed/explore_workspace")


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


@dataclass(frozen=True)
class AcceptanceRules:
    """The arm's round parameters for the freeze nomination and the verdict.

    A nominated research node fails the freeze gate on a non-finite metric or a
    research-period drawdown over ``max_drawdown`` (``evaluate``); the return
    and Sharpe targets there only warn. ``max_drawdown`` and
    ``cost_stress_multiplier`` decide the forward and Held-out verdict
    (``pipelines/verdict.py``), whose remaining thresholds are that module's
    constants.
    """

    min_return: float = 0.0
    min_sharpe: float = 0.0
    max_drawdown: float = 0.25
    # The forward neutralised excess must stay positive after paying this
    # multiple of the profile's slippage (verdict F5).
    cost_stress_multiplier: float = 2.0

    def __post_init__(self) -> None:
        for name in ("min_return", "min_sharpe", "max_drawdown", "cost_stress_multiplier"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be between zero and one")
        if self.cost_stress_multiplier < 1:
            raise ValueError("cost_stress_multiplier must be at least one")

    def to_record(self) -> dict[str, object]:
        return {
            "min_return": self.min_return,
            "min_sharpe": self.min_sharpe,
            "max_drawdown": self.max_drawdown,
            "cost_stress_multiplier": self.cost_stress_multiplier,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AcceptanceRules:
        """These rules from a ``to_record`` mapping, ignoring unknown keys.

        Run manifests are read back long after they were written, so a record
        that carries a retired key must still rebuild the rules it does name.
        """

        allowed = set(cls().to_record())
        return cls(**{key: record[key] for key in allowed if key in record})  # type: ignore[arg-type]

    def agent_facts(self) -> dict[str, object]:
        """The ``acceptance_rules`` run fact, derived from these rules and the
        verdict constants so no prompt restates a threshold. Rules only: no
        forward or Held-out date appears here."""

        return {
            "freeze_gate": {
                "span": f"the nominee replayed the whole research period (span={FULL_SPAN})",
                "finite_metrics": "total_return/max_drawdown/sharpe must be finite",
                "max_drawdown": (
                    f"<= {self.max_drawdown} over the research period, the same "
                    "limit the forward and Held-out verdict enforce"
                ),
                "full_span_validations": (
                    f">= {verdict.FREEZE_MIN_FULL_SPAN_VALIDATIONS} in the arm, the "
                    "nominee included"
                ),
                "deflated_sharpe_probability": (
                    f">= {verdict.FREEZE_MIN_DSR_PROBABILITY} for the nominee's "
                    "research-period neutralized IR; trials = distinct revisions "
                    "validated in the arm on any span"
                ),
                "freezes_per_arm": 1,
            },
            "targets": {
                "role": "warnings on a nomination, not selection criteria",
                "min_return": self.min_return,
                "min_sharpe": self.min_sharpe,
                "no_orders": "order_count=0 warns no_orders",
            },
            "graduation": {
                "evaluated_on": (
                    "one continuous replay of the frozen artifact over the twelve "
                    "months after research end and then Held-out; no session sees "
                    "either period"
                ),
                "forward": {
                    "lower_bound": (
                        f"{verdict.FORWARD_CONFIDENCE:.0%} one-sided block-bootstrap "
                        "lower bound of annualized neutralized excess > 0"
                    ),
                    "recency": (
                        f"neutralized excess of the last {verdict.RECENCY_MONTHS} "
                        "months >= 0"
                    ),
                    "max_drawdown": f"<= {self.max_drawdown}",
                    "excess_at_cost_stress": (
                        f"> 0 with slippage multiplied by {self.cost_stress_multiplier}"
                    ),
                    "round_trips": (
                        f">= {verdict.MIN_ROUND_TRIPS_PER_MONTH} per month"
                    ),
                    "mean_gross": f">= {verdict.MIN_MEAN_GROSS}",
                    "strategy_error": "none",
                    "minimum_detectable_excess": (
                        "about 2.12 x research tracking error / sqrt(years of forward "
                        "data): a lower residual tracking error is what makes a "
                        "real edge detectable"
                    ),
                },
                "heldout": {
                    "neutralized_excess": (
                        f">= -{verdict.HELDOUT_TOLERANCE_Z} x forward tracking error "
                        "/ sqrt(years)"
                    ),
                    "max_drawdown": f"<= {self.max_drawdown}",
                    "mean_gross": f">= {verdict.MIN_MEAN_GROSS}",
                    "strategy_error": "none",
                },
            },
        }

    def evaluate(self, summary: dict[str, object]) -> tuple[list[str], list[str]]:
        """(hard_reasons, warnings) of a nomination; the hard ones are the
        freeze gate's (``experiment.freeze_gate_for``).

        Two hard rejects. Non-finite metrics, because every IEEE comparison
        against NaN is False, so a NaN metric would otherwise pass every
        threshold. And a research-period drawdown over ``max_drawdown``, the
        same limit F4/H3 enforce forward: freezing a book that already breached
        it spends a forward test on a candidate the verdict must reject.

        Return and Sharpe shortfalls stay warnings -- they are targets, and an
        arm may honestly freeze a modest but real edge. A zero ``order_count``
        warns the same way: it clears every threshold without ever placing an
        order, so the warning is the only thing distinguishing it from a real
        result (``trade_count`` counts closed round trips and is 0 for
        buy-and-hold). Only a summary from a completed evaluation reaches here;
        an aborted replay never produces one."""
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
            hard.append("max_drawdown_above_limit")
        if values.get("total_return", float("-inf")) < self.min_return:
            warnings.append("return_below_target")
        if "sharpe" in values and values["sharpe"] < self.min_sharpe:
            warnings.append("sharpe_below_target")
        if summary.get("order_count") == 0:
            warnings.append("no_orders")
        return hard, warnings


@dataclass(frozen=True)
class RollingExperimentConfig:
    experiment_id: str
    experiments_root: Path
    # Research, forward and Held-out dates of the arm (pipelines/calendar.py).
    geometry: ResearchGeometry = DEFAULT_RESEARCH_GEOMETRY
    # The macro data floor is 2020-01, so 24 months before a July 2022
    # decision view is the most history every domain carries.
    window_months: int = 24
    # The research session's budgets, spent across every attempt of the arm's
    # one session. The host's forward replay is charged to none of them. One
    # replay-year is one research year replayed for one candidate: a batch of
    # three candidates on a two-year span costs six, and a full-period
    # validation costs as many as the research period has years.
    max_replay_years: int = DEFAULT_MAX_REPLAY_YEARS
    # Host-side random-portfolio null controls (K=500 replays, minutes each)
    # the session may request through ``run_null_control``; the frozen node's
    # block is reused at freeze. 0 leaves the tool out.
    max_null_controls: int = DEFAULT_MAX_NULL_CONTROLS
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    # Attempts of the research session or of the forward replay before the
    # experiment fails with the last error.
    session_max_attempts: int = 3
    # Pausable effective inference minutes of the research session.
    max_research_minutes: int = DEFAULT_MAX_RESEARCH_MINUTES
    # Trailing wrap-up grace added to the session budget and forwarded on
    # ResearchSessionRequest.deadline_grace_seconds. Implementation default only.
    deadline_grace_minutes: int = DEFAULT_DEADLINE_GRACE_MINUTES
    finalize_before_deadline_seconds: int = 300
    per_call_timeout_seconds: int = 3600
    # Wall clock for one ``fit(context)`` invocation of the formal strategy;
    # the executor default is the single source. A slower fit fails the backtest.
    strategy_fit_timeout_seconds: int = int(SandboxLimits().fit_timeout_seconds)
    # Individual NL Sub Agent failures return audited error results by default
    # so Agent code can decide whether to ignore, retry, or fail closed.
    nl_failure_policy: str = "return_error_with_audit"
    # Optional experiment-level exploration direction injected into every
    # research session prompt. Per-session directives are additive.
    research_directive: str = ""
    # Optional repo-relative directory of Agent-readable notes copied into each
    # session's workspace/refs/. Empty copies nothing; a set path must exist and
    # be a directory.
    workspace_reference: str = ""
    # Which cross-experiment memory tiers mount read-only into every session:
    # the curated repository library alone, plus the skills of every graduated
    # experiment, or nothing.
    operating_memory: str = DEFAULT_OPERATING_MEMORY
    # Also record failed validation attempts as lightweight dead-end nodes
    # (no output snapshot) so later sessions can see what was already tried.
    record_failed_attempts: bool = True
    schedule: StrategySchedule = field(default_factory=StrategySchedule)
    broker_profile: BrokerProfile = field(default_factory=BrokerProfile)
    acceptance: AcceptanceRules = field(default_factory=AcceptanceRules)
    step_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.experiment_id):
            raise ValueError(
                "experiment_id must contain only letters, digits, underscore, or dash"
            )
        if not isinstance(self.geometry, ResearchGeometry):
            raise TypeError("geometry must be a ResearchGeometry")
        for name in (
            "window_months",
            "max_replay_years",
            "max_llm_calls",
            "session_max_attempts",
            "max_research_minutes",
            "per_call_timeout_seconds",
            "strategy_fit_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_null_controls",
            "deadline_grace_minutes",
            "finalize_before_deadline_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
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


def session_deadline_seconds(
    max_research_minutes: float,
    deadline_grace_minutes: float = DEFAULT_DEADLINE_GRACE_MINUTES,
) -> float:
    """Total session budget: main deadline plus trailing wrap-up grace."""
    return float(max_research_minutes) * 60.0 + float(deadline_grace_minutes) * 60.0


@dataclass(frozen=True)
class SnapshotBundle:
    snapshot_id: str
    decision_ref: str
    # Empty for a decision-only bundle (``SnapshotProvider.prepare_decision``).
    replay_ref: str
    generation_id: str = ""


class SnapshotProvider(Protocol):
    def prepare(
        self,
        *,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle: ...

    def prepare_decision(self, *, decision_time: datetime) -> SnapshotBundle: ...


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


class ArtifactStore(Protocol):
    """What the Pipeline may read off an artifact store's records.

    Structural, not nominal: a store answers with its own record type (the
    filesystem store returns a plain namespace), so these annotations name the
    fields a caller may rely on, never an ``isinstance``.
    """

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

    def frozen(
        self,
        artifact_id: str,
        *,
        expected_path: str | Path | None = None,
        experiment_id: str | None = None,
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
    # Replay slots that continue the book after ``snapshot.replay_ref``, in
    # order: the span runs as one replay from ``start`` (the first slot's
    # start) to ``end`` (the last slot's end).
    continuation: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplaySpan:
    """Consecutive replay slots replayed as one book.

    ``snapshot`` carries the first slot and the decision view at its anchor;
    ``continuation`` names the slots after it. ``label`` names the span on
    every record (``full`` for the whole research period).
    """

    label: str
    mode: str
    start: str
    end: str
    snapshot: SnapshotBundle
    continuation: tuple[str, ...] = ()

    @property
    def slots(self) -> int:
        """Replay slots the span covers; for a research span, its years."""

        return 1 + len(self.continuation)

    def request(
        self,
        revision: ArtifactRevision,
        *,
        schedule: StrategySchedule,
        broker_profile: BrokerProfile,
    ) -> EvaluationRequest:
        return EvaluationRequest(
            revision=revision,
            snapshot=self.snapshot,
            mode=self.mode,
            start=self.start,
            end=self.end,
            schedule=schedule,
            broker_profile=broker_profile,
            continuation=self.continuation,
        )


_YEAR_SPAN = re.compile(r"Y(\d+)(?:\.\.Y(\d+))?")


def research_span(years: Sequence[ReplaySpan], label: str) -> ReplaySpan:
    """The span ``label`` names over the research years, replayed as one book.

    ``full`` is every year, ``Yk`` one year and ``Yi..Yj`` the contiguous years
    i through j. The span opens on its first year's slot and decision view and
    continues through the rest; a span covering every year is labelled
    ``full`` however it was named. Any other label, or a year the research
    period does not have, is refused, so no span reaches past research end.
    """

    count = len(years)
    text = str(label).strip()
    match = _YEAR_SPAN.fullmatch(text)
    if text == FULL_SPAN:
        first, last = 1, count
    elif match:
        first = int(match.group(1))
        last = int(match.group(2) or first)
    else:
        first = last = 0
    if not 1 <= first <= last <= count:
        raise ValueError(
            f"span must be {FULL_SPAN}, one research year Y1..Y{count} or contiguous "
            f"years such as Y1..Y{count}; got {label!r}"
        )
    chosen = years[first - 1 : last]
    if (first, last) == (1, count):
        canonical = FULL_SPAN
    elif first == last:
        canonical = f"Y{first}"
    else:
        canonical = f"Y{first}..Y{last}"
    return ReplaySpan(
        label=canonical,
        mode=chosen[0].mode,
        start=chosen[0].start,
        end=chosen[-1].end,
        snapshot=chosen[0].snapshot,
        continuation=tuple(year.snapshot.replay_ref for year in chosen[1:]),
    )


@dataclass(frozen=True)
class EvaluationResult:
    """One completed evaluation.

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
    # The ``ReplaySpan.label`` the validation replayed.
    span: str


@dataclass(frozen=True)
class BudgetUsed:
    """What the research session has consumed, cumulative over its attempts.

    Recorded on the trace with every model call and tool result, so a new
    attempt continues the counters from the last block the interrupted one
    wrote rather than from zero.
    """

    inference_seconds: float = 0.0
    llm_calls: int = 0
    main_calls: int = 0
    subagent_calls: int = 0
    compact_calls: int = 0
    replay_years: int = 0
    null_controls: int = 0

    def to_record(self) -> dict[str, object]:
        return {
            "inference_seconds": round(float(self.inference_seconds), 1),
            "llm_calls": int(self.llm_calls),
            "main_calls": int(self.main_calls),
            "subagent_calls": int(self.subagent_calls),
            "compact_calls": int(self.compact_calls),
            "replay_years": int(self.replay_years),
            "null_controls": int(self.null_controls),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BudgetUsed:
        def number(key: str, kind: type) -> object:
            value = record.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"budget_used.{key} must be a non-negative number")
            return kind(value)

        return cls(
            inference_seconds=number("inference_seconds", float),  # type: ignore[arg-type]
            llm_calls=number("llm_calls", int),  # type: ignore[arg-type]
            main_calls=number("main_calls", int),  # type: ignore[arg-type]
            subagent_calls=number("subagent_calls", int),  # type: ignore[arg-type]
            compact_calls=number("compact_calls", int),  # type: ignore[arg-type]
            replay_years=number("replay_years", int),  # type: ignore[arg-type]
            null_controls=number("null_controls", int),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SessionResume:
    """The interrupted attempts a new attempt of the research session continues."""

    # 1-based number of the attempt about to start.
    attempt: int
    # When and why the last attempt stopped (its last trace event, its error).
    interrupted_at: str
    error: str
    # The last successful compaction's summary, the checkpoint the new attempt
    # starts from; None when the interrupted attempts never compacted.
    compaction_summary: str | None
    budget_used: BudgetUsed
    # Transcript file names of the earlier attempts under the ``trace`` root.
    transcripts: tuple[str, ...]


# How the research session ended. ``freeze`` nominates a node for the freeze
# gate, ``no_edge`` ends the arm without a deliverable, and ``deadline`` is a
# session whose budget (the wrap-up grace or the model-call budget) ran out
# before it finished.
SESSION_OUTCOMES = ("freeze", "no_edge", "deadline")


@dataclass(frozen=True)
class ResearchSessionRequest:
    experiment_id: str
    run_id: str
    # The Agent's only data view: the decision view at research end, anchored
    # at ``decision_time``.
    snapshot: SnapshotBundle
    decision_time: datetime
    # The research years in order, one single-slot span each (labels Y1..Yn):
    # every span a Validation may replay is resolved from them
    # (``research_span``).
    research_years: tuple[ReplaySpan, ...]
    # First day of the decision view's history window (``window_months``
    # before research end), the Agent-visible input window.
    input_window_start: str
    # Replay-years the session may spend on Validations (see
    # ``RollingExperimentConfig.max_replay_years``).
    max_replay_years: int
    max_llm_calls: int
    deadline_seconds: float
    # Trailing wrap-up grace reserved from deadline_seconds.
    deadline_grace_seconds: float = DEFAULT_DEADLINE_GRACE_MINUTES * 60.0
    directive: str = ""
    # Per-session HITL override of the experiment's default sandbox GPU count;
    # None keeps the experiment default. The "auto" selector still picks which
    # devices by free memory at container start.
    sandbox_gpu_count: int | None = None
    acceptance_rules: Mapping[str, object] = field(default_factory=dict)
    modification_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )
    snapshot_config: Mapping[str, object] = field(default_factory=dict)
    record_failed_attempts: bool = True
    nl_failure_policy: str = "return_error_with_audit"
    finalize_before_deadline_seconds: int = 300
    # Cap on the session's own ``run_null_control`` calls; the experiment default
    # is the single source (RollingExperimentConfig.max_null_controls).
    max_null_controls: int = RollingExperimentConfig.max_null_controls
    progress_hook: Callable[[str, dict[str, object] | None], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    session_key: str = RESEARCH_SESSION_KEY
    # Trusted host source for the current experiment-level skills snapshot.
    # The sandbox adapter copies it to workspace/skills but never exposes this
    # host path through Agent-visible facts or manifests.
    skills_source_ref: str = ""
    # What the session's earlier attempts consumed and recorded: the live
    # counters continue from these amounts, the freeze gate and the finish
    # tool see these Validations, and ``resume`` says how the last attempt
    # stopped. All empty on a first attempt.
    budget_used: BudgetUsed = field(default_factory=BudgetUsed)
    steps_before: tuple[StepResult, ...] = ()
    resume: SessionResume | None = None

    @property
    def validation(self) -> ReplaySpan:
        """The whole research period as one span."""

        return research_span(self.research_years, FULL_SPAN)


@dataclass(frozen=True)
class ResearchSessionResult:
    conversation_id: str
    steps: tuple[StepResult, ...]
    # One of ``SESSION_OUTCOMES``.
    outcome: str
    # Keyword-only from here: independent optional fields, so a new one can
    # never land in an older field's positional slot.
    _: KW_ONLY
    # ``freeze``: the nominated Step.
    node_id: str | None = None
    # The Agent's own account of its outcome.
    reason: str = ""
    finish_reason: str = ""
    # Host path of this run's manifest.
    run_manifest_ref: str = ""
    # Trusted host path to this run's collected workspace/skills audit copy.
    skills_source_ref: str = ""
    # Null-control blocks the session already computed, keyed by step id; the
    # Pipeline reuses the frozen node's block instead of drawing it again.
    null_controls: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    # The session's cumulative spend at its end, and which attempt ended it.
    budget_used: BudgetUsed = field(default_factory=BudgetUsed)
    attempt: int = 1

    def __post_init__(self) -> None:
        if self.outcome not in SESSION_OUTCOMES:
            raise ValueError(f"unknown research session outcome: {self.outcome!r}")


ResearchDeveloper = Callable[[ResearchSessionRequest], ResearchSessionResult]


__all__ = [
    "DEFAULT_DEADLINE_GRACE_MINUTES",
    "DEFAULT_MAX_LLM_CALLS",
    "DEFAULT_MAX_NULL_CONTROLS",
    "DEFAULT_MAX_REPLAY_YEARS",
    "DEFAULT_MAX_RESEARCH_MINUTES",
    "DEFAULT_RESEARCH_GEOMETRY",
    "SESSION_OUTCOMES",
    "AcceptanceRules",
    "ArtifactRevision",
    "ArtifactStore",
    "BudgetUsed",
    "EvaluationBackend",
    "EvaluationRequest",
    "EvaluationResult",
    "ExecutionMode",
    "FrozenArtifact",
    "ReplaySpan",
    "ResearchDeveloper",
    "ResearchSessionRequest",
    "ResearchSessionResult",
    "RollingExperimentConfig",
    "SessionResume",
    "SnapshotBundle",
    "SnapshotProvider",
    "StepResult",
    "StrategyExperimentConfig",
    "research_span",
    "rolling_default",
    "session_deadline_seconds",
]
