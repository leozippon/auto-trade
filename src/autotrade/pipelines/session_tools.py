"""The research session's Validation engine and the Agent-facing backtest tools.

``SessionValidations`` holds the session's Step list, its step tree and its
validation record; ``smoke_backtest``, ``validate``, ``test`` and
``run_null_control`` are the replay tools a research session hands the Agent
on top of it. Nothing
here composes a session: ``research_session`` builds these tools and wires them
into the Agent runner, so this module never imports it back.
"""

from __future__ import annotations

import queue
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from autotrade.agent.compact import safe_error_summary
from autotrade.common.records import (
    read_json_object,
    redact_host_paths,
    write_json_atomic,
)
from autotrade.environment.artifacts import (
    READONLY_FILES,
    ArtifactRevision,
    ArtifactSnapshotUnstable,
    FilesystemArtifactStore,
    copy_artifact_snapshot,
)
from autotrade.environment.executor import (
    StrategyMemoryExceeded,
    StrategyStateExceeded,
    StrategyTimedOut,
    caused_by,
    gpu_contended,
    raised_by_strategy,
    strategy_resources_of,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.null_control import (
    BASELINE_DRAWS,
    NullControlSetupError,
)
from autotrade.environment.replay.stats import plain_statistics
from autotrade.environment.replay.style import (
    active_analysis,
    series_pairs,
)
from autotrade.environment.runtime import (
    AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS,
    RunManifest,
    utc_now_iso,
)
from autotrade.environment.sandbox import SandboxConfig, link_copytree
from autotrade.environment.step_tree import StepTree
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
)
from autotrade.environment.tools.base import (
    AGENT_JUSTIFICATION_MAX_CHARS,
    SessionInterrupt,
    ToolError,
    ToolResult,
    ToolSpec,
)
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.node_reference import (
    NODE_REFERENCE_DESCRIPTION,
    NODE_REFERENCE_EXAMPLE,
    session_handle,
    session_handles,
    session_node_reference,
)
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.environment.trace import TraceEvent

from .agent_views import NULL_CONTROL_KEYS, allowed_keys
from .calendar import FULL_SPAN, REVEALED_SPAN, yyyymmdd
from .config import (
    BATCH_VALIDATE_MAX_CANDIDATES,
    BATCH_VALIDATE_MIN_CANDIDATES,
    DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
    RESEARCH_PHASE,
    AcceptanceRules,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    ReplaySpan,
    ResearchSessionRequest,
    StepResult,
    research_span,
)
from .experiment import (
    FROZEN_NULL_SEED,
    check_registration,
    replay_and_analysis,
    research_step_record,
    result_analysis,
)
from .ledger import ExperimentLedger, SessionKind, link_head
from .session_resume import record_step_sidecar
from .skills import assert_skills_absent_from_formal
from .validation_record import (
    SOURCE_TEST,
    ValidationSteps,
    record_significance,
    step_entry,
    step_table,
    test_step_record,
)
from .verdict import (
    ConditionReason,
    active_daily,
    graded_statistics,
    holder_readings,
    residual_reading,
    summary_excess_at_cost_stress,
)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _public_error_text(exc: Exception) -> str:
    """The exact failure text with host paths redacted."""
    return redact_host_paths(safe_error_summary(exc))


def _public_validation_error(exc: Exception) -> str:
    """Agent-visible Validation failure: type, actionable reason, no host leaks."""
    return f"daily Validation failed: {_public_error_text(exc)}"


# A strategy directory a replay tool takes: ``output`` itself -- the working
# copy as it stands -- or a workspace directory laid out like it, such as
# ``candidates/<name>``. The other workspace roots are the working copy's own
# trees or not strategy trees at all. ``smoke_backtest`` and ``validate``
# resolve a path through :func:`strategy_directory` and give it the read-only
# template through :func:`supply_readonly_files`, so a directory one accepts is
# one the other accepts. Every replay is a snapshot verified against the bytes
# its check approved, so later edits cannot reach it.
STRATEGY_PATH_MAX_CHARS = 200
_RESERVED_WORKSPACE_ROOTS = frozenset({"output", "models", "inputs", "skills", "refs"})
_WORKING_COPY = "output"


def strategy_directory(workspace: SafeWorkspace, path: str, *, who: str) -> Path:
    """The existing workspace directory ``path`` names, refused at a reserved root."""

    directory = workspace.resolve(path, must_exist=True, directory=True)
    if directory == workspace.root or (
        PurePosixPath(path).parts[0] in _RESERVED_WORKSPACE_ROOTS
        and directory != workspace.root / _WORKING_COPY
    ):
        raise ToolError(
            f"{who} points at a reserved workspace root ({path}); pass output itself "
            "or copy the tree to its own directory, e.g. candidates/<name>/",
            error_type="path_error",
            blocked_target=path,
        )
    return directory


def supply_readonly_files(
    output_dir: Path, directory: Path, *, who: str, path: str
) -> None:
    """Give a strategy directory the read-only template files it did not write.

    ``README.md`` is part of every formal artifact but carries no strategy
    content and the Agent may not edit it, so a directory laid out from its
    strategy modules alone would be refused for "modifying" a file it never
    touched. The working copy's own read-only files are copied in where absent;
    a directory that carries a different one is still refused by
    ``modification_check``, which restores only the working copy the host
    itself seeded. Only the read-only template names are touched -- never a
    sibling module of the package.
    """

    for name in READONLY_FILES:
        source = output_dir / name
        target = directory / name
        if not source.is_file() or target.exists():
            continue
        try:
            shutil.copyfile(source, target)
        except PermissionError as exc:
            # A directory copied out of a read-only artifact tree keeps mode
            # 0444/0555, and the template cannot land in it. Say so with the
            # remedy instead of failing as an unhandled host error.
            raise ToolError(
                f"{who} is not writable, so the read-only template {name} cannot "
                f"be supplied: {_public_error_text(exc)}",
                error_type="permission_denied",
                blocked_target=path,
            ) from exc


SMOKE_BACKTEST_DEFAULT_DAYS = 3
SMOKE_BACKTEST_MAX_DAYS = 5


class SmokeBacktestTool(SessionTimeBudgetAware):
    """Run a strategy directory (the working copy by default) through the real
    replay for a few days.

    Hand-rolled shell smoke tests are what let seven of nine official backtests
    die on day one: a script that sets ``ctx.asof_dir = "/mnt/snapshot"`` and
    fakes an account object exercises the flat frozen snapshot and a dict-like
    account, while the replay hands the strategy a rolling directory-per-domain
    as-of view and a real ``AccountSnapshot``. This tool removes the reason to
    hand-roll one: same executor, same per-decision wall clock
    (``SandboxConfig.timeout_seconds``), same as-of layout, same account
    object, same modification gate — only the window is short.

    It is deliberately NOT an evaluation: no revision is committed, no step-tree
    node is written and nothing here can be tested or registered.

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
        "Unofficial rehearsal: replay output/ (or path, a directory laid out and "
        "checked like a validate candidate's) for a few research-year trading days "
        "on the real replay path: the rolling as-of view (each "
        "context.asof_dir/<domain>/ is a directory of parquet parts), a real "
        "AccountSnapshot, the same executor and per-decision timeout. The window "
        "opens at the research period's first trading day, or the first one on or "
        "after start: a fit whose training window grows is slowest late in the "
        "research years, so measure it there before a full-span validate. Commits "
        "nothing, creates no node, does not count against "
        "acceptance_rules.validation.validates_per_step. seconds_per_day.strategy "
        "is the only cost that scales with days; window_build_seconds is the "
        "one-off opening of a late window, reused by a later probe at the same "
        "start. asof_date_ranges: each domain's rows and first/last dates on the "
        "last day.",
        {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": SMOKE_BACKTEST_MAX_DAYS,
                    "description": (
                        "Trading days to replay from the window's first day "
                        f"(default {SMOKE_BACKTEST_DEFAULT_DAYS})."
                    ),
                },
                "start": {
                    "type": "string",
                    "minLength": 8,
                    "maxLength": 10,
                    "description": (
                        "YYYYMMDD (or YYYY-MM-DD) inside the research period: "
                        "the window opens at the first trading day at or after "
                        "it, instead of at the start of the research period. A "
                        "date outside the research period is refused."
                    ),
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": STRATEGY_PATH_MAX_CHARS,
                    "description": (
                        "Workspace-relative directory laid out like output/, "
                        "e.g. candidates/value (default output)."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        mutating=True,
        example={"days": 2, "start": "20241008", "path": "candidates/value"},
    )

    def __init__(
        self,
        *,
        request: ResearchSessionRequest,
        workspace: SafeWorkspace,
        models_dir: Path,
        modification_check_factory: Callable[[Path], ModificationCheckTool],
        evaluator: EvaluationBackend,
        schedule,
        broker_profile,
        scratch_root: Path,
        time_budget: InferenceTimeBudget,
    ) -> None:
        self.request = request
        self.workspace = workspace
        self.output_dir = workspace.root / _WORKING_COPY
        self.models_dir = models_dir
        # The batch's own static gate, pointed at the directory smoked.
        self.modification_check_factory = modification_check_factory
        self.evaluator = evaluator
        self.schedule = schedule
        self.broker_profile = broker_profile
        self.scratch_root = Path(scratch_root)
        self.time_budget = time_budget

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        # Argument validation is the Agent's own mistake and stays on its
        # clock; the replay itself does not.
        days = self._days(arguments)
        start = self._start(arguments)
        path, directory = self._directory(arguments)
        with self.time_budget.pause():
            return self._invoke_exempt(days, start, path, directory)

    def _invoke_exempt(
        self, days: int, start: str | None, path: str, directory: Path
    ) -> ToolResult:
        supply_readonly_files(
            self.output_dir, directory, who=f"smoke_backtest path {path}", path=path
        )
        check = self.modification_check_factory(directory).invoke({})
        if not check.ok:
            # Same static gate as a Validation candidate, so a green smoke run
            # means the gate will not be what fails the official one.
            raise ToolError(f"smoke_backtest blocked by modification_check: {check.error}")
        # The rehearsal replays an immutable snapshot outside the Agent's mounts,
        # not the live tree: it can neither be registered nor leave anything behind
        # in the directory (a trusted-mode run imports main.py and would drop
        # __pycache__ into it, which the next modification_check would
        # reject), and the Agent's own writes during the run cannot reach the
        # bytes being replayed.
        scratch = self._scratch_dir()
        evaluation = None
        try:
            models_source = self.models_dir if self.models_dir.is_dir() else None
            fingerprint = copy_artifact_snapshot(
                directory,
                models_source,
                dest_output=scratch / "output",
                dest_models=scratch / "models",
            )
            if fingerprint != check.value["fingerprint"]:
                raise ArtifactSnapshotUnstable(
                    f"{path}/ changed between modification_check and the smoke snapshot"
                )
            revision = ArtifactRevision(
                "smoke",
                scratch / "output",
                scratch / "models" if models_source is not None else None,
            )
            evaluation = self.evaluator.evaluate(
                self.request.full_span.request(
                    revision, schedule=self.schedule, broker_profile=self.broker_profile
                ),
                max_days=days,
                start_day=start,
            )
        except SessionInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - the exception text IS the result
            return ToolResult(
                True,
                value={
                    "path": path,
                    "status": "failed",
                    "start": start or self.request.full_span.start,
                    "error": _public_error_text(exc),
                    # A rehearsal that died on a clock or on a shared card says
                    # what it was using when it did.
                    **resource_reading(strategy_resources_of(exc)),
                },
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        result_dir = Path(evaluation.result_ref).parent
        try:
            return ToolResult(
                True,
                value={"path": path, **self._report(evaluation, result_dir, start)},
            )
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

    def _directory(self, arguments: Mapping[str, object]) -> tuple[str, Path]:
        """The strategy directory to smoke, under the rules a batch candidate's
        path follows (:func:`strategy_directory`); ``output`` by default."""

        raw = arguments.get("path", _WORKING_COPY)
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError(
                "smoke_backtest path must be a non-empty string",
                error_type="schema_error",
                blocked_target="path",
            )
        path = raw.strip()
        if len(path) > STRATEGY_PATH_MAX_CHARS:
            raise ToolError(
                f"smoke_backtest path exceeds {STRATEGY_PATH_MAX_CHARS} characters",
                error_type="schema_error",
                blocked_target="path",
            )
        return path, strategy_directory(self.workspace, path, who="smoke_backtest path")

    def _start(self, arguments: Mapping[str, object]) -> str | None:
        """The probe date, refused unless it is inside the research period.

        The session may measure a fit anywhere in the research years, and
        nowhere else: a date past them would replay the validation years,
        which only ``test`` reveals, and a date before them has no slot.
        """

        raw = arguments.get("start")
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise ToolError("smoke_backtest start must be a YYYYMMDD string")
        try:
            day = yyyymmdd(raw)
        except ValueError as exc:
            raise ToolError(f"smoke_backtest start is not a date: {raw}") from exc
        span = self.request.full_span
        if not span.start <= day <= span.end:
            raise ToolError(
                f"smoke_backtest start {day} is outside the research years "
                f"{span.start}..{span.end}"
            )
        return day

    def _report(
        self,
        evaluation: EvaluationResult,
        result_dir: Path,
        start: str | None,
    ) -> dict[str, object]:
        summary = dict(evaluation.summary)
        phases = summary.get("phase_seconds")
        phases = dict(phases) if isinstance(phases, dict) else {}
        replayed = summary.get("replayed_trade_days")
        replayed = int(replayed) if isinstance(replayed, int) else 0
        # Only the cost that actually scales with decision days. ``data_view``
        # is the outer phase around the as-of view, and on a late ``start`` it
        # is almost entirely the one-off window opening on the window's first
        # day (minutes, then ~0.1 s a day): dividing it by the probe's days
        # reported a per-day rate an audited session extrapolated to a full
        # span. It is reported below as the whole-run figure it is; the other
        # phases are host instrumentation the session cannot act on.
        per_day = (
            {"strategy": round(phases["strategy"] / replayed, 3)}
            if replayed and "strategy" in phases
            else {}
        )
        lifecycle = summary.get("order_lifecycle")
        lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
        report: dict[str, object] = {
            "status": "ok",
            "start": start or self.request.full_span.start,
            "replayed_trade_days": replayed,
            "orders": {
                side: {key: counts.get(key) for key in ("filled", "rejected")}
                for side, counts in lifecycle.items()
                if isinstance(counts, Mapping)
            },
            "reject_counts": summary.get("reject_counts"),
            "seconds_per_day": per_day,
            **(
                {"window_build_seconds": round(float(phases["data_view"]), 3)}
                if "data_view" in phases
                else {}
            ),
            "asof_date_ranges": _smoke_pit(result_dir).get("asof_date_ranges") or {},
            # What the rehearsal cost its container, so the Agent sizes a
            # batch on a measurement of the real replay container instead of
            # on the session container it can see from the inside.
            **resource_reading(summary.get("resources")),
        }
        return report


def _smoke_pit(result_dir: Path) -> dict[str, object]:
    """The replay record's ``pit`` block: the domain directory names the
    replay exposed under asof_dir and, for a smoke run, each one's date range
    on its last day (``pit_backend.asof_date_ranges``)."""
    try:
        record = read_json_object(result_dir / "result.json")
    except (OSError, ValueError):
        return {}
    pit = record.get("pit")
    return dict(pit) if isinstance(pit, dict) else {}


# One Validation's full replay record is attached to its step-tree node, and
# the node tree is mounted as the ``steps`` search root. This pair is the only
# Agent-readable reference to the full result, so the attachment site and the
# returned reference must name the same file.
VALIDATION_RESULT_ATTACHMENT = "validation/result.json"
# Beside it, the graded series day by day with the factor legs it is
# regressed on (``verdict.active_daily``): without it a session rebuilt proxies
# from its own equity curve to find when a drawdown happened or which period
# drives a reading.
# Only a research Validation is recorded here; the forward and Held-out replay
# writes to the host's results root and never to a Step tree.
VALIDATION_DAILY_ATTACHMENT = "validation/active_daily.csv"
# Summary blocks whose size scales with the replay: one row per closed position
# and one per week of the window. A copy of them is therefore not a fixed cost
# -- an audited session shipped 427 KB of ``per_stock`` into the conversation
# and forced a 174 s compaction plus twelve recovery calls. They stay in the
# referenced result.json; a step-tree node keeps every other metric.
REPLAY_SCALED_SUMMARY_BLOCKS = ("per_stock", "weekly_returns")


def write_active_daily(result_ref: str) -> Path:
    """Write one completed Validation's daily graded series beside its result.

    Read off the style sidecar the verdict itself reads, through a
    temporary name, so a failure leaves no partial file behind.
    """

    result_dir = Path(result_ref).parent
    analysis = result_analysis(result_ref)
    target = result_dir / PurePosixPath(VALIDATION_DAILY_ATTACHMENT).name
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        active_daily(analysis).to_csv(staging, index=False, float_format="%.10g")
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def inline_backtest_stats(summary: Mapping[str, object]) -> dict[str, object]:
    """The backtest metrics a step-tree node carries: every block but the
    ones that scale with the replay."""
    return {
        key: value
        for key, value in summary.items()
        if key not in REPLAY_SCALED_SUMMARY_BLOCKS
    }


def validation_period_stats(summary: Mapping[str, object]) -> dict[str, object]:
    """What a replay over the validation period leaves where the session can
    read it (its node and its run-manifest row): the whole-span scalars, the
    figures a test's ``whole_replay`` is read from. No block: the per-call
    clocks and container readings, the per-year windows and the rest are
    finer than a test reports."""

    return {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}


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


class SessionValidations:
    """The research session's Validation engine: one Step list, one tree, one
    validation record.

    Not an Agent tool: ``validate`` commits and replays every candidate
    through it, and ``test`` records every step through it, so neither can
    bypass the ledger behind the other's back.
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
        manifest: RunManifest,
        experiment_dir: Path,
    ) -> None:
        self.request = request
        # The run's acceptance rules (``AcceptanceRules.from_record``, which
        # refuses a record without a required rule).
        self.rules = AcceptanceRules.from_record(request.acceptance_rules)
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
        # Continued from the earlier attempts' Validations and tests: the
        # validation record's steps so far, the test horizon they imply, and how
        # many validate calls the current interval has already made.
        self.steps: list[StepResult] = list(request.steps_before)
        self.tests: list[dict[str, object]] = [dict(row) for row in request.tests_before]
        self.steps_plan = ValidationSteps.of(
            [request.validation], self.rules.test_step_months
        )
        self.test_replays = int(request.budget_used.test_replays)
        self.validates_in_interval = _validates_since(
            tree, self.tests[-1].get("recorded_at") if self.tests else None
        )
        # Candidates reserved so far, which names each result.
        self.candidates_started = 0

    @property
    def test_horizon(self) -> str:
        """The validation record's test horizon: the end of the last tested step, the
        end of the research years before the first test."""

        return self.steps_plan.horizon_after(len(self.tests))

    @property
    def validates_per_step(self) -> int:
        return self.rules.validates_per_step

    def ledger_head(self) -> dict[str, object]:
        """The link keys every ledger record of this session carries."""

        return link_head(self.request.experiment_id, self.request.session_key, self.request.run_id)

    def record_row(self, record: Mapping[str, object]) -> dict[str, object]:
        """One validation-record step as a test result lists it, as it was
        recorded: its window, the candidate by handle, the entry's figures
        without the daily rows, and, where the step replayed seed replicates,
        each one's active return and return (or its failure's class: the
        error text stays in the ledger)."""

        entry = dict(record["entry"])  # type: ignore[arg-type]
        row = _rounded_row(
            {
                "step": record["step"],
                **{key: entry.get(key) for key in ("start", "end", "days")},
                "candidate": self.named([str(record["candidate"]["step_id"])])[0],  # type: ignore[index]
                **{key: entry.get(key) for key in RECORD_ENTRY_KEYS},
            }
        )
        replicates = [dict(item) for item in record.get("replicates") or ()]  # type: ignore[attr-defined]
        if replicates:
            names = self.named([str(item["step_id"]) for item in replicates])
            row["seed_replicates"] = [
                _replicate_cell(name, item.get("entry") or str(item["failure"]))  # type: ignore[arg-type]
                for name, item in zip(names, replicates, strict=True)
            ]
        return row

    @property
    def session_ref(self) -> str:
        """The opaque session ref this session's Step nodes carry."""

        return self.ref_store.get_or_create("session", self.request.session_key)

    def handle(self, node_id: str) -> dict[str, str]:
        """``{"handle": ...}`` where the node's short handle is unambiguous."""

        return session_handle(self.tree, node_id, session_ref=self.session_ref)

    def named(self, node_ids: Sequence[str]) -> list[str]:
        """Nodes as the Agent should name them: the short handle where unambiguous."""

        return session_handles(self.tree, node_ids, session_ref=self.session_ref)

    def span(self, label: object) -> tuple[ReplaySpan, int | None]:
        """The span ``label`` names and, for the revealed span, the tested
        steps it covers (None for a research span), or a schema error naming
        the valid ones.

        A research span stays inside the research years: the session's data
        view ends with them for the whole session, so a span reaching past
        them is refused by name however far the tests have gone. ``revealed``
        is the validation span as a test replays it -- one fresh book from
        the validation start -- through the k steps tested so far, the count
        the validation record holds (``tests``, continued from the ledger on
        resume); its replay stops at the end of step k (``replay_legs``'
        ``end_day``), so it is the window the last test replayed and no day
        past it is replayed. Its label names k (``revealed:k``), so the nodes
        of one window share a label and those of another do not. Before the
        first test nothing is revealed, and it is refused.
        """

        text = str(label).strip()
        if text == REVEALED_SPAN:
            tested = len(self.tests)
            if not tested:
                raise ToolError(
                    f"span {REVEALED_SPAN} replays the validation steps test has revealed, "
                    "and no step is tested yet: before the first test validate replays only "
                    "the research years",
                    error_type="schema_error",
                    blocked_target="span",
                )
            return replace(self.request.validation, label=f"{REVEALED_SPAN}:{tested}"), tested
        try:
            span = research_span(
                self.request.years, text, research_years=self.rules.research_years
            )
        except ValueError as exc:
            raise ToolError(
                f"{exc}; or {REVEALED_SPAN}, the validation steps tested so far",
                error_type="schema_error",
                blocked_target="span",
            ) from exc
        research = self.request.full_span
        if span.end > research.end:
            raise ToolError(
                f"span {span.label} ends at {span.end}, past the end of the research years "
                f"{research.end}: a research span stays inside them "
                f"({research.start}..{research.end}); the validation years are revealed "
                f"step by step by test, and span {REVEALED_SPAN} replays the steps tested so far",
                error_type="schema_error",
                blocked_target="span",
            )
        return span, None

    def registration(
        self,
        node_id: str,
        seed_replicates: Sequence[str] = (),
    ) -> dict[str, object]:
        """Whether one Step of this session registers, as the Pipeline reads
        it (``experiment.check_registration``), with ``seed_replicates`` the
        Steps it would register as its seed replicates. Registration only,
        never a score. A node that is not a Step of this session does not
        register.
        """

        steps = {step.step_id: step for step in self.steps}
        if any(step_id not in steps for step_id in (node_id, *seed_replicates)):
            return {"passed": False, "reasons": [ConditionReason.NOT_A_SESSION_STEP]}
        return check_registration(
            research_step_record(steps[node_id]),
            experiment_dir=self.experiment_dir,
            acceptance=self.rules,
            seed_replicates=[research_step_record(steps[step_id]) for step_id in seed_replicates],
        )

    def append_manifest_summary(self, summary: dict[str, object]) -> None:
        """Every backtest attempt, successful or not, lands in the run manifest.

        It is the only durable per-run record of what the session actually ran.
        """
        self.manifest.append_backtest_summary(summary)

    def commit_revision(
        self, source_output: Path, fingerprint: str, *, label: str
    ) -> ArtifactRevision:
        """Register one approved working tree into an immutable revision.

        The revision is verified to be the bytes ``modification_check`` just
        approved, so an approval can never be transferred to a tree that was
        still being written; a mismatch discards the revision and raises.

        It also records the artifact it descends from, so the revisions the experiment
        keeps form the same lineage the Step tree draws.
        """
        revision, snapshot = self.artifact_store.create_revision(
            source_output,
            models_path=self.models_dir,
            parent_revision_id=self._parent_revision_id(),
        )
        if snapshot != fingerprint:
            self.artifact_store.discard_revision(revision.revision_id)
            raise ArtifactSnapshotUnstable(
                f"{label} changed between modification_check and the "
                "Validation snapshot"
            )
        assert_skills_absent_from_formal(revision.output_path, revision.models_path)
        return revision

    def _parent_revision_id(self) -> str | None:
        """The revision the working artifact descends from.

        A tree position the Agent set with ``step_rollback`` is the deliberate
        branch point and wins; otherwise the working copy descends from the last
        Validation the experiment recorded, across attempts. ``validate`` commits
        a whole round before it records any of it, so a round's revisions all
        resolve to the same parent and stay siblings. ``None`` before the experiment's
        first Validation, and at a position the experiment's own Validations do not
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
        fingerprint: str,
        metadata: Mapping[str, object] | None = None,
        validation_period: bool = False,
    ) -> str:
        """Append one completed Validation to the step tree under the current
        position; ``validate`` repositions the tree before each record so
        its candidates are siblings of one parent. ``fingerprint`` is the
        committed bytes' (:meth:`commit_revision`). A replay over the
        ``validation_period`` exposes what a test does: its node carries the
        whole-span figures (:func:`validation_period_stats`) and no record of
        the replay, which stays with the host sidecar."""
        node_id = self.tree.record_step(
            revision.output_path,
            epoch_id=SessionKind.RESEARCH,
            # The session id is opaqued like every other Agent-visible id.
            session_ref=self.session_ref,
            run_id=self.ref_store.get_or_create("run", self.request.run_id),
            result_name=result_name,
            revision_id=self.ref_store.get_or_create(
                "strategy", revision.revision_id
            ),
            # tree.json is Agent-readable and accumulates one node per
            # Validation for the whole experiment, so it keeps no block that
            # scales with the replay.
            metrics=(validation_period_stats if validation_period else inline_backtest_stats)(
                evaluation.summary
            ),
            models_root=revision.models_path,
            attachments=None
            if validation_period
            else {
                VALIDATION_RESULT_ATTACHMENT: evaluation.result_ref,
                VALIDATION_DAILY_ATTACHMENT: write_active_daily(evaluation.result_ref),
            },
            metadata=metadata,
        )
        record_step_sidecar(
            self.experiment_dir,
            recorded_step(node_id, revision.revision_id, evaluation, metadata or {}, fingerprint),
        )
        self.publish_tree()
        return node_id

    def publish_tree(self) -> None:
        """Publish the session's tree to the experiment, node snapshots included."""

        if self.tree.tree_path.is_file():
            link_copytree(self.tree.root, self.experiment_dir / "steps")

    def validation_request(
        self, revision: ArtifactRevision, span: ReplaySpan
    ) -> EvaluationRequest:
        """The Validation replay of ``span`` for a revision the session accepted."""
        return span.request(
            revision, schedule=self.schedule, broker_profile=self.broker_profile
        )

    def reserve(self, count: int) -> list[str]:
        """Name the results of ``count`` candidates about to replay.

        A name is never reused: the attempt is recorded under it whether the
        replay completes, fails or never starts.
        """
        names = [
            f"valid_{self.candidates_started + offset + 1:03d}" for offset in range(count)
        ]
        self.candidates_started += count
        return names

    def release(self, count: int) -> None:
        """Give back the names of candidates whose snapshot never held (no replay ran)."""
        self.candidates_started = max(0, self.candidates_started - count)

    def check_deadline(self) -> None:
        try:
            self.time_budget.check()
        except TimeoutError as exc:
            raise TimeoutError("research session deadline exceeded") from exc


def _validates_since(tree: StepTree, since: object) -> int:
    """How many counted validate calls the step tree records after ``since``
    (the last test's ledger stamp; None counts every call): the distinct batch
    ids of the complete Validations created after it, so a batch counts
    exactly when ``ValidateTool`` counted it live. A resumed attempt continues
    the interval's count from here."""

    batches = set()
    for node in tree.nodes():
        metadata = node.get("metadata")
        batch = metadata.get("batch_id") if isinstance(metadata, Mapping) else None
        if (
            batch
            and node.get("complete_validation") is True
            and (since is None or str(node.get("created_at") or "") > str(since))
        ):
            batches.add(str(batch))
    return len(batches)


def recorded_step(
    node_id: str,
    revision_id: str,
    evaluation: EvaluationResult,
    metadata: Mapping[str, object],
    fingerprint: str,
) -> StepResult:
    """The Step a recorded Validation is, from the metadata its node carries
    and the ``fingerprint`` of the bytes its revision committed."""

    span = str(metadata.get("span") or "")
    if not span:
        raise ValueError("a recorded Validation needs its span in metadata")
    return StepResult(
        node_id,
        revision_id,
        evaluation,
        span=span,
        control=metadata.get("control") is True,
        batch_id=str(metadata["batch_id"]) if metadata.get("batch_id") else None,
        fingerprint=fingerprint,
    )


# ``validate``'s candidate range and replay width live with the experiment's
# other budgets (``config.BATCH_VALIDATE_MAX_CANDIDATES``,
# ``ExperimentConfig.batch_validate_max_concurrency``).
BATCH_NAME_MAX_CHARS = 40
# A pre-registration is only binding if the falsification clause fits with it:
# signal, holding, control, and what would refute the claim. It is bounded like
# every other Agent-written justification, by what one such statement needs
# rather than by what one line is.
BATCH_HYPOTHESIS_MAX_CHARS = AGENT_JUSTIFICATION_MAX_CHARS
# Repeated identical rejections. A refused batch is free by design -- nothing
# is committed and no interval call is spent -- so one audited session spent
# 3.89 h of 10.17 h on 128 consecutive rejections carrying the same error, 393
# parent LLM calls apart. A rejection is counted per session by its signature
# (error type plus the target it names, never the message text, which carries
# digests and so changes with the file); the third identical one says so and
# names the recovery for that signature. The model-call and time budgets bound
# the loop.
BATCH_REJECTION_ESCALATE_AT = 3
# What a failed candidate's replay measured. ``StrategyRaised``
# (``environment.executor``, read off the whole cause chain by
# ``raised_by_strategy``) is the strategy's own failure -- its code raised, or
# (``StrategyMemoryExceeded``) its container was killed at the published memory
# cap: the replay measured the strategy -- it cannot run on this data. Every
# other failure measured the host: a strategy clock that ran out, a shared card
# another process took, a sandbox that never started or broke protocol. That
# produced no evidence about the candidate.
BATCH_FAILURE_STRATEGY = "strategy"
BATCH_FAILURE_ENVIRONMENT = "environment"
# A span the host cannot replay. A batch that failed on the environment is free
# to resend, which is right for a transient fault (a clock lost to load, a card
# another process took) and leaves nothing to stop the loop when the host
# fails the span every time: in round 20261015 every replay crossing
# 2022-07-01 failed in the host's as-of merge, and two experiments resent 6 and 4 such
# batches (2.1 h and 1.5 h of wall time), although nothing past that date had
# been tested. A batch in which every candidate failed on the environment
# measured nothing; the second such batch on the same span in one attempt ends
# the attempt (``EnvironmentBlocked``). A batch in which any candidate
# completed or raised in its own code proved the span runs, and does not count.
ENVIRONMENT_BLOCKED_BATCHES = 2


class EnvironmentBlocked(SessionInterrupt):
    """The host failed every candidate of ``ENVIRONMENT_BLOCKED_BATCHES``
    batches on one span: the attempt ends here with outcome
    ``environment_blocked`` and ``reason`` as its ending reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"environment_blocked: {reason}")
        self.reason = reason


def _failure_cause(error: BaseException) -> str:
    """Which of the two a failed candidate's replay measured."""

    return BATCH_FAILURE_STRATEGY if raised_by_strategy(error) else BATCH_FAILURE_ENVIRONMENT


def validation_period_failure(error: BaseException) -> str:
    """What a failed replay over the validation period tells the session: a
    fixed sentence for the failure's class, never its own text.

    The exception message, its type, the day it failed, the worker's stderr
    and the container readings are the strategy's to shape on data of steps
    the record has not taken, so any of them could carry such a step out.
    The host keeps them (:func:`record_validation_period_failure`); a replay
    of the research years reports its error as it is.
    """

    if caused_by(error, (StrategyMemoryExceeded, StrategyStateExceeded)):
        return "it exceeded budgets.strategy_memory_bytes or budgets.strategy_state_disk_bytes"
    if raised_by_strategy(error):
        return "its own code raised"
    if caused_by(error, StrategyTimedOut):
        return "a fit or generate_orders call ran past its clock"
    return "the host failed the replay"


# Host-only, beside the Step sidecars: the detail of every failed replay over
# the validation period, one file each, which the console shows the operator.
VALIDATION_PERIOD_FAILURES_DIR = ".host/validation_period_failures"


def record_validation_period_failure(
    experiment_dir: Path, error: BaseException, **identity: object
) -> None:
    """Keep the full detail of one failed replay over the validation period
    where only the host reads it: what ``identity`` names (tool, step or span,
    candidate), the cause, the error text and the container readings."""

    directory = Path(experiment_dir) / VALIDATION_PERIOD_FAILURES_DIR
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.mkdir(exist_ok=True, mode=0o700)
    recorded_at = utc_now_iso()
    write_json_atomic(
        directory / f"{recorded_at.replace(':', '')}_{uuid.uuid4().hex[:8]}.json",
        {
            "recorded_at": recorded_at,
            **identity,
            "cause": _failure_cause(error),
            "error": _public_error_text(error),
            "resources": strategy_resources_of(error),
        },
    )


def validation_period_failures(experiment_dir: Path) -> list[dict[str, object]]:
    """The failures :func:`record_validation_period_failure` kept, oldest first."""

    directory = Path(experiment_dir) / VALIDATION_PERIOD_FAILURES_DIR
    return [read_json_object(path) for path in sorted(directory.glob("*.json"))]


@dataclass(frozen=True)
class _Replayed:
    """One candidate's replay: its evaluation or the error it raised (exactly
    one of them), the ``time.perf_counter`` window it ran in, and the card it
    was pinned to (None: every card of the experiment)."""

    evaluation: EvaluationResult | None
    error: Exception | None
    started: float
    ended: float
    card: int | None = None

    def __post_init__(self) -> None:
        if (self.evaluation is None) == (self.error is None):
            raise ValueError("a replay holds either its evaluation or its error")

    @property
    def seconds(self) -> float:
        return self.ended - self.started

    def overlaps(self, other: _Replayed) -> bool:
        return self.started < other.ended and other.started < self.ended

    def shares_card_with(self, other: _Replayed) -> bool:
        """Whether the two held a card at the same time."""

        return self.overlaps(other) and (
            self.card is None or other.card is None or self.card == other.card
        )


def leg_cards(devices: Sequence[int], width: int) -> list[int]:
    """The card of each of a batch's ``width`` concurrent replay slots.

    The experiment's cards in turn, so with at least as many cards as slots no two
    legs running at once share one, and with fewer they share evenly. A batch
    running one replay at a time pins nothing: that replay attaches every card,
    which is what lets one strategy run a model across them (tensor parallel).
    """

    if width <= 1 or not devices:
        return []
    return [devices[slot % len(devices)] for slot in range(width)]


def _shared_gpu_note(siblings: Sequence[str]) -> str:
    """What a candidate the shared card failed is told about its own batch.

    The siblings that replayed on its card while it ran held memory on the
    same device. The CUDA report cannot say whose memory filled the card -- a
    sibling's, the candidate's own other strategy container's, or a process
    outside the experiment -- so the note names the siblings as sharers, not as the
    cause, and the failure stays the environment's, like any other contention.
    """

    return (
        f"; {', '.join(siblings)} of this batch replayed on the same GPU while this "
        "candidate ran, so the memory it lacked may be held by its own batch: send "
        "fewer GPU-heavy candidates per batch or lower their GPU memory "
        "(each row's resources.peak_gpu_memory_bytes is its measured peak)"
    )


# What every attempt's run-manifest row keeps of its candidate's registration.
_BATCH_MANIFEST_KEYS = (
    "batch_id",
    "candidate",
    "hypothesis",
    "control",
    "span",
)
# What the Agent reads of one replay. A tool result is a projection: the
# stored records (the node's result.json, its active_daily.csv, the ledger)
# keep everything, and a row carries each figure once under a name that says
# what it is, so no glossary is needed beside it. A batch multiplies a row by
# its width and every later model call re-sends it, so constants of the batch
# ride once beside the rows.


def active_reading(
    analysis: Mapping[str, object],
    summary: Mapping[str, object],
    *,
    rules: AcceptanceRules,
) -> dict[str, object]:
    """The graded series over one replay: the book's daily return minus its
    random baseline (``verdict.graded_statistics``: the annualised excess,
    tracking error and information ratio, and the series' loadings on the
    market, size and value legs), the residual statistic of the same
    regression at the experiment's ``alpha`` (``verdict.residual_reading``)
    with the ``threshold`` for the replay's length for the caller to place,
    the drawdown of its compounded path as the replay states it, and the
    excess after the cost stress on the replay's turnover
    (``verdict.summary_excess_at_cost_stress``). ``unmeasured`` says why a
    replay the residual statistic cannot read has none."""

    try:
        graded = graded_statistics(analysis)
    except ValueError as exc:
        return {"unmeasured": str(exc)}
    try:
        reading = residual_reading(analysis, alpha=rules.alpha)
    except ValueError as exc:
        reading = {"unmeasured": str(exc)}
    benchmark = summary.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    block = {
        "excess": graded["plain_excess"],
        "information_ratio": graded["information_ratio"],
        "residual_information_ratio": reading.get("residual_information_ratio"),
        "residual_excess": reading.get("residual_excess"),
        "tracking_error": graded["tracking_error"],
        "max_drawdown": benchmark.get("active_max_drawdown"),
        "market_beta": graded["market_beta"],
        "size_beta": graded["size_beta"],
        "value_beta": graded["value_beta"],
        "excess_at_cost_stress": summary_excess_at_cost_stress(
            graded, summary, cost_stress_multiplier=rules.cost_stress_multiplier
        ),
        "threshold": reading.get("threshold"),
        **({"unmeasured": reading["unmeasured"]} if "unmeasured" in reading else {}),
    }
    return _rounded_row(block)


def book_reading(
    analysis: Mapping[str, object], summary: Mapping[str, object]
) -> dict[str, object]:
    """The holder's money over one replay, after every cost: the replay's own
    figures, the benchmark and random-baseline comparisons as the holder
    reads them (``verdict.holder_readings``: compounded returns and their
    differences), and how it was traded."""

    def block(name: str) -> Mapping[str, object]:
        value = summary.get(name)
        return value if isinstance(value, Mapping) else {}

    holder = holder_readings(analysis)
    benchmark, exposure = block("benchmark"), block("exposure")
    return _rounded_row(
        {
            "return": summary.get("total_return"),
            "annualized_return": summary.get("annualized_return"),
            "excess_vs_benchmark": holder["raw_excess"],
            "baseline_return": holder["baseline_return"],
            "excess_vs_baseline": holder["plain_selection"],
            "max_drawdown": summary.get("max_drawdown"),
            "sharpe": summary.get("sharpe"),
            "turnover": summary.get("turnover"),
            "avg_gross": exposure.get("avg_gross"),
            "zero_position_days": exposure.get("zero_position_days"),
            "size_tilt": benchmark.get("size_tilt"),
            "top_industry_weight": benchmark.get("top_industry_weight"),
            "orders": summary.get("order_count"),
            "trades": summary.get("trade_count"),
            "win_rate": summary.get("win_rate"),
            "top5_share_of_gains": block("pnl_concentration").get("top5_share_of_gross_gains"),
            "breakeven_extra_slippage_bps": block("cost_sensitivity").get(
                "breakeven_extra_slippage_bps"
            ),
        }
    )


def year_readings(
    analysis: Mapping[str, object],
    summary: Mapping[str, object],
    years: Sequence[ReplaySpan],
) -> dict[str, object]:
    """``by_year``: one row per research year of a replay's span, read off its
    ``sub_windows`` and labelled as spans name the year (``Yk``, its place in
    ``years``); and, over two years or more, ``concentration``: the year with
    the largest active excess, its share of the positive years' sum, and the
    active excess of the span's days outside it (the annualised mean of the
    active series, as every active excess is read). A sub-window outside
    every year is a host defect and raises."""

    rows: list[dict[str, object]] = []
    bounds: list[tuple[str, str]] = []
    windows = summary.get("sub_windows")
    for window in windows if isinstance(windows, list) else ():
        first, last = str(window["start"]), str(window["end"])
        index = next(
            (place for place, year in enumerate(years, start=1) if year.start <= first <= year.end),
            None,
        )
        if index is None:
            raise ValueError(f"sub-window {first}..{last} lies in no research year")
        rows.append(
            _rounded_row(
                {
                    "label": f"Y{index}",
                    "active_excess": window.get("active_excess_return"),
                    "excess_vs_benchmark": window.get("excess_return"),
                    "return": window.get("return"),
                    "benchmark_return": window.get("benchmark_return"),
                    "sharpe": window.get("sharpe"),
                }
            )
        )
        bounds.append((first, last))
    readings: dict[str, object] = {"by_year": rows}
    excesses = [row["active_excess"] for row in rows]
    active = active_analysis(analysis)
    if len(rows) < 2 or active is None or not all(isinstance(value, float) for value in excesses):
        return readings
    best = max(range(len(rows)), key=lambda place: excesses[place])  # type: ignore[arg-type, return-value]
    positive = sum(value for value in excesses if value > 0)  # type: ignore[operator, misc]
    first, last = bounds[best]
    outside = plain_statistics(
        value for day, value in series_pairs(active["strategy_daily"]) if not first <= day <= last
    )["plain_excess"]
    readings["concentration"] = _rounded_row(
        {
            "best_year": rows[best]["label"],
            "best_year_share": excesses[best] / positive if positive > 0 else None,  # type: ignore[operator]
            "active_excess_ex_best_year": outside,
        }
    )
    return readings


def resource_reading(resources: Mapping[str, object] | None) -> dict[str, object]:
    """``{"resources": ...}``: what a replay cost its strategy container, or
    nothing when it measured nothing.

    The measured peaks (memory, the state directory, GPU memory) and the fit
    and ``generate_orders`` seconds as count, p50, p95 and max. The limits
    they ran under are the experiment's ``budgets`` facts, both strategy
    clocks multiplied by the replays running side by side, so they are not
    repeated per row. The per-call lists stay in the node's result.json under
    ``stats.resources``: a strategy that refits daily reads 1,941 fits over
    an eight-year span, and a batch result is re-sent with every later model
    call. A failed replay carries the same block off its exception.
    """

    if not resources:
        return {}
    block: dict[str, object] = {
        key: resources[key]
        for key in ("peak_memory_bytes", "peak_state_bytes", "peak_gpu_memory_bytes")
        if key in resources
    }
    for key in ("fit_seconds", "decision_seconds"):
        seconds = resources.get(key)
        if seconds:
            block[key] = _seconds_summary(seconds)  # type: ignore[arg-type]
    return {"resources": block} if block else {}


def _seconds_summary(seconds: Sequence[float]) -> dict[str, float]:
    """How many calls ran and their p50, p95 and max seconds. Percentiles are
    taken at nearest rank, so each is a measured call rather than an
    interpolation; the fit max is the reading the fit timeout bounds."""

    ordered = sorted(seconds)

    def rank(percent: int) -> float:
        return ordered[(percent * len(ordered) + 99) // 100 - 1]

    return {"count": len(ordered), "p50": rank(50), "p95": rank(95), "max": ordered[-1]}


@contextmanager
def _batch_replay_timeouts(evaluator: object, workers: int) -> Iterator[None]:
    """Widen both strategy deadlines to the replay width this batch creates.

    ``SandboxConfig.fit_timeout_seconds`` and ``timeout_seconds`` are
    runaway guards measured on host wall clock. Fanning ``workers`` replays
    out over the same host makes the same ``fit(context)`` and the same
    ``generate_orders(context)`` take longer without the strategy doing
    anything different, so a fixed cap makes the verdict depend on how many
    siblings a candidate happened to be batched with — the failure mode that
    cost one experiment four Validations to fits solo reruns finished in
    1,550-2,027 s. Scaling both caps by ``workers`` keeps the guards (a
    runaway fit or decision still dies) while removing that dependence.

    The inference cap was deliberately left fixed when the fit cap was
    scaled, on the argument that a Validation makes hundreds of decision
    calls and widening the per-call cap would stretch a slow batch instead of
    failing it. Three experiments then filed the counter-example on one day: a
    candidate measured at ~2.0 s/day over 243 serial decision days died at
    ``strategy inference exceeded 180s`` on one rebalance day inside a 2-way
    batch. Both caps bound ONE call, so both are equally distorted by the
    fan-out, and the whole-replay runaway is bounded elsewhere (the session's
    own deadline).

    The batch owns the evaluator's config for the pool's lifetime, so
    mutating and restoring it here is safe: the parent runs its replaying
    tools one at a time, and a sub-agent's ``smoke_backtest`` (the only other
    replay) belongs to a role that can write, which the runner's writer
    barrier keeps from overlapping ``validate`` or ``test``. An evaluator
    without a strategy sandbox (trusted mode, test doubles) has no clock to
    scale and is left alone.
    """

    config = getattr(evaluator, "sandbox", None)
    if not isinstance(config, SandboxConfig):
        yield
        return
    evaluator.sandbox = replace(  # type: ignore[attr-defined]
        config,
        fit_timeout_seconds=config.fit_timeout_seconds * workers,
        timeout_seconds=config.timeout_seconds * workers,
    )
    try:
        yield
    finally:
        evaluator.sandbox = config  # type: ignore[attr-defined]


def replay_legs(
    evaluator: EvaluationBackend,
    requests: Sequence[EvaluationRequest],
    *,
    max_concurrency: int,
    end_day: str | None = None,
) -> list[_Replayed]:
    """Replay prepared requests, at most ``max_concurrency`` at a time.

    The one replay rule of ``validate``'s candidates and of a ``test``'s
    candidate with its seed replicates, so a leg's card and clocks never
    depend on which tool replays it: legs running side by side each take one
    of the experiment's cards (:func:`leg_cards`) under both strategy clocks
    widened to the width (:func:`_batch_replay_timeouts`); a width of one
    attaches every card under the base clocks. ``end_day`` stops every replay
    there (a test's horizon). Each evaluation owns its result directory, its
    as-of view and its strategy container; the shared Timeview stash
    serializes part publication with its own file locks. Outcomes come back
    in input order, so the caller's bookkeeping does not depend on which
    replay finished first.
    """

    workers = min(len(requests), max_concurrency)
    # One card per running leg, taken when the leg starts and handed back
    # when it ends, so a leg never lands on a card another running leg
    # holds while a free one waits.
    free_cards: queue.SimpleQueue[int] = queue.SimpleQueue()
    sandbox = getattr(evaluator, "sandbox", None)
    pinned = sandbox.spec.gpu if isinstance(sandbox, SandboxConfig) else None
    devices = pinned if isinstance(pinned, tuple) else ()
    cards = leg_cards(devices, workers)
    for card in cards:
        free_cards.put(card)

    def run_one(index: int) -> _Replayed:
        card = free_cards.get() if cards else None
        request = requests[index]
        if card is not None:
            request = replace(request, gpu_devices=(card,))
        started = time.perf_counter()
        try:
            evaluation = (
                evaluator.evaluate(request)
                if end_day is None
                else evaluator.evaluate(request, end_day=end_day)
            )
        except SessionInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - the text IS this leg's result
            return _Replayed(None, exc, started, time.perf_counter(), card)
        finally:
            if card is not None:
                free_cards.put(card)
        return _Replayed(evaluation, None, started, time.perf_counter(), card)

    if workers <= 1:
        return [run_one(index) for index in range(len(requests))]
    outcomes: dict[int, _Replayed] = {}
    interrupt: SessionInterrupt | None = None
    with (
        _batch_replay_timeouts(evaluator, workers),
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replay-leg") as pool,
    ):
        futures = {pool.submit(run_one, index): index for index in range(len(requests))}
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes[index] = future.result()
            except SessionInterrupt as exc:
                interrupt = exc
    if interrupt is not None:
        raise interrupt
    # Every leg returned its outcome or raised: none is missing here.
    return [outcomes[index] for index in range(len(requests))]


# A validate row's keys in the order the Agent reads them; a failed row has
# name, control, status, cause, error, wall_seconds and resources (a revealed
# one the first five, its error a fixed sentence). Once a step is tested, a
# completed full-span row also carries ``revealed`` (``ValidateTool._read_revealed``).
VALIDATE_ROW_KEYS = (
    "name",
    "control",
    "status",
    "cause",
    "error",
    "handle",
    "node_id",
    "revision_id",
    "wall_seconds",
    "active",
    "book",
    "by_year",
    "by_step",
    "concentration",
    "revealed",
    "resources",
    "same_bytes_as",
    "result_ref",
)


def _validate_row(row: Mapping[str, object]) -> dict[str, object]:
    return {key: row[key] for key in VALIDATE_ROW_KEYS if key in row}


@dataclass(frozen=True)
class _BatchCandidate:
    name: str
    hypothesis: str
    path: str
    directory: Path
    control: bool


class ValidateTool(SessionTimeBudgetAware):
    """The one Validation tool: candidates as sibling Steps.

    Every candidate gets its own ``modification_check``, its own immutable
    revision, one replay over the batch's span and one Step node; over the
    full span once a step is tested, also a reading over the tested steps
    (:meth:`_read_revealed`). The interval's ``validates_per_step``
    allowance rations readings: a call
    counts once, whatever its width, when at least one of its candidates
    completed its replay and returned a reading, and not at all when none
    did, whatever the cause. The candidates of a call share
    one parent node and one span, so a round's numbers are comparable and each
    hypothesis is recorded before its result, at every width from one to six.

    Selection is never automatic: the Agent reads the table and tests a
    candidate with ``test``.
    """

    spec = ToolSpec(
        "validate",
        f"Replay {BATCH_VALIDATE_MIN_CANDIDATES}-{BATCH_VALIDATE_MAX_CANDIDATES} "
        "candidates over one span in one call; the only way to make a node that "
        "test can take. A span of years replays as one continuous book from its "
        "first year. Once a step is tested, a full-span row also carries revealed: "
        "its candidate over the tested steps, replayed and reported as test does; "
        "span revealed replays only those steps. "
        "At most acceptance_rules.validation.validates_per_step counted calls "
        "before the first test and between two tests; a call counts only if one of "
        "its candidates completes its replay.\n\nA candidate's path is a workspace directory laid out "
        "like output/ (main.py and siblings; read-only template files are supplied; "
        "models/ is shared), or output itself; copy it from output/, not refs/. A "
        "completed candidate becomes an immutable revision and Step node under the "
        "current node; a failed one is a dead end with cause strategy (its own code "
        "raised, or it exceeded budgets.strategy_memory_bytes or "
        "budgets.strategy_state_disk_bytes) or environment (the host failed it: "
        "resending it unchanged is reasonable). The whole batch is refused if the "
        "calls are used up, two candidates are byte-identical or one fails "
        "modification_check. It "
        "waits briefly for sub-agents that can write and is refused while one "
        "runs.\n\nthreshold is z(1-alpha)/sqrt(years) for this span, as "
        "acceptance_rules.significance reads it. In each row, active is the graded "
        "series, book minus its random baseline: annualised excess, tracking_error "
        "and information_ratio, residual_* and betas from its OLS on the market, "
        "size and value legs, and excess_at_cost_stress (after "
        "cost_stress_multiplier x the slippage on the span's turnover); readings, "
        "not verdicts. book is the holder's money after every cost: max_drawdown is "
        "the equity drawdown registration limits, turnover is traded notional over "
        "opening capital, summed. concentration.best_year_share is the best year's "
        "active excess over the positive years' sum. Resource limits are budgets.* "
        "facts (strategy clocks scaled by the legs replaying at once). "
        "same_bytes_as names earlier nodes with these bytes on this span: the same "
        "trial, no new evidence. result_ref (under root steps) is the full record; "
        "validation/active_daily.csv beside it is the graded series day by day with "
        "the three legs.",
        {
            "type": "object",
            "properties": {
                "span": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 20,
                    "description": (
                        "full (default: the research years, plus a reading on the "
                        "validation steps tested so far), one research year Yk, "
                        "contiguous research years Yi..Yj, or revealed (only the "
                        "validation steps tested so far). Iterate on years; test "
                        "takes a full-span validation."
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
                                    "What this candidate is and what you expect "
                                    "of it, recorded before its result; at most "
                                    f"{BATCH_HYPOTHESIS_MAX_CHARS} characters "
                                    "(detail belongs in your notes)."
                                ),
                            },
                            "path": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": STRATEGY_PATH_MAX_CHARS,
                                "description": (
                                    "Workspace-relative directory laid out "
                                    "like output/, e.g. candidates/value."
                                ),
                            },
                            "control": {
                                "type": "boolean",
                                "description": (
                                    "Optional, default false. true marks a "
                                    "comparison leg: it replays like any "
                                    "candidate, but test never takes it."
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
                    "name": "lgbm_ranker",
                    "hypothesis": "LightGBM 横截面排序（季度重训）的主动超额在两个年份都为正",
                    "path": "candidates/lgbm_ranker",
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
        trace_emit: Callable[[TraceEvent, dict[str, object]], object] | None = None,
        max_concurrency: int = DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
    ) -> None:
        self.backtest = backtest
        self.workspace = workspace
        self.modification_check_factory = modification_check_factory
        self._trace_emit = trace_emit
        # The experiment's replay width (``batch_validate_max_concurrency``), already
        # validated with its other budgets.
        self.max_concurrency = max_concurrency
        # Per-session rejection counter, keyed by signature. Not persisted:
        # the loop it bounds is one session's retry loop.
        self._rejections: dict[tuple[str, str], int] = {}
        # Batches whose every candidate failed on the environment, per span
        # label; per attempt like the rejection counter.
        self._environment_failed: dict[str, int] = {}

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.backtest.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        self.backtest.check_deadline()
        # A host condition, not the Agent's input: refused before anything is
        # reserved and outside the repeated-rejection counter, so it is never
        # charged however often it repeats.
        self.workspace.require_free_space("validate")
        with self.backtest.time_budget.pause():
            return self._invoke_exempt(arguments)

    def _invoke_exempt(self, arguments: Mapping[str, object]) -> ToolResult:
        # Everything that can refuse the batch runs before any replay starts,
        # so a rejected batch costs nothing and the Agent can fix the
        # offending input and call again. Because it costs nothing, the same
        # rejection can also repeat forever: ``_rejected`` counts it.
        if self.backtest.validates_in_interval >= self.backtest.validates_per_step:
            raise ToolError(
                f"validate refused: acceptance_rules.validation.validates_per_step allows "
                f"{self.backtest.validates_per_step} counted validate call(s) before the first test and "
                f"between two tests, and this interval has counted {self.backtest.validates_in_interval}; "
                "0 remain in this interval. shell, smoke_backtest, sub-agents and offline "
                "evaluation stay available.",
                error_type="validates_per_step_exhausted",
                details={
                    "validates_in_step": self.backtest.validates_in_interval,
                    "validates_per_step": self.backtest.validates_per_step,
                },
            )
        try:
            span, revealed = self.backtest.span(arguments.get("span", FULL_SPAN))
            candidates = self._parse(arguments)
            for candidate in candidates:
                supply_readonly_files(
                    self.backtest.output_dir,
                    candidate.directory,
                    who=f"candidate {candidate.name} ({candidate.path})",
                    path=candidate.path,
                )
            checks = self._precheck(candidates)
        except ToolError as exc:
            escalated = self._rejected(exc)
            if escalated is exc:
                raise
            raise escalated from exc
        result_names = self.backtest.reserve(len(candidates))
        batch_id = uuid.uuid4().hex[:12]
        parent_node_id = self.backtest.tree.current_node_id
        revisions = self._commit(candidates, checks, span)
        # The revealed span stops where its last tested step ends, as the
        # test that revealed that step did.
        end_day = None if revealed is None else self.backtest.steps_plan.horizon_after(revealed)
        outcomes = replay_legs(
            self.backtest.evaluator,
            [self.backtest.validation_request(revision, span) for revision in revisions],
            max_concurrency=self.max_concurrency,
            end_day=end_day,
        )
        # How many candidates the host, not the strategy, failed: measured
        # nothing, so the row says the candidate may be resent unchanged.
        environment_failures = sum(
            1
            for replayed in outcomes
            if replayed.error is not None
            and _failure_cause(replayed.error) == BATCH_FAILURE_ENVIRONMENT
        )
        # A batch does not re-check the deadline here: dropping N completed
        # Validations because the clock ran out during them would destroy real
        # evidence. The session deadline is enforced at the next dispatch and
        # LLM call.
        rows: list[dict[str, object]] = []
        recorded: list[tuple[dict[str, object], StepResult]] = []
        try:
            for index, (candidate, check, revision, result_name, replayed) in enumerate(
                zip(candidates, checks, revisions, result_names, outcomes, strict=True)
            ):
                evaluation, error = replayed.evaluation, replayed.error
                # Every candidate — recorded or dead end — hangs off the node
                # the batch started at, never off the sibling before it.
                self.backtest.tree.set_position(parent_node_id)
                row: dict[str, object] = {
                    "name": candidate.name,
                    "control": candidate.control,
                }
                if evaluation is not None or revealed is None:
                    # How long a failed revealed replay ran would date its failure.
                    row["wall_seconds"] = round(replayed.seconds, 1)
                metadata = {
                    "batch_id": batch_id,
                    "candidate": candidate.name,
                    "hypothesis": candidate.hypothesis,
                    "control": candidate.control,
                    "source_path": candidate.path,
                    "span": span.label,
                }
                if evaluation is None and revealed is not None:
                    assert error is not None  # a replay holds one of the two
                    row.update(self._record_failure(result_name, error, metadata, validation_period=True))
                elif evaluation is None:
                    assert error is not None  # a replay holds one of the two
                    shared = (
                        [
                            candidates[other].name
                            for other, sibling in enumerate(outcomes)
                            if other != index and sibling.shares_card_with(replayed)
                        ]
                        if gpu_contended(error)
                        else []
                    )
                    row.update(
                        self._record_failure(result_name, error, metadata, shared_gpu=shared)
                    )
                else:
                    row.update(
                        self._record_success(
                            revision,
                            evaluation,
                            result_name=result_name,
                            # commit_revision verified the revision holds these bytes.
                            fingerprint=str(check["fingerprint"]),
                            metadata=metadata,
                            validation_period=revealed is not None,
                        )
                    )
                    recorded.append((row, self.backtest.steps[-1]))
                    if len(recorded) == 1:
                        # The batch's first reading is what the allowance
                        # rations: the call counts here, once, and a batch
                        # that completes no candidate never does
                        # (``_validates_since`` counts the same on resume).
                        self.backtest.validates_in_interval += 1
                rows.append(row)
        finally:
            # The batch never touched the working copy, so the tree position
            # comes back to where it branched from even if recording failed:
            # the Agent moves it deliberately with step_rollback once it picks
            # a winner.
            self.backtest.tree.set_position(parent_node_id)
        if not recorded:
            if environment_failures == len(rows):
                self._environment_blocked(span, rows)
            raise ToolError(
                f"validate: all {len(rows)} candidates failed their "
                f"Validation; {len(rows) - environment_failures} of them raised in their own code, "
                f"{environment_failures} failed on the environment. No Validation was recorded "
                "and the call was not counted",
                error_type="validation_failed",
                details={"batch_id": batch_id, "candidates": [_validate_row(row) for row in rows]},
            )
        thresholds: list[object] = []
        for row, step in recorded:
            row.update(self._reading(step.validation, revealed))
            thresholds.append(row["active"].pop("threshold", None))  # type: ignore[attr-defined]
        if span.label == FULL_SPAN and self.backtest.tests:
            # Every full-span node, once a step is tested, read over it too.
            self._read_revealed(recorded, batch_id)
        # One span, one length, one threshold: a row whose regression read a
        # different number of days keeps its own.
        threshold = next((value for value in thresholds if value is not None), None)
        for (row, _step), own in zip(recorded, thresholds, strict=True):
            if own is not None and own != threshold:
                row["active"]["threshold"] = own  # type: ignore[index]
        first = recorded[0][1].validation.summary
        benchmark = first.get("benchmark")
        return ToolResult(
            True,
            value={
                "batch_id": batch_id,
                **({"parent_node_id": parent_node_id} if parent_node_id else {}),
                "validates_in_step": self.backtest.validates_in_interval,
                "validates_per_step": self.backtest.validates_per_step,
                "complete_validations": len(recorded),
                "span": {
                    "label": span.label,
                    "start": span.start,
                    "end": end_day or span.end,
                    # The revealed steps' length in years, as a year span states its own.
                    "years": span.slots
                    if revealed is None
                    else revealed * self.backtest.rules.test_step_months / 12,
                    **({} if revealed is None else {"steps": revealed}),
                    "days": first.get("replayed_trade_days"),
                },
                "threshold": threshold,
                **(
                    {
                        "benchmark": {
                            "ts_code": benchmark.get("ts_code"),
                            "label": benchmark.get("label"),
                            "return": benchmark.get("benchmark_return"),
                        }
                    }
                    if isinstance(benchmark, Mapping)
                    else {}
                ),
                "candidates": [_validate_row(row) for row in rows],
                "failed": len(rows) - len(recorded),
            },
        )

    def _environment_blocked(
        self, span: ReplaySpan, rows: Sequence[Mapping[str, object]]
    ) -> None:
        """Count a batch the environment failed whole on ``span``; raise
        :class:`EnvironmentBlocked` once the span reaches the threshold."""

        count = self._environment_failed.get(span.label, 0) + 1
        self._environment_failed[span.label] = count
        if count < ENVIRONMENT_BLOCKED_BATCHES:
            return
        reason = (
            f"span {span.label}: {count} batches failed on the environment in every "
            f"candidate; last error: {rows[-1]['error']}"
        )
        if self._trace_emit is not None:
            self._trace_emit(
                TraceEvent.ENVIRONMENT_BLOCKED,
                {"tool": "validate", "span": span.label, "batches": count, "reason": reason},
            )
        raise EnvironmentBlocked(reason)

    # ---- input ----

    def _parse(self, arguments: Mapping[str, object]) -> list[_BatchCandidate]:
        raw = arguments.get("candidates")
        if not isinstance(raw, list):
            raise ToolError(
                "validate candidates must be an array",
                error_type="schema_error",
            )
        if not (
            BATCH_VALIDATE_MIN_CANDIDATES
            <= len(raw)
            <= BATCH_VALIDATE_MAX_CANDIDATES
        ):
            raise ToolError(
                f"validate takes {BATCH_VALIDATE_MIN_CANDIDATES} to "
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
                    "hypothesis, path and an optional control",
                    error_type="schema_error",
                    blocked_target=str(index),
                )
            unknown = sorted(set(item) - {"name", "hypothesis", "path", "control"})
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
            path = _batch_text(item, "path", index, STRATEGY_PATH_MAX_CHARS)
            control = item.get("control", False)
            if not isinstance(control, bool):
                raise ToolError(
                    f"candidate {index} control must be true (a comparison leg "
                    "test never takes) or false; leave it out for a candidate",
                    error_type="schema_error",
                    blocked_target="control",
                )
            if name in names:
                raise ToolError(
                    f"duplicate candidate name: {name}",
                    error_type="schema_error",
                    blocked_target=name,
                )
            names.add(name)
            directory = strategy_directory(self.workspace, path, who=f"candidate {name}")
            if str(directory) in directories:
                raise ToolError(
                    f"duplicate candidate path: {path}",
                    error_type="schema_error",
                    blocked_target=path,
                )
            directories.add(str(directory))
            parsed.append(_BatchCandidate(name, hypothesis, path, directory, control))
        return parsed

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
        recovery = _rejection_recovery(exc.error_type)
        message = (
            f"{exc}\n[repeated rejection] validate has now refused this "
            f"exact rejection {count} times; calling it again unchanged returns "
            f"the same refusal. {recovery}"
        )
        if self._trace_emit is not None:
            self._trace_emit(
                TraceEvent.BATCH_REJECTION_ESCALATED,
                {
                    "tool": "validate",
                    "error_type": exc.error_type,
                    "blocked_target": signature[1],
                    "repeat_count": count,
                },
            )
        details = {**exc.details, "repeat_count": count}
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
            # every revision and every reserved name goes back.
            for revision in revisions:
                self.backtest.artifact_store.discard_revision(revision.revision_id)
            self.backtest.release(len(candidates))
            public_error = _public_error_text(exc)
            # Recorded as what it was: every attempt reaches the run manifest.
            self.backtest.append_manifest_summary(
                {
                    "mode": RESEARCH_PHASE,
                    "status": "infrastructure_error",
                    "complete_validation": False,
                    "span": span.label,
                    "error": public_error,
                }
            )
            raise ToolError(
                "validate could not start: " + public_error,
                error_type="infrastructure_error",
                retry_hint=(
                    "A candidate directory was still being written when its "
                    "snapshot was taken and nothing ran; no budget was "
                    "consumed. Let any in-container job finish, then call "
                    "validate again."
                ),
            ) from exc
        return revisions

    # ---- recording ----

    def _record_success(
        self,
        revision: ArtifactRevision,
        evaluation: EvaluationResult,
        *,
        result_name: str,
        fingerprint: str,
        metadata: Mapping[str, object],
        validation_period: bool = False,
    ) -> dict[str, object]:
        # Recorded before this one: Validations of the same bytes on the same
        # span, in any attempt (a resumed attempt's Steps carry their bytes).
        repeated = [
            step.step_id
            for step in self.backtest.steps
            if step.fingerprint == fingerprint and step.span == metadata["span"]
        ]
        node_id = self.backtest.record_validation(
            revision,
            evaluation,
            result_name=result_name,
            fingerprint=fingerprint,
            metadata=metadata,
            validation_period=validation_period,
        )
        self.backtest.steps.append(
            recorded_step(node_id, revision.revision_id, evaluation, metadata, fingerprint)
        )
        self.backtest.append_manifest_summary(
            {
                "result_name": result_name,
                "mode": RESEARCH_PHASE,
                "status": "ok",
                "complete_validation": True,
                **{key: metadata[key] for key in _BATCH_MANIFEST_KEYS},
                **(validation_period_stats if validation_period else manifest_backtest_stats)(
                    evaluation.summary
                ),
            }
        )
        row: dict[str, object] = {
            "status": "ok",
            # The batch's later siblings take later result names, so a handle
            # unambiguous now stays so for the rest of the batch.
            **self.backtest.handle(node_id),
            "node_id": node_id,
            "revision_id": self.backtest.ref_store.get_or_create(
                "strategy", revision.revision_id
            ),
            # The random baseline is seeded by the span, so a deterministic
            # strategy repeats these nodes' readings exactly: the same trial.
            **({"same_bytes_as": self.backtest.named(repeated)} if repeated else {}),
        }
        if validation_period:
            # What a test returns of its replay and no more: no record, and
            # no container readings, which are the strategy's to shape.
            return row
        public_result_ref = f"{node_id}/{VALIDATION_RESULT_ATTACHMENT}"
        if not (self.backtest.tree.root / public_result_ref).is_file():
            raise ToolError(
                "Validation result attachment is missing for the recorded step: "
                f"{public_result_ref}"
            )
        return {
            **row,
            **resource_reading(evaluation.summary.get("resources")),
            # The node's directory under the ``steps`` root; its
            # active_daily.csv sits beside the record.
            "result_ref": public_result_ref,
        }

    def _record_failure(
        self,
        result_name: str,
        error: Exception,
        batch_metadata: Mapping[str, object],
        *,
        shared_gpu: Sequence[str] = (),
        validation_period: bool = False,
    ) -> dict[str, object]:
        """Record one failed candidate. Over the ``validation_period`` the
        session reads its fixed sentence alone (:func:`validation_period_failure`)
        and the host keeps the rest."""

        if validation_period:
            public_error = f"daily Validation failed: {validation_period_failure(error)}"
            record_validation_period_failure(
                self.backtest.experiment_dir,
                error,
                tool="validate",
                span=batch_metadata["span"],
                candidate=batch_metadata["candidate"],
                batch_id=batch_metadata["batch_id"],
            )
        else:
            public_error = _public_validation_error(error)
        if shared_gpu:
            # The dead-end node and the manifest keep the note with the error.
            public_error += _shared_gpu_note(shared_gpu)
        cause = _failure_cause(error)
        request = self.backtest.request
        metadata = {
            **batch_metadata,
            # A later session reading this dead end needs to know whether the
            # hypothesis was falsified or the host simply got in the way.
            "cause": cause,
        }
        if request.record_failed_attempts:
            # record_failed_attempt leaves the tree position alone by design,
            # so a dead end never becomes anybody's parent.
            self.backtest.tree.record_failed_attempt(
                epoch_id=SessionKind.RESEARCH,
                session_ref=self.backtest.session_ref,
                run_id=self.backtest.ref_store.get_or_create("run", request.run_id),
                result_name=result_name,
                error=public_error,
                metadata=metadata,
            )
            self.backtest.publish_tree()
        self.backtest.append_manifest_summary(
            {
                "result_name": result_name,
                "mode": RESEARCH_PHASE,
                "status": "failed",
                "complete_validation": False,
                **{key: metadata[key] for key in (*_BATCH_MANIFEST_KEYS, "cause")},
                "error": public_error,
            }
        )
        row: dict[str, object] = {
            "status": "failed",
            "cause": cause,
            "error": public_error,
        }
        if not validation_period:
            # The failed row's telemetry comes off the exception — there is no
            # result to carry it — and lands where a successful row keeps its own:
            # a fit that ran out of clock or a card that was taken says so here,
            # measured in the container it actually ran in.
            row.update(resource_reading(strategy_resources_of(error)))
        return row

    # ---- readings ----

    def _reading(self, evaluation: EvaluationResult, revealed: int | None) -> dict[str, object]:
        """One completed replay as a row reads it: ``active`` (its
        ``threshold`` still inside), ``book``, and over research years
        ``by_year`` with ``concentration`` or, over the ``revealed`` steps,
        ``by_step``, the steps read as a test reads its candidate."""

        result_ref, summary = evaluation.result_ref, evaluation.summary
        analysis = result_analysis(result_ref)
        reading: dict[str, object] = {
            "active": active_reading(analysis, summary, rules=self.backtest.rules),
            "book": book_reading(analysis, summary),
        }
        if revealed is None:
            reading.update(year_readings(analysis, summary, self.backtest.request.years))
        else:
            reading["by_step"] = step_readings(
                *replay_and_analysis(result_ref), self.backtest.steps_plan, through=revealed
            )
        return reading

    def _read_revealed(
        self, recorded: Sequence[tuple[dict[str, object], StepResult]], batch_id: str
    ) -> None:
        """Each recorded row of a full-span call made once a step is tested
        gets ``revealed``: its candidate over the tested steps, replayed and
        read exactly as span revealed replays and reads it
        (:meth:`SessionValidations.span`), its own threshold kept in
        ``active``. A replay that fails gives its class alone, as a failed
        revealed candidate does, and the host keeps the rest.

        A reading only: no node, record or registration reads it, so each
        node, its testability and the allowance are what its research replay
        made them. Those replays have all finished, so they ran at the width
        and clocks they always do; these share the call's replay width.
        """

        backtest = self.backtest
        span, tested = backtest.span(REVEALED_SPAN)
        outcomes = replay_legs(
            backtest.evaluator,
            [
                backtest.validation_request(backtest.artifact_store.revision(step.revision_id), span)
                for _row, step in recorded
            ],
            max_concurrency=self.max_concurrency,
            end_day=backtest.steps_plan.horizon_after(tested),  # type: ignore[arg-type]
        )
        for (row, _step), replayed in zip(recorded, outcomes, strict=True):
            if replayed.evaluation is not None:
                row["revealed"] = self._reading(replayed.evaluation, tested)
                continue
            error = replayed.error
            assert error is not None  # a replay holds one of the two
            record_validation_period_failure(
                backtest.experiment_dir,
                error,
                tool="validate",
                span=span.label,
                candidate=row["name"],
                batch_id=batch_id,
            )
            row["revealed"] = {
                "status": "failed",
                "cause": _failure_cause(error),
                "error": validation_period_failure(error),
            }


TEST_DESCRIPTION = (
    "Advance the validation period one step with candidate, a complete full-span "
    "validate node of this session. It must register as "
    "acceptance_rules.registration defines; otherwise it is refused with "
    "reasons and the horizon stays. seed_replicates may name full-span nodes "
    "whose bytes differ from the candidate's in exactly one integer seed assignment "
    "(e.g. SEED_BASE = 1000 -> 2000); each test naming them replays each beside the "
    "candidate in its own replay slot (budgets.batch_validate_max_concurrency): "
    "with fewer slots than 1 + replicates a step takes about ceil((1 + replicates) "
    "/ slots) replays of wall clock, none of it session time.\n\nThe candidate "
    "replays as one fresh book from the validation start to the end of the next "
    "step, so switching candidates costs only this replay. Its newest step becomes "
    "that step's entry in the validation record, never rewritten; the horizon "
    "advances and the validate allowance restarts. If the candidate raises or the "
    "host fails a leg, nothing is recorded; a replicate whose own code raises is "
    "recorded. Failures report only their class. It waits for every background "
    "sub-agent to "
    "finish; the last step ends the session with this candidate "
    "registered.\n\nwhole_replay reads this replay as a validate row does. "
    "candidate_by_step is this replay's view of every step so far, beside each "
    "replicate's: a counterfactual for steps recorded with another candidate. "
    "record is authoritative, as recorded. significance reads the record as "
    "acceptance_rules.significance defines."
)


class TestTool(SessionTimeBudgetAware):
    """``test(candidate)``: one step of the validation period, lived through.

    The session's only way forward and the only way it ends: the test horizon
    advances one step per call, the step's entry is appended to the experiment's
    validation record in the ledger as the call completes, and the call that
    tests the last step finishes the session with its candidate as the registered
    strategy. Nothing here ranks or selects: the agent chooses the candidate,
    the host replays and records.
    """

    spec = ToolSpec(
        "test",
        TEST_DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "candidate": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": NODE_REFERENCE_DESCRIPTION,
                },
                "seed_replicates": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 500},
                    "description": (
                        "Optional: the candidate's seed replicates as "
                        "acceptance_rules.registration.seed_replicates defines them, each a "
                        "node id or short handle of this session."
                    ),
                },
            },
            "required": ["candidate"],
            "additionalProperties": False,
        },
        mutating=True,
        example={"candidate": NODE_REFERENCE_EXAMPLE},
    )

    def __init__(
        self,
        backtest: SessionValidations,
        *,
        max_concurrency: int = DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY,
    ) -> None:
        self.backtest = backtest
        # The replay width ``validate`` has (``batch_validate_max_concurrency``):
        # the candidate and its seed replicates replay as one batch's legs.
        self.max_concurrency = max_concurrency
        # Consecutive test replays the host, not the strategy, failed: the
        # second ends the attempt, as ``validate`` does for one span.
        self._environment_failed = 0

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.backtest.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        self.backtest.check_deadline()
        backtest = self.backtest
        candidates = [item.step_id for item in backtest.steps]
        node_id = session_node_reference(
            "test", backtest.tree, str(arguments.get("candidate") or ""),
            session_ref=backtest.session_ref, offered=candidates,
        )
        replicates = tuple(
            session_node_reference(
                "test", backtest.tree, str(item), session_ref=backtest.session_ref, offered=candidates
            )
            for item in arguments.get("seed_replicates") or ()  # type: ignore[union-attr]
        )
        registration = dict(backtest.registration(node_id, replicates))
        if not registration.get("passed"):
            reasons = [str(item) for item in registration.get("reasons") or ()]
            raise ToolError(
                f"test refused: {backtest.named([node_id])[0]} does not register "
                f"({', '.join(reasons) or 'no reason recorded'}); the test horizon did not move. "
                "Name a node that registers, or validate what registration lacks (a full-span "
                "validation). A seed replicate, when you name one, is what "
                "acceptance_rules.registration.seed_replicates defines.",
                error_type="test_registration_refused",
                details={
                    "registration": {
                        "passed": False,
                        "reasons": reasons,
                        **({"seed_replicates": registration["seed_replicates"]} if "seed_replicates" in registration else {}),
                    }
                },
            )
        steps = backtest.steps_plan
        step = len(backtest.tests) + 1
        if step > steps.count:
            raise ToolError("test refused: the validation period is complete", error_type="validation_complete")
        test_horizon = steps.horizon_after(step)
        # The candidate and the seed replicates named, replayed side by side.
        legs = [next(item for item in backtest.steps if item.step_id == named) for named in (node_id, *replicates)]
        names = backtest.named(replicates)
        with backtest.time_budget.pause():
            # The prebuilt as-of state the replay opens on must be in the seed before
            # the replay runs: a seed that lacks it ends the attempt by name.
            try:
                require_prebuilt = getattr(backtest.evaluator, "require_prebuilt_asof_state", None)
                requests = [
                    backtest.validation_request(
                        backtest.artifact_store.revision(leg.revision_id),
                        backtest.request.validation,
                    )
                    for leg in legs
                ]
                if callable(require_prebuilt):
                    require_prebuilt(requests[0])
            except (FileNotFoundError, RuntimeError) as exc:
                raise EnvironmentBlocked(f"test step {step}: {_public_error_text(exc)}") from exc
            started = time.perf_counter()
            outcomes = replay_legs(
                backtest.evaluator, requests, max_concurrency=self.max_concurrency, end_day=test_horizon
            )
            backtest.test_replays += 1
            failed = _failed_leg(outcomes)
            if failed is not None:
                index, exc = failed
                cause = _failure_cause(exc)
                # The step is not recorded, so the session reads the failure's
                # class alone (the reason, also read by a resumed attempt,
                # included) and the host keeps its detail.
                record_validation_period_failure(
                    backtest.experiment_dir, exc, tool="test", step=step, candidate=node_id,
                    leg=legs[index].step_id,
                )
                failure = validation_period_failure(exc)
                if cause == BATCH_FAILURE_ENVIRONMENT:
                    self._environment_failed += 1
                    if self._environment_failed >= ENVIRONMENT_BLOCKED_BATCHES:
                        raise EnvironmentBlocked(
                            f"test step {step}: {self._environment_failed} replays failed on the "
                            f"environment; last: {failure}"
                        ) from exc
                which = "The candidate" if index == 0 else f"Seed replicate {names[index - 1]}"
                raise ToolError(
                    f"test step {step} did not complete; the test horizon did not move. "
                    f"{which} failed on the validation data: {failure}"
                    + (
                        "."
                        if cause == BATCH_FAILURE_STRATEGY
                        else " (cause environment), so the same call may be resent unchanged."
                    ),
                    error_type="test_replay_failed",
                    details={
                        "cause": cause,
                        **({"seed_replicate": names[index - 1]} if index else {}),
                    },
                ) from exc
            self._environment_failed = 0
            evaluation = outcomes[0].evaluation
            assert evaluation is not None
            replay, analysis = replay_and_analysis(evaluation.result_ref)
            window = steps.window(step)
            entry = step_entry(replay, analysis, start=window[0], end=window[1])
            # Each replicate's rows over every step so far, or the class of
            # the failure its own code met: a reading, recorded beside the
            # candidate's with the error text, which only the host reads.
            replicate_rows: list[list[dict[str, object]] | str] = []
            replicate_records: list[dict[str, object]] = []
            for leg, outcome in zip(legs[1:], outcomes[1:], strict=True):
                identity = {"step_id": leg.step_id, "revision_id": leg.revision_id}
                if outcome.evaluation is None:
                    failure = validation_period_failure(outcome.error)  # type: ignore[arg-type]
                    replicate_rows.append(failure)
                    replicate_records.append(
                        {
                            **identity,
                            "result_ref": None,
                            "strategy_error": _public_error_text(outcome.error),  # type: ignore[arg-type]
                            "failure": failure,
                        }
                    )
                    continue
                leg_replay, leg_analysis = replay_and_analysis(outcome.evaluation.result_ref)
                replicate_rows.append(step_table(leg_replay, leg_analysis, steps, through=step))
                replicate_records.append(
                    {
                        **identity,
                        "result_ref": outcome.evaluation.result_ref,
                        "entry": step_entry(leg_replay, leg_analysis, start=window[0], end=window[1]),
                    }
                )
            record = test_step_record(
                head=backtest.ledger_head(),
                steps=steps,
                step=step,
                candidate={
                    "step_id": node_id,
                    "revision_id": legs[0].revision_id,
                    "fingerprint": legs[0].fingerprint,
                    "seed_replicates": list(replicates),
                },
                replay={"start": steps.start, "end": test_horizon, "result_ref": evaluation.result_ref},
                entry=entry,
                registration=registration,
                source=SOURCE_TEST,
                replicates=replicate_records,
            )
            # Durable before anything is returned: a resumed attempt
            # continues from this test horizon.
            backtest.ledger.append(record)
            backtest.tests.append(record)
            backtest.validates_in_interval = 0
            finished = step == steps.count
        # This replay's own view of every step from the validation start, the
        # replicates' beside the candidate's: a counterfactual for the steps
        # the record holds with another candidate.
        by_step = [
            {
                **row,
                **(
                    {
                        "seed_replicates": [
                            _replicate_cell(name, rows if isinstance(rows, str) else rows[index])
                            for name, rows in zip(names, replicate_rows, strict=True)
                        ]
                    }
                    if replicates
                    else {}
                ),
            }
            for index, row in enumerate(step_readings(replay, analysis, steps, through=step))
        ]
        active = active_reading(analysis, evaluation.summary, rules=backtest.rules)
        # The record's own threshold is the one significance reads.
        active.pop("threshold", None)
        significance = record_significance([row["entry"] for row in backtest.tests], alpha=backtest.rules.alpha)
        value: dict[str, object] = {
            "status": "step_recorded",
            "step": step,
            "steps": steps.count,
            "steps_left": steps.count - step,
            "test_horizon": test_horizon,
            "candidate": {
                **backtest.handle(node_id),
                # The host registers the last test's candidate and replicates
                # by these full ids.
                "node_id": node_id,
                **({"seed_replicates": list(replicates)} if replicates else {}),
            },
            "replay": {"start": steps.start, "end": test_horizon, "wall_seconds": round(time.perf_counter() - started, 1)},
            "whole_replay": {"active": active, "book": book_reading(analysis, evaluation.summary)},
            "candidate_by_step": by_step,
            "record": [backtest.record_row(row) for row in backtest.tests],
            "significance": _rounded_row(
                {
                    **{key: significance[key] for key in SIGNIFICANCE_KEYS if key in significance},
                    **({"unmeasured": significance["unmeasured"]} if significance.get("unmeasured") else {}),
                }
            ),
        }
        if finished:
            value.update(
                outcome="completed",
                pipeline_outcome=(
                    f"The validation period is complete: {backtest.named([node_id])[0]} is the registered "
                    "strategy; the host replays it over the Forward year and the Held-out months and "
                    "records the readings. The session ends here."
                ),
            )
        return ToolResult(True, value=value, finish=finished)


def _failed_leg(outcomes: Sequence[_Replayed]) -> tuple[int, Exception] | None:
    """The leg whose failure fails a test step, by its index (0 the
    candidate): the candidate's own failure, else the first replicate the
    host failed. A replicate whose own code raised fails nothing; its error
    is its reading."""

    for index, outcome in enumerate(outcomes):
        error = outcome.error
        if error is not None and (index == 0 or _failure_cause(error) == BATCH_FAILURE_ENVIRONMENT):
            return index, error
    return None


def _rounded_row(row: Mapping[str, object]) -> dict[str, object]:
    """A result row with its floats at six decimals and no daily rows."""

    return {
        key: _rounded(value) if isinstance(value, float) else value
        for key, value in row.items()
        if key != "daily"
    }


# What a test reads of each step: a candidate_by_step row (and a revealed
# validate row's by_step) and each seed replicate's cell beside it carry the
# step's active return and return; a record row carries the recorded entry's
# figures.
REPLICATE_STEP_KEYS = ("active_return", "return")
RECORD_ENTRY_KEYS = ("return", "active_return", "index_return", "max_drawdown", "turnover", "fees")
# The record's significance test as a test result carries it.
SIGNIFICANCE_KEYS = (
    "residual_information_ratio",
    "threshold",
    "p_value",
    "residual_excess",
    "residual_tracking_error",
    "market_beta",
    "size_beta",
    "value_beta",
    "days",
    "years",
)


def step_readings(
    replay: Mapping[str, object],
    analysis: Mapping[str, object],
    steps: ValidationSteps,
    *,
    through: int,
) -> list[dict[str, object]]:
    """A replay from the validation start read over steps 1..``through``
    (``validation_record.step_table``): each step's active return and return,
    the rows a test returns as ``candidate_by_step``."""

    return [
        {"step": row["step"], **_rounded_row({key: row[key] for key in REPLICATE_STEP_KEYS})}
        for row in step_table(replay, analysis, steps, through=through)
    ]


def _replicate_cell(handle: str, row: Mapping[str, object] | str) -> dict[str, object]:
    """One seed replicate in a step's row: its handle with its own active
    return and return over the step, or, where ``row`` is the class of the
    failure its own code met (:func:`validation_period_failure`), that
    sentence as ``strategy_error``."""

    if isinstance(row, str):
        return {"handle": handle, "strategy_error": row}
    return {"handle": handle, **_rounded_row({key: row[key] for key in REPLICATE_STEP_KEYS})}


class NullControlTool(SessionTimeBudgetAware):
    """Rank one complete Validation against random-name replays of its trades.

    The same null control the Pipeline runs for the registered node at registration
    (``experiment._null_control``), drawn with the same seed through the same
    backend over the node's own span, so the figure the Agent reads before
    selecting is the figure the ledger records: the block is cached per node
    and handed to the Pipeline, which reuses it for the registered node instead of
    drawing again. Each call is host replay the session does not pay for, hence
    the per-session cap.
    """

    spec = ToolSpec(
        # The tool is a verb (like ``validate``): ``null_control`` alone
        # is the result/ledger block it produces, and one name for both made
        # every mention of the block read as a tool reference.
        "run_null_control",
        "Random-portfolio null control of one complete Validation node.",
        {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": NODE_REFERENCE_DESCRIPTION,
                },
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
        # Sequential, like a formal backtest: it
        # pauses the session clock and spends a capped budget.
        mutating=True,
        example={"node_id": NODE_REFERENCE_EXAMPLE},
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
            "Rank a complete research-span Validation node of this session against "
            f"{BASELINE_DRAWS} host replays of its own trade skeleton over its span, "
            "each name replaced by a random one on the same side of the "
            "benchmark_index membership (the float-cap decile without index_weight) "
            "that one board lot of the same money could buy: the null control the "
            "Pipeline runs at registration, which reuses this cached block. "
            "excess_percentile near 0.5 means the names added nothing the timing "
            f"and sizing did not. At most {max_calls} calls per session "
            "(budgets.max_null_controls; a call the host cannot set up costs none); "
            "refused for any other node, once spent, and while a sub-agent that can "
            "write runs. Use it on full-span finalists.",
            self.spec.input_schema,
            mutating=True,
            example=self.spec.example,
        )

    @property
    def session_time_budget(self) -> InferenceTimeBudget:
        return self.backtest.time_budget

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        self.backtest.check_deadline()
        candidates = [item.step_id for item in self.backtest.steps]
        node_id = session_node_reference(
            "run_null_control",
            self.backtest.tree,
            str(arguments.get("node_id") or ""),
            session_ref=self.backtest.session_ref,
            offered=candidates,
        )
        step = next((item for item in self.backtest.steps if item.step_id == node_id), None)
        if step is None:
            raise ToolError(
                "run_null_control requires the node_id of a complete Validation of "
                f"this session; {node_id} is not one",
                details={"candidates": self.backtest.named(candidates)},
            )
        try:
            span = research_span(
                self.backtest.request.years,
                step.span,
                research_years=self.backtest.rules.research_years,
            )
        except ValueError as exc:
            # A revealed-span node: the null control, like registration,
            # reads a node over the research years.
            raise ToolError(
                "run_null_control ranks a node replayed over research years (span full, Yk "
                f"or Yi..Yj); {self.backtest.named([node_id])[0]} replayed span {step.span}",
                error_type="schema_error",
                blocked_target="node_id",
            ) from exc
        if node_id in self.blocks:
            return ToolResult(True, value=self._report(node_id, self.blocks[node_id], cached=True))
        if self.used >= self.max_calls:
            raise ToolError(
                f"run_null_control budget exhausted: {self.max_calls} per session "
                "(max_null_controls). The registered node's null control still runs at "
                "registration.",
                error_type="null_control_budget_exhausted",
            )
        runner = getattr(self.backtest.evaluator, "null_control", None)
        if not callable(runner):
            raise ToolError("run_null_control is not available on this evaluation backend")
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
                    seed=FROZEN_NULL_SEED,
                )
            except SessionInterrupt:
                raise
            except Exception as exc:
                if isinstance(exc, NullControlSetupError):
                    # It never reached its first draw, so none of the host
                    # compute the charge stands for was spent and nothing was
                    # measured: give the call back.
                    self.used -= 1
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
            "handle": self.backtest.named([node_id])[0],
            "null_control": allowed_keys(block, NULL_CONTROL_KEYS),
            "cached": cached,
            "null_controls_used": self.used,
            "null_controls_remaining": self.max_calls - self.used,
        }


def _rejection_recovery(error_type: str) -> str:
    """What to actually do about a ``validate`` rejection that repeats.

    The first rejection already names the offending candidate; what it does not
    name is the remedy, which is why the audited loop kept re-sending the same
    batch. One sentence per signature class, concrete enough to act on.
    """

    if error_type == "readonly_baseline":
        return (
            "Recovery: the refusal's details.readonly_violations name the "
            "read-only file that no longer holds the bytes this session was "
            "seeded with. Delete your copy of it from the candidate directory; "
            "validate supplies the read-only template itself."
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
