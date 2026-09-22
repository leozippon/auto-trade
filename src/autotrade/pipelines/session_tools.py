"""The research session's Validation engine and the Agent-facing backtest tools.

``SessionValidations`` holds the session's replay-year budget, its Step list and
its step tree; ``smoke_backtest``, ``batch_validate`` and ``run_null_control``
are the three tools a research session hands the Agent on top of it. Nothing
here composes a session: ``local_backend`` builds these tools and wires them
into the Agent runner, so this module never imports it back.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from autotrade.agent.compact import safe_error_summary
from autotrade.environment.artifacts import (
    READONLY_FILES,
    ArtifactSnapshotUnstable,
    FilesystemArtifactStore,
    copy_artifact_snapshot,
)
from autotrade.environment.data.summary import HOST_PATH_RE
from autotrade.environment.executor import raised_by_strategy, strategy_resources_of
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.null_control import PANEL_DRAWS, NullControlSetupError
from autotrade.environment.runtime import (
    AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS,
    RunManifest,
)
from autotrade.environment.sandbox import link_copytree
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
from autotrade.environment.tools.finish_session import SessionBudgetStatus
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.workspace import SafeWorkspace

from .agent_views import NULL_CONTROL_KEYS, allowed_keys
from .calendar import FULL_SPAN, yyyymmdd
from .config import (
    AcceptanceRules,
    ArtifactRevision,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    ReplaySpan,
    ResearchSessionRequest,
    StepResult,
    research_span,
)
from .experiment import freeze_gate_for, null_control_seed, research_step_record
from .ledger import RESEARCH_STAGE, ExperimentLedger
from .session_resume import record_step_sidecar
from .skills import _assert_skills_absent_from_formal


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
        "UNOFFICIAL smoke run of the CURRENT output/ over a few trading days of "
        "the research period, on the real replay path: real rolling "
        "as-of view (each context.asof_dir/<domain>/ is a DIRECTORY of parquet "
        "parts, read it with pd.read_parquet(directory)), real AccountSnapshot "
        "object, same sandbox executor and per-decision timeout as "
        "a Validation replay. It starts at the research period's first trading "
        "day unless start names a later one, which is how you measure a fit "
        "where a real batch will run it: a fit whose training window grows with "
        "the span is far slower late in the research period than on day one, so "
        "probe a data-dense late window BEFORE spending replay-years on a "
        "full-span batch. Returns seconds_per_day.strategy (the only figure "
        "that scales with the days replayed), window_build_seconds (the host "
        "side as-of view: on a late start it is almost all the one-off opening "
        "of that window, minutes of host wall clock paid once and reused by a "
        "later probe at the same start, so never multiply it by days; a "
        "full-span batch opens at the period start and does not pay it), "
        "phase_seconds (raw instrumentation whose phases overlap -- data_view "
        "wraps the as-of sub-phases, so they do not add up), order "
        "counts, the as-of domain directory names, the container resources block "
        "(peak memory against the container limit, per-fit seconds against the "
        "fit timeout in force, and GPU memory for a GPU strategy), and the exact "
        "exception text on failure. It commits no revision, creates no Step, "
        "cannot be frozen, and consumes no replay-years. Use it before "
        "batch_validate instead of hand-writing a shell smoke test against "
        "/mnt/snapshot.",
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
            },
            "required": [],
            "additionalProperties": False,
        },
        mutating=True,
        example={"days": 2, "start": "20241008"},
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
        start = self._start(arguments)
        with self.time_budget.pause():
            return self._invoke_exempt(days, start)

    def _invoke_exempt(self, days: int, start: str | None = None) -> ToolResult:
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
                start_day=start,
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
                    **({"start": start} if start is not None else {}),
                    # A rehearsal that died on a clock or on a shared card says
                    # what it was using when it did.
                    **(
                        {"resources": resources}
                        if (resources := strategy_resources_of(exc))
                        else {}
                    ),
                },
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        result_dir = Path(evaluation.result_ref).parent
        try:
            return ToolResult(
                True, value=self._report(evaluation, result_dir, days, start)
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

    def _start(self, arguments: Mapping[str, object]) -> str | None:
        """The probe date, refused unless it is inside the research period.

        The session may measure a fit anywhere in the period it researches, and
        nowhere else: a date past research end would replay sealed data, and a
        date before it has no slot.
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
        span = self.request.validation
        if not span.start <= day <= span.end:
            raise ToolError(
                f"smoke_backtest start {day} is outside the research period "
                f"{span.start}..{span.end}"
            )
        return day

    def _report(
        self,
        evaluation: EvaluationResult,
        result_dir: Path,
        days: int,
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
        # span. It is reported below as the whole-run figure it is.
        per_day = (
            {"strategy": round(phases["strategy"] / replayed, 3)}
            if replayed and "strategy" in phases
            else {}
        )
        report: dict[str, object] = {
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
            **(
                {"window_build_seconds": round(float(phases["data_view"]), 3)}
                if "data_view" in phases
                else {}
            ),
            "phase_seconds": phases,
            "nl_calls": summary.get("nl_calls"),
            "asof_domains": _smoke_asof_domains(result_dir),
            "hint": _SMOKE_LAYOUT_HINT,
        }
        if start is not None:
            report["start"] = start
        # What the rehearsal cost its container against the limits in force, so
        # the Agent sizes a batch on a measurement of the real replay container
        # instead of on the session container it can see from the inside.
        resources = summary.get("resources")
        if resources:
            report["resources"] = resources
        return report


_SMOKE_LAYOUT_HINT = (
    "context.asof_dir/<domain>/ is a directory of parquet parts: read it with "
    "pd.read_parquet(context.asof_dir + '/daily'), never "
    "context.asof_dir + '/daily.parquet'. context.account is an AccountSnapshot "
    "object (context.account.cash, context.account.positions), not a dict. Never "
    "fall back to context.snapshot_dir when an as-of read fails: the frozen "
    "snapshot stops at the decision time and using it is a PIT violation."
)


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
        # Continued from the earlier attempts' spend and Validations, which the
        # request reconciles: a resume never gets back the replay-years its
        # recorded Validations already cost.
        self.replay_years_used = request.replay_years_spent
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
        rules = (
            AcceptanceRules.from_record(self.request.acceptance_rules)
            if self.request.acceptance_rules
            else None
        )
        return freeze_gate_for(
            self.ledger.read(),
            rows,
            nominee,
            hard_reasons=(
                rules.evaluate(dict(nominee["summary"]))  # type: ignore[arg-type]
                if rules is not None
                else []
            ),
            acceptance=rules,
            years=[(year.start, year.end) for year in self.request.research_years],
        )

    def selection_statistics(self, step: StepResult) -> dict[str, object]:
        """The provisional freeze-gate reading one candidate row carries."""

        gate = self.freeze_gate(step.step_id)
        dsr = gate.get("deflated_sharpe")
        return {
            "freeze_gate_passed": gate["passed"],
            "freeze_gate_reasons": gate["reasons"],
            # The graded series and the two figures the gate reads off it
            # beside the deflated Sharpe: absent on a span the gate refuses
            # before measuring.
            "series": gate.get("series"),
            "information_ratio": gate.get("information_ratio"),
            "positive_years": gate.get("positive_years"),
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

    def refund(self, count: int, span: ReplaySpan) -> None:
        """Give back the replay-years of ``count`` candidates that measured nothing.

        A replay-year buys evidence about a strategy. A replay that ended in an
        environment failure -- a timeout, a shared card taken by another
        process, a sandbox that never started -- produced none, so the budget it
        claimed goes back. The result names it already spent stay spent: the
        attempt is still recorded, and a name is never reused.
        """
        self.replay_years_used = max(0, self.replay_years_used - count * span.slots)

    def release(self, count: int, span: ReplaySpan) -> None:
        """Refund candidates whose snapshot never held (no replay ran)."""
        self.refund(count, span)
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
# A pre-registration is only binding if the falsification clause fits with it:
# signal, holding, control, and what would refute the claim. It is bounded like
# every other Agent-written justification, by what one such statement needs
# rather than by what one line is.
BATCH_HYPOTHESIS_MAX_CHARS = AGENT_JUSTIFICATION_MAX_CHARS
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
# What a failed candidate's replay measured, which decides whether its
# replay-years bought anything. ``StrategyRaised`` (``environment.executor``,
# read off the whole cause chain by ``raised_by_strategy``) is the strategy's
# own exception: the replay measured the strategy -- it cannot run on this data
# -- and the charge stands. Every other failure measured the host: a strategy
# clock that ran out, a shared card another process took, a sandbox that never
# started or broke protocol. That produced no evidence about the candidate, so
# the batch gives its replay-years back instead of charging research budget for
# the environment's own trouble.
BATCH_FAILURE_STRATEGY = "strategy"
BATCH_FAILURE_ENVIRONMENT = "environment"
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
# The neutralized figures ride with the raw one because the freeze gate and the
# verdict read them, the active one above all -- it is the series they grade:
# a comparison made on the raw column alone reads a different number than they
# will.
BATCH_SUB_WINDOW_KEYS = (
    "label",
    "return",
    "excess_return",
    "neutralized_excess_return",
    "active_neutralized_excess_return",
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


def batch_candidate_resources(summary: Mapping[str, object]) -> dict[str, object]:
    """A row's container telemetry, which rides beside its metrics, not in them.

    What the replay cost its container against the limits it ran under: the
    peak memory and the per-fit seconds beside the ceilings this batch's own
    concurrency put them under. Two arms sized a batch from the session
    container's limits instead and lost 20 replay-years to timeouts. A failed
    candidate has no metrics but carries the same block from its exception, so
    every row keeps it in the one place — its own top level. Absent when the
    replay measured nothing.
    """

    resources = summary.get("resources")
    return {"resources": resources} if resources else {}


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
        "validate the working copy as it stands; build such a directory by copying "
        "output/, the code you have been editing, not refs/, which only holds the "
        "pack this arm started from. hypothesis is the falsifiable "
        "statement you register BEFORE any result exists. The batch costs one "
        "replay-year per candidate per year of the span, reserved before anything "
        "runs; a candidate whose replay completes becomes its own immutable "
        "revision and Step node under the CURRENT node as shared parent, recorded "
        "with its span. A candidate whose replay fails has no result_ref and, while "
        "record_failed_attempts is on, is recorded as a dead-end node, so later "
        "sessions see what was already tried; its row says which kind of failure it "
        "was: cause strategy means your own code raised, so the replay measured the "
        "strategy and keeps its replay-years, while cause environment means the host "
        "failed it (a strategy clock that ran out, a shared GPU taken by another "
        "process, a sandbox that never started or broke protocol), which measured "
        "nothing and gets its replay-years back, so resubmitting that candidate "
        "unchanged is a reasonable move. The returned "
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
        "carries its cause and its exact failure text instead — one failure never "
        "hides the others. Every row, failed ones included, also carries resources: "
        "what the replay cost the strategy container it ran in (peak memory against "
        "that container's own limit, the seconds of each fit against the fit timeout "
        "this batch's concurrency put in force, and for a GPU strategy its peak video "
        "memory and the free memory it was admitted with). "
        "Each completed row's result_ref reads back that candidate's full "
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
        # The one place the batch settles its reservation. Every candidate was
        # charged before anything ran; those whose replay measured the host and
        # not the strategy bought no evidence, so their claim goes back here,
        # before this call's result and its budget block reach the trace.
        refunded = sum(
            1 for row in rows if row.get("cause") == BATCH_FAILURE_ENVIRONMENT
        )
        if refunded:
            self.backtest.refund(refunded, span)
        # Every row of the round deflates against the same trial pool: the
        # whole batch is complete by the time the table is returned.
        for row, step in recorded:
            row["selection_statistics"] = self.backtest.selection_statistics(step)
        if not recorded:
            charged = len(rows) - refunded
            raise ToolError(
                f"batch_validate: all {len(rows)} candidates failed their "
                f"Validation; {charged} of them raised in their own code and "
                f"kept the replay-years, {refunded} failed on the environment "
                "and got them back",
                error_type="validation_failed",
                details={
                    "batch_id": batch_id,
                    "candidates": rows,
                    "replay_years_used": self.backtest.replay_years_used,
                    "replay_years_remaining": self.backtest.replay_years_remaining,
                },
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
        refused by ``modification_check``, which restores only the working copy
        the host itself seeded. Only the read-only template names are touched
        — never a sibling module of the package.
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
            **batch_candidate_resources(evaluation.summary),
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
        if error is None:
            error = RuntimeError("unknown replay failure")
        public_error = _public_validation_error(error)
        cause = (
            BATCH_FAILURE_STRATEGY
            if raised_by_strategy(error)
            else BATCH_FAILURE_ENVIRONMENT
        )
        request = self.backtest.request
        metadata = {
            "batch_id": batch_id,
            "candidate": candidate.name,
            "hypothesis": candidate.hypothesis,
            "source_path": candidate.path,
            "span": span.label,
            # A later session reading this dead end needs to know whether the
            # hypothesis was falsified or the host simply got in the way.
            "cause": cause,
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
                **{
                    key: metadata[key]
                    for key in ("batch_id", "candidate", "hypothesis", "span", "cause")
                },
                "error": public_error,
            }
        )
        row: dict[str, object] = {
            "status": "failed",
            "cause": cause,
            "error": public_error,
        }
        # The failed row's telemetry comes off the exception — there is no
        # result to carry it — and lands where a successful row keeps its own:
        # a fit that ran out of clock or a card that was taken says so here,
        # measured in the container it actually ran in.
        resources = strategy_resources_of(error)
        if resources:
            row["resources"] = resources
        return row


def batch_select_hint(
    rows: Sequence[Mapping[str, object]], *, replay_years_remaining: int
) -> str:
    """Name the row leading on the graded figure; select nothing.

    The active information ratio -- the row's return minus its zero-skill panel,
    neutralized, over its residual risk -- is what the freeze gate grades, so
    the hint says which row leads on it and on nothing else. A winning round
    is the starting point of the next pre-registered round, not the end of the
    session; once no batch fits the replay-year budget, the hint says the
    session is left with finishing.
    """

    ranked = [
        (_batch_row_active_ir(row), row)
        for row in rows
        if row.get("status") == "ok"
    ]
    leading = max(ranked, key=lambda item: item[0], default=(float("-inf"), None))
    lead = (
        f"leading on active information ratio: {leading[1].get('name')} "
        f"(node_id={leading[1].get('node_id')}); "
        if leading[1] is not None and math.isfinite(leading[0])
        else "no row carries an active information ratio; "
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


def _batch_row_active_ir(row: Mapping[str, object]) -> float:
    stats = row.get("stats")
    benchmark = stats.get("benchmark") if isinstance(stats, Mapping) else None
    value = (
        benchmark.get("active_information_ratio")
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
    "information the timing and sizing did not. The percentile gates nothing; "
    "what the gate grades is the node's active series against the panel its own "
    "validation drew (benchmark.active_*)"
)


class NullControlTool(SessionTimeBudgetAware):
    """Rank one complete Validation against random-name replays of its trades.

    The same null control the Pipeline runs for the frozen node at freeze
    (``experiment._null_control``), drawn with the same seed through the same
    backend over the node's own span, so the figure the Agent reads before
    selecting is the figure the ledger records: the block is cached per node
    and handed to the Pipeline, which reuses it for the frozen node instead of
    drawing again. Each call is host replay the session does not pay for, hence
    the per-session cap.
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
            f"Rank one complete Validation node of this session against K={PANEL_DRAWS} "
            "host replays of its own trade skeleton over the node's own span, every "
            "name replaced by a random one that one board lot of the same money could "
            "buy on the same side of the arm's benchmark_index membership (the "
            "float-cap decile when "
            "index_weight is not mounted): the same null control the Pipeline runs "
            "for the frozen node at freeze. Returns the null_control block the ledger "
            "will carry (observed_excess, excess_percentile — near 0.5 means the names "
            "added nothing the timing and sizing did not — the null's mean and p05/p95, "
            "rejects_mean, dropped_trips_mean). Every formal validation already "
            "reports the node against a panel of the same construction "
            "(benchmark.active_*), which is what the freeze gate grades; this "
            "percentile is a second, descriptive reading of it. Costs well under a "
            "minute of host replay per research year (the session clock pauses like a "
            f"formal backtest), consumes no replay-years and is capped at {max_calls} per "
            "session (max_null_controls in the budgets fact; every result reports "
            "null_controls_remaining); a node's block is cached, and the frozen node's "
            "block is reused at freeze instead of being drawn again. Use it on "
            "full-span finalists before finish_session, not on every candidate. "
            "Refused for a node that is not a complete Validation of this session, "
            "once the cap is spent, and while a background sub-agent that can write "
            "is still running; it is also unavailable once the session enters hard "
            "finalization, so rank the finalists before that. A call the host "
            "cannot even set up — it replays nothing and reports no block — costs "
            "no budget, so the counters it returns are the ones still available.",
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
                if isinstance(exc, NullControlSetupError):
                    # It never reached its first draw, so none of the host
                    # compute the charge stands for was spent: give the call
                    # back, as batch_validate refunds an environment failure.
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
