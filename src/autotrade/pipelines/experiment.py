"""Experiment pipeline: one research session, one freeze, one forward replay.

docs/pipeline-design.md. The Pipeline schedules Data, Environment and Agent in
time order, freezes inputs and outputs at each boundary and writes the single
experiment ledger. It implements no investment logic and never rewrites
strategy content; it only prepares, freezes, replays and records.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from autotrade.common.records import read_json_object
from autotrade.environment.artifacts import (
    ArtifactRevision,
    FrozenArtifact,
    copy_artifact,
    copy_model_artifacts,
    model_artifact_delta,
    modification_delta,
    restore_frozen_artifact_trees,
)
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    StrategyExecutor,
    TrustedStrategyExecutor,
    attach_strategy_resources,
    raised_by_strategy,
    strategy_resource_usage,
)
from autotrade.environment.replay import (
    ContextDataProvider,
    ExecutionPriceProvider,
    ReplayResult,
    run_daily_replay,
)
from autotrade.environment.replay.engine import BacktestError
from autotrade.environment.replay.stats import window_activity
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.environment.runtime import chmod_tree, utc_now_iso
from autotrade.environment.step_tree import node_handle
from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.environment.trace import agent_trace_path

from .agent_inbox import expire_experiment_session_inbox
from .calendar import FULL_SPAN, VALIDATION_SPAN, Slot, anchor_on
from .config import (
    AcceptanceRules,
    ArtifactStore,
    BudgetUsed,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    ReplaySpan,
    ResearchDeveloper,
    ResearchSessionRequest,
    ResearchSessionResult,
    RollingExperimentConfig,
    SnapshotBundle,
    SnapshotProvider,
    StepResult,
    StrategyExperimentConfig,
    research_span,
    session_deadline_seconds,
)
from .ledger import (
    ENVIRONMENT_BLOCKED,
    STRATEGY_ERROR,
    ExperimentLedger,
    FrozenArtifactMutated,
    FrozenArtifactRestoreFailed,
    RunMarkers,
    SessionKind,
    Stage,
    VerdictStatus,
    assert_no_frozen_artifact_mutation,
    forward_record,
    frozen_record,
    incubation_record,
    is_frozen_artifact_mutation,
    recorded_path,
    research_over,
    research_records,
)
from .pit_views_seed import FORWARD_PHASE, RESEARCH_PHASE
from .rolling_record import (
    SOURCE_HOST_COMPLETION,
    ValidationSteps,
    frontier_after,
    step_entry,
    test_step_record,
    test_step_records,
)
from .session_resume import (
    REVISIONS_DIR,
    load_recorded_steps,
    resume_state,
    revision_fingerprint,
)
from .skills import (
    ExperimentSkillsStore,
    SkillsPublication,
    SkillsSnapshot,
    latest_skills_snapshot,
    resolve_collected_skills_source,
)
from .verdict import (
    ConditionReason,
    ConditionStage,
    forward_mde,
    forward_slice,
    graded_statistics,
    graduation_verdict,
    heldout_slice,
    judge,
    registered_mandate,
    residual_reading,
    sealed_slice,
    sealed_strategy_error,
)

# A session deadline override may raise the research deadline above the
# configured maximum, bounded by this absolute ceiling in minutes: twice the
# default research budget (config.max_research_minutes), enough headroom for
# one slow session without letting it run unattended for weeks.
_MAX_DEADLINE_OVERRIDE_MINUTES = 4800


class EnvironmentBlockedAttempt(RuntimeError):
    """The research session's attempt ended on an environment block
    (``session_tools.EnvironmentBlocked``) while the arm still has attempts:
    recorded as a failed attempt, which the runner retries from the frontier
    and the record."""


class DailyStrategyPipeline:
    def __init__(
        self,
        config: StrategyExperimentConfig,
        *,
        context_data: ContextDataProvider | None = None,
        execution_price: ExecutionPriceProvider | None = None,
        executor_factory: Callable[[StrategyExperimentConfig], StrategyExecutor]
        | None = None,
    ) -> None:
        self.config = config
        self.context_data = context_data
        self.execution_price = execution_price
        self.executor_factory = executor_factory

    def run(
        self,
        daily: pd.DataFrame | str | Path,
        corporate_actions: pd.DataFrame | None = None,
    ) -> ReplayResult:
        """``corporate_actions`` is the replay slot's ex-date table; the Broker
        credits its cash dividends on the ex-date (the share leg comes from the
        daily frame's ``pre_close``). The PIT backend always passes it; None is
        only for the local ``daily`` development backend and unit tests."""

        frame = pd.read_parquet(daily) if isinstance(daily, (str, Path)) else daily
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("daily must be a pandas DataFrame or parquet path")
        # Every replay fits from empty: the state directory is created here and
        # discarded with the run, so nothing an earlier run wrote reaches it.
        # World-writable for the sandbox's non-root fit worker.
        state = TemporaryDirectory(prefix="strategy_state_")
        state_dir = Path(state.name)
        state_dir.chmod(0o777)
        try:
            executor = self._create_executor(state_dir)
            # The executor's own telemetry has to be read while its container
            # still exists, so it is closed before the usage is taken — on the
            # failing path too, where the peak and the clock that ran out are
            # exactly what the failure needs to report.
            try:
                try:
                    replay = run_daily_replay(
                        daily=frame,
                        strategy=executor,
                        schedule=self.config.schedule,
                        profile=self.config.broker_profile,
                        context_data=self.context_data,
                        execution_price=self.execution_price,
                        corporate_actions=corporate_actions,
                    )
                finally:
                    executor.close()
            except BaseException as exc:
                attach_strategy_resources(exc, executor)
                raise
            return replace(replay, resources=strategy_resource_usage(executor))
        finally:
            # The trusted executor leaves the tree read-only between fits.
            chmod_tree(state_dir, file_mode=0o644, dir_mode=0o755)
            state.cleanup()

    def _create_executor(self, state_dir: Path) -> StrategyExecutor:
        if self.executor_factory is not None:
            executor = self.executor_factory(self.config)
            if not isinstance(executor, StrategyExecutor):
                raise TypeError("executor_factory must return a StrategyExecutor")
            return executor
        if self.config.execution_mode == "trusted":
            return TrustedStrategyExecutor.from_path(
                self.config.strategy_path,
                state_dir=state_dir,
                models_dir=self.config.models_dir,
            )
        return DockerStrategyExecutor(
            self.config.strategy_path,
            self.config.sandbox,
            models_dir=self.config.models_dir,
            state_dir=state_dir,
        )


class RollingExperimentPipeline:
    """One research session → freeze → one continuous forward replay → verdict.

    Every stage reads what came before it from the ledger, so a resumed worker
    continues wherever the durable records end.
    """

    def __init__(
        self,
        config: RollingExperimentConfig,
        *,
        snapshots: SnapshotProvider,
        artifacts: ArtifactStore,
        evaluator: EvaluationBackend,
        developer: ResearchDeveloper,
        trading_days: Sequence[str],
        ledger: ExperimentLedger | None = None,
    ) -> None:
        self.config = config
        self.snapshots = snapshots
        self.artifacts = artifacts
        self.evaluator = evaluator
        self.developer = developer
        # The pinned release's daily dates: the Held-out slot ends at its last.
        self.trading_days = sorted(str(day) for day in trading_days)
        self.ledger = ledger or ExperimentLedger(config.ledger_path)
        self.run_markers = RunMarkers(config.experiment_dir)

    # ---- research ------------------------------------------------------

    def research_spans(self, slots: Sequence[Slot]) -> tuple[ReplaySpan, ...]:
        """One single-slot span per research year in ``slots``, over its slot
        and the decision view at its anchor. Prepared from the research
        geometry alone: every slot ends by research end and every anchor is at
        or before its decision time."""

        geometry = self.config.geometry
        spans = []
        for slot in slots:
            if (
                slot.end > geometry.research_end
                or slot.anchor > geometry.research_decision_time
            ):
                raise RuntimeError(
                    f"research slot {slot.label} {slot.start}..{slot.end} reaches past "
                    f"research end {geometry.research_end}"
                )
            bundle = self.snapshots.prepare(
                phase=RESEARCH_PHASE,
                start=slot.start,
                end=slot.end,
                decision_time=slot.anchor,
            )
            spans.append(
                ReplaySpan(
                    label=slot.label,
                    mode=RESEARCH_PHASE,
                    start=slot.start,
                    end=slot.end,
                    snapshot=bundle,
                )
            )
        return tuple(spans)

    def research_inputs(
        self, records: Sequence[Mapping[str, object]] = ()
    ) -> tuple[SnapshotBundle, tuple[ReplaySpan, ...]]:
        """The decision view at the arm's frontier and every year of the
        geometry as :meth:`research_spans`: what a research session gets.

        The first ``research_years`` of the arm's rules are the research years
        (``ResearchGeometry.split_research``); the frontier is the validation
        start before any test and the end of the last step the ledger
        ``records`` hold afterwards (``rolling_record.frontier_after``). The
        spans a Validation may replay are bounded by the frontier in the
        session (``session_tools.SessionValidations.span``), so no row past it
        reaches a research session; the tests replay the validation years
        (:meth:`validation_span`).
        """

        years = self.research_spans(self.config.geometry.research_years)
        frontier = frontier_after(records, self.validation_steps())
        return self.decision_view_at(anchor_on(frontier)), years

    def validation_steps(self) -> ValidationSteps:
        """The validation years tiled into the arm's test steps."""

        _research, validation = self.config.geometry.split_research(
            self.config.acceptance.research_years
        )
        return ValidationSteps.of(validation, self.config.acceptance.test_step_months)

    def decision_view_at(self, decision_time) -> SnapshotBundle:
        """The decision view a session mounts at a frontier: one the provider
        already holds where it can tell (``require_decision_view``, which
        refuses a view the seed lacks by name), else prepared."""

        require = getattr(self.snapshots, "require_decision_view", None)
        if callable(require):
            return require(decision_time=decision_time)
        return self.snapshots.prepare_decision(decision_time=decision_time)

    def validation_span(self) -> ReplaySpan:
        """The validation years as one span, replayed as one book from the
        decision view at their first year and labelled ``validation``: what a
        test replays up to its frontier, and what the host completes a record
        over."""

        _research, validation = self.config.geometry.split_research(
            self.config.acceptance.research_years
        )
        return replace(research_span(self.research_spans(validation), FULL_SPAN), label=VALIDATION_SPAN)

    def run_research_session(
        self, *, session_context: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Run the arm's one research session and record its outcome.

        The session sees the data up to its frontier (:meth:`research_inputs`)
        and ends when its tests reach the end of the validation years
        (``completed``): the last test's candidate is frozen, checked for
        registration alone (:func:`freeze_registration`), never for a score.
        A session whose budget ran out (``deadline``) is completed by the
        host: the last tested candidate is frozen, or before any test the last
        registrable full-span candidate; with none the arm ends without a
        deliverable. An environment block ends the attempt, not the research,
        until the arm's attempts are spent, when it ends ``environment_blocked``
        with nothing frozen. The record always carries ``frozen`` or
        ``arm_end`` and research is over once it is written. An attempt that
        failed before that record is continued, not repeated: the next attempt
        starts from the budget, the compaction summary, the Validations and
        the test steps the interrupted ones left in the trace, the step tree
        and the ledger.
        """

        records = self.ledger.read()
        assert_no_frozen_artifact_mutation(records)
        if research_over(records):
            raise RuntimeError("research is over; no further research session runs")
        if research_records(records):
            raise RuntimeError("the arm's research session is already recorded")
        resume = resume_state(self.config.experiment_dir, records)
        steps_before = load_recorded_steps(self.config.experiment_dir)
        run_started = time.monotonic()
        run_id = f"run_{uuid.uuid4().hex}"
        context = dict(session_context or {})
        progress = _optional_hook(context.get("progress_hook"), "progress_hook")
        budgets = _session_budgets(self.config, context.get("resource_override"))
        current_skills = self._current_skills()
        frozen_id: str | None = None
        wrote_ledger_record = False
        attempt = {
            "experiment_id": self.config.experiment_id,
            "epoch_id": Stage.RESEARCH,
            "fold_id": SessionKind.RESEARCH,
            "run_id": run_id,
            "session_key": SessionKind.RESEARCH,
            "phase": Stage.RESEARCH,
        }
        # Evidence for a run that never gets to run its own except branch.
        self.run_markers.begin(attempt)
        try:
            _publish_progress(progress, "pit_snapshot", run_id=run_id, phase=Stage.RESEARCH)
            decision, years = self.research_inputs(records)
            tests_before = tuple(test_step_records(records))
            frontier = frontier_after(records, self.validation_steps())
            session = self.developer(
                ResearchSessionRequest.for_arm(
                    self.config,
                    budgets=budgets,
                    run_id=run_id,
                    snapshot=decision,
                    decision_time=anchor_on(frontier),
                    years=years,
                    validation=self.validation_span(),
                    decision_view_at=self.decision_view_at,
                    tests_before=tests_before,
                    directive=str(context.get("directive") or ""),
                    sandbox_gpu_count=_optional_gpu_count(context.get("sandbox_gpu_count")),
                    snapshot_config=_snapshot_config_record(self.snapshots),
                    progress_hook=progress,
                    skills_source_ref=(
                        str(current_skills.root) if current_skills.root is not None else ""
                    ),
                    budget_used=resume.budget_used if resume is not None else BudgetUsed(),
                    steps_before=steps_before,
                    resume=resume,
                )
            )
            # Every row the ledger records names the bytes it replayed.
            step_rows = completed_rows(
                self.config.experiment_dir,
                [research_step_record(step) for step in session.steps],
            )
            gate: dict[str, object] | None = None
            frozen: dict[str, object] | None = None
            arm_end: dict[str, object] | None = None
            tests = test_step_records(self.ledger.read())
            if session.outcome == ENVIRONMENT_BLOCKED:
                if session.attempt < self.config.session_max_attempts:
                    # The attempt ends, the research does not: the next one
                    # resumes from the frontier and the record.
                    raise EnvironmentBlockedAttempt(session.reason)
                arm_end = {"status": ENVIRONMENT_BLOCKED, "reason": session.reason}
                # The worker log is the operator's other place to see it.
                print(
                    f"{utc_now_iso()} {attempt['experiment_id']} {ENVIRONMENT_BLOCKED}: "
                    f"{session.reason}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                by_step = {str(row["step_id"]): row for row in step_rows}
                steps = {step.step_id: step for step in session.steps}
                candidate = self._frozen_candidate(session, tests, by_step)
                if candidate is None:
                    arm_end = {
                        "status": VerdictStatus.NO_DELIVERABLE,
                        "reason": (
                            "research budget exhausted before any test, with no registrable "
                            f"full-span candidate ({session.finish_reason})"
                        ),
                    }
                else:
                    node_id, replicates, deterministic = candidate
                    gate = freeze_registration(
                        by_step[node_id],
                        experiment_dir=self.config.experiment_dir,
                        acceptance=self.config.acceptance,
                        seed_replicates=[by_step[step_id] for step_id in replicates],
                        deterministic=deterministic,
                    )
                    if not gate["passed"]:
                        # Every test checked this registration already.
                        raise RuntimeError(
                            f"the frozen candidate {node_id} no longer registers: "
                            + ", ".join(str(reason) for reason in gate["reasons"])
                        )
                    _publish_progress(progress, "freezing", run_id=run_id)
                    frozen = self._freeze(
                        steps[node_id],
                        gate=gate,
                        run_id=run_id,
                        null_controls=session.null_controls,
                        seed_replicates=[steps[step_id] for step_id in replicates],
                    )
                    frozen_id = str(frozen["artifact_id"])
                    if session.outcome == "deadline":
                        _publish_progress(progress, "completing_record", run_id=run_id)
                        self.complete_record(
                            frozen, gate, tests, fingerprint=str(by_step[node_id]["fingerprint"]), run_id=run_id
                        )
            _publish_progress(progress, "publishing", run_id=run_id)
            skills = self._publish_or_keep_skills(
                session.skills_source_ref,
                current=current_skills,
                generation_id=f"{SessionKind.RESEARCH}_{run_id}",
                run_id=run_id,
            )
            trace = agent_trace_path(self.config.experiment_dir / "artifacts", run_id)
            record = {
                "record_type": "research_session",
                **{key: attempt[key] for key in ("experiment_id", "epoch_id", "fold_id", "run_id")},
                "session_key": SessionKind.RESEARCH,
                "conversation_id": session.conversation_id,
                "finish_reason": session.finish_reason,
                "outcome": session.outcome,
                "reason": session.reason or None,
                "nominated_step_id": str(frozen["source_step_id"]) if frozen else None,
                "steps": step_rows,
                **self._account_record(),
                "freeze_gate": gate,
                "frozen": frozen,
                "arm_end": arm_end,
                "skills_ref": skills.skills_ref or None,
                "skills_generation_id": skills.generation_id or None,
                **skills.stats.ledger_fields(),
                "skills_published": skills.published,
                "run_manifest_ref": session.run_manifest_ref or None,
                "agent_trace_ref": str(trace) if trace.exists() else None,
                "snapshot_ids": {
                    "decision": decision.snapshot_id,
                    "research_span": years[0].snapshot.snapshot_id,
                },
                "frontier": frontier_after(self.ledger.read(), self.validation_steps()),
                "steps_tested": len(test_step_records(self.ledger.read())),
                "attempts": session.attempt,
                "budget_used": session.budget_used.to_record(),
                **_session_timing(context, run_started),
            }
            self.ledger.append(record)
            wrote_ledger_record = True
            expire_experiment_session_inbox(
                self.config.experiment_dir, SessionKind.RESEARCH, expired_by=run_id
            )
            # Every revision a Validation recorded stays: it is the arm's
            # artifact history, and the store shares the bytes between
            # revisions. Only frozen artifacts no record still references go.
            prune = getattr(self.artifacts, "prune_superseded_frozen", None)
            if callable(prune):
                prune(
                    keep_frozen_ids=_keep_frozen_artifact_ids(
                        self.ledger.read(), extra_id=frozen_id
                    )
                )
            return record
        except BaseException as exc:
            # BaseException, not Exception: a terminated worker unwinds this
            # session through SystemExit (the entrypoint's SIGTERM handler) or
            # KeyboardInterrupt, and those must leave the same evidence as any
            # other failure.
            if not wrote_ledger_record:
                self.ledger.append(
                    {
                        **attempt,
                        "record_type": "attempt_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                wrote_ledger_record = True
            raise
        finally:
            # The marker is this run's only evidence until one of its ledger
            # records is durable, so it is dropped only once one is; otherwise
            # it stays for the next worker start to record.
            if wrote_ledger_record:
                self.run_markers.finish(run_id)

    def _frozen_candidate(
        self,
        session: ResearchSessionResult,
        tests: Sequence[Mapping[str, object]],
        by_step: Mapping[str, Mapping[str, object]],
    ) -> tuple[str, tuple[str, ...], bool] | None:
        """The Step the arm freezes, with its seed replicates and declaration.

        A completed session froze its last test's candidate; a session whose
        budget ran out after a test freezes the last tested candidate as that
        test registered it; one that never tested freezes the last full-span,
        non-control Step that registers on its own (a strategy that trains a
        model owes a replicate no host can name, so it does not); none means
        no deliverable.
        """

        if session.outcome == "completed":
            if session.node_id is None:
                raise RuntimeError("a completed session names no frozen candidate")
            return session.node_id, tuple(session.seed_replicates), session.deterministic
        if tests:
            candidate = dict(tests[-1]["candidate"])  # type: ignore[arg-type]
            return (
                str(candidate["step_id"]),
                tuple(str(item) for item in candidate.get("seed_replicates") or ()),
                bool(candidate.get("deterministic")),
            )
        for step in reversed(session.steps):
            row = by_step.get(step.step_id)
            if row is None or step.control or step.span != FULL_SPAN:
                continue
            gate = freeze_registration(
                row, experiment_dir=self.config.experiment_dir, acceptance=self.config.acceptance
            )
            if gate["passed"]:
                return step.step_id, (), False
        return None

    def complete_record(
        self,
        frozen: Mapping[str, object],
        gate: Mapping[str, object],
        tests: Sequence[Mapping[str, object]],
        *,
        fingerprint: str,
        run_id: str,
    ) -> list[dict[str, object]]:
        """Fill the rolling record's remaining steps from one host replay of
        the frozen strategy over the whole validation period.

        The replay is the one a test would have run at the last step: the
        validation span as one fresh book with its matched panel, opened on
        the seed's warm start. Each step after the ``tests`` already recorded
        gets a ``test_step`` record of the same shape, read off that replay's
        window for the step and marked ``host_completion``; the records are
        appended in step order and returned. A frozen strategy whose own code
        raises here fails the attempt: the record it would fill cannot be
        read off nothing.
        """

        steps = self.validation_steps()
        done = len(tests)
        if done >= steps.count:
            return []
        artifact = self._frozen_artifact(frozen)
        span = self.validation_span()
        request = span.request(
            _frozen_revision(artifact),
            schedule=self.config.schedule,
            broker_profile=self.config.broker_profile,
        )
        require_warm = getattr(self.evaluator, "require_warm_start", None)
        if callable(require_warm):
            require_warm(request)
        result = self.evaluator.evaluate(request)
        replay, analysis = replay_and_analysis(result.result_ref)
        head = {
            "experiment_id": self.config.experiment_id,
            "epoch_id": Stage.RESEARCH,
            "fold_id": SessionKind.RESEARCH,
            "run_id": run_id,
            "session_key": SessionKind.RESEARCH,
        }
        candidate = {
            "step_id": str(frozen["source_step_id"]),
            "revision_id": str(frozen["revision_id"]),
            "fingerprint": fingerprint,
            "seed_replicates": [
                str(item["source_step_id"]) for item in frozen.get("seed_replicates") or ()  # type: ignore[union-attr]
            ],
            "deterministic": bool(frozen.get("deterministic")),
        }
        appended = []
        for step in range(done + 1, steps.count + 1):
            first, last = steps.window(step)
            record = test_step_record(
                head=head,
                steps=steps,
                step=step,
                candidate=candidate,
                replay={"start": span.start, "end": span.end, "result_ref": result.result_ref},
                entry=step_entry(replay, analysis, start=first, end=last),
                registration=gate,
                source=SOURCE_HOST_COMPLETION,
            )
            self.ledger.append(record)
            appended.append(record)
        return appended

    def _freeze(
        self,
        nominee: StepResult,
        *,
        gate: Mapping[str, object],
        run_id: str,
        null_controls: Mapping[str, Mapping[str, object]],
        seed_replicates: Sequence[StepResult] = (),
    ) -> dict[str, object]:
        """Freeze the nominee's immutable revision and state what it was frozen on.

        The seed replicates the registration accepted are frozen beside it,
        each as an artifact of its own, so the forward stage replays bytes no
        later change can reach; ``seed_replicates`` in the block names them,
        absent when there are none. The figures are the nominee's own over
        the research years, graded as it is and through the residual
        statistic.
        """

        def freeze(step: StepResult) -> tuple[str, Path, Path | None]:
            return self._freeze_revision(step.step_id, step.revision_id, run_id)

        def frozen_beside(step: StepResult) -> dict[str, object]:
            item_id, item_output, item_models = freeze(step)
            return {
                "artifact_id": item_id,
                "output_path": str(item_output),
                "models_path": (
                    str(item_models) if item_models is not None and item_models.is_dir() else None
                ),
                "source_step_id": step.step_id,
                "revision_id": step.revision_id,
                "research_result_ref": step.validation.result_ref,
            }

        artifact_id, output, models = freeze(nominee)
        analysis = _result_analysis(nominee.validation.result_ref)
        statistics = graded_statistics(analysis)
        replicates = [frozen_beside(step) for step in seed_replicates]
        forward = self.config.geometry.forward
        forward_days = sum(1 for day in self.trading_days if forward.start <= day <= forward.end)
        fit = validate_strategy_package(output / "main.py")
        research, _validation = self.config.geometry.split_research(
            self.config.acceptance.research_years
        )
        null_control = (
            dict(null_controls[nominee.step_id])
            if nominee.step_id in null_controls
            else self._null_control(
                nominee.validation.result_ref,
                start=research[0].start,
                end=research[-1].end,
                profile=self.config.broker_profile,
                seed=null_control_seed(SessionKind.RESEARCH, "frozen"),
            )
        )
        return {
            "artifact_id": artifact_id,
            "output_path": str(output),
            "models_path": str(models) if models is not None and models.is_dir() else None,
            "source_step_id": nominee.step_id,
            "revision_id": nominee.revision_id,
            "research_result_ref": nominee.validation.result_ref,
            # Which series the figures below are of: ``active`` is the
            # nominee's return minus its zero-skill panel composite.
            "series": statistics["series"],
            "mandate": gate["mandate"],
            "days": statistics["days"],
            "plain_excess": statistics["plain_excess"],
            "tracking_error": statistics["tracking_error"],
            "information_ratio": statistics["information_ratio"],
            # The residual statistic over the research years: what the
            # session read before its first test.
            "residual_reading": residual_reading(analysis, alpha=self.config.acceptance.alpha),
            "blocks": nominee.validation.summary.get("sub_windows"),
            "null_control": null_control,
            "forward_mde": forward_mde(float(statistics["tracking_error"]), forward_days),  # type: ignore[arg-type]
            "fit_plan": {
                "fit": fit is not None,
                "refit_period": fit.refit_period if fit is not None else None,
            },
            # The Agent's declaration that the package uses no randomness,
            # which waived its seed replicates, only when made: the operator
            # reviews it before the strategy goes live.
            **(
                {"deterministic": True}
                if gate["seed_replicates"].get("deterministic")  # type: ignore[union-attr]
                else {}
            ),
            **({"seed_replicates": replicates} if replicates else {}),
        }

    def _freeze_revision(
        self, step_id: str, revision_id: str, run_id: str
    ) -> tuple[str, Path, Path | None]:
        """Freeze one Step's revision as a new artifact: its id, its output
        tree and its models tree (``None`` when the store keeps none)."""

        artifact_id = f"strategy_{SessionKind.RESEARCH}_{uuid.uuid4().hex[:12]}"
        stored = self.artifacts.freeze_revision(
            revision_id,
            artifact_id=artifact_id,
            experiment_id=self.config.experiment_id,
            epoch_id=Stage.RESEARCH,
            fold_id=SessionKind.RESEARCH,
            run_id=run_id,
            step_id=step_id,
        )
        return artifact_id, stored.path, stored.model_path

    # ---- forward -------------------------------------------------------

    def run_forward(
        self, *, session_context: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Read the frozen artifact on the sealed years, then replay it once
        over forward and Held-out, and judge it (:meth:`replay_frozen`).

        The stage runs after the research session has ended for good: a
        failure here, the sealed reading's included, fails a forward-stage
        attempt, which the caller retries from the sealed replay, and never
        reopens the session. The strategy's own exception is a measurement;
        any other failure measured nothing. Frozen trees that changed during
        a replay are recorded and fail closed.
        """

        records = self.ledger.read()
        assert_no_frozen_artifact_mutation(records)
        frozen_row = frozen_record(records)
        if frozen_row is None:
            raise RuntimeError("the forward replay needs a frozen artifact")
        if forward_record(records) is not None:
            raise RuntimeError("the arm's forward verdict is already recorded")
        block: Mapping[str, object] = frozen_row["frozen"]  # type: ignore[assignment]
        artifact = self._frozen_artifact(block)
        replicates = [
            self._frozen_artifact(item)
            for item in block.get("seed_replicates") or ()  # type: ignore[union-attr]
        ]
        forward = self.config.geometry.forward
        heldout = self.config.geometry.heldout(self.trading_days)
        context = dict(session_context or {})
        progress = _optional_hook(context.get("progress_hook"), "progress_hook")
        run_id = f"run_{uuid.uuid4().hex}"
        attempt = {
            "experiment_id": self.config.experiment_id,
            "epoch_id": Stage.FORWARD,
            "fold_id": SessionKind.FORWARD,
            "run_id": run_id,
            "session_key": SessionKind.FORWARD,
            "phase": Stage.FORWARD,
        }
        head = {
            "record_type": "forward",
            **{key: attempt[key] for key in ("experiment_id", "epoch_id", "fold_id", "run_id")},
            "session_key": SessionKind.FORWARD,
        }
        self.run_markers.begin(attempt)
        wrote_ledger_record = False

        def integrity_failure(row: Mapping[str, object]) -> None:
            nonlocal wrote_ledger_record
            self.ledger.append({**head, **row})
            # The integrity row is this run's record: the fail-fast that
            # follows must not also log a failed attempt.
            wrote_ledger_record = True

        try:
            _publish_progress(progress, "pit_snapshot", run_id=run_id, phase=Stage.FORWARD)
            record = {
                **head,
                **self.replay_frozen(
                    artifact,
                    replicates,
                    progress,
                    run_id=run_id,
                    sealed=self.validation_span(),
                    forward=forward,
                    heldout=heldout,
                    broker_profile=self.config.broker_profile,
                    integrity_failure=integrity_failure,
                ),
            }
            self.ledger.append(record)
            wrote_ledger_record = True
            return record
        except BaseException as exc:
            if not wrote_ledger_record:
                self.ledger.append(
                    {
                        **attempt,
                        "record_type": "attempt_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                wrote_ledger_record = True
            raise
        finally:
            if wrote_ledger_record:
                self.run_markers.finish(run_id)

    def replay_frozen(
        self,
        artifact: FrozenArtifact,
        replicates: Sequence[FrozenArtifact],
        progress,
        *,
        run_id: str,
        sealed: ReplaySpan,
        forward: Slot,
        heldout: Slot,
        broker_profile: BrokerProfile,
        integrity_failure: Callable[[Mapping[str, object]], None] | None = None,
    ) -> dict[str, object]:
        """The forward stage of a frozen artifact and its seed replicates
        through ``broker_profile`` (its Broker and the zero-skill panel drawn
        with it), as :meth:`run_forward` records it. Writes nothing to the
        ledger.

        First the sealed reading: the artifact replayed over ``sealed``, the
        research span of the sealed years, as one book from the view at its
        first year, read by ``verdict.sealed_slice``, each seed replicate
        replayed the same way and reported beside it. Then, whether or not
        the sealed gate passed, one continuous replay over the ``forward`` and
        ``heldout`` slots and each replicate's, read and reported beside the
        verdict, which is the sealed gate's (:meth:`_judge`). The strategy's
        own exception in either replay is a reading
        (``verdict.sealed_strategy_error``, or the forward record's
        ``strategy_error`` status naming the slice it raised in, which no
        condition reads); a failure that measured nothing raises,
        and so does any replicate's failure. Frozen trees that changed during
        a replay fail closed after the integrity row (the body of what
        :meth:`run_forward` appends) is handed to ``integrity_failure``, when
        given.
        """

        forward_bundle = self._prepare_slot(forward)
        heldout_bundle = self._prepare_slot(heldout)
        span = ReplaySpan(
            label=Stage.FORWARD,
            mode=FORWARD_PHASE,
            start=forward.start,
            end=heldout.end,
            snapshot=forward_bundle,
            continuation=(heldout_bundle.replay_ref,),
        )
        base = {
            "artifact_id": artifact.artifact_id,
            **self._account_record(),
            "replay": {
                "start": forward.start,
                "forward_end": forward.end,
                "heldout_start": heldout.start,
                "replay_end": heldout.end,
                "requested_end": heldout.requested_end,
                "truncation_reason": heldout.truncation_reason,
            },
            "snapshot_ids": {
                "sealed_decision": sealed.snapshot.snapshot_id,
                "forward_decision": forward_bundle.snapshot_id,
                "heldout_decision": heldout_bundle.snapshot_id,
            },
        }

        def replay(
            item: FrozenArtifact, over: ReplaySpan
        ) -> tuple[EvaluationResult | None, BaseException | None]:
            """One guarded replay of a frozen artifact over a span."""

            result, error, changed, restore_error = _run_guarded_evaluation(
                self.evaluator,
                over.request(
                    _frozen_revision(item),
                    schedule=self.config.schedule,
                    broker_profile=broker_profile,
                ),
                item,
            )
            if changed:
                if integrity_failure is not None:
                    integrity_failure(
                        {
                            **base,
                            "artifact_id": item.artifact_id,
                            "status": "integrity_failure",
                            "state_changed_during_test": True,
                            "result_ref": result.result_ref if result is not None else None,
                            "error": _error_text(error) if error is not None else None,
                        }
                    )
                if restore_error is not None:
                    raise FrozenArtifactRestoreFailed(
                        "strategy or model artifacts changed during a forward-stage replay "
                        f"and restoring the pre-evaluation trees failed: {restore_error}"
                    ) from restore_error
                raise FrozenArtifactMutated(
                    "strategy or model artifacts changed during a forward-stage replay"
                ) from error
            return result, error

        def replicate_results(
            over: ReplaySpan, stage: str
        ) -> tuple[list[tuple[FrozenArtifact, EvaluationResult]], list[dict[str, object]]]:
            """Every seed replicate replayed over ``over``: the completed
            replays, and a row for each replicate that raised in its own code,
            which is a reading (``replicate_strategy_error``) and leaves it
            out of every seed mean. A failure that measured nothing raises: a
            replicate is never dropped for want of the host."""

            completed: list[tuple[FrozenArtifact, EvaluationResult]] = []
            raised: list[dict[str, object]] = []
            for index, item in enumerate(replicates, start=1):
                _publish_progress(
                    progress, stage, run_id=run_id, seed_replicate=f"{index}/{len(replicates)}"
                )
                item_result, item_error = replay(item, over)
                if item_error is not None:
                    if not raised_by_strategy(item_error):
                        raise RuntimeError(
                            f"seed replicate {item.source_step_id} ({item.artifact_id}) did "
                            f"not complete its {stage} replay, and its reading is never "
                            f"dropped: {_error_text(item_error)}"
                        ) from item_error
                    raised.append(
                        {
                            **_replicate_identity(item),
                            "result_ref": None,
                            "strategy_error": True,
                            "error": _error_text(item_error),
                        }
                    )
                    continue
                assert item_result is not None
                completed.append((item, item_result))
            return completed, raised

        _publish_progress(progress, "sealed_replay", run_id=run_id)
        sealed_result, sealed_error = replay(artifact, sealed)
        sealed_record: dict[str, object] = {
            "label": sealed.label,
            "start": sealed.start,
            "end": sealed.end,
        }
        sealed_raised: list[dict[str, object]] = []
        if sealed_error is not None:
            if not raised_by_strategy(sealed_error):
                raise sealed_error
            sealed_block = sealed_strategy_error(
                start=sealed.start, end=sealed.end, error=_error_text(sealed_error)
            )
            sealed_record.update(result_ref=None, slice=sealed_block)
        else:
            assert sealed_result is not None
            sealed_replicates, sealed_raised = replicate_results(sealed, "sealed_replay")
            sealed_block = judged_sealed(
                sealed_result.result_ref,
                rules=self.config.acceptance,
                start=sealed.start,
                end=sealed.end,
                seed_replicates=[
                    (_replicate_identity(item), item_result.result_ref)
                    for item, item_result in sealed_replicates
                ],
            )
            sealed_record.update(
                result_ref=sealed_result.result_ref,
                **(
                    {
                        "seed_replicates": [
                            *(
                                {**_replicate_identity(item), "result_ref": item_result.result_ref}
                                for item, item_result in sealed_replicates
                            ),
                            *sealed_raised,
                        ]
                    }
                    if replicates
                    else {}
                ),
                slice=sealed_block,
            )
        base["sealed"] = sealed_record

        _publish_progress(progress, "forward_replay", run_id=run_id)
        result, error = replay(artifact, span)
        if error is not None:
            if not raised_by_strategy(error):
                raise error
            where = "heldout" if _failure_day(error) >= heldout.start else "forward"
            return {
                **base,
                "status": STRATEGY_ERROR,
                # The slice the frozen book raised in.
                "strategy_error": where,
                "error": _error_text(error),
                "result_ref": None,
                "slices": None,
                "refits_executed": None,
                "null_control": None,
                "verdict": graduation_verdict(
                    sealed=sealed_block,
                    strategy_errors=(where, *(("replicate",) if sealed_raised else ())),
                ),
            }
        assert result is not None
        replays, forward_raised = replicate_results(span, "forward_replay")
        _publish_progress(progress, "verdict", run_id=run_id)
        return {
            **base,
            **self._judge(
                result,
                artifact,
                sealed=sealed_block,
                forward=forward,
                heldout=heldout,
                broker_profile=broker_profile,
                seed_replicates=replays,
                raised_replicates=forward_raised,
                replicate_raised=bool(sealed_raised or forward_raised),
            ),
        }

    def _prepare_slot(self, slot: Slot) -> SnapshotBundle:
        return self.snapshots.prepare(
            phase=FORWARD_PHASE,
            start=slot.start,
            end=slot.end,
            decision_time=slot.anchor,
        )

    def _judge(
        self,
        result: EvaluationResult,
        artifact: FrozenArtifact,
        *,
        sealed: Mapping[str, object],
        forward: Slot,
        heldout: Slot,
        broker_profile: BrokerProfile,
        seed_replicates: Sequence[tuple[FrozenArtifact, EvaluationResult]] = (),
        raised_replicates: Sequence[Mapping[str, object]] = (),
        replicate_raised: bool = False,
    ) -> dict[str, object]:
        """The two reported slices of one completed forward replay through
        ``broker_profile``, and the verdict on the ``sealed`` block and on
        whether a seed replicate raised in any replay (``replicate_raised``).

        A slice that cannot be measured raises ``ValueError``, which fails the
        attempt: a verdict is never read off a number that was not measured.
        Each seed replicate's completed replay is handed to both slices, named
        by its artifact and source Step: ``slices.<name>.seed_replicates`` and
        ``slices.<name>.seed_mean`` carry its readings and the mean with the
        book, ``slices.forward.lower_bound`` the F2 bound over them together,
        and the record's ``seed_replicates`` its result and refits; a
        replicate that raised in this replay (``raised_replicates``) is a row
        there and in no slice.
        """

        replay, analysis = replay_and_analysis(result.result_ref)
        replicates: list[tuple[dict[str, object], dict[str, object]]] = []
        replicate_rows: list[dict[str, object]] = []
        for item, item_result in seed_replicates:
            item_replay, item_analysis = replay_and_analysis(item_result.result_ref)
            identity = _replicate_identity(item)
            replicates.append((identity, item_analysis))
            replicate_rows.append(
                {
                    **identity,
                    "revision_id": item.revision_id,
                    "result_ref": item_result.result_ref,
                    "refits_executed": _refits_executed(
                        item, item_replay, forward=forward, heldout=heldout
                    ),
                }
            )
        judged = judged_replay(
            replay,
            analysis,
            rules=self.config.acceptance,
            sealed=sealed,
            forward=(forward.start, forward.end),
            heldout=(heldout.start, heldout.end),
            seed_key=artifact.artifact_id,
            slippage_bps=broker_profile.slippage_bps,
            seed_replicates=replicates,
            strategy_errors=("replicate",) if replicate_raised else (),
        )
        replicate_rows.extend(dict(row) for row in raised_replicates)
        return {
            "status": "ok",
            "error": None,
            "result_ref": result.result_ref,
            "slices": judged["slices"],
            "refits_executed": _refits_executed(
                artifact, replay, forward=forward, heldout=heldout
            ),
            **({"seed_replicates": replicate_rows} if replicate_rows else {}),
            # Diagnostic only: where the forward slice's excess sits among
            # random-name replays of the same skeleton.
            "null_control": self._null_control(
                result.result_ref,
                start=forward.start,
                end=heldout.end,
                profile=broker_profile,
                seed=null_control_seed(artifact.artifact_id, "forward"),
                step=(forward.start, forward.end),
            ),
            "verdict": judged["verdict"],
        }

    def _frozen_artifact(self, block: object) -> FrozenArtifact:
        """The frozen artifact a frozen block (or one of its seed replicates)
        names, validated by its store against the output path the block
        records, read under the arm's current directory
        (``ledger.recorded_path``)."""

        if not isinstance(block, Mapping):
            raise TypeError("frozen record carries no frozen block")
        return self.artifacts.frozen(
            str(block["artifact_id"]),
            expected_path=recorded_path(self.config.experiment_dir, str(block["output_path"])),
            experiment_id=self.config.experiment_id,
        )

    # ---- shared --------------------------------------------------------

    def _account_record(self) -> dict[str, object]:
        """The account and the effective acceptance rules a stage was judged
        under, defaulted and overridden alike, so the ledger states its gates."""

        return {
            "initial_cash": self.config.broker_profile.initial_cash,
            "acceptance_rules": self.config.acceptance.to_record(),
        }

    def _null_control(
        self,
        result_ref: str,
        *,
        start: str,
        end: str,
        profile: BrokerProfile,
        seed: int,
        step: tuple[str, str] | None = None,
    ) -> dict[str, object] | None:
        """Rank one completed result against random-name copies of its own
        trades, replayed through ``profile`` (the one the result ran through).

        Informational evidence beside the return: it says whether the excess
        came from WHICH names were picked or only from the timing, sizing and
        exposure the skeleton already fixed. This percentile is not part of any
        verdict -- what a verdict grades against is the panel the evaluation
        stored with the result -- so a backend that cannot run it (the local
        development backend) leaves the block absent and a failure is recorded
        rather than raised.
        """

        runner = getattr(self.evaluator, "null_control", None)
        if not callable(runner):
            return None
        try:
            return runner(
                result_ref,
                start=start,
                end=end,
                profile=profile,
                schedule=self.config.schedule,
                seed=seed,
                step=step,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, the stage still runs
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def _current_skills(self) -> SkillsSnapshot:
        return latest_skills_snapshot(
            self.ledger.read(), experiment_dir=self.config.experiment_dir
        )

    def _publish_or_keep_skills(
        self,
        source_ref: str,
        *,
        current: SkillsSnapshot,
        generation_id: str,
        run_id: str,
    ) -> SkillsPublication:
        """Publish a validated collected workspace, or retain the ledger head."""

        if not str(source_ref).strip():
            return SkillsPublication(
                current.skills_ref,
                current.generation_id,
                current.stats,
                False,
            )
        source = resolve_collected_skills_source(
            self.config.experiment_dir, run_id, source_ref
        )
        return ExperimentSkillsStore(self.config.experiment_dir).publish(
            source,
            generation_id=generation_id,
            previous=current,
        )


def research_step_record(step: StepResult) -> dict[str, object]:
    """One completed Validation as the ledger's ``steps[]`` row.

    ``graded`` carries the graded series' plain excess, tracking error, IR
    and loadings over the span (``verdict.graded_statistics``), from the
    replay's own style sidecar (``None`` when they cannot be measured).
    ``series`` inside it says whether that is the active series (the replay
    carries a zero-skill panel) or the strategy's own. ``fingerprint`` names
    the bytes the revision holds, which the seed replicates' check compares.
    """

    return {
        "step_id": step.step_id,
        "revision_id": step.revision_id,
        "fingerprint": step.fingerprint,
        "control": step.control,
        "batch_id": step.batch_id,
        "span": step.span,
        "summary": step.validation.summary,
        "validation_result_ref": step.validation.result_ref,
        "graded": validation_statistics(step.validation.result_ref),
    }


def completed_rows(
    experiment_dir: str | Path, rows: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    """``rows`` each with what the ledger and the registration read of it.

    The artifact fingerprint: the one a row carries, else the one the arm's
    revision store recorded for its revision
    (``session_resume.revision_fingerprint``) -- a Step of the host's
    deterministic baseline. And the graded statistics
    (:func:`research_step_record`): the ones a row carries, else those of its
    style sidecar, read where the arm in ``experiment_dir`` now is.
    """

    def completed(row: Mapping[str, object]) -> Mapping[str, object]:
        if not row.get("fingerprint"):
            row = {**row, "fingerprint": revision_fingerprint(experiment_dir, str(row["revision_id"]))}
        if "graded" not in row:
            row = {**row, "graded": _measured_or_none(_style_analysis(experiment_dir, row))}
        return row

    return [completed(row) for row in rows]


def freeze_registration(
    nominee: Mapping[str, object],
    *,
    experiment_dir: str | Path,
    acceptance: AcceptanceRules,
    seed_replicates: Sequence[Mapping[str, object]] = (),
    deterministic: bool = False,
) -> dict[str, object]:
    """Whether one nominated Step registers for a freeze (docs/PIPELINE_DESIGN.md).

    Registration only, never a score: the nominee's summary breaks no hard
    nomination rule (``AcceptanceRules.evaluate``: finite metrics, the
    drawdown limit), it replayed the whole research-years span and is not a
    control, its series can be measured -- its residual reading included --
    the tracking mandate holds over its span where the rules set one
    (``verdict.registered_mandate``), and its ``seed_replicates`` are what a
    replicate must be (:func:`_seed_replicates`) -- none owed by a nomination
    declared ``deterministic``. ``nominee`` and the replicates are ledger
    rows (:func:`research_step_record`) of the arm in ``experiment_dir``. The
    session reads the same function at every test
    (``session_tools.SessionValidations.registration``) and the pipeline
    again before it freezes.
    """

    summary = nominee.get("summary")
    reasons = [
        *acceptance.evaluate(dict(summary) if isinstance(summary, Mapping) else {}),
        *judge(ConditionStage.REGISTRATION, nominee, {}),
    ]
    if reasons:
        return {"passed": False, "reasons": reasons}
    try:
        analysis = _style_analysis(experiment_dir, nominee)
        series = graded_statistics(analysis)["series"]
        # The reading the freeze records: a series it cannot read (one the
        # factor legs explain exactly, an all-cash book) is refused here,
        # while the session is still open, never after the freeze.
        residual_reading(analysis, alpha=acceptance.alpha)
    except ValueError as exc:
        return {"passed": False, "reasons": [ConditionReason.UNMEASURABLE], "error": str(exc)}
    mandate = registered_mandate(analysis, rules=acceptance)
    nominee = completed_rows(experiment_dir, [nominee])[0]
    replicates = completed_rows(experiment_dir, seed_replicates)
    seeds = _seed_replicates(
        nominee,
        replicates,
        experiment_dir=experiment_dir,
        deterministic=deterministic,
        series=str(series),
    )
    reasons = [*mandate["reasons"], *judge(ConditionStage.SEEDS, seeds, {})]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "mandate": mandate["mandate"],
        "seed_replicates": seeds,
        "thresholds": {"max_drawdown": acceptance.max_drawdown, **acceptance.mandate},
    }


# The one line a seed replicate changes: an integer assignment, with an
# optional ``int`` annotation and a trailing comment, to a seed name --
# ``SEED``, ``SEED_BASE`` or one ending in ``_SEED`` or ``_SEED_BASE``, any
# case (``SEED_BASE = 2000``). A count of seeds such as ``SEEDS_PER_HEAD``
# changes the strategy, not its draw, and is not one.
_SEED_LINE = re.compile(r"\s*([A-Za-z_]\w*)\s*(?::\s*int\s*)?=\s*([+-]?\d+)\s*(?:#.*)?")
_SEED_NAME = re.compile(r"(?:\w*_)?seed(?:_base)?", re.IGNORECASE)


def one_line_change(
    experiment_dir: str | Path, nominee_revision: str, other_revision: str
) -> tuple[str, str, str, int]:
    """The one line by which another revision differs from the nominee's.

    Read off the two revisions' manifests (every ``output/`` and ``models/``
    file by SHA-256): exactly one file may differ, a ``.py`` file of the
    strategy package, in exactly one line, with as many lines on both sides.
    It is checked on bytes, so the other revision cannot differ in anything
    that line does not explain: how a seed replicate (:func:`seed_change`)
    is told from the nominee. Returns
    the file's path in the package, the nominee's line, the other's and its
    1-based number; ``ValueError`` says what else differs.
    """

    root = Path(experiment_dir) / REVISIONS_DIR
    nominee = _revision_files(root / nominee_revision)
    other = _revision_files(root / other_revision)
    changed = sorted(path for path in {*nominee, *other} if nominee.get(path) != other.get(path))
    if not changed:
        raise ValueError("holds the nominee's own bytes")
    path = changed[0]
    if len(changed) > 1 or path not in nominee or path not in other:
        raise ValueError(
            f"differs from the nominee in {len(changed)} file(s) "
            f"({', '.join(changed[:5])}), not in exactly one line"
        )
    if not (path.startswith("output/") and path.endswith(".py")):
        raise ValueError(f"differs from the nominee in {path}, which is not strategy code")
    before = (root / nominee_revision / path).read_text(encoding="utf-8").splitlines()
    after = (root / other_revision / path).read_text(encoding="utf-8").splitlines()
    if len(before) != len(after):
        raise ValueError(
            f"changes the number of lines of {path} ({len(before)} to {len(after)}), "
            "not exactly one line"
        )
    lines = [index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
    if len(lines) != 1:
        raise ValueError(
            f"differs from the nominee in more than one line of {path} "
            f"({len(lines)} lines), not exactly one line"
        )
    line = lines[0]
    return path.removeprefix("output/"), before[line], after[line], line + 1


def seed_change(experiment_dir: str | Path, nominee_revision: str, replicate_revision: str) -> str:
    """The one seed line by which a replicate's revision differs from the nominee's.

    The one changed line (:func:`one_line_change`) must on both sides assign
    an integer to the same seed name (``_SEED_NAME``), with two different
    values. That is how a strategy changes its training seed and nothing
    else -- the packs set it in one knob line. Returns ``"<path>: <line>"``;
    ``ValueError`` says what else differs.
    """

    path, before, after, number = one_line_change(
        experiment_dir, nominee_revision, replicate_revision
    )
    old, new = _SEED_LINE.fullmatch(before), _SEED_LINE.fullmatch(after)
    if (
        old is None
        or new is None
        or old[1] != new[1]
        or _SEED_NAME.fullmatch(old[1]) is None
        or int(old[2]) == int(new[2])
    ):
        raise ValueError(
            f"changes line {number} of {path} to {after.strip()!r}, which does not "
            "give a seed name (SEED, SEED_BASE, or one ending in _SEED or _SEED_BASE) "
            "another integer value"
        )
    return f"{path}: {after.strip()}"


def _revision_files(directory: Path) -> dict[str, str]:
    """``path -> sha256`` of every file one recorded revision holds."""

    manifest = read_json_object(directory / "manifest.json")
    return {str(entry["path"]): str(entry["sha256"]) for entry in manifest["files"]}


def _seed_replicates(
    nominee: Mapping[str, object],
    replicates: Sequence[Mapping[str, object]],
    *,
    experiment_dir: str | Path,
    deterministic: bool,
    series: str,
) -> dict[str, object]:
    """What the registration reads of a nominee's seed replicates.

    A strategy that trains a model (its ``main.py`` defines ``fit``, the
    frozen block's ``fit_plan``) is frozen with its training seeds, which
    the forward stage replays beside it and reports. A seed replicate is a
    full-span, non-control row of the session whose bytes are the nominee's
    but for one seed line (:func:`seed_change`) and differ from every other
    registered row's, with a measured IR on the nominee's graded ``series``.
    The block records what was read, a ``problem`` per refused replicate,
    and is what the ``seeds`` conditions of ``verdict.CONDITIONS`` judge: a
    named replicate that is not one, and a nominee that trains a model and
    names none, unless the nomination declares its package ``deterministic``
    (no randomness, so a replicate would reproduce it exactly); the block
    records the declaration when it is made.
    """

    main = Path(experiment_dir) / REVISIONS_DIR / str(nominee["revision_id"]) / "output" / "main.py"
    trains = validate_strategy_package(main) is not None
    seen = {str(nominee["fingerprint"]): "the nominee"}
    entries: list[dict[str, object]] = []
    for row in replicates:
        step_id = str(row["step_id"])
        figures = row.get("graded")
        entry: dict[str, object] = {"step_id": step_id}
        try:
            if step_id == nominee["step_id"]:
                raise ValueError("is the nominee itself")
            if row.get("span") != FULL_SPAN:
                raise ValueError(f"replayed span {row.get('span')}, not {FULL_SPAN}")
            if row.get("control") is True:
                raise ValueError("is registered as a control")
            fingerprint = str(row["fingerprint"])
            if fingerprint in seen:
                raise ValueError(f"holds the same bytes as {seen[fingerprint]}")
            seen[fingerprint] = node_handle(step_id)
            ratio = figures.get("information_ratio") if isinstance(figures, Mapping) else None
            if not (
                isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                and math.isfinite(ratio)
                and figures.get("series") == series  # type: ignore[union-attr]
            ):
                raise ValueError(f"has no measured {series} information ratio")
            entry["seed_line"] = seed_change(
                experiment_dir, str(nominee["revision_id"]), str(row["revision_id"])
            )
            entry["information_ratio"] = float(figures["information_ratio"])  # type: ignore[index]
        except ValueError as exc:
            entry["problem"] = str(exc)
        entries.append(entry)
    return {
        "trains_a_model": trains,
        **({"deterministic": True} if deterministic else {}),
        "replicates": entries,
    }


def _style_analysis(experiment_dir: str | Path, row: Mapping[str, object]) -> dict[str, object]:
    """The style sidecar beside one Validation row's result, the row of the
    arm in ``experiment_dir`` read where that arm is now
    (``ledger.recorded_path``)."""

    result = recorded_path(experiment_dir, str(row["validation_result_ref"]))
    return read_json_object(result.parent / STYLE_ARTIFACT_NAME)


def null_control_seed(key: str, role: str) -> int:
    """A stable 32-bit seed per session (or artifact) and role.

    Stable across processes and runs (``hash`` is not), so a re-run draws the
    same null and its percentile can be compared with the one the ledger
    already holds. The session's ``run_null_control`` tool draws with the
    ``frozen`` role, so its block is the one the freeze would have drawn.
    """

    digest = hashlib.blake2b(f"{key}:{role}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _measured_or_none(analysis: Mapping[str, object]) -> dict[str, object] | None:
    """``verdict.graded_statistics`` of a whole sidecar; ``None`` when its
    span cannot be measured."""

    try:
        return graded_statistics(analysis)
    except ValueError:
        return None


def validation_statistics(result_ref: str) -> dict[str, object] | None:
    """One validation's graded statistics: plain excess, tracking error, IR
    and loadings (``verdict.graded_statistics``).

    Every successful replay writes its style sidecar, so a missing or
    unreadable one raises: dropping the row would silently narrow the freeze
    gate's IR dispersion. Only an unmeasurable span is None. The console's
    live step listing reads its rows through this same function.
    """

    path = Path(result_ref).parent / STYLE_ARTIFACT_NAME
    return _measured_or_none(read_json_object(path))


def replay_and_analysis(result_ref: str) -> tuple[dict[str, object], dict[str, object]]:
    """One completed replay's result and the style sidecar beside it."""

    path = Path(result_ref)
    return (
        read_json_object(path),
        read_json_object(path.parent / STYLE_ARTIFACT_NAME),
    )


def _result_analysis(result_ref: str) -> dict[str, object]:
    """The style sidecar beside one completed replay's result."""

    return replay_and_analysis(result_ref)[1]


def _replicate_identity(item: FrozenArtifact) -> dict[str, object]:
    """What names a frozen seed replicate in a forward-stage record."""

    return {"artifact_id": item.artifact_id, "source_step_id": item.source_step_id}


def judged_sealed(
    result_ref: str,
    *,
    rules: AcceptanceRules,
    start: str,
    end: str,
    seed_replicates: Sequence[tuple[Mapping[str, object], str]] = (),
) -> dict[str, object]:
    """The sealed gate on one completed replay of the frozen book over the
    sealed years (``verdict.sealed_slice``), its seed replicates' replays
    given as what names each and its result reference. What a forward record
    stores under ``sealed.slice``, so a stored reading is judged again by
    this same function (``scripts/dev/check_verdicts.py``)."""

    return sealed_slice(
        _result_analysis(result_ref),
        rules=rules,
        start=start,
        end=end,
        seed_replicates=[
            (identity, _result_analysis(reference)) for identity, reference in seed_replicates
        ],
    )


def judged_replay(
    replay: Mapping[str, object],
    analysis: Mapping[str, object],
    *,
    rules: AcceptanceRules,
    sealed: Mapping[str, object],
    forward: tuple[str, str],
    heldout: tuple[str, str],
    seed_key: str,
    slippage_bps: float,
    seed_replicates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]] = (),
    strategy_errors: Sequence[str] = (),
) -> dict[str, object]:
    """The ``slices`` and ``verdict`` of one completed forward replay: its
    result ``replay`` and style sidecar ``analysis``, read under ``rules``
    over the ``forward`` and ``heldout`` bounds, each slice with its activity,
    and the verdict on the ``sealed`` gate's block and ``strategy_errors``
    (``verdict.graduation_verdict``).

    What a forward record stores, so a stored replay is judged again by this
    same function (``scripts/dev/check_verdicts.py``). ``seed_key`` is the
    frozen artifact's id, ``slippage_bps`` the Broker's and
    ``seed_replicates`` what names each completed replicate with its replay's
    sidecar.
    """

    curve, executions = replay["equity_curve"], replay["executions"]
    activity = {
        name: window_activity(curve, executions, start=start, end=end)  # type: ignore[arg-type]
        for name, (start, end) in (("forward", forward), ("heldout", heldout))
    }
    forward_block = forward_slice(
        analysis,
        rules=rules,
        start=forward[0],
        end=forward[1],
        seed_key=seed_key,
        slippage_bps=slippage_bps,
        turnover=float(activity["forward"]["turnover"]),  # type: ignore[arg-type]
        round_trips=int(activity["forward"]["round_trips"]),  # type: ignore[arg-type]
        mean_gross=float(activity["forward"]["mean_gross"]),  # type: ignore[arg-type]
        seed_replicates=seed_replicates,
    )
    heldout_block = heldout_slice(
        analysis,
        rules=rules,
        start=heldout[0],
        end=heldout[1],
        mean_gross=float(activity["heldout"]["mean_gross"]),  # type: ignore[arg-type]
        seed_replicates=seed_replicates,
    )
    return {
        "slices": {
            "forward": {**forward_block, "activity": activity["forward"]},
            "heldout": {**heldout_block, "activity": activity["heldout"]},
        },
        "verdict": graduation_verdict(
            sealed=sealed,
            forward=forward_block,
            heldout=heldout_block,
            strategy_errors=strategy_errors,
        ),
    }


def _refits_executed(
    artifact: FrozenArtifact, replay: Mapping[str, object], *, forward: Slot, heldout: Slot
) -> dict[str, int]:
    """How many ``fit`` calls of one forward replay fell in each slice."""

    fit = validate_strategy_package(artifact.path / "main.py")
    days = [str(value)[:10].replace("-", "") for value in replay.get("inference_dates") or ()]  # type: ignore[union-attr]
    return {"forward": _fits_in(fit, days, forward), "heldout": _fits_in(fit, days, heldout)}


def _fits_in(fit, inference_days: Sequence[str], slot: Slot) -> int:
    """How many ``fit`` calls of the replay fell inside ``slot``."""

    if fit is None:
        return 0
    count = 0
    last: str | None = None
    for day in inference_days:
        if fit.is_due(day, last):
            count += slot.start <= day <= slot.end
            last = day
    return count


def _failure_day(error: BaseException) -> str:
    """``YYYYMMDD`` of the decision whose strategy call raised."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, BacktestError) and current.inference_at is not None:
            return current.inference_at.strftime("%Y%m%d")
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    raise RuntimeError("a strategy error carries no decision time") from error


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _keep_frozen_artifact_ids(
    records: Sequence[Mapping[str, object]],
    extra_id: str | None = None,
) -> tuple[str, ...]:
    """The arm's frozen artifact, its seed replicates and, on an arm frozen
    before 2026-11-06, its ablation; those an incubation recorded before
    2026-11-03 froze, any artifact an integrity row names, and the freeze
    now being recorded."""

    keep = {str(extra_id)} if extra_id else set()
    for row in (frozen_record(records), incubation_record(records)):
        if row is None:
            continue
        block: Mapping[str, object] = row["frozen"]  # type: ignore[assignment]
        keep.add(str(block["artifact_id"]))
        keep.update(
            str(item["artifact_id"])
            for item in (*(block.get("seed_replicates") or ()), block.get("ablation"))  # type: ignore[misc]
            if item
        )
    for record in records:
        if is_frozen_artifact_mutation(record) and record.get("artifact_id"):
            keep.add(str(record["artifact_id"]))
    return tuple(sorted(keep))


def _snapshot_config_record(snapshots: SnapshotProvider) -> dict[str, object]:
    """The provider's own decision-window configuration, when it has one.

    ``build_experiment_facts`` reads ``snapshot_config.decision_windows`` for the
    visible-timeline block; the local daily provider has no such configuration.
    """
    config = getattr(snapshots, "config", None)
    to_record = getattr(config, "to_record", None)
    return dict(to_record()) if callable(to_record) else {}


def _frozen_revision(artifact: FrozenArtifact) -> ArtifactRevision:
    return ArtifactRevision(artifact.artifact_id, artifact.path, artifact.model_path)


def _snapshot_frozen_trees(artifact: FrozenArtifact, dest: Path) -> None:
    copy_artifact(artifact.path, dest / "output")
    copy_model_artifacts(artifact.model_path, dest / "models")


def _live_models_path(artifact: FrozenArtifact) -> Path:
    if artifact.model_path is not None:
        return Path(artifact.model_path)
    return artifact.path.parent / "models"


def _frozen_trees_changed(artifact: FrozenArtifact, snapshot: Path) -> bool:
    if modification_delta(snapshot / "output", artifact.path).changed_files:
        return True
    return bool(
        model_artifact_delta(snapshot / "models", _live_models_path(artifact)).changed_files
    )


def _run_guarded_evaluation(
    evaluator: EvaluationBackend,
    request: EvaluationRequest,
    artifact: FrozenArtifact,
) -> tuple[EvaluationResult | None, BaseException | None, bool, BaseException | None]:
    """Evaluate once, restore frozen trees if they changed, and report both."""
    with TemporaryDirectory() as raw:
        root = Path(raw)
        _snapshot_frozen_trees(artifact, root)
        result: EvaluationResult | None = None
        error: BaseException | None = None
        restore_error: BaseException | None = None
        try:
            result = evaluator.evaluate(request)
        except Exception as exc:  # noqa: BLE001 - caller chooses diagnostic vs fail-fast
            error = exc
        try:
            changed = _frozen_trees_changed(artifact, root)
        except Exception:  # noqa: BLE001 - comparison failure is an integrity failure
            changed = True
        if changed:
            try:
                restore_frozen_artifact_trees(
                    output_path=artifact.path,
                    snapshot_output=root / "output",
                    models_path=artifact.model_path,
                    snapshot_models=root / "models",
                )
                if _frozen_trees_changed(artifact, root):
                    raise RuntimeError(
                        "frozen trees still differ after restoring pre-evaluation bytes"
                    )
            except Exception as exc:  # noqa: BLE001 - restore failure is worse than the mutation
                restore_error = exc
        return result, error, changed, restore_error


def _optional_hook(value: object, name: str):
    if value is None:
        return None
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def _publish_progress(hook, stage: str, **progress: object) -> None:
    if hook is not None:
        hook(stage, dict(progress) if progress else None)


def _session_timing(
    context: Mapping[str, object],
    fallback_started: float,
) -> dict[str, float]:
    callback = context.get("session_timing")
    if callable(callback):
        value = callback()
        if not isinstance(value, Mapping):
            raise TypeError("session_timing must return a mapping")
        wall = float(value.get("run_wall_seconds", 0.0))
        if wall < 0:
            raise ValueError("session timing values must be non-negative")
        return {"run_wall_seconds": round(wall, 1)}
    return {
        "run_wall_seconds": round(max(0.0, time.monotonic() - fallback_started), 1)
    }


def _optional_gpu_count(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        raise ValueError("sandbox_gpu_count override must be an integer in 0..4")
    return value


def _session_budgets(
    config: RollingExperimentConfig, override: object
) -> dict[str, int | float]:
    limits: dict[str, int | float] = {
        "max_llm_calls": config.max_llm_calls,
        "max_api_spend_usd": config.max_api_spend_usd,
        "deadline_seconds": config.max_research_minutes * 60,
    }
    if override not in (None, {}):
        if not isinstance(override, dict):
            raise TypeError("resource_override must be an object")
        unknown = sorted(set(override).difference(limits))
        if unknown:
            raise ValueError(f"unknown resource override(s): {unknown}")
        for name, value in override.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} override must be positive")
            if name == "deadline_seconds":
                # The session deadline may be raised, bounded by the absolute
                # ceiling above; other budgets stay downward-only.
                if float(value) > _MAX_DEADLINE_OVERRIDE_MINUTES * 60:
                    raise ValueError(
                        "deadline_seconds override cannot exceed "
                        f"{_MAX_DEADLINE_OVERRIDE_MINUTES} minutes"
                    )
            elif float(value) > float(limits[name]):
                raise ValueError(
                    f"{name} override cannot exceed the configured session limit"
                )
            limits[name] = (
                float(value)
                if name in ("deadline_seconds", "max_api_spend_usd")
                else int(value)
            )
    # Grace is not a resource_override key: add it after the override check so
    # the session budget and the runner reservation share one config source.
    main_minutes = float(limits["deadline_seconds"]) / 60.0
    limits["deadline_seconds"] = session_deadline_seconds(
        main_minutes, config.deadline_grace_minutes
    )
    limits["deadline_grace_seconds"] = float(config.deadline_grace_minutes) * 60.0
    return limits


__all__ = [
    "DailyStrategyPipeline",
    "EnvironmentBlockedAttempt",
    "RollingExperimentPipeline",
    "completed_rows",
    "freeze_registration",
    "judged_replay",
    "judged_sealed",
    "null_control_seed",
    "one_line_change",
    "replay_and_analysis",
    "research_step_record",
    "seed_change",
    "validation_statistics",
]
