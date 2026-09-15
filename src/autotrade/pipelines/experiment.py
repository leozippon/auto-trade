"""Experiment pipeline: one research session, one freeze, one forward replay.

docs/pipeline-design.md. The Pipeline schedules Data, Environment and Agent in
time order, freezes inputs and outputs at each boundary and writes the single
experiment ledger. It implements no investment logic and never rewrites
strategy content; it only prepares, freezes, replays and records.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from autotrade.environment.artifacts import (
    copy_artifact,
    copy_model_artifacts,
    model_artifact_delta,
    modification_delta,
    restore_frozen_artifact_trees,
)
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    StrategyExecutor,
    TrustedStrategyExecutor,
    raised_by_strategy,
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
from autotrade.environment.runtime import agent_trace_path, chmod_tree
from autotrade.environment.strategy import NLQuery
from autotrade.environment.strategy_loader import validate_strategy_package

from .agent_inbox import expire_experiment_session_inbox
from .calendar import FULL_SPAN, Slot
from .config import (
    ArtifactRevision,
    ArtifactStore,
    BudgetUsed,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    FrozenArtifact,
    ReplaySpan,
    ResearchDeveloper,
    ResearchSessionRequest,
    RollingExperimentConfig,
    SnapshotBundle,
    SnapshotProvider,
    StepResult,
    StrategyExperimentConfig,
    research_span,
    session_deadline_seconds,
)
from .ledger import (
    FORWARD_SESSION_KEY,
    FORWARD_STAGE,
    RESEARCH_SESSION_KEY,
    RESEARCH_STAGE,
    STRATEGY_ERROR,
    ExperimentLedger,
    FrozenArtifactMutated,
    FrozenArtifactRestoreFailed,
    RunMarkers,
    assert_no_frozen_artifact_mutation,
    forward_record,
    frozen_record,
    is_frozen_artifact_mutation,
    research_over,
    research_records,
)
from .pit_views_seed import FORWARD_PHASE, RESEARCH_PHASE
from .session_resume import load_recorded_steps, resume_state
from .skills import (
    ExperimentSkillsStore,
    SkillsPublication,
    SkillsSnapshot,
    latest_skills_snapshot,
    resolve_collected_skills_source,
)
from .verdict import (
    forward_mde,
    forward_slice,
    freeze_gate,
    graduation_verdict,
    heldout_slice,
    neutralized_statistics,
)

# A session deadline override may raise the research deadline above the
# configured maximum, bounded by this absolute ceiling in minutes: twice the
# default research budget (config.max_research_minutes), enough headroom for
# one slow session without letting it run unattended for weeks.
_MAX_DEADLINE_OVERRIDE_MINUTES = 4800


class DailyStrategyPipeline:
    def __init__(
        self,
        config: StrategyExperimentConfig,
        *,
        nl_query: NLQuery | None = None,
        context_data: ContextDataProvider | None = None,
        execution_price: ExecutionPriceProvider | None = None,
        executor_factory: Callable[[StrategyExperimentConfig], StrategyExecutor]
        | None = None,
    ) -> None:
        self.config = config
        self.nl_query = nl_query
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
            try:
                return run_daily_replay(
                    daily=frame,
                    strategy=executor,
                    schedule=self.config.schedule,
                    profile=self.config.broker_profile,
                    nl_query=self.nl_query,
                    context_data=self.context_data,
                    execution_price=self.execution_price,
                    corporate_actions=corporate_actions,
                )
            finally:
                executor.close()
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

    def research_inputs(self) -> tuple[SnapshotBundle, tuple[ReplaySpan, ...]]:
        """The decision view at research end and the research years.

        Each year is a one-slot span over its slot and the decision view at its
        anchor; the spans a Validation replays are resolved from them
        (``config.research_span``). Prepared from the research geometry alone:
        every slot ends by research end and every anchor is at or before its
        decision time, so no row stamped after research end reaches a research
        session.
        """

        geometry = self.config.geometry
        years = geometry.research_years
        bundles = []
        for slot in years:
            if (
                slot.end > geometry.research_end
                or slot.anchor > geometry.research_decision_time
            ):
                raise RuntimeError(
                    f"research slot {slot.label} {slot.start}..{slot.end} reaches past "
                    f"research end {geometry.research_end}"
                )
            bundles.append(
                self.snapshots.prepare(
                    phase=RESEARCH_PHASE,
                    start=slot.start,
                    end=slot.end,
                    decision_time=slot.anchor,
                )
            )
        decision = self.snapshots.prepare_decision(
            decision_time=geometry.research_decision_time
        )
        return decision, tuple(
            ReplaySpan(
                label=slot.label,
                mode=RESEARCH_PHASE,
                start=slot.start,
                end=slot.end,
                snapshot=bundle,
            )
            for slot, bundle in zip(years, bundles, strict=True)
        )

    def run_research_session(
        self, *, session_context: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Run the arm's one research session and record its outcome.

        The session starts from the template and ends by freezing its nominee
        or ending the arm. A freeze passes only the freeze gate; a nomination
        the gate refuses, ``no_edge`` and an exhausted budget all end the arm
        without a deliverable, so the record always carries ``frozen`` or
        ``arm_end`` and research is over once it is written. An attempt that
        failed before that record is continued, not repeated: the next attempt
        starts from the budget, the compaction summary and the Validations the
        interrupted ones left in the trace and the step tree.
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
            "epoch_id": RESEARCH_STAGE,
            "fold_id": RESEARCH_SESSION_KEY,
            "run_id": run_id,
            "session_key": RESEARCH_SESSION_KEY,
            "phase": RESEARCH_STAGE,
        }
        # Evidence for a run that never gets to run its own except branch.
        self.run_markers.begin(attempt)
        try:
            _publish_progress(progress, "pit_snapshot", run_id=run_id, phase=RESEARCH_STAGE)
            decision, years = self.research_inputs()
            session = self.developer(
                ResearchSessionRequest(
                    experiment_id=self.config.experiment_id,
                    run_id=run_id,
                    snapshot=decision,
                    decision_time=self.config.geometry.research_decision_time,
                    research_years=years,
                    input_window_start=_months_before(
                        self.config.geometry.research_end, self.config.window_months
                    ),
                    max_replay_years=int(budgets["max_replay_years"]),
                    max_llm_calls=int(budgets["max_llm_calls"]),
                    deadline_seconds=budgets["deadline_seconds"],
                    deadline_grace_seconds=budgets["deadline_grace_seconds"],
                    directive=str(context.get("directive") or ""),
                    sandbox_gpu_count=_optional_gpu_count(context.get("sandbox_gpu_count")),
                    acceptance_rules=self.config.acceptance.to_record(),
                    modification_constraints=self.config.step_constraints,
                    snapshot_config=_snapshot_config_record(self.snapshots),
                    record_failed_attempts=self.config.record_failed_attempts,
                    nl_failure_policy=self.config.nl_failure_policy,
                    finalize_before_deadline_seconds=self.config.finalize_before_deadline_seconds,
                    max_null_controls=self.config.max_null_controls,
                    progress_hook=progress,
                    skills_source_ref=(
                        str(current_skills.root) if current_skills.root is not None else ""
                    ),
                    budget_used=resume.budget_used if resume is not None else BudgetUsed(),
                    steps_before=steps_before,
                    resume=resume,
                )
            )
            spent = sum(research_span(years, step.span).slots for step in session.steps)
            if spent > budgets["max_replay_years"]:
                raise RuntimeError(
                    f"research session replayed {spent} replay-years, over its budget "
                    f"of {budgets['max_replay_years']}"
                )
            step_rows = [research_step_record(step) for step in session.steps]
            gate: dict[str, object] | None = None
            frozen: dict[str, object] | None = None
            arm_end: dict[str, object] | None = None
            if session.outcome == "freeze":
                nominee = next(
                    (row for row in step_rows if row["step_id"] == session.node_id), None
                )
                if nominee is None:
                    raise RuntimeError(
                        f"the nominated node {session.node_id!r} is not a Step of this session"
                    )
                gate = freeze_gate_for(
                    records,
                    step_rows,
                    nominee,
                    hard_reasons=self.config.acceptance.evaluate(
                        dict(nominee["summary"])
                    )[0],
                )
                if gate["passed"]:
                    _publish_progress(progress, "freezing", run_id=run_id)
                    frozen = self._freeze(
                        next(step for step in session.steps if step.step_id == nominee["step_id"]),
                        gate=gate,
                        run_id=run_id,
                        null_controls=session.null_controls,
                    )
                    frozen_id = str(frozen["artifact_id"])
                else:
                    reasons = ", ".join(str(reason) for reason in gate["reasons"])
                    arm_end = {
                        "status": "no_deliverable",
                        "reason": f"freeze refused by the gate ({reasons})",
                    }
            elif session.outcome == "no_edge":
                arm_end = {"status": "no_deliverable", "reason": f"no_edge: {session.reason}"}
            else:
                arm_end = {
                    "status": "no_deliverable",
                    "reason": (
                        "research budget exhausted without a freeze "
                        f"({session.finish_reason})"
                    ),
                }
            _publish_progress(progress, "publishing", run_id=run_id)
            skills = self._publish_or_keep_skills(
                session.skills_source_ref,
                current=current_skills,
                generation_id=f"{RESEARCH_SESSION_KEY}_{run_id}",
                run_id=run_id,
            )
            trace = agent_trace_path(self.config.experiment_dir / "artifacts", run_id)
            record = {
                "record_type": "research_session",
                **{key: attempt[key] for key in ("experiment_id", "epoch_id", "fold_id", "run_id")},
                "session_key": RESEARCH_SESSION_KEY,
                "conversation_id": session.conversation_id,
                "finish_reason": session.finish_reason,
                "outcome": session.outcome,
                "reason": session.reason or None,
                "nominated_step_id": session.node_id if session.outcome == "freeze" else None,
                "steps": step_rows,
                "trials_to_date": len(_arm_revisions(records, step_rows)),
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
                "attempts": session.attempt,
                "budget_used": session.budget_used.to_record(),
                **_session_timing(context, run_started),
            }
            self.ledger.append(record)
            wrote_ledger_record = True
            expire_experiment_session_inbox(
                self.config.experiment_dir, RESEARCH_SESSION_KEY, expired_by=run_id
            )
            # Candidate revisions are discarded only once the session is
            # recorded: a failed attempt's revisions are what its resumed
            # attempt freezes.
            prune = getattr(self.artifacts, "prune_transient", None)
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

    def _freeze(
        self,
        nominee: StepResult,
        *,
        gate: Mapping[str, object],
        run_id: str,
        null_controls: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        """Freeze the nominee's immutable revision and state what it was frozen on."""

        artifact_id = f"strategy_{RESEARCH_SESSION_KEY}_{uuid.uuid4().hex[:12]}"
        stored = self.artifacts.freeze_revision(
            nominee.revision_id,
            artifact_id=artifact_id,
            experiment_id=self.config.experiment_id,
            epoch_id=RESEARCH_STAGE,
            fold_id=RESEARCH_SESSION_KEY,
            run_id=run_id,
            step_id=nominee.step_id,
        )
        output = Path(stored.path)
        models = Path(stored.model_path) if stored.model_path is not None else None
        forward = self.config.geometry.forward
        forward_days = sum(1 for day in self.trading_days if forward.start <= day <= forward.end)
        fit = validate_strategy_package(output / "main.py")
        null_control = (
            dict(null_controls[nominee.step_id])
            if nominee.step_id in null_controls
            else self._null_control(
                nominee.validation.result_ref,
                start=self.config.geometry.research_start,
                end=self.config.geometry.research_end,
                seed=null_control_seed(RESEARCH_SESSION_KEY, "frozen"),
            )
        )
        return {
            "artifact_id": artifact_id,
            "output_path": str(output),
            "models_path": str(models) if models is not None and models.is_dir() else None,
            "source_step_id": nominee.step_id,
            "revision_id": nominee.revision_id,
            "research_result_ref": nominee.validation.result_ref,
            "days": gate["days"],
            "neutralized_excess": gate["neutralized_excess"],
            "tracking_error": gate["tracking_error"],
            "information_ratio": gate["information_ratio"],
            "blocks": nominee.validation.summary.get("sub_windows"),
            "null_control": null_control,
            "deflated_sharpe": gate["deflated_sharpe"],
            "full_span_validations": gate["full_span_validations"],
            "forward_mde": forward_mde(float(gate["tracking_error"]), forward_days),  # type: ignore[arg-type]
            "fit_plan": {
                "fit": fit is not None,
                "refit_period": fit.refit_period if fit is not None else None,
            },
        }

    # ---- forward -------------------------------------------------------

    def run_forward(
        self, *, session_context: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Replay the frozen artifact once over forward and Held-out, then judge it.

        One continuous replay: the book is not reset at the Held-out boundary,
        and both slices are read from its one result. The strategy's own
        exception is a measurement and discards the arm; any other failure
        measured nothing and fails the attempt, which the caller retries from
        the forward start. Frozen trees that changed during the replay are
        recorded and fail closed.
        """

        records = self.ledger.read()
        assert_no_frozen_artifact_mutation(records)
        frozen_row = frozen_record(records)
        if frozen_row is None:
            raise RuntimeError("the forward replay needs a frozen artifact")
        if forward_record(records) is not None:
            raise RuntimeError("the arm's forward verdict is already recorded")
        artifact = self._frozen_artifact(frozen_row)
        forward = self.config.geometry.forward
        heldout = self.config.geometry.heldout(self.trading_days)
        context = dict(session_context or {})
        progress = _optional_hook(context.get("progress_hook"), "progress_hook")
        run_id = f"run_{uuid.uuid4().hex}"
        attempt = {
            "experiment_id": self.config.experiment_id,
            "epoch_id": FORWARD_STAGE,
            "fold_id": FORWARD_SESSION_KEY,
            "run_id": run_id,
            "session_key": FORWARD_SESSION_KEY,
            "phase": FORWARD_STAGE,
        }
        self.run_markers.begin(attempt)
        wrote_ledger_record = False
        try:
            _publish_progress(progress, "pit_snapshot", run_id=run_id, phase=FORWARD_STAGE)
            forward_bundle = self._prepare_slot(forward)
            heldout_bundle = self._prepare_slot(heldout)
            span = ReplaySpan(
                label=FORWARD_STAGE,
                mode=FORWARD_PHASE,
                start=forward.start,
                end=heldout.end,
                snapshot=forward_bundle,
                continuation=(heldout_bundle.replay_ref,),
            )
            _publish_progress(progress, "forward_replay", run_id=run_id)
            result, error, changed, restore_error = _run_guarded_evaluation(
                self.evaluator,
                span.request(
                    _frozen_revision(artifact),
                    schedule=self.config.schedule,
                    broker_profile=self.config.broker_profile,
                ),
                artifact,
            )
            base = {
                "record_type": "forward",
                **{key: attempt[key] for key in ("experiment_id", "epoch_id", "fold_id", "run_id")},
                "session_key": FORWARD_SESSION_KEY,
                "artifact_id": artifact.artifact_id,
                "replay": {
                    "start": forward.start,
                    "forward_end": forward.end,
                    "heldout_start": heldout.start,
                    "replay_end": heldout.end,
                    "requested_end": heldout.requested_end,
                    "truncation_reason": heldout.truncation_reason,
                },
                "snapshot_ids": {
                    "forward_decision": forward_bundle.snapshot_id,
                    "heldout_decision": heldout_bundle.snapshot_id,
                },
            }
            if changed:
                self.ledger.append(
                    {
                        **base,
                        "status": "integrity_failure",
                        "state_changed_during_test": True,
                        "result_ref": result.result_ref if result is not None else None,
                        "error": _error_text(error) if error is not None else None,
                    }
                )
                # The integrity row is this run's record: the fail-fast below
                # must not also log a failed attempt.
                wrote_ledger_record = True
                if restore_error is not None:
                    raise FrozenArtifactRestoreFailed(
                        "strategy or model artifacts changed during the forward replay "
                        f"and restoring the pre-evaluation trees failed: {restore_error}"
                    ) from restore_error
                raise FrozenArtifactMutated(
                    "strategy or model artifacts changed during the forward replay"
                ) from error
            if error is not None:
                if not raised_by_strategy(error):
                    raise error
                where = (
                    "heldout" if _failure_day(error) >= heldout.start else "forward"
                )
                record = {
                    **base,
                    "status": STRATEGY_ERROR,
                    "error": _error_text(error),
                    "result_ref": None,
                    "slices": None,
                    "refits_executed": None,
                    "null_control": None,
                    "verdict": graduation_verdict(
                        forward=None, heldout=None, strategy_error=where
                    ),
                }
            else:
                assert result is not None
                _publish_progress(progress, "verdict", run_id=run_id)
                record = {
                    **base,
                    **self._judge(result, artifact, forward=forward, heldout=heldout),
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
        forward: Slot,
        heldout: Slot,
    ) -> dict[str, object]:
        """The two slices of one completed replay and the verdict on them.

        A slice that cannot be measured raises ``ValueError``, which fails the
        attempt: a verdict is never read off a number that was not measured.
        """

        result_path = Path(result.result_ref)
        replay = json.loads(result_path.read_text(encoding="utf-8"))
        analysis = json.loads(
            (result_path.parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")
        )
        curve = replay["equity_curve"]
        executions = replay["executions"]
        acceptance = self.config.acceptance
        forward_activity = window_activity(
            curve, executions, start=forward.start, end=forward.end
        )
        heldout_activity = window_activity(
            curve, executions, start=heldout.start, end=heldout.end
        )
        forward_block = forward_slice(
            analysis,
            start=forward.start,
            end=forward.end,
            seed_key=artifact.artifact_id,
            max_drawdown=acceptance.max_drawdown,
            cost_stress_multiplier=acceptance.cost_stress_multiplier,
            slippage_bps=self.config.broker_profile.slippage_bps,
            turnover=float(forward_activity["turnover"]),  # type: ignore[arg-type]
            round_trips=int(forward_activity["round_trips"]),  # type: ignore[arg-type]
            mean_gross=float(forward_activity["mean_gross"]),  # type: ignore[arg-type]
        )
        heldout_block = heldout_slice(
            analysis,
            start=heldout.start,
            end=heldout.end,
            forward_tracking_error=float(forward_block["tracking_error"]),  # type: ignore[arg-type]
            max_drawdown=acceptance.max_drawdown,
            mean_gross=float(heldout_activity["mean_gross"]),  # type: ignore[arg-type]
        )
        fit = validate_strategy_package(artifact.path / "main.py")
        inference_days = [
            str(value)[:10].replace("-", "") for value in replay.get("inference_dates") or ()
        ]
        return {
            "status": "ok",
            "error": None,
            "result_ref": result.result_ref,
            "slices": {
                "forward": {**forward_block, "activity": forward_activity},
                "heldout": {**heldout_block, "activity": heldout_activity},
            },
            "refits_executed": {
                "forward": _fits_in(fit, inference_days, forward),
                "heldout": _fits_in(fit, inference_days, heldout),
            },
            # Diagnostic only: where the forward slice's excess sits among
            # random-name replays of the same skeleton.
            "null_control": self._null_control(
                result.result_ref,
                start=forward.start,
                end=heldout.end,
                seed=null_control_seed(artifact.artifact_id, "forward"),
                step=(forward.start, forward.end),
            ),
            "verdict": graduation_verdict(forward=forward_block, heldout=heldout_block),
        }

    def _frozen_artifact(self, frozen_row: Mapping[str, object]) -> FrozenArtifact:
        """The frozen artifact the ledger names, validated by its store."""

        block = frozen_row["frozen"]
        if not isinstance(block, Mapping):
            raise TypeError("frozen record carries no frozen block")
        stored = self.artifacts.frozen(
            str(block["artifact_id"]),
            expected_path=str(block["output_path"]),
            experiment_id=self.config.experiment_id,
        )
        return FrozenArtifact(
            str(stored.artifact_id),
            Path(stored.path),
            Path(stored.model_path) if stored.model_path is not None else None,
            str(stored.source_run_id),
            str(stored.source_fold_id),
            str(stored.source_step_id),
            str(stored.revision_id),
        )

    # ---- shared --------------------------------------------------------

    def _null_control(
        self,
        result_ref: str,
        *,
        start: str,
        end: str,
        seed: int,
        step: tuple[str, str] | None = None,
    ) -> dict[str, object] | None:
        """Rank one completed result against random-name copies of its own trades.

        Informational evidence beside the return: it says whether the excess
        came from WHICH names were picked or only from the timing, sizing and
        exposure the skeleton already fixed. The null is not part of any
        verdict, so a backend that cannot run it (the local development
        backend) leaves the block absent and a failure is recorded rather than
        raised.
        """

        runner = getattr(self.evaluator, "null_control", None)
        if not callable(runner):
            return None
        try:
            return runner(
                result_ref,
                start=start,
                end=end,
                profile=self.config.broker_profile,
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

    ``neutralized`` carries the span's neutralised excess, residual tracking
    error and IR from the replay's own style sidecar (``None`` when they cannot
    be measured), the figures the freeze gate counts.
    """

    return {
        "step_id": step.step_id,
        "revision_id": step.revision_id,
        "span": step.span,
        "summary": step.validation.summary,
        "validation_result_ref": step.validation.result_ref,
        "neutralized": _neutralized(step.validation.result_ref),
    }


def freeze_gate_for(
    records: Sequence[Mapping[str, object]],
    session_rows: Sequence[Mapping[str, object]],
    nominee: Mapping[str, object],
    *,
    hard_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """The freeze gate of one nominated Step against the whole arm (PL1 §4.1).

    Trials are the distinct revisions validated anywhere in the arm, earlier
    sessions' recorded Steps and this session's alike; the IR dispersion is
    taken over every measurable full-span validation. A nominee that did not
    replay the full research period, fails a hard nomination rule, or whose
    statistics cannot be measured does not pass.
    """

    reasons = list(hard_reasons)
    if nominee.get("span") != FULL_SPAN:
        reasons.append("freeze_needs_full_span_validation")
    if reasons:
        return {"passed": False, "reasons": reasons}
    rows = [*_recorded_steps(records), *session_rows]
    irs = [
        float(row["neutralized"]["information_ratio"])  # type: ignore[index]
        for row in rows
        if row.get("span") == FULL_SPAN and _finite_ir(row.get("neutralized"))
    ]
    result_ref = Path(str(nominee["validation_result_ref"]))
    analysis = json.loads(
        (result_ref.parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")
    )
    try:
        return freeze_gate(
            analysis, trials=len(_arm_revisions(records, session_rows)), full_span_irs=irs
        )
    except ValueError as exc:
        return {"passed": False, "reasons": ["freeze_unmeasurable"], "error": str(exc)}


def null_control_seed(key: str, role: str) -> int:
    """A stable 32-bit seed per session (or artifact) and role.

    Stable across processes and runs (``hash`` is not), so a re-run draws the
    same null and its percentile can be compared with the one the ledger
    already holds. The session's ``run_null_control`` tool draws with the
    ``frozen`` role, so its block is the one the freeze would have drawn.
    """

    digest = hashlib.blake2b(f"{key}:{role}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _recorded_steps(records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row
        for record in research_records(records)
        for row in (record.get("steps") or ())
        if isinstance(row, Mapping)
    ]


def _arm_revisions(
    records: Sequence[Mapping[str, object]], session_rows: Sequence[Mapping[str, object]]
) -> set[str]:
    return {
        str(row["revision_id"]) for row in (*_recorded_steps(records), *session_rows)
    }


def _neutralized(result_ref: str) -> dict[str, object] | None:
    # Every successful replay writes its style sidecar, so a missing or
    # unreadable one raises: dropping the row would silently narrow the
    # freeze gate's IR dispersion. Only an unmeasurable span is None.
    path = Path(result_ref).parent / STYLE_ARTIFACT_NAME
    analysis = json.loads(path.read_text(encoding="utf-8"))
    try:
        return neutralized_statistics(analysis)
    except ValueError:
        return None


def _finite_ir(block: object) -> bool:
    value = block.get("information_ratio") if isinstance(block, Mapping) else None
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


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


def _months_before(end: str, months: int) -> str:
    """The first day of the ``months``-long window that ends on ``end``."""

    stamp = pd.Timestamp(end) - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    return stamp.strftime("%Y%m%d")


def _keep_frozen_artifact_ids(
    records: Sequence[Mapping[str, object]],
    extra_id: str | None = None,
) -> tuple[str, ...]:
    """The arm's frozen artifact, any artifact an integrity row names, and the
    freeze now being recorded."""

    keep = {str(extra_id)} if extra_id else set()
    frozen = frozen_record(records)
    if frozen is not None:
        keep.add(str(frozen["frozen"]["artifact_id"]))  # type: ignore[index]
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
        "max_replay_years": config.max_replay_years,
        "max_llm_calls": config.max_llm_calls,
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
            limits[name] = float(value) if name == "deadline_seconds" else int(value)
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
    "RollingExperimentPipeline",
    "freeze_gate_for",
    "null_control_seed",
    "research_step_record",
]
