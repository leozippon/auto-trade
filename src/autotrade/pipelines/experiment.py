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
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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
from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR, window_activity
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.environment.runtime import agent_trace_path, chmod_tree, utc_now_iso
from autotrade.environment.step_tree import node_handle
from autotrade.environment.strategy import NLQuery
from autotrade.environment.strategy_loader import validate_strategy_package

from .agent_inbox import expire_experiment_session_inbox
from .calendar import FULL_SPAN, Slot
from .config import (
    AcceptanceRules,
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
    LINEAGE_RECORD_TYPE,
    LINEAGE_SERIES_NAME,
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
    lineage_record,
    research_over,
    research_records,
)
from .pit_views_seed import FORWARD_PHASE, RESEARCH_PHASE
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
    UNMEASURABLE,
    effective_trials,
    forward_mde,
    forward_slice,
    freeze_gate,
    graduation_verdict,
    heldout_slice,
    information_ratio_bar,
    judge,
    neutralized_statistics,
    stamps,
    trial_correlation,
    trial_family_statistics,
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
                        nl_query=self.nl_query,
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
        if self.config.lineage_arms and lineage_record(records) is None:
            # The console extracted the lineage at creation; its ledger record
            # is written here, after the worker pinned the research release.
            self.ledger.append(lineage_ledger_record(self.config.experiment_dir))
            records = self.ledger.read()
        recorded = lineage_record(records)
        recorded_arms = tuple(recorded["arms"]) if recorded is not None else ()  # type: ignore[arg-type]
        if recorded_arms != self.config.lineage_arms:
            # The gate reads the lineage the ledger holds; one the params name
            # but the record does not would silently drop from the family.
            raise RuntimeError(
                f"lineage_arms {list(self.config.lineage_arms)} do not match the "
                f"lineage the ledger records ({list(recorded_arms)})"
            )
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
                    window_months=self.config.window_months,
                    max_replay_years=int(budgets["max_replay_years"]),
                    max_llm_calls=int(budgets["max_llm_calls"]),
                    deadline_seconds=budgets["deadline_seconds"],
                    deadline_grace_seconds=budgets["deadline_grace_seconds"],
                    benchmark_index=self.config.benchmark_index,
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
            # Every row the ledger records names the bytes it replayed.
            step_rows = fingerprinted(
                self.config.experiment_dir,
                [research_step_record(step) for step in session.steps],
            )
            gate: dict[str, object] | None = None
            frozen: dict[str, object] | None = None
            arm_end: dict[str, object] | None = None
            if session.outcome == "freeze":
                by_step = {str(row["step_id"]): row for row in step_rows}
                missing = [
                    step_id
                    for step_id in (session.node_id, *session.seed_replicates)
                    if step_id not in by_step
                ]
                if missing:
                    raise RuntimeError(
                        f"the nominated node or seed replicate {missing[0]!r} is not a "
                        "Step of this session"
                    )
                nominee = by_step[str(session.node_id)]
                gate = freeze_gate_for(
                    records,
                    step_rows,
                    nominee,
                    experiment_dir=self.config.experiment_dir,
                    hard_reasons=self.config.acceptance.evaluate(dict(nominee["summary"])),
                    acceptance=self.config.acceptance,
                    years=[
                        (slot.start, slot.end)
                        for slot in self.config.geometry.research_years
                    ],
                    seed_replicates=[by_step[step_id] for step_id in session.seed_replicates],
                )
                if gate["passed"]:
                    _publish_progress(progress, "freezing", run_id=run_id)
                    steps = {step.step_id: step for step in session.steps}
                    frozen = self._freeze(
                        steps[str(nominee["step_id"])],
                        gate=gate,
                        run_id=run_id,
                        null_controls=session.null_controls,
                        seed_replicates=[steps[step_id] for step_id in session.seed_replicates],
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
                "trials_to_date": trial_family(
                    fingerprinted(
                        self.config.experiment_dir, [*_recorded_steps(records), *step_rows]
                    )
                )["trials"],
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
                "attempts": session.attempt,
                "budget_used": session.budget_used.to_record(),
                **_session_timing(context, run_started),
            }
            self.ledger.append(record)
            wrote_ledger_record = True
            expire_experiment_session_inbox(
                self.config.experiment_dir, RESEARCH_SESSION_KEY, expired_by=run_id
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

        The seed replicates the gate accepted are frozen beside it, each as an
        artifact of its own, so the forward stage replays bytes no later
        change can reach; ``seed_replicates`` in the block names them, absent
        when there are none.
        """

        def freeze(step: StepResult) -> tuple[str, Path, Path | None]:
            artifact_id = f"strategy_{RESEARCH_SESSION_KEY}_{uuid.uuid4().hex[:12]}"
            stored = self.artifacts.freeze_revision(
                step.revision_id,
                artifact_id=artifact_id,
                experiment_id=self.config.experiment_id,
                epoch_id=RESEARCH_STAGE,
                fold_id=RESEARCH_SESSION_KEY,
                run_id=run_id,
                step_id=step.step_id,
            )
            models = Path(stored.model_path) if stored.model_path is not None else None
            return artifact_id, Path(stored.path), models

        artifact_id, output, models = freeze(nominee)
        ratios = {
            str(entry["step_id"]): entry.get("information_ratio")
            for entry in (gate.get("seed_replicates") or {}).get("replicates") or ()  # type: ignore[union-attr]
        }
        replicates = []
        for step in seed_replicates:
            replicate_id, replicate_output, replicate_models = freeze(step)
            replicates.append(
                {
                    "artifact_id": replicate_id,
                    "output_path": str(replicate_output),
                    "models_path": (
                        str(replicate_models)
                        if replicate_models is not None and replicate_models.is_dir()
                        else None
                    ),
                    "source_step_id": step.step_id,
                    "revision_id": step.revision_id,
                    "research_result_ref": step.validation.result_ref,
                    "information_ratio": ratios[step.step_id],
                }
            )
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
            # Which series the figures below are of: ``active`` is the
            # nominee's return minus its zero-skill panel composite.
            "series": gate["series"],
            "mandate": gate["mandate"],
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
            **({"seed_replicates": replicates} if replicates else {}),
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

        The seed replicates the freeze registered are then replayed one after
        another exactly as the book was (same span, warm-up, refits, Broker,
        panel and style analysis) and judged with it (:meth:`_judge`). A
        replicate whose replay fails for any reason, its own strategy's
        included, fails the attempt: the seed mean is never read without it.
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
            base = {
                "record_type": "forward",
                **{key: attempt[key] for key in ("experiment_id", "epoch_id", "fold_id", "run_id")},
                "session_key": FORWARD_SESSION_KEY,
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
                    "forward_decision": forward_bundle.snapshot_id,
                    "heldout_decision": heldout_bundle.snapshot_id,
                },
            }

            def replay(
                item: FrozenArtifact,
            ) -> tuple[EvaluationResult | None, BaseException | None]:
                """One guarded replay of a frozen artifact over the span."""

                nonlocal wrote_ledger_record
                result, error, changed, restore_error = _run_guarded_evaluation(
                    self.evaluator,
                    span.request(
                        _frozen_revision(item),
                        schedule=self.config.schedule,
                        broker_profile=self.config.broker_profile,
                    ),
                    item,
                )
                if changed:
                    self.ledger.append(
                        {
                            **base,
                            "artifact_id": item.artifact_id,
                            "status": "integrity_failure",
                            "state_changed_during_test": True,
                            "result_ref": result.result_ref if result is not None else None,
                            "error": _error_text(error) if error is not None else None,
                        }
                    )
                    # The integrity row is this run's record: the fail-fast
                    # below must not also log a failed attempt.
                    wrote_ledger_record = True
                    if restore_error is not None:
                        raise FrozenArtifactRestoreFailed(
                            "strategy or model artifacts changed during the forward replay "
                            f"and restoring the pre-evaluation trees failed: {restore_error}"
                        ) from restore_error
                    raise FrozenArtifactMutated(
                        "strategy or model artifacts changed during the forward replay"
                    ) from error
                return result, error

            _publish_progress(progress, "forward_replay", run_id=run_id)
            result, error = replay(artifact)
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
                replays: list[tuple[FrozenArtifact, EvaluationResult]] = []
                for index, item in enumerate(replicates, start=1):
                    _publish_progress(
                        progress,
                        "forward_replay",
                        run_id=run_id,
                        seed_replicate=f"{index}/{len(replicates)}",
                    )
                    replicate_result, replicate_error = replay(item)
                    if replicate_error is not None:
                        raise RuntimeError(
                            f"seed replicate {item.source_step_id} ({item.artifact_id}) did "
                            "not complete its forward replay, and the seed mean is never "
                            f"read without it: {_error_text(replicate_error)}"
                        ) from replicate_error
                    assert replicate_result is not None
                    replays.append((item, replicate_result))
                _publish_progress(progress, "verdict", run_id=run_id)
                record = {
                    **base,
                    **self._judge(
                        result,
                        artifact,
                        forward=forward,
                        heldout=heldout,
                        seed_replicates=replays,
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
        seed_replicates: Sequence[tuple[FrozenArtifact, EvaluationResult]] = (),
    ) -> dict[str, object]:
        """The two slices of one completed replay and the verdict on them.

        A slice that cannot be measured raises ``ValueError``, which fails the
        attempt: a verdict is never read off a number that was not measured.
        Each seed replicate's completed replay is handed to both slices, named
        by its artifact and source Step: ``slices.<name>.seed_replicates`` and
        ``slices.<name>.seed_mean`` carry its readings and the mean with the
        book, ``slices.forward.plain_excess_lower_bound`` what F8 judges on
        them together, and the record's ``seed_replicates`` its result and
        refits.
        """

        replay, analysis = _replay_and_analysis(result)
        replicates: list[tuple[dict[str, object], dict[str, object]]] = []
        replicate_rows: list[dict[str, object]] = []
        for item, item_result in seed_replicates:
            item_replay, item_analysis = _replay_and_analysis(item_result)
            identity = {"artifact_id": item.artifact_id, "source_step_id": item.source_step_id}
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
            rules=acceptance,
            start=forward.start,
            end=forward.end,
            seed_key=artifact.artifact_id,
            slippage_bps=self.config.broker_profile.slippage_bps,
            turnover=float(forward_activity["turnover"]),  # type: ignore[arg-type]
            round_trips=int(forward_activity["round_trips"]),  # type: ignore[arg-type]
            mean_gross=float(forward_activity["mean_gross"]),  # type: ignore[arg-type]
            seed_replicates=replicates,
        )
        heldout_block = heldout_slice(
            analysis,
            rules=acceptance,
            start=heldout.start,
            end=heldout.end,
            forward_tracking_error=float(forward_block["tracking_error"]),  # type: ignore[arg-type]
            mean_gross=float(heldout_activity["mean_gross"]),  # type: ignore[arg-type]
            seed_replicates=replicates,
        )
        return {
            "status": "ok",
            "error": None,
            "result_ref": result.result_ref,
            "slices": {
                "forward": {**forward_block, "activity": forward_activity},
                "heldout": {**heldout_block, "activity": heldout_activity},
            },
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
                seed=null_control_seed(artifact.artifact_id, "forward"),
                step=(forward.start, forward.end),
            ),
            "verdict": graduation_verdict(forward=forward_block, heldout=heldout_block),
        }

    def _frozen_artifact(self, block: object) -> FrozenArtifact:
        """The frozen artifact a frozen block (or one of its seed replicates)
        names, validated by its store."""

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
        seed: int,
        step: tuple[str, str] | None = None,
    ) -> dict[str, object] | None:
        """Rank one completed result against random-name copies of its own trades.

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


def trial_fields(step: StepResult) -> dict[str, object]:
    """The fields of one Step the trial family reads (:func:`trial_family`)."""

    return {
        "step_id": step.step_id,
        "revision_id": step.revision_id,
        "fingerprint": step.fingerprint,
        "control": step.control,
        "batch_id": step.batch_id,
        "offline_trials": step.offline_trials,
    }


def research_step_record(step: StepResult) -> dict[str, object]:
    """One completed Validation as the ledger's ``steps[]`` row.

    ``neutralized`` carries the graded series' neutralised excess, residual
    tracking error and IR over the span, from the replay's own style sidecar
    (``None`` when they cannot be measured): the figures the freeze gate
    counts. ``series`` inside it says whether that is the active series (the
    replay carries a zero-skill panel) or the strategy's own. ``fingerprint``,
    ``control``, ``batch_id`` and ``offline_trials`` are what the trial family
    reads.
    """

    return {
        **trial_fields(step),
        "span": step.span,
        "summary": step.validation.summary,
        "validation_result_ref": step.validation.result_ref,
        "neutralized": neutralized(step.validation.result_ref),
    }


def trial_family(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The trials the freeze gate deflates over, from the arm's Validation rows.

    A trial is a distinct strategy validated anywhere in the arm, on any span,
    in any attempt: the bytes its revision holds (``fingerprint``, which every
    row must carry, :func:`fingerprinted`), not the revision, so validating
    the same bytes again -- a probe's finalist on the full span -- is the same
    trial. Bytes whose every Validation was registered as a control are left
    out: a control is a comparison leg, not one of the configurations the
    nominee was selected among (and can never be nominated). Bytes validated
    both as a control and as a candidate are a trial: the candidate
    registration is the selection. Each trial's series in ρ̄ is that of its
    longest validated span (the Validation with the most measured days, the
    first of equals); ``representatives`` are those Validations in revision
    order.

    The candidate configurations a batch declared as screened offline and not
    submitted (``offline_trials``; a submitted one is a host trial, each
    configuration is declared once) are trials too, counted once per batch. A
    batch whose candidates all failed leaves no row, so its declaration is not
    counted and ``batch_validate`` asks for it again in the next batch. A row
    recorded before batches declared them carries no ``offline_trials`` and
    reads as 0; ``undeclared_offline_validations`` counts such rows so the
    record says so. ``trials`` is M, the host trials plus the offline ones;
    ``controls`` counts the control-only strategies.
    """

    strategies: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        fingerprint = row.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"Validation {row.get('step_id')} carries no artifact fingerprint, "
                "so its trial cannot be identified"
            )
        strategies.setdefault(fingerprint, []).append(row)
    trials = [
        validations
        for validations in strategies.values()
        if any(row.get("control") is not True for row in validations)
    ]
    representatives = sorted(
        (max(validations, key=_measured_days) for validations in trials),
        key=lambda row: str(row["revision_id"]),
    )
    declared: dict[str, int] = {}
    undeclared = 0
    for row in rows:
        count = row.get("offline_trials")
        if count is None:
            undeclared += 1
        else:
            declared[str(row.get("batch_id") or row["step_id"])] = int(count)  # type: ignore[call-overload]
    offline = sum(declared.values())
    return {
        "trials": len(representatives) + offline,
        "representatives": representatives,
        "controls": len(strategies) - len(trials),
        "offline_trials": offline,
        "undeclared_offline_validations": undeclared,
    }


def _measured_days(row: Mapping[str, object]) -> int:
    """How many days one Validation row's graded series measured (0 when none)."""

    neutral = row.get("neutralized")
    days = neutral.get("days") if isinstance(neutral, Mapping) else None
    return days if isinstance(days, int) else 0


def fingerprinted(
    experiment_dir: str | Path, rows: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    """``rows`` each with the artifact fingerprint :func:`trial_family` keys on:
    the one a row carries, else the one the arm's revision store recorded for
    its revision (``session_resume.revision_fingerprint``) -- a ledger row
    written before rows carried it, or a Step of the host's deterministic
    baseline."""

    return [
        row
        if row.get("fingerprint")
        else {**row, "fingerprint": revision_fingerprint(experiment_dir, str(row["revision_id"]))}
        for row in rows
    ]


def _arm_trials(
    records: Sequence[Mapping[str, object]],
    session_rows: Sequence[Mapping[str, object]],
    experiment_dir: str | Path,
) -> tuple[
    list[Mapping[str, object]], dict[str, object], tuple[list[str], int, list[dict[str, float]]]
]:
    """What an arm's freeze gate deflates over: its Validation rows, earlier
    sessions' recorded Steps and this session's alike, fingerprinted from the
    revision store of the arm in ``experiment_dir`` where a row does not carry
    it; their :func:`trial_family`; and the lineage the ledger records
    (:func:`recorded_lineage`)."""

    rows = fingerprinted(experiment_dir, [*_recorded_steps(records), *session_rows])
    return rows, trial_family(rows), recorded_lineage(records)


def freeze_gate_for(
    records: Sequence[Mapping[str, object]],
    session_rows: Sequence[Mapping[str, object]],
    nominee: Mapping[str, object],
    *,
    experiment_dir: str | Path,
    acceptance: AcceptanceRules,
    hard_reasons: Sequence[str] = (),
    years: Sequence[tuple[str, str]] = (),
    seed_replicates: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """The freeze gate of one nominated Step against the whole arm
    (docs/pipeline-design.md).

    The trial family is the arm's (:func:`_arm_trials`); ρ̄ is read off each
    trial's representative sidecar and the lineage's extracted series. The
    count of full-span validations takes every measurable one, controls
    included. A nominee that did not replay the full research period, was
    registered as a control, fails a hard nomination rule, or whose statistics
    cannot be measured does not pass. ``acceptance`` is the arm's rules,
    ``hard_reasons`` what their ``evaluate`` refuses the nominee for and
    ``years`` the research years; the console, which reads only the deflated
    Sharpe of an arm's best node, passes the default rules and leaves the
    other two out. Where the rules hold
    ``require_seed_replicates`` the measured gate also judges the nominee's
    ``seed_replicates`` (rows of this session; :func:`_seed_replicate_gate`);
    naming any under rules that do not hold it is refused.
    """

    if seed_replicates and not acceptance.require_seed_replicates:
        raise ValueError("this arm's acceptance rules hold no seed-replicate condition")
    reasons = [*hard_reasons, *judge("registration", nominee, {})]
    if reasons:
        return {"passed": False, "reasons": reasons}
    rows, family, (lineage_arms, lineage_trials, lineage_series) = _arm_trials(
        records, session_rows, experiment_dir
    )
    representatives: list[Mapping[str, object]] = family["representatives"]  # type: ignore[assignment]
    summary = nominee.get("summary")
    try:
        gate = freeze_gate(
            _style_analysis(nominee),
            rules=acceptance,
            trials=len(representatives),
            offline_trials=family["offline_trials"],  # type: ignore[arg-type]
            trial_analyses=[_style_analysis(row) for row in representatives],
            lineage_trials=lineage_trials,
            lineage_series=lineage_series,
            full_span_validations=sum(
                1
                for row in rows
                if row.get("span") == FULL_SPAN and _finite_ir(row.get("neutralized"))
            ),
            years=years,
            summary=summary if isinstance(summary, Mapping) else None,
        )
    except ValueError as exc:
        return {"passed": False, "reasons": [UNMEASURABLE], "error": str(exc)}
    gate["deflated_sharpe"] = {
        **gate["deflated_sharpe"],  # type: ignore[dict-item]
        "controls": family["controls"],
        "undeclared_offline_validations": family["undeclared_offline_validations"],
        "lineage_arms": lineage_arms,
    }
    if acceptance.require_seed_replicates:
        gate = _seed_replicate_gate(
            gate,
            fingerprinted(experiment_dir, [nominee])[0],
            fingerprinted(experiment_dir, seed_replicates),
            experiment_dir=experiment_dir,
            rules=acceptance,
        )
    return gate


# The one line a seed replicate changes: an integer assignment, with an
# optional ``int`` annotation and a trailing comment, to a seed name --
# ``SEED``, ``SEED_BASE`` or one ending in ``_SEED`` or ``_SEED_BASE``, any
# case (``SEED_BASE = 2000``). A count of seeds such as ``SEEDS_PER_HEAD``
# changes the strategy, not its draw, and is not one.
_SEED_LINE = re.compile(r"\s*([A-Za-z_]\w*)\s*(?::\s*int\s*)?=\s*([+-]?\d+)\s*(?:#.*)?")
_SEED_NAME = re.compile(r"(?:\w*_)?seed(?:_base)?", re.IGNORECASE)


def seed_change(experiment_dir: str | Path, nominee_revision: str, replicate_revision: str) -> str:
    """The one seed line by which a replicate's revision differs from the nominee's.

    Read off the two revisions' manifests (every ``output/`` and ``models/``
    file by SHA-256): exactly one file may differ, a ``.py`` file of the
    strategy package, and in exactly one line, which on both sides assigns an
    integer to the same seed name (``_SEED_NAME``), with two
    different values. That is how a strategy changes its training seed and
    nothing else -- the packs set it in one knob line -- and it is checked
    on bytes, so a replicate cannot differ in anything a seed does not
    explain. Returns ``"<path>: <line>"``; ``ValueError`` says what else
    differs.
    """

    root = Path(experiment_dir) / REVISIONS_DIR
    nominee = _revision_files(root / nominee_revision)
    replicate = _revision_files(root / replicate_revision)
    changed = sorted(
        path for path in {*nominee, *replicate} if nominee.get(path) != replicate.get(path)
    )
    if not changed:
        raise ValueError("holds the nominee's own bytes")
    path = changed[0]
    if len(changed) > 1 or path not in nominee or path not in replicate:
        raise ValueError(
            f"differs from the nominee in {len(changed)} file(s) "
            f"({', '.join(changed[:5])}), not in one seed line"
        )
    if not (path.startswith("output/") and path.endswith(".py")):
        raise ValueError(f"differs from the nominee in {path}, which is not strategy code")
    before = (root / nominee_revision / path).read_text(encoding="utf-8").splitlines()
    after = (root / replicate_revision / path).read_text(encoding="utf-8").splitlines()
    lines = [index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
    if len(before) != len(after) or len(lines) != 1:
        raise ValueError(
            f"differs from the nominee in more than one line of {path}, not in one seed line"
        )
    line = lines[0]
    old, new = _SEED_LINE.fullmatch(before[line]), _SEED_LINE.fullmatch(after[line])
    if (
        old is None
        or new is None
        or old[1] != new[1]
        or _SEED_NAME.fullmatch(old[1]) is None
        or int(old[2]) == int(new[2])
    ):
        raise ValueError(
            f"changes line {line + 1} of {path} to {after[line].strip()!r}, which does not "
            "give a seed name (SEED, SEED_BASE, or one ending in _SEED or _SEED_BASE) "
            "another integer value"
        )
    return f"{path.removeprefix('output/')}: {after[line].strip()}"


def _revision_files(directory: Path) -> dict[str, str]:
    """``path -> sha256`` of every file one recorded revision holds."""

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return {str(entry["path"]): str(entry["sha256"]) for entry in manifest["files"]}


def _seed_replicate_gate(
    gate: Mapping[str, object],
    nominee: Mapping[str, object],
    replicates: Sequence[Mapping[str, object]],
    *,
    experiment_dir: str | Path,
    rules: AcceptanceRules,
) -> dict[str, object]:
    """``gate`` with the seed-replicate conditions of an arm that holds them.

    A strategy that trains a model (its ``main.py`` defines ``fit``, the
    frozen block's ``fit_plan``) is judged on its training seeds together. A
    seed replicate is a full-span, non-control row of the session whose bytes
    are the nominee's but for one seed line (:func:`seed_change`) and differ
    from every other registered row's, with a measured IR on the nominee's
    graded series. The block ``seed_replicates`` records what was read, a
    ``problem`` per refused replicate, and the ``seeds`` conditions of
    ``verdict.CONDITIONS`` are judged on it: a named replicate that is not
    one, a nominee that trains a model and names none, and a mean active IR
    over the nominee and its replicates below the nominee's own
    ``information_ratio_bar``. Every other condition stays the nominee's own.
    The thresholds name ``require_seed_replicates`` whether or not the nominee
    had anything to replicate: the stamp says the arm holds the rule.
    """

    main = Path(experiment_dir) / REVISIONS_DIR / str(nominee["revision_id"]) / "output" / "main.py"
    trains = validate_strategy_package(main) is not None
    seen = {str(nominee["fingerprint"]): "the nominee"}
    entries: list[dict[str, object]] = []
    for row in replicates:
        step_id = str(row["step_id"])
        neutral = row.get("neutralized")
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
            if not _finite_ir(neutral) or neutral.get("series") != gate.get("series"):  # type: ignore[union-attr]
                raise ValueError(f"has no measured {gate.get('series')} information ratio")
            entry["seed_line"] = seed_change(
                experiment_dir, str(nominee["revision_id"]), str(row["revision_id"])
            )
            entry["information_ratio"] = float(neutral["information_ratio"])  # type: ignore[index]
        except ValueError as exc:
            entry["problem"] = str(exc)
        entries.append(entry)
    ratios = [gate.get("information_ratio"), *(entry.get("information_ratio") for entry in entries)]
    measured = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in ratios
    )
    block = {
        "trains_a_model": trains,
        "replicates": entries,
        "mean_information_ratio": sum(ratios) / len(ratios) if entries and measured else None,  # type: ignore[arg-type]
        "information_ratio_bar": gate["deflated_sharpe"].get("information_ratio_bar"),  # type: ignore[union-attr]
    }
    reasons = judge("seeds", block, {}, rules)
    return {
        **gate,
        "passed": bool(gate["passed"]) and not reasons,
        "reasons": [*gate["reasons"], *reasons],  # type: ignore[misc]
        "seed_replicates": block,
        "thresholds": {**gate["thresholds"], **stamps("seeds", rules)},  # type: ignore[dict-item]
    }


def full_span_bar(
    records: Sequence[Mapping[str, object]],
    session_rows: Sequence[Mapping[str, object]],
    *,
    experiment_dir: str | Path,
    research_years: int,
    acceptance: AcceptanceRules,
) -> dict[str, object]:
    """The trial count, effective count and active-IR bar a full-span,
    non-control nominee faces now, for the rows :func:`freeze_gate_for`
    refuses before measuring (a sub-span, a control, a hard rule broken).

    The family is the one the gate deflates over (:func:`_arm_trials`); it
    holds at least the nominee itself. The bar is the gate's
    ``information_ratio_bar`` at that N_eff over a full research period: the
    days the arm's full-span Validations measured, or ``TRADING_DAYS_PER_YEAR``
    per research year before there is one (8 years: 1,952 against the 1,940 a
    full span measures, a bar about 0.3 % lower). The gate itself is unchanged.
    """

    rows, family, (_arms, lineage_trials, lineage_series) = _arm_trials(
        records, session_rows, experiment_dir
    )
    representatives: list[Mapping[str, object]] = family["representatives"]  # type: ignore[assignment]
    statistics = trial_family_statistics(
        trials=max(len(representatives), 1),
        offline_trials=family["offline_trials"],  # type: ignore[arg-type]
        trial_analyses=[_style_analysis(row) for row in representatives],
        lineage_trials=lineage_trials,
        lineage_series=lineage_series,
    )
    days = max(
        (_measured_days(row) for row in rows if row.get("span") == FULL_SPAN),
        default=0,
    ) or TRADING_DAYS_PER_YEAR * research_years
    return {
        "trials": statistics["trials"],
        "effective_trials": statistics["effective_trials"],
        "information_ratio_bar": information_ratio_bar(
            statistics["effective_trials"],  # type: ignore[arg-type]
            days,
            acceptance.min_dsr_probability,
        ),
    }


def lineage_summary(extraction: Mapping[str, object]) -> dict[str, object]:
    """A lineage's own trial count and what it counts as independently, from
    what ``lineage.extract_lineage`` read: the ledger record's figures, and
    what a round's ``--dry-run`` prints."""

    arms: Sequence[Mapping[str, object]] = extraction["arms"]  # type: ignore[assignment]
    host = sum(int(arm["host_trials"]) for arm in arms)  # type: ignore[call-overload]
    offline = sum(int(arm["offline_trials"]) for arm in arms)  # type: ignore[call-overload]
    correlation, pairs = trial_correlation(
        (),
        [
            {str(day): float(value) for day, value in item["daily"]}  # type: ignore[index]
            for item in extraction["series"]  # type: ignore[attr-defined]
        ],
    )
    return {
        "arms": [str(arm["experiment_id"]) for arm in arms],
        "trials": host + offline,
        "host_trials": host,
        "offline_trials": offline,
        "controls": sum(int(arm["controls"]) for arm in arms),  # type: ignore[call-overload]
        "trial_correlation": correlation,
        "trial_correlation_pairs": pairs,
        "effective_trials": effective_trials(host + offline, correlation),
    }


def lineage_ledger_record(experiment_dir: Path) -> dict[str, object]:
    """The ``lineage`` ledger record of an arm created with ``lineage_arms``,
    from the series file the console wrote beside its ledger at creation."""

    path = Path(experiment_dir) / "ledgers" / LINEAGE_SERIES_NAME
    if not path.is_file():
        raise RuntimeError(
            f"the arm names lineage_arms but its creation wrote no lineage series ({path})"
        )
    return {
        "record_type": LINEAGE_RECORD_TYPE,
        "experiment_id": Path(experiment_dir).name,
        "epoch_id": RESEARCH_STAGE,
        "fold_id": RESEARCH_SESSION_KEY,
        # No run produced it: it carries what creation read from other arms.
        "run_id": LINEAGE_RECORD_TYPE,
        **lineage_summary(json.loads(path.read_text(encoding="utf-8"))),
        "series_ref": str(path),
        "recorded_at": utc_now_iso(),
    }


def recorded_lineage(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[str], int, list[dict[str, float]]]:
    """The lineage the ledger records: its arms, the trials they add and one
    daily series per measurable lineage revision (``pipelines/lineage.py``).

    Read from this arm's own files only -- the ledger record and the series
    file it names, both written at creation -- never from the lineage arms.
    ``([], 0, [])`` for an arm created without one.
    """

    record = lineage_record(records)
    if record is None:
        return [], 0, []
    payload = json.loads(Path(str(record["series_ref"])).read_text(encoding="utf-8"))
    return (
        [str(arm) for arm in record["arms"]],  # type: ignore[union-attr]
        int(record["trials"]),  # type: ignore[call-overload]
        [
            {str(date): float(value) for date, value in item["daily"]}
            for item in payload["series"]
        ],
    )


def _style_analysis(row: Mapping[str, object]) -> dict[str, object]:
    """The style sidecar beside one Validation row's result."""

    result_ref = Path(str(row["validation_result_ref"]))
    return json.loads((result_ref.parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8"))


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


def neutralized(result_ref: str) -> dict[str, object] | None:
    """One validation's neutralised excess, tracking error and IR.

    Every successful replay writes its style sidecar, so a missing or
    unreadable one raises: dropping the row would silently narrow the freeze
    gate's IR dispersion. Only an unmeasurable span is None. The console's
    live step listing reads its rows through this same function.
    """

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


def _replay_and_analysis(result: EvaluationResult) -> tuple[dict[str, object], dict[str, object]]:
    """One completed replay's result and the style sidecar beside it."""

    path = Path(result.result_ref)
    return (
        json.loads(path.read_text(encoding="utf-8")),
        json.loads((path.parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")),
    )


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
    """The arm's frozen artifact and its seed replicates, any artifact an
    integrity row names, and the freeze now being recorded."""

    keep = {str(extra_id)} if extra_id else set()
    frozen = frozen_record(records)
    if frozen is not None:
        block: Mapping[str, object] = frozen["frozen"]  # type: ignore[assignment]
        keep.add(str(block["artifact_id"]))
        keep.update(
            str(item["artifact_id"])
            for item in block.get("seed_replicates") or ()  # type: ignore[union-attr]
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
    "fingerprinted",
    "freeze_gate_for",
    "full_span_bar",
    "lineage_ledger_record",
    "lineage_summary",
    "neutralized",
    "null_control_seed",
    "recorded_lineage",
    "research_step_record",
    "seed_change",
    "trial_family",
    "trial_fields",
]
