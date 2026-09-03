"""Experiment pipeline: Step/Fold/Epoch/Held-out orchestration.

docs/pipeline-design.md. The Pipeline schedules Data, Environment, and Agent
in time order, freezes inputs/outputs at each boundary, and writes the single
experiment ledger. It implements no investment logic and never rewrites
strategy content; it only accepts, freezes, falls back, and records.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from autotrade.agent.runner import AgentSessionDeadlineExceeded
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
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay import (
    ContextDataProvider,
    ExecutionPriceProvider,
    ReplayResult,
    run_daily_replay,
)
from autotrade.environment.replay.style import daily_returns_from_curve
from autotrade.environment.runtime import agent_trace_path, chmod_tree
from autotrade.environment.strategy import NLQuery

from .agent_inbox import expire_experiment_session_inbox
from .agent_views import (
    agent_visible_ledger_record as _agent_visible_ledger_record,
)
from .agent_views import (
    compact_fold_history as _compact_fold_history,
)
from .agent_views import (
    vs_parent_metrics as _vs_parent_metrics,
)
from .config import (
    ArtifactRevision,
    ArtifactStore,
    EvaluationBackend,
    EvaluationRequest,
    EvaluationResult,
    FoldDeveloper,
    FoldOutcome,
    FoldSessionRequest,
    FoldSessionResult,
    FrozenArtifact,
    MetaLearner,
    MetaSessionResult,
    RollingExperimentConfig,
    SnapshotBundle,
    SnapshotProvider,
    StepResult,
    StrategyExperimentConfig,
    fold_session_deadline_seconds,
)
from .folds import FoldSpec, heldout_periods
from .hitl_state import fold_session_key
from .ledger import (
    ExperimentLedger,
    FrozenArtifactMutated,
    FrozenArtifactRestoreFailed,
    RunMarkers,
    assert_no_frozen_artifact_mutation,
    deflated_sharpe,
    is_frozen_artifact_mutation,
    latest_fold_records,
    walk_forward_transitions,
)
from .meta_inputs import (
    AgentTraceFullSidecar,
    build_meta_fold_review_bundle,
    select_meta_review_folds,
)
from .meta_schedule import (
    meta_learning_id,
    meta_record_id,
    meta_session_key,
)
from .prior import PRIOR_MAX_CHARS, ExperimentPriorStore
from .skills import (
    ExperimentSkillsStore,
    SkillsPublication,
    SkillsSnapshot,
    latest_skills_snapshot,
    resolve_collected_skills_source,
)

# A per-session deadline override may raise the fold deadline above the
# configured maximum, bounded by this absolute ceiling in minutes: twice the
# default Fold budget (config.max_fold_minutes), enough headroom for one slow
# session without letting it run unattended for days.
_MAX_DEADLINE_OVERRIDE_MINUTES = 1440


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

    def run(self, daily: pd.DataFrame | str | Path) -> ReplayResult:
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
    """Step → Fold → Epoch → Held-out orchestration over injected backends."""

    def __init__(
        self,
        config: RollingExperimentConfig,
        *,
        snapshots: SnapshotProvider,
        artifacts: ArtifactStore,
        evaluator: EvaluationBackend,
        developer: FoldDeveloper,
        meta_learner: MetaLearner | None = None,
        ledger: ExperimentLedger | None = None,
    ) -> None:
        self.config = config
        self.ref_store = AgentRefStore(config.experiment_dir)
        self.snapshots = snapshots
        self.artifacts = artifacts
        self.evaluator = evaluator
        self.developer = developer
        self.meta_learner = meta_learner
        self.ledger = ledger or ExperimentLedger(config.ledger_path)
        self.run_markers = RunMarkers(config.experiment_dir)

    def run_fold(
        self,
        epoch_id: str,
        fold: FoldSpec,
        *,
        parent: FrozenArtifact | None,
        prior: str = "",
        session_context: dict[str, object] | None = None,
    ) -> FoldOutcome:
        assert_no_frozen_artifact_mutation(self.ledger.read())
        run_started = time.monotonic()
        run_id = f"run_{uuid.uuid4().hex}"
        context = dict(session_context or {})
        progress = _optional_hook(context.get("progress_hook"), "progress_hook")
        budgets = _session_budgets(self.config, context.get("resource_override"))
        current_skills = latest_skills_snapshot(
            self.ledger.read(), experiment_dir=self.config.experiment_dir
        )
        retained_artifact_id = parent.artifact_id if parent is not None else None
        wrote_business_record = False
        attempt = {
            "experiment_id": self.config.experiment_id,
            "epoch_id": epoch_id,
            "fold_id": fold.fold_id,
            "run_id": run_id,
            "session_key": fold_session_key(epoch_id, fold.fold_id),
            "phase": "fold",
        }
        # Evidence for a run that never gets to run its own except branch.
        self.run_markers.begin(attempt)
        try:
            _publish_progress(
                progress, "pit_snapshot", run_id=run_id, phase="validation"
            )
            valid_snapshot = self.snapshots.prepare(
                fold=fold,
                phase="valid",
                start=fold.validation_start,
                end=fold.validation_end,
                decision_time=fold.valid_decision_time,
            )
            control, control_error = self._parent_control(
                parent, fold, valid_snapshot, progress=progress, run_id=run_id
            )
            try:
                session = self.developer(
                    FoldSessionRequest(
                        experiment_id=self.config.experiment_id,
                        epoch_id=epoch_id,
                        fold=fold,
                        run_id=run_id,
                        parent=parent,
                        prior=prior,
                        snapshot=valid_snapshot,
                        max_steps=budgets["max_steps"],
                        max_backtests=budgets["max_backtests"],
                        max_llm_calls=budgets["max_llm_calls"],
                        deadline_seconds=budgets["deadline_seconds"],
                        deadline_grace_seconds=budgets["deadline_grace_seconds"],
                        directive=str(context.get("directive") or ""),
                        prompt_override=str(context.get("prompt_override") or ""),
                        sandbox_gpu_count=_optional_gpu_count(
                            context.get("sandbox_gpu_count")
                        ),
                        fold_period=self.config.fold_period,
                        test_stage=self.config.test_stage,
                        parent_control=control,
                        epoch_index=_epoch_index(epoch_id),
                        phase=(
                            "convergence"
                            if _epoch_index(epoch_id)
                            >= self.config.convergence_start_epoch
                            else "exploration"
                        ),
                        acceptance_rules=self.config.acceptance.to_record(),
                        modification_constraints=replace(
                            self.config.step_constraints,
                            is_initial_artifact=parent is None,
                        ).for_epoch(_epoch_index(epoch_id)),
                        snapshot_config=_snapshot_config_record(self.snapshots),
                        record_failed_attempts=self.config.record_failed_attempts,
                        nl_failure_policy=self.config.nl_failure_policy,
                        finalize_before_deadline_seconds=self.config.finalize_before_deadline_seconds,
                        step_gate_hook=_optional_hook(
                            context.get("step_gate_hook"), "step_gate_hook"
                        ),
                        user_question_hook=_optional_hook(
                            context.get("user_question_hook"),
                            "user_question_hook",
                        ),
                        progress_hook=progress,
                        session_key=str(
                            context.get("session_key")
                            or fold_session_key(epoch_id, fold.fold_id)
                        ),
                        skills_source_ref=(
                            str(current_skills.root)
                            if current_skills.root is not None
                            else ""
                        ),
                    )
                )
            except AgentSessionDeadlineExceeded as exc:
                # Expected control flow: the session already emitted
                # session_end{deadline_exceeded} after its wrap-up grace.
                # Record a no-candidate fold instead of failing the run; the
                # fallback chain below decides what the next fold inherits.
                session = FoldSessionResult(
                    conversation_id=exc.conversation_id,
                    steps=(),
                    selected_step_id=None,
                    finish_reason="deadline_grace_exhausted",
                )
            # The parent control is the host's Step, never the Agent's.
            if sum(not step.parent_control for step in session.steps) > budgets["max_steps"]:
                raise RuntimeError("Fold developer exceeded the Step budget")
            selected = _select_step(session.steps, session.selected_step_id)
            hard: list[str] = ["no_complete_validation"]
            warnings: list[str] = []
            if selected is not None:
                hard, warnings = self.config.acceptance.evaluate(
                    selected.validation.summary
                )
            if selected is not None and not hard:
                artifact_id = (
                    f"strategy_{epoch_id}_{fold.fold_id}_{uuid.uuid4().hex[:12]}"
                )
                frozen = self.artifacts.freeze_revision(
                    selected.revision_id,
                    artifact_id=artifact_id,
                    experiment_id=self.config.experiment_id,
                    epoch_id=epoch_id,
                    fold_id=fold.fold_id,
                    run_id=run_id,
                    step_id=selected.step_id,
                )
                status = "frozen"
                validation = selected.validation.summary
            elif parent is not None:
                if parent.requires_validation:
                    self._assert_parent_validated_in_fold(
                        parent, session.steps, control=control
                    )
                    parent = replace(parent, requires_validation=False)
                frozen = parent
                status = "no_update" if selected is not None else "no_valid_backtest"
                validation = (
                    selected.validation.summary if selected is not None else None
                )
            else:
                # First fold without an acceptable baseline: never terminate
                # the run. Record baseline_missing (with the rejection
                # reasons when a candidate existed) and continue with the
                # later folds; the run only fails at the end if no fold ever
                # froze an artifact.
                frozen = None
                status = "baseline_missing"
                validation = (
                    selected.validation.summary if selected is not None else None
                )
            if frozen is not None:
                retained_artifact_id = frozen.artifact_id
            test_result_ref: str | None = None
            state_changed_during_test = False
            restore_error: BaseException | None = None
            test_snapshot = None
            test_summary: dict[str, object] | None
            if frozen is None:
                test_summary = {"status": "skipped_no_frozen_artifact"} if fold.has_test else None
            elif not fold.has_test:
                # Single-window development: there is no Test stage. The frozen
                # strategy is judged by the automatic Held-out replay instead.
                test_summary = None
            else:
                assert fold.test_start is not None and fold.test_end is not None
                assert fold.test_decision_time is not None
                _publish_progress(progress, "frozen_test", run_id=run_id)
                test_snapshot = self.snapshots.prepare(
                    fold=fold,
                    phase="frozen_test",
                    start=fold.test_start,
                    end=fold.test_end,
                    decision_time=fold.test_decision_time,
                )
                test_result, test_error, state_changed_during_test, restore_error = (
                    _run_guarded_evaluation(
                        self.evaluator,
                        EvaluationRequest(
                            revision=_frozen_revision(frozen),
                            snapshot=test_snapshot,
                            mode="frozen_test",
                            start=fold.test_start,
                            end=fold.test_end,
                            schedule=self.config.schedule,
                            broker_profile=self.config.broker_profile,
                        ),
                        frozen,
                    )
                )
                if test_result is not None:
                    test_summary = test_result.summary
                    test_result_ref = test_result.result_ref
                else:
                    # Frozen Test is diagnostic only unless the frozen trees
                    # themselves changed: then the ledger records the integrity
                    # failure and the run terminates.
                    test_summary = {
                        "status": "failed",
                        "error": (
                            f"{type(test_error).__name__}: {test_error}"
                            if test_error is not None
                            else "frozen_test_failed"
                        ),
                    }
            _publish_progress(progress, "publishing", run_id=run_id)
            skills = self._publish_or_keep_skills(
                session.skills_source_ref,
                current=current_skills,
                generation_id=f"{epoch_id}_{fold.fold_id}_{run_id}",
                run_id=run_id,
            )
            record = {
                "record_type": "fold",
                "experiment_id": self.config.experiment_id,
                "epoch_id": epoch_id,
                "fold_id": fold.fold_id,
                "run_id": run_id,
                "session_key": fold_session_key(epoch_id, fold.fold_id),
                **fold.to_record(),
                "parent_strategy_artifact_id": parent.artifact_id if parent else None,
                "parent_control": _parent_control_record(
                    parent, control, control_error, session.steps
                ),
                "conversation_id": session.conversation_id,
                "finish_reason": session.finish_reason,
                "early_stop_reason": session.early_stop_reason or None,
                "fold_status": status,
                "accept_reasons": hard,
                "accept_warnings": warnings,
                "selected_step_id": selected.step_id if selected is not None else None,
                "steps": [_step_record(step) for step in session.steps],
                "frozen_strategy_artifact_id": frozen.artifact_id
                if frozen is not None
                else None,
                "frozen_strategy_artifact_path": (
                    str(frozen.path) if frozen is not None else None
                ),
                "validation_result": validation,
                # How the frozen candidate stands against this Fold's own
                # baseline, and how wide a search it won (§2.4).
                "vs_parent": _vs_parent_metrics(
                    validation, control.summary if control is not None else None
                ),
                "selection_statistics": _selection_statistics(
                    session.steps, selected
                ),
                "test_result": test_summary,
                "test_result_ref": test_result_ref,
                "run_manifest_ref": session.run_manifest_ref,
                "skills_ref": skills.skills_ref or None,
                "skills_generation_id": skills.generation_id or None,
                **skills.stats.ledger_fields(),
                "skills_published": skills.published,
                "agent_trace_ref": str(
                    agent_trace_path(self.config.experiment_dir / "artifacts", run_id)
                )
                if agent_trace_path(
                    self.config.experiment_dir / "artifacts", run_id
                ).exists()
                else None,
                # HITL re-run tag and the step-node parent override that started
                # this session: recorded for audit and for the runner's
                # "this rerun request is absorbed" check.
                "rerun_id": str(context.get("rerun_id") or "") or None,
                "parent_override": str(context.get("parent_override") or "") or None,
                "snapshot_ids": {
                    "valid_decision_input": valid_snapshot.snapshot_id,
                    "test_decision_input": (
                        test_snapshot.snapshot_id if test_snapshot is not None else None
                    ),
                },
                **_session_timing(context, run_started),
            }
            if state_changed_during_test:
                record["state_changed_during_test"] = True
            self.ledger.append(record)
            wrote_business_record = True
            expire_experiment_session_inbox(
                self.config.experiment_dir,
                str(record["session_key"]),
                expired_by=run_id,
            )
            if state_changed_during_test:
                if restore_error is not None:
                    raise FrozenArtifactRestoreFailed(
                        "strategy or model artifacts changed during frozen test "
                        "and restoring the pre-evaluation trees failed: "
                        f"{restore_error}"
                    ) from restore_error
                raise FrozenArtifactMutated(
                    "strategy or model artifacts changed during frozen test"
                )
            return FoldOutcome(
                fold.fold_id, run_id, status, frozen, validation, test_summary
            )
        except FrozenArtifactMutated:
            raise
        except Exception as exc:
            if not wrote_business_record:
                self.ledger.append(
                    {
                        **attempt,
                        "record_type": "attempt_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            raise
        finally:
            # Either a business record or an attempt_failed is now durable, so
            # this run is no longer an interrupted one.
            self.run_markers.finish(run_id)
            prune = getattr(self.artifacts, "prune_transient", None)
            if callable(prune):
                prune(
                    keep_frozen_ids=_keep_frozen_artifact_ids(
                        self.ledger.read(),
                        extra_id=retained_artifact_id,
                    )
                )

    def run_heldout(
        self,
        epoch_id: str,
        final: FrozenArtifact,
        trading_days: list[str],
        *,
        replay: bool = False,
    ) -> int:
        assert_no_frozen_artifact_mutation(self.ledger.read())
        count = 0
        # A re-run fold invalidates every earlier Held-out result: they scored a
        # frontier that no longer exists, so the caller replays them against the
        # new one (append-only ledger; consumers read latest-per-label).
        completed = (
            set()
            if replay
            else {
                str(record.get("period"))
                for record in self.ledger.read("heldout")
                if record.get("period") and not is_frozen_artifact_mutation(record)
            }
        )
        # Graduation term (b): the final Epoch's walk-forward transitions
        # (docs/pipeline-design.md §3.3), read once from the ledger and applied
        # to every Held-out period's verdict.
        walk_forward = walk_forward_transitions(
            self.ledger.read("fold"),
            epoch_id=epoch_id,
            test_stage=self.config.test_stage,
        )
        for period in heldout_periods(
            self.config.heldout_first_period,
            self.config.heldout_last_period,
            trading_days,
            period=self.config.fold_period,
            min_region_trade_days=self.config.min_region_trade_days,
        ):
            run_id = f"run_{uuid.uuid4().hex}"
            label = str(period["label"])
            if label in completed:
                continue
            attempt = {
                "experiment_id": self.config.experiment_id,
                "epoch_id": epoch_id,
                "fold_id": f"heldout_{label}",
                "run_id": run_id,
                "session_key": "heldout",
                "phase": "heldout",
            }
            # Evidence for a period that never gets to run its own except branch.
            self.run_markers.begin(attempt)
            wrote_business_record = False
            try:
                snapshot = self.snapshots.prepare(
                    fold=None,
                    phase="heldout",
                    start=str(period["start"]),
                    end=str(period["end"]),
                    decision_time=period["decision_time"],  # type: ignore[arg-type]
                )
                result, heldout_error, state_changed, restore_error = _run_guarded_evaluation(
                    self.evaluator,
                    EvaluationRequest(
                        revision=_frozen_revision(final),
                        snapshot=snapshot,
                        mode="heldout",
                        start=str(period["start"]),
                        end=str(period["end"]),
                        schedule=self.config.schedule,
                        broker_profile=self.config.broker_profile,
                    ),
                    final,
                )
                if state_changed:
                    self.ledger.append(
                        {
                            "record_type": "heldout",
                            "experiment_id": self.config.experiment_id,
                            "epoch_id": epoch_id,
                            "fold_id": f"heldout_{label}",
                            "run_id": run_id,
                            "session_key": "heldout",
                            "period": label,
                            "strategy_artifact_id": final.artifact_id,
                            "snapshot_id": snapshot.snapshot_id,
                            "result": (
                                result.summary
                                if result is not None
                                else {
                                    "status": "failed",
                                    "error": (
                                        f"{type(heldout_error).__name__}: {heldout_error}"
                                        if heldout_error is not None
                                        else "heldout_failed"
                                    ),
                                }
                            ),
                            "result_ref": result.result_ref if result is not None else None,
                            "state_changed_during_test": True,
                        }
                    )
                    # The integrity row is this run's business record: the
                    # fail-fast below must not also log a failed attempt.
                    wrote_business_record = True
                    if restore_error is not None:
                        raise FrozenArtifactRestoreFailed(
                            "strategy or model artifacts changed during held-out "
                            "and restoring the pre-evaluation trees failed: "
                            f"{restore_error}"
                        ) from restore_error
                    raise FrozenArtifactMutated(
                        "strategy or model artifacts changed during held-out"
                    ) from heldout_error
                if heldout_error is not None:
                    raise heldout_error
                if result is None:
                    raise RuntimeError("held-out evaluation returned no result")
                self.ledger.append(
                    {
                        "record_type": "heldout",
                        "experiment_id": self.config.experiment_id,
                        "epoch_id": epoch_id,
                        "fold_id": f"heldout_{label}",
                        "run_id": run_id,
                        "session_key": "heldout",
                        "period": label,
                        "strategy_artifact_id": final.artifact_id,
                        "snapshot_id": snapshot.snapshot_id,
                        "result": result.summary,
                        "result_ref": result.result_ref,
                        # Graduation verdict of this period; the experiment-level
                        # verdict (ledger.experiment_verdict) needs every period.
                        "verdict": self.config.acceptance.heldout_verdict(
                            result.summary, walk_forward
                        ),
                    }
                )
                wrote_business_record = True
            except Exception as exc:
                if not wrote_business_record:
                    self.ledger.append(
                        {
                            **attempt,
                            "record_type": "attempt_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                raise
            finally:
                # Either a held-out record or an attempt_failed is now durable,
                # so this run is no longer an interrupted one.
                self.run_markers.finish(run_id)
            count += 1
        return count

    def _run_meta(
        self,
        epoch_id: str,
        completed_folds: int,
        visible_fold: FoldSpec,
        parent: FrozenArtifact | None,
        session_context: dict[str, object] | None = None,
        previous_prior: str = "",
    ) -> tuple[str, FrozenArtifact | None]:
        if self.meta_learner is None:
            return previous_prior, parent
        run_started = time.monotonic()
        run_id = f"run_{uuid.uuid4().hex}"
        session_id = meta_learning_id(epoch_id, completed_folds)
        deadline_exceeded = False
        attempt = {
            "experiment_id": self.config.experiment_id,
            "epoch_id": epoch_id,
            "fold_id": session_id,
            "run_id": run_id,
            "session_key": meta_session_key(epoch_id, completed_folds),
            "phase": "meta_learning",
        }
        # Evidence for a run that never gets to run its own except branch.
        self.run_markers.begin(attempt)
        try:
            context = dict(session_context or {})
            current_skills = latest_skills_snapshot(
                self.ledger.read(), experiment_dir=self.config.experiment_dir
            )
            progress = _optional_hook(context.get("progress_hook"), "progress_hook")
            _publish_progress(progress, "pit_snapshot", run_id=run_id, phase="meta")
            history, agent_trace_sidecars = _development_inputs(
                self.ledger.read(),
                ref_store=self.ref_store,
                artifacts_root=self.config.experiment_dir / "artifacts",
            )
            meta_snapshot = self.snapshots.prepare(
                fold=visible_fold,
                phase="meta",
                start=visible_fold.validation_start,
                end=visible_fold.validation_end,
                decision_time=visible_fold.valid_decision_time,
            )
            try:
                session = self.meta_learner(
                    {
                        "experiment_id": self.config.experiment_id,
                        "epoch_id": epoch_id,
                        "run_id": run_id,
                        "meta_learning_id": session_id,
                        "trigger_after_folds": completed_folds,
                        "visible_fold": _agent_visible_fold(
                            visible_fold, ref_store=self.ref_store
                        ),
                        # Host-only raw identity for the audit manifest. The Meta
                        # learner removes it before writing Agent-visible facts.
                        "host_visible_fold": visible_fold.to_record(),
                        "snapshot_id": meta_snapshot.snapshot_id,
                        "data_summary_ref": meta_snapshot.data_summary_ref,
                        "parent_artifact_id": parent.artifact_id if parent else None,
                        "previous_prior": previous_prior,
                        # Internal host source; LLMMetaLearner removes it before
                        # writing Agent-visible meta_context or manifests.
                        "skills_source_ref": (
                            str(current_skills.root)
                            if current_skills.root is not None
                            else ""
                        ),
                        "development_history": history,
                        "review_window": history.get("review_window"),
                        "agent_trace_sidecars": agent_trace_sidecars,
                        "meta_learning_memory": self._prior_meta_learning_logs(
                            session_id
                        ),
                        "directive": str(context.get("directive") or ""),
                        "prompt_override": str(context.get("prompt_override") or ""),
                        "user_question_hook": _optional_hook(
                            context.get("user_question_hook"),
                            "user_question_hook",
                        ),
                        "progress_hook": progress,
                        "session_key": str(
                            context.get("session_key")
                            or meta_session_key(epoch_id, completed_folds)
                        ),
                        "network": "disabled",
                    }
                )
            except AgentSessionDeadlineExceeded as exc:
                # Expected control flow: the meta session closed gracefully at
                # its deadline. Keep the previous PRIOR and parent, record the
                # outcome, and let the run continue with the next session.
                session = MetaSessionResult(
                    prior=previous_prior,
                    conversation_id=exc.conversation_id,
                )
                deadline_exceeded = True
            prior_text, prior_published, prior_ref, prior_generation_id = (
                self._publish_or_keep_prior(
                    session,
                    previous_prior=previous_prior,
                    generation_id=f"{session_id}_{run_id}",
                    deadline_exceeded=deadline_exceeded,
                )
            )
            # Candidate selection and the freeze are the Pipeline's, exactly as
            # for a Fold's selected Step: the Meta session only nominates, and
            # adoption is decided here on the modification check's own verdict
            # (`session.allowed`), never on the nomination alone. A learner that
            # offers a revision the check refused must not have it adopted --
            # a meta-regularized artifact becomes the next Fold's parent.
            status = "prior_only"
            frozen = parent
            if deadline_exceeded:
                status = "deadline_exceeded_kept_previous"
            elif parent is not None and session.allowed and session.revision_id:
                frozen = self.artifacts.freeze_revision(
                    session.revision_id,
                    artifact_id=f"strategy_{session_id}_meta_learning",
                    experiment_id=self.config.experiment_id,
                    epoch_id=epoch_id,
                    fold_id=session_id,
                    run_id=run_id,
                    step_id="meta_learning",
                )
                frozen = FrozenArtifact(
                    frozen.artifact_id,
                    Path(frozen.path),
                    Path(frozen.model_path) if frozen.model_path is not None else None,
                    run_id,
                    session_id,
                    "meta_learning",
                    session.revision_id,
                    # Never backtested: the next Fold may only fall back to it
                    # after validating identical content itself.
                    requires_validation=True,
                )
                status = "meta_regularized"
            elif parent is not None and session.allowed:
                status = "prior_only_kept_parent"
            elif parent is not None:
                status = "rejected_kept_parent"
            _publish_progress(progress, "publishing", run_id=run_id)
            skills = self._publish_or_keep_skills(
                session.skills_source_ref,
                current=current_skills,
                generation_id=f"{session_id}_{run_id}",
                run_id=run_id,
            )
            trace_ref = agent_trace_path(
                self.config.experiment_dir / "artifacts", run_id
            )
            self.ledger.append(
                {
                    "record_type": "meta_learning",
                    "experiment_id": self.config.experiment_id,
                    "epoch_id": epoch_id,
                    "fold_id": session_id,
                    "run_id": run_id,
                    "session_key": meta_session_key(epoch_id, completed_folds),
                    "meta_learning_id": session_id,
                    "trigger_after_folds": completed_folds,
                    "prior": prior_text,
                    "prior_published": prior_published,
                    "prior_ref": prior_ref or None,
                    "prior_generation_id": prior_generation_id or None,
                    "prior_chars": len(prior_text),
                    "skills_ref": skills.skills_ref or None,
                    "skills_generation_id": skills.generation_id or None,
                    **skills.stats.ledger_fields(),
                    "skills_published": skills.published,
                    "status": status,
                    "modification_check": dict(session.modification_check),
                    "frozen_strategy_artifact_id": (
                        frozen.artifact_id
                        if status == "meta_regularized" and frozen
                        else None
                    ),
                    "frozen_strategy_artifact_path": (
                        str(frozen.path)
                        if status == "meta_regularized" and frozen
                        else None
                    ),
                    "agent_trace_ref": str(trace_ref) if trace_ref.exists() else None,
                    "review_window": history.get("review_window"),
                    **_session_timing(context, run_started),
                }
            )
            expire_experiment_session_inbox(
                self.config.experiment_dir,
                meta_session_key(epoch_id, completed_folds),
                expired_by=run_id,
            )
            return prior_text, frozen
        except Exception as exc:
            self.ledger.append(
                {
                    **attempt,
                    "record_type": "attempt_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            # Either a meta_learning record or an attempt_failed is now durable,
            # so this run is no longer an interrupted one.
            self.run_markers.finish(run_id)

    def _parent_control(
        self,
        parent: FrozenArtifact | None,
        fold: FoldSpec,
        snapshot: SnapshotBundle,
        *,
        progress,
        run_id: str,
    ) -> tuple[EvaluationResult | None, str]:
        """Replay the inherited parent unchanged on this Fold's Validation window.

        Runs before the Agent session through the same evaluator, snapshot and
        replay bounds the session's own Validations use, and is charged to no
        session budget. The result is the walk-forward evidence for the
        previous Fold's frozen strategy and the parent's completed Validation
        in this Fold; a failure is recorded explicitly and the Fold proceeds.
        """
        if parent is None:
            return None, ""
        _publish_progress(progress, "parent_control", run_id=run_id)
        try:
            return (
                self.evaluator.evaluate(
                    EvaluationRequest(
                        revision=_frozen_revision(parent),
                        snapshot=snapshot,
                        mode="valid",
                        start=fold.validation_start,
                        end=fold.validation_end,
                        schedule=self.config.schedule,
                        broker_profile=self.config.broker_profile,
                    )
                ),
                "",
            )
        except Exception as exc:  # noqa: BLE001 - recorded, the Fold still runs
            return None, f"{type(exc).__name__}: {exc}"

    def _assert_parent_validated_in_fold(
        self,
        parent: FrozenArtifact,
        steps: tuple[StepResult, ...],
        *,
        control: EvaluationResult | None = None,
    ) -> None:
        """Refuse to fall back to a meta-regularized parent this Fold never validated.

        A meta-regularized artifact enters the Fold without a backtest. Falling
        back to it silently would ship an unvalidated strategy. The host's
        parent control is that Validation when it completed and passed
        acceptance; otherwise identity is established by comparing the trees
        directly: a Step whose revision has no changed strategy or model file
        performed Validation on this parent.
        """
        if control is not None and not self.config.acceptance.evaluate(control.summary)[0]:
            return
        for step in steps:
            revision = self.artifacts.revision(step.revision_id)
            if modification_delta(parent.path, revision.output_path).changed_files:
                continue
            models = getattr(revision, "models_path", None)
            if (
                models is not None
                and parent.model_path is not None
                and model_artifact_delta(parent.model_path, models).changed_files
            ):
                continue
            hard, _ = self.config.acceptance.evaluate(step.validation.summary)
            if not hard:
                return
        raise RuntimeError(
            "Meta-regularized parent has no acceptable complete Validation in this Fold; "
            "refusing unvalidated fallback"
        )

    def _prior_meta_learning_logs(self, current_meta_learning_id: str) -> str:
        """Latest prior Meta trace from each of the most recent N Epochs.

        Periodic sessions in the current Epoch are eligible, so the immediate
        predecessor remains visible without allowing raw-memory growth to
        multiply by the number of interval triggers.
        """
        chunks: list[str] = []
        keep = max(0, self.config.meta_memory_max_epochs)
        if not keep:
            return ""
        latest_by_epoch: dict[str, dict[str, object]] = {}
        epoch_order: list[str] = []
        for record in self.ledger.read("meta_learning"):
            if meta_record_id(record) == current_meta_learning_id:
                continue
            epoch = str(record.get("epoch_id") or "")
            if epoch not in latest_by_epoch:
                epoch_order.append(epoch)
            latest_by_epoch[epoch] = record
        for epoch in epoch_order[-keep:]:
            trace = self._meta_learning_trace_ref(latest_by_epoch[epoch])
            if not trace.exists():
                continue
            text = trace.read_text(encoding="utf-8")
            if text.strip():
                chunks.append(text if text.endswith("\n") else text + "\n")
        return "".join(chunks)

    def _meta_learning_trace_ref(self, record: Mapping[str, object]) -> Path:
        ref = record.get("agent_trace_ref")
        if ref:
            return Path(str(ref))
        run_id = record.get("run_id")
        if run_id:
            return agent_trace_path(
                self.config.experiment_dir / "artifacts", str(run_id)
            )
        return Path("__missing_meta_learning_agent_trace__")

    def run_meta_session(
        self,
        epoch_id: str,
        completed_folds: int,
        visible_fold: FoldSpec,
        *,
        parent: FrozenArtifact | None,
        previous_prior: str = "",
        session_context: dict[str, object] | None = None,
    ) -> tuple[str, FrozenArtifact | None]:
        """Run one scheduled Meta session through the canonical ledger path.

        Returns the current PRIOR and the parent the next Fold must start from.
        """

        return self._run_meta(
            epoch_id,
            completed_folds,
            visible_fold,
            parent,
            session_context,
            previous_prior=previous_prior,
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

    def _publish_or_keep_prior(
        self,
        session: MetaSessionResult,
        *,
        previous_prior: str,
        generation_id: str,
        deadline_exceeded: bool,
    ) -> tuple[str, bool, str, str]:
        """Publish a non-empty new PRIOR.md, otherwise keep the previous version."""
        store = ExperimentPriorStore(self.config.experiment_dir)
        candidate = "" if deadline_exceeded else str(session.prior or "").strip()
        previous = previous_prior.strip()
        if candidate and len(candidate) > PRIOR_MAX_CHARS:
            raise ValueError(
                f"PRIOR.md is {len(candidate)} characters; keep it to {PRIOR_MAX_CHARS}"
            )
        if not candidate and not previous and not deadline_exceeded:
            raise ValueError("the first Meta session must produce a non-empty PRIOR.md")
        if candidate and candidate != previous:
            published = store.publish(candidate, generation_id=generation_id)
            return published.text, True, published.prior_ref, published.generation_id
        return (
            previous,
            False,
            store.current_ref(),
            store.current_generation_id(),
        )


def _keep_frozen_artifact_ids(
    records: list[dict[str, object]],
    extra_id: str | None = None,
) -> tuple[str, ...]:
    """Frozen trees still named by the latest fold/meta frontier, plus the fold now finishing."""
    keep: set[str] = set()
    extra = str(extra_id or "")
    if extra:
        keep.add(extra)
    for record in latest_fold_records(records).values():
        artifact_id = str(record.get("frozen_strategy_artifact_id") or "")
        if artifact_id:
            keep.add(artifact_id)
    for record in records:
        if not is_frozen_artifact_mutation(record):
            continue
        for key in ("frozen_strategy_artifact_id", "strategy_artifact_id"):
            artifact_id = str(record.get(key) or "")
            if artifact_id:
                keep.add(artifact_id)
    latest_meta: dict[str, dict[str, object]] = {}
    for record in records:
        if record.get("record_type") != "meta_learning":
            continue
        session_key = str(record.get("session_key") or "")
        if session_key:
            latest_meta[session_key] = record
    for record in latest_meta.values():
        if record.get("status") != "meta_regularized":
            continue
        artifact_id = str(record.get("frozen_strategy_artifact_id") or "")
        if artifact_id:
            keep.add(artifact_id)
    return tuple(sorted(keep))


def _select_step(
    steps: tuple[StepResult, ...], selected_id: str | None
) -> StepResult | None:
    # Only completed full-window validations ever become a StepResult.
    if selected_id is None:
        return steps[-1] if steps else None
    selected = next((step for step in steps if step.step_id == selected_id), None)
    if selected is None:
        raise RuntimeError(f"selected Step is absent: {selected_id}")
    return selected


def _epoch_index(epoch_id: str) -> int:
    _, _, number = epoch_id.rpartition("_")
    try:
        return int(number)
    except ValueError:
        return 1


def _snapshot_config_record(snapshots: SnapshotProvider) -> dict[str, object]:
    """The provider's own decision-window configuration, when it has one.

    ``build_experiment_facts`` reads ``snapshot_config.decision_windows`` for the
    visible-timeline block; the local daily provider has no such configuration.
    """
    config = getattr(snapshots, "config", None)
    to_record = getattr(config, "to_record", None)
    return dict(to_record()) if callable(to_record) else {}


def _step_record(step: StepResult) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "revision_id": step.revision_id,
        "complete_validation": True,
        "parent_control": step.parent_control,
        "summary": step.validation.summary,
        "validation_result_ref": step.validation.result_ref,
    }


def _selection_statistics(
    steps: tuple[StepResult, ...], selected: StepResult | None
) -> dict[str, object]:
    """How wide this Fold's search was, and how much of the winner it explains.

    ``candidates_evaluated`` counts every candidate the session replayed to a
    complete Validation on this Fold's window: one per ``daily_backtest`` call
    and one per ``batch_validate`` candidate that finished. A failed replay
    never becomes a Step and never counts, and the host's parent control is
    not a candidate — it is the baseline the search is measured against, not a
    trial in it.

    That same count is N for :func:`ledger.deflated_sharpe`, computed for the
    candidate ``finish_fold`` nominated. ``trials`` is the subset of those
    candidates carrying a finite Sharpe, i.e. the N the formula actually used.
    Keeping the parent (nominating the control node) or finishing with no
    candidate leaves the probability ``None``: nothing was selected out of the
    search, so there is no selection bias to correct.
    """

    candidates = [step for step in steps if not step.parent_control]
    nominated = (
        selected if selected is not None and not selected.parent_control else None
    )
    series = (
        _validation_daily_returns(nominated.validation.result_ref)
        if nominated
        else None
    )
    statistics = deflated_sharpe(
        observed_sharpe=(
            nominated.validation.summary.get("sharpe") if nominated else None
        ),
        trial_sharpes=[step.validation.summary.get("sharpe") for step in candidates],
        returns=series if series is not None else (),
    )
    if nominated is None:
        statistics["unavailable_reason"] = "no_nominated_candidate"
    elif series is None and statistics["unavailable_reason"] == "return_series_too_short":
        # The record could not be read at all; saying the window was short
        # would send a reader looking at the calendar instead of the file.
        statistics["unavailable_reason"] = "return_series_missing"
    return {"candidates_evaluated": len(candidates), **statistics}


def _validation_daily_returns(result_ref: str) -> list[float] | None:
    """Daily returns of one completed Validation, read from its own record.

    The equity curve lives in the replay's ``result.json``, never in the
    summary, and it is the only place the return series exists. ``None`` says
    the series could not be read at all -- an absent, unreadable or
    curve-less record -- which is a different fact from a window that is
    genuinely too short, and the two must not share one reason.
    """

    path = Path(str(result_ref or ""))
    if path.is_dir():
        path = path / "result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    curve = payload.get("equity_curve") if isinstance(payload, dict) else None
    if not isinstance(curve, list):
        return None
    return [
        value
        for _day, value in daily_returns_from_curve(
            [row for row in curve if isinstance(row, Mapping)]
        )
    ]


def _parent_control_record(
    parent: FrozenArtifact | None,
    control: EvaluationResult | None,
    error: str,
    steps: tuple[StepResult, ...],
) -> dict[str, object] | None:
    """Ledger projection of the host's parent control; None without a parent."""
    if parent is None:
        return None
    if control is None:
        return {
            "status": "failed",
            "parent_strategy_artifact_id": parent.artifact_id,
            "error": error,
        }
    return {
        "status": "ok",
        "parent_strategy_artifact_id": parent.artifact_id,
        # The in-session Step node the developer recorded for it, when the
        # session got that far (a deadline before the first call records none).
        "step_id": next(
            (step.step_id for step in steps if step.parent_control), None
        ),
        "validation_result": control.summary,
        "validation_result_ref": control.result_ref,
    }


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


def _development_inputs(
    records: list[dict[str, object]],
    *,
    ref_store: AgentRefStore,
    artifacts_root: str | Path | None = None,
) -> tuple[dict[str, object], list[AgentTraceFullSidecar]]:
    """Meta-visible development history plus internal full-trace sidecars.

    Every public field crosses the Agent boundary, so it is built exclusively
    from ``agent_views``: raw fold ids become opaque refs and Test evidence is
    limited to the compact frozen-test metric whitelist of already-completed
    Folds. Held-out never appears.

    ``fold_validation_history`` is the one list of compact Fold histories: the
    ``compact_fold_history`` projection of every completed Fold so far, across
    Epochs, so Meta sees the whole accumulated Validation record next to the
    PRIOR that absorbed it. The review window is named, not re-listed --
    ``review_window`` and ``fold_reviews`` identify the Folds completed after
    the previous Meta by the same opaque ``fold_id``. A second window-scoped
    copy of the same projection would put one Fold in ``development_history``
    twice, which a Meta counting rows reads as two Folds.

    ``fold_reviews`` covers only the review-window Folds, and carries frozen
    strategy source, a bounded Agent Trace index, ``agent_process_summary``,
    and ``agent_trace_full`` sidecar metadata. Each sidecar is a byte-exact copy
    of the raw Fold AgentTraceWriter JSONL; its bytes stay in the internal
    sidecar list and never enter ordinary Fold prompts or ``meta_context``.
    """

    folds, review_window = select_meta_review_folds(
        records, ref_store=ref_store
    )
    reviews, sidecars = build_meta_fold_review_bundle(
        folds, ref_store=ref_store, artifacts_root=artifacts_root
    )
    return {
        "evaluation_contract": {
            "validation": "Fold selection and iteration evidence",
            "frozen_test": "compact completed-Fold metrics are adaptive meta-development feedback",
            "heldout": "never visible; sole final untouched evaluation",
        },
        "fold_reviews": reviews,
        "review_window": review_window,
        "fold_validation_history": [
            _compact_fold_history(
                record,
                ref_store=ref_store,
                include_frozen_test_metrics=True,
            )
            for record in latest_fold_records(records).values()
        ],
        "meta_learning": [
            _agent_visible_ledger_record(
                record,
                ref_store=ref_store,
                include_frozen_test_metrics=True,
            )
            for record in records
            if record.get("record_type") == "meta_learning"
        ],
    }, sidecars


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
        wait = float(value.get("researcher_wait_seconds", 0.0))
        if wall < 0 or wait < 0:
            raise ValueError("session timing values must be non-negative")
        return {
            "run_wall_seconds": round(wall, 1),
            "researcher_wait_seconds": round(wait, 1),
        }
    return {
        "run_wall_seconds": round(max(0.0, time.monotonic() - fallback_started), 1),
        "researcher_wait_seconds": 0.0,
    }


def _agent_visible_fold(
    fold: FoldSpec, *, ref_store: AgentRefStore
) -> dict[str, object]:
    # The raw fold id encodes the held-out test period, so the Meta session
    # sees the same opaque ref every other agent-visible surface projects.
    return {
        "fold_id": ref_store.get_or_create("fold", fold.fold_id),
        "input_window": f"{fold.input_window_start}..{fold.input_window_end}",
        "validation_period": f"{fold.validation_start}..{fold.validation_end}",
        "valid_decision_time": fold.valid_decision_time.isoformat(),
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
        "max_steps": config.max_steps_per_fold,
        "max_backtests": config.max_backtests_per_fold,
        "max_llm_calls": config.max_llm_calls,
        "deadline_seconds": config.max_fold_minutes * 60,
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
                # The fold deadline may be raised per session, bounded by the
                # absolute ceiling above; other budgets stay downward-only.
                if float(value) > _MAX_DEADLINE_OVERRIDE_MINUTES * 60:
                    raise ValueError(
                        "deadline_seconds override cannot exceed "
                        f"{_MAX_DEADLINE_OVERRIDE_MINUTES} minutes"
                    )
            elif float(value) > float(limits[name]):
                raise ValueError(
                    f"{name} override cannot exceed the configured Fold limit"
                )
            limits[name] = float(value) if name == "deadline_seconds" else int(value)
    # Grace is not a resource_override key: add it after the override check so
    # the session budget and the runner reservation share one config source.
    main_minutes = float(limits["deadline_seconds"]) / 60.0
    limits["deadline_seconds"] = fold_session_deadline_seconds(
        main_minutes, config.deadline_grace_minutes
    )
    limits["deadline_grace_seconds"] = float(config.deadline_grace_minutes) * 60.0
    return limits


ExperimentPipeline = DailyStrategyPipeline
