"""Experiment pipeline: Step/Fold/Epoch/Held-out orchestration.

docs/pipeline-design.md. The Pipeline schedules Data, Environment, and Agent
in time order, freezes inputs/outputs at each boundary, and writes the single
experiment ledger. It implements no investment logic and never rewrites
strategy content; it only accepts, freezes, falls back, and records.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd

from autotrade.environment.executor import (
    DockerStrategyExecutor,
    StrategyExecutor,
    TrustedStrategyExecutor,
)
from autotrade.environment.artifacts import model_artifact_delta, modification_delta
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import agent_trace_path
from autotrade.environment.replay import (
    ContextDataProvider,
    ExecutionPriceProvider,
    ReplayResult,
    run_daily_replay,
)
from autotrade.environment.strategy import NLQuery
from autotrade.environment.strategy_loader import load_strategy

from .agent_views import (
    agent_visible_ledger_record as _agent_visible_ledger_record,
    compact_fold_history as _compact_fold_history,
)
from .meta_inputs import (
    AgentTraceFullSidecar,
    build_meta_fold_review_bundle,
    select_meta_review_folds,
)
from .prior import ExperimentPriorStore, PRIOR_MAX_CHARS
from .skills import (
    ExperimentSkillsStore,
    SkillsPublication,
    SkillsSnapshot,
    latest_skills_snapshot,
    resolve_collected_skills_source,
)
from autotrade.agent.runner import AgentSessionDeadlineExceeded
from .config import (
    ArtifactRevision,
    ArtifactStore,
    EvaluationBackend,
    EvaluationRequest,
    FoldDeveloper,
    FoldOutcome,
    FoldSessionRequest,
    FoldSessionResult,
    FrozenArtifact,
    MetaLearner,
    MetaSessionResult,
    RollingExperimentConfig,
    SnapshotProvider,
    StepResult,
    StrategyExperimentConfig,
)
from .agent_inbox import expire_experiment_session_inbox
from .folds import FoldSpec, build_fold_schedule, heldout_periods
from .hitl_state import fold_session_key
from .ledger import ExperimentLedger, latest_fold_records
from .meta_schedule import (
    meta_learning_id,
    meta_learning_trigger_counts,
    meta_record_id,
    meta_session_key,
)


# A per-session deadline override may raise the fold deadline above the
# configured maximum, bounded by this hard ceiling in minutes.
_MAX_DEADLINE_OVERRIDE_MINUTES = 480


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
        executor = self._create_executor()
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

    def _create_executor(self) -> StrategyExecutor:
        if self.executor_factory is not None:
            executor = self.executor_factory(self.config)
            if not isinstance(executor, StrategyExecutor):
                raise TypeError("executor_factory must return a StrategyExecutor")
            return executor
        if self.config.execution_mode == "trusted":
            return TrustedStrategyExecutor(load_strategy(self.config.strategy_path))
        return DockerStrategyExecutor(self.config.strategy_path, self.config.sandbox)


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

    def run(self, trading_days: list[str]) -> dict[str, object]:
        if self.ledger.read():
            raise RuntimeError("batch experiments require an empty experiment ledger")
        folds = build_fold_schedule(
            self.config.first_test_period,
            self.config.last_test_period,
            trading_days,
            window_months=self.config.window_months,
            period=self.config.fold_period,
            min_region_trade_days=self.config.min_region_trade_days,
        )
        parent: FrozenArtifact | None = None
        prior = ""
        final_epoch = ""
        for epoch_index in range(1, self.config.epochs + 1):
            epoch_id = f"epoch_{epoch_index:03d}"
            final_epoch = epoch_id
            triggers = set(
                meta_learning_trigger_counts(
                    len(folds), self.config.meta_learning_fold_interval
                )
            )
            for fold_index, fold in enumerate(folds):
                if self.meta_learner is not None and fold_index in triggers:
                    prior, parent = self._run_meta(
                        epoch_id, fold_index, fold, parent, previous_prior=prior
                    )
                outcome = self.run_fold(epoch_id, fold, parent=parent, prior=prior)
                parent = outcome.frozen
        # Fail-fast path: only reachable when every Fold ended with no freezable
        # artifact at all (integrity failures); acceptance shortfalls alone never
        # land here because they still freeze with warnings.
        if parent is None:
            raise RuntimeError("experiment produced no frozen strategy")
        heldout_count = self.run_heldout(final_epoch, parent, trading_days)
        return {
            "final_strategy_artifact": parent.artifact_id,
            "heldout_runs": heldout_count,
        }

    def run_fold(
        self,
        epoch_id: str,
        fold: FoldSpec,
        *,
        parent: FrozenArtifact | None,
        prior: str = "",
        session_context: dict[str, object] | None = None,
    ) -> FoldOutcome:
        run_started = time.monotonic()
        run_id = f"run_{uuid.uuid4().hex}"
        context = dict(session_context or {})
        progress = _optional_hook(context.get("progress_hook"), "progress_hook")
        budgets = _session_budgets(self.config, context.get("resource_override"))
        current_skills = latest_skills_snapshot(
            self.ledger.read(), experiment_dir=self.config.experiment_dir
        )
        retained_artifact_id = parent.artifact_id if parent is not None else None
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
                        directive=str(context.get("directive") or ""),
                        prompt_override=str(context.get("prompt_override") or ""),
                        sandbox_gpu_count=_optional_gpu_count(
                            context.get("sandbox_gpu_count")
                        ),
                        fold_period=self.config.fold_period,
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
            if len(session.steps) > budgets["max_steps"]:
                raise RuntimeError("Fold developer exceeded the Step budget")
            selected = _select_step(session.steps, session.selected_step_id)
            hard: list[str] = ["no_complete_validation"]
            warnings: list[str] = []
            if selected is not None:
                hard, warnings = self.config.acceptance.evaluate(
                    selected.validation.summary,
                    complete=selected.validation.complete,
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
                    self._assert_parent_validated_in_fold(parent, session.steps)
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
            test_result_ref: str | None = None
            if frozen is not None:
                _publish_progress(progress, "frozen_test", run_id=run_id)
                test_snapshot = self.snapshots.prepare(
                    fold=fold,
                    phase="frozen_test",
                    start=fold.test_start,
                    end=fold.test_end,
                    decision_time=fold.test_decision_time,
                )
                try:
                    test_result = self.evaluator.evaluate(
                        EvaluationRequest(
                            revision=_frozen_revision(frozen),
                            snapshot=test_snapshot,
                            mode="frozen_test",
                            start=fold.test_start,
                            end=fold.test_end,
                            schedule=self.config.schedule,
                            broker_profile=self.config.broker_profile,
                        )
                    )
                    test_summary = test_result.summary
                    test_result_ref = test_result.result_ref
                except Exception as exc:  # noqa: BLE001 - Frozen Test is diagnostic evidence
                    # Frozen Test is diagnostic only: it cannot undo an already
                    # accepted revision, but the failure must remain explicit.
                    test_summary = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                test_snapshot = None
                test_summary = {"status": "skipped_no_frozen_artifact"}
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
                "conversation_id": session.conversation_id,
                "finish_reason": session.finish_reason,
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
            self.ledger.append(record)
            expire_experiment_session_inbox(
                self.config.experiment_dir,
                str(record["session_key"]),
                expired_by=run_id,
            )
            retained_artifact_id = frozen.artifact_id if frozen is not None else None
            return FoldOutcome(
                fold.fold_id, run_id, status, frozen, validation, test_summary
            )
        except Exception as exc:
            self.ledger.append(
                {
                    "record_type": "attempt_failed",
                    "experiment_id": self.config.experiment_id,
                    "epoch_id": epoch_id,
                    "fold_id": fold.fold_id,
                    "run_id": run_id,
                    "phase": "fold",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
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
                if record.get("period")
            }
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
            snapshot = self.snapshots.prepare(
                fold=None,
                phase="heldout",
                start=str(period["start"]),
                end=str(period["end"]),
                decision_time=period["decision_time"],  # type: ignore[arg-type]
            )
            result = self.evaluator.evaluate(
                EvaluationRequest(
                    revision=_frozen_revision(final),
                    snapshot=snapshot,
                    mode="heldout",
                    start=str(period["start"]),
                    end=str(period["end"]),
                    schedule=self.config.schedule,
                    broker_profile=self.config.broker_profile,
                )
            )
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
                }
            )
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
                    "record_type": "attempt_failed",
                    "experiment_id": self.config.experiment_id,
                    "epoch_id": epoch_id,
                    "fold_id": session_id,
                    "run_id": run_id,
                    "session_key": meta_session_key(epoch_id, completed_folds),
                    "phase": "meta_learning",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

    def _assert_parent_validated_in_fold(
        self, parent: FrozenArtifact, steps: tuple[StepResult, ...]
    ) -> None:
        """Refuse to fall back to a meta-regularized parent this Fold never validated.

        A meta-regularized artifact enters the Fold without a backtest. Falling
        back to it silently would ship an unvalidated strategy. Identity is
        established by comparing the trees directly: a Step whose revision has
        no changed strategy or model file performed Validation on this parent.
        """
        for step in steps:
            if not step.validation.complete:
                continue
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
            hard, _ = self.config.acceptance.evaluate(
                step.validation.summary, complete=step.validation.complete
            )
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
    complete = [step for step in steps if step.validation.complete]
    if selected_id is None:
        return complete[-1] if complete else None
    selected = next((step for step in steps if step.step_id == selected_id), None)
    if selected is None:
        raise RuntimeError(f"selected Step is absent: {selected_id}")
    if not selected.validation.complete:
        raise RuntimeError("selected Step lacks complete validation")
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
        "complete_validation": step.validation.complete,
        "summary": step.validation.summary,
        "validation_result_ref": step.validation.result_ref,
    }


def _frozen_revision(artifact: FrozenArtifact) -> ArtifactRevision:
    return ArtifactRevision(artifact.artifact_id, artifact.path, artifact.model_path)


def _development_history(
    records: list[dict[str, object]],
    *,
    ref_store: AgentRefStore,
    artifacts_root: str | Path | None = None,
) -> dict[str, object]:
    history, _sidecars = _development_inputs(
        records, ref_store=ref_store, artifacts_root=artifacts_root
    )
    return history


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
    Folds. Held-out never appears. ``fold_reviews`` and
    ``fold_backtest_summaries`` only cover regular Folds completed after the
    previous Meta; older Folds are already absorbed into PRIOR.
    ``fold_reviews`` carries frozen strategy source, a bounded Agent Trace
    index, ``agent_process_summary``, and ``agent_trace_full`` sidecar metadata.
    Each sidecar is a byte-exact copy of the raw Fold AgentTraceWriter JSONL; its
    bytes stay in the internal sidecar list and never enter ordinary Fold
    prompts or ``meta_context``.
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
        "fold_backtest_summaries": [
            _compact_fold_history(
                record,
                ref_store=ref_store,
                include_frozen_test_metrics=True,
            )
            for record in folds
        ],
        "fold_reviews": reviews,
        "review_window": review_window,
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
                # 240-minute hard ceiling; the other budgets stay downward-only.
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
    # The budget handed to the session is main deadline plus the trailing
    # wrap-up grace window; the runner reserves that trailing window for
    # wrap-up (reaching the main deadline never interrupts the model).
    limits["deadline_seconds"] = (
        float(limits["deadline_seconds"]) + config.deadline_grace_minutes * 60
    )
    return limits


ExperimentPipeline = DailyStrategyPipeline
