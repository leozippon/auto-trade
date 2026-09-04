"""Pre-approval Prompt preview for Fold and Meta sessions.

The console shows this text before it approves a session, so it has to be the
prompt the session will actually receive rather than a second description of
it. Nothing here restates prompt text, budgets or contract wording: the
experiment is resolved by the worker's own ``load_worker_options`` over the
persisted parameters, the Fold schedule is rebuilt by the pipeline's own
``build_fold_schedule``, the Agent-visible facts come from the shared
``build_experiment_facts`` projection of a run manifest shaped like the one a
session writes, and the text is rendered by ``build_system_prompt`` and
``build_meta_learning_prompt``. A change to ``prompts.py``, to a budget default
or to the facts projection therefore reaches this preview with no edit here.

Only what comes into existence when the session starts is unavailable: the run
id, the sandbox runtime environment, which PIT files the session's snapshot
will carry, the parent the artifact chain will have inherited by then, and the
host's parent-control replay. Those render as ``RUNTIME_PLACEHOLDER`` instead
of being invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import (
    FOLD_DEFAULT_INSTRUCTION,
    build_meta_learning_prompt,
    build_system_prompt,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.hitl_state import (
    CONTROL_NAME,
    HITL_DIR_NAME,
    PARAMS_NAME,
    SCHEDULE_NAME,
    read_control,
    read_json,
)
from autotrade.pipelines.prior import latest_prior_text

from .registry import read_ledger_records

if TYPE_CHECKING:
    from autotrade.environment.sandbox import SandboxLimits
    from autotrade.pipelines.folds import FoldSpec
    from autotrade.pipelines.worker import InteractiveWorkerOptions

# Marks a fact that exists only once the session starts. Short and unmistakable
# inside the facts JSON, and explained by PREVIEW_NOTE above the preview.
RUNTIME_PLACEHOLDER = "<runtime>"

PREVIEW_NOTE = (
    "预览走 worker 同一条装配链：hitl/params.json → load_worker_options → "
    "build_experiment_facts → build_system_prompt，会话计划、Development Ledger 与当前 PRIOR 一并注入。"
    "只有会话启动时才产生的事实（run id、沙箱 runtime env、快照实际挂载的数据、"
    f"届时继承的父产物与父对照回放）显示为 {RUNTIME_PLACEHOLDER}。"
)

_SYSTEM_BANNER = "======== 系统提示词（build_system_prompt）========"
_USER_BANNER = "======== 开局用户消息 ========"

# Never a directory and never a file: fold_workspace_map's two optional entries
# are decided from the resolved config below instead of from a workspace that
# only exists inside a running session.
_NO_WORKSPACE = Path("/nonexistent/prompt-preview-workspace")


def build_prompt_preview(
    experiment_dir: Path, session_key: str, directive: str, *, repo_root: Path
) -> dict[str, object]:
    """Raises KeyError for an unknown session and ValueError for held-out keys."""
    # Deferred: pulls the pipeline/worker stack, which the console only needs
    # when a researcher actually opens a preview.
    from autotrade.pipelines.worker import load_worker_options

    directory = Path(experiment_dir)
    entry = _session_entry(directory, session_key)
    kind = str(entry.get("kind") or "")
    if kind not in {"fold", "meta", "meta_learning"}:
        raise ValueError(f"unsupported session kind: {kind}")
    # The same store a session projects its identities through, constructed
    # before anything else so a legacy experiment fails here exactly as a real
    # session would.
    ref_store = AgentRefStore(directory)
    options = load_worker_options(directory, repo_root=repo_root)
    control = read_control(directory / HITL_DIR_NAME / CONTROL_NAME)
    records = read_ledger_records(directory)
    prior = latest_prior_text(records)
    epoch_id = str(entry.get("epoch_id") or "")
    # What the worker would inherit as this session's parent: the console's
    # imported seed, or any frozen artifact the ledger already carries. Which
    # exact artifact that is depends on the sessions that still run first, so
    # only its presence is stated here.
    is_initial = not _has_parent_artifact(directory, records)
    context = _SessionContext(
        options=options,
        ref_store=ref_store,
        records=records,
        prior=prior,
        epoch_id=epoch_id,
        is_initial=is_initial,
    )
    if kind == "fold":
        system, instruction = _fold_prompt(
            context,
            fold_id=str(entry.get("fold_id") or ""),
            directive=directive,
            resource_override=control.resource_overrides.get(session_key),
            prompt_override=control.prompt_overrides.get(session_key, ""),
        )
    else:
        system, instruction = _meta_prompt(
            context,
            trigger_after_folds=int(entry.get("fold_index") or 0),
            directive=directive,
            prompt_override=control.prompt_overrides.get(session_key, ""),
        )
    prompt = f"{_SYSTEM_BANNER}\n{system}\n\n{_USER_BANNER}\n{instruction}"
    return {"prompt": prompt, "note": PREVIEW_NOTE}


@dataclass(frozen=True)
class _SessionContext:
    """Everything both session kinds resolve the same way."""

    options: InteractiveWorkerOptions
    ref_store: AgentRefStore
    records: list[dict[str, object]]
    prior: str
    epoch_id: str
    is_initial: bool

    @property
    def rolling(self):
        return self.options.rolling

    @property
    def strategy_limits(self) -> SandboxLimits:
        """The formal executor's strategy wall clocks, from the worker's own
        sandbox derivation rather than a second copy of the defaults."""
        from autotrade.pipelines.worker import _strategy_sandbox_from_spec

        return _strategy_sandbox_from_spec(
            self.options.agent_sandbox,
            fit_timeout_seconds=self.rolling.strategy_fit_timeout_seconds,
        ).limits

    @property
    def context_compaction(self) -> dict[str, object] | None:
        llm = self.options.llm
        if llm is None:
            return None
        return {
            "enabled": llm.compact_enabled,
            "token_threshold": llm.compaction.token_threshold,
            "max_calls": llm.compaction.max_calls,
        }


def _fold_prompt(
    context: _SessionContext,
    *,
    fold_id: str,
    directive: str,
    resource_override: object,
    prompt_override: str,
) -> tuple[str, str]:
    from autotrade.pipelines.experiment import _epoch_index, _session_budgets
    from autotrade.pipelines.local_backend import fold_workspace_map

    rolling = context.rolling
    fold = _fold_spec(context.options, fold_id)
    epoch_index = _epoch_index(context.epoch_id)
    budgets = _session_budgets(rolling, resource_override)
    limits = context.strategy_limits
    manifest: dict[str, object] = {
        "experiment_id": rolling.experiment_id,
        "epoch_id": context.epoch_id,
        "fold_id": fold.fold_id,
        "kind": "fold",
        "fold": {
            "input_window": f"{fold.input_window_start}..{fold.input_window_end}",
            "validation_period": f"{fold.validation_start}..{fold.validation_end}",
        },
        "valid_decision_time": fold.valid_decision_time.isoformat(),
        "fold_period": rolling.fold_period,
        "test_stage": rolling.test_stage,
        "snapshot_config": context.options.snapshot_config.to_record(),
        "phase": _phase(epoch_index, rolling.convergence_start_epoch),
        "is_initial_artifact": context.is_initial,
        "template_ref": "agent_output_template" if context.is_initial else None,
        "modification_constraints": rolling.step_constraints.to_record(),
        "acceptance_rules": rolling.acceptance.to_record(),
        "schedule": rolling.schedule.to_record(),
        "broker_profile": rolling.broker_profile.to_record(),
        "nl_failure_policy": rolling.nl_failure_policy,
        "step_tree_enabled": rolling.step_tree_enabled,
        "record_failed_attempts": rolling.record_failed_attempts,
        "max_steps": budgets["max_steps"],
        "max_backtests_per_fold": budgets["max_backtests"],
        "deadline_seconds": budgets["deadline_seconds"],
        "finalize_before_deadline_seconds": rolling.finalize_before_deadline_seconds,
        "sandbox_spec": (
            context.options.agent_sandbox.to_record()
            if context.options.agent_sandbox is not None
            else None
        ),
        "budgets": {
            "max_steps": budgets["max_steps"],
            "max_backtests": budgets["max_backtests"],
            "max_llm_calls": budgets["max_llm_calls"],
            "deadline_seconds": budgets["deadline_seconds"],
            "deadline_grace_seconds": budgets["deadline_grace_seconds"],
            "strategy_inference_timeout_seconds": limits.timeout_seconds,
            "strategy_fit_timeout_seconds": limits.fit_timeout_seconds,
        },
    }
    workspace = fold_workspace_map(_NO_WORKSPACE)
    if rolling.workspace_reference:
        workspace["refs"] = "refs/"
    if context.prior.strip():
        workspace["prior"] = "inputs/PRIOR.md"
    # The three blocks LLMFoldDeveloper._fold_facts adds beside the shared
    # projection: the cross-fold Validation record, the host's parent control,
    # and the fixed workspace/boundary index.
    facts: dict[str, object] = {
        **build_experiment_facts(
            manifest=manifest,
            ref_store=context.ref_store,
            max_llm_calls=int(budgets["max_llm_calls"]),
            context_compaction=context.context_compaction,
            model_artifacts_empty=True if context.is_initial else None,
        ),
        "development_history": _development_history(context),
        "parent_control": None if context.is_initial else RUNTIME_PLACEHOLDER,
        "workspace": workspace,
        "forbidden": [
            "current_test",
            "future_data",
            "heldout",
            "external_network",
            "host_control",
        ],
    }
    _mark_runtime_only(facts)
    if not context.is_initial:
        _mark_runtime_parent(facts, model_artifacts=True)
    system = build_system_prompt(
        rolling.schedule,
        mode="fold",
        experiment_facts=facts,
        phase=str(manifest["phase"]),
        step_tree_enabled=rolling.step_tree_enabled,
        prior_prompt=context.prior,
        fold_exploration_directive=rolling.fold_exploration_directive,
        fold_directive=directive,
    )
    return system, prompt_override.strip() or FOLD_DEFAULT_INSTRUCTION


def _meta_prompt(
    context: _SessionContext,
    *,
    trigger_after_folds: int,
    directive: str,
    prompt_override: str,
) -> tuple[str, str]:
    from autotrade.pipelines.meta_schedule import meta_learning_id

    rolling = context.rolling
    limits = context.strategy_limits
    session_id = meta_learning_id(context.epoch_id, trigger_after_folds)
    manifest: dict[str, object] = {
        "experiment_id": rolling.experiment_id,
        "epoch_id": context.epoch_id,
        "meta_learning_id": session_id,
        "trigger_after_folds": trigger_after_folds,
        "fold_id": session_id,
        "kind": "meta_learning",
        "development_inputs": {
            "meta_context": "/mnt/agent/workspace/inputs/meta_context.json",
            "meta_learning_memory": "/mnt/agent/workspace/inputs/meta_learning_memory.jsonl",
            "agent_traces": "/mnt/agent/workspace/inputs/agent_traces",
            "agent_trace_full": {
                "directory": "/mnt/agent/workspace/inputs/agent_traces",
                # How many completed Folds the review window covers is decided
                # when the session is assembled.
                "available": RUNTIME_PLACEHOLDER,
                "fold_count": RUNTIME_PLACEHOLDER,
            },
            "previous_prior": bool(context.prior.strip()),
        },
        "prior_output": "/mnt/agent/workspace/PRIOR.md",
        "is_initial_artifact": context.is_initial,
        "template_ref": "agent_output_template" if context.is_initial else None,
        "modification_constraints": rolling.regularization_constraints.to_record(),
        "meta_learning_directive": rolling.meta_learning_directive,
        "fold_exploration_directive": rolling.fold_exploration_directive,
        "budgets": {
            "max_llm_calls": rolling.max_llm_calls,
            "deadline_seconds": rolling.max_fold_minutes * 60,
            "strategy_inference_timeout_seconds": limits.timeout_seconds,
            "strategy_fit_timeout_seconds": limits.fit_timeout_seconds,
        },
    }
    facts = build_experiment_facts(manifest=manifest, ref_store=context.ref_store)
    _mark_runtime_only(facts)
    if not context.is_initial:
        _mark_runtime_parent(facts, model_artifacts=False)
    # A Meta session is prompted with no schedule block: it never runs a replay.
    system = build_system_prompt(mode="meta", experiment_facts=facts)
    instruction = prompt_override.strip() or build_meta_learning_prompt(
        {},
        experiment_directive=rolling.meta_learning_directive,
        fold_exploration_directive=rolling.fold_exploration_directive,
    )
    if directive.strip():
        instruction += f"\n\nSupervising user directive:\n{directive.strip()}"
    return system, instruction


def _session_entry(experiment_dir: Path, session_key: str) -> dict[str, object]:
    schedule_plan = read_json(experiment_dir / HITL_DIR_NAME / SCHEDULE_NAME)
    raw_sessions = schedule_plan.get("sessions")
    sessions: list[object] = raw_sessions if isinstance(raw_sessions, list) else []
    for item in sessions:
        if not isinstance(item, dict):
            continue
        if str(item.get("session_key") or item.get("key") or "") != session_key:
            continue
        if str(item.get("kind") or "") == "heldout":
            raise ValueError("held-out runs have no agent session or system prompt")
        return item
    if session_key == "heldout":
        raise ValueError("held-out runs have no agent session or system prompt")
    raise KeyError(f"unknown session: {session_key}")


def _fold_spec(options: InteractiveWorkerOptions, fold_id: str) -> FoldSpec:
    """The FoldSpec this session will receive, from the pipeline's own schedule.

    Same inputs as the worker: the experiment's pinned research release and its
    daily trading dates, so the visible windows and the PIT decision anchor are
    the ones the session is actually given.
    """
    from autotrade.environment.data.pit import PITDataStore
    from autotrade.environment.data.research_release import pin_research_release
    from autotrade.pipelines.folds import build_fold_schedule
    from autotrade.pipelines.pit_backend import required_release_raw_datasets

    if options.data_backend != "pit":
        raise ValueError("prompt preview requires the PIT data backend")
    release = pin_research_release(
        experiment_dir=options.experiment_dir,
        raw_dir=options.raw_dir,
        fundamental_events_root=options.fundamental_events_root,
        fundamental_events_status=options.fundamental_events_status,
        required_raw_datasets=required_release_raw_datasets(options.snapshot_config),
    )
    trading_days = PITDataStore(release.raw_dir).trade_dates("daily")
    folds = build_fold_schedule(
        options.rolling.development_first_period,
        options.rolling.development_last_period,
        trading_days,
        window_months=options.rolling.window_months,
        period=options.rolling.fold_period,
        min_region_trade_days=options.rolling.min_region_trade_days,
        test_stage=options.rolling.test_stage,
        validation_periods=options.rolling.validation_periods,
    )
    for fold in folds:
        if fold.fold_id == fold_id:
            return fold
    raise KeyError(f"unknown fold: {fold_id}")


def _development_history(context: _SessionContext) -> list[dict[str, object]]:
    from autotrade.pipelines.agent_views import compact_fold_history
    from autotrade.pipelines.ledger import latest_fold_records

    return [
        compact_fold_history(record, ref_store=context.ref_store)
        for record in latest_fold_records(context.records).values()
    ]


def _has_parent_artifact(experiment_dir: Path, records: list[dict[str, object]]) -> bool:
    """Whether this experiment already has a strategy to inherit."""
    if read_json(experiment_dir / HITL_DIR_NAME / PARAMS_NAME).get(
        "_inherited_artifact"
    ):
        return True
    return any(record.get("frozen_strategy_artifact_id") for record in records)


def _phase(epoch_index: int, convergence_start_epoch: int) -> str:
    return "convergence" if epoch_index >= convergence_start_epoch else "exploration"


def _mark_runtime_only(facts: dict[str, object]) -> None:
    identity = facts.get("identity")
    if isinstance(identity, dict):
        identity["run_id"] = RUNTIME_PLACEHOLDER
    timeline = facts.get("visible_timeline")
    policy = timeline.get("execution_policy") if isinstance(timeline, dict) else None
    if isinstance(policy, dict):
        # Which PIT files the session can read is a property of the snapshot the
        # host builds at session start, not of anything readable now.
        for key in (
            "historical_minutes_available",
            "auction_available",
            "events_available",
            "text_available",
        ):
            policy[key] = RUNTIME_PLACEHOLDER


def _mark_runtime_parent(facts: dict[str, object], *, model_artifacts: bool) -> None:
    """Which frozen artifact this session inherits depends on the sessions that
    still run before it, so only its presence is stated ahead of time."""
    contract = facts.get("artifact_contract")
    parent = contract.get("parent") if isinstance(contract, dict) else None
    if not isinstance(parent, dict):
        return
    parent["id"] = RUNTIME_PLACEHOLDER
    if model_artifacts:
        parent["model_artifacts_empty"] = RUNTIME_PLACEHOLDER
