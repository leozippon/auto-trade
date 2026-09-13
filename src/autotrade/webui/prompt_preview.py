"""Pre-session Prompt preview for research sessions.

The console shows this text before a session starts, so it has to be the
prompt the session will actually receive rather than a second description of
it. Nothing here restates prompt text, budgets or contract wording: the
experiment is resolved by the worker's own ``load_worker_options`` over the
persisted parameters, the research window comes from the experiment's own
``ResearchGeometry``, the Agent-visible facts come from the shared
``build_experiment_facts`` projection of a run manifest shaped like the one a
session writes, and the text is rendered by ``build_system_prompt``. A change
to ``prompts.py``, to a budget default or to the facts projection therefore
reaches this preview with no edit here.

Only what comes into existence when the session starts is unavailable: the run
id, the sandbox runtime environment, which PIT files the session's snapshot
will carry, and the node the session starts from when an earlier session has
not recorded it yet. Those render as ``RUNTIME_PLACEHOLDER`` instead of being
invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import FOLD_DEFAULT_INSTRUCTION, build_system_prompt
from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.hitl_state import (
    CONTROL_NAME,
    HITL_DIR_NAME,
    SCHEDULE_NAME,
    read_control,
    read_json,
)
from autotrade.pipelines.ledger import research_records
from autotrade.pipelines.prior import latest_prior_text

from .registry import read_ledger_records

if TYPE_CHECKING:
    from autotrade.environment.sandbox import SandboxLimits
    from autotrade.pipelines.worker import InteractiveWorkerOptions

# Marks a fact that exists only once the session starts. Short and unmistakable
# inside the facts JSON, and explained by PREVIEW_NOTE above the preview.
RUNTIME_PLACEHOLDER = "<runtime>"

PREVIEW_NOTE = (
    "预览走 worker 同一条装配链：hitl/params.json → load_worker_options → "
    "build_experiment_facts → build_system_prompt，此前会话的结果与当前 PRIOR 一并注入。"
    "只有会话启动时才产生的事实（run id、沙箱 runtime env、快照实际挂载的数据、"
    f"尚未由前一会话记下的起点节点）显示为 {RUNTIME_PLACEHOLDER}。"
)

_SYSTEM_BANNER = "======== 系统提示词（build_system_prompt）========"
_USER_BANNER = "======== 开局用户消息 ========"


def build_prompt_preview(
    experiment_dir: Path, session_key: str, directive: str, *, repo_root: Path
) -> dict[str, object]:
    """Raises KeyError for an unknown session and ValueError for the forward replay."""
    # Deferred: pulls the pipeline/worker stack, which the console only needs
    # when a researcher actually opens a preview.
    from autotrade.pipelines.worker import load_worker_options

    directory = Path(experiment_dir)
    # The same store a session projects its identities through, constructed
    # before anything else so a legacy experiment fails here exactly as a real
    # session would.
    ref_store = AgentRefStore(directory)
    entry = _session_entry(directory, session_key)
    options = load_worker_options(directory, repo_root=repo_root)
    control = read_control(directory / HITL_DIR_NAME / CONTROL_NAME)
    records = read_ledger_records(directory)
    index = int(entry.get("index") or 0)
    done = research_records(records)
    # The node this session starts from: the template for the first session,
    # the node its predecessor handed on once that one is recorded.
    if index == 1:
        start: object = None
    elif len(done) >= index - 1:
        start = done[index - 2].get("next_start_node_id")
    else:
        start = RUNTIME_PLACEHOLDER
    context = _SessionContext(
        options=options,
        ref_store=ref_store,
        records=records,
        prior=latest_prior_text(records),
        session_key=session_key,
        index=index,
        is_initial=start is None,
    )
    system = _research_prompt(
        context,
        directive=directive,
        resource_override=control.resource_overrides.get(session_key),
    )
    prompt = f"{_SYSTEM_BANNER}\n{system}\n\n{_USER_BANNER}\n{FOLD_DEFAULT_INSTRUCTION}"
    return {"prompt": prompt, "note": PREVIEW_NOTE}


@dataclass(frozen=True)
class _SessionContext:
    options: InteractiveWorkerOptions
    ref_store: AgentRefStore
    records: list[dict[str, object]]
    prior: str
    session_key: str
    index: int
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


def _research_prompt(
    context: _SessionContext,
    *,
    directive: str,
    resource_override: object,
) -> str:
    from autotrade.pipelines.experiment import _months_before, _session_budgets
    from autotrade.pipelines.local_backend import (
        FOLD_FORBIDDEN,
        fold_workspace_map,
        research_history,
    )

    rolling = context.rolling
    geometry = rolling.geometry
    budgets = _session_budgets(rolling, resource_override)
    limits = context.strategy_limits
    decision_time = geometry.research_decision_time.isoformat()
    manifest: dict[str, object] = {
        "experiment_id": rolling.experiment_id,
        "epoch_id": "research",
        "fold_id": context.session_key,
        "kind": "fold",
        "session_index": context.index,
        "sessions_total": rolling.research_sessions,
        "fold": {
            "input_window": (
                f"{_months_before(geometry.research_end, rolling.window_months)}.."
                f"{geometry.research_end}"
            ),
            "validation_period": f"{geometry.research_start}..{geometry.research_end}",
        },
        "valid_decision_time": decision_time,
        "snapshot_config": context.options.snapshot_config.to_record(),
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
            "max_null_controls_per_fold": rolling.max_null_controls_per_fold,
            "max_llm_calls": budgets["max_llm_calls"],
            "deadline_seconds": budgets["deadline_seconds"],
            "deadline_grace_seconds": budgets["deadline_grace_seconds"],
            "strategy_inference_timeout_seconds": limits.timeout_seconds,
            "strategy_fit_timeout_seconds": limits.fit_timeout_seconds,
            "strategy_gpu_count": limits.gpu_count,
        },
    }
    workspace = fold_workspace_map(Path("/nonexistent/prompt-preview-workspace"))
    if rolling.workspace_reference:
        workspace["refs"] = "refs/"
    # The blocks LLMFoldDeveloper._fold_facts adds beside the shared
    # projection: the earlier sessions' outcomes and the fixed
    # workspace/boundary index.
    facts: dict[str, object] = {
        **build_experiment_facts(
            manifest=manifest,
            ref_store=context.ref_store,
            max_llm_calls=int(budgets["max_llm_calls"]),
            context_compaction=context.context_compaction,
            model_artifacts_empty=True if context.is_initial else None,
        ),
        "development_history": research_history(context.records),
        "workspace": workspace,
        "forbidden": FOLD_FORBIDDEN,
    }
    _mark_runtime_only(facts)
    if not context.is_initial:
        _mark_runtime_start(facts)
    return build_system_prompt(
        rolling.schedule,
        mode="fold",
        experiment_facts=facts,
        step_tree_enabled=rolling.step_tree_enabled,
        prior_prompt=context.prior,
        fold_exploration_directive=rolling.fold_exploration_directive,
        fold_directive=directive,
    )


def _session_entry(experiment_dir: Path, session_key: str) -> dict[str, object]:
    schedule_plan = read_json(experiment_dir / HITL_DIR_NAME / SCHEDULE_NAME)
    raw_sessions = schedule_plan.get("sessions")
    sessions: list[object] = raw_sessions if isinstance(raw_sessions, list) else []
    for item in sessions:
        if not isinstance(item, dict) or str(item.get("session_key") or "") != session_key:
            continue
        if str(item.get("kind") or "") != "research":
            raise ValueError("the forward replay has no agent session or system prompt")
        return item
    raise KeyError(f"unknown session: {session_key}")


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


def _mark_runtime_start(facts: dict[str, object]) -> None:
    """Which node the session starts from is the session's own fact, so only
    its presence is stated ahead of time."""
    contract = facts.get("artifact_contract")
    parent = contract.get("parent") if isinstance(contract, dict) else None
    if not isinstance(parent, dict):
        return
    parent["id"] = RUNTIME_PLACEHOLDER
    parent["model_artifacts_empty"] = RUNTIME_PLACEHOLDER
