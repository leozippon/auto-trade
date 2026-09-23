"""Pre-session Prompt preview for the research session.

The console shows this text before the session starts, so it has to be the
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
id, the sandbox runtime environment and which PIT files the session's snapshot
will carry. Those render as ``RUNTIME_PLACEHOLDER`` instead of being invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import SESSION_DEFAULT_INSTRUCTION, build_system_prompt
from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.config import BudgetUsed
from autotrade.pipelines.hitl_state import (
    CONTROL_NAME,
    HITL_DIR_NAME,
    SCHEDULE_NAME,
    read_control,
    read_json,
)

from .registry import read_ledger_records

if TYPE_CHECKING:
    from autotrade.environment.sandbox import SandboxLimits
    from autotrade.pipelines.worker import InteractiveWorkerOptions

# Marks a fact that exists only once the session starts. Short and unmistakable
# inside the facts JSON, and explained by PREVIEW_NOTE above the preview.
RUNTIME_PLACEHOLDER = "<runtime>"

PREVIEW_NOTE = (
    "预览走 worker 同一条装配链：hitl/params.json → load_worker_options → "
    "build_experiment_facts → build_system_prompt。"
    "只有会话启动时才产生的事实（run id、沙箱 runtime env、快照实际挂载的数据）"
    f"显示为 {RUNTIME_PLACEHOLDER}。"
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
    _require_research_session(directory, session_key)
    options = load_worker_options(directory, repo_root=repo_root)
    control = read_control(directory / HITL_DIR_NAME / CONTROL_NAME)
    context = _SessionContext(
        options=options,
        ref_store=ref_store,
        records=read_ledger_records(directory),
        session_key=session_key,
    )
    system = _research_prompt(
        context,
        directive=directive,
        resource_override=control.resource_overrides.get(session_key),
    )
    prompt = f"{_SYSTEM_BANNER}\n{system}\n\n{_USER_BANNER}\n{SESSION_DEFAULT_INSTRUCTION}"
    return {"prompt": prompt, "note": PREVIEW_NOTE}


@dataclass(frozen=True)
class _SessionContext:
    options: InteractiveWorkerOptions
    ref_store: AgentRefStore
    records: list[dict[str, object]]
    session_key: str

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
    from autotrade.pipelines.experiment import _session_budgets
    from autotrade.pipelines.local_backend import (
        arm_record,
        research_geometry_record,
        session_fact_blocks,
        start_record,
    )

    rolling = context.rolling
    geometry = rolling.geometry
    budgets = _session_budgets(rolling, resource_override)
    limits = context.strategy_limits
    manifest: dict[str, object] = {
        "experiment_id": rolling.experiment_id,
        "epoch_id": "research",
        "fold_id": context.session_key,
        "kind": "research",
        "research": research_geometry_record(
            geometry.research_years,
            window_months=rolling.window_months,
            decision_time=geometry.research_decision_time.isoformat(),
        ),
        "benchmark_index": rolling.benchmark_index,
        "snapshot_config": context.options.snapshot_config.to_record(),
        "start": start_record(),
        "arm": arm_record(()),
        "modification_constraints": rolling.step_constraints.to_record(),
        "acceptance_rules": rolling.acceptance.to_record(),
        "schedule": rolling.schedule.to_record(),
        "broker_profile": rolling.broker_profile.to_record(),
        "nl_failure_policy": rolling.nl_failure_policy,
        "record_failed_attempts": rolling.record_failed_attempts,
        "attempt": 1,
        "finalize_before_deadline_seconds": rolling.finalize_before_deadline_seconds,
        "sandbox_spec": (
            context.options.agent_sandbox.to_record()
            if context.options.agent_sandbox is not None
            else None
        ),
        "budgets": {
            "max_replay_years": budgets["max_replay_years"],
            "max_null_controls": rolling.max_null_controls,
            "max_llm_calls": budgets["max_llm_calls"],
            "deadline_seconds": budgets["deadline_seconds"],
            "deadline_grace_seconds": budgets["deadline_grace_seconds"],
            "strategy_inference_timeout_seconds": limits.timeout_seconds,
            "strategy_fit_timeout_seconds": limits.fit_timeout_seconds,
            "strategy_gpu_count": limits.gpu_count,
            "used_before_this_attempt": BudgetUsed().to_record(),
        },
    }
    # The blocks LLMResearchDeveloper._session_facts adds beside the shared
    # projection: the fixed workspace/boundary index.
    blocks = session_fact_blocks(Path("/nonexistent/prompt-preview-workspace"))
    if rolling.workspace_reference:
        blocks["workspace"]["refs"] = "refs/"  # type: ignore[index]
    facts: dict[str, object] = {
        **build_experiment_facts(
            manifest=manifest,
            ref_store=context.ref_store,
            max_llm_calls=int(budgets["max_llm_calls"]),
            context_compaction=context.context_compaction,
            model_artifacts_empty=True,
        ),
        **blocks,
    }
    _mark_runtime_only(facts)
    return build_system_prompt(
        rolling.schedule,
        experiment_facts=facts,
        exploration_directive=rolling.research_directive,
        session_directive=directive,
    )


def _require_research_session(experiment_dir: Path, session_key: str) -> None:
    schedule_plan = read_json(experiment_dir / HITL_DIR_NAME / SCHEDULE_NAME)
    raw_sessions = schedule_plan.get("sessions")
    sessions: list[object] = raw_sessions if isinstance(raw_sessions, list) else []
    for item in sessions:
        if not isinstance(item, dict) or str(item.get("session_key") or "") != session_key:
            continue
        if str(item.get("kind") or "") != "research":
            raise ValueError("the forward replay has no agent session or system prompt")
        return
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
