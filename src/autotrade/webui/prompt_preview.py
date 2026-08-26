"""Pre-approval system-prompt preview for Fold and Meta sessions."""

from __future__ import annotations

from pathlib import Path

from autotrade.agent.prompts import build_meta_learning_prompt, build_system_prompt
from autotrade.environment.identity import agent_visible_ref
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines.hitl_state import (
    HITL_DIR_NAME,
    PARAMS_NAME,
    SCHEDULE_NAME,
    read_json,
)
from autotrade.pipelines.prior import latest_prior_text

from .registry import read_ledger_records

PREVIEW_NOTE = "预览由持久参数、会话计划与 Development Ledger 组装；运行时才产生的沙箱事实不会在此伪造。"


def build_prompt_preview(
    experiment_dir: Path, session_key: str, directive: str
) -> dict[str, object]:
    """Raises KeyError for an unknown session and ValueError for held-out keys."""
    hitl = Path(experiment_dir) / HITL_DIR_NAME
    schedule_plan = read_json(hitl / SCHEDULE_NAME)
    sessions = (
        schedule_plan.get("sessions")
        if isinstance(schedule_plan.get("sessions"), list)
        else []
    )
    entry = next(
        (
            item
            for item in sessions
            if isinstance(item, dict)
            and str(item.get("session_key") or item.get("key") or "") == session_key
        ),
        None,
    )
    if entry is None:
        if session_key == "heldout":
            raise ValueError("held-out runs have no agent session or system prompt")
        raise KeyError(f"unknown session: {session_key}")
    kind = str(entry.get("kind") or "")
    if kind == "heldout":
        raise ValueError("held-out runs have no agent session or system prompt")
    params = read_json(hitl / PARAMS_NAME)
    timing = StrategySchedule(
        period=str(params.get("strategy_period") or "day"),  # type: ignore[arg-type]
        inference_time=str(params.get("inference_time") or "08:30"),
    )
    records = read_ledger_records(experiment_dir)
    previous_prior = latest_prior_text(records, experiment_dir=experiment_dir)
    if kind in {"meta", "meta_learning"}:
        prompt = "\n\n".join(
            [
                build_system_prompt(timing, mode="meta"),
                build_meta_learning_prompt(None),
                f"研究者指令：{directive.strip()}"
                if directive.strip()
                else "研究者指令：无额外指令。",
            ]
        )
        return {"kind": "meta_learning", "prompt": prompt, "note": PREVIEW_NOTE}
    if kind != "fold":
        raise ValueError(f"unsupported session kind: {kind}")
    facts = {
        key: entry.get(key)
        for key in ("epoch_id", "fold_id", "fold_index")
        if entry.get(key) is not None
    }
    if "fold_id" in facts:
        facts["fold_id"] = agent_visible_ref(facts["fold_id"], prefix="fold_ref")
    facts.update(
        {
            "max_steps": params.get("max_steps_per_fold", 10),
            "max_backtests": params.get("max_backtests_per_fold", 30),
            "max_llm_calls": params.get("max_llm_calls", 200),
            "deadline_seconds": int(params.get("max_fold_minutes", 240)) * 60,  # type: ignore[arg-type]
        }
    )
    prompt = build_system_prompt(
        timing, mode="fold", experiment_facts=facts, prior_prompt=previous_prior
    )
    prompt += (
        f"\n\n研究者本 Fold 指令：{directive.strip()}"
        if directive.strip()
        else "\n\n研究者本 Fold 指令：无额外指令。"
    )
    return {"kind": "fold", "prompt": prompt, "note": PREVIEW_NOTE}
