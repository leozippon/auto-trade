"""Agent session runner: the main conversation loop for one Fold or meta-learning run.

docs/agent-design.md plus docs/environment-design.md §2.2 define the Agent
session and tool-entrypoint contract: one Agent session per Fold (one
conversation_id), Steps share the session, only documented tools are
callable, and the fold deadline is the master constraint. The session budget
the pipeline hands over includes a trailing wrap-up grace window
(``deadline_grace_seconds``): reaching the main deadline never interrupts the
model — with no complete Validation a single wrap-up prompt is injected and the
session keeps its full autonomy through the grace window; when the grace window
is exhausted the session closes gracefully (``session_end`` with status
``deadline_exceeded``) and the Pipeline records a no-candidate Fold instead of
failing the run. Inside the finalize window a current-run complete Validation
instead switches to the restricted hard-finalization capability view. Main
conversation calls and semantic compactions are logged in agent_trace.jsonl
(docs/environment-design.md §2.4 and §4.2).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from autotrade.environment.llm import (
    ChatMessage,
    LLMProxy,
    ToolCall,
    context_overflow_error,
    context_request_fits,
    is_context_overflow_error,
)
from autotrade.environment.runtime import sanitize_for_log, utc_now_iso
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
    TimeBudgetBinding,
    validate_time_budget_bindings,
)
from autotrade.environment.tools import (
    SafeWorkspace,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from autotrade.environment.tools.base import SessionInterrupt

from .compact import (
    ContextCompactor,
    drop_leading_orphan_tools,
    estimate_messages_tokens,
    fit_tool_results_to_context,
    is_compaction_message,
    is_llm_compaction_message,
    safe_error_summary,
)
from .explore import EXPLORE_ROLES, ExploreSubAgentEngine
from .prompts import (
    HARD_FINALIZATION_SYSTEM_PROMPT,
    STEP_WRAP_UP_PROMPT,
    WRAP_UP_PROMPT,
)

_LLM_FAILURE_CIRCUIT = 3

# Default wrap-up grace shared with RollingExperimentConfig.deadline_grace_minutes:
# the pipeline hands the session a budget of main deadline + grace and the runner
# reserves the trailing grace for wrap-up. Keep the two defaults aligned.
DEFAULT_DEADLINE_GRACE_SECONDS = 600.0


class AgentSessionDeadlineExceeded(SessionInterrupt):
    """The fold deadline and its wrap-up grace window were both exhausted.

    Control flow, not an error: the session has already emitted
    ``session_end{status: deadline_exceeded}``; the Pipeline converts this into
    a recorded no-candidate Fold/Meta outcome and continues the experiment
    instead of failing the run. A SessionInterrupt subclass so it re-raises
    through tool dispatch instead of being swallowed into an error observation.
    """

    def __init__(
        self,
        message: str = "Agent session deadline and wrap-up grace exhausted",
        *,
        conversation_id: str = "",
        llm_calls: int = 0,
    ) -> None:
        super().__init__(message)
        self.conversation_id = conversation_id
        self.llm_calls = llm_calls


_TERMINAL_TOOLS = frozenset({"finish_fold", "finish_meta"})
_FOLD_FINALIZATION_TOOLS = frozenset({"finish_fold", "step_rollback"})
_PARALLEL_READ_TOOLS = frozenset(
    {"glob", "grep", "modification_check", "nl_query", "read_file", "validate_strategy"}
)
_FOLD_TOOLS = frozenset(
    {
        "ask_user",
        "daily_backtest",
        "finish_fold",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "step_rollback",
        "todo",
        "validate_strategy",
    }
)
_META_TOOLS = frozenset(
    {
        "ask_user",
        "edit_file",
        "finish_meta",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "todo",
        "write_file",
        "write_taste",
    }
)
_CLEARED_TOOL_RESULT = json.dumps(
    {
        "observation": "cleared",
        "note": "旧工具原始结果已清理以节省上下文；必要结论保留在当前会话摘要与结果制品中。",
    },
    ensure_ascii=False,
)
INBOX_SAFE_BEFORE_LLM = "before_llm"
INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS = "after_llm_before_tools"
INBOX_SAFE_BETWEEN_SERIAL_TOOLS = "between_serial_tools"
INBOX_SAFE_AFTER_PARALLEL_READONLY = "after_parallel_readonly"
INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM = "after_tools_before_llm"
_INBOX_TRACE_CHARS = 400
_INTERRUPTED_BY_USER = "interrupted_by_user"


class AgentInboxHook(Protocol):
    """Current-session unconsumed notices; consume is atomic per run."""

    def pending(self) -> Sequence[object]: ...

    def consume(self, message_id: str) -> str: ...


def _inbox_trace_text(text: str) -> str:
    redacted = sanitize_for_log(text)
    if not isinstance(redacted, str):
        redacted = str(redacted)
    if len(redacted) > _INBOX_TRACE_CHARS:
        return redacted[:_INBOX_TRACE_CHARS]
    return redacted


@dataclass(frozen=True)
class AgentSessionConfig:
    mode: str = "fold"
    finalize_before_deadline_seconds: float = 300.0
    # Trailing wrap-up grace reserved from the end of the handed session
    # budget. The main deadline sits grace seconds before the budget end:
    # reaching it never interrupts the model; exhausting the budget does.
    deadline_grace_seconds: float = DEFAULT_DEADLINE_GRACE_SECONDS
    max_llm_calls: int = 200
    max_steps: int = 10
    deadline_seconds: float = 1_200.0
    max_response_tokens: int = 8_000
    max_history_messages: int = 150
    trim_message_headroom: int = 30
    trim_token_threshold: int = 60_000
    context_summary_max_items: int = 30
    context_summary_max_chars: int = 6_000
    clear_tool_results: bool = True
    tool_result_keep_recent: int = 8
    tool_result_clear_min_chars: int = 4_000
    tool_result_clear_token_threshold: int = 40_000

    def __post_init__(self) -> None:
        if self.mode not in ("fold", "meta", "meta_learning"):
            raise ValueError("Agent session mode must be fold, meta, or meta_learning")
        for name in (
            "max_llm_calls",
            "max_steps",
            "deadline_seconds",
            "max_response_tokens",
            "max_history_messages",
            "trim_token_threshold",
            "context_summary_max_items",
            "context_summary_max_chars",
            "tool_result_clear_min_chars",
            "tool_result_clear_token_threshold",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.finalize_before_deadline_seconds < 0
            or self.deadline_grace_seconds < 0
            or self.trim_message_headroom < 0
            or self.tool_result_keep_recent < 0
        ):
            raise ValueError(
                "session reserve and context editing counts cannot be negative"
            )


@dataclass(frozen=True)
class AgentSessionResult:
    conversation_id: str
    status: str
    finish_value: dict[str, object]
    llm_calls: int
    # Closed's session-summary ``token_usage`` block: the seven per-call totals
    # plus ``cache_hit_ratio`` and the Explore Sub Agent roll-up.
    usage: dict[str, object] = field(default_factory=dict)
    context_compactions: int = 0
    steps_used: int = 0


class AgentSessionRunner:
    """Drive one persistent native-tool Fold or Meta conversation."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        tools: ToolRegistry,
        system_prompt: str,
        config: AgentSessionConfig | None = None,
        compactor: ContextCompactor | None = None,
        explore: ExploreSubAgentEngine | None = None,
        time_budget: InferenceTimeBudget | None = None,
        conversation_id: str | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
        inbox: AgentInboxHook | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config or AgentSessionConfig()
        self.compactor = compactor
        self.explore = explore
        if self.explore is not None:
            explore_mode = getattr(self.explore, "mode", "fold")
            if self.config.mode in {"meta", "meta_learning"}:
                if explore_mode != "meta":
                    raise ValueError("Meta session explore sub-agent must use mode='meta'")
            elif explore_mode != "fold":
                raise ValueError("Fold session explore sub-agent must use mode='fold'")
        if self.explore is not None and self.explore.event_sink is None:
            self.explore.event_sink = event_sink
        bindings: list[TimeBudgetBinding] = []
        if isinstance(llm, SessionTimeBudgetAware):
            bindings.append(TimeBudgetBinding("main_llm", llm.session_time_budget))
        if explore is not None and explore.session_time_budget is not None:
            bindings.append(TimeBudgetBinding("explore", explore.session_time_budget))
        if compactor is not None:
            bindings.append(
                TimeBudgetBinding("compactor", compactor.session_time_budget)
            )
        bindings.extend(tools.time_budget_bindings())
        self.time_budget = validate_time_budget_bindings(
            time_budget, tuple(bindings), owner="Agent runner"
        )
        self.conversation_id = conversation_id or f"conversation_{uuid.uuid4().hex}"
        self.event_sink = event_sink
        self.inbox = inbox
        self._observation_summaries: list[dict[str, object]] = []
        self._complete_validation_nodes: list[dict[str, object]] = []
        self._hard_finalization = False
        self._hard_finalization_context_initialized = False
        self._wrap_up_sent = False
        self._explore_attempts = 0
        self._explored_roles: set[str] = set()
        self._validate_capability_boundary()

    def run(self, instruction: str) -> AgentSessionResult:
        if not instruction.strip():
            raise ValueError("Agent instruction cannot be empty")
        time_budget = self.time_budget or InferenceTimeBudget(
            duration_seconds=self.config.deadline_seconds
        )
        messages = [
            ChatMessage("system", self.system_prompt),
            ChatMessage("user", instruction.strip()),
        ]
        usage = _new_token_totals()
        explore_totals: dict[str, int] | None = None
        llm_calls = 0
        accepted_steps = 0
        step_wrap_up_sent = False
        llm_failure_streak = 0
        context_overflow_recovery_used = False
        self._complete_validation_nodes = []
        self._hard_finalization = False
        self._hard_finalization_context_initialized = False
        self._wrap_up_sent = False
        self._explore_attempts = 0
        self._explored_roles = set()
        self._emit(
            "session_start",
            {
                "mode": self.config.mode,
                "system_prompt": self.system_prompt,
                "instruction": instruction.strip(),
            },
        )

        while llm_calls < self.config.max_llm_calls:
            remaining = time_budget.remaining()
            if remaining <= 0:
                self._emit(
                    "session_end",
                    {"status": "deadline_exceeded", "llm_calls": llm_calls},
                )
                raise AgentSessionDeadlineExceeded(
                    conversation_id=self.conversation_id, llm_calls=llm_calls
                )
            self._activate_hard_finalization_if_ready(remaining)
            if self._hard_finalization:
                if not self._hard_finalization_context_initialized:
                    messages = self._hard_finalization_messages(remaining)
            else:
                if (
                    self.config.mode == "fold"
                    and not self._wrap_up_sent
                    and remaining <= self.config.deadline_grace_seconds
                ):
                    messages.append(ChatMessage("user", WRAP_UP_PROMPT))
                    self._wrap_up_sent = True
                    self._emit(
                        "wrap_up_started",
                        {
                            "remaining_seconds": round(remaining, 6),
                            "grace_seconds": self.config.deadline_grace_seconds,
                        },
                    )
                messages, _ = self._compact_if_needed(messages, remaining)
            messages = self._clear_stale_tool_results(messages)
            messages = self._trim(messages)
            provider_tools = self._provider_tools()
            messages = self._apply_inbox(
                messages, safe_point=INBOX_SAFE_BEFORE_LLM
            )

            try:
                messages = self._prepare_context_request(
                    messages,
                    provider_tools,
                    max(time_budget.remaining(), 0.0),
                    allow_semantic_compaction=not self._hard_finalization,
                )
                self._emit(
                    "llm_call_started",
                    {
                        "call_index": llm_calls + 1,
                        "status": "running",
                        "provider": getattr(self.llm, "provider", ""),
                        "model": getattr(self.llm, "model", ""),
                    },
                )
                response = self.llm.complete(
                    messages,
                    tools=provider_tools,
                    tool_choice="auto",
                    max_tokens=self.config.max_response_tokens,
                )
                llm_calls += 1
                llm_failure_streak = 0
            except Exception as exc:
                llm_calls += 1
                llm_failure_streak += 1
                error = safe_error_summary(exc)
                self._emit(
                    "llm_call",
                    {
                        "call_index": llm_calls,
                        "status": "error",
                        "provider": getattr(self.llm, "provider", ""),
                        "model": getattr(self.llm, "model", ""),
                        "error": error,
                    },
                )
                if is_context_overflow_error(exc):
                    if not context_overflow_recovery_used:
                        messages, progressed = self._recover_context_overflow(
                            messages,
                            provider_tools,
                            max(time_budget.remaining(), 0.0),
                            allow_semantic_compaction=not self._hard_finalization,
                        )
                        if progressed:
                            context_overflow_recovery_used = True
                            llm_failure_streak = 0
                            continue
                    self._emit(
                        "session_end",
                        {"status": "context_window_exceeded", "llm_calls": llm_calls},
                    )
                    raise RuntimeError(
                        "Agent context window cannot be reduced safely"
                    ) from exc
                if isinstance(exc, TimeoutError) or time_budget.remaining() <= 0:
                    self._emit(
                        "session_end",
                        {"status": "deadline_exceeded", "llm_calls": llm_calls},
                    )
                    raise AgentSessionDeadlineExceeded(
                        conversation_id=self.conversation_id, llm_calls=llm_calls
                    ) from exc
                if llm_failure_streak >= _LLM_FAILURE_CIRCUIT:
                    self._emit(
                        "session_end",
                        {"status": "llm_unavailable", "llm_calls": llm_calls},
                    )
                    raise RuntimeError(
                        "Agent language model unavailable after consecutive failures"
                    ) from exc
                observation: dict[str, object] = {
                    "observation": "llm_error",
                    "error": error,
                    "retry_hint": "Proceed with a different bounded action; do not repeat the same failing request verbatim.",
                }
                messages.append(
                    ChatMessage(
                        "user",
                        json.dumps(observation, ensure_ascii=False, allow_nan=False),
                    )
                )
                self._remember_observation("llm_call", observation)
                continue

            _accumulate_usage(usage, response.usage)
            messages.append(
                ChatMessage(
                    "assistant",
                    response.content or None,
                    response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
            )
            self._emit(
                "llm_call",
                {
                    "call_index": llm_calls,
                    "status": "ok",
                    "provider": getattr(self.llm, "provider", ""),
                    "model": response.model,
                    "usage": dict(response.usage),
                    "content": response.content,
                    "tool_names": [call.name for call in response.tool_calls],
                },
            )
            if not response.tool_calls:
                nudge: dict[str, object] = {
                    "observation": "no_tool_call",
                    "retry_hint": "Use an injected tool to advance the session; text alone does not finish it.",
                }
                messages.append(
                    ChatMessage("user", json.dumps(nudge, ensure_ascii=False))
                )
                self._remember_observation("llm_call", nudge)
                messages = self._apply_inbox(
                    messages, safe_point=INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS
                )
                continue

            if self._inbox_interrupt_pending():
                results = self._skip_tool_calls(
                    response.tool_calls,
                    safe_point=INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS,
                )
                apply_point = INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS
            else:
                parallel = self._is_parallel_readonly_batch(response.tool_calls)
                results, skipped_at = self._dispatch_tool_calls(
                    response.tool_calls, time_budget
                )
                if skipped_at:
                    apply_point = skipped_at
                elif parallel:
                    apply_point = INBOX_SAFE_AFTER_PARALLEL_READONLY
                else:
                    apply_point = INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM
            first_new_tool_index = len(messages)
            for call, record in results:
                messages.append(
                    ChatMessage(
                        "tool",
                        json.dumps(
                            sanitize_for_log(record),
                            ensure_ascii=False,
                            default=str,
                            allow_nan=False,
                        ),
                        tool_call_id=call.id,
                    )
                )
                self._remember_observation(call.name, record)
                traced_arguments = dict(call.arguments)
                if call.name == "explore":
                    traced_arguments.pop("task", None)
                self._emit(
                    "tool_call",
                    {
                        "call_index": llm_calls,
                        "tool_call_id": call.id,
                        "tool": call.name,
                        "arguments": sanitize_for_log(traced_arguments),
                        "result": sanitize_for_log(record),
                    },
                )
                explore_value = record.get("value")
                if call.name == "explore" and isinstance(explore_value, dict):
                    if explore_totals is None:
                        explore_totals = {
                            "llm_calls": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    _accumulate_explore_usage(explore_totals, explore_value)
            accepted_steps = len(self._complete_validation_nodes)
            messages = self._clear_stale_tool_results(
                messages, protect_from_index=first_new_tool_index
            )

            if self.tools.finished:
                finish = dict(self.tools.finish_value or {})
                token_usage = _token_usage_summary(usage, explore_totals)
                self._emit(
                    "session_end",
                    {
                        "status": "finished",
                        "llm_calls": llm_calls,
                        "steps_used": accepted_steps,
                        "finish": finish,
                        # Cache-hit ratio and the Explore roll-up are the levers
                        # for tuning trimming/compaction and for costing a run.
                        "token_usage": token_usage,
                    },
                )
                return AgentSessionResult(
                    conversation_id=self.conversation_id,
                    status="finished",
                    finish_value=finish,
                    llm_calls=llm_calls,
                    usage=token_usage,
                    context_compactions=(
                        self.compactor.compaction_count
                        if self.compactor is not None
                        else 0
                    ),
                    steps_used=accepted_steps,
                )
            messages = self._apply_inbox(messages, safe_point=apply_point)
            if (
                self.config.mode == "fold"
                and accepted_steps >= self.config.max_steps
                and not step_wrap_up_sent
            ):
                messages.append(ChatMessage("user", STEP_WRAP_UP_PROMPT))
                step_wrap_up_sent = True

        self._emit(
            "session_end",
            {
                "status": "call_budget_exhausted",
                "llm_calls": self.config.max_llm_calls,
                "steps_used": accepted_steps,
            },
        )
        raise RuntimeError("Agent exceeded the session call budget")

    def _provider_tools(self) -> tuple[dict[str, object], ...]:
        if self._hard_finalization:
            names = self._finalization_tool_names()
            records = json.loads(
                json.dumps(
                    self.tools.provider_tools(names),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            candidate_ids = [
                str(candidate["node_id"])
                for candidate in self._complete_validation_nodes
            ]
            for record in records:
                function = record["function"]
                parameters = function["parameters"]
                node_schema = parameters["properties"]["node_id"]
                node_schema["enum"] = candidate_ids
                parameters["required"] = ["node_id"]
            return tuple(records)
        tools = list(self.tools.provider_tools())
        if self.explore is not None:
            roles = list(EXPLORE_ROLES)
            if self.config.mode in {"meta", "meta_learning"}:
                description = (
                    "Optionally delegate one first-level read-only role from the "
                    "unified enum auditor/developer/general-purpose/Explore. Auditor "
                    "is usually the best fit for process review. Every Meta sub-role, "
                    "including developer, is read-only and may only propose candidates. "
                    "Nested explore is forbidden; the parent may finish without delegation."
                )
            else:
                description = (
                    "Optionally delegate one first-level Fold role from the unified "
                    "enum auditor/developer/general-purpose/Explore. Usually prefer "
                    "auditor for review before developer for implementation. Explore "
                    "is a read-only discovery role; the tool name remains explore. Only "
                    "developer and general-purpose may write strategy code. Nested "
                    "explore is forbidden; the parent may finish without delegation."
                )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "explore",
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": roles,
                                },
                                "task": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 8_000,
                                },
                                "max_rounds": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                },
                            },
                            "required": ["role", "task"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return tuple(tools)

    def _active_tool_names(self) -> frozenset[str]:
        if self._hard_finalization:
            return self._finalization_tool_names()
        names = {spec.name for spec in self.tools.specs()}
        if self.explore is not None:
            names.add("explore")
        return frozenset(names)

    def _finalization_tool_names(self) -> frozenset[str]:
        registered = {spec.name for spec in self.tools.specs()}
        if "finish_fold" not in registered:
            raise RuntimeError("Fold hard finalization requires finish_fold")
        return frozenset(registered.intersection(_FOLD_FINALIZATION_TOOLS))

    def _finalization_call_error(self, call: ToolCall) -> str:
        if not self._hard_finalization:
            return ""
        if call.name not in self._active_tool_names():
            return f"tool is unavailable in the current session phase: {call.name}"
        node_id = call.arguments.get("node_id")
        candidates = {
            str(candidate["node_id"]) for candidate in self._complete_validation_nodes
        }
        if not isinstance(node_id, str) or node_id not in candidates:
            return (
                f"{call.name} requires one node_id from the current run's "
                "complete Validation candidates"
            )
        return ""

    def _record_complete_validation(self, record: dict[str, object]) -> None:
        value = record.get("value")
        if not isinstance(value, Mapping):
            return
        node_id = value.get("node_id")
        revision_id = value.get("revision_id")
        if not isinstance(node_id, str) or not node_id:
            return
        if not isinstance(revision_id, str) or not revision_id:
            return
        if any(
            candidate.get("node_id") == node_id
            for candidate in self._complete_validation_nodes
        ):
            return
        raw_stats = value.get("stats")
        stats: dict[str, object] = {}
        if isinstance(raw_stats, Mapping):
            for key, metric in raw_stats.items():
                if len(stats) >= 32:
                    break
                if metric is None or isinstance(metric, (str, int, float, bool)):
                    stats[str(key)] = sanitize_for_log(metric)
        self._complete_validation_nodes.append(
            {
                "node_id": node_id,
                "revision_id": revision_id,
                "validation_index": len(self._complete_validation_nodes) + 1,
                "stats": stats,
            }
        )

    def _activate_hard_finalization_if_ready(self, remaining: float) -> bool:
        """Switch to the restricted finalization view inside the finalize window.

        The window is anchored to the MAIN deadline (budget end minus the
        wrap-up grace): ``finalize_before_deadline_seconds`` before it, exactly
        as before the grace existed. Once the main deadline has passed and the
        wrap-up prompt was injected, the session keeps its full capability
        surface through the grace window — a Validation completing during
        grace does not yank the context anymore; the wrap-up prompt already
        asks the model to finish on its own.
        """
        if (
            self._hard_finalization
            or self.config.mode != "fold"
            or self._wrap_up_sent
            or not self._complete_validation_nodes
        ):
            return False
        main_remaining = remaining - self.config.deadline_grace_seconds
        if main_remaining > self.config.finalize_before_deadline_seconds:
            return False
        tool_names = self._finalization_tool_names()
        self._hard_finalization = True
        self._hard_finalization_context_initialized = False
        self._emit(
            "hard_finalization_started",
            {
                "remaining_seconds": round(max(remaining, 0.0), 6),
                "main_deadline_remaining_seconds": round(max(main_remaining, 0.0), 6),
                "reserve_seconds": self.config.finalize_before_deadline_seconds,
                "grace_seconds": self.config.deadline_grace_seconds,
                "candidate_node_ids": [
                    str(candidate["node_id"])
                    for candidate in self._complete_validation_nodes
                ],
                "available_tools": sorted(tool_names),
            },
        )
        return True

    def _hard_finalization_messages(self, remaining: float) -> list[ChatMessage]:
        self._hard_finalization_context_initialized = True
        payload = {
            "observation": "fold_hard_finalization",
            "remaining_inference_seconds": round(max(remaining, 0.0), 6),
            "selection_contract": (
                "Choose one listed complete Validation node yourself. The Runner "
                "does not rank or auto-submit candidates. Call finish_fold with "
                "its node_id; step_rollback is optional when the workspace should "
                "be restored first."
            ),
            "complete_validation_candidates": list(self._complete_validation_nodes),
            "available_tools": sorted(self._finalization_tool_names()),
        }
        return [
            ChatMessage("system", HARD_FINALIZATION_SYSTEM_PROMPT),
            ChatMessage(
                "user",
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
            ),
        ]

    def _dispatch_tool_calls(
        self, calls: tuple[ToolCall, ...], time_budget: InferenceTimeBudget
    ) -> tuple[list[tuple[ToolCall, dict[str, object]]], str | None]:
        def run_one(index: int) -> tuple[ToolCall, dict[str, object]]:
            call = calls[index]
            self._emit(
                "tool_call_started",
                {"tool": call.name, "tool_call_id": call.id, "status": "running"},
            )
            phase_error = self._finalization_call_error(call)
            if phase_error:
                return call, {"ok": False, "error": phase_error}
            if time_budget.remaining() <= 0:
                return call, {
                    "ok": False,
                    "error": "Agent session deadline reached before tool dispatch",
                }
            if call.name == "explore":
                return call, self._dispatch_explore(call)
            record = self.tools.invoke(
                call.name,
                call.arguments,
                allowed_names=self._active_tool_names(),
            ).to_record()
            if call.name == "daily_backtest" and _is_complete_validation(record):
                self._record_complete_validation(record)
                self._activate_hard_finalization_if_ready(time_budget.remaining())
            return call, record

        if self._is_parallel_readonly_batch(calls):
            slots: list[tuple[ToolCall, dict[str, object]] | None] = [None] * len(
                calls
            )
            with ThreadPoolExecutor(max_workers=min(len(calls), 4)) as executor:
                futures = {
                    executor.submit(run_one, index): index
                    for index in range(len(calls))
                }
                for future in as_completed(futures):
                    slots[futures[future]] = future.result()
            return [item for item in slots if item is not None], None

        results: list[tuple[ToolCall, dict[str, object]]] = []
        terminal_seen = False
        for index, call in enumerate(calls):
            if index > 0 and self._inbox_interrupt_pending():
                results.extend(
                    self._skip_tool_calls(
                        calls[index:],
                        safe_point=INBOX_SAFE_BETWEEN_SERIAL_TOOLS,
                    )
                )
                return results, INBOX_SAFE_BETWEEN_SERIAL_TOOLS
            if terminal_seen:
                results.append(
                    (
                        call,
                        {
                            "ok": False,
                            "error": "terminal tool already called in this assistant turn",
                        },
                    )
                )
                continue
            results.append(run_one(index))
            if call.name in _TERMINAL_TOOLS and self.tools.finished:
                terminal_seen = True
        return results, None

    def _is_parallel_readonly_batch(self, calls: tuple[ToolCall, ...]) -> bool:
        return len(calls) > 1 and all(
            call.name in _PARALLEL_READ_TOOLS
            and (self.tools.spec(call.name) is not None)
            and not bool(self.tools.spec(call.name).mutating)  # type: ignore[union-attr]
            for call in calls
        )

    def _inbox_interrupt_pending(self) -> bool:
        if self.inbox is None:
            return False
        return any(bool(getattr(item, "interrupt", False)) for item in self.inbox.pending())

    def _skip_tool_calls(
        self, calls: tuple[ToolCall, ...], *, safe_point: str
    ) -> list[tuple[ToolCall, dict[str, object]]]:
        skipped: list[tuple[ToolCall, dict[str, object]]] = []
        record = {
            "ok": False,
            "observation": _INTERRUPTED_BY_USER,
            "error": _INTERRUPTED_BY_USER,
        }
        for call in calls:
            skipped.append((call, dict(record)))
            self._emit(
                "tool_skipped",
                {
                    "tool_call_id": call.id,
                    "tool": call.name,
                    "reason": _INTERRUPTED_BY_USER,
                    "safe_point": safe_point,
                },
            )
        return skipped

    def _apply_inbox(
        self, messages: list[ChatMessage], *, safe_point: str
    ) -> list[ChatMessage]:
        if self.inbox is None:
            return messages
        pending = tuple(self.inbox.pending())
        if not pending:
            return messages
        applied_at = utc_now_iso()
        for item in pending:
            message_id = str(getattr(item, "message_id", "") or "").strip()
            text = str(getattr(item, "text", "") or "")
            if not message_id or not text.strip():
                raise RuntimeError("inbox hook returned an invalid notice")
            interrupt = bool(getattr(item, "interrupt", False))
            messages.append(ChatMessage("user", text))
            self._emit(
                "user_message",
                {
                    "message_id": message_id,
                    "interrupt": interrupt,
                    "applied_at": applied_at,
                    "safe_point": safe_point,
                    "content": _inbox_trace_text(text),
                },
            )
            self.inbox.consume(message_id)
        return messages

    def _dispatch_explore(self, call: ToolCall) -> dict[str, object]:
        if self.explore is None:
            record: dict[str, object] = {
                "ok": False,
                "error": "Explore is not configured",
            }
            self._emit(
                "explore_attempt",
                {"ok": False, "error": record["error"]},
            )
            return record
        allowed = EXPLORE_ROLES
        raw_role = call.arguments.get("role")
        if not isinstance(raw_role, str) or raw_role not in allowed:
            record = {
                "ok": False,
                "error": "explore.role must be one of: " + ", ".join(allowed),
            }
            self._emit(
                "explore_attempt",
                {"ok": False, "error": record["error"]},
            )
            return record
        task = call.arguments.get("task")
        raw_rounds = call.arguments.get("max_rounds")
        if not isinstance(task, str) or not task.strip():
            record = {
                "ok": False,
                "error": "explore.task must be a non-empty string",
            }
            self._emit(
                "explore_attempt",
                {"ok": False, "error": record["error"]},
            )
            return record
        max_rounds: int | None = None
        if raw_rounds is not None:
            if not isinstance(raw_rounds, int) or isinstance(raw_rounds, bool):
                record = {
                    "ok": False,
                    "error": "explore.max_rounds must be an integer",
                }
                self._emit(
                    "explore_attempt",
                    {"ok": False, "error": record["error"]},
                )
                return record
            max_rounds = raw_rounds
        self._explore_attempts += 1
        self._explored_roles.add(raw_role)
        attempt = self._explore_attempts
        result = self.explore.run(
            task,
            role=raw_role,
            max_rounds=max_rounds,
            parent_call_id=call.id,
        )
        ok = result.get("status") == "completed"
        self._emit(
            "explore_attempt",
            {
                "attempt": attempt,
                "role": raw_role,
                "ok": ok,
                "status": result.get("status"),
            },
        )
        return {"ok": ok, "value": result}

    def _compact_if_needed(
        self,
        messages: list[ChatMessage],
        remaining: float,
        *,
        force: bool = False,
    ) -> tuple[list[ChatMessage], bool]:
        if self.compactor is None:
            return messages, False
        result = self.compactor.compact(
            messages,
            remaining_seconds=remaining,
            step_id=None,
            force=force,
        )
        if result is None:
            return messages, False
        self._emit("context_compaction", result.event)
        progressed = result.event.get("status") == "ok" and tuple(
            result.messages
        ) != tuple(messages)
        return list(result.messages), progressed

    def _prepare_context_request(
        self,
        messages: list[ChatMessage],
        provider_tools: tuple[dict[str, object], ...],
        remaining: float,
        *,
        allow_semantic_compaction: bool = True,
    ) -> list[ChatMessage]:
        fits, _prompt_tokens, _context_window = context_request_fits(
            self.llm,
            messages,
            tools=provider_tools,
            max_tokens=self.config.max_response_tokens,
        )
        if fits:
            return messages
        if allow_semantic_compaction:
            messages, _ = self._compact_if_needed(messages, remaining, force=True)
            fits, _prompt_tokens, _context_window = context_request_fits(
                self.llm,
                messages,
                tools=provider_tools,
                max_tokens=self.config.max_response_tokens,
            )
            if fits:
                return messages
        messages, edit = fit_tool_results_to_context(
            self.llm,
            messages,
            tools=provider_tools,
            max_tokens=self.config.max_response_tokens,
        )
        if edit:
            self._emit("context_edit", edit)
        fits, prompt_tokens, context_window = context_request_fits(
            self.llm,
            messages,
            tools=provider_tools,
            max_tokens=self.config.max_response_tokens,
        )
        if fits:
            return messages
        assert context_window is not None
        raise context_overflow_error(
            estimated_prompt_tokens=prompt_tokens,
            max_tokens=self.config.max_response_tokens,
            context_window=context_window,
        )

    def _recover_context_overflow(
        self,
        messages: list[ChatMessage],
        provider_tools: tuple[dict[str, object], ...],
        remaining: float,
        *,
        allow_semantic_compaction: bool = True,
    ) -> tuple[list[ChatMessage], bool]:
        """Perform the sole post-provider overflow recovery for a session."""

        recovered = messages
        compacted = False
        if allow_semantic_compaction:
            recovered, compacted = self._compact_if_needed(
                messages, remaining, force=True
            )
        recovered, edit = fit_tool_results_to_context(
            self.llm,
            recovered,
            tools=provider_tools,
            max_tokens=self.config.max_response_tokens,
            force=not compacted,
        )
        if edit:
            self._emit(
                "context_edit",
                {**edit, "reason": "provider_context_overflow_recovery"},
            )
        return recovered, compacted or bool(edit)

    def _clear_stale_tool_results(
        self,
        messages: list[ChatMessage],
        *,
        protect_from_index: int | None = None,
    ) -> list[ChatMessage]:
        if not self.config.clear_tool_results:
            return messages
        if (
            estimate_messages_tokens(messages)
            < self.config.tool_result_clear_token_threshold
        ):
            return messages
        tool_indices = [
            index for index, message in enumerate(messages) if message.role == "tool"
        ]
        keep_recent = self.config.tool_result_keep_recent
        protected = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()
        if protect_from_index is not None:
            protected.update(
                index for index in tool_indices if index >= protect_from_index
            )
        cleared = 0
        chars_freed = 0
        for index in tool_indices:
            if index in protected:
                continue
            message = messages[index]
            content = message.content or ""
            if (
                len(content) >= self.config.tool_result_clear_min_chars
                and content != _CLEARED_TOOL_RESULT
            ):
                chars_freed += len(content)
                messages[index] = ChatMessage(
                    "tool", _CLEARED_TOOL_RESULT, tool_call_id=message.tool_call_id
                )
                cleared += 1
        if cleared:
            self._emit(
                "context_edit",
                {
                    "cleared_tool_results": cleared,
                    "chars_freed": chars_freed,
                    "kept_recent": keep_recent,
                    "protected_from_index": protect_from_index,
                },
            )
        return messages

    def _trim(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if (
            len(messages) <= self.config.max_history_messages
            and estimate_messages_tokens(messages) < self.config.trim_token_threshold
        ):
            return messages
        if self.config.max_history_messages <= 2:
            keep = max(self.config.max_history_messages - 1, 0)
            tail = drop_leading_orphan_tools(messages[-keep:]) if keep else []
            return [messages[0], *tail]

        non_summary = [
            message for message in messages[1:] if not is_compaction_message(message)
        ]
        if (
            self.compactor is not None
            and len(messages) <= self.config.max_history_messages
            and len(non_summary) <= max(self.config.max_history_messages - 3, 0)
        ):
            return messages
        latest_llm_summary = next(
            (
                message
                for message in reversed(messages[1:])
                if is_llm_compaction_message(message)
            ),
            None,
        )
        summary = self._context_summary_payload()
        summary_items = summary.get("items")
        if not isinstance(summary_items, list):
            raise TypeError("context summary items must be a list")
        summary_message = ChatMessage(
            "user",
            json.dumps(summary, ensure_ascii=False, default=str, allow_nan=False),
        )
        kept_llm_compaction = (
            latest_llm_summary is not None and self.config.max_history_messages >= 4
        )
        reserved = 3 if kept_llm_compaction else 2
        max_tail = max(self.config.max_history_messages - reserved, 0)
        headroom = min(self.config.trim_message_headroom, max(max_tail - 1, 0))
        keep = max_tail - headroom
        if self.compactor is None and keep >= len(non_summary):
            keep = max(len(non_summary) - max(headroom, 1), 1)
        tail = drop_leading_orphan_tools(non_summary[-keep:]) if keep else []
        if kept_llm_compaction:
            trimmed = [messages[0], latest_llm_summary, summary_message, *tail]
        else:
            trimmed = [messages[0], summary_message, *tail]
        self._emit(
            "context_summary",
            {
                "summary_items": len(summary_items),
                "kept_llm_compaction": kept_llm_compaction,
                "kept_messages": len(trimmed),
                "dropped_messages": max(len(messages) - len(trimmed), 0),
                "max_history_messages": self.config.max_history_messages,
                "trim_message_headroom": headroom,
            },
        )
        return trimmed

    def _remember_observation(
        self, action: str, observation: dict[str, object]
    ) -> None:
        value = observation.get("value")
        details = value if isinstance(value, dict) else observation
        item: dict[str, object] = {
            "action": action,
            "ok": observation.get("ok"),
        }
        for key in (
            "error",
            "path",
            "node_id",
            "revision_id",
            "complete",
            "backtests_used",
            "backtests_remaining",
            "step_directive",
            "status",
        ):
            candidate = details.get(key)
            if candidate not in (None, "", {}, []):
                item[key] = _shorten(candidate, 300)
        self._observation_summaries.append(item)
        if len(self._observation_summaries) > 120:
            self._observation_summaries = self._observation_summaries[-120:]

    def _context_summary_payload(self) -> dict[str, object]:
        items = self._observation_summaries[-self.config.context_summary_max_items :]
        payload: dict[str, object] = {
            "observation": "context_summary",
            "summary_kind": "deterministic_runner_summary",
            "note": "Required conclusions remain in the current session summary and result artifacts.",
            "items": items,
        }
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= self.config.context_summary_max_chars:
            return payload
        compact_items: list[dict[str, object]] = []
        for item in reversed(items):
            compact_items.insert(0, item)
            compact_payload = {**payload, "items": compact_items}
            if (
                len(json.dumps(compact_payload, ensure_ascii=False, default=str))
                > self.config.context_summary_max_chars
            ):
                compact_items.pop(0)
                break
        return {**payload, "items": compact_items, "truncated": True}

    def _validate_capability_boundary(self) -> None:
        names = {spec.name for spec in self.tools.specs()}
        if self.config.mode in {"meta", "meta_learning"}:
            unsupported = sorted(names - _META_TOOLS)
            if unsupported:
                raise ValueError(
                    f"offline Meta session received unsupported tools: {unsupported}"
                )
        else:
            unsupported = sorted(names - _FOLD_TOOLS)
            if unsupported:
                raise ValueError(
                    f"Agent session received unsupported tools: {unsupported}"
                )
            if "daily_backtest" in names and "finish_fold" not in names:
                raise ValueError(
                    "Fold session with daily_backtest requires finish_fold"
                )

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        record = dict(payload)
        if event == "session_end" and self.explore is not None:
            record["explore_attempts"] = self._explore_attempts
            record["explored_roles"] = sorted(self._explored_roles)
        if self.event_sink is not None:
            self.event_sink(event, record)


class MetaLearningAgent:
    """Validate the small, local-only output contract of a Meta session."""

    def __init__(self, runner: AgentSessionRunner, workspace: str | Path) -> None:
        if runner.config.mode not in {"meta", "meta_learning"}:
            raise ValueError("MetaLearningAgent requires a meta session runner")
        self.runner = runner
        self.workspace = Path(workspace).resolve()

    def learn(self, instruction: str) -> dict[str, object]:
        result = self.runner.run(instruction)
        # A Taste is adopted only on an explicit meta_learning_done; a session
        # that stopped any other way must not have its Taste carried forward.
        if result.finish_value.get("status") != "meta_learning_done":
            raise RuntimeError(
                f"meta-learning did not finish with done: {result.finish_value.get('status')}"
            )
        relative = result.finish_value.get("taste_path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Meta Agent did not nominate taste_path")
        path = (self.workspace / relative).resolve()
        if self.workspace not in path.parents or not path.is_file():
            raise RuntimeError(
                "Meta Agent taste_path is outside the workspace or missing"
            )
        taste = path.read_text(encoding="utf-8").strip()
        if not taste:
            raise RuntimeError("taste.md cannot be empty")
        prior_path = self.workspace / "PRIOR.md"
        prior = ""
        if prior_path.is_file():
            prior = prior_path.read_text(encoding="utf-8").strip()
        return {
            "taste": taste,
            "prior": prior,
            "conversation_id": result.conversation_id,
        }


# A bare 4-digit number followed by one of these is a count/threshold, not a
# date (e.g. "2000 只股票"), so it must not trip the visible-window year check.
_COUNT_UNITS = "只家个支亿万元点名股条款行列页倍"

# A year welded to date syntax (年 / -MM / .MM / Qn / 季度), an 8-digit YYYYMMDD,
# or QnYYYY — a calendar date regardless of which year, so it stays correct when
# the visible fold moves to another year. Bare 4-digit numbers are NOT matched.
_DATE_EXPR = re.compile(
    r"(?:19|20)\d{2}\s*(?:年|[/.\-]\s*\d{1,2}|[Qq][1-4]|\s*[一二三四]\s*季度)"
    r"|(?:19|20)\d{6}"
    r"|[Qq][1-4]\s*(?:19|20)\d{2}"
)

# Taste is injected into every later Fold prompt. Keep a short prior.
TASTE_MAX_CHARS = 4000
# PRIOR is free-format process memory published by Meta. Resource bound, not a schema.
PRIOR_MAX_CHARS = 16_000


def visible_window_dates(manifest: Mapping[str, object]) -> set[str]:
    """Years and YYYYMMDD period bounds of the meta-learning visible fold, read
    from the manifest so the leak check targets the real window whatever year it
    is (e.g. ``{"2020", "2021", "20200101", "20210930", ...}``)."""
    fold = manifest.get("meta_learning_visible_fold") or {}
    if not isinstance(fold, Mapping):
        fold = {}
    blob = " ".join(
        str(value)
        for value in (
            fold.get("input_window"),
            fold.get("validation_period"),
            fold.get("valid_decision_time"),
            manifest.get("valid_decision_time"),
        )
        if value
    )
    return set(re.findall(r"(?:19|20)\d{6}", blob)) | set(
        re.findall(r"(?:19|20)\d{2}", blob)
    )


def calendar_policy_violation(
    text: str, *, window_dates: set[str] | None = None
) -> str:
    """Why this text contains a forbidden calendar date, or "" when it does not.

    Welded date expressions (_DATE_EXPR) are always rejected. When window_dates
    is given, the visible-window years/bounds are also rejected even if written
    bare. Cadence words (季度/月/周) and plain counts/percentages are unaffected.
    """
    dates = set(window_dates or ())
    bare_window = (
        re.compile(
            r"\b(?:"
            + "|".join(re.escape(token) for token in sorted(dates, key=len, reverse=True))
            + r")\b"
            rf"(?!\s*[{_COUNT_UNITS}])"
        )
        if dates
        else None
    )
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _DATE_EXPR.search(line) or (bare_window and bare_window.search(line)):
            return (
                f"line {lineno} contains a calendar date (non-transferable): "
                f"{line.strip()[:80]!r}"
            )
    return ""


def taste_policy_violation(taste_path: Path, *, window_dates: set[str]) -> str:
    """Why this taste.md may not be accepted, or "" when it is acceptable.

    The Taste is injected into every later Fold prompt, so a calendar date in
    it carries hidden-schedule evidence forward, and a long process ledger
    drowns the Fold contract. Empty/overlong files are rejected here; dates go
    through calendar_policy_violation.
    """
    if not taste_path.exists():
        return "write taste.md before finishing"
    text = taste_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return "taste.md must be non-empty before finishing"
    nchars = len(text.strip())
    if nchars > TASTE_MAX_CHARS:
        return (
            f"taste.md is {nchars} characters; keep it to {TASTE_MAX_CHARS} "
            "as a short directional prior, then call finish_meta again"
        )
    violation = calendar_policy_violation(text, window_dates=window_dates)
    if violation:
        return (
            f"taste.md {violation}; state it qualitatively (e.g. 样本交易日不足、按季度轮动) "
            "with no year or window date, then call finish_meta again"
        )
    return ""


_AGENT_PROCESS_HEADING = re.compile(r"(?m)^##[ \t]+Agent Process[ \t]*$")
_PRIOR_BOUNDARY_RE = re.compile(
    r"不得|不要|禁止|不可见|不能用于|不得按|不得用|不得读取|不得使用|不得写入|"
    r"永远不可见|排除|不进入|不读取|不挂载"
)
_HELDOUT_MENTION_RE = re.compile(r"held-?out|holdout|持有期外|隐藏区间", re.I)
_TEST_NUMBER_RE = re.compile(
    r"(?:逐\s*fold|每个\s*fold|fold[_\s-]?(?:ref)?\s*\d*).{0,48}(?:test|测试).{0,40}\d|"
    r"(?:test|测试).{0,24}(?:sharpe|收益|回撤|夏普|total_return|超额).{0,16}\d",
    re.I,
)
_TEST_SELECTION_RE = re.compile(
    r"(根据|按照|基于|凭).{0,20}(?:test|测试).{0,20}(选|选择|保留|淘汰|采用)|"
    r"(?:test|测试).{0,20}(更好|更差|更优|更稳).{0,16}(所以|因此|于是|选择|保留)",
    re.I,
)


def prior_content_violation(text: str) -> str:
    """Held-out leaks, per-Fold Test figures, or Test-based selection."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or _PRIOR_BOUNDARY_RE.search(stripped):
            continue
        if _HELDOUT_MENTION_RE.search(stripped):
            return (
                f"line {lineno} leaks Held-out into PRIOR; "
                "use a boundary sentence such as 不得使用 Test/Held-out"
            )
        if _TEST_NUMBER_RE.search(stripped):
            return (
                f"line {lineno} contains a per-Fold Test figure; "
                "remove Test numbers then call finish_meta again"
            )
        if _TEST_SELECTION_RE.search(stripped):
            return (
                f"line {lineno} uses Test to choose a strategy; "
                "state a transferable process rule instead"
            )
    return ""


def prior_policy_violation(prior_path: Path) -> str:
    """Why an existing PRIOR.md may not be finished, or empty when it is acceptable.

    Missing or blank PRIOR.md means this Meta round keeps the previous version.
    """
    if not prior_path.exists():
        return ""
    text = prior_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return ""
    nchars = len(text.strip())
    if nchars > PRIOR_MAX_CHARS:
        return (
            f"PRIOR.md is {nchars} characters; keep it to {PRIOR_MAX_CHARS} "
            "as process memory, then call finish_meta again"
        )
    if len(_AGENT_PROCESS_HEADING.findall(text)) > 1:
        return (
            "PRIOR.md has duplicate ## Agent Process headings; "
            "keep a single current snapshot"
        )
    leak = prior_content_violation(text)
    if leak:
        return f"PRIOR.md {leak}"
    return ""


class TasteFinishTool:
    spec = ToolSpec(
        "finish_meta",
        "Finish local Meta learning and nominate taste.md.",
        {
            "type": "object",
            "properties": {
                "taste_path": {"type": "string", "minLength": 1, "maxLength": 500}
            },
            "required": ["taste_path"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        workspace: str | Path | SafeWorkspace,
        *,
        window_dates: set[str] | None = None,
    ) -> None:
        self.workspace = (
            workspace
            if isinstance(workspace, SafeWorkspace)
            else SafeWorkspace(workspace)
        )
        self.window_dates = set(window_dates or ())

    def invoke(self, arguments) -> ToolResult:
        path = self.workspace.resolve(
            str(arguments["taste_path"]), must_exist=True, directory=False
        )
        violation = taste_policy_violation(path, window_dates=self.window_dates)
        if violation:
            raise ToolError(violation, error_type="taste_policy")
        prior_violation = prior_policy_violation(self.workspace.root / "PRIOR.md")
        if prior_violation:
            raise ToolError(prior_violation, error_type="prior_policy")
        return ToolResult(
            True,
            # Pipeline adopts a Taste only on an explicit done: the status is
            # the Runner's evidence that the session actually finished.
            value={
                "taste_path": self.workspace.relative(path),
                "status": "meta_learning_done",
            },
            finish=True,
        )


def _new_token_totals() -> dict[str, int]:
    return {
        key: 0
        for key in (
            "llm_calls_with_usage",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        )
    }


def _accumulate_usage(total: dict[str, int], usage: object) -> None:
    """Sum prompt/completion/reasoning/cache tokens across main-conversation calls.

    Cache hits are only realized while the request prefix (system prompt +
    tool schemas + early history) stays byte-stable; trimming and compaction
    rewrite history and reset that prefix, so the session ``cache_hit_ratio``
    is the lever for tuning how aggressively to trim/compact.
    """
    if not isinstance(usage, dict):
        return
    total["llm_calls_with_usage"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] += int(value)
    hit = usage.get("prompt_cache_hit_tokens")
    if not isinstance(hit, (int, float)) or isinstance(hit, bool):
        prompt_details = usage.get("prompt_tokens_details")
        hit = (
            prompt_details.get("cached_tokens")
            if isinstance(prompt_details, dict)
            else None
        )
    if isinstance(hit, (int, float)) and not isinstance(hit, bool):
        total["cache_hit_tokens"] += int(hit)
    miss = usage.get("prompt_cache_miss_tokens")
    if isinstance(miss, (int, float)) and not isinstance(miss, bool):
        total["cache_miss_tokens"] += int(miss)
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning = completion_details.get("reasoning_tokens")
        if isinstance(reasoning, (int, float)) and not isinstance(reasoning, bool):
            total["reasoning_tokens"] += int(reasoning)


def _accumulate_explore_usage(
    totals: dict[str, int], result: Mapping[str, object]
) -> None:
    """Explore Sub Agent calls bill the same provider account but bypass
    ``_accumulate_usage``; without this the session summary understates real
    cost by up to ~15% in observed sessions."""
    calls = result.get("llm_calls")
    if isinstance(calls, (int, float)) and not isinstance(calls, bool):
        totals["llm_calls"] += int(calls)
    usage = result.get("usage_totals")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += int(value)


def _token_usage_summary(
    totals: Mapping[str, int], explore: Mapping[str, int] | None
) -> dict[str, object]:
    summary: dict[str, object] = dict(totals)
    prompt = int(totals.get("prompt_tokens", 0))
    summary["cache_hit_ratio"] = (
        round(int(totals.get("cache_hit_tokens", 0)) / prompt, 4) if prompt else 0.0
    )
    if explore is not None:
        summary["explore"] = dict(explore)
        summary["total_tokens_including_explore"] = int(
            totals.get("total_tokens", 0)
        ) + int(explore.get("total_tokens", 0))
    return summary


def _is_complete_validation(record: dict[str, object]) -> bool:
    if record.get("ok") is not True:
        return False
    value = record.get("value")
    return isinstance(value, dict) and value.get("complete") is True


def _shorten(value: object, max_chars: int) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
