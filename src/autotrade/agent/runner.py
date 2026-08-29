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
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from autotrade.environment.llm import (
    AGENT_MAX_OUTPUT_TOKENS,
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
from autotrade.environment.tools.base import (
    SessionInterrupt,
    ToolError,
    ToolRegistry,
    is_sequential_tool,
)

from .compact import (
    ContextCompactor,
    fit_tool_results_to_context,
    safe_error_summary,
)
from .subagent import (
    OUTPUT_TRUNCATED_CONTINUATION,
    SubAgentEngine,
    AgentTool,
    _copy_chat_message,
    deliver_subagent_report,
    normalize_subagent_thinking,
    resolve_subagent_max_turns,
)
from .prompts import (
    HARD_FINALIZATION_SYSTEM_PROMPT,
    STEP_WRAP_UP_PROMPT,
    WRAP_UP_PROMPT,
)

_LLM_FAILURE_CIRCUIT = 3
SUBAGENT_TEARDOWN_WAIT_SECONDS = 30.0


def _subagent_teardown_timeout(requested: float | None = None) -> float:
    if requested is None:
        return SUBAGENT_TEARDOWN_WAIT_SECONDS
    return min(max(0.0, requested), SUBAGENT_TEARDOWN_WAIT_SECONDS)

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
# A completed Validation can switch the session into hard finalization, and
# the documented contract is that the remaining research calls of that same
# turn are then refused. That only holds when the batch runs in order, so the
# backtest gate is sequential by name regardless of how its spec is declared.
_PHASE_GATE_TOOLS = frozenset({"daily_backtest"})
_FOLD_TOOLS = frozenset(
    {
        "ask_user",
        "daily_backtest",
        "agent",
        "finish_fold",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "smoke_backtest",
        "step_rollback",
        "write_file",
        "edit_file",
        "write_skill",
        "delete_skill",
    }
)
_META_TOOLS = frozenset(
    {
        "ask_user",
        "edit_file",
        "agent",
        "finish_meta",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "write_file",
        "write_skill",
        "delete_skill",
    }
)
INBOX_SAFE_BEFORE_LLM = "before_llm"
INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS = "after_llm_before_tools"
INBOX_SAFE_BETWEEN_SERIAL_TOOLS = "between_serial_tools"
INBOX_SAFE_AFTER_PARALLEL_READONLY = "after_parallel_readonly"
INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM = "after_tools_before_llm"
_INBOX_TRACE_CHARS = 400
_INTERRUPTED_BY_USER = "interrupted_by_user"
# Consecutive own read/search/shell/write calls that trigger the delegation
# reminder while no child is running. The streak resets on an ``agent``
# launch and on every reminder, so a parent that keeps working alone is
# reminded again after each further streak.
DELEGATION_NUDGE_AFTER_CALLS = 8
_OWN_WORK_TOOLS = frozenset(
    {"read_file", "grep", "glob", "shell", "write_file", "edit_file"}
)
# Elapsed fractions of the session's inference time budget at which one
# ``time_budget_notice`` observation states the remaining minutes and, for a
# Fold, how many backtests have run so far.
TIME_BUDGET_NOTICE_FRACTIONS = (0.5, 0.75, 0.9)
_BACKTEST_TOOLS = ("smoke_backtest", "daily_backtest")


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
    # Completion-token safety ceiling shared with the sub-agents; see
    # ``AGENT_MAX_OUTPUT_TOKENS``.
    max_response_tokens: int = AGENT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if self.mode not in ("fold", "meta", "meta_learning"):
            raise ValueError("Agent session mode must be fold, meta, or meta_learning")
        for name in (
            "max_llm_calls",
            "max_steps",
            "deadline_seconds",
            "max_response_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.finalize_before_deadline_seconds < 0
            or self.deadline_grace_seconds < 0
        ):
            raise ValueError("session reserve cannot be negative")


@dataclass
class _SubAgentJob:
    task_id: str
    call_id: str
    role: str
    attempt: int
    future: Future
    # The parent's one-line label; echoed in the live picture so a duplicate
    # scope is visible before it is launched again.
    description: str = ""
    # Collected result (usage accounted, ``subagent_attempt`` emitted).
    record: dict[str, object] | None = None
    # Whether the ``subagent_completed`` observation reached the conversation.
    delivered: bool = False
    # The finished child's own transcript, kept for ``resume``.
    messages: tuple[ChatMessage, ...] | None = None
    # Parent instructions not yet read by the child (``action="message"``);
    # the child drains it before each model round.
    steer: deque[str] = field(default_factory=deque)


@dataclass(frozen=True)
class AgentSessionResult:
    conversation_id: str
    status: str
    finish_value: dict[str, object]
    llm_calls: int
    # Closed's session-summary ``token_usage`` block: the seven per-call totals
    # plus ``cache_hit_ratio`` and the Sub Agent roll-up.
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
        subagent: SubAgentEngine | None = None,
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
        self.subagent = subagent
        if self.subagent is not None:
            subagent_mode = getattr(self.subagent, "mode", "fold")
            if self.config.mode in {"meta", "meta_learning"}:
                if subagent_mode != "meta":
                    raise ValueError("Meta session sub-agent must use mode='meta'")
            elif subagent_mode != "fold":
                raise ValueError("Fold session sub-agent must use mode='fold'")
        self._event_lock = threading.Lock()
        self._subagent_lock = threading.Lock()
        # The tool call id of the invocation running on the current thread;
        # the agent launch reads it to attribute the child to its call.
        self._call_context = threading.local()
        if self.subagent is not None:
            if self.subagent.event_sink is None:
                self.subagent.event_sink = self._locked_event_sink
            if self.subagent.compactor is None and compactor is not None:
                # Children get the parent's context window discipline: the
                # same compaction gateway and thresholds, one fresh instance
                # per child conversation.
                self.subagent.compactor = compactor
            self.tools.register(AgentTool(self._launch_subagent))
        bindings: list[TimeBudgetBinding] = []
        if isinstance(llm, SessionTimeBudgetAware):
            bindings.append(TimeBudgetBinding("main_llm", llm.session_time_budget))
        if subagent is not None and subagent.session_time_budget is not None:
            bindings.append(TimeBudgetBinding("subagent", subagent.session_time_budget))
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
        self._complete_validation_nodes: list[dict[str, object]] = []
        self._hard_finalization = False
        self._hard_finalization_context_initialized = False
        self._wrap_up_sent = False
        self._subagent_attempts = 0
        self._subagent_roles: set[str] = set()
        self._subagent_jobs: list[_SubAgentJob] = []
        self._subagent_pool: ThreadPoolExecutor | None = None
        self._subagent_totals: dict[str, int] | None = None
        self._usage = _new_token_totals()
        # Snapshot of the conversation at the last tool dispatch; a child with
        # inherit_context forks from it.
        self._live_messages: list[ChatMessage] = []
        # Set only when the session closes; children poll it to stop early.
        self._cancelled = threading.Event()
        if self.subagent is not None:
            self.subagent.attach_cancel_event(self._cancelled)
        self._validate_capability_boundary()

    def run(self, instruction: str) -> AgentSessionResult:
        if not instruction.strip():
            raise ValueError("Agent instruction cannot be empty")
        time_budget = self.time_budget or InferenceTimeBudget(
            duration_seconds=self.config.deadline_seconds
        )
        budget_total = max(time_budget.remaining(), 0.0)
        notice_index = 0
        backtests = dict.fromkeys(_BACKTEST_TOOLS, 0)
        messages = [
            ChatMessage("system", self.system_prompt),
            ChatMessage("user", instruction.strip()),
        ]
        self._usage = _new_token_totals()
        self._subagent_totals = None
        llm_calls = 0
        accepted_steps = 0
        step_wrap_up_sent = False
        llm_failure_streak = 0
        context_overflow_recovery_used = False
        self._complete_validation_nodes = []
        self._hard_finalization = False
        self._hard_finalization_context_initialized = False
        self._wrap_up_sent = False
        self._subagent_attempts = 0
        self._subagent_roles = set()
        self._subagent_jobs = []
        self._live_messages = []
        own_work_streak = 0
        self._cancelled.clear()
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
                self._close_session(
                    {"status": "deadline_exceeded", "llm_calls": llm_calls}
                )
                raise AgentSessionDeadlineExceeded(
                    conversation_id=self.conversation_id, llm_calls=llm_calls
                )
            self._activate_hard_finalization_if_ready(remaining)
            provider_tools = self._provider_tools()
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
                if not self._wrap_up_sent:
                    notice_index = self._time_budget_notice(
                        messages, remaining, budget_total, notice_index, backtests
                    )
                messages, _ = self._compact_if_needed(
                    messages, remaining, provider_tools
                )
            messages = self._append_subagent_observations(messages)
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
                    self._close_session(
                        {"status": "context_window_exceeded", "llm_calls": llm_calls}
                    )
                    raise RuntimeError(
                        "Agent context window cannot be reduced safely"
                    ) from exc
                if isinstance(exc, TimeoutError) or time_budget.remaining() <= 0:
                    self._close_session(
                        {"status": "deadline_exceeded", "llm_calls": llm_calls}
                    )
                    raise AgentSessionDeadlineExceeded(
                        conversation_id=self.conversation_id, llm_calls=llm_calls
                    ) from exc
                if "LLM call budget exhausted" in error:
                    self._close_session(
                        {
                            "status": "call_budget_exhausted",
                            "llm_calls": llm_calls,
                        }
                    )
                    raise RuntimeError(
                        "Agent exceeded the session call budget"
                    ) from exc
                if llm_failure_streak >= _LLM_FAILURE_CIRCUIT:
                    self._close_session(
                        {"status": "llm_unavailable", "llm_calls": llm_calls}
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
                continue

            _accumulate_usage(self._usage, response.usage)
            messages.append(
                ChatMessage(
                    "assistant",
                    response.content or None if response.tool_calls else response.content,
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
                if _output_truncated(response.usage, self.config.max_response_tokens):
                    # The whole budget went into thinking (or an unfinished
                    # essay): say so, ask for a concise continuation, and do
                    # not let the next turn silently repeat the same.
                    completion = int(dict(response.usage).get("completion_tokens") or 0)
                    self._emit(
                        "output_truncated",
                        {
                            "call_index": llm_calls,
                            "completion_tokens": completion,
                            "max_tokens": self.config.max_response_tokens,
                        },
                    )
                    messages.append(
                        ChatMessage(
                            "user",
                            json.dumps(
                                {
                                    "observation": "output_truncated",
                                    "completion_tokens": completion,
                                    "max_tokens": self.config.max_response_tokens,
                                    "message": OUTPUT_TRUNCATED_CONTINUATION.format(
                                        limit=self.config.max_response_tokens
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )
                    continue
                if self._yield_for_pending_subagent(time_budget):
                    continue
                nudge: dict[str, object] = {
                    "observation": "no_tool_call",
                    "retry_hint": "Use an injected tool to advance the session; text alone does not finish it.",
                }
                messages.append(
                    ChatMessage("user", json.dumps(nudge, ensure_ascii=False))
                )
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
                parallel = self._is_parallel_batch(response.tool_calls)
                self._live_messages = messages
                results, skipped_at = self._dispatch_tool_calls(
                    response.tool_calls, time_budget
                )
                if skipped_at:
                    apply_point = skipped_at
                elif parallel:
                    apply_point = INBOX_SAFE_AFTER_PARALLEL_READONLY
                else:
                    apply_point = INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM
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
                traced_arguments = dict(call.arguments)
                if call.name == "agent":
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
            accepted_steps = len(self._complete_validation_nodes)
            for call, _record in results:
                if call.name in backtests:
                    backtests[call.name] += 1
            if self.subagent is not None:
                for call, _record in results:
                    if call.name == "agent":
                        own_work_streak = 0
                    elif call.name in _OWN_WORK_TOOLS:
                        own_work_streak += 1
                    else:
                        own_work_streak = 0
                picture = self._subagent_live_picture()
                if (
                    own_work_streak >= DELEGATION_NUDGE_AFTER_CALLS
                    and not picture["running_children"]
                ):
                    reminded = own_work_streak
                    own_work_streak = 0
                    messages.append(
                        ChatMessage(
                            "user",
                            json.dumps(
                                {
                                    "observation": "delegation_reminder",
                                    "message": (
                                        f"现在没有子代理在运行，而你已连续 {reminded} 次自行读取/执行/写入。"
                                        "把接下来可并行的块（实现、统计、审计、性能剖析）作为并行 agent 子代理启动"
                                        "（范围互斥），再继续本轮设计；串行自做占用的是本会话最稀缺的资源。"
                                    ),
                                    **picture,
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )
                    self._emit(
                        "delegation_reminder",
                        {"own_work_calls": reminded, **picture},
                    )

            if self.tools.finished:
                self._append_subagent_observations(messages, wait=True)
                finish = dict(self.tools.finish_value or {})
                token_usage = _token_usage_summary(self._usage, self._subagent_totals)
                self._close_session(
                    {
                        "status": "finished",
                        "llm_calls": llm_calls,
                        "steps_used": accepted_steps,
                        "finish": finish,
                        "token_usage": token_usage,
                    }
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

        self._close_session(
            {
                "status": "call_budget_exhausted",
                "llm_calls": self.config.max_llm_calls,
                "steps_used": accepted_steps,
            }
        )
        raise RuntimeError("Agent exceeded the session call budget")

    def _time_budget_notice(
        self,
        messages: list[ChatMessage],
        remaining: float,
        total: float,
        notice_index: int,
        backtests: Mapping[str, int],
    ) -> int:
        """Inject one budget notice per crossed fraction; return the next index."""

        fractions = TIME_BUDGET_NOTICE_FRACTIONS
        if total <= 0 or notice_index >= len(fractions):
            return notice_index
        elapsed = 1.0 - remaining / total
        crossed: float | None = None
        while notice_index < len(fractions) and elapsed >= fractions[notice_index]:
            crossed = fractions[notice_index]
            notice_index += 1
        if crossed is None:
            return notice_index
        remaining_minutes = round(max(remaining, 0.0) / 60.0, 1)
        payload: dict[str, object] = {
            "observation": "time_budget_notice",
            "elapsed_fraction": crossed,
            "remaining_minutes": remaining_minutes,
        }
        if self.config.mode == "fold":
            complete = len(self._complete_validation_nodes)
            payload.update(
                smoke_backtests=backtests["smoke_backtest"],
                daily_backtests=backtests["daily_backtest"],
                complete_validations=complete,
            )
            payload["message"] = (
                f"推理时间预算已用去 {crossed:.0%}，剩余约 {remaining_minutes:g} 分钟；"
                f"至今 smoke_backtest {backtests['smoke_backtest']} 次、"
                f"daily_backtest {backtests['daily_backtest']} 次、完整 Validation {complete} 个。"
                "一次完整 Validation 通常需要半小时以上：请在剩余时间内完成正式回测并 finish_fold。"
            )
        else:
            payload["message"] = (
                f"推理时间预算已用去 {crossed:.0%}，剩余约 {remaining_minutes:g} 分钟；"
                "请在剩余时间内完成 PRIOR 并 finish_meta。"
            )
        messages.append(ChatMessage("user", json.dumps(payload, ensure_ascii=False)))
        self._emit(
            "time_budget_notice",
            {
                key: value
                for key, value in payload.items()
                if key not in {"observation", "message"}
            },
        )
        return notice_index

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
        return self.tools.provider_tools()

    def _active_tool_names(self) -> frozenset[str]:
        if self._hard_finalization:
            return self._finalization_tool_names()
        return frozenset(spec.name for spec in self.tools.specs())

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
        """Run one assistant turn's tool calls.

        A batch of parallel-safe calls (reads, checks, agent launches) runs
        concurrently; a batch containing any sequential tool runs in order,
        stops after a terminal tool, and honours inbox interrupts between calls.
        """

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
            if call.name in _TERMINAL_TOOLS or call.name in _PHASE_GATE_TOOLS:
                # Barrier: a formal backtest or finish must not overlap a
                # developer child that may still be writing the workspace.
                self._wait_subagent_jobs()
            self._call_context.call_id = call.id
            record = self.tools.invoke(
                call.name,
                call.arguments,
                allowed_names=self._active_tool_names(),
            ).to_record()
            if call.name == "daily_backtest" and record.get("ok") is True:
                # A successful backtest result is a complete Validation node.
                self._record_complete_validation(record)
                self._activate_hard_finalization_if_ready(time_budget.remaining())
            return call, record

        if self._is_parallel_batch(calls):
            slots: list[tuple[ToolCall, dict[str, object]] | None] = [None] * len(
                calls
            )
            interrupt: SessionInterrupt | None = None
            with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as executor:
                futures = {
                    executor.submit(run_one, index): index
                    for index in range(len(calls))
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        slots[index] = future.result()
                    except SessionInterrupt as exc:
                        interrupt = exc
                    except Exception as exc:  # noqa: BLE001 - one call must not drop its siblings
                        slots[index] = (
                            calls[index],
                            {"ok": False, "error": safe_error_summary(exc)},
                        )
            if interrupt is not None:
                raise interrupt
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

    def _is_parallel_batch(self, calls: tuple[ToolCall, ...]) -> bool:
        return len(calls) > 1 and not any(
            call.name in _PHASE_GATE_TOOLS
            or is_sequential_tool(self.tools.spec(call.name))
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

    def _launch_subagent(self, arguments: Mapping[str, object]) -> dict[str, object]:
        """Start one background child from registry-validated ``agent`` arguments.

        Never blocks the parent turn: the child runs in the sub-agent pool,
        whose worker count is the concurrency cap, so launches beyond the cap
        queue until a slot frees. Completion is delivered later as a
        ``subagent_completed`` observation. ``resume`` continues a finished
        child's own transcript with the new task; a running or unknown child
        is refused. ``action="message"`` instead queues an instruction for a
        child that has not finished (see ``_steer_subagent``).
        """

        if self.subagent is None:
            raise ToolError("Sub-agent is not configured")
        action = str(arguments.get("action") or "launch")
        if action == "message":
            return self._steer_subagent(arguments)
        if "agent" not in arguments or "task" not in arguments:
            raise ToolError(
                "agent: launch requires agent and task", error_type="schema_error"
            )
        role = str(arguments["agent"])
        task = str(arguments["task"])
        if not task.strip():
            raise ToolError("agent.task must be a non-empty string")
        try:
            # Validated here so a bad value is a tool error on the parent's
            # turn; the engine resolves the same precedence for the record.
            thinking = normalize_subagent_thinking(arguments.get("thinking"), role)
            max_rounds = resolve_subagent_max_turns(
                arguments.get("max_turns"), role, self.subagent.config.max_rounds
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        inherit = bool(arguments.get("inherit_context", False))
        description = str(arguments.get("description") or "").strip()
        resume = str(arguments.get("resume") or "").strip() or None
        call_id = getattr(self._call_context, "call_id", None)
        cap = self.subagent.config.max_concurrent
        transcript: tuple[ChatMessage, ...] | None = None
        if resume is not None:
            with self._subagent_lock:
                previous = next(
                    (job for job in self._subagent_jobs if job.task_id == resume), None
                )
            if previous is None:
                raise ToolError(
                    f"resume: unknown sub-agent task_id {resume}",
                    error_type="unknown_subagent",
                )
            if previous.record is None or not previous.future.done():
                raise ToolError(
                    f"resume: sub-agent {resume} is still running; wait for its "
                    "subagent_completed message",
                    error_type="subagent_running",
                )
            if previous.role != role:
                raise ToolError(
                    f"resume: sub-agent {resume} ran as {previous.role}; "
                    "a follow-up keeps that agent role",
                    error_type="subagent_role_mismatch",
                )
            if not previous.messages:
                raise ToolError(
                    f"resume: sub-agent {resume} left no transcript to continue",
                    error_type="subagent_no_transcript",
                )
            transcript = previous.messages
            inherit = False
        parent_messages = (
            tuple(_copy_chat_message(message) for message in self._live_messages)
            if inherit and self._live_messages
            else None
        )
        with self._subagent_lock:
            pending = [
                job
                for job in self._subagent_jobs
                if job.record is None and not job.future.done()
            ]
            queued = len(pending) >= cap
            self._subagent_attempts += 1
            self._subagent_roles.add(role)
            attempt = self._subagent_attempts
            task_id = f"agent_{uuid.uuid4().hex[:12]}"
            if self._subagent_pool is None:
                self._subagent_pool = ThreadPoolExecutor(
                    max_workers=max(1, cap),
                    thread_name_prefix="agent",
                )
            steer: deque[str] = deque()
            future = self._subagent_pool.submit(
                self.subagent.run_with_transcript,
                task,
                role=role,
                max_rounds=max_rounds,
                parent_call_id=call_id,
                thinking=thinking,
                inherit_context=inherit,
                parent_messages=parent_messages,
                transcript=transcript,
                resumed_from=resume,
                description=description,
                task_id=task_id,
                steer_queue=steer,
            )
            self._subagent_jobs.append(
                _SubAgentJob(
                    task_id=task_id,
                    call_id=str(call_id or ""),
                    role=role,
                    attempt=attempt,
                    future=future,
                    description=description,
                    steer=steer,
                )
            )
            picture = self._subagent_live_picture()
        record: dict[str, object] = {
            "status": "started",
            "background": True,
            "task_id": task_id,
            "role": role,
            "attempt": attempt,
        }
        if resume is not None:
            record["resumed_from"] = resume
        if queued:
            record["queued"] = True
        record.update(picture)
        return record

    def _steer_subagent(self, arguments: Mapping[str, object]) -> dict[str, object]:
        """Queue one parent instruction for a child that has not finished.

        The child reads it before its next model round (a queued child before
        its first), so the ack only says it was queued; delivery shows up as
        the child's ``subagent_steer`` event and as ``steers`` /
        ``steers_undelivered`` in its ``subagent_completed`` observation. A
        finished child takes follow-ups through ``resume``.
        """

        if "task_id" not in arguments or "text" not in arguments:
            raise ToolError(
                "agent: message requires task_id and text", error_type="schema_error"
            )
        task_id = str(arguments["task_id"]).strip()
        text = str(arguments["text"]).strip()
        if not text:
            raise ToolError("agent.text must be a non-empty string")
        with self._subagent_lock:
            job = next((job for job in self._subagent_jobs if job.task_id == task_id), None)
            if job is None:
                raise ToolError(
                    f"message: unknown sub-agent task_id {task_id}",
                    error_type="unknown_subagent",
                )
            if job.record is not None or job.future.done():
                raise ToolError(
                    f"message: sub-agent {task_id} has finished; give it a follow-up "
                    "with resume instead",
                    error_type="subagent_finished",
                )
            job.steer.append(text)
            child_queued = not job.future.running()
        self._emit(
            "subagent_steer",
            {
                "task_id": task_id,
                "role": job.role,
                "chars": len(text),
                "delivery": "queued",
                "parent_call_id": job.call_id or None,
            },
        )
        record: dict[str, object] = {
            "status": "queued",
            "task_id": task_id,
            "delivered_at_round": None,
        }
        if child_queued:
            record["child_queued"] = True
        return record

    def _subagent_live_picture(self) -> dict[str, object]:
        """Children still in flight, as the parent should see them.

        The pool is FIFO with ``max_concurrent`` workers, so the oldest
        uncollected children are the running ones and the rest wait for a
        slot. Each entry carries the parent's own ``description`` so a scope
        that is already in progress is visible before it is launched twice.
        Callers on a tool thread hold ``_subagent_lock``.
        """

        in_flight = [
            job
            for job in self._subagent_jobs
            if job.record is None and not job.future.done()
        ]
        cap = self.subagent.config.max_concurrent if self.subagent is not None else 0

        def entry(job: _SubAgentJob) -> dict[str, str]:
            return {
                "task_id": job.task_id,
                "role": job.role,
                "description": job.description,
            }

        return {
            "running_children": [entry(job) for job in in_flight[:cap]],
            "queued_children": [entry(job) for job in in_flight[cap:]],
        }

    def _append_subagent_observations(
        self, messages: list[ChatMessage], *, wait: bool = False
    ) -> list[ChatMessage]:
        """Collect finished children and deliver each result once as a message."""

        self._collect_finished_subagents(
            wait=wait, timeout=_subagent_teardown_timeout() if wait else None
        )
        for job in self._subagent_jobs:
            if job.record is None or job.delivered:
                continue
            job.delivered = True
            value = job.record.get("value")
            payload = {
                "observation": "subagent_completed",
                "ok": job.record.get("ok"),
                "status": job.record.get("status"),
                "task_id": job.task_id,
                "role": job.role,
            }
            if isinstance(value, dict):
                # The bounded report (clipped and spilled at collection).
                payload.update(job.record.get("report") or {"summary": ""})
                for key in ("rounds", "tool_calls"):
                    if isinstance(value.get(key), int):
                        payload[key] = value[key]
                if value.get("truncated"):
                    # A cut-off report is not a complete one; the marker at
                    # the tail of a long summary is easy to miss.
                    payload["truncated"] = True
                for key in ("truncated_rounds", "llm_errors", "steers"):
                    if isinstance(value.get(key), int) and value[key] > 0:
                        payload[key] = value[key]
                if value.get("error"):
                    payload["error"] = value.get("error")
            if job.steer:
                # An instruction the child never read is not silently lost.
                payload["steers_undelivered"] = len(job.steer)
            messages.append(
                ChatMessage(
                    "user",
                    json.dumps(payload, ensure_ascii=False, default=str),
                )
            )
        return messages

    def _uncollected_subagent_jobs(self) -> list[_SubAgentJob]:
        return [job for job in self._subagent_jobs if job.record is None]

    def _yield_for_pending_subagent(self, time_budget: InferenceTimeBudget) -> bool:
        """Skip no_tool_call when a sub-agent is still running; wait for progress."""
        uncollected = self._uncollected_subagent_jobs()
        if not uncollected:
            return False
        if not any(job.future.done() for job in uncollected):
            self._wait_first_pending_subagent(time_budget)
        return True

    def _wait_first_pending_subagent(self, time_budget: InferenceTimeBudget) -> None:
        completed = threading.Event()

        def _on_done(_future: Future) -> None:
            completed.set()

        pending = [
            job.future
            for job in self._uncollected_subagent_jobs()
            if not job.future.done()
        ]
        if not pending:
            return
        self._emit("subagent_wait_started", {"pending": len(pending)})
        for future in pending:
            future.add_done_callback(_on_done)
            if future.done():
                return
        while not completed.is_set() and not self._cancelled.is_set():
            remaining = time_budget.remaining()
            if remaining <= 0:
                return
            completed.wait(timeout=min(0.05, remaining))

    def _wait_subagent_jobs(
        self, timeout: float = SUBAGENT_TEARDOWN_WAIT_SECONDS
    ) -> list[dict[str, object]]:
        return self._collect_finished_subagents(
            wait=True, timeout=_subagent_teardown_timeout(timeout)
        )

    def _collect_finished_subagents(
        self, *, wait: bool = False, timeout: float | None = None
    ) -> list[dict[str, object]]:
        """The one place a child's result is taken: usage and Trace here, the
        conversation observation later via ``_append_subagent_observations``."""

        finished: list[dict[str, object]] = []
        wait_timeout = _subagent_teardown_timeout(timeout) if wait else None
        deadline = (
            time.monotonic() + wait_timeout if wait_timeout is not None else None
        )
        for job in self._subagent_jobs:
            if job.record is not None:
                continue
            result: object
            if wait:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0:
                    continue
                try:
                    result, job.messages = job.future.result(timeout=remaining)
                except TimeoutError:
                    continue
                except Exception as exc:  # noqa: BLE001 - child failure stays an observation
                    result = {
                        "task_id": job.task_id,
                        "status": "error",
                        "error": safe_error_summary(exc),
                        "role": job.role,
                    }
            else:
                if not job.future.done():
                    continue
                try:
                    result, job.messages = job.future.result()
                except Exception as exc:  # noqa: BLE001 - child failure stays an observation
                    result = {
                        "task_id": job.task_id,
                        "status": "error",
                        "error": safe_error_summary(exc),
                        "role": job.role,
                    }
            ok = result.get("status") == "completed" if isinstance(result, dict) else False
            record = {
                "ok": ok,
                "status": result.get("status") if isinstance(result, dict) else "error",
                "task_id": job.task_id,
                "role": job.role,
                "value": result,
            }
            job.record = record
            attempt: dict[str, object] = {
                "attempt": job.attempt,
                "role": job.role,
                "ok": ok,
                "status": record["status"],
                "task_id": job.task_id,
            }
            if isinstance(result, dict):
                if self._subagent_totals is None:
                    self._subagent_totals = {
                        "llm_calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                _accumulate_subagent_usage(self._subagent_totals, result)
                # Bound the report once, here, so the Trace records what the
                # parent will receive and where the rest went.
                report = deliver_subagent_report(
                    str(result.get("summary") or ""), self.tools.result_store()
                )
                record["report"] = report
                attempt.update(
                    (key, report[key])
                    for key in (
                        "summary_chars",
                        "summary_delivered_chars",
                        "summary_lines",
                        "resume_line",
                        "summary_truncated",
                        "result_ref",
                    )
                    if key in report
                )
            self._emit("subagent_attempt", attempt)
            finished.append(record)
        return finished

    def _cancel_pending_subagents(self) -> None:
        for job in self._subagent_jobs:
            if job.record is not None or job.future.done():
                continue
            if not job.future.cancel():
                continue
            record = {
                "ok": False,
                "status": "cancelled",
                "task_id": job.task_id,
                "role": job.role,
                "value": {
                    "task_id": job.task_id,
                    "status": "cancelled",
                    "error": "Sub-agent cancelled",
                    "role": job.role,
                },
            }
            job.record = record
            self._emit(
                "subagent_attempt",
                {
                    "attempt": job.attempt,
                    "role": job.role,
                    "ok": False,
                    "status": "cancelled",
                    "task_id": job.task_id,
                },
            )

    def _close_session(self, payload: dict[str, object]) -> None:
        self._cancelled.set()
        if self.subagent is not None:
            self.subagent.cancel()
        # Bound the wait; do not abort an in-flight LLM or tool invoke.
        # After an uncancelable LLM returns, the child checks this event and
        # exits without dispatching. Pending (not started) jobs are cancelled.
        self._collect_finished_subagents(
            wait=True, timeout=_subagent_teardown_timeout()
        )
        self._cancel_pending_subagents()
        payload = dict(payload)
        payload.setdefault(
            "token_usage", _token_usage_summary(self._usage, self._subagent_totals)
        )
        self._emit("session_end", payload)
        pool = self._subagent_pool
        self._subagent_pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _compact_if_needed(
        self,
        messages: list[ChatMessage],
        remaining: float,
        provider_tools: tuple[dict[str, object], ...],
        *,
        force: bool = False,
    ) -> tuple[list[ChatMessage], bool]:
        if self.compactor is None:
            return messages, False
        result = self.compactor.compact(
            messages,
            tools=provider_tools,
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
            messages, _ = self._compact_if_needed(
                messages, remaining, provider_tools, force=True
            )
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
        if context_window is None:
            raise RuntimeError(
                "context window is unavailable after the request did not fit"
            )
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
                messages, remaining, provider_tools, force=True
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

    def _validate_capability_boundary(self) -> None:
        """Fail closed on any tool outside the session's documented set."""

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

    def _locked_event_sink(self, event: str, payload: dict[str, object]) -> None:
        with self._event_lock:
            if self.event_sink is not None:
                self.event_sink(event, payload)

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        record = dict(payload)
        if event == "session_end" and self.subagent is not None:
            record["subagent_attempts"] = self._subagent_attempts
            record["subagent_roles"] = sorted(self._subagent_roles)
        self._locked_event_sink(event, record)


class MetaLearningAgent:
    """Validate the fixed, local-only PRIOR.md output contract of a Meta session."""

    def __init__(self, runner: AgentSessionRunner, workspace: str | Path) -> None:
        if runner.config.mode not in {"meta", "meta_learning"}:
            raise ValueError("MetaLearningAgent requires a meta session runner")
        self.runner = runner
        self.workspace = Path(workspace).resolve()

    def learn(self, instruction: str) -> dict[str, object]:
        result = self.runner.run(instruction)
        if result.finish_value.get("status") != "meta_learning_done":
            raise RuntimeError(
                f"meta-learning did not finish with done: {result.finish_value.get('status')}"
            )
        prior_path = self.workspace / "PRIOR.md"
        if not prior_path.is_file():
            raise RuntimeError("Meta Agent did not produce PRIOR.md")
        prior = prior_path.read_text(encoding="utf-8").strip()
        if not prior:
            raise RuntimeError("PRIOR.md cannot be empty")
        return {
            "prior": prior,
            "conversation_id": result.conversation_id,
        }


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


def _output_truncated(usage: object, max_tokens: int) -> bool:
    """A reply that used its whole completion budget (the transport surfaces no
    ``finish_reason``, so the usage count is the signal)."""

    if not isinstance(usage, Mapping):
        return False
    completion = usage.get("completion_tokens")
    return (
        isinstance(completion, (int, float))
        and not isinstance(completion, bool)
        and completion >= max_tokens
    )


def _accumulate_usage(total: dict[str, int], usage: object) -> None:
    """Sum prompt/completion/reasoning/cache tokens across main-conversation calls.

    Cache hits are only realized while the request prefix (system prompt +
    tool schemas + early history) stays byte-stable; semantic compaction and
    emergency tool-result fitting rewrite history and reset that prefix.
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


def _accumulate_subagent_usage(
    totals: dict[str, int], result: Mapping[str, object]
) -> None:
    """Sub Agent calls bill the same provider account but bypass
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
    totals: Mapping[str, int], subagent: Mapping[str, int] | None
) -> dict[str, object]:
    summary: dict[str, object] = dict(totals)
    prompt = int(totals.get("prompt_tokens", 0))
    summary["cache_hit_ratio"] = (
        round(int(totals.get("cache_hit_tokens", 0)) / prompt, 4) if prompt else 0.0
    )
    if subagent is not None:
        summary["subagent"] = dict(subagent)
        summary["total_tokens_including_subagents"] = int(
            totals.get("total_tokens", 0)
        ) + int(subagent.get("total_tokens", 0))
    return summary

