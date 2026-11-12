"""Agent session runner: the main conversation loop of the research session.

docs/AGENT_DESIGN.md plus docs/ENVIRONMENT_DESIGN.md §2.2 define the Agent
session and tool-entrypoint contract: one conversation per session attempt,
Steps share the session, only documented tools are callable, and the session
budget is the master constraint. The session finishes when the ``test`` that
brings the test horizon to the end of the validation years returns; no other
action ends it. The budget the pipeline hands over includes a trailing
wrap-up grace window (``deadline_grace_seconds``): reaching the main deadline
never interrupts the model — a single wrap-up prompt is injected and the
session keeps its full tool surface through the grace window. The same single
prompt goes out instead when the API spend or the model calls reach
``BUDGET_WRAP_UP_FRACTION`` of their caps first, since running out of either
ends the session at once. When the grace window, the model-call budget or the
API spend cap is exhausted the session closes gracefully (``session_end`` with
status ``deadline_exceeded``, ``call_budget_exhausted`` or
``api_spend_exhausted``) and the Pipeline
records a ``deadline`` outcome and completes the validation record from the
test horizon instead of failing the run; the readout follows from the host's
forward replay. Main conversation calls and compactions
are logged in the Agent trace (docs/ENVIRONMENT_DESIGN.md §2.4 and §4.2).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol

from autotrade.environment.llm import (
    AGENT_MAX_OUTPUT_TOKENS,
    ApiSpendExhausted,
    CallBudgetExhausted,
    ChatMessage,
    LLMProxy,
    LLMProxyError,
    MalformedToolCallError,
    ProviderRefusalError,
    ToolCall,
    estimate_chat_request_tokens,
    is_context_overflow_error,
    malformed_tool_call_messages,
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
from autotrade.environment.tools.search import trace_read_stub
from autotrade.environment.trace import TraceEvent, trace_reasoning

from .compact import (
    DEFAULT_COMPACTION_TOKEN_THRESHOLD,
    ContextCompactionConfig,
    ContextCompactor,
    ContextFitter,
    compact_with_summary,
    safe_error_summary,
)
from .prompts import (
    WRAP_UP_PROMPT,
)
from .subagent import (
    OUTPUT_TRUNCATED_CONTINUATION,
    PROVIDER_TRANSIENT_FAILURE_HINT,
    AgentTool,
    SubAgentEngine,
    accumulate_usage,
    deliver_subagent_report,
    new_token_totals,
    normalize_subagent_thinking,
    output_truncated,
    resolve_subagent_max_turns,
    subagent_role,
)

_LLM_FAILURE_CIRCUIT = 3
# How long a teardown or a terminal-tool barrier waits for the children still
# running. Read at each call site, so a test can shorten it on the module.
SUBAGENT_TEARDOWN_WAIT_SECONDS = 30.0

# Default wrap-up grace shared with ExperimentConfig.deadline_grace_minutes:
# the pipeline hands the session a budget of main deadline + grace and the runner
# reserves the trailing grace for wrap-up. Keep the two defaults aligned.
DEFAULT_DEADLINE_GRACE_SECONDS = 600.0


# ``finish_reason`` values of a session that ended because a budget ran out.
DEADLINE_GRACE_EXHAUSTED = "deadline_grace_exhausted"
LLM_CALL_BUDGET_EXHAUSTED = "llm_call_budget_exhausted"
API_SPEND_EXHAUSTED = "api_spend_exhausted"


class AgentSessionBudgetExhausted(SessionInterrupt):
    """A session budget ran out: the deadline with its wrap-up grace, the
    model-call budget or the API spend cap.

    Control flow, not an error: the session has already emitted
    ``session_end`` (status ``deadline_exceeded``, ``call_budget_exhausted``
    or ``api_spend_exhausted``);
    the Pipeline records the ``deadline`` outcome with ``finish_reason`` and
    ends the experiment instead of failing the run. A SessionInterrupt subclass so it
    re-raises through tool dispatch instead of being swallowed into an error
    observation.
    """

    def __init__(
        self,
        message: str = "Agent session budget exhausted",
        *,
        finish_reason: str = DEADLINE_GRACE_EXHAUSTED,
        conversation_id: str = "",
        llm_calls: int = 0,
    ) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.conversation_id = conversation_id
        self.llm_calls = llm_calls


# ``test`` advances the test horizon and, on the last step, ends the
# session: it waits for every sub-agent (the last step's close would discard
# their reports) and a finishing call ends the turn.
_TERMINAL_TOOLS = frozenset({"test"})
# The Agent's own compaction: the tool validates the summary, the runner
# rebuilds the conversation once the turn's tool results are in.
_COMPACT_TOOL = "compact"
# The estimated request size, as a share of the runtime compaction threshold,
# at which the Agent is told once to compact itself: close to the threshold, so
# a session compacts only once its window is nearly spent, with the host's own
# compaction at the threshold as the backstop. Re-armed by every compaction, so
# a long session hears it once per context window.
COMPACT_ADVISORY_FRACTION = 0.9
# The tool that produces complete Validation nodes: one per finished candidate.
_VALIDATION_TOOLS = frozenset({"validate"})
# A batch runs in order whatever its spec declares, so its rows are recorded
# before the turn's later calls read the tree.
_PHASE_GATE_TOOLS = _VALIDATION_TOOLS
# Host jobs that pause the session clock like a formal backtest take the same
# writer barrier: the null control is a multi-minute host replay of a recorded
# result, dispatched in order and never beside a child still writing.
_WRITER_BARRIER_TOOLS = _PHASE_GATE_TOOLS | frozenset({"run_null_control"})
_SESSION_TOOLS = frozenset(
    {
        "validate",
        "test",
        "run_null_control",
        "agent",
        "compact",
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
        "report_issue",
        "skill_feedback",
    }
)
INBOX_SAFE_BEFORE_LLM = "before_llm"
INBOX_SAFE_AFTER_LLM_BEFORE_TOOLS = "after_llm_before_tools"
INBOX_SAFE_BETWEEN_SERIAL_TOOLS = "between_serial_tools"
INBOX_SAFE_AFTER_PARALLEL_READONLY = "after_parallel_readonly"
INBOX_SAFE_AFTER_TOOLS_BEFORE_LLM = "after_tools_before_llm"
_INBOX_TRACE_CHARS = 400
_INTERRUPTED_BY_USER = "interrupted_by_user"
# Used fractions of each session budget -- the inference time, the API spend
# and the model calls -- at which one ``budget_notice`` observation states what
# is left of that budget and how many backtests have run so far.
BUDGET_NOTICE_FRACTIONS = (0.5, 0.75, 0.9)
# The used fraction of the API spend or of the model calls at which the wrap-up
# prompt goes out, as it does for the time budget at its grace window: running
# out of either stops the session at once, so the session gets its chance to
# ``test`` its best candidate first. It takes the place of that budget's last
# notice.
BUDGET_WRAP_UP_FRACTION = 0.9
# What is left of each budget, in its own unit, as the notice payload names it.
_BUDGET_REMAINING_KEYS = {
    "inference_time": "remaining_minutes",
    "api_spend": "remaining_usd",
    "llm_calls": "remaining_calls",
}
# What a spend or call notice adds about the rest of that budget: it is
# charged per model call, the session's own validate and test included. The
# inference-time notice states the used share and the minutes left only.
_BUDGET_NOTICE_ADVICE = {
    "api_spend": "子代理与压缩的调用也计入花费，为仍要跑的 validate 与 test 留出花费",
    "llm_calls": "子代理与压缩的调用也计入，为仍要跑的 validate 与 test 留出调用次数",
}
_BACKTEST_TOOLS = ("smoke_backtest", "validate", "test")
# The trace events that carry one tool outcome. The session_end tool_failures
# scalar and the Agent Trace summary must count the same events, or an audit
# that reads only the scalar undercounts a session whose children did the work.
_TOOL_OUTCOME_EVENTS = frozenset({TraceEvent.TOOL_CALL, TraceEvent.SUBAGENT_TOOL})


def _failed_tool_event(payload: Mapping[str, object]) -> bool:
    result = payload.get("result")
    return isinstance(result, Mapping) and result.get("ok") is False


class AgentInboxHook(Protocol):
    """Current-session unconsumed notices; consume is atomic per run."""

    def pending(self) -> Sequence[object]: ...

    def consume(self, message_id: str) -> str: ...


class SessionBudgetMeter(Protocol):
    """The session's shared model-call and API-spend counters, every role and
    earlier attempt included: the pipeline's ``SessionCallBudget``, which
    charges each response as it lands. The runner only reads them, for the
    budget notices and the wrap-up."""

    calls: int
    max_calls: int
    api_spend_usd: float
    max_api_spend_usd: float


@dataclass(frozen=True)
class _BudgetLevel:
    """How far one session budget is spent: the used fraction of its cap and
    what is left in its own unit (minutes, USD or calls)."""

    budget: str
    used: float
    left: float


def _budget_spent_phrase(level: _BudgetLevel, fraction: float) -> str:
    """How much of a budget is spent and what is left: the opening of its
    notice, and of the wrap-up a spend or call budget triggers."""

    if level.budget == "inference_time":
        return f"推理时间预算已用去 {fraction:.0%}，剩余约 {level.left:g} 分钟"
    if level.budget == "api_spend":
        return f"API 花费已用去上限的 {fraction:.0%}，剩余约 ${level.left:.2f}"
    return f"模型调用已用去上限的 {fraction:.0%}，剩余 {level.left:.0f} 次"


def _inbox_trace_text(text: str) -> str:
    redacted = sanitize_for_log(text)
    if not isinstance(redacted, str):
        redacted = str(redacted)
    if len(redacted) > _INBOX_TRACE_CHARS:
        return redacted[:_INBOX_TRACE_CHARS]
    return redacted


@dataclass(frozen=True)
class AgentSessionConfig:
    # Trailing wrap-up grace reserved from the end of the handed session
    # budget. The main deadline sits grace seconds before the budget end:
    # reaching it never interrupts the model; exhausting the budget does.
    deadline_grace_seconds: float = DEFAULT_DEADLINE_GRACE_SECONDS
    max_llm_calls: int = 200
    deadline_seconds: float = 1_200.0
    # Completion-token safety ceiling shared with the sub-agents; see
    # ``AGENT_MAX_OUTPUT_TOKENS``.
    max_response_tokens: int = AGENT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "max_llm_calls",
            "deadline_seconds",
            "max_response_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.deadline_grace_seconds < 0:
            raise ValueError("deadline_grace_seconds cannot be negative")


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
    # the child drains it before each model turn.
    steer: deque[str] = field(default_factory=deque)


@dataclass(frozen=True)
class AgentSessionResult:
    """A finished session (every other ending raises); ``session_end`` carries
    the full closing record."""

    conversation_id: str
    finish_value: dict[str, object]
    llm_calls: int
    # Closed's session-summary ``token_usage`` block: the eight per-call totals
    # plus ``cache_hit_ratio`` and the Sub Agent roll-up with its own ratio.
    usage: dict[str, object] = field(default_factory=dict)


class AgentSessionRunner:
    """Drive one persistent native-tool research-session conversation."""

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
        session_budget: SessionBudgetMeter | None = None,
        conversation_id: str | None = None,
        event_sink: Callable[[TraceEvent, dict[str, object]], None] | None = None,
        inbox: AgentInboxHook | None = None,
        trace_ref: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config or AgentSessionConfig()
        # The shared call and spend meter; without one only the time budget
        # has notices and a wrap-up.
        self.session_budget = session_budget
        self.compactor = compactor
        self.subagent = subagent
        # The run's opaque ref: the transcript file the compaction summary
        # points the Agent at.
        self.trace_ref = trace_ref
        # Compaction bookkeeping: the main-loop call count as the events read
        # it, the call the last compaction reached, and whether the advisory
        # went out since then.
        self._llm_calls = 0
        self._compacted_through_call = 0
        self._context_notice_sent = False
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
        # The context-window discipline around every main-conversation call;
        # each child runs its own fitter over its own model and compactor.
        self._context = ContextFitter(
            llm,
            max_tokens=self.config.max_response_tokens,
            compactor=compactor,
            result_store=self.tools.result_store(),
            record=self._record_context,
        )
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
        # The tail of the session clock the parent keeps for its wrap-up;
        # children and the parent's waits for them stop there.
        self._child_reserve_seconds = self._child_reserve_for_budget()
        if self.subagent is not None:
            self.subagent.parent_reserve_seconds = self._child_reserve_seconds
        self.conversation_id = conversation_id or f"conversation_{uuid.uuid4().hex}"
        self.event_sink = event_sink
        self.inbox = inbox
        # Node ids of the complete Validations: what the budget notices and
        # ``steps_used`` count.
        self._complete_validation_nodes: set[str] = set()
        self._wrap_up_sent = False
        self._subagent_attempts = 0
        self._subagent_roles: set[str] = set()
        # Failed tool calls of the session, the parent's and its children's, so
        # a session_end-only audit sees the same total the trace summary counts
        # without walking every event. Both kinds of event pass through
        # ``_locked_event_sink``, which is where they are counted.
        self._tool_failures = 0
        self._subagent_jobs: list[_SubAgentJob] = []
        self._subagent_pool: ThreadPoolExecutor | None = None
        self._subagent_totals: dict[str, int] | None = None
        self._usage = new_token_totals()
        # Snapshot of the conversation at the last tool dispatch; a child with
        # inherit_context forks from it.
        self._live_messages: list[ChatMessage] = []
        # Set only when the session closes; children poll it to stop early.
        self._cancelled = threading.Event()
        # Whether the session has already been closed: the teardown runs once,
        # whether the loop closed it itself or ``run`` is leaving on a failure.
        self._session_closed = False
        if self.subagent is not None:
            self.subagent.attach_cancel_event(self._cancelled)
        self._validate_capability_boundary()

    def run(
        self,
        instruction: str,
        *,
        preamble: Sequence[ChatMessage] = (),
        complete_validations: Sequence[Mapping[str, object]] = (),
        budget_total_seconds: float | None = None,
    ) -> AgentSessionResult:
        """Drive the conversation to its finish.

        ``preamble`` is inserted between the system prompt and the instruction:
        a resumed attempt puts the interrupted one's compaction summary there.
        ``complete_validations`` are the Validations earlier attempts recorded,
        seeded so the resumed session can test them.
        ``budget_total_seconds`` is the whole session budget the time-budget
        notices are measured against; a resumed attempt hands over the experiment
        total while its time budget only holds what is left of it.

        However the conversation ends — a finish, an exhausted budget, an
        interrupt raised through tool dispatch, a trace sink that failed or an
        unexpected bug — the session is closed exactly once: the children are
        cancelled, their pool is shut down and the failure reaches the caller
        unchanged.
        """

        try:
            return self._run_conversation(
                instruction,
                preamble=preamble,
                complete_validations=complete_validations,
                budget_total_seconds=budget_total_seconds,
            )
        except BaseException as exc:
            self._abort_session(exc)
            raise

    def _run_conversation(
        self,
        instruction: str,
        *,
        preamble: Sequence[ChatMessage] = (),
        complete_validations: Sequence[Mapping[str, object]] = (),
        budget_total_seconds: float | None = None,
    ) -> AgentSessionResult:
        """The conversation loop itself; ``run`` owns the session's teardown."""

        if not instruction.strip():
            raise ValueError("Agent instruction cannot be empty")
        time_budget = self.time_budget or InferenceTimeBudget(
            duration_seconds=self.config.deadline_seconds
        )
        budget_total = (
            float(budget_total_seconds)
            if budget_total_seconds is not None
            else max(time_budget.remaining(), 0.0)
        )
        notice_indices = dict.fromkeys(_BUDGET_REMAINING_KEYS, 0)
        backtests = dict.fromkeys(_BACKTEST_TOOLS, 0)
        messages = [
            ChatMessage("system", self.system_prompt),
            *preamble,
            ChatMessage("user", instruction.strip()),
        ]
        self._usage = new_token_totals()
        self._subagent_totals = None
        llm_calls = 0
        self._llm_calls = 0
        self._compacted_through_call = 0
        self._context_notice_sent = False
        accepted_steps = 0
        llm_failure_streak = 0
        malformed_reissue_used = False
        context_overflow_recovery_used = False
        self._complete_validation_nodes = set()
        for row in complete_validations:
            self._record_complete_validation(row)
        self._wrap_up_sent = False
        self._subagent_attempts = 0
        self._subagent_roles = set()
        self._tool_failures = 0
        self._subagent_jobs = []
        self._live_messages = []
        self._cancelled.clear()
        self._session_closed = False
        self._emit(
            TraceEvent.SESSION_START,
            {
                "system_prompt": self.system_prompt,
                "instruction": instruction.strip(),
                **(
                    {"preamble": [message.content or "" for message in preamble]}
                    if preamble
                    else {}
                ),
            },
        )

        while llm_calls < self.config.max_llm_calls:
            remaining = time_budget.remaining()
            if remaining <= 0:
                self._close_session(
                    {"status": "deadline_exceeded", "llm_calls": llm_calls}
                )
                raise AgentSessionBudgetExhausted(
                    "Agent session deadline and wrap-up grace exhausted",
                    conversation_id=self.conversation_id,
                    llm_calls=llm_calls,
                )
            provider_tools = self.tools.provider_tools()
            if not self._wrap_up_sent:
                levels = self._budget_levels(remaining, budget_total)
                self._wrap_up_if_due(messages, remaining, levels)
                if not self._wrap_up_sent:
                    self._budget_notices(messages, levels, notice_indices, backtests)
            messages, _ = self._context.compact(messages, provider_tools, remaining)
            messages = self._context_notice(messages, provider_tools)
            messages = self._append_subagent_observations(messages)
            messages = self._apply_inbox(
                messages, safe_point=INBOX_SAFE_BEFORE_LLM
            )

            try:
                messages, output_cap = self._context.fit(
                    messages, provider_tools, max(time_budget.remaining(), 0.0)
                )
                self._emit(
                    TraceEvent.LLM_CALL_STARTED,
                    {
                        "call_index": llm_calls + 1,
                        "status": "running",
                        "provider": self.llm.provider,
                        "model": self.llm.model,
                    },
                )
                response = self.llm.complete(
                    messages,
                    tools=provider_tools,
                    tool_choice="auto",
                    max_tokens=output_cap,
                )
                llm_calls += 1
                self._llm_calls = llm_calls
                llm_failure_streak = 0
                malformed_reissue_used = False
            except Exception as exc:
                llm_calls += 1
                self._llm_calls = llm_calls
                llm_failure_streak += 1
                error = safe_error_summary(exc)
                malformed = isinstance(exc, MalformedToolCallError)
                failure: dict[str, object] = {
                    "call_index": llm_calls,
                    "status": "error",
                    "provider": self.llm.provider,
                    "model": self.llm.model,
                    "error": error,
                }
                if malformed:
                    # Countable in an audit: this class costs one turn, not a
                    # repeated generation.
                    failure["error_type"] = "malformed_tool_call"
                    failure.update(trace_reasoning(exc.reasoning_content))
                elif isinstance(exc, ProviderRefusalError):
                    failure["error_type"] = "provider_refusal"
                self._emit(TraceEvent.LLM_CALL, failure)
                if is_context_overflow_error(exc):
                    if not context_overflow_recovery_used:
                        messages, progressed = self._context.recover(
                            messages, provider_tools, max(time_budget.remaining(), 0.0)
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
                    raise AgentSessionBudgetExhausted(
                        "Agent session deadline and wrap-up grace exhausted",
                        conversation_id=self.conversation_id,
                        llm_calls=llm_calls,
                    ) from exc
                if isinstance(exc, ApiSpendExhausted):
                    self._close_session(
                        {"status": API_SPEND_EXHAUSTED, "llm_calls": llm_calls}
                    )
                    raise AgentSessionBudgetExhausted(
                        str(exc),
                        finish_reason=API_SPEND_EXHAUSTED,
                        conversation_id=self.conversation_id,
                        llm_calls=llm_calls,
                    ) from exc
                if isinstance(exc, CallBudgetExhausted):
                    self._close_session(
                        {
                            "status": "call_budget_exhausted",
                            "llm_calls": llm_calls,
                        }
                    )
                    raise AgentSessionBudgetExhausted(
                        "Agent session LLM call budget exhausted",
                        finish_reason=LLM_CALL_BUDGET_EXHAUSTED,
                        conversation_id=self.conversation_id,
                        llm_calls=llm_calls,
                    ) from exc
                if llm_failure_streak >= _LLM_FAILURE_CIRCUIT:
                    self._close_session(
                        {"status": "llm_unavailable", "llm_calls": llm_calls}
                    )
                    raise RuntimeError(
                        "Agent language model unavailable after consecutive "
                        f"failures; last error: {safe_error_summary(exc, 240)}"
                    ) from exc
                if malformed and not malformed_reissue_used:
                    # One re-issue per streak: keep the text that did arrive
                    # and ask only for the call again. A second failure falls
                    # through to the generic handling below.
                    malformed_reissue_used = True
                    messages.extend(malformed_tool_call_messages(exc, error=error))
                    continue
                observation: dict[str, object] = {
                    "observation": "llm_error",
                    "error": error,
                    "retry_hint": PROVIDER_TRANSIENT_FAILURE_HINT
                    if isinstance(exc, LLMProxyError) and exc.retryable
                    else "Proceed with a different bounded action; do not repeat the same failing request verbatim.",
                }
                messages.append(
                    ChatMessage(
                        "user",
                        json.dumps(observation, ensure_ascii=False, allow_nan=False),
                    )
                )
                continue

            accumulate_usage(self._usage, response.usage)
            messages.append(
                ChatMessage(
                    "assistant",
                    response.content or None if response.tool_calls else response.content,
                    response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
            )
            self._emit(
                TraceEvent.LLM_CALL,
                {
                    "call_index": llm_calls,
                    "status": "ok",
                    "provider": self.llm.provider,
                    "model": response.model,
                    "usage": dict(response.usage),
                    "content": response.content,
                    **trace_reasoning(response.reasoning_content),
                    "tool_names": [call.name for call in response.tool_calls],
                },
            )
            if not response.tool_calls:
                if output_truncated(response.usage, output_cap):
                    # The whole budget went into thinking (or an unfinished
                    # essay): say so, ask for a concise continuation, and do
                    # not let the next turn silently repeat the same.
                    completion = int(dict(response.usage).get("completion_tokens") or 0)
                    self._emit(
                        TraceEvent.OUTPUT_TRUNCATED,
                        {
                            "call_index": llm_calls,
                            "completion_tokens": completion,
                            "max_tokens": output_cap,
                        },
                    )
                    messages.append(
                        ChatMessage(
                            "user",
                            json.dumps(
                                {
                                    "observation": "output_truncated",
                                    "completion_tokens": completion,
                                    "max_tokens": output_cap,
                                    "message": OUTPUT_TRUNCATED_CONTINUATION.format(
                                        limit=output_cap
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
                    "retry_hint": "Text alone does nothing: continue the research with the injected tools.",
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
                    TraceEvent.TOOL_CALL,
                    {
                        "call_index": llm_calls,
                        "tool_call_id": call.id,
                        "tool": call.name,
                        "arguments": sanitize_for_log(traced_arguments),
                        # A read of the trace is logged as the call and its
                        # size, never as the content it would re-log.
                        "result": sanitize_for_log(
                            trace_read_stub(call.name, call.arguments, record)
                        ),
                    },
                )
            messages = self._compact_on_request(messages, results)
            accepted_steps = len(self._complete_validation_nodes)
            for call, _record in results:
                if call.name in backtests:
                    backtests[call.name] += 1

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
                    finish_value=finish,
                    llm_calls=llm_calls,
                    usage=token_usage,
                )
            messages = self._apply_inbox(messages, safe_point=apply_point)

        self._close_session(
            {
                "status": "call_budget_exhausted",
                "llm_calls": self.config.max_llm_calls,
                "steps_used": accepted_steps,
            }
        )
        raise AgentSessionBudgetExhausted(
            "Agent session LLM call budget exhausted",
            finish_reason=LLM_CALL_BUDGET_EXHAUSTED,
            conversation_id=self.conversation_id,
            llm_calls=self.config.max_llm_calls,
        )

    def _budget_levels(self, remaining: float, total: float) -> list[_BudgetLevel]:
        """Where each session budget stands: the inference time against the
        whole session budget, and the shared API spend and model calls when the
        pipeline handed over their meter."""

        levels: list[_BudgetLevel] = []
        if total > 0:
            levels.append(
                _BudgetLevel(
                    "inference_time",
                    1.0 - remaining / total,
                    round(max(remaining, 0.0) / 60.0, 1),
                )
            )
        meter = self.session_budget
        if meter is not None:
            spend, spend_cap = meter.api_spend_usd, meter.max_api_spend_usd
            levels.append(
                _BudgetLevel(
                    "api_spend", spend / spend_cap, round(max(spend_cap - spend, 0.0), 2)
                )
            )
            levels.append(
                _BudgetLevel(
                    "llm_calls",
                    meter.calls / meter.max_calls,
                    max(meter.max_calls - meter.calls, 0),
                )
            )
        return levels

    def _wrap_up_if_due(
        self,
        messages: list[ChatMessage],
        remaining: float,
        levels: Sequence[_BudgetLevel],
    ) -> None:
        """Send the wrap-up prompt when the first budget reaches its wrap-up
        point: the time budget its grace window, the API spend or the model
        calls ``BUDGET_WRAP_UP_FRACTION`` of their caps. One sentence naming
        that budget leads the prompt."""

        extra: dict[str, object] = {}
        if remaining <= self.config.deadline_grace_seconds:
            budget = "inference_time"
            minutes = round(max(remaining, 0.0) / 60.0, 1)
            lead = f"推理时间已到主截止，剩下的约 {minutes:g} 分钟是收尾宽限。"
        else:
            level = next(
                (
                    level
                    for level in levels
                    if level.budget != "inference_time"
                    and level.used >= BUDGET_WRAP_UP_FRACTION
                ),
                None,
            )
            if level is None:
                return
            budget = level.budget
            lead = f"{_budget_spent_phrase(level, level.used)}。"
            extra = {
                "used_fraction": round(level.used, 4),
                _BUDGET_REMAINING_KEYS[budget]: level.left,
            }
        messages.append(ChatMessage("user", lead + WRAP_UP_PROMPT))
        self._wrap_up_sent = True
        self._emit(
            TraceEvent.WRAP_UP_STARTED,
            {
                "budget": budget,
                **extra,
                "remaining_seconds": round(remaining, 6),
                "grace_seconds": self.config.deadline_grace_seconds,
            },
        )

    def _budget_notices(
        self,
        messages: list[ChatMessage],
        levels: Sequence[_BudgetLevel],
        notice_indices: dict[str, int],
        backtests: Mapping[str, int],
    ) -> None:
        """Inject one ``budget_notice`` per budget that crossed a notice
        fraction since its last one; crossing several at once yields one."""

        fractions = BUDGET_NOTICE_FRACTIONS
        for level in levels:
            index = notice_indices[level.budget]
            crossed: float | None = None
            while index < len(fractions) and level.used >= fractions[index]:
                crossed = fractions[index]
                index += 1
            notice_indices[level.budget] = index
            if crossed is None:
                continue
            complete = len(self._complete_validation_nodes)
            payload: dict[str, object] = {
                "observation": "budget_notice",
                "budget": level.budget,
                "used_fraction": crossed,
                _BUDGET_REMAINING_KEYS[level.budget]: level.left,
                "smoke_backtests": backtests["smoke_backtest"],
                "validates": backtests["validate"],
                "tests": backtests["test"],
                "complete_validations": complete,
            }
            advice = _BUDGET_NOTICE_ADVICE.get(level.budget)
            payload["message"] = (
                f"{_budget_spent_phrase(level, crossed)}；"
                f"至今 smoke_backtest {backtests['smoke_backtest']} 次、"
                f"validate {backtests['validate']} 次、test {backtests['test']} 次、"
                f"完整 Validation {complete} 个。"
                + (f"{advice}；" if advice else "")
                + "预算耗尽时宿主以最近一次 test 的候选补完记录。"
            )
            messages.append(
                ChatMessage("user", json.dumps(payload, ensure_ascii=False))
            )
            self._emit(
                TraceEvent.BUDGET_NOTICE,
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"observation", "message"}
                },
            )

    def _record_complete_validations(self, record: dict[str, object]) -> None:
        """Register every complete Validation node one tool result carries.

        ``validate`` reports one row per candidate and only the rows that
        finished are nodes: the complete Validations the budget notices count
        and a resumed attempt is handed.
        """
        value = record.get("value")
        rows = value.get("candidates") if isinstance(value, Mapping) else None
        for row in rows if isinstance(rows, list) else ():
            if isinstance(row, Mapping):
                self._record_complete_validation(row)

    def _record_complete_validation(self, value: Mapping[str, object]) -> None:
        # A complete Validation node carries both ids; a failed row has neither.
        node_id = value.get("node_id")
        revision_id = value.get("revision_id")
        if (
            isinstance(node_id, str)
            and node_id
            and isinstance(revision_id, str)
            and revision_id
        ):
            self._complete_validation_nodes.add(node_id)

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
            started: dict[str, object] = {
                "tool": call.name,
                "tool_call_id": call.id,
                "status": "running",
            }
            if call.name in _VALIDATION_TOOLS:
                # A batch runs for hours, and the tool_call event that also
                # carries its arguments comes only once it returns: record its
                # candidates, hypotheses, controls and span now.
                started["arguments"] = sanitize_for_log(dict(call.arguments))
            self._emit(TraceEvent.TOOL_CALL_STARTED, started)
            if time_budget.remaining() <= 0:
                return call, {
                    "ok": False,
                    "error": "Agent session deadline reached before tool dispatch",
                }
            if call.name in _TERMINAL_TOOLS or call.name in _WRITER_BARRIER_TOOLS:
                # Barrier: a formal backtest, null control or finish must not
                # overlap a developer child that may still be writing the
                # workspace. A finish waits for every child (the close discards
                # their reports); a backtest waits only for children whose role
                # can write, so a read-only audit never delays a ready turn,
                # and refuses rather than snapshot candidates a writer may
                # still be changing.
                self._wait_subagent_jobs(
                    writers_only=call.name in _WRITER_BARRIER_TOOLS
                )
                if call.name in _TERMINAL_TOOLS:
                    in_flight = self._in_flight_subagent_error()
                    if in_flight:
                        return call, in_flight
                else:
                    writers = [
                        {"task_id": job.task_id, "role": job.role}
                        for job in self._subagent_jobs
                        if not job.future.done() and self._subagent_can_write(job)
                    ]
                    if writers:
                        return call, {
                            "ok": False,
                            "error_type": "subagent_writers_in_flight",
                            "error": (
                                "a formal backtest or null control cannot start "
                                "while a child that can write the workspace is "
                                "still running; "
                                "wait for its subagent_completed or message it "
                                "to report now"
                            ),
                            "writers": writers,
                        }
            self._call_context.call_id = call.id
            record = self.tools.invoke(
                call.name,
                call.arguments,
            ).to_record()
            if call.name in _VALIDATION_TOOLS and record.get("ok") is True:
                # A successful backtest result is a complete Validation node;
                # a batch result carries one per candidate that finished.
                self._record_complete_validations(record)
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

    def _in_flight_subagent_error(self) -> dict[str, object] | None:
        """Refuse a terminal tool while a child launched by this session runs.

        The barrier before this already waited ``SUBAGENT_TEARDOWN_WAIT_SECONDS``
        for the children to return. Whatever is still running would be cancelled
        by the session close that a successful finish triggers: a session that
        launched three sub-agents and finished in the same turn discarded all
        three reports.

        Two exemptions, because a refusal that cannot be satisfied loses more
        than the reports it protects. The session's own tail, the wrap-up grace,
        is where finishing is the only remaining move (the host completes the
        record once the budget is spent). And whatever the phase, a child
        that outlives the teardown barrier's own window is one the session no
        longer has time to wait for.
        """

        if self._subagent_wait_floor() is None:
            return None
        remaining = self.time_budget.remaining() if self.time_budget is not None else None
        if remaining is not None and remaining <= SUBAGENT_TEARDOWN_WAIT_SECONDS:
            return None
        with self._subagent_lock:
            picture = self._subagent_live_picture()
        running = picture["running_children"]
        queued = picture["queued_children"]
        if not running and not queued:
            return None
        return {
            "ok": False,
            "error": (
                "terminal tool refused: "
                f"{len(running) + len(queued)} sub-agent(s) launched by this "
                "session have not returned. Their results would be discarded "
                "with the session. Wait for their subagent_completed "
                "observations, or steer them to report now with "
                "agent(action=\"message\"), then call this tool again."
            ),
            "error_type": "subagents_in_flight",
            **picture,
        }

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
                TraceEvent.TOOL_SKIPPED,
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
                TraceEvent.USER_MESSAGE,
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
            max_turns = resolve_subagent_max_turns(
                arguments.get("max_turns"), role, self.subagent.config.max_turns
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
            tuple(self._live_messages)
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
                max_turns=max_turns,
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

        The child reads it before its next model turn (a queued child before
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
            TraceEvent.SUBAGENT_STEER,
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
            "delivered_at_turn": None,
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
            wait=wait, timeout=SUBAGENT_TEARDOWN_WAIT_SECONDS if wait else None
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
                for key in ("turns", "tool_calls"):
                    if isinstance(value.get(key), int):
                        payload[key] = value[key]
                if value.get("truncated"):
                    # A cut-off report is not a complete one; the marker at
                    # the tail of a long summary is easy to miss.
                    payload["truncated"] = True
                for key in ("truncated_turns", "llm_errors", "steers"):
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

    def _child_time_reserve_seconds(self) -> float:
        """The tail of the session budget that belongs to the parent alone:
        the wrap-up grace, in which the parent writes up what it knows while
        the host completes the record. Nothing else may consume it -- neither
        the parent's own wait for a child nor the child's own loop, which the
        engine bounds with the same figure.
        """

        return self.config.deadline_grace_seconds

    def _child_reserve_for_budget(self) -> float:
        """The tail to withhold from the children of THIS session.

        A child shares the session clock, so it must stop before the parent's
        wrap-up window opens: the child cancelled at the parent's deadline
        had spent shared LLM calls on a report nobody could receive. The tail
        is withheld only while there is a session outside it — a budget no
        larger than the tail belongs to a session that starts inside its own
        wrap-up window and has no tail left to reserve.
        """

        reserve = self._child_time_reserve_seconds()
        total = (
            self.time_budget.remaining()
            if self.time_budget is not None
            else self.config.deadline_seconds
        )
        return reserve if reserve < total else 0.0

    def _subagent_wait_floor(self) -> float | None:
        """Remaining seconds at which the parent stops waiting for a child.

        ``None`` means it must not wait at all: once the wrap-up prompt is out,
        what is left is finishing time. A session once spent 55 minutes inside
        one such wait and crossed the main deadline and the whole grace without
        ever reaching the loop top.
        """

        if self._wrap_up_sent:
            return None
        return self._child_reserve_seconds

    def _yield_for_pending_subagent(self, time_budget: InferenceTimeBudget) -> bool:
        """Skip no_tool_call when a sub-agent is still running; wait for progress.

        Returns False once the wait floor is reached, so the turn falls through
        to the ordinary nudge and the loop reaches its next top — where the
        wrap-up notice and the deadline are checked — instead of yielding the
        rest of the session.
        """
        uncollected = self._uncollected_subagent_jobs()
        if not uncollected:
            return False
        if any(job.future.done() for job in uncollected):
            return True
        floor = self._subagent_wait_floor()
        if floor is None or time_budget.remaining() <= floor:
            return False
        self._wait_first_pending_subagent(time_budget, floor)
        return True

    def _wait_first_pending_subagent(
        self, time_budget: InferenceTimeBudget, floor: float = 0.0
    ) -> None:
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
        self._emit(
            TraceEvent.SUBAGENT_WAIT_STARTED,
            {"pending": len(pending), "wait_floor_seconds": round(floor, 6)},
        )
        for future in pending:
            future.add_done_callback(_on_done)
            if future.done():
                return
        while not completed.is_set() and not self._cancelled.is_set():
            waitable = time_budget.remaining() - floor
            if waitable <= 0:
                return
            completed.wait(timeout=min(0.05, waitable))

    def _subagent_can_write(self, job: _SubAgentJob) -> bool:
        """Whether the child's role can mutate the workspace: the writing role
        is the one with the shell; the read-only role has no mutating tool."""

        return subagent_role(job.role).shell

    def _wait_subagent_jobs(self, *, writers_only: bool = False) -> list[dict[str, object]]:
        return self._collect_finished_subagents(
            wait=True, timeout=SUBAGENT_TEARDOWN_WAIT_SECONDS, writers_only=writers_only
        )

    def _collect_finished_subagents(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        writers_only: bool = False,
    ) -> list[dict[str, object]]:
        """The one place a child's result is taken: usage and Trace here, the
        conversation observation later via ``_append_subagent_observations``.

        ``writers_only`` limits the wait to children whose role can write;
        finished read-only children are still collected, running ones skipped."""

        finished: list[dict[str, object]] = []
        wait_timeout = timeout if wait else None
        deadline = (
            time.monotonic() + wait_timeout if wait_timeout is not None else None
        )
        for job in self._subagent_jobs:
            if job.record is not None:
                continue
            result: object
            if wait and not job.future.done() and writers_only and not self._subagent_can_write(job):
                continue
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
                    self._subagent_totals = {"llm_calls": 0, **new_token_totals()}
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
            self._emit(TraceEvent.SUBAGENT_ATTEMPT, attempt)
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
                TraceEvent.SUBAGENT_ATTEMPT,
                {
                    "attempt": job.attempt,
                    "role": job.role,
                    "ok": False,
                    "status": "cancelled",
                    "task_id": job.task_id,
                },
            )

    def _close_session(self, payload: dict[str, object]) -> None:
        if self._session_closed:
            return
        self._session_closed = True
        self._cancelled.set()
        if self.subagent is not None:
            self.subagent.cancel()
        try:
            # Bound the wait; do not abort an in-flight LLM or tool invoke.
            # After an uncancelable LLM returns, the child checks this event and
            # exits without dispatching. Pending (not started) jobs are cancelled.
            self._collect_finished_subagents(
                wait=True, timeout=SUBAGENT_TEARDOWN_WAIT_SECONDS
            )
            self._cancel_pending_subagents()
            payload = dict(payload)
            payload.setdefault(
                "token_usage", _token_usage_summary(self._usage, self._subagent_totals)
            )
            payload.setdefault("tool_failures", self._tool_failures)
            self._emit(TraceEvent.SESSION_END, payload)
        finally:
            # Collecting a child and writing the terminal event both touch the
            # trace sink and the result store; a failure there still leaves the
            # pool this session owns to release.
            pool = self._subagent_pool
            self._subagent_pool = None
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

    def _abort_session(self, exc: BaseException) -> None:
        """Close a session ``run`` is leaving on an exception.

        A session that already closed itself — a finish, an exhausted budget —
        is done; any other exit still has to cancel its children and shut their
        pool down. The terminal event names the failure that is on its way out,
        and a sink that cannot record it does not get to replace it.
        """

        if self._session_closed:
            return
        # The original failure must propagate, whatever the teardown hits.
        with contextlib.suppress(Exception):
            self._close_session({"status": "error", "error": safe_error_summary(exc)})

    def _record_context(self, event: TraceEvent, payload: dict[str, object]) -> None:
        """What the context fitter did, as this loop's own events."""

        if event == TraceEvent.CONTEXT_COMPACTION:
            self._record_compaction("runtime", payload)
        else:
            self._emit(event, payload)

    def _compact_on_request(
        self,
        messages: list[ChatMessage],
        results: Sequence[tuple[ToolCall, dict[str, object]]],
    ) -> list[ChatMessage]:
        """Rebuild the conversation around the summary a ``compact`` call handed in.

        Runs once the turn's tool results are appended, so the tail keeps the
        protocol pairs of that turn; the last accepted call of a turn wins.
        """

        summaries = [
            str(call.arguments.get("summary") or "")
            for call, record in results
            if call.name == _COMPACT_TOOL and record.get("ok") is True
        ]
        if not summaries:
            return messages
        result = compact_with_summary(
            messages,
            summaries[-1],
            keep_recent_messages=self._compact_keep_recent_messages(),
            trace_ref=self.trace_ref,
        )
        self._record_compaction("agent", result.event)
        return list(result.messages)

    def _record_compaction(self, trigger: str, event: Mapping[str, object]) -> None:
        """One ``context_compaction`` event for either trigger, locating the
        calls whose messages it replaced so the trace doubles as the
        compaction checkpoint a resume starts from. A failed attempt is
        recorded but moves neither the checkpoint nor the advisory."""

        payload: dict[str, object] = {
            "trigger": trigger,
            "call_index": self._llm_calls,
            "replaced_call_range": [self._compacted_through_call + 1, self._llm_calls],
            **event,
        }
        self._emit(TraceEvent.CONTEXT_COMPACTION, payload)
        if event.get("status") == "ok":
            self._compacted_through_call = self._llm_calls
            self._context_notice_sent = False

    def _compact_keep_recent_messages(self) -> int:
        # One tail length for both triggers: the runtime compactor's, when one
        # is wired, else the dataclass default.
        if self.compactor is not None:
            return self.compactor.config.keep_recent_messages
        return ContextCompactionConfig().keep_recent_messages

    def _context_notice(
        self,
        messages: list[ChatMessage],
        provider_tools: tuple[dict[str, object], ...],
    ) -> list[ChatMessage]:
        """Tell the Agent once per context window that it is time to compact."""

        if self._context_notice_sent:
            return messages
        threshold = (
            self.compactor.config.token_threshold
            if self.compactor is not None
            else DEFAULT_COMPACTION_TOKEN_THRESHOLD
        )
        estimated = estimate_chat_request_tokens(messages, tools=provider_tools)
        if estimated < COMPACT_ADVISORY_FRACTION * threshold:
            return messages
        self._context_notice_sent = True
        payload = {
            "observation": "context_notice",
            "estimated_tokens": estimated,
            "token_threshold": threshold,
            "message": (
                f"上下文估算已达到压缩阈值的 {COMPACT_ADVISORY_FRACTION:.0%}。请在当前这一回合的结果消化完后"
                "调用 compact(summary=...)：把策略现状、已做的决定与被否定的方向（带读数与节点 id）、"
                "未完成的线索与 trace 里的定位写进摘要；达到阈值仍未压缩时宿主用压缩模型代劳。"
            ),
        }
        messages.append(ChatMessage("user", json.dumps(payload, ensure_ascii=False)))
        self._emit(
            TraceEvent.CONTEXT_NOTICE,
            {key: value for key, value in payload.items() if key not in ("observation", "message")},
        )
        return messages

    def _validate_capability_boundary(self) -> None:
        """Fail closed on any tool outside the session's documented set."""

        names = {spec.name for spec in self.tools.specs()}
        unsupported = sorted(names - _SESSION_TOOLS)
        if unsupported:
            raise ValueError(f"Agent session received unsupported tools: {unsupported}")
        producing = sorted(names & _VALIDATION_TOOLS)
        if producing and "test" not in names:
            raise ValueError(f"a session with {producing[0]} requires test")

    def _locked_event_sink(self, event: TraceEvent, payload: dict[str, object]) -> None:
        with self._event_lock:
            if event in _TOOL_OUTCOME_EVENTS and _failed_tool_event(payload):
                self._tool_failures += 1
            if self.event_sink is not None:
                self.event_sink(event, payload)

    def _emit(self, event: TraceEvent, payload: dict[str, object]) -> None:
        record = dict(payload)
        if event == TraceEvent.SESSION_END and self.subagent is not None:
            record["subagent_attempts"] = self._subagent_attempts
            record["subagent_roles"] = sorted(self._subagent_roles)
        self._locked_event_sink(event, record)


def _accumulate_subagent_usage(
    totals: dict[str, int], result: Mapping[str, object]
) -> None:
    """Roll one child's totals into the session's ``subagent`` block.

    Sub-agent calls bill the same provider account but are summed by the
    child's own loop, into the same counters (``usage_totals``); without this
    roll-up the session summary understates real cost by up to ~15% in
    observed sessions. A child that crashed before reporting adds nothing."""

    calls = result.get("llm_calls")
    if isinstance(calls, int) and not isinstance(calls, bool):
        totals["llm_calls"] += calls
    usage = result.get("usage_totals")
    if isinstance(usage, Mapping):
        for key, value in usage.items():
            totals[key] += int(value)


def _with_cache_hit_ratio(totals: Mapping[str, int]) -> dict[str, object]:
    summary: dict[str, object] = dict(totals)
    prompt = totals["prompt_tokens"]
    hits = totals["cache_hit_tokens"]
    reported = hits + totals["cache_miss_tokens"]
    # ``None`` when no call reported a cache count (vLLM): unknown, not zero.
    summary["cache_hit_ratio"] = round(hits / prompt, 4) if prompt and reported else None
    return summary


def _token_usage_summary(
    totals: Mapping[str, int], subagent: Mapping[str, int] | None
) -> dict[str, object]:
    summary = _with_cache_hit_ratio(totals)
    if subagent is not None:
        summary["subagent"] = _with_cache_hit_ratio(subagent)
        summary["total_tokens_including_subagents"] = (
            totals["total_tokens"] + subagent["total_tokens"]
        )
    return summary
