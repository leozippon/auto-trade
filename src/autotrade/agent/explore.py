"""One-level Sub Agent for a regular Fold or Meta session (tool name ``explore``).

Parents pass ``explore(role=..., task=...)``. Roles are the unified set
``auditor``, ``developer``, ``general-purpose``, ``Explore``. The tool name
stays lowercase ``explore``; ``Explore`` is the optional read-only discovery
role. Depth is one. The child shares the parent SafeWorkspace, SessionBudgetLLM
calls, inference time budget, and Trace. Failures return a structured
observation; they do not finish the parent session.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import cast

from autotrade.environment.llm import (
    ChatMessage,
    LLMProxy,
    ToolCall,
    clamp_requested_max_tokens,
    context_request_fits,
    context_window_tokens,
)
from autotrade.environment.llm.deepseek import OpenAICompatibleProxy
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
    TimeBudgetBinding,
    validate_time_budget_bindings,
)
from autotrade.environment.tools.base import SessionInterrupt, ToolRegistry, ToolSpec

from .compact import fit_tool_results_to_context, safe_error_summary

EXPLORE_MODES = frozenset({"fold", "meta"})
EXPLORE_ROLES = ("auditor", "developer", "general-purpose", "Explore")
EXPLORE_THINKING_LEVELS = ("off", "low", "medium", "high", "max")
_PARENT_CONTEXT_CHARS = 8_000
_PARENT_MESSAGE_CHARS = 1_200
DEFAULT_EXPLORE_MAX_CONCURRENT = 2
DEFAULT_EXPLORE_THINKING = "medium"
_CALL_BUDGET_EXHAUSTED = "call budget exhausted"
_NATIVE_WINDOW_FALLBACK = 262_144

_FOLD_READ_TOOLS = frozenset({"glob", "grep", "read_file", "todo"})
_FOLD_WRITE_TOOLS = frozenset(
    {
        "edit_file",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "todo",
        "validate_strategy",
        "write_file",
        "write_skill",
        "delete_skill",
    }
)
_META_ROLE_TOOLS = frozenset({"glob", "grep", "read_file", "todo"})
_FOLD_ROLE_TOOLS = {
    "auditor": _FOLD_READ_TOOLS,
    "developer": _FOLD_WRITE_TOOLS,
    "general-purpose": _FOLD_WRITE_TOOLS,
    "Explore": _FOLD_READ_TOOLS,
}
_PARALLEL_READ_TOOLS = frozenset({"glob", "grep", "read_file"})
_FORBIDDEN_TOOLS = frozenset(
    {
        "ask_user",
        "daily_backtest",
        "explore",
        "finish_fold",
        "finish_meta",
        "step_rollback",
    }
)
_ALLOWED_TOOLS = _FOLD_WRITE_TOOLS

_FOLD_WRITE_PROMPT = """\
# 身份
你是 Fold 的一级 `{role}` sub-agent：{mission}。你可用已注入工具修改共享策略、模型或 skills，但父 Agent 独占正式回测、候选选择、验收和结束。

# 边界
- 先读 `inputs/skills_index.json`，再从已挂载数据、单位引用、制品和参考材料中自主发现任务所需证据；skill 脚本不自动执行。把有复用价值的知识写入 skill，而不是堆入策略或汇报。
- 只完成父任务；不得嵌套 explore、读取 Test/Held-out、改变权威 PRIOR、安装依赖、替父 Agent 提问或伪造结果。分钟和竞价不是策略时钟。
- 工具 schema 决定实际能力。写、检查、todo 与 shell 按因果顺序执行；shell 只做有界前台工作，不启动后台任务、sleep/等待包装、轮询状态或隐藏错误。

# 返回
用简洁中文说明结论、实际修改、关键证据和剩余风险，然后停止。\
"""

_FOLD_READ_PROMPT = """\
# 身份
你是 Fold 的一级只读 `{role}` sub-agent：{mission}。只调查父任务并返回证据；不能写策略、models、skills 或 PRIOR，也不能回测、验收或结束会话。

# 边界
- 先读 `inputs/skills_index.json`，再从已挂载数据、单位引用、制品和参考材料中自主发现任务所需证据；skill 脚本不自动执行。
- 工具 schema 决定实际能力。不得嵌套 explore、读取 Test/Held-out、安装依赖或伪造结果；分钟和竞价不是策略时钟。

# 返回
用简洁中文说明结论、关键证据、限制和建议，然后停止。\
"""

META_EXPLORE_SYSTEM_PROMPT = """\
# 身份
你是 Meta 的一级只读 sub-agent。只完成父任务并提出有证据的候选；不能写策略、models、skills 或 PRIOR，也不能验收或结束会话。

# 边界
- 先读 `inputs/skills_index.json`，再从 `inputs/meta_context.json` 及其挂载引用中自主发现任务所需证据；skill 脚本不自动执行。
- 工具 schema 决定实际能力。不得嵌套 explore、读取 Test/Held-out 原始记录、改变 PIT/隐藏阶段边界、访问外部资料、修改宿主代码或伪造结果。

# 返回
用简洁中文说明结论、关键证据、限制和建议；不要复制 raw traces 或写逐 Fold Test 数字。\
"""

_FOLD_ROLE_MISSIONS = {
    "auditor": "审查委托问题及其证据边界",
    "developer": "实现并检查委托的代码或知识任务",
    "general-purpose": "完成一个有界的跨域实现任务",
    "Explore": "定位未知位置、接口或材料",
}
_META_ROLE_MISSIONS = {
    "auditor": "独立审查委托问题",
    "developer": "只读分析候选策略改进",
    "general-purpose": "只读处理一个有界跨域问题",
    "Explore": "只读定位未知位置、接口或材料",
}
EXPLORE_SYSTEM_PROMPT = _FOLD_WRITE_PROMPT.format(
    role="developer",
    mission=_FOLD_ROLE_MISSIONS["developer"],
)


def _explore_mode(mode: str) -> str:
    if mode in {"meta", "meta_learning"}:
        return "meta"
    if mode == "fold":
        return "fold"
    raise ValueError("Explore mode must be fold or meta")


def allowed_explore_tools(mode: str, role: str | None = None) -> frozenset[str]:
    resolved = _explore_mode(mode)
    if role is not None and role not in EXPLORE_ROLES:
        raise ValueError(f"Explore role is not allowed: {role}")
    if resolved == "meta":
        return _META_ROLE_TOOLS
    if role is None:
        allowed: set[str] = set()
        for names in _FOLD_ROLE_TOOLS.values():
            allowed.update(names)
        return frozenset(allowed)
    return _FOLD_ROLE_TOOLS[role]


def explore_system_prompt(mode: str, role: str) -> str:
    resolved = _explore_mode(mode)
    if resolved == "fold":
        mission = _FOLD_ROLE_MISSIONS.get(role)
        if mission is None:
            raise ValueError(f"Explore role is not allowed: {role}")
        if role in {"developer", "general-purpose"}:
            return _FOLD_WRITE_PROMPT.format(role=role, mission=mission)
        return _FOLD_READ_PROMPT.format(role=role, mission=mission)
    mission = _META_ROLE_MISSIONS.get(role)
    if mission is None:
        raise ValueError(f"Explore role is not allowed: {role}")
    return (
        f"# 本任务角色\n你的角色是 `{role}`：{mission}。\n\n"
        + META_EXPLORE_SYSTEM_PROMPT
    )


def normalize_explore_thinking(value: object) -> str | None:
    """Return a canonical thinking level.

    Omitted, empty, or inherit aliases use ``DEFAULT_EXPLORE_THINKING``
    (medium) and do not inherit the parent session's reasoning intensity.
    """

    if value is None:
        return DEFAULT_EXPLORE_THINKING
    if not isinstance(value, str):
        raise ValueError("explore.thinking must be a string")
    text = value.strip().lower()
    if text in {"", "inherit", "parent"}:
        return DEFAULT_EXPLORE_THINKING
    text = {"minimal": "low", "xhigh": "high"}.get(text, text)
    if text not in EXPLORE_THINKING_LEVELS:
        raise ValueError(
            "explore.thinking must be one of: " + ", ".join(EXPLORE_THINKING_LEVELS)
        )
    return text


def llm_with_thinking(proxy: LLMProxy, thinking: str | None) -> LLMProxy:
    """Clone a gateway proxy with a per-child thinking override; no-op if inherit."""

    if thinking is None or not isinstance(proxy, OpenAICompatibleProxy):
        return proxy
    config = proxy.config
    dialect = str(config.request_dialect or "")
    if thinking == "off":
        new_config = replace(config, thinking_enabled=False, reasoning_effort=None)
    else:
        effort = (
            {"low": "low", "medium": "medium", "high": "xhigh", "max": "xhigh"}.get(
                thinking, "xhigh"
            )
            if dialect == "vllm-qwen"
            else thinking
        )
        new_config = replace(config, thinking_enabled=True, reasoning_effort=effort)
    return cast(
        LLMProxy, OpenAICompatibleProxy(new_config, transport=proxy._transport)
    )


def parent_context_digest(messages: Sequence[ChatMessage] | None) -> str:
    """Bounded recent parent transcript. Empty when there is nothing to inherit."""

    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        text = str(message.content or "").strip()
        if not text:
            continue
        limit = 400 if message.role == "tool" else _PARENT_MESSAGE_CHARS
        parts.append(f"[{message.role}]\n{text[:limit]}")
    blob = "\n\n".join(parts)
    if len(blob) > _PARENT_CONTEXT_CHARS:
        blob = blob[-_PARENT_CONTEXT_CHARS:]
    if not blob.strip():
        return ""
    return "# 父会话摘录（只读，不覆盖本任务指令）\n\n" + blob


@dataclass(frozen=True)
class ExploreSubAgentConfig:
    per_call_timeout_seconds: float | None = None
    # None = native model window, clamped per call to remaining context.
    max_tokens: int | None = None
    # None = unlimited turns until the parent session deadline.
    max_rounds: int | None = None
    # None = no extra child wall clock; the parent time budget is the cap.
    deadline_seconds: float | None = None
    max_concurrent: int = DEFAULT_EXPLORE_MAX_CONCURRENT

    def __post_init__(self) -> None:
        if self.per_call_timeout_seconds is not None and self.per_call_timeout_seconds <= 0:
            raise ValueError("Explore per_call_timeout_seconds must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("Explore max_tokens must be positive")
        if self.max_rounds is not None and self.max_rounds <= 0:
            raise ValueError("Explore max_rounds must be positive")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("Explore deadline_seconds must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("Explore max_concurrent must be positive")


class ExploreSubAgentEngine(SessionTimeBudgetAware):
    """Bounded native-tool loop over the shared parent workspace."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        tools: ToolRegistry,
        config: ExploreSubAgentConfig | None = None,
        deadline_at: datetime | None = None,
        time_budget: InferenceTimeBudget | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
        mode: str = "fold",
        cancel_event: threading.Event | None = None,
    ) -> None:
        if mode not in EXPLORE_MODES:
            raise ValueError("Explore mode must be fold or meta")
        self.mode = mode
        self.system_prompt = (
            META_EXPLORE_SYSTEM_PROMPT if mode == "meta" else EXPLORE_SYSTEM_PROMPT
        )
        self.llm = llm
        self.tools = tools
        self.config = config or ExploreSubAgentConfig()
        self.deadline_at = deadline_at
        self.event_sink = event_sink
        self._cancel_event = cancel_event or threading.Event()
        bindings = (
            (TimeBudgetBinding("explore_llm", llm.session_time_budget),)
            if isinstance(llm, SessionTimeBudgetAware)
            else ()
        )
        self.time_budget = validate_time_budget_bindings(
            time_budget, bindings, owner="Explore"
        )
        self._validate_tools()

    @property
    def session_time_budget(self) -> InferenceTimeBudget | None:
        return self.time_budget

    def attach_cancel_event(self, event: threading.Event) -> None:
        self._cancel_event = event

    def cancel(self) -> None:
        self._cancel_event.set()

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(
        self,
        task: str,
        *,
        role: str,
        max_rounds: int | None = None,
        parent_call_id: str | None = None,
        thinking: str | None = None,
        inherit_context: bool = False,
        parent_messages: Sequence[ChatMessage] | None = None,
        description: str = "",
        task_id: str | None = None,
    ) -> dict[str, object]:
        if not task.strip():
            raise ValueError("Explore task cannot be empty")
        allowed = allowed_explore_tools(self.mode, role)
        self._validate_tools()
        rounds_limit = (
            max_rounds
            if isinstance(max_rounds, int) and max_rounds > 0
            else self.config.max_rounds
        )
        task_id = task_id or f"explore_{uuid.uuid4().hex[:12]}"
        child_cap = (
            time.monotonic() + self.config.deadline_seconds
            if self.config.deadline_seconds is not None
            else float("inf")
        )
        deadline = min(child_cap, self._deadline_monotonic())
        thinking = normalize_explore_thinking(thinking)
        llm = llm_with_thinking(self.llm, thinking)
        started = {
            "task_id": task_id,
            "role": role,
            "parent_call_id": parent_call_id,
            "status": "started",
            "mode": self.mode,
            "model": getattr(llm, "model", "") or getattr(self.llm, "model", ""),
            "thinking": thinking or "inherit",
            "inherit_context": bool(inherit_context),
        }
        if description:
            started["description"] = description
        self._emit("explore_task", started)
        messages = [ChatMessage("system", explore_system_prompt(self.mode, role))]
        if inherit_context and parent_messages:
            messages.extend(
                _copy_chat_message(message)
                for message in parent_messages
                if message.role != "system"
            )
        messages.append(ChatMessage("user", task.strip()))
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        rounds = 0
        tool_calls_made = 0
        summary = ""
        status = "completed"
        error = ""
        llm_calls = 0
        try:
            while rounds_limit is None or rounds < rounds_limit:
                if self._cancelled():
                    status = "cancelled"
                    error = "Explore cancelled"
                    break
                if self._deadline_reached(deadline):
                    status = "timeout"
                    error = "Explore deadline reached"
                    break
                rounds += 1
                provider_tools = self._provider_tools(allowed)
                output_tokens = self._output_tokens(llm, messages, provider_tools)
                messages, _context_edit = fit_tool_results_to_context(
                    llm,
                    messages,
                    tools=provider_tools,
                    max_tokens=output_tokens,
                )
                output_tokens = self._output_tokens(llm, messages, provider_tools)
                try:
                    response = llm.complete(
                        messages,
                        tools=provider_tools,
                        tool_choice="auto",
                        max_tokens=output_tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - child retry must not kill parent
                    llm_calls += 1
                    error = safe_error_summary(exc)
                    if self._cancelled():
                        status = "cancelled"
                        error = "Explore cancelled"
                        break
                    if _is_nonretryable_explore_error(exc) or self._deadline_reached(
                        deadline
                    ):
                        status = (
                            "timeout"
                            if isinstance(exc, TimeoutError)
                            or self._deadline_reached(deadline)
                            else "error"
                        )
                        break
                    messages.append(
                        ChatMessage(
                            "user",
                            json.dumps(
                                {
                                    "observation": "llm_error",
                                    "error": error,
                                    "retry_hint": "Continue with a different bounded action.",
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )
                    continue
                llm_calls += 1
                _add_usage(usage, response.usage)
                messages.append(
                    ChatMessage(
                        "assistant",
                        response.content,
                        response.tool_calls,
                        reasoning_content=response.reasoning_content,
                    )
                )
                self._emit(
                    "explore_llm",
                    {
                        "task_id": task_id,
                        "round": rounds,
                        "provider": getattr(llm, "provider", ""),
                        "model": response.model,
                        "usage": dict(response.usage),
                        "content": response.content,
                        "tool_names": [call.name for call in response.tool_calls],
                        "parent_call_id": parent_call_id,
                    },
                )
                if self._cancelled():
                    status = "cancelled"
                    error = "Explore cancelled"
                    break
                if not response.tool_calls:
                    text = (response.content or "").strip()
                    if text:
                        summary = text
                        break
                    messages.append(
                        ChatMessage(
                            "user",
                            json.dumps(
                                {
                                    "observation": "no_tool_call",
                                    "retry_hint": "Use an injected tool; thinking-only replies do not finish the task.",
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )
                    continue
                results = self._dispatch_calls(
                    response.tool_calls, deadline, allowed=allowed
                )
                for call, record, attempted in results:
                    tool_calls_made += int(attempted)
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
                    self._emit(
                        "explore_tool",
                        {
                            "task_id": task_id,
                            "round": rounds,
                            "tool": call.name,
                            "result": record,
                            "parent_call_id": parent_call_id,
                        },
                    )
            # Force a concise final summary when the loop ended without one
            # (rounds exhausted, or a round was cut off by output length).
            if (
                status == "completed"
                and not summary
                and not self._deadline_reached(deadline)
                and not self._cancelled()
            ):
                messages.append(
                    ChatMessage(
                        "user",
                        "请立即用简洁中文说明结论、关键证据、剩余风险或建议，不要再调用工具。",
                    )
                )
                provider_tools = self._provider_tools(allowed)
                output_tokens = self._output_tokens(llm, messages, provider_tools)
                messages, _context_edit = fit_tool_results_to_context(
                    llm,
                    messages,
                    tools=provider_tools,
                    max_tokens=output_tokens,
                )
                output_tokens = self._output_tokens(llm, messages, provider_tools)
                response = llm.complete(
                    messages,
                    tools=provider_tools,
                    tool_choice="none",
                    max_tokens=output_tokens,
                )
                llm_calls += 1
                _add_usage(usage, response.usage)
                summary = response.content.strip()
        except Exception as exc:  # noqa: BLE001 - a sub-agent failure must not kill the parent
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            error = safe_error_summary(exc)

        result: dict[str, object] = {
            "task_id": task_id,
            "status": status,
            "rounds": rounds,
            "tool_calls": tool_calls_made,
            "llm_calls": llm_calls,
            "provider": getattr(llm, "provider", ""),
            "model": getattr(llm, "model", "") or getattr(self.llm, "model", ""),
            "usage_totals": usage,
            "summary": summary,
            "mode": self.mode,
            "role": role,
            "thinking": thinking or "inherit",
            "inherit_context": bool(inherit_context),
        }
        if error:
            result["error"] = error
        self._emit(
            "explore",
            {
                **result,
                "parent_call_id": parent_call_id,
            },
        )
        return result

    def _output_tokens(
        self,
        llm: LLMProxy,
        messages: Sequence[ChatMessage],
        tools: Sequence[object],
    ) -> int:
        window = context_window_tokens(llm)
        requested = self.config.max_tokens or window or _NATIVE_WINDOW_FALLBACK
        _fits, prompt_tokens, resolved_window = context_request_fits(
            llm,
            messages,
            tools=tuple(tools),  # type: ignore[arg-type]
            max_tokens=requested,
        )
        clamped, _prompt_fits = clamp_requested_max_tokens(
            requested_max_tokens=requested,
            estimated_prompt_tokens=max(prompt_tokens, 1),
            context_window=resolved_window,
        )
        return clamped

    def _provider_tools(self, allowed: frozenset[str]) -> tuple[dict[str, object], ...]:
        visible = {spec.name for spec in self.tools.specs() if spec.name in allowed}
        return self.tools.provider_tools(visible)

    def _dispatch_calls(
        self,
        calls: tuple[ToolCall, ...],
        deadline: float,
        *,
        allowed: frozenset[str],
    ) -> list[tuple[ToolCall, dict[str, object], bool]]:

        def run_one(index: int) -> tuple[ToolCall, dict[str, object], bool]:
            call = calls[index]
            rejection = _reject_tool_call(
                self.tools.spec(call.name), call.arguments, allowed=allowed
            )
            if rejection:
                return call, {"ok": False, "error": rejection}, False
            if self._cancelled():
                return call, {"ok": False, "error": "Explore cancelled"}, False
            if self._deadline_reached(deadline):
                return call, {"ok": False, "error": "Explore deadline reached"}, False
            return call, self.tools.invoke(call.name, call.arguments).to_record(), True

        can_parallel = len(calls) > 1 and all(
            call.name in _PARALLEL_READ_TOOLS
            and not _reject_tool_call(
                self.tools.spec(call.name), call.arguments, allowed=allowed
            )
            for call in calls
        )
        if not can_parallel:
            return [run_one(index) for index in range(len(calls))]
        results: list[tuple[ToolCall, dict[str, object], bool] | None] = [None] * len(
            calls
        )
        with ThreadPoolExecutor(max_workers=min(len(calls), 4)) as executor:
            futures = {
                executor.submit(run_one, index): index for index in range(len(calls))
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [item for item in results if item is not None]

    def _deadline_monotonic(self) -> float:
        if self.deadline_at is None:
            return float("inf")
        remaining = (self.deadline_at - datetime.now(UTC)).total_seconds()
        return time.monotonic() + max(remaining, 0.0)

    def _deadline_reached(self, local_deadline: float) -> bool:
        return time.monotonic() >= local_deadline or (
            self.time_budget is not None and self.time_budget.remaining() <= 0
        )

    def _validate_tools(self) -> None:
        allowed = allowed_explore_tools(self.mode)
        for spec in self.tools.specs():
            if spec.name not in allowed or spec.name in _FORBIDDEN_TOOLS:
                raise ValueError(f"Explore tool is not allowed: {spec.name}")

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self.event_sink is not None:
            self.event_sink(event, dict(payload))


def _is_nonretryable_explore_error(exc: Exception) -> bool:
    if isinstance(exc, (SessionInterrupt, TimeoutError)):
        return True
    text = f"{exc} {safe_error_summary(exc)}"
    return _CALL_BUDGET_EXHAUSTED in text


def _copy_chat_message(message: ChatMessage) -> ChatMessage:
    return ChatMessage(
        message.role,
        message.content,
        message.tool_calls,
        tool_call_id=message.tool_call_id,
        reasoning_content=message.reasoning_content,
    )


def _add_usage(total: dict[str, int], usage: object) -> None:
    if not isinstance(usage, dict):
        return
    for key in total:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] += value


def _reject_tool_call(
    spec: ToolSpec | None,
    arguments: object,
    *,
    allowed: frozenset[str] = _ALLOWED_TOOLS,
) -> str:
    del arguments
    if spec is None:
        return "unknown Explore tool"
    if spec.name not in allowed or spec.name in _FORBIDDEN_TOOLS:
        return f"Explore tool is not allowed: {spec.name}"
    return ""
