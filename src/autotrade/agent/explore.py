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
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

from autotrade.environment.llm import ChatMessage, LLMProxy, ToolCall
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
    TimeBudgetBinding,
    validate_time_budget_bindings,
)
from autotrade.environment.tools.base import ToolRegistry, ToolSpec

from .compact import fit_tool_results_to_context, safe_error_summary

EXPLORE_MODES = frozenset({"fold", "meta"})
EXPLORE_ROLES = ("auditor", "developer", "general-purpose", "Explore")
OPTIONAL_EXPLORE_ROLES = frozenset({"general-purpose", "Explore"})
FOLD_REQUIRED_EXPLORE_ROLES = ("auditor", "developer")
META_REQUIRED_EXPLORE_ROLES = ("auditor",)

_FOLD_AUDIT_TOOLS = frozenset({"glob", "grep", "read_file", "shell", "todo"})
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
    }
)
_META_ROLE_TOOLS = frozenset({"glob", "grep", "read_file", "todo"})
_FOLD_ROLE_TOOLS = {
    "auditor": _FOLD_AUDIT_TOOLS,
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
        "write_taste",
    }
)
_ALLOWED_TOOLS = _FOLD_WRITE_TOOLS

_FOLD_WRITE_PROMPT = """\
# 角色
你是 sub-agent，角色 {role}：{mission}。禁止再嵌套子代理。写能力来自已注入的工具，而不是本提示。你可以在共享 Fold 工作树上用已注册工具读、改、跑轻量检查，但不要替主 Agent 做最终提交、回测选择或提问。

# 方法
- 用 grep/glob/read_file 做定向检索；用 write_file/edit_file 修改文本产物；用完整 shell 做隔离分析、轻量验证和必要的文件操作；用 `todo` 维护与主 Agent 共享的本会话研究计划。
- 静态类型疑问可用现有前台 shell 运行 `pyright --project /opt/autotrade/pyrightconfig.json /mnt/agent/workspace /mnt/agent/output`；它是 debug 顾问，不替代 validate_strategy 或 modification_check。不得后台运行。
- 不得调用 explore（禁止嵌套），也没有 daily_backtest、finish_fold、step_rollback 或 ask_user。
- 一轮内相互独立的只读检索可并行；写入、edit、shell、todo、modification_check、validate_strategy 必须按调用顺序串行。
- 工具错误要如实保留，不要猜测成功。shell 不要用 `2>/dev/null` 隐藏错误。
- 不得安装依赖，不得读取 Test/Held-out。
- 权威 PRIOR 不在本 Fold 可写树中；即使改了工作区副本，也不能改变已注入的 PRIOR 或制品库中的权威版本。
- 历史分钟和竞价仅是日级推断时点之前的研究证据，不是执行时钟。

# 交付
任务完成后停止调用工具，直接用简洁中文返回四部分：结论、已做修改、证据、风险与限制、建议主 Agent 下一步。证据包含关键路径、字段、数字或覆盖范围，不罗列原始长输出。\
"""

_FOLD_AUDIT_PROMPT = """\
# 角色
你是 sub-agent，角色 {role}：{mission}。只完成委托给你的具体检查任务，禁止再嵌套子代理。你与主 Fold 共享工作树和预算，但不能替主 Agent 写策略、提交、回测或提问。

# 方法
- 只用 read_file/grep/glob 做有界只读定位；用 `todo` 维护本会话研究计划；可用前台 shell 做只读检查（查看 schema、抽样、计数、类型询问）。
- shell 只能运行只读命令，不得创建、修改、删除或覆盖任何文件，也不得改策略产物。
- 没有 write_file、edit_file。不得调用 explore（禁止嵌套），也没有 daily_backtest、finish_fold、step_rollback 或 ask_user。
- 一轮内相互独立的只读检索可并行；shell 与 todo 必须按调用顺序串行。
- 工具错误要如实保留，不要猜测成功。shell 不要用 `2>/dev/null` 隐藏错误。
- 不得安装依赖，不得读取 Test/Held-out。
- 权威 PRIOR 不在本 Fold 可写树中。历史分钟和竞价仅是日级推断时点之前的研究证据，不是执行时钟。

# 交付
任务完成后停止调用工具，直接用简洁中文返回：结论、证据、风险与限制、建议主 Agent 下一步。证据包含关键路径、字段、数字或覆盖范围，不罗列原始长输出。\
"""

META_EXPLORE_SYSTEM_PROMPT = """\
# 角色
你是 sub-agent：Meta 主协调者的一层只读审计/分析子代理，只完成委托给你的具体独立任务，禁止再嵌套子代理。你与父 Meta 共享同一 SafeWorkspace、总预算、deadline 和 Trace。结果只返回父 Meta；你不能发布 PRIOR、Taste 或结束本会话。

# 方法
- 只用 read_file/grep/glob 做有界只读定位；用 `todo` 维护本会话研究计划。todo 可以写会话计划，但不得改 PRIOR、Taste 或策略产物。
- 不得调用 explore（禁止嵌套）。没有 write_file、edit_file、shell、daily_backtest、write_taste、finish_meta、modification_check 或提问。
- 一轮内相互独立的只读检索可并行；todo 必须按调用顺序串行。
- 工具错误要如实保留，不要猜测成功。
- 不得读取 Test/Held-out 原始记录，不得改宿主代码。

# 交付
任务完成后停止调用工具，直接用简洁中文返回：结论、证据、风险与限制、建议父 Meta 下一步。证据包含关键字段、计数或覆盖范围，不罗列原始长输出，不写入逐 Fold Test 数字或 Held-out。\
"""

_FOLD_EXPLORE_PROMPT = """\
# 角色
你是 sub-agent，角色 {role}：{mission}。只完成委托给你的具体探索任务，禁止再嵌套子代理。你与主 Fold 共享工作树和预算，但不能替主 Agent 写策略、提交、回测或提问。

# 方法
- 只用 read_file/grep/glob 做有界只读定位；用 `todo` 维护本会话研究计划。
- 没有 write_file、edit_file、shell。不得调用 explore（禁止嵌套），也没有 daily_backtest、finish_fold、step_rollback 或 ask_user。
- 一轮内相互独立的只读检索可并行；todo 必须按调用顺序串行。
- 工具错误要如实保留，不要猜测成功。
- 不得安装依赖，不得读取 Test/Held-out。
- 权威 PRIOR 不在本 Fold 可写树中。历史分钟和竞价仅是日级推断时点之前的研究证据，不是执行时钟。

# 交付
任务完成后停止调用工具，直接用简洁中文返回：结论、证据、风险与限制、建议主 Agent 下一步。证据包含关键路径、字段、数字或覆盖范围，不罗列原始长输出。\
"""

_FOLD_ROLE_MISSIONS = {
    "auditor": (
        "在开发前检查 PIT 可见数据、单位/可用性、父策略、历史制品与已有结果；"
        "必要时可多次"
    ),
    "developer": "主 Fold Agent 的一层可写 coding 子代理，只完成委托给你的真实代码开发任务",
    "general-purpose": "可选通用可写同事；只处理跨域有界任务，不能替代 auditor 或 developer",
    "Explore": "只读探索未知位置、资料或接口；不能替代 auditor 或 developer",
}
_META_ROLE_MISSIONS = {
    "auditor": (
        "非空窗口检查常规 Fold Trace、process summary、冻结策略、"
        "Train/Validation 及允许的紧凑 Test 反馈；空窗口检查 Taste、PRIOR 与输入边界；"
        "必要时可多次"
    ),
    "developer": "只读，仅能提出候选改进，不能写 PRIOR、Taste 或策略",
    "general-purpose": "可选只读跨域有界任务，不能替代 auditor",
    "Explore": "只读探索未知位置、资料或接口，不能替代 auditor",
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


def session_explore_roles(mode: str) -> tuple[str, ...]:
    if _explore_mode(mode) == "meta":
        return META_REQUIRED_EXPLORE_ROLES
    return FOLD_REQUIRED_EXPLORE_ROLES


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
        if role == "auditor":
            return _FOLD_AUDIT_PROMPT.format(role=role, mission=mission)
        return _FOLD_EXPLORE_PROMPT.format(role=role, mission=mission)
    mission = _META_ROLE_MISSIONS.get(role)
    if mission is None:
        raise ValueError(f"Explore role is not allowed: {role}")
    return (
        f"# 本任务角色\n你的角色是 `{role}`：{mission}。\n\n"
        + META_EXPLORE_SYSTEM_PROMPT
    )


@dataclass(frozen=True)
class ExploreSubAgentConfig:
    per_call_timeout_seconds: float = 120.0
    # Room for a tool-call round (long DuckDB/SQL arguments) plus a concise
    # digest; too small a cap makes a round stop on finish_reason=length.
    max_tokens: int = 6_000
    max_rounds: int = 6
    deadline_seconds: float = 600.0

    def __post_init__(self) -> None:
        if (
            self.per_call_timeout_seconds <= 0
            or self.max_tokens <= 0
            or self.max_rounds <= 0
            or self.deadline_seconds <= 0
        ):
            raise ValueError("Explore budgets must be positive")


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

    def run(
        self,
        task: str,
        *,
        role: str,
        max_rounds: int | None = None,
        parent_call_id: str | None = None,
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
        task_id = f"explore_{uuid.uuid4().hex[:12]}"
        deadline = min(
            time.monotonic() + self.config.deadline_seconds,
            self._deadline_monotonic(),
        )
        self._emit(
            "explore_task",
            {
                "task_id": task_id,
                "role": role,
                "parent_call_id": parent_call_id,
                "status": "started",
                "mode": self.mode,
            },
        )
        messages = [
            ChatMessage("system", explore_system_prompt(self.mode, role)),
            ChatMessage("user", task.strip()),
        ]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        rounds = 0
        tool_calls_made = 0
        digest = ""
        status = "completed"
        error = ""
        llm_calls = 0
        try:
            while rounds < rounds_limit:
                if self._deadline_reached(deadline):
                    status = "timeout"
                    error = "Explore deadline reached"
                    break
                rounds += 1
                provider_tools = self._provider_tools(allowed)
                messages, _context_edit = fit_tool_results_to_context(
                    self.llm,
                    messages,
                    tools=provider_tools,
                    max_tokens=self.config.max_tokens,
                )
                response = self.llm.complete(
                    messages,
                    tools=provider_tools,
                    tool_choice="auto",
                    max_tokens=self.config.max_tokens,
                )
                llm_calls += 1
                _add_usage(usage, response.usage)
                messages.append(
                    ChatMessage(
                        "assistant",
                        response.content or None,
                        response.tool_calls,
                        reasoning_content=response.reasoning_content,
                    )
                )
                self._emit(
                    "explore_llm",
                    {
                        "task_id": task_id,
                        "round": rounds,
                        "provider": getattr(self.llm, "provider", ""),
                        "model": response.model,
                        "usage": dict(response.usage),
                        "content": response.content,
                        "tool_names": [call.name for call in response.tool_calls],
                        "parent_call_id": parent_call_id,
                    },
                )
                if not response.tool_calls:
                    digest = response.content.strip()
                    break
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
            # Force a concise final digest when the loop ended without one
            # (rounds exhausted, or a round was cut off by output length).
            if (
                status == "completed"
                and not digest
                and not self._deadline_reached(deadline)
            ):
                messages.append(
                    ChatMessage(
                        "user",
                        "请立即按“结论 / 已做修改 / 证据 / 风险与限制 / 建议主 Agent 下一步”给出简洁中文摘要，不要再调用工具。",
                    )
                )
                provider_tools = self._provider_tools(allowed)
                messages, _context_edit = fit_tool_results_to_context(
                    self.llm,
                    messages,
                    tools=provider_tools,
                    max_tokens=self.config.max_tokens,
                )
                response = self.llm.complete(
                    messages,
                    tools=provider_tools,
                    tool_choice="none",
                    max_tokens=self.config.max_tokens,
                )
                llm_calls += 1
                _add_usage(usage, response.usage)
                digest = response.content.strip()
        except Exception as exc:  # noqa: BLE001 - a sub-agent failure must not kill the parent
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            error = safe_error_summary(exc)

        result: dict[str, object] = {
            "task_id": task_id,
            "status": status,
            "rounds": rounds,
            "tool_calls": tool_calls_made,
            "llm_calls": llm_calls,
            "provider": getattr(self.llm, "provider", ""),
            "model": getattr(self.llm, "model", ""),
            "usage_totals": usage,
            "digest": digest,
            "mode": self.mode,
            "role": role,
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
