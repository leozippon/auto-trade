"""Read-only data-exploration Sub Agent (Claude-Code "Explore" pattern).

The Fold/meta-learning Agent delegates a concrete read-only investigation to a
cheaper-model sub-agent that may call ``shell``/``grep``/``glob`` over the
visible sandbox. It returns a compact evidence digest, so the expensive main
context stays small and routine probing runs on the cheaper model. It never
writes formal artifacts.
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

EXPLORE_SYSTEM_PROMPT = """\
# 角色
你是主 Agent 的只读调查员，只回答委托给你的具体问题。你可以用被注入的只读工具读取与统计可见的 PIT 数据、策略产物、Validation 结果和 Step 记录，但不要修改任何文件，不要写正式产物，不要替主 Agent 作最终决策。

# 方法
- 优先用 grep/glob 做定向搜索，用 shell 做目录、metadata、head/count/limit、轻量 Python/DuckDB 只读抽样；不要全量读取大表。
- shell 是轻量合同 guard，不是只读 Bash 解析器；不要写文件、不要重定向到文件、不要隐藏错误。只读约定由本提示约束，硬隔离和产物校验兜底。
- 一轮可并行发起多个相互独立的只读检索；工具错误要如实保留，不要猜测成功。
- shell 命令不要用 `2>/dev/null` 隐藏错误。
- 不得安装依赖，不得读取 Test/Held-out。
- 历史分钟和竞价仅是日级推断时点之前的研究证据，不是执行时钟。

# 交付
信息足够后停止调用工具，直接用简洁中文返回四部分：结论、证据、风险与限制、建议主 Agent 下一步。证据包含关键路径、字段、数字或覆盖范围，不罗列原始长输出。\
"""

_READ_ONLY_TOOLS = frozenset(
    {
        "glob",
        "grep",
        "read_file",
        "shell",
        "validate_strategy",
    }
)
_PARALLEL_READ_TOOLS = frozenset({"glob", "grep", "read_file"})
_READ_ONLY_SHELL_COMMANDS = frozenset(
    {
        "cat",
        "cut",
        "du",
        "grep",
        "head",
        "ls",
        "pwd",
        "readlink",
        "realpath",
        "rg",
        "stat",
        "tail",
        "tr",
        "uniq",
        "wc",
    }
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
    """Bounded native-tool exploration loop over read-only tools."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        tools: ToolRegistry,
        config: ExploreSubAgentConfig | None = None,
        deadline_at: datetime | None = None,
        time_budget: InferenceTimeBudget | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
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
        max_rounds: int | None = None,
        parent_call_id: str | None = None,
    ) -> dict[str, object]:
        if not task.strip():
            raise ValueError("Explore task cannot be empty")
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
                "task": task.strip()[:8_000],
                "parent_call_id": parent_call_id,
                "status": "started",
            },
        )
        messages = [
            ChatMessage("system", EXPLORE_SYSTEM_PROMPT),
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
                provider_tools = self.tools.provider_tools()
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
                results = self._dispatch_calls(response.tool_calls, deadline)
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
                        "请立即按“结论 / 证据 / 风险与限制 / 建议主 Agent 下一步”四部分给出简洁中文摘要，不要再调用工具。",
                    )
                )
                provider_tools = self.tools.provider_tools()
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
        except Exception as exc:  # noqa: BLE001 - a sub-agent failure must not kill the Fold
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
        }
        if error:
            result["error"] = error
        self._emit(
            "explore",
            {
                **result,
                "task": task[:500],
                "parent_call_id": parent_call_id,
            },
        )
        return result

    def _dispatch_calls(
        self, calls: tuple[ToolCall, ...], deadline: float
    ) -> list[tuple[ToolCall, dict[str, object], bool]]:
        def run_one(index: int) -> tuple[ToolCall, dict[str, object], bool]:
            call = calls[index]
            rejection = _reject_tool_call(self.tools.spec(call.name), call.arguments)
            if rejection:
                return call, {"ok": False, "error": rejection}, False
            if self._deadline_reached(deadline):
                return call, {"ok": False, "error": "Explore deadline reached"}, False
            return call, self.tools.invoke(call.name, call.arguments).to_record(), True

        can_parallel = len(calls) > 1 and all(
            call.name in _PARALLEL_READ_TOOLS
            and not _reject_tool_call(self.tools.spec(call.name), call.arguments)
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
        for spec in self.tools.specs():
            if spec.mutating:
                raise ValueError(f"Explore tool must be non-mutating: {spec.name}")
            if spec.name not in _READ_ONLY_TOOLS:
                raise ValueError(
                    f"Explore tool is not on the read-only whitelist: {spec.name}"
                )

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


def _reject_tool_call(spec: ToolSpec | None, arguments: object) -> str:
    if spec is None:
        return "unknown Explore tool"
    if spec.mutating or spec.name not in _READ_ONLY_TOOLS:
        return f"Explore tool is not read-only: {spec.name}"
    if spec.name != "shell":
        return ""
    if not isinstance(arguments, dict):
        return "Explore shell arguments must be an object"
    argv = arguments.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) for item in argv)
    ):
        return "Explore shell argv must be a non-empty string array"
    command = argv[0]
    if command not in _READ_ONLY_SHELL_COMMANDS:
        return f"Explore shell command is not on the read-only whitelist: {command}"
    if command == "rg" and any(
        argument in {"--pre", "--hostname-bin"}
        or argument.startswith(("--pre=", "--hostname-bin="))
        for argument in argv[1:]
    ):
        return "Explore shell command may not execute helper programs"
    return ""
