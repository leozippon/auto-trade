"""One-level writable coding Sub Agent for a regular Fold (tool name ``explore``).

The Fold Agent delegates a concrete coding or inspection task to a cheaper-model
sub-agent that may read and write the shared Fold workspace. Write capability
comes from the registered tools, not from the prompt. Depth is one: the
sub-agent cannot spawn another ``explore``. It shares the parent SafeWorkspace,
sandbox runner, SessionBudgetLLM calls, and inference time budget. Failures
return a structured observation; they do not finish the Fold or roll back writes.
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
你是主 Agent 的一层可写 coding 子代理，只完成委托给你的具体任务。写能力来自已注入的工具，而不是本提示。你可以在共享 Fold 工作树上用已注册工具读、改、跑轻量检查，但不要替主 Agent 做最终提交、回测选择或提问。

# 方法
- 用 grep/glob/read_file 做定向检索；用 write_file/edit_file 修改文本产物；用完整 shell 做隔离分析、轻量验证和必要的文件操作。
- 不得调用 explore（禁止嵌套），也没有 daily_backtest、finish_fold、step_rollback 或 ask_user。
- 一轮内相互独立的只读检索可并行；写入、edit、shell、modification_check、validate_strategy 必须按调用顺序串行。
- 工具错误要如实保留，不要猜测成功。shell 不要用 `2>/dev/null` 隐藏错误。
- 不得安装依赖，不得读取 Test/Held-out。
- 权威 PRIOR 不在本 Fold 可写树中；即使改了工作区副本，也不能改变已注入的 PRIOR 或制品库中的权威版本。
- 历史分钟和竞价仅是日级推断时点之前的研究证据，不是执行时钟。

# 交付
任务完成后停止调用工具，直接用简洁中文返回四部分：结论、已做修改、证据、风险与限制、建议主 Agent 下一步。证据包含关键路径、字段、数字或覆盖范围，不罗列原始长输出。\
"""

_ALLOWED_TOOLS = frozenset(
    {
        "edit_file",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "validate_strategy",
        "write_file",
    }
)
_PARALLEL_READ_TOOLS = frozenset({"glob", "grep", "read_file"})
_FORBIDDEN_TOOLS = frozenset(
    {
        "ask_user",
        "daily_backtest",
        "explore",
        "finish_fold",
        "step_rollback",
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
    """Bounded native-tool coding loop over the shared Fold workspace."""

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
                        "请立即按“结论 / 已做修改 / 证据 / 风险与限制 / 建议主 Agent 下一步”给出简洁中文摘要，不要再调用工具。",
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
            if spec.name not in _ALLOWED_TOOLS or spec.name in _FORBIDDEN_TOOLS:
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


def _reject_tool_call(spec: ToolSpec | None, arguments: object) -> str:
    del arguments
    if spec is None:
        return "unknown Explore tool"
    if spec.name not in _ALLOWED_TOOLS or spec.name in _FORBIDDEN_TOOLS:
        return f"Explore tool is not allowed: {spec.name}"
    return ""
