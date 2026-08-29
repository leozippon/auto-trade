"""One-level Sub Agent for a regular Fold or Meta session (tool name ``agent``).

Parents call ``agent(agent=<role>, task=...)`` like any other registered tool:
the registry validates the arguments, :class:`AgentTool` hands them to the
runner, and the runner starts the child in the background and returns at once.
Roles are the unified set ``auditor``, ``developer``, ``general-purpose``,
``Explore``; ``Explore`` is the optional read-only discovery role. Depth is
one. The child shares the parent SafeWorkspace, SessionBudgetLLM calls,
inference time budget, output ceiling, context window with its compaction
gateway and thresholds, and Trace. A finished child keeps its transcript for
the session so ``resume=<task_id>`` can hand it a follow-up task. Failures
return a structured observation; they do not finish the parent session.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from autotrade.environment.llm import (
    AGENT_MAX_OUTPUT_TOKENS,
    ChatMessage,
    LLMProxy,
    ToolCall,
    clamp_requested_max_tokens,
    context_request_fits,
    is_context_overflow_error,
)
from autotrade.environment.runtime import sanitize_for_log
from autotrade.environment.time_budget import (
    InferenceTimeBudget,
    SessionTimeBudgetAware,
    TimeBudgetBinding,
    validate_time_budget_bindings,
)
from autotrade.environment.tools.base import (
    SessionInterrupt,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    is_sequential_tool,
)

from .compact import (
    ContextCompactor,
    drop_trailing_unanswered_tool_calls,
    fit_tool_results_to_context,
    safe_error_summary,
)

SUBAGENT_MODES = frozenset({"fold", "meta"})
SUBAGENT_THINKING_LEVELS = ("off", "low", "medium", "xhigh")
# Read-compatible aliases from older traces and prompts; on the wire they were
# never distinct from xhigh.
_LEGACY_THINKING_ALIASES = {"minimal": "low", "high": "xhigh", "max": "xhigh"}
DEFAULT_SUBAGENT_MAX_CONCURRENT = 4
# Global default child turn budget (the last tier of the launch precedence:
# call argument > role default > this). A child has the parent's context
# window and compaction, so the round cap bounds work, not memory: under the
# earlier cap of 24, one child in eleven hit the wrap-up notice while still
# issuing tool calls every round at 25k-144k prompt tokens, i.e. bounded work
# was cut short, not wandering. The last ``SUBAGENT_GRACE_ROUNDS`` turns start
# with a wrap-up notice (Pi's grace-turn pattern); an explicit ``max_turns``
# is honoured as given.
DEFAULT_SUBAGENT_MAX_ROUNDS = 48
SUBAGENT_GRACE_ROUNDS = 2
# Global default thinking level; a child never inherits the parent's level.
# The parent lowers it per launch for bounded mechanical work, where a round
# that spends the whole completion cap on reasoning costs more than it adds.
DEFAULT_SUBAGENT_THINKING = "xhigh"
SUBAGENT_DESCRIPTION_MAX_CHARS = 200
# A parent's mid-run instruction to a child (``action="message"``): bounded
# like a brief, queued on the job, and delivered as one labelled user message
# before the child's next model round.
SUBAGENT_STEER_MAX_CHARS = 2_000
STEER_MESSAGE_LABEL = "[父代理指令]"
# Characters of a child's final report delivered inline in the parent's
# ``subagent_completed`` observation (Pi's preview-then-fetch shape). A longer
# report is clipped there and spilled in full through the search tools'
# result store, so the parent reads it back in pages instead of receiving it
# whole.
SUBAGENT_REPORT_MAX_CHARS = 6_000
# Bounded argument echo for the child's Trace: a write/edit call can carry a
# whole file, and the trace writer replaces any event above its per-event cap
# with a stub that loses ``task_id`` — the console would then lose the call.
TRACE_ARGUMENT_MAX_CHARS = 2_000
SUBAGENT_TASK_ID_PREFIX = "agent_"
_CALL_BUDGET_EXHAUSTED = "call budget exhausted"
# Appended to a child's summary when its reply hit the output ceiling, so the
# parent never receives a silently cut half-sentence.
OUTPUT_TRUNCATED_MARKER = "[输出在 {limit} token 上限被截断]"
# A reply that spent its whole completion budget on reasoning (no content, no
# tool call) gets a forced concise continuation — the same observation the
# parent conversation receives — at most this many times per child before the
# launch is reported as failed instead of terminated as ``completed``.
SUBAGENT_MAX_TRUNCATION_CONTINUATIONS = 2
OUTPUT_TRUNCATED_CONTINUATION = (
    "上一轮输出在 {limit} token 上限被截断且没有工具调用。"
    "请把已有结论压缩成几句话，然后直接调用下一步工具；不要重新展开完整推理。"
)

_FOLD_READ_TOOLS = frozenset({"glob", "grep", "read_file"})
_FOLD_WRITE_TOOLS = frozenset(
    {
        "edit_file",
        "glob",
        "grep",
        "modification_check",
        "read_file",
        "shell",
        "write_file",
        "write_skill",
        "delete_skill",
    }
)
# Every role is read-only in a Meta session.
_META_ROLE_TOOLS = frozenset({"glob", "grep", "read_file"})


@dataclass(frozen=True)
class SubAgentRole:
    """One row of the role table: a role's identity, Fold tool set, missions
    and launch defaults, the way a Pi agent-definition file carries them.

    ``thinking`` and ``max_turns`` are the role tier of the launch precedence
    (call argument > role default > global default); ``None`` defers to the
    global default, which every shipped role does today.
    """

    name: str
    description: str
    fold_tools: frozenset[str]
    fold_mission: str
    meta_mission: str
    thinking: str | None = None
    max_turns: int | None = None

    def __post_init__(self) -> None:
        if self.thinking is not None and self.thinking not in SUBAGENT_THINKING_LEVELS:
            raise ValueError(f"role {self.name}: unknown thinking level {self.thinking}")
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError(f"role {self.name}: max_turns must be at least 1")

    @property
    def shell(self) -> bool:
        return "shell" in self.fold_tools

    @property
    def default_thinking(self) -> str:
        return self.thinking or DEFAULT_SUBAGENT_THINKING

    def default_max_turns(self, global_default: int | None) -> int | None:
        return self.max_turns if self.max_turns is not None else global_default


SUBAGENT_ROLE_TABLE: tuple[SubAgentRole, ...] = (
    SubAgentRole(
        "auditor",
        "只读审计：核对数据、单位、代码与证据边界",
        _FOLD_READ_TOOLS,
        fold_mission="审查委托问题及其证据边界",
        meta_mission="独立审查委托问题",
    ),
    SubAgentRole(
        "developer",
        "读写实现：跑 Python、写策略、模型与 skills",
        _FOLD_WRITE_TOOLS,
        fold_mission="实现并检查委托的代码或知识任务",
        meta_mission="只读分析候选策略改进",
    ),
    SubAgentRole(
        "general-purpose",
        "读写通用：一个有界的跨域实现或计算任务",
        _FOLD_WRITE_TOOLS,
        fold_mission="完成一个有界的跨域实现任务",
        meta_mission="只读处理一个有界跨域问题",
    ),
    SubAgentRole(
        "Explore",
        "只读探索：定位文件、接口与材料",
        _FOLD_READ_TOOLS,
        fold_mission="定位未知位置、接口或材料",
        meta_mission="只读定位未知位置、接口或材料",
    ),
)
_ROLES_BY_NAME = {role.name: role for role in SUBAGENT_ROLE_TABLE}
SUBAGENT_ROLES = tuple(role.name for role in SUBAGENT_ROLE_TABLE)


def subagent_role(name: object) -> SubAgentRole:
    role = _ROLES_BY_NAME.get(name)  # type: ignore[arg-type]
    if role is None:
        raise ValueError(f"Sub-agent role is not allowed: {name}")
    return role


def _role_schema_text() -> str:
    """Per-role capability and launch defaults for the ``agent`` field."""

    lines = []
    for role in SUBAGENT_ROLE_TABLE:
        tools = "有 Sandbox shell、可写" if role.shell else "只读 glob/grep/read_file，不能执行"
        lines.append(
            f"{role.name}：{role.description}（{tools}；默认 thinking {role.default_thinking}、"
            f"max_turns {role.default_max_turns(DEFAULT_SUBAGENT_MAX_ROUNDS)}）"
        )
    return "；".join(lines) + "。Meta 会话中全部角色只读。"

_FOLD_WRITE_PROMPT = """\
# 身份
你是 Fold 的一级 `{role}` sub-agent：{mission}。你可用已注入工具修改共享策略、模型或 skills，但父 Agent 独占正式回测、候选选择、验收和结束。

# 边界
- 先读 `inputs/skills_index.json`，再从已挂载数据、单位引用、制品和参考材料中自主发现任务所需证据；skill 脚本不自动执行。把有复用价值的知识写入 skill，而不是堆入策略或汇报。
- 只完成父任务；不得再委托子代理、读取 Test/Held-out、改变权威 PRIOR、安装依赖、替父 Agent 提问或伪造结果。分钟和竞价不是策略时钟。
- 工具 schema 决定实际能力。同一轮的只读调用并发执行；写、检查与 shell 按因果顺序分轮调用。shell 只做有界前台工作，不启动后台任务、sleep/等待包装、轮询状态或隐藏错误；shell 写入工作区的文件会保留。全市场逐股或全历史的计算先在抽样上验证脚本，再分块运行并把中间结果落盘，每块都要在 shell 超时内完成。
- 运行中收到以 `[父代理指令]` 开头的消息时，它是父 Agent 的补充要求，优先于原 task。

# 返回
用简洁中文说明结论、实际修改、关键证据和剩余风险，然后停止。\
"""

_FOLD_READ_PROMPT = """\
# 身份
你是 Fold 的一级只读 `{role}` sub-agent：{mission}。只调查父任务并返回证据；不能写策略、models、skills 或 PRIOR，也不能回测、验收或结束会话。

# 边界
- 先读 `inputs/skills_index.json`，再从已挂载数据、单位引用、制品和参考材料中自主发现任务所需证据；skill 脚本不自动执行。
- 工具 schema 决定实际能力；同一轮的多个只读调用并发执行。不得再委托子代理、读取 Test/Held-out、安装依赖或伪造结果；分钟和竞价不是策略时钟。
- 运行中收到以 `[父代理指令]` 开头的消息时，它是父 Agent 的补充要求，优先于原 task。

# 返回
用简洁中文说明结论、关键证据、限制和建议，然后停止。\
"""

META_SUBAGENT_SYSTEM_PROMPT = """\
# 身份
你是 Meta 的一级只读 sub-agent。只完成父任务并提出有证据的候选；不能写策略、models、skills 或 PRIOR，也不能验收或结束会话。

# 边界
- 先读 `inputs/skills_index.json`，再从 `inputs/meta_context.json` 及其挂载引用中自主发现任务所需证据；skill 脚本不自动执行。
- 工具 schema 决定实际能力；同一轮的多个只读调用并发执行。不得再委托子代理、读取 Test/Held-out 原始记录、改变 PIT/隐藏阶段边界、访问外部资料、修改宿主代码或伪造结果。
- 运行中收到以 `[父代理指令]` 开头的消息时，它是父 Agent 的补充要求，优先于原 task。

# 返回
用简洁中文说明结论、关键证据、限制和建议；不要复制 raw traces 或写逐 Fold Test 数字。\
"""

SUBAGENT_SYSTEM_PROMPT = _FOLD_WRITE_PROMPT.format(
    role="developer",
    mission=subagent_role("developer").fold_mission,
)

# The single place the sub-agent mechanism is explained to the model; the
# system prompt only points here. Modeled on Pi's Agent tool description.
AGENT_TOOL_DESCRIPTION = (
    "启动一个后台子代理并立即返回；它完成后结果以 subagent_completed 消息送回，不要轮询。"
    "用于读库、探索、计算、实现或审计等能独立完成的任务：把大量阅读、计算和实现留在子代理里以保护主上下文；"
    "目标已知的单个文件直接用 read_file/grep/glob；不要重复子代理正在做的搜索。"
    "同一轮可发起多个（默认同时运行 4 个，超出排队），并行的子代理范围须互斥。"
    "子代理拥有与你相同的上下文窗口、压缩阈值和输出上限（达到阈值时自动压缩，不会因上下文写满而失败），"
    f"可以承担较大的有界块；省略 max_turns 时最多 {DEFAULT_SUBAGENT_MAX_ROUNDS} 轮：倒数第 {SUBAGENT_GRACE_ROUNDS} 轮起收到收尾提示，"
    "到上限后强制一次简洁总结。几个并行的有界子代理仍好过一个很长的串行子代理；确需更多轮次时显式给 max_turns。"
    "角色能力：developer/general-purpose 有 Sandbox shell（可跑 Python 读 PIT parquet、算 IC 表、做冒烟测试）并可写策略、模型与 skills；"
    "auditor/Explore 只能用 glob/grep/read_file 读文本与代码，不能执行任何命令——任何需要计算的任务用 general-purpose 或 developer；"
    "Meta 会话中全部角色只读。子代理只看到自己的角色提示和你的 task（inherit_context=true 时另带你的对话），"
    f"所以 task 要写全路径、约束和期望的返回格式。thinking 默认 {DEFAULT_SUBAGENT_THINKING}，适合需要判断的审计、设计与实现；"
    "有界的机械工作（按给定路径读取并摘录、跑一段已写好的脚本、逐文件核对）显式降到 low/medium："
    f"每轮输出上限 {AGENT_MAX_OUTPUT_TOKENS} token，把它全部耗在思考里而发不出工具调用的一轮只得到最多 "
    f"{SUBAGENT_MAX_TRUNCATION_CONTINUATIONS} 次强制简洁续写，之后该次委托记为 error。"
    "thinking 与 max_turns 由你按次决定，生效顺序：本次调用参数 > 角色默认（见 agent 字段） > 全局默认"
    f"（{DEFAULT_SUBAGENT_THINKING}、{DEFAULT_SUBAGENT_MAX_ROUNDS} 轮）；生效值记入该子代理的 subagent_task 事件。"
    f"子代理的汇报最多内联 {SUBAGENT_REPORT_MAX_CHARS} 字符：更长的汇报只内联开头（summary_truncated=true），"
    "全文落盘并以 result_root/result_ref 返回，用 read_file 从 resume_line 起分页读回（offset 是行号，不是字符数）；"
    "要求子代理把长材料写进工作区文件而不是塞进汇报。"
    "子代理不能嵌套、正式回测、结束会话、改 PRIOR 或自行验收；它的汇报描述意图而非结果，其写入须由你验收。"
    "resume=<task_id> 让一个已完成的子代理在自己的对话上继续新的 task（保留它读过的上下文，角色须相同）；"
    "仍在运行或未知的 task_id 会被拒绝。只在后续任务确实需要它已有的上下文时 resume；独立的后续工作另起并行的全新子代理，不要串成 resume 链。"
    "action=message（带 task_id 与 text）给一个仍在运行的子代理发中途指令：立即返回 status=queued，"
    "指令在它下一轮模型调用前作为一条 `[父代理指令]` 消息送达（尚未开始的排队子代理在第一轮前收到），"
    "它的 subagent_completed 里 steers/steers_undelivered 记送达与未送达条数。"
    "只在需要改变范围、追加刚发现的约束或让它提前收尾汇报时使用；不为催促而发，"
    "后续任务用 resume 或新子代理，已完成的子代理不能 message。"
)

AGENT_TOOL_SPEC = ToolSpec(
    "agent",
    AGENT_TOOL_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["launch", "message"],
                "description": (
                    "省略或 launch：启动子代理（须给 agent 与 task）；"
                    "message：给仍在运行的子代理发中途指令（须给 task_id 与 text）。"
                ),
            },
            "agent": {
                "type": "string",
                "enum": list(SUBAGENT_ROLES),
                "description": "launch：角色、能力与默认档位——" + _role_schema_text(),
            },
            "task": {
                "type": "string",
                "minLength": 1,
                "description": "launch：完整的委托任务：目标、范围、已知事实与期望的返回内容。",
            },
            "task_id": {
                "type": "string",
                "minLength": 1,
                "description": "message：仍在运行（或排队）的子代理的 task_id。",
            },
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": SUBAGENT_STEER_MAX_CHARS,
                "description": (
                    f"message：中途指令正文（≤{SUBAGENT_STEER_MAX_CHARS} 字符），"
                    "在子代理下一轮模型调用前作为 `[父代理指令]` 消息送达。"
                ),
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": SUBAGENT_DESCRIPTION_MAX_CHARS,
                "description": "控制台显示的一句话标签。",
            },
            "max_turns": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    f"子代理最多的模型轮次，≥1；省略时按角色默认（当前各角色均为 {DEFAULT_SUBAGENT_MAX_ROUNDS} 轮），"
                    f"倒数第 {SUBAGENT_GRACE_ROUNDS} 轮起收到收尾提示，上限后强制简洁总结；显式值按原值生效。"
                    "子代理的上下文与你同样大并自动压缩，轮次只约束工作量。没有 schema 上限，但每轮占用会话共享调用配额中的一次子代理调用"
                    "（全部子代理累计不超过会话 max_llm_calls 的一半），配额耗尽的子代理以 error 结束，所以不要设得远超任务需要。"
                ),
            },
            "thinking": {
                "type": "string",
                "enum": list(SUBAGENT_THINKING_LEVELS),
                "description": (
                    f"子代理思考强度 off/low/medium/xhigh；省略时按角色默认（当前各角色均为 {DEFAULT_SUBAGENT_THINKING}），不继承父会话。"
                    "需要判断的任务保留默认；有界的机械工作显式给 low/medium，"
                    f"因为把 {AGENT_MAX_OUTPUT_TOKENS} token 的输出预算全部耗在思考里而发不出工具调用的一轮"
                    f"只得到最多 {SUBAGENT_MAX_TRUNCATION_CONTINUATIONS} 次强制简洁续写，之后记为 error"
                    "（旧值 high/max 等同 xhigh）；off 关闭扩展思考。"
                ),
            },
            "inherit_context": {
                "type": "boolean",
                "description": "true 时把当前对话分叉给子代理；默认 false，独立上下文。resume 时忽略。",
            },
            "resume": {
                "type": "string",
                "minLength": 1,
                "description": "本会话中一个已完成子代理的 task_id：在它自己的对话上继续执行新的 task。",
            },
        },
        # ``launch`` needs agent+task and ``message`` needs task_id+text; the
        # dispatcher enforces the pair the action requires.
        "required": [],
        "additionalProperties": False,
    },
    example={
        "agent": "auditor",
        "task": "读 workspace 根下 inputs/data_summary.json，返回可用字段、单位与 available_at 规则。",
        "description": "数据摘要与单位",
        "thinking": "medium",
    },
)


class AgentTool:
    """The parent-facing ``agent`` tool.

    Registered in the parent's tool registry so arguments go through the
    standard schema validation path; ``launch`` is the runner's background
    dispatcher and returns the ``started`` observation.
    """

    spec = AGENT_TOOL_SPEC

    def __init__(
        self, launch: Callable[[Mapping[str, object]], Mapping[str, object]]
    ) -> None:
        self._launch = launch

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        return ToolResult(True, value=dict(self._launch(arguments)))


def _subagent_mode(mode: str) -> str:
    if mode in {"meta", "meta_learning"}:
        return "meta"
    if mode == "fold":
        return "fold"
    raise ValueError("Sub-agent mode must be fold or meta")


def allowed_subagent_tools(mode: str, role: str | None = None) -> frozenset[str]:
    resolved = _subagent_mode(mode)
    spec = subagent_role(role) if role is not None else None
    if resolved == "meta":
        return _META_ROLE_TOOLS
    if spec is None:
        return frozenset().union(*(item.fold_tools for item in SUBAGENT_ROLE_TABLE))
    return spec.fold_tools


def subagent_system_prompt(mode: str, role: str) -> str:
    resolved = _subagent_mode(mode)
    spec = subagent_role(role)
    if resolved == "fold":
        template = _FOLD_WRITE_PROMPT if spec.shell else _FOLD_READ_PROMPT
        return template.format(role=role, mission=spec.fold_mission)
    return (
        f"# 本任务角色\n你的角色是 `{role}`：{spec.meta_mission}。\n\n"
        + META_SUBAGENT_SYSTEM_PROMPT
    )


def normalize_subagent_thinking(value: object, role: str | None = None) -> str:
    """Return the effective thinking level for one launch.

    Precedence: the call argument, then the role default, then
    ``DEFAULT_SUBAGENT_THINKING``. Omitted, empty, or inherit aliases fall
    through to the defaults; a child never inherits the parent session's
    reasoning intensity.
    """

    default = subagent_role(role).default_thinking if role is not None else DEFAULT_SUBAGENT_THINKING
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("agent.thinking must be a string")
    text = value.strip().lower()
    if text in {"", "inherit", "parent"}:
        return default
    text = _LEGACY_THINKING_ALIASES.get(text, text)
    if text not in SUBAGENT_THINKING_LEVELS:
        raise ValueError(
            "agent.thinking must be one of: " + ", ".join(SUBAGENT_THINKING_LEVELS)
        )
    return text


def resolve_subagent_max_turns(
    value: object, role: str, global_default: int | None
) -> int | None:
    """The effective turn budget: call argument > role default > global default.

    ``None`` means only the session deadline and call quota bound the child.
    """

    if isinstance(value, int) and not isinstance(value, bool):
        if value < 1:
            raise ValueError("agent.max_turns must be at least 1")
        return value
    if value is not None:
        raise ValueError("agent.max_turns must be an integer")
    return subagent_role(role).default_max_turns(global_default)


class ToolResultStore(Protocol):
    """Where an oversized result goes when it must leave the conversation
    (the search tools' spill store)."""

    def store_tool_result(
        self, *, tool: str, kind: str, content: str
    ) -> dict[str, object]: ...


def deliver_subagent_report(
    summary: str, store: ToolResultStore | None
) -> dict[str, object]:
    """Bound what a child's report puts into the parent's context.

    Up to ``SUBAGENT_REPORT_MAX_CHARS`` is delivered inline; a longer report
    is clipped there, marked ``summary_truncated``, and spilled in full to the
    result store as ``result_root``/``result_ref`` so the parent reads it back
    in pages. Without a store the clip is still explicit, never silent.

    The spill is read back with ``read_file``, which pages by LINE, so the clip
    is measured in lines as well as characters: a parent told only how many
    characters it received has to guess where to resume and reads past the end
    of the file for nothing. ``resume_line`` is that 0-based offset.
    """

    total = len(summary)
    if total <= SUBAGENT_REPORT_MAX_CHARS:
        return {"summary": summary, "summary_chars": total}
    delivered = summary[:SUBAGENT_REPORT_MAX_CHARS]
    payload: dict[str, object] = {
        "summary": delivered,
        "summary_chars": total,
        "summary_delivered_chars": SUBAGENT_REPORT_MAX_CHARS,
        "summary_lines": len(summary.splitlines()),
        # The line the clip fell inside is re-delivered whole rather than lost.
        "resume_line": delivered.count("\n"),
        "summary_truncated": True,
    }
    if store is not None:
        payload.update(store.store_tool_result(tool="agent", kind="report", content=summary))
    if "result_ref" in payload:
        # The store's generic hint names the file; only this caller knows the
        # report's line geometry, so it says where to resume too.
        payload["result_hint"] = (
            f"full report spilled ({payload['summary_lines']} lines); read the rest "
            f"with: read_file root='{payload['result_root']}' "
            f"path='{payload['result_ref']}' offset={payload['resume_line']}"
        )
    else:
        payload["result_hint"] = (
            "full report was not persisted; the rest is lost — have the child write "
            "long findings to a workspace file next time"
        )
    return payload


def llm_with_thinking(proxy: LLMProxy, thinking: str) -> LLMProxy:
    """Clone the gateway with this child's thinking level.

    The session hands the engine a budget wrapper, not the gateway itself; the
    wrapper clones itself over a re-configured gateway, so the level really
    reaches the request. A proxy that cannot take one (test doubles) is
    returned as is, and ``thinking_applied`` in the child's trace says which
    happened.
    """

    clone = getattr(proxy, "with_thinking", None)
    if clone is None:
        return proxy
    enabled = thinking != "off"
    # low/medium/xhigh are native levels for every supported dialect.
    return cast(
        LLMProxy,
        clone(enabled=enabled, reasoning_effort=thinking if enabled else None),
    )


@dataclass(frozen=True)
class SubAgentConfig:
    per_call_timeout_seconds: float | None = None
    # None = the shared ``AGENT_MAX_OUTPUT_TOKENS`` ceiling (same as the
    # parent conversation), clamped per call to the remaining context.
    max_tokens: int | None = None
    # Turn budget for a child whose launch omits ``max_turns``; None = only the
    # parent session deadline bounds it. Context is not a bound: a child
    # compacts at the parent's threshold (``SubAgentEngine.compactor``).
    max_rounds: int | None = DEFAULT_SUBAGENT_MAX_ROUNDS
    # None = no extra child wall clock; the parent time budget is the cap.
    deadline_seconds: float | None = None
    # Children running at once; further launches queue in the same pool.
    max_concurrent: int = DEFAULT_SUBAGENT_MAX_CONCURRENT

    def __post_init__(self) -> None:
        if self.per_call_timeout_seconds is not None and self.per_call_timeout_seconds <= 0:
            raise ValueError("Sub-agent per_call_timeout_seconds must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("Sub-agent max_tokens must be positive")
        if self.max_rounds is not None and self.max_rounds <= 0:
            raise ValueError("Sub-agent max_rounds must be positive")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("Sub-agent deadline_seconds must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("Sub-agent max_concurrent must be positive")


class SubAgentEngine(SessionTimeBudgetAware):
    """Bounded native-tool loop over the shared parent workspace."""

    def __init__(
        self,
        *,
        llm: LLMProxy,
        tools: ToolRegistry,
        config: SubAgentConfig | None = None,
        deadline_at: datetime | None = None,
        time_budget: InferenceTimeBudget | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
        mode: str = "fold",
        cancel_event: threading.Event | None = None,
        compactor: ContextCompactor | None = None,
    ) -> None:
        if mode not in SUBAGENT_MODES:
            raise ValueError("Sub-agent mode must be fold or meta")
        self.mode = mode
        self.system_prompt = (
            META_SUBAGENT_SYSTEM_PROMPT if mode == "meta" else SUBAGENT_SYSTEM_PROMPT
        )
        self.llm = llm
        self.tools = tools
        self.config = config or SubAgentConfig()
        self.deadline_at = deadline_at
        self.event_sink = event_sink
        # The parent session's compactor (gateway and thresholds); each child
        # runs a fresh instance of it over its own conversation. The runner
        # hands its own compactor down when none is given here.
        self.compactor = compactor
        self._cancel_event = cancel_event or threading.Event()
        bindings: list[TimeBudgetBinding] = []
        if isinstance(llm, SessionTimeBudgetAware):
            bindings.append(TimeBudgetBinding("subagent_llm", llm.session_time_budget))
        if compactor is not None:
            bindings.append(
                TimeBudgetBinding("subagent_compactor", compactor.session_time_budget)
            )
        self.time_budget = validate_time_budget_bindings(
            time_budget, tuple(bindings), owner="Sub-agent"
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

    def run(self, task: str, **kwargs: object) -> dict[str, object]:
        """Run one child and return its result record (see ``run_with_transcript``)."""

        result, _transcript = self.run_with_transcript(task, **kwargs)  # type: ignore[arg-type]
        return result

    def run_with_transcript(
        self,
        task: str,
        *,
        role: str,
        max_rounds: int | None = None,
        parent_call_id: str | None = None,
        thinking: str | None = None,
        inherit_context: bool = False,
        parent_messages: Sequence[ChatMessage] | None = None,
        transcript: Sequence[ChatMessage] | None = None,
        resumed_from: str | None = None,
        description: str = "",
        task_id: str | None = None,
        steer_queue: deque[str] | None = None,
    ) -> tuple[dict[str, object], tuple[ChatMessage, ...]]:
        """Run one child; return its result record and final transcript.

        ``transcript`` resumes a finished child's own conversation with the
        new task appended; otherwise the child starts from its role prompt,
        optionally forked from ``parent_messages``. ``steer_queue`` is the
        parent's mid-run instructions: drained before every model round, each
        one a labelled user message and a ``subagent_steer`` event.
        """

        if not task.strip():
            raise ValueError("Sub-agent task cannot be empty")
        allowed = allowed_subagent_tools(self.mode, role)
        self._validate_tools()
        # Launch precedence (Pi's): call argument > role default > global.
        rounds_limit = resolve_subagent_max_turns(max_rounds, role, self.config.max_rounds)
        thinking = normalize_subagent_thinking(thinking, role)
        task_id = task_id or f"{SUBAGENT_TASK_ID_PREFIX}{uuid.uuid4().hex[:12]}"
        child_cap = (
            time.monotonic() + self.config.deadline_seconds
            if self.config.deadline_seconds is not None
            else float("inf")
        )
        deadline = min(child_cap, self._deadline_monotonic())
        llm = llm_with_thinking(self.llm, thinking)
        thinking_applied = llm is not self.llm
        compactor = self.compactor.fresh() if self.compactor is not None else None
        started = {
            "task_id": task_id,
            "role": role,
            "parent_call_id": parent_call_id,
            "status": "started",
            "mode": self.mode,
            "model": getattr(llm, "model", "") or getattr(self.llm, "model", ""),
            # The effective launch values, whichever tier they came from.
            "thinking": thinking,
            "thinking_applied": thinking_applied,
            "rounds_limit": rounds_limit,
            "inherit_context": bool(inherit_context),
            # The brief is what delegation quality is judged by; clipped like
            # every other traced argument.
            "task": _traced_arguments(task.strip()),
        }
        if description:
            started["description"] = description
        if resumed_from:
            started["resumed_from"] = resumed_from
        self._emit("subagent_task", started)
        if transcript:
            # Resume: the child's own conversation continues; a transcript cut
            # off mid-batch still ends at its last answered turn.
            messages = drop_trailing_unanswered_tool_calls(
                [_copy_chat_message(message) for message in transcript]
            )
        else:
            messages = [ChatMessage("system", subagent_system_prompt(self.mode, role))]
            if inherit_context and parent_messages:
                # The parent snapshot is taken mid-batch: its last assistant
                # turn may carry tool calls (this launch among them) with no
                # results yet, so the fork ends at the last answered turn.
                messages.extend(
                    drop_trailing_unanswered_tool_calls(
                        [
                            _copy_chat_message(message)
                            for message in parent_messages
                            if message.role != "system"
                        ]
                    )
                )
        messages.append(ChatMessage("user", task.strip()))
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        rounds = 0
        tool_calls_made = 0
        summary = ""
        status = "completed"
        error = ""
        llm_calls = 0
        llm_errors = 0
        truncated_rounds = 0
        continuations = 0
        steers = 0
        try:
            while rounds_limit is None or rounds < rounds_limit:
                if self._cancelled():
                    status = "cancelled"
                    error = "Sub-agent cancelled"
                    break
                if self._deadline_reached(deadline):
                    status = "timeout"
                    error = "Sub-agent deadline reached"
                    break
                rounds += 1
                # The child is this queue's only consumer, so a non-empty
                # check followed by popleft cannot race the parent's appends.
                while steer_queue:
                    text = steer_queue.popleft()
                    steers += 1
                    messages.append(ChatMessage("user", f"{STEER_MESSAGE_LABEL} {text}"))
                    self._emit(
                        "subagent_steer",
                        {
                            "task_id": task_id,
                            "role": role,
                            "round": rounds,
                            "chars": len(text),
                            "delivery": "delivered",
                            "parent_call_id": parent_call_id,
                        },
                    )
                if (
                    rounds_limit is not None
                    and rounds_limit > SUBAGENT_GRACE_ROUNDS
                    and rounds == rounds_limit - SUBAGENT_GRACE_ROUNDS
                ):
                    # Grace turns: the child hears about the ceiling while it
                    # can still finish cleanly, instead of being cut off.
                    remaining_rounds = rounds_limit - rounds + 1
                    messages.append(
                        ChatMessage(
                            "user",
                            f"还剩 {remaining_rounds} 轮模型调用（上限 {rounds_limit}）："
                            "请立即收尾，用简洁中文汇报结论、关键证据与剩余风险，不要再开新的探索。",
                        )
                    )
                    self._emit(
                        "subagent_wrap_up",
                        {
                            "task_id": task_id,
                            "role": role,
                            "round": rounds,
                            "rounds_limit": rounds_limit,
                            "parent_call_id": parent_call_id,
                        },
                    )
                provider_tools = self._provider_tools(allowed)
                messages, output_tokens = self._fit_context(
                    compactor,
                    llm,
                    messages,
                    provider_tools,
                    deadline,
                    task_id=task_id,
                    role=role,
                    round_index=rounds,
                    parent_call_id=parent_call_id,
                )
                try:
                    response = llm.complete(
                        messages,
                        tools=provider_tools,
                        tool_choice="auto",
                        max_tokens=output_tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - child retry must not kill parent
                    llm_calls += 1
                    llm_errors += 1
                    error = safe_error_summary(exc)
                    self._emit(
                        "subagent_llm_error",
                        {
                            "task_id": task_id,
                            "role": role,
                            "round": rounds,
                            "provider": getattr(llm, "provider", ""),
                            "model": getattr(llm, "model", ""),
                            "llm_error": error,
                            "parent_call_id": parent_call_id,
                        },
                    )
                    if self._cancelled():
                        status = "cancelled"
                        error = "Sub-agent cancelled"
                        break
                    if _is_nonretryable_subagent_error(exc) or self._deadline_reached(
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
                # A transient provider error that a later round survived is
                # not this child's outcome.
                error = ""
                _add_usage(usage, response.usage)
                cut_off = _output_truncated(response.usage, output_tokens)
                truncated_rounds += int(cut_off)
                messages.append(
                    ChatMessage(
                        "assistant",
                        response.content,
                        response.tool_calls,
                        reasoning_content=response.reasoning_content,
                    )
                )
                self._emit(
                    "subagent_llm",
                    {
                        "task_id": task_id,
                        "role": role,
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
                    error = "Sub-agent cancelled"
                    break
                if not response.tool_calls:
                    text = (response.content or "").strip()
                    if text:
                        summary = text
                        if cut_off:
                            summary += "\n" + OUTPUT_TRUNCATED_MARKER.format(limit=output_tokens)
                        break
                    if cut_off:
                        # The whole budget went into thinking: ask for a
                        # concise continuation like the parent does, a bounded
                        # number of times; then the launch failed.
                        if continuations < SUBAGENT_MAX_TRUNCATION_CONTINUATIONS:
                            continuations += 1
                            completion = int(
                                dict(response.usage).get("completion_tokens") or 0
                            )
                            self._emit(
                                "subagent_output_truncated",
                                {
                                    "task_id": task_id,
                                    "role": role,
                                    "round": rounds,
                                    "completion_tokens": completion,
                                    "max_tokens": output_tokens,
                                    "continuation": continuations,
                                    "parent_call_id": parent_call_id,
                                },
                            )
                            messages.append(
                                ChatMessage(
                                    "user",
                                    json.dumps(
                                        {
                                            "observation": "output_truncated",
                                            "completion_tokens": completion,
                                            "max_tokens": output_tokens,
                                            "message": OUTPUT_TRUNCATED_CONTINUATION.format(
                                                limit=output_tokens
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                )
                            )
                            continue
                        status = "error"
                        error = (
                            f"output budget exhausted on reasoning in {truncated_rounds} "
                            "rounds without a tool call or report"
                        )
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
                        "subagent_tool",
                        {
                            "task_id": task_id,
                            "role": role,
                            "round": rounds,
                            "tool": call.name,
                            "arguments": _traced_arguments(call.arguments),
                            "result": sanitize_for_log(record),
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
                messages, output_tokens = self._fit_context(
                    compactor,
                    llm,
                    messages,
                    provider_tools,
                    deadline,
                    task_id=task_id,
                    role=role,
                    round_index=rounds,
                    parent_call_id=parent_call_id,
                )
                response = llm.complete(
                    messages,
                    tools=provider_tools,
                    tool_choice="none",
                    max_tokens=output_tokens,
                )
                llm_calls += 1
                _add_usage(usage, response.usage)
                summary = response.content.strip()
                if _output_truncated(response.usage, output_tokens):
                    truncated_rounds += 1
                    summary = (
                        summary + "\n" if summary else ""
                    ) + OUTPUT_TRUNCATED_MARKER.format(limit=output_tokens)
                if summary:
                    messages.append(ChatMessage("assistant", summary))
        except Exception as exc:  # noqa: BLE001 - a sub-agent failure must not kill the parent
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            error = safe_error_summary(exc)
        if status == "completed" and not summary:
            # ``completed`` means a report reached the parent; nothing else.
            if self._deadline_reached(deadline):
                status, error = "timeout", "Sub-agent deadline reached before a report"
            elif self._cancelled():
                status, error = "cancelled", "Sub-agent cancelled"
            else:
                status, error = "error", "Sub-agent ended without a report"

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
            "thinking": thinking,
            "thinking_applied": thinking_applied,
            "rounds_limit": rounds_limit,
            "inherit_context": bool(inherit_context),
        }
        if resumed_from:
            result["resumed_from"] = resumed_from
        if truncated_rounds:
            result["truncated"] = True
            result["truncated_rounds"] = truncated_rounds
        if llm_errors:
            result["llm_errors"] = llm_errors
        if steers:
            result["steers"] = steers
        if error:
            result["error"] = error
        self._emit(
            "subagent",
            {
                **result,
                "parent_call_id": parent_call_id,
            },
        )
        return result, tuple(messages)

    def _fit_context(
        self,
        compactor: ContextCompactor | None,
        llm: LLMProxy,
        messages: list[ChatMessage],
        provider_tools: tuple[dict[str, object], ...],
        deadline: float,
        *,
        task_id: str,
        role: str,
        round_index: int,
        parent_call_id: str | None,
    ) -> tuple[list[ChatMessage], int]:
        """The parent loop's context discipline before one child request.

        Semantic compaction at the shared threshold (forced when the request
        does not fit), then the emergency in-place tool-result summary; a
        request that still does not fit fails at the gateway, which ends the
        child instead of being retried.
        """

        output_tokens = self._output_tokens(llm, messages, provider_tools)
        identity = {
            "task_id": task_id,
            "role": role,
            "round": round_index,
            "parent_call_id": parent_call_id,
        }
        if compactor is not None:
            fits, _prompt_tokens, _window = context_request_fits(
                llm, messages, tools=provider_tools, max_tokens=output_tokens
            )
            result = compactor.compact(
                messages,
                tools=provider_tools,
                remaining_seconds=self._remaining_seconds(deadline),
                force=not fits,
            )
            if result is not None:
                # Nested: the compaction record's own ``status``/``summary``
                # must not read as the child's outcome or report.
                self._emit(
                    "subagent_context_compaction",
                    {**identity, "compaction": dict(result.event)},
                )
                messages = list(result.messages)
                output_tokens = self._output_tokens(llm, messages, provider_tools)
        messages, edit = fit_tool_results_to_context(
            llm, messages, tools=provider_tools, max_tokens=output_tokens
        )
        if edit:
            self._emit("subagent_context_edit", {**identity, "context_edit": edit})
            output_tokens = self._output_tokens(llm, messages, provider_tools)
        return messages, output_tokens

    def _remaining_seconds(self, local_deadline: float) -> float:
        remaining = local_deadline - time.monotonic()
        if self.time_budget is not None:
            remaining = min(remaining, self.time_budget.remaining())
        return max(remaining, 0.0)

    def _output_tokens(
        self,
        llm: LLMProxy,
        messages: Sequence[ChatMessage],
        tools: Sequence[object],
    ) -> int:
        requested = self.config.max_tokens or AGENT_MAX_OUTPUT_TOKENS
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

        rejections = [
            _reject_tool_call(self.tools.spec(call.name), allowed=allowed)
            for call in calls
        ]

        def run_one(index: int) -> tuple[ToolCall, dict[str, object], bool]:
            call = calls[index]
            if rejections[index]:
                return call, {"ok": False, "error": rejections[index]}, False
            if self._cancelled():
                return call, {"ok": False, "error": "Sub-agent cancelled"}, False
            if self._deadline_reached(deadline):
                return call, {"ok": False, "error": "Sub-agent deadline reached"}, False
            return call, self.tools.invoke(call.name, call.arguments).to_record(), True

        # Same rule as the parent runner: the whole batch runs concurrently
        # unless one call is sequential (mutating, gate, or rejected).
        can_parallel = (
            len(calls) > 1
            and not any(rejections)
            and all(
                not is_sequential_tool(self.tools.spec(call.name)) for call in calls
            )
        )
        if not can_parallel:
            return [run_one(index) for index in range(len(calls))]
        results: list[tuple[ToolCall, dict[str, object], bool] | None] = [None] * len(
            calls
        )
        interrupt: SessionInterrupt | None = None
        with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as executor:
            futures = {
                executor.submit(run_one, index): index for index in range(len(calls))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except SessionInterrupt as exc:
                    interrupt = exc
                except Exception as exc:  # noqa: BLE001 - one call must not drop its siblings
                    results[index] = (
                        calls[index],
                        {"ok": False, "error": safe_error_summary(exc)},
                        True,
                    )
        if interrupt is not None:
            raise interrupt
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
        # The role tables are the single allowlist: nesting, backtest, finish,
        # rollback, and ask_user are absent from every role by construction.
        allowed = allowed_subagent_tools(self.mode)
        for spec in self.tools.specs():
            if spec.name not in allowed:
                raise ValueError(f"Sub-agent tool is not allowed: {spec.name}")

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self.event_sink is not None:
            self.event_sink(event, dict(payload))


def _output_truncated(usage: object, max_tokens: int) -> bool:
    """True when a reply used its whole completion budget (finish_reason=length
    is not surfaced by the transport, so the usage count is the signal)."""

    if not isinstance(usage, Mapping):
        return False
    completion = usage.get("completion_tokens")
    return (
        isinstance(completion, (int, float))
        and not isinstance(completion, bool)
        and completion >= max_tokens
    )


def _is_nonretryable_subagent_error(exc: Exception) -> bool:
    if isinstance(exc, (SessionInterrupt, TimeoutError)):
        return True
    if is_context_overflow_error(exc):
        # Compaction and tool-result editing already ran; the next round
        # would only overflow again.
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


def _traced_arguments(arguments: object) -> object:
    """Sanitized, length-bounded arguments for the Trace record.

    Mirrors the parent ``tool_call`` echo; only the recorded copy is clipped,
    the tool itself still receives the full arguments.
    """

    return _clip_traced(sanitize_for_log(arguments))


def _clip_traced(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _clip_traced(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip_traced(item) for item in value]
    if isinstance(value, str) and len(value) > TRACE_ARGUMENT_MAX_CHARS:
        return f"{value[: TRACE_ARGUMENT_MAX_CHARS - 1]}…"
    return value


def _add_usage(total: dict[str, int], usage: object) -> None:
    if not isinstance(usage, dict):
        return
    for key in total:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] += value


def _reject_tool_call(spec: ToolSpec | None, *, allowed: frozenset[str]) -> str:
    if spec is None:
        return "unknown sub-agent tool"
    if spec.name not in allowed:
        return f"Sub-agent tool is not allowed: {spec.name}"
    return ""
