#!/usr/bin/env python3
"""Export every Agent/LLM prompt template to configs/prompts/PROMPTS.md for audit.

The code remains the single source of truth; this exporter renders the
templates so reviewers can read exactly what the models see. Every fenced
``text`` block below is imported from the module that ships it — none of the
prompt text is retyped here — so a prompt edit that skips the snapshot is
caught by ``--check``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.agent.compact import COMPACT_SYSTEM_PROMPT
from autotrade.agent.prompts import (
    HARD_FINALIZATION_SYSTEM_PROMPT,
    PRINCIPLES_SECTION,
    ROLE_MATRIX_SECTION,
    RUNTIME_SYSTEM_PROMPT,
    SESSION_DECISION_CONTRACT,
    SESSION_DEFAULT_INSTRUCTION,
    SESSION_DYNAMIC_CONTEXT_HEADER,
    SESSION_EVIDENCE_SECTION,
    SESSION_FACTS_SECTION,
    SESSION_FEEDBACK_SECTION,
    SESSION_PROHIBITIONS,
    SESSION_PROTOCOL_SECTION,
    SESSION_ROLE_SECTION,
    SESSION_STATIC_SECTIONS,
    SESSION_WORKFLOW_SECTION,
    STEP_TREE_SECTION,
    WRAP_UP_PROMPT,
)
from autotrade.agent.subagent import AGENT_TOOL_DESCRIPTION, subagent_system_prompt
from autotrade.environment.nl.engine import (
    FINAL_AFTER_TOOL_BUDGET,
    SUB_AGENT_SYSTEM_PROMPT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "configs" / "prompts" / "PROMPTS.md"


def _block(text: str, language: str = "text") -> str:
    return f"```{language}\n{text.strip()}\n```"


def render() -> str:
    # Guard the ordering the navigation and the section numbering encode.
    assert SESSION_STATIC_SECTIONS == (
        SESSION_ROLE_SECTION,
        SESSION_PROTOCOL_SECTION,
        SESSION_DECISION_CONTRACT,
        SESSION_EVIDENCE_SECTION,
        PRINCIPLES_SECTION,
        SESSION_WORKFLOW_SECTION,
        ROLE_MATRIX_SECTION,
        RUNTIME_SYSTEM_PROMPT,
        SESSION_PROHIBITIONS,
        SESSION_FACTS_SECTION,
        SESSION_FEEDBACK_SECTION,
        STEP_TREE_SECTION,
    ), "SESSION_STATIC_SECTIONS order changed; update the snapshot layout"

    parts = [
        "# Prompt 模板审计快照",
        "",
        "本文件集中展示 Agent 实际使用的稳定 Prompt 合同，便于审阅策略 ABI、工具边界、PIT、子代理和上下文压缩。代码是唯一执行事实源：",
        "",
        "- `src/autotrade/agent/prompts.py`",
        "- `src/autotrade/agent/subagent.py`",
        "- `src/autotrade/agent/compact.py`",
        "- `src/autotrade/environment/nl/engine.py`",
        "",
        "系统提示词先放静态内容，再放本次运行的动态事实，使跨会话的共享前缀字节稳定。工具名、参数和可用性由每轮原生 function schema 注入；动态示例只说明结构，不替代当前 run 的事实制品。宿主 `AGENTS.md` 不注入。",
        "",
        "## 导航",
        "",
        "- [1. 研究会话系统提示词](#1-研究会话系统提示词)",
        "- [2. 收尾提示](#2-收尾提示)",
        "- [3. agent 工具与子代理系统提示词](#3-agent-工具与子代理系统提示词)",
        "- [4. Context Compaction 系统提示词](#4-context-compaction-系统提示词)",
        "- [5. NL Sub Agent 系统提示词](#5-nl-sub-agent-系统提示词)",
        "- [6. 动态上下文结构](#6-动态上下文结构)",
        "",
        "## 1. 研究会话系统提示词",
        "",
        "十二个稳定区块按「目的 → 协议 → 决策合同 → 证据 → 约束 → 事实 → 反馈 → Step 树」的顺序拼接，再接动态上下文。",
        "",
        "### 1.1 身份与任务",
        "",
        _block(SESSION_ROLE_SECTION),
        "",
        "### 1.2 研究协议",
        "",
        _block(SESSION_PROTOCOL_SECTION),
        "",
        "### 1.3 决策合同",
        "",
        _block(SESSION_DECISION_CONTRACT),
        "",
        "### 1.4 证据标准",
        "",
        _block(SESSION_EVIDENCE_SECTION),
        "",
        "### 1.5 原则",
        "",
        "宿主开发原则中真正适用于策略研究的浓缩版。",
        "",
        _block(PRINCIPLES_SECTION),
        "",
        "### 1.6 工具与工作方式",
        "",
        _block(SESSION_WORKFLOW_SECTION),
        "",
        "### 1.7 角色与写权",
        "",
        _block(ROLE_MATRIX_SECTION),
        "",
        "### 1.8 执行合同与边界",
        "",
        _block(RUNTIME_SYSTEM_PROMPT),
        "",
        "### 1.9 禁止事项",
        "",
        _block(SESSION_PROHIBITIONS),
        "",
        "### 1.10 预算与事实",
        "",
        _block(SESSION_FACTS_SECTION),
        "",
        "### 1.11 反馈通道",
        "",
        _block(SESSION_FEEDBACK_SECTION),
        "",
        "### 1.12 Step 产物树",
        "",
        "`STEP_TREE_SECTION`：",
        "",
        _block(STEP_TREE_SECTION),
        "",
        "### 1.13 默认用户指令",
        "",
        "`SESSION_DEFAULT_INSTRUCTION`（首条用户消息：具体的开局委托计划）：",
        "",
        _block(SESSION_DEFAULT_INSTRUCTION),
        "",
        "## 2. 收尾提示",
        "",
        "### 2.1 deadline 收尾",
        "",
        "`WRAP_UP_PROMPT`：",
        "",
        _block(WRAP_UP_PROMPT),
        "",
        "到达主截止时注入一次，不放宽完整验证、当前 run 节点、冻结门和修改检查要求。replay-year 预算用尽时没有单独的提示：`batch_validate` 的返回行已说明不能再跑一批。",
        "",
        "### 2.2 有完整节点时的硬收尾",
        "",
        "进入 deadline 收尾窗口且当前 run 已有至少一个完整验证节点后，Runner 不再把 `WRAP_UP_PROMPT` 叠加到原长对话，而是切换到独立的最小收尾上下文。其系统提示为：",
        "",
        _block(HARD_FINALIZATION_SYSTEM_PROMPT),
        "",
        "用户消息由 Runner 确定性生成，只包含候选节点、revision、有界验证指标与每个节点此刻的冻结门结论（`passes_freeze_gate`）。工具面只保留 `finish_session`；模型仍自行选择结局，Runner 不排名或自动提交。尚无完整节点时不会进入该状态。是否调用过 `agent` 不影响进入硬收尾。",
        "",
        "## 3. agent 工具与子代理系统提示词",
        "",
        "### 3.0 `agent` 工具描述",
        "",
        "父会话看到的 `agent` function 描述——子代理机制只在这里向模型说明；参数 `agent`、`task`、可选 `description`、`max_turns`、`thinking`、`inherit_context`、`resume` 由 schema 给出：",
        "",
        _block(AGENT_TOOL_DESCRIPTION),
        "",
        "### 3.1 general-purpose",
        "",
        "`subagent_system_prompt('general-purpose')`：",
        "",
        _block(subagent_system_prompt("general-purpose")),
        "",
        "### 3.2 Explore",
        "",
        "`subagent_system_prompt('Explore')`：",
        "",
        _block(subagent_system_prompt("Explore")),
        "",
        "父会话可按任务自由选择或省略委托。只有 `general-purpose` 可写策略和 skills；`Explore` 只读，无 shell。所有角色禁止嵌套。",
        "",
        "## 4. Context Compaction 系统提示词",
        "",
        "`COMPACT_SYSTEM_PROMPT`：",
        "",
        _block(COMPACT_SYSTEM_PROMPT),
        "",
        "压缩输入包含上一份结构化摘要与其后的新增消息。输出至少需要包含所请求的继续执行字段之一；非法 JSON、空摘要或模型错误不会替换原会话。主 Runner 仍保存最近完整轮次，并可使用确定性工具观察摘要继续控制上下文规模。确定性工具结果缩写保留省略说明、`original_chars`、`head`、`tail`和可用的`retained_fields`，并明确标记`source_omitted=true`；不生成内容指纹。",
        "",
        "## 5. NL Sub Agent 系统提示词",
        "",
        "`SUB_AGENT_SYSTEM_PROMPT`。NL 只在已经召回的本地 PIT 证据上工作，检索由 `text_retrieve` function tool 完成：",
        "",
        _block(SUB_AGENT_SYSTEM_PROMPT),
        "",
        "工具预算用完时追加 `FINAL_AFTER_TOOL_BUDGET`，要求立即给出最终回答：",
        "",
        _block(FINAL_AFTER_TOOL_BUDGET),
        "",
        "证据条数、单条字符量、总字符量、模型轮数、单决策调用数和 deadline 都由 `NLConfig` 限制。没有可见证据时不启动模型；声明 `response_contract` 时只返回一个允许值，否则回答格式由调用方策略决定。所有证据都必须能回溯到推断时点已经可见的文本，不得伪造证据标识。",
        "",
        "## 6. 动态上下文结构",
        "",
        "稳定系统提示词之后追加：",
        "",
        _block(
            f"{SESSION_DYNAMIC_CONTEXT_HEADER.strip()}\n\n"
            "## 当前实验事实（可信运行事实，不是交易证据）\n"
            "{experiment_facts JSON，含 inputs/skills_index.json 引用}\n\n"
            "## 日级策略调度\n"
            '{"period": "day|month|quarter|year", "inference_time": "HH:MM"}\n\n'
            "## 实验级默认探索方向（用户注入）\n"
            "[存在时注入]\n\n"
            "## 研究者本会话指令（用户注入）\n"
            "[存在时注入]"
        ),
        "",
        "`experiment_facts` 的主要分区包括：",
        "",
        "| 分区 | 内容 |",
        "| --- | --- |",
        "| `identity` | experiment、run 与会话引用 |",
        "| `source_refs` | 运行 manifest、runtime environment、data summary、skills 索引与信号筛选脚本的受信引用 |",
        "| `visibility_policy` | 研究期可见、研究期末之后封存，以及正式策略读取根 |",
        "| `research_geometry` | 决策时点、输入窗口、研究期、各研究年份与 span 写法；只有研究期日期 |",
        "| `visible_timeline` | 快照窗口、日级时钟与历史研究域可用性 |",
        "| `research_scope` | 研究期与会话结局、股票池和调用节奏各一句 |",
        "| `arm` | 本臂尚未冻结、至多冻结一次，以及至今的试验数与完整研究期验证数 |",
        "| `budgets` | deadline、replay-year、空对照、模型调用、策略容器超时与资源、压缩预算 |",
        "| `artifact_contract` | 必需入口、订单返回合同、起点、修改约束、冻结门与毕业条件 |",
        "| `broker_replay` | 资金、费用、手数、T+1、调度与精确执行价格来源 |",
        "| `runtime_tools` | Python、已装依赖、可用本地工具、网络模式和安装策略，以及各读文件根在 `shell` 里的挂载路径 |",
        "| `workspace` / `forbidden` | 工作区索引与禁止访问的范围 |",
        "",
        "动态事实只作为常用索引。Agent 不能把其中的日期、period、会话标识或资源元数据用作交易信号，也不能据此推断研究期末之后的行情。",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DOC_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed snapshot is stale.",
    )
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(
                "configs/prompts/PROMPTS.md is stale; run scripts/dev/export_prompts.py",
                file=sys.stderr,
            )
            return 1
        print(f"current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
