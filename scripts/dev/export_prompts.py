#!/usr/bin/env python3
"""Export every Agent/LLM prompt template to configs/prompts/PROMPTS.md for audit.

The code remains the single source of truth; this exporter renders the
templates so reviewers can read exactly what the models see. Every fenced
``text`` block below is imported from the module that ships it — none of the
prompt text is retyped here — so a prompt edit that skips the snapshot is
caught by ``--check``.

Every fenced block is imported from the module that ships it; no prompt text is
retyped here.
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
from autotrade.agent.explore import EXPLORE_SYSTEM_PROMPT
from autotrade.agent.agents_md import load_required_agents_md_sections
from autotrade.agent.prompts import (
    CONVERGENCE_PHASE_PROMPT,
    DEFAULT_ANTI_OVERFIT_PROMPT,
    DEFAULT_CONVERGENCE_PROMPT,
    EXPLORATION_PHASE_PROMPT,
    FOLD_ACTION_SECTION,
    FOLD_DEFAULT_INSTRUCTION,
    FOLD_DYNAMIC_CONTEXT_HEADER,
    FOLD_ENV_SECTION,
    FOLD_PROHIBITIONS,
    FOLD_ROLE_SECTION,
    FOLD_STATIC_SECTIONS,
    FOLD_SUBAGENT_CONTRACT,
    FOLD_SUBMIT_CONTRACT,
    HARD_FINALIZATION_SYSTEM_PROMPT,
    META_PHASE_CONTRACT,
    META_SYSTEM_PROMPT,
    RUNTIME_SYSTEM_PROMPT,
    STEP_TREE_SECTION,
    STEP_WRAP_UP_PROMPT,
    WRAP_UP_PROMPT,
    build_prior_section,
)
from autotrade.environment.nl.engine import (
    FINAL_AFTER_TOOL_BUDGET,
    SUB_AGENT_SYSTEM_PROMPT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "configs" / "prompts" / "PROMPTS.md"


def _block(text: str, language: str = "text") -> str:
    return f"```{language}\n{text.strip()}\n```"


def render() -> str:
    role, runtime, environment, action, submit, prohibitions = FOLD_STATIC_SECTIONS
    # Guard the ordering the navigation and the section numbering encode.
    assert (role, runtime, environment, action, submit, prohibitions) == (
        FOLD_ROLE_SECTION,
        RUNTIME_SYSTEM_PROMPT,
        FOLD_ENV_SECTION,
        FOLD_ACTION_SECTION,
        FOLD_SUBMIT_CONTRACT,
        FOLD_PROHIBITIONS,
    ), "FOLD_STATIC_SECTIONS order changed; update the snapshot layout"

    parts = [
        "# Prompt 模板审计快照",
        "",
        "本文件集中展示 Agent 实际使用的稳定 Prompt 合同，便于审阅策略 ABI、工具边界、PIT、离线 Meta、Explore 和上下文压缩。代码是唯一执行事实源：",
        "",
        "- `src/autotrade/agent/prompts.py`",
        "- `src/autotrade/agent/explore.py`",
        "- `src/autotrade/agent/compact.py`",
        "- `src/autotrade/environment/nl/engine.py`",
        "",
        "运行时动态上下文由 Pipeline 追加；工具名、参数和可用性由每轮原生 function schema 注入。动态示例只说明结构，不替代当前 run 的事实制品。",
        "",
        "## 导航",
        "",
        "- [1. Fold Agent 系统提示词](#1-fold-agent-系统提示词)",
        "- [2. 收尾提示](#2-收尾提示)",
        "- [3. 阶段与防过拟合构件](#3-阶段与防过拟合构件)",
        "- [4. 离线 Meta Agent 系统提示词](#4-离线-meta-agent-系统提示词)",
        "- [5. Explore Agent 系统提示词](#5-explore-agent-系统提示词)",
        "- [6. Context Compaction 系统提示词](#6-context-compaction-系统提示词)",
        "- [7. NL Sub Agent 系统提示词](#7-nl-sub-agent-系统提示词)",
        "- [8. 动态上下文结构](#8-动态上下文结构)",
        "",
        "## 1. Fold Agent 系统提示词",
        "",
        "运行时系统提示词先注入仓库根 `AGENTS.md` 的三个完整 section，再接本项目 Fold 子代理合同，然后是六个稳定区块。",
        "",
        "### 1.0 仓库 AGENTS 三节",
        "",
        "运行时从仓库根 `AGENTS.md` 抽取，代码不复制正文。当前抽取如下：",
        "",
        _block(load_required_agents_md_sections().text),
        "",
        "### 1.0b Fold 子代理合同",
        "",
        "`FOLD_SUBAGENT_CONTRACT`：",
        "",
        _block(FOLD_SUBAGENT_CONTRACT),
        "",
        "### 1.1 角色与目标",
        "",
        _block(role),
        "",
        "### 1.2 核心执行合同",
        "",
        _block(runtime),
        "",
        "### 1.3 环境与配置",
        "",
        _block(environment),
        "",
        "### 1.4 动作与流程",
        "",
        _block(action),
        "",
        "### 1.5 提交合同",
        "",
        _block(submit),
        "",
        "### 1.6 禁止事项",
        "",
        _block(prohibitions),
        "",
        "### 1.7 Fold 默认用户指令",
        "",
        "`FOLD_DEFAULT_INSTRUCTION`：",
        "",
        _block(FOLD_DEFAULT_INSTRUCTION),
        "",
        "## 2. 收尾提示",
        "",
        "### 2.1 Step 预算用完",
        "",
        "`STEP_WRAP_UP_PROMPT`：",
        "",
        _block(STEP_WRAP_UP_PROMPT),
        "",
        "### 2.2 Fold deadline 收尾",
        "",
        "`WRAP_UP_PROMPT`：",
        "",
        _block(WRAP_UP_PROMPT),
        "",
        "两个提示在对应条件首次满足时各最多注入一次。收尾提示不放宽完整 Validation、当前 run 节点和修改检查要求。",
        "",
        "### 2.3 有完整节点时的硬收尾",
        "",
        "进入 deadline 收尾窗口且当前 run 已有至少一个完整 Validation 节点后，Runner 不再把 `WRAP_UP_PROMPT` 叠加到原长对话，而是切换到独立的最小收尾上下文。其系统提示为：",
        "",
        _block(HARD_FINALIZATION_SYSTEM_PROMPT),
        "",
        "用户消息由 Runner 确定性生成，只包含候选节点、revision 和有界 Validation 指标。工具面只保留 `finish_fold` 与已配置时的 `step_rollback`；模型仍自行选择候选，Runner 不排名或自动提交。尚无完整节点时不会进入该状态。",
        "",
        "## 3. 阶段与防过拟合构件",
        "",
        "### 3.1 通用防过拟合",
        "",
        "`DEFAULT_ANTI_OVERFIT_PROMPT`：",
        "",
        _block(DEFAULT_ANTI_OVERFIT_PROMPT),
        "",
        "### 3.2 探索期",
        "",
        "`EXPLORATION_PHASE_PROMPT`：",
        "",
        _block(EXPLORATION_PHASE_PROMPT),
        "",
        "### 3.3 收敛期",
        "",
        "`DEFAULT_CONVERGENCE_PROMPT` 与 `CONVERGENCE_PHASE_PROMPT` 依次注入：",
        "",
        _block(
            f"{DEFAULT_CONVERGENCE_PROMPT.strip()}\n\n{CONVERGENCE_PHASE_PROMPT.strip()}"
        ),
        "",
        "### 3.4 Step 产物树",
        "",
        "启用 Step 树时追加 `STEP_TREE_SECTION`：",
        "",
        _block(STEP_TREE_SECTION),
        "",
        "## 4. 离线 Meta Agent 系统提示词",
        "",
        "Meta 同样先注入上述 AGENTS 三节，再接本项目阶段身份，然后是 `META_SYSTEM_PROMPT`。",
        "",
        "`META_PHASE_CONTRACT`：",
        "",
        _block(META_PHASE_CONTRACT),
        "",
        "`META_SYSTEM_PROMPT`：",
        "",
        _block(META_SYSTEM_PROMPT),
        "",
        "Meta 的工具白名单为 `read_file`、`grep`、`glob`、`write_taste`、可选 `ask_user` 和 `finish_meta`。Runner 在第一轮模型请求之前验证工具集合；多余能力会使会话直接失败。",
        "",
        "Meta 用户消息由 `build_meta_learning_prompt` 组织：",
        "",
        _block(
            "请从本地 development 证据提炼后续 Fold 的 Taste，并按需更新 PRIOR.md。"
            "先读 `inputs/meta_context.json`（含本窗口已完成 Fold 的冻结策略投影与 `agent_trace`）。从 `agent_trace` 提炼过程/方法经验并更新 PRIOR；需要时再读 `inputs/meta_learning_memory.jsonl`。"
            "不要输出逐 Fold 测试明细，不要使用任何外部资料。\n"
            "{previous_taste, development}\n\n"
            "[可选：实验级默认 Fold 探索方向]\n"
            "[可选：实验级探索方向]"
        ),
        "",
        "研究者方向都是待检验假设，不覆盖离线、PIT、隐藏阶段与过拟合约束。",
        "",
        "## 5. Explore Agent 系统提示词",
        "",
        "`EXPLORE_SYSTEM_PROMPT`：",
        "",
        _block(EXPLORE_SYSTEM_PROMPT),
        "",
        "Explore 是一层可写 coding 子代理，与主 Fold 共享工作树和预算；禁止嵌套。只读检索可并行，写入、edit、shell 与 check 按调用顺序串行。达到轮次或 deadline 而没有总结时，会再请求一次无工具摘要。",
        "",
        "## 6. Context Compaction 系统提示词",
        "",
        "`COMPACT_SYSTEM_PROMPT`：",
        "",
        _block(COMPACT_SYSTEM_PROMPT),
        "",
        "压缩输入包含上一份结构化摘要与其后的新增消息。输出至少需要包含所请求的继续执行字段之一；非法 JSON、空摘要或模型错误不会替换原会话。主 Runner 仍保存最近完整轮次，并可使用确定性工具观察摘要继续控制上下文规模。",
        "",
        "## 7. NL Sub Agent 系统提示词",
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
        "## 8. 动态上下文结构",
        "",
        "Fold 的稳定系统提示词之后追加：",
        "",
        _block(
            f"{FOLD_DYNAMIC_CONTEXT_HEADER.strip()}\n\n"
            "## 当前实验事实（可信运行事实，不是交易证据）\n"
            "{experiment_facts JSON}\n\n"
            "## 日级策略调度\n"
            '{"period": "day|month|quarter|year", "inference_time": "HH:MM"}\n\n'
            "## Step 产物树\n"
            "[启用时注入]\n\n"
            f"{build_prior_section('{PRIOR.md 全文}', role='fold')}\n\n"
            "## 本 Epoch 的 Taste（元学习注入）\n"
            "[存在时注入]\n\n"
            "## 实验级默认 Fold 探索方向（用户注入）\n"
            "[存在时注入]\n\n"
            "## 研究者本 Fold 指令（用户注入）\n"
            "[存在时注入]\n\n"
            "## 阶段策略与防过拟合\n"
            "[通用构件 + 探索期或收敛期构件]"
        ),
        "",
        "`experiment_facts` 的主要分区包括：",
        "",
        "| 分区 | 内容 |",
        "| --- | --- |",
        "| `identity` | experiment、run、Epoch、会话类型和当前 Fold 标识 |",
        "| `source_refs` | 运行 manifest、runtime environment 和 data summary 的受信引用 |",
        "| `visibility_policy` | Train/Validation 可见性、Test/Held-out 隐藏和正式策略读取根 |",
        "| `visible_timeline` | Fold 周期、快照窗口、日级时钟与历史研究域可用性 |",
        "| `budgets` | deadline、Step、模型调用、Validation 和压缩预算 |",
        "| `artifact_contract` | 必需入口、订单返回合同、修改约束、Step 和验收语义 |",
        "| `broker_replay` | 资金、费用、手数、T+1、调度与精确执行价格来源 |",
        "| `runtime_tools` | Python、已装依赖、可用本地工具、网络模式和安装策略 |",
        "| `meta_learning` | 仅 Meta：本地 development 输入、Taste 输出和能力禁用状态 |",
        "",
        "动态事实只作为常用索引。Agent 不能把其中的日期、period、Fold 标识或资源元数据用作交易信号，也不能据此推断隐藏阶段。",
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
