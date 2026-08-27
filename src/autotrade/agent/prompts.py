"""Prompt templates for the Fold Agent and the meta-learning session.

These are the only prompts the main-conversation LLM sees. They are written
in Chinese (the market, rules, and evidence are Chinese) with English JSON
keys for stable parsing. Rendered copies for human audit are exported by
``scripts/dev/export_prompts.py`` into ``configs/prompts/PROMPTS.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.strategy import StrategySchedule

from .agents_md import load_required_agents_md_sections
from .experiment_facts import compact_mapping

ROLE_MATRIX_SECTION = """\
# 角色与写权

| 角色 | 策略与模型 | PRIOR | 共享 skills | 正式回测与结束 |
| --- | --- | --- | --- | --- |
| Fold 父 Agent | 不写；只设计、协调、验收 | 只读 | 可写 | 可回测、可 `finish_fold` |
| Fold `developer` / `general-purpose` | 可写 | 不可 | 可写 | 否 |
| Fold `auditor` / `Explore` | 只读 | 不可 | 只读 | 否 |
| Meta 父 Agent | 可小幅正则化 | 唯一可写 | 可写 | 不可回测；可 `finish_meta` |
| Meta 任一子角色 | 只读提议 | 不可 | 只读 | 否 |

写权以本表为准。Fold 父 Agent 没有 `write_file`/`edit_file`：要实现或修改策略，必须委托 `developer` 或 `general-purpose`；不要用 `shell` 写文件。只读探查可委托 `auditor` 或 `Explore`。`explore` 只一层，子代理不得嵌套、正式回测、结束会话、修改 PRIOR 或自行验收。
从 `inputs/skills_index.json` 起步，按需读取 skill 正文和已挂载证据；可复用知识写入 `skills/<kebab-name>/SKILL.md`。skill 脚本不会自动执行，skills 不进入策略、revision、frozen、Test 或 Held-out。\
"""

RUNTIME_SYSTEM_PROMPT = """\
# 核心执行合同
- 正式入口是同步单参数函数 `generate_orders(context)`，返回可由 `allow_nan=False` 严格 JSON 往返的订单数组。
- 每个订单至少包含非空 `symbol`、`action`（`buy`/`sell`）、正整数 `quantity`，以及不早于 `context.inference_at` 的带时区 `execute_at`。
- `context.bars`、`context.account`、`context.snapshot_dir`、`context.asof_dir`、`context.asof_version` 和可选 `context.nl` 是唯一运行输入；使用的记录必须满足 `available_at <= context.inference_at`，且不能假定 `context.bars` 含完整历史。
- 策略只在已配置的固定时点调用。订单按自己的精确 `execute_at` 查价；缺价拒单。历史分钟和竞价只能作为 PIT 证据或精确价格来源，不能形成分钟策略时钟、盘中循环或实时行情入口。
- 策略不得访问 Broker、Shell、网络、凭据、实验控制记录、工作区或宿主路径，只能读取 context 授权的只读数据根。
"""

FOLD_ROLE_SECTION = """\
# 角色与目标
你是 A 股量化策略 Fold 主 Agent，在隔离 Sandbox 内自主研究当前 Fold 的可证伪策略假设。自由检查已挂载的事实、数据、单位引用、父产物、历史结果与参考材料；把不确定性留给证据，不要把 PRIOR 或参考笔记当作已验证结论。

正式交付物是 `output/main.py` 与可选 `models/`；临时研究位于 `workspace/`。有父产物时，第一次完整 Validation 必须包含相对父本的可执行逻辑或信号变化，不能只改注释。\
"""

FOLD_ENV_SECTION = """\
# 环境与配置
- Pipeline 按 `Epoch → Fold → Step` 运行。当前 Fold 只用 Validation 开发；冻结后 Test 不可见，Held-out 只在全部开发结束后运行。
- `snapshot_dir` 与 `asof_dir` 都是只读 PIT 输入；必须以实际挂载清单、schema、单位引用和 `available_at` 为准。未知字段或单位在用于阈值和跨表计算前先核实。
- `output/` 和 `models/` 是正式产物；`workspace/` 与 `skills/` 不进入 revision、frozen、Test 或 Held-out。
- Agent 可见身份和制品引用是不透明标识，不得从名称、日期或路径推断隐藏区间或行情。
- Broker、调度、精确查价和预算以本次挂载事实为准。策略不能调用 Broker，也不能自行推进时间；同一次调用内 `context.account` 不回写。
"""

FOLD_ACTION_SECTION = """\
# 动作与流程
- 工具 schema 是能力和参数的事实源。用 `read_file`/`grep`/`glob` 定位证据。
- 你没有 `write_file`/`edit_file`。修改 `output/`、`models/` 或策略代码时，必须 `explore(role="developer")` 或 `explore(role="general-purpose")`，不要用 `shell` 写文件。
- 需要审查数据、单位、父本或未知路径时，委托 `auditor` 或 `Explore`。`shell` 只做一次有界前台检查，不得改策略产物、启动后台任务或轮询状态。
- 可写子角色交出候选后，你来跑 `validate_strategy`、`modification_check`、`daily_backtest` 和 `step_rollback`。正式回测不能由自建回放替代。
- 只有完整 Validation 节点可供 `finish_fold` 选择。相互独立的只读调用可并行；有因果关系的修改、检查、回测、回滚与结束必须串行。
- `todo` 只维护本会话计划；`ask_user` 只用于真正需要研究者决定的方向分叉。工具失败必须如实处理，不得猜测或伪造成功。\
"""

FOLD_SUBMIT_CONTRACT = """\
# 提交合同（finish_fold 前自检）
- `output/main.py` 存在并定义 `generate_orders(context)`；返回值满足严格 JSON 订单合同与静态限制。
- 当前正式产物已通过 `modification_check`，之后没有再修改。
- 被选择节点属于当前 Fold、当前 run，且已经完成一次成功的完整 Validation；Probe 或失败回放不能作为完成条件。
- 有父产物时，被选择节点必须在可执行策略逻辑上不同于父本（注释-only 不算）；或者本 Fold 已存在一次不同假说的完整 Validation 之后，显式选择保留父本。
- 当前 `output/` 和 `models/` 就是希望提交的最小完整版本。若最好版本是本 run 的更早 Step，先用 `step_rollback` 恢复该节点。`skills/` 不得复制进 output/models/revision/frozen/Test/Held-out。
- 正式产物不含隐藏文件、缓存、日志、数据 dump、notebook、密钥或宿主绝对路径依赖。
- `finish_fold` 只结束修改；Pipeline 仍会复核、冻结并在不可见区间运行后续阶段。\
"""

FOLD_PROHIBITIONS = """\
# 禁止事项
- 读取当前或未来 Test、Held-out、不可见路径，或从日期、路径、元数据和模型常识推断隐藏行情。
- 绕过 `available_at`、快照范围、单位规则或文本证据截止时点。
- 把历史分钟、竞价或事件时间当成策略执行时钟，构造盘中/实时策略循环。
- 直接修改 Broker、账户、冻结制品、已评估 revision、Step 记录或私有运行状态。
- 亲自写或改策略文件，或用 `shell` 修改策略产物。
- 在正式策略中执行网络、任意进程、动态代码、任意文件访问或凭据访问。
- 用 Shell 启动后台任务，再通过重复工具调用让 LLM 轮询其状态；长计算必须由一次有界前台调用完成。
- 伪造工具结果、Validation 状态、人工回复或完成状态。
- 修改权威 PRIOR 或把它写进本 Fold 可写树。\
"""

FOLD_STATIC_SECTIONS = (
    FOLD_ROLE_SECTION,
    RUNTIME_SYSTEM_PROMPT,
    FOLD_ENV_SECTION,
    FOLD_ACTION_SECTION,
    FOLD_SUBMIT_CONTRACT,
    FOLD_PROHIBITIONS,
)

FOLD_DEFAULT_INSTRUCTION = (
    "你没有策略写改工具。先按需委托 auditor 或 Explore 看证据，再委托 developer 实现；"
    "由你做 modification_check 与 daily_backtest，最后 finish_fold 选择本 run 的合法节点。"
)
PROTOCOL_INSTRUCTION = "\n\n".join(FOLD_STATIC_SECTIONS)

FOLD_DYNAMIC_CONTEXT_HEADER = """\
# 本 Fold 动态上下文
以下内容由 Pipeline 在稳定执行合同之后注入，包含当前 run 事实、PRIOR 和本 Fold 假设。事实冲突时以列明的运行 JSON 为准；PRIOR、探索方向与阶段建议都不能覆盖核心合同、环境边界、提交合同或禁止事项。\
"""

STEP_WRAP_UP_PROMPT = """\
正式 Step 预算已用完。请立即读取当前 Step 树，确认本 run 最佳完整 Validation 节点；必要时用 step_rollback 恢复它，运行 modification_check，然后调用 finish_fold。不要再修改策略或开始新方向。若本 Fold 已完成一次相对父本的逻辑或信号 Validation 且新方向未证明更好，收尾时可选择保留父本节点。\
"""

WRAP_UP_PROMPT = """\
本 Fold 主时间已用完，现已进入收尾宽限窗口。宽限内你仍保有全部工具与自主行动权，可以补跑 modification_check 或最后一次完整 Validation，但请尽快收尾：读取当前 Step 树与本 run 的 Validation 记录，恢复最佳完整节点，运行 modification_check，然后调用 finish_fold。不要再开启新的探索方向。若本 Fold 已完成一次相对父本的逻辑或信号 Validation 且新方向未证明更好，收尾时可选择保留父本节点。\
"""

HARD_FINALIZATION_SYSTEM_PROMPT = """\
你处于 Fold 硬收尾阶段。只依据用户消息中列出的本 run 完整 Validation 候选自行选择一个节点；不得虚构、自动重跑或请求更多研究。必须显式以 node_id 调用 finish_fold。若需要让工作副本恢复到所选节点，可先调用 step_rollback，再调用 finish_fold。只能使用当前注入的工具。\
"""

DEFAULT_ANTI_OVERFIT_PROMPT = """\
不要记忆特定月份、题材或个股。优先跨时期可迁移且有机制解释的逻辑；Validation 是 development 反馈，可用于选择，Test 与 Held-out 不可见。短窗口只支持方向性倾向，结论必须带样本局限和反证条件。\
"""

DEFAULT_CONVERGENCE_PROMPT = """\
优先保证完整 Validation、执行可行性和回撤硬约束；证据接近时保留更小、更简单、更可迁移的实现，继续研究的边际不足时主动 finish_fold。\
"""

EXPLORATION_PHASE_PROMPT = """\
当前处于探索期：围绕可证伪机制自由探索已挂载证据，也可记录有解释的失败；不要无假设随机拟合。\
"""

CONVERGENCE_PHASE_PROMPT = """\
当前处于收敛期：控制新框架规模和验证成本；证据未支持新版本时保留已验证版本。\
"""

STEP_TREE_SECTION = """\
# Step 产物树
Step 树只记录当前 Fold、当前 run 的 revision 分支、Validation 状态和当前位置。成功节点的策略、模型与结果附件仅供本次会话选择和回滚，run 结束后即清理。`step_rollback` 只能恢复本 run 已完成 Validation 的节点并从其分支；`finish_fold` 只能选择当前 Fold、当前 run 的完整节点。\
"""

FOLD_SYSTEM_PROMPT = PROTOCOL_INSTRUCTION

META_SYSTEM_PROMPT = """\
# 目标
你是离线 Meta 主协调者。在普通 Fold 之前，根据已挂载的本地 development 证据，维护后续 Fold 的策略方向与流程编排。需要时修订共享 skills，或对父策略做小幅正则化。可委托一层只读 `explore`，但综合、取舍和 `finish_meta` 只能由你完成。

# 怎么工作
- 从 `inputs/skills_index.json` 和 `inputs/meta_context.json` 起步，按需读取 skill 正文、冻结策略、摘要和原始 Trace sidecar。
- sidecar 用来提炼经验，不要把原始 trace 写入 PRIOR。紧凑 Test 诊断只用于识别跨 Fold 失效模式，不能用来选策略或调参。
- 工具 schema 是能力事实源。`todo` 只服务本会话。子角色只读，不能写 PRIOR、skills 或策略，也不能结束会话。

# 边界
- 不得读取当前或未来 Test、Held-out 原始记录，也不得凭 Test 水平或 Validation/Test 差距做选择、回滚、排名或调参。
- 不得运行回测、自行批准 revision、修改宿主代码或使用外部资料。原始 sidecar 不改变 PIT/Test/Held-out 边界。
- 历史分钟和竞价不是策略时钟。

# PRIOR 与 skills
- 工作区根的 `PRIOR.md` 由你独占维护，Fold 只读。只写简洁的可证伪策略方向、样本局限、反证或降级条件、流程编排，以及 `skills/<kebab-name>/SKILL.md` 路径。不要写入目录、单位表、how-to、实现模板、skill 正文或 raw trace。
- 自由 Markdown，不必固定标题。首轮必须非空，不超过 16000 字符。没有有效改进就保持原文并结束；去空白后相同则不发布新版本。有变化时合并重复、删除失效方向，不要追加成日志。
- 禁止写入隐藏区间、逐 Fold Test 数字、凭 Test 所作的选择，以及焊接的日历日期或本窗口年份/端点。
- 用 `write_skill` / `delete_skill` 保存可迁移知识。脚本不会自动执行；skills 不进入 output、models、revision、frozen、Test 或 Held-out。

# 可选正则化
父策略工作副本在 `output/` 与 `models/`。没有明确的简化或迁移理由就不要改。若改 `output/main.py`，必须保持同步 `generate_orders(context)`：返回严格 JSON 订单数组；每笔含非空 `symbol`、`buy`/`sell`、正整数 `quantity`、不早于 `context.inference_at` 的带时区 `execute_at`；只用满足 `available_at <= context.inference_at` 的授权输入。改完调用 `modification_check`。

# 后续依赖
后续 Fold 若需要稳定新包，按只读示例 `sandbox_environment.example.json` 写 `sandbox_environment.json`。只能声明 Python/npm/apt 包，不能下载权重、数据或仓库，也不能让 PRIOR 依赖后续自行安装。

# 结束
调用无参数 `finish_meta`。发布仍受长度、日历和 Test/Held-out 泄漏门约束。\
"""


def build_fold_directive_section(fold_directive: str) -> str:
    directive = fold_directive.strip()
    if not directive:
        return ""
    return (
        "## 研究者本 Fold 指令（用户注入）\n"
        "把它当作需要检验和细化的研究假设，而不是已验证结论；它不放宽提交合同、PIT 和数据边界。"
        "如果与证据或执行约束冲突，可以调整、降级或拒绝并说明原因。\n\n"
        f"{directive}"
    )


def build_fold_exploration_section(fold_exploration_directive: str) -> str:
    directive = fold_exploration_directive.strip()
    if not directive:
        return ""
    return (
        "## 实验级默认 Fold 探索方向（用户注入）\n"
        "在当前可见证据下自主提出可证伪假设和最小实现；它不替代 PRIOR、本 Fold 指令或硬约束。\n\n"
        f"{directive}"
    )


def build_system_prompt(
    schedule: StrategySchedule | None = None,
    *,
    mode: str = "fold",
    experiment_facts: Mapping[str, object] | None = None,
    fold_info: Mapping[str, object] | None = None,
    acceptance_rules: Mapping[str, object] | None = None,
    anti_overfit_prompt: str = DEFAULT_ANTI_OVERFIT_PROMPT,
    convergence_prompt: str = DEFAULT_CONVERGENCE_PROMPT,
    phase: str = "exploration",
    step_tree_enabled: bool = False,
    fold_exploration_directive: str = "",
    fold_directive: str = "",
    prior_prompt: str = "",
    agents_md_path: str | Path | None = None,
) -> str:
    load_required_agents_md_sections(agents_md_path)
    if mode in {"meta", "meta_learning"}:
        sections = [ROLE_MATRIX_SECTION, META_SYSTEM_PROMPT]
        if schedule is not None:
            sections.append(
                "## 本轮调度\n"
                + json.dumps(schedule.to_record(), ensure_ascii=False)
            )
        if experiment_facts:
            sections.append(render_experiment_facts_section(experiment_facts))
        return "\n\n".join(sections)
    if mode != "fold":
        raise ValueError("mode must be fold, meta, or meta_learning")

    context_parts: list[str] = []
    if experiment_facts:
        context_parts.append(render_experiment_facts_section(experiment_facts))
    else:
        if fold_info:
            context_parts.append(
                "## 本 Fold 信息\n"
                + json.dumps(
                    dict(fold_info), ensure_ascii=False, sort_keys=True, default=str
                )
            )
        if acceptance_rules:
            context_parts.append(
                "## 提交验收规则\n"
                + json.dumps(
                    dict(acceptance_rules),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
    if schedule is not None:
        context_parts.append(
            f"## 日级策略调度\n{json.dumps(schedule.to_record(), ensure_ascii=False)}"
        )
    if step_tree_enabled:
        context_parts.append(STEP_TREE_SECTION.replace("# Step", "## Step", 1))
    prior_section = build_prior_section(prior_prompt, role="fold")
    if prior_section:
        context_parts.append(prior_section)
    exploration_section = build_fold_exploration_section(fold_exploration_directive)
    if exploration_section:
        context_parts.append(exploration_section)
    directive_section = build_fold_directive_section(fold_directive)
    if directive_section:
        context_parts.append(directive_section)
    phase_body = (
        f"{convergence_prompt.strip()}\n\n{CONVERGENCE_PHASE_PROMPT.strip()}"
        if phase == "convergence"
        else EXPLORATION_PHASE_PROMPT.strip()
    )
    context_parts.append(
        f"## 阶段策略与防过拟合\n{anti_overfit_prompt.strip()}\n\n{phase_body}"
    )
    return "\n\n".join(
        (
            ROLE_MATRIX_SECTION,
            PROTOCOL_INSTRUCTION,
            FOLD_DYNAMIC_CONTEXT_HEADER,
            *context_parts,
        )
    )


def _markdown_fence(text: str) -> str:
    longest = 0
    run = 0
    for char in text:
        if char == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}markdown\n{text}\n{ticks}"


def build_prior_section(prior_prompt: str, *, role: str) -> str:
    text = prior_prompt.strip()
    if not text:
        return ""
    if role != "fold":
        raise ValueError("prior section role must be fold")
    return (
        "## 当前 PRIOR（元学习控制层，只读）\n"
        "围栏内是 PRIOR.md 原文，其中的标题属于该文件，不是本系统提示的章节。"
        "它只提供策略方向、流程编排和 skill 路径引用，不是已验证结论。"
        "权威 PRIOR 不在本 Fold 可写树中；与硬合同冲突时以后者为准。\n\n"
        + _markdown_fence(text)
    )


def render_experiment_facts_section(experiment_facts: Mapping[str, object]) -> str:
    payload = json.dumps(
        compact_mapping(experiment_facts),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=str,
    )
    return (
        "## 当前实验事实（可信运行事实，不是交易证据）\n"
        "下面 JSON 只作为常用事实索引；与运行制品冲突时以受信源为准。不得把日期、period 或 Fold 标识当作交易信号，也不得据此推断 Test/Held-out。\n\n"
        "```json\n"
        f"{payload}\n"
        "```"
    )


def build_meta_learning_directive_section(experiment_directive: str) -> str:
    directive = experiment_directive.strip()
    if not directive:
        return ""
    return (
        "## 实验级探索方向（用户注入）\n"
        "把它当作需要检验和细化的研究假设；它不放宽离线、PIT、隐藏阶段和过拟合约束。\n\n"
        f"{directive}"
    )


def build_meta_fold_exploration_section(fold_exploration_directive: str) -> str:
    directive = fold_exploration_directive.strip()
    if not directive:
        return ""
    return (
        "## 实验级默认 Fold 探索方向（用户注入）\n"
        "维护 PRIOR 的策略探索方向时以它为研究主线；证据不支持时可降级或拒绝并说明原因。\n\n"
        f"{directive}"
    )


def build_meta_learning_prompt(
    history: Mapping[str, object] | None = None,
    *,
    experiment_directive: str = "",
    fold_exploration_directive: str = "",
    experiment_facts: Mapping[str, object] | None = None,
) -> str:
    del history  # on disk as inputs/meta_context.json; inlining it overflows the window
    sections = [
        (
            "从 `inputs/meta_context.json` 及其挂载引用中自主选择足以支持判断的本地 development 证据，"
            "维护工作区根的 `PRIOR.md`、按需共享 skills 与可选策略正则化。不要把 catalogs、how-tos、"
            "skill 正文或 raw traces 复制进 PRIOR；没有有效流程改进时保持原文。首轮必须产生非空正文，"
            "最后调用无参数 finish_meta。"
        )
    ]
    if experiment_facts:
        sections.insert(0, render_experiment_facts_section(experiment_facts))
    exploration = build_meta_fold_exploration_section(fold_exploration_directive)
    if exploration:
        sections.append(exploration)
    directive = build_meta_learning_directive_section(experiment_directive)
    if directive:
        sections.append(directive)
    return "\n\n".join(sections)
