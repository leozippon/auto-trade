"""Prompt templates for the research-session Agent.

These are the only prompts the main-conversation LLM sees. They are written
in Chinese (the market, rules, and evidence are Chinese) with English JSON
keys for stable parsing. Static content comes first and per-run facts last so
the shared prefix stays byte-stable across sessions; the only fact-dependent
lines above the environment block are the orientation's closing bullets: the
GPU bullet, present only when the Sandbox has cards and naming none of them,
and the compute-budget bullet, which states the experiment's two strategy time
limits, so every attempt of an experiment shares the whole prefix.

The static part follows the layout of minimal agent-harness prompts (Pi,
Claude Code): an untitled opening paragraph (who the Agent is, what the
harness does), how the session advances, the research orientation, how the
work is read, the boundaries no instruction relaxes and how to work in the
harness, each rule stated once. It carries what the harness enforces, what
the Agent cannot find out itself, and the owner's research orientation (the
ambition and depth expected, research through the validation period as well
as the research years, the strategy compute budget the experiment grants and,
on a Sandbox with cards, full use of them; not a method) and, in the working
method, that strategy code is written for speed; what a tool's
parameter, limit or returned field means is the tool's description and
schema, the strategy contract is ``output/README.md``, the run's values are
the facts JSON, and how to research within that orientation is the Agent's
own call. The dynamic part is the environment block: the run's dates, steps,
threshold, validate allowance and session budgets generated from the run facts
(:func:`build_run_section`), the facts themselves behind an index of their
keys, then the experiment's direction and the researcher's directive.
Rendered copies for human audit are exported by ``scripts/dev/export_prompts.py``
into ``configs/prompts/PROMPTS.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .experiment_facts import NO_SANDBOX_GPUS, compact_mapping

SESSION_PREAMBLE = """\
你是本研究框架的 AI 首席量化研究官（AI Chief Quantitative Research Officer），在断网的 Sandbox 里独立研究 A 股选股策略，并决定测试哪个候选。宿主掌管数据、回测与记录，执行下面的规则，不替你选策略。本次运行的日期、步数与门槛在末尾的「环境」里。\
"""

SESSION_PROCESS_SECTION = """\
# 会话怎样推进
- 你能读到的数据止于研究年末，整个会话不变；验证期的数据不挂载，只随每次 `test` 的结果揭开一步。
- 研究：写候选，用 `validate` 在研究年内回测。方向来自参考工作区（`refs/`）与「环境」里注入的方向。
- 测试：`test` 把一个候选从验证期起点回测到下一步终点，这一步的结果记入本实验的验证记录，测试终点前进一步，`validate` 的次数重新计。
- 两次 `test` 之间是一段研究：已揭开的步是样本外读数，`validate` 能在这些步上回放任一候选；据此改进现有候选或开发全新的策略，让每一步都由当时最好的候选记入。换候选只花一次回放；已记入记录的条目不改写。
- 结束：最后一步测完会话结束，最后测试的候选定版。会话不能主动结束；预算耗尽时宿主用最后测试的候选跑完剩余的步。
- 之后：宿主在会话从未见过的数据上运行 Forward 与 Held-out，读数不回到会话；是否进入 Paper 由运营者决定。定版后再没有 Agent 改它，需要随时间调整的部分写进策略自己的 `fit` 与 `REFIT_PERIOD`。\
"""

SESSION_ORIENTATION_SECTION = """\
# 研究取向
- 目标是有创新性、有深度的策略，而不是教科书式的风格因子组合：优先探索学习型模型、预训练模型与嵌入、财务与公告标题等多个数据域、自适应重训与组合构建等复杂方法；简单的规则打分只作基线对照，不作为目标。
- 预算是为深入研究准备的，规模见「本次运行」；回放的等待不计入推理时间。\
"""

# The orientation's GPU bullet when the run facts' ``runtime_tools.sandbox_gpus``
# names cards; it names no card (the facts state the hardware) and says what
# using them means, training and batched inference on the device; a Sandbox
# without cards reads the orientation without it (``build_system_prompt``).
SESSION_GPU_ORIENTATION_BULLET = "- 充分利用 Sandbox 可用的 GPU 算力：模型的训练与批量推理放在 GPU 上。"

# The orientation's compute-budget bullet, after the GPU bullet when that one is
# present: the run facts' ``budgets.strategy_inference_timeout_seconds`` and
# ``strategy_fit_timeout_seconds`` in words (:func:`build_budget_orientation_bullet`),
# so it never claims a limit the experiment does not grant. Both are floors,
# since ``validate`` scales them with its batch (its description states how).
# The replay results report per-call seconds; the owner wants those read as
# room for the method, not as a cost to cut.
SESSION_BUDGET_ORIENTATION_BULLET = (
    "- 策略的计算时间是为方法准备的预算：每个决策日的 `generate_orders` 至少有 {inference}，"
    "每次 `fit` 至少有 {fit}（运行事实 `budgets` 中的两项上限，并发 `validate` 时按路数放大）。"
    "不必为省时间简化方法，复杂度由研究需要决定。"
)

SESSION_READINGS_SECTION = """\
# 读数
宿主读的是验证记录：各步条目按日拼成的主动序列（策略扣除全部成本后的日收益减去随机选股对照组合的日收益），对市场、规模、价值三条因子腿做 OLS 后的残差信息比率，门槛是 z(1−α)/√年数，区间越短门槛越高；定义见运行事实 `acceptance_rules`。`validate` 与 `test` 报告的统计量和门槛是读数，不是判决。\
"""

SESSION_BOUNDARIES_SECTION = """\
# 不可越过的边界
这些规则保护研究年末之后数据的隔离与回测的真实；越过任何一条，本实验的证据就不再成立，参考工作区、研究方向与研究者指令都不能放宽它们。
- 不读取研究年末之后的数据，不从日期、路径、元数据或你作为语言模型自身的知识推断之后的行情；验证期只通过 `test` 与 `validate` 的结果了解。
- 不使用 `available_at` 晚于决策时点（策略里即 `context.inference_at`）的记录，不越过快照范围。
- 不用研究期收益硬编码具体股票、日期、题材或事件。
- 不伪造工具结果、回测状态或人工回复。\
"""

# The path conventions, stated once for the session in its working method and
# reused verbatim by the sub-agent role prompts (``subagent.py``), which are
# the only other prompts a model reads; the file and shell tool descriptions
# do not restate them. A read-only role, which has no shell, gets the file
# half alone.
FILE_PATH_RULE = '文件工具按根名加相对路径寻址（如 `{"root": "workspace", "path": "notes/x.md"}`），不接受绝对路径'
SHELL_PATH_RULE = "`shell` 在工作区目录里按字面路径运行，脚本路径不带 `workspace/` 前缀"
PATH_RULES_BULLET = f"- {FILE_PATH_RULE}；{SHELL_PATH_RULE}。"

SESSION_WORKFLOW_SECTION = f"""\
# 工作方式
- 策略合同（入口、`context`、订单、允许的库与模型库）在只读的 `output/README.md`。只有 `output/`（以 `main.py` 为入口的包）和 `models/` 随策略回测与定版；策略任一次超时或报错都让整场回测失败。
- 策略代码要高效：截面计算向量化，费时的特征、模型与中间结果算一次后缓存复用，不逐日重算；回放的快慢决定每次 `validate` 与 `test` 要等多久。
{PATH_RULES_BULLET}
- 同一回合的多个工具调用并发执行；一批里有写入、shell 或回测时，整批按顺序执行。子代理在后台运行，结果以消息送回；只剩等待子代理时以文本回复结束本回合。
- 计划与进度记在工作区根的 `TODO.md`，验证与测试节点在根 `steps` 的 Step 树（`tree.txt`）；中断后的续跑从这两处恢复。
- 技能库（`inputs/skills_index.json`）是建议，不是规则：先核实再依赖，与实测相抵触时用 `skill_feedback` 报告。环境、工具、数据或文档里的可疑缺陷用 `report_issue` 报告，然后继续。\
"""

SESSION_STATIC_SECTIONS = (
    SESSION_PREAMBLE,
    SESSION_PROCESS_SECTION,
    SESSION_ORIENTATION_SECTION,
    SESSION_READINGS_SECTION,
    SESSION_BOUNDARIES_SECTION,
    SESSION_WORKFLOW_SECTION,
)
PROTOCOL_INSTRUCTION = "\n\n".join(SESSION_STATIC_SECTIONS)

SESSION_DEFAULT_INSTRUCTION = "开始研究。何时开始测试由你决定。"

SESSION_DYNAMIC_CONTEXT_HEADER = """\
# 环境
这部分由宿主为本次尝试生成，各会话不同。\
"""

# Opens the run facts in the environment block; the exporter renders the
# same constant, so the heading has one source.
EXPERIMENT_FACTS_HEADER = """\
## 运行事实
下面的 JSON 是宿主在本次尝试开始时写下的受信事实（同一份在 workspace 的 inputs/session_context.json），会话中不更新；它描述环境，不是交易证据，其中的日期与标识不是交易信号。主要的键：research_geometry（研究年、验证期、span 写法）、budgets（预算与策略容器）、artifact_contract.acceptance_rules（主动序列、统计量、登记条件）、broker_replay（资金、费用、交易规则）、runtime_tools 与 source_refs（Sandbox、读文件根的 shell 路径、数据摘要与单位引用）。\
"""

RESUME_INSTRUCTION = """\
本次是本实验研究会话的第 {attempt} 次尝试：上一次尝试在 {interrupted_at} 中断（{error}）。已用预算：推理 {used_minutes:g}/{total_minutes:g} 分钟、模型调用 {used_calls}/{total_calls}、API 花费 ${used_usd:.2f}/${total_usd:.2f}、随机对照 {used_nulls}/{total_nulls}。工作区、`output/`、`TODO.md`、skills、Step 树与验证记录都保持中断时的状态，测试终点与已测试的步数见「环境」；之前尝试的 transcript 在根 `trace` 下（{transcripts}）。{summary_note}先读 `TODO.md` 与 Step 树（根 `steps` 的 `tree.txt`），核对工作区现状后继续，不要重做已经验证过的工作。\
"""

WRAP_UP_PROMPT = """\
现在进入收尾：任一预算耗尽时会话停止，宿主用最后一次 `test` 的候选把验证期剩下的步跑完；还没测试过时，用最后一个满足登记条件的完整研究年候选。停止前你仍可使用全部工具：手上有比最后一次测试更好的候选，就用 `test` 换上它；不要再开新的研究方向。\
"""

# How each step length of ``research_geometry.step`` reads in the run section.
STEP_LENGTH_WORDS = {"quarter": "一个季度", "month": "一个月"}


def build_session_directive_section(session_directive: str) -> str:
    directive = session_directive.strip()
    if not directive:
        return ""
    return (
        "## 研究者本会话指令（用户注入）\n"
        "把它当作需要检验和细化的研究假设，而不是已验证结论；"
        "与证据或执行约束冲突时，可以调整、降级或拒绝并说明原因。\n\n"
        f"{directive}"
    )


def build_exploration_section(exploration_directive: str) -> str:
    directive = exploration_directive.strip()
    if not directive:
        return ""
    return (
        "## 本实验的研究方向（用户注入）\n"
        "方向约束研究什么，不约束怎样做。\n\n"
        f"{directive}"
    )


def build_resume_instruction(
    *,
    attempt: int,
    interrupted_at: str,
    error: str,
    has_summary: bool,
    transcripts: Sequence[str],
    used: Mapping[str, object],
    totals: Mapping[str, object],
) -> str:
    """The opening message of an attempt that continues an interrupted one."""

    summary_note = (
        "上面是中断前最近一次压缩的摘要，从它继续。"
        if has_summary
        else "中断前没有压缩摘要：从 transcript 的末尾往前读，恢复计划。"
    )
    return RESUME_INSTRUCTION.format(
        attempt=attempt,
        interrupted_at=interrupted_at or "未知时间",
        error=error or "未记录原因",
        used_minutes=round(float(used.get("inference_seconds", 0.0)) / 60.0, 1),
        total_minutes=round(float(totals.get("inference_seconds", 0.0)) / 60.0, 1),
        used_calls=int(used.get("llm_calls", 0)),
        total_calls=int(totals.get("llm_calls", 0)),
        used_usd=float(used.get("api_spend_usd", 0.0)),
        total_usd=float(totals.get("api_spend_usd", 0.0)),
        used_nulls=int(used.get("null_controls", 0)),
        total_nulls=int(totals.get("null_controls", 0)),
        transcripts="、".join(transcripts) if transcripts else "尚无",
        summary_note=summary_note,
    )


def build_run_section(experiment_facts: Mapping[str, object]) -> str:
    """This run's data view, years, steps, threshold, validate allowance and budgets in words.

    Read off the facts the prompt also carries, so the list and the JSON
    cannot disagree: ``research_geometry`` (``research_period``, ``years``
    -- the research years alone, as the facts list them --,
    ``validation_period``, ``step``, ``steps``, ``steps_tested``,
    ``test_horizon``), ``artifact_contract.acceptance_rules.significance``
    (``alpha``, ``threshold_over_one_year``, ``threshold_over_validation_period``) and
    ``budgets.validates_per_step``. Empty for facts whose geometry states no
    test horizon; a geometry that does state one must carry every other key.
    The session budgets (``budgets.deadline_seconds``, ``max_llm_calls``,
    ``max_api_spend_usd``) add a closing line with their scale per research
    interval; a budget the facts do not carry is left out of it.
    """

    geometry = experiment_facts.get("research_geometry")
    if not isinstance(geometry, Mapping) or "test_horizon" not in geometry:
        return ""
    rules = experiment_facts["artifact_contract"]["acceptance_rules"]  # type: ignore[index]
    significance = rules["significance"]
    budgets = experiment_facts["budgets"]
    allowance = budgets["validates_per_step"]  # type: ignore[index]
    step = STEP_LENGTH_WORDS[str(geometry["step"])]
    research_end = str(geometry["research_period"]).split("..")[1]
    years = len(geometry["years"])  # type: ignore[arg-type]
    tested = int(geometry["steps_tested"])  # type: ignore[call-overload]
    lines = [
        "## 本次运行",
        f"- 数据：会话挂载的数据止于研究年末 {research_end}，整个会话都不变；验证期只通过 `test` 的结果揭开。",
        f"- 研究年：{years} 个，{geometry['research_period']}。",
        f"- 验证期：{geometry['validation_period']}，共 {geometry['steps']} 步，每步{step}；本次尝试开始时已测试 {tested} 步"
        + (f"，测到 {geometry['test_horizon']}。" if tested else "。"),
        f"- 门槛：α = {significance['alpha']}；一年区间约 {significance['threshold_over_one_year']}，整个验证期约 {significance['threshold_over_validation_period']}。",
        f"- `validate`：第一次 `test` 之前、以及相邻两次 `test` 之间，各至多 {allowance} 次。",
    ]
    budget_line = _run_budget_line(budgets)  # type: ignore[arg-type]
    if budget_line:
        lines.append(budget_line)
    return "\n".join(lines)


def _run_budget_line(budgets: Mapping[str, object]) -> str:
    """The session budgets; empty when the facts carry none of the three."""

    seconds = budgets.get("deadline_seconds")
    calls = budgets.get("max_llm_calls")
    spend = budgets.get("max_api_spend_usd")
    totals: list[str] = []
    if seconds is not None:
        totals.append(f"推理时间 {chinese_duration(seconds)}")  # type: ignore[arg-type]
    if calls is not None:
        totals.append(f"模型调用 {calls} 次")
    if spend is not None:
        totals.append(f"API 花费 {spend:g} 美元")
    if not totals:
        return ""
    return f"- 预算（整个会话共用）：{'、'.join(totals)}。"


def chinese_duration(seconds: float) -> str:
    """Whole seconds in Chinese, each non-zero unit once: 5400 is "1 小时 30 分钟"."""

    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    units = ((hours, "小时"), (minutes, "分钟"), (secs, "秒"))
    return " ".join(f"{count} {unit}" for count, unit in units if count)


def build_budget_orientation_bullet(experiment_facts: Mapping[str, object]) -> str:
    """The orientation's compute-budget bullet; empty unless the facts state both strategy limits."""

    budgets = experiment_facts.get("budgets")
    if not isinstance(budgets, Mapping):
        return ""
    inference = budgets.get("strategy_inference_timeout_seconds")
    fit = budgets.get("strategy_fit_timeout_seconds")
    if inference is None or fit is None:
        return ""
    return SESSION_BUDGET_ORIENTATION_BULLET.format(
        inference=chinese_duration(inference),  # type: ignore[arg-type]
        fit=chinese_duration(fit),  # type: ignore[arg-type]
    )


def build_system_prompt(
    *,
    experiment_facts: Mapping[str, object] | None = None,
    exploration_directive: str = "",
    session_directive: str = "",
) -> str:
    # The orientation closes with the GPU bullet when the same
    # ``runtime_tools.sandbox_gpus`` the facts JSON below carries names cards,
    # then with the compute-budget bullet when the facts state both limits.
    facts = experiment_facts or {}
    closing: list[str] = []
    runtime_tools = facts.get("runtime_tools")
    gpus = runtime_tools.get("sandbox_gpus") if isinstance(runtime_tools, Mapping) else None
    if gpus and gpus != NO_SANDBOX_GPUS:
        closing.append(SESSION_GPU_ORIENTATION_BULLET)
    budget = build_budget_orientation_bullet(facts)
    if budget:
        closing.append(budget)
    protocol = PROTOCOL_INSTRUCTION.replace(
        SESSION_ORIENTATION_SECTION,
        "\n".join((SESSION_ORIENTATION_SECTION, *closing)),
        1,
    )
    context_parts: list[str] = []
    if experiment_facts:
        run = build_run_section(experiment_facts)
        if run:
            context_parts.append(run)
        context_parts.append(render_experiment_facts_section(experiment_facts))
    for section in (
        build_exploration_section(exploration_directive),
        build_session_directive_section(session_directive),
    ):
        if section:
            context_parts.append(section)
    return "\n\n".join(
        (
            protocol,
            SESSION_DYNAMIC_CONTEXT_HEADER,
            *context_parts,
        )
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
        f"{EXPERIMENT_FACTS_HEADER}\n\n"
        "```json\n"
        f"{payload}\n"
        "```"
    )
