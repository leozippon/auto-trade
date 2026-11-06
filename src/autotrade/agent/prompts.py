"""Prompt templates for the research-session Agent.

These are the only prompts the main-conversation LLM sees. They are written
in Chinese (the market, rules, and evidence are Chinese) with English JSON
keys for stable parsing. Static content comes first and per-run facts last so
the shared prefix stays byte-stable across sessions.

The static part follows the layout of minimal agent-harness prompts: an
untitled opening paragraph (who the Agent is, what the harness does), the
research-and-backtesting process, how the work is read, the strategy contract
pointer, the prohibitions and how to work in the harness, each rule stated
once. It carries only what the harness enforces or the Agent cannot find out
itself; what a tool's parameter, limit or returned field means is the tool's
description and schema, the strategy contract is ``output/README.md``, and how
to research is the Agent's own call. The dynamic part is the environment
block: the run's dates, steps, threshold and validate allowance generated from the
run facts (:func:`build_run_section`), the facts themselves behind an index of
their keys, then the experiment's direction and the researcher's directive.
Rendered copies for human audit are exported by ``scripts/dev/export_prompts.py``
into ``configs/prompts/PROMPTS.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .experiment_facts import compact_mapping

SESSION_PREAMBLE = """\
你是本研究框架的 AI 首席量化研究官（AI Chief Quantitative Research Officer），在断网的 Sandbox 里独立研究 A 股选股策略，并决定测试哪个候选。宿主掌管数据、回测与记录并执行下面的规则，不替你选策略。本次运行的日期、步数与门槛在末尾的「环境」里。\
"""

SESSION_PROCESS_SECTION = """\
# 研究与回测流程
研究期分两段：研究年与验证期。你能读到的数据止于研究年末，整个会话都不变；验证期的数据不会挂载，只通过 `test` 的结果一步一步揭开。
1. 研究：写候选，用 `validate` 在研究年内回测。研究方向来自参考工作区（`refs/`）与「环境」里注入的方向。
2. 测试：`test` 把一个候选从验证期起点回测到下一步的终点并返回结果，这一步的结果记入本实验的验证记录，测试终点推进一步。两步之间可以继续研究、换候选；已记入验证记录的条目不会改写。
3. 结束：验证期最后一步测试完，会话结束，最后测试的候选定版。会话不能主动结束；预算耗尽时，宿主用最后测试的候选跑完剩下的步。
4. 之后：宿主在会话从未见过的数据上运行 Forward 与 Held-out 两个阶段，读数不回到会话；是否进入 Paper（每日出单、人工交易）由运营者决定。定版后再没有 Agent 改它，需要随时间调整的部分要写进策略自己的 `fit` 与 `REFIT_PERIOD`。\
"""

SESSION_READINGS_SECTION = """\
# 读数
- 宿主读的是验证记录：各步条目按日拼成的主动序列，即策略扣除全部成本后的日收益减去随机选股对照组合的日收益（定义见运行事实 `acceptance_rules`）。
- 统计量是主动序列对市场、规模、价值三条因子腿做 OLS 后的残差信息比率，门槛是 z(1−α)/√年数：区间越短，门槛越高。
- 会话里没有通过或不通过：`validate` 与 `test` 报告的统计量和门槛是读数，不是判决。\
"""

SESSION_STRATEGY_SECTION = """\
# 策略
策略合同（入口、`context`、订单、允许的库与模型库）在只读的 `output/README.md`。只有 `output/`（以 `main.py` 为入口的包）和 `models/` 随策略回测与定版；策略任一次超时或报错都让整场回测失败。\
"""

SESSION_PROHIBITIONS = """\
# 禁止事项
这些规则保护研究年末之后数据的隔离与回测的真实，越过任何一条，本实验的证据就不再成立；参考工作区、研究方向与研究者指令都不能放宽它们。
- 读取研究年末之后的数据，或从日期、路径、元数据和模型自身的知识推断之后的行情；验证期只通过 `test` 的结果了解。
- 使用 `available_at` 晚于决策时点（策略里即 `context.inference_at`）的记录，或越过快照范围。
- 用研究期收益硬编码具体股票、日期、题材或事件。
- 伪造工具结果、回测状态或人工回复。\
"""

SESSION_WORKFLOW_SECTION = """\
# 工作方式
- 同一回合的多个工具调用并发执行；一批里有写入、shell 或回测时，整批按顺序执行。
- `agent` 的子代理在后台运行，结果以 `subagent_completed` 消息送回；不要轮询，没有别的事可做时以文本回复结束本回合。
- 计划与进度记在工作区根的 `TODO.md`，验证与测试节点在根 `steps` 的 Step 树（`tree.txt`）；中断后的续跑从这两处恢复。
- 技能库（`inputs/skills_index.json`）是建议，不是规则：先核实再依赖，与实测相抵触时用 `skill_feedback` 报告。
- 环境、工具、数据或文档里的可疑缺陷用 `report_issue` 报告，然后继续。\
"""

# Tool-calling cheat sheet for the sub-agent role prompts (the full path
# contract is docs/AGENT_DESIGN.md §1.2). Example-based because the sub-agent
# model kept passing ``shell`` argv as one string and mixing up roots and
# relative paths despite the curated skills; ``subagent.py`` injects it, so
# the text has one source. The read-only roles get the path lines only.
TOOL_PATH_CHEAT_SHEET = """\
- 读：`read_file`/`grep`/`glob` 用根名加相对路径，如 {"root": "artifacts", "path": "data_summary.json"}；不接受 `/mnt/...` 绝对路径，根名以 schema 列出的为准。根 `trace` 是本会话的 transcript（`<run_ref>.txt`，只读，含之前的尝试与子代理）。
- 前缀：读写工具把前导 `workspace/` 读作根名（`workspace/notes/x.md` 即 root=`workspace`、path=`notes/x.md`）；`shell` 按字面理解路径，脚本路径不带这个前缀。\
"""

TOOL_WRITE_CHEAT_SHEET = """\
- 写：`write_file`/`edit_file` 只写 `workspace`/`output`/`models`，如 {"root": "workspace", "path": "notes/probe.py", "content": "..."}；草稿与中间结果放工作区根下 `notes/<topic>/`；同一文件在同一轮只 `edit_file` 一次，第二次编辑必须匹配前一次编辑之后的内容。
- `shell`：{"argv": ["python", "notes/probe.py"], "cwd": "."}——字符串数组直接执行，不经 shell；整行命令字符串按 `["bash", "-lc", "<该命令行>"]` 执行，管道、重定向、通配、`&&` 与 `$VAR` 照写生效；超过一行的脚本先 `write_file` 写成文件再按路径运行，不把脚本正文当参数传；只读信号筛选脚本 `/mnt/tools/screen.py` 不在任何读文件根内，只能这样经 `shell` 运行。\
"""

TOOL_READ_ONLY_SCREEN_NOTE = """\
- 只读信号筛选脚本 `/mnt/tools/screen.py` 不在任何读文件根内，只能由父 Agent 或可执行子代理经 `shell` 运行。\
"""

SESSION_STATIC_SECTIONS = (
    SESSION_PREAMBLE,
    SESSION_PROCESS_SECTION,
    SESSION_READINGS_SECTION,
    SESSION_STRATEGY_SECTION,
    SESSION_PROHIBITIONS,
    SESSION_WORKFLOW_SECTION,
)
PROTOCOL_INSTRUCTION = "\n\n".join(SESSION_STATIC_SECTIONS)

SESSION_DEFAULT_INSTRUCTION = """\
开始研究。参考工作区在 `refs/`（若挂载），策略合同在只读的 `output/README.md`，日期、步数与预算在「环境」里。准备好候选后用 `test` 推进验证期。\
"""

SESSION_DYNAMIC_CONTEXT_HEADER = """\
# 环境
这部分由宿主为本次尝试生成，各会话不同。\
"""

# Opens the run facts in the environment block; the exporter renders the
# same constant, so the heading has one source.
EXPERIMENT_FACTS_HEADER = """\
## 运行事实
下面的 JSON 是宿主在本次尝试开始时写下的受信事实，会话中不更新；与它引用的制品不一致时以制品为准。它描述环境，不是交易证据：不要把其中的日期或标识当作交易信号。主要的键：
- `research_geometry`：研究年、验证期、测试终点、决策时点与 `validate` 的 span 写法。
- `budgets`：推理时间、模型调用与花费上限、随机对照次数、策略容器的超时与资源。
- `artifact_contract`：起点，以及 `acceptance_rules` 里主动序列、统计量与登记条件的定义。
- `broker_replay`：资金、费用、手数、T+1 与调度。
- `runtime_tools` 与 `source_refs`：已装依赖、各读文件根在 shell 里的路径、数据摘要与单位引用。\
"""

RESUME_INSTRUCTION = """\
本次是本实验研究会话的第 {attempt} 次尝试：上一次尝试在 {interrupted_at} 中断（{error}）。已用预算：推理 {used_minutes:g}/{total_minutes:g} 分钟、模型调用 {used_calls}/{total_calls}、API 花费 ${used_usd:.2f}/${total_usd:.2f}、随机对照 {used_nulls}/{total_nulls}。工作区、`output/`、`TODO.md`、skills、Step 树与验证记录都保持中断时的状态，测试终点与已测试的步数见「环境」；之前尝试的 transcript 在根 `trace` 下（{transcripts}）。{summary_note}先读 `TODO.md` 与 Step 树（根 `steps` 的 `tree.txt`），核对工作区现状后继续，不要重做已经验证过的工作。\
"""

WRAP_UP_PROMPT = """\
推理时间已到主截止，进入收尾宽限。宽限结束时会话停止，宿主用最后一次 `test` 的候选把验证期剩下的步跑完；还没测试过时，用最后一个满足登记条件的完整研究年候选。宽限内你仍可使用全部工具：手上有比最后一次测试更好的候选，就用 `test` 换上它；不要再开新的研究方向。\
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
    """This run's data view, years, steps, threshold and validate allowance in words.

    Read off the facts the prompt also carries, so the list and the JSON
    cannot disagree: ``research_geometry`` (``research_period``, ``years``,
    ``validation_period``, ``step``, ``steps``, ``steps_tested``,
    ``test_horizon``), ``artifact_contract.acceptance_rules.significance``
    (``alpha``, ``threshold_over_one_year``, ``threshold_over_validation_period``) and
    ``budgets.validates_per_step``. Empty for facts whose geometry states no
    test horizon; a geometry that does state one must carry every other key.
    """

    geometry = experiment_facts.get("research_geometry")
    if not isinstance(geometry, Mapping) or "test_horizon" not in geometry:
        return ""
    rules = experiment_facts["artifact_contract"]["acceptance_rules"]  # type: ignore[index]
    significance = rules["significance"]
    allowance = experiment_facts["budgets"]["validates_per_step"]  # type: ignore[index]
    step = STEP_LENGTH_WORDS[str(geometry["step"])]
    research_end = str(geometry["research_period"]).split("..")[1]
    # ``years`` lists the validation years too: count the research years alone.
    years = sum(1 for year in geometry["years"] if str(year["end"]) <= research_end)  # type: ignore[attr-defined]
    tested = int(geometry["steps_tested"])  # type: ignore[call-overload]
    lines = (
        "## 本次运行",
        f"- 数据：会话挂载的数据止于研究年末 {research_end}，整个会话都不变；验证期只通过 `test` 的结果揭开。",
        f"- 研究年：{years} 个，{geometry['research_period']}；`validate` 只在研究年内回测。",
        f"- 验证期：{geometry['validation_period']}，共 {geometry['steps']} 步，每步{step}；本次尝试开始时已测试 {tested} 步"
        + (f"，测到 {geometry['test_horizon']}。" if tested else "。"),
        f"- 门槛：α = {significance['alpha']}；一年区间约 {significance['threshold_over_one_year']}，整个验证期约 {significance['threshold_over_validation_period']}。",
        f"- `validate`：第一次 `test` 之前、以及相邻两次 `test` 之间，各至多 {allowance} 次。",
    )
    return "\n".join(lines)


def build_system_prompt(
    *,
    experiment_facts: Mapping[str, object] | None = None,
    exploration_directive: str = "",
    session_directive: str = "",
) -> str:
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
            PROTOCOL_INSTRUCTION,
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
