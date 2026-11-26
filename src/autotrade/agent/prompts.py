"""Prompt templates for the research-session Agent.

These are the only prompts the main-conversation LLM sees. They are written
in Chinese (the market, rules, and evidence are Chinese) with English JSON
keys for stable parsing. Static content comes first and per-run facts last so
the shared prefix stays byte-stable across sessions; the only fact-dependent
lines above the environment block are the orientation's closing bullets: the
GPU bullet, present only when the Sandbox has cards and naming none of them,
and the timeout bullet, which states the experiment's two strategy time
limits, so every attempt of an experiment shares the whole prefix.

The static part follows the layout of minimal agent-harness prompts (Pi,
Claude Code): an untitled opening paragraph (who the Agent is, what the
harness does), how the session advances, the research orientation, how the
work is read, the boundaries no instruction relaxes and how to work in the
harness, each rule stated once. It carries what the harness enforces, what
the Agent cannot find out itself, and the owner's research orientation (the
ambition and depth expected, research through the validation period as well
as the research years, the strategy timeouts as guards rather than allowances,
replay wall clock as no reason to shrink a model and, on a Sandbox with cards,
what the cards are for; not a method). The working method points where the
rest lives instead of restating it: what a tool's parameter, limit or
returned field means is the tool's description and schema, every
strategy-side mechanic (what persists between calls, days and replays
included) is ``output/README.md``, measured facts are the skills, the run's
values are the facts JSON, and how to research within that orientation is the
Agent's own call. The dynamic part is the environment block: the run's dates,
steps, threshold, validate allowance, session budgets and workspace storage
allowance generated from the run facts (:func:`build_run_section`), the facts
themselves as compact JSON, then the experiment's direction and the
researcher's directive.
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
- 研究：写候选，用 `validate` 回测。方向来自参考工作区（`refs/`）与「环境」里注入的方向。
- 测试：`test` 把一个候选从验证期起点回测到下一步终点，这一步的结果记入本实验的验证记录，测试终点前进一步，`validate` 的次数重新计。
- 两次 `test` 之间是一段研究：已揭开的步是样本外读数。据此改进现有候选或开发全新的策略，让每一步都由当时最好的候选记入；按这些读数选择、更换候选是正当的：已记入记录的条目不改写，定版后还有 Forward 与 Held-out 检验。
- 结束：最后一步测完会话结束，最后测试的候选定版。会话不能主动结束；预算耗尽时宿主用最后测试的候选跑完剩余的步，还没测试过时用最后一个满足登记条件的完整研究年候选，没有则不定版。
- 之后：宿主在会话从未见过的数据上运行 Forward 与 Held-out，读数不回到会话；是否进入 Paper 由运营者决定。定版后再没有 Agent 改它，需要随时间调整的部分写进策略自己的 `fit` 与 `REFIT_PERIOD`。\
"""

SESSION_ORIENTATION_SECTION = """\
# 研究取向
- 目标是有创新性、有深度的策略，而不是教科书式的风格因子组合：优先探索学习型模型、预训练模型与嵌入、财务与公告标题等多个数据域、自适应重训与组合构建等复杂方法；简单的规则打分只作基线对照，不作为目标。
- 预算是为深入研究准备的，规模与计法见「本次运行」。\
"""

# The orientation's GPU bullet when the run facts' ``runtime_tools.sandbox_gpus``
# names cards; it names no card (the facts state the hardware). It says what the
# cards are for and what counts as using them, and that a method's scale is
# bounded by card memory and the per-call timeouts, not by replay speed (sessions
# built sub-0.5 GiB networks to keep replays short); the measured costs are the
# curated ``pretrained-costs`` skill. A Sandbox without cards reads the
# orientation without it (``build_system_prompt``).
SESSION_GPU_ORIENTATION_BULLET = (
    "- GPU 留给需要它的方法：在全市场、长历史上训练的序列或注意力模型，以及用模型库里的预训练语言与嵌入模型"
    "对文本编码、打分或微调（路径与加载见 `output/README.md` 的 Pretrained models，实测开销见技能 `pretrained-costs`）。"
    "只占几百 MiB 显存的小网络不算用上了卡；方法的规模受卡的显存与单次调用超时约束，不受回放快慢约束。"
)

# The orientation's strategy-timeout bullet, after the GPU bullet when that one is
# present: the run facts' ``budgets.strategy_inference_timeout_seconds`` and
# ``strategy_fit_timeout_seconds`` in words (:func:`build_budget_orientation_bullet`),
# so it never claims a limit the experiment does not grant. Both are floors,
# since ``validate`` scales them with its batch (its description states how).
# They are timeouts against a runaway call: sessions read them as a daily
# allowance and priced a daily fit at 1,701 times the limit, so the bullet says
# they are neither an allowance nor a target.
SESSION_BUDGET_ORIENTATION_BULLET = (
    "- 策略单次调用的超时：每个决策日的 `generate_orders` 至少 {inference}，每次 `fit` 至少 {fit}"
    "（`budgets` 的两项）。它们只拦下失控的调用，不是每天的额度或目标；"
    "不必为省时间简化方法，复杂度由研究需要决定。"
)

SESSION_READINGS_SECTION = """\
# 读数
宿主读的是验证记录：各步条目按日拼成的主动序列（策略扣除全部成本后的日收益减去随机选股对照组合的日收益），对市场、规模、价值、低换手四条因子腿做 OLS 后的残差信息比率，门槛是 z(1−α)/√年数，区间越短门槛越高；定义见运行事实 `acceptance_rules`。`validate` 与 `test` 报告的统计量和门槛是读数，不是判决。\
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

# The working method points instead of restating: the contract README owns
# every strategy-side mechanic, the skills the measured facts. Its second
# bullet keeps the one orientation clause about replay speed (sessions shrank
# their models to shorten replays no budget charges) and points, prominently,
# at what persists between calls, days and replays, which the owner asked the
# sessions to be made aware of (2026-11-16).
SESSION_WORKFLOW_SECTION = f"""\
# 工作方式
- 策略合同（入口、包与 `models/`、限额、`context`、订单、允许的库与模型库）在只读的 `output/README.md`。
- 策略代码要高效，但回放的墙钟不计入任何会话预算，也不是缩小模型的理由。调用之间、交易日之间与各场回放之间留下什么，怎样不重建数据、不重载权重，见 `output/README.md` 的 State across calls and days。
{PATH_RULES_BULLET}
- 同一回合的多个工具调用并发执行；一批里有写入、shell 或回测时，整批按顺序执行。子代理在后台运行，结果以消息送回；只剩等待子代理时以文本回复结束本回合。
- 计划与进度记在工作区根的 `TODO.md`，验证与测试节点在根 `steps` 的 Step 树（`tree.txt`）；中断后的续跑从这两处恢复。
- 技能库（`inputs/skills_index.json`）是建议，不是规则：先核实再依赖，与实测相抵触时用 `skill_feedback` 报告；环境、工具、数据或文档里的可疑缺陷用 `report_issue` 报告后继续。\
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
# same constant, so the heading has one source. The JSON shows its own keys,
# so the header names none of them.
EXPERIMENT_FACTS_HEADER = """\
## 运行事实
下面的 JSON 是宿主在本次尝试开始时写下的受信事实（同 `inputs/session_context.json`），会话中不更新；它描述环境，不是交易证据，其中的日期与标识不是交易信号。\
"""

# Labels the reference Pack's README where the host restates it
# (``compact.reference_pack_message``): the Pack's requirements live only
# there and a session reads it at its start, but an arm's second summary
# dropped its Pack's required controls (round 20261119), and sessions told to
# re-read it did not, one last at hour 2 of 52 across three compactions (round
# 20261121). So the first turn after every compaction of the session and the
# first turn of a resumed attempt carry the text itself, from the host's copy.
REFERENCE_PACK_NOTE = (
    "宿主重述的参考包 README：每次压缩之后与续跑开始时，宿主把本实验参考包 `refs/README.md` 的原文"
    "（取自宿主保存的原件，不是工作区里的副本）放在 readme 里，下一次压缩时换上新的一份，不必为此重读。"
)

RESUME_INSTRUCTION = """\
本次续接本实验中断的研究会话：上一次尝试在 {interrupted_at} 中断（{error}）。已用预算：推理 {used_minutes:g}/{total_minutes:g} 分钟、模型调用 {used_calls}/{total_calls}、API 花费 ${used_usd:.2f}/${total_usd:.2f}、随机对照 {used_nulls}/{total_nulls}。工作区、`output/`、`TODO.md`、skills、Step 树与验证记录都保持中断时的状态；之前尝试的 transcript 在根 `trace` 下（{transcripts}）。{summary_note}先读 `TODO.md` 与 Step 树（根 `steps` 的 `tree.txt`），核对工作区现状后继续，不要重做已经验证过的工作。{record_note}\
"""

WRAP_UP_PROMPT = """\
现在进入收尾：任一预算耗尽时会话停止，宿主用最后一次 `test` 的候选把验证期剩下的步跑完；还没测试过时，用最后一个满足登记条件的完整研究年候选，没有则不定版。停止前你仍可使用全部工具：手上有比最后一次测试更好的候选，就用 `test` 换上它；不要再开新的研究方向。\
"""

# How each step length of ``research_geometry.step`` reads in the run section.
STEP_LENGTH_WORDS = {"quarter": "一个季度", "month": "一个月"}

# Used fractions of each session budget -- the inference time, the API spend
# and the model calls -- at which the runner injects one ``budget_notice``
# observation stating what is left of that budget and how many backtests have
# run so far. Defined here because the run section states them; the runner
# reads the same tuple.
BUDGET_NOTICE_FRACTIONS = (0.5, 0.75, 0.9)


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
    interrupted_at: str,
    error: str,
    summary_at: str | None,
    transcripts: Sequence[str],
    used: Mapping[str, object],
    totals: Mapping[str, object],
    record: Mapping[str, object] | None,
) -> str:
    """The opening message of an attempt that continues an interrupted one.

    ``summary_at`` is when the carried summary was written (None: there is
    none), and ``record`` the validation record as a test result returns it
    (``session_tools.SessionValidations.record_reading``; None: no step is
    tested). The summary may predate tests the record holds, so the note
    says which one is authoritative.
    """

    summary_note = (
        f"上面是中断前最近一次压缩的摘要，写于 {summary_at}；此后到中断的工作只在 transcript 里。"
        if summary_at is not None
        else "中断前没有压缩摘要：从 transcript 的末尾往前读，恢复计划。"
    )
    record_note = (
        "验证期还没有测试过任何一步。"
        if record is None
        else "\n\n验证记录（同 `test` 返回的 `record` 与 `significance`；摘要或 `TODO.md` 与它不符时以它为准）：\n"
        + json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    )
    return RESUME_INSTRUCTION.format(
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
        record_note=record_note,
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
    ``max_api_spend_usd``) add a line with their totals, how inference time is
    counted and when the host reports use (:func:`_run_budget_line`); a budget
    the facts do not carry is left out of it. ``budgets.workspace_storage_bytes``
    adds the closing storage line (:func:`_run_storage_line`).
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
    for line in (_run_budget_line(budgets), _run_storage_line(budgets)):  # type: ignore[arg-type]
        if line:
            lines.append(line)
    return "\n".join(lines)


def _run_budget_line(budgets: Mapping[str, object]) -> str:
    """The session budgets, how inference time is counted, that there is no
    token budget, and when the runner reports use; empty when the facts carry
    none of the three. Sessions with totals only invented their own usage
    ("27k tokens used out of 32k", "28 hours with 55 % remaining")."""

    seconds = budgets.get("deadline_seconds")
    calls = budgets.get("max_llm_calls")
    spend = budgets.get("max_api_spend_usd")
    totals: list[str] = []
    if seconds is not None:
        # The clock pauses only inside smoke_backtest, validate, test and
        # run_null_control (``InferenceTimeBudget.pause`` in session_tools).
        totals.append(
            f"推理时间 {chinese_duration(seconds)}（会话的墙钟时间，回放与随机对照的等待不计）"  # type: ignore[arg-type]
        )
    if calls is not None:
        totals.append(f"模型调用 {calls} 次")
    if spend is not None:
        totals.append(f"API 花费 {spend:g} 美元")
    if not totals:
        return ""
    fractions = "、".join(f"{fraction:.0%}" for fraction in BUDGET_NOTICE_FRACTIONS)
    return (
        f"- 预算（整个会话共用）：{'、'.join(totals)}；没有 token 预算，上下文靠压缩（`compact`）管理。"
        f"每项用到 {fractions} 时宿主各提醒一次（`budget_notice`，或最后一档的收尾提示），没收到提醒就是还没用到 {BUDGET_NOTICE_FRACTIONS[0]:.0%}。"
    )


def _run_storage_line(budgets: Mapping[str, object]) -> str:
    """The workspace storage allowance, what counts toward it and that the
    session keeps to it itself; empty when the facts carry none. Without it
    sessions filled the shared storage with derived arrays until writes
    failed, and ``df`` there shows the shared filesystem, not the quota."""

    allowance = budgets.get("workspace_storage_bytes")
    if allowance is None:
        return ""
    return (
        f"- 工作区存储：额度 {int(allowance) / 1024**3:g} GiB，工作区里的全部文件都算，"  # type: ignore[call-overload]
        "缓存、中间数组与留下的模型文件在内（用 `du -sh` 量）；宿主不检查，由你守住，不再需要的文件及时删。"
    )


def chinese_duration(seconds: float) -> str:
    """Whole seconds in Chinese, each non-zero unit once: 5400 is "1 小时 30 分钟"."""

    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    units = ((hours, "小时"), (minutes, "分钟"), (secs, "秒"))
    return " ".join(f"{count} {unit}" for count, unit in units if count)


def build_budget_orientation_bullet(experiment_facts: Mapping[str, object]) -> str:
    """The orientation's strategy-timeout bullet; empty unless the facts state both strategy limits."""

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
    # then with the strategy-timeout bullet when the facts state both limits.
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
    # Compact: indentation cost a quarter of the facts' tokens on every call.
    payload = json.dumps(
        compact_mapping(experiment_facts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return (
        f"{EXPERIMENT_FACTS_HEADER}\n\n"
        "```json\n"
        f"{payload}\n"
        "```"
    )
