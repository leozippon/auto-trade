"""The Agent experiences a research session, not a Fold.

Every surface a session model reads -- the system prompt with its dynamic
half, the opening message, the wrap-up and hard-finalization prompts, the
sub-agent role prompts and every tool description -- carries the research
workflow and none of the retired Fold/Meta vocabulary.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from autotrade.agent import prompts
from autotrade.agent.subagent import (
    AGENT_TOOL_DESCRIPTION,
    SUBAGENT_ROLES,
    subagent_system_prompt,
)

# Words of the Fold/Meta workflow a research session must never read. ASCII
# words match on letter boundaries, so ``fold_id`` and ``fold_ref_`` count
# while ``scaffold`` does not; ``phase`` also stops at an underscore, since a
# replay's ``phase_seconds`` timing is not the retired stage vocabulary.
RETIRED_WORDS = (
    "Fold",
    "fold",
    "Meta",
    "meta_learning",
    "Epoch",
    "parent_control",
    "parent-control",
    "confirmation",
    "anchor",
    "deployment",
    "transition",
    "phase",
    "父本",
    "父产物",
    "确认折",
    "锚点",
    "baseline_anchor",
    "no_update",
    "baseline_missing",
    "vs_parent",
    "部署调整",
    "探索期",
    "收敛期",
    "Test",
    "新季度",
    "Development",
    "finish_fold",
    "finish_meta",
    "terminate",
)


def retired_vocabulary(text: str) -> list[str]:
    """The retired words ``text`` carries."""

    found = []
    for word in RETIRED_WORDS:
        if not word.isascii():
            hit = word in text
        else:
            right = "A-Za-z_" if word == "phase" else "A-Za-z"
            hit = re.search(rf"(?<![A-Za-z_]){re.escape(word)}(?![{right}])", text) is not None
        if hit:
            found.append(word)
    return found


def _full_prompt() -> str:
    return prompts.build_system_prompt(
        experiment_facts={"identity": {"session_kind": "research"}},
        exploration_directive="长期方向",
        session_directive="本会话指令",
    )


def _tool_descriptions() -> dict[str, str]:
    from autotrade.environment.tools import (
        CompactTool,
        EditFileTool,
        GlobTool,
        GrepTool,
        ModificationCheckTool,
        ReadFileTool,
        ReportIssueTool,
        SafeWorkspace,
        SandboxShellTool,
        SearchRoots,
        SkillFeedbackTool,
        StepRollbackTool,
        WriteFileTool,
    )
    from autotrade.pipelines.session_tools import (
        NullControlTool,
        SmokeBacktestTool,
        TestTool,
        ValidateTool,
    )
    from autotrade.pipelines.skills import DeleteSkillTool, WriteSkillTool

    with tempfile.TemporaryDirectory() as tmp:
        workspace = SafeWorkspace(Path(tmp))
        roots = SearchRoots(workspace)
        specs = [
            tool.spec
            for tool in (
                GlobTool(roots),
                GrepTool(roots),
                ReadFileTool(roots),
                SandboxShellTool(workspace, object()),  # type: ignore[arg-type]
            )
        ]
    specs.extend(
        cls.spec
        for cls in (
            CompactTool,
            EditFileTool,
            WriteFileTool,
            ModificationCheckTool,
            StepRollbackTool,
            ReportIssueTool,
            SkillFeedbackTool,
            ValidateTool,
            TestTool,
            SmokeBacktestTool,
            WriteSkillTool,
            DeleteSkillTool,
        )
    )
    # The null control's description is built per session around its cap.
    specs.append(NullControlTool(object(), max_calls=3).spec)  # type: ignore[arg-type]
    return {
        spec.name: f"{spec.description} {spec.input_schema}" for spec in specs
    } | {"agent": AGENT_TOOL_DESCRIPTION}


@pytest.mark.parametrize(
    "surface",
    [
        "system_prompt",
        "default_instruction",
        "wrap_up",
        *(f"subagent:{role}" for role in SUBAGENT_ROLES),
    ],
)
def test_no_session_surface_carries_fold_or_meta_vocabulary(surface: str) -> None:
    text = {
        "system_prompt": _full_prompt(),
        "default_instruction": prompts.SESSION_DEFAULT_INSTRUCTION,
        "wrap_up": prompts.WRAP_UP_PROMPT,
    }.get(surface) or subagent_system_prompt(surface.split(":", 1)[1])
    assert retired_vocabulary(text) == [], surface


def test_no_tool_description_carries_fold_or_meta_vocabulary() -> None:
    for name, text in _tool_descriptions().items():
        assert retired_vocabulary(text) == [], name


def test_the_tools_state_the_validate_allowance_the_prompt_states() -> None:
    """The replay-year budget is retired: no tool text charges one, and the
    budget validate and smoke_backtest name is the one the run section
    states, validates_per_step calls before the first test and between tests."""

    descriptions = _tool_descriptions()
    for name, text in descriptions.items():
        assert not re.search(r"replay[-_ ]years?", text, re.IGNORECASE), name
    for name in ("validate", "smoke_backtest"):
        assert "acceptance_rules.validation.validates_per_step" in descriptions[name], name
    assert "before the first test and between two tests" in descriptions["validate"]
    run = prompts.build_run_section(_run_facts())
    assert "- `validate`：第一次 `test` 之前、以及相邻两次 `test` 之间，各至多 6 次。" in run
    # A control is a comparison leg test never takes.
    assert "test never takes it" in descriptions["validate"]


# The static part's headings, in the order the prompt reads them (Pi and
# Claude Code's layout: who, how the session advances, the orientation, how
# the work is read, the hard boundaries, how to work).
STATIC_HEADINGS = (
    "# 会话怎样推进",
    "# 研究取向",
    "# 读数",
    "# 不可越过的边界",
    "# 工作方式",
)


def test_the_static_prompt_reads_in_its_sections_without_the_retired_test_gate() -> None:
    """The owner wants research to go on through the validation period: the
    orientation no longer asks for directions compared before the first test,
    and that wording cannot return."""

    stable = _full_prompt().split("# 环境")[0]
    assert stable.startswith("你是本研究框架的 AI 首席量化研究官")
    places = [stable.index(f"\n{heading}\n") for heading in STATIC_HEADINGS]
    assert places == sorted(places)
    assert [line for line in stable.splitlines() if line.startswith("# ")] == list(STATIC_HEADINGS)
    for retired in ("之后再开始测试", "充分比较多个方向", "# 研究与回测流程", "# 禁止事项", "# 策略"):
        assert retired not in stable, retired


def test_the_prompt_states_the_research_and_backtesting_process() -> None:
    """The static part says who the Agent is, how the session advances and
    the rules that matter, each once; the numbers are the run section's."""

    prompt = _full_prompt()
    stable = prompt.split("# 环境")[0]
    process = stable[stable.index("# 会话怎样推进") : stable.index("# 研究取向")]
    orientation = stable[stable.index("# 研究取向") : stable.index("# 读数")]
    readings = stable[stable.index("# 读数") : stable.index("# 不可越过的边界")]
    boundaries = stable[stable.index("# 不可越过的边界") : stable.index("# 工作方式")]
    workflow = stable[stable.index("# 工作方式") :]
    assert "AI 首席量化研究官（AI Chief Quantitative Research Officer）" in stable
    # Research on the visible years, whose data view never moves; each test
    # extends the backtest one step, reveals it through its results and
    # restarts the validate allowance; research goes on between tests; then
    # the host-run Forward and Held-out stages, then Paper.
    for clause in (
        "你能读到的数据止于研究年末，整个会话不变",
        "验证期的数据不挂载，只随每次 `test` 的结果揭开一步",
        "用 `validate` 在研究年内回测",
        "`test` 把一个候选从验证期起点回测到下一步终点，这一步的结果记入本实验的验证记录",
        "测试终点前进一步，`validate` 的次数重新计",
        "两次 `test` 之间是一段研究：用刚揭开的区间检验候选的分布外泛化",
        "新候选在已揭开区间上的表现随之可见",
        "改进现有候选或开发全新的策略，在研究年上检验，让每一步都由当时最好的候选记入",
        "换候选只花一次回放；已记入记录的条目不改写",
        "最后一步测完会话结束，最后测试的候选定版",
        "会话不能主动结束",
        "宿主用最后测试的候选跑完剩余的步",
        "运行 Forward 与 Held-out，读数不回到会话",
        "是否进入 Paper 由运营者决定",
        "`fit` 与 `REFIT_PERIOD`",
    ):
        assert clause in process, clause
    # The host reads the record; the statistic and its threshold; readings,
    # not verdicts.
    for clause in (
        "宿主读的是验证记录",
        "随机选股对照组合",
        "残差信息比率",
        "z(1−α)/√年数",
        "是读数，不是判决",
    ):
        assert clause in readings, clause
    # The owner's orientation: innovative, deep strategies over textbook style
    # factors, with rule scoring as the baseline, and the budget, whose scale
    # the run section states, spent on research through the research years
    # and the validation period alike.
    for clause in (
        "有创新性、有深度的策略，而不是教科书式的风格因子组合",
        "学习型模型、预训练模型与嵌入",
        "简单的规则打分只作基线对照，不作为目标",
        "规模见「本次运行」：研究年与验证期的每一步都是研究的时间，回放的等待不计入推理时间",
        "预算大半未用就测完最后一步，说明研究做得太浅",
    ):
        assert clause in orientation, clause
    assert "参考工作区、研究方向与研究者指令都不能放宽它们" in boundaries
    # The Agent's own knowledge as a language model, not a pretrained model a
    # strategy loads, is what the boundary rules out.
    assert "不读取研究年末之后的数据" in boundaries and "你作为语言模型自身的知识" in boundaries
    assert "验证期只通过 `test` 的结果了解" in boundaries
    # The working method points at the contract, asks for fast strategy code
    # in one line (the owner's guidance, 2026-11-09) and states the path
    # conventions once, for every file tool and the shell.
    assert "只读的 `output/README.md`" in workflow
    assert "策略代码要高效：截面计算向量化，费时的特征、模型与中间结果算一次后缓存复用，不逐日重算" in workflow
    assert "回放的快慢决定每次 `validate` 与 `test` 要等多久" in workflow
    assert '`{"root": "workspace", "path": "notes/x.md"}`' in workflow
    assert "脚本路径不带 `workspace/` 前缀" in workflow
    assert "## 本实验的研究方向（用户注入）\n方向约束研究什么，不约束怎样做" in prompt
    # The retired action set and its wording are gone from every surface the
    # session reads under this contract: no registration or finishing action, no
    # "no excess" outcome, no pass/fail verdict, no replay-year budget.
    surfaces = {
        "system_prompt": prompt,
        "default_instruction": prompts.SESSION_DEFAULT_INSTRUCTION,
        "wrap_up": prompts.WRAP_UP_PROMPT,
        "resume": prompts.RESUME_INSTRUCTION,
    }
    for name, text in surfaces.items():
        for retired in (
            "freeze",
            "finish_session",
            "冻结",
            "no_edge",
            "未发现超额",
            "没有超额",
            "deadline",
            "验收",
            "即通过",
            "封存",
            "sealed",
            "回测年",
            "replay_year",
            "batch_validate",
            "step_rollback",
        ):
            assert retired not in text, (name, retired)
    # Nothing of the retired deflation, its trial count or its incubation
    # route survives anywhere the session reads.
    for retired in (
        "offline_trials",
        "information_ratio_bar",
        "selection_statistics",
        "去膨胀",
        "试验数",
        "passes_freeze_gate",
        "模拟观察",
    ):
        for surface in (prompt, prompts.WRAP_UP_PROMPT):
            assert retired not in surface, retired
    # Beyond the orientation's ambition, how to research is the Agent's own
    # call, which a reference workspace may suggest: the stable prompt carries
    # no research protocol, evidence recipe, process quota or delegation
    # style. Compute guidance is one line each: the orientation's strategy
    # limits as a budget and, with cards, the GPU bullet
    # (test_experiment_facts_scope); the working method's line on fast
    # strategy code (test_the_prompt_states_the_research_and_backtesting_process).
    for retired in (
        "# 研究协议",
        "# 证据标准",
        "# 原则",
        "至少跑完两轮",
        "比过对照",
        "核心打分器",
        "这是建议，不是要求",
        "示例（只示形式）",
        "以包为准",
        "开放搜索（",
        "默认探索方向",
        "按季重训",
    ):
        assert retired not in prompt, retired
    # Operational detail is the tools' and the contract's: no registration
    # field, row field, threshold or mount path is restated here.
    for statement in (
        "raw_excess_at_cost_stress",
        "plain_selection",
        "seed_replicates",
        "SEED_BASE",
        "residual_reading",
        "/mnt/",
        "500 字符",
    ):
        assert statement not in stable, statement
    from autotrade.pipelines.session_tools import ValidateTool

    assert "offline_trials" not in ValidateTool.spec.input_schema["properties"]
    # Round 20260927 wrote 296 skills, most re-deriving one data contract,
    # because every finishing text asked for them first; none does now.
    for surface in surfaces.values():
        assert "写好 skills" not in surface and "写进 skills" not in surface
    # The run facts open their block with an index of their keys, say they
    # are a snapshot the session does not see updated, and name the file the
    # same facts are in; no package list is promised.
    facts = prompt[prompt.index("## 运行事实") : prompt.index("```json")]
    assert "会话中不更新" in facts and "inputs/session_context.json" in facts
    for fact in ("budgets（", "research_geometry（", "artifact_contract.acceptance_rules（"):
        assert fact in facts, fact
    assert "已装依赖" not in facts


# What the session surfaces call the experiment, its control portfolio, its
# finalised delivery and its out-of-sample acceptance. These are the coinages
# the surfaces used to carry; each one is a different word for a thing the
# contract already names, so a surface that brings one back is drifting again.
RETIRED_NARRATION = ("本臂", "零技能面板", "一本书", "席位", "回合数", "孵化", "毕业")


def test_the_session_surfaces_word_the_experiment_the_current_way() -> None:
    surfaces = {
        "system_prompt": _full_prompt(),
        "default_instruction": prompts.SESSION_DEFAULT_INSTRUCTION,
        "wrap_up": prompts.WRAP_UP_PROMPT,
        **{f"subagent:{role}": subagent_system_prompt(role) for role in SUBAGENT_ROLES},
        **_tool_descriptions(),
    }
    for name, text in surfaces.items():
        for retired in RETIRED_NARRATION:
            assert retired not in text, (name, retired)
    # The wording the surfaces do use, stated once where each is defined.
    assert "AI 首席量化研究官" in surfaces["system_prompt"]
    assert "随机选股对照组合" in surfaces["system_prompt"]
    assert "定版" in surfaces["system_prompt"]


# Forward and Held-out results never enter a session (agent-design §1.2): the
# prompt and the curated memory state rules about those periods, never a
# reading of them. A clause that names either period carries no figure.
FORWARD_MARKERS = ("前推", "Held-out", "heldout", "forward months", "会话看不到的")
FIGURE = re.compile(r"\d+\.\d+|\d\s*%")
# The row blocks the host builds; a path the contract names under them must
# exist on a real row, so a renamed field cannot leave the contract pointing
# at nothing (``raw_readings.last_two_years.return`` did).
ROW_BLOCKS = ("active", "book", "concentration")


def _contract_texts() -> dict[str, str]:
    from autotrade.pipelines.skills import SKILL_LIBRARY_PATH

    library = Path(__file__).resolve().parents[2] / SKILL_LIBRARY_PATH
    return {
        "system_prompt": prompts.PROTOCOL_INSTRUCTION,
        "default_instruction": prompts.SESSION_DEFAULT_INSTRUCTION,
        "wrap_up": prompts.WRAP_UP_PROMPT,
        **{
            f"memory:{path.parent.name}": path.read_text(encoding="utf-8")
            for path in sorted(library.glob("*/SKILL.md"))
        },
    }


def test_the_contract_quotes_no_forward_figure_and_names_only_real_row_fields() -> None:
    from .test_batch_validate import _Session, _strategy

    texts = _contract_texts()
    for name, text in texts.items():
        for clause in re.split(r"[。；\n]", text):
            if any(marker in clause for marker in FORWARD_MARKERS):
                assert not FIGURE.search(clause), (name, clause)

    with tempfile.TemporaryDirectory() as tmp:
        session = _Session(Path(tmp))
        row = session.validate_one("a", _strategy("1"))
    named = {
        path
        for text in texts.values()
        for path in re.findall(r"`([a-z_]+(?:\.[a-z_0-9]+)+)`", text)
        if path.split(".")[0] in ROW_BLOCKS
    }
    assert named, "the contract names no row field"
    for path in sorted(named):
        node: object = row
        for key in path.split("."):
            assert isinstance(node, dict) and key in node, path
            node = node[key]


def _run_facts(**geometry: object) -> dict[str, object]:
    """Facts carrying what the run section reads, at a five-year research and
    five-year validation geometry with quarterly steps; ``years`` lists the
    research years, as the facts do."""

    from autotrade.pipelines.verdict import residual_threshold

    return {
        "research_geometry": {
            "test_horizon": "20200731",
            "research_period": "20150801..20200731",
            "years": [{"label": f"Y{index}", "end": f"{2015 + index}0731"} for index in range(1, 6)],
            "validation_period": "20200801..20250731",
            "step": "quarter",
            "steps": 20,
            "steps_tested": 0,
            **geometry,
        },
        "artifact_contract": {
            "acceptance_rules": {
                "significance": {
                    "alpha": 0.05,
                    "threshold_over_one_year": round(residual_threshold(1.0, 0.05), 3),
                    "threshold_over_validation_period": round(residual_threshold(5.0, 0.05), 3),
                }
            }
        },
        "budgets": {"validates_per_step": 6},
    }


def test_the_run_section_states_this_runs_view_steps_and_threshold() -> None:
    """The environment tells the session, concretely, that its data ends at
    the research years' end for the whole session, the research years, the
    number and length of the steps, the threshold and the validate allowance, read
    off the facts the prompt also carries."""

    facts = _run_facts()
    prompt = prompts.build_system_prompt(experiment_facts=facts)
    section = prompt[prompt.index("## 本次运行") : prompt.index("## 运行事实")]
    assert prompt.index("# 环境") < prompt.index("## 本次运行")
    for clause in (
        "- 数据：会话挂载的数据止于研究年末 20200731，整个会话都不变；验证期只通过 `test` 的结果揭开。",
        "- 研究年：5 个，20150801..20200731；`validate` 只在研究年内回测。",
        "- 验证期：20200801..20250731，共 20 步，每步一个季度；本次尝试开始时已测试 0 步。",
        "- 门槛：α = 0.05；一年区间约 1.645，整个验证期约 0.736。",
        "- `validate`：第一次 `test` 之前、以及相邻两次 `test` 之间，各至多 6 次。",
    ):
        assert clause in section, clause
    # A resumed attempt reads where its predecessor stopped on the same data
    # view; monthly steps read as months.
    later = prompts.build_run_section(
        _run_facts(test_horizon="20211031", step="month", steps=60, steps_tested=15)
    )
    assert "止于研究年末 20200731，" in later
    assert "共 60 步，每步一个月；本次尝试开始时已测试 15 步，测到 20211031。" in later
    # A geometry that states a test horizon must state the rest: a missing key or
    # an unknown step length fails the prompt instead of dropping the line.
    broken = _run_facts()
    del broken["budgets"]["validates_per_step"]  # type: ignore[attr-defined]
    with pytest.raises(KeyError):
        prompts.build_run_section(broken)
    with pytest.raises(KeyError):
        prompts.build_run_section(_run_facts(step="week"))
    # Facts without a test horizon add no section.
    assert "## 本次运行" not in _full_prompt()


def test_the_run_section_states_the_session_budgets_per_research_interval() -> None:
    """The three session budgets in words, and their average over the
    research intervals: one before each test, so as many as the steps. A
    budget the facts do not carry is left out, never printed as a blank."""

    facts = _run_facts()
    facts["budgets"].update(  # type: ignore[attr-defined]
        deadline_seconds=600_000.0, max_llm_calls=6400, max_api_spend_usd=50.0
    )
    assert prompts.build_run_section(facts).endswith(
        "\n- 预算（整个会话共用）：推理时间 166 小时 40 分钟、模型调用 6400 次、API 花费 50 美元；"
        "每次 `test` 之前是一段研究，20 段平均每段约 8 小时 20 分钟推理时间、320 次模型调用。"
    )
    del facts["budgets"]["max_api_spend_usd"]  # type: ignore[attr-defined]
    del facts["budgets"]["max_llm_calls"]  # type: ignore[attr-defined]
    assert prompts.build_run_section(facts).endswith(
        "\n- 预算（整个会话共用）：推理时间 166 小时 40 分钟；"
        "每次 `test` 之前是一段研究，20 段平均每段约 8 小时 20 分钟推理时间。"
    )
    assert "预算" not in prompts.build_run_section(_run_facts())
