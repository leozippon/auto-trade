"""The Agent experiences a research session, not a Fold.

Every surface a session model reads -- the system prompt with its dynamic
half, the opening message, the wrap-up and hard-finalization prompts, the
sub-agent role prompts and every tool description -- carries the research
workflow and none of the retired Fold/Meta vocabulary, and no retired tool is
registrable for any role.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from autotrade.agent import prompts
from autotrade.agent.runner import _FINALIZATION_TOOLS, _SESSION_TOOLS, _TERMINAL_TOOLS
from autotrade.agent.subagent import (
    AGENT_TOOL_DESCRIPTION,
    SUBAGENT_ROLES,
    allowed_subagent_tools,
    subagent_system_prompt,
)
from autotrade.environment.tools.base import SEQUENTIAL_TOOL_NAMES

RETIRED_TOOLS = ("finish_fold", "finish_meta", "daily_backtest", "terminate")

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
        FinishSessionTool,
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
        BatchValidateTool,
        NullControlTool,
        SmokeBacktestTool,
    )
    from autotrade.pipelines.skills import DeleteSkillTool, WriteSkillTool

    with tempfile.TemporaryDirectory() as tmp:
        roots = SearchRoots(SafeWorkspace(Path(tmp)))
        specs = [tool.spec for tool in (GlobTool(roots), GrepTool(roots), ReadFileTool(roots))]
    specs.extend(
        cls.spec
        for cls in (
            CompactTool,
            EditFileTool,
            WriteFileTool,
            SandboxShellTool,
            ModificationCheckTool,
            StepRollbackTool,
            ReportIssueTool,
            SkillFeedbackTool,
            FinishSessionTool,
            BatchValidateTool,
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
        "hard_finalization",
        *(f"subagent:{role}" for role in SUBAGENT_ROLES),
    ],
)
def test_no_session_surface_carries_fold_or_meta_vocabulary(surface: str) -> None:
    text = {
        "system_prompt": _full_prompt(),
        "default_instruction": prompts.SESSION_DEFAULT_INSTRUCTION,
        "wrap_up": prompts.WRAP_UP_PROMPT,
        "hard_finalization": prompts.HARD_FINALIZATION_SYSTEM_PROMPT,
    }.get(surface) or subagent_system_prompt(surface.split(":", 1)[1])
    assert retired_vocabulary(text) == [], surface


def test_no_tool_description_carries_fold_or_meta_vocabulary() -> None:
    for name, text in _tool_descriptions().items():
        assert retired_vocabulary(text) == [], name


def test_retired_tools_are_absent_from_every_role() -> None:
    surfaces = {
        "session": _SESSION_TOOLS,
        "terminal": _TERMINAL_TOOLS,
        "finalization": _FINALIZATION_TOOLS,
        "sequential": SEQUENTIAL_TOOL_NAMES,
        **{f"subagent:{role}": allowed_subagent_tools(role) for role in SUBAGENT_ROLES},
    }
    for surface, names in surfaces.items():
        for retired in RETIRED_TOOLS:
            assert retired not in names, (surface, retired)
    assert _TERMINAL_TOOLS == _FINALIZATION_TOOLS == {"finish_session"}
    # The finish and the Agent's own compaction rebuild or end the turn, so
    # neither runs beside other calls.
    assert SEQUENTIAL_TOOL_NAMES == {"finish_session", "compact"}
    # Sub-agents never finish, validate, roll back or compact the parent.
    for role in SUBAGENT_ROLES:
        assert not allowed_subagent_tools(role) & {
            "finish_session",
            "batch_validate",
            "run_null_control",
            "step_rollback",
            "agent",
            "compact",
        }
    import autotrade.environment.tools as tools_package

    for retired in ("FinishFoldTool", "FinishMetaTool"):
        assert not hasattr(tools_package, retired)


def test_the_prompt_states_the_research_and_backtesting_process() -> None:
    """The static part says who the Agent is, the process it is engaged in and
    the rules that matter, each once; the numbers are the run section's."""

    prompt = _full_prompt()
    stable = prompt.split("# 环境")[0]
    process = stable[stable.index("# 研究与回测流程") : stable.index("# 读数")]
    readings = stable[stable.index("# 读数") : stable.index("# 策略")]
    assert "AI 首席量化研究官（AI Chief Quantitative Research Officer）" in stable
    # Research on the visible years, each test extends the backtest one step,
    # then the host-run Forward and Held-out stages, then Paper.
    for clause in (
        "研究期分两段：研究年一开始就可见，验证期由 `test` 逐步揭开",
        "你能读到的数据止于当前的前沿日期",
        "用 `validate` 回测",
        "`test` 把一个候选从验证期起点回测到下一步的终点，这一步的结果记入本实验的滚动记录",
        "随后前沿推进一步",
        "已记入滚动记录的条目不会改写",
        "验证期最后一步测试完，会话结束，最后测试的候选被冻结",
        "会话不能主动结束",
        "宿主用最后测试的候选跑完剩下的步",
        "运行 Forward 与 Held-out 两个阶段，读数不回到会话",
        "是否进入 Paper（每日出单、人工交易）由运营者决定",
        "`fit` 与 `REFIT_PERIOD`",
    ):
        assert clause in process, clause
    # The host reads the record; the statistic and its bar; no pass or fail.
    for clause in (
        "宿主读的是滚动记录",
        "随机选股对照组合",
        "残差信息比率",
        "z(1−α)/√年数",
        "会话里没有通过或不通过",
        "是读数，不是判决",
    ):
        assert clause in readings, clause
    assert "参考包、研究方向与研究者指令都不能放宽它们" in stable
    assert "读取前沿之后的数据" in stable and "模型自身的知识" in stable
    assert "## 本实验的研究方向（用户注入）\n方向约束研究什么，不约束怎样做" in prompt
    # The retired action set and its wording are gone from every surface the
    # session reads under this contract: no freeze or finishing action, no
    # "no excess" outcome, no pass/fail verdict, no replay-year budget. (The
    # runner's finish-based hard finalization keeps its own prompt until that
    # mechanism goes.)
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
            "定版",
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
    # How to research is the Agent's own call, which a reference pack may
    # suggest: the stable prompt carries no research protocol, evidence
    # recipe, process quota, delegation style or compute advice.
    for retired in (
        "# 研究协议",
        "# 证据标准",
        "# 原则",
        "至少跑完两轮",
        "比过对照",
        "核心打分器",
        "向量化",
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
    from autotrade.pipelines.session_tools import BatchValidateTool

    assert "offline_trials" not in BatchValidateTool.spec.input_schema["properties"]
    # Round 20260927 wrote 296 skills, most re-deriving one data contract,
    # because every finishing text asked for them first; none does now.
    for surface in surfaces.values():
        assert "写好 skills" not in surface and "写进 skills" not in surface
    # The run facts open their block with an index of their keys, and say
    # they are a snapshot the session does not see updated.
    facts = prompt[prompt.index("## 运行事实") : prompt.index("```json")]
    assert "会话中不更新" in facts
    for fact in ("`budgets`", "`research_geometry`", "`artifact_contract`"):
        assert fact in facts, fact


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
        "hard_finalization": prompts.HARD_FINALIZATION_SYSTEM_PROMPT,
        **{f"subagent:{role}": subagent_system_prompt(role) for role in SUBAGENT_ROLES},
        **_tool_descriptions(),
    }
    for name, text in surfaces.items():
        for retired in RETIRED_NARRATION:
            assert retired not in text, (name, retired)
    # The wording the surfaces do use, stated once where each is defined.
    assert "AI 首席量化研究官" in surfaces["system_prompt"]
    assert "随机选股对照组合" in surfaces["system_prompt"]
    assert "冻结" in surfaces["system_prompt"]


# Forward and Held-out results never enter a session (agent-design §1.2): the
# prompt and the curated memory state rules about those periods, never a
# reading of them. A clause that names either period carries no figure.
FORWARD_MARKERS = ("前推", "Held-out", "heldout", "forward months", "会话看不到的")
FIGURE = re.compile(r"\d+\.\d+|\d\s*%")
# The row blocks the host builds; a path the contract names under them must
# exist on a real row, so a renamed field cannot leave the contract pointing
# at nothing (``raw_readings.last_two_years.return`` did).
ROW_BLOCKS = ("residual_reading", "raw_readings")


def _contract_texts() -> dict[str, str]:
    from autotrade.pipelines.skills import OPERATING_MEMORY_LIBRARY

    library = Path(__file__).resolve().parents[2] / OPERATING_MEMORY_LIBRARY
    return {
        "system_prompt": prompts.PROTOCOL_INSTRUCTION,
        "default_instruction": prompts.SESSION_DEFAULT_INSTRUCTION,
        "wrap_up": prompts.WRAP_UP_PROMPT,
        "hard_finalization": prompts.HARD_FINALIZATION_SYSTEM_PROMPT,
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
    five-year validation geometry with quarterly steps."""

    from autotrade.pipelines.verdict import residual_bar

    return {
        "research_geometry": {
            "frontier": "20200731",
            "research_period": "20150801..20200731",
            "years": [{"label": f"Y{index}"} for index in range(1, 6)],
            "validation_period": "20200801..20250731",
            "step": "quarter",
            "steps": 20,
            "steps_tested": 0,
            **geometry,
        },
        "artifact_contract": {
            "acceptance_rules": {
                "instrument": {
                    "alpha": 0.05,
                    "bar_over_one_year": round(residual_bar(1.0, 0.05), 3),
                    "bar_over_validation_period": round(residual_bar(5.0, 0.05), 3),
                }
            }
        },
        "budgets": {"validates_per_step": 6},
    }


def test_the_run_section_states_this_runs_frontier_steps_and_bar() -> None:
    """The environment tells the session, concretely, the frontier date, the
    research years, the number and length of the steps, the bar and the
    validate allowance, read off the facts the prompt also carries."""

    facts = _run_facts()
    prompt = prompts.build_system_prompt(experiment_facts=facts)
    section = prompt[prompt.index("## 本次运行") : prompt.index("## 运行事实")]
    assert prompt.index("# 环境") < prompt.index("## 本次运行")
    for clause in (
        "- 前沿：20200731；会话里的数据止于这一天。",
        "- 研究年：5 个，20150801..20200731。",
        "- 验证期：20200801..20250731，共 20 步，每步一个季度；本次尝试开始时已测试 0 步。",
        "- 门槛：α = 0.05；一年区间约 1.645，整个验证期约 0.736。",
        "- `validate`：第一次 `test` 之前、以及相邻两次 `test` 之间，各至多 6 次。",
    ):
        assert clause in section, clause
    # A resumed attempt reads where its predecessor stopped; monthly steps
    # read as months.
    later = prompts.build_run_section(
        _run_facts(frontier="20211031", step="month", steps=60, steps_tested=15)
    )
    assert "- 前沿：20211031；" in later
    assert "共 60 步，每步一个月；本次尝试开始时已测试 15 步" in later
    # A geometry that states a frontier must state the rest: a missing key or
    # an unknown step length fails the prompt instead of dropping the line.
    broken = _run_facts()
    del broken["budgets"]["validates_per_step"]  # type: ignore[attr-defined]
    with pytest.raises(KeyError):
        prompts.build_run_section(broken)
    with pytest.raises(KeyError):
        prompts.build_run_section(_run_facts(step="week"))
    # Facts without a frontier add no section.
    assert "## 本次运行" not in _full_prompt()
