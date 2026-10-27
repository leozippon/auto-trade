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


def test_the_prompt_states_the_research_session_contract() -> None:
    prompt = _full_prompt()
    stable = prompt.split("# 环境")[0]
    role = prompt[: prompt.index("# 结束会话")]
    # Purpose: one mechanism on the research period; graduation is decided on
    # later sealed data; adaptation lives inside the artifact.
    for clause in (
        "本臂只有这一个研究会话",
        "前推期与 Held-out 上被连续回放一次并裁决",
        "滚动重拟合",
        "用多长的历史、多快的节奏由策略自己决定",
        # The direction binds what is researched, not how: a pack's advice on
        # method, models and compute is optional, and beyond it a pack binds
        # only the data boundary, costs and resource limits.
        "每个候选都属于这个方向",
        "方向之外的想法记进收尾 `reason` 的未测项",
        "包里关于方法、模型与算力的建议可以不采纳",
        "包的其余文字只在 PIT 与数据边界、交易成本与资源上限上约束本臂",
        "参考包、研究方向与研究者指令都不能放宽下面的结束条件、策略合同或禁止事项",
    ):
        assert clause in role, clause
    assert "## 本臂的研究方向（用户注入）\n方向约束研究什么，不约束怎样做" in prompt
    contract = prompt[prompt.index("# 结束会话") : prompt.index("# 策略合同")]
    for clause in (
        "`freeze`",
        "`no_edge`",
        "之后没有别的会话接手",
        '`span="full"`',
        "`deadline`",
        "一条臂至多冻结一次",
        # Which conditions bind is the acceptance_rules fact alone.
        "只看运行事实 `acceptance_rules`",
        "`freeze_gate` 是 `finish_session` 提名时执行的冻结门",
        "`graded_series` 定义的主动序列",
        "没列出的条件本臂没有",
        "`selection_statistics.information_ratio_bar`",
        # Screening saves replay years, not trials, and has one home.
        "包括 `source_refs.signal_screen_ref` 的筛选脚本",
        "按 `offline_trials` 如实申报",
        "种子的写法见 `output/README.md`",
        # Only a package with randomness owes seed replicates; one without is
        # nominated deterministic, which the ledger keeps for the operator.
        "策略包含随机性（训练、采样）的提名要带种子复现",
        "完全不含随机性的包以 `deterministic` 提名，不需要种子复现",
        "这一声明记入账本，供运营者在上线前终审",
        # A node that fails only the deflation is a deliverable by incubation,
        # and a no_edge names it first.
        "却仍是本臂的交付",
        "`freeze_deflated_sharpe_below_threshold`",
        "运营者可以把它孵化到 Paper 上裁决",
        "多搜索不会让这个交付变远",
        "以 `no_edge` 结束时，`reason` 先点名这样的节点",
    ):
        assert clause in contract, clause
    # The hard finalization replaces the system prompt, so it carries the
    # incubation naming rule itself.
    assert "freeze_deflated_sharpe_below_threshold" in prompts.HARD_FINALIZATION_SYSTEM_PROMPT
    # How to research is the Agent's own call, which a reference pack may
    # suggest: the stable prompt carries no research protocol, evidence
    # recipe, process quota, delegation style or compute advice, and none of
    # the rules that bound a session to its pack's axes or a fixed refit pace.
    for retired in (
        "# 研究协议",
        "# 证据标准",
        "# 原则",
        "至少跑完两轮",
        "比过对照",
        "不得再以这个假设冻结",
        "核心打分器",
        "向量化",
        "这是建议，不是要求",
        "示例（只示形式）",
        "诚实指的是",
        "下一根登记轴",
        "让位于包",
        "不换家族",
        "只在这些轴上",
        "以包为准",
        "尽早让第一个真正的候选",
        "开放搜索（",
        "默认探索方向",
        "按季重训",
    ):
        assert retired not in prompt, retired
    # Each kind of statement has one home. Which conditions bind and at what
    # threshold is the arm's acceptance_rules fact: the prompt enumerates none
    # and carries no sentence that holds only for arms whose rules list an
    # optional condition, which a resumed older arm would misread.
    for statement in (
        "列有",
        "raw_excess_at_cost_stress",
        "plain_selection",
        "seed_replicates",
        "arm.lineage",
        "SEED_BASE",
        "研究期权益回撤与主动回撤",
        "按字节认",
    ):
        assert statement not in stable, statement
    # A registration field is defined by its parameter, not here.
    from autotrade.environment.tools.finish_session import FinishSessionTool
    from autotrade.pipelines.session_tools import BatchValidateTool

    offline = BatchValidateTool.spec.input_schema["properties"]["offline_trials"]
    assert "does NOT submit" in offline["description"]
    assert "**没有**提交" not in stable and "不重复申报" not in stable
    # The early-finish account is enforced by finish_session and stated in its
    # description; the prompt does not restate it.
    assert "more than a third" in FinishSessionTool.spec.description
    assert "三分之一" not in prompt
    # Round 20260927 wrote 296 skills, most re-deriving one data contract,
    # because every finishing text asked for them first; none does now.
    for surface in (prompt, prompts.SESSION_DEFAULT_INSTRUCTION, prompts.WRAP_UP_PROMPT):
        assert "写好 skills" not in surface and "写进 skills" not in surface
    feedback = prompt[prompt.index("# 反馈通道") : prompt.index("# 环境")]
    assert "只写挂载的运行记忆与数据引用里还没有的可复用做法" in feedback
    assert "本臂的结论与证据写进 `finish_session` 的 `reason`，不另写成 skill" in feedback
    assert "`continue`" not in contract and "PRIOR" not in prompt
    # The run facts open the environment block with an index of their keys,
    # and say they are a snapshot the session does not see updated.
    facts = prompt[prompt.index("## 运行事实") : prompt.index("```json")]
    assert "会话中不更新" in facts
    for fact in ("`budgets`", "`research_geometry`", "`arm`", "`artifact_contract`"):
        assert fact in facts, fact
    # Enforced limits and mount paths sit in the schema and the facts.
    assert "500 字符" not in prompt
    assert "/mnt/tools/screen.py" not in stable


# Forward and Held-out results never enter a session (agent-design §1.2): the
# prompt and the curated memory state rules about those periods, never a
# reading of them. A clause that names either period carries no figure.
FORWARD_MARKERS = ("前推", "Held-out", "heldout", "forward months", "会话看不到的")
FIGURE = re.compile(r"\d+\.\d+|\d\s*%")
# The row blocks the host builds; a path the contract names under them must
# exist on a real row, so a renamed field cannot leave the contract pointing
# at nothing (``raw_readings.last_two_years.return`` did).
ROW_BLOCKS = ("selection_statistics", "raw_readings")


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
