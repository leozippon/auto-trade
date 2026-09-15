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
        step_tree_enabled=True,
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
        StepRollbackTool,
        WriteFileTool,
    )
    from autotrade.pipelines.local_backend import (
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
    role = prompt[: prompt.index("# 研究协议")]
    # Purpose: one mechanism on the research period; graduation is decided on
    # later sealed data; adaptation lives inside the artifact.
    for clause in (
        "本臂只有这一个研究会话",
        "前推期与 Held-out 上被连续回放一次并裁决",
        "滚动重拟合",
        "尾部窗口通常取 2–3 年、按季重训",
        "不是替代它",
    ):
        assert clause in role, clause
    protocol = prompt[prompt.index("# 研究协议") : prompt.index("# 决策合同")]
    for clause in (
        "`hypothesis` 参数就是有约束力的预登记记录",
        "调用之后补写的笔记不算预登记",
        "迭代用子区间，结论用全期",
        '`span="full"`',
        "冻结只接受完整研究期节点",
        "`run_null_control` 的随机同名组合零假设",
        "`source_refs.signal_screen_ref`",
        "机制家族指收益来源的经济解释",
        "同一特征集换个估计器不算",
        "示例（只示形式）",
        '`finish_session(outcome="no_edge", reason=<触发它的读数>)`',
    ):
        assert clause in protocol, clause
    contract = prompt[prompt.index("# 决策合同") : prompt.index("# 证据标准")]
    for clause in (
        "`freeze`",
        "`no_edge`",
        "之后没有别的会话接手",
        "`deadline`",
        "`acceptance_rules.freeze_gate`",
        "本臂至少两个完整研究期验证",
        "任何 span、任何尝试、对照都算",
        "一条臂至多冻结一次",
        "`finish_session` 之后不能再写",
    ):
        assert clause in contract, clause
    assert "`continue`" not in contract and "PRIOR" not in prompt
    evidence = prompt[prompt.index("# 证据标准") : prompt.index("# 原则")]
    for clause in (
        "中性化超额约为 0",
        "半数以上研究年份的中性化超额为负",
        "`excess_percentile` 在 0.5 附近",
        "`selection_statistics.deflated_sharpe_probability`",
        "`acceptance_rules.graduation.forward.minimum_detectable_excess`",
    ):
        assert clause in evidence, clause
    facts = prompt[prompt.index("# 预算与事实") : prompt.index("# 反馈通道")]
    for fact in ("`budgets`", "`research_geometry`", "`arm`", "`artifact_contract`"):
        assert fact in facts, fact
    feedback = prompt[prompt.index("# 反馈通道") :]
    assert "留给后来者的只有两处" in feedback
    # Each shared rule has one home.
    assert prompt.count("三分之一") == 1
    assert prompt.count("不是选择标准") == 1
    # Enforced limits and mount paths sit in the schema and the facts.
    assert "500 字符" not in prompt
    assert "/mnt/tools/screen.py" not in prompt.split("# 本会话动态上下文")[0].split("# 工具与工作方式")[0]
