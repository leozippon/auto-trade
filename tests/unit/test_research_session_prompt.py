"""Invariants of what a research session reads that the byte comparison of
the exported prompts (``configs/prompts/PROMPTS.md``) cannot show: the
numbers the prompt renders from the run facts, the allowance the tools and
the prompt state alike, and the Forward/Held-out readings the contract never
quotes.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from autotrade.agent import prompts


def _full_prompt() -> str:
    return prompts.build_system_prompt(
        experiment_facts={"identity": {"session_kind": "research"}},
        exploration_directive="长期方向",
        session_directive="本会话指令",
    )


def test_the_tools_state_the_validate_allowance_the_prompt_states() -> None:
    """validate and smoke_backtest name the allowance at its facts path, and
    the run section states the configured number over the same interval:
    validates_per_step calls before the first test and between tests."""

    from autotrade.pipelines.session_tools import SmokeBacktestTool, ValidateTool

    for tool in (ValidateTool, SmokeBacktestTool):
        assert "acceptance_rules.validation.validates_per_step" in tool.spec.description
    assert "before the first test and between two tests" in ValidateTool.spec.description
    run = prompts.build_run_section(_run_facts())
    assert "- `validate`：第一次 `test` 之前、以及相邻两次 `test` 之间，各至多 6 次。" in run


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
        "- 研究年：5 个，20150801..20200731。",
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


def test_the_run_section_states_the_session_budgets() -> None:
    """The three session budgets in words, totals only. A budget the facts do
    not carry is left out, never printed as a blank."""

    facts = _run_facts()
    facts["budgets"].update(  # type: ignore[attr-defined]
        deadline_seconds=600_000.0, max_llm_calls=6400, max_api_spend_usd=50.0
    )
    assert prompts.build_run_section(facts).endswith(
        "\n- 预算（整个会话共用）：推理时间 166 小时 40 分钟、模型调用 6400 次、API 花费 50 美元。"
    )
    del facts["budgets"]["max_api_spend_usd"]  # type: ignore[attr-defined]
    del facts["budgets"]["max_llm_calls"]  # type: ignore[attr-defined]
    assert prompts.build_run_section(facts).endswith(
        "\n- 预算（整个会话共用）：推理时间 166 小时 40 分钟。"
    )
    assert "预算" not in prompts.build_run_section(_run_facts())
