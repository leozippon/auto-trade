"""``finish_session``: the two ways the research session ends.

The tool decides nothing about the market: it refuses a freeze the gate
rejects and an unexplained decision, and it hands the Pipeline an outcome the
pipeline records as ``freeze`` or ``no_edge``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.step_tree import StepTree
from autotrade.environment.tools.base import (
    AGENT_JUSTIFICATION_MAX_CHARS,
    ToolError,
    ToolRegistry,
)
from autotrade.environment.tools.finish_session import (
    REASON_MAX_CHARS,
    REASON_MIN_CHARS,
    FinishSessionTool,
    SessionBudgetStatus,
)
from autotrade.pipelines.config import SESSION_OUTCOMES
from autotrade.pipelines.local_backend import _session_outcome

SESSION = "session_ref_ab"
RUN = "run_x"
REASON = "neutralized excess is negative in three of four research years; the null percentile is 0.48"


def _node(tree: StepTree, root: Path, marker: str, *, session: str = SESSION, run: str = RUN) -> str:
    output = root / f"cand_{marker}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text(
        f"def generate_orders(context):\n    _ = {marker!r}\n    return []\n",
        encoding="utf-8",
    )
    return tree.record_step(
        output,
        epoch_id="research",
        session_ref=session,
        run_id=run,
        result_name=f"valid_{marker}",
        revision_id=new_revision_id("revision"),
        metrics={},
        metadata={"span": "full"},
    )


class _Gate:
    """The Pipeline's gate as the tool sees it: node id -> verdict."""

    def __init__(self, passing: set[str]) -> None:
        self.passing = passing

    def __call__(self, node_id: str) -> dict[str, object]:
        if node_id in self.passing:
            return {
                "passed": True,
                "reasons": [],
                "full_span_validations": 2,
                "information_ratio": 0.9,
                "deflated_sharpe": {"deflated_sharpe_probability": 0.71, "trials": 5},
            }
        return {
            "passed": False,
            "reasons": ["freeze_deflated_sharpe_below_threshold"],
            "full_span_validations": 2,
            "information_ratio": 0.2,
            "deflated_sharpe": {"deflated_sharpe_probability": 0.12, "trials": 5},
        }


def _tool(tree: StepTree, gate: _Gate, **kwargs: object) -> FinishSessionTool:
    return FinishSessionTool(tree, session_ref=SESSION, freeze_gate=gate, **kwargs)  # type: ignore[arg-type]


def test_every_outcome_maps_to_the_pipeline_outcome_it_names(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    winner = _node(tree, tmp_path, "a")
    registry = ToolRegistry([_tool(tree, _Gate({winner}))])
    results = {
        "freeze": registry.invoke("finish_session", {"outcome": "freeze", "node_id": winner}),
        "no_edge": registry.invoke("finish_session", {"outcome": "no_edge", "reason": REASON}),
    }
    assert set(results) == set(SESSION_OUTCOMES) - {"deadline"}
    mapped = {name: _session_outcome(result.value) for name, result in results.items()}
    assert mapped == {
        "freeze": ("freeze", winner, ""),
        "no_edge": ("no_edge", None, REASON),
    }
    assert all(result.ok and result.finish for result in results.values())
    assert results["freeze"].value["freeze_gate"]["deflated_sharpe_probability"] == 0.71
    # The outcome is required; the retired outcomes (the multi-session
    # ``continue`` among them) are not in the enum.
    for arguments in (
        {},
        {"outcome": "continue", "node_id": winner, "reason": REASON},
        {"outcome": "terminate", "reason": REASON},
        {"outcome": "select"},
    ):
        assert registry.invoke("finish_session", arguments).ok is False


def test_a_nomination_the_gate_rejects_is_refused_with_its_reasons_and_numbers(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    weak = _node(tree, tmp_path, "weak")
    strong = _node(tree, tmp_path, "strong")
    finish = _tool(tree, _Gate({strong}))
    with pytest.raises(ToolError) as refused:
        finish.invoke({"outcome": "freeze", "node_id": weak})
    message = str(refused.value)
    assert refused.value.error_type == "freeze_gate_refused"
    assert "freeze_deflated_sharpe_below_threshold" in message
    assert "deflated_sharpe_probability=0.12" in message and "trials=5" in message
    assert strong in message
    assert refused.value.details["passing_nodes"] == [strong]
    assert refused.value.details["freeze_gate"]["reasons"] == [
        "freeze_deflated_sharpe_below_threshold"
    ]
    # Nothing ended: the tree still points where it was, and the passing node freezes.
    assert finish.invoke({"outcome": "freeze", "node_id": strong}).finish


def test_no_edge_names_no_node_and_needs_a_reason_and_a_validation(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    empty = _tool(tree, _Gate(set()))
    with pytest.raises(ToolError, match="at least one complete"):
        empty.invoke({"outcome": "no_edge", "reason": REASON})
    node = _node(tree, tmp_path, "a")
    with pytest.raises(ToolError, match="node_id must be absent"):
        empty.invoke({"outcome": "no_edge", "node_id": node, "reason": REASON})
    with pytest.raises(ToolError, match="requires reason"):
        empty.invoke({"outcome": "no_edge", "reason": "x" * (REASON_MIN_CHARS - 1)})
    result = empty.invoke({"outcome": "no_edge", "reason": REASON})
    assert result.value["candidates_evaluated"] == 1
    assert "no forward test" in result.value["pipeline_outcome"]


def test_the_reason_is_bounded_like_every_other_written_justification(tmp_path: Path):
    """The reason carries the gate readouts and the directions that were
    closed, the same account a pre-registered hypothesis carries, so it gets
    the same bound from the same constant -- and the bound the Agent is told
    up front, in the description and in the parameter, is the one enforced."""

    assert REASON_MAX_CHARS == AGENT_JUSTIFICATION_MAX_CHARS
    reason_schema = FinishSessionTool.spec.input_schema["properties"]["reason"]
    assert reason_schema["maxLength"] == REASON_MAX_CHARS
    assert f"{REASON_MIN_CHARS}-{REASON_MAX_CHARS}" in reason_schema["description"]
    assert f"{REASON_MIN_CHARS}-{REASON_MAX_CHARS}" in FinishSessionTool.spec.description

    tree = StepTree(tmp_path / "steps")
    _node(tree, tmp_path, "a")
    registry = ToolRegistry([_tool(tree, _Gate(set()))])
    at_cap = registry.invoke(
        "finish_session", {"outcome": "no_edge", "reason": "e" * REASON_MAX_CHARS}
    )
    assert at_cap.ok and at_cap.finish
    over = registry.invoke(
        "finish_session", {"outcome": "no_edge", "reason": "e" * (REASON_MAX_CHARS + 1)}
    )
    assert not over.ok
    assert f"limit {REASON_MAX_CHARS}" in over.error


def test_only_a_complete_node_of_this_session_can_be_named(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    earlier = _node(tree, tmp_path, "earlier", session="session_ref_older")
    mine = _node(tree, tmp_path, "mine")
    finish = _tool(tree, _Gate({earlier, mine}))
    with pytest.raises(ToolError, match="not a Step of this session"):
        finish.invoke({"outcome": "freeze", "node_id": earlier})
    with pytest.raises(ToolError, match="not a Step node"):
        finish.invoke({"outcome": "freeze", "node_id": "missing"})
    # One own node: a freeze may omit node_id; with two it must name one.
    assert finish.invoke({"outcome": "freeze"}).value["node_id"] == mine
    _node(tree, tmp_path, "second")
    with pytest.raises(ToolError, match="requires node_id") as ambiguous:
        finish.invoke({"outcome": "freeze"})
    assert len(ambiguous.value.details["candidates"]) == 2
    # A Validation an interrupted attempt of this session recorded is this
    # session's Step: nominable, and counted when a freeze must name one.
    resumed = _node(tree, tmp_path, "resumed", run="run_before")
    with pytest.raises(ToolError, match="requires node_id") as three:
        finish.invoke({"outcome": "freeze"})
    assert resumed in three.value.details["candidates"]
    result = _tool(tree, _Gate({resumed})).invoke({"outcome": "freeze", "node_id": resumed})
    assert result.value["node_id"] == resumed


def test_an_early_freeze_must_say_why_while_another_batch_fits(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    node = _node(tree, tmp_path, "a")

    def status(remaining: int) -> SessionBudgetStatus:
        return SessionBudgetStatus(
            replay_years_remaining=remaining,
            replay_years_total=24,
            inference_seconds_remaining=5400.0,
        )

    early = _tool(tree, _Gate({node}), budget_status=lambda: status(12))
    with pytest.raises(ToolError, match="again with reason") as refused:
        early.invoke({"outcome": "freeze", "node_id": node})
    assert "12/24 replay-years" in str(refused.value) and "90 min" in str(refused.value)
    explained = early.invoke({"outcome": "freeze", "node_id": node, "reason": REASON})
    assert explained.value["budget_at_finish"]["replay_years_remaining"] == 12
    # Exactly a third left, or no batch left to run, needs no reason.
    assert _tool(tree, _Gate({node}), budget_status=lambda: status(8)).invoke(
        {"outcome": "freeze", "node_id": node}
    ).finish
    assert _tool(
        tree, _Gate({node}), budget_status=lambda: status(12), another_round_fits=lambda: False
    ).invoke({"outcome": "freeze", "node_id": node}).finish
