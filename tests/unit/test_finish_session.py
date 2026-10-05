"""``finish_session``: the two ways the research session ends.

The tool decides nothing about the market: it refuses a freeze the gate
rejects and an unexplained decision, and it hands the Pipeline an outcome the
pipeline records as ``freeze`` or ``no_edge``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.environment.artifacts import new_revision_id
from autotrade.environment.step_tree import StepTree, node_handle
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
from autotrade.environment.tools.step_rollback import StepRollbackTool
from autotrade.pipelines.config import SESSION_OUTCOMES
from autotrade.pipelines.research_session import _seed_replicates, _session_outcome

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

    def __call__(self, node_id: str, seed_replicates: tuple[str, ...] = ()) -> dict[str, object]:
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
            "reasons": [
                "freeze_deflated_sharpe_below_threshold",
                "freeze_active_drawdown_exceeded",
            ],
            "full_span_validations": 2,
            "information_ratio": 0.2,
            "positive_years": 5,
            "active_max_drawdown": 0.369,
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
    # Every reason comes with the number it was refused on.
    assert "active_max_drawdown=0.369" in message and "positive_years=5" in message
    # Passing nodes are named the way the Agent can type them back.
    assert node_handle(strong) in message
    assert refused.value.details["passing_nodes"] == [node_handle(strong)]
    assert refused.value.details["freeze_gate"]["reasons"] == [
        "freeze_deflated_sharpe_below_threshold",
        "freeze_active_drawdown_exceeded",
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
    with pytest.raises(ToolError, match="neither a node_id nor the short handle") as unknown:
        finish.invoke({"outcome": "freeze", "node_id": "missing"})
    # The refusal names what can be named, by handle, and never another session's node.
    assert unknown.value.error_type == "unknown_node"
    assert unknown.value.details["nodes"] == ["valid_mine"]
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
    assert node_handle(resumed) in three.value.details["candidates"]
    result = _tool(tree, _Gate({resumed})).invoke({"outcome": "freeze", "node_id": resumed})
    assert result.value["node_id"] == resumed


def test_a_node_is_named_by_its_short_handle_only_while_it_is_unambiguous(tmp_path: Path):
    """The ~110-character ids are what the session's model mistypes, so a
    node's result name names it too -- but never by guessing: each attempt
    numbers its results afresh, and a handle two attempts share is refused
    with the full ids it could mean, as is a handle no node carries."""

    tree = StepTree(tmp_path / "steps")
    other = _node(tree, tmp_path, "b", session="session_ref_older")
    first = _node(tree, tmp_path, "a")
    second = _node(tree, tmp_path, "b")
    finish = _tool(tree, _Gate({first, second, other}))
    # Another session's node shares the handle but not the session: no ambiguity.
    frozen = finish.invoke({"outcome": "freeze", "node_id": "valid_b"}).value
    assert (frozen["node_id"], frozen["handle"]) == (second, "valid_b")
    # A resumed attempt of this session numbers its results from the start again.
    repeat = _node(tree, tmp_path, "a", run="run_resumed")
    with pytest.raises(ToolError) as ambiguous:
        _tool(tree, _Gate({first})).invoke({"outcome": "freeze", "node_id": "valid_a"})
    assert ambiguous.value.error_type == "unknown_node"
    assert first in str(ambiguous.value) and repeat in str(ambiguous.value)
    # Its nodes stay nameable by full id, the unique one by handle, and the
    # refusal lists them that way.
    assert sorted(ambiguous.value.details["nodes"]) == sorted([first, "valid_b", repeat])
    full = _tool(tree, _Gate({first})).invoke({"outcome": "freeze", "node_id": first}).value
    assert full["node_id"] == first and "handle" not in full
    # The Step tree the Agent reads shows each unique handle beside its id.
    rendered = tree.render_ascii()
    assert f"[valid_b] {second}" in rendered and f"[valid_b] {other}" in rendered
    assert f"[valid_a] {first}" not in rendered and f"- {first}" in rendered
    # step_rollback takes the same two forms under the same rule.
    output, models = tmp_path / "work" / "output", tmp_path / "work" / "models"
    output.mkdir(parents=True)
    rollback = StepRollbackTool(tree, output, models, session_ref=SESSION)
    restored = rollback.invoke({"node_id": "valid_b"}).value
    assert (restored["node_id"], restored["handle"]) == (second, "valid_b")
    assert (output / "main.py").read_text(encoding="utf-8").count("'b'") == 1
    with pytest.raises(ToolError, match="short handle of 2 nodes"):
        rollback.invoke({"node_id": "valid_a"})
    with pytest.raises(ToolError, match="neither a node_id nor the short handle"):
        rollback.invoke({"node_id": "valid_zz"})


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


def test_seed_replicates_are_offered_where_the_arm_holds_them_and_go_to_the_gate(tmp_path: Path):
    """Only an arm whose rules hold the condition is offered seed_replicates;
    the tool resolves them like node_id and hands them to the gate, which
    judges them. A nominee refused for want of them is told what a replicate
    is, a node that fails only for want of them is listed as passing, and the
    freeze hands the Pipeline their full ids."""

    tree = StepTree(tmp_path / "steps")
    nominee = _node(tree, tmp_path, "a")
    replicate = _node(tree, tmp_path, "b")
    calls: list[tuple[str, tuple[str, ...]]] = []

    def gate(node_id: str, seed_replicates: tuple[str, ...] = ()) -> dict[str, object]:
        calls.append((node_id, seed_replicates))
        if seed_replicates == (replicate,):
            return {"passed": True, "reasons": []}
        return {
            "passed": False,
            "reasons": ["freeze_too_few_seed_replicates"],
            "seed_replicates": {"trains_a_model": True, "replicates": []},
        }

    plain = ToolRegistry([_tool(tree, gate)])  # type: ignore[arg-type]
    assert "seed_replicates" not in FinishSessionTool.spec.input_schema["properties"]
    named = {"outcome": "freeze", "node_id": nominee, "seed_replicates": [replicate]}
    assert "unknown argument" in str(plain.invoke("finish_session", named).error)

    seeded = _tool(tree, gate, seed_replicates=True)  # type: ignore[arg-type]
    assert "seed_replicates" in seeded.spec.input_schema["properties"]
    # The rule a replicate meets is the arm's freeze_gate fact, stated there
    # alone: the schema and a refusal that concerns one point to it.
    fact = "acceptance_rules.freeze_gate.seed_replicates"
    assert "seed name" not in seeded.spec.description
    assert fact in str(seeded.spec.input_schema["properties"]["seed_replicates"]["description"])
    with pytest.raises(ToolError) as refused:
        seeded.invoke({"outcome": "freeze", "node_id": nominee})
    message = str(refused.value)
    assert "freeze_too_few_seed_replicates" in message and fact in message
    assert "seed name" not in message
    assert refused.value.details["passing_nodes"] == [node_handle(replicate)]
    assert '"seed_replicates"' in str(refused.value.retry_hint)

    calls.clear()
    finish = ToolRegistry([seeded]).invoke(
        "finish_session",
        {"outcome": "freeze", "node_id": "valid_a", "seed_replicates": ["valid_b"]},
    )
    assert finish.ok and finish.finish
    assert calls == [(nominee, (replicate,))]
    assert _seed_replicates(finish.value) == (replicate,)
    with pytest.raises(ToolError, match="so must seed_replicates"):
        seeded.invoke({"outcome": "no_edge", "reason": REASON, "seed_replicates": [replicate]})
