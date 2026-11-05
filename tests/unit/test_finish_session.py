"""``finish_session``: the two ways the research session ends.

The tool decides nothing about the market: it refuses a freeze that does not
register and an unexplained decision, ends the session on every freeze that
registers, whatever the node's readings, and hands the Pipeline an outcome it
records as ``freeze`` or ``no_edge``. It returns no reading of the sealed
years: the host reads them after the session has ended.
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
from autotrade.pipelines.research_session import (
    _deterministic,
    _seed_replicates,
    _session_outcome,
)

SESSION = "session_ref_ab"
RUN = "run_x"
REASON = "active excess is negative in three of four research years; the null percentile is 0.48"


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
    """The Pipeline's registration as the tool sees it: node id -> verdict.
    A node in ``passing`` registers; any other is refused, for its span."""

    def __init__(self, passing: set[str]) -> None:
        self.passing = passing

    def __call__(
        self,
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        deterministic: bool = False,
    ) -> dict[str, object]:
        if node_id in self.passing:
            return {"passed": True, "reasons": []}
        return {"passed": False, "reasons": ["freeze_needs_full_span_validation"]}


def _tool(tree: StepTree, gate: _Gate, **kwargs: object) -> FinishSessionTool:
    return FinishSessionTool(tree, session_ref=SESSION, registration=gate, **kwargs)  # type: ignore[arg-type]


def test_every_outcome_maps_to_the_pipeline_outcome_it_names(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    winner = _node(tree, tmp_path, "a")
    registry = ToolRegistry([_tool(tree, _Gate({winner}))])
    results = {
        "freeze": registry.invoke("finish_session", {"outcome": "freeze", "node_id": winner}),
        "no_edge": registry.invoke("finish_session", {"outcome": "no_edge", "reason": REASON}),
    }
    # The other two are the host's: a spent budget, a span it cannot replay.
    assert set(results) == set(SESSION_OUTCOMES) - {"deadline", "environment_blocked"}
    mapped = {name: _session_outcome(result.value) for name, result in results.items()}
    assert mapped == {
        "freeze": ("freeze", winner, ""),
        "no_edge": ("no_edge", None, REASON),
    }
    assert all(result.ok and result.finish for result in results.values())
    # The freeze returns no score of any kind: the sealed years are read by
    # the host after the session has ended, once.
    frozen = results["freeze"].value
    assert not {"freeze_gate", "registration", "residual_reading", "information_ratio"} & set(frozen)
    assert "once on the sealed years" in frozen["pipeline_outcome"]
    # The outcome is required; the retired outcomes (the multi-session
    # ``continue`` among them) are not in the enum.
    for arguments in (
        {},
        {"outcome": "continue", "node_id": winner, "reason": REASON},
        {"outcome": "terminate", "reason": REASON},
        {"outcome": "select"},
    ):
        assert registry.invoke("finish_session", arguments).ok is False


def test_a_nomination_that_does_not_register_is_refused_with_its_reasons_alone(tmp_path: Path):
    """A refusal names the registration's reasons and no number: registration
    reads no score, and no other node is offered as one that would pass."""

    tree = StepTree(tmp_path / "steps")
    weak = _node(tree, tmp_path, "weak")
    strong = _node(tree, tmp_path, "strong")
    finish = _tool(tree, _Gate({strong}))
    with pytest.raises(ToolError) as refused:
        finish.invoke({"outcome": "freeze", "node_id": weak})
    message = str(refused.value)
    assert refused.value.error_type == "freeze_registration_refused"
    assert "freeze_needs_full_span_validation" in message
    assert "=" not in message and node_handle(strong) not in message
    assert refused.value.details == {
        "registration": {"passed": False, "reasons": ["freeze_needs_full_span_validation"]}
    }
    # Nothing ended: the node that registers freezes and ends the session.
    assert finish.invoke({"outcome": "freeze", "node_id": strong}).finish


def test_a_freeze_that_registers_ends_the_session_whatever_its_readings(tmp_path: Path):
    """Registration is the only thing a freeze is checked for: a node whose
    replay lost money every year registers and ends the session as one that
    earned. The budget left is recorded with it, never a reason to refuse."""

    tree = StepTree(tmp_path / "steps")
    losing = _node(tree, tmp_path, "losing")
    budget = SessionBudgetStatus(inference_seconds_remaining=5400.0)
    frozen = _tool(tree, _Gate({losing}), budget_status=lambda: budget).invoke(
        {"outcome": "freeze", "node_id": losing}
    )
    assert frozen.ok and frozen.finish
    assert frozen.value["budget_at_finish"]["inference_seconds_remaining"] == 5400.0
    assert "reason" not in frozen.value
    # The retired early-finish account is no parameter, and no schema field
    # declares offline screens any more.
    properties = FinishSessionTool.spec.input_schema["properties"]
    assert set(properties) == {"outcome", "node_id", "seed_replicates", "deterministic", "reason"}


def test_no_edge_names_no_node_and_needs_a_reason_and_a_validation(tmp_path: Path):
    tree = StepTree(tmp_path / "steps")
    empty = _tool(tree, _Gate(set()))
    with pytest.raises(ToolError, match="at least one complete"):
        empty.invoke({"outcome": "no_edge", "reason": REASON})
    node = _node(tree, tmp_path, "a")
    with pytest.raises(ToolError, match="node_id and seed_replicates must be absent"):
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


def test_seed_replicates_are_resolved_and_go_to_the_registration(tmp_path: Path):
    """The tool resolves seed_replicates like node_id and hands them to the
    registration, which checks them. A nominee refused for want of them is
    told what a replicate is, and the freeze hands the Pipeline their full
    ids."""

    tree = StepTree(tmp_path / "steps")
    nominee = _node(tree, tmp_path, "a")
    replicate = _node(tree, tmp_path, "b")
    calls: list[tuple[str, tuple[str, ...]]] = []

    def gate(
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        deterministic: bool = False,
    ) -> dict[str, object]:
        calls.append((node_id, seed_replicates))
        if seed_replicates == (replicate,):
            return {"passed": True, "reasons": []}
        return {
            "passed": False,
            "reasons": ["freeze_too_few_seed_replicates"],
            "seed_replicates": {"trains_a_model": True, "replicates": []},
        }

    seeded = _tool(tree, gate)  # type: ignore[arg-type]
    assert "seed_replicates" in FinishSessionTool.spec.input_schema["properties"]
    # The rule a replicate meets is the arm's freeze_registration fact, stated
    # there alone: the schema and a refusal that concerns one point to it.
    fact = "acceptance_rules.freeze_registration.seed_replicates"
    assert "seed name" not in seeded.spec.description
    assert fact in str(seeded.spec.input_schema["properties"]["seed_replicates"]["description"])
    with pytest.raises(ToolError) as refused:
        seeded.invoke({"outcome": "freeze", "node_id": nominee})
    message = str(refused.value)
    assert "freeze_too_few_seed_replicates" in message and fact in message
    assert "seed name" not in message
    assert '"seed_replicates"' in str(refused.value.retry_hint)

    calls.clear()
    finish = ToolRegistry([seeded]).invoke(
        "finish_session",
        {"outcome": "freeze", "node_id": "valid_a", "seed_replicates": ["valid_b"]},
    )
    assert finish.ok and finish.finish
    assert calls == [(nominee, (replicate,))]
    assert _seed_replicates(finish.value) == (replicate,)
    with pytest.raises(ToolError, match="seed_replicates must be absent"):
        seeded.invoke({"outcome": "no_edge", "reason": REASON, "seed_replicates": [replicate]})


def test_a_deterministic_declaration_is_the_nominees_and_reaches_the_pipeline(
    tmp_path: Path,
):
    """deterministic goes to the registration with the nominee, and the
    freeze hands it to the Pipeline; a nomination that omits it is not
    deterministic."""

    tree = StepTree(tmp_path / "steps")
    nominee = _node(tree, tmp_path, "a")
    _node(tree, tmp_path, "b")
    calls: list[tuple[str, bool]] = []

    def gate(
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        deterministic: bool = False,
    ) -> dict[str, object]:
        calls.append((node_id, deterministic))
        return {
            "passed": deterministic,
            "reasons": [] if deterministic else ["freeze_too_few_seed_replicates"],
        }

    finish = _tool(tree, gate)  # type: ignore[arg-type]
    registry = ToolRegistry([finish])
    assert FinishSessionTool.spec.input_schema["properties"]["deterministic"]["type"] == "boolean"
    assert "deterministic=true" in FinishSessionTool.spec.description
    with pytest.raises(ToolError, match="freeze_too_few_seed_replicates"):
        finish.invoke({"outcome": "freeze", "node_id": nominee})
    assert calls == [(nominee, False)]
    refused = registry.invoke(
        "finish_session", {"outcome": "freeze", "node_id": nominee, "deterministic": "yes"}
    )
    assert refused.ok is False

    calls.clear()
    frozen = registry.invoke(
        "finish_session", {"outcome": "freeze", "node_id": "valid_a", "deterministic": True}
    )
    assert frozen.ok and frozen.finish
    assert calls == [(nominee, True)]
    assert frozen.value["deterministic"] is True and _deterministic(frozen.value) is True
