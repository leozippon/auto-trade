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
    OFFLINE_TRIALS_MAX,
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

    def __call__(
        self,
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        offline_trials: int = 0,
        deterministic: bool = False,
    ) -> dict[str, object]:
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
    # The other two are the host's: a spent budget, a span it cannot replay.
    assert set(results) == set(SESSION_OUTCOMES) - {"deadline", "environment_blocked"}
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


def test_an_early_no_edge_owes_the_same_account_as_an_early_freeze(tmp_path: Path):
    """Ending with no_edge also ends research for the whole arm, so with two
    thirds of the replay years left it is refused, naming the early-finish
    rule, until a reason says what stays untested; later the reason is still
    required, as the evidence of the decision."""
    tree = StepTree(tmp_path / "steps")
    _node(tree, tmp_path, "a")

    def status(remaining: int) -> SessionBudgetStatus:
        return SessionBudgetStatus(
            replay_years_remaining=remaining,
            replay_years_total=24,
            inference_seconds_remaining=5400.0,
        )

    early = _tool(tree, _Gate(set()), budget_status=lambda: status(16))
    with pytest.raises(ToolError, match="more than a third") as refused:
        early.invoke({"outcome": "no_edge"})
    message = str(refused.value)
    assert "this no_edge leaves 16/24 replay-years" in message and "90 min" in message
    assert "naming the hypotheses that stay untested" in message
    assert '"outcome": "no_edge"' in str(refused.value.retry_hint)
    explained = early.invoke({"outcome": "no_edge", "reason": REASON})
    assert explained.finish and explained.value["budget_at_finish"]["replay_years_remaining"] == 16
    # Once no batch fits, the early rule lapses but the evidence is still owed.
    late = _tool(tree, _Gate(set()), budget_status=lambda: status(16), another_round_fits=lambda: False)
    with pytest.raises(ToolError, match="requires reason"):
        late.invoke({"outcome": "no_edge"})


def test_seed_replicates_are_resolved_and_go_to_the_gate(tmp_path: Path):
    """The tool resolves seed_replicates like node_id and hands them to the
    gate, which judges them. A nominee refused for want of them is told what
    a replicate is, a node that fails only for want of them is listed as
    passing, and the freeze hands the Pipeline their full ids."""

    tree = StepTree(tmp_path / "steps")
    nominee = _node(tree, tmp_path, "a")
    replicate = _node(tree, tmp_path, "b")
    calls: list[tuple[str, tuple[str, ...], int]] = []

    def gate(
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        offline_trials: int = 0,
        deterministic: bool = False,
    ) -> dict[str, object]:
        calls.append((node_id, seed_replicates, offline_trials))
        if seed_replicates == (replicate,):
            return {"passed": True, "reasons": []}
        return {
            "passed": False,
            "reasons": ["freeze_too_few_seed_replicates"],
            "seed_replicates": {"trains_a_model": True, "replicates": []},
        }

    seeded = _tool(tree, gate)  # type: ignore[arg-type]
    assert "seed_replicates" in FinishSessionTool.spec.input_schema["properties"]
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
    assert calls == [(nominee, (replicate,), 0)]
    assert _seed_replicates(finish.value) == (replicate,)
    with pytest.raises(ToolError, match="so must seed_replicates"):
        seeded.invoke({"outcome": "no_edge", "reason": REASON, "seed_replicates": [replicate]})


def test_a_deterministic_declaration_is_the_nominees_and_reaches_the_pipeline(
    tmp_path: Path,
):
    """deterministic goes to the gate with the nominee alone, never with the
    alternates a refusal lists, and the freeze hands it to the Pipeline; a
    nomination that omits it is not deterministic."""

    tree = StepTree(tmp_path / "steps")
    nominee = _node(tree, tmp_path, "a")
    other = _node(tree, tmp_path, "b")
    calls: list[tuple[str, bool]] = []

    def gate(
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        offline_trials: int = 0,
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
    assert calls == [(nominee, False), (other, False)]
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


def test_offline_trials_reach_the_gate_before_it_verdicts(tmp_path: Path):
    """The terminal offline-screen delta is read by the gate before its
    verdict: a count that pushes the nominee over a bar is refused, an
    alternate candidate is compared at the same count, and the accepted
    freeze carries the count the Pipeline persists."""

    tree = StepTree(tmp_path / "steps")
    weak = _node(tree, tmp_path, "weak")
    strong = _node(tree, tmp_path, "strong")
    seen: list[tuple[str, tuple[str, ...], int]] = []

    def gate(
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        offline_trials: int = 0,
        deterministic: bool = False,
    ) -> dict[str, object]:
        seen.append((node_id, seed_replicates, offline_trials))
        passed = node_id == strong and offline_trials < 10
        return {
            "passed": passed,
            "reasons": [] if passed else ["freeze_deflated_sharpe_below_threshold"],
            "deflated_sharpe": {"trials": 5 + offline_trials},
        }

    finish = _tool(tree, gate)  # type: ignore[arg-type]
    with pytest.raises(ToolError) as refused:
        finish.invoke({"outcome": "freeze", "node_id": weak, "offline_trials": 5})
    assert "trials=10" in str(refused.value)
    # The alternate candidate the refusal lists was read at the same count.
    assert node_handle(strong) in str(refused.value)
    assert seen[-1] == (strong, (), 5)

    with pytest.raises(ToolError) as over:
        finish.invoke({"outcome": "freeze", "node_id": strong, "offline_trials": 12})
    assert "trials=17" in str(over.value)

    frozen = finish.invoke({"outcome": "freeze", "node_id": strong, "offline_trials": 5})
    assert frozen.value["terminal_offline_trials"] == 5


def test_offline_trials_is_optional_and_never_assumed(tmp_path: Path):
    """The parameter is optional and defaults to 0; an explicitly null or
    malformed count is refused rather than read as none. Both outcomes carry
    the count, so the Pipeline persists it whether the arm froze or ended."""

    tree = StepTree(tmp_path / "steps")
    node = _node(tree, tmp_path, "a")
    calls: list[int] = []

    def gate(
        node_id: str,
        seed_replicates: tuple[str, ...] = (),
        offline_trials: int = 0,
        deterministic: bool = False,
    ) -> dict[str, object]:
        calls.append(offline_trials)
        return {"passed": True, "reasons": []}

    finish = _tool(tree, gate)  # type: ignore[arg-type]
    assert finish.invoke({"outcome": "freeze", "node_id": node}).value["terminal_offline_trials"] == 0
    assert finish.invoke({"outcome": "no_edge", "reason": REASON}).value["terminal_offline_trials"] == 0
    assert calls == [0]

    for invalid in (None, True, False, 1.5, -1, OFFLINE_TRIALS_MAX + 1):
        with pytest.raises(ToolError) as bad:
            finish.invoke({"outcome": "no_edge", "reason": REASON, "offline_trials": invalid})
        assert bad.value.error_type == "schema_error"
        assert bad.value.blocked_target == "offline_trials"

    at_cap = finish.invoke(
        {"outcome": "no_edge", "reason": REASON, "offline_trials": OFFLINE_TRIALS_MAX}
    )
    assert at_cap.value["terminal_offline_trials"] == OFFLINE_TRIALS_MAX
