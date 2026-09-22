"""End the research session: freeze a nominee, or end the arm without one."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from autotrade.environment.step_tree import StepTree, node_in_session

from .base import AGENT_JUSTIFICATION_MAX_CHARS, ToolError, ToolResult, ToolSpec

# The outcomes the Agent may state; the Pipeline's SESSION_OUTCOMES adds the
# host-recorded ``deadline``.
FINISH_OUTCOMES = ("freeze", "no_edge")
# A voluntary freeze that leaves more than this share of the replay-year budget
# unused must say why. It is a justification, not a block, and it lapses once
# another batch cannot fit.
EARLY_FINISH_BUDGET_FRACTION = 1 / 3
# One ``reason`` serves every justification -- the evidence behind ending the
# arm, and the account of an early freeze -- and a one-word reason is none of
# them. Its upper bound is the shared one every Agent-written justification
# gets, so the reason and a candidate's hypothesis cannot drift apart.
REASON_MIN_CHARS = 40
REASON_MAX_CHARS = AGENT_JUSTIFICATION_MAX_CHARS

# The freeze gate as the Pipeline would read one node of this session now:
# ``passed``, ``reasons`` (named in ``pipelines/verdict.py``) and the gate's
# numbers. The Environment never imports the Pipeline, so the gate is handed
# over as a callable; the Pipeline recomputes it when it records the freeze.
FreezeGate = Callable[[str], Mapping[str, object]]


@dataclass(frozen=True)
class SessionBudgetStatus:
    """What the session has left when ``finish_session`` is called."""

    replay_years_remaining: int
    replay_years_total: int
    inference_seconds_remaining: float

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["inference_seconds_remaining"] = round(
            max(self.inference_seconds_remaining, 0.0), 1
        )
        return record

    @property
    def early_finish(self) -> bool:
        return (
            self.replay_years_total > 0
            and self.replay_years_remaining
            > self.replay_years_total * EARLY_FINISH_BUDGET_FRACTION
        )


_DESCRIPTION = (
    "End the arm's research session with one outcome; no other session follows. "
    'outcome="freeze" nominates node_id, a complete Validation of this session '
    "that replayed the whole research period (span=full), and passes only the "
    "freeze gate the acceptance_rules fact states, thresholds included: at least "
    "two full-span validations in the arm, and on the nominee's active series "
    "(its return minus the host's zero-skill panel) the information ratio, its "
    "deflated Sharpe probability with trials counted over every revision the arm "
    "has validated on any span, the share of research years with a positive "
    "excess and the drawdown limit, plus the equity drawdown limit and the "
    "tracking mandate when the arm has one. A nomination that fails the gate "
    "is refused with its named reasons and numbers and the session goes on; a "
    "freeze ends research for the whole arm, and the frozen artifact is then "
    "tested once on later data no session sees. node_id may be omitted only while "
    "this session has exactly one complete Validation. "
    'outcome="no_edge" ends the arm without a deliverable and takes no node_id; '
    "it needs at least one complete Validation of this session and a reason "
    f"({REASON_MIN_CHARS}-{REASON_MAX_CHARS} chars), the evidence behind the "
    "decision; a freeze needs a reason only when it leaves more than a third of "
    "the replay-year budget while another batch still fits. The reason is "
    "recorded on the arm's session record; write any skills before this call. "
    "The call starts only after every background sub-agent has finished, is "
    "refused while one is still running, and once it succeeds the remaining tool "
    "calls of the turn are cancelled."
)


class FinishSessionTool:
    spec = ToolSpec(
        "finish_session",
        _DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": list(FINISH_OUTCOMES),
                    "description": (
                        "freeze: nominate node_id for the freeze gate. no_edge: end "
                        "the arm without a deliverable."
                    ),
                },
                "node_id": {"type": "string", "minLength": 1, "maxLength": 500},
                "reason": {
                    "type": "string",
                    "minLength": REASON_MIN_CHARS,
                    "maxLength": REASON_MAX_CHARS,
                    "description": (
                        "Required for no_edge: the evidence read (neutralized excess "
                        "and its yearly blocks, null percentile, deflated Sharpe with "
                        "its trials) and what stays untested. "
                        "Required for a freeze only when it leaves more than a third "
                        f"of the replay-year budget. {REASON_MIN_CHARS}-"
                        f"{REASON_MAX_CHARS} characters."
                    ),
                },
            },
            "required": ["outcome"],
            "additionalProperties": False,
        },
        example={"outcome": "freeze", "node_id": "<complete full-span Validation node_id>"},
    )

    def __init__(
        self,
        tree: StepTree,
        *,
        session_ref: str,
        freeze_gate: FreezeGate,
        another_round_fits: Callable[[], bool] | None = None,
        budget_status: Callable[[], SessionBudgetStatus] | None = None,
    ) -> None:
        self.tree = tree
        self.session_ref = session_ref
        self._freeze_gate = freeze_gate
        self._another_round_fits = another_round_fits or (lambda: True)
        self._budget_status = budget_status

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        outcome = str(arguments.get("outcome") or "")
        reason = str(arguments.get("reason") or "").strip()
        node_id = str(arguments.get("node_id") or "")
        if outcome == "freeze":
            return self._freeze(node_id, reason)
        return self._no_edge(node_id, reason)

    # ---- outcomes ----

    def _freeze(self, node_id: str, reason: str) -> ToolResult:
        node_id = node_id or self._sole_candidate()
        node = self._complete_node(node_id)
        gate = dict(self._freeze_gate(node_id))
        if not gate.get("passed"):
            reasons = [str(item) for item in gate.get("reasons") or ()]
            passing = [
                candidate
                for candidate in self._session_candidates()
                if candidate != node_id and self._freeze_gate(candidate).get("passed")
            ]
            raise ToolError(
                f"finish_session refused: {node_id} fails the freeze gate "
                f"({', '.join(reasons) or 'no reason recorded'}); "
                f"{_gate_numbers(gate)}. "
                + (
                    f"These nodes of this session pass it now: {', '.join(passing)}. "
                    if passing
                    else "No node of this session passes it now. "
                )
                + "Nominate a passing node, validate what the gate lacks (a "
                "full-span validation, a control), or finish with no_edge.",
                error_type="freeze_gate_refused",
                retry_hint='finish_session({"outcome": "no_edge", "reason": "<evidence>"})',
                details={"freeze_gate": _gate_record(gate), "passing_nodes": passing},
            )
        budget = self._early_finish_budget(reason)
        self.tree.set_position(node_id)
        return ToolResult(
            True,
            value={
                "status": "session_finished",
                "outcome": "freeze",
                "node_id": node_id,
                "revision_id": str(node["revision_id"]),
                **({"reason": reason} if reason else {}),
                **budget,
                "freeze_gate": _gate_record(gate),
                "pipeline_outcome": (
                    f"The Pipeline freezes {node_id} as the arm's artifact; research "
                    "ends and no further session runs."
                ),
            },
            finish=True,
        )

    def _no_edge(self, node_id: str, reason: str) -> ToolResult:
        if node_id:
            raise ToolError(
                'finish_session: outcome="no_edge" nominates nothing, so node_id must '
                "be absent",
                retry_hint='finish_session({"outcome": "no_edge", "reason": "<evidence>"})',
            )
        self._require_reason("no_edge", reason)
        candidates = self._session_candidates()
        if not candidates:
            raise ToolError(
                'finish_session: outcome="no_edge" needs at least one complete '
                "Validation of this session to have found no edge in; run "
                "batch_validate first"
            )
        return ToolResult(
            True,
            value={
                "status": "session_finished",
                "outcome": "no_edge",
                "reason": reason,
                "candidates_evaluated": len(candidates),
                **self._budget_record(),
                "pipeline_outcome": (
                    "The arm ends without a deliverable: no forward test."
                ),
            },
            finish=True,
        )

    # ---- checks ----

    def _require_reason(self, outcome: str, reason: str) -> None:
        if len(reason) < REASON_MIN_CHARS:
            raise ToolError(
                f'finish_session: outcome="{outcome}" requires reason (between '
                f"{REASON_MIN_CHARS} and {REASON_MAX_CHARS} chars) citing the "
                "evidence behind it; it is recorded on the arm's session record",
                retry_hint=f'finish_session({{"outcome": "{outcome}", "reason": "<evidence>"}})',
            )

    def _session_candidates(self) -> list[str]:
        return [
            str(node["node_id"])
            for node in self.tree.nodes()
            if node_in_session(node, session_ref=self.session_ref)
            and node.get("complete_validation")
            and node.get("revision_id")
        ]

    def _sole_candidate(self) -> str:
        candidates = self._session_candidates()
        if len(candidates) == 1:
            return candidates[0]
        raise ToolError(
            f"finish_session requires node_id: this session has {len(candidates)} "
            "complete Validations"
            + (f" ({', '.join(candidates)})" if candidates else ""),
            retry_hint='finish_session({"outcome": "freeze", "node_id": "<node_id>"})',
            details={"candidates": candidates},
        )

    def _complete_node(self, node_id: str) -> Mapping[str, object]:
        try:
            node = self.tree.get_node(node_id)
        except ValueError as exc:
            raise ToolError(f"finish_session: {node_id} is not a Step node") from exc
        if not node_in_session(node, session_ref=self.session_ref):
            raise ToolError(f"finish_session: {node_id} is not a Step of this session")
        if not node.get("complete_validation") or not node.get("revision_id"):
            raise ToolError(f"finish_session: {node_id} is not a complete Validation")
        if not (self.tree.node_output_dir(node_id) / "main.py").is_file():
            raise ToolError(
                f"finish_session: {node_id} has no strategy snapshot to hand on"
            )
        return node

    def _budget_record(self) -> dict[str, object]:
        status = self._budget_status() if self._budget_status is not None else None
        return {} if status is None else {"budget_at_finish": status.to_record()}

    def _early_finish_budget(self, reason: str) -> dict[str, object]:
        """The budget left at a freeze, refusing an early one that gives no reason.

        Early means more than a third of the replay-year budget is left while
        another batch still fits. Without a wired budget nothing is refused.
        """

        status = self._budget_status() if self._budget_status is not None else None
        if status is None:
            return {}
        recorded: dict[str, object] = {"budget_at_finish": status.to_record()}
        if reason or not status.early_finish or not self._another_round_fits():
            return recorded
        minutes = max(status.inference_seconds_remaining, 0.0) / 60
        raise ToolError(
            "finish_session refused: this freeze leaves "
            f"{status.replay_years_remaining}/{status.replay_years_total} replay-years "
            f"and about {minutes:.0f} min of inference time unused, more than a third "
            "of the replay-year budget. Freezing ends research for the whole arm: "
            "either run the rounds that still test the nominee, or call "
            f"finish_session again with reason ({REASON_MIN_CHARS}-{REASON_MAX_CHARS} "
            "chars) naming the hypotheses that stay untested and why the remaining "
            "budget is better left unused.",
            retry_hint='finish_session({"outcome": "freeze", "node_id": ..., "reason": "..."})',
            details=status.to_record(),
        )


def _gate_record(gate: Mapping[str, object]) -> dict[str, object]:
    """The gate's verdict and numbers, bounded for an observation."""

    dsr = gate.get("deflated_sharpe")
    dsr = dsr if isinstance(dsr, Mapping) else {}
    record = {
        "passed": bool(gate.get("passed")),
        "reasons": [str(item) for item in gate.get("reasons") or ()],
        "deflated_sharpe_probability": dsr.get("deflated_sharpe_probability"),
        "trials": dsr.get("trials"),
        "effective_trials": dsr.get("effective_trials"),
        "information_ratio_bar": dsr.get("information_ratio_bar"),
        "full_span_validations": gate.get("full_span_validations"),
        "information_ratio": gate.get("information_ratio"),
        "neutralized_excess": gate.get("neutralized_excess"),
        "unavailable_reason": dsr.get("unavailable_reason"),
        "thresholds": gate.get("thresholds"),
    }
    return {key: value for key, value in record.items() if value is not None}


def _gate_numbers(gate: Mapping[str, object]) -> str:
    record = _gate_record(gate)
    parts = [
        f"{name}={record[name]:.4g}" if isinstance(record[name], float) else f"{name}={record[name]}"
        for name in (
            "deflated_sharpe_probability",
            "trials",
            "effective_trials",
            "full_span_validations",
            "information_ratio",
            "information_ratio_bar",
        )
        if name in record
    ]
    return "readings: " + (", ".join(parts) if parts else "none measurable")


__all__ = [
    "EARLY_FINISH_BUDGET_FRACTION",
    "FINISH_OUTCOMES",
    "REASON_MAX_CHARS",
    "REASON_MIN_CHARS",
    "FinishSessionTool",
    "FreezeGate",
    "SessionBudgetStatus",
]
