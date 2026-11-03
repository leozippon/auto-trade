"""End the research session: freeze a nominee, or end the arm without one."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from autotrade.environment.step_tree import StepTree, node_in_session

from .base import AGENT_JUSTIFICATION_MAX_CHARS, ToolError, ToolResult, ToolSpec
from .node_reference import (
    NODE_REFERENCE_DESCRIPTION,
    NODE_REFERENCE_EXAMPLE,
    session_handle,
    session_handles,
    session_node_reference,
)

# The outcomes the Agent may state; the Pipeline's SESSION_OUTCOMES adds the
# host-recorded ``deadline``.
FINISH_OUTCOMES = ("freeze", "no_edge")
# One ``reason`` serves every justification -- the evidence behind ending the
# arm -- and a one-word reason is none. Its upper bound is the shared one
# every Agent-written justification gets, so the reason and a candidate's
# hypothesis cannot drift apart.
REASON_MIN_CHARS = 40
REASON_MAX_CHARS = AGENT_JUSTIFICATION_MAX_CHARS

# Whether one node of this session registers for a freeze, with the nodes it
# would register as its seed replicates, whether the nomination declares its
# package deterministic and the node it would register as its ablation:
# ``passed``, ``reasons`` (named in ``pipelines/verdict.py`` and
# ``pipelines/experiment.py``) and what the registration read of the
# replicates and the ablation. Registration only, never a score. The
# Environment never imports the Pipeline, so the check is handed over as a
# callable; the Pipeline checks again when it records the freeze.
Registration = Callable[[str, tuple[str, ...], bool, str | None], Mapping[str, object]]

# The registration's reasons that concern a nominee's seed replicates: a
# refusal that names one points to the fact that defines a replicate.
SEED_REASONS = frozenset(
    {
        "freeze_seed_replicate_invalid",
        "freeze_too_few_seed_replicates",
    }
)
# Those that concern its ablation, which point to the fact that defines one.
ABLATION_REASONS = frozenset({"freeze_ablation_invalid", "freeze_ablation_missing"})


@dataclass(frozen=True)
class SessionBudgetStatus:
    """What the session has left when ``finish_session`` is called: recorded
    on the finish, never a reason to refuse one."""

    replay_years_remaining: int
    replay_years_total: int
    inference_seconds_remaining: float

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["inference_seconds_remaining"] = round(
            max(self.inference_seconds_remaining, 0.0), 1
        )
        return record


_DESCRIPTION = (
    "End the arm's research session with one outcome; no other session follows. "
    'outcome="freeze" nominates node_id, a complete Validation of this session '
    "that replayed the whole visible research period (span=full), and succeeds "
    "only if that node registers: every condition the "
    "acceptance_rules.freeze_registration fact lists for this arm. Registration "
    "reads no score: a nomination that does not register is refused with its "
    "named reasons and the session goes on, and one that registers ends the "
    "session at once. The host then reads the frozen node once on the sealed "
    "research years (acceptance_rules.sealed_gate); this call returns no reading "
    "and there is no second one. node_id is the full id or the node's short "
    "handle (valid_002); it may be omitted only while this session has exactly "
    "one complete Validation. deterministic=true declares that the nominee's "
    "package uses no randomness at all, so seed replicates would reproduce it "
    "exactly: registration then asks for none, and the declaration is recorded "
    "in the ledger for the operator's review before the strategy goes live. "
    "ablation names the nominee's ablation exactly where "
    "acceptance_rules.freeze_registration.ablation asks for one. "
    'outcome="no_edge" ends the arm without freezing a node and takes no node_id; '
    "it needs at least one complete Validation of this session and a reason "
    f"({REASON_MIN_CHARS}-{REASON_MAX_CHARS} chars), the evidence behind the "
    "decision. The reason is recorded on the arm's session record and is where "
    "the arm's conclusion belongs; a freeze may carry one too. No skill is "
    "needed to finish, and none can be written after this call. The call starts "
    "only after every background sub-agent has finished, is refused while one is "
    "still running, and once it succeeds the remaining tool calls of the turn "
    "are cancelled."
)


class FinishSessionTool:
    # The rule a replicate meets is the arm's
    # ``acceptance_rules.freeze_registration`` fact, which the parameter and a
    # refusal that concerns one point to.
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
                        "freeze: nominate node_id and end the session. no_edge: end "
                        "the arm without a deliverable."
                    ),
                },
                "node_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": NODE_REFERENCE_DESCRIPTION,
                },
                "seed_replicates": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 500},
                    "description": (
                        "freeze only: the nominee's seed replicates as "
                        "acceptance_rules.freeze_registration.seed_replicates "
                        "defines them, each a node id or short handle of this "
                        "session; required (at least one) when the nominee's "
                        "main.py defines fit, unless deterministic is true."
                    ),
                },
                "ablation": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": (
                        "freeze only: the nominee's ablation as "
                        "acceptance_rules.freeze_registration.ablation defines it, "
                        "a node id or short handle of this session; required "
                        "exactly where that fact asks for one, refused otherwise."
                    ),
                },
                "deterministic": {
                    "type": "boolean",
                    "description": (
                        "freeze only: true declares that the nominee's package "
                        "uses no randomness at all, so seed replicates would "
                        "reproduce it exactly; registration then asks for none, "
                        "and the declaration is recorded in the ledger for the "
                        "operator's review. Default false."
                    ),
                },
                "reason": {
                    "type": "string",
                    "minLength": REASON_MIN_CHARS,
                    "maxLength": REASON_MAX_CHARS,
                    "description": (
                        "Required for no_edge, optional for a freeze: the "
                        "evidence read and what stays untested. "
                        f"{REASON_MIN_CHARS}-{REASON_MAX_CHARS} characters."
                    ),
                },
            },
            "required": ["outcome"],
            "additionalProperties": False,
        },
        example={"outcome": "freeze", "node_id": NODE_REFERENCE_EXAMPLE},
    )

    def __init__(
        self,
        tree: StepTree,
        *,
        session_ref: str,
        registration: Registration,
        budget_status: Callable[[], SessionBudgetStatus] | None = None,
    ) -> None:
        self.tree = tree
        self.session_ref = session_ref
        self._registration = registration
        self._budget_status = budget_status

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        outcome = str(arguments.get("outcome") or "")
        reason = str(arguments.get("reason") or "").strip()
        node_id = str(arguments.get("node_id") or "")
        replicates = [str(item) for item in arguments.get("seed_replicates") or ()]  # type: ignore[union-attr]
        ablation = str(arguments.get("ablation") or "")
        if outcome == "freeze":
            deterministic = arguments.get("deterministic") is True
            return self._freeze(node_id, reason, replicates, deterministic, ablation)
        return self._no_edge(node_id, reason, replicates, ablation)

    # ---- outcomes ----

    def _freeze(
        self,
        reference: str,
        reason: str,
        replicate_refs: list[str],
        deterministic: bool,
        ablation_ref: str,
    ) -> ToolResult:
        node_id = (
            self._reference(reference) if reference else self._sole_candidate()
        )
        node = self._complete_node(node_id)
        replicates = tuple(self._reference(item) for item in replicate_refs)
        for replicate in replicates:
            self._complete_node(replicate)
        ablation = self._reference(ablation_ref) if ablation_ref else None
        if ablation is not None:
            self._complete_node(ablation)
        if reason:
            self._require_reason("freeze", reason)
        registration = dict(self._registration(node_id, replicates, deterministic, ablation))
        if not registration.get("passed"):
            reasons = [str(item) for item in registration.get("reasons") or ()]
            seeds = SEED_REASONS.intersection(reasons)
            control = ABLATION_REASONS.intersection(reasons)
            raise ToolError(
                f"finish_session refused: {self._named([node_id])[0]} does not register "
                f"for a freeze ({', '.join(reasons) or 'no reason recorded'}). "
                + (
                    f"{self._seed_text(registration)} A seed replicate is what "
                    "acceptance_rules.freeze_registration.seed_replicates defines. "
                    if seeds
                    else ""
                )
                + (
                    f"{self._ablation_text(registration)} An ablation is what "
                    "acceptance_rules.freeze_registration.ablation defines. "
                    if control
                    else ""
                )
                + "Nominate a node that registers, validate what registration lacks "
                "(a full-span validation, a seed replicate, an ablation), or finish "
                "with no_edge.",
                error_type="freeze_registration_refused",
                retry_hint=(
                    'finish_session({"outcome": "freeze", "node_id": "<nominee>", '
                    '"seed_replicates": ["<replicate>"]})'
                    if seeds
                    else 'finish_session({"outcome": "no_edge", "reason": "<evidence>"})'
                ),
                details={"registration": self._registration_record(registration)},
            )
        self.tree.set_position(node_id)
        return ToolResult(
            True,
            value={
                "status": "session_finished",
                "outcome": "freeze",
                "node_id": node_id,
                **session_handle(self.tree, node_id, session_ref=self.session_ref),
                "revision_id": str(node["revision_id"]),
                **({"seed_replicates": list(replicates)} if replicates else {}),
                "deterministic": deterministic,
                **({"ablation": ablation} if ablation is not None else {}),
                **({"reason": reason} if reason else {}),
                **self._budget(),
                "pipeline_outcome": (
                    f"The Pipeline freezes {node_id} as the arm's artifact"
                    + (" with its seed replicates" if replicates else "")
                    + (" and its ablation" if ablation is not None else "")
                    + "; research ends, no further session runs, and the host reads "
                    "the frozen artifact once on the sealed years."
                ),
            },
            finish=True,
        )

    def _no_edge(
        self, node_id: str, reason: str, replicates: list[str], ablation: str
    ) -> ToolResult:
        if node_id or replicates or ablation:
            raise ToolError(
                'finish_session: outcome="no_edge" nominates nothing, so node_id, '
                "seed_replicates and ablation must be absent",
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
                **self._budget(),
                "pipeline_outcome": (
                    "The arm ends without a frozen artifact: no sealed reading and no "
                    "forward test."
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

    def _reference(self, reference: str) -> str:
        return session_node_reference(
            "finish_session",
            self.tree,
            reference,
            session_ref=self.session_ref,
            offered=self._session_candidates(),
        )

    def _registration_record(self, registration: Mapping[str, object]) -> dict[str, object]:
        """The registration's verdict and what it read of the seed replicates,
        named by their handles."""

        record: dict[str, object] = {
            "passed": bool(registration.get("passed")),
            "reasons": [str(item) for item in registration.get("reasons") or ()],
        }
        block = registration.get("seed_replicates")
        if isinstance(block, Mapping):
            record["seed_replicates"] = {
                **block,
                "replicates": [
                    {**entry, "step_id": self._named([str(entry["step_id"])])[0]}
                    for entry in block.get("replicates") or ()  # type: ignore[union-attr]
                ],
            }
        ablation = registration.get("ablation")
        if isinstance(ablation, Mapping):
            record["ablation"] = {
                **ablation,
                **(
                    {"step_id": self._named([str(ablation["step_id"])])[0]}
                    if "step_id" in ablation
                    else {}
                ),
            }
        return record

    def _ablation_text(self, registration: Mapping[str, object]) -> str:
        """What registration read off the nominee's ablation, in one sentence."""

        block = registration.get("ablation")
        block = block if isinstance(block, Mapping) else {}
        if "step_id" not in block:
            return "Ablation: none named, and this arm requires one."
        named = self._named([str(block["step_id"])])[0]
        return f"Ablation: {named} {block.get('problem') or 'registers'}."

    def _seed_text(self, registration: Mapping[str, object]) -> str:
        """What registration read off the nominee's seed replicates, in one sentence."""

        block = registration.get("seed_replicates")
        block = block if isinstance(block, Mapping) else {}
        parts = [
            f"{self._named([str(entry['step_id'])])[0]} "
            + (str(entry["problem"]) if "problem" in entry else f"registers ({entry['seed_line']})")
            for entry in block.get("replicates") or ()  # type: ignore[union-attr]
        ]
        return (
            "Seed replicates: "
            + ("; ".join(parts) if parts else "none named, and the nominee trains a model")
            + "."
        )

    def _session_candidates(self) -> list[str]:
        return [
            str(node["node_id"])
            for node in self.tree.nodes()
            if node_in_session(node, session_ref=self.session_ref)
            and node.get("complete_validation")
            and node.get("revision_id")
        ]

    def _named(self, node_ids: list[str]) -> list[str]:
        """Nodes as the Agent should name them: the short handle where unambiguous."""

        return session_handles(self.tree, node_ids, session_ref=self.session_ref)

    def _sole_candidate(self) -> str:
        candidates = self._session_candidates()
        if len(candidates) == 1:
            return candidates[0]
        named = self._named(candidates)
        raise ToolError(
            f"finish_session requires node_id: this session has {len(candidates)} "
            "complete Validations"
            + (f" ({', '.join(named)})" if named else ""),
            retry_hint=(
                f'finish_session({{"outcome": "freeze", "node_id": "{NODE_REFERENCE_EXAMPLE}"}})'
            ),
            details={"candidates": named},
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

    def _budget(self) -> dict[str, object]:
        """The budget left at the finish, recorded with it; nothing without a
        wired budget."""

        status = self._budget_status() if self._budget_status is not None else None
        return {} if status is None else {"budget_at_finish": status.to_record()}


__all__ = [
    "ABLATION_REASONS",
    "FINISH_OUTCOMES",
    "REASON_MAX_CHARS",
    "REASON_MIN_CHARS",
    "SEED_REASONS",
    "FinishSessionTool",
    "Registration",
    "SessionBudgetStatus",
]
