"""Select one immutable, fully evaluated Step as the Fold result."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from autotrade.environment.runtime import redact_host_paths
from autotrade.environment.step_tree import StepTree, node_in_session

from .base import ToolError, ToolResult, ToolSpec

# A voluntary finish that leaves more than this share of the backtest budget
# unused must say why. Reviewed Folds finished at 27-50 % backtest usage with
# open hypotheses listed, and Meta only caught it afterwards; the reason is
# recorded with the Fold result so the review sees the Agent's own
# justification. It is a justification, not a block, and it lapses once another
# round cannot fit.
FINISH_FOLD_EARLY_STOP_BUDGET_FRACTION = 1 / 3
EARLY_STOP_REASON_MAX_CHARS = 500

# The Pipeline's hard acceptance rules, handed over as a callable that maps one
# node's recorded metrics to its hard-reject reasons (empty = passes). The
# Environment never imports the Pipeline; the rules live there and the freeze
# decision stays there, but a nomination the Pipeline will certainly reject has
# to be visible to the session that makes it — reviewed Folds nominated a node
# the rules rejected, were recorded ``baseline_missing``, and only learned it in
# the next Meta session while a sibling node would have frozen. Which rules are
# hard is the Pipeline's to say; the tool only reports what the callable returns.
HardRuleCheck = Callable[[Mapping[str, object]], Sequence[str]]


@dataclass(frozen=True)
class FoldBudgetStatus:
    """What the Fold session has left when ``finish_fold`` is called.

    The Pipeline owns the budget counters; it hands them over through a
    callable so the tool can decide whether a finish is early and tell the
    Agent exactly what it is leaving unused.
    """

    backtests_remaining: int
    backtests_total: int
    steps_remaining: int
    steps_total: int
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
            self.backtests_total > 0
            and self.backtests_remaining
            > self.backtests_total * FINISH_FOLD_EARLY_STOP_BUDGET_FRACTION
        )


def executable_source_structure(source: str) -> str:
    """Return the directly comparable executable structure of one module.

    Comments, module/class/function docstrings, and whitespace are ignored so
    a comment-only harvest has the parent's structure, while a logic or signal
    change does not.
    """

    tree = ast.parse(source)
    _strip_docstrings(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def executable_output_structure(root: Path) -> str:
    """Structure of a whole strategy package: every ``.py`` below ``root``.

    A helper module edited while ``main.py`` stays the same is a different
    hypothesis, so the comparison covers the package, keyed by relative path.
    """

    parts = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        parts.append(
            f"{relative.as_posix()}\0{executable_source_structure(path.read_text(encoding='utf-8'))}"
        )
    return "\n".join(parts)


def _node_metrics(node: Mapping[str, object]) -> Mapping[str, object]:
    metrics = node.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _tree_bytes(root: Path | None) -> dict[str, bytes]:
    if root is None or not root.is_dir():
        return {}
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


class FinishFoldTool:
    spec = ToolSpec(
        "finish_fold",
        "Finish this Fold by nominating one complete Validation node of the current "
        "run (its node_id comes from daily_backtest or a batch_validate row). Pass "
        "node_id explicitly: after a batch_validate round the tree position is the "
        "round's parent, so a bare call there is refused instead of silently keeping "
        "the parent. Outside the deadline window a voluntary finish that leaves more "
        "than a third of the backtest budget unused must carry early_stop_reason "
        "(which hypotheses stay untested and why they are not worth the remaining "
        "budget); the reason is recorded with the Fold result for the Meta review. "
        "The nominated node is also checked against the Pipeline's hard acceptance "
        "rules (the acceptance_rules fact marks which rules are hard and which only "
        "warn): outside the deadline "
        "window a breaching node is refused while another recorded node still passes, "
        "and the refusal lists which ones do; inside the window, or when nothing "
        "recorded passes, the nomination is accepted and the result states that the "
        "Pipeline will not freeze it. The call starts only after every background "
        "sub-agent has finished, is refused while one is still running, and once it "
        "succeeds the remaining tool calls of this assistant turn are cancelled.",
        {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "minLength": 1, "maxLength": 500},
                "early_stop_reason": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": EARLY_STOP_REASON_MAX_CHARS,
                    "description": (
                        "Why this Fold stops while more than a third of its backtest "
                        "budget remains: the untested hypotheses and why the remaining "
                        "budget is better left unused. Required only in that case."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        example={"node_id": "<complete Validation node_id>"},
    )

    def __init__(
        self,
        tree: StepTree,
        *,
        fold_id: str,
        run_id: str,
        parent_main_py: str | Path | None = None,
        current_output: str | Path | None = None,
        current_models: str | Path | None = None,
        another_round_fits: Callable[[], bool] | None = None,
        budget_status: Callable[[], FoldBudgetStatus] | None = None,
        hard_rule_check: HardRuleCheck | None = None,
    ) -> None:
        self.tree = tree
        self.fold_id = fold_id
        self.run_id = run_id
        self._current_output = Path(current_output) if current_output is not None else None
        self._current_models = Path(current_models) if current_models is not None else None
        # The early-stop justification and the hard-rule refusal apply only
        # while the session could still run a round; the caller says whether
        # time and budget allow one, and (when wired) what is left of each
        # budget.
        self._another_round_fits = another_round_fits or (lambda: True)
        self._budget_status = budget_status
        self._hard_rule_check = hard_rule_check
        self._parent_structure: str | None = None
        if parent_main_py is not None:
            # The parent package is the directory that holds its main.py.
            path = Path(parent_main_py)
            if not path.is_file():
                raise ValueError(f"parent strategy structure is invalid: missing {path.name}")
            try:
                self._parent_structure = executable_output_structure(path.parent)
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"parent strategy structure is invalid: {exc}") from exc

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        node_id = self._resolve_node_id(arguments)
        try:
            node = self.tree.get_node(node_id)
        except ValueError as exc:
            raise ToolError("finish_fold cannot select an absent Step") from exc
        if not node_in_session(node, fold_id=self.fold_id, run_id=self.run_id):
            raise ToolError("finish_fold can select only a Step from the current Fold session")
        if not node.get("complete_validation") or not node.get("revision_id"):
            raise ToolError("finish_fold requires successful complete validation")
        # Ahead of the working-copy check: a node the Pipeline would reject has
        # to be replaced rather than restored, so the refusal that names the
        # passing nodes must not cost a step_rollback to the wrong one first.
        acceptance = self._check_hard_acceptance(node_id, node)
        # The working-copy check comes before the budget gates so a winner
        # nominated without step_rollback costs one refusal, not two.
        self._require_current_matches_revision(node_id)
        early_stop = self._require_early_stop_reason(arguments)
        nominated_structure = self._node_structure(node_id)
        if self._parent_structure is not None:
            self._require_different_hypothesis(node_id, nominated_structure)
        self.tree.set_position(node_id)
        return ToolResult(
            True,
            value={
                "node_id": node_id,
                "revision_id": str(node["revision_id"]),
                "status": "fold_finished",
                # Candidate selection, AcceptanceRules and the final freeze are
                # the Pipeline's, not the Agent's: finishing only nominates.
                "fold_status": "pending_pipeline_review",
                "write_locked": True,
                # The Agent's own account of an early finish and the budget it
                # left, for the fold ledger and the Meta review.
                **early_stop,
                # Present only when the Pipeline will reject this nomination.
                **acceptance,
            },
            finish=True,
        )

    def _check_hard_acceptance(
        self, node_id: str, node: Mapping[str, object]
    ) -> dict[str, object]:
        """Hard acceptance verdict for the nominated node.

        Empty when the node passes, or when no rules are wired. A breach is
        refused while another recorded node passes and another round still
        fits, so the session can select that one or run a risk-reduced round
        instead of learning in the next Meta session that its Fold froze
        nothing. Inside the deadline window, or with nothing recorded that
        passes, the nomination is accepted and the record states what the
        Pipeline will do with it — the session's ``early_stop_reason`` and the
        Meta review then read the same outcome the fold ledger will carry.
        """

        check = self._hard_rule_check
        if check is None:
            return {}
        reasons = [str(reason) for reason in check(_node_metrics(node))]
        if not reasons:
            return {}
        candidates = self._hard_rule_candidates(check)
        passing = [
            row
            for row in candidates
            if row["node_id"] != node_id and row["passes_hard_rules"]
        ]
        if passing and self._another_round_fits():
            listed = "; ".join(
                f"{row['node_id']} ({row['result_name']}"
                + (
                    f", max_drawdown={row['max_drawdown']:.4f})"
                    if isinstance(row.get("max_drawdown"), float)
                    else ")"
                )
                for row in passing
            )
            raise ToolError(
                f"finish_fold refused: {node_id} fails the Pipeline's hard "
                f"acceptance rules ({', '.join(reasons)}), so the Fold would "
                "freeze nothing. These recorded nodes pass them: "
                f"{listed}. Select one of those (step_rollback to it first), or "
                "pre-register a risk-reduced round and run batch_validate.",
                error_type="acceptance_hard_reject",
                retry_hint=(
                    "step_rollback(<passing node_id>) then "
                    "finish_fold({\"node_id\": <passing node_id>})"
                ),
                details={
                    "hard_reject_reasons": reasons,
                    "candidates": candidates,
                },
            )
        return {
            "acceptance_hard_reject_reasons": reasons,
            # The fold status the Pipeline will record, in its own words: it
            # falls back to the inherited parent, or — with no parent to fall
            # back to — records the Fold with no frozen artifact at all. The
            # Agent's early_stop_reason and the Meta review then read what the
            # ledger reads.
            "pipeline_fold_status": (
                "no_update" if self._parent_structure is not None else "baseline_missing"
            ),
            "pipeline_will_freeze": False,
        }

    def _hard_rule_candidates(self, check: HardRuleCheck) -> list[dict[str, object]]:
        """Every complete Validation of this session with its hard-rule verdict.

        The host's ``parent_control`` node is one of them: selecting it is the
        documented way to keep the parent, so it must be visible here exactly
        like the session's own Validations.
        """

        rows: list[dict[str, object]] = []
        for node in self.tree.nodes():
            if (
                not node_in_session(node, fold_id=self.fold_id, run_id=self.run_id)
                or not node.get("complete_validation")
                or not node.get("revision_id")
            ):
                continue
            metrics = _node_metrics(node)
            reasons = [str(reason) for reason in check(metrics)]
            row: dict[str, object] = {
                "node_id": str(node["node_id"]),
                "result_name": str(node.get("result_name") or ""),
                "passes_hard_rules": not reasons,
            }
            drawdown = metrics.get("max_drawdown")
            if isinstance(drawdown, (int, float)) and not isinstance(drawdown, bool):
                row["max_drawdown"] = float(drawdown)
            if reasons:
                row["hard_reject_reasons"] = reasons
            rows.append(row)
        return rows

    def _resolve_node_id(self, arguments: Mapping[str, object]) -> str:
        """The nominated node: the argument, or the tree position when no
        batch round hangs below it.

        ``batch_validate`` leaves the position on the round's parent, so a
        bare call there would freeze the parent even when a candidate won;
        that call is refused with the candidates listed instead.
        """

        explicit = str(arguments.get("node_id") or "")
        if explicit:
            return explicit
        cursor = self.tree.current_node_id
        if not cursor:
            raise ToolError("finish_fold requires a fully evaluated Step")
        candidates = self._batch_candidates_under(cursor)
        if not candidates:
            return cursor
        listed = "; ".join(
            f"{row['node_id']} ({row['candidate']}: {row['result']})" for row in candidates
        )
        raise ToolError(
            "finish_fold requires an explicit node_id here: the tree position is "
            f"the parent of a batch_validate round ({cursor}), so a bare call "
            "would select the parent, not a candidate. Candidates under it: "
            f"{listed}. Pass the winner's node_id (step_rollback to it first so "
            "the working copy matches), or the parent's own node_id to keep it "
            "deliberately.",
            details={"tree_position": cursor, "candidates": candidates},
        )

    def _batch_candidates_under(self, parent_id: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for node in self.tree.nodes():
            metadata = node.get("metadata")
            if (
                node.get("parent_node_id") != parent_id
                or not node_in_session(node, fold_id=self.fold_id, run_id=self.run_id)
                or not isinstance(metadata, Mapping)
                or not metadata.get("batch_id")
            ):
                continue
            metrics = node.get("metrics") if isinstance(node.get("metrics"), Mapping) else {}
            if node.get("status") == "failed":
                result = "failed"
            else:
                parts = [
                    f"{key}={metrics[key]:.4f}"
                    for key in ("total_return", "sharpe")
                    if isinstance(metrics.get(key), (int, float))
                ]
                result = " ".join(parts) or "complete"
            rows.append(
                {
                    "node_id": str(node["node_id"]),
                    "candidate": str(metadata.get("candidate") or node.get("result_name")),
                    "result": result,
                }
            )
        return rows

    def _require_early_stop_reason(
        self, arguments: Mapping[str, object]
    ) -> dict[str, object]:
        """The early-stop fields to record, refusing a voluntary early finish
        that gives no reason.

        Voluntary means another round still fits (the same waiver as the round
        floor); early means more than a third of the backtest budget is left.
        Without a wired budget the tool cannot tell, so it only records a
        reason the Agent chose to give.
        """

        reason = str(arguments.get("early_stop_reason") or "").strip()
        status = self._budget_status() if self._budget_status is not None else None
        recorded: dict[str, object] = {}
        if reason:
            recorded["early_stop_reason"] = reason
        if status is None:
            return recorded
        recorded["budget_at_finish"] = status.to_record()
        if reason or not status.early_finish or not self._another_round_fits():
            return recorded
        minutes = max(status.inference_seconds_remaining, 0.0) / 60
        raise ToolError(
            "finish_fold refused: this voluntary finish leaves "
            f"{status.backtests_remaining}/{status.backtests_total} backtests, "
            f"{status.steps_remaining}/{status.steps_total} Steps and about "
            f"{minutes:.0f} min of inference time unused, more than a third of the "
            "backtest budget. Either pre-register another round and run "
            "batch_validate, or call finish_fold again with early_stop_reason "
            f"(<= {EARLY_STOP_REASON_MAX_CHARS} chars) naming the hypotheses that "
            "stay untested and why the remaining budget is better left unused; "
            "the reason is recorded with the Fold result for the Meta review.",
            retry_hint=(
                "finish_fold({\"node_id\": ..., \"early_stop_reason\": \"...\"}) "
                "or run another batch_validate round"
            ),
            details=status.to_record(),
        )

    def _require_different_hypothesis(self, node_id: str, nominated_structure: str) -> None:
        parent_structure = self._parent_structure
        if parent_structure is None:
            return
        different_ids = {
            candidate_id
            for candidate_id, structure in self._session_complete_structures()
            if structure != parent_structure
        }
        if not different_ids:
            raise ToolError(
                "finish_fold requires a complete Validation whose executable "
                "strategy logic differs from the parent; comment-only changes do not count"
            )
        if nominated_structure == parent_structure or node_id in different_ids:
            return
        raise ToolError(
            "finish_fold can select only a different-hypothesis Validation "
            "or an explicit keep-parent after one existed"
        )

    def _session_complete_structures(self) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for node in self.tree.nodes():
            if (
                not node_in_session(node, fold_id=self.fold_id, run_id=self.run_id)
                or not node.get("complete_validation")
                or not node.get("revision_id")
            ):
                continue
            candidate_id = str(node["node_id"])
            output_dir = self.tree.node_output_dir(candidate_id)
            if not (output_dir / "main.py").is_file():
                continue
            try:
                found.append((candidate_id, executable_output_structure(output_dir)))
            except (OSError, SyntaxError):
                continue
        return found

    def _require_current_matches_revision(self, node_id: str) -> None:
        if self._current_output is None:
            return
        nominated = self.tree.node_output_dir(node_id)
        if _tree_bytes(nominated) != _tree_bytes(self._current_output):
            raise ToolError(
                "finish_fold requires the current output to match the selected "
                "Validation revision; restore that Step or run a new complete "
                "daily_backtest"
            )
        nominated_models = self.tree.node_models_dir(node_id)
        current_models = self._current_models
        if _tree_bytes(nominated_models) != _tree_bytes(current_models):
            raise ToolError(
                "finish_fold requires the current models to match the selected "
                "Validation revision; restore that Step or run a new complete "
                "daily_backtest"
            )

    def _node_structure(self, node_id: str) -> str:
        output_dir = self.tree.node_output_dir(node_id)
        if not (output_dir / "main.py").is_file():
            raise ToolError(
                "finish_fold cannot select a Step whose strategy snapshot is absent"
            )
        try:
            return executable_output_structure(output_dir)
        except (OSError, SyntaxError) as exc:
            raise ToolError(
                f"finish_fold cannot compare {node_id}: {redact_host_paths(str(exc))}"
            ) from exc


__all__ = [
    "EARLY_STOP_REASON_MAX_CHARS",
    "FINISH_FOLD_EARLY_STOP_BUDGET_FRACTION",
    "FinishFoldTool",
    "FoldBudgetStatus",
    "HardRuleCheck",
    "executable_output_structure",
    "executable_source_structure",
]
