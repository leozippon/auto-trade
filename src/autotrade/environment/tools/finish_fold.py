"""Select one immutable, fully evaluated Step as the Fold result."""

from __future__ import annotations

import ast
import re
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
# A finish that nominates nothing must cite the evidence that showed no edge;
# a one-word reason is not evidence.
NO_EDGE_REASON_MIN_CHARS = 40

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


# The declared-knob convention of the strategy contract: a module-level
# ``UPPER_CASE`` name bound to a literal (``REFIT_PERIOD``, ``HOLD``, ``TOP_N``,
# a board or column list written as a constant).
_KNOB_NAME = re.compile(r"_*[A-Z][A-Z0-9_]*")
_KNOB_PLACEHOLDER = "<knob>"


def mechanism_structure(root: Path) -> str:
    """The strategy package's structure with every declared knob blanked.

    Two packages are the same mechanism iff their strings are equal. Same
    reading as :func:`executable_output_structure` (every ``.py`` below
    ``root``, keyed by relative path, comments and docstrings ignored) except
    that what a deployment refit may change is replaced by a placeholder:
    every numeric, boolean or ``None`` literal anywhere (a signed number
    counts as one literal) by a placeholder of its type, and the value of
    every module-level ``UPPER_CASE`` assignment whose value is a literal of
    any shape by one placeholder. Anything else -- a file added, removed or
    renamed, a function, class, branch, loop, call, comparison, import,
    decorator or argument, or a string literal used inline in the logic -- is
    the mechanism, and changing it changes the string.
    """

    return "\n".join(
        f"{relative}\0{structure}"
        for relative, structure in _mechanism_parts(root).items()
    )


def mechanism_difference(parent: Path, candidate: Path) -> str | None:
    """The first file whose mechanism differs, or None when both are the same
    mechanism; the same reading as :func:`mechanism_structure`, file by file."""

    before = _mechanism_parts(parent)
    after = _mechanism_parts(candidate)
    for relative in sorted(set(before) | set(after)):
        if relative not in after:
            return f"{relative} removed"
        if relative not in before:
            return f"{relative} added"
        if before[relative] != after[relative]:
            return f"{relative} changes the executable logic beyond its declared knobs"
    return None


def _mechanism_parts(root: Path) -> dict[str, str]:
    parts: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _strip_docstrings(tree)
        _blank_declared_knobs(tree)
        parts[relative.as_posix()] = ast.dump(
            tree, annotate_fields=True, include_attributes=False
        )
    return parts


def _blank_declared_knobs(tree: ast.Module) -> None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if all(
            isinstance(target, ast.Name) and _KNOB_NAME.fullmatch(target.id)
            for target in targets
        ) and _is_literal(value):
            node.value = ast.Constant(value=_KNOB_PLACEHOLDER)
    for node in ast.walk(tree):
        for field, child in ast.iter_fields(node):
            if isinstance(child, list):
                for index, item in enumerate(child):
                    if _is_scalar_knob(item):
                        child[index] = _typed_placeholder(item)
            elif _is_scalar_knob(child):
                setattr(node, field, _typed_placeholder(child))


def _is_literal(node: ast.AST) -> bool:
    """A literal of any shape: what ``ast.literal_eval`` accepts."""

    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _is_scalar_knob(node: object) -> bool:
    """A numeric, boolean or ``None`` literal, optionally signed."""

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (bool, int, float, complex))
    )


def _typed_placeholder(node: ast.AST) -> ast.Constant:
    inner = node.operand if isinstance(node, ast.UnaryOp) else node
    assert isinstance(inner, ast.Constant)
    value = inner.value
    kind = "none" if value is None else "bool" if isinstance(value, bool) else type(value).__name__
    return ast.Constant(value=f"<{kind}>")


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
        "Finish this Fold. outcome=\"select\" (the default) nominates one complete "
        "Validation node of the current run by node_id (from daily_backtest or a "
        "batch_validate row); pass node_id explicitly: after a batch_validate round "
        "the tree position is the round's parent, so a bare call there is refused "
        "instead of silently keeping the parent, and keeping the parent is done by "
        "nominating the parent_control node. outcome=\"no_edge\" nominates nothing "
        "(node_id must be absent) and requires reason, citing the evidence that no "
        "candidate proved an edge; the Fold then records no_update with a parent "
        "(the parent stays the lineage head) or baseline_missing without one. "
        "A freeze without a frozen parent (parent_control_available=false, the "
        "parent is the template) is recorded baseline_anchor=true: the lineage "
        "gets a control the next Fold replays as its parent_control, but an "
        "anchor is never delivered. Use no_edge "
        "instead of nominating a node you do not want frozen: the Pipeline "
        "freezes every nomination whose metrics are finite. Outside the deadline "
        "window a voluntary finish that leaves more than a third of the backtest "
        "budget unused must also carry early_stop_reason (which hypotheses stay "
        "untested and why they are not worth the remaining budget). Both reasons "
        f"are capped at {EARLY_STOP_REASON_MAX_CHARS} characters -- write them "
        "compactly, an over-long one is refused -- and are recorded with the Fold "
        "result for the Meta review. A nominated node "
        "is checked against the Pipeline's hard acceptance rules (the "
        "acceptance_rules fact marks which rules are hard and which only warn): "
        "outside the deadline window a breaching node is refused while another "
        "recorded node still passes, and the refusal lists which ones do; inside "
        "the window, or when nothing recorded passes, the nomination is accepted "
        "and the result states that the Pipeline will not freeze it. Every accepted "
        "call returns pipeline_fold_status and a one-line pipeline_outcome saying "
        "what the Pipeline will freeze. The call starts only after every background "
        "sub-agent has finished, is refused while one is still running, and once it "
        "succeeds the remaining tool calls of this assistant turn are cancelled.",
        {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "minLength": 1, "maxLength": 500},
                "outcome": {
                    "type": "string",
                    "enum": ["select", "no_edge"],
                    "description": (
                        "select (default): freeze node_id. no_edge: nominate "
                        "nothing; requires reason and no node_id."
                    ),
                },
                "reason": {
                    "type": "string",
                    "minLength": NO_EDGE_REASON_MIN_CHARS,
                    "maxLength": EARLY_STOP_REASON_MAX_CHARS,
                    "description": (
                        "outcome=\"no_edge\" only: the evidence that no candidate "
                        "proved an edge (neutralized excess, vs_parent.beats_parent, "
                        "the new-quarter sub_window, any null_control or deflated "
                        f"Sharpe figure read); {NO_EDGE_REASON_MIN_CHARS}-"
                        f"{EARLY_STOP_REASON_MAX_CHARS} characters."
                    ),
                },
                "early_stop_reason": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": EARLY_STOP_REASON_MAX_CHARS,
                    "description": (
                        "Why this Fold stops while more than a third of its backtest "
                        "budget remains: the untested hypotheses and why the remaining "
                        "budget is better left unused. Required only in that case; at "
                        f"most {EARLY_STOP_REASON_MAX_CHARS} characters."
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
        null_controls: Callable[[], Mapping[str, Mapping[str, object]]] | None = None,
        same_mechanism: bool = False,
    ) -> None:
        self.tree = tree
        self.fold_id = fold_id
        self.run_id = run_id
        # A deployment adjustment refits the graduated mechanism: the nominated
        # node must be the parent's mechanism (mechanism_structure), and
        # nominating the parent itself is the normal no-adjustment outcome, so
        # the different-hypothesis rule does not apply.
        self.same_mechanism = same_mechanism
        self._current_output = Path(current_output) if current_output is not None else None
        self._current_models = Path(current_models) if current_models is not None else None
        # The early-stop justification and the hard-rule refusal apply only
        # while the session could still run a round; the caller says whether
        # time and budget allow one, and (when wired) what is left of each
        # budget.
        self._another_round_fits = another_round_fits or (lambda: True)
        self._budget_status = budget_status
        self._hard_rule_check = hard_rule_check
        # The null-control blocks the session already drew, by node id, so a
        # candidate listing can show the percentile the Agent has read.
        self._null_controls = null_controls or dict
        self._parent_structure: str | None = None
        self._parent_dir: Path | None = None
        if parent_main_py is not None:
            # The parent package is the directory that holds its main.py.
            path = Path(parent_main_py)
            if not path.is_file():
                raise ValueError(f"parent strategy structure is invalid: missing {path.name}")
            try:
                self._parent_structure = executable_output_structure(path.parent)
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"parent strategy structure is invalid: {exc}") from exc
            self._parent_dir = path.parent
        if same_mechanism and self._parent_dir is None:
            raise ValueError("same_mechanism needs the parent package to compare against")

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        if str(arguments.get("outcome") or "select") == "no_edge":
            return self._finish_no_edge(arguments)
        if str(arguments.get("reason") or "").strip():
            raise ToolError(
                'finish_fold: reason belongs to outcome="no_edge"; a nomination '
                "explains an early finish with early_stop_reason instead",
                retry_hint=(
                    'finish_fold({"node_id": ...}) or '
                    'finish_fold({"outcome": "no_edge", "reason": "<evidence>"})'
                ),
            )
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
        hard_reject_reasons = self._check_hard_acceptance(node_id, node)
        # The working-copy check comes before the budget gates so a winner
        # nominated without step_rollback costs one refusal, not two.
        self._require_current_matches_revision(node_id)
        early_stop = self._require_early_stop_reason(arguments)
        nominated_structure = self._node_structure(node_id)
        if self.same_mechanism:
            self._require_same_mechanism(node_id)
        elif self._parent_structure is not None:
            self._require_different_hypothesis(node_id, nominated_structure)
        self.tree.set_position(node_id)
        return ToolResult(
            True,
            value={
                "node_id": node_id,
                "revision_id": str(node["revision_id"]),
                "status": "fold_finished",
                "outcome": "select",
                # Candidate selection, AcceptanceRules and the final freeze are
                # the Pipeline's, not the Agent's: finishing only nominates.
                "fold_status": "pending_pipeline_review",
                "write_locked": True,
                # The Agent's own account of an early finish and the budget it
                # left, for the fold ledger and the Meta review.
                **early_stop,
                # What the Pipeline will do with this nomination, in its words.
                **self._nomination_verdict(node_id, node, hard_reject_reasons),
            },
            finish=True,
        )

    def _finish_no_edge(self, arguments: Mapping[str, object]) -> ToolResult:
        """Finish without a nomination: no candidate proved an edge.

        The Pipeline freezes every nomination whose metrics are finite, so a
        session that found nothing worth freezing has to say so here instead
        of nominating "the least bad node"; the reason is recorded with the
        Fold result so the Meta review reads the evidence, not a guess. The
        inherited parent, when there is one, stays the lineage head.
        """

        if str(arguments.get("node_id") or ""):
            raise ToolError(
                'finish_fold: outcome="no_edge" nominates nothing, so node_id must be '
                "absent; to freeze a node call finish_fold with node_id alone, to keep "
                "the parent nominate the parent_control node",
                retry_hint='finish_fold({"outcome": "no_edge", "reason": "<evidence>"})',
            )
        reason = str(arguments.get("reason") or "").strip()
        if len(reason) < NO_EDGE_REASON_MIN_CHARS:
            raise ToolError(
                'finish_fold: outcome="no_edge" requires reason (between '
                f"{NO_EDGE_REASON_MIN_CHARS} and {EARLY_STOP_REASON_MAX_CHARS} chars) "
                "citing the evidence that no candidate proved an edge: the "
                "neutralized excess, vs_parent.beats_parent, the new-quarter "
                "sub_window, and any null_control or deflated Sharpe figure read; "
                "it is recorded with the Fold result for the Meta review",
                retry_hint='finish_fold({"outcome": "no_edge", "reason": "<evidence>"})',
            )
        candidates = self._session_candidates()
        if not candidates:
            raise ToolError(
                'finish_fold: outcome="no_edge" needs at least one complete '
                "Validation of this session to have found no edge in; run "
                "daily_backtest or batch_validate first"
            )
        early_stop = self._require_early_stop_reason(arguments)
        return ToolResult(
            True,
            value={
                "status": "fold_finished",
                "outcome": "no_edge",
                "reason": reason,
                "fold_status": "pending_pipeline_review",
                "write_locked": True,
                "candidates_evaluated": len(candidates),
                **early_stop,
                "pipeline_fold_status": self._fallback_status(),
                "pipeline_will_freeze": False,
                "pipeline_outcome": f"No candidate frozen; {self._fallback_sentence()}",
            },
            finish=True,
        )

    def _session_candidates(self) -> list[str]:
        """Complete Validations of this session that are the Agent's own, not
        the host's parent control."""

        return [
            str(node["node_id"])
            for node in self.tree.nodes()
            if node_in_session(node, fold_id=self.fold_id, run_id=self.run_id)
            and node.get("complete_validation")
            and node.get("revision_id")
            and not (
                isinstance(node.get("metadata"), Mapping)
                and node["metadata"].get("parent_control")
            )
        ]

    def _nomination_verdict(
        self, node_id: str, node: Mapping[str, object], reasons: list[str]
    ) -> dict[str, object]:
        """The fold status the Pipeline will record for this nomination, and
        one line saying what it will freeze: the session's early_stop_reason
        and the Meta review then read what the ledger reads."""

        label = f"{node_id} ({node.get('result_name') or 'validation'})"
        if reasons:
            return {
                "acceptance_hard_reject_reasons": reasons,
                "pipeline_fold_status": self._fallback_status(),
                "pipeline_will_freeze": False,
                "pipeline_outcome": (
                    f"No candidate frozen: {label} fails the Pipeline's hard "
                    f"acceptance rules ({', '.join(reasons)}); "
                    f"{self._fallback_sentence()}"
                ),
            }
        return {
            "pipeline_fold_status": "frozen",
            "pipeline_will_freeze": True,
            "pipeline_outcome": f"Fold will freeze {label} as this Fold's strategy",
        }

    def _fallback_status(self) -> str:
        # Without a node to freeze the Pipeline falls back to the inherited
        # parent, or -- with no parent to fall back to -- records the Fold
        # with no frozen artifact at all.
        return "no_update" if self._parent_structure is not None else "baseline_missing"

    def _fallback_sentence(self) -> str:
        if self._parent_structure is not None:
            return "the inherited parent stays the lineage head (no_update)"
        return (
            "there is no parent, so the Fold records baseline_missing and the "
            "next Fold starts from the template again"
        )

    def _check_hard_acceptance(
        self, node_id: str, node: Mapping[str, object]
    ) -> list[str]:
        """Hard-reject reasons of the nominated node; empty when it passes or
        no rules are wired.

        A breach is refused while another recorded node passes and another
        round still fits, so the session can select that one or run a
        risk-reduced round instead of learning in the next Meta session that
        its Fold froze nothing. Inside the deadline window, or with nothing
        recorded that passes, the nomination is accepted and the reasons ride
        into the result's verdict.
        """

        check = self._hard_rule_check
        if check is None:
            return []
        reasons = [str(reason) for reason in check(_node_metrics(node))]
        if not reasons:
            return []
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
        return reasons

    def _hard_rule_candidates(
        self, check: HardRuleCheck | None
    ) -> list[dict[str, object]]:
        """Every complete Validation of this session with its hard-rule verdict
        and the figures a choice between them is made on.

        The host's ``parent_control`` node is one of them: selecting it is the
        documented way to keep the parent, so it must be visible here exactly
        like the session's own Validations. Without wired rules every node
        passes, as the Pipeline would then freeze any of them.
        """

        null_controls = self._null_controls()
        rows: list[dict[str, object]] = []
        for node in self.tree.nodes():
            if (
                not node_in_session(node, fold_id=self.fold_id, run_id=self.run_id)
                or not node.get("complete_validation")
                or not node.get("revision_id")
            ):
                continue
            node_id = str(node["node_id"])
            metrics = _node_metrics(node)
            reasons = [] if check is None else [str(reason) for reason in check(metrics)]
            row: dict[str, object] = {
                "node_id": node_id,
                "result_name": str(node.get("result_name") or ""),
                "passes_hard_rules": not reasons,
            }
            benchmark = metrics.get("benchmark")
            null = null_controls.get(node_id)
            for key, value in (
                ("max_drawdown", metrics.get("max_drawdown")),
                (
                    "neutralized_excess_return",
                    benchmark.get("neutralized_excess_return")
                    if isinstance(benchmark, Mapping)
                    else None,
                ),
                (
                    "null_excess_percentile",
                    null.get("excess_percentile") if isinstance(null, Mapping) else None,
                ),
            ):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[key] = float(value)
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

    def _require_same_mechanism(self, node_id: str) -> None:
        assert self._parent_dir is not None
        try:
            difference = mechanism_difference(
                self._parent_dir, self.tree.node_output_dir(node_id)
            )
        except (OSError, SyntaxError) as exc:
            raise ToolError(
                f"finish_fold cannot compare {node_id}: {redact_host_paths(str(exc))}"
            ) from exc
        if difference is None:
            return
        raise ToolError(
            f"finish_fold refused: {node_id} changes the graduated mechanism "
            f"({difference}). A deployment adjustment may only change models/, "
            "numeric/bool/None literals and module-level UPPER_CASE literal "
            "constants; nominate the parent_control node or a node that keeps "
            "the mechanism.",
            error_type="mechanism_changed",
            retry_hint='finish_fold({"node_id": "<parent_control or knob-only node>"})',
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
    "NO_EDGE_REASON_MIN_CHARS",
    "FinishFoldTool",
    "FoldBudgetStatus",
    "HardRuleCheck",
    "executable_output_structure",
    "executable_source_structure",
    "mechanism_difference",
    "mechanism_structure",
]
