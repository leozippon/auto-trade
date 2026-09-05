"""Agent-visible projections of ledger, fold-history, and metric records.

The single test-leakage allowlist surface (docs/agent-design.md): everything an
Agent or Meta session may read from experiment history passes through these
whitelisting projections. ``metrics`` is also the host-side compact metric
block written into ledger records; the ``agent_visible_*`` functions
additionally opaque raw fold/strategy identifiers and strip Test/Held-out
evidence except through the explicit frozen-test metric whitelist.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from autotrade.environment.identity import AgentRefStore


def metrics(summary: dict[str, object] | None) -> dict[str, object] | None:
    if not summary:
        return None
    keys = (
        "total_return",
        "long_return",
        "sharpe",
        "max_drawdown",
        "order_count",
        "trade_count",
        "turnover",
        # Compact Barra-lite block (benchmark/excess return, beta, size tilt)
        # from the backtest tool — descriptive attribution per step.
        "benchmark",
    )
    compact = {key: summary.get(key) for key in keys if key in summary}
    exposure = summary.get("exposure")
    if isinstance(exposure, dict):
        compact["exposure"] = {
            key: exposure.get(key)
            for key in ("avg_gross", "max_gross", "zero_position_days", "replay_days")
            if key in exposure
        }
    # One scalar each from the cost-sensitivity and concentration blocks: this
    # block rides in every ledger record and Meta view, so it keeps the two
    # numbers that change a judgement — whether the excess survives twice the
    # modelled slippage, and how much of the gain one name produced — and
    # leaves the full blocks in the backtest summary.
    for source, field in (
        ("cost_sensitivity", "excess_at_2x_slippage"),
        ("pnl_concentration", "top_name_share_of_gross_gains"),
    ):
        block = summary.get(source)
        if isinstance(block, dict) and field in block:
            compact[field] = block.get(field)
    return compact


# The benchmark fields one compact metric block keeps. Raw excess alone cannot
# separate real edge from a small-cap or high-beta tilt, so the size/beta
# neutralized excess -- the tie-breaker both the Fold and the Meta guidance name
# -- and the caliber it was computed under ride beside it. Descriptive
# attribution only: nothing here is Test or Held-out evidence, which the
# ``include_frozen_test_metrics`` gate governs, not this whitelist.
_BENCHMARK_TEXT_KEYS = frozenset({"label", "neutralized_excess_method"})
_BENCHMARK_KEYS = (
    "label",
    "benchmark_return",
    "excess_return",
    "beta",
    "n_days",
    "size_tilt",
    "neutralized_excess_return",
    "neutralized_excess_method",
)


def _visible_metrics(value: object) -> dict[str, object] | None:
    return agent_visible_metrics(value if isinstance(value, dict) else None)


def agent_visible_metrics(summary: dict[str, object] | None) -> dict[str, object] | None:
    """Compact metric projection safe for Meta workspace history."""

    compact = metrics(summary)
    if compact is None:
        return None
    compact = {
        key: value
        for key, value in compact.items()
        if key in {"benchmark", "exposure"}
        or (isinstance(value, (int, float)) and not isinstance(value, bool))
    }
    benchmark = compact.get("benchmark")
    if isinstance(benchmark, dict):
        compact["benchmark"] = {
            key: benchmark.get(key)
            for key in _BENCHMARK_KEYS
            if key in benchmark
            and (
                isinstance(benchmark.get(key), str)
                if key in _BENCHMARK_TEXT_KEYS
                else isinstance(benchmark.get(key), (int, float))
                and not isinstance(benchmark.get(key), bool)
            )
        }
    else:
        compact.pop("benchmark", None)
    exposure = compact.get("exposure")
    if isinstance(exposure, dict):
        compact_exposure = {
            key: exposure.get(key)
            for key in ("avg_gross", "max_gross", "zero_position_days", "replay_days")
            if key in exposure
            and isinstance(exposure.get(key), (int, float))
            and not isinstance(exposure.get(key), bool)
        }
        if compact_exposure:
            compact["exposure"] = compact_exposure
        else:
            compact.pop("exposure", None)
    else:
        compact.pop("exposure", None)
    return compact


# ``vs_parent``: one candidate's Validation minus the Fold's parent control.
#
# Every candidate of a Fold is quoted on the same Validation window as the
# host's parent control, so the number that carries information is the
# difference, not two absolute figures read side by side. ``beats_parent`` is
# True only when BOTH excess deltas are > 0: the raw excess alone cannot
# separate an edge from a small-cap or high-beta tilt, and the neutralized
# excess alone cannot show what the tilt actually contributed.
_VS_PARENT_KEYS = ("excess_return", "neutralized_excess_return")


def vs_parent_metrics(
    summary: Mapping[str, object] | None,
    control_summary: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Candidate-minus-control deltas for one completed Validation.

    ``None`` when the Fold has no parent control: a Fold that inherited
    nothing has no baseline on this window, and borrowing another Fold's or
    another candidate's numbers would invent one. Both excess figures come
    from the summaries' own ``benchmark`` blocks (window excess return and the
    annualized size/beta neutralized excess); ``max_drawdown_delta`` compares
    magnitudes, so it is positive when the candidate drew down more than the
    parent whichever sign convention the summary uses. ``beats_parent`` is
    True only when both excess deltas are > 0, and ``None`` when either delta
    could not be computed.
    """

    if not isinstance(summary, Mapping) or not isinstance(control_summary, Mapping):
        return None
    deltas: dict[str, object] = {}
    for key in _VS_PARENT_KEYS:
        candidate = _benchmark_number(summary, key)
        control = _benchmark_number(control_summary, key)
        deltas[f"{key}_delta"] = (
            candidate - control
            if candidate is not None and control is not None
            else None
        )
    candidate_drawdown = _finite(summary.get("max_drawdown"))
    control_drawdown = _finite(control_summary.get("max_drawdown"))
    deltas["max_drawdown_delta"] = (
        abs(candidate_drawdown) - abs(control_drawdown)
        if candidate_drawdown is not None and control_drawdown is not None
        else None
    )
    excess = deltas["excess_return_delta"]
    neutralized = deltas["neutralized_excess_return_delta"]
    deltas["beats_parent"] = (
        bool(excess > 0 and neutralized > 0)
        if isinstance(excess, float) and isinstance(neutralized, float)
        else None
    )
    return deltas


def _benchmark_number(summary: Mapping[str, object], key: str) -> float | None:
    benchmark = summary.get("benchmark")
    return _finite(benchmark.get(key)) if isinstance(benchmark, Mapping) else None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# One completed Validation as a later Fold or the Meta session reads it back.
# ``neutralized_excess_method`` is deliberately absent: the caliber is one
# constant sentence, and the session facts state it once
# (``build_experiment_facts``) instead of once per summary per fold.
_SUMMARY_KEYS = (
    "result_name",
    "mode",
    "status",
    "complete_validation",
    "total_return",
    "long_return",
    "sharpe",
    "max_drawdown",
    "order_count",
    "trade_count",
    # Exit health + benchmark-relative view: a lineage whose "gains" trail the
    # index must stay visible to later epochs.
    "strategy_exit_fill_count",
    "benchmark",
    # Overfitting tell (lzp-test21 post-mortem): turnover cost drove the
    # held-out loss while the dev metrics looked healthy — meta-learning must
    # see it, not just returns.
    "turnover",
    # The same tell priced out: what one more bp of slippage per side costs and
    # what the excess is worth at twice it, beside how few trades and names
    # produced the gains.
    "cost_sensitivity",
    "pnl_concentration",
    # Cost of the Validation itself, so Meta can weigh a direction's replay and
    # NL spend against its evidence.
    "replay_wall_seconds",
    "replayed_trade_days",
    "nl_calls",
    "nl_llm_calls",
    "nl_wall_seconds",
    # Selection evidence: this candidate against the Fold's own parent control
    # on the same window.
    "vs_parent",
    "error",
)
_NL_SUMMARY_KEYS = frozenset({"nl_calls", "nl_llm_calls", "nl_wall_seconds"})


def _nl_service_disabled(manifest: Mapping[str, object]) -> bool:
    """True when this run mounted no text corpus for the NL sub-agent.

    The NL service answers out of the replay slot's text domain. With that
    domain switched off (``snapshot_config.replay.include_text``) every query
    can only return ``no_evidence``, so the counters carry no information about
    how the Fold spent its budget. A zero under a mounted corpus is a real
    reading — the strategy never asked — and stays. An older manifest that does
    not record the switch keeps the counters rather than hiding them.
    """

    snapshot_config = manifest.get("snapshot_config")
    replay = (
        snapshot_config.get("replay") if isinstance(snapshot_config, Mapping) else None
    )
    return isinstance(replay, Mapping) and replay.get("include_text") is False


def compact_fold_history(
    record: dict[str, object],
    *,
    ref_store: AgentRefStore,
    include_frozen_test_metrics: bool = False,
) -> dict[str, object]:
    manifest = _read_json(Path(str(record.get("run_manifest_ref", ""))))
    keys = _SUMMARY_KEYS
    if _nl_service_disabled(manifest):
        keys = tuple(key for key in keys if key not in _NL_SUMMARY_KEYS)
    backtests = []
    raw_backtests = manifest.get("backtest_summaries")
    if isinstance(raw_backtests, list):
        for summary in raw_backtests:
            if not isinstance(summary, dict):
                continue
            backtests.append(
                _without_benchmark_method(
                    {key: summary.get(key) for key in keys if key in summary}
                )
            )
    compact = {
        "epoch_id": record.get("epoch_id"),
        "fold_id": ref_store.get_or_create("fold", str(record.get("fold_id"))),
        # The window these results were replayed on. Without it a reader has to
        # borrow the label of whatever neighbouring node names a period, and a
        # benchmark figure gets attributed to the wrong year.
        "validation_period": record.get("validation_period"),
        "run_id": (
            ref_store.get_or_create("run", str(record["run_id"]))
            if record.get("run_id")
            else None
        ),
        "fold_status": record.get("fold_status"),
        "finish_reason": record.get("finish_reason"),
        "early_stop_reason": record.get("early_stop_reason"),
        "validation_result": _visible_metrics(record.get("validation_result")),
        "accept_reasons": record.get("accept_reasons"),
        "accept_warnings": record.get("accept_warnings"),
        "backtest_summaries": backtests,
    }
    if include_frozen_test_metrics and record.get("record_type") == "fold":
        compact["test_result"] = _visible_metrics(record.get("test_result"))
    return compact


def _without_benchmark_method(metrics: dict[str, object] | None) -> dict[str, object] | None:
    """Drop the neutralization caliber sentence: the session facts state it once."""

    if metrics is None:
        return None
    benchmark = metrics.get("benchmark")
    if isinstance(benchmark, dict) and "neutralized_excess_method" in benchmark:
        return {
            **metrics,
            "benchmark": {
                key: value
                for key, value in benchmark.items()
                if key != "neutralized_excess_method"
            },
        }
    return metrics


# Host-computed selection evidence a later Fold or a Meta review reads verbatim
# from the Fold record: how wide the search was and how much of the winner's
# Sharpe that width alone explains (pipelines/ledger.deflated_sharpe), where
# the observed excess sits inside random-name replays of the Fold's own trade
# skeleton (environment/replay/null_control.py), and how the frozen candidate
# stood against the Fold's parent control (``vs_parent_metrics``). Development
# statistics only — no Test or Held-out evidence enters through these.
SELECTION_STATISTICS_KEYS = (
    "candidates_evaluated",
    # Whether the deflated trial pool includes the parent control, which it
    # does exactly when the Fold kept the parent: without it a kept-parent
    # probability reads as if a challenger had won the search.
    "parent_included",
    "deflated_sharpe_probability",
    "trials",
    "sharpe_star",
    "trial_sharpe_std",
    "observed_sharpe",
    "return_days",
    "return_skew",
    "return_kurtosis",
    "unavailable_reason",
)
# ``rejects_mean`` rides along because a null whose orders are mostly rejected
# is a weaker comparison, ``status`` because a failed null must not read as a
# missing one, and the null's own centre and spread over its ``k`` draws
# because a percentile alone does not say how far the observed excess sits
# from them. Informational: nothing in the pipeline gates on it.
NULL_CONTROL_KEYS = (
    "status",
    "observed_excess",
    "excess_percentile",
    "null_excess_mean",
    "null_excess_p05",
    "null_excess_p95",
    "k",
    "rejects_mean",
    "dropped_trips_mean",
    "step",
)
VS_PARENT_DELTA_KEYS = (
    "excess_return_delta",
    "neutralized_excess_return_delta",
    "max_drawdown_delta",
    "beats_parent",
)


def allowed_keys(block: object, keys: Sequence[str]) -> dict[str, object] | None:
    """Whitelisted projection of one host-computed block; None when absent."""

    if not isinstance(block, Mapping):
        return None
    return {key: block.get(key) for key in keys if key in block}


def fold_development_summary(
    record: dict[str, object], *, ref_store: AgentRefStore
) -> dict[str, object]:
    """One completed Fold as a later Fold session reads it in its run facts.

    The verdict rather than the trial log: the frozen node's metrics, how it
    stood against the Fold's own parent control (``vs_parent``) and against
    random-name replays of its trades (``null_control``), how wide the search
    was that it won (``selection_statistics``), and how the inherited parent
    itself fared on the Fold's new period (``parent_control.step_result``).
    Per-candidate backtest summaries stay out: they are already in the Step
    tree and in the Meta history (``compact_fold_history``), and a system
    prompt that carried them grew by ~25k characters per completed Fold, so
    this projection's size does not depend on how many candidates a Fold ran.
    Test metrics never enter it.
    """

    control = record.get("parent_control")
    parent_control = None
    if isinstance(control, Mapping):
        parent_control = {
            "status": control.get("status"),
            "step_result": _visible_step_result(control.get("step_result")),
            "null_control": allowed_keys(
                control.get("null_control"), NULL_CONTROL_KEYS
            ),
        }
    return {
        "epoch_id": record.get("epoch_id"),
        "fold_id": ref_store.get_or_create("fold", str(record.get("fold_id"))),
        "validation_period": record.get("validation_period"),
        "run_id": (
            ref_store.get_or_create("run", str(record["run_id"]))
            if record.get("run_id")
            else None
        ),
        "fold_status": record.get("fold_status"),
        "finish_reason": record.get("finish_reason"),
        "early_stop_reason": record.get("early_stop_reason"),
        "accept_reasons": record.get("accept_reasons"),
        "accept_warnings": record.get("accept_warnings"),
        "validation_result": _without_benchmark_method(
            _visible_metrics(record.get("validation_result"))
        ),
        "vs_parent": allowed_keys(record.get("vs_parent"), VS_PARENT_DELTA_KEYS),
        "selection_statistics": allowed_keys(
            record.get("selection_statistics"), SELECTION_STATISTICS_KEYS
        ),
        "null_control": allowed_keys(record.get("null_control"), NULL_CONTROL_KEYS),
        "parent_control": parent_control,
    }


def _visible_step_result(value: object) -> dict[str, object] | None:
    """The parent control's new-period row: its window labels plus the compact
    metric block every other result is read through."""

    if not isinstance(value, Mapping):
        return None
    labels = {
        key: value.get(key) for key in ("label", "start", "end", "partial") if key in value
    }
    return {**labels, **(agent_visible_metrics(dict(value)) or {})}


def agent_visible_ledger_record(
    record: dict[str, object],
    *,
    ref_store: AgentRefStore,
    include_frozen_test_metrics: bool = False,
) -> dict[str, object]:
    public = json.loads(json.dumps(record, ensure_ascii=False, default=str))
    if not isinstance(public, dict):
        return {}
    allowed = {
        "record_type",
        "experiment_id",
        "epoch_id",
        "meta_learning_id",
        "trigger_after_folds",
        "run_id",
        "parent_strategy_artifact_id",
        "finish_reason",
        "fold_status",
        "accept_reasons",
        "accept_warnings",
        "selected_step_id",
        "steps",
        "frozen_strategy_artifact_id",
        "validation_result",
        "state_changed_during_test",
        "snapshot_ids",
        "status",
        "modification_check",
        "prior_chars",
        "prior_published",
        "prior_generation_id",
        "agent_session_summary",
        "meta_learning_directive",
        "fold_exploration_directive",
        "input_window",
        "validation_period",
        "valid_decision_time",
    }
    public = {key: value for key, value in public.items() if key in allowed}
    if "validation_result" in public:
        public["validation_result"] = _visible_metrics(public.get("validation_result"))
    if include_frozen_test_metrics and record.get("record_type") == "fold":
        public["test_result"] = _visible_metrics(record.get("test_result"))
    if record.get("fold_id"):
        namespace = "meta" if record.get("record_type") == "meta_learning" else "fold"
        public["fold_id"] = ref_store.get_or_create(
            namespace, str(record["fold_id"])
        )
    if public.get("run_id"):
        public["run_id"] = ref_store.get_or_create("run", str(public["run_id"]))
    if public.get("meta_learning_id"):
        public["meta_learning_id"] = ref_store.get_or_create(
            "meta", str(public["meta_learning_id"])
        )
    for key in ("parent_strategy_artifact_id", "frozen_strategy_artifact_id"):
        if public.get(key):
            public[key] = ref_store.get_or_create("strategy", str(public[key]))
    steps = public.get("steps")
    if isinstance(steps, list):
        public["steps"] = [agent_visible_step_record(step) for step in steps if isinstance(step, dict)]
    snapshot_ids = public.get("snapshot_ids")
    if isinstance(snapshot_ids, dict):
        public["snapshot_ids"] = {
            key: value
            for key, value in snapshot_ids.items()
            if not str(key).startswith("test_") and not str(key).startswith("heldout_")
        }
    return public


def agent_visible_step_record(record: dict[str, object]) -> dict[str, object]:
    allowed = {
        "step_id",
        "status",
        "strategy_artifact_ref",
        "model_artifact_ref",
        "combined_artifact_ref",
        "modification_delta_summary",
        "timing",
        "decision_reason",
        "summary",
    }
    public = {key: value for key, value in record.items() if key in allowed}
    if "summary" in public:
        public["summary"] = _visible_metrics(public.get("summary"))
    return public


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
