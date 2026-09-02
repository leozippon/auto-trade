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
from collections.abc import Mapping
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


def compact_fold_history(
    record: dict[str, object],
    *,
    ref_store: AgentRefStore,
    include_frozen_test_metrics: bool = False,
) -> dict[str, object]:
    manifest = _read_json(Path(str(record.get("run_manifest_ref", ""))))
    backtests = []
    raw_backtests = manifest.get("backtest_summaries")
    if isinstance(raw_backtests, list):
        for summary in raw_backtests:
            if not isinstance(summary, dict):
                continue
            backtests.append(
                {
                    key: summary.get(key)
                    for key in (
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
                        # Exit health + benchmark-relative view: a lineage whose
                        # "gains" trail the index must stay visible to later epochs.
                        "strategy_exit_fill_count",
                        "benchmark",
                        # Overfitting tell (lzp-test21 post-mortem): turnover cost
                        # drove the held-out loss while the dev metrics looked
                        # healthy — meta-learning must see it, not just returns.
                        "turnover",
                        # Cost of the Validation itself, so Meta can weigh a
                        # direction's replay and NL spend against its evidence.
                        "replay_wall_seconds",
                        "replayed_trade_days",
                        "nl_calls",
                        "nl_llm_calls",
                        "nl_wall_seconds",
                        # Selection evidence: this candidate against the
                        # Fold's own parent control on the same window.
                        "vs_parent",
                        "error",
                    )
                    if key in summary
                }
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
