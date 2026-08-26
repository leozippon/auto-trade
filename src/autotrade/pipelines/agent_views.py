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
            for key in ("label", "benchmark_return", "excess_return", "beta", "n_days", "size_tilt")
            if key in benchmark
            and (
                (key == "label" and isinstance(benchmark.get(key), str))
                or (
                    key != "label"
                    and isinstance(benchmark.get(key), (int, float))
                    and not isinstance(benchmark.get(key), bool)
                )
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


def compact_fold_history(
    record: dict[str, object],
    *,
    ref_store: AgentRefStore,
    include_frozen_test_metrics: bool = False,
) -> dict[str, object]:
    manifest = _read_json(Path(str(record.get("run_manifest_ref", ""))))
    backtests = []
    if isinstance(manifest.get("backtest_summaries"), list):
        for summary in manifest["backtest_summaries"]:
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
                        # every exit is a host liquidation, or whose "gains" trail
                        # the index, must stay visible to later epochs.
                        "host_exit_liquidation_count",
                        "strategy_exit_fill_count",
                        "liquidation_complete",
                        "benchmark",
                        # Overfitting tells (lzp-test21 post-mortem): structural
                        # low exposure and turnover cost drove the held-out loss
                        # while the dev metrics looked healthy — meta-learning
                        # must see them, not just returns.
                        "exposure",
                        "turnover",
                        "error",
                    )
                    if key in summary
                }
            )
    compact = {
        "epoch_id": record.get("epoch_id"),
        "fold_id": ref_store.get_or_create("fold", str(record.get("fold_id"))),
        "run_id": (
            ref_store.get_or_create("run", str(record["run_id"]))
            if record.get("run_id")
            else None
        ),
        "fold_status": record.get("fold_status"),
        "finish_reason": record.get("finish_reason"),
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
