"""Agent-visible experiment facts: the manifest/runtime_env/data_summary projection.

``build_experiment_facts`` is the visibility contract for what a Fold or
meta-learning session may know about its own run (budgets, snapshot windows,
broker replay policy, artifact contract, runtime tools). Pure data shaping —
the prompt text that wraps it lives in ``prompts.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

from autotrade.environment.identity import agent_visible_ref

EXPERIMENT_FACTS_SCHEMA_VERSION = 1


def build_experiment_facts(
    *,
    manifest: Mapping[str, object],
    runtime_env: Mapping[str, object] | None = None,
    data_summary: Mapping[str, object] | None = None,
    max_llm_calls: int | None = None,
    context_compaction: Mapping[str, object] | None = None,
    model_artifacts_empty: bool | None = None,
) -> dict[str, object]:
    """Build the short Agent-visible operational-facts projection.

    This is a convenience index, not a security boundary. It intentionally
    omits test/held-out schedule fields; exact trusted details remain in the
    referenced JSON files.
    """

    runtime_env = runtime_env or {}
    data_summary = data_summary or {}
    kind = str(manifest.get("kind") or "fold")
    is_meta = kind == "meta_learning"
    snapshot_config = _as_mapping(manifest.get("snapshot_config"))
    if is_meta:
        experiment_parameters = _as_mapping(manifest.get("experiment_parameters"))
        snapshot_config = _as_mapping(experiment_parameters.get("snapshot_config")) or snapshot_config
        fold_period = experiment_parameters.get("fold_period")
    else:
        fold_period = manifest.get("fold_period")

    facts: dict[str, object] = {
        "identity": compact_mapping(
            {
                "facts_schema_version": EXPERIMENT_FACTS_SCHEMA_VERSION,
                "experiment_id": manifest.get("experiment_id"),
                "run_id": manifest.get("run_id"),
                "epoch_id": manifest.get("epoch_id"),
                "meta_learning_id": manifest.get("meta_learning_id") if is_meta else None,
                "trigger_after_folds": manifest.get("trigger_after_folds") if is_meta else None,
                "session_kind": kind,
                "fold_sequence_or_opaque_id": _opaque_fold_ref(manifest.get("fold_id")),
                "phase": None if is_meta else manifest.get("phase"),
            }
        ),
        "source_refs": {
            "run_manifest_ref": "/mnt/artifacts/run_manifest.json",
            "runtime_env_ref": str(manifest.get("runtime_env_ref") or "/mnt/artifacts/runtime_env.json"),
            "data_summary_ref": str(manifest.get("data_summary_ref") or "/mnt/artifacts/data_summary.json"),
        },
        "visibility_policy": {
            "train_visible": True,
            "valid_visible": True,
            # Raw Test data remains unmounted. Meta alone receives compact
            # metrics from already-completed frozen Fold tests via workspace.
            "test_visible": False,
            "historical_frozen_test_metrics_visible": is_meta,
            "heldout_visible": False,
            "hidden_schedule_redacted": True,
            "formal_strategy_read_roots": ["snapshot_dir", "asof_dir"],
        },
        "visible_timeline": _visible_timeline(
            manifest=manifest,
            data_summary=data_summary,
            snapshot_config=snapshot_config,
            fold_period=fold_period,
            is_meta=is_meta,
        ),
        "budgets": _budget_facts(manifest, max_llm_calls=max_llm_calls, context_compaction=context_compaction),
        # No "paths" table and no per-file "data_profile": every production
        # consumer of this object is the prompt renderer, which dropped both
        # unconditionally (the same information lives in data_summary.json and
        # the fixed mount layout) — building always-dropped sections was shaping
        # work with no reader.
        "artifact_contract": _artifact_contract_facts(
            manifest, model_artifacts_empty=model_artifacts_empty, is_meta=is_meta
        ),
        "broker_replay": _broker_replay_facts(manifest),
        "runtime_tools": _runtime_tool_facts(runtime_env, manifest=manifest, is_meta=is_meta),
    }
    if is_meta:
        facts["meta_learning"] = _meta_learning_facts(manifest)
    return compact_mapping(facts)


def _visible_timeline(
    *,
    manifest: Mapping[str, object],
    data_summary: Mapping[str, object],
    snapshot_config: Mapping[str, object],
    fold_period: object,
    is_meta: bool,
) -> dict[str, object]:
    execution_policy = _execution_policy(data_summary)
    snapshot_windows = _snapshot_windows(snapshot_config)
    timeline = {
        "fold_period": fold_period,
        "snapshot_windows": snapshot_windows,
        "decision_snapshot_intraday_lookback_trade_days": snapshot_windows.get("intraday_trade_days"),
        "validation_intraday_scope": "historical_pit_features_and_exact_execution_prices",
        "execution_policy": execution_policy,
    }
    if is_meta:
        timeline["sample_window_only"] = True
        timeline["exact_sample_coverage_ref"] = "/mnt/artifacts/data_summary.json"
    else:
        fold = _as_mapping(manifest.get("fold"))
        timeline.update(
            {
                "current_decision_time": manifest.get("valid_decision_time")
                or fold.get("valid_decision_time"),
                "visible_input_window": fold.get("input_window"),
                "visible_validation_replay_period": fold.get("validation_period"),
            }
        )
    return compact_mapping(timeline)


def _snapshot_windows(snapshot_config: Mapping[str, object]) -> dict[str, object]:
    windows = _as_mapping(snapshot_config.get("decision_windows"))
    return compact_mapping(
        {
            "daily_months": windows.get("daily_months"),
            "fundamentals_months": windows.get("fundamentals_months"),
            "events_months": windows.get("events_months"),
            "macro_months": windows.get("macro_months"),
            "text_months": windows.get("text_months"),
            "intraday_trade_days": windows.get("intraday_trade_days"),
        }
    )


def _execution_policy(data_summary: Mapping[str, object]) -> dict[str, object]:
    visible_files = _visible_file_names(data_summary)
    return {
        "historical_minutes_available": "intraday_1min.parquet" in visible_files,
        "auction_available": "auction.parquet" in visible_files,
        "events_available": "events.parquet" in visible_files,
        "text_available": "text_index.parquet" in visible_files,
        "strategy_clock": "configured_schedule_only",
        "execution_time": "order_execute_at_exact",
        "missing_exact_price": "reject",
        "historical_minutes_drive_strategy": False,
    }


def _budget_facts(
    manifest: Mapping[str, object],
    *,
    max_llm_calls: int | None,
    context_compaction: Mapping[str, object] | None,
) -> dict[str, object]:
    budgets = _as_mapping(manifest.get("budgets"))
    return compact_mapping(
        {
            "deadline_seconds": manifest.get("deadline_seconds") or budgets.get("deadline_seconds"),
            "finalize_before_deadline_seconds": manifest.get("finalize_before_deadline_seconds"),
            "max_steps": manifest.get("max_steps") or budgets.get("max_steps"),
            "max_llm_calls": max_llm_calls
            or manifest.get("max_llm_calls")
            or budgets.get("max_llm_calls"),
            "max_backtests_per_fold": manifest.get("max_backtests_per_fold")
            or budgets.get("max_backtests"),
            "context_compaction": context_compaction,
        }
    )


def _artifact_contract_facts(
    manifest: Mapping[str, object],
    *,
    model_artifacts_empty: bool | None,
    is_meta: bool,
) -> dict[str, object]:
    is_initial = bool(manifest.get("is_initial_artifact", manifest.get("template_ref") is not None))
    parent_id = manifest.get("parent_strategy_artifact_id") or manifest.get("parent_artifact_id")
    parent = {
        "kind": "initial_template" if is_initial else "frozen_artifact",
        # Artifact ids embed the raw fold label (strategy_<epoch>_fold_<period>);
        # project them like every other agent-visible surface.
        "id": agent_visible_ref(parent_id, prefix="strategy_ref") if parent_id else None,
        "model_artifacts_empty": model_artifacts_empty,
    }
    return compact_mapping(
        {
            "required_entry": "output/main.py",
            "strategy_entry_function": "generate_orders",
            "strategy_return_contract": "strict_json_order_array",
            "model_artifacts_allowed": True,
            "workspace_frozen": False,
            "parent": compact_mapping(parent),
            "modification_constraints": manifest.get("modification_constraints"),
            "acceptance_rules": None if is_meta else manifest.get("acceptance_rules"),
            # Semantics: max_drawdown + complete validation are HARD gates;
            # min_return / min_sharpe are targets — shortfalls freeze WITH a
            # recorded warning instead of resetting the fold.
            "acceptance_semantics": None if is_meta else "drawdown+complete=hard; return/sharpe=warn-only targets",
            "step_tree_enabled": manifest.get("step_tree_enabled"),
            "record_failed_attempts": manifest.get("record_failed_attempts"),
            "nl_failure_policy": manifest.get("nl_failure_policy"),
        }
    )


def _broker_replay_facts(manifest: Mapping[str, object]) -> dict[str, object]:
    profile = _as_mapping(manifest.get("broker_profile"))
    if not profile:
        experiment_parameters = _as_mapping(manifest.get("experiment_parameters"))
        profile = _as_mapping(experiment_parameters.get("broker_profile"))
    schedule = _as_mapping(manifest.get("schedule"))
    return compact_mapping(
        {
            "profile_id": profile.get("profile_id"),
            "initial_cash": profile.get("initial_cash"),
            "commission_bps": profile.get("commission_bps"),
            "min_commission_cny": profile.get("min_commission_cny"),
            "stamp_duty_policy": compact_mapping(
                {
                    "sell_bps_before_cutover": profile.get("stamp_duty_sell_bps_before_cutover"),
                    "sell_bps_from_cutover": profile.get("stamp_duty_sell_bps_from_cutover"),
                    "cutover_date": profile.get("stamp_duty_cutover_date"),
                }
            ),
            "slippage_bps": profile.get("slippage_bps"),
            "t_plus_one": True,
            "order_lot_size": 100,
            "price_limit_enforced": True,
            "suspension_enforced": True,
            "schedule": schedule,
            "decision_frequency": "day_month_quarter_or_year",
            "execution_time": "order_execute_at_exact",
            "missing_exact_price": "reject",
            "nl_max_calls_per_decision_day": manifest.get("nl_max_calls_per_decision_day"),
            "nl_max_calls_per_backtest": manifest.get("nl_max_calls_per_backtest"),
        }
    )


def _runtime_tool_facts(
    runtime_env: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    is_meta: bool,
) -> dict[str, object]:
    tools = _as_mapping(runtime_env.get("tools"))
    available = sorted(name for name, record in tools.items() if _as_mapping(record).get("available") is True)
    missing = sorted(name for name, record in tools.items() if _as_mapping(record).get("available") is False)
    sandbox_spec = _as_mapping(runtime_env.get("sandbox_spec")) or _as_mapping(manifest.get("sandbox_spec"))
    sandbox_runtime = _as_mapping(manifest.get("sandbox_runtime"))
    proxy_aliases = [
        str(item.get("container_env"))
        for item in _as_list(sandbox_runtime.get("active_env_aliases"))
        if isinstance(item, Mapping) and str(item.get("container_env", "")).startswith("AT_PROXY_")
    ]
    active_env_passthrough = [
        str(name)
        for name in _as_list(sandbox_runtime.get("active_env_passthrough"))
        if str(name).strip()
    ]
    network = runtime_env.get("network") or sandbox_spec.get("network")
    return compact_mapping(
        {
            "python": runtime_env.get("python"),
            "python_packages": dict(_as_mapping(runtime_env.get("python_packages"))),
            "cli_tools_available": available,
            "cli_tools_missing": missing,
            "network_mode": network,
            "credential_env_names_active": active_env_passthrough,
            "proxy_alias_names_active": proxy_aliases,
            "network_install_policy": {
                "ordinary_fold": "no_network_prebuilt_dependencies_only",
                "meta_learning": (
                    "workspace_only_if_network_enabled"
                    if is_meta and str(network or "none") != "none"
                    else "blocked_unless_runtime_env_enables_network"
                ),
                "sandbox_environment_scope": "python_npm_apt_packages_only_no_weights_data_or_repositories",
            },
        }
    )


def _meta_learning_facts(manifest: Mapping[str, object]) -> dict[str, object]:
    development_inputs = _as_mapping(manifest.get("development_inputs"))
    return compact_mapping(
        {
            "prior_output_path": manifest.get("prior_output") or "/mnt/agent/workspace/PRIOR.md",
            "prior_injected_scope": "subsequent_fold_prompts_until_next_meta_trigger",
            "development_inputs": {
                key: value
                for key, value in development_inputs.items()
                if key
                in {
                    "agent_trace_full",
                    "agent_traces",
                    "development_history",
                    "experiment_ledger_full",
                    "meta_learning_memory",
                }
            },
            "previous_prior_available": bool(development_inputs.get("previous_prior")),
            "history_available": bool(development_inputs),
            "sample_window_only": True,
            "backtest_allowed": False,
            "meta_learning_directive_present": bool(str(manifest.get("meta_learning_directive") or "").strip()),
            "fold_exploration_directive_present": bool(
                str(manifest.get("fold_exploration_directive") or "").strip()
            ),
        }
    )


def _visible_file_names(data_summary: Mapping[str, object]) -> set[str]:
    names: set[str] = set()
    for view in _as_mapping(data_summary.get("views")).values():
        for item in _as_list(_as_mapping(view).get("files")):
            path = str(_as_mapping(item).get("path") or "")
            if path:
                names.add(path.rsplit("/", 1)[-1])
    return names


def _opaque_fold_ref(value: object) -> str | None:
    if value is None or str(value) == "":
        return None
    return agent_visible_ref(value, prefix="fold_ref")


def _as_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def compact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            item = compact_mapping(item)
        elif isinstance(item, list):
            item = [compact_mapping(x) if isinstance(x, Mapping) else x for x in item]
        if item is None or item == "" or item == {} or item == []:
            continue
        compact[str(key)] = item
    return compact
