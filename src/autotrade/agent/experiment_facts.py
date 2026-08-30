"""Agent-visible experiment facts: the manifest/runtime_env/data_summary projection.

``build_experiment_facts`` is the visibility contract for what a Fold or
meta-learning session may know about its own run (budgets, snapshot windows,
broker replay policy, artifact contract, runtime tools). Pure data shaping —
the prompt text that wraps it lives in ``prompts.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

from autotrade.environment.identity import AgentRefStore

EXPERIMENT_FACTS_SCHEMA_VERSION = 1


def build_experiment_facts(
    *,
    manifest: Mapping[str, object],
    ref_store: AgentRefStore,
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
                "run_id": (
                    ref_store.get_or_create("run", str(manifest["run_id"]))
                    if manifest.get("run_id")
                    else None
                ),
                "epoch_id": manifest.get("epoch_id"),
                "meta_learning_id": (
                    ref_store.get_or_create("meta", str(manifest["meta_learning_id"]))
                    if is_meta and manifest.get("meta_learning_id")
                    else None
                ),
                "trigger_after_folds": manifest.get("trigger_after_folds") if is_meta else None,
                "session_kind": kind,
                "fold_sequence_or_opaque_id": _opaque_fold_ref(
                    manifest.get("fold_id"), ref_store=ref_store, is_meta=is_meta
                ),
                "phase": None if is_meta else manifest.get("phase"),
            }
        ),
        "source_refs": {
            "run_manifest_ref": "/mnt/artifacts/run_manifest.json",
            "runtime_env_ref": str(manifest.get("runtime_env_ref") or "/mnt/artifacts/runtime_env.json"),
            "data_summary_ref": str(manifest.get("data_summary_ref") or "/mnt/artifacts/data_summary.json"),
            "skills_index_ref": str(
                _as_mapping(manifest.get("skills")).get("index_path")
                or "inputs/skills_index.json"
            ),
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
        "research_scope": _research_scope(
            manifest=manifest,
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
            manifest,
            ref_store=ref_store,
            model_artifacts_empty=model_artifacts_empty,
            is_meta=is_meta,
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


def _research_scope(
    *,
    manifest: Mapping[str, object],
    snapshot_config: Mapping[str, object],
    fold_period: object,
    is_meta: bool,
) -> dict[str, object]:
    """One sentence each on the development window, the universe and the cadence."""
    fold = _as_mapping(manifest.get("fold"))
    window = fold.get("validation_period")
    if manifest.get("test_stage") is False:
        development = (
            f"This Fold's validation period is {window}. The development window is "
            f"split into one Fold per {fold_period or 'period'}, developed in "
            "chronological order with a Meta-learning session between Folds; there "
            "is no frozen Test stage, and the frozen strategy is judged only by the "
            "automatic Held-out replay."
        )
    elif manifest.get("test_stage") is True:
        development = (
            f"This Fold's development window is its validation period {window}; "
            "development rolls period by period inside the configured window."
        )
    else:
        development = f"The development window of this session is {window}."
    screen = _as_mapping(snapshot_config.get("universe_screen"))
    active = {
        key: value
        for key, value in screen.items()
        if value not in (None, False, 0, [], ())
    }
    if active:
        universe = (
            "The universe is screened at the decision anchor "
            f"({', '.join(f'{key}={value}' for key, value in active.items())}); "
            "the strategy may filter further."
        )
    else:
        universe = (
            "The universe is unfiltered: every listed A share on all boards, ST "
            "names included and no new-listing exclusion; the strategy applies its "
            "own universe filters."
        )
    schedule = _as_mapping(manifest.get("schedule"))
    if not schedule:
        schedule = _as_mapping(_as_mapping(manifest.get("experiment_parameters")).get("schedule"))
    period = str(schedule.get("period") or "day")
    inference_time = schedule.get("inference_time")
    when = (
        "every trading day"
        if period == "day"
        else f"on the first available trading day of each {period}"
    )
    cadence = (
        f"generate_orders is called {when} at {inference_time}; the strategy chooses "
        "its own rebalance cadence by returning no orders on days it does not want to "
        "trade."
    )
    return compact_mapping(
        {
            "development_window": None if is_meta else development,
            "universe": universe,
            "strategy_cadence": cadence,
        }
    )


def _snapshot_windows(snapshot_config: Mapping[str, object]) -> dict[str, object]:
    """The decision-input windows the session can actually read.

    ``SnapshotConfig.to_record`` always carries ``intraday_trade_days`` because
    that record is the on-disk PIT cache contract and its shape must stay
    stable. With minute bars off there is no minute file to look back over, so
    the projection drops the key instead of advertising a window over data the
    execution policy reports as unavailable.
    """

    windows = _as_mapping(snapshot_config.get("decision_windows"))
    return compact_mapping(
        {
            "daily_months": windows.get("daily_months"),
            "fundamentals_months": windows.get("fundamentals_months"),
            "events_months": windows.get("events_months"),
            "macro_months": windows.get("macro_months"),
            "text_months": windows.get("text_months"),
            "intraday_trade_days": (
                windows.get("intraday_trade_days")
                if snapshot_config.get("include_intraday")
                else None
            ),
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
            # The formal executor's per-trading-day inference wall clock; a
            # slower generate_orders fails the whole backtest.
            "strategy_inference_timeout_seconds": budgets.get(
                "strategy_inference_timeout_seconds"
            ),
            # The separate wall clock of one fit(context) call, which the
            # prompts tell the agent to read before moving work into fit.
            "strategy_fit_timeout_seconds": budgets.get(
                "strategy_fit_timeout_seconds"
            ),
            "context_compaction": context_compaction,
        }
    )


def _artifact_contract_facts(
    manifest: Mapping[str, object],
    *,
    ref_store: AgentRefStore,
    model_artifacts_empty: bool | None,
    is_meta: bool,
) -> dict[str, object]:
    is_initial = bool(manifest.get("is_initial_artifact", manifest.get("template_ref") is not None))
    parent_id = manifest.get("parent_strategy_artifact_id") or manifest.get("parent_artifact_id")
    parent = {
        "kind": "initial_template" if is_initial else "frozen_artifact",
        # Artifact ids embed the raw fold label (strategy_<epoch>_fold_<period>);
        # project them like every other agent-visible surface.
        "id": (
            ref_store.get_or_create("strategy", str(parent_id)) if parent_id else None
        ),
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
            # Semantics (AcceptanceRules.evaluate): the max_drawdown cap and
            # non-finite total_return/max_drawdown/sharpe are HARD rejects;
            # min_return / min_sharpe are targets — shortfalls freeze WITH a
            # recorded warning instead of resetting the fold.
            "acceptance_semantics": None if is_meta else "max_drawdown+finite_metrics=hard; return/sharpe=warn-only targets",
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


def _opaque_fold_ref(
    value: object, *, ref_store: AgentRefStore, is_meta: bool
) -> str | None:
    if value is None or str(value) == "":
        return None
    namespace = "meta" if is_meta else "fold"
    return ref_store.get_or_create(namespace, str(value))


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
