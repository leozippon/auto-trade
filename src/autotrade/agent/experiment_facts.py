"""Agent-visible experiment facts: the manifest/runtime_env/data_summary projection.

``build_experiment_facts`` is the visibility contract for what a research
session may know about its own run (budgets, snapshot windows, broker replay
policy, artifact contract, runtime tools). Pure data shaping —
the prompt text that wraps it lives in ``prompts.py``. The facts hold values
and structured definitions; a rule the system prompt or a tool description
already states is not repeated here as prose.
"""

from __future__ import annotations

from collections.abc import Mapping

from autotrade.environment.broker_core import STAMP_DUTY_CUTOVER
from autotrade.environment.dividend_tax import (
    BONUS_SHARE_PAR_CNY,
    DIVIDEND_TAX_CUTOVER,
    NEW_RULE,
    OLD_RULE,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.style import regression_method
from autotrade.environment.sandbox import SCREENING_TOOL_MOUNT

EXPERIMENT_FACTS_SCHEMA_VERSION = 2

# ``runtime_tools.sandbox_gpus`` of a Sandbox without cards; the prompt's
# orientation reads the same fact (``prompts.build_system_prompt``).
NO_SANDBOX_GPUS = "none"


def _sandbox_gpus(spec: Mapping[str, object]) -> str | None:
    """The session Sandbox's cards as one fact: ``"1 × A100"``, or ``NO_SANDBOX_GPUS``.

    Read off the session's own ``SandboxSpec`` record, not nvidia-smi: the
    count is the cards its container attaches (a per-session override
    included) and the model is ``gpu_name_filter``, the host's device name
    every one of them was selected by (``environment.gpu.host_gpu_name``).
    ``None`` for a record that states no count, rather than a guess.
    """

    if "gpu_count" not in spec:
        return None
    count = int(spec["gpu_count"])  # type: ignore[call-overload]
    if count == 0:
        return NO_SANDBOX_GPUS
    return f"{count} × {spec.get('gpu_name_filter') or 'GPU'}"


# Where the read roots are in the Sandbox filesystem. The root names are a
# file-tool convention, not paths: two experiments read `steps` as a shell path,
# failed, and concluded the tree was invisible to `shell`.
SEARCH_ROOT_SHELL_NOTE = (
    "`read_file`/`grep`/`glob` 的根名是工具约定而不是路径；`shell` 里同一批目录在各自的挂载点下："
    "`workspace`/`output`/`models` 在 `/mnt/agent/workspace` 下，`snapshot` 是 `/mnt/snapshot`，"
    "`artifacts` 是 `/mnt/artifacts`，`steps`、`parent_output`、`parent_models` 在它下面"
    "（Step 产物树即 `/mnt/artifacts/steps`）。只有根 `trace` 在容器里没有路径，"
    "transcript 只能用 `read_file`/`grep` 读。"
)


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

    This is a convenience index, not a security boundary. It carries research
    dates only; exact trusted details remain in the referenced JSON files.
    """

    runtime_env = runtime_env or {}
    data_summary = data_summary or {}
    snapshot_config = _as_mapping(manifest.get("snapshot_config"))
    benchmark_index = str(manifest["benchmark_index"])

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
                "session_kind": manifest.get("kind"),
                "session_ref": (
                    ref_store.get_or_create("session", str(manifest["fold_id"]))
                    if manifest.get("fold_id")
                    else None
                ),
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
            # The read-only signal screen is a Docker bind mount outside every
            # file-tool root, so the fact carries its argv contract next to the
            # path; a local-dev session has no such path, so the fact stays out
            # rather than lie.
            "signal_screen_ref": (
                {
                    "path": SCREENING_TOOL_MOUNT,
                    "usage": (
                        "shell only, e.g. argv "
                        f'["python", "{SCREENING_TOOL_MOUNT}", "--help"]; '
                        "not under any read_file/grep/glob root"
                    ),
                }
                if runtime_env.get("mode") == "docker"
                else None
            ),
        },
        "visibility_policy": {
            "visible_research_period": True,
            "formal_strategy_read_roots": ["snapshot_dir", "asof_dir"],
        },
        "research_geometry": _research_geometry(manifest),
        "visible_timeline": _visible_timeline(
            data_summary=data_summary, snapshot_config=snapshot_config
        ),
        # The experiment's state at session start: nothing registered yet, and the
        # Validations earlier attempts recorded. ``False`` and zero counts are
        # facts, not absences.
        "experiment": dict(_as_mapping(manifest.get("experiment"))) or None,
        "budgets": _budget_facts(
            manifest,
            max_llm_calls=max_llm_calls,
            context_compaction=context_compaction,
        ),
        # No "paths" table and no per-file "data_profile": every production
        # consumer of this object is the prompt renderer, which dropped both
        # unconditionally (the same information lives in data_summary.json and
        # the fixed mount layout) — building always-dropped sections was shaping
        # work with no reader.
        "artifact_contract": _artifact_contract_facts(
            manifest, model_artifacts_empty=model_artifacts_empty
        ),
        "broker_replay": _broker_replay_facts(manifest),
        # The index this experiment is measured against: the benchmark leg of every
        # factor loading below, and the membership the random-selection control
        # draws its replacement names from.
        "benchmark_index": benchmark_index,
        # The caliber every factor loading and residual tracking error in this
        # session was computed under. One sentence, naming that benchmark:
        # stating it here keeps it out of every backtest summary.
        "regression_method": regression_method(benchmark_index),
        "runtime_tools": _runtime_tool_facts(runtime_env, manifest=manifest),
    }
    return compact_mapping(facts)


def _research_geometry(manifest: Mapping[str, object]) -> dict[str, object] | None:
    """The manifest's research geometry with the research years alone in
    ``years``: the validation years reach the session as
    ``validation_period`` and through ``test``, and the manifest keeps every
    year because the acceptance facts count the validation years off it."""

    geometry = _as_mapping(manifest.get("research"))
    if not geometry:
        return None
    research_end = str(geometry["research_period"]).split("..")[1]
    years = [year for year in _as_list(geometry.get("years")) if str(_as_mapping(year)["end"]) <= research_end]
    return {**geometry, "years": years}


def _visible_timeline(
    *,
    data_summary: Mapping[str, object],
    snapshot_config: Mapping[str, object],
) -> dict[str, object]:
    snapshot_windows = _snapshot_windows(snapshot_config)
    return compact_mapping(
        {
            "snapshot_windows": snapshot_windows,
            "decision_snapshot_intraday_lookback_trade_days": snapshot_windows.get(
                "intraday_trade_days"
            ),
            "validation_intraday_scope": "historical_pit_features_and_exact_execution_prices",
            "execution_policy": _execution_policy(data_summary),
            "universe_screen": _universe_screen(snapshot_config),
        }
    )


def _universe_screen(snapshot_config: Mapping[str, object]) -> dict[str, object]:
    """The screen the decision view's universe is built under, its active
    settings only: absent for the default, unfiltered universe (every listed
    A share on all boards, ST names and new listings included), which the
    strategy filters itself."""

    screen = _as_mapping(snapshot_config.get("universe_screen"))
    return {key: value for key, value in screen.items() if value not in (None, False, 0, [], ())}


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
    populated_files = _populated_file_names(data_summary)
    return {
        "historical_minutes_available": "intraday_1min.parquet" in populated_files,
        "auction_available": "auction.parquet" in populated_files,
        "events_available": "events.parquet" in populated_files,
        "text_available": "text_index.parquet" in populated_files,
        "strategy_clock": "configured_schedule_only",
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
            # Pausable effective reasoning time: the main deadline PLUS the
            # trailing wrap-up grace, so the grace rides beside it: without
            # the split a session plans against a deadline that is already
            # ``deadline_grace_seconds`` later than the one its directive and
            # wrap-up prompt talk about.
            "deadline_seconds": budgets.get("deadline_seconds"),
            "deadline_grace_seconds": budgets.get("deadline_grace_seconds"),
            "max_llm_calls": max_llm_calls or budgets.get("max_llm_calls"),
            "max_api_spend_usd": budgets.get("max_api_spend_usd"),
            # The validate allowance of one interval between tests
            # (``AcceptanceRules.validates_per_step``), which the prompt's run
            # section reads from here.
            "validates_per_step": budgets.get("validates_per_step"),
            # Host null controls the session may request before it selects,
            # minutes of replay each; every run_null_control result says what is left.
            "max_null_controls": budgets.get("max_null_controls"),
            # The formal executor's per-trading-day inference wall clock; a
            # slower generate_orders fails the whole backtest.
            "strategy_inference_timeout_seconds": budgets.get(
                "strategy_inference_timeout_seconds"
            ),
            # The separate wall clock of one fit(context) call; a fit past it
            # fails the whole backtest, as an inference past its own does.
            "strategy_fit_timeout_seconds": budgets.get(
                "strategy_fit_timeout_seconds"
            ),
            # GPUs the formal strategy container (fit worker included) is
            # started with; 0 means every formal replay runs on CPU. This is
            # the authoritative device fact: a per-session GPU override moves
            # only the session's own container, never this number.
            "strategy_gpu_count": budgets.get("strategy_gpu_count"),
            # The totals above are the experiment's, spent across every attempt of
            # its one session; what earlier attempts already used of them.
            "used_before_this_attempt": budgets.get("used_before_this_attempt"),
            # CPU quota that same container is started with, from which its
            # OMP/MKL/OPENBLAS/NUMEXPR thread variables are derived. Read
            # from the run manifest, which carries the evaluator's own
            # strategy SandboxSpec -- the experiment's, and the single source the executor
            # builds the `--cpus` flag from -- so a strategy that sets its own
            # thread count has the real number instead of a guess: a session
            # that guessed low spent two backtest slots on fit timeouts it had
            # the cores to avoid.
            "strategy_cpus": budgets.get("strategy_cpus"),
            # The memory ceiling of that same container, in the unit a replay
            # reports its measured peak in (``resources.peak_memory_bytes``).
            # The session container carries the same ceiling, so a fit profiled
            # under `shell` is measured against the number the formal replay
            # will enforce rather than extrapolated across two boundaries.
            "strategy_memory_bytes": budgets.get("strategy_memory_bytes"),
            # The disk the replay's state directory may hold, measured after
            # every call, in the unit a replay reports its own in
            # (``resources.peak_state_bytes``); past it the backtest fails.
            "strategy_state_disk_bytes": budgets.get("strategy_state_disk_bytes"),
            # Replays one validate call runs at once (the experiment's
            # ``batch_validate_max_concurrency``), each with its own strategy
            # container (two when the candidate declares fit). The host is
            # shared with the other running experiments, so wall clock per
            # replay is not exclusive; the two strategy timeouts above are
            # measured on the strategy's own call and start only once the
            # worker holds the decision inputs, so neither bills container
            # scheduling or host contention to the strategy.
            # How the two strategy clocks scale with it is the validate
            # description's rule.
            "batch_validate_max_concurrency": budgets.get("batch_validate_max_concurrency"),
            "context_compaction": context_compaction,
        }
    )


def _artifact_contract_facts(
    manifest: Mapping[str, object],
    *,
    model_artifacts_empty: bool | None,
) -> dict[str, object]:
    # Local import: the pipelines package imports this module, so binding the
    # acceptance rules at module scope would close an import cycle.
    from autotrade.pipelines.config import AcceptanceRules

    start = dict(_as_mapping(manifest.get("start")))
    start["model_artifacts_empty"] = model_artifacts_empty
    # The registration every test enforces, the validation protocol and the
    # readings the registered artifact is reported on later, all derived from the
    # run's own rules and the count of validation years the manifest's
    # geometry leaves after the research years. No date after the geometry's
    # last year.
    acceptance = _as_mapping(manifest.get("acceptance_rules"))
    research = _as_mapping(manifest.get("research"))
    years = research.get("years")
    validation_years = (
        len(years) - int(acceptance["research_years"])  # type: ignore[call-overload]
        if acceptance and isinstance(years, list)
        else None
    )
    return compact_mapping(
        {
            "required_entry": "output/main.py",
            "strategy_entry_function": "generate_orders",
            "strategy_return_contract": "strict_json_order_array",
            "model_artifacts_allowed": True,
            "workspace_frozen": False,
            "start": compact_mapping(start),
            "modification_constraints": manifest.get("modification_constraints"),
            "acceptance_rules": (
                AcceptanceRules.from_record(acceptance).agent_facts(validation_years=validation_years)
                if validation_years is not None
                else None
            ),
            "record_failed_attempts": manifest.get("record_failed_attempts"),
        }
    )


# What ``BrokerProfile.permitted_boards`` means for the Agent, stated beside the
# boards on an experiment whose account carries them.
PERMITTED_BOARDS_RULE = (
    "The account may buy only names listed on these boards (main = Shanghai and Shenzhen "
    "main boards, gem = ChiNext, star = STAR Market, bj = Beijing Stock Exchange): a buy of "
    "a name on any other board is rejected whole as board_not_permitted, a sale never is, "
    "and the random-selection control draws its replacement names from these boards only."
)


def _broker_replay_facts(manifest: Mapping[str, object]) -> dict[str, object]:
    profile = _as_mapping(manifest["broker_profile"])
    schedule = _as_mapping(manifest.get("schedule"))
    boards = profile.get("permitted_boards")
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
                    "cutover_date": STAMP_DUTY_CUTOVER,
                }
            ),
            "slippage_bps": profile.get("slippage_bps"),
            "dividend_tax_policy": (
                {
                    "charged": True,
                    "charged_at": "old_rule_withholding_on_ex_date_then_fifo_sale_supplement",
                    "regime_selected_by": "dividend_record_date_not_ex_date_or_sale_date",
                    "record_date_cutover": DIVIDEND_TAX_CUTOVER,
                    "holding_period": "acquisition_through_day_before_sale_natural_months_and_years",
                    "taxable_income_per_share": "gross_cash_dividend_plus_bonus_shares_times_par",
                    "bonus_share_par_cny": BONUS_SHARE_PAR_CNY,
                    "rules": {
                        rule.key: {
                            "record_dates": "before_cutover_since_2013" if rule == OLD_RULE else "from_cutover",
                            "withholding_rate": rule.withholding_rate,
                            "final_rate_by_months_held_at_most": {
                                str(months): rate for months, rate in rule.tiers
                            },
                            "final_rate_beyond": rule.rate_beyond,
                            "sale_supplement": "final_rate_minus_withholding_rate",
                        }
                        for rule in (OLD_RULE, NEW_RULE)
                    },
                    "conventions": "ex_date_cash_settlement_bonus_shares_keep_original_acquisition_date_conversions_not_income",
                }
                if profile["dividend_tax"]
                else {"charged": False}
            ),
            # Only where the profile pins them: an account without the field
            # buys on every board, and its facts say nothing new.
            "permitted_boards": (
                None if boards is None else {"boards": boards, "rule": PERMITTED_BOARDS_RULE}
            ),
            "t_plus_one": True,
            "ex_date_settlement": "pre_close_reset_cash_dividend_credited",
            "order_lot_size": 100,
            "price_limit_enforced": True,
            "suspension_enforced": True,
            "schedule": schedule,
            "decision_frequency": "day_month_quarter_or_year",
            "execution_time": "order_session_open_close_or_intraday_else_execute_at_exact",
            "session_fills": "daily_bar_open_close_or_vwap_plus_slippage_intraday_limit_price_fills_once_day_low_or_high_reaches_it_locked_or_unreached_unfilled_same_day",
            "missing_exact_price": "reject",
        }
    )


def _runtime_tool_facts(
    runtime_env: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    tools = _as_mapping(runtime_env.get("tools"))
    available = sorted(name for name, record in tools.items() if _as_mapping(record).get("available") is True)
    missing = sorted(name for name, record in tools.items() if _as_mapping(record).get("available") is False)
    sandbox_spec = _as_mapping(runtime_env.get("sandbox_spec")) or _as_mapping(manifest.get("sandbox_spec"))
    network = runtime_env.get("network") or sandbox_spec.get("network")
    return compact_mapping(
        {
            "python": runtime_env.get("python"),
            "cli_tools_available": available,
            "cli_tools_missing": missing,
            "network_mode": network,
            "sandbox_gpus": _sandbox_gpus(sandbox_spec),
            "network_install_policy": {
                "session": "no_network_prebuilt_dependencies_only",
            },
            "file_root_shell_paths": SEARCH_ROOT_SHELL_NOTE,
        }
    )


def _populated_file_names(data_summary: Mapping[str, object]) -> set[str]:
    """File names that actually carry rows in at least one visible view.

    A switched-off domain (minute bars by default) or a domain with nothing in
    the visible window (the auction file before 2025) is still written as a
    zero-row Parquet, so file presence alone would advertise research columns
    that do not exist and execution prices every order would be rejected for as
    ``missing_execution_price``. The replay engine already gates its minute
    execution-price source on the file's row count, and this projection states
    the same fact to the session.
    """

    names: set[str] = set()
    for view in _as_mapping(data_summary.get("views")).values():
        for item in _as_list(_as_mapping(view).get("files")):
            record = _as_mapping(item)
            path = str(record.get("path") or "")
            rows = record.get("rows")
            if path and isinstance(rows, int) and rows > 0:
                names.add(path.rsplit("/", 1)[-1])
    return names


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
