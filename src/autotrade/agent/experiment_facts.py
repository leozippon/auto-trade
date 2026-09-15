"""Agent-visible experiment facts: the manifest/runtime_env/data_summary projection.

``build_experiment_facts`` is the visibility contract for what a research
session may know about its own run (budgets, snapshot windows, broker replay
policy, artifact contract, runtime tools). Pure data shaping —
the prompt text that wraps it lives in ``prompts.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.style import NEUTRALIZATION_METHOD
from autotrade.environment.sandbox import SCREENING_TOOL_MOUNT, SandboxLimits

EXPERIMENT_FACTS_SCHEMA_VERSION = 2

# Agent-visible clock contract for ``budgets.deadline_seconds``, read from the
# injected facts; every replay tool pauses the clock,
# while shell and sub-agent waits do not and must not be named here.
DEADLINE_SECONDS_NOTE = (
    "`deadline_seconds` 统计可暂停的有效推理时间；"
    "`smoke_backtest`、`batch_validate` 和 `run_null_control` "
    "调用期间暂停计时，"
    "因此会话总墙钟可能更长。"
)

# Agent-visible rule for the two strategy clocks under batch fan-out. The
# environment, not the strategy, creates the contention a batch adds, so both
# caps move with it (pipelines.local_backend._batch_replay_timeouts).
BATCH_VALIDATE_FIT_TIMEOUT_NOTE = (
    "`batch_validate` 并发回放期间，`strategy_fit_timeout_seconds` 与单次推断上限"
    "`strategy_inference_timeout_seconds` 都按本批实际并发路数成倍放大（并发 3 路即 3 倍），"
    "因为并发是环境引入的、不该记到策略头上；单个候选的调用两者都用基准值。"
)

# How ``budgets.max_replay_years`` is spent.
REPLAY_YEARS_NOTE = (
    "一个候选按其 span 覆盖的研究年份计 replay-year：一年的 span 计 1，完整研究期计研究年数；"
    "一批的花费是候选数乘年数，开跑前整批预留；`smoke_backtest` 与 `run_null_control` 不计。"
)

# What a session knows about the periods after research end: that they exist
# and are sealed, never their dates, slots or results.
SEALED_PERIODS_NOTE = (
    "研究期末之后是前推期与 Held-out：冻结产物在那里被连续回放一次并裁决是否毕业；"
    "它们的日期、数据、回放槽与结果不进入任何会话。"
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
            "research_period_visible": True,
            "after_research_end": SEALED_PERIODS_NOTE,
            "formal_strategy_read_roots": ["snapshot_dir", "asof_dir"],
        },
        "research_geometry": _as_mapping(manifest.get("research")) or None,
        "visible_timeline": _visible_timeline(
            data_summary=data_summary, snapshot_config=snapshot_config
        ),
        "research_scope": _research_scope(
            manifest=manifest, snapshot_config=snapshot_config
        ),
        # The arm's selection state at session start: nothing frozen yet, and
        # the trial pool the freeze gate deflates over before this session.
        # ``False`` and zero counts are facts, not absences.
        "arm": dict(_as_mapping(manifest.get("arm"))) or None,
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
        # The caliber every ``neutralized_excess_return`` in this session was
        # computed under. One constant sentence: stating it here keeps it out
        # of every backtest summary.
        "neutralized_excess_method": NEUTRALIZATION_METHOD,
        "runtime_tools": _runtime_tool_facts(runtime_env, manifest=manifest),
    }
    return compact_mapping(facts)


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
        }
    )


def _research_scope(
    *,
    manifest: Mapping[str, object],
    snapshot_config: Mapping[str, object],
) -> dict[str, object]:
    """One sentence each on the research period, the universe and the cadence."""
    research = _as_mapping(manifest.get("research"))
    period = research.get("research_period")
    research_sentence = (
        f"The arm's one research session researches the period {period}, one "
        "July-June year per label, and ends by freezing one full-period node "
        "through the freeze gate or by ending the arm without a deliverable; no "
        "other session follows, and the frozen artifact is judged only on later, "
        "sealed data."
        if period
        else None
    )
    screen = _as_mapping(snapshot_config.get("universe_screen"))
    active = {
        key: value
        for key, value in screen.items()
        if value not in (None, False, 0, [], ())
    }
    if active:
        universe = (
            "The universe is screened when the decision view is built "
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
    period_name = str(schedule.get("period") or "day")
    inference_time = schedule.get("inference_time")
    when = (
        "every trading day"
        if period_name == "day"
        else f"on the first available trading day of each {period_name}"
    )
    cadence = (
        f"generate_orders is called {when} at {inference_time}; the strategy chooses "
        "its own rebalance cadence by returning no orders on days it does not want to "
        "trade."
    )
    return compact_mapping(
        {
            "research": research_sentence,
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
    populated_files = _populated_file_names(data_summary)
    return {
        "historical_minutes_available": "intraday_1min.parquet" in populated_files,
        "auction_available": "auction.parquet" in populated_files,
        "events_available": "events.parquet" in populated_files,
        "text_available": "text_index.parquet" in populated_files,
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
    # Local import: the pipelines package imports this module, so binding the
    # batch cap at module scope would close an import cycle.
    from autotrade.pipelines.local_backend import BATCH_VALIDATE_MAX_CONCURRENCY

    budgets = _as_mapping(manifest.get("budgets"))
    return compact_mapping(
        {
            # Pausable effective reasoning time: the main deadline PLUS the
            # trailing wrap-up grace, so the grace rides beside it: without
            # the split a session plans against a deadline that is already
            # ``deadline_grace_seconds`` later than the one its directive and
            # wrap-up prompt talk about.
            "deadline_seconds": budgets.get("deadline_seconds"),
            "deadline_seconds_note": DEADLINE_SECONDS_NOTE,
            "deadline_grace_seconds": budgets.get("deadline_grace_seconds"),
            "finalize_before_deadline_seconds": manifest.get("finalize_before_deadline_seconds"),
            "max_llm_calls": max_llm_calls or budgets.get("max_llm_calls"),
            "max_replay_years": budgets.get("max_replay_years"),
            "max_replay_years_note": (
                REPLAY_YEARS_NOTE if budgets.get("max_replay_years") is not None else None
            ),
            # Host null controls the session may request before it selects,
            # minutes of replay each; every run_null_control result says what is left.
            "max_null_controls": budgets.get("max_null_controls"),
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
            # GPUs the formal strategy container (fit worker included) is
            # started with; 0 means every formal replay runs on CPU. This is
            # the authoritative device fact: a per-session GPU override moves
            # only the session's own container, never this number.
            "strategy_gpu_count": budgets.get("strategy_gpu_count"),
            # The totals above are the arm's, spent across every attempt of
            # its one session; what earlier attempts already used of them.
            "used_before_this_attempt": budgets.get("used_before_this_attempt"),
            # CPU quota that same container is started with, and therefore the
            # value its OMP/MKL/OPENBLAS/NUMEXPR thread variables carry. Read
            # from the environment's own SandboxLimits, the single source the
            # executor builds the `--cpus` flag from, so a strategy that sets
            # its own thread count has the real number instead of a guess: a
            # session that guessed low spent two backtest slots on fit
            # timeouts it had the cores to avoid.
            "strategy_cpus": SandboxLimits().cpus,
            # Replays one batch_validate call runs at once, each with its own
            # strategy container (two when the candidate declares fit). The
            # host is shared with the other running experiments, so wall clock
            # per replay is not exclusive; the two strategy timeouts above are
            # measured on the strategy's own call and start only once the
            # worker holds the decision inputs, so neither bills container
            # scheduling or host contention to the strategy.
            "batch_validate_max_concurrency": BATCH_VALIDATE_MAX_CONCURRENCY,
            "batch_validate_fit_timeout_note": BATCH_VALIDATE_FIT_TIMEOUT_NOTE,
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
    # The freeze gate the finish tool enforces and the graduation rules the
    # frozen artifact is judged by later, both derived from the run's own
    # rules. Rules only: no date after research end.
    acceptance = _as_mapping(manifest.get("acceptance_rules"))
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
                AcceptanceRules.from_record(acceptance).agent_facts()
                if acceptance
                else None
            ),
            "record_failed_attempts": manifest.get("record_failed_attempts"),
            "nl_failure_policy": manifest.get("nl_failure_policy"),
        }
    )


def _broker_replay_facts(manifest: Mapping[str, object]) -> dict[str, object]:
    profile = _as_mapping(manifest.get("broker_profile"))
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
) -> dict[str, object]:
    tools = _as_mapping(runtime_env.get("tools"))
    available = sorted(name for name, record in tools.items() if _as_mapping(record).get("available") is True)
    missing = sorted(name for name, record in tools.items() if _as_mapping(record).get("available") is False)
    sandbox_spec = _as_mapping(runtime_env.get("sandbox_spec")) or _as_mapping(manifest.get("sandbox_spec"))
    network = runtime_env.get("network") or sandbox_spec.get("network")
    return compact_mapping(
        {
            "python": runtime_env.get("python"),
            "python_packages": dict(_as_mapping(runtime_env.get("python_packages"))),
            "cli_tools_available": available,
            "cli_tools_missing": missing,
            "network_mode": network,
            "network_install_policy": {
                "session": "no_network_prebuilt_dependencies_only",
            },
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
