"""An experiment's parameters (``hitl/params.json``) as one typed record.

The file is flat: a console create, a round file and the command line all
write one JSON object of parameters. :class:`ExperimentParams` groups those
parameters by what they configure, one sub-record each, and each field is
one parameter: its type, its default and who may set it (:class:`Surface`).
The console's creation defaults, the values it stamps, the keys it refuses
and requires, the worker's accepted keys and the creation form's field list
are all read off these fields, so adding a parameter means adding a field.

A value reads back as the type its field declares or as null, which every
reader of the file takes as unset (as it takes an empty string for a
parameter that is not text). A key no field declares is refused. Keys
beginning with ``_`` are the file's own stamps (who created it and when),
not parameters. A missing key takes its field's default. That is the
parameter's one default: the console stamps it at creation, and the worker
reads it for a worker-side file that leaves the key out (:func:`read_param`). A parameter
the worker derives from another one when unset says so beside its field (a
worker-side ``llm_model`` stands in for an unset main model,
:func:`session_models`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import (
    MISSING,
    Field,
    dataclass,
    field,
    fields,
    is_dataclass,
    make_dataclass,
)
from enum import StrEnum
from functools import cache
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from autotrade.common.ids import EXPERIMENT_ID, is_experiment_id
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.llm.lighting import (
    DEEPSEEK_FLASH_MODEL,
    GPT_ASTRA_MODEL,
)
from autotrade.environment.llm.model_profiles import (
    AGENT_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
)
from autotrade.environment.sandbox import DEFAULT_IMAGE, SandboxSpec

from .config import (
    ACCEPTANCE_KEYS,
    DEFAULT_PIT_VIEWS_SEED,
    DEFAULT_RESEARCH_GEOMETRY,
    REQUIRED_ACCEPTANCE_KEYS,
    AcceptanceRules,
    acceptance_default,
    experiment_default,
)


class Surface(StrEnum):
    """Who may set a parameter."""

    # A console create request (and so a round file) may name it.
    CONSOLE = "console"
    # The console stamps its one supported value and refuses a request that
    # names it; a worker-side params.json may set another.
    MANAGED = "managed"
    # Only a worker-side params.json or the command line sets it; the console
    # refuses it.
    HOST = "host"


def _param(
    default: object = None,
    *,
    surface: Surface = Surface.CONSOLE,
    required: bool = False,
    stamp: object = None,
) -> Any:
    """One parameter. ``required``: a create request must state it (its
    default is only what the form shows). ``stamp``: the value the console
    writes for a ``MANAGED`` parameter."""

    return field(default=default, metadata={"surface": surface, "required": required, "stamp": stamp})


def _host(default: object = None) -> Any:
    """A host parameter. ``default`` is what the worker reads when a
    params.json leaves it out; None where the worker derives it from another
    parameter or it has none."""

    return _param(default, surface=Surface.HOST)


# Former operator overrides of the provider endpoint and credential. Endpoints
# and credentials come only from the trusted model profile's fixed environment
# keys, so no parameter can redirect either: the console and the worker refuse
# them, and the command line names them in its own refusal.
NON_PERSISTABLE_PARAMS = frozenset({"llm_api_key_env", "llm_base_url"})


@dataclass(frozen=True, kw_only=True)
class GeometryParams:
    """Research, forward and Held-out dates (``calendar.ResearchGeometry``)."""

    research_start: str = _param(DEFAULT_RESEARCH_GEOMETRY.research_start, required=True)
    research_end: str = _param(DEFAULT_RESEARCH_GEOMETRY.research_end, required=True)
    forward_end: str = _param(DEFAULT_RESEARCH_GEOMETRY.forward_end, required=True)
    heldout_end: str = _param(DEFAULT_RESEARCH_GEOMETRY.heldout_end, required=True)


@dataclass(frozen=True, kw_only=True)
class PackParams:
    """What the experiment's sessions are handed: the directive, the reference workspace
    and whether the curated skill library mounts."""

    research_directive: str = _param(experiment_default("research_directive"))
    workspace_reference: str = _param(experiment_default("workspace_reference"))
    skill_library: str = _param(experiment_default("skill_library"))


@dataclass(frozen=True, kw_only=True)
class ScheduleParams:
    """When the strategy decides and what it is measured against."""

    strategy_period: str = _param("day")
    inference_time: str = _param("08:00")
    # Sets nothing since the approval mode was removed; still written because
    # the round launcher pins it (scripts/experiments/_round.py).
    initial_control_mode: str = _param("auto")
    benchmark_index: str = _param(experiment_default("benchmark_index"))


@dataclass(frozen=True, kw_only=True)
class DataParams:
    """The decision view's windows and domains, its prebuilt view cache and the
    universe screen."""

    window_months: int = _param(experiment_default("window_months"))
    daily_window_months: int | None = _param()
    fundamentals_window_months: int | None = _param()
    events_window_months: int | None = _param()
    macro_window_months: int | None = _param()
    text_window_months: int | None = _param()
    intraday_trade_days: int = _param(SnapshotConfig().intraday_trade_days)
    # The default selection is the default prebuilt view cache's: the ten
    # default fundamentals, the index and Shenwan macro datasets and the
    # announcement titles, with events and intraday off.
    include_fundamentals: bool = _param(True)
    include_macro: bool = _param(True)
    include_events: bool = _param(False)
    include_text: bool = _param(True)
    include_intraday: bool = _param(False)
    fundamental_datasets: tuple[str, ...] = _param(())
    macro_datasets: tuple[str, ...] = _param(("index_daily", "index_dailybasic", "sw_daily", "index_weight"))
    events_datasets: tuple[str, ...] = _param(())
    text_datasets: tuple[str, ...] = _param(("anns_d",))
    # Which prebuilt PIT view tree this experiment hardlinks from and whose
    # research release it pins. A named tree must exist and carry exactly this
    # experiment's selection, so an experiment that selects other datasets names
    # a tree prebuilt for them, or "" for none (it cold-builds every view on the
    # newest release).
    pit_views_seed: str = _param(str(DEFAULT_PIT_VIEWS_SEED))
    # The universe reaches the agent unfiltered (all boards, ST included, no
    # new-listing exclusion); the strategy applies its own filters.
    screen_exclude_st: bool = _param(False)
    screen_exclude_new_listed_days: int = _param(0)
    screen_min_circ_mv_yi: float | None = _param()
    screen_max_circ_mv_yi: float | None = _param()
    screen_min_price: float | None = _param()
    screen_max_price: float | None = _param()
    screen_boards: tuple[str, ...] = _param(())
    max_intraday_row_group_rows: int = _param(2_000_000)


@dataclass(frozen=True, kw_only=True)
class BudgetParams:
    """The research session's budgets and attempts (``ExperimentConfig``)."""

    max_null_controls: int = _param(experiment_default("max_null_controls"))
    batch_validate_max_concurrency: int = _param(experiment_default("batch_validate_max_concurrency"))
    max_llm_calls: int = _param(experiment_default("max_llm_calls"))
    max_api_spend_usd: float = _param(experiment_default("max_api_spend_usd"))
    session_max_attempts: int = _param(experiment_default("session_max_attempts"))
    max_research_minutes: int = _param(experiment_default("max_research_minutes"))
    per_call_timeout_seconds: int = _param(experiment_default("per_call_timeout_seconds"))
    record_failed_attempts: bool = _param(experiment_default("record_failed_attempts"))


@dataclass(frozen=True, kw_only=True)
class SandboxParams:
    """The strategy container's boundary, the experiment's GPU request and the
    session container's limits (``sandbox.SandboxSpec``)."""

    strategy_fit_timeout_seconds: int = _param(experiment_default("strategy_fit_timeout_seconds"))
    strategy_inference_timeout_seconds: int = _param(experiment_default("strategy_inference_timeout_seconds"))
    strategy_cpus: float = _param(experiment_default("strategy_cpus"))
    strategy_memory: str = _param(experiment_default("strategy_memory"))
    strategy_state_disk: str = _param(experiment_default("strategy_state_disk"))
    gpu_count: int = _param(SandboxSpec().gpu_count)
    agent_sandbox_image: str | None = _host(DEFAULT_IMAGE)
    # Left out: the strategy container's strategy_cpus and strategy_memory.
    agent_sandbox_cpus: float | None = _host()
    agent_sandbox_memory: str | None = _host()
    agent_sandbox_pids: int | None = _host(SandboxSpec().pids)
    agent_sandbox_tmpfs: str | None = _host(SandboxSpec().tmpfs_size)
    # Stamped by the console at creation from the host's pretrained store
    # (``environment.pretrained``), never requested.
    pretrained_manifest_sha256: str | None = _host()


@dataclass(frozen=True, kw_only=True)
class BrokerParams:
    """The account (``broker.BrokerProfile``)."""

    initial_cash: float = _param(1_000_000.0)
    max_total_holdings: int | None = _param()
    max_single_name_weight: float | None = _param()
    commission_bps: float = _param(BrokerProfile().commission_bps)
    slippage_bps: float = _param(BrokerProfile().slippage_bps)
    # The boards the account may buy on. Left null, creation stamps the ones
    # its initial cash qualifies for (``broker.default_permitted_boards``), so
    # the stamped params.json always names them; an experiment without the key is
    # unrestricted.
    permitted_boards: tuple[str, ...] | None = _param()


@dataclass(frozen=True, kw_only=True)
class AgentParams:
    """The session's models and context management (``worker.LLMWorkerSettings``)."""

    # The main conversation runs on the relay's gpt-6-astra, the sub-agents
    # and compaction on its DeepSeek-V4.1-Flash: hosted APIs are the operating
    # default, the local model the fallback once the API quota is exhausted.
    model: str = _param(GPT_ASTRA_MODEL)
    subagent_model: str = _param(DEEPSEEK_FLASH_MODEL)
    compact_model: str = _param(DEEPSEEK_FLASH_MODEL)
    # Which relay credential the experiment's calls start on (``primary`` or
    # ``alternate``); the other is the fallback.
    llm_key: str = _param("primary")
    reasoning_effort: str = _param("xhigh")
    no_thinking: bool = _param(False)
    disable_context_compact: bool = _param(False)
    # Null: each conversation role compacts at its bound, its model's and the
    # compaction model's window less the output budget and a margin
    # (``worker.LLMWorkerSettings.compaction_for``). A value is a lower
    # threshold; one above the bound is refused.
    compact_token_threshold: int | None = _param()
    compact_keep_recent_messages: int = _param(10)
    compact_max_tokens: int = _param(10_000)
    compact_max_calls: int = _param(10)
    llm_env_file: str | None = _host(".env")
    # Stands in for a main role the file leaves out or empty (:func:`session_models`).
    llm_model: str | None = _host()
    # Left out: per_call_timeout_seconds.
    llm_timeout_seconds: float | None = _host()
    llm_max_retries: int | None = _host(DEFAULT_LLM_MAX_RETRIES)
    llm_retry_backoff_seconds: float | None = _host(DEFAULT_LLM_RETRY_BACKOFF_SECONDS)
    llm_temperature: float | None = _host(0.0)
    llm_max_response_tokens: int | None = _host(AGENT_MAX_OUTPUT_TOKENS)


def _managed(stamp: str, *, default: object = MISSING) -> Any:
    """A console-managed parameter the console stamps as ``stamp``.
    ``default``, what the worker reads when a worker-side params.json leaves
    it out, is the stamp unless named."""

    return _param(stamp if default is MISSING else default, surface=Surface.MANAGED, stamp=stamp)


@dataclass(frozen=True, kw_only=True)
class EnvironmentParams:
    """Where the experiment's code, data and work live and how it runs. The console
    stamps the only supported research environment; the rest belongs to a
    worker-side params.json or the command line."""

    strategy_path: str | None = _managed("configs/agent_output_template/main.py", default=None)
    # Left out: pit for an LLM experiment on a host with the raw lake, else daily.
    data_backend: str | None = _managed("pit", default=None)
    raw_dir: str | None = _managed("data/raw")
    fundamental_events_root: str | None = _managed("data/pit/fundamental_events")
    fundamental_events_status: str | None = _managed(
        "results/data_quality/fundamental_events_status.json"
    )
    execution_mode: str | None = _managed("sandbox")
    developer_mode: str | None = _managed("llm", default="baseline")
    daily_path: str | None = _host()
    experiments_root: str | None = _host()
    work_root: str | None = _host(".runtime/sandboxes")


# The console's own value of each required acceptance rule
# (``config.REQUIRED_ACCEPTANCE_KEYS``), which the rules themselves leave
# without a default. The residual statistic's level is one for the whole
# program, so that every reading is stated against the same bar
# (``_round.BASE_EXPECTED_DEFAULTS`` pins it). How many of the geometry's years
# a session researches on is the experiment's experimental choice, so the console has
# no value for it: every create request states it.
_REQUIRED_ACCEPTANCE_DEFAULTS: dict[str, object] = {
    "alpha": 0.05,
    "research_years": None,
}


def _acceptance_fields() -> list[tuple[str, object, Any]]:
    """The acceptance parameters, read off the rules' own declaration
    (``config.AcceptanceRules``) and, for a rule without a default, the
    console's own value above.

    The limits a request may leave blank come first: the tracking mandate,
    which naming ``tracking_error_cap`` switches on (a blank beta band then
    takes ``config.MANDATED_DEFAULTS``), and the two drawdowns, which stay the
    rules' own 0.45 / 0.30 unless the request names them. Every other rule is
    filled, so the form shows the values an empty request would stamp
    (``config.acceptance_for``).
    """

    if set(_REQUIRED_ACCEPTANCE_DEFAULTS) != set(REQUIRED_ACCEPTANCE_KEYS):
        raise RuntimeError("every required acceptance rule needs its console value")
    rules = {
        **{
            name: _REQUIRED_ACCEPTANCE_DEFAULTS[name]
            if name in REQUIRED_ACCEPTANCE_KEYS
            else acceptance_default(name)
            for name in ACCEPTANCE_KEYS
        },
        "max_drawdown": None,
        "active_max_drawdown": None,
    }
    ordered = {**{name: None for name, value in rules.items() if value is None}, **rules}
    hints = get_type_hints(AcceptanceRules)
    return [
        (name, hints[name] | None, _param(value, required=value is None and name in REQUIRED_ACCEPTANCE_KEYS))
        for name, value in ordered.items()
    ]


AcceptanceParams = make_dataclass(
    "AcceptanceParams", _acceptance_fields(), frozen=True, kw_only=True
)
AcceptanceParams.__module__ = __name__
AcceptanceParams.__doc__ = (
    "The acceptance rules as a create request states them (``config.acceptance_for``); "
    "null is a limit the request left blank."
)


@dataclass(frozen=True, kw_only=True)
class ExperimentParams:
    """One experiment's parameters, grouped; ``from_record`` reads the flat
    params.json form. Nothing writes a file from the record: the console
    and the command line write the request they validated through it."""

    experiment_id: str | None = _param(required=True)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    packs: PackParams = field(default_factory=PackParams)
    schedule: ScheduleParams = field(default_factory=ScheduleParams)
    data: DataParams = field(default_factory=DataParams)
    budgets: BudgetParams = field(default_factory=BudgetParams)
    sandbox: SandboxParams = field(default_factory=SandboxParams)
    acceptance: AcceptanceParams = field(default_factory=AcceptanceParams)  # type: ignore[valid-type]
    broker: BrokerParams = field(default_factory=BrokerParams)
    agent: AgentParams = field(default_factory=AgentParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ExperimentParams:
        """The parameters a params.json (or a merged create request) states.

        Refuses a key no parameter declares and a value whose JSON type is
        not its parameter's; ``_``-prefixed stamps are skipped. A missing key
        takes its parameter's default.
        """

        unknown = sorted(
            key for key in record if not str(key).startswith("_") and key not in PARAMETERS
        )
        if unknown:
            raise ValueError(f"unknown experiment parameters: {unknown}")
        values: dict[str | None, dict[str, object]] = {}
        for name, spec in PARAMETERS.items():
            if name in record:
                values.setdefault(spec.group, {})[name] = _checked(name, spec.kind, record[name])
        top = values.pop(None, {})
        groups = {
            item.name: item.type_(**values.get(item.name, {}))  # type: ignore[operator]
            for item in _groups()
        }
        return cls(**top, **groups)  # type: ignore[arg-type]



@dataclass(frozen=True)
class ParameterSpec:
    """One parameter: its group (None for a top-level field), its declared
    type, who may set it and its default."""

    group: str | None
    kind: object
    surface: Surface
    required: bool
    default: object
    stamp: object


@dataclass(frozen=True)
class _Group:
    name: str
    type_: type


def _groups() -> Iterator[_Group]:
    hints = get_type_hints(ExperimentParams)
    for item in fields(ExperimentParams):
        kind = hints[item.name]
        if isinstance(kind, type) and is_dataclass(kind):
            yield _Group(item.name, kind)


def _spec(group: str | None, item: Field[object], kind: object) -> ParameterSpec:
    default = item.default if item.default is not MISSING else None
    return ParameterSpec(
        group=group,
        kind=kind,
        surface=item.metadata["surface"],
        required=item.metadata["required"],
        default=default,
        stamp=item.metadata["stamp"],
    )


@cache
def _parameters() -> dict[str, ParameterSpec]:
    specs: dict[str, ParameterSpec] = {}
    hints = get_type_hints(ExperimentParams)
    groups = {group.name for group in _groups()}
    for item in fields(ExperimentParams):
        if item.name in groups:
            continue
        specs[item.name] = _spec(None, item, hints[item.name])
    for group in _groups():
        group_hints = get_type_hints(group.type_)
        for item in fields(group.type_):
            if item.name in specs:
                raise RuntimeError(f"parameter {item.name} is declared twice")
            specs[item.name] = _spec(group.name, item, group_hints[item.name])
    return specs


PARAMETERS: Mapping[str, ParameterSpec] = _parameters()


def _checked(name: str, kind: object, value: object) -> object:
    """``value`` as parameter ``name`` holds it, or ValueError when its JSON
    type is not the declared one. Null is always accepted: unset; so is the
    empty string a cleared form field sends for a parameter that is not
    text."""

    union = get_origin(kind) in (Union, UnionType)
    bases = {arg for arg in (get_args(kind) if union else (kind,)) if arg is not type(None)}
    if value is None or (value == "" and str not in bases):
        return None
    for base in bases:
        origin = getattr(base, "__origin__", base)
        if origin is bool and isinstance(value, bool):
            return value
        if origin is int and isinstance(value, int) and not isinstance(value, bool):
            return value
        if origin is float and isinstance(value, int | float) and not isinstance(value, bool):
            return value
        if origin is str and isinstance(value, str):
            return value
        if origin is tuple and isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
            return tuple(value)
    raise ValueError(f"experiment parameter {name} must be {_kind_label(bases)}, not {type(value).__name__}")


def _kind_label(bases: set[object]) -> str:
    labels = {bool: "a boolean", int: "an integer", float: "a number", str: "a string"}
    return " or ".join(sorted(labels.get(getattr(base, "__origin__", base), "a list of strings") for base in bases))


def parameters(surface: Surface) -> list[str]:
    """The parameters ``surface`` sets, in declaration order."""

    return [name for name, spec in PARAMETERS.items() if spec.surface is surface]


def console_defaults() -> dict[str, object]:
    """What the console stamps for every parameter a create request leaves out."""

    return {name: PARAMETERS[name].default for name in parameters(Surface.CONSOLE)}


def managed_values() -> dict[str, object]:
    """The values the console stamps for the parameters it manages."""

    return {name: PARAMETERS[name].stamp for name in parameters(Surface.MANAGED)}


def read_param(params: Mapping[str, object], name: str) -> object:
    """Parameter ``name`` as a params.json states it, null included, or its
    declared default when the file leaves it out. A name no parameter
    declares raises ``KeyError``."""

    return params.get(name, PARAMETERS[name].default)


def read_param_or_default(params: Mapping[str, object], name: str) -> object:
    """:func:`read_param`, with a stated null or empty string also taking
    the declared default: for a parameter whose reader takes an empty value
    as unset."""

    value = read_param(params, name)
    return PARAMETERS[name].default if value in (None, "") else value


def session_models(params: Mapping[str, object]) -> tuple[str, str, str]:
    """The models an experiment's main, sub-agent and compaction roles run on.

    A role the file leaves out or empty takes its own default, except that
    a worker-side ``llm_model`` stands in for the main role.
    """

    return (
        str(params.get("model") or params.get("llm_model") or PARAMETERS["model"].default),
        str(read_param_or_default(params, "subagent_model")),
        str(read_param_or_default(params, "compact_model")),
    )


def console_request(
    request: Mapping[str, object],
    *,
    defaults: Mapping[str, object],
    experiments_root: str,
    work_root: str,
) -> dict[str, object]:
    """A console create request merged into the params it creates, before
    the console's own stamps that need the deployment (the pretrained pin,
    the resolved rules, the boards, the time).

    The request-level checks of a create, shared by the console and the
    round launcher: no console-managed or host parameter, no unknown one,
    every required one stated, a valid id, and every value of its declared
    type. ``defaults`` are the console's (``hitl_state.WEB_CREATE_DEFAULTS``).
    """

    closed = sorted(
        key for key in request if key in NON_PERSISTABLE_PARAMS
        or (key in PARAMETERS and PARAMETERS[key].surface in (Surface.MANAGED, Surface.HOST))
    )
    if closed:
        raise ValueError("console-managed parameters are not accepted: " + ", ".join(closed))
    unknown = sorted(set(request) - set(defaults))
    if unknown:
        raise ValueError("unknown experiment parameters: " + ", ".join(unknown))
    merged = {**defaults, **request}
    missing = sorted(
        name
        for name, spec in PARAMETERS.items()
        if spec.required and merged.get(name) in (None, "")
    )
    if missing:
        raise ValueError("missing required experiment parameters: " + ", ".join(missing))
    experiment_id = str(request.get("experiment_id") or "").strip()
    if not is_experiment_id(experiment_id):
        raise ValueError(
            f"experiment_id must match {EXPERIMENT_ID.pattern} (letters, digits, _ and -)"
        )
    merged.update(
        {
            **managed_values(),
            "experiment_id": experiment_id,
            "experiments_root": experiments_root,
            "work_root": work_root,
        }
    )
    ExperimentParams.from_record(merged)
    return merged


__all__ = [
    "NON_PERSISTABLE_PARAMS",
    "PARAMETERS",
    "AcceptanceParams",
    "AgentParams",
    "BrokerParams",
    "BudgetParams",
    "DataParams",
    "EnvironmentParams",
    "ExperimentParams",
    "GeometryParams",
    "PackParams",
    "ParameterSpec",
    "SandboxParams",
    "ScheduleParams",
    "Surface",
    "console_defaults",
    "console_request",
    "managed_values",
    "parameters",
    "read_param",
    "read_param_or_default",
    "session_models",
]
