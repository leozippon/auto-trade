"""Configuration for one research arm."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, MISSING, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from autotrade.common.ids import EXPERIMENT_ID, is_experiment_id
from autotrade.environment.artifacts import (
    ArtifactRevision,
    FrozenArtifact,
    ModificationConstraints,
)
from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.contracts import (
    DEFAULT_BENCHMARK_INDEX,
    benchmark_index_label,
)
from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.sandbox import SandboxConfig, SandboxSpec
from autotrade.environment.strategy import StrategySchedule

from . import verdict
from .calendar import FULL_SPAN, ResearchGeometry, step_windows
from .ledger import ENVIRONMENT_BLOCKED, SessionKind
from .skills import DEFAULT_OPERATING_MEMORY

ExecutionMode = Literal["sandbox", "trusted"]

# Increment only when the cached snapshot/replay on-disk contract changes.
# Source revisions are intentionally not cache inputs: harmless code changes
# should not invalidate every expensive data view.
# v6: events/macro unions carry the full configured-dataset schema (typed
# zero-row contributions for datasets without visible rows in the window).
# v7: the snapshot manifest carries dataset_columns for the unit reference.
# v8: incomplete/unusable vendor columns are dropped from the union domains
# (SNAPSHOT_EXCLUDED_COLUMNS in environment.data.snapshot), so a v7 view still
# carries fields the Agent must no longer see.
# v9: the daily join drops its duplicate close_basic/pre_close_limit columns.
# v10: same-day macro tables carry the close contract stamp
# (MACRO_DATASET_CONTRACTS) instead of the raw date-EOD placeholder, so a v9
# macro view shows T rows a trading day later than a v10 one.
# v11: daily.is_suspended counts only suspend_d halts (suspend_type "S"); v10
# also flagged resumption rows ("R"), i.e. normally traded sessions.
# v12: cyq_perf.his_high (a proven post-decision price leak) and its unverified
# same-source his_low leave the events union (SNAPSHOT_EXCLUDED_COLUMNS), so a
# v11 decision/replay view still carries his_high. A cached view, or a PIT view
# seed, built under v11 must be rebuilt; SNAPSHOT_CACHE_FORMAT_VERSION is the
# only reuse gate and nothing rebuilds a stale view automatically.
# v13: top_list.name leaves the events union (SNAPSHOT_EXCLUDED_COLUMNS). It is
# the name as of the vendor's pull, not the day's, so a v12 view that loads
# top_list carries later names and ST marks into history.
# v14: a row whose key the raw lake first held after its vendor-date stamp
# becomes visible only from that first-seen time, so a v13 view shows rows the
# vendor added after the fact from their vendor date.
# v15: a fina_indicator_vip row is visible no earlier than the latest statement
# announcement of its period (the vendor files restated indicators under the
# original ann_date); a v14 view shows restated indicators up to years early.
# The stamp lives in the PIT event index a release clones, so only a view built
# from a release whose index this code wrote carries it.
# v16: one anns_d text-index row per filing (ts_code + cninfo id); a midnight rec_time now stamps date-only (via the restamped lake).
SNAPSHOT_CACHE_FORMAT_VERSION = 16

# Trailing wrap-up grace added to the research session budget. Not a console or
# worker knob; ResearchSessionRequest carries the seconds to the Agent runner.
DEFAULT_DEADLINE_GRACE_MINUTES = 10
# A research session's time, null-control and model-call budgets are
# arm-level: one session per arm spends them across every attempt.
DEFAULT_MAX_RESEARCH_MINUTES = 2400
DEFAULT_MAX_NULL_CONTROLS = 12
DEFAULT_MAX_LLM_CALLS = 6400
# What one arm's research session may spend on hosted model APIs, in USD at the
# catalog's list prices (``environment.llm.pricing``); the local model is free.
DEFAULT_MAX_API_SPEND_USD = 50.0
# ``validate``: one formal step that fans out a pre-registered candidate
# set over one span. Audited sessions reached at most two formal Validations
# each when every candidate was its own serial step, each branching off the
# last, which made the evidence per decision too thin. A batch fixes the parent
# and the span for every candidate, so their numbers are comparable, and
# pre-registers each hypothesis before any result exists. One candidate is a
# round too: the hypothesis is the same binding pre-registration at any width.
BATCH_VALIDATE_MIN_CANDIDATES = 1
# The rules' ``validates_per_step`` bounds how many calls an interval between
# tests may make; this cap only bounds what one observation may carry, and six
# screening candidates already make a wide round.
BATCH_VALIDATE_MAX_CANDIDATES = 6
# Concurrent replays per batch, the arm's ``batch_validate_max_concurrency``
# (1 up to the candidate cap; more workers than candidates run nothing). Each
# replay holds its own result/as-of directory and its own strategy container —
# two when the candidate declares fit, whose read-write state bind needs a
# second worker — and the Timeview stash serializes part publication across
# evaluations, so the bound is host capacity (up to two containers per replay
# at the strategy container's CPUs (SandboxSpec.cpus), and on a GPU arm every one of them on the arm's own
# cards), not correctness — which only holds because both strategy deadlines
# scale with the batch's own width (``session_tools._batch_replay_timeouts``);
# a fixed clock made the outcome depend on how many siblings happened to share
# the host.
DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY = 3

# Research on four July-June years, a twelve-month forward test after them,
# and a Held-out quarter the replay clips to the release end
# (docs/PIPELINE_DESIGN.md). The PIT view seed is planned over it.
DEFAULT_RESEARCH_GEOMETRY = ResearchGeometry(
    research_start="20210701",
    research_end="20250630",
    forward_end="20260630",
    heldout_end="20260930",
)

# The exploration PIT view seed an experiment hardlinks completed views from
# unless its ``pit_views_seed`` parameter names another tree, and the scratch
# directory the offline prebuild pins its research release in. Repo-relative
# and gitignored; the contract that decides reuse lives in
# ``autotrade.pipelines.pit_views_seed``. Defined here because both the console
# creation defaults and that module need the path.
DEFAULT_PIT_VIEWS_SEED = Path("data/pit_views_seed/explore")
DEFAULT_PIT_VIEWS_SEED_WORKSPACE = Path("data/pit_views_seed/explore_workspace")


@dataclass(frozen=True)
class StrategyExperimentConfig:
    strategy_path: Path
    schedule: StrategySchedule = field(default_factory=StrategySchedule)
    broker_profile: BrokerProfile = field(default_factory=BrokerProfile)
    execution_mode: ExecutionMode = "sandbox"
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    # The strategy's frozen ``models/`` tree, mounted read-only when it has
    # one. The per-replay ``fit`` state directory is not a configuration knob:
    # the replay creates it empty and discards it with the run.
    models_dir: Path | None = None

    def __post_init__(self) -> None:
        path = Path(self.strategy_path).resolve()
        if not path.is_file():
            raise ValueError(f"strategy file does not exist: {path}")
        if self.execution_mode not in ("sandbox", "trusted"):
            raise ValueError("execution_mode must be sandbox or trusted")
        object.__setattr__(self, "strategy_path", path)
        if self.models_dir is not None:
            models = Path(self.models_dir).resolve()
            if not models.is_dir():
                raise ValueError(f"models directory does not exist: {models}")
            object.__setattr__(self, "models_dir", models)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _whole_int(value: object, name: str) -> int:
    """A JSON-safe integer: ``2`` or ``2.0``, never ``2.5`` and never a bool."""

    number = _finite_number(value, name)
    integer = int(number)
    if number != integer:
        raise ValueError(f"{name} must be an integer")
    return integer


def _open_unit(value: object, name: str, *, include_one: bool) -> float:
    """A probability-like knob: ``(0, 1]`` or ``(0, 1)``."""

    number = _finite_number(value, name)
    ok = 0 < number <= 1 if include_one else 0 < number < 1
    if not ok:
        raise ValueError(f"{name} must be in {'(0, 1]' if include_one else '(0, 1)'}")
    return number


def _rule(default: object, text: str) -> Any:
    """One acceptance rule's field: its default, and in one line what it limits."""

    return field(default=default, metadata={"help": text})


def _required(text: str) -> Any:
    """One acceptance rule without a default (``REQUIRED_ACCEPTANCE_KEYS``):
    every record and every create request names it, so a record written
    before the rule existed is refused rather than read under a default."""

    return field(kw_only=True, metadata={"help": text})


# The step lengths a test may advance the validation frontier by, in months:
# a quarter by default, a month where a round wants finer steps.
TEST_STEP_MONTHS = (1, 3)


@dataclass(frozen=True)
class AcceptanceRules:
    """The arm's create-time rules: the research geometry's split, the test
    steps, the registration every ``test`` checks its candidate against and
    the readings the forward stage reports.

    A candidate is refused at registration on a non-finite metric or a
    drawdown over ``max_drawdown`` over its span (``evaluate``), and on a
    broken tracking mandate when one is set. Nothing else gates: the host
    reads the rolling record and the forward reading with the residual
    statistic (``pipelines/verdict.py``) at ``alpha`` and records them. Every
    field is stamped into ``params.json`` at creation so a later change to
    the defaults cannot move an arm that already exists.

    This is the one declaration of the rules. The record (``to_record``), the
    parameters a worker accepts (``ACCEPTANCE_KEYS``), the console's creation
    defaults (``hitl_state.WEB_CREATE_DEFAULTS``) and the audit entrypoint's
    flags (``scripts/experiments/_cli.py``) are read off its fields, and the
    verdict functions take the rules whole. A rule without a default
    (``REQUIRED_ACCEPTANCE_KEYS``) is named by every record, so a record
    written before it existed is refused rather than judged under it.
    """

    # The registration's limit over the candidate's span, and the Paper book's
    # kill limits.
    max_drawdown: float = _rule(0.45, "Equity drawdown limit of the freeze registration and of Paper.")
    cost_stress_multiplier: float = _rule(
        2.0,
        "Forward reading: multiple of the profile's slippage its reported active excess is stressed at.",
    )
    active_max_drawdown: float = _rule(
        0.30, "Paper's drawdown limit of the active series (strategy minus zero-skill panel)."
    )
    # The tracking mandate: the strategy's own residual tracking error against
    # the factor legs and its market beta over the nominee's span, a
    # registration check, and reported forward. ``None`` means no mandate:
    # both are measured and reported, not graded. The band is set exactly when the cap is -- a cap alone is
    # cheapest to meet by dropping beta.
    tracking_error_cap: float | None = _rule(
        None,
        "Tracking mandate: residual tracking error cap against the arm's benchmark index; "
        "giving it turns the mandate on for this arm.",
    )
    beta_min: float | None = _rule(None, "Tracking mandate: lower end of the market beta band.")
    beta_max: float | None = _rule(None, "Tracking mandate: upper end of the market beta band.")
    forward_confidence: float = _rule(
        verdict.FORWARD_CONFIDENCE,
        "Forward reading: one-sided block-bootstrap confidence of its reported lower bound.",
    )
    # The residual statistic's one-sided level (``verdict.residual_bar``):
    # the bar every reading is stated against, never a gate.
    alpha: float = _required(
        "Residual statistic: one-sided level of the bar stated beside every reading."
    )
    # The geometry's split: the first research_years are the research years a
    # session sees; the rest are the validation years its tests reveal.
    research_years: int = _required(
        "Research years the session sees; the rest of the geometry are the validation years."
    )
    # How far one test advances the validation frontier, and how many
    # validate calls an interval between tests may make.
    test_step_months: int = _rule(3, "Months the validation frontier advances per test (1 or 3).")
    validates_per_step: int = _rule(
        6, "validate calls allowed before the first test and between consecutive tests."
    )

    def __post_init__(self) -> None:
        mandate = (self.tracking_error_cap, self.beta_min, self.beta_max)
        for name, value in zip(
            ("max_drawdown", "cost_stress_multiplier", "active_max_drawdown"),
            (self.max_drawdown, self.cost_stress_multiplier, self.active_max_drawdown),
            strict=True,
        ):
            _finite_number(value, name)
        for name in ("max_drawdown", "active_max_drawdown"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.cost_stress_multiplier < 1:
            raise ValueError("cost_stress_multiplier must be at least one")
        object.__setattr__(
            self,
            "forward_confidence",
            _open_unit(self.forward_confidence, "forward_confidence", include_one=False),
        )
        object.__setattr__(self, "alpha", _open_unit(self.alpha, "alpha", include_one=False))
        object.__setattr__(self, "research_years", _whole_int(self.research_years, "research_years"))
        if self.research_years < 1:
            raise ValueError("research_years must be an integer >= 1")
        object.__setattr__(
            self, "test_step_months", _whole_int(self.test_step_months, "test_step_months")
        )
        if self.test_step_months not in TEST_STEP_MONTHS:
            raise ValueError(f"test_step_months must be one of {TEST_STEP_MONTHS}")
        object.__setattr__(
            self, "validates_per_step", _whole_int(self.validates_per_step, "validates_per_step")
        )
        if self.validates_per_step < 1:
            raise ValueError("validates_per_step must be an integer >= 1")
        if any(value is None for value in mandate):
            if any(value is not None for value in mandate):
                raise ValueError(
                    "tracking_error_cap, beta_min and beta_max are set together or not at all"
                )
            return
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in mandate):
            raise ValueError("tracking_error_cap, beta_min and beta_max must be finite")
        if not self.tracking_error_cap > 0:
            raise ValueError("tracking_error_cap must be positive")
        if not self.beta_min < self.beta_max:
            raise ValueError("beta_min must be below beta_max")

    def to_record(self) -> dict[str, object]:
        return {rule.name: getattr(self, rule.name) for rule in fields(self)}

    @property
    def mandate(self) -> dict[str, float | None]:
        """The tracking mandate as a record's thresholds state it."""

        return {
            "tracking_error_cap": self.tracking_error_cap,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AcceptanceRules:
        """These rules from a ``to_record`` mapping, ignoring unknown keys.

        Run manifests and ``hitl/params.json`` are read back long after they
        were written, so a record that carries a retired key (the former
        ``min_return``/``min_sharpe`` targets, the deflated-Sharpe bars, the
        ``RETIRED_SWITCHES``, the sealed gate's ``OBSOLETE_KEYS``) must still
        rebuild the rules it does name. A record that lacks a required rule
        (``REQUIRED_ACCEPTANCE_KEYS``) is refused: it was written under rules
        these are not.
        """

        missing = [
            rule.name for rule in fields(cls) if rule.default is MISSING and rule.name not in record
        ]
        if missing:
            raise ValueError(
                f"the acceptance rules lack the required {', '.join(missing)}: a record "
                "written before those rules existed is not read under them"
            )
        return cls(**{rule.name: record[rule.name] for rule in fields(cls) if rule.name in record})  # type: ignore[arg-type]

    def agent_facts(self, *, validation_years: int) -> dict[str, object]:
        """The ``acceptance_rules`` run fact, derived from these rules so no
        prompt restates a threshold. ``validation_years`` is how many years
        the arm's geometry leaves after the research years, for the bar over
        the whole validation period. Rules and that count only: the dates of
        the research and validation years are the research geometry's
        (``research_session.research_geometry_record``), and no forward or
        Held-out date appears anywhere a session reads."""

        if isinstance(validation_years, bool) or not isinstance(validation_years, int) or validation_years < 1:
            raise ValueError(f"validation_years must be a positive integer, got {validation_years!r}")

        if self.tracking_error_cap is None:
            mandate: object = (
                "none for this arm: benchmark.tracking_error and "
                "benchmark.market_beta are reported, not graded"
            )
        else:
            mandate = {
                "tracking_error": (
                    f"<= {self.tracking_error_cap}: the node's own residual tracking "
                    "error against the factor legs (benchmark.tracking_error)"
                ),
                "market_beta": (
                    f"within {self.beta_min}..{self.beta_max} "
                    "(benchmark.market_beta)"
                ),
                "role": (
                    "a constraint the book must satisfy over the candidate's span "
                    "and again forward; meeting it is not evidence of skill"
                ),
            }
        years = TRADING_DAYS_PER_YEAR
        return {
            "graded_series": (
                "every return statistic below is measured on the ACTIVE series: the "
                "node's daily return after every cost minus the host's zero-skill panel "
                f"composite, the equal-weight mean of {verdict.PANEL_DRAWS} random-name "
                "copies of the node's own filled trades (same entry and exit instants, "
                "same money per round trip, same Broker and costs; each name replaced by "
                "one on the same side of the arm's benchmark_index membership that one "
                "board lot of that money could buy). Its annualized mean is the active "
                "excess, its annualized standard deviation the active tracking error "
                "and their ratio the active information ratio. Every formal validation "
                "reports them as benchmark.active_excess, "
                "benchmark.active_tracking_error, benchmark.active_information_ratio "
                "and benchmark.active_max_drawdown (over the validation's own span), "
                "per research year as sub_windows[].active_excess_return, and day by "
                "day as the active_return of its daily_series. The strategy exposes "
                "nothing for this beyond its orders"
            ),
            "instrument": {
                "statistic": (
                    "the residual information ratio of the active series: its daily "
                    "return regressed by OLS on an intercept, the benchmark_index return "
                    "(market), the size leg and the value leg (regression_method), the "
                    f"intercept times sqrt({years}) over the residual standard deviation; "
                    f"years = trading days / {years}"
                ),
                "bar": "z(1 - alpha) / sqrt(years): the level a zero-skill reading reaches with probability alpha; stated beside every reading, never a gate",
                "alpha": self.alpha,
                "bar_over_one_year": round(verdict.residual_bar(1.0, self.alpha), 3),
                "bar_over_validation_period": round(
                    verdict.residual_bar(float(validation_years), self.alpha), 3
                ),
                "on_spans": (
                    "every validate row reports it over its own span (residual_reading) "
                    "with the bar for that span's length; every test reports it over the "
                    "rolling record so far with the bar for the record's length"
                ),
            },
            "validation": {
                "research_years": self.research_years,
                "test_step_months": self.test_step_months,
                "validates_per_step": self.validates_per_step,
                "test": (
                    "test(candidate) advances the validation frontier one step: the host "
                    "replays the candidate as a fresh book from the validation start to the "
                    "new frontier with its matched panel, records the newest step as the "
                    "rolling record's entry for that step and returns the whole replay's "
                    "statistics, a per-month table, the record to date and its residual "
                    "statistic. The session's data view stays at the end of the research "
                    "years, so the validation years reach it only through these results. "
                    "The candidate must register (registration below) and may change at any step"
                ),
                "validates_per_step_rule": (
                    f"at most {self.validates_per_step} validate calls before the first "
                    "test and between two tests; a further call is refused until the next test"
                ),
                "end": (
                    "the session ends when the frontier reaches the end of the validation "
                    "years; the candidate of the last test is the frozen strategy. There is "
                    "no action that ends the session earlier. If the time, call or spend "
                    "budget runs out after a test, the host completes the record with the "
                    "last tested candidate; before any test, with the last registrable "
                    "full-span candidate"
                ),
                "after": (
                    "the frozen strategy and its seed replicates are then replayed by the "
                    "host over the Forward year and the Held-out months as one book; those "
                    "readings are recorded, gate nothing and reach no session"
                ),
            },
            "registration": {
                "span": f"the candidate replayed the whole research-years span (span={FULL_SPAN})",
                "control": "a node registered as a control (control: true) is never a candidate",
                "finite_metrics": "total_return/max_drawdown/sharpe must be finite",
                "max_drawdown": (
                    f"<= {self.max_drawdown}: equity drawdown over the candidate's span"
                ),
                "seed_replicates": (
                    "a candidate whose main.py defines fit (it trains a model) names at "
                    "least one seed replicate in test's seed_replicates: another complete "
                    "full-span Validation of this session, registered as a candidate (not "
                    "a control), whose strategy is the candidate's bytes with exactly one "
                    "line changed, an integer assignment to a seed name (SEED, SEED_BASE, "
                    "or one ending in _SEED or _SEED_BASE; e.g. SEED_BASE = 1000 in the "
                    "candidate, 2000 in a replicate), and whose bytes differ from every "
                    "other registered one. The replicates named at the last test are "
                    "frozen beside the strategy and replayed forward with it. A candidate "
                    "without fit needs none, and one declared deterministic in test (its "
                    "package uses no randomness) owes none"
                ),
                "tracking_mandate": mandate,
                "checked": "at every test; a candidate that fails is refused and the frontier does not move",
            },
            "forward": {
                "reported": (
                    "one continuous replay of the frozen strategy over the twelve months "
                    "after the research geometry's last year and then Held-out, graded on its "
                    "active series against a panel drawn from that replay: the "
                    f"{self.forward_confidence:.0%} block-bootstrap lower bound of active "
                    "excess over the frozen book and its seed replicates, the active excess "
                    f"with slippage multiplied by {self.cost_stress_multiplier}, drawdowns, "
                    "turnover and the residual statistic; recorded beside the rolling "
                    "record, pooled across frozen strategies, deciding nothing"
                ),
            },
        }

    def evaluate(self, summary: dict[str, object]) -> list[str]:
        """The hard reasons a nomination is refused (``experiment.freeze_registration``):
        the ``nomination`` conditions of ``verdict.CONDITIONS``, a non-finite
        metric and a research-period drawdown over ``max_drawdown``, read off
        the nominee's replay summary. Only a summary from a completed
        evaluation reaches here; an aborted replay never produces one."""

        return verdict.judge(verdict.ConditionStage.NOMINATION, summary, {"max_drawdown": self.max_drawdown})


# Every rule's name, in declaration order: the keys of a rules record and of
# the arm parameters that set one.
ACCEPTANCE_KEYS: tuple[str, ...] = tuple(rule.name for rule in fields(AcceptanceRules))
# The rules without a default, which every record and create request names.
REQUIRED_ACCEPTANCE_KEYS: tuple[str, ...] = tuple(
    rule.name for rule in fields(AcceptanceRules) if rule.default is MISSING
)


def acceptance_default(name: str) -> object:
    """The default of one acceptance rule; ``KeyError`` for a required one."""

    rule = next(rule for rule in fields(AcceptanceRules) if rule.name == name)
    if rule.default is MISSING:
        raise KeyError(f"{name} is a required acceptance rule and has no default")
    return rule.default


# The per-arm on/off switches retired on 2026-10-07, when what they switched on
# became the only rules (docs/PIPELINE_DESIGN.md §6). Every params.json written
# before then carries them, and the run records of that time state the first
# four in their acceptance rules and thresholds. Readers accept exactly these
# keys and derive nothing from them; any other unknown parameter is refused.
RETIRED_SWITCHES = frozenset(
    {
        "require_raw_excess_at_cost_stress",
        "require_forward_plain_selection",
        "require_seed_replicates",
        "independent_offline_trials",
        "dividend_tax",
    }
)

# The rules and the create parameter retired on 2026-11-03 with the
# deflated-Sharpe freeze gate, its trial family and its lineage, and the
# forward and Held-out bars, when the sealed gate became the verdict's one
# condition. Every params.json written before then carries them; readers
# accept exactly these keys beside ``RETIRED_SWITCHES`` and derive nothing
# from them, so an earlier arm still lists in the console.
RETIRED_RULES = frozenset(
    {
        "min_active_ir",
        "min_dsr_probability",
        "min_positive_year_share",
        "min_full_span_validations",
        "recency_months",
        "lineage_arms",
        "min_mean_gross",
        "min_round_trips_per_month",
        "heldout_tolerance_z",
    }
)

# The keys retired with the freeze's ablation on 2026-11-05 and with the
# rolling validation on 2026-11-06: the ablation rule, the sealed gate's rule,
# the research session's replay-year budget and the hard-finalization window.
# Every params.json and ledger record written before then carries some of
# them; readers accept exactly these keys beside the two sets above and derive
# nothing from them, so an archived arm still loads and lists, while a create
# request naming one is refused like any unknown key.
OBSOLETE_KEYS = frozenset(
    {
        "sealed_years",
        "ablation_required",
        "max_replay_years",
        "finalize_before_deadline_seconds",
    }
)

# The beta band that travels with a tracking mandate when the create request
# names a cap and leaves the band blank. A cap alone is cheapest to meet by
# dropping beta. Drawdowns are not here: they stay the rules' own 45 % / 30 %
# unless the request names them.
MANDATED_DEFAULTS: Mapping[str, float] = {
    "beta_min": 0.85,
    "beta_max": 1.15,
}


def acceptance_for(params: Mapping[str, object]) -> AcceptanceRules:
    """An arm's acceptance rules from its own create-time parameters.

    The tracking mandate is a manual setting and no capital anywhere turns it
    on: it exists exactly when ``params`` names a ``tracking_error_cap``. A cap
    without a beta band takes ``MANDATED_DEFAULTS``. Drawdowns stay the rules'
    own 45 % equity and 30 % active unless the request names them. Without a
    cap the strategy's tracking error and beta are still measured and reported,
    just not graded.

    Only the rules' own keys are read; one that is absent or ``None`` takes the
    field default (today's verdict constants for the statistical bars), a
    required rule (``REQUIRED_ACCEPTANCE_KEYS``) has none and is refused, and
    a beta band named without a cap is refused by ``AcceptanceRules`` itself.
    """

    return AcceptanceRules.from_record(
        {
            **defaulted_acceptance(params),
            **{key: params[key] for key in REQUIRED_ACCEPTANCE_KEYS if params.get(key) is not None},
        }
    )


def defaulted_acceptance(params: Mapping[str, object]) -> dict[str, object]:
    """Every rule that has a default, as :func:`acceptance_for` resolves it
    from an arm's parameters: the value they name, else the default, and a
    blank beta band beside a named cap filled from ``MANDATED_DEFAULTS``.
    Read without the required rules, so a reader that needs only these (the
    console's threshold preview) reads an arm stamped before a rule became
    required as it was stamped."""

    rules = {
        rule.name: rule.default for rule in fields(AcceptanceRules) if rule.default is not MISSING
    }
    rules.update({key: params[key] for key in rules if params.get(key) is not None})
    if rules["tracking_error_cap"] is not None:
        for key, value in MANDATED_DEFAULTS.items():
            if params.get(key) is None:
                rules[key] = value
    return rules


@dataclass(frozen=True)
class RollingExperimentConfig:
    experiment_id: str
    experiments_root: Path
    # Research, forward and Held-out dates of the arm (pipelines/calendar.py).
    geometry: ResearchGeometry = DEFAULT_RESEARCH_GEOMETRY
    # The macro data floor is 2020-01, so 24 months before a July 2022
    # decision view is the most history every domain carries.
    window_months: int = 24
    # The index this arm is measured against: the benchmark leg of the style
    # attribution and of the factor loadings, and the membership the
    # zero-skill panel draws its replacements from. One of
    # ``contracts.BENCHMARK_INDEXES``.
    benchmark_index: str = DEFAULT_BENCHMARK_INDEX
    # The research session's budgets, spent across every attempt of the arm's
    # one session. The host's forward replay is charged to none of them.
    # Host-side random-portfolio null controls the session may request through
    # ``run_null_control`` (``null_control.PANEL_DRAWS`` replays each); the
    # frozen node's block is reused at freeze. 0 leaves the tool out.
    max_null_controls: int = DEFAULT_MAX_NULL_CONTROLS
    # Replays one ``validate`` call runs at once (see
    # ``DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY``).
    batch_validate_max_concurrency: int = DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    # Spent by the session's main, sub-agent and compaction calls together.
    max_api_spend_usd: float = DEFAULT_MAX_API_SPEND_USD
    # Attempts of the research session or of the forward replay before the
    # experiment fails with the last error.
    session_max_attempts: int = 3
    # Pausable effective inference minutes of the research session.
    max_research_minutes: int = DEFAULT_MAX_RESEARCH_MINUTES
    # Trailing wrap-up grace added to the session budget and forwarded on
    # ResearchSessionRequest.deadline_grace_seconds. Implementation default only.
    deadline_grace_minutes: int = DEFAULT_DEADLINE_GRACE_MINUTES
    per_call_timeout_seconds: int = 3600
    # Wall clock for one ``fit(context)`` invocation of the formal strategy;
    # the executor default is the single source. A slower fit fails the backtest.
    strategy_fit_timeout_seconds: int = int(SandboxConfig().fit_timeout_seconds)
    # The rest of the strategy container's boundary: the wall clock of one
    # ``generate_orders(context)`` call, the CPU quota its thread variables are
    # derived from, and its memory ceiling (a Docker limit). The executor's
    # defaults are the single source, so an arm whose params.json predates these
    # knobs replays exactly as before; the session container takes the same CPUs
    # and memory (``worker._llm_settings``).
    strategy_inference_timeout_seconds: int = int(SandboxConfig().timeout_seconds)
    strategy_cpus: float = SandboxConfig().spec.cpus
    strategy_memory: str = SandboxConfig().spec.memory
    # Optional experiment-level exploration direction injected into every
    # research session prompt. Per-session directives are additive.
    research_directive: str = ""
    # Optional repo-relative directory of Agent-readable notes copied into each
    # session's workspace/refs/. Empty copies nothing; a set path must exist and
    # be a directory.
    workspace_reference: str = ""
    # Whether the curated repository library mounts read-only into every
    # session (``curated``) or nothing does (``none``).
    operating_memory: str = DEFAULT_OPERATING_MEMORY
    # Also record failed validation attempts as lightweight dead-end nodes
    # (no output snapshot) so later sessions can see what was already tried.
    record_failed_attempts: bool = True
    schedule: StrategySchedule = field(default_factory=StrategySchedule)
    broker_profile: BrokerProfile = field(default_factory=BrokerProfile)
    # The arm's effective rules (``acceptance_for``): a tracking mandate only
    # when the create request named a ``tracking_error_cap``. No default: the
    # rules carry required keys only an arm's own parameters can state.
    acceptance: AcceptanceRules = field(kw_only=True)
    step_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )

    def __post_init__(self) -> None:
        if not is_experiment_id(self.experiment_id):
            raise ValueError(f"experiment_id must match {EXPERIMENT_ID.pattern}")
        if not isinstance(self.geometry, ResearchGeometry):
            raise TypeError("geometry must be a ResearchGeometry")
        for name in (
            "window_months",
            "max_llm_calls",
            "session_max_attempts",
            "max_research_minutes",
            "per_call_timeout_seconds",
            "strategy_fit_timeout_seconds",
            "strategy_inference_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # The executor's own checks on a CPU quota and a Docker memory limit.
        SandboxSpec(cpus=self.strategy_cpus, memory=self.strategy_memory, gpu=None)
        for name in (
            "max_null_controls",
            "deadline_grace_minutes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        width = self.batch_validate_max_concurrency
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or not 1 <= width <= BATCH_VALIDATE_MAX_CANDIDATES
        ):
            raise ValueError(
                "batch_validate_max_concurrency must be an integer from 1 to "
                f"{BATCH_VALIDATE_MAX_CANDIDATES}"
            )
        positive_spend_cap(self.max_api_spend_usd)
        benchmark_index_label(self.benchmark_index)
        # The research years must leave at least one validation year, and the
        # validation years must tile into whole test steps.
        _research, validation = self.geometry.split_research(self.acceptance.research_years)
        step_windows(validation[0].start, validation[-1].end, self.acceptance.test_step_months)
        object.__setattr__(self, "experiments_root", Path(self.experiments_root))

    @property
    def experiment_dir(self) -> Path:
        return self.experiments_root / self.experiment_id

    @property
    def ledger_path(self) -> Path:
        return self.experiment_dir / "ledgers" / "experiment_ledger.jsonl"


def positive_spend_cap(value: object) -> float:
    """A spend cap is a positive finite number of USD; there is no uncapped arm."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("max_api_spend_usd must be a positive finite number of USD")
    return float(value)


_ROLLING_FIELDS = {field_obj.name: field_obj for field_obj in fields(RollingExperimentConfig)}


def rolling_default(name: str) -> object:
    """Default of one ``RollingExperimentConfig`` field.

    The dataclass is the single source of truth for the pipeline defaults. A
    caller that fills in an absent value (the ``params.json`` loader, a CLI)
    reads it from here instead of restating a literal that can drift.
    """
    field_obj = _ROLLING_FIELDS.get(name)
    if field_obj is None or field_obj.default is MISSING:
        raise KeyError(f"{name} is not a defaulted RollingExperimentConfig field")
    return field_obj.default


def session_deadline_seconds(
    max_research_minutes: float,
    deadline_grace_minutes: float = DEFAULT_DEADLINE_GRACE_MINUTES,
) -> float:
    """Total session budget: main deadline plus trailing wrap-up grace."""
    return float(max_research_minutes) * 60.0 + float(deadline_grace_minutes) * 60.0


@dataclass(frozen=True)
class SnapshotBundle:
    snapshot_id: str
    decision_ref: str
    # Empty for a decision-only bundle (``SnapshotProvider.prepare_decision``).
    replay_ref: str
    generation_id: str = ""


class SnapshotProvider(Protocol):
    def prepare(
        self,
        *,
        phase: str,
        start: str,
        end: str,
        decision_time: datetime,
    ) -> SnapshotBundle: ...

    def prepare_decision(self, *, decision_time: datetime) -> SnapshotBundle: ...


class ArtifactStore(Protocol):
    """The artifact store the Pipeline freezes and reloads through
    (``environment.artifacts.FilesystemArtifactStore``)."""

    def revision(self, revision_id: str) -> ArtifactRevision: ...

    def freeze_revision(
        self,
        revision_id: str,
        *,
        artifact_id: str,
        experiment_id: str,
        epoch_id: str,
        fold_id: str,
        run_id: str,
        step_id: str,
    ) -> FrozenArtifact: ...

    def frozen(
        self,
        artifact_id: str,
        *,
        expected_path: str | Path | None = None,
        experiment_id: str | None = None,
    ) -> FrozenArtifact: ...


@dataclass(frozen=True)
class EvaluationRequest:
    revision: ArtifactRevision
    snapshot: SnapshotBundle
    mode: str
    start: str
    end: str
    schedule: StrategySchedule
    broker_profile: BrokerProfile
    # Replay slots that continue the book after ``snapshot.replay_ref``, in
    # order: the span runs as one replay from ``start`` (the first slot's
    # start) to ``end`` (the last slot's end).
    continuation: tuple[str, ...] = ()
    # The research years this replay covers, as ``(start, end)`` calendar
    # bounds in order, which is how the sub-window statistics divide the
    # window. Empty when the request is not over research years -- the forward
    # and Held-out span, a plain daily file -- so a replay can never report
    # years it does not have.
    year_windows: tuple[tuple[str, str], ...] = ()
    # The cards this replay's strategy containers attach: a leg of a
    # concurrent batch gets one of the arm's (``session_tools.leg_cards``);
    # None attaches every card the evaluator's sandbox pins.
    gpu_devices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ReplaySpan:
    """Consecutive replay slots replayed as one book.

    ``snapshot`` carries the first slot and the decision view at its anchor;
    ``continuation`` names the slots after it. ``label`` names the span on
    every record (``full`` for the whole research period).
    """

    label: str
    mode: str
    start: str
    end: str
    snapshot: SnapshotBundle
    continuation: tuple[str, ...] = ()
    # The research years the span is made of, as ``(start, end)`` bounds in
    # order; empty for a span that is not over research years (the forward and
    # Held-out span). It rides on the evaluation request so the sub-window
    # statistics use the years the arm really researches.
    year_windows: tuple[tuple[str, str], ...] = ()

    @property
    def slots(self) -> int:
        """Replay slots the span covers; for a research span, its years."""

        return 1 + len(self.continuation)

    def request(
        self,
        revision: ArtifactRevision,
        *,
        schedule: StrategySchedule,
        broker_profile: BrokerProfile,
    ) -> EvaluationRequest:
        return EvaluationRequest(
            revision=revision,
            snapshot=self.snapshot,
            mode=self.mode,
            start=self.start,
            end=self.end,
            schedule=schedule,
            broker_profile=broker_profile,
            continuation=self.continuation,
            year_windows=self.year_windows,
        )


_YEAR_SPAN = re.compile(r"Y(\d+)(?:\.\.Y(\d+))?")


def research_span(
    years: Sequence[ReplaySpan], label: str, *, research_years: int | None = None
) -> ReplaySpan:
    """The span ``label`` names over ``years``, replayed as one book.

    ``full`` is the first ``research_years`` years (every year when none is
    given), ``Yk`` one year and ``Yi..Yj`` the contiguous years i through j.
    The span opens on its first year's slot and decision view and continues
    through the rest; a span covering exactly the ``full`` years is labelled
    ``full`` however it was named. Any other label, or a year ``years`` does
    not have, is refused. Whether a year may be replayed (a Validation stays
    inside the research years) is the caller's rule, not the parser's.
    """

    count = len(years)
    full = count if research_years is None else int(research_years)
    if not 1 <= full <= count:
        raise ValueError(f"research_years must be 1..{count}, got {research_years!r}")
    text = str(label).strip()
    match = _YEAR_SPAN.fullmatch(text)
    if text == FULL_SPAN:
        first, last = 1, full
    elif match:
        first = int(match.group(1))
        last = int(match.group(2) or first)
    else:
        first = last = 0
    if not 1 <= first <= last <= count:
        raise ValueError(
            f"span must be {FULL_SPAN}, one year Y1..Y{count} or contiguous "
            f"years such as Y1..Y{count}; got {label!r}"
        )
    chosen = years[first - 1 : last]
    if (first, last) == (1, full):
        canonical = FULL_SPAN
    elif first == last:
        canonical = f"Y{first}"
    else:
        canonical = f"Y{first}..Y{last}"
    return ReplaySpan(
        label=canonical,
        mode=chosen[0].mode,
        start=chosen[0].start,
        end=chosen[-1].end,
        snapshot=chosen[0].snapshot,
        continuation=tuple(year.snapshot.replay_ref for year in chosen[1:]),
        # Each chosen span is one research year of the geometry, so its own
        # bounds are that year's: the one place the years are read from.
        year_windows=tuple((year.start, year.end) for year in chosen),
    )


@dataclass(frozen=True)
class EvaluationResult:
    """One completed evaluation.

    A backend either returns this or raises: a partial or aborted replay never
    produces an EvaluationResult, so there is no "incomplete" variant to carry.
    """

    summary: dict[str, object]
    result_ref: str


class EvaluationBackend(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


@dataclass(frozen=True)
class StepResult:
    step_id: str
    revision_id: str
    validation: EvaluationResult
    # The ``ReplaySpan.label`` the validation replayed.
    span: str
    # Registered by the Agent as a comparison leg: never nominated.
    control: bool = False
    # The ``validate`` call that recorded it.
    batch_id: str | None = None
    # The artifact fingerprint of the bytes the revision holds. ``None`` where
    # the caller did not commit the revision through a session (the host's
    # deterministic baseline); the ledger row then reads it from the
    # revision's manifest (``experiment.completed_rows``).
    fingerprint: str | None = None


@dataclass(frozen=True)
class BudgetUsed:
    """What the research session has consumed, cumulative over its attempts.

    Recorded on the trace with every model call and tool result, so a new
    attempt continues the counters from the last block the interrupted one
    wrote rather than from zero.
    """

    inference_seconds: float = 0.0
    llm_calls: int = 0
    main_calls: int = 0
    subagent_calls: int = 0
    compact_calls: int = 0
    # Test replays the session's tests ran (``session_tools.TestTool``): a
    # counter of its own, outside every budget.
    test_replays: int = 0
    null_controls: int = 0
    # USD charged against ``max_api_spend_usd``; a block written before the cap
    # existed reads as 0.
    api_spend_usd: float = 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "inference_seconds": round(float(self.inference_seconds), 1),
            "llm_calls": int(self.llm_calls),
            "main_calls": int(self.main_calls),
            "subagent_calls": int(self.subagent_calls),
            "compact_calls": int(self.compact_calls),
            "test_replays": int(self.test_replays),
            "null_controls": int(self.null_controls),
            "api_spend_usd": round(float(self.api_spend_usd), 4),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BudgetUsed:
        def number(key: str, kind: type) -> object:
            value = record.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"budget_used.{key} must be a non-negative number")
            return kind(value)

        return cls(
            inference_seconds=number("inference_seconds", float),  # type: ignore[arg-type]
            llm_calls=number("llm_calls", int),  # type: ignore[arg-type]
            main_calls=number("main_calls", int),  # type: ignore[arg-type]
            subagent_calls=number("subagent_calls", int),  # type: ignore[arg-type]
            compact_calls=number("compact_calls", int),  # type: ignore[arg-type]
            test_replays=number("test_replays", int),  # type: ignore[arg-type]
            null_controls=number("null_controls", int),  # type: ignore[arg-type]
            api_spend_usd=number("api_spend_usd", float),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SessionResume:
    """The interrupted attempts a new attempt of the research session continues."""

    # 1-based number of the attempt about to start.
    attempt: int
    # When and why the last attempt stopped (its last trace event, its error).
    interrupted_at: str
    error: str
    # The last successful compaction's summary, the checkpoint the new attempt
    # starts from; None when the interrupted attempts never compacted.
    compaction_summary: str | None
    budget_used: BudgetUsed
    # Transcript file names of the earlier attempts under the ``trace`` root.
    transcripts: tuple[str, ...]


# How the research session ended. ``completed`` is a session whose last test
# reached the end of the validation years (its candidate is the frozen
# strategy), ``deadline`` one whose budget (the wrap-up grace, the model-call
# budget or the API spend cap) ran out first (the host completes the record),
# and ``environment_blocked`` one the host ended because its replays kept
# failing on the environment (``session_tools.EnvironmentBlocked``), which
# ends the attempt, not the research, until the attempts are spent.
SESSION_OUTCOMES = ("completed", "deadline", ENVIRONMENT_BLOCKED)


@dataclass(frozen=True)
class ResearchSessionRequest:
    experiment_id: str
    run_id: str
    # The Agent's data view for the whole session: the decision view at the
    # end of the research years, anchored at ``decision_time``; no test moves it.
    snapshot: SnapshotBundle
    decision_time: datetime
    # Every year of the research geometry in order, one single-slot span each
    # (labels Y1..Yn): the first ``acceptance_rules.research_years`` are the
    # research years and ``full`` (``research_span``) is their span; the rest
    # are the validation years, which only the tests replay (a Validation
    # stays inside the research years).
    years: tuple[ReplaySpan, ...]
    # Months of history every decision view carries up to its anchor
    # (``RollingExperimentConfig.window_months``): the view the session reads
    # and the view each research year's replay starts from.
    window_months: int
    max_llm_calls: int
    deadline_seconds: float
    # The arm's benchmark index (``RollingExperimentConfig.benchmark_index``),
    # published to the session through the run manifest so the Agent grades
    # itself against the index the host grades it on. Required: a request that
    # omitted it would tell the Agent the default index whatever the arm's is.
    benchmark_index: str
    # Trailing wrap-up grace reserved from deadline_seconds.
    deadline_grace_seconds: float = DEFAULT_DEADLINE_GRACE_MINUTES * 60.0
    directive: str = ""
    # Per-session HITL override of the experiment's default sandbox GPU count;
    # None keeps the experiment default. The session takes that many of the
    # cards its arm holds.
    sandbox_gpu_count: int | None = None
    # The arm's ``AcceptanceRules.to_record()``; no default, as the rules
    # carry required keys.
    acceptance_rules: Mapping[str, object] = field(kw_only=True)
    # The validation years as one span (``experiment.validation_span``): what
    # a test replays from its first day to the new frontier, and the host
    # completes a record over.
    validation: ReplaySpan = field(kw_only=True)
    # The arm's ``test_step`` ledger records so far (``rolling_record``): the
    # record the attempt continues and the frontier it reached.
    tests_before: tuple[Mapping[str, object], ...] = ()
    modification_constraints: ModificationConstraints = field(
        default_factory=ModificationConstraints
    )
    snapshot_config: Mapping[str, object] = field(default_factory=dict)
    record_failed_attempts: bool = True
    # Cap on the session's own ``run_null_control`` calls; the experiment default
    # is the single source (RollingExperimentConfig.max_null_controls).
    max_null_controls: int = RollingExperimentConfig.max_null_controls
    # Replays one ``validate`` call runs at once
    # (RollingExperimentConfig.batch_validate_max_concurrency).
    batch_validate_max_concurrency: int = (
        RollingExperimentConfig.batch_validate_max_concurrency
    )
    # The arm's API spend cap (RollingExperimentConfig.max_api_spend_usd), after
    # any HITL resource override.
    max_api_spend_usd: float = RollingExperimentConfig.max_api_spend_usd
    progress_hook: Callable[[str, dict[str, object] | None], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    session_key: str = SessionKind.RESEARCH
    # Trusted host source for the current experiment-level skills snapshot.
    # The sandbox adapter copies it to workspace/skills but never exposes this
    # host path through Agent-visible facts or manifests.
    skills_source_ref: str = ""
    # What the session's earlier attempts consumed and recorded: the live
    # counters continue from these amounts, the registration and the finish
    # tool see these Validations, and ``resume`` says how the last attempt
    # stopped. All empty on a first attempt.
    budget_used: BudgetUsed = field(default_factory=BudgetUsed)
    steps_before: tuple[StepResult, ...] = ()
    resume: SessionResume | None = None

    @classmethod
    def for_arm(
        cls,
        config: RollingExperimentConfig,
        *,
        budgets: Mapping[str, float],
        **session: Any,
    ) -> ResearchSessionRequest:
        """The request for one session of ``config``'s arm.

        The arm-level fields come from the config, the four budgets
        (``max_llm_calls``, ``max_api_spend_usd``, ``deadline_seconds``,
        ``deadline_grace_seconds``) as given after any HITL override, and
        every other field is the session's own.
        """

        return cls(
            experiment_id=config.experiment_id,
            window_months=config.window_months,
            benchmark_index=config.benchmark_index,
            acceptance_rules=config.acceptance.to_record(),
            modification_constraints=config.step_constraints,
            record_failed_attempts=config.record_failed_attempts,
            max_null_controls=config.max_null_controls,
            batch_validate_max_concurrency=config.batch_validate_max_concurrency,
            max_llm_calls=int(budgets["max_llm_calls"]),
            max_api_spend_usd=float(budgets["max_api_spend_usd"]),
            deadline_seconds=float(budgets["deadline_seconds"]),
            deadline_grace_seconds=float(budgets["deadline_grace_seconds"]),
            **session,
        )

    @property
    def research_years(self) -> int:
        """How many of ``years`` are the research years (the rules' value)."""

        return int(self.acceptance_rules["research_years"])  # type: ignore[call-overload]

    @property
    def full_span(self) -> ReplaySpan:
        """The research years as one span: what ``full`` names."""

        return research_span(self.years, FULL_SPAN, research_years=self.research_years)


@dataclass(frozen=True)
class ResearchSessionResult:
    conversation_id: str
    steps: tuple[StepResult, ...]
    # One of ``SESSION_OUTCOMES``.
    outcome: str
    # Keyword-only from here: independent optional fields, so a new one can
    # never land in an older field's positional slot.
    _: KW_ONLY
    # The frozen candidate: the Step the last test took, the Steps it
    # registered as its seed replicates and whether it declared its package
    # deterministic (``experiment.freeze_registration``). None when no test
    # ran (``deadline`` before the first test, or ``environment_blocked``).
    node_id: str | None = None
    seed_replicates: tuple[str, ...] = ()
    deterministic: bool = False
    # The Agent's own account of its outcome.
    reason: str = ""
    finish_reason: str = ""
    # Host path of this run's manifest.
    run_manifest_ref: str = ""
    # Trusted host path to this run's collected workspace/skills audit copy.
    skills_source_ref: str = ""
    # Null-control blocks the session already computed, keyed by step id; the
    # Pipeline reuses the frozen node's block instead of drawing it again.
    null_controls: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    # The session's cumulative spend at its end, and which attempt ended it.
    budget_used: BudgetUsed = field(default_factory=BudgetUsed)
    attempt: int = 1

    def __post_init__(self) -> None:
        if self.outcome not in SESSION_OUTCOMES:
            raise ValueError(f"unknown research session outcome: {self.outcome!r}")


ResearchDeveloper = Callable[[ResearchSessionRequest], ResearchSessionResult]


__all__ = [
    "BATCH_VALIDATE_MAX_CANDIDATES",
    "BATCH_VALIDATE_MIN_CANDIDATES",
    "DEFAULT_BATCH_VALIDATE_MAX_CONCURRENCY",
    "DEFAULT_DEADLINE_GRACE_MINUTES",
    "DEFAULT_MAX_API_SPEND_USD",
    "DEFAULT_MAX_LLM_CALLS",
    "DEFAULT_MAX_NULL_CONTROLS",
    "DEFAULT_MAX_RESEARCH_MINUTES",
    "DEFAULT_RESEARCH_GEOMETRY",
    "OBSOLETE_KEYS",
    "REQUIRED_ACCEPTANCE_KEYS",
    "RETIRED_RULES",
    "SESSION_OUTCOMES",
    "TEST_STEP_MONTHS",
    "AcceptanceRules",
    "ArtifactRevision",
    "ArtifactStore",
    "BudgetUsed",
    "EvaluationBackend",
    "EvaluationRequest",
    "EvaluationResult",
    "ExecutionMode",
    "FrozenArtifact",
    "ReplaySpan",
    "ResearchDeveloper",
    "ResearchSessionRequest",
    "ResearchSessionResult",
    "RollingExperimentConfig",
    "SessionResume",
    "SnapshotBundle",
    "SnapshotProvider",
    "StepResult",
    "StrategyExperimentConfig",
    "acceptance_default",
    "acceptance_for",
    "defaulted_acceptance",
    "positive_spend_cap",
    "research_span",
    "rolling_default",
    "session_deadline_seconds",
]
