"""One launcher for every checked-in round definition.

A round is data: which arms it starts, which reference pack and exploration
directive each arm gets, which PIT view seed and dataset selection they share,
and the handful of parameters that round decides differently from the console
creation form. Everything around that data -- merging the console defaults,
refusing a default drift, running the console's own create-time validation
offline, reporting a dry-run and POSTing the create requests -- is the same for
every round and lives here, so a new round file is its arms plus a two-line
entry point.

Two decisions are shared rather than per-round because the console runs several
rounds side by side and a walk-forward comparison across them only means
anything while they agree: BASE_OVERRIDES carries the account, the graduation
gates, the quarterly schedule and the per-Fold budgets, and
BASE_EXPECTED_DEFAULTS pins the console creation defaults every round relies
on -- above all the six model roles, which no round overrides, so a rename of
the local model must stop the launcher rather than silently move an arm onto a
hosted stream. A round states what it decides for itself in `overrides`, and
anything it states there stops being an expected default.

`normalize` runs the request-level checks the console applies on POST
/api/experiments (ExperimentManager.create_experiment's closed, unknown,
required, id and stamp rules, then the worker's own resolve_worker_options
pre-flight, which is what actually type-checks every knob) and additionally
refuses a directive the PRIOR calendar policy would reject -- which
resolve_worker_options enforces on fold_exploration_directive too, so a
directive carrying a literal calendar date fails the worker at start rather
than merely reading badly. The console's deployment-state checks -- an
experiment directory that already exists and a free running slot -- can only be
decided against the live server and stay at POST time.

A round that names a PIT view seed explicitly makes that seed required: the
tree must exist, its recorded snapshot configuration must be exactly this
round's, and the prebuild must have left no staged slot behind. An arm refused
for a staged slot is waiting for the prebuild, not misconfigured, and the
refusal says so.

RETIRED_IDS records the experiment ids that have been used and archived but no
longer appear in any round file, so a new round cannot quietly reuse one after
its definition is dropped. `logs/archive/` is not part of the repository, which
is why the list is checked in rather than read from disk; `archived_ids` reads
the archive where it exists so the two can be compared.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.tools.prior_policy import calendar_policy_violation
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
from autotrade.pipelines.hitl_state import (
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    WEB_REQUIRED_PARAMS,
)
from autotrade.pipelines.ledger import latest_fold_records
from autotrade.pipelines.worker import resolve_worker_options

# The console's own id rule and roots; importing them keeps this module from
# growing a second copy of the create contract.
from autotrade.webui.manager import _ID as EXPERIMENT_ID_RE

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
ARCHIVE_ROOT = REPO_ROOT / "logs" / "archive"

# Console defaults every round relies on. Values, not commentary: if the console
# changes any of them the round has to be re-decided, not silently re-run.
BASE_EXPECTED_DEFAULTS: dict[str, object] = {
    "test_stage": False,
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
    "meta_learning_fold_interval": 1,
    "window_months": 24,
    "include_fundamentals": True,
    "include_macro": True,
    "include_events": True,
    "include_text": True,
    "include_intraday": False,
    # A round that does not name a seed reuses the default tree the quarterly
    # views were prebuilt into. A drift here would not fail it, it would
    # cold-build every view of every arm.
    "pit_views_seed": "data/pit_views_seed/explore",
    "screen_exclude_st": False,
    "screen_exclude_new_listed_days": 0,
    "screen_boards": (),
    "screen_min_circ_mv_yi": None,
    "screen_max_circ_mv_yi": None,
    # Decides which Folds the schedule keeps, so a drift would silently move a
    # plan away from the tree its seed was built over.
    "min_region_trade_days": 2,
    # Both the packs and the directives promise the Agent this many host null
    # controls per Fold.
    "max_null_controls_per_fold": 3,
    # Derived from SandboxLimits.fit_timeout_seconds; the directives promise the
    # Agent a fit budget of this size.
    "strategy_fit_timeout_seconds": 3600,
    "operating_memory": "curated+graduated",
    "initial_control_mode": "auto",
    "reasoning_effort": "xhigh",
    "inference_time": "08:30",
    "strategy_period": "day",
    "inherit_from": "",
    "inherit_memory_from": "",
    # All six model roles. Every round runs entirely on the local model and
    # overrides none of them, so the console default is what actually decides
    # them; spelled out as literals on purpose, since a rename of the local
    # model is exactly the drift this has to catch.
    "model": "qwen-3.8-27b-fp8",
    "meta_model": "qwen-3.8-27b-fp8",
    "subagent_model": "qwen-3.8-27b-fp8",
    "analysis_model": "qwen-3.8-27b-fp8",
    "nl_model": "qwen-3.8-27b-fp8",
    "compact_model": "qwen-3.8-27b-fp8",
}

# What every round decides the same way. Values, not commentary: the schedule,
# the account, the graduation gates and the per-step budgets.
BASE_OVERRIDES: dict[str, object] = {
    # No GPU. The request travels with the experiment: it reaches the Agent
    # session sandbox and the strategy container of every formal replay alike;
    # an arm that really trains on a card overrides this with 1.
    "gpu_count": 0,
    # Quarterly walk-forward: each Fold is validated on the trailing four
    # quarters ending at its own quarter, so every step adds exactly one new
    # quarter and the chain never revisits a window.
    "fold_period": "quarter",
    "validation_periods": 4,
    "development_first_period": "2022Q1",
    "development_last_period": "2025Q4",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
    # One pass: with twelve forward transitions there is nothing a second pass
    # over the same windows could add that would still be out of sample.
    "epochs": 1,
    # The researcher's real account, where the CNY 5 minimum commission and lot
    # sizes are a real cost rather than a rounding error.
    "initial_cash": 100_000,
    # Graduation must survive twice the profile slippage, rest on more than a
    # handful of trades, and the artifact that ships must carry its own forward
    # evidence rather than inheriting the parent's -- which the reserved
    # confirmation Folds at the end of the window are what makes possible.
    "cost_stress_multiplier": 2.0,
    "heldout_min_trades": 20,
    "confirmation_folds": 2,
    # Per-Fold budgets: under the yearly-window defaults because at each step
    # three of the four quarters and the parent's own result on the window are
    # already known, but wide enough for several pre-registered rounds.
    "max_fold_minutes": 600,
    "max_steps_per_fold": 24,
    "max_backtests_per_fold": 24,
    "max_llm_calls": 1600,
}

# Two lines shared by every directive: what the quarterly schedule makes of the
# host's parent control, and the one selection criterion the static prompt does
# not already state. Everything else the directives used to repeat -- the budget
# figures, pre-registration, fitting parameters in fit(context) -- is in the
# static system prompt or the injected run facts.
PARENT_CONTROL_LINE = (
    "本 Fold 的验证窗口是截至本季的连续四个季度，其中只有最后一个季度是上一 Fold 之后新出现的："
    "父本对照节点的最后一个子窗口才是父策略在这个新季度上的样本外记录，更早的子窗口它已经见过。"
    "比较对照与候选时按这个口径读 `sub_windows`。"
)
ROBUSTNESS_LINE = (
    "并列候选之间还要看对参数与阈值的敏感度：结论在邻域里翻转的候选不算稳健。"
)

# Reported for every arm on --dry-run: what a round decides, plus the defaults
# it depends on. A round adds the keys its own decisions need.
BASE_REPORT_KEYS: tuple[str, ...] = (
    "experiment_id",
    "workspace_reference",
    "inherit_from",
    "inherit_memory_from",
    "gpu_count",
    "reasoning_effort",
    "initial_control_mode",
    "fold_period",
    "development_first_period",
    "development_last_period",
    "validation_periods",
    "test_stage",
    "heldout_first_period",
    "heldout_last_period",
    "epochs",
    "meta_learning_fold_interval",
    "window_months",
    "include_intraday",
    "pit_views_seed",
    "operating_memory",
    "max_fold_minutes",
    "max_steps_per_fold",
    "max_backtests_per_fold",
    "max_llm_calls",
    "max_null_controls_per_fold",
    "initial_cash",
    "cost_stress_multiplier",
    "heldout_min_trades",
    "confirmation_folds",
    "deployment_adjustment_start",
    "deployment_max_backtests",
    "deployment_pit_views_seed",
    "strategy_fit_timeout_seconds",
    "inference_time",
    "strategy_period",
    "model",
    "meta_model",
    "subagent_model",
    "analysis_model",
    "compact_model",
    "nl_model",
)

# Experiment ids that were used and archived and are not in any round file any
# more. An id is never reused: the console keys the experiment directory, the
# sandbox work root, the Docker image tag and the archive path on it, so a
# second run under an old name would be indistinguishable from the first in
# every record that survives it. Kept here because `logs/archive/` is
# gitignored, so a fresh checkout can still enforce this; `archived_ids` reads
# the archive where it exists so the two can be compared.
# The two ways an arm can start from another experiment: a parent artifact
# copied in read-only, and that experiment's PRIOR and skills imported as this
# one's own first generation.
INHERITANCE_KEYS: tuple[str, ...] = ("inherit_from", "inherit_memory_from")

RETIRED_IDS: frozenset[str] = frozenset(
    {
        "cb_linkage_20260914",
        "corner_cases_20260907",
        "corner_cases_20260910",
        "explore_platform_strategies_20260910",
        "factor_cs_20260910",
        "factor_cs_allflash_20260910",
        "site_visits_20260914",
        "value_regime_20260914",
    }
)


def quarter_key(period: str) -> tuple[int, int]:
    """``"2024Q3"`` -> ``(2024, 3)``; raises on anything else."""
    text = str(period).strip().upper()
    if len(text) != 6 or text[4] != "Q" or not text[:4].isdigit() or text[5] not in "1234":
        raise ValueError(f"not a quarter: {period!r}")
    return int(text[:4]), int(text[5])


def quarter_shift(period: str, quarters: int) -> str:
    """The quarter ``quarters`` steps after ``period`` (negative steps go back)."""
    year, quarter = quarter_key(period)
    index = year * 4 + (quarter - 1) + quarters
    return f"{index // 4}Q{index % 4 + 1}"


def quarters_between(first: str, last: str) -> int:
    """How many quarters ``first..last`` spans, both ends included; 0 when empty."""
    start, end = quarter_key(first), quarter_key(last)
    span = (end[0] * 4 + end[1]) - (start[0] * 4 + start[1]) + 1
    return max(span, 0)


def last_completed_fold(experiment_id: str, experiments_root: Path | None = None) -> str:
    """The quarter of the last Fold ``experiment_id`` finished, e.g. ``"2024Q3"``.

    Read from that experiment's own ledger when the round is created, so a
    round planned to continue where another one froze is planned against what
    the source has actually done rather than against a quarter typed into a
    round file and gone stale by the time it is launched.
    """

    root = experiments_root if experiments_root is not None else EXPERIMENTS_ROOT
    ledger = Path(root) / experiment_id / "ledgers" / "experiment_ledger.jsonl"
    if not ledger.is_file():
        raise ValueError(
            f"{experiment_id} has no ledger at {ledger}, so there is no completed "
            "Fold to continue from"
        )
    records = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quarters = sorted(
        (fold_id.removeprefix("fold_") for _epoch, fold_id in latest_fold_records(records)),
        key=quarter_key,
    )
    if not quarters:
        raise ValueError(
            f"{experiment_id} has completed no Fold yet, so there is nothing to continue from"
        )
    return quarters[-1]


def archived_ids() -> set[str]:
    """Experiment ids with an archived tree under ``logs/archive/<batch>/<id>``.

    Empty where the archive was never created -- it is a local operator
    artifact, not part of the repository -- so a caller must treat an empty
    result as "no evidence", never as "nothing was archived".
    """

    if not ARCHIVE_ROOT.is_dir():
        return set()
    return {
        arm.name
        for batch in ARCHIVE_ROOT.iterdir()
        if batch.is_dir() and not batch.is_symlink()
        for arm in batch.iterdir()
        if arm.is_dir() and not arm.is_symlink()
    }


def already_created(experiment_id: str) -> bool:
    """Whether this experiment has been created at least once on this console.

    Inheritance -- a parent artifact, another experiment's PRIOR and skills --
    is copied once, while the console creates the experiment. After that the
    copy is this experiment's own read-only tree and the source is free to be
    retired and archived. So the experiment's own directory, live or archived,
    is the evidence that whatever it inherited has already been taken, and its
    sources no longer have to be anywhere.
    """

    return (EXPERIMENTS_ROOT / experiment_id).is_dir() or experiment_id in archived_ids()


def normalize(params: dict[str, object]) -> dict[str, object]:
    """Run the console's create-time validation offline and return params.json.

    Same order and same request-level checks as
    ExperimentManager.create_experiment, then the worker's own
    resolve_worker_options pre-flight. The console's duplicate-directory and
    running-slot checks need the live deployment and stay at POST time; the
    calendar-policy gate below is stricter than create.
    """
    closed = sorted(set(params) & WEB_CLOSED_PARAMS)
    if closed:
        raise ValueError("console-managed parameters are not accepted: " + ", ".join(closed))
    unknown = sorted(set(params) - set(WEB_CREATE_DEFAULTS))
    if unknown:
        raise ValueError("unknown experiment parameters: " + ", ".join(unknown))
    merged = {**WEB_CREATE_DEFAULTS, **params}
    missing = sorted(key for key in WEB_REQUIRED_PARAMS if merged.get(key) in (None, ""))
    if missing:
        raise ValueError("missing required experiment parameters: " + ", ".join(missing))
    experiment_id = str(params.get("experiment_id") or "").strip()
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError("experiment_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,99}")
    directive = str(merged.get("fold_exploration_directive") or "")
    # Not enforced on create, but the same text is injected into PRIOR-facing
    # prompts, is re-sendable through set_directive, and is refused by
    # resolve_worker_options itself.
    violation = calendar_policy_violation(directive)
    if violation:
        raise ValueError(f"fold_exploration_directive {violation}")
    merged.update(
        {
            **WEB_INTERNAL_PARAMS,
            "experiment_id": experiment_id,
            "experiments_root": str(EXPERIMENTS_ROOT),
            "work_root": str(REPO_ROOT / ".runtime/sandboxes"),
            "_creation_surface": "webui",
        }
    )
    resolve_worker_options(
        merged,
        experiment_dir=EXPERIMENTS_ROOT / experiment_id,
        repo_root=REPO_ROOT,
        preflight=True,
    )
    return merged


def post(port: int, params: dict[str, object]) -> bool:
    """POST one create request; report whether the console accepted it."""
    experiment_id = params["experiment_id"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/experiments",
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            print(experiment_id, response.status, response.read(400).decode("utf-8", "replace"))
            return True
    except urllib.error.HTTPError as exc:
        print(experiment_id, "HTTP", exc.code, exc.read(800).decode("utf-8", "replace"), file=sys.stderr)
    except urllib.error.URLError as exc:
        # No console on that port, or it dropped the connection: an operator
        # error, not a traceback.
        print(experiment_id, "console unreachable:", exc.reason, file=sys.stderr)
    return False


@dataclass(frozen=True)
class Round:
    """One round definition: its arms and what it decides differently.

    ``arms`` maps experiment id to the per-arm part of the create request --
    normally ``workspace_reference`` and ``fold_exploration_directive``, plus
    any parameter that arm alone changes. ``overrides`` is what the whole round
    decides on top of BASE_OVERRIDES, ``expected_defaults`` pins console
    defaults this round relies on beyond BASE_EXPECTED_DEFAULTS, and
    ``report_keys`` adds what its dry-run should show.
    """

    arms: Mapping[str, Mapping[str, object]]
    overrides: Mapping[str, object] = field(default_factory=dict)
    expected_defaults: Mapping[str, object] = field(default_factory=dict)
    report_keys: tuple[str, ...] = ()
    # The prebuilt view tree every arm hardlinks from, when the round needs one
    # of its own. Empty means the console default seed, which carries the
    # default dataset selection.
    pit_views_seed: str = ""

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("a round with no arms cannot be created")
        reused = sorted(set(self.arms) & RETIRED_IDS)
        if reused:
            raise ValueError(
                "experiment ids that were already used and archived cannot be "
                "reused: " + ", ".join(reused)
            )
        # An arm may inherit a parent artifact or another experiment's memory,
        # and both are read out of the source's live directory at create time.
        # A retired id names a tree that only exists in the archive now, so it
        # cannot be a source for an arm still to be created; an arm that was
        # already created took its copy while the source was live, and the
        # source is allowed to have been retired since.
        retired_sources = sorted(
            {
                source
                for experiment_id, arm in self.arms.items()
                if not already_created(experiment_id)
                for key in INHERITANCE_KEYS
                for source in (str(arm.get(key) or "").strip(),)
                if source in RETIRED_IDS
            }
        )
        if retired_sources:
            raise ValueError(
                "cannot inherit from an experiment that was retired and archived: "
                + ", ".join(retired_sources)
            )

    @property
    def common_overrides(self) -> dict[str, object]:
        """What every arm of this round sends, seed included."""
        overrides = {**BASE_OVERRIDES, **self.overrides}
        if self.pit_views_seed:
            overrides["pit_views_seed"] = self.pit_views_seed
        return overrides

    @property
    def expected(self) -> dict[str, object]:
        """Console defaults this round still relies on.

        Anything the round decides for itself stops being a default it depends
        on, so it is dropped here rather than pinned twice.
        """
        common = self.common_overrides
        return {
            key: value
            for key, value in BASE_EXPECTED_DEFAULTS.items()
            if key not in common
        } | dict(self.expected_defaults)

    @property
    def keys_reported(self) -> tuple[str, ...]:
        return (*BASE_REPORT_KEYS, *self.report_keys)

    def check_console_defaults(self) -> None:
        drift = {
            key: (value, WEB_CREATE_DEFAULTS[key])
            for key, value in self.expected.items()
            if WEB_CREATE_DEFAULTS[key] != value
        }
        if drift:
            raise SystemExit(
                "console creation defaults drifted from what this round assumes; "
                "re-decide the round before creating: "
                + json.dumps(
                    {k: {"round": str(v[0]), "console": str(v[1])} for k, v in drift.items()},
                    ensure_ascii=False,
                )
            )

    def request_params(self, experiment_id: str) -> dict[str, object]:
        """The create request body: console defaults, the round's decisions, the id."""
        base = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in WEB_CREATE_DEFAULTS.items()
        }
        # An arm value may be a zero-argument callable when the round can only
        # decide it against live state -- reading where another experiment got
        # to, say. It is resolved here, at create time, so the dry-run and the
        # POST see the same answer and a refusal surfaces as a rejection.
        arm = {
            key: (value() if callable(value) else value)
            for key, value in self.arms[experiment_id].items()
        }
        return {
            **base,
            **self.common_overrides,
            **arm,
            "experiment_id": experiment_id,
        }

    def validated(self, experiment_id: str) -> tuple[dict[str, object] | None, str]:
        """params.json for one arm, or ``(None, reason)``.

        The refusal is returned rather than raised because a dry-run should
        read every arm before it stops -- the seed is shared, so the first
        arm's refusal says nothing about whether the others are well-formed.
        """
        try:
            merged = normalize(self.request_params(experiment_id))
            missing = self.missing_sources(merged)
            if missing:
                raise ValueError(
                    "inheritance source is not an experiment on this console: "
                    + ", ".join(missing)
                )
            return merged, ""
        except ValueError as exc:
            reason = f"{experiment_id}: parameters rejected, nothing was sent: {exc}"
            if "unfinished build" in str(exc):
                reason += (
                    "\n  the tree is there and its contract is this round's; the"
                    " prebuild is still staging views into it. Wait for"
                    " scripts/data/prebuild_pit_views_seed.py to report status ok,"
                    " then re-run -- no parameter needs changing"
                )
            elif "pit_views_seed" in str(exc) or "view seed" in str(exc):
                reason += (
                    f"\n  every arm shares one prebuilt view tree ({self.pit_views_seed});"
                    " build it with scripts/data/prebuild_pit_views_seed.py for exactly"
                    " this dataset selection and calendar before creating or dry-running"
                    " the round"
                )
            return None, reason

    def missing_sources(self, merged: Mapping[str, object]) -> list[str]:
        """Inheritance sources this arm names that are not on the console.

        The console reads the source's directory while it creates the
        experiment and discards the half-created tree when that fails, so a
        source that is not there is a create error worth catching offline --
        but only for an arm that has still to be created. Once the arm exists
        the copy has been taken and is its own, which is exactly how a round
        retires the predecessors it inherited from: create the successors
        first, archive the sources after. Deployment state, not a request-level
        rule: where `experiments/` does not exist at all -- a fresh checkout --
        there is nothing to judge and nothing is reported.
        """

        if not EXPERIMENTS_ROOT.is_dir():
            return []
        if already_created(str(merged.get("experiment_id") or "").strip()):
            return []
        return [
            source
            for key in INHERITANCE_KEYS
            for source in (str(merged.get(key) or "").strip(),)
            if source and not (EXPERIMENTS_ROOT / source).is_dir()
        ]

    def seed_status(self) -> str:
        """One line on the shared seed, printed before the arms.

        What the per-arm pre-flight decides has three parts, and this line says
        which of them can already be read: the tree exists, `provider.json`
        records this round's snapshot configuration under the current cache
        format, and the build left no staged slot behind. The first two are
        read here; the third is the pre-flight's own, so an arm's refusal names
        it rather than this line.
        """
        if not self.pit_views_seed:
            return (
                "seed: this round names none, so its arms use the console default tree"
                " and cold-build anything it does not carry"
            )
        provider = REPO_ROOT / self.pit_views_seed / "provider.json"
        if not provider.is_file():
            return (
                f"seed {self.pit_views_seed}: provider.json absent -- the prebuild has not"
                " written its contract yet, so the snapshot-configuration pre-flight"
                " cannot run and every arm below is refused"
            )
        try:
            record = json.loads(provider.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return (
                f"seed {self.pit_views_seed}: provider.json unreadable ({exc});"
                " every arm below is refused"
            )
        version = record.get("schema_version")
        if version != SNAPSHOT_CACHE_FORMAT_VERSION:
            return (
                f"seed {self.pit_views_seed}: built under snapshot cache format {version!r}"
                f" but this code writes {SNAPSHOT_CACHE_FORMAT_VERSION} -- the pre-flight"
                " refuses it; rebuild the seed under a new directory"
            )
        return (
            f"seed {self.pit_views_seed}: contract present (snapshot cache format"
            f" {version}), so the per-arm pre-flight below compares this round's"
            " selection against it and, on top of that, refuses the tree while a"
            " prebuild is still staging views into it -- an arm refused for an"
            " unfinished build is a wait, not a wrong parameter."
        )

    def main(self, argv: list[str], usage: str | None = None) -> int:
        """`<port> [--dry-run] [experiment_id ...]`, shared by every round file."""
        # A mistyped flag must never fall through to the real POST path: without
        # this, --dryrun is read as an experiment-id filter and creates the round.
        mistyped = [arg for arg in argv[1:] if arg.startswith("--") and arg != "--dry-run"]
        if mistyped:
            print("unknown option: " + ", ".join(mistyped), file=sys.stderr)
            return 2
        if len(argv) < 2 or not argv[1].isdigit():
            raise SystemExit(usage or self.main.__doc__)
        port = int(argv[1])
        dry_run = "--dry-run" in argv
        wanted = {arg for arg in argv[2:] if not arg.startswith("--")}
        unknown_ids = sorted(wanted - set(self.arms))
        if unknown_ids:
            raise SystemExit("not in this round: " + ", ".join(unknown_ids))
        self.check_console_defaults()
        print(self.seed_status())
        failed: list[str] = []
        for experiment_id in self.arms:
            if wanted and experiment_id not in wanted:
                continue
            merged, reason = self.validated(experiment_id)
            if merged is None:
                # On the POST path nothing may be sent for a rejected arm; on a
                # dry-run the refusal is a reading, so the remaining arms are
                # read too and the exit code still reports it.
                if not dry_run:
                    raise SystemExit(reason)
                print(reason, file=sys.stderr)
                failed.append(experiment_id)
                continue
            if dry_run:
                directive = str(merged["fold_exploration_directive"])
                print(json.dumps({key: merged[key] for key in self.keys_reported}, ensure_ascii=False))
                print(f"  directive: {len(directive.splitlines())} lines, {len(directive)} chars")
                for line in directive.splitlines():
                    print("   |", line)
                continue
            if not post(port, self.request_params(experiment_id)):
                failed.append(experiment_id)
        if failed:
            print(("refused: " if dry_run else "not created: ") + ", ".join(failed), file=sys.stderr)
            return 1
        return 0
