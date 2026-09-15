"""One launcher for every checked-in round definition.

A round is data: which arms it starts, which reference pack and exploration
directive each arm gets, which PIT view seed and dataset selection they share,
and the handful of parameters that round decides differently from the console
creation form. Everything around that data -- merging the console defaults,
refusing a default drift, running the console's own create-time validation
offline, reporting a dry-run and POSTing the create requests -- is the same for
every round and lives here, so a round file is its arms plus a two-line entry
point.

Two decisions are shared rather than per-round because the console runs rounds
side by side and their verdicts only compare while they agree: BASE_OVERRIDES
carries the research geometry, the account, the verdict parameters and the
per-session budgets, and BASE_EXPECTED_DEFAULTS pins the console creation
defaults every round relies on -- above all the model roles, which no round
overrides, so a rename of the local model must stop the launcher rather than
silently move an arm onto a hosted stream. A round states what it decides for
itself in `overrides`, and anything it states there stops being an expected
default.

`normalize` runs the request-level checks the console applies on POST
/api/experiments (ExperimentManager.create_experiment's closed, unknown,
required and id rules, then the worker's own resolve_worker_options pre-flight,
which type-checks every knob, refuses a malformed research geometry and, for a
named PIT view seed, refuses a tree whose cache format or snapshot
configuration is not this round's or whose prebuild is still staging views,
and reads the research release the seed was built from -- the release every
arm will pin -- refusing it unless it is published here and reaches Held-out).
The console's deployment-state checks -- an experiment directory that already
exists and a free running slot -- can only be decided against the live server
and stay at POST time.

`--dry-run` judges the round before its arms: a probe request carrying only
what every arm shares goes through the same pre-flight, so the geometry and the
seed contract are read even for a round that has no arms yet. Such a round can
be dry-run but not created.

RETIRED_IDS records the experiment ids that have been used and archived, so a
new round cannot quietly reuse one. `logs/archive/` is not part of the
repository, which is why the list is checked in rather than read from disk;
`archived_ids` reads the archive where it exists so the two can be compared.
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

from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
from autotrade.pipelines.hitl_state import (
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    WEB_REQUIRED_PARAMS,
)
from autotrade.pipelines.worker import resolve_worker_options

# The console's own id rule; importing it keeps this module from growing a
# second copy of the create contract.
from autotrade.webui.manager import _ID as EXPERIMENT_ID_RE

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
ARCHIVE_ROOT = REPO_ROOT / "logs" / "archive"
# The id the round-level dry-run validates under. Never created: it only
# carries the parameters every arm shares through the pre-flight.
PROBE_ID = "round_dry_run_probe"

# Console defaults every round relies on. Values, not commentary: if the console
# changes any of them the round has to be re-decided, not silently re-run.
BASE_EXPECTED_DEFAULTS: dict[str, object] = {
    "include_fundamentals": True,
    "include_macro": True,
    "include_events": True,
    "include_text": True,
    "include_intraday": False,
    "screen_exclude_st": False,
    "screen_exclude_new_listed_days": 0,
    "screen_boards": (),
    "screen_min_circ_mv_yi": None,
    "screen_max_circ_mv_yi": None,
    # Both the packs and the directives promise the Agent this many host null
    # controls per research session.
    "max_null_controls_per_session": 3,
    # Replay-years one research session may spend on validations; the packs
    # size their rounds against it.
    "max_replay_years_per_session": 24,
    # Derived from SandboxLimits.fit_timeout_seconds; packs promise the Agent a
    # fit budget of this size.
    "strategy_fit_timeout_seconds": 3600,
    "operating_memory": "curated+graduated",
    "initial_control_mode": "auto",
    "reasoning_effort": "xhigh",
    "inference_time": "08:30",
    "strategy_period": "day",
    # Every model role. Every round runs entirely on the local model and
    # overrides none of them, so the console default is what actually decides
    # them; spelled out as literals on purpose, since a rename of the local
    # model is exactly the drift this has to catch.
    "model": "qwen-3.8-27b-fp8",
    "subagent_model": "qwen-3.8-27b-fp8",
    "nl_model": "qwen-3.8-27b-fp8",
    "compact_model": "qwen-3.8-27b-fp8",
}

# What every round decides the same way. Values, not commentary.
BASE_OVERRIDES: dict[str, object] = {
    # Research on four July-June years, the twelve forward months after them,
    # and a Held-out quarter the replay clips to the release end. Required by
    # the console, and the geometry the PIT view seeds are planned over.
    "research_start": "20210701",
    "research_end": "20250630",
    "forward_end": "20260630",
    "heldout_end": "20260930",
    "research_sessions": 4,
    # Sixty months behind the research-end decision view reach 2020-07, just
    # inside the 2020 floor of fundamentals and macro; part of every seed's
    # snapshot configuration.
    "window_months": 60,
    # No GPU. The request travels with the experiment to the Agent session
    # sandbox and the strategy container of every replay alike; an arm that
    # really trains on a card overrides this with 1.
    "gpu_count": 0,
    # The researcher's real account, where the CNY 5 minimum commission and lot
    # sizes are a real cost rather than a rounding error.
    "initial_cash": 100_000,
    # The forward verdict's drawdown limit and cost stress.
    "max_drawdown": 0.25,
    "cost_stress_multiplier": 2.0,
    # Per research session.
    "max_session_minutes": 600,
    "max_llm_calls": 1600,
}

# Reported once per round on --dry-run: what every arm shares.
ROUND_REPORT_KEYS: tuple[str, ...] = (
    "research_start",
    "research_end",
    "forward_end",
    "heldout_end",
    "research_sessions",
    "pit_views_seed",
    "window_months",
    "include_fundamentals",
    "fundamental_datasets",
    "include_macro",
    "macro_datasets",
    "include_events",
    "events_datasets",
    "include_text",
    "text_datasets",
    "include_intraday",
    "operating_memory",
    "max_session_minutes",
    "max_replay_years_per_session",
    "max_llm_calls",
    "max_null_controls_per_session",
    "strategy_fit_timeout_seconds",
    "initial_cash",
    "max_drawdown",
    "cost_stress_multiplier",
    "gpu_count",
    "reasoning_effort",
    "initial_control_mode",
    "inference_time",
    "strategy_period",
    "model",
    "subagent_model",
    "nl_model",
    "compact_model",
)

# Experiment ids that were used and archived. An id is never reused: the
# console keys the experiment directory, the sandbox work root, the Docker
# image tag and the archive path on it, so a second run under an old name would
# be indistinguishable from the first in every record that survives it.
RETIRED_IDS: frozenset[str] = frozenset(
    {
        "alt_events_ranker_20260916",
        "alt_events_ranker_20260917",
        "analyst_revision_20260916",
        "analyst_revision_20260917",
        "cb_linkage_20260914",
        "corner_cases_20260907",
        "corner_cases_20260910",
        "defensive_quality_20260918",
        "earnings_surprise_20260918",
        "explore_github_strategies_20260910",
        "explore_platform_strategies_20260910",
        "factor_cs_20260910",
        "factor_cs_allflash_20260910",
        "github_confirm_20260917",
        "margin_flow_20260916",
        "margin_flow_20260917",
        "ml_ranker_20260910",
        "open_mechanism_20260910",
        "order_flow_ranker_20260917",
        "site_visits_20260914",
        "value_regime_20260914",
    }
)


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


def normalize(params: dict[str, object]) -> dict[str, object]:
    """Run the console's create-time validation offline and return params.json.

    Same order and same request-level checks as
    ExperimentManager.create_experiment, then the worker's own
    resolve_worker_options pre-flight. The console's duplicate-directory and
    running-slot checks need the live deployment and stay at POST time.
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
    normally ``workspace_reference`` and ``research_directive``, plus
    any parameter that arm alone changes. ``overrides`` is what the whole round
    decides on top of BASE_OVERRIDES -- normally its dataset selection -- and
    ``pit_views_seed`` the prebuilt view tree every arm hardlinks from.
    """

    arms: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    overrides: Mapping[str, object] = field(default_factory=dict)
    # Empty means the console default seed, which carries the default dataset
    # selection.
    pit_views_seed: str = ""

    def __post_init__(self) -> None:
        reused = sorted(set(self.arms) & RETIRED_IDS)
        if reused:
            raise ValueError(
                "experiment ids that were already used and archived cannot be "
                "reused: " + ", ".join(reused)
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
        return {key: value for key, value in BASE_EXPECTED_DEFAULTS.items() if key not in common}

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
        """The create request body: console defaults, the round's decisions, the id.

        ``PROBE_ID`` stands for the part every arm shares; any other id must be
        one of the round's arms.
        """
        base = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in WEB_CREATE_DEFAULTS.items()
        }
        arm = {} if experiment_id == PROBE_ID else dict(self.arms[experiment_id])
        return {**base, **self.common_overrides, **arm, "experiment_id": experiment_id}

    def validated(self, experiment_id: str) -> tuple[dict[str, object] | None, str]:
        """params.json for one arm (or the probe), or ``(None, reason)``.

        The refusal is returned rather than raised because a dry-run should
        read every arm before it stops.
        """
        try:
            return normalize(self.request_params(experiment_id)), ""
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
                    " this dataset selection and research geometry before creating or"
                    " dry-running the round"
                )
            return None, reason

    def seed_status(self) -> str:
        """One line on the shared seed, printed before anything is validated.

        Says whether the tree's contract can be read at all and which release
        every arm will pin; whether it matches this round's selection, whether
        its build finished and whether that release is published and reaches
        Held-out is the pre-flight's own verdict below.
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
                " written its contract yet, so the round is refused below"
            )
        try:
            record = json.loads(provider.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"seed {self.pit_views_seed}: provider.json unreadable ({exc}); the round is refused below"
        version = record.get("schema_version")
        if version != SNAPSHOT_CACHE_FORMAT_VERSION:
            return (
                f"seed {self.pit_views_seed}: built under snapshot cache format {version!r}"
                f" but this code writes {SNAPSHOT_CACHE_FORMAT_VERSION} -- the pre-flight"
                " refuses it; rebuild the seed under a new directory"
            )
        return (
            f"seed {self.pit_views_seed}: contract present (snapshot cache format {version},"
            f" release {record.get('generation_id')}, which every arm pins); the pre-flight"
            " below compares this round's selection against it, refuses a tree a prebuild is"
            " still staging and checks that release is published and reaches Held-out."
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
        if not dry_run and not self.arms:
            raise SystemExit("this round has no arms to create; --dry-run validates its geometry and seed")
        self.check_console_defaults()
        print(self.seed_status())
        shared, reason = self.validated(PROBE_ID)
        if shared is None:
            # Every arm sends these parameters, so no arm can pass either.
            print(reason, file=sys.stderr)
            return 1
        if dry_run:
            print(json.dumps({key: shared[key] for key in ROUND_REPORT_KEYS}, ensure_ascii=False))
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
                # What this arm decides for itself: everything it sends that
                # differs from the round it belongs to.
                own = {
                    key: value
                    for key, value in merged.items()
                    if key != "research_directive" and value != shared.get(key)
                }
                directive = str(merged["research_directive"])
                print(json.dumps(own, ensure_ascii=False))
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
