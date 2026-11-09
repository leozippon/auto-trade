"""One launcher for every checked-in round definition.

A round is data: which experiments it starts, which reference workspace and exploration
directive each experiment gets, which prebuilt view cache and dataset selection they share,
and the handful of parameters that round decides differently from the console
creation form. Everything around that data -- merging the console defaults,
refusing a default drift, running the console's own create-time validation
offline, reporting a dry-run and POSTing the create requests -- is the same for
every round and lives here, so a round file is its experiments plus a two-line entry
point.

Two decisions are shared rather than per-round because the console runs rounds
side by side and their verdicts only compare while they agree: BASE_OVERRIDES
carries the account, the cost stress and the per-session budgets, and
BASE_EXPECTED_DEFAULTS pins the console creation defaults every round relies
on -- the research geometry and dataset selection the default seed was built
for, and above all the model roles an experiment does not name, so a change of
the console's model defaults must stop the launcher rather than silently move
an experiment onto another model, and the residual statistic's level. A round states what it decides for itself in `overrides`,
and anything it states there stops being an expected default; the rules the
console has no value for (`research_years`) it must state, for the round or
per experiment, or every request is refused. What
several rounds send alike -- a geometry, a model pair's hosted roles -- lives
in `_profiles.py`; an open round imports from there and from here, never from
another round file.

A round's id is the day it is launched, in Beijing time, with a letter suffix
(b, c, ...) when an earlier round of the same day took that id. Its file is
`create_round_<round id>.py`, and `main` refuses an experiment id that does not
end with `_<round id>` (ROUND_FILE_PREFIX says why). Nothing reads an order into
these ids: the queue is the experiment order of a round file.

A round whose experiments have all been created is marked `closed`, and so is one
whose remaining experiments are withdrawn: they leave its experiment list. Its `main`
refuses every mode: the queue reads an experiment without an experiment directory as
pending, so an experiment whose directory was deleted would be created again
(archiving keeps the directory, see archive_experiment.py). Once every experiment of
a closed round has ended, its file leaves the tree (git history keeps it); no later
round takes its id, since a later round is launched on a later day.

`normalize` runs the request-level checks the console applies on POST
/api/experiments (ExperimentManager.create_experiment's closed, unknown,
required and id rules, then the worker's own resolve_worker_options pre-flight,
which type-checks every knob, refuses a malformed research geometry and, for a
named prebuilt view cache, refuses a tree whose cache format or snapshot
configuration is not this round's or whose prebuild has not finished,
and reads the research release the seed was built from -- the release every
experiment will pin -- refusing it unless it is published here and reaches Held-out).
The console's deployment-state checks -- an experiment directory that already
exists and a free running slot -- can only be decided against the live server
and stay at POST time.

`--dry-run` judges the round before its experiments: a probe request carrying only
what every experiment shares goes through the same pre-flight, so the geometry and the
seed contract are read even for a round that has no experiments yet. Such a round can
be dry-run but not created.

`--fill` reads the experiment list as an ordered queue and keeps the console's running
slots occupied: it asks /api/health how many are free under each of the two
running-experiment limits and which GPUs a GPU experiment could take, skips the experiments whose
experiment directory already exists -- running, completed and failed alike, the
console's own rule -- and creates the next pending ones in file order through
the same validation and POST path. A local experiment reached while the local-model
limit is full, and a GPU experiment reached while too few cards are free, stays
pending without stopping the queue, so the experiments behind it still take the free
slots; so does an experiment whose prebuilt view cache has no contract yet or whose
prebuild has not finished, which the next run creates once the seed reports
status ok. Nothing pending or nothing free is the steady state and exits 0, so
the mode is idempotent and safe on a timer; only a creation that was attempted
and refused exits non-zero. A timer run writes one timestamped summary line --
slots under both limits and the free GPUs, and how many experiments were skipped,
created, refused and are still pending, and how many of those the local-model
limit, the free GPUs and an unfinished seed hold back -- plus one line per experiment
it created or that was refused; with `--dry-run` it also lists each pending
experiment and whether it takes a free slot. The research cron
(`ops/cron/research_fill.cron`) runs it over several round files in turn.

Before an experiment is created, and on every dry-run reading of it, `smoke` replays
what the experiment's `output/` starts as -- its reference workspace's starter, else the contract
template -- through the host's own evaluator on the experiment's seed: a few days from
each research-year start inside the full span, and the full span once. A seed
whose views break the replay (round 20261015's five-year seed failed every
replay crossing 2022-07-01, the untouched starter included) refuses the experiment
with the failure instead of letting a session discover it. The smoke exercises
only what that strategy does; the contract template trades nothing (see
`smoke`). The outcome is recorded under logs/research/smoke/, keyed by the
seed, its contract, the geometry and the starter's bytes: a pass is not
replayed again, and a failure keeps refusing until its record is deleted. A GPU
experiment's starter replays on cards the smoke takes for its duration, admitted as
the console admits them (``gpu.host_min_free_gpu_memory_mib``: the cards that
keep its margin free and that no running experiment claims); with too few of those the
experiment is refused for that run only, and nothing is recorded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.common.records import read_json_object
from autotrade.environment import pretrained
from autotrade.environment.artifacts import ArtifactRevision, FilesystemArtifactStore
from autotrade.environment.gpu import GpuUnavailableError, host_min_free_gpu_memory_mib
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import rmtree_keeping_file_modes
from autotrade.environment.sandbox import GpuMode
from autotrade.pipelines.calendar import FULL_SPAN
from autotrade.pipelines.config import (
    SNAPSHOT_CACHE_FORMAT_VERSION,
    acceptance_for,
    research_span,
)
from autotrade.pipelines.experiment_params import Surface, console_request, parameters
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    select_gpus,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.pit_views_seed import UNFINISHED_BUILD_MARKER
from autotrade.pipelines.worker import build_experiment_pipeline, resolve_worker_options

# The console's own id, local-experiment and GPU-request rules; importing them keeps
# this module from growing a second copy of the create contract.
from autotrade.webui.manager import gpu_request, uses_local_model

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
# Where an experiment's reference workspace lives when the experiment does not name one.
PACKS_DIR = "configs/workspace_refs"
# The id the round-level dry-run validates under. Never created: it only
# carries the parameters every experiment shares through the pre-flight.
PROBE_ID = "round_dry_run_probe"
# A round's id is its file's name after this prefix, and every experiment id of the
# round ends with `_<round id>`. An id is never reused: the console keys the
# experiment directory, the sandbox work root, the Docker image tag (on the ZGCA
# instance too) and the archive path on it, so a second run under an old id would
# be indistinguishable from the first in every record that survives it. Round ids
# are unique by file name, so experiment ids ending in them cannot collide across
# rounds.
ROUND_FILE_PREFIX = "create_round_"

# Console defaults every round relies on. Values, not commentary: if the console
# changes any of them the round has to be re-decided, not silently re-run.
BASE_EXPECTED_DEFAULTS: dict[str, object] = {
    # The ten-year geometry and dataset selection the console's default seed
    # was prebuilt for (scripts/experiments/_profiles.py), part of every seed's
    # snapshot configuration.
    "research_start": "20150801",
    "research_end": "20250731",
    "forward_end": "20260731",
    "heldout_end": "20261031",
    "window_months": 132,
    "include_fundamentals": True,
    "fundamental_datasets": (),
    "include_macro": True,
    "macro_datasets": ("index_daily", "index_dailybasic", "sw_daily", "index_weight"),
    "include_events": False,
    "include_text": True,
    "text_datasets": ("anns_d",),
    "include_intraday": False,
    "screen_exclude_st": False,
    "screen_exclude_new_listed_days": 0,
    "screen_boards": (),
    "screen_min_circ_mv_yi": None,
    "screen_max_circ_mv_yi": None,
    # Both the reference workspaces and the directives promise the Agent this many host null
    # controls over the research session.
    "max_null_controls": 12,
    # Derived from SandboxConfig.fit_timeout_seconds; reference workspaces promise the Agent a
    # fit budget of this size.
    "strategy_fit_timeout_seconds": 3600,
    "skill_library": "curated",
    "initial_control_mode": "auto",
    "reasoning_effort": "xhigh",
    "inference_time": "08:00",
    "strategy_period": "day",
    # Every model role, as the console defaults it: the relay's gpt-6-astra
    # for the main conversation, its DeepSeek-V4.1-Flash for the sub-agents
    # and compaction. A role an experiment does not name runs on these; spelled out
    # as literals on purpose, since a change of a console default is exactly
    # the drift this has to catch. An experiment that wants the local model for a
    # role names it.
    "model": "gpt-6-astra",
    "subagent_model": "DeepSeek-V4.1-Flash",
    "compact_model": "DeepSeek-V4.1-Flash",
    # Left to the console, which stamps the boards the experiment's initial cash
    # qualifies for (`broker.default_permitted_boards`); a round that wants
    # others names them.
    "permitted_boards": None,
    # The residual statistic's one-sided level, one for every round so that
    # every reading is stated against the same bar.
    "alpha": 0.05,
}

# What every round decides the same way. Values, not commentary.
BASE_OVERRIDES: dict[str, object] = {
    # No GPU. The request travels with the experiment to the Agent session
    # sandbox and the strategy container of every replay alike; an experiment that
    # really trains on a card overrides this with 1.
    "gpu_count": 0,
    # The researcher's real account, where the CNY 5 minimum commission and lot
    # sizes are a real cost rather than a rounding error.
    "initial_cash": 100_000,
    # The forward verdict's cost stress. The drawdown limits and the tracking
    # mandate are not stated here: an experiment gets a mandate only by naming its own
    # `tracking_error_cap` (config.acceptance_for). Unnamed drawdowns stay
    # 0.45 / 0.30; an experiment that wants other bars names them.
    "cost_stress_multiplier": 2.0,
    # The one research session's budgets, spent across every attempt.
    "max_research_minutes": 10_080,
    "max_llm_calls": 6400,
}

# Both tables name console parameters of the experiment record
# (pipelines/experiment_params.py): a pin or an override of anything else is a typo
# that would otherwise surface only when a round runs.
_NOT_CONSOLE = sorted(
    {*BASE_EXPECTED_DEFAULTS, *BASE_OVERRIDES} - set(parameters(Surface.CONSOLE))
)
if _NOT_CONSOLE:
    raise RuntimeError(f"not console parameters of the experiment record: {_NOT_CONSOLE}")

# Reported once per round on --dry-run: what every experiment shares.
ROUND_REPORT_KEYS: tuple[str, ...] = (
    "research_start",
    "research_end",
    "forward_end",
    "heldout_end",
    "pit_views_seed",
    "benchmark_index",
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
    "skill_library",
    "max_research_minutes",
    "max_llm_calls",
    "max_null_controls",
    "strategy_fit_timeout_seconds",
    "strategy_inference_timeout_seconds",
    "strategy_cpus",
    "strategy_memory",
    "strategy_state_disk",
    "initial_cash",
    # null: the console stamps the boards the initial cash qualifies for.
    "permitted_boards",
    "cost_stress_multiplier",
    "gpu_count",
    "reasoning_effort",
    "initial_control_mode",
    "inference_time",
    "strategy_period",
    "model",
    "subagent_model",
    "compact_model",
)

def round_id(program: str) -> str:
    """The id of the round whose file ``program`` (a round's ``argv[0]``) is."""

    stem = Path(program).stem
    if not stem.startswith(ROUND_FILE_PREFIX) or stem == ROUND_FILE_PREFIX:
        raise SystemExit(
            f"{program} is not a round file: a round runs from"
            f" scripts/experiments/{ROUND_FILE_PREFIX}<round id>.py"
        )
    return stem.removeprefix(ROUND_FILE_PREFIX)


def normalize(params: dict[str, object]) -> dict[str, object]:
    """Run the console's create-time validation offline and return params.json.

    Same order and same request-level checks as
    ExperimentManager.create_experiment, then the worker's own
    resolve_worker_options pre-flight. The console's duplicate-directory and
    running-slot checks need the live deployment and stay at POST time.
    """
    merged = console_request(
        params,
        defaults=WEB_CREATE_DEFAULTS,
        experiments_root=str(EXPERIMENTS_ROOT),
        work_root=str(REPO_ROOT / ".runtime/sandboxes"),
    )
    merged["_creation_surface"] = "webui"
    resolve_worker_options(
        merged,
        experiment_dir=EXPERIMENTS_ROOT / str(merged["experiment_id"]),
        repo_root=REPO_ROOT,
        preflight=True,
    )
    return merged


# Trading days each research-year start is replayed for by the smoke.
SMOKE_DAYS = 3
# What decides a starter's replay besides its own bytes and its seed's contract.
SMOKE_KEYS = (
    "pit_views_seed",
    "research_start",
    "research_end",
    "window_months",
    "benchmark_index",
    "initial_cash",
    "permitted_boards",
)


def smoke(merged: Mapping[str, object]) -> str:
    """Replay an experiment's starter on its seed before the experiment is created; "" when it
    passed, else why not.

    ``merged`` is the experiment's validated params.json. What the session's
    ``output/`` starts as is replayed: the reference workspace's ``starter/``, or the experiment's
    strategy template (``strategy_path``) when the reference workspace ships no ``starter/``
    or there is no reference workspace. A ``starter/`` without ``main.py`` is refused, since
    the session would silently start from the template instead. It is replayed
    by the host's own evaluator as a session's smoke_backtest and Validation
    replay it: over the full research span, a few days from each research-year
    start (the late-start path a defective seed breaks first), then the whole
    span. The contract template reads nothing and places no orders, so its
    smoke shows only that the seed's views build for every decision, the
    strategy container starts and the contract holds; a seed defect that only a
    strategy's own reads, its ``fit`` or the order and fill path would hit is
    not caught. The outcome is recorded once per seed contract, geometry and
    starter bytes, so a pass is not replayed by the next timer run and a
    failure is not retried until its record is deleted. A starter whose experiment
    asks for GPUs replays on cards it takes for the smoke under the console's
    admission margin (``gpu.host_min_free_gpu_memory_mib``); when too few pass
    it, ``GpuUnavailableError`` propagates and nothing is recorded, so a later
    run smokes the experiment again. An admission setting that cannot be read is
    refused the same way: it is no verdict on the starter either.
    """

    reference = str(merged.get("workspace_reference") or "")
    starter = REPO_ROOT / reference / "starter"
    if not reference or not starter.exists():
        starter = (REPO_ROOT / str(merged["strategy_path"])).parent
    elif not (starter / "main.py").is_file():
        return f"the reference workspace's starter cannot be smoked: {starter} lacks main.py"
    digest = hashlib.sha256(
        json.dumps({key: merged.get(key) for key in SMOKE_KEYS}, sort_keys=True, default=str).encode("utf-8")
    )
    seed = str(merged.get("pit_views_seed") or "")
    if seed:
        digest.update((REPO_ROOT / seed / "provider.json").read_bytes())
    for path in sorted(starter.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(starter)).encode("utf-8") + b"\0" + path.read_bytes())
    record_path = REPO_ROOT / "logs" / "research" / "smoke" / f"{digest.hexdigest()[:20]}.json"
    if record_path.is_file():
        record = read_json_object(record_path)
        if record["passed"]:
            return ""
        return f"{record['failure']} (recorded {record_path.relative_to(REPO_ROOT)}; delete it to smoke again)"
    started = time.monotonic()
    failure = _smoke_replays(merged, starter)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "experiment_id": merged["experiment_id"],
                "starter": str(starter.relative_to(REPO_ROOT)),
                **{key: merged.get(key) for key in SMOKE_KEYS},
                "smoke_days": SMOKE_DAYS,
                "passed": not failure,
                "failure": failure,
                "seconds": round(time.monotonic() - started, 1),
                "at": _now(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return failure


def _smoke_replays(merged: Mapping[str, object], starter: Path) -> str:
    """The replays of :func:`smoke`, in a scratch experiment assembled exactly
    as the worker assembles one (its seed hardlinked in, the
    configured sandbox image, the pretrained store pinned as the console will
    pin it, no Agent run); "" or the first failure."""

    scratch = REPO_ROOT / ".runtime" / "round_smoke" / uuid.uuid4().hex
    experiment_dir = scratch / str(merged["experiment_id"])
    experiment_dir.mkdir(parents=True)
    try:
        options = resolve_worker_options(
            {
                **merged,
                "experiments_root": str(scratch),
                "pretrained_manifest_sha256": pretrained.manifest_digest(),
            },
            experiment_dir=experiment_dir,
            repo_root=REPO_ROOT,
        )
        # No console claims cards for the smoke, so it takes free ones for its
        # duration, as scripts/experiments/_cli.py does for a command-line run:
        # a strategy container that asks for a GPU and names none refuses to start.
        # The admission margin is the console's own, so a card the console would
        # start the experiment on is not refused here for being busy.
        spec = options.agent_sandbox
        if spec is not None and spec.gpu == GpuMode.AUTO:
            try:
                margin = host_min_free_gpu_memory_mib()
            except ValueError as exc:
                # An admission margin that cannot be read is no verdict on the
                # starter either: it is reported as the GPU refusal it is.
                raise GpuUnavailableError(str(exc)) from exc
            devices = select_gpus(
                EXPERIMENTS_ROOT,
                spec.gpu_count,
                require_name=spec.gpu_name_filter,
                min_free_memory_mib=margin,
            )
            options = replace(options, agent_sandbox=replace(spec, gpu=tuple(devices)))
        pipeline = build_experiment_pipeline(
            options,
            ledger=ExperimentLedger(experiment_dir / "ledgers" / "experiment_ledger.jsonl"),
            store=FilesystemArtifactStore(experiment_dir / "artifacts" / "strategy"),
            ref_store=AgentRefStore(experiment_dir),
        ).pipeline
        output = scratch / "output"
        shutil.copytree(starter, output, ignore=shutil.ignore_patterns("__pycache__"))
        # Every year of the geometry the experiment replays, the validation years
        # too: its tests replay those on the same seed.
        years = pipeline.research_spans(options.experiment.geometry.research_years)
        request = research_span(years, FULL_SPAN).request(
            ArtifactRevision("smoke", output),
            schedule=options.experiment.schedule,
            broker_profile=options.experiment.broker_profile,
        )
        runs = [(year.label, year.start, SMOKE_DAYS) for year in years] + [(FULL_SPAN, None, None)]
        for label, start, days in runs:
            try:
                pipeline.evaluator.evaluate(request, max_days=days, start_day=start)
            except Exception as exc:  # noqa: BLE001 - the exception text IS the result
                where = f"{days} days from {start}" if start else f"the full span {request.start}..{request.end}"
                return f"{label}, {where}: {type(exc).__name__}: {exc}"
        return ""
    except GpuUnavailableError:
        # Too few admissible cards says nothing about the starter or its seed.
        raise
    except Exception as exc:  # noqa: BLE001 - an experiment that cannot even be assembled fails its smoke
        return f"the smoke could not be assembled: {type(exc).__name__}: {exc}"
    finally:
        rmtree_keeping_file_modes(scratch)


def seed_ready(seed: str) -> bool:
    """Whether a named prebuilt view cache has its contract and a finished prebuild;
    the console default (no seed) always is. Readiness alone: whether the tree
    is this experiment's is the pre-flight's to judge."""

    root = REPO_ROOT / seed
    return not seed or ((root / "provider.json").is_file() and not (root / UNFINISHED_BUILD_MARKER).exists())


def _now() -> str:
    """Local time with its offset: the cron log's runs are told apart by it."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def post(port: int, params: dict[str, object]) -> bool:
    """POST one create request; print one line and report whether the console
    accepted it."""
    experiment_id = params["experiment_id"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/experiments",
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Flushed: stdout and stderr share the cron log, and a buffered line would
    # land after the refusals that followed it.
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read(400).decode("utf-8", "replace")
            print(f"{experiment_id}: created, HTTP {response.status} {body}", flush=True)
            return True
    except urllib.error.HTTPError as exc:
        body = exc.read(800).decode("utf-8", "replace")
        print(f"{experiment_id}: refused, HTTP {exc.code} {body}", file=sys.stderr)
    except urllib.error.URLError as exc:
        # No console on that port, or it dropped the connection: an operator
        # error, not a traceback.
        print(f"{experiment_id}: not sent, console unreachable: {exc.reason}", file=sys.stderr)
    return False


# What --fill reads from /api/health.
HEALTH_KEYS = (
    "max_running_experiments",
    "max_running_local_experiments",
    "running",
    "running_local",
    "gpus_free",
)


def health(port: int) -> dict[str, object]:
    """The console's running roster, its two running-experiment limits and its free GPUs.

    Read rather than assumed: the limits are the console's constants and the
    roster and the cards change under the operator's hands, so a fill that
    cannot read them -- a console too old to report them included -- refuses
    instead of creating blind against limits it guessed.
    """
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            record = json.loads(response.read())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{_now()} console health unreadable at {url}: {exc}") from exc
    missing = [key for key in HEALTH_KEYS if key not in record]
    if missing:
        raise SystemExit(
            f"{_now()} console health at {url} lacks {', '.join(missing)}: "
            "restart the console onto this checkout"
        )
    return record


@dataclass(frozen=True)
class Round:
    """One round definition: its experiments and what it decides differently.

    ``experiments`` maps experiment id to the per-experiment part of the create request --
    normally ``research_directive`` (the reference workspace defaults to the one named after
    the experiment, see :meth:`request_params`), plus any parameter that experiment alone
    changes. ``overrides`` is what the whole round decides on top of
    BASE_OVERRIDES -- normally its dataset selection, and any default every experiment
    shares -- and ``pit_views_seed`` the prebuilt view tree every experiment hardlinks
    from. ``closed`` says every experiment has been created: the file is then only
    the record of what was run, and :meth:`main` refuses it.
    """

    experiments: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    overrides: Mapping[str, object] = field(default_factory=dict)
    # Empty means the console default seed, which carries the default dataset
    # selection.
    pit_views_seed: str = ""
    # Stated, not read from disk: an experiment whose directory was deleted would look
    # pending.
    closed: bool = False

    def check_ids(self, round_id: str) -> None:
        """Refuse an experiment id that does not end with ``_<round_id>``
        (ROUND_FILE_PREFIX says why)."""

        foreign = [experiment for experiment in self.experiments if not experiment.endswith(f"_{round_id}")]
        if foreign:
            raise SystemExit(
                f"an experiment id of round {round_id} must end with _{round_id}: " + ", ".join(foreign)
            )

    @property
    def common_overrides(self) -> dict[str, object]:
        """What every experiment of this round sends, seed included."""
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

        ``PROBE_ID`` stands for the part every experiment shares; any other id must be
        one of the round's experiments. An experiment's reference workspace is the one named after it,
        ``configs/workspace_refs/<id>``, unless the experiment or the round names
        another (``""`` for none); the pre-flight refuses one that is missing.
        """
        base = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in WEB_CREATE_DEFAULTS.items()
        }
        common = self.common_overrides
        own = {} if experiment_id == PROBE_ID else dict(self.experiments[experiment_id])
        if experiment_id != PROBE_ID and "workspace_reference" not in {*own, *common}:
            own["workspace_reference"] = f"{PACKS_DIR}/{experiment_id}"
        return {**base, **common, **own, "experiment_id": experiment_id}

    def validated(self, experiment_id: str) -> tuple[dict[str, object] | None, str]:
        """params.json for one experiment (or the probe), or ``(None, reason)``.

        The refusal is returned rather than raised because a dry-run should
        read every experiment before it stops.
        """
        try:
            return normalize(self.request_params(experiment_id)), ""
        except ValueError as exc:
            reason = f"{experiment_id}: parameters rejected, nothing was sent: {exc}"
            if "unfinished build" in str(exc):
                reason += (
                    "\n  the tree is there and its contract is this round's, but its"
                    " prebuild has not finished: wait for a running"
                    " scripts/data/prebuild_pit_views_seed.py, or re-run one that"
                    " failed, until it reports status ok, then re-run --"
                    " no parameter needs changing"
                )
            elif "pit_views_seed" in str(exc) or "view cache" in str(exc):
                reason += (
                    f"\n  every experiment shares one prebuilt view tree ({self.pit_views_seed});"
                    " build it with scripts/data/prebuild_pit_views_seed.py for exactly"
                    " this dataset selection and research geometry before creating or"
                    " dry-running the round"
                )
            return None, reason

    def seed_status(self) -> str:
        """One line on the shared seed, printed before anything is validated.

        Says whether the tree's contract can be read at all and which release
        every experiment will pin; whether it matches this round's selection, whether
        its build finished and whether that release is published and reaches
        Held-out is the pre-flight's own verdict below.
        """
        if not self.pit_views_seed:
            return (
                "seed: this round names none, so its experiments use the console default tree"
                " and cold-build anything it does not carry"
            )
        provider = REPO_ROOT / self.pit_views_seed / "provider.json"
        if not provider.is_file():
            return (
                f"seed {self.pit_views_seed}: provider.json absent -- the prebuild has not"
                " written its contract yet, so the round is refused below"
            )
        try:
            record = read_json_object(provider)
        except (OSError, ValueError) as exc:
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
            f" release {record.get('generation_id')}, which every experiment pins); the pre-flight"
            " below compares this round's selection against it, refuses a tree whose prebuild"
            " has not finished and checks that release is published and reaches Held-out."
        )

    def fill(
        self, port: int, *, dry_run: bool
    ) -> tuple[list[str], list[str], dict[str, list[str]], str]:
        """The experiments to create now, every pending experiment, the pending experiments held
        back by each reason that clears by itself, and the slot part of the
        summary line.

        The queue is the experiment order of the round file. An experiment whose
        directory exists has been created already -- the console refuses a
        second one whatever state it reached -- so only the rest are pending.
        They take the console's free slots in order, except that an experiment the
        console would refuse for a reason that clears by itself is held: it
        stays pending and the experiments behind it go ahead. That is a local experiment (the
        console's own ``uses_local_model``) reached while the local-model limit
        is full, an experiment asking for more GPUs (``gpu_request``) than the console
        has free cards for, and an experiment whose seed has no contract yet or an
        unfinished prebuild (``seed_ready``). A dry-run lists the pending experiments;
        a timer run only counts them.
        """
        record = health(port)
        running = sorted(str(name) for name in record["running"])
        running_local = sorted(str(name) for name in record["running_local"])
        cap = int(record["max_running_experiments"])
        local_cap = int(record["max_running_local_experiments"])
        free = max(cap - len(running), 0)
        local_free = max(local_cap - len(running_local), 0)
        cards = [str(device) for device in record["gpus_free"]]  # type: ignore[attr-defined]
        gpus_free = len(cards)
        slots = (
            f"slots {len(running)}/{cap} in use ({', '.join(running) or 'none'}), {free} free; "
            f"local-model slots {len(running_local)}/{local_cap} in use "
            f"({', '.join(running_local) or 'none'}), {local_free} free; "
            f"GPUs {gpus_free} free ({', '.join(cards) or 'none'})"
            + (f", unreadable: {record['gpu_error']}" if record.get("gpu_error") else "")
        )
        pending = [experiment for experiment in self.experiments if not (EXPERIMENTS_ROOT / experiment).exists()]
        chosen: list[str] = []
        held: dict[str, list[str]] = {"local": [], "gpu": [], "seed": []}
        for experiment_id in pending:
            if len(chosen) == free:
                break
            params = self.request_params(experiment_id)
            if not seed_ready(str(params.get("pit_views_seed") or "")):
                held["seed"].append(experiment_id)
                continue
            local = uses_local_model(params)
            if local and local_free == 0:
                held["local"].append(experiment_id)
                continue
            gpus = gpu_request(params)
            if gpus > gpus_free:
                held["gpu"].append(experiment_id)
                continue
            local_free -= local
            gpus_free -= gpus
            chosen.append(experiment_id)
        if dry_run:
            waits = {
                "local": "waits for a local-model slot",
                "gpu": "waits for a free GPU",
                "seed": "waits for its seed's prebuild to finish",
            }
            for experiment_id in pending:
                slot = next(
                    (waits[reason] for reason, experiments in held.items() if experiment_id in experiments),
                    "takes a free slot" if experiment_id in chosen else "waits for a free slot",
                )
                print(f"{experiment_id}: pending, {slot}")
        return chosen, pending, held, slots

    def main(self, argv: list[str], usage: str | None = None) -> int:
        """`<port> [--dry-run] [--fill] [experiment_id ...]`, shared by every round file."""
        if self.closed:
            raise SystemExit(
                "this round is closed: every experiment it defines has been created, and the file"
                " is only the record of what was run"
            )
        self.check_ids(round_id(argv[0]))
        # A mistyped flag must never fall through to the real POST path: without
        # this, --dryrun is read as an experiment-id filter and creates the round.
        mistyped = [
            arg for arg in argv[1:] if arg.startswith("--") and arg not in ("--dry-run", "--fill")
        ]
        if mistyped:
            print("unknown option: " + ", ".join(mistyped), file=sys.stderr)
            return 2
        if len(argv) < 2 or not argv[1].isdigit():
            raise SystemExit(usage or self.main.__doc__)
        port = int(argv[1])
        dry_run = "--dry-run" in argv
        fill = "--fill" in argv
        wanted = {arg for arg in argv[2:] if not arg.startswith("--")}
        unknown_ids = sorted(wanted - set(self.experiments))
        if unknown_ids:
            raise SystemExit("not in this round: " + ", ".join(unknown_ids))
        if fill and wanted:
            raise SystemExit("--fill reads the queue itself; name no experiment id")
        if not dry_run and not self.experiments:
            raise SystemExit("this round has no experiments to create; --dry-run validates its geometry and seed")
        self.check_console_defaults()
        if not fill or dry_run:
            # A timer run's log carries its summary line, not the seed's
            # contract every ten minutes.
            print(self.seed_status())
        shared, reason = self.validated(PROBE_ID)
        if shared is None:
            # Every experiment sends these parameters, so no experiment can pass either.
            print(f"{_now()} {reason}" if fill else reason, file=sys.stderr)
            return 1
        if dry_run:
            print(json.dumps({key: shared[key] for key in ROUND_REPORT_KEYS}, ensure_ascii=False))
        if fill:
            # The queue decides the selection; --dry-run still decides whether
            # anything is sent.
            selected, pending, held, slots = self.fill(port, dry_run=dry_run)
        else:
            selected = [experiment for experiment in self.experiments if not wanted or experiment in wanted]
        created: list[str] = []
        failed: list[str] = []
        for experiment_id in selected:
            merged, reason = self.validated(experiment_id)
            if merged is None:
                # On a dry-run the refusal is a reading, so the remaining experiments
                # are read too and the exit code still reports it; on the POST
                # path nothing more is sent after a rejected experiment.
                print(reason, file=sys.stderr)
                failed.append(experiment_id)
                if dry_run:
                    continue
                break
            try:
                failure = smoke(merged)
                if failure:
                    failure = f"the starter's smoke failed on {merged.get('pit_views_seed')}, nothing was sent: {failure}"
            except GpuUnavailableError as exc:
                failure = (
                    "no GPU is admissible for the starter's smoke, so nothing was smoked,"
                    f" recorded or sent: {exc}"
                )
            if failure:
                print(f"{experiment_id}: {failure}", file=sys.stderr)
                failed.append(experiment_id)
                if dry_run:
                    continue
                break
            if dry_run:
                # What this experiment decides for itself: everything it sends that
                # differs from the round it belongs to.
                own = {
                    key: value
                    for key, value in merged.items()
                    if key != "research_directive" and value != shared.get(key)
                }
                directive = str(merged["research_directive"])
                print(json.dumps(own, ensure_ascii=False))
                print("  smoke: the starter replayed from every research-year start and over the full span")
                # The gates this experiment would be judged by: the defaults with
                # whatever the experiment names, mandate included.
                rules = acceptance_for(merged)
                print("  acceptance:", json.dumps(rules.to_record(), ensure_ascii=False))
                print(f"  directive: {len(directive.splitlines())} lines, {len(directive)} chars")
                for line in directive.splitlines():
                    print("   |", line)
                continue
            if post(port, self.request_params(experiment_id)):
                created.append(experiment_id)
            else:
                failed.append(experiment_id)
        if fill:
            done = len(selected) - len(failed) if dry_run else len(created)
            print(
                f"{_now()} fill {Path(argv[0]).stem}: {slots}; queue {len(self.experiments)}: "
                f"{len(self.experiments) - len(pending)} skipped (created already), "
                f"{done} {'would be created' if dry_run else 'created'}, "
                f"{len(failed)} refused, {len(pending) - done - len(failed)} pending "
                f"({len(held['local'])} held by the local-model limit, {len(held['gpu'])} by the free GPUs,"
                f" {len(held['seed'])} by an unfinished seed)",
                flush=True,
            )
        if failed:
            print(("refused: " if dry_run else "not created: ") + ", ".join(failed), file=sys.stderr)
            return 1
        return 0
