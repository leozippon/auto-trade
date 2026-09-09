"""Every checked-in round, read through the one launcher that creates it.

The round files are data -- arms, seed, dataset selection, directives -- and
`scripts/experiments/_round.py` is the behaviour, so these checks are written
once and parametrised over whatever round files exist. A new round file is
covered the moment it is added.
"""

from __future__ import annotations

import importlib
import json

import pytest

from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL
from autotrade.environment.tools.prior_policy import calendar_policy_violation
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
from autotrade.webui.manager import MAX_RUNNING_EXPERIMENTS
from scripts.experiments._round import (
    BASE_EXPECTED_DEFAULTS,
    PARENT_CONTROL_LINE,
    REPO_ROOT,
    RETIRED_IDS,
    ROBUSTNESS_LINE,
    Round,
    archived_ids,
)

# Every model role a create request carries.
MODEL_ROLES = (
    "model",
    "meta_model",
    "subagent_model",
    "analysis_model",
    "compact_model",
    "nl_model",
)
# The one arm that really trains on a device.
GPU_ARM = "ml_ranker_20260910"


def _rounds() -> dict[str, Round]:
    """Every checked-in round file, by module name."""
    paths = sorted((REPO_ROOT / "scripts" / "experiments").glob("create_round_*.py"))
    assert paths, "no round definitions found"
    return {
        path.stem: importlib.import_module(f"scripts.experiments.{path.stem}").ROUND
        for path in paths
    }


ROUNDS = _rounds()
ROUND_IDS = sorted(ROUNDS)
ARMS = [(name, arm) for name, rnd in sorted(ROUNDS.items()) for arm in rnd.arms]


def _seed_not_ready(rnd: Round) -> str:
    """Why this round's seed cannot be read yet, or "" when it can.

    Operator state only. A named seed tree is a gitignored artifact built by
    scripts/data/prebuild_pit_views_seed.py, so a fresh checkout, an older
    cache format and a prebuild still writing into the tree are all waits
    rather than failures. A seed that exists and is finished but carries a
    different selection is NOT a wait -- that is the drift these checks exist
    to catch, so it is deliberately left to fail.
    """
    if not rnd.pit_views_seed:
        return ""
    seed = REPO_ROOT / rnd.pit_views_seed
    if not seed.is_dir():
        return f"prebuilt PIT view seed {rnd.pit_views_seed} is not present"
    try:
        recorded = json.loads((seed / "provider.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"prebuilt PIT view seed {rnd.pit_views_seed} has no readable provider.json"
    cache_format = recorded.get("schema_version")
    if cache_format != SNAPSHOT_CACHE_FORMAT_VERSION:
        return (
            f"prebuilt PIT view seed {rnd.pit_views_seed} was built under snapshot cache "
            f"format {cache_format}; this code writes {SNAPSHOT_CACHE_FORMAT_VERSION}"
        )
    _, reason = rnd.validated(next(iter(rnd.arms)))
    # assert_seed_snapshot_config refuses a tree a prebuild is still staging
    # views into, and names it in exactly these words.
    if "unfinished build" in reason:
        return f"prebuilt PIT view seed {rnd.pit_views_seed} has an unfinished build"
    return ""


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_every_arm_runs_every_role_on_the_local_model(round_name: str, experiment_id: str) -> None:
    """Cost policy: no arm may open a hosted stream.

    No round overrides a model role, so the guarantee rests entirely on the
    console defaults -- which is why BASE_EXPECTED_DEFAULTS pins all six and
    check_console_defaults refuses a drift.
    """
    rnd = ROUNDS[round_name]
    assert LOCAL_QWEN_MODEL == "qwen-3.8-27b-fp8"
    assert set(MODEL_ROLES).isdisjoint(rnd.common_overrides)
    rnd.check_console_defaults()
    params = rnd.request_params(experiment_id)
    for role in MODEL_ROLES:
        assert BASE_EXPECTED_DEFAULTS[role] == LOCAL_QWEN_MODEL, role
        assert params[role] == LOCAL_QWEN_MODEL, (experiment_id, role)


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_only_the_ml_ranker_arm_takes_a_gpu(round_name: str, experiment_id: str) -> None:
    """The GPU request travels with the experiment to the Agent's session
    sandbox and to the strategy container of every formal replay, so an
    accidental card is a real cost on a shared machine."""
    params = ROUNDS[round_name].request_params(experiment_id)
    assert params["gpu_count"] == (1 if experiment_id == GPU_ARM else 0), experiment_id


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_every_directive_is_usable_by_the_worker(round_name: str, experiment_id: str) -> None:
    """A directive must be non-empty, carry the two shared readings, and hold
    no literal calendar date.

    worker.resolve_worker_options runs fold_exploration_directive through
    prior_policy.calendar_policy_violation, so a date written into one is not a
    style issue: the arm would be refused at every start. Data windows that
    must be excluded are named by their cause and defined in the reference pack
    instead.
    """
    directive = str(ROUNDS[round_name].request_params(experiment_id)["fold_exploration_directive"])
    assert directive.strip(), experiment_id
    assert calendar_policy_violation(directive) == "", experiment_id
    assert PARENT_CONTROL_LINE in directive, experiment_id
    assert ROBUSTNESS_LINE in directive, experiment_id


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_every_reference_pack_a_round_names_exists(round_name: str, experiment_id: str) -> None:
    """A workspace_reference that does not exist fails the session at start.

    An arm may name none -- open_mechanism starts from the empty template with
    nothing mounted but the operating memory -- but one it names must be there.
    """
    reference = ROUNDS[round_name].request_params(experiment_id).get("workspace_reference")
    if not reference:
        return
    assert (REPO_ROOT / str(reference)).is_dir(), reference
    assert (REPO_ROOT / str(reference) / "README.md").is_file(), reference


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_every_arm_passes_the_offline_create_validation(round_name: str, experiment_id: str) -> None:
    """What `--dry-run` reports: the console's own create-time checks and the
    worker pre-flight accept this arm as it stands."""
    rnd = ROUNDS[round_name]
    waiting = _seed_not_ready(rnd)
    if waiting:
        pytest.skip(waiting)
    merged, reason = rnd.validated(experiment_id)
    assert merged is not None, reason
    for key in rnd.keys_reported:
        assert key in merged, key


@pytest.mark.parametrize("round_name", ROUND_IDS)
def test_a_round_that_names_a_seed_sends_one_selection_for_all_its_arms(round_name: str) -> None:
    """A seed's identity is the whole snapshot configuration.

    One differing dataset name, or a domain switch off (which drops the whole
    selection in worker._snapshot_config), and that arm no longer matches the
    prebuilt tree -- it would cold-build every view instead. Nothing an arm
    states for itself may touch that selection.
    """
    rnd = ROUNDS[round_name]
    if not rnd.pit_views_seed:
        return
    selections = set()
    for experiment_id in rnd.arms:
        params = rnd.request_params(experiment_id)
        assert params["pit_views_seed"] == rnd.pit_views_seed, experiment_id
        assert params["include_macro"] is True, experiment_id
        assert params["include_events"] is True, experiment_id
        selections.add((tuple(params["macro_datasets"]), tuple(params["events_datasets"])))
    assert len(selections) == 1


@pytest.mark.parametrize("round_name", ROUND_IDS)
def test_the_selection_matches_the_prebuilt_seed(round_name: str) -> None:
    """The round and the tree its arms hardlink from agree.

    The pre-flight refuses a mismatch on its own; comparing here as well is
    what says which half drifted when it does.
    """
    rnd = ROUNDS[round_name]
    if not rnd.pit_views_seed:
        return
    waiting = _seed_not_ready(rnd)
    if waiting:
        pytest.skip(waiting)
    recorded = json.loads(
        (REPO_ROOT / rnd.pit_views_seed / "provider.json").read_text(encoding="utf-8")
    )["snapshot_config"]["datasets"]
    params = rnd.request_params(next(iter(rnd.arms)))
    assert recorded["macro"] == list(params["macro_datasets"])
    assert recorded["events"] == list(params["events_datasets"])


def test_no_round_reuses_an_experiment_id() -> None:
    """An id is never reused, by any round.

    The console keys the experiment directory, the sandbox work root, the
    Docker image tag and the archive path on the experiment id, so a second run
    under an old name would be indistinguishable from the first in every record
    that survives it. Three sources of "already used" are checked: the other
    round files, the ids retired out of them, and -- where the operator's
    archive exists -- what is actually archived on this machine.
    """
    seen: dict[str, str] = {}
    for round_name in ROUND_IDS:
        for experiment_id in ROUNDS[round_name].arms:
            assert experiment_id not in seen, (
                f"{experiment_id} is defined by both {seen.get(experiment_id)} and {round_name}"
            )
            assert experiment_id not in RETIRED_IDS, (
                f"{round_name} reuses the retired id {experiment_id}"
            )
            seen[experiment_id] = round_name

    archived = archived_ids()
    if not archived:
        pytest.skip("logs/archive/ is not present, so archived ids cannot be cross-checked")
    # An arm archived from an earlier attempt of a round that still defines it
    # is not retired; everything else that was archived is.
    assert archived - set(seen) <= RETIRED_IDS, (
        "these ids are archived but are neither in a round file nor in "
        f"RETIRED_IDS, so a later round could silently reuse one: "
        f"{sorted(archived - set(seen) - RETIRED_IDS)}"
    )


def test_the_console_could_hold_every_arm_the_round_files_define() -> None:
    """The console refuses a create or resume past MAX_RUNNING_EXPERIMENTS, so
    the checked-in rounds together must not describe more arms than it can
    run."""
    total = sum(len(rnd.arms) for rnd in ROUNDS.values())
    assert total <= MAX_RUNNING_EXPERIMENTS
