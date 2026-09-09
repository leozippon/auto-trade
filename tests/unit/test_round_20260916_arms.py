"""2026-09-16 arms: three new information sources refill the slots the retired
arms freed, every model role stays on the local Qwen and no arm asks for a GPU,
and all three ask for the one extended dataset selection their shared PIT view
seed was prebuilt for."""

from __future__ import annotations

import json

import pytest

from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL
from autotrade.environment.tools.prior_policy import calendar_policy_violation
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
from autotrade.pipelines.hitl_state import WEB_CLOSED_PARAMS, WEB_CREATE_DEFAULTS
from autotrade.webui.manager import MAX_RUNNING_EXPERIMENTS
from scripts.experiments.create_round_20260910 import ROUND as ROUND_20260910
from scripts.experiments.create_round_20260914 import ROUND as ROUND_20260914
from scripts.experiments.create_round_20260916 import (
    COMMON_OVERRIDES,
    EVENTS_DATASETS,
    EXPECTED_DEFAULTS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    REPO_ROOT,
    REPORT_KEYS,
    ROUND,
    check_console_defaults,
    normalize,
    request_params,
)

# The arms of the two earlier rounds that keep running beside this one.
# `value_regime_20260914` is not among them: it was asked to stop at a session
# boundary in the same batch that freed these slots, so its round module is now
# a record rather than a live definition.
SURVIVING = (
    "explore_github_strategies_20260910",
    "ml_ranker_20260910",
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
# What this round adds to round 20260914's events selection: one table per arm.
NEW_EVENTS_TABLES = {"margin_secs", "report_rc"}

SEED_DIR = REPO_ROOT / PIT_VIEWS_SEED


def _seed_cache_format() -> object | None:
    try:
        return json.loads((SEED_DIR / "provider.json").read_text(encoding="utf-8")).get("schema_version")
    except (OSError, json.JSONDecodeError):
        return None


def _seed_not_ready() -> str:
    """Why the seed cannot be read yet, or "" when it can.

    Operator state only. The tree is a gitignored artifact built by
    scripts/data/prebuild_pit_views_seed.py, so a fresh checkout, an older
    cache format and a prebuild still writing into the tree are all waits
    rather than failures. A seed that exists and is finished but carries a
    different selection is NOT a wait -- that is the drift this test exists to
    catch, so it is deliberately left to fail.
    """
    if not SEED_DIR.is_dir():
        return f"prebuilt PIT view seed {PIT_VIEWS_SEED} is not present"
    cache_format = _seed_cache_format()
    if cache_format != SNAPSHOT_CACHE_FORMAT_VERSION:
        return (
            f"prebuilt PIT view seed {PIT_VIEWS_SEED} was built under snapshot cache "
            f"format {cache_format}; this code writes {SNAPSHOT_CACHE_FORMAT_VERSION}"
        )
    try:
        normalize(request_params(next(iter(ROUND))))
    except ValueError as exc:
        # pit_views_seed.assert_seed_snapshot_config refuses a tree a prebuild
        # is still staging views into, and names it in exactly these words.
        if "unfinished build" in str(exc):
            return f"prebuilt PIT view seed {PIT_VIEWS_SEED} has an unfinished build"
    return ""


_SEED_NOT_READY = _seed_not_ready()
needs_seed = pytest.mark.skipif(bool(_SEED_NOT_READY), reason=_SEED_NOT_READY)


def test_the_round_is_three_arms_the_console_has_room_for() -> None:
    """Three new arms, none of them a re-created earlier arm, inside the cap.

    The console refuses a create or resume past MAX_RUNNING_EXPERIMENTS, so
    what must hold is that this round plus the arms still running fits under
    it -- not an exact fill, which depends on when the stopping arm exits.
    """
    assert len(ROUND) == 3
    assert set(SURVIVING) < set(ROUND_20260910)
    assert set(ROUND).isdisjoint(set(ROUND_20260910) | set(ROUND_20260914))
    assert len(ROUND) + len(SURVIVING) <= MAX_RUNNING_EXPERIMENTS


def test_every_arm_runs_every_role_on_the_local_model() -> None:
    """Cost policy: no arm may open a hosted stream.

    The round overrides no model role, so the guarantee rests entirely on the
    console defaults -- which is why EXPECTED_DEFAULTS pins all six and
    check_console_defaults refuses a drift.
    """
    assert LOCAL_QWEN_MODEL == "qwen-3.8-27b-fp8"
    assert set(MODEL_ROLES).isdisjoint(COMMON_OVERRIDES)
    check_console_defaults()
    for role in MODEL_ROLES:
        assert EXPECTED_DEFAULTS[role] == LOCAL_QWEN_MODEL, role
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        # Including the LightGBM ranker: its pack forbids torch and fits on
        # the CPU, so unlike ml_ranker_20260910 it must not attach a card.
        assert params["gpu_count"] == 0, experiment_id
        for role in MODEL_ROLES:
            assert params[role] == LOCAL_QWEN_MODEL, (experiment_id, role)


def test_every_arm_sends_the_same_seed_and_the_same_dataset_selection() -> None:
    """A seed's identity is the whole snapshot configuration.

    One differing dataset name, or a domain switch off (which drops the whole
    selection in worker._snapshot_config), and that arm no longer matches the
    prebuilt tree -- it would cold-build every view instead.
    """
    selections = set()
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        assert params["pit_views_seed"] == PIT_VIEWS_SEED, experiment_id
        assert params["include_macro"] is True, experiment_id
        assert params["include_events"] is True, experiment_id
        assert params["include_intraday"] is False, experiment_id
        assert params["macro_datasets"] == MACRO_DATASETS, experiment_id
        assert params["events_datasets"] == EVENTS_DATASETS, experiment_id
        selections.add(
            (tuple(params["macro_datasets"]), tuple(params["events_datasets"]))
        )
    assert len(selections) == 1
    # What this round selects beyond the console's default scope; naming them
    # here is what makes an accidental drop visible. The last two are this
    # round's own additions -- the roster and the analyst-forecast slice each
    # arm exists to read.
    assert {"cb_basic", "cb_daily", "cb_call"} <= set(MACRO_DATASETS)
    assert {"fut_basic", "fut_mapping", "fut_daily"} <= set(MACRO_DATASETS)
    assert {"opt_basic", "opt_daily"} <= set(MACRO_DATASETS)
    assert {"stk_surv", "top10_floatholders"} <= set(EVENTS_DATASETS)
    assert NEW_EVENTS_TABLES <= set(EVENTS_DATASETS)


def test_no_arm_moves_the_shared_calendar() -> None:
    """Both packs verified their source tables cover the whole input window, so
    unlike site_visits neither arm starts its Development later; the Folds are
    exactly the ones the shared seed was planned over."""
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        assert params["development_first_period"] == "2022Q1", experiment_id
        assert params["development_last_period"] == "2025Q4", experiment_id
        assert params["heldout_first_period"] == "20260101..20260630", experiment_id
        assert params["heldout_last_period"] == "20260101..20260630", experiment_id
        assert params["fold_period"] == "quarter", experiment_id
        assert params["validation_periods"] == 4, experiment_id


def test_the_round_states_the_heldout_own_transition_gate() -> None:
    """heldout_min_final_transitions is stated, accepted and reported.

    It equals the console default today, so the point of stating it is that the
    round manifest names the gate both packs' end-game rules are written
    against. That only works while the console still accepts it as a create
    parameter.
    """
    assert COMMON_OVERRIDES["heldout_min_final_transitions"] == 1
    assert "heldout_min_final_transitions" in WEB_CREATE_DEFAULTS
    assert "heldout_min_final_transitions" not in WEB_CLOSED_PARAMS
    assert "heldout_min_final_transitions" in REPORT_KEYS
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        assert params["heldout_min_final_transitions"] == 1, experiment_id
        assert params["heldout_min_trades"] == 20, experiment_id
        assert params["cost_stress_multiplier"] == 2.0, experiment_id


def test_no_directive_carries_a_calendar_date() -> None:
    """A literal date in fold_exploration_directive fails the worker at start.

    worker.resolve_worker_options runs the directive through
    prior_policy.calendar_policy_violation, so a date written into a directive
    is not a style issue: the arm would be refused at every start. Data windows
    that must be excluded are named by their cause and defined in the reference
    pack instead.
    """
    for experiment_id in ROUND:
        directive = str(request_params(experiment_id)["fold_exploration_directive"])
        assert directive.strip(), experiment_id
        assert calendar_policy_violation(directive) == "", experiment_id


def test_every_arm_points_at_an_existing_reference_pack() -> None:
    """A workspace_reference that does not exist fails the session at start."""
    for experiment_id in ROUND:
        reference = request_params(experiment_id)["workspace_reference"]
        assert reference, experiment_id
        assert (REPO_ROOT / str(reference)).is_dir(), reference
        assert (REPO_ROOT / str(reference) / "README.md").is_file(), reference


@needs_seed
def test_the_selection_matches_the_prebuilt_seed_and_the_preflight_accepts_it() -> None:
    """The round and the tree it hardlinks from agree, arm by arm.

    normalize runs the worker pre-flight, which refuses a seed prebuilt for a
    different snapshot configuration or under an older cache format; the
    explicit comparison above it says which half drifted when it does.
    """
    recorded = json.loads((SEED_DIR / "provider.json").read_text(encoding="utf-8"))
    datasets = recorded["snapshot_config"]["datasets"]
    assert datasets["macro"] == MACRO_DATASETS
    assert datasets["events"] == EVENTS_DATASETS
    for experiment_id in ROUND:
        merged = normalize(request_params(experiment_id))
        for role in MODEL_ROLES:
            assert merged[role] == LOCAL_QWEN_MODEL, (experiment_id, role)
        assert merged["pit_views_seed"] == PIT_VIEWS_SEED, experiment_id
