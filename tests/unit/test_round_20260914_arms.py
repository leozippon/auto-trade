"""2026-09-14 arms: three new directions refill the three slots round 20260910
freed, every model role stays on the local Qwen, and all three ask for the one
extended dataset selection their shared PIT view seed was prebuilt for."""

from __future__ import annotations

import json

import pytest

from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL
from autotrade.webui.manager import MAX_RUNNING_EXPERIMENTS
from scripts.experiments.create_round_20260910 import ROUND as ROUND_20260910
from scripts.experiments.create_round_20260914 import (
    COMMON_OVERRIDES,
    EVENTS_DATASETS,
    EXPECTED_DEFAULTS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    REPO_ROOT,
    ROUND,
    check_console_defaults,
    normalize,
    request_params,
)

SITE_VISITS_ID = "site_visits_20260914"
# The arms of round 20260910 this round replaces. Which three keep running is
# the whole reason this round is three arms and not six.
RETIRED_20260910 = (
    "corner_cases_20260910",
    "explore_platform_strategies_20260910",
    "factor_cs_20260910",
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

SEED_DIR = REPO_ROOT / PIT_VIEWS_SEED
# The seed tree is a gitignored operator artifact built by
# scripts/data/prebuild_pit_views_seed.py, so the checks that read it are
# skipped where it was never built rather than failing a fresh checkout.
needs_seed = pytest.mark.skipif(
    not (SEED_DIR / "provider.json").is_file(),
    reason=f"prebuilt PIT view seed {PIT_VIEWS_SEED} is not present",
)


def test_the_three_new_arms_refill_the_console_exactly() -> None:
    """Three new arms beside round 20260910's three survivors fill the console.

    A fourth arm here could not be started: the console refuses a create or
    resume past MAX_RUNNING_EXPERIMENTS.
    """
    assert len(ROUND) == 3
    assert set(RETIRED_20260910) < set(ROUND_20260910)
    assert set(ROUND).isdisjoint(ROUND_20260910)
    surviving = set(ROUND_20260910) - set(RETIRED_20260910)
    assert len(ROUND) + len(surviving) == MAX_RUNNING_EXPERIMENTS


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
    # What this round adds beyond the console's default scope; naming them here
    # is what makes an accidental drop visible.
    assert {"cb_basic", "cb_daily", "cb_call"} <= set(MACRO_DATASETS)
    assert {"fut_basic", "fut_mapping", "fut_daily"} <= set(MACRO_DATASETS)
    assert {"opt_basic", "opt_daily"} <= set(MACRO_DATASETS)
    assert {"stk_surv", "top10_floatholders"} <= set(EVENTS_DATASETS)


def test_only_site_visits_starts_its_development_later() -> None:
    """stk_surv starts two years into the lake, so that arm's first fully
    covered validation window is later than the round's; its Folds stay a
    subset of the shared seed's plan, which is why the seed still applies."""
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        expected = "2024Q1" if experiment_id == SITE_VISITS_ID else "2022Q1"
        assert params["development_first_period"] == expected, experiment_id
        assert params["development_last_period"] == "2025Q4", experiment_id
        assert params["fold_period"] == "quarter", experiment_id
        assert params["validation_periods"] == 4, experiment_id
    assert COMMON_OVERRIDES["development_first_period"] == "2022Q1"


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
    different snapshot configuration; the explicit comparison above it says
    which half drifted when it does.
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
