"""2026-09-10 arms: every model role of every arm is the local Qwen, the six
arms fill the console's running slots exactly, only the ML ranker arm takes a
GPU, and every reference pack exists."""

from __future__ import annotations

from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL
from autotrade.webui.manager import MAX_RUNNING_EXPERIMENTS
from scripts.experiments.create_round_20260910 import (
    COMMON_OVERRIDES,
    EXPECTED_DEFAULTS,
    REPO_ROOT,
    ROUND,
    check_console_defaults,
    normalize,
    request_params,
)

ML_ID = "ml_ranker_20260910"
# Every model role a create request carries.
MODEL_ROLES = (
    "model",
    "meta_model",
    "subagent_model",
    "analysis_model",
    "compact_model",
    "nl_model",
)


def test_every_arm_runs_every_role_on_the_local_model() -> None:
    """Cost policy: no arm may open a hosted stream.

    The round overrides no model role, so the guarantee rests entirely on the
    console defaults -- which is why EXPECTED_DEFAULTS pins all six and
    check_console_defaults refuses a drift. Asserting on request_params (what
    is POSTed) and on normalize (what the worker pre-flight accepts) checks
    the value that actually reaches params.json, not just the constant.
    """
    assert LOCAL_QWEN_MODEL == "qwen-3.8-27b-fp8"
    assert set(MODEL_ROLES).isdisjoint(COMMON_OVERRIDES)
    check_console_defaults()
    for role in MODEL_ROLES:
        assert EXPECTED_DEFAULTS[role] == LOCAL_QWEN_MODEL, role
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        merged = normalize(params)
        for role in MODEL_ROLES:
            assert params[role] == LOCAL_QWEN_MODEL, (experiment_id, role)
            assert merged[role] == LOCAL_QWEN_MODEL, (experiment_id, role)


def test_the_round_fills_the_consoles_running_slots_exactly() -> None:
    """A seventh arm could not be started beside the other six: the console
    refuses a create or resume past MAX_RUNNING_EXPERIMENTS."""
    assert len(ROUND) == MAX_RUNNING_EXPERIMENTS


def test_only_the_ml_ranker_arm_takes_a_gpu_and_every_pack_exists() -> None:
    """The GPU request travels with the experiment to the ML arm's session
    sandbox and to the strategy container of every formal replay; a
    workspace_reference that does not exist fails the session at start, so
    the packs the round points at must be in the repository."""
    for experiment_id in ROUND:
        params = request_params(experiment_id)
        assert params["gpu_count"] == (1 if experiment_id == ML_ID else 0), experiment_id
        reference = params.get("workspace_reference")
        if reference:
            assert (REPO_ROOT / reference).is_dir(), reference
            assert (REPO_ROOT / reference / "README.md").is_file(), reference
