"""2026-09-10 arms: the five locally served arms put only the Meta parent on
DeepSeek-v4-flash, the ML ranker arm and the allflash control put every role
on it, only the ML ranker arm takes a GPU, and every reference pack exists."""

from __future__ import annotations

from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL
from scripts.experiments.create_round_20260910 import (
    ALL_FLASH_ROLES,
    EXPECTED_DEFAULTS,
    REPO_ROOT,
    ROUND,
    check_console_defaults,
    normalize,
    request_params,
)

FLASH = "deepseek-v4-flash"
ALLFLASH_ID = "factor_cs_allflash_20260910"
ML_ID = "ml_ranker_20260910"
HOSTED_IDS = (ML_ID, ALLFLASH_ID)
LOCAL_IDS = tuple(
    experiment_id for experiment_id in ROUND if experiment_id not in HOSTED_IDS
)
# Every model role a create request carries; a hosted arm must cover them all,
# or it opens a local stream the console's sixth slot cannot afford.
MODEL_ROLES = (
    "model",
    "meta_model",
    "subagent_model",
    "analysis_model",
    "compact_model",
    "nl_model",
)


def _assert_local_roles(params: dict[str, object]) -> None:
    assert params["meta_model"] == FLASH
    for role in MODEL_ROLES:
        if role != "meta_model":
            assert params[role] == LOCAL_QWEN_MODEL, role


def _assert_hosted_roles(params: dict[str, object]) -> None:
    for role in MODEL_ROLES:
        assert params[role] == FLASH, role


def test_local_arms_use_flash_only_for_meta() -> None:
    """Five locally served arms sit at the top of the local gateway's measured
    band; webui.manager.MAX_RUNNING_EXPERIMENTS keeps the sixth slot for an
    experiment that opens no local stream."""
    assert LOCAL_QWEN_MODEL == "qwen-3.8-27b-fp8"
    assert len(LOCAL_IDS) == 5
    assert EXPECTED_DEFAULTS["analysis_model"] == LOCAL_QWEN_MODEL
    check_console_defaults()
    for experiment_id in LOCAL_IDS:
        params = request_params(experiment_id)
        _assert_local_roles(params)
        _assert_local_roles(normalize(params))


def test_hosted_arms_put_every_role_on_flash() -> None:
    """The ML ranker arm is the sixth running arm, so every role -- not only
    the conversation ones -- is hosted; the allflash control follows the same
    mapping. The offline pre-flight must accept a hosted arm that also asks
    for a GPU: model choice and gpu_count are independent knobs."""
    assert set(ALL_FLASH_ROLES) == set(MODEL_ROLES)
    for experiment_id in HOSTED_IDS:
        params = request_params(experiment_id)
        _assert_hosted_roles(params)
        _assert_hosted_roles(normalize(params))


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
