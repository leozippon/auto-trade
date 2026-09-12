"""The 2026-09-18 round is two arms on the 2026-09-17 round's seed.

`test_round_arms.py` already parametrises every launcher rule over every round
file, so this file only pins what this round decides for itself: two arms
created one at a time within the console's running cap, no inheritance, the
shared seed and selection imported rather than retyped, and reference packs
whose starters are valid strategy packages with the pre-registered legs the
packs name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.webui.manager import MAX_RUNNING_EXPERIMENTS
from scripts.experiments._round import REPO_ROOT
from scripts.experiments.create_round_20260917 import ROUND as SHARED
from scripts.experiments.create_round_20260918 import ROUND

EARNINGS = "earnings_surprise_20260918"
DEFENSIVE = "defensive_quality_20260918"
ARMS = (EARNINGS, DEFENSIVE)
PACK_FILES = ("README.md", "families.md", "exploration-plan.md", "pit-field-map.md", "sources.md")
# Every pre-registered leg each pack's starter must carry, by the module that names them.
LEGS = {
    EARNINGS: ("lib/surprise.py", ('"s1"', '"c_growth"', '"c_timing"')),
    DEFENSIVE: ("lib/face.py", ('"s1"', '"s_cfq"', '"s_lowvol"', '"c_vol20"', '"c_growth"', '"c_es"')),
}


def _starter(arm: str) -> Path:
    return Path(REPO_ROOT / str(ROUND.request_params(arm)["workspace_reference"]) / "starter")


def test_the_round_is_two_arms_within_the_running_cap_without_inheritance() -> None:
    """The arms fill slots one at a time (the launcher creates only the ids it
    is given), but the round as a whole must still fit the console's cap."""
    assert list(ROUND.arms) == list(ARMS)
    assert len(ROUND.arms) <= MAX_RUNNING_EXPERIMENTS == 4
    for arm in ARMS:
        params = ROUND.request_params(arm)
        assert params["inherit_from"] == ""
        assert params["inherit_memory_from"] == ""
        assert params["gpu_count"] == 0


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_shares_the_previous_round_seed_and_selection(arm: str) -> None:
    """A seed's identity is the whole snapshot configuration: each arm must send
    the 2026-09-17 selection byte for byte or it would cold-build every view."""
    assert ROUND.pit_views_seed == SHARED.pit_views_seed
    mine = ROUND.request_params(arm)
    theirs = SHARED.request_params(next(iter(SHARED.arms)))
    for key in (
        "macro_datasets",
        "events_datasets",
        "include_intraday",
        "development_first_period",
        "development_last_period",
        "heldout_first_period",
        "heldout_last_period",
        "deployment_adjustment_start",
        "window_months",
    ):
        assert mine[key] == theirs[key], key
    assert "report_rc" in mine["events_datasets"]
    assert "index_daily" in mine["macro_datasets"]
    assert mine["include_fundamentals"] is True


@pytest.mark.parametrize("arm", ARMS)
def test_every_starter_is_a_valid_strategy_package_with_its_legs(arm: str) -> None:
    pack = _starter(arm).parent
    for name in PACK_FILES:
        assert (pack / name).is_file(), name
    starter = _starter(arm)
    assert validate_strategy_package(starter / "main.py") is None  # no fit, no REFIT_PERIOD
    source = (starter / "main.py").read_text(encoding="utf-8")
    assert 'CANDIDATE = "s1"' in source
    module, legs = LEGS[arm]
    text = (starter / module).read_text(encoding="utf-8")
    for leg in legs:
        assert leg in text, leg
    assert not any(literal in text + source for literal in ("/mnt/", "/Data2", "/home/"))


@pytest.mark.parametrize("arm", ARMS)
def test_no_starter_falls_back_to_the_frozen_snapshot(arm: str) -> None:
    """Reading `snapshot_dir` during a replay substitutes stale data for the
    rolling view; both packs' own rule is that an as-of read failure fails."""
    for path in _starter(arm).rglob("*.py"):
        assert "snapshot_dir" not in path.read_text(encoding="utf-8"), path


def test_the_defensive_controls_are_not_neutralized_on_their_own_face() -> None:
    """`c_vol20` scored on -vol_20 and then residualized on vol_20 would be
    noise, and the same for `c_growth` on yoy_np; the candidates keep the
    full set."""
    text = (_starter(DEFENSIVE) / "lib" / "face.py").read_text(encoding="utf-8")
    assert 'CONTROL_EXCLUDES = {"vol20": ("vol_20", "max_20"), "growth": ("yoy_np",)}' in text
    assert '"vol_20", "max_20", "ep", "yoy_np"]' in text  # the full neutralization set stays declared
