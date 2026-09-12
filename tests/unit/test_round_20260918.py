"""The 2026-09-18 round is one arm on the 2026-09-17 round's seed.

`test_round_arms.py` already parametrises every launcher rule over every round
file, so this file only pins what this round decides for itself: one arm, no
inheritance, the shared seed and selection imported rather than retyped, and
a reference pack whose starter is a valid strategy package with the three
pre-registered legs the pack names.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.strategy_loader import validate_strategy_package
from scripts.experiments._round import REPO_ROOT
from scripts.experiments.create_round_20260917 import ROUND as SHARED
from scripts.experiments.create_round_20260918 import ROUND

ARM = "earnings_surprise_20260918"


def test_the_round_is_one_arm_without_inheritance() -> None:
    assert list(ROUND.arms) == [ARM]
    params = ROUND.request_params(ARM)
    assert params["inherit_from"] == ""
    assert params["inherit_memory_from"] == ""
    assert params["gpu_count"] == 0


def test_the_round_shares_the_previous_round_seed_and_selection() -> None:
    """A seed's identity is the whole snapshot configuration: the arm must send
    the 2026-09-17 selection byte for byte or it would cold-build every view."""
    assert ROUND.pit_views_seed == SHARED.pit_views_seed
    mine = ROUND.request_params(ARM)
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
    assert mine["include_fundamentals"] is True


def test_the_starter_is_a_valid_strategy_package_with_the_three_legs() -> None:
    pack = REPO_ROOT / str(ROUND.request_params(ARM)["workspace_reference"])
    for name in ("README.md", "families.md", "exploration-plan.md", "pit-field-map.md", "sources.md"):
        assert (pack / name).is_file(), name
    starter = pack / "starter"
    assert validate_strategy_package(starter / "main.py") is None  # no fit, no REFIT_PERIOD
    source = (starter / "main.py").read_text(encoding="utf-8")
    assert 'CANDIDATE = "s1"' in source
    surprise = (starter / "lib" / "surprise.py").read_text(encoding="utf-8")
    for leg in ('"s1"', '"c_growth"', '"c_timing"'):
        assert leg in surprise, leg
    assert not any(literal in surprise + source for literal in ("/mnt/", "/Data2", "/home/"))


def test_the_starter_never_falls_back_to_the_frozen_snapshot() -> None:
    """Reading `snapshot_dir` during a replay substitutes stale data for the
    rolling view; the pack's own rule is that an as-of read failure fails."""
    starter = Path(REPO_ROOT / str(ROUND.request_params(ARM)["workspace_reference"]) / "starter")
    for path in starter.rglob("*.py"):
        assert "snapshot_dir" not in path.read_text(encoding="utf-8"), path
