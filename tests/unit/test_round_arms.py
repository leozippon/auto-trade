"""Every checked-in round and reference pack, read through the one launcher.

The round files are data -- arms, seed, dataset selection, directives -- and
`scripts/experiments/_round.py` is the behaviour, so these checks are written
once and parametrised over whatever round files exist. A new round file or arm
is covered the moment it is added.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
from pathlib import Path

import pytest

from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL
from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    SNAPSHOT_CACHE_FORMAT_VERSION,
)
from autotrade.pipelines.pit_backend import required_release_raw_datasets
from autotrade.pipelines.pit_views_seed import pit_cache_provider_record
from autotrade.pipelines.worker import _snapshot_config
from scripts.experiments import _round
from scripts.experiments._round import (
    BASE_EXPECTED_DEFAULTS,
    BASE_OVERRIDES,
    PROBE_ID,
    REPO_ROOT,
    RETIRED_IDS,
    Round,
    archived_ids,
)
from tests.unit.research_release_fixture import publish_release

MODEL_ROLES = ("model", "subagent_model", "nl_model", "compact_model")
# A four-digit calendar year, the shape every literal date in a directive takes.
CALENDAR_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:\d{4})?(?!\d)")
PACKS = sorted(path for path in (REPO_ROOT / "configs" / "workspace_refs").iterdir() if path.is_dir())


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


def _synthetic_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rnd: Round) -> Path:
    """A repository root holding a finished seed prebuilt for ``rnd``'s selection.

    The contract is what the prebuild writes to ``provider.json``, over the
    published release it names, which reaches the round's Held-out; the tree
    needs nothing else for the create-time pre-flight to accept it.
    """
    monkeypatch.setattr(_round, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_round, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    config = _snapshot_config(rnd.request_params(PROBE_ID))
    release = publish_release(
        tmp_path, "synthetic", datasets=required_release_raw_datasets(config)
    )
    seed = tmp_path / rnd.pit_views_seed
    seed.mkdir(parents=True)
    record = pit_cache_provider_record(
        generation_id=release.generation_id,
        release_raw_dir=release.raw_dir,
        snapshot_config=config,
    )
    (seed / "provider.json").write_text(json.dumps(record), encoding="utf-8")
    return seed


@pytest.mark.parametrize("round_name", ROUND_IDS)
def test_every_round_runs_every_model_role_on_the_local_model(round_name: str) -> None:
    """Cost policy: no arm may open a hosted stream.

    No round overrides a model role, so the guarantee rests entirely on the
    console defaults -- which is why BASE_EXPECTED_DEFAULTS pins every role and
    check_console_defaults refuses a drift.
    """
    rnd = ROUNDS[round_name]
    rnd.check_console_defaults()
    for experiment_id in (PROBE_ID, *rnd.arms):
        params = rnd.request_params(experiment_id)
        for role in MODEL_ROLES:
            assert BASE_EXPECTED_DEFAULTS[role] == LOCAL_QWEN_MODEL, role
            assert params[role] == LOCAL_QWEN_MODEL, (experiment_id, role)


def test_the_shared_geometry_is_the_one_seeds_are_planned_over() -> None:
    """The prebuild plans a seed over DEFAULT_RESEARCH_GEOMETRY; a round on any
    other geometry would cold-build every view its seed does not carry."""
    assert {key: BASE_OVERRIDES[key] for key in DEFAULT_RESEARCH_GEOMETRY.to_record()} == (
        DEFAULT_RESEARCH_GEOMETRY.to_record()
    )


@pytest.mark.parametrize("round_name", ROUND_IDS)
def test_a_round_dry_runs_against_its_seed_contract(
    round_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What `--dry-run` reports on a finished seed for this round's selection:
    the shared parameters pass the console's own pre-flight, geometry included,
    and every arm passes after them."""
    rnd = ROUNDS[round_name]
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    for arm in rnd.arms.values():
        reference = arm.get("workspace_reference")
        if reference:
            (tmp_path / str(reference)).mkdir(parents=True)
    assert rnd.main(["launcher", "0", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "release synthetic, which every arm pins" in out[0]
    report = json.loads(out[1])
    assert report["research_end"] == BASE_OVERRIDES["research_end"]
    assert report["pit_views_seed"] == rnd.pit_views_seed


def test_the_dry_run_refuses_a_seed_built_for_another_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd = Round(pit_views_seed="data/seed_probe", overrides={"text_datasets": ["report_rc"]})
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    other = Round(pit_views_seed=rnd.pit_views_seed, overrides={"text_datasets": ["anns_d"]})
    assert other.main(["launcher", "0", "--dry-run"]) == 1
    assert "different snapshot configuration" in capsys.readouterr().err


def test_the_dry_run_refuses_a_seed_whose_release_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every arm pins the release its seed was built from, so a release this
    repository does not hold is refused before anything is POSTed."""
    rnd = Round(pit_views_seed="data/seed_probe")
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    shutil.rmtree(tmp_path / "data" / "research_releases" / "synthetic")
    assert rnd.main(["launcher", "0", "--dry-run"]) == 1
    assert "research release synthetic is missing or incomplete" in capsys.readouterr().err


def test_the_dry_run_refuses_a_seed_still_being_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd = Round(pit_views_seed="data/seed_probe")
    seed = _synthetic_repo(tmp_path, monkeypatch, rnd)
    (seed / "decision" / ".20210630T235959+0800.0123abcd.tmp").mkdir(parents=True)
    assert rnd.main(["launcher", "0", "--dry-run"]) == 1
    assert "unfinished build" in capsys.readouterr().err


def test_the_dry_run_refuses_a_research_period_that_is_not_whole_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd = Round(pit_views_seed="data/seed_probe")
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    shifted = Round(pit_views_seed=rnd.pit_views_seed, overrides={"research_end": "20250331"})
    assert shifted.main(["launcher", "0", "--dry-run"]) == 1
    assert "whole July-June years" in capsys.readouterr().err


def test_a_round_without_arms_is_never_posted() -> None:
    with pytest.raises(SystemExit, match="no arms to create"):
        Round().main(["launcher", "0"])


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_every_arm_directive_is_usable(round_name: str, experiment_id: str) -> None:
    """A directive is copied into every research session of the arm and must
    hold no calendar year: data windows that must be excluded are named by
    their cause and defined in the reference pack, and no forward or Held-out
    date may reach the Agent through it."""
    directive = str(ROUNDS[round_name].request_params(experiment_id)["research_directive"])
    assert directive.strip(), experiment_id
    assert not CALENDAR_YEAR.search(directive), (experiment_id, CALENDAR_YEAR.findall(directive))


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_an_arm_requests_a_gpu_exactly_when_its_starter_needs_cuda(
    round_name: str, experiment_id: str
) -> None:
    """A starter with no CPU path fails every replay of an arm created without
    a card, and a card requested for a starter that never touches CUDA is taken
    from the shared pool for nothing."""
    params = ROUNDS[round_name].request_params(experiment_id)
    starter = REPO_ROOT / str(params.get("workspace_reference") or "") / "starter"
    needs_cuda = starter.is_dir() and any(
        "torch.cuda" in path.read_text(encoding="utf-8") for path in starter.rglob("*.py")
    )
    assert (int(params["gpu_count"]) >= 1) is needs_cuda, (experiment_id, params["gpu_count"])


@pytest.mark.parametrize("round_name", ROUND_IDS)
def test_the_selection_matches_the_prebuilt_seed(round_name: str) -> None:
    """The round and the real tree its arms hardlink from agree, byte for byte.

    The pre-flight refuses a mismatch on its own; comparing the whole snapshot
    configuration here says which half drifted when it does. The tree is
    gitignored operator state, so its absence is a wait, not a failure.
    """
    rnd = ROUNDS[round_name]
    provider = REPO_ROOT / rnd.pit_views_seed / "provider.json"
    if not rnd.pit_views_seed or not provider.is_file():
        pytest.skip(f"prebuilt PIT view seed {rnd.pit_views_seed or '(default)'} is not present")
    recorded = json.loads(provider.read_text(encoding="utf-8"))
    if recorded.get("schema_version") != SNAPSHOT_CACHE_FORMAT_VERSION:
        pytest.skip(f"{rnd.pit_views_seed} was built under another snapshot cache format")
    expected = _snapshot_config(rnd.request_params(PROBE_ID)).to_record()
    assert recorded["snapshot_config"] == expected


def test_no_round_reuses_an_experiment_id() -> None:
    """An id is never reused, by any round.

    Three sources of "already used" are checked: the other round files, the
    retired ids, and -- where the operator's archive exists -- what is actually
    archived on this machine.
    """
    seen: dict[str, str] = {}
    for round_name in ROUND_IDS:
        for experiment_id in ROUNDS[round_name].arms:
            assert experiment_id not in seen, (experiment_id, seen.get(experiment_id), round_name)
            seen[experiment_id] = round_name
    with pytest.raises(ValueError, match="already used and archived"):
        Round(arms={min(RETIRED_IDS): {}})
    archived = archived_ids()
    if not archived:
        pytest.skip("logs/archive/ is not present, so archived ids cannot be cross-checked")
    assert archived - set(seen) <= RETIRED_IDS, sorted(archived - set(seen) - RETIRED_IDS)


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_every_reference_pack_is_readable_and_its_starter_loads(pack: Path) -> None:
    """A pack mounts into every research session: it needs a README, and a
    starter is a valid strategy package that neither hard-codes a host path nor
    falls back to the frozen decision snapshot during a replay."""
    assert (pack / "README.md").is_file()
    starter = pack / "starter"
    if not (starter / "main.py").is_file():
        return
    validate_strategy_package(starter / "main.py")
    for path in starter.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(literal in text for literal in ("/mnt/", "/Data2", "/home/")), path
        assert "snapshot_dir" not in text, path
