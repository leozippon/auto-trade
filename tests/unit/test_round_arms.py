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
from autotrade.pipelines.hitl_state import WEB_CLOSED_PARAMS, WEB_CREATE_DEFAULTS
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
    """A repository root holding a finished seed for every tree ``rnd`` names.

    Each tree is prebuilt for the selection of the first request naming it --
    the round's shared part first, then its arms -- so an arm that names its
    own tree gets one for its own selection, while an arm that names the
    round's tree with another selection is still refused. The contract is what
    the prebuild writes to ``provider.json``, over the one published release
    they all name, which reaches the round's Held-out; a tree needs nothing
    else for the create-time pre-flight to accept it. Returns the round's tree.
    """
    monkeypatch.setattr(_round, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_round, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    configs = {}
    for experiment_id in (PROBE_ID, *rnd.arms):
        params = rnd.request_params(experiment_id)
        configs.setdefault(str(params["pit_views_seed"]), _snapshot_config(params))
    release = publish_release(
        tmp_path,
        "synthetic",
        datasets=dict.fromkeys(
            name for config in configs.values() for name in required_release_raw_datasets(config)
        ),
    )
    for name, config in configs.items():
        seed = tmp_path / name
        seed.mkdir(parents=True)
        record = pit_cache_provider_record(
            generation_id=release.generation_id,
            release_raw_dir=release.raw_dir,
            snapshot_config=config,
        )
        (seed / "provider.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path / rnd.pit_views_seed


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
            # Arms that are independent seeds of one search share a pack.
            (tmp_path / str(reference)).mkdir(parents=True, exist_ok=True)
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


# --fill reads the arm list as a queue against the live console, so these are
# the only tests here that fake it: the health record it reads and the create
# request it sends.
FILL_ARMS = ("fill_first", "fill_second", "fill_third")


def _fill_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    running: int,
    created: tuple[str, ...] = (),
    accepted: bool = True,
) -> tuple[Round, list[str]]:
    """A three-arm round on a finished seed, with the console's two calls faked.

    ``running`` is how many of the console's four slots are in use, ``created``
    the arms whose experiment directory already exists. The returned list
    records, in order, the ids a create request was actually sent for.
    """
    rnd = Round(arms={arm: {} for arm in FILL_ARMS}, pit_views_seed="data/seed_probe")
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    for experiment_id in created:
        (tmp_path / "experiments" / experiment_id).mkdir(parents=True)
    monkeypatch.setattr(
        _round,
        "health",
        lambda port: {
            "max_running_experiments": 4,
            "running": [f"other_{index}" for index in range(running)],
        },
    )
    posted: list[str] = []

    def fake_post(port: int, params: dict[str, object]) -> bool:
        posted.append(str(params["experiment_id"]))
        return accepted

    monkeypatch.setattr(_round, "post", fake_post)
    return rnd, posted


def test_a_fill_creates_pending_arms_in_queue_order_up_to_the_free_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One free slot, one arm already created: the next pending arm takes it and
    the one behind it waits."""
    rnd, posted = _fill_round(tmp_path, monkeypatch, running=3, created=("fill_first",))
    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == ["fill_second"]
    out = capsys.readouterr().out
    assert "fill_first: created already, skipped" in out
    assert "fill_third: pending, waits for a free slot" in out


@pytest.mark.parametrize(
    ("running", "created"),
    [(4, ()), (0, FILL_ARMS)],
    ids=["no slot free", "nothing pending"],
)
def test_a_fill_with_nothing_to_do_is_the_steady_state(
    running: int, created: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode runs on a timer, so an idle fill exits 0 and sends nothing."""
    rnd, posted = _fill_round(tmp_path, monkeypatch, running=running, created=created)
    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == []


def test_a_fill_reports_a_creation_the_console_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd, posted = _fill_round(tmp_path, monkeypatch, running=2, accepted=False)
    assert rnd.main(["launcher", "0", "--fill"]) == 1
    assert posted == ["fill_first", "fill_second"]
    assert "not created: fill_first, fill_second" in capsys.readouterr().err


def test_a_fill_dry_run_plans_without_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd, posted = _fill_round(tmp_path, monkeypatch, running=3)
    assert rnd.main(["launcher", "0", "--fill", "--dry-run"]) == 0
    assert posted == []
    assert "fill_first: pending, takes a free slot" in capsys.readouterr().out


def test_a_fill_never_takes_an_experiment_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming an arm would be silently ignored: the queue decides the selection."""
    rnd, _ = _fill_round(tmp_path, monkeypatch, running=0)
    with pytest.raises(SystemExit, match="reads the queue itself"):
        rnd.main(["launcher", "0", "--fill", "fill_first"])


def test_an_arm_is_tracked_only_when_it_names_a_tracking_error_cap() -> None:
    """A round states the account and nothing about the gates: the request
    carries the optional drawdown and mandate limits as null, and a null cap
    is no mandate at any account size. An arm that wants one names the cap,
    which fills a blank beta band; drawdowns stay 0.45 / 0.30 unless named."""
    from autotrade.pipelines.config import AcceptanceRules, acceptance_for

    optional = (
        "max_drawdown",
        "active_max_drawdown",
        "tracking_error_cap",
        "beta_min",
        "beta_max",
    )
    assert all(key in WEB_CREATE_DEFAULTS and key not in BASE_OVERRIDES for key in optional)
    rnd = Round(
        arms={
            "large": {"initial_cash": 1_000_000},
            "small": {"initial_cash": 100_000},
            "tracked": {
                "initial_cash": 1_000_000,
                "tracking_error_cap": 0.08,
                "max_drawdown": 0.4,
            },
        }
    )

    def rules(experiment_id: str) -> dict[str, object]:
        return acceptance_for(rnd.request_params(experiment_id)).to_record()

    assert [rnd.request_params("large")[key] for key in optional] == [None] * len(optional)
    # The account decides nothing: two arms an order of magnitude apart, both
    # silent about the mandate, are judged by exactly the same rules.
    assert rules("large") == rules("small") == AcceptanceRules().to_record()
    tracked = rules("tracked")
    assert (tracked["tracking_error_cap"], tracked["beta_min"], tracked["beta_max"]) == (
        0.08,
        0.85,
        1.15,
    )
    assert (tracked["max_drawdown"], tracked["active_max_drawdown"]) == (0.4, 0.30)


def test_the_20260921_mandated_arms_keep_unmandated_drawdowns() -> None:
    """Those 1M arms named only a cap. Drawdowns stay 0.45 / 0.30; the cap
    still fills the default beta band."""
    from autotrade.pipelines.config import acceptance_for

    rnd = ROUNDS["create_round_20260921"]
    for experiment_id in (
        "capital_aware_1m_enhanced_20260921",
        "capital_aware_1m_riskmodel_20260921",
    ):
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.45, 0.30)
        assert rules["tracking_error_cap"] == 0.08
        assert (rules["beta_min"], rules["beta_max"]) == (0.85, 1.15)


def test_the_20260921b_arms_name_their_graduation_bars() -> None:
    """The b-round records a create-time choice for every bar, not a hidden
    pair of packages. Mandated arms keep the 35/15 drawdowns they named;
    un-mandated arms keep 45/30; statistical bars are the defaults of their day."""
    from autotrade.pipelines.config import AcceptanceRules, acceptance_for

    rnd = ROUNDS["create_round_20260921b"]
    # The defaults these arms were created under: DSR1 later raised
    # min_dsr_probability to 0.975, and a round file is the record of its day.
    defaults = {**AcceptanceRules().to_record(), "min_dsr_probability": 0.90}
    statistical = (
        "min_active_ir",
        "min_dsr_probability",
        "min_positive_year_share",
        "min_full_span_validations",
        "forward_confidence",
        "recency_months",
        "min_mean_gross",
        "min_round_trips_per_month",
        "heldout_tolerance_z",
    )
    mandated = (
        "fullcash_overlay_1m_20260921b",
        "indneutral_value_1m_20260921b",
    )
    unmandated = (
        "resid_momentum_100k_20260921b",
        "eyield_concentrated_100k_20260921b",
    )
    for experiment_id in (*mandated, *unmandated):
        request = rnd.request_params(experiment_id)
        rules = acceptance_for(request).to_record()
        for key in statistical:
            assert request[key] == defaults[key], (experiment_id, key)
            assert rules[key] == defaults[key], (experiment_id, key)
    for experiment_id in mandated:
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.35, 0.15)
        assert rules["tracking_error_cap"] == 0.08
        assert (rules["beta_min"], rules["beta_max"]) == (0.85, 1.15)
    for experiment_id in unmandated:
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.45, 0.30)
        assert rules["tracking_error_cap"] is None


def test_the_20260921c_arms_name_their_graduation_bars() -> None:
    """The c-round records a create-time choice for every bar, not a hidden
    pair of packages. Mandated arms keep the 35/15 drawdowns they named;
    un-mandated arms keep 45/30; statistical bars are the defaults of their day.
    `--fill` takes the first four and queues the last two."""
    from autotrade.pipelines.config import AcceptanceRules, acceptance_for

    rnd = ROUNDS["create_round_20260921c"]
    # The defaults these arms were created under: DSR1 later raised
    # min_dsr_probability to 0.975, and a round file is the record of its day.
    defaults = {**AcceptanceRules().to_record(), "min_dsr_probability": 0.90}
    statistical = (
        "min_active_ir",
        "min_dsr_probability",
        "min_positive_year_share",
        "min_full_span_validations",
        "forward_confidence",
        "recency_months",
        "min_mean_gross",
        "min_round_trips_per_month",
        "heldout_tolerance_z",
    )
    assert list(rnd.arms) == [
        "quality_overlay_1m_20260921c",
        "pacc_index_100k_20260921c",
        "high52_overlay_1m_20260921c",
        "lottery_reverse_100k_20260921c",
        "rmax_overlay_1m_20260921c",
        "net_issuance_100k_20260921c",
    ]
    mandated = (
        "quality_overlay_1m_20260921c",
        "high52_overlay_1m_20260921c",
        "rmax_overlay_1m_20260921c",
    )
    unmandated = (
        "pacc_index_100k_20260921c",
        "lottery_reverse_100k_20260921c",
        "net_issuance_100k_20260921c",
    )
    for experiment_id in (*mandated, *unmandated):
        request = rnd.request_params(experiment_id)
        rules = acceptance_for(request).to_record()
        for key in statistical:
            assert request[key] == defaults[key], (experiment_id, key)
            assert rules[key] == defaults[key], (experiment_id, key)
    for experiment_id in mandated:
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.35, 0.15)
        assert rules["tracking_error_cap"] == 0.08
        assert (rules["beta_min"], rules["beta_max"]) == (0.85, 1.15)
    for experiment_id in unmandated:
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.45, 0.30)
        assert rules["tracking_error_cap"] is None


def test_the_20260921d_arms_name_their_graduation_bars() -> None:
    """The d-round records a create-time choice for every bar, not a hidden
    pair of packages. Mandated arms keep the 35/15 drawdowns they named;
    un-mandated arms keep 45/30; statistical bars are the defaults of their day.
    Four arms in file order: an empty slot takes the first."""
    from autotrade.pipelines.config import AcceptanceRules, acceptance_for

    rnd = ROUNDS["create_round_20260921d"]
    # The defaults these arms were created under: DSR1 later raised
    # min_dsr_probability to 0.975, and a round file is the record of its day.
    defaults = {**AcceptanceRules().to_record(), "min_dsr_probability": 0.90}
    statistical = (
        "min_active_ir",
        "min_dsr_probability",
        "min_positive_year_share",
        "min_full_span_validations",
        "forward_confidence",
        "recency_months",
        "min_mean_gross",
        "min_round_trips_per_month",
        "heldout_tolerance_z",
    )
    assert list(rnd.arms) == [
        "gp_overlay_1m_20260921d",
        "noa_index_100k_20260921d",
        "cma_overlay_1m_20260921d",
        "cashdiv_pool_100k_20260921d",
    ]
    mandated = (
        "gp_overlay_1m_20260921d",
        "cma_overlay_1m_20260921d",
    )
    unmandated = (
        "noa_index_100k_20260921d",
        "cashdiv_pool_100k_20260921d",
    )
    for experiment_id in (*mandated, *unmandated):
        request = rnd.request_params(experiment_id)
        rules = acceptance_for(request).to_record()
        for key in statistical:
            assert request[key] == defaults[key], (experiment_id, key)
            assert rules[key] == defaults[key], (experiment_id, key)
    for experiment_id in mandated:
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.35, 0.15)
        assert rules["tracking_error_cap"] == 0.08
        assert (rules["beta_min"], rules["beta_max"]) == (0.85, 1.15)
    for experiment_id in unmandated:
        rules = acceptance_for(rnd.request_params(experiment_id)).to_record()
        assert (rules["max_drawdown"], rules["active_max_drawdown"]) == (0.45, 0.30)
        assert rules["tracking_error_cap"] is None


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


def test_an_arm_may_run_its_own_account_and_an_arm_that_states_none_inherits() -> None:
    """The account is a per-arm parameter, not only a per-round one.

    Two arms of the 2026-09-19 round mount one pack and differ in the account
    they research on, because at CNY 100k a constituent book is cost-feasible
    to about 30 names and at CNY 1M to 50. That only works while `initial_cash`
    stays an open create parameter the arm entry may override: a closed or
    unknown key would be refused before anything is sent, and an ignored one
    would run both arms on the same account while the round file claimed
    otherwise. Stated as the two relations that must hold.
    """
    assert "initial_cash" in WEB_CREATE_DEFAULTS
    assert "initial_cash" not in WEB_CLOSED_PARAMS
    rnd = Round(
        arms={"states_it": {"initial_cash": 250_000}, "states_none": {}},
        overrides={"initial_cash": 100_000},
    )
    assert rnd.request_params("states_it")["initial_cash"] == 250_000
    assert rnd.request_params("states_none")["initial_cash"] == 100_000
    accounts = {
        experiment_id: float(ROUNDS[name].request_params(experiment_id)["initial_cash"])
        for name, experiment_id in ARMS
    }
    assert all(value > 0 for value in accounts.values()), accounts
    paired = {
        experiment_id: value
        for experiment_id, value in accounts.items()
        if experiment_id.startswith("index_relative_")
    }
    # The id names the account, and the group really does run more than one.
    assert len(set(paired.values())) > 1, paired
    for experiment_id, value in paired.items():
        assert value == (100_000.0 if "_100k_" in experiment_id else 1_000_000.0), (
            experiment_id,
            value,
        )


@pytest.mark.parametrize(("round_name", "experiment_id"), ARMS)
def test_every_arm_mounts_a_checked_in_reference_pack(
    round_name: str, experiment_id: str
) -> None:
    """An arm that names a pack must name one this repository holds.

    The create-time pre-flight resolves `workspace_reference` against the
    repository root, but the dry-run test above points that root at a synthetic
    tree it populates from the round itself, so a mistyped or deleted pack
    passes every other check here and first fails when the console mounts it.
    """
    reference = ROUNDS[round_name].request_params(experiment_id).get("workspace_reference")
    if not reference:
        return
    assert (REPO_ROOT / str(reference) / "README.md").is_file(), (experiment_id, reference)


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


def test_mounting_index_weight_leaves_every_other_round_byte_for_byte() -> None:
    """A newly selectable dataset reaches exactly the round that asked for it.

    A round's snapshot configuration IS the contract its prebuilt seed and the
    arms hardlinking that tree were built under, so adding a dataset to a round
    with running arms would make their seed unusable. Stated as the relation
    that has to hold: `index_weight` reaches exactly the rounds built on the
    benchmark round's selection -- the one that introduced it and whichever
    later rounds import it -- each of their macro selections is the 2026-09-20
    one plus that name, and each record is otherwise identical to it byte for
    byte. The tree those arms actually read is compared separately, above,
    against its own provider.json.
    """
    records = {
        name: _snapshot_config(ROUNDS[name].request_params(PROBE_ID)).to_record()
        for name in ROUND_IDS
    }
    carrying = {
        name for name, record in records.items() if "index_weight" in record["datasets"]["macro"]
    }
    assert carrying == {
        "create_round_20260919",
        "create_round_20260921",
        "create_round_20260921b",
        "create_round_20260921c",
        "create_round_20260921d",
        "create_round_20260922",
        "create_round_20260923",
        "create_round_20260924",
        "create_round_20260925",
    }, sorted(carrying)
    base = records["create_round_20260920"]
    for name in sorted(carrying):
        benchmark = records[name]
        assert benchmark["datasets"]["macro"] == [*base["datasets"]["macro"], "index_weight"], name
        assert json.dumps(
            {**benchmark, "datasets": base["datasets"]}, sort_keys=True
        ) == json.dumps(base, sort_keys=True), name
    # The pin-time check reads the same selection: a round that does not select
    # the dataset must not start requiring its raw directory either.
    for name, record in records.items():
        required = required_release_raw_datasets(
            _snapshot_config(ROUNDS[name].request_params(PROBE_ID))
        )
        assert ("index_weight" in required) == (name in carrying), name
        assert "index_weight" not in record["datasets"]["events"], name


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


# The import allowlist lives in configs/agent_output_template/README.md, which
# every session mounts read-only as output/README.md. Packs used to re-copy it
# verbatim; a line naming four or more of these libraries is that copy coming
# back, and it goes stale silently the moment the contract's list moves.
ALLOWED_IMPORT_MARKERS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
    "xgboost",
    "statsmodels",
    "torch",
)


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_no_reference_pack_restates_the_import_allowlist(pack: Path) -> None:
    """A pack states the arm's deltas and defers the contract to output/README.md."""
    for path in sorted(pack.glob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            named = [marker for marker in ALLOWED_IMPORT_MARKERS if marker in line]
            assert len(named) < 4, (path.name, number, named)
