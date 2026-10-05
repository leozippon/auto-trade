"""Every open round and the reference packs its arms mount, read through the one launcher.

The round files are data -- arms, seed, dataset selection, directives -- and
`scripts/experiments/_round.py` is the behaviour, so these checks are written
once and parametrised over whatever open round files exist. A new round file or
arm is covered the moment it is added. A closed round is the record of arms
that were created and is only imported: it must refuse to run, and its ids stay
used.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import shutil
from pathlib import Path

import pytest

from autotrade.environment.data.contracts import BENCHMARK_INDEXES
from autotrade.environment.llm.model_profiles import LOCAL_QWEN_MODEL, MIMO_FLASH_MODEL
from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    SNAPSHOT_CACHE_FORMAT_VERSION,
)
from autotrade.pipelines.hitl_state import (
    CREATION_STAMPS,
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
)
from autotrade.pipelines.pit_backend import required_release_raw_datasets
from autotrade.pipelines.pit_views_seed import pit_cache_provider_record
from autotrade.pipelines.worker import _snapshot_config
from autotrade.webui.manager import (
    MAX_RUNNING_EXPERIMENTS,
    MAX_RUNNING_LOCAL_EXPERIMENTS,
)
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
from tests.unit.research_release_fixture import (
    BACKFILLED_HISTORY_START,
    publish_release,
)

MODEL_ROLES = ("model", "subagent_model", "nl_model", "compact_model")
# A four-digit calendar year, the shape every literal date in a directive takes.
CALENDAR_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:\d{4})?(?!\d)")
ROUND_FILES = sorted((REPO_ROOT / "scripts" / "experiments").glob("create_round_*.py"))


def _rounds() -> dict[str, Round]:
    """Every checked-in round file, closed ones included, by module name."""
    assert ROUND_FILES, "no round definitions found"
    return {
        path.stem: importlib.import_module(f"scripts.experiments.{path.stem}").ROUND
        for path in ROUND_FILES
    }


ROUNDS = _rounds()
ROUND_IDS = sorted(name for name, rnd in ROUNDS.items() if not rnd.closed)
ARMS = [(name, arm) for name in ROUND_IDS for arm in ROUNDS[name].arms]
# The packs the open rounds' arms mount: every research attempt copies its
# pack from the repository, so these are the packs a session can still read.
PACKS = sorted(
    {
        REPO_ROOT / str(reference)
        for name, arm in ARMS
        if (reference := ROUNDS[name].request_params(arm)["workspace_reference"])
    }
)


def _synthetic_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rnd: Round) -> Path:
    """A repository root holding a finished seed for every tree ``rnd`` names.

    Each tree is prebuilt for the selection of the first request naming it --
    the round's shared part first, then its arms -- so an arm that names its
    own tree gets one for its own selection, while an arm that names the
    round's tree with another selection is still refused. The contract is what
    the prebuild writes to ``provider.json``, over the one published release
    they all name, which reaches the round's Held-out and holds benchmark
    history back to the backfill floor, as the lake now does; a tree needs
    nothing else for the create-time pre-flight to accept it. Every arm's
    reference pack exists, as the repository holds them. Returns the round's
    tree.
    """
    monkeypatch.setattr(_round, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_round, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    configs = {}
    for experiment_id in (PROBE_ID, *rnd.arms):
        params = rnd.request_params(experiment_id)
        configs.setdefault(str(params["pit_views_seed"]), _snapshot_config(params))
        if params["workspace_reference"]:
            # Arms that are independent seeds of one search share a pack.
            (tmp_path / str(params["workspace_reference"])).mkdir(parents=True, exist_ok=True)
    release = publish_release(
        tmp_path,
        "synthetic",
        datasets=dict.fromkeys(
            name for config in configs.values() for name in required_release_raw_datasets(config)
        ),
        history_start=BACKFILLED_HISTORY_START,
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
def test_every_round_runs_on_the_local_model_unless_an_arm_declares_a_hosted_role(round_name: str) -> None:
    """Cost policy: an arm opens a hosted stream only where its own entry says so.

    The console defaults pin every role to the local model (BASE_EXPECTED_DEFAULTS,
    which check_console_defaults guards against drift), so a hosted role is always
    an arm's explicit declaration in its round file. Text-evidence scoring is never
    hosted: it also runs in the forward and Held-out replays, whose data must not
    leave the machine.
    """
    rnd = ROUNDS[round_name]
    rnd.check_console_defaults()
    for experiment_id in (PROBE_ID, *rnd.arms):
        params = rnd.request_params(experiment_id)
        declared = rnd.arms.get(experiment_id, {})
        for role in MODEL_ROLES:
            assert BASE_EXPECTED_DEFAULTS[role] == LOCAL_QWEN_MODEL, role
            if role in declared and role != "nl_model":
                continue
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
    and every arm passes after them. Each lineage arm a round names exists, as
    the experiments it continues do."""
    from tests.unit.test_lineage import _arm

    rnd = ROUNDS[round_name]
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    for experiment_id in rnd.arms:
        params = rnd.request_params(experiment_id)
        for seed, name in enumerate(params.get("lineage_arms") or ()):
            if not (tmp_path / "experiments" / name).exists():
                research = (str(params["research_start"]), str(params["research_end"]))
                _arm(tmp_path / "experiments", name, [{"seed": seed, "loading": 0.5}], research=research)
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


# The eight-year research period the backfilled lake opens: research from
# 2017-07 on a 108-month window of index and Shenwan data alone, since the
# fundamental, event and text history starts inside that window.
EIGHT_YEAR = {
    "research_start": "20170701",
    "window_months": 108,
    "include_fundamentals": False,
    "include_events": False,
    "include_text": False,
    "macro_datasets": ["index_daily", "index_dailybasic", "sw_daily", "index_weight"],
}
STAR_50 = "000688.SH"


def test_an_eight_year_round_dry_runs_on_the_backfilled_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every benchmark but STAR 50 has history before 2017-07 in the backfilled
    lake -- CSI 1000's constituents from their first section in 2014-10 -- so
    an eight-year arm on any of them passes the create pre-flight."""
    rnd = Round(
        arms={
            f"eight_year_{code[:6]}": {"benchmark_index": code}
            for code in BENCHMARK_INDEXES
            if code != STAR_50
        },
        overrides=EIGHT_YEAR,
        pit_views_seed="data/seed_probe_8y",
    )
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    assert rnd.main(["launcher", "0", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out.splitlines()[1])
    assert (report["research_start"], report["window_months"]) == ("20170701", 108)


def test_an_eight_year_round_refuses_star_50_and_fundamentals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """STAR 50's history starts at its 2019-12-31 base date, inside the research
    period, so its arm is refused naming how far back each table goes. An arm
    adding fundamentals is refused here by the eight-year seed's contract; the
    missing fundamental history itself (no PIT partition before the first
    decision view) is refused only when that view is built, which no dry-run
    reaches."""
    rnd = Round(
        arms={
            "eight_year_star50": {"benchmark_index": STAR_50},
            "eight_year_fundamentals": {"benchmark_index": "000905.SH", "include_fundamentals": True},
        },
        overrides=EIGHT_YEAR,
        pit_views_seed="data/seed_probe_8y",
    )
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    assert rnd.main(["launcher", "0", "--dry-run"]) == 1
    err = capsys.readouterr().err.splitlines()
    reasons = {line.split(":", 1)[0]: line for line in err if line.startswith("eight_year_")}
    assert "benchmark_index 000688.SH (科创50) has no history before research_start 20170701" in (
        reasons["eight_year_star50"]
    )
    assert (
        "{'index_daily/ts_code=000688.SH': '20191231', 'index_weight/index_code=000688.SH': '20200731'}"
        in reasons["eight_year_star50"]
    )
    assert "different snapshot configuration" in reasons["eight_year_fundamentals"]
    assert err[-1] == "refused: eight_year_star50, eight_year_fundamentals"


def test_an_arm_mounts_the_pack_named_after_it_unless_it_names_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An arm may leave out its pack when the pack is configs/workspace_refs/<id>.

    Every checked-in arm that names exactly that path sends the same create
    request byte for byte without it; an arm or a round that names another
    pack, or "" for none, keeps it; and a default pack that is not there is
    refused like a named one."""
    rnd = Round(
        arms={
            "conventional": {},
            "elsewhere": {"workspace_reference": "configs/workspace_refs/shared"},
            "packless": {"workspace_reference": ""},
        },
        pit_views_seed="data/seed_probe",
    )
    assert {arm: rnd.request_params(arm)["workspace_reference"] for arm in rnd.arms} == {
        "conventional": "configs/workspace_refs/conventional",
        "elsewhere": "configs/workspace_refs/shared",
        "packless": "",
    }
    assert rnd.request_params(PROBE_ID)["workspace_reference"] == WEB_CREATE_DEFAULTS["workspace_reference"]
    shared = Round(arms={"seed_a": {}}, overrides={"workspace_reference": "configs/workspace_refs/shared"})
    assert shared.request_params("seed_a")["workspace_reference"] == "configs/workspace_refs/shared"
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    (tmp_path / "configs/workspace_refs/conventional").rmdir()
    assert rnd.main(["launcher", "0", "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "conventional: parameters rejected" in err and "workspace_reference" in err
    assert err.splitlines()[-1] == "refused: conventional"


def test_a_dry_run_reads_each_lineage_and_refuses_one_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An arm names its lineage as a list of experiment ids. The dry-run reads
    those arms through the console's own pre-flight and prints what they add
    to the arm's freeze gate; a lineage arm that was never created refuses the
    arm that names it, before anything is sent."""
    from tests.unit.test_lineage import DAYS, _arm

    rnd = Round(
        arms={
            "heir": {"lineage_arms": ["earlier_one", "earlier_two"]},
            "orphan": {"lineage_arms": ["never_created"]},
        },
        pit_views_seed="data/seed_probe",
    )
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    for name, seed in (("earlier_one", 1), ("earlier_two", 2)):
        _arm(
            tmp_path / "experiments",
            name,
            [{"seed": seed, "loading": 0.5, "control": True}, {"seed": seed + 10, "loading": 0.5}],
        )
    assert rnd.main(["launcher", "0", "--dry-run"]) == 1
    captured = capsys.readouterr()
    [line] = [line for line in captured.out.splitlines() if line.startswith("  lineage: ")]
    reading = json.loads(line.removeprefix("  lineage: "))
    assert '"lineage_arms": ["earlier_one", "earlier_two"]' in captured.out
    assert (reading["trials"], reading["host_trials"], reading["controls"]) == (2, 2, 2)
    assert 1.0 < reading["effective_trials"] <= 2.0
    assert reading["bar_days"] == len(DAYS)
    assert reading["information_ratio_bar_floor"] > 0.98
    assert "orphan: parameters rejected, nothing was sent: lineage arm never_created does not exist" in (
        captured.err
    )
    assert captured.err.splitlines()[-1] == "refused: orphan"


def test_a_round_without_arms_is_never_posted() -> None:
    with pytest.raises(SystemExit, match="no arms to create"):
        Round().main(["launcher", "0"])


def test_a_closed_round_refuses_every_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every arm of a closed round was created, and archiving an arm deletes
    its directory, so a mode that reads the queue would create it again. The
    round refuses before it reads the console or the seed."""

    def unreachable(*args: object) -> object:
        raise AssertionError("a closed round reached the console")

    monkeypatch.setattr(_round, "health", unreachable)
    monkeypatch.setattr(_round, "post", unreachable)
    rnd = Round(arms={"done": {}}, closed=True)
    for flags in ([], ["done"], ["--dry-run"], ["--fill"], ["--fill", "--dry-run"]):
        with pytest.raises(SystemExit, match="this round is closed"):
            rnd.main(["launcher", "0", *flags])


def test_an_open_round_imports_no_other_round() -> None:
    """What rounds share lives in `_profiles.py`, so a round can close, and stay
    the record of its arms, without an open round reading its values from it."""
    for name in ROUND_IDS:
        source = (REPO_ROOT / "scripts" / "experiments" / f"{name}.py").read_text(encoding="utf-8")
        modules = {
            node.module
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not [module for module in modules if "create_round_" in module], (name, sorted(modules))


def test_a_change_of_the_rule_set_stops_the_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """A round's queued arms are created days after its first ones, with the
    rules the console then stamps on a new arm. Every rule it stamps is pinned,
    so turning one off, or stamping one nobody pinned, stops a round that has
    not decided that rule for itself."""
    Round().check_console_defaults()
    assert set(CREATION_STAMPS) <= set(BASE_EXPECTED_DEFAULTS)

    monkeypatch.setitem(_round.WEB_CREATE_DEFAULTS, "require_seed_replicates", False)
    drift = '"require_seed_replicates": {"round": "True", "console": "False"}'
    with pytest.raises(SystemExit, match=re.escape(drift)):
        Round().check_console_defaults()
    Round(overrides={"require_seed_replicates": False}).check_console_defaults()

    monkeypatch.setitem(_round.WEB_CREATE_DEFAULTS, "require_seed_replicates", True)
    monkeypatch.delitem(_round.BASE_EXPECTED_DEFAULTS, "require_seed_replicates")
    undecided = '"require_seed_replicates": {"round": "undecided", "console": "True"}'
    with pytest.raises(SystemExit, match=re.escape(undecided)):
        Round().check_console_defaults()
    Round(overrides={"require_seed_replicates": True}).check_console_defaults()


# --fill reads the arm list as a queue against the live console, so these are
# the only tests here that fake it: the health record it reads and the create
# request it sends.
FILL_ARMS = ("fill_first", "fill_second", "fill_third")
# The one line a timer run logs: its local time with offset, then the counts.
SUMMARY = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} fill launcher: (.*)$")


def _fill_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    local: int,
    hosted: int = 0,
    created: tuple[str, ...] = (),
    accepted: bool = True,
    arms: dict[str, dict[str, object]] | None = None,
) -> tuple[Round, list[str]]:
    """A three-arm round on a finished seed, with the console's two calls faked.

    ``local`` and ``hosted`` are how many arms of each kind the console is
    running against its two limits (it has no free GPU), ``created`` the arms
    whose experiment directory already exists, ``arms`` what each arm decides
    for itself (an arm that names no model role is local, one that names no
    ``gpu_count`` runs on CPU like the round). The returned list records, in
    order, the ids a create request was actually sent for.
    """
    rnd = Round(arms=arms or {arm: {} for arm in FILL_ARMS}, pit_views_seed="data/seed_probe")
    _synthetic_repo(tmp_path, monkeypatch, rnd)
    for experiment_id in created:
        (tmp_path / "experiments" / experiment_id).mkdir(parents=True)
    running_local = [f"local_{index}" for index in range(local)]
    monkeypatch.setattr(
        _round,
        "health",
        lambda port: {
            "max_running_experiments": MAX_RUNNING_EXPERIMENTS,
            "max_running_local_experiments": MAX_RUNNING_LOCAL_EXPERIMENTS,
            "running": running_local + [f"hosted_{index}" for index in range(hosted)],
            "running_local": running_local,
            "gpus_free": [],
        },
    )
    posted: list[str] = []

    def fake_post(port: int, params: dict[str, object]) -> bool:
        posted.append(str(params["experiment_id"]))
        return accepted

    monkeypatch.setattr(_round, "post", fake_post)
    return rnd, posted


def _summary(out: str) -> str:
    """The run's summary line, which must be the only line a timer run logs
    on stdout when the console's own create lines are faked away."""
    lines = out.splitlines()
    assert len(lines) == 1, lines
    match = SUMMARY.match(lines[0])
    assert match, lines[0]
    return match.group(1)


def test_a_fill_creates_pending_arms_in_queue_order_up_to_the_free_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One free slot, one arm already created: the next pending arm takes it and
    the one behind it waits, and the run logs one line saying so."""
    rnd, posted = _fill_round(
        tmp_path, monkeypatch, local=2, hosted=MAX_RUNNING_EXPERIMENTS - 3, created=("fill_first",)
    )
    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == ["fill_second"]
    hosted = ", ".join(f"hosted_{index}" for index in range(MAX_RUNNING_EXPERIMENTS - 3))
    assert _summary(capsys.readouterr().out) == (
        f"slots {MAX_RUNNING_EXPERIMENTS - 1}/{MAX_RUNNING_EXPERIMENTS} in use "
        f"({hosted}, local_0, local_1), 1 free; "
        f"local-model slots 2/{MAX_RUNNING_LOCAL_EXPERIMENTS} in use (local_0, local_1), "
        f"{MAX_RUNNING_LOCAL_EXPERIMENTS - 2} free; GPUs 0 free (none); queue 3: "
        "1 skipped (created already), 1 created, 0 refused, 1 pending "
        "(0 held by the local-model limit, 0 by the free GPUs)"
    )


@pytest.mark.parametrize(
    ("local", "hosted", "created", "counts"),
    [
        (
            MAX_RUNNING_LOCAL_EXPERIMENTS,
            MAX_RUNNING_EXPERIMENTS - MAX_RUNNING_LOCAL_EXPERIMENTS,
            (),
            (
                "0 skipped (created already), 0 created, 0 refused, 3 pending "
                "(0 held by the local-model limit, 0 by the free GPUs)"
            ),
        ),
        (
            MAX_RUNNING_LOCAL_EXPERIMENTS,
            0,
            (),
            (
                "0 skipped (created already), 0 created, 0 refused, 3 pending "
                "(3 held by the local-model limit, 0 by the free GPUs)"
            ),
        ),
        (
            0,
            0,
            FILL_ARMS,
            (
                "3 skipped (created already), 0 created, 0 refused, 0 pending "
                "(0 held by the local-model limit, 0 by the free GPUs)"
            ),
        ),
    ],
    ids=["no slot free", "no local slot for a local queue", "nothing pending"],
)
def test_a_fill_with_nothing_to_do_is_the_steady_state(
    local: int,
    hosted: int,
    created: tuple[str, ...],
    counts: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mode runs on a timer, so an idle fill exits 0, sends nothing and
    logs one line however long the queue behind it is."""
    rnd, posted = _fill_round(tmp_path, monkeypatch, local=local, hosted=hosted, created=created)
    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == []
    assert _summary(capsys.readouterr().out).endswith(counts)


# Both session roles hosted: the arm does not count against the local limit.
HOSTED_ROLES = {"model": MIMO_FLASH_MODEL, "subagent_model": MIMO_FLASH_MODEL}


def test_a_fill_holds_a_local_arm_at_the_local_limit_and_creates_the_hosted_one_behind_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The local arms at the head and the tail of the queue stay pending, the
    hosted arm between them takes a free slot, and holding is the steady
    state, not a refusal."""
    arms = {arm: {} for arm in FILL_ARMS}
    arms["fill_second"] = dict(HOSTED_ROLES)
    rnd, posted = _fill_round(tmp_path, monkeypatch, local=MAX_RUNNING_LOCAL_EXPERIMENTS, arms=arms)
    assert rnd.main(["launcher", "0", "--fill", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "fill_first: pending, waits for a local-model slot" in out
    assert "fill_second: pending, takes a free slot" in out
    assert "fill_third: pending, waits for a local-model slot" in out
    assert posted == []

    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == ["fill_second"]
    assert _summary(capsys.readouterr().out).endswith(
        f"local-model slots {MAX_RUNNING_LOCAL_EXPERIMENTS}/{MAX_RUNNING_LOCAL_EXPERIMENTS} in use "
        f"({', '.join(f'local_{index}' for index in range(MAX_RUNNING_LOCAL_EXPERIMENTS))}), 0 free; "
        "GPUs 0 free (none); queue 3: 0 skipped (created already), 1 created, 0 refused, "
        "2 pending (2 held by the local-model limit, 0 by the free GPUs)"
    )


def test_a_fill_holds_a_gpu_arm_while_no_card_is_free_and_creates_the_cpu_arm_behind_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With every card in use or held, the GPU arms stay pending and the CPU
    arm between them is created; one free card takes the first GPU arm only,
    since the console hands a card to one arm at a time."""
    arms = {arm: {"gpu_count": 1} for arm in FILL_ARMS}
    arms["fill_second"] = {}
    rnd, posted = _fill_round(tmp_path, monkeypatch, local=0, arms=arms)
    assert rnd.main(["launcher", "0", "--fill", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "fill_first: pending, waits for a free GPU" in out
    assert "fill_second: pending, takes a free slot" in out
    assert "fill_third: pending, waits for a free GPU" in out
    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == ["fill_second"]
    assert _summary(capsys.readouterr().out).endswith(
        "GPUs 0 free (none); queue 3: 0 skipped (created already), 1 created, 0 refused, "
        "2 pending (0 held by the local-model limit, 2 by the free GPUs)"
    )

    # Nothing was really created (the POST is faked), so the queue is unchanged.
    no_card = _round.health(0)
    monkeypatch.setattr(_round, "health", lambda port: {**no_card, "gpus_free": [5]})
    posted.clear()
    assert rnd.main(["launcher", "0", "--fill"]) == 0
    assert posted == ["fill_first", "fill_second"]
    assert _summary(capsys.readouterr().out).endswith(
        "GPUs 1 free (5); queue 3: 0 skipped (created already), 2 created, 0 refused, "
        "1 pending (0 held by the local-model limit, 1 by the free GPUs)"
    )


def test_a_fill_past_a_held_local_arm_still_stops_at_any_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the local limit holds an arm back; the pre-flight refusing the
    hosted arm behind it stops the run as before, with nothing sent after it."""
    arms = {arm: dict(HOSTED_ROLES) for arm in FILL_ARMS}
    arms["fill_first"] = {}
    arms["fill_second"] = {**HOSTED_ROLES, "research_end": "20250331"}
    rnd, posted = _fill_round(tmp_path, monkeypatch, local=MAX_RUNNING_LOCAL_EXPERIMENTS, arms=arms)
    assert rnd.main(["launcher", "0", "--fill"]) == 1
    assert posted == []
    captured = capsys.readouterr()
    assert _summary(captured.out).endswith(
        "0 created, 1 refused, 2 pending (1 held by the local-model limit, 0 by the free GPUs)"
    )
    assert "fill_second: parameters rejected" in captured.err


def test_a_fill_reports_a_creation_the_console_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd, posted = _fill_round(
        tmp_path, monkeypatch, local=2, hosted=MAX_RUNNING_EXPERIMENTS - 4, accepted=False
    )
    assert rnd.main(["launcher", "0", "--fill"]) == 1
    assert posted == ["fill_first", "fill_second"]
    captured = capsys.readouterr()
    assert _summary(captured.out).endswith(
        "0 created, 2 refused, 1 pending (0 held by the local-model limit, 0 by the free GPUs)"
    )
    assert "not created: fill_first, fill_second" in captured.err


def test_a_fill_stops_at_an_arm_the_preflight_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is sent for the refused arm or after it, the reason and the
    summary still reach the log, and the run exits non-zero."""
    arms = {arm: {} for arm in FILL_ARMS}
    arms["fill_second"] = {"research_end": "20250331"}
    rnd, posted = _fill_round(tmp_path, monkeypatch, local=0, arms=arms)
    assert rnd.main(["launcher", "0", "--fill"]) == 1
    assert posted == ["fill_first"]
    captured = capsys.readouterr()
    assert _summary(captured.out).endswith(
        "1 created, 1 refused, 1 pending (0 held by the local-model limit, 0 by the free GPUs)"
    )
    assert "fill_second: parameters rejected" in captured.err
    assert "whole July-June years" in captured.err


def test_a_fill_dry_run_plans_without_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rnd, posted = _fill_round(tmp_path, monkeypatch, local=0, hosted=MAX_RUNNING_EXPERIMENTS - 1)
    assert rnd.main(["launcher", "0", "--fill", "--dry-run"]) == 0
    assert posted == []
    out = capsys.readouterr().out.splitlines()
    assert "fill_first: pending, takes a free slot" in out
    assert "fill_third: pending, waits for a free slot" in out
    match = SUMMARY.match(out[-1])
    assert match, out[-1]
    assert match.group(1).endswith(
        "0 skipped (created already), 1 would be created, 0 refused, 2 pending "
        "(0 held by the local-model limit, 0 by the free GPUs)"
    )


def test_a_fill_never_takes_an_experiment_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming an arm would be silently ignored: the queue decides the selection."""
    rnd, _ = _fill_round(tmp_path, monkeypatch, local=0)
    with pytest.raises(SystemExit, match="reads the queue itself"):
        rnd.main(["launcher", "0", "--fill", "fill_first"])


def test_an_arm_is_tracked_only_when_it_names_a_tracking_error_cap() -> None:
    """A round states the account and nothing about the gates: the request
    carries the optional drawdown and mandate limits as null, and a null cap
    is no mandate at any account size. An arm that wants one names the cap,
    which fills a blank beta band; drawdowns stay 0.45 / 0.30 unless named."""
    from autotrade.pipelines.config import acceptance_for

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
    # A new arm is also held to every rule creation stamps on, which the
    # rules themselves leave off for arms recorded without them.
    assert rules("large") == rules("small") == acceptance_for(CREATION_STAMPS).to_record()
    tracked = rules("tracked")
    assert (tracked["tracking_error_cap"], tracked["beta_min"], tracked["beta_max"]) == (
        0.08,
        0.85,
        1.15,
    )
    assert (tracked["max_drawdown"], tracked["active_max_drawdown"]) == (0.4, 0.30)


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

    Arms that mount one pack may differ in the account they research on,
    because at CNY 100k a constituent book is cost-feasible to about 30 names
    and at CNY 1M to 50. That only works while `initial_cash` stays an open
    create parameter the arm entry may override: a closed or unknown key would
    be refused before anything is sent, and an ignored one would run both arms
    on the same account while the round file claimed otherwise.
    """
    assert "initial_cash" in WEB_CREATE_DEFAULTS
    assert "initial_cash" not in WEB_CLOSED_PARAMS
    rnd = Round(
        arms={"states_it": {"initial_cash": 250_000}, "states_none": {}},
        overrides={"initial_cash": 100_000},
    )
    assert rnd.request_params("states_it")["initial_cash"] == 250_000
    assert rnd.request_params("states_none")["initial_cash"] == 100_000


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


def test_no_round_reuses_an_experiment_id() -> None:
    """An id is never reused, by any round.

    Three sources of "already used" are checked: the other round files, closed
    ones included, the retired ids, and -- where the operator's archive
    exists -- what is actually archived on this machine.
    """
    seen: dict[str, str] = {}
    for round_name in sorted(ROUNDS):
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


# Forward and Held-out results never enter a session (agent-design §1.2), yet in
# round 20260926 other arms' forward readings reached refs/ through the packs'
# closed-direction and prior tables: the tables whose header row names a
# closure (关闭) or the register (登记册). Those tables cite the register row
# and research-period readings only.
FORWARD_PERIOD_MARKERS = ("前推", "held-out", "forward ir", "f2 下界")


def _forward_period_table_rows(pack: Path) -> list[str]:
    """Every closed-direction or prior table row in ``pack`` naming the forward
    period or Held-out, as ``file:line: row``."""
    rows: list[str] = []
    for path in sorted(pack.rglob("*.md")):
        in_scope = False
        previous = ""
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = line.strip()
            if row.startswith("|"):
                if not previous.startswith("|"):
                    in_scope = "关闭" in row or "登记册" in row
                if in_scope and any(marker in row.lower() for marker in FORWARD_PERIOD_MARKERS):
                    rows.append(f"{path.relative_to(pack)}:{number}: {row}")
            previous = row
    return rows


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_no_pack_table_quotes_the_forward_period(pack: Path) -> None:
    """A pack quotes no forward or Held-out reading."""
    rows = _forward_period_table_rows(pack)
    assert not rows, f"{pack.name} quotes the forward period:\n" + "\n".join(rows)


# Round 20260927 copied the book into 225 starters in 38 variants, so a fixed
# defect (a full book buying a 13th seat) kept shipping in the copies for 70
# hours. A starter's book is the canonical file itself.
STARTER_LIB = REPO_ROOT / "configs" / "starter_lib"
CANONICAL_BOOK = {
    path.name: path.read_bytes()
    for path in STARTER_LIB.iterdir()
    if path.is_file() and path.name != "README.md"
}


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_a_starter_carries_the_canonical_book_byte_for_byte(pack: Path) -> None:
    """A starter file named like one in configs/starter_lib/ is that file."""
    assert CANONICAL_BOOK, "configs/starter_lib/ holds no canonical module"
    differing = [
        str(path.relative_to(pack))
        for path in sorted((pack / "starter").rglob("*"))
        if path.name in CANONICAL_BOOK and path.read_bytes() != CANONICAL_BOOK[path.name]
    ]
    assert not differing, (
        f"{pack.name} edits its copy of configs/starter_lib/ ({', '.join(differing)}): copy the "
        "canonical file unchanged, or change it there first"
    )
