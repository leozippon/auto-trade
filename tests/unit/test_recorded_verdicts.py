"""The standing recompute check (``scripts/dev/recompute_verdicts.py``) on one
synthetic arm the real worker ran to its verdict.

The run over the real records reads ``experiments/`` and is made by hand (the
script's docstring says how); here the same two modes read an arm small enough
for the unit suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from autotrade.pipelines import worker
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from scripts.dev.recompute_verdicts import (
    BASELINE_PATHS,
    MANDATED,
    REPO_ROOT,
    VARIANTS,
    WITH_REPLICATES,
    baseline_dump,
    changed,
    check_records,
    differential,
    dump,
)
from tests.unit.synthetic_arm import SyntheticPITProvider, make_arm

ARM = "arm"
LEDGER = Path("ledgers") / "experiment_ledger.jsonl"


@pytest.fixture(scope="module")
def experiments(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An experiments root holding one arm with a recorded freeze gate and a
    recorded forward verdict."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker, "ResearchPITSnapshotProvider", SyntheticPITProvider)
        repo, experiment = make_arm(tmp_path_factory.mktemp("recorded"))
        result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["verdict"] == {"status": "graduated", "reasons": []}
    return experiment.parent


@pytest.fixture(scope="module")
def current(experiments: Path) -> dict:
    return dump(experiments, jobs=1)


def _edited(experiments: Path, tmp_path: Path, edit: Callable[[dict, dict], None]) -> Path:
    """A second root with the arm's parameters and its ledger after ``edit``
    changed the research and forward records. The stored inputs stay where the
    records point."""

    source, target = experiments / ARM, tmp_path / "experiments" / ARM
    (target / "hitl").mkdir(parents=True)
    (target / "ledgers").mkdir()
    shutil.copy(source / "hitl" / "params.json", target / "hitl" / "params.json")
    research, forward = (
        json.loads(line) for line in (source / LEDGER).read_text(encoding="utf-8").splitlines()
    )
    edit(research, forward)
    (target / LEDGER).write_text(
        json.dumps(research) + "\n" + json.dumps(forward) + "\n", encoding="utf-8"
    )
    return target.parent


def _moved(research: dict, forward: dict) -> None:
    research["freeze_gate"]["thresholds"]["min_information_ratio"] = 9.0
    research["freeze_gate"]["reasons"] = ["freeze_information_ratio_below_threshold"]
    forward["verdict"]["status"] = "discarded"


def test_the_recorded_gate_and_verdict_reproduce(experiments: Path):
    judgements = check_records(experiments, jobs=1, exceptions={})

    assert [tuple(item) for item in judgements] == [
        ("freeze_gate", ARM, "reproduced", ""),
        ("forward", ARM, "reproduced", ""),
    ]


def test_a_recorded_value_the_code_does_not_reproduce_fails(experiments: Path, tmp_path: Path):
    gate, verdict = check_records(_edited(experiments, tmp_path, _moved), jobs=1, exceptions={})

    assert (gate.outcome, gate.detail) == (
        "failed",
        "differs at ['reasons', 'thresholds.min_information_ratio']",
    )
    assert (verdict.outcome, verdict.detail) == ("failed", "differs at ['verdict.status']")


def test_a_key_added_since_the_record_is_listed_and_reproduces(experiments: Path, tmp_path: Path):
    def recorded_before_the_holder_readings(_research: dict, forward: dict) -> None:
        del forward["slices"]["forward"]["raw_readings"]
        del forward["verdict"]["holder_line"]

    root = _edited(experiments, tmp_path, recorded_before_the_holder_readings)
    _gate, verdict = check_records(root, jobs=1, exceptions={})

    assert (verdict.outcome, verdict.detail) == (
        "reproduced",
        "added: verdict.holder_line, slices.forward.raw_readings",
    )


def test_a_declared_exception_covers_exactly_the_keys_it_names(experiments: Path, tmp_path: Path):
    moved = _edited(experiments, tmp_path, _moved)
    keys = frozenset({"reasons", "thresholds.min_information_ratio"})

    gate, verdict = check_records(
        moved, jobs=1, exceptions={(ARM, "freeze_gate"): ("an old rule", keys)}
    )
    assert (gate.outcome, gate.detail) == ("declared", "2 keys: an old rule")
    # The exception is the gate's: the verdict beside it still fails.
    assert verdict.outcome == "failed"

    # Fewer keys than differ, a judgement that reproduces after all, and an
    # arm that records no such judgement are each a failure.
    narrower = {(ARM, "freeze_gate"): ("an old rule", frozenset({"reasons"}))}
    assert check_records(moved, jobs=1, exceptions=narrower)[0].outcome == "failed"
    stale = check_records(experiments, jobs=1, exceptions={(ARM, "freeze_gate"): ("gone", keys)})
    assert stale[0].outcome == "failed" and "no longer matches" in stale[0].detail
    absent = check_records(experiments, jobs=1, exceptions={("retired", "forward"): ("x", keys)})
    assert tuple(absent[-1]) == (
        "forward",
        "retired",
        "failed",
        "the declared exception names no recorded judgement",
    )


def test_the_dump_judges_every_stored_input_under_every_variant_of_its_rules(current: dict):
    assert list(VARIANTS) == ["R0", "R1", "R3", MANDATED]
    variants = ["own", *VARIANTS, WITH_REPLICATES]
    arm = current["arms"][ARM]
    assert list(arm["rules"]) == list(arm["facts"]) == ["own", *VARIANTS]
    (session,) = arm["sessions"].values()
    assert session["freeze_gate"]["passed"] is True
    assert session["full_span_bar"]["trials"] >= 1
    assert len(session["nominees"]) == 2
    for gates in session["nominees"].values():
        assert list(gates) == variants
        # The raw condition's reading and stamp exist exactly where an era holds it.
        assert "raw_excess_at_cost_stress" not in gates["R0"]
        assert "cost_stress_multiplier" not in gates["R0"]["thresholds"]
        assert gates["R1"]["thresholds"]["cost_stress_multiplier"] == 2.0
        assert "require_seed_replicates" not in gates["R1"]["thresholds"]
        assert gates["R3"]["thresholds"]["require_seed_replicates"] is True
        # Named, the session's other Step is read as a replicate and refused:
        # it holds the nominee's own bytes.
        assert gates["R3"]["seed_replicates"]["replicates"] == []
        assert gates[WITH_REPLICATES]["reasons"] == ["freeze_seed_replicate_invalid"]
        # No arm on disk holds a tracking mandate; under one, a one-name book
        # is refused for the tracking error it cannot keep.
        assert gates["own"]["thresholds"]["tracking_error_cap"] is None
        assert gates[MANDATED]["thresholds"]["tracking_error_cap"] == 0.08
        assert "freeze_tracking_error_above_cap" in gates[MANDATED]["reasons"]
    (verdicts,) = arm["forward"].values()
    assert list(verdicts) == ["recorded", *VARIANTS, WITH_REPLICATES]
    stamped = {
        name: sorted(key for key in block["verdict"]["thresholds"] if key.startswith("require_"))
        for name, block in verdicts.items()
    }
    assert stamped == {
        "recorded": [],
        "R0": [],
        "R1": [],
        "R3": ["require_forward_plain_selection"],
        MANDATED: [],
        WITH_REPLICATES: ["require_forward_plain_selection"],
    }
    assert "forward_tracking_error_above_cap" in verdicts[MANDATED]["verdict"]["reasons"]
    assert verdicts["recorded"]["verdict"]["reasons"] == []
    # The series F8 judges exists exactly where an era holds the condition:
    # the book's own, or the book with the replicates named.
    assert "judged_selection" not in verdicts["R1"]["slices"]["forward"]
    alone = verdicts["R3"]["slices"]["forward"]
    named = verdicts[WITH_REPLICATES]["slices"]["forward"]
    assert "seed_mean" not in alone and alone["judged_selection"]["members"] == 1
    assert named["seed_mean"]["members"] == named["judged_selection"]["members"] == 2
    # The creation contract rides along, in the console's own order.
    assert [key for key, _value in current["creation"]["defaults"]] == list(WEB_CREATE_DEFAULTS)
    assert "min_dsr_probability" in current["creation"]["accepted"]


def test_two_dumps_differ_by_a_value_its_type_a_one_sided_key_or_key_order():
    same = {"a": 1, "b": [1.5, {"c": None}], "n": float("nan")}
    assert changed(same, json.loads(json.dumps(same))) == []
    assert changed({"a": {"b": [1.5, {"c": 2}]}}, {"a": {"b": [1.5, {"c": 3}]}}) == ["a.b[1].c"]
    assert changed({"a": 1}, {"a": 1.0}) == ["a"]
    assert changed({"a": True}, {"a": 1}) == ["a"]
    assert changed({"a": {"x": 1}}, {"a": {"x": 2, "y": 2}}) == ["a.y (one side only)", "a.x"]
    assert changed({"a": {"x": 1, "y": 2}}, {"a": {"y": 2, "x": 1}}) == ["a (key order)"]
    assert changed({"a": [1, 2]}, {"a": [1, 2, 3]}) == ["a"]


def test_the_differential_names_exactly_what_a_baseline_judges_differently(
    experiments: Path, current: dict, tmp_path: Path
):
    """A baseline tree whose bootstrap draws half as often: its own script,
    run on its own code, reads another forward lower bound and nothing else."""

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "baseline"
    for relative in BASELINE_PATHS:
        source, target = REPO_ROOT / relative, repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(source, target)
    module = repo / "src" / "autotrade" / "pipelines" / "verdict.py"
    text = module.read_text(encoding="utf-8")
    assert "BOOTSTRAP_DRAWS = 2_000\n" in text
    module.write_text(
        text.replace("BOOTSTRAP_DRAWS = 2_000\n", "BOOTSTRAP_DRAWS = 1_000\n"), encoding="utf-8"
    )
    git = ["git", "-C", str(repo)]
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run([*git, "init", "--quiet"], check=True)
    subprocess.run([*git, "add", "--all"], check=True)
    subprocess.run(
        [*git, *identity, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "baseline"],
        check=True,
    )

    differing, moved = differential(baseline_dump("HEAD", experiments, jobs=1, repo=repo), current)

    assert differing and not moved
    assert {path.rsplit(".", 1)[-1] for path in differing} == {"lower_bound", "bootstrap_draws"}
    assert all(path.startswith(f"arms.{ARM}.forward.") for path in differing)


def test_an_arm_whose_stored_inputs_changed_between_the_readings_is_named_not_compared(
    current: dict,
):
    later = json.loads(json.dumps(current))
    later["arms"][ARM]["inputs"] = "the ledger grew"
    later["arms"][ARM]["forward"] = {}
    later["arms"]["created_meanwhile"] = {"inputs": "a new arm"}

    assert differential(current, later) == ([], [ARM, "created_meanwhile"])
    assert differential(current, json.loads(json.dumps(current))) == ([], [])
