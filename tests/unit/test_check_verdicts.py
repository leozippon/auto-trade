"""The consistency check (``scripts/dev/check_verdicts.py``) on one synthetic
experiment the real worker ran to its readout.

The run over the real experiments reads ``experiments/`` and is made by hand (the
script's docstring says how); here it reads an experiment small enough for the unit
suite: every reading the experiment stores reproduces, a float that differs only in
its last bits still does, the two keys a retired gate stored do, and a tampered one is
named with both values and fails the run.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from autotrade.common.time import utc_now
from autotrade.pipelines import pit_backend, worker
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from scripts.dev.check_verdicts import (
    _ABSENT,
    check_experiment,
    current_experiments,
    main,
)
from tests.unit.support.synthetic_experiment import (
    SyntheticPITProvider,
    few_baseline_draws,
    make_experiment,
)

# Each test runs a synthetic experiment end to end before it checks a reading.
pytestmark = pytest.mark.slow

LEDGER = Path("ledgers") / "experiment_ledger.jsonl"


@pytest.fixture(scope="module", autouse=True)
def few_draws():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pit_backend, "run_null_control", few_baseline_draws)
        yield


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An experiment created now, with a recorded registration check, validation record and readout."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker, "ResearchPITSnapshotProvider", SyntheticPITProvider)
        repo, experiment = make_experiment(tmp_path_factory.mktemp("judged"), _created_at=utc_now("auto"))
        result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["verdict"] == {"status": "registered_with_record", "reasons": []}
    return experiment


def _copy(experiment: Path, root: Path, edit=None, **params: object) -> Path:
    """The experiment copied under ``root``, its ledger lines passed through
    ``edit`` and its parameters updated with ``params``. The records still
    name the paths the worker wrote at, and are read under the copy."""

    target = root / experiment.name
    shutil.copytree(experiment, target)
    path = target / "hitl" / "params.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **params}), encoding="utf-8")
    records = [json.loads(line) for line in (target / LEDGER).read_text(encoding="utf-8").splitlines()]
    if edit is not None:
        edit(records)
    (target / LEDGER).write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return target


def _record(records: list[dict], kind: str) -> dict:
    return next(row for row in records if row["record_type"] == kind)


def test_every_reading_of_a_current_experiment_reproduces(experiment: Path) -> None:
    judgements = check_experiment(experiment)
    assert [item.reading for item in judgements] == ["check_registration", "forward", "readout"]
    assert [item.differences for item in judgements] == [[], [], []]
    assert current_experiments(experiment.parent) == [experiment]
    assert main(["--experiments", str(experiment.parent)]) == 0


def test_a_moved_experiment_is_judged_from_where_it_now_lives(experiment: Path, tmp_path: Path) -> None:
    """An experiment whose records name a root it no longer lives at is judged from
    its own files; a record naming a place outside the experiment fails the check
    instead of being read from there."""

    gone = tmp_path / "gone" / "experiments" / experiment.name

    def moved(records: list[dict]) -> None:
        records[:] = json.loads(json.dumps(records).replace(str(experiment), str(gone)))

    copy = _copy(experiment, tmp_path / "experiments", moved)
    records = [json.loads(line) for line in (copy / LEDGER).read_text(encoding="utf-8").splitlines()]
    assert _record(records, "forward")["result_ref"].startswith(str(gone))
    judgements = check_experiment(copy)
    assert [(item.reading, item.differences) for item in judgements] == [
        ("check_registration", []),
        ("forward", []),
        ("readout", []),
    ]

    def elsewhere(records: list[dict]) -> None:
        moved(records)
        forward = _record(records, "forward")
        forward["result_ref"] = forward["result_ref"].replace(f"/{experiment.name}/", "/another_experiment/")

    with pytest.raises(FileNotFoundError, match="outside the experiment"):
        check_experiment(_copy(experiment, tmp_path / "elsewhere" / "experiments", elsewhere))


def _move_floats(records: list[dict], by) -> None:
    """Move the stored forward slice's lower bound and the readout's residual IR with ``by``."""

    forward = _record(records, "forward")
    for block, key in (
        (forward["slices"]["forward"], "lower_bound"),
        (forward["readout"]["record"], "residual_information_ratio"),
    ):
        block[key] = by(block[key])


def test_a_float_that_differs_only_in_its_last_bits_reproduces(experiment: Path, tmp_path: Path) -> None:
    """A reading recorded on another machine differs from today's in the last
    bits of its floats (at most 139 ulps on the real experiments)."""

    def last_bits(value: float) -> float:
        for _ in range(200):
            value = math.nextafter(value, 0.0)
        return value

    copy = _copy(experiment, tmp_path / "experiments", lambda records: _move_floats(records, last_bits))
    assert [item.differences for item in check_experiment(copy)] == [[], [], []]
    assert main(["--experiments", str(copy.parent)]) == 0


def test_a_tampered_reading_is_named_with_both_values_and_fails_the_run(
    experiment: Path, tmp_path: Path
) -> None:
    """A float moved by a millionth of itself, and a registration's verdict,
    are enough."""

    def tampered(records: list[dict]) -> None:
        _record(records, "research_session")["registration"]["passed"] = False
        _move_floats(records, lambda value: value * (1 - 1e-6))

    copy = _copy(experiment, tmp_path / "experiments", tampered)
    records = [json.loads(line) for line in (experiment / LEDGER).read_text(encoding="utf-8").splitlines()]
    forward = _record(records, "forward")
    bound = forward["slices"]["forward"]["lower_bound"]
    ratio = forward["readout"]["record"]["residual_information_ratio"]
    judgements = {item.reading: item.differences for item in check_experiment(copy)}
    assert judgements == {
        "check_registration": [("passed", False, True)],
        "forward": [("forward.lower_bound", bound * (1 - 1e-6), bound)],
        "readout": [("record.residual_information_ratio", ratio * (1 - 1e-6), ratio)],
    }
    assert main(["--experiments", str(copy.parent)]) == 1


def test_only_the_retired_seed_gate_keys_a_registration_stored_are_ignored(
    experiment: Path, tmp_path: Path
) -> None:
    """A registration stored before the seed-replicate gate retired keeps its two
    flags, which today's reading no longer emits; any other key it alone holds is
    still a difference."""

    def stored_before(records: list[dict]) -> None:
        seeds = _record(records, "research_session")["registration"]["seed_replicates"]
        seeds.update(deterministic=True, trains_a_model=False, another_flag=True)

    copy = _copy(experiment, tmp_path / "experiments", stored_before)
    judgements = {item.reading: item.differences for item in check_experiment(copy)}
    assert judgements == {
        "check_registration": [("seed_replicates.another_flag", True, _ABSENT)],
        "forward": [],
        "readout": [],
    }
    assert main(["--experiments", str(copy.parent)]) == 1
